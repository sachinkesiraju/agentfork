"""The project engine: experiment nodes as real agentfork branches.

One project owns one :class:`ForkOrchestrator`, so every node in the
experiment tree is a genuine branch — a git worktree (sandbox half) plus a KV
context slice (kv half) under a single branch ID, cleaned up by
``kill_losers``/``kill`` rather than ad-hoc bookkeeping.

Scoring follows the map-reduce contract: candidates are scored by grepping the
eval run's own log (``runner.extract_score``), never by a worker's claim, and
reduction is ``amr.reduce`` — survive only by beating your own parent by more
than the baseline noise margin, within cost guards.

Runs are detached processes (``runner``); a polling thread adopts them, reads
exit codes, extracts scores, and logs to ``results.tsv``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from agentfork.app import amr, git, runner
from agentfork.app.harness import Idea, build_harness
from agentfork.app.store import Store
from agentfork.orchestrator import ForkOrchestrator, SandboxBackend

_log = logging.getLogger("agentfork.app.engine")

DEFAULT_PARAMS = {
    "eval_cmd": "",
    "metric_grep": "",
    "minimize": True,
    "k": 4, "b": 2,
    "eval_slots": 2,
    "timeout_s": 600,
    "max_gens": 3,
    "baseline_runs": 2,
    "holdout_cmd": "",
    "cost_guards": {},      # name -> relative tolerance
    "model": "",           # passed to the harness (cli flag or api model id)
    "api_base": "",        # harness=api: OpenAI-compatible endpoint base
}

_WORKTREES = "worktrees"
_RUNS = "runs"


class EngineError(RuntimeError):
    pass


def _tokenize(text: str) -> list[int]:
    """Deterministic stand-in tokenizer for the CPU reference cache: the KV
    bookkeeping (dedup ratio, prefill tokens saved) is what matters here."""
    return [ord(c) for c in text]


class WorktreeSandbox(SandboxBackend):
    """Sandbox half of a node: a git worktree under the project's run dir.

    ``spawn`` creates branch ``af/<branch_id>`` from the parent's branch (or
    the baseline ref at the root) and checks it out; ``kill`` removes the
    worktree but keeps the branch — it is the node's evidence. Both halves
    are restart-tolerant: a worktree that already exists is reattached as-is
    (the checkout is disposable, the branch is not), and a kill of a branch
    that still has a run inside it leaves the worktree alone.
    """

    parallel_lifecycle = True

    def __init__(self, repo: str, base_dir: Path, baseline: str,
                 is_busy: Callable[[str], bool] | None = None):
        self.repo = repo
        self.base_dir = Path(base_dir)
        self.baseline = baseline
        self.is_busy = is_busy or (lambda branch_id: False)

    def _branch(self, branch_id: str) -> str:
        # orchestrator child ids are "parent/n"; a git ref cannot be both a
        # leaf and a directory, so flatten the slash
        return "af/" + branch_id.replace("/", "__")

    def _path(self, branch_id: str) -> Path:
        return self.base_dir / branch_id

    def spawn(self, branch_id: str, parent_id: str | None) -> None:
        path = self._path(branch_id)
        if path.exists():
            return  # reattach: the checkout is still on this node's branch
        branch = self._branch(branch_id)
        if not git.branch_exists(self.repo, branch):
            start = self._branch(parent_id) if parent_id else self.baseline
        else:
            start = branch  # add_worktree checks an existing branch out as-is
        git.add_worktree(self.repo, path, branch, start)

    def kill(self, branch_id: str) -> None:
        if self.is_busy(branch_id):
            # a detached run is still executing inside the worktree; leave the
            # checkout in place — a later idle-time kill collects it
            return
        git.remove_worktree(self.repo, self._path(branch_id))

    def alive(self, branch_id: str) -> bool:
        return self._path(branch_id).exists()

    def worktree(self, branch_id: str) -> str | None:
        path = self._path(branch_id)
        return str(path) if path.exists() else None


class Engine:
    def __init__(self, store: Store, home: Path | None = None):
        self.store = store
        self.home = home or (store.path.parent)
        self._orchestrators: dict[str, ForkOrchestrator] = {}
        self._sandboxes: dict[str, WorktreeSandbox] = {}
        self._reconciled: set[str] = set()
        # projects currently driven by an AutoresearchLoop; the loop owns the
        # loop state while it runs, so single steps must not overwrite it
        self.driven: set[str] = set()
        self._stop = threading.Event()
        # a worker row with no pid is a legacy in-process worker orphaned by
        # a previous dashboard — it can never finish, so settle it now.
        # pid-ful rows are detached processes the poller adopts normally.
        for run in self.store.active_runs():
            if run["kind"] == "worker" and not run.get("pid"):
                self.store.update_run(run["id"], status="failed",
                                      ended_at=time.time())
                node = self.store.node(run["node_id"])
                if node and node["status"] in ("proposed", "implementing"):
                    self.store.update_node(node["id"], status="crash")
        self._poller = threading.Thread(target=self._poll_loop, daemon=True)
        self._poller.start()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._stop.set()
        self._poller.join(timeout=5)
        for orch in self._orchestrators.values():
            try:
                # shutdown only stops the reaper and drops the registry lock —
                # it must not kill branches, or restarting the dashboard would
                # delete worktrees under evals that are still running detached
                orch.stop_reaper()
                orch._release_registry_lock()
            except Exception:
                pass

    def project_dir(self, project_id: str) -> Path:
        d = self.home / "projects" / project_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def results(self, project_id: str) -> amr.Results:
        return amr.Results(self.project_dir(project_id) / "results.tsv")

    def _orchestrator(self, project: dict,
                      reconcile: bool = True) -> ForkOrchestrator:
        """The project's orchestrator, built on first use.

        ``reconcile`` replays kill over rows loaded from a previous dashboard
        process (the runtime cannot adopt another owner's sandbox handles or
        KV state), which also removes their worktrees — so read-only callers
        pass ``False`` and the collection happens on the first mutating use.
        """
        orch = self._orchestrators.get(project["id"])
        if orch is None:
            base = self.project_dir(project["id"]) / _WORKTREES
            sandbox = WorktreeSandbox(
                project["repo_path"], base, project["baseline_branch"],
                is_busy=lambda bid, pid=project["id"]: self._node_busy(pid, bid))
            orch = ForkOrchestrator(
                sandbox=sandbox,
                registry_path=self.project_dir(project["id"]) / "orch.journal")
            self._orchestrators[project["id"]] = orch
            self._sandboxes[project["id"]] = sandbox
        if reconcile and project["id"] not in self._reconciled:
            self._reconciled.add(project["id"])
            orch.reconcile()
        return orch

    def _emit(self, project_id: str, kind: str, **payload) -> None:
        self.store.add_event(project_id, kind, payload)

    def _set_step_state(self, project_id: str, **changes) -> None:
        """Loop state for a single manual step, ignored while a loop drives
        the project (its own state machine is authoritative then)."""
        if project_id in self.driven:
            return
        self.store.set_loop_state(project_id, **changes)

    # -- projects ------------------------------------------------------------

    def create_project(self, name: str, repo_path: str, baseline_branch: str,
                       harness_name: str, params: dict) -> dict:
        repo = Path(repo_path).expanduser()
        if not repo.exists() or not git.is_repo(repo):
            raise EngineError(f"{repo_path}: not a git repository")
        repo_path = git.toplevel(repo)
        if not git.branch_exists(repo_path, baseline_branch):
            raise EngineError(f"{repo_path}: no branch {baseline_branch!r}")
        merged = {**DEFAULT_PARAMS, **params}
        if not merged["eval_cmd"]:
            raise EngineError("eval_cmd is required — it is the project's "
                              "fixed run command")
        if not merged["metric_grep"]:
            raise EngineError("metric_grep is required — scores are grepped "
                              "from eval logs, never taken from a worker")
        git.gitignore(repo_path, ["results.tsv", ".agentfork/",
                                  ".agentfork-candidate"])
        project = self.store.create_project(
            name=name, repo_path=repo_path, baseline_branch=baseline_branch,
            tag=name, harness=harness_name, params=merged)
        self._emit(project["id"], "project.created",
                   name=name, repo=repo_path)
        return project

    # -- baseline -----------------------------------------------------------

    def baseline(self, project_id: str) -> dict:
        """Create the root node and queue the baseline evals that set the
        noise margin. Baseline = eval_cmd on the untouched baseline branch,
        ``baseline_runs`` times."""
        project = self.store.project(project_id)
        if self.store.nodes(project_id, gen=0):
            raise EngineError("baseline already exists")
        orch = self._orchestrator(project)
        orch.create_parent(project_id,
                           tokens=_tokenize(self._context(project_id)))
        node = self.store.create_node(
            project_id=project_id, parent_id=None, gen=0, slug="baseline",
            # the worktree sits on the branch the orchestrator just made
            # (af/<project_id>) — never on the user's baseline ref
            branch_name="af/" + project_id, title="baseline",
            node_id=project_id, status="baseline",
            worktree_path=self._sandboxes[project_id].worktree(project_id),
            commit_sha=git.head_sha(project["repo_path"],
                                    project["baseline_branch"]))
        self.store.set_loop_state(project_id, state="baseline",
                                  frontier=[node["id"]], gen=0)
        self._emit(project_id, "node.created", node_id=node["id"],
                   title="baseline")
        for _ in range(project["params"]["baseline_runs"]):
            self.start_eval(project_id, node["id"])
        return node

    # -- fan out -------------------------------------------------------------

    def _build_harness(self, project: dict):
        params = project["params"]
        return build_harness(project["harness"],
                             model=params.get("model") or None,
                             api_base=params.get("api_base") or None)

    def propose(self, project_id: str, parent_node_id: str, k: int) -> list[Idea]:
        project = self.store.project(project_id)
        harness = self._build_harness(project)
        ctx = self._context(project_id)
        parent = self.store.node(parent_node_id)
        if parent is not None:
            ctx += (f"\n\nThe parent candidate was {parent['title']!r} "
                    f"(score {parent['score']}). Propose its children — each "
                    "idea forks THIS parent's code and is scored relative to "
                    "it.\n" + (parent["description"] or ""))
        return harness.propose(ctx, k, cwd=project["repo_path"])

    def fan_out(self, project_id: str, parent_node_id: str,
                ideas: list[Idea]) -> list[dict]:
        """Fork the parent node once per idea: worktree + KV context + a
        worker run that applies the idea, then an eval run."""
        project = self.store.project(project_id)
        parent = self.store.node(parent_node_id)
        if parent is None:
            raise EngineError(f"no such node: {parent_node_id}")
        # A node must be settled before children fork from it: its worktree
        # HEAD is the start point, and mid-work the candidate isn't committed
        # yet. Scored/kept nodes are frozen but still expand — descending from
        # a winner is the whole point of the search.
        if parent["status"] in ("proposed", "implementing", "ready",
                                "running"):
            raise EngineError(f"{parent_node_id} is still {parent['status']} — "
                              "wait for its runs to finish")
        orch = self._orchestrator(project)
        self._adopt_branch(orch, project, parent)
        # name children ourselves: the orchestrator's next-id counter only
        # knows live branches, but node ids are the tsv's commits — a reused
        # id would collide with a discarded child's row (and after a
        # restart, with every previous child's)
        used = {n["id"] for n in self.store.nodes(project_id)}
        child_ids = []
        i = 1
        while len(child_ids) < len(ideas):
            cid = f"{parent_node_id}/{i}"
            i += 1
            if cid not in used:
                child_ids.append(cid)
        children = orch.fork(parent_node_id, n=len(ideas),
                             child_ids=child_ids)
        nodes = []
        for branch, idea in zip(children, ideas):
            orch.extend(branch.branch_id, _tokenize(idea.description))
            node = self.store.create_node(
                project_id=project_id, parent_id=parent_node_id,
                gen=parent["gen"] + 1, slug=git.slugify(idea.title),
                branch_name="af/" + branch.branch_id.replace("/", "__"),
                title=idea.title,
                description=idea.description, regions=idea.regions,
                status="implementing",
                worktree_path=self._sandboxes[project_id].worktree(
                    branch.branch_id),
                node_id=branch.branch_id)
            nodes.append(node)
            self._emit(project_id, "node.created", node_id=node["id"],
                       title=idea.title, gen=node["gen"],
                       parent_id=parent_node_id)
        self._set_step_state(project_id, gen=parent["gen"] + 1,
                             state="running",
                             message=f"gen {parent['gen'] + 1}: "
                                     f"{len(nodes)} candidates")
        harness = self._build_harness(project)
        for node, idea in zip(nodes, ideas):
            self.start_worker(project, node, harness, idea)
        return nodes

    # -- runs ---------------------------------------------------------------

    def start_worker(self, project: dict, node: dict, harness, idea: Idea):
        """A worker run is a detached process (``agentfork.app.worker``) —
        the harness applies the idea inside the node's worktree, the worker
        commits it and queues the eval. A real pid means a worker can be
        killed mid-edit, streams its log like an eval's, and survives the
        dashboard (its worktree is left alone on close)."""
        run = self.store.create_run(project_id=project["id"],
                                    node_id=node["id"], kind="worker",
                                    command=f"harness:{harness.name}",
                                    run_dir="", status="queued")
        run_dir = self.project_dir(project["id"]) / _RUNS / run["id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "payload.json").write_text(json.dumps({
            "db": str(self.store.path),
            "project_id": project["id"],
            "project_dir": str(self.project_dir(project["id"])),
            "node_id": node["id"],
            "title": node["title"],
            "description": node["description"],
            "regions": node["regions"],
            "context": self._context(project["id"]),
        }))
        import agentfork
        package_root = str(Path(agentfork.__file__).resolve().parent.parent)
        env = {"PYTHONPATH": package_root + os.pathsep
               + os.environ.get("PYTHONPATH", "")}
        command = f"'{sys.executable}' -m agentfork.app.worker '{run_dir}'"
        handle = runner.start(run_dir, command, run_dir, env=env)
        return self.store.update_run(
            run["id"], status="running", run_dir=str(run_dir),
            command=command, pid=handle.pid, started_at=time.time())

    def start_eval(self, project_id: str, node_id: str,
                   kind: str = "eval", force: bool = False) -> dict:
        project = self.store.project(project_id)
        node = self.store.node(node_id)
        if node is None or node["worktree_path"] is None:
            raise EngineError(f"node {node_id} has no worktree")
        self._ensure_worktree(project, node)
        # Frozen means a run already answered the node; only an explicit
        # caller re-answer (``force``) or a holdout may re-run it.
        if node["frozen"] and not (force or kind == "holdout"):
            raise EngineError(f"node {node_id} is frozen — it already has a "
                              "score; pass force to re-answer it")
        params = project["params"]
        command = params["holdout_cmd"] if kind == "holdout" else params["eval_cmd"]
        if not command:
            raise EngineError(f"{kind} command is empty")
        # the run row comes first so its id keys the run directory: two runs
        # of the same node (the baseline pair, a re-run) must not share a
        # log, pid or exit_code file
        run_dir = self.project_dir(project_id) / _RUNS / (
            f"{kind}-{node_id}-{time.time_ns()}")
        run = self.store.create_run(project_id=project_id, node_id=node_id,
                                    kind=kind, command=command,
                                    run_dir=str(run_dir), status="queued")
        if self._slots_free(project_id) <= 0:
            self._emit(project_id, "run.queued", run_id=run["id"],
                       node_id=node_id, run_kind=kind)
            return run
        return self._launch(run)

    def _launch(self, run: dict) -> dict:
        """Start a queued eval's process if a slot is free, else leave it
        queued — the poller drains the queue as slots open up.
        ``eval_slots`` is the project's fixed resource budget for concurrent
        evaluators."""
        node = self.store.node(run["node_id"])
        project = self.store.project(run["project_id"])
        self._ensure_worktree(project, node)
        run_dir = Path(run["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        if run["kind"] == "eval":
            # a re-answer unfreezes the node so the new score lands on it;
            # a holdout is validation, not an answer — the node stays put
            self.store.update_node(node["id"], status="running", frozen=False)
        handle = runner.start(run_dir, run["command"], node["worktree_path"],
                              env={"AGENTFORK_NODE": run["node_id"],
                                   "AGENTFORK_RUN": run["id"],
                                   "AGENTFORK_RUN_DIR": str(run_dir)})
        run = self.store.update_run(run["id"], status="running",
                                    pid=handle.pid, started_at=time.time())
        self._emit(run["project_id"], "run.started", run_id=run["id"],
                   node_id=run["node_id"], run_kind=run["kind"],
                   pid=handle.pid)
        return run

    def _slots_free(self, project_id: str) -> int:
        """How many more evaluators may start under ``eval_slots``;
        ``None``/0 means unbounded."""
        params = self.store.project(project_id)["params"]
        slots = int(params.get("eval_slots") or 0)
        if not slots:
            return 1 << 30
        running = [r for r in self.store.runs(project_id)
                   if r["kind"] in ("eval", "holdout")
                   and r["status"] == "running"]
        return max(0, slots - len(running))

    def _ensure_worktree(self, project: dict, node: dict) -> None:
        """Re-check out a node's worktree if it is gone.

        A node's evidence is its git branch and commit, not its working
        directory: shutting the dashboard down collects the sandboxes, so a
        node answered in an earlier session has a branch but no checkout.
        """
        path = Path(node["worktree_path"])
        if path.exists():
            return
        branch = node["branch_name"]
        if not branch or not git.branch_exists(project["repo_path"], branch):
            raise EngineError(
                f"node {node['id']}: worktree and branch are both gone")
        git.add_worktree(project["repo_path"], path, branch,
                         node["commit_sha"] or branch)
        self._emit(project["id"], "node.restored", node_id=node["id"],
                   worktree=str(path))

    def kill_run(self, run_id: str) -> dict:
        run = self.store.run(run_id)
        if run is None:
            raise EngineError(f"no such run: {run_id}")
        if run["status"] not in ("queued", "running"):
            return run  # already settled — nothing to kill
        if run["status"] == "running":
            runner.kill(run["run_dir"])
        self._emit(run["project_id"], "run.killed", run_id=run_id)
        # settle the node too: a killed run still answers it (as a crash),
        # otherwise the node is stuck "running" with nothing left to finish it
        self._finish_run(run, runner_exit=runner.exit_code(run["run_dir"]),
                         killed="killed")
        return self.store.run(run_id)

    def _node_busy(self, project_id: str, node_id: str) -> bool:
        return any(r["node_id"] == node_id
                   and r["status"] in ("queued", "running")
                   for r in self.store.runs(project_id))

    def _adopt_branch(self, orch: ForkOrchestrator, project: dict,
                      node: dict) -> None:
        """Re-register a node's branch after a restart.

        The runtime deliberately cannot adopt a previous process's branches
        (sandbox handles and KV trees are process-local), so a node whose
        branch row was collected is re-parented fresh: the worktree is
        reattached at its own branch, the KV tree gets the node's context
        re-tokenized. Children then fork from it normally.
        """
        rows = {b.branch_id: b for b in orch.branches()}
        row = rows.get(node["id"])
        if row is not None and row.state == "live":
            return
        if row is not None:
            orch.kill(node["id"])  # clear the dead row (busy-safe on worktree)
        orch.create_parent(node["id"],
                           tokens=_tokenize(node["description"] or
                                            node["title"]))

    # -- reduce ---------------------------------------------------------------

    def reduce(self, project_id: str, gen: int,
               margin: float | None = None) -> amr.Reduction:
        project = self.store.project(project_id)
        params = project["params"]
        if margin is None:
            margin = self._margin(project_id)
        in_flight = [n["id"] for n in self.store.nodes(project_id, gen=gen)
                     if n["status"] in ("proposed", "implementing", "ready",
                                        "running")]
        if in_flight:
            raise EngineError(
                f"gen {gen} still has {len(in_flight)} candidate(s) in flight "
                f"({', '.join(in_flight[:3])}…) — wait for their evals")
        red = self.results(project_id).reduce(
            gen, margin, beam=params["b"], minimize=params["minimize"],
            guards=params["cost_guards"])
        orch = self._orchestrator(project)
        kept = {s.commit for s in red.kept}
        # commits in the tsv are node ids
        for node in self.store.nodes(project_id, gen=gen):
            if node["id"] in kept:
                self.store.update_node(node["id"], status="kept", frozen=True)
            elif node["status"] in ("ran", "crash", "cost_fail", "discarded"):
                if node["id"] not in kept and node["gen"] == gen:
                    self.store.update_node(node["id"], status="discarded",
                                           frozen=True)
        for fail_commit, _reason, _desc in red.cost_failures:
            self.store.update_node(fail_commit, status="cost_fail",
                                   frozen=True)
        if red.stall:
            law = (f"STALL at gen {gen}: no candidate beat its parent by "
                   f"more than the margin ({margin:g}).")
            self.store.add_law(project_id, gen, law)
            self.results(project_id).log(gen, f"stall-{gen}", "-", None,
                                         "note", law)
            self._emit(project_id, "law", gen=gen, text=law)
        if red.kept:
            losers = [n["id"] for n in self.store.nodes(project_id, gen=gen)
                      if n["id"] not in kept]
            for loser in losers:
                try:
                    orch.kill(loser)
                except Exception:  # noqa: BLE001 — keep going, reconcile later
                    _log.exception("kill failed for %s", loser)
            # the frontier is state the loop reads back, so it is always
            # written; only the human-facing state/message is step-scoped
            self.store.set_loop_state(project_id, gen=gen,
                                      frontier=[s.commit for s in red.kept])
            self._set_step_state(
                project_id, state="ready",
                message=f"gen {gen}: keeping "
                        f"{', '.join(s.commit for s in red.kept)}")
        else:
            self._set_step_state(
                project_id, gen=gen, state="stalled" if red.stall else "ready",
                message=(f"gen {gen}: STALL — frontier unchanged"
                         if red.stall else f"gen {gen}: nothing kept"))

        self._emit(project_id, "reduced", gen=gen,
                   kept=[s.commit for s in red.kept], stall=red.stall)
        return red

    def descend(self, project_id: str, node_id: str, k: int) -> list[dict]:
        """Propose + fan out in one step: the UI's "fan out" button."""
        ideas = self.propose(project_id, node_id, k)
        return self.fan_out(project_id, node_id, ideas)

    # -- context / margin -----------------------------------------------------

    def _context(self, project_id: str) -> str:
        project = self.store.project(project_id)
        params = project["params"]
        parts = [
            f"You are improving the repository at {project['repo_path']}.",
            f"The fixed eval command is: {params['eval_cmd']}",
            f"Metric is grepped from eval output with: {params['metric_grep']} "
            f"({'minimize' if params['minimize'] else 'maximize'}).",
        ]
        try:
            text = self.results(project_id).tree(
                minimize=params["minimize"]).render()
        except amr.AmrError:
            text = ""
        if text:
            parts.append("Experiment history:\n" + text)
        laws = self.store.laws(project_id)
        if laws:
            parts.append("Laws learned so far:\n" +
                         "\n".join(f"- {law['text']}" for law in laws))
        return "\n\n".join(parts)

    def _margin(self, project_id: str) -> float:
        """Noise margin = abs difference between the two baseline scores, per
        the map-reduce program."""
        scores = [r.score for r in self.results(project_id).rows()
                  if r.gen == 0 and r.score is not None]
        if len(scores) < 2:
            raise EngineError("margin needs both baseline runs to finish")
        # spread over every baseline run (baseline_runs can exceed two),
        # floored at an epsilon: a deterministic eval measures 0 noise, and
        # then an improvement still has to be a real nonzero delta
        return max(max(scores) - min(scores), 1e-9)

    # -- poller ---------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:  # noqa: BLE001
                _log.exception("poll tick failed")
            self._stop.wait(0.5)

    def _poll_once(self) -> None:
        # first drain queued evals into whatever slots are free
        for run in self.store.active_runs():
            if run["status"] != "queued" or run["kind"] == "worker":
                continue
            if self._slots_free(run["project_id"]) <= 0:
                continue
            try:
                self._launch(run)
            except Exception:  # noqa: BLE001
                _log.exception("launch failed for %s", run["id"])
                self._finish_run(run, runner_exit=127, killed=None)
        for run in self.store.active_runs():
            if run["status"] == "queued":
                continue  # waits on a slot; the drain above starts it
            run_dir = run["run_dir"]
            if runner.alive(run_dir):
                started = run.get("started_at") or run["created_at"]
                project = self.store.project(run["project_id"])
                timeout = project["params"]["timeout_s"]
                if timeout and time.time() - started > timeout:
                    runner.kill(run_dir)
                    self._finish_run(run, runner_exit=None, killed="timeout")
                continue
            # a run killed by a signal never writes exit_code
            exit_code = runner.exit_code(run_dir)
            self._finish_run(run, runner_exit=exit_code,
                             killed=None if exit_code is not None else "killed")

    def _finish_run(self, run: dict, runner_exit: int | None,
                    killed: str | None) -> None:
        project = self.store.project(run["project_id"])
        params = project["params"]
        node = self.store.node(run["node_id"])
        score = runner.extract_score(run["run_dir"], params["metric_grep"]) \
            if run["kind"] in ("eval", "holdout") else None
        status = (killed if killed else
                  "done" if runner_exit == 0 else
                  "failed" if runner_exit is not None else "killed")
        # atomic: if a kill already settled this row, nothing below may run —
        # otherwise the poller would resurrect it and log a duplicate row
        if not self.store.settle_run(run["id"], status=status,
                                     exit_code=runner_exit, score=score,
                                     ended_at=time.time()):
            return
        self._emit(run["project_id"], "run.finished", run_id=run["id"],
                   node_id=run["node_id"], status=status,
                   exit_code=runner_exit, score=score)
        if run["kind"] == "worker":
            # a failed/killed worker can't drive its node to ready — mark
            # the node so the tree doesn't sit on "implementing" forever
            if status != "done" and node and node["status"] == "implementing":
                self.store.update_node(node["id"], status="crash",
                                       frozen=node["gen"] > 0)
                self.results(run["project_id"]).log(
                    node["gen"], node["id"], node["parent_id"] or "-",
                    None, "crash", f"worker exited {runner_exit}",
                    regions=node["regions"])
                self._emit(run["project_id"], "node.crashed",
                           node_id=node["id"],
                           reason=f"worker exited {runner_exit}")
            self._settle_loop_state(run["project_id"])
            return
        if run["kind"] == "holdout":
            # validation, not a candidate answer: the score stays on the run
            # row; the tsv gets a note (byte-compatible) and the node's
            # status/score/delta are left alone
            if node is not None:
                self.results(run["project_id"]).log(
                    node["gen"], node["id"], node["parent_id"] or "-",
                    score, "note",
                    f"holdout {status}" +
                    (f" score={score:g}" if score is not None else ""),
                    costs=self._costs(run), regions=node["regions"])
            self._settle_loop_state(run["project_id"])
            return
        if run["kind"] == "eval" and node and not node["frozen"]:
            # while a sibling run of this node is still queued/running the
            # node stays in flight — the baseline pair only settles once
            # every eval for it has finished
            siblings = [r for r in self.store.runs(run["project_id"],
                                                 node_id=node["id"])
                        if r["id"] != run["id"]
                        and r["status"] in ("queued", "running")]
            if siblings:
                self.results(run["project_id"]).log(
                    node["gen"], node["id"], node["parent_id"] or "-",
                    score, "ran" if (score is not None and status == "done")
                    else "crash",
                    node["title"] if score is not None else
                    f"eval exited {runner_exit}",
                    costs=self._costs(run), regions=node["regions"])
            else:
                # the baseline node is answered by a *pair* of runs, so it
                # only freezes once _baseline_done has seen them all
                freeze = node["gen"] > 0
                if score is None or status != "done":
                    self.store.update_node(node["id"], status="crash",
                                           frozen=freeze)
                    self.results(run["project_id"]).log(
                        node["gen"], node["id"], node["parent_id"] or "-",
                        None, "crash", f"eval exited {runner_exit}")
                else:
                    parent = self.store.node(node["parent_id"]) \
                        if node["parent_id"] else None
                    delta = (score - parent["score"]
                             if parent and parent["score"] is not None
                             else None)
                    self.store.update_node(node["id"], status="ran",
                                           score=score, delta=delta,
                                           frozen=freeze)
                    self.results(run["project_id"]).log(
                        node["gen"], node["id"], node["parent_id"] or "-",
                        score, "ran", node["title"],
                        costs=self._costs(run), regions=node["regions"])
                if node["gen"] == 0:
                    self._baseline_done(run["project_id"], node)
                if score is not None and status == "done":
                    self._emit(run["project_id"], "node.scored",
                               node_id=node["id"], score=score, delta=delta)
        self._settle_loop_state(run["project_id"])

    def _settle_loop_state(self, project_id: str) -> None:
        """Leave a hand-driven project idle once nothing is in flight, so the
        header stops offering to stop a generation that already finished."""
        if any(r["status"] in ("queued", "running")
               for r in self.store.runs(project_id)):
            return
        state = self.store.loop_state(project_id)
        if state["state"] in ("running", "baseline"):
            self._set_step_state(
                project_id, state="ready",
                message=f"gen {state['gen']}: all runs settled")

    def _costs(self, run: dict) -> dict[str, float]:
        """Cost guards: wall-clock seconds are the built-in cost."""
        started = run.get("started_at") or run["created_at"]
        return {"seconds": round(time.time() - started, 3)}

    def _baseline_done(self, project_id: str, node: dict) -> None:
        """Fix the noise margin once every baseline eval has settled.

        Called after each baseline eval; the last one to finish wins. A
        baseline eval that crashed or was killed still counts as settled, so
        the margin is fixed from whatever scores exist rather than leaving the
        project stuck in ``baseline`` forever.
        """
        evals = [r for r in self.store.runs(project_id, node["id"])
                 if r["kind"] == "eval"]
        if any(r["status"] in ("queued", "running") for r in evals):
            return
        scores = [r.score for r in self.results(project_id).rows()
                  if r.gen == 0 and r.score is not None]
        if len(scores) >= 2:
            margin = self._margin(project_id)
            self.store.update_node(node["id"], status="ran", score=scores[0],
                                   frozen=True)
            self.store.set_loop_state(project_id, margin=margin,
                                      frontier=[node["id"]])
            self._set_step_state(
                project_id, state="ready",
                message=f"baseline {scores[0]:g}; noise margin {margin:g}")
            self._emit(project_id, "baseline.ready", node_id=node["id"],
                       score=scores[0], margin=margin)
        else:
            self._set_step_state(
                project_id, state="error",
                message="baseline needs two scored runs to fix the noise "
                        "margin; re-run it")

    def delete_project(self, project_id: str) -> None:
        """Tear a project down for real: kill its runs (caller does this),
        collect worktrees + af/* branches it created in the user's repo,
        and remove its data dir under ~/.agentfork. The SQLite rows are
        deleted by the caller."""
        project = self.store.project(project_id)
        orch = self._orchestrators.pop(project_id, None)
        self._sandboxes.pop(project_id, None)
        if orch is not None:
            try:
                orch.stop_reaper()
                orch._release_registry_lock()
            except Exception:  # noqa: BLE001
                pass
        if project is not None:
            repo = project["repo_path"]
            for n in self.store.nodes(project_id):
                if n["worktree_path"]:
                    git.remove_worktree(repo, n["worktree_path"])
                # only branches agentfork itself created — the user's
                # baseline_branch is stored on the project, not the node
                if n["branch_name"].startswith("af/"):
                    git.delete_branch(repo, n["branch_name"])
            shutil.rmtree(self.project_dir(project_id), ignore_errors=True)

    # -- snapshots for the API -------------------------------------------------

    def tree(self, project_id: str) -> dict:
        loop = self.store.loop_state(project_id)
        loop["driven"] = project_id in self.driven
        return {"nodes": self.store.nodes(project_id),
                "loop": loop,
                "results_tsv": self.results(project_id).path.exists() and
                str(self.results(project_id).path) or None}

    def metrics(self, project_id: str) -> dict:
        project = self.store.project(project_id)
        if project is None:
            return {"orchestrator": {}, "kv": {}, "branches": 0}
        # built on demand (read-only) so the ledger still reports a project's
        # branches after a dashboard restart
        orch = self._orchestrator(project, reconcile=False)
        kv_stats = getattr(orch.kv, "stats", None)
        kv = kv_stats() if callable(kv_stats) else (kv_stats or {})
        if hasattr(kv, "__dict__"):
            kv = {**vars(kv), "dedup_ratio": kv.dedup_ratio}
        live = [b for b in orch.branches() if b.state == "live"]
        snap = orch.metrics_snapshot()
        # only the counters this product can move — the VMM-lifecycle ones
        # (execs, swept_dead, restarted) are always 0 under WorktreeSandbox
        orch_counters = {k: snap[k] for k in
                         ("forks", "kills", "kill_failures", "reconciles",
                          "reaped_expired") if k in snap}
        return {"orchestrator": orch_counters, "kv": kv,
                "branches": len(live),
                "worktrees": sum(
                    1 for n in self.store.nodes(project_id)
                    if n["worktree_path"]
                    and Path(n["worktree_path"]).exists())}
