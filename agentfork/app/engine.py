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

import logging
import threading
import time
from pathlib import Path

from agentfork.app import amr, git, runner
from agentfork.app.harness import HarnessError, Idea, build_harness
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
    worktree but keeps the branch — it is the node's evidence.
    """

    parallel_lifecycle = True

    def __init__(self, repo: str, base_dir: Path, baseline: str):
        self.repo = repo
        self.base_dir = Path(base_dir)
        self.baseline = baseline

    def _branch(self, branch_id: str) -> str:
        # orchestrator child ids are "parent/n"; a git ref cannot be both a
        # leaf and a directory, so flatten the slash
        return "af/" + branch_id.replace("/", "__")

    def _path(self, branch_id: str) -> Path:
        return self.base_dir / branch_id

    def spawn(self, branch_id: str, parent_id: str | None) -> None:
        start = self._branch(parent_id) if parent_id else self.baseline
        git.add_worktree(self.repo, self._path(branch_id),
                         self._branch(branch_id), start)

    def kill(self, branch_id: str) -> None:
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
        self._stop = threading.Event()
        self._poller = threading.Thread(target=self._poll_loop, daemon=True)
        self._poller.start()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._stop.set()
        self._poller.join(timeout=5)
        for orch in self._orchestrators.values():
            try:
                orch.close()
            except Exception:
                pass

    def project_dir(self, project_id: str) -> Path:
        d = self.home / "projects" / project_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def results(self, project_id: str) -> amr.Results:
        return amr.Results(self.project_dir(project_id) / "results.tsv")

    def _orchestrator(self, project: dict) -> ForkOrchestrator:
        orch = self._orchestrators.get(project["id"])
        if orch is None:
            base = self.project_dir(project["id"]) / _WORKTREES
            sandbox = WorktreeSandbox(project["repo_path"], base,
                                      project["baseline_branch"])
            orch = ForkOrchestrator(
                sandbox=sandbox,
                registry_path=self.project_dir(project["id"]) / "orch.journal")
            orch.reconcile()
            self._orchestrators[project["id"]] = orch
            self._sandboxes[project["id"]] = sandbox
        return orch

    def _emit(self, project_id: str, kind: str, **payload) -> None:
        self.store.add_event(project_id, kind, payload)

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
            branch_name=project["baseline_branch"], title="baseline",
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

    def propose(self, project_id: str, parent_node_id: str, k: int) -> list[Idea]:
        project = self.store.project(project_id)
        harness = build_harness(project["harness"])
        return harness.propose(self._context(project_id), k)

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
        children = orch.fork(parent_node_id, n=len(ideas))
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
        loop = self.store.loop_state(project_id)
        self.store.set_loop_state(project_id, gen=parent["gen"] + 1,
                                  state="running",
                                  frontier=loop["frontier"])
        harness = build_harness(project["harness"])
        for node, idea in zip(nodes, ideas):
            self.start_worker(project, node, harness, idea)
        return nodes

    # -- runs ---------------------------------------------------------------

    def start_worker(self, project: dict, node: dict, harness, idea: Idea):
        """A worker run is a detached run too: the harness applies the idea
        inside the node's worktree, then the node is committed and queued for
        eval. For CLI harnesses the work is a process; for in-process
        harnesses (fake/api) we run it in a helper thread."""
        run_dir = self.project_dir(project["id"]) / _RUNS / f"w-{node['id']}"
        run = self.store.create_run(project_id=project["id"],
                                    node_id=node["id"], kind="worker",
                                    command=f"harness:{harness.name}",
                                    run_dir=str(run_dir), status="running")
        ctx = self._context(project["id"])
        t = threading.Thread(target=self._do_worker,
                             args=(project, node, harness, idea, ctx, run["id"]),
                             daemon=True)
        t.start()
        return run

    def _do_worker(self, project, node, harness, idea, context, run_id):
        run_dir = Path(self.store.run(run_id)["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            with (run_dir / runner.LOG).open("w") as log:
                try:
                    summary = harness.implement(node["worktree_path"], idea,
                                                context)
                    log.write(summary + "\n")
                except HarnessError as exc:
                    log.write(f"worker failed: {exc}\n")
                    raise
            sha = git.commit_all(node["worktree_path"],
                                 f"agentfork: {idea.title}")
            self.store.update_run(run_id, status="done", ended_at=time.time())
            self.store.update_node(node["id"], status="ready",
                                   commit_sha=sha or node["commit_sha"])
            self._emit(project["id"], "node.ready", node_id=node["id"])
            self.start_eval(project["id"], node["id"])
        except Exception as exc:  # noqa: BLE001 — worker failure → node crash
            _log.exception("worker failed for %s", node["id"])
            self.store.update_run(run_id, status="failed", ended_at=time.time())
            self.store.update_node(node["id"], status="crash", frozen=True)
            self.results(project["id"]).log(
                node["gen"], node["id"], node["parent_id"] or "-", None,
                "crash", f"worker failed: {exc}",
                regions=node["regions"])
            self._emit(project["id"], "node.crashed", node_id=node["id"],
                       reason=str(exc)[:200])

    def start_eval(self, project_id: str, node_id: str,
                   kind: str = "eval", force: bool = False) -> dict:
        project = self.store.project(project_id)
        node = self.store.node(node_id)
        if node is None or node["worktree_path"] is None:
            raise EngineError(f"node {node_id} has no worktree")
        # Frozen means a run already answered the node; only an explicit
        # caller re-answer (``force``) or a holdout may re-run it.
        if node["frozen"] and not (force or kind == "holdout"):
            raise EngineError(f"node {node_id} is frozen — it already has a "
                              "score; pass force to re-answer it")
        params = project["params"]
        command = params["holdout_cmd"] if kind == "holdout" else params["eval_cmd"]
        if not command:
            raise EngineError(f"{kind} command is empty")
        run_dir = self.project_dir(project_id) / _RUNS / f"{kind}-{node_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        handle = runner.start(run_dir, command, node["worktree_path"],
                              env={"AGENTFORK_NODE": node_id,
                                   "AGENTFORK_RUN_DIR": str(run_dir)})
        self.store.update_node(node_id, status="running")
        run = self.store.create_run(project_id=project_id, node_id=node_id,
                                    kind=kind, command=command,
                                    run_dir=str(run_dir), status="running")
        self.store.update_run(run["id"], pid=handle.pid,
                              started_at=time.time())
        self._emit(project_id, "run.started", run_id=run["id"],
                   node_id=node_id, run_kind=kind, pid=handle.pid)
        return run

    def kill_run(self, run_id: str) -> dict:
        run = self.store.run(run_id)
        if run is None:
            raise EngineError(f"no such run: {run_id}")
        runner.kill(run["run_dir"])
        self.store.update_run(run_id, status="killed", ended_at=time.time())
        self._emit(run["project_id"], "run.killed", run_id=run_id)
        return self.store.run(run_id)

    # -- reduce ---------------------------------------------------------------

    def reduce(self, project_id: str, gen: int,
               margin: float | None = None) -> amr.Reduction:
        project = self.store.project(project_id)
        params = project["params"]
        if margin is None:
            margin = self._margin(project_id)
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
            self.store.set_loop_state(project_id, state="running", gen=gen,
                                      frontier=[s.commit for s in red.kept])
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
        return abs(scores[0] - scores[1])

    # -- poller ---------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:  # noqa: BLE001
                _log.exception("poll tick failed")
            self._stop.wait(0.5)

    def _poll_once(self) -> None:
        for run in self.store.active_runs():
            if run["kind"] == "worker":
                continue  # in-process; managed by its own thread
            run_dir = run["run_dir"]
            if runner.alive(run_dir):
                started = run.get("started_at") or run["created_at"]
                project = self.store.project(run["project_id"])
                timeout = project["params"]["timeout_s"]
                if timeout and time.time() - started > timeout:
                    runner.kill(run_dir)
                    self._finish_run(run, runner_exit=None, killed="timeout")
                continue
            # a signalled run never writes exit_code; a run that was killed
            # here one tick ago is finished again — distinguish by what the
            # run was doing when we last saw it
            exit_code = runner.exit_code(run_dir)
            was_killed = (exit_code is None and run["status"] == "running"
                          and (run.get("started_at") or 0) > 0)
            self._finish_run(run, runner_exit=exit_code,
                             killed="killed" if was_killed else None)

    def _finish_run(self, run: dict, runner_exit: int | None,
                    killed: str | None) -> None:
        project = self.store.project(run["project_id"])
        params = project["params"]
        node = self.store.node(run["node_id"])
        score = runner.extract_score(run["run_dir"], params["metric_grep"]) \
            if run["kind"] in ("eval", "holdout") else None
        status = ("timeout" if killed else
                  "done" if runner_exit == 0 else
                  "failed" if runner_exit is not None and runner_exit != 0
                  else "killed")
        self.store.update_run(run["id"], status=status, exit_code=runner_exit,
                              score=score, ended_at=time.time())
        self._emit(run["project_id"], "run.finished", run_id=run["id"],
                   node_id=run["node_id"], status=status,
                   exit_code=runner_exit, score=score)
        if run["kind"] in ("eval", "holdout") and node:
            if score is None or status != "done":
                self.store.update_node(node["id"], status="crash", frozen=True)
                self.results(run["project_id"]).log(
                    node["gen"], node["id"], node["parent_id"] or "-",
                    None, "crash", f"{run['kind']} exited {runner_exit}")
            else:
                parent = self.store.node(node["parent_id"]) \
                    if node["parent_id"] else None
                delta = (score - parent["score"]
                         if parent and parent["score"] is not None else None)
                self.store.update_node(node["id"], status="ran", score=score,
                                       delta=delta, frozen=True)
                self.results(run["project_id"]).log(
                    node["gen"], node["id"], node["parent_id"] or "-",
                    score, "ran", node["title"],
                    costs=self._costs(run), regions=node["regions"])
                if node["gen"] == 0:
                    self._baseline_done(run["project_id"], node)
                self._emit(run["project_id"], "node.scored",
                           node_id=node["id"], score=score, delta=delta)

    def _costs(self, run: dict) -> dict[str, float]:
        """Cost guards: wall-clock seconds are the built-in cost."""
        started = run.get("started_at") or run["created_at"]
        return {"seconds": round(time.time() - started, 3)}

    def _baseline_done(self, project_id: str, node: dict) -> None:
        scores = [r.score for r in self.results(project_id).rows()
                  if r.gen == 0 and r.score is not None]
        done = [r for r in self.store.runs(project_id, node["id"])
                if r["status"] in ("done", "failed", "killed", "timeout")]
        total = [r for r in self.store.runs(project_id, node["id"])
                 if r["kind"] == "eval"]
        if len(done) < len(total):
            return  # other baseline evals still running
        if len(scores) >= 2:
            margin = abs(scores[0] - scores[1])
            node_score = scores[0]
            self.store.update_node(node["id"], score=node_score)
            self.store.set_loop_state(project_id, state="ready", margin=margin,
                                      frontier=[node["id"]])
            self._emit(project_id, "baseline.ready", node_id=node["id"],
                       score=node_score, margin=margin)

    # -- snapshots for the API -------------------------------------------------

    def tree(self, project_id: str) -> dict:
        return {"nodes": self.store.nodes(project_id),
                "loop": self.store.loop_state(project_id),
                "results_tsv": self.results(project_id).path.exists() and
                str(self.results(project_id).path) or None}

    def metrics(self, project_id: str) -> dict:
        orch = self._orchestrators.get(project_id)
        if orch is None:
            return {"orchestrator": {}, "kv": {}}
        kv_stats = getattr(orch.kv, "stats", None)
        kv = kv_stats() if callable(kv_stats) else (kv_stats or {})
        if hasattr(kv, "__dict__"):
            kv = {**vars(kv), "dedup_ratio": kv.dedup_ratio}
        return {"orchestrator": orch.metrics_snapshot(), "kv": kv,
                "branches": len([b for b in orch.branches()
                                 if b.state == "live"])}
