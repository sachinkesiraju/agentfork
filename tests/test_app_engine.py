"""The engine end to end: nodes are real branches, scores come from artifacts.

Uses the fake harness and a tiny git repo whose "eval" prints a score that
depends on the candidate file the harness writes, so a whole generation of the
map-reduce loop runs with no network and no model.
"""

import subprocess
import time
from pathlib import Path

import pytest

from agentfork.app import git
from agentfork.app.engine import Engine, EngineError
from agentfork.app.harness import Idea
from agentfork.app.store import Store

# eval: score = 1.0 unless the candidate marker names a better idea
EVAL = r"""#!/bin/sh
score=1.0
if [ -f .agentfork-candidate ]; then
  case "$(head -1 .agentfork-candidate)" in
    good) score=0.20 ;;
    okay) score=0.95 ;;
    slow) score=0.10; sleep 1 ;;
  esac
fi
echo "val_loss=$score"
"""


@pytest.fixture()
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    git.git(path, "init", "-q", "-b", "main")
    git.git(path, "config", "user.email", "t@example.com")
    git.git(path, "config", "user.name", "t")
    (path / "eval.sh").write_text(EVAL)
    (path / "eval.sh").chmod(0o755)
    git.git(path, "add", "-A")
    git.git(path, "commit", "-qm", "init")
    return path


@pytest.fixture()
def engine(tmp_path):
    store = Store(tmp_path / "home" / "agentfork.db")
    eng = Engine(store, home=tmp_path / "home")
    yield eng
    eng.close()
    store.close()


def _project(engine, repo, **params):
    return engine.create_project(
        "p", str(repo), "main", "fake",
        {"eval_cmd": "./eval.sh", "metric_grep": r"val_loss=([0-9.]+)",
         "minimize": True, "k": 2, "b": 1, "baseline_runs": 2,
         "timeout_s": 60, **params})


def _wait(engine, node_ids, timeout=30.0):
    deadline = time.monotonic() + timeout
    pending = set(node_ids)
    while pending and time.monotonic() < deadline:
        for nid in list(pending):
            node = engine.store.node(nid)
            runs = engine.store.runs(node["project_id"], node_id=nid)
            in_flight = any(r["status"] in ("queued", "running") for r in runs)
            if not in_flight and node["status"] not in (
                    "proposed", "implementing", "ready", "running"):
                pending.discard(nid)
        time.sleep(0.05)
    assert not pending, f"nodes never settled: {pending}"


def test_create_project_requires_a_git_repo(engine, tmp_path):
    with pytest.raises(EngineError):
        engine.create_project("x", str(tmp_path), "main", "fake",
                              {"eval_cmd": "true", "metric_grep": "x"})


def test_create_project_requires_the_eval_contract(engine, repo):
    with pytest.raises(EngineError):
        engine.create_project("p", str(repo), "main", "fake", {})
    with pytest.raises(EngineError):
        engine.create_project("p", str(repo), "main", "fake",
                              {"eval_cmd": "./eval.sh"})


def test_create_project_requires_an_existing_baseline(engine, repo):
    with pytest.raises(EngineError):
        engine.create_project("p", str(repo), "nope", "fake",
                              {"eval_cmd": "./eval.sh", "metric_grep": "x"})


def test_create_project_gitignores_agentfork_state(engine, repo):
    _project(engine, repo)
    assert "results.tsv" in (repo / ".gitignore").read_text()


def test_baseline_fixes_the_noise_margin(engine, repo):
    p = _project(engine, repo)
    node = engine.baseline(p["id"])
    _wait(engine, [node["id"]])
    assert engine.store.node(node["id"])["score"] == pytest.approx(1.0)
    assert engine._margin(p["id"]) == pytest.approx(1e-9)  # floored, never exactly 0
    loop = engine.store.loop_state(p["id"])
    assert loop["state"] == "ready" and loop["frontier"] == [node["id"]]
    rows = engine.results(p["id"]).rows()
    assert len(rows) == 2 and all(r.gen == 0 for r in rows)


def test_baseline_is_not_repeatable(engine, repo):
    p = _project(engine, repo)
    engine.baseline(p["id"])
    with pytest.raises(EngineError):
        engine.baseline(p["id"])


def test_margin_needs_both_baseline_runs(engine, repo):
    p = _project(engine, repo, baseline_runs=1)
    engine.baseline(p["id"])
    with pytest.raises(EngineError):
        engine._margin(p["id"])


def test_fan_out_creates_worktrees_branches_and_scores(engine, repo):
    p = _project(engine, repo)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    ideas = [Idea("good", "the winning change"), Idea("okay", "a small win")]
    nodes = engine.fan_out(p["id"], root["id"], ideas)
    assert len(nodes) == 2
    for n in nodes:
        assert git.branch_exists(repo, n["branch_name"])
    _wait(engine, [n["id"] for n in nodes])
    scored = {engine.store.node(n["id"])["title"]:
              engine.store.node(n["id"])["score"] for n in nodes}
    assert scored["good"] == pytest.approx(0.20)
    assert scored["okay"] == pytest.approx(0.95)
    # every candidate is committed on its own branch
    for n in nodes:
        assert engine.store.node(n["id"])["commit_sha"]


def test_answered_nodes_freeze(engine, repo):
    p = _project(engine, repo)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    nodes = engine.fan_out(p["id"], root["id"], [Idea("good", "x")])
    _wait(engine, [n["id"] for n in nodes])
    node = engine.store.node(nodes[0]["id"])
    assert node["frozen"]
    # a scored node can't be silently re-answered, but it CAN be descended
    # from — descending from a winner is the search's whole point
    with pytest.raises(EngineError):
        engine.start_eval(p["id"], nodes[0]["id"])
    engine.start_eval(p["id"], nodes[0]["id"], force=True)
    _wait(engine, [nodes[0]["id"]])
    children = engine.fan_out(p["id"], nodes[0]["id"], [Idea("gen2", "y")])
    assert children[0]["gen"] == 2
    _wait(engine, [c["id"] for c in children])


def test_reduce_keeps_the_winner_and_reaps_the_losers(engine, repo):
    p = _project(engine, repo)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    nodes = engine.fan_out(p["id"], root["id"],
                           [Idea("good", "winner"), Idea("okay", "loser")])
    _wait(engine, [n["id"] for n in nodes])
    red = engine.reduce(p["id"], 1, margin=0.1)
    kept = [s.commit for s in red.kept]
    assert len(kept) == 1
    winner = engine.store.node(kept[0])
    loser = next(n for n in nodes if n["id"] != kept[0])
    assert winner["title"] == "good"
    assert winner["status"] == "kept"
    assert engine.store.node(loser["id"])["status"] == "discarded"
    # the loser's branch survives as evidence; its worktree is reaped
    assert git.branch_exists(repo, loser["branch_name"])
    assert not engine._sandboxes[p["id"]].worktree(loser["id"])
    assert engine.store.loop_state(p["id"])["frontier"] == [winner["id"]]


def test_reduce_writes_a_law_on_stall(engine, repo):
    p = _project(engine, repo)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    nodes = engine.fan_out(p["id"], root["id"], [Idea("okay", "barely")])
    _wait(engine, [n["id"] for n in nodes])
    red = engine.reduce(p["id"], 1, margin=0.5)
    assert red.stall and red.kept == []
    laws = engine.store.laws(p["id"])
    assert laws and "STALL" in laws[0]["text"]
    assert any(r.status == "note" for r in engine.results(p["id"]).rows())


def test_crashing_eval_marks_the_node_and_logs_a_crash(engine, repo):
    p = _project(engine, repo, eval_cmd="exit 9")
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    node = engine.store.node(root["id"])
    assert node["status"] == "crash"
    assert any(r.status == "crash" for r in engine.results(p["id"]).rows())


def test_eval_timeout_is_killed_and_recorded(engine, repo):
    p = _project(engine, repo, eval_cmd="sleep 30", timeout_s=1)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]], timeout=30)
    runs = engine.store.runs(p["id"])
    assert runs and all(r["status"] == "timeout" for r in runs)
    assert engine.store.node(root["id"])["status"] == "crash"


def test_worker_failure_crashes_the_node(engine, repo):
    p = _project(engine, repo)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    # "fail" is the fake harness's scripted failure — the worker is a real
    # subprocess, so its crash comes back via the run's exit code
    nodes = engine.fan_out(p["id"], root["id"], [Idea("fail", "x")])
    _wait(engine, [n["id"] for n in nodes])
    assert engine.store.node(nodes[0]["id"])["status"] == "crash"
    runs = [r for r in engine.store.runs(p["id"]) if r["kind"] == "worker"]
    assert runs and runs[0]["status"] == "failed"
    assert runs[0]["pid"]  # workers are real processes, killable like evals


def test_cost_guard_rejects_a_slow_winner(engine, repo):
    p = _project(engine, repo, cost_guards={"seconds": 0.2})
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    nodes = engine.fan_out(p["id"], root["id"], [Idea("slow", "fast but slow")])
    _wait(engine, [n["id"] for n in nodes], timeout=40)
    red = engine.reduce(p["id"], 1, margin=0.05)
    assert red.kept == []
    assert red.cost_failures and "seconds" in red.cost_failures[0][1]
    assert engine.store.node(nodes[0]["id"])["status"] == "cost_fail"


def test_kv_ledger_reflects_forked_branches(engine, repo):
    p = _project(engine, repo)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    before = engine.metrics(p["id"])
    nodes = engine.fan_out(p["id"], root["id"],
                           [Idea("good", "a"), Idea("okay", "b")])
    _wait(engine, [n["id"] for n in nodes])
    after = engine.metrics(p["id"])
    assert after["orchestrator"]["forks"] == before["orchestrator"]["forks"] + 2
    assert after["branches"] == 3
    assert after["kv"]["dedup_ratio"] >= 1.0
    engine.reduce(p["id"], 1, margin=0.1)
    assert engine.metrics(p["id"])["orchestrator"]["kills"] >= 1


def test_context_includes_history_and_laws(engine, repo):
    p = _project(engine, repo)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    engine.store.add_law(p["id"], 1, "LAW: do not touch the tokenizer")
    ctx = engine._context(p["id"])
    assert "./eval.sh" in ctx and "minimize" in ctx
    assert "Experiment history" in ctx
    assert "do not touch the tokenizer" in ctx


def test_descend_uses_the_harness_to_propose(engine, repo):
    p = _project(engine, repo)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    nodes = engine.descend(p["id"], root["id"], 2)
    assert len(nodes) == 2
    assert all(n["gen"] == 1 for n in nodes)
    _wait(engine, [n["id"] for n in nodes])


def test_holdout_requires_a_command(engine, repo):
    p = _project(engine, repo)
    root = engine.baseline(p["id"])
    with pytest.raises(EngineError):
        engine.start_eval(p["id"], root["id"], kind="holdout")


def test_holdout_may_reanswer_a_frozen_node(engine, repo):
    p = _project(engine, repo, holdout_cmd="./eval.sh")
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    assert engine.store.node(root["id"])["frozen"]
    run = engine.start_eval(p["id"], root["id"], kind="holdout")
    assert run["kind"] == "holdout"


def test_kill_run_marks_it_killed(engine, repo):
    p = _project(engine, repo, eval_cmd="sleep 30", timeout_s=300)
    engine.baseline(p["id"])
    runs = engine.store.runs(p["id"])
    engine.kill_run(runs[0]["id"])
    assert engine.store.run(runs[0]["id"])["status"] == "killed"


def test_worktrees_are_isolated_between_candidates(engine, repo):
    p = _project(engine, repo)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    nodes = engine.fan_out(p["id"], root["id"],
                           [Idea("good", "a"), Idea("okay", "b")])
    markers = {
        (engine.store.node(n["id"])["worktree_path"] + "/.agentfork-candidate")
        for n in nodes}
    assert len(markers) == 2
    _wait(engine, [n["id"] for n in nodes])
    heads = {subprocess.run(["git", "-C", str(repo), "rev-parse",
                             engine.store.node(n["id"])["branch_name"]],
                            capture_output=True, text=True).stdout.strip()
             for n in nodes}
    assert len(heads) == 2


def test_each_run_gets_its_own_run_directory(engine, repo):
    """Two runs of one node must not share a log, pid or exit_code file."""
    p = _project(engine, repo, eval_cmd="sleep 30", timeout_s=300)
    engine.baseline(p["id"])
    runs = engine.store.runs(p["id"])
    assert len(runs) == 2
    assert len({r["run_dir"] for r in runs}) == 2
    for run in runs:
        engine.kill_run(run["id"])


def test_killing_one_run_leaves_its_sibling_alone(engine, repo):
    p = _project(engine, repo, eval_cmd="sleep 30", timeout_s=300)
    engine.baseline(p["id"])
    first, second = engine.store.runs(p["id"])
    engine.kill_run(first["id"])
    assert engine.store.run(first["id"])["status"] == "killed"
    assert engine.store.run(second["id"])["status"] == "running"
    engine.kill_run(second["id"])


def test_a_killed_run_settles_its_node(engine, repo):
    p = _project(engine, repo)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    nodes = engine.fan_out(p["id"], root["id"], [Idea("good", "a")])
    _wait(engine, [n["id"] for n in nodes])
    node = nodes[0]
    run = engine.start_eval(p["id"], node["id"], force=True)
    engine.kill_run(run["id"])
    assert engine.store.run(run["id"])["status"] == "killed"
    assert engine.store.node(node["id"])["status"] != "running"


def test_baseline_with_one_unscored_run_does_not_stay_in_baseline(engine, repo):
    p = _project(engine, repo, eval_cmd="sleep 30", timeout_s=300)
    engine.baseline(p["id"])
    for run in engine.store.runs(p["id"]):
        engine.kill_run(run["id"])
    assert engine.store.loop_state(p["id"])["state"] == "error"


def test_manual_reduce_leaves_the_loop_idle_not_running(engine, repo):
    p = _project(engine, repo)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    nodes = engine.fan_out(p["id"], root["id"],
                           [Idea("good", "a"), Idea("okay", "b")])
    _wait(engine, [n["id"] for n in nodes])
    engine.reduce(p["id"], 1)
    assert engine.store.loop_state(p["id"])["state"] in ("ready", "stalled")


def test_ledger_reports_the_tree_after_a_restart(engine, repo, tmp_path):
    """A restarted dashboard cannot adopt the previous process's branches, but
    the ledger must still account for the persisted tree instead of zeroes."""
    p = _project(engine, repo)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    engine.close()
    restarted = Engine(engine.store, home=tmp_path / "home")
    try:
        metrics = restarted.metrics(p["id"])
        assert restarted.store.nodes(p["id"])          # tree persisted
        assert "reconciles" in metrics["orchestrator"]
        assert "worktrees" in metrics
        assert "dedup_ratio" in metrics["kv"]
    finally:
        restarted.close()


def test_killing_every_candidate_leaves_the_loop_idle(engine, repo):
    p = _project(engine, repo, eval_cmd="sleep 5", timeout_s=300)
    root = engine.baseline(p["id"])
    for run in engine.store.runs(p["id"]):
        engine.kill_run(run["id"])
    engine.store.update_node(root["id"], status="ran", score=1.0, frozen=True)
    nodes = engine.fan_out(p["id"], root["id"],
                           [Idea("good", "a"), Idea("okay", "b")])
    _wait(engine, [n["id"] for n in nodes])
    for run in engine.store.runs(p["id"]):
        if run["status"] in ("queued", "running"):
            engine.kill_run(run["id"])
    assert engine.store.loop_state(p["id"])["state"] != "running"


def test_killing_a_settled_run_is_a_noop(engine, repo):
    """kill on an already-finished run must not rewrite its terminal state."""
    p = _project(engine, repo)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    run = engine.store.runs(p["id"])[0]
    assert run["status"] == "done"
    again = engine.kill_run(run["id"])
    assert again["status"] == "done"
    assert engine.store.node(root["id"])["status"] != "crash"


def test_reduce_refuses_while_candidates_are_in_flight(engine, repo):
    """A mid-generation reduce would kill worktrees under running evals and
    then mark the corpses discarded — refuse instead."""
    p = _project(engine, repo, eval_cmd="sleep 5; ./eval.sh", timeout_s=300)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]], timeout=30)
    nodes = engine.fan_out(p["id"], root["id"],
                           [Idea("good", "a"), Idea("okay", "b")])
    with pytest.raises(EngineError, match="in flight"):
        engine.reduce(p["id"], 1)
    _wait(engine, [n["id"] for n in nodes], timeout=30)
    engine.reduce(p["id"], 1)  # once settled, it goes through


def test_eval_slots_serialize_the_baseline_pair(engine, repo):
    """eval_slots is the resource budget: with one slot the second baseline
    eval stays queued until the first finishes."""
    p = _project(engine, repo, eval_slots=1, eval_cmd="sleep 0.4; ./eval.sh")
    root = engine.baseline(p["id"])
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        runs = engine.store.runs(p["id"])
        if runs and all(r["status"] not in ("queued", "running")
                        for r in runs):
            break
        time.sleep(0.1)
    assert engine.store.node(root["id"])["status"] != "running"
    runs = sorted(engine.store.runs(p["id"]), key=lambda r: r["started_at"])
    assert len(runs) == 2 and all(r["status"] == "done" for r in runs)
    first, second = runs
    assert second["started_at"] >= first["ended_at"] - 0.01


def test_close_leaves_a_running_evals_worktree_alone(engine, repo):
    """Shutting the dashboard down must not delete the checkout under an eval
    that is still running detached — the new process can adopt or collect it."""
    p = _project(engine, repo, eval_cmd="sleep 30", timeout_s=300)
    engine.baseline(p["id"])
    node = engine.store.nodes(p["id"])[0]
    worktree = Path(node["worktree_path"])
    engine.close()
    assert worktree.exists()
    for run in engine.store.runs(p["id"]):
        subprocess.run(["kill", "-9", str(run["pid"])], check=False)


def test_a_restarted_dashboard_can_still_fan_out(engine, repo, tmp_path):
    """The prior process's branch rows are journaled but its sandboxes and KV
    trees are gone; fanning out re-adopts the parent (worktree reattached,
    context re-tokenized) instead of failing on a dead branch."""
    p = _project(engine, repo)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    engine.close()
    restarted = Engine(engine.store, home=tmp_path / "home")
    try:
        nodes = restarted.fan_out(p["id"], root["id"], [Idea("good", "a")])
        assert nodes[0]["gen"] == 1
        assert Path(nodes[0]["worktree_path"]).exists()
        _wait(restarted, [n["id"] for n in nodes])
        assert restarted.store.node(nodes[0]["id"])["score"] == 0.20
    finally:
        restarted.close()


def test_child_ids_never_collide_with_prior_nodes(engine, repo, tmp_path):
    """Node ids are the tsv's commits: the orchestrator's next-id counter
    only knows live branches, so children must be named from the store —
    otherwise a discarded (or pre-restart) child's id gets reused and its
    row collides."""
    p = _project(engine, repo)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    first = engine.fan_out(p["id"], root["id"], [Idea("good", "a")])
    _wait(engine, [n["id"] for n in first])
    engine.close()
    restarted = Engine(engine.store, home=tmp_path / "home")
    try:
        second = restarted.fan_out(p["id"], root["id"], [Idea("okay", "b")])
        assert second[0]["id"] != first[0]["id"]
        _wait(restarted, [n["id"] for n in second])
    finally:
        restarted.close()


def test_holdout_does_not_contaminate_the_record(engine, repo):
    """A holdout is revalidation, not an answer: the node's score/status
    don't move, and the tsv records a note — never a 'ran' row."""
    p = _project(engine, repo, holdout_cmd="./eval.sh")
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    nodes = engine.fan_out(p["id"], root["id"], [Idea("good", "a")])
    _wait(engine, [n["id"] for n in nodes])
    node = engine.store.node(nodes[0]["id"])
    run = engine.start_eval(p["id"], node["id"], kind="holdout")
    deadline = time.monotonic() + 30
    while engine.store.run(run["id"])["status"] in ("queued", "running"):
        assert time.monotonic() < deadline, "holdout never finished"
        time.sleep(0.05)
    run = engine.store.run(run["id"])
    assert run["status"] == "done" and run["score"] == pytest.approx(0.20)
    after = engine.store.node(node["id"])
    assert after["score"] == node["score"] and after["status"] == node["status"]
    rows = [r for r in engine.results(p["id"]).rows()
            if r.commit == node["id"]]
    assert [r.status for r in rows][-1] == "note"


def test_score_is_grepped_from_the_end_of_the_log(engine, repo, tmp_path):
    """A long eval that scrolls the metric past the reader window still
    reports its final score — extract reads the tail, not the head."""
    from agentfork.app import runner
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / runner.LOG).open("wb") as f:
        f.write(b"val_loss=9.99\n")
        f.write(b"x" * 9_000_000)
        f.write(b"\nval_loss=0.5\n")
    assert runner.extract_score(run_dir, r"val_loss=([0-9.]+)") == 0.5


def test_a_node_whose_worktree_is_gone_is_checked_out_again(engine, repo):
    """The node's evidence is its branch and commit; the checkout is
    disposable and comes back when a run needs it."""
    p = _project(engine, repo)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])
    nodes = engine.fan_out(p["id"], root["id"], [Idea("good", "a")])
    _wait(engine, [n["id"] for n in nodes])
    node = engine.store.node(nodes[0]["id"])
    git.remove_worktree(repo, node["worktree_path"])
    assert not Path(node["worktree_path"]).exists()
    engine.start_eval(p["id"], node["id"], force=True)
    assert Path(node["worktree_path"]).exists()
    _wait(engine, [node["id"]])
    assert engine.store.node(node["id"])["score"] == 0.20
