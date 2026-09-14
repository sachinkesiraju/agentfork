"""The engine end to end: nodes are real branches, scores come from artifacts.

Uses the fake harness and a tiny git repo whose "eval" prints a score that
depends on the candidate file the harness writes, so a whole generation of the
map-reduce loop runs with no network and no model.
"""

import subprocess
import time

import pytest

from agentfork.app import git
from agentfork.app.engine import Engine, EngineError
from agentfork.app.harness import FakeHarness, Idea
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
            status = engine.store.node(nid)["status"]
            if status not in ("proposed", "implementing", "ready", "running"):
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
    assert engine._margin(p["id"]) == pytest.approx(0.0)
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


def test_worker_failure_crashes_the_node(engine, repo, monkeypatch):
    p = _project(engine, repo)
    root = engine.baseline(p["id"])
    _wait(engine, [root["id"]])

    def boom(self, worktree, idea, context):
        raise RuntimeError("harness exploded")

    monkeypatch.setattr(FakeHarness, "implement", boom)
    nodes = engine.fan_out(p["id"], root["id"], [Idea("bad", "x")])
    _wait(engine, [n["id"] for n in nodes])
    assert engine.store.node(nodes[0]["id"])["status"] == "crash"


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
