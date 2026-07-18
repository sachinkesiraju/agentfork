import json
import sys
from dataclasses import FrozenInstanceError

import pytest

from agentfork.kill.reaper import BranchReaper
from agentfork.kv.tree_cache import TreeKVCache
from agentfork.orchestrator import (
    Branch,
    ForkOrchestrator,
    NullSandbox,
    ReaperSandbox,
)

PREFIX = list(range(1000))
SUFFIX = list(range(5000, 5080))


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class RecordingSandbox:
    """In-process fake: records spawn/kill order, idempotent kill."""

    def __init__(self, fail_spawn_for=()):
        self.live = set()
        self.events = []
        self.fail_spawn_for = set(fail_spawn_for)

    def spawn(self, branch_id, parent_id):
        if branch_id in self.fail_spawn_for:
            raise RuntimeError(f"spawn failed: {branch_id}")
        self.live.add(branch_id)
        self.events.append(("spawn", branch_id))

    def kill(self, branch_id):
        self.live.discard(branch_id)
        self.events.append(("kill", branch_id))

    def alive(self, branch_id):
        return branch_id in self.live


class ExternalSandbox:
    """Simulates external state that survives a supervisor crash: one file
    per live branch under ``root``."""

    def __init__(self, root):
        self.root = root
        self.root.mkdir(exist_ok=True)

    def spawn(self, branch_id, parent_id):
        (self.root / branch_id.replace("/", "_")).write_text("live")

    def kill(self, branch_id):
        path = self.root / branch_id.replace("/", "_")
        if path.exists():
            path.unlink()

    def alive(self, branch_id):
        return (self.root / branch_id.replace("/", "_")).exists()


def test_fork_and_kill_lifecycle_zero_ledger():
    kv = TreeKVCache()
    sandbox = RecordingSandbox()
    orch = ForkOrchestrator(kv=kv, sandbox=sandbox)
    orch.create_parent("parent", tokens=PREFIX)
    children = orch.fork("parent", n=10)

    assert len(children) == 10
    assert all(orch.alive(c.branch_id) for c in children)
    # CoW: 10 forks add no resident tokens until suffixes diverge
    assert kv.resident_tokens() == len(PREFIX)

    # the trailing token must differ per child or siblings share the whole
    # suffix and the first kill frees nothing (hash() is salted per process,
    # so hash-derived tokens collide in ~4% of runs)
    for i, c in enumerate(children):
        orch.extend(c.branch_id, SUFFIX + [i])
    receipts = [orch.kill(c.branch_id) for c in children]
    assert all(r.kv_freed_tokens > 0 for r in receipts)
    orch.kill("parent")

    assert kv.resident_tokens() == 0
    assert orch.branches() == []
    assert sandbox.live == set()


def test_fork_rolls_back_kv_and_registry_on_sandbox_failure(tmp_path):
    kv = TreeKVCache()
    sandbox = RecordingSandbox(fail_spawn_for={"parent/loser"})
    orch = ForkOrchestrator(kv=kv, sandbox=sandbox,
                            registry_path=tmp_path / "registry.json")
    orch.create_parent("parent", tokens=PREFIX)
    orch.fork("parent", child_ids=["parent/ok"])

    with pytest.raises(RuntimeError, match="spawn failed"):
        orch.fork("parent", child_ids=["parent/loser"])

    assert "parent/loser" not in kv.trees
    assert {b.branch_id for b in orch.branches()} == {"parent", "parent/ok"}
    on_disk = json.loads((tmp_path / "registry.json").read_text())
    assert {b["branch_id"] for b in on_disk["branches"]} == {"parent", "parent/ok"}
    # the surviving sibling is untouched
    assert orch.alive("parent/ok")


def test_create_parent_rolls_back_when_initial_extend_fails():
    kv = TreeKVCache(per_tree_budget=10)
    orch = ForkOrchestrator(kv=kv, sandbox=RecordingSandbox())
    with pytest.raises(MemoryError):
        orch.create_parent("parent", tokens=PREFIX)
    assert orch.branches() == []
    assert "parent" not in kv.trees


def test_kill_losers_keeps_winner_and_ancestors():
    orch = ForkOrchestrator(sandbox=RecordingSandbox())
    orch.create_parent("parent", tokens=PREFIX)
    children = orch.fork("parent", n=5)
    winner = children[2].branch_id
    grandchildren = orch.fork(winner, n=2)

    receipts = orch.kill_losers(grandchildren[0].branch_id)

    survivors = {b.branch_id for b in orch.branches()}
    assert survivors == {"parent", winner, grandchildren[0].branch_id}
    assert len(receipts) == 4 + 1  # 4 losing children + 1 losing grandchild


def test_lease_expiry_reaps_only_lapsed_branches():
    clock = FakeClock()
    orch = ForkOrchestrator(sandbox=RecordingSandbox(), clock=clock)
    orch.create_parent("parent", tokens=PREFIX)
    orch.fork("parent", n=2, lease_s=30)
    orch.fork("parent", n=1, lease_s=300)

    clock.now += 60
    reaped = orch.reap_expired()

    assert len(reaped) == 2
    assert len(orch.branches()) == 2  # parent (no lease) + long-lease child
    orch.renew_lease("parent/3", 10)
    clock.now += 20
    assert len(orch.reap_expired()) == 1


def test_reconcile_collects_crashed_supervisor_leftovers(tmp_path):
    registry = tmp_path / "registry.json"
    sandbox_root = tmp_path / "sandboxes"
    clock = FakeClock()

    first = ForkOrchestrator(sandbox=ExternalSandbox(sandbox_root),
                             registry_path=registry, clock=clock,
                             default_lease_s=100)
    first.create_parent("parent", tokens=PREFIX)
    first.fork("parent", n=3)
    # simulate a crash mid-fork: journal says forking, sandbox never spawned
    first._record(Branch("parent/ghost", "parent", "forking", clock.now, None))
    # a real crash ends the process, so the kernel drops its registry flock;
    # release it by hand since this "crash" stays in-process
    first._release_registry_lock()
    del first  # crash: no close(), external sandbox files remain

    assert len(list(sandbox_root.iterdir())) == 4

    clock.now += 200  # every lease has lapsed
    second = ForkOrchestrator(sandbox=ExternalSandbox(sandbox_root),
                              registry_path=registry, clock=clock)
    assert len(second.branches()) == 5
    receipts = second.reconcile()

    assert len(receipts) == 5
    assert second.branches() == []
    assert list(sandbox_root.iterdir()) == []
    assert json.loads(registry.read_text())["branches"] == []


def test_kill_is_idempotent_and_close_reaps_everything():
    sandbox = RecordingSandbox()
    with ForkOrchestrator(sandbox=sandbox) as orch:
        orch.create_parent("parent", tokens=PREFIX)
        orch.fork("parent", n=3)
        assert orch.kill("nonexistent").kv_freed_tokens == 0
    assert sandbox.live == set()


def test_fork_from_dead_parent_is_rejected():
    orch = ForkOrchestrator(sandbox=RecordingSandbox())
    orch.create_parent("parent")
    orch.kill("parent")
    with pytest.raises(KeyError, match="no live branch"):
        orch.fork("parent", n=1)


def test_null_sandbox_supports_kv_only_orchestration():
    orch = ForkOrchestrator(sandbox=NullSandbox())
    orch.create_parent("parent", tokens=PREFIX)
    children = orch.fork("parent", n=4)
    assert all(orch.alive(c.branch_id) for c in children)
    orch.close()
    assert orch.kv.resident_tokens() == 0


@pytest.mark.skipif(not BranchReaper.supported(),
                    reason="requires Linux pidfd support")
def test_reaper_sandbox_integration_kills_real_processes():
    argv = [sys.executable, "-c", "import time; time.sleep(60)"]
    sandbox = ReaperSandbox(argv)
    with ForkOrchestrator(sandbox=sandbox) as orch:
        orch.create_parent("parent", tokens=PREFIX)
        children = orch.fork("parent", n=3)
        assert all(orch.alive(c.branch_id) for c in children)
        receipt = orch.kill(children[0].branch_id)
        assert receipt.kv_freed_tokens == 0  # no unique suffix yet
        assert not orch.alive(children[0].branch_id)
    assert sandbox.reaper._branches == {}


class FlakyKillSandbox(RecordingSandbox):
    """Kill fails once per branch, then succeeds — a transient reap error."""

    def __init__(self):
        super().__init__()
        self.failed_once = set()

    def kill(self, branch_id):
        if branch_id in self.live and branch_id not in self.failed_once:
            self.failed_once.add(branch_id)
            raise RuntimeError(f"transient kill failure: {branch_id}")
        super().kill(branch_id)


def test_failed_kill_is_journaled_and_retried_by_reconcile(tmp_path):
    sandbox = FlakyKillSandbox()
    orch = ForkOrchestrator(sandbox=sandbox,
                            registry_path=tmp_path / "registry.json")
    orch.create_parent("parent", tokens=PREFIX)

    with pytest.raises(RuntimeError, match="transient kill failure"):
        orch.kill("parent")

    # intent survived the failure: the record is journaled as mid-kill
    assert orch.branches()[0].state == "killing"
    on_disk = json.loads((tmp_path / "registry.json").read_text())
    assert on_disk["branches"][0]["state"] == "killing"

    receipts = orch.reconcile()

    assert [r.branch_id for r in receipts] == ["parent"]
    assert orch.branches() == []
    assert sandbox.live == set()


class ExecSandbox(RecordingSandbox):
    """RecordingSandbox that also supports the optional exec channel."""

    def exec(self, branch_id, argv, timeout_s=None, stdin=None):
        self.events.append(("exec", branch_id, tuple(argv), timeout_s))
        return f"ran:{argv[0]}"

    def exec_detached(self, branch_id, argv):
        self.events.append(("exec_detached", branch_id, tuple(argv)))
        return ("detached", branch_id)


def test_exec_delegates_to_backends_that_support_it():
    sandbox = ExecSandbox()
    orch = ForkOrchestrator(sandbox=sandbox)
    orch.create_parent("parent", tokens=PREFIX)

    assert orch.exec("parent", ["echo", "hi"], timeout_s=5.0) == "ran:echo"
    assert sandbox.events[-1] == ("exec", "parent", ("echo", "hi"), 5.0)


def test_exec_requires_a_live_branch():
    orch = ForkOrchestrator(sandbox=ExecSandbox())
    orch.create_parent("parent")
    orch.kill("parent")
    with pytest.raises(KeyError, match="no live branch"):
        orch.exec("parent", ["true"])
    with pytest.raises(KeyError, match="no live branch"):
        orch.exec("never-existed", ["true"])


def test_exec_on_backend_without_exec_raises():
    orch = ForkOrchestrator(sandbox=RecordingSandbox())
    orch.create_parent("parent")
    with pytest.raises(RuntimeError, match="does not support exec"):
        orch.exec("parent", ["true"])
    with pytest.raises(RuntimeError, match="does not support exec_detached"):
        orch.exec_detached("parent", ["true"])


def test_exec_detached_delegates_to_backends_that_support_it():
    sandbox = ExecSandbox()
    orch = ForkOrchestrator(sandbox=sandbox)
    orch.create_parent("parent")

    assert orch.exec_detached("parent", ["srv"]) == ("detached", "parent")
    assert sandbox.events[-1] == ("exec_detached", "parent", ("srv",))
    with pytest.raises(KeyError, match="no live branch"):
        orch.exec_detached("ghost", ["srv"])


def test_closed_orchestrator_refuses_mutation(tmp_path):
    registry = tmp_path / "registry.json"
    first = ForkOrchestrator(sandbox=RecordingSandbox(), registry_path=registry)
    first.create_parent("parent", tokens=PREFIX)
    first.close()
    first.close()  # idempotent

    second = ForkOrchestrator(registry_path=registry)  # ownership transferred
    for call in (lambda: first.create_parent("x"),
                 lambda: first.fork("parent"),
                 lambda: first.extend("parent", [1]),
                 lambda: first.kill("parent"),
                 lambda: first.reconcile()):
        with pytest.raises(RuntimeError, match="closed"):
            call()
    assert first.branches() == []  # reads stay allowed
    second.close()


def test_auto_child_ids_continue_after_registry_reload(tmp_path):
    registry = tmp_path / "registry.json"
    kv = TreeKVCache()
    sandbox = RecordingSandbox()
    first = ForkOrchestrator(
        kv=kv, sandbox=sandbox, registry_path=registry)
    first.create_parent("parent")
    first.fork("parent", n=2)
    first._release_registry_lock()  # simulate kernel release after crash

    second = ForkOrchestrator(
        kv=kv, sandbox=sandbox, registry_path=registry)
    child = second.fork("parent")[0]

    assert child.branch_id == "parent/3"
    second.close()


def test_reconcile_collects_all_rows_loaded_from_previous_owner(tmp_path):
    registry = tmp_path / "registry.json"
    sandbox_root = tmp_path / "sandboxes"
    first = ForkOrchestrator(
        sandbox=ExternalSandbox(sandbox_root), registry_path=registry)
    first.create_parent("parent")
    first.fork("parent", n=2)
    first._release_registry_lock()

    second = ForkOrchestrator(
        sandbox=ExternalSandbox(sandbox_root), registry_path=registry)
    receipts = second.reconcile()

    assert {receipt.branch_id for receipt in receipts} == {
        "parent", "parent/1", "parent/2"}
    assert second.branches() == []
    assert list(sandbox_root.iterdir()) == []


def test_branch_snapshots_are_immutable():
    orch = ForkOrchestrator(sandbox=RecordingSandbox())
    branch = orch.create_parent("parent")

    with pytest.raises(FrozenInstanceError):
        branch.state = "killing"

    orch.close()


def test_persist_failure_after_spawn_rolls_back_both_backends(monkeypatch):
    kv = TreeKVCache()
    sandbox = RecordingSandbox()
    orch = ForkOrchestrator(kv=kv, sandbox=sandbox)
    original_persist = orch._persist

    def fail_live_transition():
        if any(branch.state == "live" for branch in orch._branches.values()):
            raise OSError("disk full")
        original_persist()

    monkeypatch.setattr(orch, "_persist", fail_live_transition)

    with pytest.raises(OSError, match="disk full"):
        orch.create_parent("parent")

    assert orch.branches() == []
    assert kv.trees == {}
    assert sandbox.live == set()


def test_reconcile_attempts_expired_branches_after_stuck_failure():
    clock = FakeClock()
    sandbox = FlakyKillSandbox()
    orch = ForkOrchestrator(sandbox=sandbox, clock=clock)
    orch.create_parent("stuck")
    orch.create_parent("expired", lease_s=1)
    with pytest.raises(RuntimeError):
        orch.kill("stuck")
    clock.now += 2

    with pytest.raises(RuntimeError):
        orch.reconcile()

    assert sandbox.failed_once == {"stuck", "expired"}
