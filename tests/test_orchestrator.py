import json
import sys

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

    for c in children:
        orch.extend(c.branch_id, SUFFIX + [hash(c.branch_id) % 1000])
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
