"""Concurrency behavior of the components.

Threaded hammers assert the invariants the coarse per-component locks are
supposed to protect (unique branch IDs, consistent registry and cache state
after fork/extend/kill storms), and the registry-ownership tests pin the
one-orchestrator-per-registry-file rule enforced with ``flock``.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from agentfork import ForkOrchestrator, TreeKVCache
from agentfork.kill.reaper import BranchReaper

PREFIX = list(range(128))


def test_concurrent_forks_mint_unique_live_children():
    with ForkOrchestrator() as orch:
        orch.create_parent("root", tokens=PREFIX)

        with ThreadPoolExecutor(8) as pool:
            children = list(pool.map(
                lambda _: orch.fork("root", n=1)[0], range(64)))

        ids = [c.branch_id for c in children]
        assert len(set(ids)) == 64
        assert len(orch.branches()) == 65
        assert all(orch.alive(i) for i in ids)


def test_concurrent_fork_extend_kill_leaves_consistent_state():
    orch = ForkOrchestrator()
    orch.create_parent("root", tokens=PREFIX)

    def churn(i):
        child = orch.fork("root", n=1)[0]
        orch.extend(child.branch_id, [10_000 + i])
        orch.kill(child.branch_id)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(churn, range(200)))

    assert [b.branch_id for b in orch.branches()] == ["root"]
    orch.kill("root")
    assert orch.branches() == []
    assert orch.kv.trees == {}


def test_tree_cache_survives_concurrent_fork_and_kill():
    cache = TreeKVCache()
    cache.create_tree("t")
    cache.extend("t", PREFIX)

    def fork_extend_kill(i):
        tid = cache.fork_branch("t")
        cache.extend(tid.tree_id, [10_000 + i])
        cache.kill(tid.tree_id)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(fork_extend_kill, range(200)))

    assert list(cache.trees) == ["t"]


def test_second_orchestrator_on_same_registry_is_refused(tmp_path):
    registry = tmp_path / "registry.json"
    first = ForkOrchestrator(registry_path=registry)
    first.create_parent("root", tokens=PREFIX)

    with pytest.raises(RuntimeError, match="owned by another orchestrator"):
        ForkOrchestrator(registry_path=registry)

    first.close()  # releases ownership; a successor may now take over
    second = ForkOrchestrator(registry_path=registry)
    assert second.branches() == []
    second.close()


def test_closing_after_failed_kill_still_releases_ownership(tmp_path):
    class ExplodingSandbox:
        def spawn(self, branch_id, parent_id):
            pass

        def kill(self, branch_id):
            raise RuntimeError("kill failed")

        def alive(self, branch_id):
            return True

    registry = tmp_path / "registry.json"
    first = ForkOrchestrator(sandbox=ExplodingSandbox(), registry_path=registry)
    first.create_parent("root")

    with pytest.raises(RuntimeError, match="kill failed"):
        first.close()

    # ownership was released despite the error; the record survives for
    # the successor's reconcile() to retry
    second = ForkOrchestrator(registry_path=registry)
    assert [b.branch_id for b in second.branches()] == ["root"]
    second.close()


class SlowSandbox:
    """Parallel-safe sandbox whose spawn/kill take real wall-clock time, to
    observe whether multi-branch operations overlap."""

    parallel_lifecycle = True

    def __init__(self, delay=0.15):
        import threading
        self.delay = delay
        self.live = set()
        self._lock = threading.Lock()

    def spawn(self, branch_id, parent_id):
        import time
        time.sleep(self.delay)
        with self._lock:
            self.live.add(branch_id)

    def kill(self, branch_id):
        import time
        time.sleep(self.delay)
        with self._lock:
            self.live.discard(branch_id)

    def alive(self, branch_id):
        return branch_id in self.live


def test_fork_fans_out_when_sandbox_is_parallel_safe():
    import time

    sandbox = SlowSandbox(delay=0.15)
    with ForkOrchestrator(sandbox=sandbox) as orch:
        orch.create_parent("root", tokens=PREFIX)
        t0 = time.perf_counter()
        children = orch.fork("root", n=6)
        fork_s = time.perf_counter() - t0
        assert len(children) == 6
        assert all(orch.alive(c.branch_id) for c in children)
        # six 0.15s spawns serialized would take >=0.9s
        assert fork_s < 0.7

        t0 = time.perf_counter()
        receipts = orch.kill_losers(children[0].branch_id)
        kill_s = time.perf_counter() - t0
        assert len(receipts) == 5
        assert kill_s < 0.7


def test_parallel_fork_failure_rolls_back_only_failed_children():
    class HalfFailingSandbox(SlowSandbox):
        def spawn(self, branch_id, parent_id):
            if branch_id.endswith(("1", "3")):
                raise RuntimeError(f"spawn failed: {branch_id}")
            super().spawn(branch_id, parent_id)

    orch = ForkOrchestrator(sandbox=HalfFailingSandbox(delay=0.01))
    orch.create_parent("root", tokens=PREFIX)

    with pytest.raises(RuntimeError, match="spawn failed"):
        orch.fork("root", child_ids=[f"root/c{i}" for i in range(5)], n=5)

    survivors = {b.branch_id for b in orch.branches()}
    assert survivors == {"root", "root/c0", "root/c2", "root/c4"}
    assert all(orch.alive(b) for b in survivors)


def test_kill_is_noop_while_same_branch_kill_is_in_flight():
    import threading

    sandbox = SlowSandbox(delay=0.3)
    orch = ForkOrchestrator(sandbox=sandbox)
    orch.create_parent("root", tokens=PREFIX)

    slow = threading.Thread(target=orch.kill, args=("root",))
    slow.start()
    try:
        import time
        time.sleep(0.05)  # let the slow kill journal intent and enter I/O
        receipt = orch.kill("root")  # concurrent duplicate: no-op
        assert receipt.kv_freed_tokens == 0
    finally:
        slow.join()
    assert orch.branches() == []


def test_reaper_pdeathsig_flag():
    assert BranchReaper().pdeathsig is True
    assert BranchReaper(pdeathsig=False).pdeathsig is False


def test_reaper_sandbox_forwards_pdeathsig():
    from agentfork import ReaperSandbox

    assert ReaperSandbox(["true"]).reaper.pdeathsig is True
    assert ReaperSandbox(["true"], pdeathsig=False).reaper.pdeathsig is False
