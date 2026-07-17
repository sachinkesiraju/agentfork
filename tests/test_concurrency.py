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


def test_reaper_pdeathsig_flag():
    assert BranchReaper().pdeathsig is True
    assert BranchReaper(pdeathsig=False).pdeathsig is False
