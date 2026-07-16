import pytest

from agentfork.kv.tree_cache import TreeKVCache


def toks(s):
    return [ord(c) for c in s]


def test_fork_inherits_prefix_zero_copy():
    kv = TreeKVCache()
    kv.create_tree("parent")
    charged = kv.extend("parent", toks("SHARED-PREFIX" * 100))
    assert charged == 1300  # cold prefill
    before = kv.resident_tokens()
    kv.fork_branch("parent", "child-1")
    assert kv.resident_tokens() == before  # CoW: nothing copied
    # child extends: pays only its unique suffix
    charged = kv.extend("child-1", toks("unique-1"))
    assert charged == len("unique-1")


def test_ten_way_fanout_prefix_reuse_gate():
    """Gate G2: >=90% of shared prefix tokens saved across 10 children."""
    kv = TreeKVCache()
    kv.create_tree("p")
    shared = toks("S" * 32000)
    kv.extend("p", shared)
    for i in range(10):
        kv.fork_branch("p", f"c{i}")
        kv.extend(f"c{i}", toks(f"unique-{i}" * 100))
    s = kv.stats
    # every child inherited the full 32k shared prefix without re-prefill
    child_reuse = s.prefill_tokens_saved / (10 * 32000)
    assert child_reuse >= 0.99
    # vs a no-sharing baseline that prefills S+unique per child (parent's one
    # cold prefill is inherently paid in both worlds)
    no_sharing = 10 * (32000 + 800) + 32000
    assert s.prefill_tokens_charged / no_sharing < 0.12
    # residency ~= shared + sum(unique), not N*(shared+unique)
    assert kv.resident_tokens() < 32000 + 10 * 1000 + 100
    assert s.dedup_ratio > 5.0


def test_kill_frees_only_unshared_pages():
    kv = TreeKVCache()
    kv.create_tree("p")
    kv.extend("p", toks("SHARED" * 50))
    kv.fork_branch("p", "c")
    kv.extend("c", toks("child-only-suffix"))
    before = kv.resident_tokens()
    freed = kv.kill("c")
    assert freed == len("child-only-suffix")
    assert kv.resident_tokens() == before - freed
    # parent prefix still matchable
    assert kv.match_prefix(toks("SHARED" * 50)) == 300


def test_kill_last_owner_frees_whole_path():
    kv = TreeKVCache()
    kv.create_tree("p")
    kv.extend("p", toks("ONLY-TREE"))
    assert kv.kill("p") == len("ONLY-TREE")
    assert kv.resident_tokens() == 0
    assert kv.kill("p") == 0  # idempotent


def test_eviction_spares_pinned_trees():
    kv = TreeKVCache(capacity_tokens=1000)
    kv.create_tree("live")
    kv.extend("live", toks("L" * 400))
    kv.create_tree("dead")
    kv.extend("dead", toks("D" * 400))
    kv.kill("dead")  # now unreferenced -> evictable
    kv.create_tree("new")
    kv.extend("new", toks("N" * 500))  # forces eviction of dead pages
    assert kv.match_prefix(toks("L" * 400)) == 400  # pinned tree survived
    assert kv.stats.evicted_tokens >= 0


def test_pinned_capacity_exhaustion_raises_not_deadlocks():
    kv = TreeKVCache(capacity_tokens=500)
    kv.create_tree("a")
    kv.extend("a", toks("A" * 400))
    kv.create_tree("b")
    with pytest.raises(MemoryError):
        kv.extend("b", toks("B" * 400))
    kv.kill("a")
    assert kv.extend("b", toks("B" * 400)) == 400  # recoverable after kill


def test_per_tree_budget_enforced():
    kv = TreeKVCache(per_tree_budget=100)
    kv.create_tree("t")
    with pytest.raises(MemoryError):
        kv.extend("t", toks("X" * 200))


def test_refcount_no_leak_over_fork_kill_cycles():
    kv = TreeKVCache()
    kv.create_tree("p")
    kv.extend("p", toks("BASE" * 100))
    for cycle in range(50):
        kv.fork_branch("p", f"c{cycle}")
        kv.extend(f"c{cycle}", toks(f"u{cycle}" * 10))
        kv.kill(f"c{cycle}")
    kv.kill("p")
    assert kv.resident_tokens() == 0
    assert not kv.root.children
