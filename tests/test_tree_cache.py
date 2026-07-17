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


def test_ten_way_fanout_prefix_reuse():
    """>=90% of shared prefix tokens are saved across 10 children."""
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


def test_independent_trees_do_not_share_prefixes():
    kv = TreeKVCache()
    shared = toks("SAME-PREFIX")
    kv.create_tree("a")
    kv.create_tree("b")

    assert kv.extend("a", shared) == len(shared)
    assert kv.match_tree_prefix("b", shared) == 0
    assert kv.extend("b", shared) == len(shared)
    assert kv.resident_tokens() == 2 * len(shared)


def test_forked_branches_share_one_namespace():
    kv = TreeKVCache()
    shared = toks("SHARED")
    kv.create_tree("parent")
    kv.extend("parent", shared)
    child = kv.fork_branch("parent", "child")

    assert child.namespace == "parent"
    assert kv.match_tree_prefix("child", shared) == len(shared)
    assert kv.resident_tokens() == len(shared)


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
    assert kv.match_tree_prefix("p", toks("SHARED" * 50)) == 300


def test_kill_last_owner_frees_whole_path():
    kv = TreeKVCache()
    kv.create_tree("p")
    kv.extend("p", toks("ONLY-TREE"))
    assert kv.kill("p") == len("ONLY-TREE")
    assert kv.resident_tokens() == 0
    assert kv.kill("p") == 0  # idempotent


def test_kill_releases_capacity_without_disturbing_other_trees():
    kv = TreeKVCache(capacity_tokens=1000)
    kv.create_tree("live")
    kv.extend("live", toks("L" * 400))
    kv.create_tree("temporary")
    kv.extend("temporary", toks("T" * 400))

    assert kv.kill("temporary") == 400
    kv.create_tree("new")
    assert kv.extend("new", toks("N" * 500)) == 500
    assert kv.match_tree_prefix("live", toks("L" * 400)) == 400
    assert kv.resident_tokens() == 900


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


def test_per_tree_budget_covers_all_branches():
    kv = TreeKVCache(per_tree_budget=10)
    kv.create_tree("p")
    kv.extend("p", toks("BASE"))
    kv.fork_branch("p", "a")
    kv.fork_branch("p", "b")
    kv.extend("a", toks("AAA"))

    with pytest.raises(MemoryError):
        kv.extend("b", toks("BBBB"))
    assert kv.resident_tokens() == 7


def test_negative_limits_are_rejected():
    with pytest.raises(ValueError):
        TreeKVCache(capacity_tokens=-1)
    with pytest.raises(ValueError):
        TreeKVCache(per_tree_budget=-1)


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
