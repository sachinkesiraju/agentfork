"""Unit tests for agentfork.kv.sglang_backend.SGLangKVBackend.

Exercises the adapter's translation logic (bookkeeping of per-branch token
lengths, deriving a charged/cache-miss count from the patch's cache-hit
return value) against a small in-process fake standing in for
``sglang.srt.mem_cache.tree_radix_cache.TreeRadixCache``. The radix-tree
internals themselves are already covered by the patch's own test file
(``patches/0001-sglang-tree-radix-cache.patch``); this file never imports
``sglang`` and never touches a real TreeRadixCache.
"""

from types import SimpleNamespace

from agentfork.kv.sglang_backend import SGLangKVBackend


class FakeTreeRadixCache:
    """Mimics just enough of the real patch's semantics to verify the
    adapter's translation logic, not the radix-tree internals.

    ``force_hit`` lets a test dictate exactly what ``extend_tree`` reports as
    the cache-hit length for the *next* call against a given branch, so both
    the all-new and fully-cached cases can be exercised deterministically.
    Without an override, ``extend_tree`` behaves as a cold/all-new extend:
    hit equals the branch's length before this call.
    """

    def __init__(self):
        self.sequences: dict[str, list[int]] = {}
        self.parents: dict[str, str | None] = {}
        self._seq = 0
        self.force_hit: dict[str, int] = {}

    def create_agent_tree(self, branch_id: str):
        self.sequences[branch_id] = []
        self.parents[branch_id] = None
        return SimpleNamespace(branch_id=branch_id)

    def fork_branch(self, parent_id: str, child_id: str | None = None):
        if child_id is None:
            self._seq += 1
            child_id = f"{parent_id}/{self._seq}"
        self.sequences[child_id] = list(self.sequences[parent_id])
        self.parents[child_id] = parent_id
        return SimpleNamespace(branch_id=child_id)

    def kill_tree(self, branch_id: str) -> int:
        tokens = self.sequences.pop(branch_id, [])
        self.parents.pop(branch_id, None)
        return len(tokens)

    def extend_tree(self, branch_id: str, tokens: list[int]) -> int:
        old = self.sequences[branch_id]
        hit = self.force_hit.pop(branch_id, None)
        if hit is None:
            hit = len(old)  # default: nothing new was cached
        self.sequences[branch_id] = old + list(tokens)
        return hit


def test_create_tree_calls_through_and_initializes_bookkeeping():
    cache = FakeTreeRadixCache()
    backend = SGLangKVBackend(cache)

    branch = backend.create_tree("root")

    assert branch.branch_id == "root"
    assert "root" in cache.sequences
    assert backend._lengths["root"] == 0

    # all-new tokens right after create: charged == full length, not 0
    charged = backend.extend("root", [1, 2, 3])
    assert charged == 3
    assert backend._lengths["root"] == 3


def test_fork_branch_propagates_parent_length_to_child():
    cache = FakeTreeRadixCache()
    backend = SGLangKVBackend(cache)
    backend.create_tree("p")
    backend.extend("p", [1, 2, 3, 4, 5])  # backend._lengths["p"] == 5
    assert backend._lengths["p"] == 5

    child = backend.fork_branch("p", "c")

    assert child.branch_id == "c"
    # inherited length, not reset to zero
    assert backend._lengths["c"] == 5

    # extend on the child after fork must charge relative to the inherited
    # length (5), not from zero: two brand-new tokens should charge 2.
    charged = backend.extend("c", [6, 7])
    assert charged == 2
    assert backend._lengths["c"] == 7


def test_fork_branch_auto_generated_child_id_also_inherits_length():
    cache = FakeTreeRadixCache()
    backend = SGLangKVBackend(cache)
    backend.create_tree("p")
    backend.extend("p", [1, 2, 3])

    child = backend.fork_branch("p")  # no child_id: adapter must use whatever the cache returns

    assert backend._lengths[child.branch_id] == 3


def test_kill_calls_through_and_removes_bookkeeping():
    cache = FakeTreeRadixCache()
    backend = SGLangKVBackend(cache)
    backend.create_tree("t")
    backend.extend("t", [1, 2, 3])

    freed = backend.kill("t")

    assert freed == 3  # FakeTreeRadixCache.kill_tree returns len(tokens)
    assert "t" not in backend._lengths
    assert "t" not in cache.sequences


def test_extend_charges_full_length_when_entirely_new():
    cache = FakeTreeRadixCache()
    backend = SGLangKVBackend(cache)
    backend.create_tree("t")

    # hit == old length (0): nothing new was cached, so charged == len(tokens)
    charged = backend.extend("t", [10, 20, 30, 40])

    assert charged == 4


def test_extend_charges_zero_when_fully_recached():
    cache = FakeTreeRadixCache()
    backend = SGLangKVBackend(cache)
    backend.create_tree("t")
    backend.extend("t", [1, 2, 3])  # old length now 3

    # simulate the whole new sequence (old + new) being cache-hit
    cache.force_hit["t"] = 3 + 2
    charged = backend.extend("t", [4, 5])

    assert charged == 0
    assert backend._lengths["t"] == 5
