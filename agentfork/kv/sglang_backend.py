"""Adapts a live SGLang ``TreeRadixCache`` to the ``KVBackend`` surface
``ForkOrchestrator`` expects.

Takes an already-constructed ``TreeRadixCache`` instance (built against a
running SGLang engine, see ``patches/0001-sglang-tree-radix-cache.patch``)
rather than importing sglang itself, so this module has no hard dependency on
an SGLang installation.

``extend_tree`` on the patch returns the cache-hit length, not the charged
(cache-miss) count that ``TreeKVCache.extend`` and ``ForkOrchestrator``
callers expect, so this adapter tracks each branch's token-sequence length
itself and derives the charged count locally, reproducing the same
``len(new_sequence) - max(hit, old_len)`` arithmetic the patch computes
internally for its own accounting. The ``max`` matters when part of a
branch's previously-cached prefix has since been evicted (the patch's own
``fork_branch`` anticipates this: "parent's node may have been partially
evicted; re-match"), so ``hit`` can fall below ``old_len``; charging
``new_len - hit`` in that case would double-charge for tokens the branch
already paid for.

This adapter is unit-tested against a fake standing in for ``TreeRadixCache``
(see tests/test_sglang_backend.py); it has not been exercised against a live
SGLang engine.
"""

from __future__ import annotations

import threading

from agentfork._locking import locked


class SGLangKVBackend:
    def __init__(self, cache):
        self._cache = cache
        self._lock = threading.RLock()
        self._lengths: dict[str, int] = {}

    @locked
    def create_tree(self, tree_id: str):
        branch = self._cache.create_agent_tree(tree_id)
        self._lengths[tree_id] = 0
        return branch

    @locked
    def fork_branch(self, parent_id: str, child_id: str | None = None):
        parent_len = self._lengths[parent_id]
        branch = self._cache.fork_branch(parent_id, child_id)
        self._lengths[branch.branch_id] = parent_len
        return branch

    @locked
    def kill(self, tree_id: str) -> int:
        self._lengths.pop(tree_id, None)
        return self._cache.kill_tree(tree_id)

    @locked
    def extend(self, tree_id: str, tokens: list[int]) -> int:
        old_len = self._lengths[tree_id]  # before extend_tree: never mutate the engine for an untracked branch
        hit = self._cache.extend_tree(tree_id, tokens)
        new_total = old_len + len(tokens)
        charged = new_total - max(hit, old_len)
        self._lengths[tree_id] = new_total
        return charged

    @locked
    def has_tree(self, tree_id: str) -> bool:
        return tree_id in self._lengths
