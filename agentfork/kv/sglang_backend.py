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
``len(new_sequence) - hit`` arithmetic the patch computes internally for its
own accounting.

This adapter is unit-tested against a fake standing in for ``TreeRadixCache``
(see tests/test_sglang_backend.py); it has not been exercised against a live
SGLang engine.
"""

from __future__ import annotations


class SGLangKVBackend:
    def __init__(self, cache):
        self._cache = cache
        self._lengths: dict[str, int] = {}

    def create_tree(self, tree_id: str):
        branch = self._cache.create_agent_tree(tree_id)
        self._lengths[tree_id] = 0
        return branch

    def fork_branch(self, parent_id: str, child_id: str | None = None):
        branch = self._cache.fork_branch(parent_id, child_id)
        self._lengths[branch.branch_id] = self._lengths.get(parent_id, 0)
        return branch

    def kill(self, tree_id: str) -> int:
        self._lengths.pop(tree_id, None)
        return self._cache.kill_tree(tree_id)

    def extend(self, tree_id: str, tokens: list[int]) -> int:
        hit = self._cache.extend_tree(tree_id, tokens)
        new_total = self._lengths.get(tree_id, 0) + len(tokens)
        charged = new_total - hit
        self._lengths[tree_id] = new_total
        return charged
