"""Tree-keyed copy-on-write radix KV cache.

A CPU-side reference implementation of the cache semantics the SGLang patch
(`patches/`) adds to ``sglang.srt.mem_cache.radix_cache``:

- every cached prefix belongs to a *tree* (an agent process tree), identified
  by ``TreeId(tree_id, parent_id, child_seq)``;
- ``fork_branch(parent)`` creates a child tree that inherits the parent's
  prefix pages copy-on-write: no page is copied, refcounts are bumped, and the
  child pays zero prefill for the shared prefix;
- ``kill(tree_id)`` drops every reference the tree holds and frees pages whose
  refcount reaches zero — the engine-side half of the pidfd kill path;
- eviction respects pinned live trees and enforces a per-tree token budget.

Tokens are the unit of accounting in this reference model. The structure mirrors
SGLang's refcounted radix tree; token ids stand in for KV tensors so the
semantics are testable without a GPU.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass

from agentfork._locking import locked


@dataclass(frozen=True)
class TreeId:
    tree_id: str
    parent_id: str | None = None
    child_seq: int = 0
    namespace: str = ""


class _Node:
    __slots__ = ("tokens", "children", "parent", "ref", "owners", "last_access")

    def __init__(self, tokens: tuple[int, ...], parent: "_Node | None"):
        self.tokens = tokens
        self.children: dict[int, _Node] = {}
        self.parent = parent
        self.ref = 0                      # live-tree references (lock_ref)
        self.owners: set[str] = set()     # tree_ids referencing this node
        self.last_access = time.monotonic()


@dataclass
class CacheStats:
    resident_tokens: int = 0
    logical_tokens: int = 0     # what residency would be without CoW sharing
    prefill_tokens_charged: int = 0
    prefill_tokens_saved: int = 0
    evicted_tokens: int = 0

    @property
    def dedup_ratio(self) -> float:
        if self.resident_tokens == 0:
            return 1.0
        return self.logical_tokens / self.resident_tokens


class TreeKVCache:
    """Radix cache whose entries are keyed by agent-tree identity."""

    def __init__(self, capacity_tokens: int = 1_000_000,
                 per_tree_budget: int | None = None):
        if capacity_tokens < 0:
            raise ValueError("capacity_tokens must be nonnegative")
        if per_tree_budget is not None and per_tree_budget < 0:
            raise ValueError("per_tree_budget must be nonnegative")
        self.root = _Node((), None)
        self.capacity = capacity_tokens
        self.per_tree_budget = per_tree_budget
        self.trees: dict[str, TreeId] = {}
        self._tree_tokens: dict[str, list[int]] = {}   # full token seq per tree
        self._child_seq = itertools.count(1)
        self._lock = threading.RLock()
        self.stats = CacheStats()

    # -- tree lifecycle ----------------------------------------------------

    @locked
    def create_tree(self, tree_id: str) -> TreeId:
        if tree_id in self.trees:
            raise ValueError(f"tree exists: {tree_id}")
        tid = TreeId(tree_id, namespace=tree_id)
        self.trees[tree_id] = tid
        self._tree_tokens[tree_id] = []
        self.root.children[tree_id] = _Node((), self.root)
        return tid

    @locked
    def fork_branch(self, parent_id: str, child_id: str | None = None) -> TreeId:
        """CoW fork: child inherits the parent's cached prefix, zero copy."""
        if parent_id not in self.trees:
            raise KeyError(f"no such tree: {parent_id}")
        seq = next(self._child_seq)
        child_id = child_id or f"{parent_id}/{seq}"
        if child_id in self.trees:
            raise ValueError(f"tree exists: {child_id}")
        namespace = self.trees[parent_id].namespace
        tid = TreeId(child_id, parent_id=parent_id, child_seq=seq,
                     namespace=namespace)
        self.trees[child_id] = tid
        parent_tokens = list(self._tree_tokens[parent_id])
        self._tree_tokens[child_id] = list(parent_tokens)
        # bump refcounts along the parent's cached path — no data copied
        node = self.root.children[namespace]
        matched = 0
        while matched < len(parent_tokens):
            nxt = node.children.get(parent_tokens[matched])
            if nxt is None or parent_tokens[matched:matched + len(nxt.tokens)] != list(nxt.tokens):
                break
            nxt.ref += 1
            nxt.owners.add(child_id)
            matched += len(nxt.tokens)
            node = nxt
        self.stats.prefill_tokens_saved += matched
        self.stats.logical_tokens += len(parent_tokens)
        return tid

    @locked
    def kill(self, tree_id: str) -> int:
        """Drop the tree's references; free pages that hit refcount zero.

        Returns the number of tokens freed. O(path length) — this is the
        engine half of a <10 ms kill.
        """
        if tree_id not in self.trees:
            return 0
        freed = 0
        namespace = self.trees[tree_id].namespace
        node = self.root.children[namespace]
        tokens = self._tree_tokens.pop(tree_id)
        path: list[_Node] = []
        matched = 0
        while matched < len(tokens):
            nxt = node.children.get(tokens[matched])
            if nxt is None or tokens[matched:matched + len(nxt.tokens)] != list(nxt.tokens):
                break
            path.append(nxt)
            matched += len(nxt.tokens)
            node = nxt
        for n in reversed(path):
            n.owners.discard(tree_id)
            n.ref -= 1
            if n.ref <= 0 and not n.children:
                del n.parent.children[n.tokens[0]]
                freed += len(n.tokens)
                self.stats.resident_tokens -= len(n.tokens)
        self.stats.logical_tokens -= len(tokens)
        del self.trees[tree_id]
        if not any(t.namespace == namespace for t in self.trees.values()):
            del self.root.children[namespace]
        return freed

    # -- extend / lookup ----------------------------------------------------

    @locked
    def extend(self, tree_id: str, tokens: list[int]) -> int:
        """Append tokens to a tree (decode/prefill). Returns tokens charged
        (i.e. not already cached along this tree's path — the CoW miss)."""
        if tree_id not in self.trees:
            raise KeyError(f"no such tree: {tree_id}")
        namespace = self.trees[tree_id].namespace
        seq = self._tree_tokens[tree_id]
        full = seq + tokens
        node, matched = self._walk(full, namespace)
        charged = len(full) - matched
        if (self.per_tree_budget is not None
                and self._namespace_tokens(namespace) + charged
                > self.per_tree_budget):
            raise MemoryError(
                f"tree {namespace} exceeds budget {self.per_tree_budget}")
        # insert the uncached suffix as a new node chain owned by this tree
        pos = matched
        if pos < len(full):
            self._maybe_evict(len(full) - pos)
            new = _Node(tuple(full[pos:]), node)
            new.ref, new.owners = 1, {tree_id}
            node.children[full[pos]] = new
            self.stats.resident_tokens += len(new.tokens)
        # ensure this tree holds refs on the whole path (idempotent)
        self._add_refs(full, tree_id, namespace)
        self.stats.logical_tokens += len(tokens)
        self.stats.prefill_tokens_charged += charged
        self.stats.prefill_tokens_saved += len(tokens) - charged if charged < len(tokens) else 0
        self._tree_tokens[tree_id] = full
        return charged

    @locked
    def match_prefix(self, tokens: list[int]) -> int:
        return max((self._walk(tokens, namespace)[1]
                    for namespace in self.root.children), default=0)

    @locked
    def match_tree_prefix(self, tree_id: str, tokens: list[int]) -> int:
        if tree_id not in self.trees:
            raise KeyError(f"no such tree: {tree_id}")
        _, matched = self._walk(tokens, self.trees[tree_id].namespace)
        return matched

    @locked
    def resident_tokens(self) -> int:
        return self.stats.resident_tokens

    @locked
    def has_tree(self, tree_id: str) -> bool:
        return tree_id in self.trees

    # -- internals -----------------------------------------------------------

    def _walk(self, tokens: list[int], namespace: str) -> tuple[_Node, int]:
        node, matched = self.root.children[namespace], 0
        while matched < len(tokens):
            nxt = node.children.get(tokens[matched])
            if nxt is None:
                break
            n = len(nxt.tokens)
            if tokens[matched:matched + n] != list(nxt.tokens):
                # partial match within a node: split it (radix behavior)
                common = 0
                while (common < n and matched + common < len(tokens)
                       and nxt.tokens[common] == tokens[matched + common]):
                    common += 1
                if common == 0:
                    break
                self._split(nxt, common)
                continue
            nxt.last_access = time.monotonic()
            matched += n
            node = nxt
        return node, matched

    def _split(self, node: _Node, at: int) -> None:
        head = _Node(node.tokens[:at], node.parent)
        head.ref, head.owners = node.ref, set(node.owners)
        node.parent.children[node.tokens[0]] = head
        tail_tokens = node.tokens[at:]
        node.tokens = tail_tokens
        node.parent = head
        head.children[tail_tokens[0]] = node

    def _add_refs(self, tokens: list[int], tree_id: str, namespace: str) -> None:
        node, matched = self.root.children[namespace], 0
        while matched < len(tokens):
            nxt = node.children.get(tokens[matched])
            if nxt is None or tokens[matched:matched + len(nxt.tokens)] != list(nxt.tokens):
                break
            if tree_id not in nxt.owners:
                nxt.owners.add(tree_id)
                nxt.ref += 1
            matched += len(nxt.tokens)
            node = nxt

    def _namespace_tokens(self, namespace: str) -> int:
        root = self.root.children[namespace]
        stack = list(root.children.values())
        total = 0
        while stack:
            node = stack.pop()
            total += len(node.tokens)
            stack.extend(node.children.values())
        return total

    def _maybe_evict(self, need: int) -> None:
        if self.stats.resident_tokens + need <= self.capacity:
            return
        # evict unreferenced leaves LRU-first; pinned (ref>0) nodes are safe
        leaves = [n for n in self._all_nodes() if not n.children and n.ref <= 0]
        leaves.sort(key=lambda n: n.last_access)
        for leaf in leaves:
            if self.stats.resident_tokens + need <= self.capacity:
                return
            del leaf.parent.children[leaf.tokens[0]]
            self.stats.resident_tokens -= len(leaf.tokens)
            self.stats.evicted_tokens += len(leaf.tokens)
        if self.stats.resident_tokens + need > self.capacity:
            raise MemoryError(
                "capacity exhausted by pinned trees; kill a branch or raise "
                "capacity (per-tree budgets prevent unbounded pinning)")

    def _all_nodes(self):
        stack = [node for root in self.root.children.values()
                 for node in root.children.values()]
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            yield n
