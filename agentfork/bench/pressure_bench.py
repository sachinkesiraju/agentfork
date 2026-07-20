"""Cache-pressure simulator: validate the pressure cost model against caches.

The pressure cost model (`agentfork.bench.cost_model.pressure_model`) predicts,
for an N-way fanout over a shared prefix under interleaved unrelated traffic,
how often a stock LRU radix cache keeps the parent prefix resident versus a
tree-pinned cache, and the resulting prefill-token compute ratio.

This module replays that workload against two *real* cache objects and reports
measured hit rates and prefill-tokens-charged, so the model can be checked:

  * pinned arm  -> ``agentfork.kv.tree_cache.TreeKVCache`` (the CPU reference
    implementation, with real capacity, eviction, and pinning). The parent
    tree stays alive, so its prefix is pinned and survives eviction; unrelated
    traffic is transient (freed on completion), matching a real engine where a
    finished unrelated request's exclusive KV is released while the pinned
    parent is protected.
  * stock arm   -> ``StockRadixCache`` below, a compact single-namespace radix
    cache with deterministic LRU eviction and no cross-request pinning, i.e.
    the behavior of stock SGLang RadixAttention / vLLM APC. A finished
    request's prefix stays cached until LRU evicts it.

Token ids stand in for KV tensors, exactly as in ``TreeKVCache``. The workload
matches the GPU harness in ``modal_gpu_validation.py``: prefill the parent,
then before each of N children inject ``U`` unrelated prefill tokens and run
the child (shared prefix + unique suffix).
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, field

from agentfork.bench.cost_model import PressureScenario, pressure_model
from agentfork.kv.tree_cache import TreeKVCache


# ---------------------------------------------------------------------------
# Stock LRU radix cache (no tree pinning) -- models stock RadixAttention.
# ---------------------------------------------------------------------------

class _RNode:
    __slots__ = ("tokens", "children", "parent", "last_access")

    def __init__(self, tokens: tuple[int, ...], parent: "_RNode | None"):
        self.tokens = tokens
        self.children: dict[int, _RNode] = {}
        self.parent = parent
        self.last_access = 0


class StockRadixCache:
    """Single-namespace radix cache with deterministic LRU eviction.

    No request pins anything after it completes: every cached node is
    eligible for LRU eviction. This is the stock prefix-caching baseline.
    ``last_access`` uses a monotonic counter (not wall-clock) so eviction
    order is deterministic and testable.
    """

    def __init__(self, capacity_tokens: int):
        if capacity_tokens <= 0:
            raise ValueError("capacity_tokens must be positive")
        self.capacity = capacity_tokens
        self.root = _RNode((), None)
        self.resident = 0
        self.prefill_charged = 0
        self._clock = itertools.count(1)

    def _tick(self) -> int:
        return next(self._clock)

    def match_prefix(self, tokens: list[int]) -> int:
        """Longest cached prefix of ``tokens`` (does not mutate LRU order)."""
        node, matched = self.root, 0
        while matched < len(tokens):
            nxt = node.children.get(tokens[matched])
            if nxt is None:
                break
            n = len(nxt.tokens)
            if tokens[matched:matched + n] != list(nxt.tokens):
                common = 0
                while (common < n and matched + common < len(tokens)
                       and nxt.tokens[common] == tokens[matched + common]):
                    common += 1
                matched += common
                break
            matched += n
            node = nxt
        return matched

    def run_request(self, tokens: list[int]) -> tuple[int, int]:
        """Serve a request for ``tokens``. Returns (matched_prefix, charged).

        Charges the uncached suffix, inserts it, and refreshes LRU order along
        the whole path. Evicts LRU-first when over capacity.
        """
        matched = self._insert_and_touch(tokens)
        charged = len(tokens) - matched
        self.prefill_charged += charged
        return matched, charged

    def _insert_and_touch(self, tokens: list[int]) -> int:
        node, matched = self.root, 0
        now = self._tick()
        while matched < len(tokens):
            nxt = node.children.get(tokens[matched])
            if nxt is None:
                break
            n = len(nxt.tokens)
            if tokens[matched:matched + n] != list(nxt.tokens):
                common = 0
                while (common < n and matched + common < len(tokens)
                       and nxt.tokens[common] == tokens[matched + common]):
                    common += 1
                if common == 0:
                    break
                self._split(nxt, common)
                continue
            nxt.last_access = now
            matched += n
            node = nxt
        if matched < len(tokens):
            self._evict(len(tokens) - matched)
            new = _RNode(tuple(tokens[matched:]), node)
            new.last_access = now
            node.children[tokens[matched]] = new
            self.resident += len(new.tokens)
        return matched

    def _split(self, node: _RNode, at: int) -> None:
        head = _RNode(node.tokens[:at], node.parent)
        head.last_access = node.last_access
        node.parent.children[node.tokens[0]] = head
        tail = node.tokens[at:]
        node.tokens = tail
        node.parent = head
        head.children[tail[0]] = node

    def _all_leaves(self):
        return [n for n in self._all_nodes() if not n.children]

    def _all_nodes(self):
        stack = list(self.root.children.values())
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            yield n

    def _evict(self, need: int) -> None:
        # evict LRU leaves first; removing a leaf may expose its parent as a
        # new leaf, so re-scan until we have room or nothing is evictable.
        while self.resident + need > self.capacity:
            leaves = self._all_leaves()
            if not leaves:
                break
            victim = min(leaves, key=lambda n: n.last_access)
            del victim.parent.children[victim.tokens[0]]
            self.resident -= len(victim.tokens)


# ---------------------------------------------------------------------------
# Workload replay
# ---------------------------------------------------------------------------

@dataclass
class Measurement:
    arm: str
    parent_hits: int
    n_children: int
    prefill_charged: int
    child_prefill_charged: int  # parent + children, excludes unrelated traffic

    @property
    def parent_hit_rate(self) -> float:
        return self.parent_hits / self.n_children


@dataclass
class _TokenGen:
    """Distinct token-id ranges so nothing collides accidentally."""
    _next: int = 0
    _ranges: dict = field(default_factory=dict)

    def block(self, name: str, size: int) -> list[int]:
        start = self._ranges.get(name)
        if start is None:
            start = self._next
            self._ranges[name] = start
            self._next += size
        return list(range(start, start + size))

    def fresh(self, size: int) -> list[int]:
        start = self._next
        self._next += size
        return list(range(start, start + size))


def run_stock(ps: PressureScenario) -> Measurement:
    gen = _TokenGen()
    cache = StockRadixCache(ps.capacity_tokens)
    prefix = gen.block("parent", ps.prefix_tokens)
    cache.run_request(prefix)  # cold parent prefill
    parent_charged_start = cache.prefill_charged
    child_charged = parent_charged_start
    hits = 0
    for i in range(ps.n_children):
        apply = ps.interleaved_tokens > 0 and (
            ps.pattern == "sustained" or i == 0)
        if apply:
            cache.run_request(gen.fresh(ps.interleaved_tokens))
        before = cache.prefill_charged
        matched = cache.match_prefix(prefix)
        if matched >= ps.prefix_tokens:
            hits += 1
        suffix = gen.block(f"suffix-{i}", ps.suffix_tokens)
        cache.run_request(prefix + suffix)
        child_charged += cache.prefill_charged - before
    return Measurement(
        arm="stock",
        parent_hits=hits,
        n_children=ps.n_children,
        prefill_charged=cache.prefill_charged,
        child_prefill_charged=child_charged,
    )


def run_pinned(ps: PressureScenario) -> Measurement:
    gen = _TokenGen()
    # capacity must hold the pinned parent + its live children; unrelated
    # traffic is transient. Size the cache generously for the pinned arm so a
    # legitimately pinned working set never raises -- pinning's whole point is
    # that the parent is protected, not that unrelated traffic never arrives.
    cache = TreeKVCache(capacity_tokens=max(
        ps.capacity_tokens,
        ps.prefix_tokens + ps.n_children * ps.suffix_tokens
        + ps.interleaved_tokens + 1))
    prefix = gen.block("parent", ps.prefix_tokens)
    cache.create_tree("parent")
    if prefix:
        cache.extend("parent", prefix)
    parent_charged = cache.stats.prefill_tokens_charged
    child_charged = parent_charged
    hits = 0
    for i in range(ps.n_children):
        apply = ps.interleaved_tokens > 0 and (
            ps.pattern == "sustained" or i == 0)
        if apply:
            # unrelated request: its own tree, freed on completion (kill).
            uname = f"noise-{i}"
            cache.create_tree(uname)
            cache.extend(uname, gen.fresh(ps.interleaved_tokens))
            cache.kill(uname)
        child = f"child-{i}"
        cache.fork_branch("parent", child)
        matched = cache.match_tree_prefix(child, prefix)
        if matched >= ps.prefix_tokens:
            hits += 1
        before = cache.stats.prefill_tokens_charged
        suffix = gen.block(f"suffix-{i}", ps.suffix_tokens)
        if suffix:
            cache.extend(child, suffix)
        child_charged += cache.stats.prefill_tokens_charged - before
    return Measurement(
        arm="pinned",
        parent_hits=hits,
        n_children=ps.n_children,
        prefill_charged=cache.stats.prefill_tokens_charged,
        child_prefill_charged=child_charged,
    )


def compare(ps: PressureScenario) -> dict:
    """Run both arms, the model, and report predicted vs measured."""
    predicted = pressure_model(ps)
    stock = run_stock(ps)
    pinned = run_pinned(ps)
    measured_ratio = stock.child_prefill_charged / pinned.child_prefill_charged
    return {
        "scenario": vars(ps),
        "model": {
            "stock_hit_rate": predicted["stock"]["parent_hit_rate"],
            "stock_prefill_charged": predicted["stock"]["prefill_charged"],
            "pinned_prefill_charged": predicted["pinned"]["prefill_charged"],
            "compute_ratio_prefill": predicted["compute_ratio_prefill"],
        },
        "measured": {
            "stock_hit_rate": round(stock.parent_hit_rate, 4),
            "pinned_hit_rate": round(pinned.parent_hit_rate, 4),
            "stock_child_prefill_charged": stock.child_prefill_charged,
            "pinned_child_prefill_charged": pinned.child_prefill_charged,
            "compute_ratio_prefill": round(measured_ratio, 4),
        },
    }


def sweep(prefix: int, n: int, suffix: int, capacity: int,
          u_values: list[int]) -> list[dict]:
    rows = []
    for u in u_values:
        ps = PressureScenario(
            prefix_tokens=prefix, n_children=n, suffix_tokens=suffix,
            interleaved_tokens=u, capacity_tokens=capacity)
        rows.append(compare(ps))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=int, default=2400)
    ap.add_argument("--children", type=int, default=10)
    ap.add_argument("--suffix", type=int, default=8)
    ap.add_argument("--capacity", type=int, default=32768)
    ap.add_argument("--interleaved", type=int, nargs="*",
                    default=[0, 8000, 24000, 48000, 96000],
                    help="one or more U values to sweep")
    a = ap.parse_args()
    print(json.dumps(
        sweep(a.prefix, a.children, a.suffix, a.capacity, a.interleaved),
        indent=2))
