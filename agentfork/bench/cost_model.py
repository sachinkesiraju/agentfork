"""Fanout cost model: tree-keyed KV fork vs composed baselines.

Baselines compared for an N-way sibling fanout over a shared prefix of
``prefix_tokens`` with ``suffix_tokens`` unique work per child:

  A. independent   — every child re-prefills the full prefix (no caching).
  B. provider      — provider prompt caching: cached-input tokens billed at a
                     discount (e.g. 0.1x); cache writes at write_mult.
  C. self_hosted   — stock SGLang/vLLM prefix caching, same-namespace requests:
                     prefix compute and physical residency are amortized when
                     the engine retains and reuses the identical prefix.
  D. agentfork     — tree-keyed lifecycle controls over the same physical
                     prefix sharing; children pay only their unique suffix.

Compute cost is proxied by prefill token-charges; cache residency by resident
tokens. This is token arithmetic, not a latency, dollar-cost, or byte-level
measurement.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass
class Scenario:
    n_children: int
    prefix_tokens: int
    suffix_tokens: int
    provider_cached_discount: float = 0.1
    provider_write_mult: float = 1.25

    def __post_init__(self) -> None:
        if self.n_children < 1:
            raise ValueError("n_children must be at least 1")
        if self.prefix_tokens < 0 or self.suffix_tokens < 0:
            raise ValueError("token counts must be nonnegative")
        if self.prefix_tokens + self.suffix_tokens == 0:
            raise ValueError("prefix_tokens and suffix_tokens cannot both be zero")
        if self.provider_cached_discount < 0 or self.provider_write_mult < 0:
            raise ValueError("provider price multipliers must be nonnegative")


def model(s: Scenario) -> dict:
    n, p, u = s.n_children, s.prefix_tokens, s.suffix_tokens
    independent = {"prefill_charged": n * (p + u), "resident": n * (p + u)}
    provider = {
        "prefill_charged": p * s.provider_write_mult
        + (n - 1) * p * s.provider_cached_discount + n * u,
        "resident": None,  # opaque, provider-side
    }
    # agentfork and self-hosted radix are defined identically on token
    # arithmetic: both prefill the shared prefix once and charge per-branch
    # suffixes. So the two *_vs_self_hosted ratios below are 1.0 by
    # construction (parity by model definition), not a measured result. The
    # modeled advantage is versus independent prefill and provider caching.
    self_hosted = {"prefill_charged": p + n * u, "resident": p + n * u}
    agentfork = {"prefill_charged": p + n * u, "resident": p + n * u}
    out = {
        "scenario": vars(s),
        "independent": independent,
        "provider_cached": provider,
        "self_hosted_radix": self_hosted,
        "agentfork": agentfork,
    }
    af = agentfork["prefill_charged"]
    out["compute_gain_vs_independent"] = round(independent["prefill_charged"] / af, 2)
    out["compute_gain_vs_provider"] = round(provider["prefill_charged"] / af, 2)
    out["compute_gain_vs_self_hosted"] = round(self_hosted["prefill_charged"] / af, 2)
    out["cache_residency_gain_vs_self_hosted"] = round(
        self_hosted["resident"] / agentfork["resident"], 2)
    return out


# ---------------------------------------------------------------------------
# Cache-pressure dimension
# ---------------------------------------------------------------------------
#
# The plain ``model`` above assumes the shared prefix is *resident* when every
# child runs. On a stock LRU-style radix cache (SGLang RadixAttention, vLLM
# APC) that assumption breaks under memory pressure: unrelated traffic between
# children pushes the parent prefix toward the LRU tail and evicts it, so a
# child re-prefills the whole prefix. agentfork's tree cache *pins* the parent
# prefix to the live tree, so it survives eviction and every child hits.
#
# ``pressure_model`` makes that difference quantitative. It models, for a
# single N-way fanout replayed against a token-capacity ``C`` cache with ``U``
# unrelated prefill tokens injected before each child:
#
#   * the parent-prefix hit rate for stock LRU vs tree-pinned caching, and
#   * the resulting prefill-token compute ratio (stock / pinned).
#
# LRU eviction argument (see report/PRESSURE.md for the full derivation, and
# ``agentfork.bench.pressure_bench`` for the empirical confirmation):
#
# Radix caches (SGLang, vLLM) evict *leaves* LRU-first. Once a child has
# extended the parent, the parent node has a child suffix hanging off it, so
# it is an *internal* node and cannot be evicted until that suffix leaf is
# evicted first. Any content older than the parent -- previous gaps' unrelated
# tokens, older child suffixes -- is a leaf and is evicted before the parent.
# So during a gap the eviction order is: old unrelated, then old suffixes,
# then (only if still short) the current suffix, then finally the parent.
# LRU therefore only has to evict the parent when the parent plus this gap's
# ``U`` unrelated tokens cannot coexist:
#
#   the parent survives a gap iff  P + U <= C   <=>   U <= C - P
#
# The suffix ``S`` drops out: it is evicted before the parent, so it never
# counts against the parent's survival. The break-even is exactly
# ``U* = C - P`` (the cache headroom above the pinned prefix), verified to the
# token against the reference caches in test_pressure_bench.py. The fanout N
# does not move this on/off boundary, but it scales the aggregate advantage:
# every missed child re-prefills P tokens.
#
# Two pressure patterns are modeled:
#   * "sustained": U unrelated tokens before *every* child (the GPU sustained
#     run). Under this pattern the outcome is all-or-nothing: U <= C-P keeps
#     the parent for all N children, U > C-P evicts it for all N.
#   * "burst": U unrelated tokens once, before the first child only (the GPU
#     single-burst run). Only the first child can miss; the re-prefill it pays
#     re-pins the parent for the remaining N-1 children.


@dataclass
class PressureScenario:
    """An N-way fanout replayed under interleaved cache pressure.

    prefix_tokens  P : shared parent prefix length.
    n_children     N : number of sibling branches.
    suffix_tokens  S : unique tokens each child adds after the shared prefix.
    interleaved_tokens U : unrelated prefill tokens injected into the cache
                           before each child (per-gap pressure). This is the
                           reuse-distance pressure the parent must survive.
    capacity_tokens C : token capacity of the KV cache.
    pattern           : "sustained" (U before every child) or "burst" (U once,
                        before the first child only).
    """

    prefix_tokens: int
    n_children: int = 10
    suffix_tokens: int = 8
    interleaved_tokens: int = 0
    capacity_tokens: int = 65_536
    pattern: str = "sustained"

    def __post_init__(self) -> None:
        if self.n_children < 1:
            raise ValueError("n_children must be at least 1")
        if self.prefix_tokens < 0 or self.suffix_tokens < 0:
            raise ValueError("token counts must be nonnegative")
        if self.interleaved_tokens < 0:
            raise ValueError("interleaved_tokens must be nonnegative")
        if self.capacity_tokens <= 0:
            raise ValueError("capacity_tokens must be positive")
        if self.prefix_tokens + self.suffix_tokens == 0:
            raise ValueError("prefix_tokens and suffix_tokens cannot both be zero")
        if self.prefix_tokens > self.capacity_tokens:
            raise ValueError("prefix_tokens must fit in capacity_tokens")
        if self.pattern not in ("sustained", "burst"):
            raise ValueError("pattern must be 'sustained' or 'burst'")


def break_even_u(prefix_tokens: int, capacity_tokens: int) -> int:
    """Per-gap unrelated-token pressure ``U*`` above which stock loses the
    pinned prefix: ``U* = C - P`` (clamped at 0). ``U > U*`` evicts the parent
    during the gap; ``U <= U*`` keeps it resident.
    """
    return max(0, capacity_tokens - prefix_tokens)


def stock_parent_hits(ps: PressureScenario) -> int:
    """Number of children (0..N-1) that find the parent prefix resident on a
    stock LRU radix cache, under the leaf-LRU eviction argument above."""
    n, u = ps.n_children, ps.interleaved_tokens
    ustar = break_even_u(ps.prefix_tokens, ps.capacity_tokens)
    evicts = u > ustar
    if ps.pattern == "burst":
        # pressure hits only the first child; a re-prefill re-pins the parent
        # for the remaining N-1 children.
        return n if not evicts else n - 1
    # sustained: every child faces a gap -> all-or-nothing.
    return n if not evicts else 0


def pressure_model(ps: PressureScenario) -> dict:
    """Model hit rates and prefill-token compute ratio for stock vs pinned."""
    p, n, s = ps.prefix_tokens, ps.n_children, ps.suffix_tokens
    hits = stock_parent_hits(ps)
    misses = n - hits

    # prefill charged for the children path (parent prefilled once up front):
    #   pinned: every child hits -> only its S-token suffix is charged
    #   stock : a miss re-prefills the whole parent prefix (P) plus its suffix
    pinned_prefill = p + n * s
    stock_prefill = p + n * s + misses * p

    out = {
        "scenario": vars(ps),
        "break_even_u": break_even_u(p, ps.capacity_tokens),
        "stock": {
            "parent_hits": hits,
            "parent_hit_rate": round(hits / n, 4),
            "misses": misses,
            "prefill_charged": stock_prefill,
        },
        "pinned": {
            "parent_hits": n,
            "parent_hit_rate": 1.0,
            "misses": 0,
            "prefill_charged": pinned_prefill,
        },
        # prefill-token compute ratio (a proxy for compute, not wall-clock VGE:
        # decode time is unaffected by prefix residency, so measured VGE is
        # smaller than this ratio).
        "compute_ratio_prefill": round(stock_prefill / pinned_prefill, 4),
        "pinning_wins": misses > 0,
    }
    return out


def break_even_surface(prefix_tokens: int, capacity_tokens: int,
                       n_children: int, suffix_tokens: int = 8,
                       u_fractions: tuple[float, ...] = (
                           0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
                       ) -> list[dict]:
    """Sweep U as a fraction of capacity and report the model at each point.

    Returns rows suitable for a markdown table: (U, U/C, stock hit rate,
    prefill compute ratio, pinning_wins).
    """
    rows = []
    for frac in u_fractions:
        u = int(round(frac * capacity_tokens))
        ps = PressureScenario(
            prefix_tokens=prefix_tokens, n_children=n_children,
            suffix_tokens=suffix_tokens, interleaved_tokens=u,
            capacity_tokens=capacity_tokens)
        m = pressure_model(ps)
        rows.append({
            "U": u,
            "U_over_C": round(u / capacity_tokens, 3),
            "stock_hit_rate": m["stock"]["parent_hit_rate"],
            "compute_ratio_prefill": m["compute_ratio_prefill"],
            "pinning_wins": m["pinning_wins"],
        })
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--children", type=int, default=10)
    ap.add_argument("--prefix", type=int, default=32000)
    ap.add_argument("--suffix", type=int, default=2000)
    ap.add_argument("--pressure", action="store_true",
                    help="run the cache-pressure model instead of the base model")
    ap.add_argument("--interleaved", type=int, default=0,
                    help="unrelated prefill tokens between children (U)")
    ap.add_argument("--capacity", type=int, default=65536,
                    help="cache token capacity (C)")
    a = ap.parse_args()
    if a.pressure:
        ps = PressureScenario(
            prefix_tokens=a.prefix, n_children=a.children,
            suffix_tokens=a.suffix, interleaved_tokens=a.interleaved,
            capacity_tokens=a.capacity)
        print(json.dumps(pressure_model(ps), indent=2))
    else:
        print(json.dumps(model(Scenario(a.children, a.prefix, a.suffix)), indent=2))
