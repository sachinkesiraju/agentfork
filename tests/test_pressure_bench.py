import pytest

from agentfork.bench.cost_model import PressureScenario, break_even_u
from agentfork.bench.pressure_bench import (
    StockRadixCache,
    compare,
    run_pinned,
    run_stock,
)


# --- StockRadixCache unit behavior ------------------------------------------


def test_stock_cache_hits_cached_prefix():
    c = StockRadixCache(capacity_tokens=1000)
    matched, charged = c.run_request(list(range(100)))
    assert (matched, charged) == (0, 100)  # cold
    matched, charged = c.run_request(list(range(100)) + [500, 501])
    assert matched == 100 and charged == 2  # prefix reused, only suffix charged


def test_stock_cache_evicts_lru_leaf_under_pressure():
    c = StockRadixCache(capacity_tokens=100)
    c.run_request(list(range(50)))            # tenant A prefix
    c.run_request(list(range(1000, 1100)))    # 100 fresh tokens -> evicts A
    assert c.match_prefix(list(range(50))) == 0  # A gone
    assert c.resident <= c.capacity


def test_stock_cache_protects_internal_node_until_leaf_evicted():
    # a prefix with a child suffix is internal; a fresh request that fits
    # alongside it must not evict it.
    c = StockRadixCache(capacity_tokens=120)
    c.run_request(list(range(40)))                    # parent
    c.run_request(list(range(40)) + [900, 901])       # child suffix on parent
    c.run_request(list(range(2000, 2040)))            # 40 fresh, fits (P+U<=C)
    assert c.match_prefix(list(range(40))) == 40      # parent survived


# --- model vs reference-cache agreement -------------------------------------

# (prefix, n, suffix, capacity, pattern)
_SHAPES = [
    (2400, 10, 8, 32768, "sustained"),
    (2400, 10, 8, 32768, "burst"),
    (2400, 12, 8, 32768, "sustained"),   # locked-holdout fanout
    (800, 6, 16, 8192, "sustained"),
    (4096, 4, 64, 16384, "sustained"),
    (4096, 4, 64, 16384, "burst"),
    # large suffixes, still within the P + S <= C domain: confirms the suffix
    # is evicted before the parent and does not move the U* = C - P boundary.
    (2000, 5, 2000, 8192, "sustained"),
    (2048, 3, 2048, 4096, "sustained"),
    (2048, 3, 2048, 4096, "burst"),
]

_U_FRACS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.95, 1.0, 1.5, 2.0]


@pytest.mark.parametrize("prefix,n,suffix,capacity,pattern", _SHAPES)
def test_model_matches_reference_caches(prefix, n, suffix, capacity, pattern):
    for frac in _U_FRACS:
        u = int(round(frac * capacity))
        ps = PressureScenario(
            prefix_tokens=prefix, n_children=n, suffix_tokens=suffix,
            interleaved_tokens=u, capacity_tokens=capacity, pattern=pattern)
        r = compare(ps)
        model, measured = r["model"], r["measured"]
        # pinning keeps every child hitting on the real TreeKVCache
        assert measured["pinned_hit_rate"] == 1.0
        # model hit rate and compute ratio match the measured caches exactly
        assert model["stock_hit_rate"] == measured["stock_hit_rate"], (ps, r)
        assert model["compute_ratio_prefill"] == measured["compute_ratio_prefill"], (
            ps, r)


def test_break_even_boundary_is_exact_against_reference():
    # U* = C - P: at U* the parent survives, at U*+1 it is evicted -- checked
    # against the real StockRadixCache / TreeKVCache, not just the formula.
    p, c, n, s = 2400, 32768, 10, 8
    ustar = break_even_u(p, c)

    at = compare(PressureScenario(
        prefix_tokens=p, n_children=n, suffix_tokens=s,
        interleaved_tokens=ustar, capacity_tokens=c))
    assert at["measured"]["stock_hit_rate"] == 1.0

    above = compare(PressureScenario(
        prefix_tokens=p, n_children=n, suffix_tokens=s,
        interleaved_tokens=ustar + 1, capacity_tokens=c))
    assert above["measured"]["stock_hit_rate"] == 0.0


def test_large_suffix_does_not_shift_break_even():
    # even with a suffix as large as the prefix (still P + S <= C), the boundary
    # stays U* = C - P: the suffix is a leaf, evicted before the parent.
    p, s, c, n = 2000, 2000, 8192, 6
    ustar = break_even_u(p, c)  # 6192, independent of S
    at = compare(PressureScenario(
        prefix_tokens=p, n_children=n, suffix_tokens=s,
        interleaved_tokens=ustar, capacity_tokens=c))
    above = compare(PressureScenario(
        prefix_tokens=p, n_children=n, suffix_tokens=s,
        interleaved_tokens=ustar + 1, capacity_tokens=c))
    assert at["measured"]["stock_hit_rate"] == 1.0
    assert above["measured"]["stock_hit_rate"] == 0.0


def test_pinned_arm_recharges_only_suffixes():
    ps = PressureScenario(prefix_tokens=2400, n_children=10, suffix_tokens=8,
                          interleaved_tokens=96000, capacity_tokens=32768)
    pinned = run_pinned(ps)
    stock = run_stock(ps)
    # pinned: parent once + N suffixes; stock under this pressure re-prefills
    # the parent for every child.
    assert pinned.child_prefill_charged == 2400 + 10 * 8
    assert stock.parent_hits == 0
    assert stock.child_prefill_charged == 2400 + 10 * 8 + 10 * 2400
