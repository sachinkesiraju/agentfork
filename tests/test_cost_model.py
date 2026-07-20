import pytest

from agentfork.bench.cost_model import (
    PressureScenario,
    Scenario,
    break_even_surface,
    break_even_u,
    model,
    pressure_model,
    stock_parent_hits,
)


def test_self_hosted_prefix_cache_is_strong_baseline():
    result = model(Scenario(n_children=10, prefix_tokens=32000, suffix_tokens=2000))

    assert result["self_hosted_radix"] == result["agentfork"]
    assert result["compute_gain_vs_self_hosted"] == 1.0
    assert result["cache_residency_gain_vs_self_hosted"] == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_children": 0, "prefix_tokens": 1, "suffix_tokens": 1},
        {"n_children": 1, "prefix_tokens": -1, "suffix_tokens": 1},
        {"n_children": 1, "prefix_tokens": 1, "suffix_tokens": -1},
        {"n_children": 1, "prefix_tokens": 0, "suffix_tokens": 0},
        {
            "n_children": 1,
            "prefix_tokens": 1,
            "suffix_tokens": 1,
            "provider_cached_discount": -0.1,
        },
    ],
)
def test_invalid_scenarios_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Scenario(**kwargs)


# --- cache-pressure model ---------------------------------------------------


def test_break_even_u_is_capacity_headroom_above_prefix():
    assert break_even_u(prefix_tokens=2400, capacity_tokens=32768) == 30368
    # clamped at 0 when the prefix already fills capacity
    assert break_even_u(prefix_tokens=40000, capacity_tokens=32768) == 0


def test_no_pressure_means_no_advantage():
    ps = PressureScenario(prefix_tokens=2400, n_children=10, suffix_tokens=8,
                          interleaved_tokens=0, capacity_tokens=32768)
    out = pressure_model(ps)
    assert out["stock"]["parent_hit_rate"] == 1.0
    assert out["pinned"]["parent_hit_rate"] == 1.0
    assert out["compute_ratio_prefill"] == 1.0
    assert out["pinning_wins"] is False


def test_sustained_pressure_is_all_or_nothing_at_break_even():
    p, c, n, s = 2400, 32768, 10, 8
    ustar = break_even_u(p, c)  # 30368

    at = pressure_model(PressureScenario(
        prefix_tokens=p, n_children=n, suffix_tokens=s,
        interleaved_tokens=ustar, capacity_tokens=c))
    assert at["stock"]["parent_hit_rate"] == 1.0  # still fits: no eviction
    assert at["pinning_wins"] is False

    above = pressure_model(PressureScenario(
        prefix_tokens=p, n_children=n, suffix_tokens=s,
        interleaved_tokens=ustar + 1, capacity_tokens=c))
    assert above["stock"]["parent_hits"] == 0     # every child misses
    assert above["pinned"]["parent_hit_rate"] == 1.0
    # every miss re-prefills P: stock = P + N*S + N*P
    assert above["stock"]["prefill_charged"] == p + n * s + n * p
    assert above["pinned"]["prefill_charged"] == p + n * s
    assert above["compute_ratio_prefill"] == round(
        (p + n * s + n * p) / (p + n * s), 4)


def test_burst_pressure_costs_exactly_one_reprefill():
    p, c, n, s = 2400, 32768, 10, 8
    ustar = break_even_u(p, c)
    out = pressure_model(PressureScenario(
        prefix_tokens=p, n_children=n, suffix_tokens=s,
        interleaved_tokens=ustar + 50_000, capacity_tokens=c,
        pattern="burst"))
    # only the first child misses; it re-pins the parent for the rest
    assert out["stock"]["misses"] == 1
    assert out["stock"]["parent_hits"] == n - 1
    assert out["stock"]["prefill_charged"] == p + n * s + p


def test_fanout_scales_advantage_but_not_break_even():
    # the on/off boundary U* does not depend on N ...
    assert break_even_u(2400, 32768) == break_even_u(2400, 32768)
    small = pressure_model(PressureScenario(
        prefix_tokens=2400, n_children=2, suffix_tokens=8,
        interleaved_tokens=40000, capacity_tokens=32768))
    big = pressure_model(PressureScenario(
        prefix_tokens=2400, n_children=50, suffix_tokens=8,
        interleaved_tokens=40000, capacity_tokens=32768))
    # ... but the compute ratio grows with N (more re-prefills amortized over
    # a fixed parent prefill)
    assert big["compute_ratio_prefill"] > small["compute_ratio_prefill"]


def test_break_even_surface_flips_once():
    rows = break_even_surface(prefix_tokens=2400, capacity_tokens=32768,
                              n_children=10, suffix_tokens=8)
    wins = [r["pinning_wins"] for r in rows]
    # monotone: once pinning wins (enough pressure) it keeps winning
    assert wins == sorted(wins, key=lambda w: w)
    assert any(wins) and not all(wins)
    # low pressure fraction -> no win, high fraction -> win
    assert rows[0]["pinning_wins"] is False
    assert rows[-1]["pinning_wins"] is True


def test_stock_parent_hits_matches_pressure_model():
    ps = PressureScenario(prefix_tokens=2400, n_children=10, suffix_tokens=8,
                          interleaved_tokens=40000, capacity_tokens=32768)
    assert stock_parent_hits(ps) == pressure_model(ps)["stock"]["parent_hits"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prefix_tokens": 1, "capacity_tokens": 0},        # capacity must be > 0
        {"prefix_tokens": -1, "capacity_tokens": 10},      # negative tokens
        {"prefix_tokens": 20, "capacity_tokens": 10},      # prefix > capacity
        {"prefix_tokens": 8, "suffix_tokens": 8, "capacity_tokens": 10},  # P+S > C
        {"prefix_tokens": 0, "suffix_tokens": 0, "capacity_tokens": 10},
        {"prefix_tokens": 1, "capacity_tokens": 10, "interleaved_tokens": -1},
        {"prefix_tokens": 1, "capacity_tokens": 10, "n_children": 0},
        {"prefix_tokens": 1, "capacity_tokens": 10, "pattern": "nope"},
    ],
)
def test_invalid_pressure_scenarios_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PressureScenario(**kwargs)
