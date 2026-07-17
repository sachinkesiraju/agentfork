import pytest

from agentfork.bench.cost_model import Scenario, model


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
