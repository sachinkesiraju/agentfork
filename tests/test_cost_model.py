from agentfork.bench.cost_model import Scenario, model


def test_self_hosted_prefix_cache_is_strong_baseline():
    result = model(Scenario(n_children=10, prefix_tokens=32000, suffix_tokens=2000))

    assert result["self_hosted_radix"] == result["agentfork"]
    assert result["compute_gain_vs_self_hosted"] == 1.0
    assert result["hbm_gain_vs_self_hosted"] == 1.0
