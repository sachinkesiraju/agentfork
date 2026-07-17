import pytest

from agentfork.workload.census import analyze, cluster_bursts


def test_empty_census_returns_zero_summary():
    assert analyze([]) == {
        "sessions": 0,
        "clusters": 0,
        "width_histogram": {},
        "fanout_p95": 0,
        "fanout_max": 0,
        "bursts": [],
        "recurring_identical_prompt_sessions": 0,
    }


def test_missing_titles_do_not_break_burst_analysis():
    sessions = [
        {"id": "a", "created": 0, "prompt": "shared-one"},
        {"id": "b", "created": 1, "prompt": "shared-two"},
    ]

    result = analyze(sessions)

    assert result["bursts"][0]["titles"] == ["", ""]
    assert result["bursts"][0]["shared_prefix_fraction_f"] == 0.7


def test_negative_burst_window_is_rejected():
    with pytest.raises(ValueError):
        cluster_bursts([], burst_window_s=-1)
