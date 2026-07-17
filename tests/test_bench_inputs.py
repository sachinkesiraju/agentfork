import pytest

from agentfork.bench.crash_bench import run as run_crash_bench
from agentfork.bench.kill_bench import run as run_kill_bench


def test_kill_benchmark_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        run_kill_bench(0, 1, 1)
    with pytest.raises(ValueError):
        run_kill_bench(1, -1, 1)


def test_crash_benchmark_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        run_crash_bench(0, 1)
    with pytest.raises(ValueError):
        run_crash_bench(1, 0)
