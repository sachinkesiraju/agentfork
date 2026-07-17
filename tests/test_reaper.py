import sys

import pytest

from agentfork.kill.reaper import BranchReaper
from agentfork.kv.tree_cache import TreeKVCache

pytestmark = pytest.mark.skipif(
    not BranchReaper.supported(), reason="requires Linux pidfd support")


def toks(s):
    return [ord(c) for c in s]


def _spawn_sleeper(reaper, tree_id):
    return reaper.spawn(tree_id, [sys.executable, "-c", "import time; time.sleep(300)"])


def test_kill_reaps_process_and_kv_under_10ms():
    kv = TreeKVCache()
    kv.create_tree("p")
    kv.extend("p", toks("SHARED" * 1000))
    reaper = BranchReaper(kv)
    kv.fork_branch("p", "b1")
    kv.extend("b1", toks("branch-work" * 50))
    _spawn_sleeper(reaper, "b1")
    assert reaper.alive("b1")
    res = reaper.kill("b1")
    assert res.kv_freed_tokens == len("branch-work" * 50)
    assert res.total_ms < 10.0, f"kill took {res.total_ms:.2f} ms"


def test_hundred_kill_cycles_no_orphans_no_leaks():
    kv = TreeKVCache()
    kv.create_tree("p")
    kv.extend("p", toks("BASE" * 200))
    reaper = BranchReaper(kv)
    worst = 0.0
    for i in range(100):
        tid = f"b{i}"
        kv.fork_branch("p", tid)
        kv.extend(tid, toks(f"work-{i}" * 20))
        _spawn_sleeper(reaper, tid)
        res = reaper.kill(tid)
        worst = max(worst, res.total_ms)
    assert not reaper._branches          # no orphaned handles
    kv.kill("p")
    assert kv.resident_tokens() == 0     # no leaked pages
    assert worst < 50.0, f"worst kill {worst:.2f} ms"
