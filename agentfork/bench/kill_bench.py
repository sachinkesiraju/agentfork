"""Integrated kill-path benchmark: one kill() reaps sandbox process + KV.

Measures, over N fork/kill cycles:
  - pidfd_send_signal syscall latency (signal submission, not confirmed exit);
  - full reap latency (zombie collected, no orphan possible);
  - KV reclaim latency (tree-keyed refcount drop + page free);
  - end-to-end kill (signal -> both halves reclaimed).

Run: python -m agentfork.bench.kill_bench --cycles 100
"""

from __future__ import annotations

import argparse
import json
import sys

from agentfork.kill.reaper import BranchReaper
from agentfork.kv.tree_cache import TreeKVCache


def run(cycles: int, prefix_tokens: int, suffix_tokens: int) -> dict:
    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    if prefix_tokens < 0 or suffix_tokens < 0:
        raise ValueError("token counts must be nonnegative")
    kv = TreeKVCache()
    kv.create_tree("parent")
    kv.extend("parent", list(range(prefix_tokens)))
    reaper = BranchReaper(kv)
    sig, reap, kvf, total = [], [], [], []
    for i in range(cycles):
        tid = f"branch-{i}"
        kv.fork_branch("parent", tid)
        kv.extend(tid, [10_000_000 + i * suffix_tokens + j for j in range(suffix_tokens)])
        reaper.spawn(tid, [sys.executable, "-c", "import time; time.sleep(60)"])
        r = reaper.kill(tid)
        sig.append(r.signal_us / 1000)
        reap.append(r.reaped_us / 1000)
        kvf.append(r.kv_free_us / 1000)
        total.append(r.total_ms)
    kv.kill("parent")
    def pct(v, q):
        return round(sorted(v)[min(int(q * len(v)), len(v) - 1)], 3)
    return {
        "cycles": cycles,
        "signal_ms": {"p50": pct(sig, 0.5), "p95": pct(sig, 0.95), "max": round(max(sig), 3)},
        "reap_ms": {"p50": pct(reap, 0.5), "p95": pct(reap, 0.95), "max": round(max(reap), 3)},
        "kv_free_ms": {"p50": pct(kvf, 0.5), "p95": pct(kvf, 0.95), "max": round(max(kvf), 3)},
        "total_ms": {"p50": pct(total, 0.5), "p95": pct(total, 0.95), "max": round(max(total), 3)},
        "leaked_kv_tokens": kv.resident_tokens(),
        "orphaned_branches": len(reaper._branches),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=100)
    ap.add_argument("--prefix", type=int, default=32000)
    ap.add_argument("--suffix", type=int, default=500)
    a = ap.parse_args()
    print(json.dumps(run(a.cycles, a.prefix, a.suffix), indent=2))
