"""Crash-injection benchmark: no orphans when the supervisor dies violently.

For each cycle, a supervisor process spawns N sandbox children through
BranchReaper (PR_SET_PDEATHSIG backstop), then the supervisor itself is
SIGKILLed with no chance to clean up. The children must be reaped by the
kernel, not leaked. This exercises the ``PR_SET_PDEATHSIG`` path used by the
current implementation; it does not exercise ``CLONE_PIDFD_AUTOKILL``.

Run: python -m agentfork.bench.crash_bench --cycles 50 --children 5
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time


def _supervisor(n_children: int, pipe_w: int):
    from agentfork.kill.reaper import BranchReaper
    from agentfork.kv.tree_cache import TreeKVCache

    kv = TreeKVCache()
    kv.create_tree("p")
    kv.extend("p", list(range(1000)))
    reaper = BranchReaper(kv)
    pids = []
    for i in range(n_children):
        kv.fork_branch("p", f"c{i}")
        pids.append(reaper.spawn(
            f"c{i}", [sys.executable, "-c", "import time; time.sleep(120)"]))
    os.write(pipe_w, (json.dumps(pids) + "\n").encode())
    os.close(pipe_w)
    time.sleep(120)  # wait to be murdered


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def run(cycles: int, n_children: int) -> dict:
    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    if n_children < 1:
        raise ValueError("n_children must be at least 1")
    orphans_total = 0
    timed_out_cycles = 0
    reap_times = []
    for _ in range(cycles):
        r, w = os.pipe()
        sup = os.fork()
        if sup == 0:
            os.close(r)
            _supervisor(n_children, w)
            os._exit(0)
        os.close(w)
        with os.fdopen(r) as f:
            child_pids = json.loads(f.readline())
        assert all(_alive(p) for p in child_pids)
        # violent crash: supervisor gets SIGKILL, no cleanup handlers run
        t0 = time.perf_counter()
        os.kill(sup, signal.SIGKILL)
        os.waitpid(sup, 0)
        deadline = time.monotonic() + 5.0
        while any(_alive(p) for p in child_pids):
            if time.monotonic() > deadline:
                break
            time.sleep(0.001)
        alive_pids = [p for p in child_pids if _alive(p)]
        orphans_total += len(alive_pids)
        if alive_pids:
            timed_out_cycles += 1
            for pid in alive_pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        else:
            reap_times.append(time.perf_counter() - t0)

    def pct(v, q):
        return round(sorted(v)[min(int(q * len(v)), len(v) - 1)] * 1000, 2)
    timings = None
    if reap_times:
        timings = {"p50": pct(reap_times, 0.5),
                   "p95": pct(reap_times, 0.95),
                   "max": round(max(reap_times) * 1000, 2)}
    return {
        "cycles": cycles,
        "children_per_cycle": n_children,
        "orphans": orphans_total,
        "timed_out_cycles": timed_out_cycles,
        "kernel_reap_ms": timings,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=50)
    ap.add_argument("--children", type=int, default=5)
    a = ap.parse_args()
    print(json.dumps(run(a.cycles, a.children), indent=2))
