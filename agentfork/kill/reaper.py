"""pidfd-based branch kill path.

One kill call sequentially reaps BOTH halves of a branch: the child process
and its reference-cache pages. The process path uses ``pidfd_send_signal``;
``PR_SET_PDEATHSIG`` is the orphan backstop if the supervisor exits. This
module does not use ``CLONE_PIDFD_AUTOKILL`` or manage Firecracker directly.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import time
from dataclasses import dataclass

PR_SET_PDEATHSIG = 1
_libc = ctypes.CDLL(None, use_errno=True)


def _preexec_pdeathsig():
    """Child dies with SIGKILL if the supervisor dies first (orphan backstop)."""
    _libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)


@dataclass
class KillResult:
    pid: int
    signal_us: float      # pidfd_send_signal round trip
    reaped_us: float      # until waitid(WEXITED) confirms the zombie is gone
    kv_freed_tokens: int
    kv_free_us: float

    @property
    def total_ms(self) -> float:
        return (self.reaped_us + self.kv_free_us) / 1000.0


class BranchReaper:
    """Owns (process, tree_id) pairs; kill() reaps both in one call."""

    def __init__(self, kv_cache=None):
        self.kv = kv_cache
        self._branches: dict[str, tuple[subprocess.Popen, int]] = {}

    def spawn(self, tree_id: str, argv: list[str]) -> int:
        proc = subprocess.Popen(argv, preexec_fn=_preexec_pdeathsig,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        pidfd = os.pidfd_open(proc.pid)
        self._branches[tree_id] = (proc, pidfd)
        return proc.pid

    def kill(self, tree_id: str) -> KillResult:
        proc, pidfd = self._branches.pop(tree_id)
        t0 = time.perf_counter_ns()
        signal.pidfd_send_signal(pidfd, signal.SIGKILL)
        t1 = time.perf_counter_ns()
        os.waitid(os.P_PIDFD, pidfd, os.WEXITED)
        proc.wait()
        t2 = time.perf_counter_ns()
        os.close(pidfd)
        freed = 0
        t3 = time.perf_counter_ns()
        if self.kv is not None:
            freed = self.kv.kill(tree_id)
        t4 = time.perf_counter_ns()
        return KillResult(
            pid=proc.pid,
            signal_us=(t1 - t0) / 1e3,
            reaped_us=(t2 - t0) / 1e3,
            kv_freed_tokens=freed,
            kv_free_us=(t4 - t3) / 1e3,
        )

    def alive(self, tree_id: str) -> bool:
        proc, _ = self._branches[tree_id]
        return proc.poll() is None
