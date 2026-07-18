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
import sys
import threading
import time
from dataclasses import dataclass
from functools import partial

from agentfork._locking import locked

PR_SET_PDEATHSIG = 1
_libc = ctypes.CDLL(None, use_errno=True)


def _preexec_pdeathsig(parent_pid: int):
    """Child dies with SIGKILL if the supervisor dies first (orphan backstop)."""
    if _libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        os._exit(127)
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


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
    """Owns (process, tree_id) pairs; kill() reaps both in one call.

    Orphan-backstop modes (``pdeathsig``): the ``PR_SET_PDEATHSIG`` signal
    makes a branch process die with the supervisor.

    - ``True`` (default) arms it via ``preexec_fn`` — fast, but CPython
      documents ``preexec_fn`` as unsafe if any other thread exists at spawn
      time, so spawns must be serialized.
    - ``"shim"`` arms it via a re-exec launcher (``_pdeathsig``) with no
      ``preexec_fn``: thread-safe (nothing runs in the fork child before
      exec), at the cost of one interpreter startup per spawn. Spawns may
      then run concurrently.
    - ``False`` skips the backstop entirely (also thread-safe); orphans of a
      died supervisor are collected by ``ForkOrchestrator.reconcile()`` on
      the next start rather than by the kernel immediately.

    ``thread_safe`` reports whether spawns may run concurrently under this
    mode.
    """

    @staticmethod
    def supported() -> bool:
        return all((hasattr(os, "pidfd_open"), hasattr(os, "P_PIDFD"),
                    hasattr(signal, "pidfd_send_signal")))

    def __init__(self, kv_cache=None, pdeathsig: bool | str = True):
        if pdeathsig not in (True, False, "shim"):
            raise ValueError("pdeathsig must be True, False, or 'shim'")
        self.kv = kv_cache
        self.pdeathsig = pdeathsig
        self._lock = threading.RLock()
        self._branches: dict[str, tuple[subprocess.Popen, int]] = {}

    @property
    def thread_safe(self) -> bool:
        """True when spawn() is safe to call from multiple threads at once
        (no ``preexec_fn``)."""
        return self.pdeathsig is not True

    @locked
    def spawn(self, tree_id: str, argv: list[str]) -> int:
        if not self.supported():
            raise RuntimeError("BranchReaper requires Linux pidfd support")
        if tree_id in self._branches:
            raise ValueError(f"branch exists: {tree_id}")
        if not argv:
            raise ValueError("argv must not be empty")
        preexec = None
        if self.pdeathsig == "shim":
            # thread-safe: no Python runs in the fork child; the launcher
            # arms the death signal after exec'ing, then exec's the command
            argv = [sys.executable, "-m", "agentfork.kill._pdeathsig",
                    str(os.getpid()), *argv]
        elif self.pdeathsig is True:
            preexec = partial(_preexec_pdeathsig, os.getpid())
        proc = subprocess.Popen(
            argv,
            preexec_fn=preexec,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            pidfd = os.pidfd_open(proc.pid)
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        self._branches[tree_id] = (proc, pidfd)
        return proc.pid

    @locked
    def kill(self, tree_id: str) -> KillResult:
        proc, pidfd = self._branches[tree_id]
        t0 = time.perf_counter_ns()
        try:
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            t1 = time.perf_counter_ns()
            try:
                os.waitid(os.P_PIDFD, pidfd, os.WEXITED)
            except ChildProcessError:
                pass
            proc.wait()
            t2 = time.perf_counter_ns()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            os.close(pidfd)
            self._branches.pop(tree_id, None)
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

    @locked
    def alive(self, tree_id: str) -> bool:
        proc, _ = self._branches[tree_id]
        return proc.poll() is None

    @locked
    def close(self) -> None:
        error = None
        for tree_id in list(self._branches):
            try:
                self.kill(tree_id)
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
