"""Adapts Firecracker snapshot/restore to ``SandboxBackend`` (Linux +
/dev/kvm only; see ``agentfork/sandbox/fc_bench.py`` for the underlying
``MicroVM`` primitives and the benchmark that measures them directly).

A root branch (``parent_id is None``) boots fresh from ``kernel``/``rootfs``.
Every branch, root or child, is paused, snapshotted, and resumed immediately
after it starts, so it can itself serve as a fork parent later — ``fc_bench.py``'s own
benchmark only measures one flat level of fanout (boot -> snapshot -> restore
N children -> kill); supporting a multi-level branch tree the way
``ForkOrchestrator.fork()`` allows (fork a child, then fork its child) is new
behavior added by this adapter and has not itself been benchmarked.

Each spawn records the VMM's pid in ``<work_dir>/<branch>/fc.pid`` so that a
restarted adapter, which has no in-memory ``MicroVM`` handles, can still kill
branches recorded by a dead supervisor when the orchestrator replays
``kill()`` during ``reconcile()``. That fallback is a plain SIGKILL by pid,
best effort: the pid may have been recycled since the supervisor died.

This adapter is unit-tested against a fake standing in for ``MicroVM`` (see
tests/test_firecracker_backend.py) and has been run end to end against real
Firecracker v1.16.1 (aarch64, idle 256 MiB guests) via ``demo/fc_demo.py``.
Guest networking, identity regeneration, and readiness probes remain
unimplemented, and multi-level forking (a child of a child) is still
unbenchmarked.
"""

from __future__ import annotations

import os
import signal
import threading
from typing import Callable

from agentfork._locking import locked
from agentfork.sandbox.fc_bench import MicroVM


class FirecrackerSandbox:
    def __init__(self, fc_bin: str, kernel: str, rootfs: str, work_dir: str,
                 microvm_factory: Callable[[str, str], MicroVM] = MicroVM):
        self.fc_bin = fc_bin
        self.kernel = kernel
        self.rootfs = rootfs
        self.work_dir = work_dir
        self._microvm_factory = microvm_factory
        self._lock = threading.RLock()
        self._vms: dict[str, object] = {}
        self._snapshots: dict[str, tuple[str, str]] = {}

    def _vm_dir(self, branch_id: str) -> str:
        d = os.path.join(self.work_dir, branch_id.replace("/", "_"))
        os.makedirs(d, exist_ok=True)
        return d

    def _pid_path(self, branch_id: str) -> str:
        return os.path.join(self.work_dir, branch_id.replace("/", "_"), "fc.pid")

    @locked
    def spawn(self, branch_id: str, parent_id: str | None) -> None:
        if branch_id in self._vms:
            raise ValueError(f"branch exists: {branch_id}")
        d = self._vm_dir(branch_id)
        vm = self._microvm_factory(self.fc_bin, d)
        try:
            if parent_id is None:
                vm.boot(self.kernel, self.rootfs)
            else:
                mem, state = self._snapshots[parent_id]
                vm.restore(mem, state)
            self._vms[branch_id] = vm
            mem_path = os.path.join(d, "mem")
            state_path = os.path.join(d, "state")
            vm.pause()
            vm.snapshot(mem_path, state_path)
            vm.resume()
            self._snapshots[branch_id] = (mem_path, state_path)
            with open(self._pid_path(branch_id), "w", encoding="utf-8") as f:
                f.write(str(vm.proc.pid))
        except BaseException:
            self._vms.pop(branch_id, None)
            try:
                vm.kill()
            except Exception:
                pass
            raise

    @locked
    def kill(self, branch_id: str) -> None:
        vm = self._vms.pop(branch_id, None)
        self._snapshots.pop(branch_id, None)
        pid_path = self._pid_path(branch_id)
        if vm is not None:
            try:
                vm.kill()
            except RuntimeError:
                pass
            self._remove(pid_path)
            return
        # crash recovery: a restarted adapter has no handle for this branch,
        # so fall back to the pid recorded at spawn. Best effort; the pid may
        # have been recycled since the supervisor died.
        try:
            with open(pid_path, encoding="utf-8") as f:
                pid = int(f.read().strip())
        except (FileNotFoundError, ValueError):
            return
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self._remove(pid_path)

    @staticmethod
    def _remove(path: str) -> None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    @locked
    def alive(self, branch_id: str) -> bool:
        vm = self._vms.get(branch_id)
        if vm is None:
            return False
        return vm.proc.poll() is None
