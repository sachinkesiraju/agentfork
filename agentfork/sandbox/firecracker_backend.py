"""Adapts Firecracker snapshot/restore to ``SandboxBackend`` (Linux +
/dev/kvm only; see ``agentfork/sandbox/fc_bench.py`` for the underlying
``MicroVM`` primitives and the benchmark that measures them directly).

A root branch (``parent_id is None``) boots fresh from ``kernel``/``rootfs``.
Every branch, root or child, is paused and snapshotted immediately after it
starts, so it can itself serve as a fork parent later — ``fc_bench.py``'s own
benchmark only measures one flat level of fanout (boot -> snapshot -> restore
N children -> kill); supporting a multi-level branch tree the way
``ForkOrchestrator.fork()`` allows (fork a child, then fork its child) is new
behavior added by this adapter and has not itself been benchmarked.

This adapter is unit-tested against a fake standing in for ``MicroVM`` (see
tests/test_firecracker_backend.py); it has not been exercised against a real
Firecracker binary or guest kernel.
"""

from __future__ import annotations

import os
from typing import Callable

from agentfork.sandbox.fc_bench import MicroVM


class FirecrackerSandbox:
    def __init__(self, fc_bin: str, kernel: str, rootfs: str, work_dir: str,
                 microvm_factory: Callable[[str, str], MicroVM] = MicroVM):
        self.fc_bin = fc_bin
        self.kernel = kernel
        self.rootfs = rootfs
        self.work_dir = work_dir
        self._microvm_factory = microvm_factory
        self._vms: dict[str, object] = {}
        self._snapshots: dict[str, tuple[str, str]] = {}

    def _vm_dir(self, branch_id: str) -> str:
        d = os.path.join(self.work_dir, branch_id.replace("/", "_"))
        os.makedirs(d, exist_ok=True)
        return d

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
            self._snapshots[branch_id] = (mem_path, state_path)
        except BaseException:
            self._vms.pop(branch_id, None)
            try:
                vm.kill()
            except Exception:
                pass
            raise

    def kill(self, branch_id: str) -> None:
        vm = self._vms.pop(branch_id, None)
        self._snapshots.pop(branch_id, None)
        if vm is None:
            return
        try:
            vm.kill()
        except RuntimeError:
            pass

    def alive(self, branch_id: str) -> bool:
        vm = self._vms.get(branch_id)
        if vm is None:
            return False
        return vm.proc.poll() is None
