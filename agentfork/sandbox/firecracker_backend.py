"""Adapts Firecracker snapshot/restore to ``SandboxBackend`` (Linux +
/dev/kvm only; see ``agentfork/sandbox/fc_bench.py`` for the underlying
``MicroVM`` primitives and the benchmark that measures them directly).

A root branch (``parent_id is None``) boots fresh from ``kernel``/``rootfs``.
Snapshots are taken lazily, at fork time: the first fork from a branch (and
any fork after the branch has run ``exec``) pauses it, snapshots it, and
resumes it, so children always inherit the parent's *current* state — not
the state it happened to have when it was spawned — and branches that are
never forked never pay the snapshot write.

Data plane: with ``vsock=True`` (default) every guest gets a vsock device
and ``exec(branch_id, argv)`` runs a command inside the guest through
``agentfork/sandbox/guest_agent.py``, which the rootfs must be running (see
that module's docstring). With ``overlay_mib`` set, every root gets a
formatted writable scratch drive (``/dev/vdb`` in the guest) and every child
gets its own copy of its parent's overlay, copied while the parent is
paused. The copy is crash-consistent, so ``_fork_source`` first asks the
agent to ``sync`` (best effort) before pausing. Per-VM files (vsock UDS,
overlay) are configured as *relative* paths and each VMM runs in its own
directory — that is what lets N children restored from one snapshot each
bind their own socket and open their own overlay copy.

Each spawn records the VMM's pid in ``<work_dir>/<branch>/fc.pid`` so that a
restarted adapter, which has no in-memory ``MicroVM`` handles, can still kill
branches recorded by a dead supervisor when the orchestrator replays
``kill()`` during ``reconcile()``. That fallback is a plain SIGKILL by pid,
best effort: the pid may have been recycled since the supervisor died. A
restarted adapter can also still fork from a branch whose snapshot files a
previous life wrote to disk.

With ``jailer=JailerConfig(...)`` every VMM runs under Firecracker's
``jailer``: chrooted, cgroup-ready, deprivileged to the configured uid/gid.
The chroot becomes the branch's per-VM directory, shared inputs are
hard-linked in, and children get the parent's snapshot pair plus the rootfs
staged into their own jail (a jailed child cannot see its parent's chroot).
Environmental constraints observed on a real host: ``chroot_base`` must not
sit on a ``nodev`` filesystem (``/tmp`` usually is one — device nodes in the
jail then fail with EACCES even as root), and the jailed gid needs access to
``/dev/kvm`` (pass the ``kvm`` group's gid).

A freshly booted root's userspace takes seconds to come up;
``wait_ready()`` blocks until the guest agent answers, and forking only
after readiness means children inherit a booted, agent-serving guest and
are ready the moment they resume.

This adapter is unit-tested against fakes standing in for ``MicroVM`` and
the vsock client (tests/test_firecracker_backend.py), and the full data
plane was validated end to end against real Firecracker v1.16.1 (aarch64,
Lima nested KVM, 2026-07-17; ``demo/fc_demo.py --exec`` and jailed runs):
vsock exec in every child, per-child overlay mount+write, fork-after-exec
freshness (children see state the parent wrote after boot; divergence stays
isolated), and zero leaked VMMs. Guest networking (tap/netns) and identity
regeneration remain unimplemented.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from typing import Callable

from agentfork._locking import locked
from agentfork.sandbox.fc_bench import JailerConfig, MicroVM, jail_root
from agentfork.sandbox.vsock import DEFAULT_PORT, ExecResult, VsockError, VsockExecClient

_log = logging.getLogger("agentfork.sandbox.firecracker")

_OVERLAY = "overlay.ext4"
_VSOCK = "v.sock"


class FirecrackerSandbox:
    """Locking is narrow: ``_lock`` guards the bookkeeping dicts only, and
    all VMM I/O (boot, restore, snapshot, kill, guest exec) runs outside it,
    so concurrent spawns/kills of different branches proceed in parallel.
    Snapshot refresh serializes per *parent* (one pause/snapshot even with N
    concurrent forks), and snapshot freshness is a generation counter: every
    exec bumps the branch's generation, a snapshot records the generation it
    captured, and a mismatch means the next fork must re-snapshot — an exec
    that lands while a snapshot is being written leaves the branch stale
    rather than being lost to a cleared dirty flag."""

    parallel_lifecycle = True

    def __init__(self, fc_bin: str, kernel: str, rootfs: str, work_dir: str,
                 microvm_factory: Callable[[str, str], MicroVM] = MicroVM,
                 vsock: bool = True, vsock_port: int = DEFAULT_PORT,
                 overlay_mib: int | None = None, mkfs: str = "mkfs.ext4",
                 jailer: JailerConfig | None = None,
                 exec_client_factory: Callable[..., VsockExecClient] = VsockExecClient):
        if overlay_mib is not None and overlay_mib <= 0:
            raise ValueError("overlay_mib must be positive")
        self.fc_bin = fc_bin
        self.kernel = kernel
        self.rootfs = rootfs
        self.work_dir = work_dir
        self.vsock = vsock
        self.vsock_port = vsock_port
        self.overlay_mib = overlay_mib
        self.mkfs = mkfs
        self.jailer = jailer
        self._microvm_factory = microvm_factory
        self._exec_client_factory = exec_client_factory
        self._lock = threading.RLock()
        self._vms: dict[str, object] = {}
        self._pending: set[str] = set()  # spawns in flight
        self._snapshots: dict[str, tuple[str, str]] = {}
        self._gen: dict[str, int] = {}       # bumped by every exec
        self._snap_gen: dict[str, int] = {}  # generation a snapshot captured
        self._parent_locks: dict[str, threading.Lock] = {}

    def _vm_dir(self, branch_id: str) -> str:
        d = os.path.join(self.work_dir, branch_id.replace("/", "_"))
        os.makedirs(d, exist_ok=True)
        return d

    def _pid_path(self, branch_id: str) -> str:
        return os.path.join(self._vm_dir(branch_id), "fc.pid")

    def _host_dir(self, branch_id: str) -> str:
        """Where this branch's per-VM files (vsock UDS, overlay, snapshots)
        live on the host: the jail chroot when jailed, else the branch's
        work directory."""
        if self.jailer is None:
            return self._vm_dir(branch_id)
        d = jail_root(self.jailer, self.fc_bin, self._vm_dir(branch_id))
        os.makedirs(d, exist_ok=True)
        return d

    def _new_vm(self, vm_dir: str):
        if self.jailer is None:
            return self._microvm_factory(self.fc_bin, vm_dir)
        return self._microvm_factory(self.fc_bin, vm_dir, self.jailer)

    def _create_overlay(self, path: str) -> None:
        with open(path, "wb") as f:
            f.truncate(self.overlay_mib * 1024 * 1024)
        subprocess.run([self.mkfs, "-q", "-F", path],
                       check=True, capture_output=True)
        self._chown_into_jail(path)

    def _chown_into_jail(self, path: str) -> None:
        """Files the (root) supervisor creates must be writable by the
        deprivileged uid the jailer drops the VMM to."""
        if self.jailer is not None:
            os.chown(path, self.jailer.uid, self.jailer.gid)

    @staticmethod
    def _link_into(src: str, dst_dir: str, name: str) -> str:
        """Hard-link ``src`` into ``dst_dir`` (same filesystem; jail
        chroots live under one base). The distinct ``name`` matters: a
        child's own future snapshot writes ``mem``/``state`` in its chroot,
        and truncating a hard link shared with the parent would corrupt the
        snapshot other children restore from."""
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst):
            os.unlink(dst)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)  # cross-filesystem fallback
        return dst

    def _parent_lock(self, parent_id: str) -> threading.Lock:
        with self._lock:
            return self._parent_locks.setdefault(parent_id, threading.Lock())

    def _fork_source(self, parent_id: str, child_dir: str) -> tuple[str, str]:
        """Return (mem, state) snapshot paths for restoring a child, taking
        or refreshing the parent's snapshot first if its state has moved on,
        and copying the parent's overlay into the child's directory.

        Serialized per parent: N concurrent forks from one parent produce
        one pause/snapshot, then N independent restores. Snapshot and
        overlay paths are deterministic per branch directory, so a restarted
        adapter with no live parent handle can still fork from the files a
        previous life wrote.
        """
        pdir = self._host_dir(parent_id)
        mem, state = os.path.join(pdir, "mem"), os.path.join(pdir, "state")
        with self._parent_lock(parent_id):
            with self._lock:
                parent_vm = self._vms.get(parent_id)
                gen = self._gen.get(parent_id, 0)
                stale = (parent_id not in self._snapshots
                         or self._snap_gen.get(parent_id) != gen)
            if parent_vm is None:
                if not (os.path.exists(mem) and os.path.exists(state)):
                    raise KeyError(f"no snapshot for parent: {parent_id}")
                if self.overlay_mib is not None:
                    child_overlay = os.path.join(child_dir, _OVERLAY)
                    shutil.copyfile(os.path.join(pdir, _OVERLAY), child_overlay)
                    self._chown_into_jail(child_overlay)
                return mem, state
            if stale and self.overlay_mib is not None and self.vsock:
                try:  # flush guest page cache so the overlay copy is coherent
                    self._exec_live(parent_id, ["sync"], timeout_s=10.0)
                except VsockError:
                    _log.debug("pre-fork sync of %s failed; overlay copy "
                               "will be crash-consistent", parent_id)
            if stale or self.overlay_mib is not None:
                parent_vm.pause()
                try:
                    if stale:
                        t0 = time.perf_counter()
                        parent_vm.snapshot(mem, state)
                        _log.debug("snapshot of %s refreshed in %.1f ms",
                                   parent_id,
                                   (time.perf_counter() - t0) * 1000)
                        with self._lock:
                            self._snapshots[parent_id] = (mem, state)
                            # an exec landing after the `gen` read above will
                            # have bumped _gen past this, keeping the branch
                            # stale for the next fork — never lost
                            self._snap_gen[parent_id] = gen
                    if self.overlay_mib is not None:
                        child_overlay = os.path.join(child_dir, _OVERLAY)
                        shutil.copyfile(os.path.join(pdir, _OVERLAY),
                                        child_overlay)
                        self._chown_into_jail(child_overlay)
                finally:
                    parent_vm.resume()
        return mem, state

    def spawn(self, branch_id: str, parent_id: str | None) -> None:
        with self._lock:
            if branch_id in self._vms or branch_id in self._pending:
                raise ValueError(f"branch exists: {branch_id}")
            self._pending.add(branch_id)
        vm = None
        try:
            d = self._vm_dir(branch_id)
            if parent_id is None:
                overlay = None
                if self.overlay_mib is not None:
                    self._create_overlay(
                        os.path.join(self._host_dir(branch_id), _OVERLAY))
                    overlay = _OVERLAY
                vm = self._new_vm(d)
                vm.boot(self.kernel, self.rootfs, overlay=overlay,
                        vsock_uds=_VSOCK if self.vsock else None)
            else:
                child_host = self._host_dir(branch_id)
                mem, state = self._fork_source(parent_id, child_host)
                if self.jailer is not None:
                    # a jailed child cannot see its parent's chroot, and the
                    # snapshot records device backing files by their names
                    # inside the parent's chroot — stage both the snapshot
                    # pair and the shared rootfs into the child's jail
                    mem = self._link_into(mem, child_host, "pmem")
                    state = self._link_into(state, child_host, "pstate")
                    self._link_into(os.path.abspath(self.rootfs),
                                    child_host, "rootfs.img")
                vm = self._new_vm(d)
                vm.restore(mem, state)
            with open(self._pid_path(branch_id), "w", encoding="utf-8") as f:
                f.write(str(vm.proc.pid))
            with self._lock:
                self._vms[branch_id] = vm
            _log.info("spawned %s (%s)", branch_id,
                      "boot" if parent_id is None else f"fork of {parent_id}")
        except BaseException:
            _log.warning("spawn of %s failed; killing its VMM", branch_id,
                         exc_info=True)
            if vm is not None:
                with self._lock:
                    self._vms.pop(branch_id, None)
                try:
                    vm.kill()
                except Exception:
                    pass
            raise
        finally:
            with self._lock:
                self._pending.discard(branch_id)

    def _exec_live(self, branch_id: str, argv: list[str],
                   timeout_s: float | None) -> ExecResult:
        if not self.vsock:
            raise RuntimeError("exec requires vsock=True")
        uds = os.path.join(self._host_dir(branch_id), _VSOCK)
        client = self._exec_client_factory(uds, self.vsock_port)
        return client.exec(argv, timeout_s)

    def exec(self, branch_id: str, argv: list[str],
             timeout_s: float | None = None) -> ExecResult:
        """Run a command inside the branch's guest via the vsock agent.

        Only branch lookup holds the lock; the guest I/O runs outside it so
        a long command never blocks spawn/kill of other branches. A kill
        racing an exec surfaces here as ``VsockError``.
        """
        with self._lock:
            if branch_id not in self._vms:
                raise KeyError(f"no such branch: {branch_id}")
            # conservatively stale from here on: even a failing command may
            # have changed guest state before failing
            self._gen[branch_id] = self._gen.get(branch_id, 0) + 1
        return self._exec_live(branch_id, argv, timeout_s)

    def wait_ready(self, branch_id: str, timeout_s: float = 60.0) -> None:
        """Block until the branch's guest agent answers a no-op exec — i.e.
        the guest has booted far enough to serve the data plane.

        A freshly booted root is NOT immediately ready: its userspace (and
        the guest agent) is still starting. Fork from a branch only after
        it is ready — children inherit the booted, agent-running state in
        the snapshot and are ready the moment they resume, which is the
        entire point of forking instead of booting N times."""
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                self.exec(branch_id, ["true"], timeout_s=5.0)
                return
            except VsockError as exc:
                if time.monotonic() >= deadline:
                    raise VsockError(
                        f"branch {branch_id} guest agent not ready after "
                        f"{timeout_s}s") from exc
                time.sleep(0.25)

    def kill(self, branch_id: str) -> None:
        with self._lock:
            vm = self._vms.pop(branch_id, None)
            self._snapshots.pop(branch_id, None)
            self._gen.pop(branch_id, None)
            self._snap_gen.pop(branch_id, None)
            self._parent_locks.pop(branch_id, None)
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
        _log.warning("no live handle for %s; SIGKILL by recorded pid %d "
                     "(best effort: the pid may have been recycled)",
                     branch_id, pid)
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
