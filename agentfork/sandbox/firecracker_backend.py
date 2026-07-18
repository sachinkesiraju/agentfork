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
``kill()`` during ``reconcile()``. The record includes Linux process start
time and executable identity; recovery verifies both and signals through a
pidfd, refusing a recycled PID. A restarted adapter can also still fork from
a branch whose snapshot files a previous life wrote to disk.

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
are ready the moment they resume. ``ForkOrchestrator`` invokes this probe
automatically for sandboxes that expose it.

This adapter is unit-tested against fakes standing in for ``MicroVM`` and
the vsock client (tests/test_firecracker_backend.py), and the full data
plane was validated end to end against real Firecracker v1.16.1 (aarch64,
Lima nested KVM, 2026-07-17; ``demo/fc_demo.py --exec`` and jailed runs):
vsock exec in every child, per-child overlay mount+write, fork-after-exec
freshness (children see state the parent wrote after boot; divergence stays
isolated), and zero leaked VMMs. Optional ``NetworkConfig`` gives each
branch its own network namespace so snapshot clones do not collide on
tap/MAC/IP; identity reseeding after restore uses the stdin exec path.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from typing import Callable

from agentfork._locking import locked
from agentfork.sandbox.fc_bench import JailerConfig, MicroVM, jail_root
from agentfork.sandbox.netns import TAP_DEV as _TAP
from agentfork.sandbox.netns import NetnsManager, NetworkConfig
from agentfork.sandbox.vsock import (
    DEFAULT_PORT,
    DetachedExec,
    ExecResult,
    VsockError,
    VsockExecClient,
)

_log = logging.getLogger("agentfork.sandbox.firecracker")

_OVERLAY = "overlay.ext4"
_VSOCK = "v.sock"
_ENCODED_BRANCH_PREFIX = "__agentfork__"
_SAFE_BRANCH_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


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
                 network: NetworkConfig | None = None,
                 exec_client_factory: Callable[..., VsockExecClient] = VsockExecClient,
                 readiness_timeout_s: float = 60.0,
                 exec_timeout_s: float = 60.0):
        if overlay_mib is not None and overlay_mib <= 0:
            raise ValueError("overlay_mib must be positive")
        if readiness_timeout_s <= 0 or exec_timeout_s <= 0:
            raise ValueError("readiness and exec timeouts must be positive")
        self.fc_bin = fc_bin
        self.kernel = kernel
        self.rootfs = rootfs
        self.work_dir = work_dir
        os.makedirs(self.work_dir, mode=0o700, exist_ok=True)
        os.chmod(self.work_dir, 0o700)
        self.vsock = vsock
        self.vsock_port = vsock_port
        self.overlay_mib = overlay_mib
        self.mkfs = mkfs
        self.jailer = jailer
        self.network = network
        self._netns = NetnsManager(network) if network is not None else None
        self.readiness_timeout_s = readiness_timeout_s
        self.exec_timeout_s = exec_timeout_s
        self._microvm_factory = microvm_factory
        self._exec_client_factory = exec_client_factory
        self._lock = threading.RLock()
        self._lifecycle_changed = threading.Condition(self._lock)
        self._vms: dict[str, object] = {}
        self._pending: set[str] = set()  # spawns in flight
        self._parent_forks: dict[str, int] = {}
        self._snapshots: dict[str, tuple[str, str]] = {}
        self._gen: dict[str, int] = {}       # bumped by every exec
        self._snap_gen: dict[str, int] = {}  # generation a snapshot captured
        self._netns_index: dict[str, int] = {}  # branch -> its /30 index
        self._parent_locks: dict[str, threading.Lock] = {}

    @staticmethod
    def _branch_dir_name(branch_id: str) -> str:
        if not isinstance(branch_id, str) or not branch_id:
            raise ValueError("branch_id must be a non-empty string")
        if len(branch_id.encode()) > 512 or "\x00" in branch_id:
            raise ValueError("branch_id is too long or contains NUL")
        segments = branch_id.split("/")
        if any(segment in ("", ".", "..")
               or not _SAFE_BRANCH_SEGMENT.fullmatch(segment)
               for segment in segments):
            raise ValueError(
                "branch_id segments must contain only letters, digits, "
                "dot, underscore, or dash, and may not be '.' or '..'")
        if len(segments) == 1:
            if branch_id.startswith(_ENCODED_BRANCH_PREFIX):
                raise ValueError(
                    f"branch_id prefix {_ENCODED_BRANCH_PREFIX!r} is reserved")
            return branch_id
        readable = "_".join(segments)[:80]
        digest = hashlib.sha256(branch_id.encode()).hexdigest()[:16]
        return f"{_ENCODED_BRANCH_PREFIX}{digest}-{readable}"

    def _vm_dir(self, branch_id: str, *, create: bool = True) -> str:
        # 0700: the API socket and vsock UDS inside grant full control of
        # the VMM and guest to anyone who can reach them
        d = os.path.join(self.work_dir, self._branch_dir_name(branch_id))
        if create:
            os.makedirs(d, mode=0o700, exist_ok=True)
            os.chmod(d, 0o700)
        return d

    def _pid_path(self, branch_id: str) -> str:
        return os.path.join(self._vm_dir(branch_id, create=False), "fc.pid")

    def _host_dir(self, branch_id: str) -> str:
        """Where this branch's per-VM files (vsock UDS, overlay, snapshots)
        live on the host: the jail chroot when jailed, else the branch's
        work directory."""
        if self.jailer is None:
            return self._vm_dir(branch_id)
        d = jail_root(self.jailer, self.fc_bin, self._vm_dir(branch_id))
        os.makedirs(d, mode=0o700, exist_ok=True)
        return d

    def _vmm_comm(self) -> str:
        """The /proc comm a live VMM for this sandbox should have: the
        fc binary's basename, kernel-truncated to 15 bytes (the jailer
        execve()s the fc binary, so jailed VMMs match too)."""
        return os.path.basename(os.path.abspath(self.fc_bin))[:15]

    def _pid_is_our_vmm(self, pid: int) -> bool | None:
        """True/False when /proc can answer whether ``pid`` is one of our
        VMMs; None when it cannot (no /proc, or the pid is gone)."""
        try:
            with open(f"/proc/{pid}/comm", encoding="ascii") as f:
                return f.read().strip() == self._vmm_comm()
        except OSError:
            return None

    def _new_vm(self, vm_dir: str, netns: str | None = None):
        # keep the call minimal so simple 2-arg fakes still work; only pass
        # jailer/netns when actually in use
        if self.jailer is None and netns is None:
            return self._microvm_factory(self.fc_bin, vm_dir)
        return self._microvm_factory(self.fc_bin, vm_dir, self.jailer, netns)

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
    def _copy_overlay(src: str, dst: str) -> None:
        """Copy an overlay image without paying for its full logical size:
        FICLONE reflink where the filesystem supports it (btrfs/XFS —
        instant, shares extents copy-on-write), else a sparse-aware copy
        that only reads and writes data extents (SEEK_DATA/SEEK_HOLE — a
        mostly-empty scratch disk copies in milliseconds), else a plain
        byte copy."""
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            if hasattr(fcntl, "FICLONE"):
                try:
                    fcntl.ioctl(fdst.fileno(), fcntl.FICLONE, fsrc.fileno())
                    return
                except OSError:
                    pass  # not a reflink-capable filesystem
            try:
                FirecrackerSandbox._sparse_copy(fsrc.fileno(), fdst.fileno())
                return
            except OSError:
                pass  # filesystem cannot enumerate holes
        shutil.copyfile(src, dst)

    @staticmethod
    def _sparse_copy(src_fd: int, dst_fd: int) -> None:
        end = os.lseek(src_fd, 0, os.SEEK_END)
        os.ftruncate(dst_fd, end)
        pos = 0
        while pos < end:
            try:
                data_start = os.lseek(src_fd, pos, os.SEEK_DATA)
            except OSError as exc:
                if exc.errno == errno.ENXIO:
                    break  # only holes remain to the end
                raise  # a real I/O error must not masquerade as end-of-data
            hole = os.lseek(src_fd, data_start, os.SEEK_HOLE)
            os.lseek(src_fd, data_start, os.SEEK_SET)
            os.lseek(dst_fd, data_start, os.SEEK_SET)
            remaining = hole - data_start
            while remaining:
                chunk = os.read(src_fd, min(remaining, 1 << 20))
                view = memoryview(chunk)
                while view:
                    written = os.write(dst_fd, view)
                    view = view[written:]
                remaining -= len(chunk)
            pos = hole

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

    @staticmethod
    def _process_identity(pid: int) -> dict | None:
        """Return stable Linux process identity fields used to reject PID reuse."""
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
                stat = f.read()
            # comm is parenthesized and may contain spaces; fields after the
            # final ')' start at field 3, making starttime field 22 index 19.
            fields = stat.rsplit(")", 1)[1].split()
            start_time = int(fields[19])
            exe = os.path.realpath(f"/proc/{pid}/exe")
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            return None
        return {"pid": pid, "start_time": start_time, "exe": exe}

    def _write_pid_record(self, branch_id: str, pid: int) -> None:
        record = self._process_identity(pid) or {"pid": pid}
        path = self._pid_path(branch_id)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

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
                    self._copy_overlay(os.path.join(pdir, _OVERLAY),
                                       child_overlay)
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
                snapshot_error = None
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
                        self._copy_overlay(os.path.join(pdir, _OVERLAY),
                                           child_overlay)
                        self._chown_into_jail(child_overlay)
                except BaseException as exc:
                    snapshot_error = exc
                    raise
                finally:
                    try:
                        parent_vm.resume()
                    except Exception:
                        if snapshot_error is None:
                            raise  # parent left paused: that IS the failure
                        # snapshot already failed (parent killed mid-fork?);
                        # don't let the resume failure mask the primary error
                        _log.warning("resume of %s failed after snapshot "
                                     "error", parent_id, exc_info=True)
        return mem, state

    def spawn(self, branch_id: str, parent_id: str | None) -> None:
        with self._lock:
            if branch_id in self._vms or branch_id in self._pending:
                raise ValueError(f"branch exists: {branch_id}")
            self._pending.add(branch_id)
            if parent_id is not None:
                self._parent_forks[parent_id] = (
                    self._parent_forks.get(parent_id, 0) + 1)
        vm = None
        try:
            # inside the try so a netns failure still runs the finally that
            # clears _pending (and the except that tears the netns down)
            netns = self._setup_netns(branch_id)
            d = self._vm_dir(branch_id)
            tap = _TAP if self._netns is not None else None
            if parent_id is None:
                overlay = None
                if self.overlay_mib is not None:
                    self._create_overlay(
                        os.path.join(self._host_dir(branch_id), _OVERLAY))
                    overlay = _OVERLAY
                vm = self._new_vm(d, netns)
                vm.boot(self.kernel, self.rootfs, overlay=overlay,
                        vsock_uds=_VSOCK if self.vsock else None, tap=tap)
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
                vm = self._new_vm(d, netns)
                vm.restore(mem, state)
            self._write_pid_record(branch_id, vm.proc.pid)
            with self._lock:
                self._vms[branch_id] = vm
            if parent_id is not None:
                self._reseed_identity(branch_id)
            _log.info("spawned %s (%s%s)", branch_id,
                      "boot" if parent_id is None else f"fork of {parent_id}",
                      f", netns {netns}" if netns else "")
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
            self._teardown_netns(branch_id)
            self._cleanup_artifacts(branch_id)
            raise
        finally:
            with self._lifecycle_changed:
                self._pending.discard(branch_id)
                if parent_id is not None:
                    remaining = self._parent_forks[parent_id] - 1
                    if remaining:
                        self._parent_forks[parent_id] = remaining
                    else:
                        self._parent_forks.pop(parent_id, None)
                self._lifecycle_changed.notify_all()

    def _netns_idx_path(self, branch_id: str) -> str:
        return os.path.join(self._vm_dir(branch_id), "netns.idx")

    def _setup_netns(self, branch_id: str) -> str | None:
        """Build this branch's network namespace, if networking is enabled.
        Every guest boots believing it is the same tap/MAC/IP; the namespace
        keeps N snapshot clones from colliding on one host network. The /30
        index is journaled next to fc.pid so a restarted adapter (which has
        no in-memory _netns_index) can still tear the namespace down."""
        if self._netns is None:
            return None
        name, index = self._netns.setup(branch_id)
        with open(self._netns_idx_path(branch_id), "w", encoding="utf-8") as f:
            f.write(str(index))
        with self._lock:
            self._netns_index[branch_id] = index
        return name

    def _teardown_netns(self, branch_id: str) -> None:
        if self._netns is None:
            return
        with self._lock:
            index = self._netns_index.pop(branch_id, None)
        if index is None:  # restarted adapter: recover the index from disk
            try:
                with open(self._netns_idx_path(branch_id), encoding="utf-8") as f:
                    index = int(f.read().strip())
            except (FileNotFoundError, ValueError):
                return
        self._netns.teardown(branch_id, index)
        self._remove(self._netns_idx_path(branch_id))

    def _reseed_identity(self, branch_id: str) -> None:
        """Best-effort de-correlation of a restored clone from its siblings.

        Every child restored from one snapshot resumes with the parent's RNG
        state, so without this they'd draw identical 'random' values (TLS
        nonces, session tokens, UUIDs). Feeding host-fresh entropy into the
        guest's pool via the exec channel stirs each clone uniquely. Machine
        identity (hostname, SSH host keys, DHCP client-id) is the rootfs's
        job — see tools/build_rootfs.sh, which regenerates them on boot.

        Purely best-effort: a concurrent kill of this branch, or any
        transport hiccup, just skips the reseed (OSError covers a raw
        ConnectionReset that never became a VsockError)."""
        if not self.vsock:
            return
        try:
            self._exec_live(branch_id, ["tee", "/dev/urandom"],
                            timeout_s=5.0, stdin=os.urandom(32))
        except (VsockError, OSError):
            _log.debug("entropy reseed of %s skipped (agent gone or racing "
                       "a kill)", branch_id)

    def _exec_client(self, branch_id: str) -> VsockExecClient:
        if not self.vsock:
            raise RuntimeError("exec requires vsock=True")
        uds = os.path.join(self._host_dir(branch_id), _VSOCK)
        return self._exec_client_factory(uds, self.vsock_port)

    def _exec_live(self, branch_id: str, argv: list[str],
                   timeout_s: float | None,
                   stdin: bytes | None = None,
                   handshake_timeout_s: float | None = None) -> ExecResult:
        client = self._exec_client(branch_id)
        if handshake_timeout_s is not None and hasattr(
                client, "handshake_timeout_s"):
            client.handshake_timeout_s = min(
                client.handshake_timeout_s, handshake_timeout_s)
            if hasattr(client, "host_grace_s") and timeout_s is not None:
                client.host_grace_s = min(
                    client.host_grace_s,
                    max(handshake_timeout_s - timeout_s, 0.001),
                )
        return client.exec(argv, timeout_s, stdin=stdin)

    def _exec_branch(
        self,
        branch_id: str,
        argv: list[str],
        timeout_s: float,
        stdin: bytes | None = None,
        handshake_timeout_s: float | None = None,
    ) -> ExecResult:
        # The same per-branch lock guards snapshots: a fork cannot capture a
        # command halfway through and then incorrectly mark that snapshot
        # current after the command mutates more state.
        with self._parent_lock(branch_id):
            with self._lock:
                if branch_id not in self._vms:
                    raise KeyError(f"no such branch: {branch_id}")
            try:
                return self._exec_live(
                    branch_id, argv, timeout_s, stdin=stdin,
                    handshake_timeout_s=handshake_timeout_s)
            finally:
                with self._lock:
                    if branch_id in self._vms:
                        self._gen[branch_id] = (
                            self._gen.get(branch_id, 0) + 1)

    def exec(self, branch_id: str, argv: list[str],
             timeout_s: float | None = None,
             stdin: bytes | None = None) -> ExecResult:
        """Run a command inside the branch's guest via the vsock agent."""
        return self._exec_branch(
            branch_id,
            argv,
            self.exec_timeout_s if timeout_s is None else timeout_s,
            stdin=stdin,
        )

    def exec_detached(self, branch_id: str,
                      argv: list[str]) -> DetachedExec:
        """Start a background process in the branch's guest (dev servers,
        watchers). Returns immediately with the guest pid and the guest-side
        log path; follow output via ``exec(["tail", ...])`` on that path."""
        with self._parent_lock(branch_id):
            with self._lock:
                if branch_id not in self._vms:
                    raise KeyError(f"no such branch: {branch_id}")
            try:
                return self._exec_client(branch_id).exec_detached(argv)
            finally:
                with self._lock:
                    if branch_id in self._vms:
                        self._gen[branch_id] = (
                            self._gen.get(branch_id, 0) + 1)

    def wait_ready(self, branch_id: str, timeout_s: float | None = None) -> None:
        """Block until the branch's guest agent answers a no-op exec — i.e.
        the guest has booted far enough to serve the data plane.

        A freshly booted root is NOT immediately ready: its userspace (and
        the guest agent) is still starting. Fork from a branch only after
        it is ready — children inherit the booted, agent-running state in
        the snapshot and are ready the moment they resume, which is the
        entire point of forking instead of booting N times."""
        if not self.vsock:
            return
        timeout_s = (
            self.readiness_timeout_s if timeout_s is None else timeout_s)
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VsockError(
                    f"branch {branch_id} guest agent not ready after "
                    f"{timeout_s}s")
            try:
                self._exec_branch(
                    branch_id,
                    ["true"],
                    timeout_s=min(1.0, remaining),
                    handshake_timeout_s=remaining,
                )
                return
            except VsockError as exc:
                if time.monotonic() >= deadline:
                    raise VsockError(
                        f"branch {branch_id} guest agent not ready after "
                        f"{timeout_s}s") from exc
                time.sleep(0.25)

    def kill(self, branch_id: str) -> None:
        with self._lifecycle_changed:
            while self._parent_forks.get(branch_id, 0):
                self._lifecycle_changed.wait()
            vm = self._vms.pop(branch_id, None)
            self._snapshots.pop(branch_id, None)
            self._gen.pop(branch_id, None)
            self._snap_gen.pop(branch_id, None)
        pid_path = self._pid_path(branch_id)
        if vm is not None:
            try:
                vm.kill()
            except Exception:  # best effort: teardown must run regardless
                _log.warning("vm.kill of %s raised; continuing teardown",
                             branch_id, exc_info=True)
            self._teardown_netns(branch_id)  # after the VMM releases the tap
            self._remove(pid_path)
            self._cleanup_artifacts(branch_id)
            with self._lock:
                self._parent_locks.pop(branch_id, None)
            return
        # Crash recovery: a restarted adapter has no handle for this branch.
        # Prefer start_time+exe identity from the JSON pid record; fall back
        # to /proc/comm for legacy plain-pid files. Refuse a recycled pid.
        try:
            with open(pid_path, encoding="utf-8") as f:
                record = self._parse_pid_record(f.read().strip())
            pid = int(record["pid"])
        except FileNotFoundError:
            self._teardown_netns(branch_id)
            self._cleanup_artifacts(branch_id)
            return
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"invalid Firecracker pid record for {branch_id}") from exc
        expected = {
            "pid": pid,
            "start_time": record.get("start_time"),
            "exe": record.get("exe"),
        }
        current = self._process_identity(pid)
        if current is None:
            self._remove(pid_path)
            self._teardown_netns(branch_id)
            self._cleanup_artifacts(branch_id)
            return
        if (expected["start_time"] is not None and expected["exe"] is not None
                and current != expected):
            _log.error(
                "refusing to kill pid %d for %s: process identity changed",
                pid, branch_id)
            self._remove(pid_path)
            self._teardown_netns(branch_id)
            self._cleanup_artifacts(branch_id)
            return
        if ((expected["start_time"] is None or expected["exe"] is None)
                and self._pid_is_our_vmm(pid) is False):
            _log.warning("recorded pid %d for %s is not a %s process "
                         "(recycled); not killing it", pid, branch_id,
                         self._vmm_comm())
            self._remove(pid_path)
            self._teardown_netns(branch_id)
            self._cleanup_artifacts(branch_id)
            return
        _log.warning("no live handle for %s; SIGKILL by recorded pid %d",
                     branch_id, pid)
        try:
            pidfd = os.pidfd_open(pid)
            try:
                if self._process_identity(pid) == current:
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
            finally:
                os.close(pidfd)
        except ProcessLookupError:
            pass
        except (AttributeError, PermissionError):
            # pidfd unavailable or denied: fall back to plain SIGKILL
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self._teardown_netns(branch_id)
        self._remove(pid_path)
        self._cleanup_artifacts(branch_id)
        with self._lock:
            self._parent_locks.pop(branch_id, None)

    @staticmethod
    def _parse_pid_record(raw: str) -> dict:
        """Parse a pid file body: JSON object, bare JSON int, or plain digits."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"pid": int(raw)}
        if isinstance(parsed, int) and not isinstance(parsed, bool):
            return {"pid": parsed}
        if not isinstance(parsed, dict):
            raise ValueError("pid record must be an object or integer")
        return {
            "pid": int(parsed["pid"]),
            "start_time": parsed.get("start_time"),
            "exe": parsed.get("exe"),
        }

    def _recorded_pid(self, branch_id: str) -> int | None:
        record = self._read_pid_record(branch_id)
        return None if record is None else record["pid"]

    def _read_pid_record(self, branch_id: str) -> dict | None:
        try:
            with open(self._pid_path(branch_id), encoding="utf-8") as f:
                return self._parse_pid_record(f.read().strip())
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _remove(path: str) -> None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    def _cleanup_artifacts(self, branch_id: str) -> None:
        vm_dir = self._vm_dir(branch_id, create=False)
        paths = [vm_dir]
        if self.jailer is not None:
            host_dir = jail_root(
                self.jailer, self.fc_bin, vm_dir)
            paths.insert(0, os.path.dirname(host_dir))
        for path in paths:
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass

    @locked
    def alive(self, branch_id: str) -> bool:
        vm = self._vms.get(branch_id)
        if vm is not None:
            return vm.proc.poll() is None
        # restart recovery: a restarted adapter has no handle, but the pid
        # record plus identity/comm checks can still answer
        record = self._read_pid_record(branch_id)
        if record is None:
            return False
        pid = record["pid"]
        current = self._process_identity(pid)
        if (current is not None
                and record.get("start_time") is not None
                and record.get("exe") is not None):
            return current == {
                "pid": pid,
                "start_time": record["start_time"],
                "exe": record["exe"],
            }
        return self._pid_is_our_vmm(pid) is True

    @locked
    def sweep_dead(self) -> list[str]:
        """Branch IDs whose VMM process has exited without a kill — the
        supervision hook the orchestrator's background reaper collects."""
        return [branch_id for branch_id, vm in self._vms.items()
                if vm.proc.poll() is not None]
