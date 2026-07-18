"""Firecracker microVM snapshot and fanout benchmark.

Measures on the current host:
  1. cold boot latency of a microVM;
  2. snapshot creation latency (paused parent -> memory file + vmstate);
  3. restore latency (child resumes from the parent snapshot);
  4. N-way fanout: N children restored from ONE parent snapshot, sharing the
     memory file read-only (page-cache backed CoW-style sharing);
  5. kill latency per child via pidfd SIGKILL.

Usage:
  python -m agentfork.sandbox.fc_bench --fc ./firecracker \
      --kernel vmlinux --rootfs rootfs.ext4 --children 10

Requires /dev/kvm access. Results print as JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class JailerConfig:
    """Run each VMM under Firecracker's ``jailer``: chroot, cgroup-ready,
    privileges dropped to ``uid``/``gid``. The supervisor itself must run as
    root (jailer chroots and setuids), and kernel/rootfs must be readable by
    ``uid`` after being hard-linked into the jail. One ``chroot_base`` must
    not be shared by two sandboxes that can mint the same branch IDs."""

    jailer_bin: str
    uid: int
    gid: int
    chroot_base: str = "/srv/jailer"


def jail_id_for(vm_dir: str) -> str:
    """Jailer IDs allow only alphanumerics and hyphens, max 64 chars.

    Sanitizing can merge distinct names (``a_b`` and ``a-b``) and truncation
    can merge long ones — and two branches sharing one jail chroot silently
    clobber each other's snapshot and overlay files. Whenever the name had
    to be altered, a short digest of the original is appended so distinct
    inputs keep distinct jail IDs. The result never starts with ``-``: the
    jailer's CLI would parse a leading-dash ``--id`` value as another flag
    and refuse to start (a real branch dir like ``__agentfork__…`` sanitizes
    to a ``--…`` prefix)."""
    name = os.path.basename(vm_dir)
    sanitized = re.sub(r"[^A-Za-z0-9-]", "-", name)
    if sanitized == name and len(sanitized) <= 64 and not name.startswith("-"):
        return sanitized
    digest = hashlib.sha256(name.encode()).hexdigest()[:8]
    core = sanitized.lstrip("-")[:55] or "b"
    return f"{core}-{digest}"


def jail_root(jailer: JailerConfig, fc_bin: str, vm_dir: str) -> str:
    """The chroot directory the jailer builds for this VM: every per-VM file
    (API socket, vsock UDS, overlay, snapshots) lives here when jailed."""
    return os.path.join(jailer.chroot_base,
                        os.path.basename(os.path.abspath(fc_bin)),
                        jail_id_for(vm_dir), "root")


def jailer_argv(jailer: JailerConfig, fc_bin: str, vm_dir: str,
                netns: str | None = None) -> list[str]:
    argv = [os.path.abspath(jailer.jailer_bin),
            "--id", jail_id_for(vm_dir),
            "--exec-file", os.path.abspath(fc_bin),
            "--uid", str(jailer.uid), "--gid", str(jailer.gid),
            "--chroot-base-dir", jailer.chroot_base]
    if netns is not None:  # jailer joins the namespace itself
        argv += ["--netns", netns]
    return argv + ["--", "--api-sock", "fc.sock"]


class _UDSHTTP:
    """Minimal HTTP-over-unix-socket client for the Firecracker API."""

    def __init__(self, sock_path: str):
        self.sock_path = sock_path

    def _connect(self) -> socket.socket:
        # the socket file appears at bind() but connects succeed only after
        # listen(); a request racing that startup window sees ECONNREFUSED
        last: Exception | None = None
        for _ in range(200):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            try:
                sock.connect(self.sock_path)
                return sock
            except (ConnectionRefusedError, FileNotFoundError) as exc:
                sock.close()
                last = exc
                time.sleep(0.005)
        raise last

    def request(self, method: str, path: str, body: dict | None = None) -> int:
        payload = json.dumps(body).encode() if body is not None else b""
        req = (f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n"
               f"Content-Type: application/json\r\n"
               f"Content-Length: {len(payload)}\r\n\r\n").encode() + payload
        with self._connect() as sock:
            sock.sendall(req)
            response = http.client.HTTPResponse(sock)
            response.begin()
            response.read()
            return response.status


class MicroVM:
    """One Firecracker VMM.

    The VMM runs with ``cwd=vm_dir`` and every per-VM path (API socket,
    vsock UDS, overlay drive) is configured *relative*. Relative paths are
    resolved against the VMM's own cwd, and device paths are recorded
    verbatim in snapshots — so a child restored in its own directory from a
    parent's snapshot binds its own ``v.sock`` and opens its own copy of
    ``overlay.ext4`` instead of colliding on the parent's files. Shared
    read-only inputs (kernel, rootfs) are passed absolute for the same
    reason: every VM must resolve the same file.
    """

    def __init__(self, fc_bin: str, vm_dir: str,
                 jailer: JailerConfig | None = None,
                 netns: str | None = None):
        self.vm_dir = vm_dir
        self.jailer = jailer
        self.netns = netns
        # jailed VMMs live in the jailer-built chroot; per-VM paths resolve
        # there instead of vm_dir (which keeps host-side logs and pid files)
        self.host_dir = jail_root(jailer, fc_bin, vm_dir) if jailer else vm_dir
        if jailer:
            os.makedirs(self.host_dir, exist_ok=True)
            argv = jailer_argv(jailer, fc_bin, vm_dir,
                               netns=f"/var/run/netns/{netns}" if netns else None)
        else:
            argv = [os.path.abspath(fc_bin), "--api-sock", "fc.sock"]
            if netns is not None:
                # unjailed: enter the netns via `ip netns exec` (the jailer
                # does this itself with --netns, so only the plain path needs it)
                argv = ["ip", "netns", "exec", netns] + argv
        self.sock = os.path.join(self.host_dir, "fc.sock")
        # output goes to a file, not DEVNULL: a VMM that dies at startup is
        # otherwise undiagnosable
        with open(os.path.join(vm_dir, "fc.log"), "wb") as log:
            self.proc = subprocess.Popen(
                argv, cwd=vm_dir, stdout=log, stderr=subprocess.STDOUT)
        self.pidfd = None
        try:
            self.pidfd = os.pidfd_open(self.proc.pid)
            for _ in range(200):
                if os.path.exists(self.sock):
                    break
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"Firecracker exited with status {self.proc.returncode}")
                time.sleep(0.005)
            else:
                raise TimeoutError(
                    f"Firecracker API socket did not appear: {self.sock}")
        except BaseException:
            self._force_cleanup()
            raise
        self.api = _UDSHTTP(self.sock)

    def _force_cleanup(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        if self.pidfd is not None:
            os.close(self.pidfd)
            self.pidfd = None

    def _request(self, method: str, path: str, body: dict) -> None:
        status = self.api.request(method, path, body)
        if not 200 <= status < 300:
            raise RuntimeError(
                f"Firecracker {method} {path} returned HTTP {status}")

    def boot(self, kernel: str, rootfs: str, overlay: str | None = None,
             vsock_uds: str | None = None, tap: str | None = None) -> float:
        """Configure and start the guest.

        ``overlay`` and ``vsock_uds`` must be paths relative to ``vm_dir``
        (e.g. ``"overlay.ext4"``, ``"v.sock"``): they are per-VM and their
        configured strings survive into snapshots, where a restoring child
        must resolve its own copies. Kernel and rootfs are absolutized:
        they are shared, read-only, and must mean the same file from every
        VM's cwd.
        """
        for rel in (overlay, vsock_uds):
            if rel is not None and os.path.isabs(rel):
                raise ValueError(f"per-VM path must be relative: {rel}")
        t0 = time.perf_counter()
        self._request("PUT", "/boot-source", {
            "kernel_image_path": self._shared_input(kernel, "vmlinux"),
            "boot_args": "console=ttyS0 reboot=k panic=1 pci=off quiet"})
        self._request("PUT", "/drives/rootfs", {
            "drive_id": "rootfs",
            "path_on_host": self._shared_input(rootfs, "rootfs.img"),
            "is_root_device": True, "is_read_only": True})
        if overlay is not None:
            self._request("PUT", "/drives/overlay", {
                "drive_id": "overlay", "path_on_host": overlay,
                "is_root_device": False, "is_read_only": False})
        if vsock_uds is not None:
            self._request("PUT", "/vsock", {
                "guest_cid": 3, "uds_path": vsock_uds})
        if tap is not None:
            # the tap lives in the VMM's netns; guest MAC/IP are identical
            # across clones because each runs in its own namespace
            from agentfork.sandbox.netns import GUEST_MAC
            self._request("PUT", "/network-interfaces/eth0", {
                "iface_id": "eth0", "host_dev_name": tap,
                "guest_mac": GUEST_MAC})
        self._request("PUT", "/machine-config", {
            "vcpu_count": 1, "mem_size_mib": 256})
        self._request("PUT", "/actions", {"action_type": "InstanceStart"})
        return time.perf_counter() - t0

    def pause(self) -> None:
        self._request("PATCH", "/vm", {"state": "Paused"})

    def resume(self) -> None:
        self._request("PATCH", "/vm", {"state": "Resumed"})

    def _shared_input(self, path: str, jail_name: str) -> str:
        """Resolve a shared read-only input (kernel, rootfs) for the VMM.
        Unjailed: absolutize. Jailed: the VMM cannot see outside its chroot,
        so hard-link (or copy, across filesystems) the file into the jail
        and return the chroot-relative name."""
        if self.jailer is None:
            return os.path.abspath(path)
        dst = os.path.join(self.host_dir, jail_name)
        if not os.path.exists(dst):
            try:
                os.link(os.path.abspath(path), dst)
            except OSError:
                shutil.copyfile(path, dst)
        return jail_name

    def _vm_path(self, host_path: str) -> str:
        """Translate a host-side path to what the VMM should be told.
        Unjailed: absolutize (the VMM's cwd is vm_dir, not the caller's).
        Jailed: the path must live in the chroot; make it chroot-relative."""
        if self.jailer is None:
            return os.path.abspath(host_path)
        rel = os.path.relpath(os.path.abspath(host_path), self.host_dir)
        if rel.startswith(".."):
            raise ValueError(f"path outside jail chroot: {host_path}")
        return rel

    def snapshot(self, mem_path: str, state_path: str) -> float:
        t0 = time.perf_counter()
        self._request("PUT", "/snapshot/create", {
            "snapshot_type": "Full",
            "snapshot_path": self._vm_path(state_path),
            "mem_file_path": self._vm_path(mem_path)})
        return time.perf_counter() - t0

    def restore(self, mem_path: str, state_path: str) -> float:
        t0 = time.perf_counter()
        self._request("PUT", "/snapshot/load", {
            "snapshot_path": self._vm_path(state_path),
            "mem_backend": {"backend_type": "File",
                            "backend_path": self._vm_path(mem_path)},
            "resume_vm": True})
        return time.perf_counter() - t0

    def kill(self) -> tuple[float, float]:
        """Returns (signal_s, reaped_s): signal_s measures signal submission;
        reap includes confirmed exit and full VMM teardown."""
        if self.pidfd is None:
            raise RuntimeError("microVM is already reaped")
        t0 = time.perf_counter()
        try:
            try:
                signal.pidfd_send_signal(self.pidfd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            t1 = time.perf_counter()
            try:
                os.waitid(os.P_PIDFD, self.pidfd, os.WEXITED)
            except ChildProcessError:
                pass
            self.proc.wait()
            return t1 - t0, time.perf_counter() - t0
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
            os.close(self.pidfd)
            self.pidfd = None


def run(fc_bin: str, kernel: str, rootfs: str, n_children: int) -> dict:
    if n_children < 1:
        raise ValueError("n_children must be at least 1")
    results: dict = {"host": os.uname().release, "children": n_children}
    parent, kids = None, []
    with tempfile.TemporaryDirectory() as td:
        try:
            parent = MicroVM(fc_bin, td)
            results["cold_boot_s"] = round(parent.boot(kernel, rootfs), 4)
            time.sleep(2.0)  # let the guest settle
            t0 = time.perf_counter()
            parent.pause()
            mem, state = os.path.join(td, "mem"), os.path.join(td, "state")
            results["parent_pause_s"] = round(time.perf_counter() - t0, 4)
            results["snapshot_create_s"] = round(parent.snapshot(mem, state), 4)
            parent.kill()

            # N-way fanout from one snapshot (children mmap the same mem file)
            restores = []
            t_fan = time.perf_counter()
            for i in range(n_children):
                d = os.path.join(td, f"c{i}")
                os.mkdir(d)
                vm = MicroVM(fc_bin, d)
                kids.append(vm)
                restores.append(vm.restore(mem, state))
            results["fanout_total_s"] = round(time.perf_counter() - t_fan, 4)
            results["restore_p50_ms"] = round(
                sorted(restores)[len(restores) // 2] * 1000, 2)
            results["restore_max_ms"] = round(max(restores) * 1000, 2)

            # CoW proof: children mmap one snapshot file; PSS splits shared pages
            rss_kib = pss_kib = 0
            for vm in kids:
                with open(f"/proc/{vm.proc.pid}/smaps_rollup") as f:
                    for line in f:
                        if line.startswith("Rss:"):
                            rss_kib += int(line.split()[1])
                        elif line.startswith("Pss:"):
                            pss_kib += int(line.split()[1])
            results["children_rss_mib"] = round(rss_kib / 1024, 1)
            results["children_pss_mib"] = round(pss_kib / 1024, 1)
            results["mem_sharing_ratio"] = round(rss_kib / max(pss_kib, 1), 2)

            kills = [vm.kill() for vm in kids]
            sigs = sorted(k[0] for k in kills)
            reaps = sorted(k[1] for k in kills)
            results["kill_signal_p50_ms"] = round(sigs[len(sigs) // 2] * 1000, 3)
            results["kill_signal_max_ms"] = round(sigs[-1] * 1000, 3)
            results["kill_reaped_p50_ms"] = round(reaps[len(reaps) // 2] * 1000, 3)
            results["kill_reaped_max_ms"] = round(reaps[-1] * 1000, 3)
        finally:
            for vm in kids:
                if vm.pidfd is not None:
                    vm.kill()
            if parent is not None and parent.pidfd is not None:
                parent.kill()
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--rootfs", required=True)
    ap.add_argument("--children", type=int, default=10)
    a = ap.parse_args()
    print(json.dumps(run(a.fc, a.kernel, a.rootfs, a.children), indent=2))
