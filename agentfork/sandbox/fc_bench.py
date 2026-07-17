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
import http.client
import json
import os
import signal
import socket
import subprocess
import tempfile
import time


class _UDSHTTP:
    """Minimal HTTP-over-unix-socket client for the Firecracker API."""

    def __init__(self, sock_path: str):
        self.sock_path = sock_path

    def request(self, method: str, path: str, body: dict | None = None) -> int:
        payload = json.dumps(body).encode() if body is not None else b""
        req = (f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n"
               f"Content-Type: application/json\r\n"
               f"Content-Length: {len(payload)}\r\n\r\n").encode() + payload
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            sock.connect(self.sock_path)
            sock.sendall(req)
            response = http.client.HTTPResponse(sock)
            response.begin()
            response.read()
            return response.status


class MicroVM:
    def __init__(self, fc_bin: str, vm_dir: str):
        self.vm_dir = vm_dir
        self.sock = os.path.join(vm_dir, "fc.sock")
        self.proc = subprocess.Popen(
            [fc_bin, "--api-sock", self.sock],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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

    def boot(self, kernel: str, rootfs: str) -> float:
        t0 = time.perf_counter()
        self._request("PUT", "/boot-source", {
            "kernel_image_path": kernel,
            "boot_args": "console=ttyS0 reboot=k panic=1 pci=off quiet"})
        self._request("PUT", "/drives/rootfs", {
            "drive_id": "rootfs", "path_on_host": rootfs,
            "is_root_device": True, "is_read_only": True})
        self._request("PUT", "/machine-config", {
            "vcpu_count": 1, "mem_size_mib": 256})
        self._request("PUT", "/actions", {"action_type": "InstanceStart"})
        return time.perf_counter() - t0

    def pause(self) -> None:
        self._request("PATCH", "/vm", {"state": "Paused"})

    def snapshot(self, mem_path: str, state_path: str) -> float:
        t0 = time.perf_counter()
        self._request("PUT", "/snapshot/create", {
            "snapshot_type": "Full",
            "snapshot_path": state_path, "mem_file_path": mem_path})
        return time.perf_counter() - t0

    def restore(self, mem_path: str, state_path: str) -> float:
        t0 = time.perf_counter()
        self._request("PUT", "/snapshot/load", {
            "snapshot_path": state_path,
            "mem_backend": {"backend_type": "File", "backend_path": mem_path},
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
