"""Firecracker microVM fork benchmark (Gate G4 evidence).

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
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self.sock_path)
        payload = json.dumps(body).encode() if body is not None else b""
        req = (f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n"
               f"Content-Type: application/json\r\n"
               f"Content-Length: {len(payload)}\r\n\r\n").encode() + payload
        s.sendall(req)
        resp = s.recv(4096).decode()
        s.close()
        return int(resp.split(" ")[1])


class MicroVM:
    def __init__(self, fc_bin: str, vm_dir: str):
        self.vm_dir = vm_dir
        self.sock = os.path.join(vm_dir, "fc.sock")
        self.proc = subprocess.Popen(
            [fc_bin, "--api-sock", self.sock],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.pidfd = os.pidfd_open(self.proc.pid)
        for _ in range(200):
            if os.path.exists(self.sock):
                break
            time.sleep(0.005)
        self.api = _UDSHTTP(self.sock)

    def boot(self, kernel: str, rootfs: str) -> float:
        t0 = time.perf_counter()
        assert self.api.request("PUT", "/boot-source", {
            "kernel_image_path": kernel,
            "boot_args": "console=ttyS0 reboot=k panic=1 pci=off quiet"}) < 300
        assert self.api.request("PUT", "/drives/rootfs", {
            "drive_id": "rootfs", "path_on_host": rootfs,
            "is_root_device": True, "is_read_only": True}) < 300
        assert self.api.request("PUT", "/machine-config", {
            "vcpu_count": 1, "mem_size_mib": 256}) < 300
        assert self.api.request("PUT", "/actions",
                                {"action_type": "InstanceStart"}) < 300
        return time.perf_counter() - t0

    def pause(self) -> None:
        assert self.api.request("PATCH", "/vm", {"state": "Paused"}) < 300

    def snapshot(self, mem_path: str, state_path: str) -> float:
        t0 = time.perf_counter()
        assert self.api.request("PUT", "/snapshot/create", {
            "snapshot_type": "Full",
            "snapshot_path": state_path, "mem_file_path": mem_path}) < 300
        return time.perf_counter() - t0

    def restore(self, mem_path: str, state_path: str) -> float:
        t0 = time.perf_counter()
        assert self.api.request("PUT", "/snapshot/load", {
            "snapshot_path": state_path,
            "mem_backend": {"backend_type": "File", "backend_path": mem_path},
            "resume_vm": True}) < 300
        return time.perf_counter() - t0

    def kill(self) -> tuple[float, float]:
        """Returns (signal_s, reaped_s): signal delivery is the moment the VM
        stops consuming GPU/CPU; reap includes full VMM teardown."""
        t0 = time.perf_counter()
        signal.pidfd_send_signal(self.pidfd, signal.SIGKILL)
        t1 = time.perf_counter()
        os.waitid(os.P_PIDFD, self.pidfd, os.WEXITED)
        self.proc.wait()
        os.close(self.pidfd)
        return t1 - t0, time.perf_counter() - t0


def run(fc_bin: str, kernel: str, rootfs: str, n_children: int) -> dict:
    results: dict = {"host": os.uname().release, "children": n_children}
    with tempfile.TemporaryDirectory() as td:
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
        kids, restores = [], []
        t_fan = time.perf_counter()
        for i in range(n_children):
            d = os.path.join(td, f"c{i}")
            os.mkdir(d)
            vm = MicroVM(fc_bin, d)
            restores.append(vm.restore(mem, state))
            kids.append(vm)
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
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--rootfs", required=True)
    ap.add_argument("--children", type=int, default=10)
    a = ap.parse_args()
    print(json.dumps(run(a.fc, a.kernel, a.rootfs, a.children), indent=2))
