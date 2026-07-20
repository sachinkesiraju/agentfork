"""Per-branch network namespaces so snapshot clones don't collide.

N microVMs restored from one snapshot share the recorded network config: the
same tap name, the same guest MAC, the same guest IP. On one host network
that is an instant conflict. The fix Firecracker documents is one network
namespace per clone: inside each netns the tap and guest IP are *identical*
(so the snapshot's recorded devices still resolve), but the namespaces are
isolated, so no two clones fight over the same tap or address. A veth pair
bridges each netns to the host, and the host masquerades the shared guest
subnet out a real interface.

    host uplink ── (NAT) ── veth-h │ veth-g ── netns ── tap0 ── guest eth0
                                    │ 172.16.0.1        172.16.0.2

Every guest believes it is 172.16.0.2 behind gateway 172.16.0.1; the host
side of each veth gets a unique /30 so return traffic routes to the right
namespace. This requires root (or CAP_NET_ADMIN) and ``iptables``/``ip``;
without them, run with ``NetworkConfig`` unset and guests stay offline.

The ``ip``/``iptables`` command *construction* here is pure and unit-tested
(tests/test_netns.py); a live NAT path is exercised only in the Firecracker
demo on a KVM host.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import subprocess
import threading
from dataclasses import dataclass

_log = logging.getLogger("agentfork.sandbox.netns")

TAP_DEV = "tap0"
GUEST_GW = "172.16.0.1"   # tap IP inside every netns (the guest's gateway)
GUEST_MAC = "06:00:ac:10:00:02"
GUEST_MASK = 30


@dataclass(frozen=True)
class NetworkConfig:
    """Host-side networking for guests. ``uplink`` is the interface that
    reaches the outside world (``eth0``, ``ens5`` …); ``host_subnet`` is a
    private range carved into a /30 per branch for the host↔netns veths.
    ``nameserver`` is written into the guest's resolv.conf by the rootfs."""

    uplink: str
    host_subnet: str = "10.200.0.0/16"
    nameserver: str = "1.1.1.1"


def _run(argv: list[str], *, check: bool = True) -> None:
    _log.debug("net: %s", " ".join(argv))
    subprocess.run(argv, check=check, capture_output=True)


class NetnsManager:
    """Creates and destroys a network namespace per branch. Command building
    is split into pure ``*_cmds`` methods for testing; ``setup``/``teardown``
    run them."""

    def __init__(self, config: NetworkConfig, runner=_run):
        self.config = config
        self._run = runner
        # one /30 per branch; the index space is finite, so freed indices are
        # recycled (a long-running fork/kill churn would otherwise exhaust it)
        self._capacity = ipaddress.ip_network(config.host_subnet).num_addresses // 4
        self._alloc_lock = threading.Lock()
        self._free: list[int] = []
        self._next = 0

    def _allocate(self) -> int:
        with self._alloc_lock:
            if self._free:
                return self._free.pop()
            if self._next >= self._capacity:
                raise RuntimeError(
                    f"network subnet {self.config.host_subnet} exhausted "
                    f"({self._capacity} concurrent branches); free some or "
                    "widen host_subnet")
            index = self._next
            self._next += 1
            return index

    def release(self, index: int) -> None:
        with self._alloc_lock:
            if index not in self._free:
                self._free.append(index)

    def reserve(self, index: int) -> None:
        """Mark ``index`` as already in use, e.g. recovered from a journal
        after a restart, so it is never handed to a new branch that would
        collide with a surviving VMM's /30. Indices skipped below the cursor
        are added to the free list so they are still recyclable."""
        if index < 0:
            raise ValueError(f"index must be non-negative: {index}")
        with self._alloc_lock:
            if index < self._next:
                if index in self._free:
                    self._free.remove(index)
                return
            self._free.extend(
                i for i in range(self._next, index) if i not in self._free)
            self._next = index + 1

    def netns_name(self, branch_id: str) -> str:
        # ip netns names: keep them short and filesystem-safe. Sanitizing or
        # truncating can merge distinct branch IDs, so append a digest of the
        # original whenever the name is altered (mirrors jail_id_for).
        safe = "".join(c if c.isalnum() else "-" for c in branch_id)
        if safe == branch_id and len(safe) <= 40:
            return f"af-{safe}"
        digest = hashlib.sha256(branch_id.encode()).hexdigest()[:8]
        return f"af-{safe[:31]}-{digest}"

    def _veth_pair(self, index: int) -> tuple[str, str, str, str]:
        """(host_veth, netns_veth, host_ip, netns_ip) for the /30 at index."""
        net = list(ipaddress.ip_network(self.config.host_subnet).subnets(
            new_prefix=30))[index]
        hosts = list(net.hosts())
        return (f"afh{index}", f"afg{index}", str(hosts[0]), str(hosts[1]))

    def _veth_net(self, index: int) -> str:
        return str(list(ipaddress.ip_network(self.config.host_subnet).subnets(
            new_prefix=30))[index])

    def setup_cmds(self, branch_id: str, index: int) -> list[list[str]]:
        """Every command to stand up the netns, tap, veth, and NAT for one
        branch. Ordered; idempotent teardown undoes them.

        Two-stage NAT: inside the netns the guest (172.16.0.2) is SNAT'd to
        the veth IP, so the host — which only knows the veth /30, not the
        guest subnet behind it — can route return traffic back. The host
        then SNATs the veth /30 out the uplink. A host rule matching the
        guest subnet would never fire: by the time the packet reaches the
        host it already carries the veth source.
        """
        ns = self.netns_name(branch_id)
        h_veth, g_veth, h_ip, g_ip = self._veth_pair(index)
        veth_net = self._veth_net(index)
        up = self.config.uplink
        return [
            ["ip", "netns", "add", ns],
            # tap the guest attaches to (same name/IP in every namespace)
            ["ip", "netns", "exec", ns, "ip", "tuntap", "add", TAP_DEV,
             "mode", "tap"],
            ["ip", "netns", "exec", ns, "ip", "addr", "add",
             f"{GUEST_GW}/{GUEST_MASK}", "dev", TAP_DEV],
            ["ip", "netns", "exec", ns, "ip", "link", "set", TAP_DEV, "up"],
            # veth: host side stays in root ns, peer moves into the netns
            ["ip", "link", "add", h_veth, "type", "veth", "peer", "name",
             g_veth],
            ["ip", "link", "set", g_veth, "netns", ns],
            ["ip", "addr", "add", f"{h_ip}/30", "dev", h_veth],
            ["ip", "link", "set", h_veth, "up"],
            ["ip", "netns", "exec", ns, "ip", "addr", "add", f"{g_ip}/30",
             "dev", g_veth],
            ["ip", "netns", "exec", ns, "ip", "link", "set", g_veth, "up"],
            ["ip", "netns", "exec", ns, "ip", "link", "set", "lo", "up"],
            # default route out of the netns, then NAT the guest subnet
            ["ip", "netns", "exec", ns, "ip", "route", "add", "default",
             "via", h_ip],
            ["ip", "netns", "exec", ns, "iptables", "-t", "nat", "-A",
             "POSTROUTING", "-o", g_veth, "-j", "MASQUERADE"],
            ["iptables", "-t", "nat", "-A", "POSTROUTING", "-s", veth_net,
             "-o", up, "-j", "MASQUERADE"],
            ["iptables", "-A", "FORWARD", "-i", h_veth, "-o", up, "-j",
             "ACCEPT"],
            ["iptables", "-A", "FORWARD", "-i", up, "-o", h_veth, "-m",
             "state", "--state", "RELATED,ESTABLISHED", "-j", "ACCEPT"],
        ]

    def teardown_cmds(self, branch_id: str, index: int) -> list[list[str]]:
        ns = self.netns_name(branch_id)
        h_veth, _, _, _ = self._veth_pair(index)
        veth_net = self._veth_net(index)
        up = self.config.uplink
        # reverse the host NAT/forward rules, then drop the veth and netns
        # (the netns-internal rules die with the namespace)
        return [
            ["iptables", "-D", "FORWARD", "-i", up, "-o", h_veth, "-m",
             "state", "--state", "RELATED,ESTABLISHED", "-j", "ACCEPT"],
            ["iptables", "-D", "FORWARD", "-i", h_veth, "-o", up, "-j",
             "ACCEPT"],
            ["iptables", "-t", "nat", "-D", "POSTROUTING", "-s", veth_net,
             "-o", up, "-j", "MASQUERADE"],
            ["ip", "link", "del", h_veth],
            ["ip", "netns", "del", ns],
        ]

    def setup(self, branch_id: str) -> tuple[str, int]:
        """Build the netns for a branch; return (netns_name, index). The
        index picks the branch's /30 and is needed to tear it down. On any
        partial failure the half-built namespace is rolled back and the index
        freed, so a failed setup leaks nothing and never wedges the caller."""
        index = self._allocate()
        try:
            for cmd in self.setup_cmds(branch_id, index):
                self._run(cmd)
        except BaseException:
            self.teardown(branch_id, index)  # frees the index too
            raise
        _log.info("netns %s up for branch %s", self.netns_name(branch_id),
                  branch_id)
        return self.netns_name(branch_id), index

    def teardown(self, branch_id: str, index: int) -> None:
        # best effort: a half-built netns must still tear down cleanly
        for cmd in self.teardown_cmds(branch_id, index):
            try:
                self._run(cmd, check=False)
            except Exception:
                _log.debug("net teardown step failed (continuing): %s", cmd)
        self.release(index)
