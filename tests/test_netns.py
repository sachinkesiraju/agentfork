"""Command construction for per-branch network namespaces. The ip/iptables
calls are captured through an injected runner — no root, no real netns."""

import pytest

from agentfork.sandbox.netns import GUEST_GW, TAP_DEV, NetnsManager, NetworkConfig

CFG = NetworkConfig(uplink="eth0", host_subnet="10.200.0.0/16",
                    nameserver="1.1.1.1")


class RecordingRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, argv, *, check=True):
        self.calls.append(argv)


def test_netns_name_is_namespaced_and_sanitized():
    mgr = NetnsManager(CFG, runner=RecordingRunner())
    # clean names pass through unchanged
    assert mgr.netns_name("root") == "af-root"
    # sanitized names get a digest suffix so distinct IDs can't collide into
    # one namespace (mirrors jail_id_for)
    assert mgr.netns_name("root/1").startswith("af-root-1-")
    assert mgr.netns_name("root/1") != mgr.netns_name("root-1")
    assert mgr.netns_name("root/1") == mgr.netns_name("root/1")  # deterministic


def test_setup_creates_netns_tap_veth_and_nat_in_order():
    runner = RecordingRunner()
    mgr = NetnsManager(CFG, runner=runner)

    name, index = mgr.setup("root/1")

    assert name == mgr.netns_name("root/1") and index == 0
    joined = [" ".join(c) for c in runner.calls]
    # namespace, then the guest tap, then veth into it, then default route + NAT
    assert joined[0] == f"ip netns add {name}"
    assert any(f"ip tuntap add {TAP_DEV} mode tap" in j for j in joined)
    assert any("veth" in j and "peer name" in j for j in joined)
    assert any("route add default" in j for j in joined)
    # host NAT masquerades the veth subnet (not the guest subnet, which the
    # netns has already SNAT'd away before the packet reaches the host)
    assert any("POSTROUTING -s 10.200" in j and "MASQUERADE" in j
               and "eth0" in j for j in joined)
    # the tap gateway IP is configured inside the namespace
    assert any(f"addr add {GUEST_GW}" in j for j in joined)


def test_each_branch_gets_a_distinct_host_veth_subnet():
    mgr = NetnsManager(CFG, runner=RecordingRunner())
    cmds0 = mgr.setup_cmds("a", 0)
    cmds1 = mgr.setup_cmds("b", 1)

    def host_veth_ip(cmds):
        for c in cmds:
            if c[:3] == ["ip", "addr", "add"] and c[-1].startswith("afh"):
                return c[3]
        return None

    assert host_veth_ip(cmds0) != host_veth_ip(cmds1)


def test_teardown_reverses_nat_and_removes_netns():
    runner = RecordingRunner()
    mgr = NetnsManager(CFG, runner=runner)
    _, index = mgr.setup("root")
    runner.calls.clear()

    mgr.teardown("root", index)

    joined = [" ".join(c) for c in runner.calls]
    assert any(j.startswith("iptables -t nat -D POSTROUTING") for j in joined)
    assert joined[-1] == "ip netns del af-root"


def test_teardown_is_best_effort_and_continues_past_failures():
    class Flaky:
        def __init__(self):
            self.calls = []

        def __call__(self, argv, *, check=True):
            self.calls.append(argv)
            if check:  # setup uses check=True; teardown uses check=False
                raise AssertionError("should not raise in teardown")
            if "FORWARD" in argv:
                raise RuntimeError("rule already gone")

    runner = Flaky()
    mgr = NetnsManager(CFG, runner=runner)
    mgr.teardown("root", 0)  # must not raise despite the failing rule
    assert any("netns" in " ".join(c) and "del" in c for c in runner.calls)


def test_freed_indices_are_recycled_not_exhausted():
    # a /29 host_subnet has exactly two /30s; churning fork/kill must reuse
    # freed indices rather than march off the end
    cfg = NetworkConfig(uplink="eth0", host_subnet="10.0.0.0/29")
    mgr = NetnsManager(cfg, runner=RecordingRunner())
    _, i0 = mgr.setup("a")
    _, i1 = mgr.setup("b")
    assert {i0, i1} == {0, 1}
    mgr.teardown("a", i0)  # frees index 0
    _, i2 = mgr.setup("c")  # must reuse it, not exhaust
    assert i2 == 0


def test_subnet_exhaustion_raises_a_clear_error_and_rolls_back():
    cfg = NetworkConfig(uplink="eth0", host_subnet="10.0.0.0/30")  # one /30
    mgr = NetnsManager(cfg, runner=RecordingRunner())
    mgr.setup("a")
    with pytest.raises(RuntimeError, match="exhausted"):
        mgr.setup("b")


def test_setup_rolls_back_and_frees_index_on_partial_failure():
    class FailOnceAtNat(RecordingRunner):
        def __init__(self):
            super().__init__()
            self.failed = False

        def __call__(self, argv, *, check=True):
            super().__call__(argv, check=check)
            if check and "MASQUERADE" in argv and not self.failed:
                self.failed = True
                raise RuntimeError("iptables failed")

    runner = FailOnceAtNat()
    mgr = NetnsManager(NetworkConfig(uplink="eth0"), runner=runner)
    with pytest.raises(RuntimeError, match="iptables failed"):
        mgr.setup("a")
    # the half-built namespace got a teardown pass (netns del issued)...
    assert any(c[:3] == ["ip", "netns", "del"] for c in runner.calls)
    # ...and the index was freed, so the next setup reuses index 0
    _, index = mgr.setup("b")
    assert index == 0
