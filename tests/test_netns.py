"""Command construction for per-branch network namespaces. The ip/iptables
calls are captured through an injected runner — no root, no real netns."""

from agentfork.sandbox.netns import (
    GUEST_GW,
    TAP_DEV,
    NetnsManager,
    NetworkConfig,
    netns_exec_prefix,
)

CFG = NetworkConfig(uplink="eth0", host_subnet="10.200.0.0/16",
                    nameserver="1.1.1.1")


class RecordingRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, argv, *, check=True):
        self.calls.append(argv)


def test_netns_name_is_namespaced_and_sanitized():
    mgr = NetnsManager(CFG, runner=RecordingRunner())
    assert mgr.netns_name("root") == "af-root"
    assert mgr.netns_name("root/1") == "af-root-1"


def test_setup_creates_netns_tap_veth_and_nat_in_order():
    runner = RecordingRunner()
    mgr = NetnsManager(CFG, runner=runner)

    name, index = mgr.setup("root/1")

    assert name == "af-root-1" and index == 0
    joined = [" ".join(c) for c in runner.calls]
    # namespace, then the guest tap, then veth into it, then default route + NAT
    assert joined[0] == "ip netns add af-root-1"
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


def test_netns_exec_prefix():
    assert netns_exec_prefix("af-x") == ["ip", "netns", "exec", "af-x"]
