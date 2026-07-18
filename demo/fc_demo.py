"""Real-Firecracker demo: ForkOrchestrator driving FirecrackerSandbox.

Boots one root microVM, snapshots it, forks N children (each restored
copy-on-write from the parent's memory file), extends each child's KV
branch, kills the losers, and tears everything down. Prints per-phase
latencies and verifies no Firecracker process survives.

The KV half is the CPU reference TreeKVCache; the sandbox half is real.
Guests are idle Ubuntu microVMs: this demonstrates and times the branch
lifecycle, not a workload running inside the guests.

Requires Linux with /dev/kvm, a firecracker binary, and a matching guest
kernel + rootfs (see Firecracker's getting-started artifacts):

  python demo/fc_demo.py --fc ./firecracker --kernel vmlinux \
      --rootfs ubuntu.squashfs --children 5

With --exec CMD, the demo also runs CMD inside every child through the vsock
data plane and prints each child's exit code and first output line. That
path requires agentfork/sandbox/guest_agent.py to be running inside the
rootfs (e.g. started from init) and a guest kernel with virtio-vsock. Use
--overlay-mib N to give every branch a writable scratch drive (children
inherit a copy of the parent's; the guest must mount /dev/vdb itself), and
--no-vsock to reproduce the plain idle-guest lifecycle run on images without
the agent.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

from agentfork import ForkOrchestrator
from agentfork.sandbox.firecracker_backend import FirecrackerSandbox

PREFIX_TOKENS = 32_000
SUFFIX_TOKENS = 500


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fc", required=True, help="firecracker binary")
    parser.add_argument("--kernel", required=True, help="guest kernel image")
    parser.add_argument("--rootfs", required=True, help="guest rootfs image")
    parser.add_argument("--children", type=int, default=5)
    parser.add_argument("--work-dir", default="/tmp/agentfork-fc-demo")
    parser.add_argument("--exec", dest="exec_cmd", default=None,
                        help="shell command to run in every child via vsock "
                             "(needs guest_agent.py in the rootfs)")
    parser.add_argument("--overlay-mib", type=int, default=None,
                        help="give each branch a writable overlay drive")
    parser.add_argument("--no-vsock", action="store_true",
                        help="skip the vsock device (idle-guest lifecycle only)")
    parser.add_argument("--jailer", default=None,
                        help="jailer binary; chroots each VMM and drops it "
                             "to --jailer-uid/gid (demo must run as root)")
    parser.add_argument("--jailer-uid", type=int, default=None)
    parser.add_argument("--jailer-gid", type=int, default=None)
    parser.add_argument("--network-uplink", default=None,
                        help="host interface to NAT guests out of (enables "
                             "per-branch netns networking; needs root)")
    args = parser.parse_args()

    if not os.path.exists("/dev/kvm"):
        print("error: /dev/kvm not present; Firecracker needs KVM", file=sys.stderr)
        return 1
    if args.exec_cmd and args.no_vsock:
        print("error: --exec needs vsock", file=sys.stderr)
        return 1

    jailer = None
    if args.jailer:
        from agentfork.sandbox.fc_bench import JailerConfig
        if args.jailer_uid is None or args.jailer_gid is None:
            print("error: --jailer needs --jailer-uid and --jailer-gid",
                  file=sys.stderr)
            return 1
        jailer = JailerConfig(
            jailer_bin=args.jailer, uid=args.jailer_uid, gid=args.jailer_gid,
            chroot_base=os.path.join(args.work_dir, "jail"))

    network = None
    if args.network_uplink:
        from agentfork.sandbox.netns import NetworkConfig
        network = NetworkConfig(uplink=args.network_uplink)

    shutil.rmtree(args.work_dir, ignore_errors=True)
    os.makedirs(args.work_dir)
    sandbox = FirecrackerSandbox(args.fc, args.kernel, args.rootfs,
                                 args.work_dir, vsock=not args.no_vsock,
                                 overlay_mib=args.overlay_mib, jailer=jailer,
                                 network=network)
    registry = os.path.join(args.work_dir, "registry.json")

    with ForkOrchestrator(sandbox=sandbox, registry_path=registry) as orch:
        t0 = time.perf_counter()
        orch.create_parent("root", tokens=list(range(PREFIX_TOKENS)))
        root_ms = (time.perf_counter() - t0) * 1000
        print(f"root: boot                                   {root_ms:8.1f} ms")

        if args.exec_cmd:
            # children snapshot whatever state the parent is in; wait for
            # its userspace so they inherit a booted, agent-serving guest
            t0 = time.perf_counter()
            sandbox.wait_ready("root")
            ready_ms = (time.perf_counter() - t0) * 1000
            print(f"root: guest agent ready                      {ready_ms:8.1f} ms")

        t0 = time.perf_counter()
        children = orch.fork("root", n=args.children)
        fork_ms = (time.perf_counter() - t0) * 1000
        print(f"fork {args.children} children (restore each, CoW memory) "
              f"{fork_ms:8.1f} ms total, {fork_ms / args.children:.1f} ms avg")

        for i, child in enumerate(children):
            start = 1_000_000 + i * SUFFIX_TOKENS
            charged = orch.extend(child.branch_id,
                                  list(range(start, start + SUFFIX_TOKENS)))
            assert charged == SUFFIX_TOKENS  # suffix is new; prefix was shared
        assert all(orch.alive(c.branch_id) for c in children)

        if args.exec_cmd:
            for child in children:
                t0 = time.perf_counter()
                result = orch.exec(child.branch_id,
                                   ["/bin/sh", "-c", args.exec_cmd],
                                   timeout_s=30.0)
                exec_ms = (time.perf_counter() - t0) * 1000
                first = result.stdout.decode(errors="replace").splitlines()
                print(f"exec in {child.branch_id}: exit {result.exit_code} "
                      f"in {exec_ms:6.1f} ms  {first[0] if first else ''}")
        print(f"KV: {args.children} children share one {PREFIX_TOKENS}-token "
              f"prefix, {SUFFIX_TOKENS} own tokens each")

        winner = children[0].branch_id
        t0 = time.perf_counter()
        receipts = orch.kill_losers(winner)
        kill_ms = (time.perf_counter() - t0) * 1000
        print(f"kill {len(receipts)} losers                            "
              f"{kill_ms:8.1f} ms, "
              f"{sum(r.kv_freed_tokens for r in receipts)} KV tokens freed")
        assert orch.alive(winner)
        print(f"winner {winner} still alive; closing")

    leftovers = subprocess.run(
        ["pgrep", "-x", os.path.basename(args.fc)],
        capture_output=True, text=True)
    if leftovers.returncode == 0:
        print(f"error: surviving firecracker pids: {leftovers.stdout.split()}",
              file=sys.stderr)
        return 1
    print("no surviving firecracker processes; registry empty")
    return 0


if __name__ == "__main__":
    sys.exit(main())
