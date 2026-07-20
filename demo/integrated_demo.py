"""Integrated end-to-end demo: ONE ForkOrchestrator driving a LIVE inference
backend and a REAL sandbox under one branch ID -- the "fork N fixes" shape.

This closes the gap where each half (patched-SGLang KV cache, Firecracker
sandbox) had only been validated on its own: here a single orchestrator run
forks N candidate branches off a shared repo-context prompt, and for EACH
branch, under the same branch id, it:

  * runs a real ``generate()`` against a live HTTP SGLang tree-cache server
    (demo/sglang_tree_server.py -- the real patched TreeRadixCache + KV pool +
    admin auth over a socket, via agentfork.SGLangHTTPBackend), and
  * runs a real candidate command inside that branch's sandbox.

Then it kills the losers (tearing down BOTH halves per branch), keeps the
winner, exports the winner's artifact, and verifies the live server's KV
allocator reclaims back to its baseline.

Sandbox backends (``--sandbox``):

  firecracker  Real microVMs (needs Linux /dev/kvm, a firecracker binary, a
               guest kernel, and an agent rootfs built by tools/build_rootfs.sh
               -- see demo/fc_demo.py). Each branch is a real VM; commands run
               in-guest over vsock; the winner's artifact is tar'd out over the
               same channel. Usually needs root for /dev/kvm.

  reaper       Real per-branch subprocess via agentfork.ReaperSandbox (the
               shipped pidfd kill path). Use when Firecracker isn't runnable.
               ReaperSandbox runs one argv template and has no in-guest exec,
               so the per-candidate command + winner artifact are run/exported
               at the host level here and clearly labelled as such.

Example (Firecracker, as root):

    # 1. start the live tree-cache server (as your normal user)
    PYTHONPATH=/path/to/sglang/python python demo/sglang_tree_server.py \
        --port 30000 --admin-api-key admin-secret &
    # 2. run the integrated demo
    sudo .venv/bin/python demo/integrated_demo.py \
        --sglang-url http://127.0.0.1:30000 --admin-api-key admin-secret \
        --sandbox firecracker --fc ./firecracker --kernel ./vmlinux \
        --rootfs ./agent-rootfs.squashfs --children 10
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from agentfork import ForkOrchestrator, ReaperSandbox, SGLangHTTPBackend

# A stand-in for a real shared repo context: long enough that its cached reuse
# across children is a meaningful number of tokens.
SHARED_CONTEXT = (
    "# repo: agentfork\n"
    "# bug: kill_losers() must free each loser's KV hold and leave the\n"
    "#      allocator at baseline, while the winner's branch stays pinned.\n"
    "# task: each candidate proposes a one-line fix and proves it in its\n"
    "#       own sandbox; the orchestrator forks all candidates off this\n"
    "#       shared prefix so they reuse its cached KV instead of re-prefilling.\n"
) * 12


def pool_stats(base_url: str) -> dict:
    with urllib.request.urlopen(f"{base_url}/pool_stats", timeout=10) as r:
        return json.loads(r.read())


def candidate_prompt(i: int) -> str:
    # Distinct suffix per child so each loser pins (and later frees) its own
    # unique KV tokens -- otherwise identical branches share everything and a
    # kill frees zero.
    return (f"{SHARED_CONTEXT}\n# candidate {i}: proposed fix number {i}\n"
            f"# rationale token stream unique to branch {i}: "
            + f"fix{i}-" * 16 + "\n")


def candidate_argv(i: int, winner: int) -> list[str]:
    """A real command the branch's sandbox runs. The 'winner' candidate writes
    a durable artifact and exits 0; others do real work but 'fail' their check."""
    ok = "PASS" if i == winner else "FAIL"
    script = (
        "mkdir -p /tmp/agentfork && "
        f"echo 'candidate {i} building fix...' && "
        f"printf 'diff --git a/fix{i} b/fix{i}\\n+candidate {i} {ok}\\n' "
        f"> /tmp/agentfork/fix.diff && "
        f"echo 'candidate {i} check: {ok}' && "
        f"test '{ok}' = PASS"
    )
    return ["/bin/sh", "-c", script]


def build_sandbox(args):
    if args.sandbox == "firecracker":
        from agentfork.sandbox.firecracker_backend import FirecrackerSandbox
        if not (args.fc and args.kernel and args.rootfs):
            sys.exit("firecracker sandbox needs --fc, --kernel and --rootfs")
        if not os.path.exists("/dev/kvm"):
            sys.exit("firecracker sandbox needs /dev/kvm")
        shutil.rmtree(args.work_dir, ignore_errors=True)
        os.makedirs(args.work_dir)
        return FirecrackerSandbox(
            args.fc, args.kernel, args.rootfs, args.work_dir,
            vsock=True, overlay_mib=args.overlay_mib)
    return ReaperSandbox(["/bin/sh", "-c", "exec sleep 3600"])


def run(args) -> int:
    base = args.sglang_url.rstrip("/")
    kv = SGLangHTTPBackend(base, admin_api_key=args.admin_api_key)
    sandbox = build_sandbox(args)
    fc = args.sandbox == "firecracker"
    n = args.children
    winner_idx = n - 1  # last candidate is the one whose in-sandbox check PASSes

    baseline = pool_stats(base)
    print(f"live server pool baseline: used={baseline['used']} "
          f"available={baseline['available']}/{baseline['pool_tokens']}")

    registry = os.path.join(tempfile.mkdtemp(), "registry.json")
    host_art = {}  # reaper-only: per-branch host workspace

    with ForkOrchestrator(kv=kv, sandbox=sandbox, registry_path=registry) as orch:
        t0 = time.perf_counter()
        orch.create_parent("root")
        # populate the parent KV branch on the live server (long shared context)
        pr = orch.generate("root", SHARED_CONTEXT, {"max_new_tokens": 8})
        print(f"parent 'root': boot+prefill {(time.perf_counter()-t0)*1e3:7.1f} ms"
              f"  KV cached_tokens={pr['meta_info']['cached_tokens']}"
              f" prompt_tokens={pr['meta_info']['prompt_tokens']}")

        if fc:
            sandbox.wait_ready("root")

        t0 = time.perf_counter()
        children = orch.fork("root", n=n)
        fork_ms = (time.perf_counter() - t0) * 1e3
        print(f"fork {n} children (KV CoW + sandbox restore) "
              f"{fork_ms:7.1f} ms total, {fork_ms/n:5.1f} ms/branch\n")

        reuse = []
        for i, child in enumerate(children):
            bid = child.branch_id
            gr = orch.generate(bid, candidate_prompt(i), {"max_new_tokens": 64},
                               reserve_tokens=64)
            cached = gr["meta_info"]["cached_tokens"]
            charged = gr["meta_info"]["charged_tokens"]
            reuse.append(cached)

            t0 = time.perf_counter()
            if fc:
                res = orch.exec(bid, candidate_argv(i, winner_idx), timeout_s=30.0)
                code = res.exit_code
                out = res.stdout.decode(errors="replace").splitlines()
            else:
                work = tempfile.mkdtemp(prefix=f"cand{i}-")
                host_art[bid] = work
                res = subprocess.run(
                    candidate_argv(i, winner_idx), cwd=work,
                    capture_output=True, text=True,
                    env={**os.environ, "TMPDIR": work})
                # candidate writes to /tmp/agentfork; mirror into host workspace
                if os.path.exists("/tmp/agentfork/fix.diff"):
                    shutil.copy("/tmp/agentfork/fix.diff",
                                os.path.join(work, "fix.diff"))
                code = res.returncode
                out = res.stdout.splitlines()
            exec_ms = (time.perf_counter() - t0) * 1e3
            tag = "" if fc else " [host subprocess]"
            print(f"  {bid:>8}: KV cached={cached:>4} charged={charged:>3} | "
                  f"sandbox exit={code} {exec_ms:6.1f} ms{tag}  "
                  f"{out[-1] if out else ''}")

        tel = kv.telemetry("root")
        peak = pool_stats(base)
        print(f"\ntelemetry(root): charged_tokens={tel['charged_tokens']} "
              f"pinned_tokens={tel['pinned_tokens']} "
              f"saved_tokens={tel['saved_tokens']} "
              f"live_branches={tel['live_branches']}")
        print(f"KV reuse per child (cached_tokens): {reuse}")
        print(f"live server pool at peak: used={peak['used']} "
              f"live_branches={peak['live_branches']}")

        winner = children[winner_idx].branch_id
        # export the winner's artifact (the durable handoff)
        dest = os.path.join(os.path.dirname(registry), "winner_artifact.tar")
        if fc:
            nbytes = orch.export_artifact(winner, "/tmp/agentfork/fix.diff", dest)
            print(f"\nwinner {winner}: exported artifact -> {dest} ({nbytes} bytes)")
        else:
            src = os.path.join(host_art[winner], "fix.diff")
            nbytes = os.path.getsize(src) if os.path.exists(src) else 0
            shutil.copy(src, dest)
            print(f"\nwinner {winner}: exported artifact -> {dest} "
                  f"({nbytes} bytes) [host copy]")

        t0 = time.perf_counter()
        receipts = orch.kill_losers(winner)
        kill_ms = (time.perf_counter() - t0) * 1e3
        freed = {r.branch_id: r.kv_freed_tokens for r in receipts}
        total_freed = sum(freed.values())
        print(f"kill_losers({winner}): {len(receipts)} branches "
              f"{kill_ms:6.1f} ms, KV freed per loser={freed}")
        print(f"total KV tokens freed on kill: {total_freed}")
        assert orch.alive(winner), "winner must survive kill_losers"
        after = pool_stats(base)
        print(f"live server pool after kill_losers: used={after['used']} "
              f"live_branches={after['live_branches']} (winner still pinned)")

    # orchestrator closed -> winner killed too; allocator must reach baseline
    final = pool_stats(base)
    print(f"live server pool after close: used={final['used']} "
          f"available={final['available']} (baseline={baseline['available']})")

    ok = (all(c > 0 for c in reuse[1:] or reuse)  # children reused the prefix
          and total_freed > 0                      # losers freed KV
          and final['used'] == baseline['used']    # allocator back to baseline
          and final['available'] == baseline['available'])
    print("\nINTEGRATED E2E: " + ("PASS" if ok else "FAIL"))
    shutil.rmtree(os.path.dirname(registry), ignore_errors=True)
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    p.add_argument("--admin-api-key", required=True)
    p.add_argument("--children", type=int, default=10)
    p.add_argument("--sandbox", choices=["firecracker", "reaper"],
                   default="firecracker")
    p.add_argument("--fc", help="firecracker binary")
    p.add_argument("--kernel", help="guest kernel image")
    p.add_argument("--rootfs", help="agent rootfs (tools/build_rootfs.sh)")
    p.add_argument("--overlay-mib", type=int, default=64)
    p.add_argument("--work-dir", default="/tmp/agentfork-integrated")
    args = p.parse_args()
    try:
        return run(args)
    except urllib.error.URLError as exc:
        sys.exit(f"cannot reach live SGLang server at {args.sglang_url}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
