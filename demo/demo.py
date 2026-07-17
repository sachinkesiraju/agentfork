"""agentfork demo: fork a live agent 10 ways, race the branches, kill the losers.

Runs on Linux without a GPU or Firecracker: each branch is a real OS process
supervised through pidfd, and its KV context lives in the CPU reference cache.
Watch the ledger: KV forks require no re-prefill or page copies, each kill reaps
the process and reference-cache entry, and the cache ends exactly empty.
Timings depend on the host.

    python -m demo.demo            # or: python demo/demo.py

For separate patched-SGLang allocator and stock-engine prefix-cache checks on a
GPU, see modal_gpu_validation.py.
"""

from __future__ import annotations

import random
import sys
import time

from agentfork.kill.reaper import BranchReaper
from agentfork.kv.tree_cache import TreeKVCache

PREFIX = 32_000   # parent context tokens (system prompt + repo + history)
SUFFIX = 800      # tokens each branch generates on its own
CHILDREN = 10

BOLD, DIM, GREEN, RED, CYAN, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[36m", "\033[0m")


def say(msg: str = "") -> None:
    print(msg)
    sys.stdout.flush()


def main() -> None:
    kv = TreeKVCache(capacity_tokens=2_000_000)
    reaper = BranchReaper(kv)
    sleeper = [sys.executable, "-c", "import time; time.sleep(600)"]

    say(f"{BOLD}== agentfork demo: one parent, {CHILDREN}-way fork, "
        f"kill the losers =={RESET}\n")

    # 1. parent agent builds up a big context (the expensive part)
    say(f"{CYAN}[parent]{RESET} spawning agent + prefilling "
        f"{PREFIX:,}-token context ...")
    kv.create_tree("parent")
    reaper.spawn("parent", sleeper)
    t0 = time.perf_counter()
    charged = kv.extend("parent", list(range(PREFIX)))
    say(f"{CYAN}[parent]{RESET} prefill charged {charged:,} tokens "
        f"(cold — someone always pays once)  "
        f"resident={kv.resident_tokens():,}\n")

    # 2. fork N sibling branches — each inherits the prefix CoW
    say(f"{BOLD}forking {CHILDREN} branches ...{RESET}")
    t0 = time.perf_counter()
    for i in range(CHILDREN):
        kv.fork_branch("parent", f"branch-{i}")
        reaper.spawn(f"branch-{i}", sleeper)
    fork_ms = (time.perf_counter() - t0) * 1e3
    s = kv.stats
    say(f"  {CHILDREN} forks in {GREEN}{fork_ms:.1f} ms{RESET} — "
        f"re-prefill charged: {GREEN}0 tokens{RESET} "
        f"(saved {s.prefill_tokens_saved:,})")
    say(f"  resident {kv.resident_tokens():,} tokens for "
        f"{s.logical_tokens:,} logical → "
        f"{GREEN}{s.dedup_ratio:.2f}x KV dedup{RESET}\n")

    # 3. branches diverge: each decodes its own suffix
    for i in range(CHILDREN):
        kv.extend(f"branch-{i}", [10_000_000 + i * SUFFIX + j
                                  for j in range(SUFFIX)])
    say(f"branches diverged (+{SUFFIX} tokens each): "
        f"resident={kv.resident_tokens():,}, "
        f"dedup {kv.stats.dedup_ratio:.2f}x\n")

    # 4. race over — one winner, kill the rest with one call each
    winner = random.randrange(CHILDREN)
    say(f"{BOLD}branch-{winner} wins — killing the other "
        f"{CHILDREN - 1} branches:{RESET}")
    for i in range(CHILDREN):
        if i == winner:
            continue
        r = reaper.kill(f"branch-{i}")
        say(f"  {RED}kill branch-{i}{RESET}: pid {r.pid} signaled in "
            f"{r.signal_us:.0f} µs, reaped in {r.reaped_us:.0f} µs, "
            f"{r.kv_freed_tokens:,} KV tokens freed → "
            f"total {GREEN}{r.total_ms:.2f} ms{RESET}")
    say(f"\nafter kills: resident={kv.resident_tokens():,} "
        f"(winner + shared prefix only)\n")

    # 5. teardown: kill winner and parent; the ledger must read zero
    reaper.kill(f"branch-{winner}")
    r = reaper.kill("parent")
    say(f"{CYAN}[parent]{RESET} killed: {r.kv_freed_tokens:,} tokens freed "
        f"in {r.total_ms:.2f} ms")
    leaked = kv.resident_tokens()
    live = len(kv.trees)
    verdict = (f"{GREEN}CLEAN{RESET}" if leaked == 0 and live == 0
               else f"{RED}LEAK{RESET}")
    say(f"\n{BOLD}final ledger:{RESET} resident KV tokens = {leaked}, "
        f"live trees = {live} → {verdict}")
    if leaked or live:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
