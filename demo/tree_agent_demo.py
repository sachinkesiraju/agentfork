"""Live agent-tree demo: fix a bug with a real LLM over a fork tree.

Runs the ``agentfork.harness`` golden path against a real model on plain Linux
with no GPU (``NullSandbox`` + ``TreeKVCache``):

    prepare a shared context (a buggy module + its failing test) on a root
    branch -> ask the LLM for N candidate fixes from that one context -> fork
    one branch per candidate, each committing a continuation that strictly
    extends the shared prefix -> apply the fix in the branch's workdir and run
    the test as the cheap check -> keep the branch whose tests pass, kill the
    rest -> fork the winner again and re-run against a fuller (edge-case) suite
    as a verification round -> keep the verified winner.

The shared context is prefilled once and inherited copy-on-write by every
candidate; the ledger at the end shows the KV dedup and that killing the losers
leaves only the winner's chain resident.

    export ANTHROPIC_API_KEY=...        # default provider
    python demo/tree_agent_demo.py
    python demo/tree_agent_demo.py --provider together --candidates 4

``--provider together`` uses ``TOGETHER_API_KEY`` and an OpenAI-compatible
endpoint instead. No network calls happen except the model requests.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

from agentfork.harness import (
    AnthropicLLM,
    BranchResult,
    OpenAICompatLLM,
    Round,
    TreeAgent,
)
from agentfork.kv.tree_cache import TreeKVCache
from agentfork.orchestrator import ForkOrchestrator, NullSandbox

BOLD, DIM, GREEN, RED, CYAN, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[36m", "\033[0m")

FIX_MARKER = "\n### CANDIDATE clamp.py ###\n"
VERIFY_MARKER = "\n### VERIFICATION ROUND ###\n"

BUGGY = '''\
def clamp(x, lo, hi):
    """Clamp x into the inclusive range [lo, hi]."""
    return min(x, hi)
'''

BASIC_TEST = '''\
from clamp import clamp

def test_within():
    assert clamp(5, 0, 10) == 5

def test_below():
    assert clamp(-3, 0, 10) == 0
'''

EDGE_TEST = '''\
from clamp import clamp

def test_above():
    assert clamp(100, 0, 10) == 10

def test_on_bounds():
    assert clamp(0, 0, 10) == 0
    assert clamp(10, 0, 10) == 10

def test_negative_range():
    assert clamp(-50, -10, -1) == -10
'''

CONTEXT = (
    "A tiny Python project has a bug. `clamp.py`:\n\n"
    f"{BUGGY}\n"
    "Its test `test_clamp.py` fails:\n\n"
    f"{BASIC_TEST}\n"
    "Return a corrected `clamp.py` that makes the tests pass. Output only the "
    "Python source for clamp.py, nothing else."
)


def say(msg: str = "") -> None:
    print(msg)
    sys.stdout.flush()


def _extract_code(text: str) -> str:
    """Pull the Python source out of a candidate, tolerating ``` fences."""
    text = text.strip()
    if "```" in text:
        blocks = text.split("```")
        # blocks[1] is the first fenced body; drop a leading "python" tag
        body = blocks[1]
        if body.lstrip().lower().startswith("python"):
            body = body.split("\n", 1)[1] if "\n" in body else ""
        return body.strip() + "\n"
    return text + "\n"


def _run_tests(workdir: str, code: str, tests: dict[str, str]) -> bool:
    """Write ``clamp.py`` and test files into ``workdir`` and run pytest."""
    with open(os.path.join(workdir, "clamp.py"), "w") as f:
        f.write(code)
    for name, body in tests.items():
        with open(os.path.join(workdir, name), "w") as f:
            f.write(body)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", workdir],
        capture_output=True, text=True, cwd=workdir, timeout=120)
    return proc.returncode == 0


def make_work(root: str, tests: dict[str, str]):
    """Build a stateless work callable: it recovers the candidate fix from the
    branch's committed prefix, writes it into a per-branch workdir, and runs
    the given tests. Failures raise, which the harness records per branch."""

    def work(branch_id: str, prefix) -> dict:
        text = bytes(prefix).decode()
        candidate = text.split(FIX_MARKER, 1)[1].split(VERIFY_MARKER, 1)[0]
        code = _extract_code(candidate)
        workdir = os.path.join(root, branch_id.replace("/", "_"))
        os.makedirs(workdir, exist_ok=True)
        passed = _run_tests(workdir, code, tests)
        return {"passed": passed, "code": code, "workdir": workdir}

    return work


def tests_pass(result: BranchResult) -> float:
    return 1.0 if result.output and result.output["passed"] else 0.0


def build_llm(provider: str):
    if provider == "anthropic":
        return AnthropicLLM()
    if provider == "together":
        return OpenAICompatLLM()
    raise ValueError(f"unknown provider: {provider}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["anthropic", "together"],
                        default="anthropic")
    parser.add_argument("--candidates", type=int, default=3,
                        help="number of candidate fixes to fan out")
    args = parser.parse_args()

    llm = build_llm(args.provider)
    say(f"{BOLD}== agentfork agent-tree demo: fix a bug, race the branches, "
        f"verify the winner =={RESET}\n")

    say(f"{CYAN}[llm]{RESET} asking {args.provider} for {args.candidates} "
        f"candidate fixes from one shared context ...")
    candidates = llm.propose(CONTEXT, args.candidates)
    say(f"{CYAN}[llm]{RESET} got {len(candidates)} candidate(s)\n")

    root = tempfile.mkdtemp(prefix="agentfork-treedemo-")
    orch = ForkOrchestrator(kv=TreeKVCache(capacity_tokens=2_000_000),
                            sandbox=NullSandbox())
    agent = TreeAgent(orch)
    try:
        round1 = Round(
            continuations=[CONTEXT + FIX_MARKER + c for c in candidates],
            work=make_work(root, {"test_clamp.py": BASIC_TEST}),
            evaluator=tests_pass)

        say(f"{BOLD}round 1:{RESET} fan out {len(candidates)} branches, run "
            f"the basic test as the cheap check ...")
        agent.prepare_root("root", CONTEXT)
        results = agent.fan_out(
            "root", round1.continuations, round1.work, round1.evaluator)
        for result in results:
            status = (f"{GREEN}PASS{RESET}" if result.error is None
                      and result.output["passed"] else f"{RED}FAIL{RESET}")
            note = "error" if result.error else "tests"
            say(f"  branch {result.branch_id}: {status} ({note})")
        winner1 = agent.select_winner(results)
        agent.kill_losers(winner1.branch_id)
        s = orch.kv.stats
        say(f"  winner {GREEN}{winner1.branch_id}{RESET}; killed the rest — "
            f"resident {orch.kv.resident_tokens():,} tokens, "
            f"{s.dedup_ratio:.2f}x KV dedup, saved "
            f"{s.prefill_tokens_saved:,} prefill tokens\n")

        say(f"{BOLD}round 2 (verification):{RESET} re-fork the winner, run the "
            f"fuller edge-case suite on the finalists ...")
        winner_text = bytes(winner1.prefix).decode()
        round2 = Round(
            continuations=[winner_text + VERIFY_MARKER + tag
                           for tag in ("audit-a", "audit-b")],
            work=make_work(root, {"test_clamp.py": BASIC_TEST,
                                  "test_edge.py": EDGE_TEST}),
            evaluator=tests_pass)
        results2 = agent.fan_out(
            winner1.branch_id, round2.continuations, round2.work,
            round2.evaluator)
        for result in results2:
            status = (f"{GREEN}PASS{RESET}" if result.error is None
                      and result.output["passed"] else f"{RED}FAIL{RESET}")
            say(f"  branch {result.branch_id}: {status} (basic + edge)")
        winner2 = agent.select_winner(results2)
        agent.kill_losers(winner2.branch_id)

        verified = winner2.output["passed"]
        say(f"\n{BOLD}verified winner:{RESET} {winner2.branch_id} "
            f"({'passes' if verified else 'FAILS'} basic + edge)\n")
        say(f"{BOLD}winning clamp.py:{RESET}")
        say(DIM + winner2.output["code"].rstrip() + RESET + "\n")

        live = {b.branch_id for b in orch.branches()}
        say(f"{BOLD}tree at finish:{RESET} live branches = {sorted(live)} "
            f"(root + winner chain only)")
    finally:
        orch.close()
        shutil.rmtree(root, ignore_errors=True)

    leaked = orch.kv.resident_tokens()
    verdict = f"{GREEN}CLEAN{RESET}" if leaked == 0 else f"{RED}LEAK{RESET}"
    say(f"{BOLD}final ledger:{RESET} resident KV tokens after teardown = "
        f"{leaked} -> {verdict}")
    if leaked:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
