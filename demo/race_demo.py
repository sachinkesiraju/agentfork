"""Browser race demo: 10 candidate fixes, one inference server, a noisy neighbor.

Two arms solve the *same* task against the *same* live tree-cache server, one
after the other, while an unrelated tenant streams traffic through that server
the whole time:

The tree is not flat: the root holds the shared repo context, a handful of
*approach* branches commit their own reasoning under it, the candidates fork
off those approaches, and the verification round forks off the winning leaf.
So a candidate can hit the root context and still be charged for the approach
above it -- the demo reports the root hit rate and the full-lineage (subtree)
hit rate separately, because only the second one says the subtree survived.

  STOCK      no pinning, no branches. The shared repo context is prefilled
             once and left in the cache like any other prefix; each candidate
             is an ordinary request that hopes the prefix is still resident,
             and each gets a cold sandbox workspace built from scratch.
  AGENTFORK  ``create_parent`` warms the shared context on a pinned branch,
             ``fork`` gives each candidate a copy-on-write child that is
             charged only its own suffix, cheap checks kill the losers (their
             KV is reclaimed immediately), and the winner is re-forked for a
             verification round. Each child inherits the parent's already
             prepared workspace instead of rebuilding it.

The neighbor is sized with the break-even math in ``agentfork.bench.cost_model``
(``U* = C - P``): it injects more than ``C - P`` unrelated tokens between
candidates, which is exactly the pressure a stock LRU radix cache cannot keep
the shared prefix through -- and which a pinned branch is immune to. Nothing
about the eviction is simulated: it is the real cache's own LRU running out of
real KV pool slots.

HONESTY (read this before quoting any number)
---------------------------------------------
This runs on a CPU box with no model weights, so **the transformer forward
pass is stubbed**. Real here: the KV pool and allocator, ``TreeRadixCache``
(prefix matching, charge accounting, ``lock_ref`` pinning, LRU eviction),
branch lifecycle over real HTTP with admin auth, the sandboxes (real
subprocesses), and the candidate checks (real ``pytest`` runs). Stubbed: the
generated text, and therefore *generation latency*. The headline numbers are
therefore **prefill tokens charged** and **parent-prefix hit rate**, which are
the cache's own measurements; wall-clock is reported as a secondary number and
must not be read as model speed.

The race renders itself as a live browser dashboard (``demo/race_web.py``):
branch trees per arm, KV bars, per-candidate HIT/MISS, and the scoreboard.

Run (needs the patched SGLang checkout on PYTHONPATH -- tools/setup_sglang.sh):

    # opens http://127.0.0.1:8765 and streams the race into it
    PYTHONPATH=/path/to/sglang/python .venv/bin/python demo/race_demo.py

    # headless: sequential log + machine-readable JSON summary (CI/tests)
    PYTHONPATH=/path/to/sglang/python .venv/bin/python demo/race_demo.py --no-ui

``--provider together|anthropic`` swaps the deterministic ``FakeLLM`` for a
real model through the existing adapters; the demo does not require it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field

from agentfork import ForkOrchestrator, ReaperSandbox, SGLangHTTPBackend
from agentfork.bench.cost_model import (
    PressureScenario,
    break_even_u,
    pressure_model,
)
from agentfork.harness import BranchResult, FakeLLM, TreeAgent

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(REPO, "demo", "sglang_tree_server.py")
if REPO not in sys.path:  # run as a script from anywhere
    sys.path.insert(0, REPO)

FIX_MARKER = "\n### CANDIDATE clamp.py ###\n"
VERIFY_MARKER = "\n### VERIFICATION ROUND ###\n"
APPROACH_MARKER = "\n### APPROACH ###\n"

# Level 1 of the tree: distinct strategies the agent commits to *before* it
# writes any code. Each one is real KV -- a few hundred tokens of reasoning
# that its own candidates inherit and that its siblings must not be charged
# for. This is what makes a subtree hit different from a root hit: a leaf
# under "boundary-first" reuses the shared repo context *and* that approach's
# reasoning, none of which it pays for.
APPROACHES = [
    ("boundary-first",
     "# plan: reason about the inclusive boundaries first, then write the "
     "comparison so lo/hi are both respected.\n"),
    ("test-driven",
     "# plan: read the failing assertions, derive the contract from them, "
     "then write the smallest expression that satisfies it.\n"),
    ("rewrite",
     "# plan: ignore the existing body and re-derive clamp from its "
     "docstring.\n"),
]

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

# The one candidate that actually fixes the bug, plus the near-misses a model
# realistically produces. The cheap check (pytest) is what tells them apart:
# every near-miss fails the basic suite, so the round-1 fan-out has exactly
# one survivor no matter how wide it is.
GOOD_FIX = "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n"
BAD_FIXES = [
    "def clamp(x, lo, hi):\n    return min(x, hi)\n",
    "def clamp(x, lo, hi):\n    return max(x, hi)\n",
    "def clamp(x, lo, hi):\n    return min(max(x, hi), lo)\n",
    "def clamp(x, lo, hi):\n    return x if lo < x < hi else hi\n",
    "def clamp(x, lo, hi):\n    return sorted([x, lo, hi])[2]\n",
    "def clamp(x, lo, hi):\n    return abs(min(x, hi))\n",
    "def clamp(x, lo, hi):\n    return max(lo, x, hi)\n",
    "def clamp(x, lo, hi):\n    return x % (hi - lo) + lo\n",
    "def clamp(x, lo, hi):\n    return lo if x < lo else x + 1\n",
    "def clamp(x, lo, hi):\n    return min(x, lo)\n",
]
# The good fix is not first: winner selection has to come from the checks.
CANDIDATES = [BAD_FIXES[0], GOOD_FIX] + BAD_FIXES[1:]


def shared_context(prefix_tokens: int) -> str:
    """The repo context every candidate inherits, padded to ``prefix_tokens``.

    The server tokenizes bytes, so one ASCII character is one token and the
    padded length is the shared prefix length ``P`` the cost model talks
    about."""
    head = (
        "# repo: tinyproj\n"
        "# task: clamp.py is wrong; propose a fix that passes the suite.\n\n"
        f"clamp.py:\n{BUGGY}\n"
        f"test_clamp.py (failing):\n{BASIC_TEST}\n"
        "# --- surrounding repo context the agent was given ---\n")
    filler = ("# ctx: unrelated but in-context repo lines the agent read "
              "while triaging this bug\n")
    while len(head) + len(filler) <= prefix_tokens:
        head += filler
    pad = prefix_tokens - len(head)
    if pad > 0:
        head += "#" * (pad - 1) + "\n" if pad > 1 else "#"
    return head


# ---------------------------------------------------------------------------
# live server (a real subprocess, restarted between arms so each arm starts
# from an identical, empty KV pool)
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LiveServer:
    """The tree-cache HTTP server from demo/sglang_tree_server.py."""

    def __init__(self, capacity_tokens: int, admin_key: str):
        self.capacity_tokens = capacity_tokens
        self.admin_key = admin_key
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc = subprocess.Popen(
            [sys.executable, SERVER, "--port", str(self.port),
             "--admin-api-key", admin_key,
             "--pool-tokens", str(capacity_tokens)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=REPO)
        self._wait_ready()

    def _wait_ready(self, timeout_s: float = 120.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                err = (self.proc.stderr.read() or b"").decode(errors="replace")
                raise RuntimeError(
                    f"tree-cache server exited early:\n{err[-2000:]}")
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2):
                    return
            except (urllib.error.URLError, OSError):
                time.sleep(0.2)
        raise TimeoutError(f"server at {self.url} never became healthy")

    def pool_stats(self) -> dict:
        with urllib.request.urlopen(f"{self.url}/pool_stats", timeout=10) as r:
            return json.loads(r.read())

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# KV backends: the only difference between the two arms
# ---------------------------------------------------------------------------


class RecordingBackend(SGLangHTTPBackend):
    """``SGLangHTTPBackend`` that keeps each branch's last ``meta_info``.

    ``TreeAgent`` discards the generate response, but the cache's own
    ``cached_tokens``/``charged_tokens`` per branch is the measurement this
    demo exists to report."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta: dict[str, dict] = {}
        self.meta_lock = threading.Lock()

    def generate(self, branch_id, prompt, sampling_params, **kwargs) -> dict:
        out = super().generate(branch_id, prompt, sampling_params, **kwargs)
        meta = out.get("meta_info") or {}
        with self.meta_lock:
            self.meta[branch_id] = meta
        return out


class StockBackend(RecordingBackend):
    """The same live server driven the way a *stock* engine is driven.

    Two differences from the agentfork path, and nothing else:

    * ``fork_branch`` does not fork. A stock engine has no branch primitive,
      so each candidate is an ordinary new request context created in the same
      cache namespace; it reuses the shared prefix only if ordinary radix
      prefix matching still finds it resident.
    * nothing stays pinned. After a request completes, its pages are released
      to the LRU (a ``demote``, i.e. ``lock_ref`` back to zero) the way a
      finished request's pages are on a stock engine -- so the shared prefix
      is ordinary evictable cache content, not a pinned tree.
    """

    def fork_branch(self, parent_id: str, child_id: str | None = None):
        if child_id is None:
            raise ValueError("child_id is required")
        with self._condition:
            namespace = self._namespaces[parent_id]
        self._operation("create", child_id, tree_id=namespace)
        with self._condition:
            self._namespaces[child_id] = namespace
            self._parents[child_id] = None  # no branch lineage on a stock engine
        return _StockBranch(child_id)

    def generate(self, branch_id, prompt, sampling_params, **kwargs) -> dict:
        out = super().generate(branch_id, prompt, sampling_params, **kwargs)
        self._operation("demote", branch_id)  # request finished: unpin
        return out


@dataclass(frozen=True)
class _StockBranch:
    branch_id: str


# ---------------------------------------------------------------------------
# the noisy neighbor
# ---------------------------------------------------------------------------


class NoisyNeighbor:
    """An unrelated tenant streaming prompts through the same server.

    Each request is its own short-lived context in its own namespace, unpinned
    as soon as it finishes -- an ordinary tenant, not an adversary. The only
    thing that makes it hostile is volume: ``request_tokens`` x
    ``requests_per_gap`` tokens land between two candidates, which is the
    per-gap pressure ``U`` in the cost model.

    The traffic is metered rather than free-running: ``gap()`` admits exactly
    ``requests_per_gap`` requests and waits for them, so the interleaved token
    count between two children is exactly ``U`` and does not drift with how
    long a candidate's checks happen to take. That is what makes the
    comparison against ``U* = C - P`` meaningful in both directions.
    """

    def __init__(self, url: str, admin_key: str, request_tokens: int,
                 requests_per_gap: int, on_event=None):
        self.url = url.rstrip("/")
        self.admin_key = admin_key
        self.request_tokens = request_tokens
        self.requests_per_gap = requests_per_gap
        self.on_event = on_event or (lambda **kw: None)
        self.served = 0
        self.served_tokens = 0
        self.deferred = 0          # server pushed back: no capacity right now
        self._seq = 0
        self._done = threading.Condition()
        self._completed = 0
        self._permits = threading.Semaphore(0)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30)

    def gap(self) -> None:
        """Admit exactly one gap's worth of neighbor traffic, and wait for it."""
        with self._done:
            target = self._completed + self.requests_per_gap
        for _ in range(self.requests_per_gap):
            self._permits.release()
        with self._done:
            while self._completed < target and self._thread.is_alive():
                self._done.wait(timeout=0.5)

    # -- traffic -----------------------------------------------------------

    def _post(self, path: str, body: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            self.url + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.admin_key}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    def _run(self) -> None:
        rng = random.Random(20260729)
        while not self._stop.is_set():
            if not self._permits.acquire(timeout=0.2):
                continue
            self._seq += 1
            branch = f"neighbor/{self._seq}"
            self._post("/tree_cache",
                       {"operation": "create", "branch_id": branch,
                        "tree_id": "neighbor"})
            # unique body so every neighbor request is genuinely new content
            tag = f"[{self._seq:06d}:{rng.randrange(10**6):06d}]"
            text = tag + "z" * (self.request_tokens - len(tag))
            status, payload = self._post(
                "/tree_generate",
                {"branch_id": branch, "tree_id": "neighbor", "text": text,
                 "sampling_params": {"max_new_tokens": 1}})
            if status == 200:
                self.served += 1
                self.served_tokens += payload["meta_info"]["charged_tokens"]
                self._post("/tree_cache",
                           {"operation": "demote", "branch_id": branch})
            else:
                # No capacity: the pinned tenant's prefix is not evictable, so
                # the neighbor waits instead of taking it. Real backpressure.
                self.deferred += 1
                self._post("/tree_cache",
                           {"operation": "kill", "branch_id": branch})
                self.on_event(kind="neighbor_deferred",
                              detail=payload.get("error", ""))
            with self._done:
                self._completed += 1
                self._done.notify_all()


# ---------------------------------------------------------------------------
# the candidate workspaces (sandbox side)
# ---------------------------------------------------------------------------

PROJECT_FILES = {"clamp.py": BUGGY, "test_clamp.py": BASIC_TEST}


def _write(dirpath: str, files: dict[str, str]) -> None:
    for name, body in files.items():
        with open(os.path.join(dirpath, name), "w") as f:
            f.write(body)


def cold_workspace(dest: str, files: dict[str, str]) -> float:
    """Build a candidate workspace from scratch: the cold-boot the stock arm
    pays per child. Returns the measured seconds."""
    t0 = time.perf_counter()
    os.makedirs(dest, exist_ok=True)
    _write(dest, files)
    # a real (small) dependency/bytecode warm-up, from cold, in this workspace
    subprocess.run([sys.executable, "-m", "compileall", "-q", dest],
                   capture_output=True, check=False)
    return time.perf_counter() - t0


def warm_workspace(template: str, dest: str) -> float:
    """Inherit the parent branch's already-prepared workspace (hardlink clone,
    the filesystem analogue of the KV copy-on-write fork)."""
    t0 = time.perf_counter()
    shutil.copytree(template, dest, copy_function=os.link, dirs_exist_ok=True)
    return time.perf_counter() - t0


def run_check(workdir: str, code: str, tests: dict[str, str]) -> bool:
    """The cheap check: write the candidate fix and run pytest for real."""
    _write(workdir, {"clamp.py": code, **tests})
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         workdir],
        capture_output=True, text=True, cwd=workdir, timeout=300)
    return proc.returncode == 0


def extract_code(candidate: str) -> str:
    text = candidate.strip()
    if "```" in text:
        body = text.split("```")[1]
        if body.lstrip().lower().startswith("python"):
            body = body.split("\n", 1)[1] if "\n" in body else ""
        return body.strip() + "\n"
    return text + "\n"


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass
class ChildRecord:
    branch_id: str
    cached_tokens: int = 0
    charged_tokens: int = 0
    parent_hit: bool = False        # reused the shared root context
    subtree_hit: bool = False       # ... and its own parent branch's tokens
    hit_level: str = "miss"         # "subtree" | "root" | "miss"
    depth: int = 1
    parent_branch: str = ""
    title: str = ""                 # short human label (an approach's name)
    sandbox_setup_s: float = 0.0
    check_passed: bool = False
    error: str | None = None


@dataclass
class ArmResult:
    name: str
    approaches: list[ChildRecord] = field(default_factory=list)
    children: list[ChildRecord] = field(default_factory=list)
    verify_children: list[ChildRecord] = field(default_factory=list)
    round1_winner: str | None = None
    parent_prefill_charged: int = 0
    wall_clock_s: float = 0.0
    peak_kv_used: int = 0
    pinned_after_kills: int = 0
    kv_freed_by_kills: int = 0
    sandbox_setup_s: float = 0.0
    template_setup_s: float = 0.0   # one-time parent workspace prep (agentfork)
    winner: str | None = None
    verified: bool = False
    neighbor_served: int = 0
    neighbor_tokens: int = 0
    neighbor_deferred: int = 0

    @property
    def parent_hit_rate(self) -> float:
        if not self.children:
            return 0.0
        return sum(c.parent_hit for c in self.children) / len(self.children)

    @property
    def subtree_hit_rate(self) -> float:
        """Leaves that reused their *own* approach branch, not just the root.

        A leaf can hit the shared repo context and still be charged for its
        approach's reasoning if that subtree was evicted; only this rate says
        the whole lineage survived."""
        if not self.children:
            return 0.0
        return sum(c.subtree_hit for c in self.children) / len(self.children)

    @property
    def prefill_charged(self) -> int:
        return (self.parent_prefill_charged
                + sum(c.charged_tokens for c in self.approaches)
                + sum(c.charged_tokens for c in self.children)
                + sum(c.charged_tokens for c in self.verify_children))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["parent_hit_rate"] = round(self.parent_hit_rate, 4)
        d["subtree_hit_rate"] = round(self.subtree_hit_rate, 4)
        d["prefill_tokens_charged"] = self.prefill_charged
        return d


# ---------------------------------------------------------------------------
# the two arms
# ---------------------------------------------------------------------------


class Race:
    def __init__(self, args, ui):
        self.args = args
        self.ui = ui
        self.context = shared_context(args.prefix_tokens)
        self.prefix_tokens = len(self.context.encode())
        self.llm = build_llm(args.provider)
        self.candidates = self.llm.propose(self.context, args.children)
        self.root = tempfile.mkdtemp(prefix="agentfork-race-")

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _suffix(self, body: str, index: int) -> str:
        """A candidate's unique continuation, padded to ``S`` tokens.

        The reasoning a model emits alongside a patch is the branch's own KV:
        unique to it, charged to it, and freed when it is killed. Padding it
        to a fixed ``S`` keeps the arms comparable and makes the reclaim on
        ``kill_losers`` a number you can see."""
        head = f"{body}# branch {index} rationale:\n"
        line = f"# note {index}: this branch's own tokens, not the parent's\n"
        while len(head) + len(line) <= self.args.suffix_tokens:
            head += line
        pad = self.args.suffix_tokens - len(head)
        if pad > 1:
            head += "#" * (pad - 1) + "\n"
        elif pad == 1:
            head += "\n"
        return head

    def _plans(self) -> list[tuple[str, str]]:
        return [APPROACHES[i % len(APPROACHES)]
                for i in range(self.args.approaches)]

    def _plan_text(self, name: str, body: str) -> str:
        """An approach's own committed reasoning, padded to ``A`` tokens."""
        head = f"# approach: {name}\n{body}"
        line = f"# {name}: reasoning that only this subtree inherits\n"
        while len(head) + len(line) <= self.args.approach_tokens:
            head += line
        pad = self.args.approach_tokens - len(head)
        if pad > 1:
            head += "#" * (pad - 1) + "\n"
        elif pad == 1:
            head += "\n"
        return head

    @staticmethod
    def _bucket(candidates: list[str],
                n_buckets: int) -> list[list[tuple[int, str]]]:
        """Spread the candidates over the approach subtrees, round-robin."""
        buckets: list[list[tuple[int, str]]] = [[] for _ in range(n_buckets)]
        for i, candidate in enumerate(candidates):
            buckets[i % n_buckets].append((i, candidate))
        return buckets

    # -- shared plumbing ---------------------------------------------------

    def _round(self, arm: ArmResult, agent: TreeAgent, backend, server,
               parent_id: str, continuations: list[str],
               tests: dict[str, str], template: str | None,
               records: list[ChildRecord], neighbor: NoisyNeighbor,
               label: str, *, depth: int = 1, base_tokens: int | None = None,
               check: bool = True,
               index_base: int = 0, total: int | None = None,
               workspace_out: dict[str, str] | None = None,
               gap_first: bool = False, titles: list[str] | None = None,
               ) -> list[BranchResult]:
        """One fan-out round: fork every candidate off ``parent_id``, then
        commit and check them one at a time with a full gap of neighbor
        traffic between consecutive candidates -- the reuse distance the
        shared prefix has to survive.

        ``TreeAgent.fan_out`` commits a child, runs its work, then moves to
        the next child, so injecting the gap at the end of ``work`` puts
        exactly ``U`` unrelated tokens between one child's commit and the
        next's.
        """
        setup_s: dict[str, float] = {}
        n_here = len(continuations)
        total = n_here if total is None else total
        base_tokens = (self.prefix_tokens if base_tokens is None
                       else base_tokens)
        done = {"n": 0}

        def work(branch_id: str, prefix: str) -> dict:
            code = ""
            if check:
                candidate = (prefix.split(FIX_MARKER, 1)[1]
                             .split(VERIFY_MARKER, 1)[0])
                code = extract_code(candidate)
            workdir = os.path.join(
                self.root, f"{arm.name}-{branch_id.replace('/', '_')}")
            if template is None:
                # stock: no parent branch to inherit from -> cold workspace
                setup_s[branch_id] = cold_workspace(
                    workdir, dict(PROJECT_FILES, **tests))
            else:
                setup_s[branch_id] = warm_workspace(template, workdir)
            if workspace_out is not None:
                workspace_out[branch_id] = workdir
            # an approach branch has committed a plan, not a patch: there is
            # nothing to test yet, so it just prepares the workspace its own
            # candidates will inherit.
            passed = run_check(workdir, code, tests) if check else True
            used = server.pool_stats()["used"]
            arm.peak_kv_used = max(arm.peak_kv_used, used)
            self.ui.kv(arm.name, used)
            done["n"] += 1
            neighbor.gap()  # unrelated traffic between consecutive branches
            return {"passed": passed, "code": code, "workdir": workdir}

        if gap_first:
            neighbor.gap()  # pressure before the very first branch, too
        results = agent.fan_out(parent_id, continuations, work,
                                lambda r: 1.0 if r.output["passed"] else 0.0)
        for idx, result in enumerate(results):
            meta = backend.meta.get(result.branch_id, {})
            record = ChildRecord(
                branch_id=result.branch_id,
                cached_tokens=int(meta.get("cached_tokens", 0)),
                charged_tokens=int(meta.get("charged_tokens", 0)),
                sandbox_setup_s=setup_s.get(result.branch_id, 0.0),
                check_passed=bool(result.error is None
                                  and result.output["passed"]),
                depth=depth,
                parent_branch=parent_id,
                title=(titles[idx] if titles else ""),
                error=(None if result.error is None
                       else f"{type(result.error).__name__}: {result.error}"))
            record.parent_hit = record.cached_tokens >= self.prefix_tokens
            # the branch reused everything its parent had committed -- the
            # shared context *and* the lineage above it
            record.subtree_hit = record.cached_tokens >= base_tokens
            record.hit_level = ("subtree" if record.subtree_hit else
                                ("root" if record.parent_hit else "miss"))
            records.append(record)
            self.ui.child(arm.name, label, index_base + idx, record, total)
        return results

    # -- arms --------------------------------------------------------------

    def run_arm(self, name: str) -> ArmResult:
        """Run one arm end to end against its own freshly started server."""
        args = self.args
        arm = ArmResult(name=name)
        stock = name == "stock"
        server = LiveServer(args.capacity_tokens, args.admin_api_key)
        neighbor = NoisyNeighbor(
            server.url, args.admin_api_key, args.noise_request_tokens,
            args.noise_requests_per_gap,
            on_event=lambda **kw: self.ui.event(name, **kw))
        cls = StockBackend if stock else RecordingBackend
        backend = cls(server.url, admin_api_key=args.admin_api_key)
        sandbox = ReaperSandbox(["/bin/sh", "-c", "exec sleep 3600"])
        registry = os.path.join(self.root, f"{name}-registry.json")
        template = None
        approach_templates: dict[str, str] = {}
        self.ui.arm_start(name, server.url)
        neighbor.start()
        t0 = time.perf_counter()
        try:
            with ForkOrchestrator(kv=backend, sandbox=sandbox,
                                  registry_path=registry) as orch:
                agent = TreeAgent(orch)
                root_id = f"{name}/root"
                agent.prepare_root(root_id, self.context)
                meta = backend.meta.get(root_id, {})
                arm.parent_prefill_charged = int(meta.get("charged_tokens", 0))
                if not stock:
                    # the parent also prepares the workspace its children
                    # inherit; the stock arm has no branch to hang this on, so
                    # every child rebuilds it from cold.
                    template = os.path.join(self.root, f"{name}-template")
                    arm.template_setup_s = cold_workspace(
                        template, PROJECT_FILES)
                self.ui.parent(name, self.prefix_tokens,
                               arm.parent_prefill_charged)

                # ---- level 1: divergent approaches off the root -------
                plans = self._plans()
                approach_conts = [
                    self.context + APPROACH_MARKER
                    + self._plan_text(name, body) for name, body in plans]
                approach_results = self._round(
                    arm, agent, backend, server, root_id, approach_conts,
                    {}, template, arm.approaches, neighbor, "approach",
                    depth=1, gap_first=True, check=False,
                    titles=[name for name, _ in plans],
                    workspace_out=(None if stock else approach_templates))
                live = [r for r in approach_results if r.error is None]
                if not live:
                    raise RuntimeError("every approach branch failed")

                # ---- level 2: candidates forked off each approach -----
                # A leaf inherits its approach's plan as well as the shared
                # context, so its cache hit is against the whole lineage.
                buckets = self._bucket(self.candidates, len(live))
                leaf_results: list[BranchResult] = []
                leaf_parent: dict[str, str] = {}
                idx = 0
                for approach, bucket in zip(live, buckets):
                    if not bucket:
                        continue
                    plan_prefix = agent.committed_prefix(approach.branch_id)
                    leaf_conts = [
                        plan_prefix + FIX_MARKER + self._suffix(c, i)
                        for i, c in bucket]
                    round_results = self._round(
                        arm, agent, backend, server, approach.branch_id,
                        leaf_conts, {"test_clamp.py": BASIC_TEST},
                        (approach_templates.get(approach.branch_id)
                         if not stock else None),
                        arm.children, neighbor, "fix",
                        depth=2, base_tokens=len(plan_prefix.encode()),
                        index_base=idx, total=len(self.candidates))
                    leaf_results += round_results
                    for r in round_results:
                        leaf_parent[r.branch_id] = approach.branch_id
                    idx += len(bucket)
                winner = agent.select_winner(leaf_results)
                arm.winner = arm.round1_winner = winner.branch_id

                before = server.pool_stats()["used"]
                receipts = agent.kill_losers(winner.branch_id)
                arm.kv_freed_by_kills = sum(r.kv_freed_tokens for r in receipts)
                after = server.pool_stats()["used"]
                self.ui.kills(name, len(receipts), arm.kv_freed_by_kills,
                              before - after)
                self.ui.kv(name, after)

                # verification round: re-fork the winner
                winner_text = winner.prefix
                verify = [winner_text + VERIFY_MARKER
                          + self._suffix(f"audit pass {i}\n", i)
                          for i in range(args.verify_children)]
                verified = agent.select_winner(self._round(
                    arm, agent, backend, server, winner.branch_id, verify,
                    {"test_clamp.py": BASIC_TEST, "test_edge.py": EDGE_TEST},
                    (approach_templates.get(leaf_parent[winner.branch_id],
                                            template)
                     if not stock else None),
                    arm.verify_children, neighbor, "verify",
                    depth=3, base_tokens=len(winner.prefix.encode())))
                agent.kill_losers(verified.branch_id)
                arm.winner = verified.branch_id
                arm.verified = bool(verified.output["passed"])
                arm.pinned_after_kills = int(
                    backend.telemetry(verified.branch_id)["pinned_tokens"])
        finally:
            arm.wall_clock_s = time.perf_counter() - t0
            neighbor.stop()
            arm.neighbor_served = neighbor.served
            arm.neighbor_tokens = neighbor.served_tokens
            arm.neighbor_deferred = neighbor.deferred
            server.stop()
        arm.sandbox_setup_s = sum(
            c.sandbox_setup_s
            for c in arm.approaches + arm.children + arm.verify_children)
        self.ui.arm_done(arm)
        return arm


def build_llm(provider: str):
    if provider == "fake":
        return FakeLLM(CANDIDATES)
    if provider == "anthropic":
        from agentfork.harness import AnthropicLLM
        return AnthropicLLM()
    if provider == "together":
        from agentfork.harness import OpenAICompatLLM
        return OpenAICompatLLM()
    raise ValueError(f"unknown provider: {provider}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

HEADER = (
    "10 fixes over 3 approach branches, 1 inference server, and someone else "
    "is using it too.\n"
    "CPU-only: KV pool, tree cache, eviction, pinning, auth, sandboxes and "
    "candidate checks are REAL;\nthe transformer forward pass is STUBBED "
    "(no GPU/weights), so read prefill tokens and hit rate,\nnot generation "
    "latency.")


class LogUI:
    """Plain sequential log (``--no-ui``): what tests and CI read."""

    def __init__(self, out=sys.stdout):
        self.out = out

    def say(self, msg: str = "") -> None:
        print(msg, file=self.out, flush=True)

    def banner(self, race: Race, args) -> None:
        self.say("== agentfork split-screen race ==")
        self.say(HEADER)
        self.say("")
        ustar = break_even_u(race.prefix_tokens, args.capacity_tokens)
        u = args.noise_request_tokens * args.noise_requests_per_gap
        self.say(f"P (shared prefix)      = {race.prefix_tokens} tokens")
        self.say(f"C (KV pool capacity)   = {args.capacity_tokens} tokens")
        self.say(f"U (neighbor per gap)   = {u} tokens "
                 f"({args.noise_requests_per_gap} x "
                 f"{args.noise_request_tokens})")
        self.say(f"U* = C - P (break-even) = {ustar} tokens  -> "
                 f"{'U > U*: stock loses the prefix' if u > ustar else 'U <= U*: stock keeps the prefix'}")
        self.say(f"N (candidates)         = {args.children} spread over "
                 f"{args.approaches} approach branches of "
                 f"{args.approach_tokens} tokens each, "
                 f"verification forks = {args.verify_children}")
        self.say("tree: root context -> approach -> candidate -> "
                 "verification (a leaf's hit is against its whole lineage)")
        self.say("")

    def arm_start(self, name: str, url: str) -> None:
        self.say(f"--- {name.upper()} arm: fresh server at {url} ---")

    def parent(self, name, prefix_tokens, charged) -> None:
        self.say(f"[{name}] shared context prefilled: {charged} tokens charged "
                 f"(P={prefix_tokens})")

    def child(self, name, label, idx, rec: ChildRecord, total) -> None:
        status = "PASS" if rec.check_passed else ("ERR" if rec.error else "fail")
        self.say(f"[{name}/{label} {idx + 1}/{total}] {rec.branch_id}: "
                 f"depth={rec.depth} cached={rec.cached_tokens} "
                 f"charged={rec.charged_tokens} "
                 f"cache={rec.hit_level} "
                 f"sandbox_setup={rec.sandbox_setup_s * 1e3:.0f}ms "
                 f"check={status}")

    def kills(self, name, n, freed, pool_delta) -> None:
        self.say(f"[{name}] killed {n} loser branch(es): KV freed={freed} "
                 f"tokens, pool used dropped by {pool_delta}")

    def event(self, name, kind, detail="") -> None:
        if kind == "neighbor_deferred":
            return  # counted, not logged per occurrence (it is high volume)
        self.say(f"[{name}] {kind}: {detail}")

    def kv(self, name, used) -> None:
        pass

    def arm_done(self, arm: ArmResult) -> None:
        self.say(f"[{arm.name}] done in {arm.wall_clock_s:.1f}s: "
                 f"hit_rate={arm.parent_hit_rate:.2f} "
                 f"prefill_charged={arm.prefill_charged} "
                 f"peak_kv={arm.peak_kv_used} verified={arm.verified}")
        self.say("")


# ---------------------------------------------------------------------------
# scoreboard + JSON summary
# ---------------------------------------------------------------------------


def scoreboard_rows(stock: ArmResult,
                    agentfork: ArmResult) -> list[tuple[str, str, str]]:
    return [
        ("root-context hit rate", f"{stock.parent_hit_rate * 100:.0f}%",
         f"{agentfork.parent_hit_rate * 100:.0f}%"),
        ("full-lineage (subtree) hit rate",
         f"{stock.subtree_hit_rate * 100:.0f}%",
         f"{agentfork.subtree_hit_rate * 100:.0f}%"),
        ("prefill tokens charged (real)", f"{stock.prefill_charged:,}",
         f"{agentfork.prefill_charged:,}"),
        ("peak KV pool used", f"{stock.peak_kv_used:,}",
         f"{agentfork.peak_kv_used:,}"),
        ("KV released when losers died", f"{stock.kv_freed_by_kills:,}",
         f"{agentfork.kv_freed_by_kills:,}"),
        ("KV still pinned after the kills", f"{stock.pinned_after_kills:,}",
         f"{agentfork.pinned_after_kills:,}"),
        ("sandbox setup (s, total)",
         f"{stock.sandbox_setup_s + stock.template_setup_s:.2f}",
         f"{agentfork.sandbox_setup_s + agentfork.template_setup_s:.2f}"),
        ("wall clock (s)* -- not a speed claim", f"{stock.wall_clock_s:.1f}",
         f"{agentfork.wall_clock_s:.1f}"),
        ("verified winner",
         f"{stock.winner if stock.verified else 'none'}",
         f"{agentfork.winner if agentfork.verified else 'none'}"),
        ("neighbor requests served", f"{stock.neighbor_served:,}",
         f"{agentfork.neighbor_served:,}"),
        ("neighbor requests deferred", f"{stock.neighbor_deferred:,}",
         f"{agentfork.neighbor_deferred:,}"),
    ]


SCOREBOARD_NOTES = """\
* the forward pass is stubbed, so wall clock here is mostly HTTP, pytest and \
process spawning: the gap between the arms is noise, not a speedup. The \
headline numbers are prefill tokens charged and the parent-prefix hit rate, \
which the cache measures for real.
Note: the stock arm also releases KV when its branches die -- but those were \
ordinary evictable pages that the neighbour had already been recycling; it had \
nothing pinned to protect, which is exactly why its shared prefix did not \
survive to the next candidate."""


def scoreboard(stock: ArmResult, agentfork: ArmResult) -> str:
    rows = scoreboard_rows(stock, agentfork)
    width = max(len(r[0]) for r in rows)
    out = [f"{'metric':<{width}}  {'STOCK':>18}  {'AGENTFORK':>18}",
           "-" * (width + 40)]
    out += [f"{k:<{width}}  {a:>18}  {b:>18}" for k, a, b in rows]
    out.append("")
    out.append("* the forward pass is stubbed, so wall clock here is mostly "
               "HTTP, pytest and process spawning:")
    out.append("  the gap between the arms is noise, not a speedup. The "
               "headline numbers are prefill tokens")
    out.append("  charged and the parent-prefix hit rate, which the cache "
               "measures for real.")
    out.append("Note: the stock arm also releases KV when its branches die -- "
               "but those were ordinary")
    out.append("evictable pages that the neighbour had already been "
               "recycling; it had nothing pinned to")
    out.append("protect, which is exactly why its shared prefix did not "
               "survive to the next candidate.")
    return "\n".join(out)


def summary_json(args, race: Race, stock: ArmResult,
                 agentfork: ArmResult) -> dict:
    u = args.noise_request_tokens * args.noise_requests_per_gap
    ustar = break_even_u(race.prefix_tokens, args.capacity_tokens)
    model = pressure_model(PressureScenario(
        prefix_tokens=race.prefix_tokens, n_children=args.children,
        suffix_tokens=args.suffix_tokens, interleaved_tokens=u,
        capacity_tokens=args.capacity_tokens))
    # After the losers die, the only pinned KV is the winner's own committed
    # prefix -- the shared context plus its lineage's unique suffixes, and
    # nothing else. Every loser's KV is gone.
    final = next((c for c in agentfork.verify_children
                  if c.branch_id == agentfork.winner), None)
    expected_pinned = (final.cached_tokens + final.charged_tokens
                       if final else 0)
    return {
        "config": {
            "children": args.children,
            "approaches": args.approaches,
            "approach_tokens": args.approach_tokens,
            "verify_children": args.verify_children,
            "prefix_tokens": race.prefix_tokens,
            "suffix_tokens": args.suffix_tokens,
            "capacity_tokens": args.capacity_tokens,
            "noise_request_tokens": args.noise_request_tokens,
            "noise_requests_per_gap": args.noise_requests_per_gap,
            "interleaved_tokens_u": u,
            "break_even_u": ustar,
            "pressure_regime": "above" if u > ustar else "at_or_below",
            "provider": args.provider,
            "model_output": False,
        },
        "pressure_model": model,
        "arms": {"stock": stock.to_dict(), "agentfork": agentfork.to_dict()},
        "claims": {
            "agentfork_parent_hit_rate": round(agentfork.parent_hit_rate, 4),
            "stock_parent_hit_rate": round(stock.parent_hit_rate, 4),
            "agentfork_subtree_hit_rate": round(agentfork.subtree_hit_rate, 4),
            "stock_subtree_hit_rate": round(stock.subtree_hit_rate, 4),
            "agentfork_prefill_charged": agentfork.prefill_charged,
            "stock_prefill_charged": stock.prefill_charged,
            "agentfork_verified": agentfork.verified,
            "stock_verified": stock.verified,
            "agentfork_kv_freed_by_kills": agentfork.kv_freed_by_kills,
            "agentfork_pinned_after_kills": agentfork.pinned_after_kills,
            "agentfork_expected_pinned_after_kills": expected_pinned,
        },
    }


JSON_BEGIN = "===RACE_DEMO_JSON_BEGIN==="
JSON_END = "===RACE_DEMO_JSON_END==="


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--children", type=int, default=10,
                   help="candidate fixes per fan-out (N)")
    p.add_argument("--approaches", type=int, default=3,
                   help="divergent approach branches forked off the root (M); "
                        "candidates are spread over them, so the tree is "
                        "root -> approach -> candidate -> verification")
    p.add_argument("--approach-tokens", type=int, default=None,
                   help="tokens of reasoning each approach commits (A); "
                        "default max(2*S, P//8)")
    p.add_argument("--verify-children", type=int, default=3,
                   help="re-forks of the winner in the verification round")
    p.add_argument("--prefix-tokens", type=int, default=8192,
                   help="shared repo-context length (P)")
    p.add_argument("--suffix-tokens", type=int, default=256,
                   help="unique tokens each candidate adds after the shared "
                        "prefix (S)")
    p.add_argument("--capacity-tokens", type=int, default=24576,
                   help="KV pool token capacity of the server (C)")
    p.add_argument("--noise-request-tokens", type=int, default=1024,
                   help="tokens per noisy-neighbor request")
    p.add_argument("--noise-requests-per-gap", type=int, default=None,
                   help="neighbor requests between candidates; default is the "
                        "smallest count with U > C - P")
    p.add_argument("--provider", choices=["fake", "anthropic", "together"],
                   default="fake",
                   help="candidate source; 'fake' is deterministic + offline")
    p.add_argument("--port", type=int, default=8765,
                   help="port for the browser dashboard")
    p.add_argument("--no-open", action="store_true",
                   help="do not open a browser automatically")
    p.add_argument("--no-ui", action="store_true",
                   help="headless: no dashboard, plain sequential log plus a "
                        "JSON summary (what tests and CI read)")
    p.add_argument("--json-out", default=None,
                   help="also write the JSON summary to this path")
    p.add_argument("--admin-api-key", default="race-demo-admin-key")
    args = p.parse_args(argv)
    if args.approach_tokens is None:
        args.approach_tokens = max(2 * args.suffix_tokens,
                                   args.prefix_tokens // 8)
    if args.noise_requests_per_gap is None:
        headroom = max(0, args.capacity_tokens - args.prefix_tokens)
        args.noise_requests_per_gap = (
            math.floor(headroom / args.noise_request_tokens) + 1)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.no_ui:
        ui = LogUI()
    else:
        from demo.race_web import WebUI
        ui = WebUI(args, LogUI(), port=args.port,
                   open_browser=not args.no_open)
    race = Race(args, ui)
    ui.banner(race, args)
    try:
        stock = race.run_arm("stock")
        agentfork = race.run_arm("agentfork")
    finally:
        race.close()

    summary = summary_json(args, race, stock, agentfork)
    print()
    print(scoreboard(stock, agentfork), flush=True)
    if not args.no_ui:
        ui.scoreboard(scoreboard_rows(stock, agentfork), SCOREBOARD_NOTES)
        ui.finish()
        print(f"\nrace finished; dashboard still served at {ui.url} "
              "(Ctrl-C to exit)", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
    if args.no_ui:
        print(JSON_BEGIN)
        print(json.dumps(summary, indent=2))
        print(JSON_END, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
