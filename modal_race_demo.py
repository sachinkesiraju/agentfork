"""The split-screen race on a real GPU with real weights (Modal A10G).

``demo/race_demo.py`` runs the race on CPU with the transformer forward pass
stubbed: honest about token accounting, silent about time. This runs the same
race against a real ``sgl.Engine`` serving **Qwen3-0.6B on an A10G**, so every
number below -- prefill, decode, wall clock -- comes from a real model:

  STOCK      stock SGLang RadixAttention. The shared repo context is prefilled
             once; each candidate is an ordinary request that hopes the prefix
             is still resident, and gets a cold sandbox workspace.
  AGENTFORK  the patched ``tree_radix`` backend. The context is pinned on a
             parent branch, candidates are forked children charged only their
             own suffix, losers are killed (KV reclaimed immediately), and the
             winner is re-forked for a verification round.

Between candidates an unrelated tenant injects ``U`` tokens of traffic, with
``U > U* = C - P`` from ``agentfork/bench/cost_model.py`` -- the regime where a
stock LRU cannot hold the shared prefix and a pinned branch is immune.

WHAT IS REAL / WHAT IS NOT
--------------------------
Real: the model and its weights, prefill and decode, the KV pool on real HBM,
prefix matching and eviction, branch pinning/kill, the neighbour's traffic, the
sandboxes (real subprocesses) and the candidate checks (real ``pytest``).
**Wall clock here IS a model-speed measurement** -- unlike the CPU demo.

Not real: which patch a branch proposes. Qwen3-0.6B cannot reliably write a
correct fix, so each branch's *patch* comes from the same deterministic
candidate list the CPU demo uses, while the model genuinely generates on that
branch's prompt. The model's own output is checked too and reported separately
as ``model_generated_fix_passed`` -- an honest 0-or-so, not folded into the
race result.

Run:  SGLANG_DIR=/path/to/sglang python3 -m modal run modal_race_demo.py
(SGLANG_DIR must be a checkout at 40517b593 with patches 0001-0003 applied.)
"""

import json
import os

import modal

SGLANG_DIR = os.path.expanduser(os.environ.get("SGLANG_DIR", "~/sglang"))
if not SGLANG_DIR.startswith("/root/") and not os.path.isdir(
    os.path.join(SGLANG_DIR, "python", "sglang")
):
    raise FileNotFoundError(f"SGLANG_DIR is not an SGLang checkout: {SGLANG_DIR}")

app = modal.App("agentfork-race-demo")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "sglang[srt]==0.5.14",
        "pytest>=8.0",
        "typing_extensions>=4.15",
        index_url="https://pypi.org/simple",
    )
    .env({
        "AGENTFORK_VALIDATION": "1",
        "SGLANG_DIR": "/root/sgl",
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
    })
    .add_local_dir(f"{SGLANG_DIR}/python/sglang", remote_path="/root/sgl/sglang")
)

MODEL = "Qwen/Qwen3-0.6B"
CAPACITY = 49152          # C: KV pool cap, so U* = C - P lands in range
N_CHILDREN = 10           # N: candidate fixes per fan-out
VERIFY_CHILDREN = 3       # re-forks of the winner
MAX_NEW = 32              # decode length per candidate (real generation)
NOISE_WORDS = 900         # ~unique tokens per neighbour request
CONTEXT_REPEATS = 900     # shared repo context length knob (-> P ~= 12k)

BUGGY = '''def clamp(x, lo, hi):
    """Clamp x into the inclusive range [lo, hi]."""
    return min(x, hi)
'''

BASIC_TEST = '''from clamp import clamp

def test_within():
    assert clamp(5, 0, 10) == 5

def test_below():
    assert clamp(-3, 0, 10) == 0
'''

EDGE_TEST = '''from clamp import clamp

def test_above():
    assert clamp(100, 0, 10) == 10

def test_on_bounds():
    assert clamp(0, 0, 10) == 0
    assert clamp(10, 0, 10) == 10
'''

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
CANDIDATES = [BAD_FIXES[0], GOOD_FIX] + BAD_FIXES[1:]


@app.function(image=image, gpu="A10G", memory=32768, timeout=3600)
def race() -> str:
    import re
    import shutil
    import subprocess
    import sys
    import tempfile
    import time

    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--upgrade", "typing_extensions>=4.15"], check=True)
    sys.path.insert(0, "/root/sgl")
    os.environ["PYTHONPATH"] = "/root/sgl"

    import torch
    from transformers import AutoTokenizer

    import sglang as sgl

    tok = AutoTokenizer.from_pretrained(MODEL)

    def ids(text: str) -> list:
        return tok(text, add_special_tokens=False).input_ids

    out = {"gpu": torch.cuda.get_device_name(0), "model": MODEL,
           "config": {"capacity_C": CAPACITY, "n_children": N_CHILDREN,
                      "verify_children": VERIFY_CHILDREN,
                      "max_new_tokens": MAX_NEW}}

    context = (
        "# repo: tinyproj\n"
        "# task: clamp.py is wrong; propose a fix that passes the suite.\n\n"
        f"clamp.py:\n{BUGGY}\n"
        f"test_clamp.py (failing):\n{BASIC_TEST}\n"
        "# --- surrounding repo context the agent was given ---\n"
        + "# ctx: unrelated but in-context repo lines read while triaging\n"
        * CONTEXT_REPEATS)

    # -- real sandboxes and real checks ------------------------------------

    def make_workspace() -> str:
        """A cold workspace: create the dir and write the test suite."""
        d = tempfile.mkdtemp(prefix="race-")
        for name, body in (("test_clamp.py", BASIC_TEST),
                           ("test_edge.py", EDGE_TEST)):
            with open(os.path.join(d, name), "w") as f:
                f.write(body)
        return d

    def run_check(workdir: str, code: str) -> bool:
        with open(os.path.join(workdir, "clamp.py"), "w") as f:
            f.write(code)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", workdir],
            capture_output=True, text=True, cwd=workdir, timeout=120)
        return proc.returncode == 0

    def extract_code(text: str) -> str:
        """Best-effort: pull a def clamp(...) out of what the model wrote."""
        m = re.search(r"def clamp\(.*?\n(?:[ \t]+.*\n?)+", text)
        return m.group(0) if m else ""

    noise_counter = [0]

    def inject_noise(engine, target_tokens: int) -> int:
        injected = 0
        while injected < target_tokens:
            noise_counter[0] += 1
            base = noise_counter[0] * NOISE_WORDS
            words = " ".join(f"z{base + j}" for j in range(NOISE_WORDS))
            r = engine.generate(words + " Task:", {"max_new_tokens": 1})
            injected += r["meta_info"]["prompt_tokens"]
        return injected

    # -- one arm ------------------------------------------------------------

    def run_arm(engine, tree_aware: bool, u_tokens: int, p_tokens: int):
        name = "agentfork" if tree_aware else "stock"
        arm = {"name": name, "children": [], "verification": [],
               "model_generated_fix_passed": 0, "sandbox_setup_s": 0.0,
               "injected_per_gap": []}
        engine.flush_cache()
        time.sleep(0.5)

        t_arm = time.perf_counter()
        # Prompts are built from token ids rather than concatenated text: a
        # branch commits ``origin_input_ids + output_ids``, and a child must
        # extend that exactly -- re-tokenizing a concatenation can merge
        # across the seam and lose the prefix.
        parent_ids = ids(context + "\n# propose a fix:\n")
        t0 = time.perf_counter()
        if tree_aware:
            parent = engine.generate(
                input_ids=parent_ids, sampling_params={"max_new_tokens": 4},
                tree_id="race", branch_id="parent", branch_reserve_tokens=64)
        else:
            parent = engine.generate(input_ids=parent_ids,
                                     sampling_params={"max_new_tokens": 4})
        base_ids = parent_ids + list(parent.get("output_ids") or [])
        arm["parent_prefill_s"] = round(time.perf_counter() - t0, 4)
        arm["parent_prompt_tokens"] = parent["meta_info"]["prompt_tokens"]

        # the agentfork arm prepares the workspace once, on the parent, and
        # children inherit it; the stock arm rebuilds it per child.
        t0 = time.perf_counter()
        template = make_workspace() if tree_aware else None
        arm["template_setup_s"] = round(time.perf_counter() - t0, 4)

        def one_child(index: int, label: str, branch_id: str,
                      prompt_ids: list, code: str, parent_branch: str):
            arm["injected_per_gap"].append(inject_noise(engine, u_tokens))
            t0 = time.perf_counter()
            if tree_aware:
                res = engine.generate(
                    input_ids=prompt_ids,
                    sampling_params={"max_new_tokens": MAX_NEW},
                    tree_id="race", branch_id=branch_id,
                    parent_id=parent_branch, branch_reserve_tokens=64)
            else:
                res = engine.generate(
                    input_ids=prompt_ids,
                    sampling_params={"max_new_tokens": MAX_NEW})
            gen_s = time.perf_counter() - t0
            meta = res["meta_info"]

            t0 = time.perf_counter()
            if tree_aware:
                workdir = tempfile.mkdtemp(prefix="race-child-")
                shutil.copytree(template, workdir, dirs_exist_ok=True)
            else:
                workdir = make_workspace()
            setup_s = time.perf_counter() - t0
            arm["sandbox_setup_s"] += setup_s

            passed = run_check(workdir, code)
            # separately: did what the model actually wrote pass? (reported,
            # never folded into the race result)
            model_code = extract_code(res.get("text", ""))
            if model_code:
                probe_dir = make_workspace()
                if run_check(probe_dir, model_code):
                    arm["model_generated_fix_passed"] += 1
                shutil.rmtree(probe_dir, ignore_errors=True)
            shutil.rmtree(workdir, ignore_errors=True)

            return {
                "index": index, "label": label, "branch_id": branch_id,
                "prompt_tokens": meta["prompt_tokens"],
                "cached_tokens": meta["cached_tokens"],
                "charged_tokens": meta["prompt_tokens"] - meta["cached_tokens"],
                "completion_tokens": meta.get("completion_tokens", 0),
                "parent_hit": meta["cached_tokens"] >= p_tokens - 8,
                "gen_s": round(gen_s, 4),
                "sandbox_setup_s": round(setup_s, 4),
                "check_passed": passed,
                "committed_ids": prompt_ids + list(res.get("output_ids") or []),
                "text": res.get("text", "")[:160],
            }

        for i, code in enumerate(CANDIDATES[:N_CHILDREN]):
            suffix = (f"\n### CANDIDATE clamp.py ###\n{code}\n"
                      f"# rationale {i}:\n")
            arm["children"].append(one_child(
                i, "fix", f"child-{i}", base_ids + ids(suffix), code,
                "parent"))

        winner = next((c for c in arm["children"] if c["check_passed"]), None)
        arm["round1_winner"] = winner["branch_id"] if winner else None

        # kill the losers and measure what comes back
        if tree_aware:
            before = engine.tree_cache_op("telemetry", "parent").value
            freed = 0
            for c in arm["children"]:
                if c["branch_id"] != arm["round1_winner"]:
                    v = engine.tree_cache_op("kill", c["branch_id"]).value
                    # kill returns released tokens, either bare or in a dict
                    freed += (v.get("released_tokens", 0) if isinstance(v, dict)
                              else (v or 0))
            after = engine.tree_cache_op("telemetry", "parent").value
            arm["telemetry_before_kills"] = before
            arm["telemetry_after_kills"] = after
            arm["kv_freed_by_kills"] = freed

        # verification round: re-fork the winner
        if winner:
            wcode = CANDIDATES[winner["index"]]
            wids = winner["committed_ids"]
            for j in range(VERIFY_CHILDREN):
                arm["verification"].append(one_child(
                    j, "verify", f"verify-{j}",
                    wids + ids(f"\n### VERIFICATION ROUND ###\n# audit {j}\n"),
                    wcode, winner["branch_id"] if tree_aware else "parent"))
        if tree_aware:
            engine.tree_cache_op("kill", "parent")
        if template:
            shutil.rmtree(template, ignore_errors=True)

        arm["wall_clock_s"] = round(time.perf_counter() - t_arm, 3)
        rows = arm["children"] + arm["verification"]
        arm["parent_hit_rate"] = round(
            sum(1 for r in rows if r["parent_hit"]) / len(rows), 4)
        arm["prefill_charged"] = sum(r["charged_tokens"] for r in rows)
        arm["prefill_cached"] = sum(r["cached_tokens"] for r in rows)
        arm["generation_s"] = round(sum(r["gen_s"] for r in rows), 4)
        arm["sandbox_setup_s"] = round(arm["sandbox_setup_s"], 4)
        arm["verified"] = bool(arm["verification"]) and all(
            r["check_passed"] for r in arm["verification"])
        for r in rows:
            r.pop("committed_ids", None)   # keep the JSON readable
        return arm

    # -- measure P with the real tokenizer, then size the neighbour ---------

    eng = sgl.Engine(model_path=MODEL, mem_fraction_static=0.6,
                     max_total_tokens=CAPACITY, log_level="warning",
                     disable_cuda_graph=True)
    try:
        probe = eng.generate(context + "\n# propose a fix:\n",
                             {"max_new_tokens": 1})
        p_tokens = probe["meta_info"]["prompt_tokens"]
        ustar = CAPACITY - p_tokens
        u_tokens = ustar + NOISE_WORDS      # comfortably above break-even
        out["config"].update({"P_measured": p_tokens, "break_even_U": ustar,
                              "U": u_tokens, "pressure_regime": "above"})
        eng.flush_cache()
        out["stock"] = run_arm(eng, False, u_tokens, p_tokens)
    finally:
        eng.shutdown()

    eng = sgl.Engine(model_path=MODEL, mem_fraction_static=0.6,
                     max_total_tokens=CAPACITY, log_level="warning",
                     radix_cache_backend="tree_radix",
                     tree_cache_quota_tokens=CAPACITY,
                     disable_cuda_graph=True)
    try:
        out["agentfork"] = run_arm(eng, True, u_tokens, p_tokens)
    finally:
        eng.shutdown()

    s, a = out["stock"], out["agentfork"]
    out["claims"] = {
        "stock_parent_hit_rate": s["parent_hit_rate"],
        "agentfork_parent_hit_rate": a["parent_hit_rate"],
        "stock_prefill_charged": s["prefill_charged"],
        "agentfork_prefill_charged": a["prefill_charged"],
        "prefill_ratio": round(
            s["prefill_charged"] / max(1, a["prefill_charged"]), 3),
        # real weights: this one IS a speed measurement
        "stock_generation_s": s["generation_s"],
        "agentfork_generation_s": a["generation_s"],
        "generation_speedup": round(
            s["generation_s"] / max(1e-9, a["generation_s"]), 3),
        "stock_verified": s["verified"],
        "agentfork_verified": a["verified"],
        "agentfork_kv_freed_by_kills": a.get("kv_freed_by_kills"),
        "model_generated_fix_passed": {
            "stock": s["model_generated_fix_passed"],
            "agentfork": a["model_generated_fix_passed"],
            "of": N_CHILDREN + VERIFY_CHILDREN,
        },
    }
    return json.dumps(out, indent=2)


@app.local_entrypoint()
def main():
    print(race.remote())
