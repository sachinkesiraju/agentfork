"""Live A10G validation of the cache-pressure break-even (U* = C - P).

Runs the sustained-pressure workload from report/PRESSURE.md on a real patched
SGLang engine at several interleaved-traffic levels U, for both stock
RadixAttention and the tree-pinned backend, and records the measured
parent-prefix reuse (``cached_tokens``) and wall-clock generation time next to
the cost model's prediction.

The KV pool is capped to a known capacity ``C`` (``max_total_tokens``) so the
break-even ``U* = C - P`` lands inside the swept U range: below it stock keeps
the prefix (child reuses ~P cached tokens), above it stock re-prefills it
(child reuses ~0), while the tree pin holds the prefix at every U.

Run: SGLANG_DIR=/path/to/sglang python3 -m modal run modal_pressure_sweep.py
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

app = modal.App("agentfork-pressure-sweep")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "sglang[srt]==0.5.14",
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

# workload shape (mirrors report/PRESSURE.md and the cost-model example)
CAPACITY = 32768          # C: KV pool cap so U* = C - P falls inside the sweep
N_CHILDREN = 10           # N: fanout
U_LEVELS = [0, 24000, 48000, 96000]   # unrelated tokens interleaved per child
MAX_NEW = 8               # decode length per child
NOISE_WORDS = 1000        # ~unique tokens per noise chunk (< pool)


@app.function(image=image, gpu="A10G", timeout=3600)
def sweep() -> str:
    import subprocess
    import sys
    import time

    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--upgrade", "typing_extensions>=4.15"], check=True)
    sys.path.insert(0, "/root/sgl")
    os.environ["PYTHONPATH"] = "/root/sgl"

    import torch

    import sglang as sgl

    out = {"gpu": torch.cuda.get_device_name(0), "config": {
        "capacity_C": CAPACITY, "n_children": N_CHILDREN,
        "u_levels": U_LEVELS, "max_new_tokens": MAX_NEW}}

    prefix = "You are a helpful assistant. " * 400  # ~2.4k shared tokens
    parent_prompt = prefix + "Parent:"

    noise_counter = [0]

    def inject_noise(engine, target):
        """Insert ~target unrelated prefill tokens as unique short requests."""
        injected = 0
        while injected < target:
            noise_counter[0] += 1
            base = noise_counter[0] * NOISE_WORDS
            words = " ".join(f"z{base + j}" for j in range(NOISE_WORDS))
            r = engine.generate(words + " Task:", {"max_new_tokens": 1})
            injected += r["meta_info"]["prompt_tokens"]
        return injected

    def run_engine(engine, tree_aware):
        rows = []
        for u in U_LEVELS:
            engine.flush_cache()
            time.sleep(0.5)
            # (re-)prime the parent prefix
            if tree_aware:
                parent = engine.generate(
                    parent_prompt, {"max_new_tokens": 4},
                    tree_id="sweep", branch_id="parent",
                    branch_reserve_tokens=4)
            else:
                parent = engine.generate(parent_prompt, {"max_new_tokens": 4})
            parent_text = parent.get("text", "")
            p_tokens = parent["meta_info"]["prompt_tokens"]

            cached, times, injected = [], [], []
            for i in range(N_CHILDREN):
                injected.append(inject_noise(engine, u))
                child_prompt = parent_prompt + parent_text + f" Child {i}:"
                t0 = time.perf_counter()
                if tree_aware:
                    res = engine.generate(
                        child_prompt, {"max_new_tokens": MAX_NEW},
                        tree_id="sweep", branch_id=f"child-{i}",
                        parent_id="parent", branch_end=True,
                        branch_reserve_tokens=4)
                else:
                    res = engine.generate(
                        child_prompt, {"max_new_tokens": MAX_NEW})
                times.append(time.perf_counter() - t0)
                cached.append(res["meta_info"]["cached_tokens"])
            if tree_aware:
                engine.tree_cache_op("kill", "parent")
            rows.append({
                "U": u,
                "parent_prompt_tokens": p_tokens,
                "mean_injected_per_gap": round(sum(injected) / len(injected)),
                "child_cached_tokens": cached,
                "mean_cached": round(sum(cached) / len(cached), 1),
                "parent_hit_rate": round(
                    sum(1 for c in cached if c >= p_tokens - 8) / len(cached), 3),
                "child_gen_s": [round(t, 4) for t in times],
                "child_total_s": round(sum(times), 4),
            })
        return rows

    # --- stock RadixAttention ---
    eng = sgl.Engine(model_path="Qwen/Qwen3-0.6B", mem_fraction_static=0.6,
                     max_total_tokens=CAPACITY, log_level="warning",
                     disable_cuda_graph=True)
    try:
        out["stock"] = run_engine(eng, tree_aware=False)
    finally:
        eng.shutdown()

    # --- tree-pinned backend ---
    eng = sgl.Engine(model_path="Qwen/Qwen3-0.6B", mem_fraction_static=0.6,
                     max_total_tokens=CAPACITY, log_level="warning",
                     radix_cache_backend="tree_radix",
                     tree_cache_quota_tokens=CAPACITY,
                     disable_cuda_graph=True)
    try:
        out["tree"] = run_engine(eng, tree_aware=True)
    finally:
        eng.shutdown()

    # --- per-U reconciliation: measured VGE + model prediction ---
    p = out["stock"][0]["parent_prompt_tokens"]
    ustar = CAPACITY - p
    recon = []
    for si, ti in zip(out["stock"], out["tree"]):
        u = si["U"]
        vge = si["child_total_s"] / ti["child_total_s"]
        recon.append({
            "U": u,
            "break_even_U": ustar,
            "model_predicts_win": u > ustar,
            "stock_parent_hit_rate": si["parent_hit_rate"],
            "tree_parent_hit_rate": ti["parent_hit_rate"],
            "measured_vge": round(vge, 4),
        })
    out["reconciliation"] = {"P_measured": p, "break_even_U": ustar,
                             "rows": recon}
    return json.dumps(out, indent=2)


@app.local_entrypoint()
def main():
    print(sweep.remote())
