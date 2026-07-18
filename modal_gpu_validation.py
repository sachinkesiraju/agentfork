"""GPU validation of the TreeRadixCache patch on Modal.

Mounts the local sglang checkout (main @ 40517b593 + tree_radix_cache patch)
over the lmsysorg/sglang image (for CUDA torch + deps) and, on a real GPU:

1. real-HBM pool test: TreeRadixCache + MHATokenToKVPool/TokenToKVPoolAllocator
   on device=cuda — measures actual HBM bytes, slot dedup, and reclaim-to-zero.
2. the TreeRadixCache unit-test file, on the GPU host.
3. stock live-engine baseline: sgl.Engine (Qwen3-0.6B) — 10 requests sharing
   a long prefix. This measures SGLang's existing RadixAttention reuse; the
   engine is not configured to call TreeRadixCache's branch APIs.
4. patched live-engine branch path and sustained-pressure comparison using
   request-carried tree/branch identity.

Run: SGLANG_DIR=/path/to/sglang python3 -m modal run modal_gpu_validation.py
(SGLANG_DIR defaults to ~/sglang; it must be a checkout with patches 0001,
0002, and 0003 applied in order.)
"""

import json
import os
import random

import modal

SGLANG_DIR = os.path.expanduser(os.environ.get("SGLANG_DIR", "~/sglang"))
if not SGLANG_DIR.startswith("/root/") and not os.path.isdir(
    os.path.join(SGLANG_DIR, "python", "sglang")
):
    raise FileNotFoundError(f"SGLANG_DIR is not an SGLang checkout: {SGLANG_DIR}")

app = modal.App("agentfork-gpu-validation")

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
    .add_local_dir(f"{SGLANG_DIR}/test", remote_path="/root/sgltest")
)


@app.function(image=image, gpu="A10G", timeout=1800)
def validate() -> str:
    import os
    import subprocess
    import sys
    import time

    # modal's baked requirements downgrade typing_extensions below what
    # pydantic_core in the sglang image needs; restore it before importing
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--upgrade", "typing_extensions>=4.15"], check=True)
    sys.path.insert(0, "/root/sgl")
    os.environ["PYTHONPATH"] = "/root/sgl"
    import torch

    out = {"gpu": torch.cuda.get_device_name(0)}

    ENGINE_CHILDREN = 12
    PRESSURE_REQUESTS_PER_CHILD = 80

    def apply_cache_pressure(engine, count=PRESSURE_REQUESTS_PER_CHILD):
        for index in range(count):
            engine.generate(
                (f"Noise tenant {index} unique context. " * 400) + "Task:",
                {"max_new_tokens": 1},
            )
        return count

    # --- 1. real HBM pool ---
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.tree_radix_cache import TreeRadixCache

    POOL, PREFIX, SUFFIX, N = 65536, 32000, 500, 10
    base = torch.cuda.memory_allocated()
    kvcache = MHATokenToKVPool(
        size=POOL, page_size=1, dtype=torch.float16, head_num=8, head_dim=64,
        layer_num=16, device="cuda", enable_memory_saver=False)
    alloc = TokenToKVPoolAllocator(size=POOL, dtype=torch.float16,
                                   device="cuda", kvcache=kvcache, need_sort=False)
    out["hbm_pool_gib"] = round((torch.cuda.memory_allocated() - base) / 2**30, 2)
    cache = TreeRadixCache(CacheInitParams(
        disable=False, req_to_token_pool=None,
        token_to_kv_pool_allocator=alloc, page_size=1))
    used = lambda: POOL - alloc.available_size()  # noqa: E731

    cache.create_agent_tree("parent")
    ptok = list(range(PREFIX))
    pslots = alloc.alloc(PREFIX)
    if pslots is None:
        raise RuntimeError("failed to allocate parent KV slots")
    cache.extend_tree("parent", ptok, value=pslots.clone())
    out["parent_used"] = used()
    t0 = time.perf_counter()
    for i in range(N):
        br = cache.fork_branch("parent", f"c{i}")
        assert cache.match_tree_prefix(br.branch_id, ptok) == PREFIX
        cs = alloc.alloc(SUFFIX)
        if cs is None:
            raise RuntimeError(f"failed to allocate KV slots for {br.branch_id}")
        cache.extend_tree(br.branch_id,
                          [10_000_000 + i * SUFFIX + j for j in range(SUFFIX)],
                          value=torch.cat([pslots, cs]))
    torch.cuda.synchronize()
    out["fork_10_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    out["after_forks_used"] = used()
    out["dedup_x"] = round(((PREFIX + SUFFIX) * N + PREFIX) / used(), 2)
    t0 = time.perf_counter()
    for i in range(N):
        cache.kill_tree(f"c{i}")
    cache.kill_tree("parent")
    torch.cuda.synchronize()
    out["kill_all_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    out["after_kill_used"] = used()
    assert out["after_kill_used"] == 0

    # --- 2. unit tests on GPU host ---
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q",
         "/root/sgltest/registered/unit/mem_cache/test_tree_radix_cache.py"],
        capture_output=True, text=True, env={**os.environ})
    out["unit_tests"] = (
        r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])
    if r.returncode != 0:
        raise RuntimeError(f"TreeRadixCache tests failed: {r.stderr[-1000:]}")

    # --- 3. stock live-engine prefix-cache baseline ---
    import sglang as sgl
    eng = sgl.Engine(
        model_path="Qwen/Qwen3-0.6B",
        mem_fraction_static=0.6,
        log_level="warning",
        disable_cuda_graph=True,
    )
    try:
        prefix = "You are a helpful assistant. " * 400  # ~2.4k tokens shared
        parent_prompt = prefix + "Parent:"
        t0 = time.perf_counter()
        first = eng.generate(parent_prompt, {"max_new_tokens": 4})
        parent_s = time.perf_counter() - t0
        parent_text = first.get("text", "")
        pressure_requests = 0
        cached, times = [], []
        for i in range(ENGINE_CHILDREN):
            pressure_requests += apply_cache_pressure(eng)
            t0 = time.perf_counter()
            result = eng.generate(
                parent_prompt + parent_text + f" Child {i}:",
                {"max_new_tokens": 4},
            )
            times.append(time.perf_counter() - t0)
            cached.append(result["meta_info"]["cached_tokens"])
        out["engine"] = {
            "prompt_tokens": first["meta_info"]["prompt_tokens"],
            "parent_prefill_s": round(parent_s, 2),
            "parent_generation_s": parent_s,
            "pressure_requests": pressure_requests,
            "sibling_cached_tokens": cached,
            "sibling_gen_s_p50": round(sorted(times)[len(times) // 2], 3),
            "total_generation_s": round(parent_s + sum(times), 4),
            "sibling_generation_s": times,
        }
    finally:
        eng.shutdown()

    # --- 4. live tree-aware engine request path ---
    eng = sgl.Engine(
        model_path="Qwen/Qwen3-0.6B",
        mem_fraction_static=0.6,
        log_level="warning",
        radix_cache_backend="tree_radix",
        tree_cache_quota_tokens=65536,
        disable_cuda_graph=True,
    )
    try:
        prefix = "You are a helpful assistant. " * 400
        parent_prompt = prefix + "Parent:"
        t0 = time.perf_counter()
        parent = eng.generate(
            parent_prompt,
            {"max_new_tokens": 4},
            tree_id="validation-tree",
            branch_id="parent",
            branch_reserve_tokens=4,
        )
        tree_parent_s = time.perf_counter() - t0
        parent_text = parent.get("text", "")
        pressure_requests = 0
        cached, times = [], []
        for i in range(ENGINE_CHILDREN):
            pressure_requests += apply_cache_pressure(eng)
            t0 = time.perf_counter()
            result = eng.generate(
                parent_prompt + parent_text + f" Child {i}:",
                {"max_new_tokens": 4},
                tree_id="validation-tree",
                branch_id=f"child-{i}",
                parent_id="parent",
                branch_end=True,
                branch_reserve_tokens=4,
            )
            times.append(time.perf_counter() - t0)
            cached.append(result["meta_info"]["cached_tokens"])
        telemetry = eng.tree_cache_op("telemetry", "parent")
        killed = eng.tree_cache_op("kill", "parent")
        out["tree_engine"] = {
            "parent_prompt_tokens": parent["meta_info"]["prompt_tokens"],
            "parent_generation_s": tree_parent_s,
            "pressure_requests": pressure_requests,
            "sibling_cached_tokens": cached,
            "sibling_gen_s_p50": round(sorted(times)[len(times) // 2], 3),
            "total_generation_s": round(tree_parent_s + sum(times), 4),
            "sibling_generation_s": times,
            "telemetry_before_kill": telemetry.value,
            "kill_value": killed.value,
        }
    finally:
        eng.shutdown()

    stock_total = sum(out["engine"]["sibling_generation_s"])
    tree_total = sum(out["tree_engine"]["sibling_generation_s"])
    ratios = []
    rng = random.Random(0)
    stock_samples = out["engine"]["sibling_generation_s"]
    tree_samples = out["tree_engine"]["sibling_generation_s"]
    paired = list(zip(stock_samples, tree_samples))
    for _ in range(1000):
        sample = [rng.choice(paired) for _ in paired]
        ratios.append(sum(item[0] for item in sample) / sum(item[1] for item in sample))
    ratios.sort()
    out["vge"] = {
        "uplift": round(stock_total / tree_total, 4),
        "ci95_low": round(ratios[25], 4),
        "ci95_high": round(ratios[974], 4),
        "target_passed": stock_total / tree_total >= 1.5 and ratios[25] >= 1.2,
    }

    return json.dumps(out, indent=2)


@app.local_entrypoint()
def main():
    print(validate.remote())
