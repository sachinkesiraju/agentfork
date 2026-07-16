"""GPU validation of the TreeRadixCache patch on Modal.

Mounts the local sglang checkout (main @ 40517b593 + tree_radix_cache patch)
over the lmsysorg/sglang image (for CUDA torch + deps) and, on a real GPU:

1. real-HBM pool test: TreeRadixCache + MHATokenToKVPool/TokenToKVPoolAllocator
   on device=cuda — measures actual HBM bytes, slot dedup, and reclaim-to-zero.
2. the TreeRadixCache unit-test file, on the GPU host.
3. stock live-engine baseline: sgl.Engine (Qwen3-0.6B) — 10 requests sharing
   a long prefix. This measures SGLang's existing RadixAttention reuse; the
   engine is not configured to call TreeRadixCache's branch APIs.

Run: SGLANG_DIR=/path/to/sglang python3 -m modal run modal_gpu_validation.py
(SGLANG_DIR defaults to ~/sglang; it must be a checkout with the
patches/0001-sglang-tree-radix-cache.patch applied.)
"""

import json
import os

import modal

SGLANG_DIR = os.path.expanduser(os.environ.get("SGLANG_DIR", "~/sglang"))

app = modal.App("agentfork-g10")

image = (
    modal.Image.from_registry("lmsysorg/sglang:latest")
    # modal's baked requirements downgrade typing_extensions below what
    # pydantic_core in the sglang image needs
    .run_commands("python3 -m pip install --upgrade 'typing_extensions>=4.15' "
                  "--index-url https://pypi.org/simple")
    # the mounted sglang main wants flashinfer 0.6.14 but only 0.6.13 cubins
    # are published; skip the version assert and use the image's 0.6.12
    .env({"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1"})
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
    cache.extend_tree("parent", ptok, value=pslots.clone())
    out["parent_used"] = used()
    t0 = time.perf_counter()
    for i in range(N):
        br = cache.fork_branch("parent", f"c{i}")
        assert cache.match_tree_prefix(br.branch_id, ptok) == PREFIX
        cs = alloc.alloc(SUFFIX)
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
    out["unit_tests"] = r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:]

    # --- 3. stock live-engine prefix-cache baseline ---
    try:
        import sglang as sgl
        eng = sgl.Engine(model_path="Qwen/Qwen3-0.6B", mem_fraction_static=0.6,
                         log_level="warning")
        prefix = "You are a helpful assistant. " * 400  # ~2.4k tokens shared
        t0 = time.perf_counter()
        first = eng.generate(prefix + "Task 0:", {"max_new_tokens": 4})
        parent_s = time.perf_counter() - t0
        cached, times = [], []
        for i in range(1, 11):
            t0 = time.perf_counter()
            r = eng.generate(prefix + f"Task {i}:", {"max_new_tokens": 4})
            times.append(time.perf_counter() - t0)
            cached.append(r["meta_info"]["cached_tokens"])
        out["engine"] = {
            "prompt_tokens": first["meta_info"]["prompt_tokens"],
            "parent_prefill_s": round(parent_s, 2),
            "sibling_cached_tokens": cached,
            "sibling_gen_s_p50": round(sorted(times)[5], 3),
        }
        eng.shutdown()
    except Exception as e:  # noqa: BLE001
        out["engine"] = f"FAILED: {type(e).__name__}: {e}"

    return json.dumps(out, indent=2)


@app.local_entrypoint()
def main():
    print(validate.remote())
