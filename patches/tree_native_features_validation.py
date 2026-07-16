"""Quantitative validation of the tree-native features on the real
SGLang pool/allocator (CPU device): per-tree quotas + fairness, fork-time
reservations, priority demotion/promotion under real memory pressure,
ABI invalidation + fallback re-prefill, and per-tree telemetry.
"""

import json
import time

import torch

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.tree_radix_cache import TreeQuotaExceeded, TreeRadixCache

POOL_TOKENS = 300_000


def make_cache(quota=None):
    kvcache = MHATokenToKVPool(
        size=POOL_TOKENS, page_size=1, dtype=torch.float16,
        head_num=2, head_dim=16, layer_num=2, device="cpu",
        enable_memory_saver=False,
    )
    alloc = TokenToKVPoolAllocator(
        size=POOL_TOKENS, dtype=torch.float16, device="cpu",
        kvcache=kvcache, need_sort=False,
    )
    params = CacheInitParams(
        disable=False, req_to_token_pool=None,
        token_to_kv_pool_allocator=alloc, page_size=1,
    )
    return TreeRadixCache(params, tree_quota_tokens=quota), alloc


results = {}

# --- 1. per-tree quota + fairness: a runaway tree cannot starve others ----
cache, alloc = make_cache(quota=50_000)
cache.create_agent_tree("greedy")
slots = alloc.alloc(40_000)
cache.extend_tree("greedy", list(range(40_000)), value=slots)
try:
    s2 = alloc.alloc(20_000)
    cache.extend_tree("greedy", list(range(100_000, 120_000)), value=None)
    quota_hit = False
except TreeQuotaExceeded:
    quota_hit = True
cache.create_agent_tree("victim")
vslots = alloc.alloc(10_000)
hit = cache.extend_tree("victim", list(range(200_000, 210_000)), value=vslots)
results["quota"] = {
    "quota_tokens": 50_000,
    "runaway_blocked": quota_hit,
    "victim_tree_unaffected": hit == 0,
}
assert quota_hit and hit == 0

# --- 2. fork-time reservations: admission decided before requests arrive --
cache, alloc = make_cache(quota=100_000)
cache.create_agent_tree("p")
pslots = alloc.alloc(32_000)
cache.extend_tree("p", list(range(32_000)), value=pslots.clone())
admitted, rejected = 0, 0
t0 = time.perf_counter()
for i in range(100):
    br = cache.fork_branch("p", f"c{i}")
    try:
        cache.reserve(br.branch_id, 2_000)   # 2k-token decode budget each
        admitted += 1
    except TreeQuotaExceeded:
        cache.kill_tree(br.branch_id)
        rejected += 1
admit_ms = (time.perf_counter() - t0) * 1e3
# quota 100k - 32k prefix = 68k -> exactly 34 children of 2k admitted
results["reservations"] = {
    "children_requested": 100,
    "admitted": admitted,
    "rejected_at_fork_time": rejected,
    "admission_control_ms_for_100": round(admit_ms, 2),
}
assert admitted == 34 and rejected == 66

# --- 3. priority demotion under real memory pressure ----------------------
cache, alloc = make_cache()
cache.create_agent_tree("p")
pslots = alloc.alloc(32_000)
cache.extend_tree("p", list(range(32_000)), value=pslots.clone())
spec = []
for i in range(10):
    br = cache.fork_branch("p", f"spec{i}")
    sl = alloc.alloc(8_000)
    cache.extend_tree(
        br.branch_id,
        [1_000_000 + i * 8_000 + j for j in range(8_000)],
        value=torch.cat([pslots, sl]),
    )
    spec.append(br.branch_id)
before = cache.evictable_size()
t0 = time.perf_counter()
made_evictable = sum(cache.demote_branch(b) for b in spec)
demote_ms = (time.perf_counter() - t0) * 1e3
# real pressure: evict half of what was demoted
cache.evict(EvictParams(num_tokens=40_000))
survived = [cache.promote_branch(b) for b in spec]
full = sum(1 for s in survived if s == 40_000)
partial = sum(1 for s in survived if 32_000 <= s < 40_000)
results["demotion"] = {
    "speculative_branches": 10,
    "tokens_made_evictable": made_evictable,
    "demote_ms_for_10": round(demote_ms, 2),
    "evicted_under_pressure": 40_000,
    "promoted_fully_cached": full,
    "promoted_prefix_only_reprefill_suffix": partial,
}
assert made_evictable == 80_000 and before == 0
assert full + partial == 10 and partial >= 5

# --- 4. invalidation + fallback re-prefill ---------------------------------
cache, alloc = make_cache()
cache.create_agent_tree("t")
slots = alloc.alloc(16_000)
cache.extend_tree("t", list(range(16_000)), value=slots)
released = cache.invalidate_tree("t")
used_after = POOL_TOKENS - alloc.available_size()
slots2 = alloc.alloc(16_000)
hit = cache.extend_tree("t", list(range(16_000)), value=slots2)
results["invalidation"] = {
    "released_tokens": released,
    "allocator_used_after_invalidate": used_after,
    "fallback_reprefill_hit": hit,
}
assert released == 16_000 and used_after == 0 and hit == 0

# --- 5. per-tree telemetry at 10k-branch scale ------------------------------
cache, alloc = make_cache()
cache.create_agent_tree("p")
pslots = alloc.alloc(32_000)
cache.extend_tree("p", list(range(32_000)), value=pslots.clone())
for i in range(10_000):
    cache.fork_branch("p", f"c{i}")
tel = cache.tree_telemetry("p")
results["telemetry_10k"] = {
    "live_branches": tel.live_branches,
    "charged_tokens": tel.charged_tokens,
    "pinned_tokens": tel.pinned_tokens,
    "saved_tokens": tel.saved_tokens,
}
assert tel.live_branches == 10_001
assert tel.charged_tokens == 32_000
assert tel.saved_tokens == 10_000 * 32_000

print(json.dumps(results, indent=2))
print("TREE-NATIVE FEATURES VALIDATION PASS")
