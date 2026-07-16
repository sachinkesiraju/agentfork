"""Scale test: fork one 32k-token prefix into 10,000 logical branches
without physical KV copies, on the real SGLang pool/allocator (CPU device).

Verifies the key scale primitive from the tree-native requirements list:
- N=10,000 fork_branch() calls from one parent;
- allocator slot usage stays at prefix + sum(suffixes), never N x prefix;
- bulk cancellation (kill_tree x 10,000) returns the allocator to exactly 0;
- fork and kill throughput are reported.
"""

import time

import torch

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.tree_radix_cache import TreeRadixCache

PREFIX = 32_000
CHILDREN = 10_000
SUFFIX = 16
POOL_TOKENS = PREFIX + CHILDREN * SUFFIX + 1024

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
cache = TreeRadixCache(params)

def used() -> int:
    return POOL_TOKENS - alloc.available_size()

cache.create_agent_tree("parent")
parent_tokens = list(range(PREFIX))
parent_slots = alloc.alloc(PREFIX)
cache.extend_tree("parent", parent_tokens, value=parent_slots.clone())
print(f"parent prefilled: allocator_used={used()}")

t0 = time.perf_counter()
branches = [cache.fork_branch("parent", f"c{i}") for i in range(CHILDREN)]
fork_s = time.perf_counter() - t0
assert used() == PREFIX, f"fork must not allocate: used={used()}"
print(f"forked {CHILDREN} branches in {fork_s:.2f}s "
      f"({CHILDREN / fork_s:,.0f} forks/s); allocator_used={used()} "
      f"(still exactly the parent prefix -> zero physical copies)")

# spot-check prefix visibility across the fanout
for i in (0, CHILDREN // 2, CHILDREN - 1):
    assert cache.match_tree_prefix(f"c{i}", parent_tokens) == PREFIX
print("prefix visibility: 100% for spot-checked branches (0, mid, last)")

# every branch diverges with its own small suffix
t0 = time.perf_counter()
for i, br in enumerate(branches):
    suffix = [10_000_000 + i * SUFFIX + j for j in range(SUFFIX)]
    slots = alloc.alloc(SUFFIX)
    full = torch.cat([parent_slots, slots])
    hit = cache.extend_tree(br.branch_id, suffix, value=full)
    assert hit >= PREFIX
extend_s = time.perf_counter() - t0
expect = PREFIX + CHILDREN * SUFFIX
unshared = CHILDREN * (PREFIX + SUFFIX)
print(f"extended {CHILDREN} branches in {extend_s:.2f}s; "
      f"allocator_used={used()} (expect {expect}); "
      f"dedup {unshared / used():.1f}x vs unshared")
assert used() == expect

t0 = time.perf_counter()
for br in branches:
    cache.kill_tree(br.branch_id)
cache.kill_tree("parent")
kill_s = time.perf_counter() - t0
print(f"bulk-killed {CHILDREN + 1} branches in {kill_s:.2f}s "
      f"({(CHILDREN + 1) / kill_s:,.0f} kills/s); allocator_used={used()} "
      f"(expect 0); live_branches={cache.live_branches()}")
assert used() == 0 and cache.live_branches() == 0
print("10K-BRANCH SCALE VALIDATION PASS")
