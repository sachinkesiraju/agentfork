"""Validate TreeRadixCache against a REAL SGLang KV pool + allocator (CPU).

Uses the actual MHATokenToKVPool (real fp16 KV tensors, Llama-1B-like shape)
and TokenToKVPoolAllocator instead of mocks: allocates real KV slots for the
parent prefill, forks 10 children (allocating real slots only for each
child's unique suffix), and measures true allocator occupancy to prove CoW
dedup and full reclamation at the allocator level.
"""

import time

import torch

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.tree_radix_cache import TreeRadixCache

POOL_TOKENS = 65536
PREFIX, SUFFIX, CHILDREN = 32000, 500, 10

kvcache = MHATokenToKVPool(
    size=POOL_TOKENS, page_size=1, dtype=torch.float16,
    head_num=8, head_dim=64, layer_num=16, device="cpu",
    enable_memory_saver=False,
)
alloc = TokenToKVPoolAllocator(
    size=POOL_TOKENS, dtype=torch.float16, device="cpu",
    kvcache=kvcache, need_sort=False,
)
kv_bytes_per_token = 2 * 16 * 8 * 64 * 2  # k+v, layers, heads, dim, fp16
print(f"real KV pool: {POOL_TOKENS} tokens x {kv_bytes_per_token/1024:.0f} KiB/token "
      f"= {POOL_TOKENS*kv_bytes_per_token/2**30:.2f} GiB backing tensors")

params = CacheInitParams(
    disable=False, req_to_token_pool=None,
    token_to_kv_pool_allocator=alloc, page_size=1,
)
cache = TreeRadixCache(params)


def used() -> int:
    return POOL_TOKENS - alloc.available_size()


# parent: allocate REAL kv slots for the prefix and register them
cache.create_agent_tree("parent")
parent_tokens = list(range(PREFIX))
parent_slots = alloc.alloc(PREFIX)
assert parent_slots is not None
hit = cache.extend_tree("parent", parent_tokens, value=parent_slots.clone())
print(f"parent prefill: hit={hit}, allocator_used={used()}")
assert used() == PREFIX

# fork children: real slots are allocated ONLY for each child's unique suffix
t0 = time.perf_counter()
for i in range(CHILDREN):
    br = cache.fork_branch("parent", f"c{i}")
    assert cache.match_tree_prefix(br.branch_id, parent_tokens) == PREFIX
    child_suffix = [10_000_000 + i * SUFFIX + j for j in range(SUFFIX)]
    child_slots = alloc.alloc(SUFFIX)
    assert child_slots is not None
    full_value = torch.cat([parent_slots, child_slots])
    hit = cache.extend_tree(br.branch_id, child_suffix, value=full_value)
    assert hit == PREFIX, hit  # child re-used the parent's real KV slots
fork_s = time.perf_counter() - t0
print(f"forked {CHILDREN} children in {fork_s*1000:.1f} ms; allocator_used={used()}")

no_sharing = (PREFIX + SUFFIX) * CHILDREN + PREFIX
expected = PREFIX + CHILDREN * SUFFIX
assert used() == expected, (used(), expected)
print(f"real allocator dedup: {used()} slots used vs {no_sharing} without sharing "
      f"({no_sharing/used():.2f}x; {(no_sharing-used())*kv_bytes_per_token/2**30:.2f} GiB "
      f"saved at this 1B-model shape, scales linearly with model size)")

# kill all children then parent: allocator must return to exactly zero
t0 = time.perf_counter()
for i in range(CHILDREN):
    cache.kill_tree(f"c{i}")
cache.kill_tree("parent")
kill_s = time.perf_counter() - t0
print(f"killed all branches in {kill_s*1000:.1f} ms; allocator_used={used()} (expect 0)")
assert used() == 0, used()
print("REAL-POOL VALIDATION PASS")
