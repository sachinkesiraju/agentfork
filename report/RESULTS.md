# Measured Results & Gate Table

All numbers below were measured on the stated host unless marked **UNTESTED**.
Nothing here is extrapolated from marketing claims.

**Host:** Ubuntu 22.04, kernel 5.15.200, 2 vCPU, 7 GB RAM, `/dev/kvm` present,
**no GPU**. Firecracker v1.7.0, SGLang @ `40517b593` (main), Python 3.10.

## Gate table

| # | Gate | Target | Measured | Verdict |
|---|------|--------|----------|---------|
| G1 | SGLang patch size | ≤ 1.5 kLOC | **524 LOC** (additive: `tree_radix_cache.py` 279 + unit tests 245; incl. quotas/reservations/demotion/invalidation/telemetry) | PASS |
| G2 | SGLang patch: existing radix tests unaffected | no new failures | 26 pass / 7 fail **before and after** (all 7 pre-existing env failures: missing `-lcrypto` for HiCache cpp hash ext; `torch.cpu.memory_allocated` on CPU-only torch) | PASS |
| G3 | Prefix reuse on 10-way fanout | ≥ 90% | **100%** — every sibling hit the full parent prefix (SGLang `TreeRadixCache` unit test + reference core: 320,063 tokens saved vs 39,937 charged) | PASS |
| G4 | Parent pause during fork | < 100 ms p50 | **~1 ms** (Firecracker PATCH /vm Paused) + 75–82 ms one-time snapshot create | PASS |
| G5 | Per-child marginal sandbox cost | < 150 ms | **2.1–2.3 ms p50 restore** per child; 10-way fanout total 58 ms, 25-way 149 ms (~6 ms/child amortized) | PASS |
| G6 | Kill latency (engine side, kill→KV reclaimed) | < 10 ms | **0.53 ms p50 / 1.46 ms max** over 100 cycles (pidfd SIGKILL 6 µs; process reap 65 µs p50; KV refcount-drop+free 0.46 ms on a 32k-token tree) | PASS |
| G7 | Leaks / orphans under kill cycles | 0 in ≥ 50 cycles | **0 leaked KV tokens, 0 orphaned processes in 100 cycles** (`kill_bench`), 0 leaks in 50-cycle unit test | PASS |
| G8 | Cost vs strong composed baselines | ≥ 20% advantage | Mixed — see below | PARTIAL |
| G9 | Workload shape: shared-prefix fraction f ≥ 0.2 and organic fanout ≥ 8 | f ≥ 0.2, fanout ≥ 8 | **FAIL on observed traces** (see census) | FAIL (with caveat) |
| G10 | GPU KV CoW fork on a real engine | measured on GPU | **Validated on NVIDIA A10 (Modal)**: (a) real 2 GiB fp16 HBM pool — 10-way fork over 32k prefix in 22 ms, **9.65× HBM slot dedup**, kill-all 0.44 ms, allocator back to exactly 0; (b) 7/7 `TreeRadixCache` unit tests pass on GPU; (c) **live SGLang engine** (Qwen3-0.6B): 10 siblings each hit **2402/2404 cached tokens (99.9% prefix reuse)**, sibling completion 33 ms p50 vs 9.07 s parent prefill (`modal_gpu_validation.py`; raw output below) | **PASS** |
| G11 | KV CoW on a **real SGLang allocator/pool** (not mocks) | dedup + full reclaim | **9.65× slot dedup** (37,000 used vs 357,000 unshared; 10-way fanout over 32k prefix on a real 2 GiB fp16 `MHATokenToKVPool` + `TokenToKVPoolAllocator`); allocator returns to **exactly 0** after killing all branches (`patches/real_pool_validation.py`) | PASS |
| G12 | Crash injection: supervisor SIGKILLed, no cleanup | 0 orphans | **0 orphans in 50 cycles × 5 children**; kernel reaps children in 1.5 ms p50 via PDEATHSIG backstop (`agentfork/bench/crash_bench.py`) | PASS |
| G13 | microVM fanout is real page-level CoW | shared pages measured | 25 children: **RSS 117.7 MiB vs PSS 23.8 MiB → 4.95× kernel-measured page sharing**, ~0.95 MiB marginal per 256 MiB child (`smaps_rollup` in `fc_bench`) | PASS |
| G14 | Scale primitive: fork one prefix into **10,000** logical branches without physical copies | N=10,000, 0 copies, exact reclaim | On the real pool/allocator: **10,000 forks in 0.95 s (10.5k forks/s)** with allocator usage unchanged (exactly the 32k parent prefix — zero physical copies); all branches see 100% of the prefix; after per-branch divergence, **1,667× dedup** vs unshared; **bulk kill of 10,001 branches in 0.17 s (59k kills/s)** returns the allocator to exactly 0 (`patches/scale_10k_branch_validation.py`) | PASS |
| G15 | Tree-native control features on the real pool: quotas, reservations, demotion, invalidation, telemetry | each measured | **Quota/fairness**: a runaway tree is blocked at its 50k-token quota while a second tree proceeds unaffected. **Fork-time reservations**: 100 children requesting 2k-token budgets against a 68k remainder → exactly 34 admitted / 66 rejected *at fork time*, 3.3 ms for all 100 admission decisions. **Priority demotion**: 10 speculative branches unpinned in 0.03 ms (80k tokens made evictable); after evicting 40k under real pressure, promote finds 5 fully cached and 5 needing only suffix re-prefill (shared prefix survives). **Invalidation**: releases all 16k pinned tokens, allocator to 0, next extend is a cold re-prefill (hit=0). **Telemetry at 10k branches**: exact per-tree live/charged/pinned/saved counters (saved=320M tokens) (`patches/tree_native_features_validation.py`; raw output below) | PASS |

## Tree-native requirements matrix (10k-branch router shape)

What the current SGLang patch (`TreeRadixCache`, 279 LOC + tests) covers of
the full tree-native requirement list (all YES rows measured — G14/G15):

| Requirement | Status | Notes |
|---|---|---|
| Explicit `tree_id` / `branch_id` / `parent_kv_id` | **YES** | `AgentBranch(branch_id, namespace=tree, parent_id, child_seq)` |
| Immutable shared-prefix KV roots + CoW suffix blocks | **YES** | shared radix nodes pinned via `lock_ref`; suffixes are new nodes; *logical* CoW at page granularity |
| Fork N logical children without N physical copies (key primitive) | **YES — measured at N=10,000** | G14 |
| Bulk child creation and cancellation | **YES** | 10.5k forks/s, 59k kills/s single-threaded CPU; no batched API yet (loop over O(depth) primitives) |
| Cache pinning with budgeted leases | **YES** | `lock_ref` pinning + per-tree token quotas (`tree_quota_tokens`), G15 |
| Fork-time reservations before requests arrive | **YES** | `reserve()` pre-charges suffix budgets against the tree quota; admission decided at fork time (34/100 admitted in 3.3 ms), G15 |
| Priority demotion of speculative branches | **YES** | `demote_branch()` unpins (evictable under pressure, still cached), `promote_branch()` re-pins survivors, G15 |
| ABI invalidation and fallback re-prefill | **YES** | `invalidate_tree()` releases pins + evicts; next extend is a cold re-prefill, G15 |
| KV residency and movement telemetry | **YES** | per-tree `TreeTelemetry`: live branches, charged/pinned/reserved/saved tokens, invalidations, demotions, G15 |
| Per-tree HBM quotas and fairness | **YES** | quota blocks the runaway tree; other trees unaffected, G15 |
| Subtree-aware worker placement | **NO** | single-worker cache; no router changes |
| Hierarchical routing fleet → cell → worker → KV radix node | **NO** | orchestrator work above the engine; the per-tree telemetry and quota hooks are the interface it would consume |

The patch now validates every *engine-local* requirement with measured
results. The two NO rows (placement, fleet routing) are inherently
multi-worker orchestrator features — out of scope for a single-engine patch.

Full VMM process teardown (waitid on the Firecracker process) is 31 ms p50 —
the *branch stops consuming resources* at signal time (µs), teardown is
asynchronous cleanup.

## Cost model (agentfork vs composed baselines)

Prefill-token charges for N-way fanout, shared prefix P, unique suffix U per
child (`agentfork/bench/cost_model.py`):

| Scenario (N, P, U) | vs independent | vs provider cache (0.1× cached reads) | vs self-hosted radix (compute) | HBM residency vs self-hosted |
|---|---|---|---|---|
| 10, 32k, 2k | **6.5×** | 1.7× | **1.0×** | **6.5×** |
| 4, 8k, 2k | 2.5× | 1.3× | 1.0× | 2.5× |
| 25, 32k, 1k | **14.5×** | 2.5× | **1.0×** | **14.5×** |

Honest reading: against the *fairest* baseline (self-hosted SGLang radix cache
with sibling batching) the **compute** win is ~1.0×—the radix cache already
amortizes prefill. The durable wins are (a) **HBM residency** (2.5–14.5× fewer
resident KV tokens under fanout → more concurrent branches per GPU), (b)
**branch latency** (ms-scale sandbox fork instead of seconds-scale re-prefill
+ cold boot), and (c) **integrated kill** (sub-ms vs seconds, with zero
orphan/leak risk tied into one refcount).

## Workload-shape census (de-risk track)

31 accessible Devin sessions of this org (all sessions visible to the
requesting user), mined live via MCP (raw session prompts are private and excluded from this repo).
Run the same analysis on your own session export with
`python -m agentfork.workload.census sessions.json`:

- width histogram of creation-bursts (≤120 s): {1: 12, 2: 1, 5: 1, 12: 1};
  p95 width 5, max 12 — and the width-12 burst is our own validation batch.
- sibling prompt shared-*leading*-prefix fraction f: **0.03–0.15**, below the
  0.2 gate.
- Prior 90-session census (independent session, `fanout_workload_report.md`):
  max organic fanout 6, 69% singletons, flight-search siblings 32.6% leading
  prefix / 98.5% char-identical overall.

**Caveat that keeps this honest in both directions:** prompt-text prefix is a
*lower bound* on KV prefix. In a real agent runtime the shared prefix is
system prompt + tool schemas + repo/context, typically 10–50× larger than the
user-visible prompt, and siblings forked from a live parent share 100% of the
parent's context by construction. The census measures *today's* session-level
workload (which does NOT justify the architecture); a fork-native runtime
would itself generate the tree-shaped workload (induced demand). Ship
decision should therefore hinge on a design partner with an actual fanout
workload, not on organic traces.

## What is NOT validated

- **Live SGLang engine end-to-end on this host:** attempted with Qwen3-0.6B on
  the CPU backend; blocked — SGLang's CPU RoPE path imports `vllm._custom_ops`,
  and vLLM CPU wheels are not installable here. The closest achievable
  validation (real `MHATokenToKVPool` + real allocator, G11) passes.

- **Production-scale GPU behavior:** G10 is now validated on an A10 (real HBM
  pool, unit tests, live engine with 99.9% sibling prefix reuse), but only at
  0.6B-model scale with a single GPU; allocator pressure at 70B scale, TP,
  and scheduler interaction under contention remain unmeasured.
- microVM + GPU colocation (vfio passthrough or API-proxied inference) —
  unmeasured.
- Provider-cache baseline uses public pricing ratios (0.1× cached reads,
  1.25× write), not measured bills.
- Firecracker numbers are CPU-only 256 MB guests; guests with real agent
  workloads inside will restore slower (page-cache cold misses).

## Raw benchmark outputs

These are the captured outputs from the rerunnable validation scripts.

### GPU validation

```json
{
  "gpu": "NVIDIA A10",
  "provider": "Modal",
  "sglang": "main@40517b593 + tree_radix_cache patch",
  "real_hbm_pool": {
    "hbm_pool_gib": 2.0,
    "parent_used": 32000,
    "fork_10_ms": 22.26,
    "after_forks_used": 37000,
    "dedup_x": 9.65,
    "kill_all_ms": 0.44,
    "after_kill_used": 0
  },
  "unit_tests_on_gpu": "7 passed",
  "live_engine": {
    "model": "Qwen/Qwen3-0.6B",
    "prompt_tokens": 2404,
    "parent_prefill_s": 9.07,
    "sibling_cached_tokens_min": 2402,
    "sibling_cached_tokens_max": 2403,
    "siblings": 10,
    "sibling_gen_s_p50": 0.033
  }
}
```

### Tree-native features

```json
{
  "quota": {"quota_tokens": 50000, "runaway_blocked": true, "victim_tree_unaffected": true},
  "reservations": {"children_requested": 100, "admitted": 34, "rejected_at_fork_time": 66, "admission_control_ms_for_100": 3.27},
  "demotion": {"speculative_branches": 10, "tokens_made_evictable": 80000, "demote_ms_for_10": 0.03, "evicted_under_pressure": 40000, "promoted_fully_cached": 5, "promoted_prefix_only_reprefill_suffix": 5},
  "invalidation": {"released_tokens": 16000, "allocator_used_after_invalidate": 0, "fallback_reprefill_hit": 0},
  "telemetry_10k": {"live_branches": 10001, "charged_tokens": 32000, "pinned_tokens": 32000, "saved_tokens": 320000000}
}
```
