# Measured Results and Validation Status

This report combines captured benchmark outputs, test results, and an explicit
token-arithmetic cost model. Hardware-dependent CPU numbers were recorded on
Ubuntu 22.04, kernel 5.15.200, 2 vCPU, 7 GB RAM, `/dev/kvm` present, with
Firecracker v1.7.0, SGLang @ `40517b593`, and Python 3.10. GPU results were
recorded separately on a Modal NVIDIA A10. Results whose private inputs or full
environments are not checked in are labeled below.

## Validation summary

| Check | Target | Measured | Verdict |
|---|---|---|---|
| SGLang patch size | ≤ 1.5 kLOC | **547 LOC** (additive: `tree_radix_cache.py` 299 + unit tests 248; incl. quotas/reservations/demotion/invalidation/telemetry) | PASS |
| Existing SGLang radix tests unaffected | no new failures | 26 pass / 7 fail **before and after** (all 7 pre-existing env failures: missing `-lcrypto` for HiCache cpp hash ext; `torch.cpu.memory_allocated` on CPU-only torch) | PASS |
| Prefix reuse on 10-way fanout | ≥ 90% | **100%** — every sibling hits the full parent prefix in the SGLang `TreeRadixCache` unit test and CPU reference test. The previously quoted 320,063/39,937 run is omitted because its raw output is not checked in. | PASS |
| Parent pause during fork | < 100 ms p50 | **~76–83 ms total recorded pause window**: ~1 ms pause API call followed by 75–82 ms full snapshot creation while the parent remains paused | PASS |
| Per-child marginal sandbox cost | < 150 ms | **2.1–2.3 ms p50 snapshot-load API time** per child; 10-way fanout total 58 ms, 25-way 149 ms (~6 ms/child amortized). Guest application readiness was not probed. | PASS |
| Subprocess + CPU reference-cache reclaim | < 10 ms | **0.53 ms p50 / 1.46 ms max** over 100 cycles (pidfd SIGKILL 6 µs; process reap 65 µs p50; Python refcount-drop+free 0.46 ms on a synthetic 32k-token tree) | PASS |
| Reference-cache leaks / subprocess handles under kill cycles | 0 in ≥ 50 cycles | **0 resident reference tokens and 0 registered subprocesses after 100 waited kill cycles** (`kill_bench`); 0 reference-cache tokens after the 50-cycle unit test | PASS |
| Cost vs strong self-hosted prefix-cache baseline | ≥ 20% compute or residency advantage | Corrected model: **1.0× compute and 1.0× cache residency** versus stock same-namespace prefix sharing; lifecycle value is not monetized | FAIL |
| Workload shape: shared-prefix fraction f ≥ 0.2 and organic fanout ≥ 8 | f ≥ 0.2, fanout ≥ 8 | **FAIL on observed traces** (see census) | FAIL (with caveat) |
| Patched cache against a real GPU pool | measured on GPU | **Validated on NVIDIA A10 (Modal)**: a real 2 GiB fp16 pool completed 10 create+extend operations over a 32k prefix in 22 ms, used 37k slots rather than the explicitly unshared 357k allocation, and returned the allocator to 0 after kill-all. The captured GPU run also passed the then-current 7-test patch suite. | **PASS for direct cache API; engine integration UNTESTED** |
| KV sharing on a **real SGLang allocator/pool** (not mocks) | sharing + full reclaim | **9.65× versus an explicitly unshared allocation** (37,000 used vs 357,000). This is not a gain over stock SGLang RadixAttention, which already shares identical cached prefixes. The allocator returns to **exactly 0** after killing all branches (`patches/real_pool_validation.py`). | PASS |
| Crash injection: supervisor SIGKILLed, no cleanup | 0 surviving subprocesses | **0 surviving Python children in 50 cycles × 5 children**; children disappear in 1.5 ms p50 through the `PR_SET_PDEATHSIG` backstop (`agentfork/bench/crash_bench.py`). This does not test external GPU cleanup. | PASS for tested subprocess path |
| microVM snapshot loads share host pages | shared pages measured | 25 idle VMM processes: **RSS 117.7 MiB vs PSS 23.8 MiB → 4.95× RSS/PSS ratio**, ~0.95 MiB PSS per 256 MiB configured guest (`smaps_rollup` in `fc_bench`). This does not measure a resident 256 MiB working set per child. | PASS for tested idle guests |
| Fork one prefix into **10,000** logical branches without physical copies | N=10,000, 0 copies, exact reclaim | On a real SGLang pool/allocator backed by small CPU tensors: **10,000 forks in 0.95 s (10.5k forks/s)** with allocator usage unchanged; after per-branch divergence, **1,667× vs unshared**; **bulk kill of 10,001 branches in 0.17 s (59k kills/s)** returns the allocator to 0 (`patches/scale_10k_branch_validation.py`). This is metadata scale, not concurrent inference scale. | PASS |
| Tree-native cache controls: quotas, reservations, demotion, invalidation, telemetry | each measured | Cache-level accounting produced the recorded quota and 34/66 reservation decisions; demotion/promotion, invalidation, and counters behaved as asserted against a CPU-backed SGLang pool (`patches/tree_native_features_validation.py`). Quotas/reservations are logical hooks and do not control allocator calls until a scheduler is wired to them. | PASS for direct cache API |

## Tree-native requirements matrix (10k-branch router shape)

What the current SGLang patch (`TreeRadixCache`, 299 implementation lines plus
tests) covers of the tree-native requirement list. Status refers to the direct
cache API unless the notes say otherwise:

| Requirement | Status | Notes |
|---|---|---|
| Explicit tree namespace / `branch_id` / `parent_id` metadata | **YES** | `AgentBranch(branch_id, namespace, parent_id, child_seq)`; there is no separate `parent_kv_id` field |
| Shared-prefix radix nodes + divergent suffix nodes | **YES** | existing SGLang KV slots are pinned via `lock_ref`; “CoW” is logical reference sharing, not copied GPU state |
| Fork N logical children without N physical copies (key primitive) | **YES — measured at N=10,000** | See the 10,000-branch validation above |
| Bulk child creation and cancellation | **YES** | 10.5k forks/s, 59k kills/s single-threaded CPU; no batched API yet (loop over O(depth) primitives) |
| Cache pinning with logical token budgets | **DIRECT API ONLY** | `lock_ref` pinning + per-tree accounting (`tree_quota_tokens`); no scheduler enforcement |
| Fork-time token-budget reservations | **DIRECT API ONLY** | `reserve()` pre-charges logical suffix budgets; it does not reserve physical allocator slots |
| Priority demotion of speculative branches | **YES** | `demote_branch()` unpins pages and `promote_branch()` re-pins survivors |
| ABI invalidation and fallback re-prefill | **YES** | `invalidate_tree()` releases pins and evicts; the next extend is a cold re-prefill |
| Cache accounting telemetry | **YES** | per-tree live/charged/pinned/reserved/saved counters plus invalidations/demotions; no cross-worker movement is tracked |
| Per-tree HBM enforcement | **NO** | the patch accounts tokens but is not wired ahead of allocator calls in the scheduler |
| Subtree-aware worker placement | **NO** | single-worker cache; no router changes |
| Hierarchical routing fleet → cell → worker → KV radix node | **NO** | orchestrator work above the engine; the per-tree telemetry and quota hooks are the interface it would consume |

The patch validates its direct cache API. Scheduler-enforced allocation,
subtree-aware placement, and fleet routing remain outside the patch, and no
end-to-end request path currently supplies branch identity to `TreeRadixCache`.

Full VMM process teardown (`waitid` on the Firecracker process) was 31 ms p50.
Signal delivery stops execution first; process resources are not fully released
until teardown completes.

## Cost model (agentfork vs composed baselines)

Prefill-token charges for N-way fanout, shared prefix P, unique suffix U per
child (`agentfork/bench/cost_model.py`):

| Scenario (N, P, U) | vs independent | vs provider cache (0.1× cached reads) | vs self-hosted prefix cache (compute) | Cache residency vs self-hosted |
|---|---|---|---|---|
| 10, 32k, 2k | **6.5×** | 1.7× | **1.0×** | **1.0×** |
| 4, 8k, 2k | 2.5× | 1.3× | 1.0× | 1.0× |
| 25, 32k, 1k | **14.5×** | 2.5× | **1.0×** | **1.0×** |

The original model incorrectly assigned N physical copies of the shared prefix
to stock SGLang. RadixAttention already stores and reuses an identical cached
prefix once, so both compute and physical cache residency are approximately
1.0× for this idealized same-namespace comparison. The patch's proposed value
against that baseline is explicit ownership, pinning, branch policy, telemetry,
and reclaim. Those lifecycle benefits have not been converted into a measured
end-to-end cost advantage.

## Workload-shape census (de-risk track)

This recorded census used 31 private Devin sessions visible to the requesting
user. The prompts and export are not checked in, so the figures below cannot be
independently reproduced from this repository. Run the same analysis on your own session export with
`python -m agentfork.workload.census sessions.json`:

- width histogram of creation-bursts (≤120 s): {1: 12, 2: 1, 5: 1, 12: 1};
  p95 width 5, max 12 — and the width-12 burst is our own validation batch.
- sibling prompt shared-*leading*-prefix fraction f: **0.03–0.15**, below the
  0.2 evaluation threshold.
- Prior 90-session census (independent session, `fanout_workload_report.md`):
  max organic fanout 6, 69% singletons, flight-search siblings 32.6% leading
  prefix / 98.5% char-identical overall.

Prompt-text prefix can be a lower bound on KV prefix because system prompts,
tool schemas, and retrieved context may precede the visible prompt. This report
did not measure that hidden-prefix ratio. Siblings forked from a live parent
would share the parent's context by construction, but the observed session-level
workload does not establish demand for that architecture. A deployment decision
needs an end-to-end trace from an actual fanout workload.

## What is NOT validated

- **Patched live-engine path:** no scheduler/model-runner/server integration
  selects `TreeRadixCache` or propagates branch IDs. The Qwen3-0.6B live-engine
  result below is a stock RadixAttention baseline, not an end-to-end patch test.
  A CPU-engine attempt was also blocked by unavailable vLLM CPU wheels.
- **Unified runtime:** no API atomically coordinates Firecracker restore, patched
  cache fork, inference submission, and rollback. The reaper's two cleanup steps
  are sequential.
- **Production-scale GPU behavior:** the direct cache API was tested on one A10
  with a synthetic 2 GiB pool. 70B-class models, tensor/pipeline parallelism,
  mixed workloads, and scheduler contention remain unmeasured.
- **Sandbox integration:** microVM + GPU colocation or API-proxied inference,
  guest networking/identity, and real agent readiness are unmeasured. The
  Firecracker runs used idle CPU-only 256 MiB guests.
- **Admission enforcement:** token quotas and reservations are direct cache API
  accounting, not scheduler-enforced physical HBM reservations.
- **Provider economics:** ratios use assumed cached-read/write prices rather
  than measured bills, latency, or total infrastructure cost.
- **Crash hardening:** `PR_SET_PDEATHSIG` passed the recorded injection test and
  the child now rechecks its parent PID after setup, but setting it still
  requires `preexec_fn`, which is unsafe in threaded supervisors; the backstop
  is opt-out (`BranchReaper(pdeathsig=False)`) rather than fixed.

## Raw benchmark outputs

These are captured outputs from validation scripts. Reruns may vary with
hardware and dependencies; the Modal base image is not digest-pinned.

### GPU validation

The `real_hbm_pool` object exercises the patch directly. The `live_engine`
object is the stock RadixAttention baseline and does not invoke branch APIs.
The captured run predates expansion of the patch test file from 7 to 17 tests.

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
