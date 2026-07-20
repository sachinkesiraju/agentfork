# When KV pinning pays off: the cache-pressure break-even

`report/RESULTS.md` records that agentfork's tree-pinned KV cache **does not**
beat stock SGLang RadixAttention when the cache is uncontended (measured
**1.017×** VGE, FAIL) — stock already stores and reuses an identical shared
prefix once. The advantage only appears under **cache pressure**, where
unrelated traffic evicts the shared prefix from stock's LRU but not from
agentfork's pinned tree. The sustained-pressure run passed at **1.596×** and a
locked holdout at **1.537×**; the single-burst run scored **1.186×** (FAIL on
the ≥1.2× CI lower bound).

This document maps *exactly where* that turn-on happens. It adds a
cache-pressure dimension to the cost model
(`agentfork/bench/cost_model.py`), derives the break-even surface in closed
form, and validates every prediction against two real cache objects with a
deterministic simulator (`agentfork/bench/pressure_bench.py`).

## Setup and parameters

| Symbol | Meaning |
|---|---|
| `P` | shared parent-prefix length (tokens) |
| `N` | fanout: number of sibling branches |
| `S` | unique suffix each child adds after the shared prefix (tokens) |
| `U` | unrelated prefill tokens interleaved before a child (pressure) |
| `C` | KV cache token capacity |

Workload (matches the GPU harness in `modal_gpu_validation.py`): prefill the
parent once, then run `N` children, each `P + S` tokens (shared prefix + unique
suffix). Two pressure patterns:

- **sustained** — `U` unrelated tokens injected before *every* child;
- **burst** — `U` unrelated tokens injected once, before the first child only.

Cost is proxied by **prefill tokens charged** (a compute proxy, not wall-clock;
see caveats). "Compute ratio" = stock prefill charged ÷ pinned prefill charged
over the parent+children path (the shared unrelated traffic is excluded, since
both arms pay it identically).

## The model

Two caching disciplines are modeled over the same radix structure:

- **pinned (agentfork)** — the parent prefix is `lock_ref`-pinned to the live
  tree, so it is never evicted while the tree is alive. Every child hits;
  each child charges only its `S`-token suffix. Total prefill = `P + N·S`.
- **stock (RadixAttention/APC)** — no cross-request pin; the shared prefix is
  ordinary cache content subject to LRU eviction. A child that finds the
  prefix evicted re-prefills the whole `P + S`.

### Break-even derivation (leaf-LRU)

Radix caches evict **leaves** LRU-first. The key observation: once a child has
extended the parent, the parent node has a child-suffix node hanging off it, so
the parent is an **internal** node and cannot be evicted until that suffix leaf
is evicted first. During any gap, the eviction order is therefore:

> older unrelated tokens → older child suffixes → the current suffix → *only
> then* the parent.

Everything older than the parent's most recent touch is a leaf and gets
sacrificed before the parent. So the parent is only forced out when the parent
itself plus the current gap's `U` unrelated tokens cannot coexist in `C`:

> **the parent survives a gap ⟺ P + U ≤ C ⟺ U ≤ C − P.**

The suffix `S` drops out entirely — it is a leaf and is evicted before the
parent, so it never counts against the parent's survival. The break-even is
exactly

> **U\* = C − P**  (the cache headroom above the pinned prefix).

Equivalently, normalizing by capacity, **pinning turns on when `U/C > 1 − P/C`.**

Consequences:

- **sustained** pressure is all-or-nothing: `U ≤ C−P` keeps the parent for all
  `N` children (stock hit rate 1.0, no advantage); `U > C−P` evicts it for all
  `N` (stock hit rate 0.0, `N` re-prefills).
- **burst** pressure can only cost the *first* child: its re-prefill re-pins
  the parent for the remaining `N−1`, so at most one miss (hit rate `(N−1)/N`).
- `N` does **not** move the on/off boundary, but it scales the magnitude:
  every miss re-prefills `P`, so under full sustained pressure the compute
  ratio is `1 + N·P/(P + N·S)`.

### Break-even surface (`U*/C` vs `P/C`, C = 32768)

| `P` | `P/C` | `U* = C−P` | `U*/C` | pinning wins when |
|---:|---:|---:|---:|---|
| 800 | 0.02 | 31,968 | 0.98 | `U/C > 0.98` |
| 2,400 | 0.07 | 30,368 | 0.93 | `U/C > 0.93` |
| 4,096 | 0.12 | 28,672 | 0.88 | `U/C > 0.88` |
| 8,192 | 0.25 | 24,576 | 0.75 | `U/C > 0.75` |
| 16,384 | 0.50 | 16,384 | 0.50 | `U/C > 0.50` |

The bigger the pinned prefix relative to capacity, the *less* per-gap traffic
it takes to evict it — the boundary is a straight line `U*/C = 1 − P/C`.

### Compute ratio scales with fanout (full sustained pressure)

`P = 2400`, `S = 8`, `C = 32768`, `U = 96000` (i.e. `U > U*`, every child misses):

| `N` | compute ratio (prefill) | stock charged | pinned charged |
|---:|---:|---:|---:|
| 1 | 2.00× | 4,808 | 2,408 |
| 2 | 2.99× | 7,216 | 2,416 |
| 4 | 4.95× | 12,032 | 2,432 |
| 10 | 10.68× | 26,480 | 2,480 |
| 25 | 24.08× | 62,600 | 2,600 |
| 50 | 43.86× | 122,800 | 2,800 |
| 100 | 76.00× | 243,200 | 3,200 |

## Empirical validation against the reference caches

`agentfork/bench/pressure_bench.py` replays the workload against two **real**
cache objects and measures hit rates and prefill tokens charged:

- **pinned arm → `agentfork/kv/tree_cache.py::TreeKVCache`** (the CPU reference
  with real capacity, eviction, and pinning). The parent tree stays alive so
  its prefix is pinned; unrelated traffic is a separate tree freed on
  completion (`kill`), exactly as a real engine releases a finished unrelated
  request's exclusive KV while the pinned parent is protected.
- **stock arm → `StockRadixCache`** — a compact single-namespace radix cache
  with deterministic leaf-LRU eviction and no cross-request pinning (stock
  RadixAttention / vLLM APC behavior).

### Sweep (P = 2400, N = 10, S = 8, C = 32768), U in {0, 24k, 48k, 96k}

`U* = C − P = 30,368`.

**Sustained:**

| `U` | `U/C` | model hit | measured hit | model ratio | measured ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.00 | 1.00 | 1.00 | 1.000× | 1.000× |
| 24,000 | 0.73 | 1.00 | 1.00 | 1.000× | 1.000× |
| 48,000 | 1.46 | 0.00 | 0.00 | 10.677× | 10.677× |
| 96,000 | 2.93 | 0.00 | 0.00 | 10.677× | 10.677× |

**Burst:**

| `U` | `U/C` | model hit | measured hit | model ratio | measured ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.00 | 1.00 | 1.00 | 1.000× | 1.000× |
| 24,000 | 0.73 | 1.00 | 1.00 | 1.000× | 1.000× |
| 48,000 | 1.46 | 0.90 | 0.90 | 1.968× | 1.968× |
| 96,000 | 2.93 | 0.90 | 0.90 | 1.968× | 1.968× |

Model and reference caches agree **to the token** on hit rate and compute ratio
across six workload shapes (`tests/test_pressure_bench.py`), and the break-even
is exact against the real caches: at `U = U*` the parent survives (stock hit
1.0), at `U = U* + 1` it is evicted (stock hit 0.0).

Reproduce:

```bash
python -m agentfork.bench.pressure_bench \
  --prefix 2400 --children 10 --suffix 8 --capacity 32768 \
  --interleaved 0 24000 48000 96000
python -m agentfork.bench.cost_model --pressure \
  --prefix 2400 --children 10 --suffix 8 --capacity 32768 --interleaved 96000
```

## Reconciling with the GPU VGE numbers

The compute ratio here is a **prefill-token** proxy; the GPU VGE in
`report/RESULTS.md` is **wall-clock sibling generation time**, which includes
decode. Decode cost is unaffected by whether the prefix was cached, so it
dilutes the prefill advantage: the sustained run's ~10× prefill-token ratio
shows up as **1.596×** wall-clock VGE, and the single-burst run (one re-prefill
out of ten children, model 1.97× prefill / hit 0.90) shows up as **1.186×**
VGE. The model predicts the *direction, ordering, and hit-rate structure* of
those runs (sustained ≫ burst ≫ uncontended = 1.0), and matches the reference
cache's hit rates and prefill charges exactly. It does not predict absolute
wall-clock VGE, which depends on the prefill/decode time split of the specific
model and GPU.

## GPU validation status

Skipped: no Modal credentials are available in this environment (`modal` is not
installed, there is no `~/.modal.toml`, and no `MODAL_*` environment variables
are set). No GPU numbers are fabricated. To run the live sweep on a patched
A10G engine, apply patches `0001`–`0003` to an SGLang checkout and run
`SGLANG_DIR=/path/to/sglang python3 -m modal run modal_gpu_validation.py`
(see `modal_gpu_validation.py`, whose `apply_cache_pressure`/`ENGINE_CHILDREN`
knobs implement the sustained pattern at `P≈2.4k, N∈{10,12}`). Record the
measured VGE at each `U ∈ {0, 24k, 48k, 96k}` beside the model's prediction
here when Modal is reachable.

## Assumptions and limits (honest)

- **Prefill-token proxy, not wall-clock.** Compute ratio counts prefill tokens;
  it deliberately excludes decode, batching, attention-kernel scaling, and
  memory bandwidth. Absolute VGE is smaller (see above).
- **Uniform per-gap pressure, single parent.** The model assumes one shared
  prefix and either uniform per-child pressure (sustained) or a single leading
  burst. Multi-parent contention, variable gaps, and partial-prefix reuse are
  not modeled.
- **Leaf-LRU, single worker.** The eviction argument assumes leaf-LRU eviction
  (SGLang/vLLM behavior) on one worker/pool. Segmented, priority, or
  cross-worker eviction policies would shift `U*`.
- **Tokens stand in for KV tensors.** Page size 1, no fragmentation; a real
  allocator's page granularity rounds `U*` to the nearest page.
- **Demand still unproven.** `agentfork/workload/census.py` finds observed
  traces fail the required shape (`f ≥ 0.2`, fanout ≥ 8). This document says
  *when* pinning would pay off; it does not establish that production workloads
  reach that regime.

## Bottom line

Tree KV pinning pays off precisely when **interleaved unrelated traffic per gap
exceeds the cache headroom above the pinned prefix, `U > C − P`** (equivalently
`U/C > 1 − P/C`). Below that line stock RadixAttention already retains the
shared prefix and agentfork's advantage is ~1.0×. Above it, stock re-prefills
the prefix — once per miss — while pinning holds it, for a prefill-token
compute ratio of `1 + m·P/(P + N·S)` where `m` is the number of missing
children (`m = N` under sustained pressure, `m = 1` under a single burst).
