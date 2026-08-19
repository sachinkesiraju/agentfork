# agentfork.kvzip_router v1

A router-programmable KV-cache compression primitive for long-context LLM serving.

The core idea is to treat KV-cache memory as a schedulable, quality-guaranteed
resource that the control plane can trade against output quality, the same way a
router already trades model cost against quality.  v1 implements a minimal,
fully runnable reference stack in pure PyTorch on top of Hugging Face
`transformers`, and includes an `agentfork` `KVBackend` adapter that proves the
serving-system integration end to end.

## What v1 proves

1. A learned **sink-attention gate** predicts per-token, per-head KV importance.
2. A **control-plane budget** (`KVBudgetHint`) maps task / adapter / quality /
   latency signals to a concrete `target_ratio`.
3. A **budget allocator** turns that ratio into per-layer / per-head water-fill
   budgets and a pruned `DynamicCache`.
4. The **compressed cache** can be decoded against with correct RoPE positions,
   preserving next-token quality.
5. **Online distillation** from a full-cache attention teacher trains the gate.
6. **Trace-driven online distillation** (`TraceJournal` + `TraceTrainer`) closes
   the flywheel from production requests to gate updates.
7. **Prune + quantize** stack multiplicatively: 4-bit retained KV is smaller
   than full-precision retained KV.
8. The **agentfork backend** (`KVZipBackend`) lets `ForkOrchestrator` fork,
   extend, and generate from a compressed, query-agnostic prefix.
9. A **vLLM V1 CPU attention backend** (`agentfork.kvzip_router.vllm_backend`) performs
   block-granular query-aware KV pruning inside `LLM.generate` with the custom
   `AttentionBackendEnum.CUSTOM` backend.
10. A **real PEFT LoRA adapter** changes the gate's per-token scores and
    per-head budgets (`tests/kvzip/test_lora_conditioning.py`).
11. A **downstream cloze-style QA benchmark** on `distilgpt2` shows that
    compressing the context still leaves the answer token in the top-5 and
    keeps its probability within 10% of the full-cache value
    (`tests/kvzip/test_downstream_qa.py`).

## Components

| Module | File | Purpose |
|--------|------|---------|
| `KVCacheGate` | `gate.py` | Per-token, per-head importance predictor. Optional task/LoRA conditioning. |
| `BudgetAllocator` | `budget.py` | Water-fill allocation of a target token budget across layers/heads. |
| `KVCompressor` | `compressor.py` | Prefill, score, allocate, prune and return a `DynamicCache` for HF models. |
| `TokenRelevanceRerank` | `rerank.py` | Query-aware rerank over retained token hidden states. |
| `KVBudgetRouter` / `KVBudgetHint` | `router.py` | Control-plane API that maps task/adapter/quality/latency to a `target_ratio`. |
| `FullCacheTeacher` / `OnlineDistiller` | `distill.py` | Offline teacher labels from attention weights; MSE distillation of the gate. |
| KV quantization helpers | `quantize.py` | Symmetric 8/4-bit round-trip for memory/savings tests. |
| `uniform_compress` / `full_cache_reference` | `baselines.py` | Naive uniform-drop and uncompressed baselines for value tests. |
| `KVZipBackend` | `agentfork.py` | `agentfork` `KVBackend` implementation; plugs into `ForkOrchestrator`. |
| `TraceJournal` / `TraceTrainer` | `trace.py` | Production-trace buffer and online distillation loop. |
| `FastKVzipCPUAttentionBackend` | `vllm_backend.py` | vLLM V1 custom attention backend with block-level KV pruning. |

## v1 spec

### Control-plane hint

```python
KVBudgetHint(
    task_id="summarize",
    adapter_id="legal",
    quality_floor="standard",   # strict | standard | aggressive
    latency_class="balanced",   # latency | balanced | throughput
    target_ratio=0.4,           # fraction of KV cache to retain
    max_drop=0.03,
)
```

`KVBudgetRouter` maps `quality_floor` and `latency_class` to a concrete
`target_ratio`, with per-task and per-adapter overrides.

### Gate

* Input: hidden state at layer `l`, shape `[batch, seq_len, hidden_dim]`.
* Optional condition vector derived from `(task_id, adapter_id)`.
* Output: `sigmoid` scores, shape `[batch, seq_len, n_kv_heads]`.

The gate is trained by `OnlineDistiller` to match a `FullCacheTeacher`, which
scores each token by summing normalized attention weights over all query
positions.  `TraceJournal` records production `(input_ids, budget)` requests and
`TraceTrainer` turns them into gradient steps, completing the live retraining
flywheel.

### Budget allocation

`BudgetAllocator` computes a water-fill threshold over all layer/head/token
scores so the total retained count matches the budget.  It returns:

* `per_head`: per-layer/per-head target counts.
* `global_indices`: a single set of retained token indices used for v1's
  uniform-sequence-length `DynamicCache`.

A paged-attention backend (v2) will consume `per_head` directly and keep
variable-length per-layer caches; v1 uses a global index set for HF
`past_key_values` compatibility.

### Prefill + decode

1. Run `model(..., output_hidden_states=True, use_cache=True)` on the prefix.
2. Score every token with the gate.
3. Allocate budgets and gather the retained KV blocks.
4. Build a new `DynamicCache` from the gathered blocks.
5. Decode new tokens with `model(input_ids, past_key_values=compressed_cache,
   position_ids=[[prefix_len]])`.

The correct `position_ids` preserve RoPE absolute positions even when the cache
is shorter than the original prompt.

### Quantization

`quantize_tensor` / `dequantize_tensor` provide symmetric 8-bit (or 4-bit
integer-stored) round-trips.  v1 keeps dequantized float caches for generation;
a real backend would fuse 4-bit dequantization into the attention kernel.

### Query-aware rerank

`TokenRelevanceRerank` scores retained hidden states against the query hidden
state.  v1 uses this as an independent module; a backend integration would turn
the selected indices into an attention mask over the compressed cache.

### agentfork integration

`KVZipBackend` (`kvzip_backend.py`) implements the `KVBackend` protocol expected by
`agentfork.orchestrator.ForkOrchestrator`:

```python
from agentfork import ForkOrchestrator, NullSandbox
from agentfork.kvzip_router import KVCompressor, KVCacheGate, KVBudgetHint, KVZipBackend

backend = KVZipBackend(compressor, default_budget=KVBudgetHint(target_ratio=0.5))
orch = ForkOrchestrator(kv=backend, sandbox=NullSandbox())
parent = orch.create_parent("root", tokens=shared_tokens)
children = orch.fork("root", n=4)
```

Forking deep-copies the compressed cache so each child starts from the same
query-agnostic prefix; the child's own decode suffixes do not affect the parent.

## Run tests

```bash
pip install -e ".[dev]"
pytest -q              # fast CPU tests
pytest -q -m slow      # include slow tests (LLM.generate, vLLM backend)
```

The HF path uses `HuggingFaceH4/tiny-random-LlamaForCausalLM` and the vLLM path
uses a self-contained `tiny-llama-64` model fixture with `head_dim=32`.
The vLLM custom backend is exercised with the CPU vLLM wheel, so these tests
do not require a GPU.

## Validation results

* `pytest -q` passes **285 tests** locally (including `agentfork` + kvzip slow tests).
* `pytest -q -m "not slow"` passes **270 tests** locally.
* `ruff check agentfork/kvzip_router tests/kvzip` is clean.
* `ruff format --check agentfork/kvzip_router tests/kvzip` is clean.
* `python -m build --wheel` succeeds and produces an `agentfork` wheel.
* The stress/flagship suite was run three consecutive times without failure.

## Remote verification (Modal)

`modal_verify.py` runs the full suite in a clean, CPU-only Modal container:

* **51 passed, 14 warnings** in ~8 minutes.

`modal_gpu_verify.py` runs the non-vLLM suite plus a GPU micro-benchmark on a
`Tesla T4`:

* **44 passed, 7 deselected** in ~7 minutes.
* The block-mask kernel runs on CUDA and zeroes the expected low-score blocks.
* 4-bit symmetric quantize round-trips on the GPU with max error < 0.4 and
  4× memory reduction.
* `Qwen/Qwen2.5-0.5B` on a **1500-token** context keeps the answer token `" Paris"`
  at rank **#0** under 50% and 25% compression and at rank **#1** with the full
  cache (peak GPU memory ~5.7 GB).

These runs were executed from the local checkout and are reproduced by the
`pytest` commands above on any matching environment.

### Stress and flagship scenario coverage

`tests/kvzip/test_stress_and_flagship.py` adds end-to-end scenario tests and robustness
exercises:

* **Multi-query / RAG reuse:** a long document is compressed once and then
  queried by several distinct prompt tokens; each query produces a different
  next-token distribution (`test_multi_query_reuse_from_one_compressed_prefix`).
* **Agent fork fan-out:** 8 branches fork from a shared compressed prefix,
  extend with unique tokens, and generate independently.  Branch outputs are
  all distinct and total KV memory stays below a naive full-cache fan-out
  baseline (`test_agentfork_fanout_memory_and_independence`).
* **Router-driven budgets:** `KVBudgetRouter` task budgets produce monotonically
  sized compressed caches (`test_router_task_budgets_translate_to_real_kv_sizes`).
* **Randomized robustness:** the gate and allocator are exercised over random
  shapes, seeds, and extreme budgets (`0.05` to `1.0`)
  (`test_gate_robustness_random_inputs`, `test_allocator_edge_budgets`).
* **Mixed-length online distillation:** the trace loop trains on varied-length
  inputs and improves held-out gate fit
  (`test_trace_trainer_on_mixed_length_traces`).
* **vLLM backend budget sweep:** `LLM.generate` runs end-to-end under the custom
  CPU backend at `target_ratio` values of `0.25`, `0.5`, `0.75`, and `1.0`
  (`test_vllm_backend_across_budgets`).

### Thesis claim → test mapping

| Thesis claim / scenario | Verified by |
|-------------------------|-------------|
| Router maps task/adapter/quality/latency to a budget | `test_router_budget_maps_task_quality_and_latency` |
| LoRA/task condition changes gate scores | `test_gate_condition_changes_scores` |
| Budgets are per-head / per-layer, not a single global ratio | `test_per_head_budgets_vary_across_layers_and_heads` |
| Online distillation produces a usable student gate | `test_online_distillation_improves_gate` |
| Prune + quantize is multiplicative | `test_prune_plus_quantize_multiplicative` |
| A compressed prefix can decode multiple distinct queries | `test_multi_query_reuse_from_compressed_prefix` |
| Learned compression beats a uniform-drop baseline at the same ratio | `test_kvzip_beats_uniform_baseline` |
| Tighter quality budgets retain more cache and lower KL | `test_quality_savings_tradeoff` |
| Compressed decode matches the full-cache reference | `test_compressed_decode_matches_full_reference` |
| KVZipBackend satisfies agentfork lifecycle and fork/generate | `tests/kvzip/test_agentfork_integration.py` |
| Forked branches share a compressed prefix (memory < naive fan-out) | `test_forked_branches_share_compressed_prefix` |
| Live trace journal buffers and trains from synthetic traces | `tests/kvzip/test_trace_loop.py` |
| Trace training improves gate fit to a held-out prompt | `test_trace_trainer_improves_gate` |
| The gate output changes for different task/adapter conditions | `test_trace_trainer_learns_per_adapter_condition` |
| vLLM custom backend zeroes low-scoring blocks | `test_block_mask_zeroes_low_scoring_blocks` |
| vLLM `LLM.generate` runs end-to-end under the custom backend | `test_vllm_custom_backend_generates` |
| vLLM custom backend with no pruning is deterministic | `test_vllm_custom_backend_with_no_compression_matches_cpu_attn` |
| Real LoRA adapter changes gate scores and per-head budgets | `test_lora_condition_changes_retention_and_loss` |
| Downstream QA: answer token stays high-probability after compression | `test_compression_preserves_answer_probability` |
| Multi-query reuse from one compressed prefix | `test_multi_query_reuse_from_one_compressed_prefix` |
| Agent fork fan-out memory bound and independence | `test_agentfork_fanout_memory_and_independence` |
| Router budgets translate to real KV sizes | `test_router_task_budgets_translate_to_real_kv_sizes` |
| Randomized gate and allocator robustness | `test_gate_robustness_random_inputs`, `test_allocator_edge_budgets` |
| Mixed-length trace training robustness | `test_trace_trainer_on_mixed_length_traces` |
| vLLM backend across budgets | `test_vllm_backend_across_budgets` |
| Cross-model gate transfer across a real model ladder | `test_warm_started_gate_beats_random_on_target_model` |
| Long-context QA: fact preserved in a ~500-token document | `test_long_context_fact_preserved_under_compression` |
| KV cache transfer: serialized cache decoded in a fresh process | `test_cross_process_kv_transfer` |

A representative end-to-end run on the tiny CPU model:

* `KVCompressor` at `target_ratio=0.5` vs full cache: KL divergence `< 0.05` and
  max logit difference `< 0.1`.
* `KVZipBackend` + `ForkOrchestrator`: 8 children fork from a compressed prefix
  and extend independently; total KV bytes stay well below a full-cache fan-out
  baseline.

## Packaging and production readiness

* The package builds a wheel with `python -m build` and declares `kvzip`
  optional extras (`pip install -e ".[kvzip]"`).
* `pyproject.toml` registers `slow`, `gpu`, and `leaderboard` pytest markers.

### What is verified for production

* **Algorithm and API:** router-programmable budgets, per-head/per-layer allocation,
  online distillation, prune+quantize, query-aware rerank, LoRA/task
  conditioning, agentfork backend, vLLM V1 custom attention backend (CPU),
  cross-model gate transfer, long-context QA, and cross-process KV transfer.
* **Quality:** `distilgpt2` answer token stays in the top-10 with probability
  within 10% of the full cache; `Qwen/Qwen2.5-0.5B` answer token stays at rank
  #0/#1 under 50% and 25% compression on a 1500-token GPU context.
* **Reliability:** full local test suite + Modal CPU + Modal GPU all pass.

### What is NOT production-ready yet

* **vLLM/SGLang GPU backend:** the custom backend in `vllm_backend.py` inherits
  from vLLM's CPU attention backend and only runs on CPU. A production build
  needs a CUDA paged-attention backend integration.
* **Fused 4-bit KV kernels:** `quantize.py` is a reference round-trip. Real
  throughput requires fused 4-bit (or lower) dequantization inside the
  attention kernel.
* **Paged block manager integration:** freed blocks are zeroed, not returned to
  the block allocator.
* **Leaderboards:** no RULER / KVPress / LM-Evaluation-Harness runs on 100k+
  contexts yet.
* **Live trace retraining:** `TraceJournal`/`TraceTrainer` are exercised on
  synthetic buffers, not a real production router or gateway.
* **Multi-node disaggregated transfer:** cross-process `to_bytes()`/`from_bytes()`
  proves the serialization primitive, but network transfer between prefill and
  decode nodes is not exercised at scale.

These are the remaining v2/v3 backend-engineering steps.  v1 is a runnable,
testable, packageable reference implementation of the full control-plane and
algorithm stack, but it needs a GPU kernel backend before it can be deployed
as a vLLM/SGLang attention plugin.
