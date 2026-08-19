"""Stress and flagship-scenario tests for kvzip_router v1.

These tests exercise the system-level claims: multi-query reuse, agent fork
fan-out, router-driven budgets, and robustness under randomized inputs and
extreme budgets.
"""

from __future__ import annotations

import pytest
import torch
from agentfork import ForkOrchestrator, NullSandbox
from transformers import AutoTokenizer

from agentfork.kvzip_router import (
    BudgetAllocator,
    KVBudgetHint,
    KVBudgetRouter,
    KVCacheGate,
    KVCompressor,
    KVZipBackend,
    TraceJournal,
    TraceTrainer,
)
from agentfork.kvzip_router.distill import FullCacheTeacher
from agentfork.kvzip_router.trace import TraceRecord

try:
    import vllm  # noqa: F401

    HAS_VLLM = True
except Exception:
    HAS_VLLM = False


def _clone_compressed(compressed):
    """Return a deep-enough copy of a CompressedKVCache for parallel decodes."""
    new_tuples = [
        (layer.keys.clone(), layer.values.clone()) for layer in compressed.cache.layers
    ]
    from transformers import DynamicCache

    return type(compressed)(
        cache=DynamicCache(ddp_cache_data=new_tuples),
        retained_indices=compressed.retained_indices.clone(),
        original_seq_len=compressed.original_seq_len,
        target_ratio=compressed.target_ratio,
        retained_hidden=(
            compressed.retained_hidden.clone()
            if compressed.retained_hidden is not None
            else None
        ),
        per_head_budgets=(
            compressed.per_head_budgets.clone()
            if compressed.per_head_budgets is not None
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Flagship: multi-turn / RAG reuse
# ---------------------------------------------------------------------------


def test_multi_query_reuse_from_one_compressed_prefix(trained_compressor):
    """A long document is compressed once and then queried multiple times."""
    compressor = trained_compressor
    tokenizer = AutoTokenizer.from_pretrained(compressor.model.config.name_or_path)
    document = (
        "The quick brown fox jumps over the lazy dog. "
        "In 1492, Columbus sailed the ocean blue. "
        "Machine learning models compress key-value caches to save memory. "
        "RAG systems retrieve documents before generating an answer."
    )
    prefix = tokenizer(document, return_tensors="pt").input_ids
    # Clone so each query decodes from the same compressed prefix independently.
    compressed = _clone_compressed(
        compressor.prefill(prefix, KVBudgetHint(target_ratio=0.5))
    )

    queries = ["fox", "1492", "RAG", "answer"]
    results = []
    for q in queries:
        query_ids = torch.tensor(
            [[tokenizer.encode(q, add_special_tokens=False)[0]]],
            dtype=torch.long,
        )
        comp_copy = _clone_compressed(compressed)
        logits, _ = compressor.decode_step(
            comp_copy,
            query_ids,
            position_ids=torch.tensor([[prefix.shape[-1]]]),
        )
        results.append(logits[:, -1, :])

    # Distinct queries should produce distinct next-token distributions.
    for i in range(len(results) - 1):
        assert not torch.allclose(results[i], results[i + 1], atol=1e-6)


# ---------------------------------------------------------------------------
# Flagship: agent fork fan-out
# ---------------------------------------------------------------------------


def _bytes_per_token(cfg):
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    return cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * head_dim * 4


def test_agentfork_fanout_memory_and_independence(trained_compressor, tiny_tokenizer):
    """Many branches fork from a shared compressed prefix and then diverge."""
    compressor = trained_compressor
    backend = KVZipBackend(
        compressor,
        default_budget=KVBudgetHint(target_ratio=0.4),
    )
    orch = ForkOrchestrator(kv=backend, sandbox=NullSandbox())

    prefix = (
        tiny_tokenizer(
            "the quick brown fox jumps over the lazy dog and runs away quickly then stops and looks around the corner",
            return_tensors="pt",
        )
        .input_ids[0]
        .tolist()
    )
    n = 8
    orch.create_parent("root", tokens=prefix)
    children = orch.fork("root", n=n)

    outputs = []
    for i, child in enumerate(children):
        # Each branch extends with a unique token and generates 3 new tokens.
        unique = [1000 + i]
        orch.extend(child.branch_id, unique)
        out = orch.generate(
            child.branch_id,
            [],
            {"max_new_tokens": 3, "temperature": 1.0},
        )
        outputs.append(out["token_ids"])
        assert len(out["token_ids"]) == 3

    # Branches must have diverged.
    assert len({tuple(o) for o in outputs}) == n

    # Total memory must be far below a naive full-cache fan-out.
    cfg = compressor.model.config
    bytes_per_tok = _bytes_per_token(cfg)
    uncompressed = n * (len(prefix) + 1 + 3) * bytes_per_tok
    assert backend.total_kv_bytes() < uncompressed


# ---------------------------------------------------------------------------
# Flagship: router-driven budget per task
# ---------------------------------------------------------------------------


def test_router_task_budgets_translate_to_real_kv_sizes(
    trained_compressor, tiny_tokenizer
):
    """The router emits different budgets and the compressor honors them."""
    compressor = trained_compressor
    router = KVBudgetRouter(
        task_overrides={"summarize": 0.8, "code": 0.5, "batch": 0.25},
    )
    text = (
        "The quick brown fox jumps over the lazy dog and runs away quickly then "
        "stops and looks around the corner of the old stone building"
    )
    ids = tiny_tokenizer(text, return_tensors="pt").input_ids

    budgets = [
        router.hint(task_id="summarize"),
        router.hint(task_id="code"),
        router.hint(task_id="batch"),
    ]
    sizes = []
    for budget in budgets:
        compressed = compressor.prefill(ids, budget)
        sizes.append(compressed.seq_len)

    # Strict/richer tasks should retain more tokens than aggressive/cheap ones.
    assert sizes[0] >= sizes[1] >= sizes[2]
    assert all(s > 0 for s in sizes)


# ---------------------------------------------------------------------------
# Stress: randomized component invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(range(5)))
def test_gate_random_inputs_invariant(seed):
    """The gate always returns valid scores for random shapes and conditions."""
    torch.manual_seed(seed)
    hidden_dim = torch.randint(8, 64, (1,)).item()
    n_kv_heads = torch.randint(1, 8, (1,)).item()
    n_layers = torch.randint(1, 5, (1,)).item()
    cond_dim = torch.randint(0, 16, (1,)).item()
    seq_len = torch.randint(4, 40, (1,)).item()
    batch = torch.randint(1, 4, (1,)).item()

    gate = KVCacheGate(
        hidden_dim=hidden_dim,
        n_kv_heads=n_kv_heads,
        n_layers=n_layers,
        cond_dim=cond_dim,
    )
    hidden = torch.randn(batch, seq_len, hidden_dim)
    cond = torch.randn(batch, cond_dim) if cond_dim > 0 else None

    for layer_idx in range(n_layers):
        scores = gate(hidden, layer_idx, condition=cond)
        assert scores.shape == (batch, seq_len, n_kv_heads)
        assert (scores >= 0).all() and (scores <= 1).all()


@pytest.mark.parametrize("seed", list(range(5)))
def test_allocator_edge_budgets(seed):
    """BudgetAllocator handles extreme ratios and random score landscapes."""
    torch.manual_seed(seed)
    n_layers = torch.randint(1, 5, (1,)).item()
    n_kv_heads = torch.randint(1, 6, (1,)).item()
    seq_len = torch.randint(10, 60, (1,)).item()

    scores = [torch.rand(1, seq_len, n_kv_heads) for _ in range(n_layers)]
    allocator = BudgetAllocator(num_layers=n_layers, num_kv_heads=n_kv_heads)

    for ratio in [0.05, 0.25, 0.5, 0.75, 1.0]:
        allocation = allocator.allocate(scores, target_ratio=ratio, seq_len=seq_len)
        retained = allocation["global_indices"]
        assert 1 <= retained.shape[-1] <= seq_len
        assert allocation["per_head"].shape == (n_layers, n_kv_heads)

    # With ratio 1.0 the allocator should keep essentially everything.
    full = allocator.allocate(scores, target_ratio=1.0, seq_len=seq_len)
    assert full["global_indices"].shape[-1] == seq_len


# ---------------------------------------------------------------------------
# Stress: distillation from mixed-length traces
# ---------------------------------------------------------------------------


def test_trace_trainer_on_mixed_length_traces(tiny_tokenizer, tiny_model):
    """The online trace loop trains on varied-length inputs and improves."""
    gate = KVCacheGate(
        hidden_dim=tiny_model.config.hidden_size,
        n_kv_heads=tiny_model.config.num_key_value_heads,
        n_layers=tiny_model.config.num_hidden_layers,
        cond_dim=8,
    )
    compressor = KVCompressor(tiny_model, gate=gate)
    teacher = FullCacheTeacher(tiny_model)

    texts = [
        "hello world",
        "the quick brown fox jumps over the lazy dog",
        "one two three four five six seven eight nine ten",
        "a",
    ]
    journal = TraceJournal(capacity=8)
    for i, text in enumerate(texts):
        ids = tiny_tokenizer(text, return_tensors="pt").input_ids
        journal.add(
            TraceRecord(
                ids,
                budget=KVBudgetHint(task_id=f"t{i}", adapter_id=f"a{i}"),
            )
        )

    def eval_loss(text):
        ids = tiny_tokenizer(text, return_tensors="pt").input_ids
        budget = KVBudgetHint(task_id="eval", adapter_id="eval")
        condition = compressor._condition(budget, device=ids.device)
        with torch.no_grad():
            outputs = tiny_model(
                ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            targets = teacher.score(ids, n_kv_heads=compressor.n_kv_heads)
            from agentfork.kvzip_router import OnlineDistiller

            loss = OnlineDistiller(gate).distill_loss(
                outputs.hidden_states, targets, condition=condition
            )
        return float(loss)

    initial = eval_loss("the cat sat on the mat and looked around")
    trainer = TraceTrainer(compressor, teacher=teacher, lr=5e-3)
    trainer.fit(journal.sample(), epochs=20)
    final = eval_loss("the cat sat on the mat and looked around")

    assert final < initial * 0.95


# ---------------------------------------------------------------------------
# Stress: vLLM backend over multiple budgets
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")
@pytest.mark.slow
@pytest.mark.parametrize("target_ratio", [0.25, 0.5, 0.75, 1.0])
def test_vllm_backend_across_budgets(vllm_tiny_model_path, target_ratio):
    """vLLM generation works and stays deterministic when no blocks are dropped."""
    import os

    os.environ["KVZIP_TARGET_RATIO"] = str(target_ratio)
    os.environ["KVZIP_MIN_BLOCKS"] = "1"

    from agentfork.kvzip_router.vllm_backend import (
        FastKVzipAttentionImpl,
        set_kvzip_budget,
    )

    set_kvzip_budget(target_ratio, 1)
    FastKVzipAttentionImpl.target_ratio = target_ratio
    FastKVzipAttentionImpl.min_blocks = 1

    from vllm import LLM, SamplingParams

    dtype = "float16" if torch.cuda.is_available() else "float32"
    llm = LLM(
        model=vllm_tiny_model_path,
        dtype=dtype,
        max_model_len=128,
        enforce_eager=True,
        gpu_memory_utilization=0.2,
        attention_config={"backend": "CUSTOM"},
    )
    outputs = llm.generate(
        "hello world",
        SamplingParams(max_tokens=2, temperature=0),
    )
    text1 = outputs[0].outputs[0].text
    assert isinstance(text1, str)

    if target_ratio == 1.0:
        outputs2 = llm.generate(
            "hello world",
            SamplingParams(max_tokens=2, temperature=0),
        )
        assert outputs2[0].outputs[0].text == text1
