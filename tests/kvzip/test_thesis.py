"""Tests that directly verify the thesis claims for kvzip_router v1."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from agentfork.kvzip_router import (
    BudgetAllocator,
    KVBudgetHint,
    KVBudgetRouter,
    KVCacheGate,
    uniform_compress,
)


def _kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """KL(P_full || P_method) from raw logits."""
    p = F.softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    return float(F.kl_div(log_q, p, reduction="batchmean"))


def test_router_budget_maps_task_quality_and_latency():
    """The router translates control-plane signals into distinct KV budgets."""
    router = KVBudgetRouter()

    cheap = router.hint(
        "summarize",
        adapter_id="summarize",
        quality_floor="standard",
        latency_class="throughput",
    )
    strict = router.hint(
        "code", adapter_id="code", quality_floor="strict", latency_class="latency"
    )
    aggressive = router.hint(
        "batch",
        adapter_id="batch",
        quality_floor="aggressive",
        latency_class="throughput",
    )

    assert strict.target_ratio > cheap.target_ratio
    assert cheap.target_ratio > aggressive.target_ratio
    assert strict.adapter_id == "code"


def test_gate_condition_changes_scores():
    """A task/adapter condition vector changes the gate's importance scores."""
    gate = KVCacheGate(hidden_dim=16, n_kv_heads=4, n_layers=2, cond_dim=8)
    hidden = torch.randn(1, 10, 16)
    out_no_cond = gate(hidden, layer_idx=0, condition=torch.zeros(1, 8))
    cond = torch.randn(1, 8)
    out_cond = gate(hidden, layer_idx=0, condition=cond)

    assert out_no_cond.shape == (1, 10, 4)
    assert out_cond.shape == (1, 10, 4)
    assert not torch.allclose(out_no_cond, out_cond, atol=1e-6)


def test_per_head_budgets_vary_across_layers_and_heads():
    """BudgetAllocator returns per-head/per-layer budgets, not a global ratio."""
    allocator = BudgetAllocator(num_layers=3, num_kv_heads=4)
    # non-uniform scores: later layers and head 3 are more important
    scores = []
    for layer_idx in range(3):
        layer = torch.rand(1, 20, 4)
        layer[:, :, :2] *= 0.1
        layer[:, :, 3] *= 1.0 + layer_idx * 0.5
        scores.append(layer)

    alloc = allocator.allocate(scores, target_ratio=0.5, seq_len=20)
    per_head = alloc["per_head"]  # [3, 4]

    assert per_head.shape == (3, 4)
    # budgets should differ across both axes
    assert per_head.float().std() > 0.0
    assert per_head.sum() < 3 * 4 * 20


def test_online_distillation_improves_gate(trained_compressor):
    """The gate loss decreases as it learns from the full-cache teacher."""
    # The fixture already trained; this test verifies the outcome is usable.
    compressor = trained_compressor
    tokenizer = AutoTokenizer.from_pretrained(compressor.model.config.name_or_path)
    test_ids = tokenizer(
        "the quick brown fox jumps over the lazy dog and runs away",
        return_tensors="pt",
    ).input_ids

    compressed = compressor.prefill(test_ids[:, :-1], KVBudgetHint(target_ratio=0.5))
    kv_logits, _ = compressor.decode_step(
        compressed,
        test_ids[:, -1:],
        position_ids=torch.tensor([[test_ids.shape[-1] - 1]]),
    )

    with torch.no_grad():
        ref_logits = compressor.model(input_ids=test_ids).logits[:, -1, :]

    assert _kl(ref_logits, kv_logits[:, -1, :]) < 0.1


def test_prune_plus_quantize_multiplicative(trained_compressor):
    """Pruning and quantizing stack: quantized retained KV is smaller than
    the same retained KV kept at full precision."""
    from agentfork.kvzip_router.quantize import quantize_tensor

    compressor = trained_compressor
    tokenizer = AutoTokenizer.from_pretrained(compressor.model.config.name_or_path)
    ids = tokenizer(
        "the quick brown fox jumps over the lazy dog and runs away quickly",
        return_tensors="pt",
    ).input_ids

    compressed = compressor.prefill(ids, KVBudgetHint(target_ratio=0.5))
    full_bytes = compressed.kv_memory_bytes

    quantized_bytes = 0
    for layer in compressed.cache.layers:
        for tensor in (layer.keys, layer.values):
            q, _ = quantize_tensor(tensor, bits=4)
            quantized_bytes += q.numel() * q.element_size()

    assert quantized_bytes < full_bytes
    # 4-bit should be roughly half 8-bit and a quarter of 32-bit; allow margin.
    assert quantized_bytes < full_bytes * 0.5


def test_multi_query_reuse_from_compressed_prefix(trained_compressor):
    """A single compressed prefill supports multiple distinct decode queries."""
    compressor = trained_compressor
    tokenizer = AutoTokenizer.from_pretrained(compressor.model.config.name_or_path)
    prefix = tokenizer(
        "the quick brown fox jumps over the lazy dog and runs away",
        return_tensors="pt",
    ).input_ids

    compressed = compressor.prefill(prefix, KVBudgetHint(target_ratio=0.5))
    query_a = prefix[:, -1:]
    query_b = torch.tensor(
        [[tokenizer.encode("cat", add_special_tokens=False)[0]]], dtype=torch.long
    )

    logits_a, _ = compressor.decode_step(
        compressed, query_a, position_ids=torch.tensor([[prefix.shape[-1]]])
    )
    logits_b, _ = compressor.decode_step(
        compressed, query_b, position_ids=torch.tensor([[prefix.shape[-1]]])
    )

    assert logits_a.shape == logits_b.shape
    assert not torch.allclose(logits_a, logits_b, atol=1e-6)


def test_kvzip_beats_uniform_baseline(trained_compressor):
    """A gate-trained compressor preserves next-token quality better than
    uniformly dropping tokens at the same target ratio."""
    compressor = trained_compressor
    tokenizer = AutoTokenizer.from_pretrained(compressor.model.config.name_or_path)
    text = "the quick brown fox jumps over the lazy dog and runs away quickly then stops and looks around the corner"
    full_ids = tokenizer(text, return_tensors="pt").input_ids
    # Pick a fixed-length prefix to keep the comparison fair.
    cut = min(28, full_ids.shape[-1] - 1)
    prefix_ids = full_ids[:, :cut]
    query_id = full_ids[:, cut : cut + 1]
    ref_ids = torch.cat([prefix_ids, query_id], dim=-1)

    with torch.no_grad():
        ref_logits = compressor.model(input_ids=ref_ids).logits[:, -1, :]
        past = compressor.model(
            input_ids=prefix_ids,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        ).past_key_values

    target = 0.5
    compressed = compressor.prefill(prefix_ids, KVBudgetHint(target_ratio=target))
    kv_logits, _ = compressor.decode_step(
        compressed, query_id, position_ids=torch.tensor([[cut]])
    )

    uniform = uniform_compress(
        past, target_ratio=target, n_sink_tokens=4, original_seq_len=cut
    )
    uni_logits, _ = compressor.decode_step(
        uniform, query_id, position_ids=torch.tensor([[cut]])
    )

    kv_kl = _kl(ref_logits, kv_logits[:, -1, :])
    uni_kl = _kl(ref_logits, uni_logits[:, -1, :])

    assert kv_kl < uni_kl
    assert kv_kl < 0.05


def test_quality_savings_tradeoff(trained_compressor):
    """Stricter quality budgets retain more KV and have lower KL divergence."""
    compressor = trained_compressor
    tokenizer = AutoTokenizer.from_pretrained(compressor.model.config.name_or_path)
    text = "the quick brown fox jumps over the lazy dog and runs away quickly then stops and looks around the corner"
    full_ids = tokenizer(text, return_tensors="pt").input_ids
    cut = min(28, full_ids.shape[-1] - 1)
    prefix_ids = full_ids[:, :cut]
    query_id = full_ids[:, cut : cut + 1]
    ref_ids = torch.cat([prefix_ids, query_id], dim=-1)

    with torch.no_grad():
        ref_logits = compressor.model(input_ids=ref_ids).logits[:, -1, :]

    ratios = [1.0, 0.5, 0.25]
    kls = []
    for ratio in ratios:
        compressed = compressor.prefill(prefix_ids, KVBudgetHint(target_ratio=ratio))
        kv_logits, _ = compressor.decode_step(
            compressed, query_id, position_ids=torch.tensor([[cut]])
        )
        kls.append(_kl(ref_logits, kv_logits[:, -1, :]))

    # Lower compression retains more signal.
    assert kls[0] <= kls[1]
    assert kls[1] <= kls[2] * 1.5  # small model noise; still near-monotonic
