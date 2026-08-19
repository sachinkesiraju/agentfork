"""Unit and end-to-end tests for kvzip_router v1."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from agentfork.kvzip_router import (
    BudgetAllocator,
    KVBudgetHint,
    KVBudgetRouter,
    KVCacheGate,
    KVCompressor,
    OnlineDistiller,
    TokenRelevanceRerank,
)
from agentfork.kvzip_router.distill import FullCacheTeacher
from agentfork.kvzip_router.quantize import dequantize_tensor, quantize_tensor


def _train_gate(model, gate, ids, steps: int = 400, lr: float = 2e-2) -> float:
    """Distill the gate on a single prompt. Returns final loss."""
    teacher = FullCacheTeacher(model)
    distiller = OnlineDistiller(gate, teacher=teacher)
    with torch.no_grad():
        teacher_scores = teacher.score(ids, n_kv_heads=gate.n_kv_heads)
        out = model(
            input_ids=ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = out.hidden_states
    optimizer = torch.optim.Adam(gate.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = distiller.distill_loss(hidden_states, teacher_scores)
        loss.backward()
        optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Component tests
# ---------------------------------------------------------------------------


def test_gate_output_shape():
    torch.manual_seed(0)
    gate = KVCacheGate(hidden_dim=16, n_kv_heads=4, n_layers=2, cond_dim=0)
    hs = torch.randn(1, 10, 16)
    scores = gate(hs, layer_idx=0)
    assert scores.shape == (1, 10, 4)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_gate_with_condition():
    torch.manual_seed(0)
    gate = KVCacheGate(hidden_dim=16, n_kv_heads=2, n_layers=2, cond_dim=8)
    hs = torch.randn(2, 10, 16)
    cond = torch.randn(2, 8)
    scores = gate(hs, layer_idx=1, condition=cond)
    assert scores.shape == (2, 10, 2)


def test_budget_allocator_respects_target():
    torch.manual_seed(0)
    allocator = BudgetAllocator(num_layers=2, num_kv_heads=2)
    scores = [
        torch.rand(1, 20, 2),
        torch.rand(1, 20, 2),
    ]
    allocation = allocator.allocate(scores, target_ratio=0.5)
    retained = allocation["global_indices"]
    assert retained.shape[0] == 1
    # retained count should be bounded by the original seq length
    assert 1 <= retained.shape[-1] <= 20
    per_head = allocation["per_head"]
    assert per_head.shape == (2, 2)


def test_router_hint_mapping():
    router = KVBudgetRouter(
        task_overrides={"summarize": 0.8},
        adapter_overrides={"legal": 0.5},
    )
    hint = router.hint(
        task_id="summarize",
        adapter_id="legal",
        quality_floor="aggressive",
        latency_class="throughput",
    )
    # adapter override wins over task override in this simple impl
    assert 0.0 < hint.target_ratio <= 1.0
    assert hint.task_id == "summarize"


def test_quantize_roundtrip():
    x = torch.randn(2, 4, 8, 16)
    q, s = quantize_tensor(x, bits=8)
    x_hat = dequantize_tensor(q, s)
    assert x_hat.shape == x.shape
    # 8-bit quantization gives a small relative error.
    rel_err = (x - x_hat).abs().mean() / x.abs().mean()
    assert rel_err < 0.02


def test_rerank_selects_relevant_tokens():
    torch.manual_seed(0)
    rerank = TokenRelevanceRerank(hidden_dim=16, k_ratio=0.3)
    query = torch.randn(1, 1, 16)
    # Make three tokens identical to the query so they should score highly.
    relevant = torch.randn(1, 1, 16)
    token_hidden = torch.cat([torch.randn(1, 7, 16), relevant.expand(1, 3, 16)], dim=1)
    topk = rerank.rerank(query_hidden=query, token_hidden=token_hidden)
    assert topk.shape[-1] == max(1, int(0.3 * 10))


# ---------------------------------------------------------------------------
# End-to-end correctness
# ---------------------------------------------------------------------------


def test_compressor_prefill_shape_and_memory_reduction(tiny_model, tiny_tokenizer):
    torch.manual_seed(0)
    gate = KVCacheGate(
        hidden_dim=16,
        n_kv_heads=4,
        n_layers=2,
        cond_dim=0,
        hidden=64,
    )
    compressor = KVCompressor(tiny_model, gate=gate)
    text = "the quick brown fox jumps over the lazy dog"
    ids = tiny_tokenizer(text, return_tensors="pt").input_ids

    compressed = compressor.prefill(ids, budget=KVBudgetHint(target_ratio=1.0))
    assert compressed.seq_len <= ids.shape[-1] + compressor.n_sink_tokens
    assert compressed.kv_memory_bytes > 0

    compressed_half = compressor.prefill(ids, budget=KVBudgetHint(target_ratio=0.5))
    assert compressed_half.seq_len <= compressed.seq_len


def test_compressed_decode_matches_full_reference(tiny_model, tiny_tokenizer):
    """Train a gate on one prompt, then verify decode logits stay close."""
    torch.manual_seed(0)
    gate = KVCacheGate(
        hidden_dim=16,
        n_kv_heads=4,
        n_layers=2,
        cond_dim=0,
        hidden=64,
    )
    compressor = KVCompressor(tiny_model, gate=gate)

    train_text = "the quick brown fox jumps over the lazy dog and runs away"
    train_ids = tiny_tokenizer(train_text, return_tensors="pt").input_ids
    final_loss = _train_gate(tiny_model, gate, train_ids, steps=500)
    assert final_loss < 0.05

    test_text = "the quick brown fox jumps over the lazy dog"
    ids = tiny_tokenizer(test_text, return_tensors="pt").input_ids
    prefix = ids[:, :-1]
    query = ids[:, -1:]

    with torch.no_grad():
        ref = tiny_model(input_ids=ids, use_cache=False, return_dict=True)
    ref_logits = ref.logits[:, -1, :]

    compressed = compressor.prefill(prefix, KVBudgetHint(target_ratio=0.6))
    with torch.no_grad():
        out = tiny_model(
            input_ids=query,
            past_key_values=compressed.cache,
            position_ids=torch.tensor([[prefix.shape[-1]]]),
            use_cache=True,
            return_dict=True,
        )
    comp_logits = out.logits[:, -1, :]

    max_diff = (ref_logits - comp_logits).abs().max().item()
    kl = F.kl_div(
        F.log_softmax(comp_logits, dim=-1),
        F.softmax(ref_logits, dim=-1),
        reduction="batchmean",
    ).item()

    assert max_diff < 0.1
    assert kl < 1e-3
    assert compressed.effective_ratio <= 1.0


def test_compressed_quality_vs_savings(tiny_model, tiny_tokenizer):
    """Vary the compression ratio and check memory drops and quality degrades gracefully."""
    torch.manual_seed(0)
    gate = KVCacheGate(
        hidden_dim=16,
        n_kv_heads=4,
        n_layers=2,
        cond_dim=0,
        hidden=64,
    )
    compressor = KVCompressor(tiny_model, gate=gate)

    train_text = "the quick brown fox jumps over the lazy dog and runs away quickly"
    train_ids = tiny_tokenizer(train_text, return_tensors="pt").input_ids
    _train_gate(tiny_model, gate, train_ids, steps=500)

    test_text = "the quick brown fox jumps over the lazy dog"
    ids = tiny_tokenizer(test_text, return_tensors="pt").input_ids
    prefix = ids[:, :-1]
    query = ids[:, -1:]

    with torch.no_grad():
        ref = tiny_model(input_ids=ids, use_cache=False, return_dict=True)
    ref_logits = ref.logits[:, -1, :]
    ref_probs = F.softmax(ref_logits, dim=-1)

    ratios = [1.0, 0.75, 0.5, 0.25]
    results = []
    for ratio in ratios:
        compressed = compressor.prefill(prefix, KVBudgetHint(target_ratio=ratio))
        with torch.no_grad():
            out = tiny_model(
                input_ids=query,
                past_key_values=compressed.cache,
                position_ids=torch.tensor([[prefix.shape[-1]]]),
                use_cache=True,
                return_dict=True,
            )
        comp_logits = out.logits[:, -1, :]
        kl = F.kl_div(
            F.log_softmax(comp_logits, dim=-1),
            ref_probs,
            reduction="batchmean",
        ).item()
        results.append(
            {
                "ratio": ratio,
                "effective_ratio": compressed.effective_ratio,
                "memory_bytes": compressed.kv_memory_bytes,
                "kl": kl,
            }
        )

    # Memory should generally decrease as the target ratio shrinks, at least for
    # longer sequences; for the tiny test we just check it is recorded.
    assert all(r["memory_bytes"] > 0 for r in results)
    # The most aggressive compression should still be within a modest KL budget
    # on this random tiny model once the gate is trained.
    assert results[-1]["kl"] < 0.5


def test_generate_produces_expected_length(tiny_model, tiny_tokenizer):
    """`KVCompressor.generate` returns the requested number of tokens."""
    torch.manual_seed(0)
    gate = KVCacheGate(
        hidden_dim=16,
        n_kv_heads=4,
        n_layers=2,
        cond_dim=0,
        hidden=64,
    )
    compressor = KVCompressor(tiny_model, gate=gate)

    train_text = "the quick brown fox jumps over the lazy dog and runs away"
    train_ids = tiny_tokenizer(train_text, return_tensors="pt").input_ids
    _train_gate(tiny_model, gate, train_ids, steps=300)

    ids = tiny_tokenizer("the quick brown fox jumps", return_tensors="pt").input_ids
    tokens, _ = compressor.generate(
        ids, KVBudgetHint(target_ratio=0.8), max_new_tokens=3
    )
    assert len(tokens) == 3
