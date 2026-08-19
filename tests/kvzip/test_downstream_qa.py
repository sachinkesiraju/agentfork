"""A real downstream cloze benchmark on a pretrained model.

The question-answering setup:
- context: a short passage containing facts about capital cities
- question: a fill-in-the-blank prompt
- answer: the capital token (e.g. " Paris")

We assert that even after aggressive KV-cache compression the answer token
remains high-probability and close to the full-cache distribution.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentfork.kvzip_router import (
    FullCacheTeacher,
    KVBudgetHint,
    KVCacheGate,
    KVCompressor,
    OnlineDistiller,
)


def _answer_probability(logits: torch.Tensor, answer_id: int) -> float:
    probs = F.softmax(logits, dim=-1)
    return float(probs[0, answer_id].item())


def _answer_rank(logits: torch.Tensor, answer_id: int) -> int:
    probs = F.softmax(logits, dim=-1)
    return int((probs[0] > probs[0, answer_id]).sum().item())


@pytest.mark.slow
def test_compression_preserves_answer_probability():
    """distilgpt2: compressed KV cache keeps the answer token ranked high."""
    torch.manual_seed(0)

    model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    context = (
        "The capital of France is Paris. The capital of Germany is Berlin. "
        "The capital of Italy is Rome."
    )
    question = "The capital of France is"
    answer_text = " Paris"

    context_ids = tokenizer(context, return_tensors="pt").input_ids
    question_ids = tokenizer(question, return_tensors="pt").input_ids
    answer_ids = tokenizer(answer_text, return_tensors="pt").input_ids
    answer_id = int(answer_ids[0, 0])

    # Train a small gate on the passage it will compress (online per-context distillation).
    gate = KVCacheGate(
        hidden_dim=model.config.hidden_size,
        n_kv_heads=model.config.n_head,
        n_layers=model.config.n_layer,
        cond_dim=0,
    )
    compressor = KVCompressor(model, gate=gate)
    teacher = FullCacheTeacher(model)
    opt = torch.optim.Adam(gate.parameters(), lr=2e-2)

    for _ in range(300):
        opt.zero_grad()
        with torch.no_grad():
            scores = teacher.score(context_ids, n_kv_heads=compressor.n_kv_heads)
            outputs = model(
                input_ids=context_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        loss = OnlineDistiller(gate).distill_loss(outputs.hidden_states, scores)
        loss.backward()
        opt.step()

    # Full-cache reference: context + question -> next-token distribution.
    full_ids = torch.cat([context_ids, question_ids], dim=-1)
    with torch.no_grad():
        ref = model(input_ids=full_ids, use_cache=False, return_dict=True)
    ref_logits = ref.logits[:, -1, :]
    ref_prob = _answer_probability(ref_logits, answer_id)
    ref_rank = _answer_rank(ref_logits, answer_id)

    # The pretrained model should place the answer token in the top few.
    assert ref_prob > 0.1
    assert ref_rank <= 5

    # Compress the context and decode the question through the cache.
    full_prob = ref_prob
    prev_prob = full_prob
    for ratio in [1.0, 0.75, 0.5, 0.25]:
        compressed = compressor.prefill(context_ids, KVBudgetHint(target_ratio=ratio))
        pos = context_ids.shape[-1]
        with torch.no_grad():
            out = model(
                input_ids=question_ids,
                past_key_values=compressed.cache,
                position_ids=torch.arange(pos, pos + question_ids.shape[-1]).unsqueeze(
                    0
                ),
                use_cache=True,
                return_dict=True,
            )
        comp_logits = out.logits[:, -1, :]
        comp_prob = _answer_probability(comp_logits, answer_id)
        comp_rank = _answer_rank(comp_logits, answer_id)

        # The answer token should stay high-probability and high-rank.
        assert comp_rank <= 5, f"ratio={ratio} dropped answer out of top-5"
        assert abs(comp_prob - full_prob) < 0.1, (
            f"ratio={ratio} changed answer probability too much: {full_prob:.3f} -> {comp_prob:.3f}"
        )

        # Tighter budgets should not drift further from full than looser budgets.
        # (The assertion is intentionally loose to survive tiny-model variance.)
        if ratio <= 0.5:
            assert abs(comp_prob - full_prob) <= abs(prev_prob - full_prob) + 0.05

        prev_prob = comp_prob
