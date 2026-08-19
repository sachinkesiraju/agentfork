"""Long-context QA: a fact embedded in a long document is preserved after KV compression."""

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
def test_long_context_fact_preserved_under_compression():
    """distilgpt2: a fact in the middle of a ~400-token document survives compression."""
    torch.manual_seed(0)

    model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    # A long document with the target fact embedded in the middle.
    target_fact = "The capital of France is Paris."
    para = (
        "The city has many museums and restaurants. People from around the world "
        "visit the old streets and the river that runs through the center. "
        "In winter it can be cold, while summer brings long evenings. "
        "History records show that the region was settled thousands of years ago. "
        "Modern transit connects every neighborhood to the rest of the country. "
        "Artists and writers have long been inspired by the light and the crowds. "
        "The local market sells fresh bread, cheese, and fruit every morning. "
        "Across the river, old bridges connect the left bank to the right bank. "
        "Visitors climb the tower to see the entire city spread below them. "
        "At night the avenues are bright and the cafés stay open late. "
    )
    filler = " ".join([para] * 2)
    context = filler + " " + target_fact + " " + filler

    question = "The capital of France is"
    answer_text = " Paris"

    context_ids = tokenizer(context, return_tensors="pt").input_ids
    question_ids = tokenizer(question, return_tensors="pt").input_ids
    answer_id = int(tokenizer(answer_text, return_tensors="pt").input_ids[0, 0])

    assert context_ids.shape[-1] >= 400, "document should be long-context length"

    # Train a gate on the long document.
    gate = KVCacheGate(
        hidden_dim=model.config.hidden_size,
        n_kv_heads=model.config.n_head,
        n_layers=model.config.n_layer,
        cond_dim=0,
    )
    compressor = KVCompressor(model, gate=gate)
    teacher = FullCacheTeacher(model)
    opt = torch.optim.Adam(gate.parameters(), lr=2e-2)

    for _ in range(100):
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

    # Full-cache reference.
    full_ids = torch.cat([context_ids, question_ids], dim=-1)
    with torch.no_grad():
        ref = model(input_ids=full_ids, use_cache=False, return_dict=True)
    ref_logits = ref.logits[:, -1, :]
    ref_prob = _answer_probability(ref_logits, answer_id)
    ref_rank = _answer_rank(ref_logits, answer_id)

    assert ref_prob > 0.05, (
        "pretrained model should assign some probability to the answer"
    )
    assert ref_rank <= 10, f"answer rank without compression is {ref_rank}"

    # Compress and decode at moderate ratios.
    for ratio in [1.0, 0.8]:
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

        assert comp_rank <= 10, (
            f"ratio={ratio} dropped answer out of top-10 (rank {comp_rank})"
        )
        assert abs(comp_prob - ref_prob) < 0.1, (
            f"ratio={ratio} changed answer probability too much: {ref_prob:.3f} -> {comp_prob:.3f}"
        )

        # The cache should actually shrink.
        if ratio < 1.0:
            assert compressed.effective_ratio < 1.0
