"""RULER-lite leaderboard: needle-in-haystack fact retrieval under compression."""

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


def _answer_stats(logits: torch.Tensor, answer_id: int) -> tuple[float, int]:
    probs = F.softmax(logits, dim=-1)
    rank = int((probs[0] > probs[0, answer_id]).sum().item())
    return float(probs[0, answer_id].item()), rank


@pytest.mark.slow
def test_ruler_lite_needle_in_haystack():
    """A fact buried in 256 tokens of filler is retrievable at 50% and 75% compression."""
    torch.manual_seed(0)

    model_name = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    target_fact = "The secret code is 7391."
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
    filler = " ".join([para] * 1)
    context = filler + " " + target_fact + " " + filler
    context_ids = tokenizer(context, return_tensors="pt").input_ids
    # Trim/pad to the target length.
    if context_ids.shape[-1] > 256:
        context_ids = context_ids[:, -256:]
    elif context_ids.shape[-1] < 256:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        context_ids = torch.nn.functional.pad(
            context_ids, (0, 256 - context_ids.shape[-1]), value=pad_id
        )

    question = "The secret code is"
    question_ids = tokenizer(question, return_tensors="pt").input_ids
    answer_id = int(tokenizer(" 7391", add_special_tokens=False).input_ids[0])

    gate = KVCacheGate(
        hidden_dim=model.config.hidden_size,
        n_kv_heads=getattr(model.config, "num_key_value_heads", model.config.n_head),
        n_layers=model.config.n_layer,
        cond_dim=0,
    )
    compressor = KVCompressor(model, gate=gate)
    teacher = FullCacheTeacher(model)
    opt = torch.optim.Adam(gate.parameters(), lr=2e-2)

    for _ in range(50):
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

    full_ids = torch.cat([context_ids, question_ids], dim=-1)
    with torch.no_grad():
        ref_logits = model(
            input_ids=full_ids, use_cache=False, return_dict=True
        ).logits[:, -1, :]
    ref_prob, ref_rank = _answer_stats(ref_logits, answer_id)
    assert ref_rank <= 10, f"full-cache rank {ref_rank} is unexpectedly poor"

    leaderboard: dict = {}
    for ratio in [1.0, 0.75, 0.5]:
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
        comp_prob, comp_rank = _answer_stats(comp_logits, answer_id)
        leaderboard[ratio] = {
            "answer_rank": comp_rank,
            "answer_prob": comp_prob,
            "effective_ratio": compressed.effective_ratio,
        }
        assert comp_rank <= 10, (
            f"compression ratio={ratio} lost the needle (rank {comp_rank})"
        )
        assert abs(comp_prob - ref_prob) < 0.15, (
            f"ratio={ratio} changed answer probability too much: {ref_prob:.3f} -> {comp_prob:.3f}"
        )

    print("leaderboard:", leaderboard)
