"""Cross-model gate transfer: a small model's gate warms up a larger model's gate."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentfork.kvzip_router import FullCacheTeacher, KVCacheGate, OnlineDistiller
from agentfork.kvzip_router.transfer import warm_start_gate


def _train_gate(model, gate, text_ids, steps: int, lr: float = 2e-2) -> float:
    """Train ``gate`` to match ``model``'s full-cache attention teacher."""
    teacher = FullCacheTeacher(model)
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        with torch.no_grad():
            scores = teacher.score(text_ids, n_kv_heads=gate.n_kv_heads)
            outputs = model(
                input_ids=text_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        loss = OnlineDistiller(gate).distill_loss(outputs.hidden_states, scores)
        loss.backward()
        opt.step()
    return float(loss.item())


@pytest.mark.slow
def test_warm_started_gate_beats_random_on_target_model():
    """A gate trained on ``distilgpt2`` and transferred to ``gpt2`` outperforms
    a randomly initialized ``gpt2`` gate after the same fine-tuning budget."""
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = (
        "The capital of France is Paris. The capital of Germany is Berlin. "
        "The capital of Italy is Rome."
    )
    text_ids = tokenizer(text, return_tensors="pt").input_ids

    # Source model: distilgpt2 (6 layers)
    src_model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2",
        attn_implementation="eager",
    )
    src_gate = KVCacheGate(
        hidden_dim=src_model.config.n_embd,
        n_kv_heads=src_model.config.n_head,
        n_layers=src_model.config.n_layer,
        cond_dim=0,
    )
    _train_gate(src_model, src_gate, text_ids, steps=100)

    # Target model: gpt2 (12 layers)
    tgt_model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        attn_implementation="eager",
    )

    # Warm-start a target gate from the source gate.
    tgt_gate_warm = warm_start_gate(
        src_gate,
        target_hidden_dim=tgt_model.config.n_embd,
        target_n_kv_heads=tgt_model.config.n_head,
        target_n_layers=tgt_model.config.n_layer,
    )
    warm_loss = _train_gate(tgt_model, tgt_gate_warm, text_ids, steps=50)

    # Random target gate trained for the same number of steps.
    tgt_gate_random = KVCacheGate(
        hidden_dim=tgt_model.config.n_embd,
        n_kv_heads=tgt_model.config.n_head,
        n_layers=tgt_model.config.n_layer,
        cond_dim=0,
    )
    random_loss = _train_gate(tgt_model, tgt_gate_random, text_ids, steps=50)

    # The warm-started gate should reach a lower distillation loss.
    assert warm_loss < random_loss, (
        f"Warm-started gate ({warm_loss:.4f}) did not beat random gate "
        f"({random_loss:.4f}) on the target model"
    )
