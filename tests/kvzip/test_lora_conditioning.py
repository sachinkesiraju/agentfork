"""Prove the gate can be conditioned on a real PEFT LoRA adapter."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentfork.kvzip_router import (
    FullCacheTeacher,
    KVBudgetHint,
    KVCacheGate,
    KVCompressor,
    OnlineDistiller,
)

peft = pytest.importorskip("peft")

_TINY_MODEL = "HuggingFaceH4/tiny-random-LlamaForCausalLM"


def test_lora_condition_changes_retention_and_loss():
    """A gate trained with the LoRA adapter active outperforms the base condition."""
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(_TINY_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        _TINY_MODEL,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    lora_config = peft.LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
    )
    peft_model = peft.get_peft_model(base_model, lora_config)

    hidden_dim = peft_model.config.hidden_size
    n_kv_heads = peft_model.config.num_key_value_heads
    n_layers = peft_model.config.num_hidden_layers

    gate = KVCacheGate(
        hidden_dim=hidden_dim,
        n_kv_heads=n_kv_heads,
        n_layers=n_layers,
        cond_dim=8,
    )
    compressor = KVCompressor(peft_model, gate=gate)

    train_text = (
        "the quick brown fox jumps over the lazy dog and runs away quickly "
        "then stops and looks around the corner of the old building"
    )
    train_ids = tokenizer(train_text, return_tensors="pt").input_ids

    teacher = FullCacheTeacher(peft_model)
    with torch.no_grad():
        teacher_scores = teacher.score(train_ids, n_kv_heads=compressor.n_kv_heads)
        outputs = peft_model(
            input_ids=train_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    lora_condition = compressor._condition(
        KVBudgetHint(task_id="math", adapter_id="lora"), device=train_ids.device
    )
    base_condition = compressor._condition(
        KVBudgetHint(task_id="math", adapter_id=""), device=train_ids.device
    )

    opt = torch.optim.Adam(gate.parameters(), lr=2e-2)
    distiller = OnlineDistiller(gate)
    for _ in range(200):
        opt.zero_grad()
        loss = distiller.distill_loss(
            outputs.hidden_states, teacher_scores, condition=lora_condition
        )
        loss.backward()
        opt.step()

    loss_lora = distiller.distill_loss(
        outputs.hidden_states, teacher_scores, condition=lora_condition
    ).item()
    loss_base = distiller.distill_loss(
        outputs.hidden_states, teacher_scores, condition=base_condition
    ).item()

    # The LoRA-aware condition should fit the LoRA teacher better than a generic base condition.
    assert loss_lora < loss_base

    # The condition should change the gate's per-token scores for the same input.
    full_text = "KV cache compression reduces memory for long context serving"
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids
    with torch.no_grad():
        hidden = peft_model(
            full_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        ).hidden_states
    scores_lora = gate(hidden[0], 0, condition=lora_condition)
    scores_base = gate(hidden[0], 0, condition=base_condition)
    assert not torch.allclose(scores_lora, scores_base, atol=1e-6)

    # Per-head budgets should also differ because the condition reallocates KV heads.
    compressed_lora = compressor.prefill(
        full_ids, KVBudgetHint(task_id="math", adapter_id="lora", target_ratio=0.5)
    )
    compressed_base = compressor.prefill(
        full_ids, KVBudgetHint(task_id="math", adapter_id="", target_ratio=0.5)
    )
    assert not torch.equal(
        compressed_lora.per_head_budgets, compressed_base.per_head_budgets
    )
