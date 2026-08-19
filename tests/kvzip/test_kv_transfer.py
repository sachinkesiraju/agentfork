"""KV cache transfer: a compressed cache can be serialized, moved, and reused."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentfork.kvzip_router import (
    CompressedKVCache,
    FullCacheTeacher,
    KVBudgetHint,
    KVCacheGate,
    KVCompressor,
    OnlineDistiller,
)


@pytest.mark.slow
def test_compressed_cache_round_trip_preserves_decode_logits():
    """Serialize a ``CompressedKVCache`` to bytes and reconstruct it; decoding
    from the reconstructed cache yields the same next-token distribution."""
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
    context_ids = tokenizer(context, return_tensors="pt").input_ids
    question_ids = tokenizer(question, return_tensors="pt").input_ids

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

    compressed = compressor.prefill(context_ids, KVBudgetHint(target_ratio=0.5))

    # Round-trip through bytes before any decode mutates the cache in place.
    payload = compressed.to_bytes()
    transferred = CompressedKVCache.from_bytes(payload)

    pos = context_ids.shape[-1]
    with torch.no_grad():
        out1 = model(
            input_ids=question_ids,
            past_key_values=compressed.cache,
            position_ids=torch.arange(pos, pos + question_ids.shape[-1]).unsqueeze(0),
            use_cache=True,
            return_dict=True,
        )
        out2 = model(
            input_ids=question_ids,
            past_key_values=transferred.cache,
            position_ids=torch.arange(pos, pos + question_ids.shape[-1]).unsqueeze(0),
            use_cache=True,
            return_dict=True,
        )
    logits1 = out1.logits[:, -1, :]
    logits2 = out2.logits[:, -1, :]

    assert torch.allclose(logits1, logits2, atol=1e-5), (
        "Transferred cache did not reproduce the same decode logits"
    )
    assert transferred.original_seq_len == compressed.original_seq_len
    assert transferred.target_ratio == compressed.target_ratio
    assert torch.equal(transferred.retained_indices, compressed.retained_indices)


@pytest.mark.slow
def test_cross_process_kv_transfer():
    """A compressed cache can be handed to a fresh Python process and decoded."""
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
    context_ids = tokenizer(context, return_tensors="pt").input_ids
    question_ids = tokenizer(question, return_tensors="pt").input_ids

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

    compressed = compressor.prefill(context_ids, KVBudgetHint(target_ratio=0.5))

    # In-process reference decode.
    pos = context_ids.shape[-1]
    with torch.no_grad():
        ref_out = model(
            input_ids=question_ids,
            past_key_values=compressed.cache,
            position_ids=torch.arange(pos, pos + question_ids.shape[-1]).unsqueeze(0),
            use_cache=True,
            return_dict=True,
        )
    ref_top_id = int(torch.argmax(ref_out.logits[0, -1, :]).item())
    ref_token = tokenizer.decode([ref_top_id])

    # Write the compressed cache to disk and hand it to a separate process.
    worker = os.path.join(os.path.dirname(__file__), "_transfer_worker.py")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".kvzip") as f:
        f.write(compressed.to_bytes())
        cache_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, worker, cache_path, str(context_ids.shape[-1]), question],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
    finally:
        os.unlink(cache_path)

    data = json.loads(result.stdout)
    assert data["top_tokens"][0].strip() == ref_token.strip()
    assert "Paris" in data["top_tokens"][0]
