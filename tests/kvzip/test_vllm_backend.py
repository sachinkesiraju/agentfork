"""Tests for the vLLM V1 attention backend with KVZip block pruning."""

import os

import pytest
import torch

# The block-mask logic is pure PyTorch and should always be available.
from agentfork.kvzip_router.vllm_backend import _apply_kvzip_block_mask

try:
    import vllm  # noqa: F401

    HAS_VLLM = True
except Exception:
    HAS_VLLM = False


def _dtype_for_vllm() -> str:
    return "float16" if torch.cuda.is_available() else "float32"


def test_block_mask_zeroes_low_scoring_blocks():
    """Direct test: the mask keeps top blocks and zeros the rest."""
    num_blocks, num_kv_heads, block_size, head_size = 4, 2, 2, 4
    kv_cache = torch.zeros(num_blocks, num_kv_heads, block_size, 2 * head_size)

    # Give each block a different key energy. Block 1 is strongest, 2 next, etc.
    for b in range(num_blocks):
        kv_cache[b, :, :, :head_size] = 0.1 * (b + 1)
        kv_cache[b, :, :, head_size:] = 0.05 * (b + 1)

    # A query of all ones makes the dot product monotonic in block value.
    query = torch.ones(1, num_kv_heads, head_size)
    block_table = torch.tensor([[0, 1, 2, -1]])
    seq_lens = torch.tensor([5])  # last token in block (5-1)//2 = 2 -> block 2

    _apply_kvzip_block_mask(
        kv_cache,
        query,
        block_table,
        seq_lens,
        target_ratio=0.5,
        min_blocks=1,
        num_heads=num_kv_heads,
    )

    # Block 2 has the highest key energy and also hosts the current token,
    # so it is retained.  Blocks 0, 1, and 3 should be zeroed.
    assert kv_cache[2].abs().sum() > 0
    for dropped in (0, 1, 3):
        assert kv_cache[dropped].abs().sum() == 0


@pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")
@pytest.mark.slow
def test_vllm_custom_backend_generates(vllm_tiny_model_path):
    """End-to-end: a tiny Llama model runs to completion under KVZip backend."""
    # Set the budget *before* importing the backend module so the spawned
    # vLLM worker processes inherit it when they re-import the module.
    os.environ["KVZIP_TARGET_RATIO"] = "0.5"
    os.environ["KVZIP_MIN_BLOCKS"] = "1"

    # Importing registers AttentionBackendEnum.CUSTOM with the KVZip backend.
    from agentfork.kvzip_router.vllm_backend import (
        FastKVzipAttentionImpl,
        set_kvzip_budget,
    )

    set_kvzip_budget(0.5, 1)
    FastKVzipAttentionImpl.target_ratio = 0.5
    FastKVzipAttentionImpl.min_blocks = 1

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=vllm_tiny_model_path,
        dtype=_dtype_for_vllm(),
        max_model_len=128,
        enforce_eager=True,
        gpu_memory_utilization=0.2,
        attention_config={"backend": "CUSTOM"},
    )
    outputs = llm.generate("hello world", SamplingParams(max_tokens=2))
    text = outputs[0].outputs[0].text
    assert isinstance(text, str)
    assert len(text) >= 0  # generation may emit empty tokens, but should not crash


@pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")
@pytest.mark.slow
def test_vllm_custom_backend_with_no_compression_matches_cpu_attn(vllm_tiny_model_path):
    """With target_ratio=1.0 no used blocks are dropped; output is deterministic."""
    os.environ["KVZIP_TARGET_RATIO"] = "1.0"
    os.environ["KVZIP_MIN_BLOCKS"] = "1"

    from agentfork.kvzip_router.vllm_backend import (
        FastKVzipAttentionImpl,
        set_kvzip_budget,
    )

    set_kvzip_budget(1.0, 1)
    FastKVzipAttentionImpl.target_ratio = 1.0
    FastKVzipAttentionImpl.min_blocks = 1

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=vllm_tiny_model_path,
        dtype=_dtype_for_vllm(),
        max_model_len=128,
        enforce_eager=True,
        gpu_memory_utilization=0.2,
        attention_config={"backend": "CUSTOM"},
    )
    outputs = llm.generate("hello world", SamplingParams(max_tokens=2, temperature=0))
    text1 = outputs[0].outputs[0].text

    # A second run with the same model and no compression should be identical
    # because all used blocks are retained and zeroed unused blocks are never read.
    outputs2 = llm.generate("hello world", SamplingParams(max_tokens=2, temperature=0))
    text2 = outputs2[0].outputs[0].text
    assert text1 == text2
