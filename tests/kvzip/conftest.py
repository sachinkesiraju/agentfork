"""Shared test fixtures."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig  # noqa: E402
from agentfork.kvzip_router import (  # noqa: E402
    FullCacheTeacher,
    KVCacheGate,
    KVCompressor,
    OnlineDistiller,
)

_TINY_MODEL = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
_TRAIN_TEXT = "the quick brown fox jumps over the lazy dog and runs away quickly then stops and looks around"


@pytest.fixture(scope="module")
def tiny_tokenizer():
    return AutoTokenizer.from_pretrained(_TINY_MODEL)


@pytest.fixture(scope="module")
def tiny_model():
    """A tiny, randomly initialized Llama model used for CPU-friendly tests."""
    return AutoModelForCausalLM.from_pretrained(
        _TINY_MODEL,
        dtype=torch.float32,
        attn_implementation="eager",
    )


@pytest.fixture(scope="module")
def trained_compressor(tiny_model, tiny_tokenizer):
    """A KVCompressor whose gate is trained to mimic full-cache attention."""
    torch.manual_seed(0)
    gate = KVCacheGate(
        hidden_dim=tiny_model.config.hidden_size,
        n_kv_heads=tiny_model.config.num_key_value_heads,
        n_layers=tiny_model.config.num_hidden_layers,
        cond_dim=0,
    )
    compressor = KVCompressor(tiny_model, gate=gate)
    train_ids = tiny_tokenizer(_TRAIN_TEXT, return_tensors="pt").input_ids

    teacher = FullCacheTeacher(tiny_model)
    with torch.no_grad():
        teacher_scores = teacher.score(train_ids, n_kv_heads=compressor.n_kv_heads)
        outputs = tiny_model(
            input_ids=train_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    opt = torch.optim.Adam(gate.parameters(), lr=2e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = OnlineDistiller(gate).distill_loss(outputs.hidden_states, teacher_scores)
        loss.backward()
        opt.step()

    return compressor


@pytest.fixture(scope="function")
def vllm_tiny_model_path(tmp_path):
    """A tiny Llama model saved to disk that vLLM can load."""
    # head_dim = hidden_size / num_attention_heads = 64, which satisfies
    # FlashAttention's head-size constraints on both CPU and GPU.
    cfg = LlamaConfig(
        vocab_size=32000,
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-06,
        tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    path = tmp_path / "tiny-llama-128"
    path.mkdir()
    model.save_pretrained(path)
    tok = AutoTokenizer.from_pretrained(_TINY_MODEL)
    tok.save_pretrained(path)
    return str(path)
