"""End-to-end integration of kvzip_router with agentfork ForkOrchestrator."""

import pytest
from agentfork import ForkOrchestrator, NullSandbox

from agentfork.kvzip_router import KVBudgetHint, KVCacheGate, KVCompressor, KVZipBackend


@pytest.fixture
def make_backend(tiny_model):
    """Factory returning a fresh KVZipBackend around the tiny model."""

    def _make(budget: KVBudgetHint | None = None, n_sink_tokens: int = 2):
        gate = KVCacheGate(
            hidden_dim=tiny_model.config.hidden_size,
            n_kv_heads=tiny_model.config.num_key_value_heads,
            n_layers=tiny_model.config.num_hidden_layers,
            cond_dim=0,
        )
        compressor = KVCompressor(tiny_model, gate=gate, n_sink_tokens=n_sink_tokens)
        return KVZipBackend(
            compressor, default_budget=budget or KVBudgetHint(target_ratio=0.5)
        )

    return _make


def _tokens(tok, text):
    return tok(text, return_tensors="pt").input_ids[0].tolist()


def _bytes_per_token(cfg):
    """Full-cache bytes for a single token, including all layers and k+v."""
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    return cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * head_dim * 4


def test_backend_implements_kv_backend(make_backend):
    """The adapter satisfies the lifecycle surface agentfork expects."""
    backend = make_backend()
    backend.create_tree("root")
    assert backend.has_tree("root")
    backend.extend("root", [1, 2, 3, 4])
    backend.fork_branch("root", "child")
    assert backend.has_tree("child")
    freed = backend.kill("child")
    assert freed == 4
    assert not backend.has_tree("child")


def test_fork_independent_generation(make_backend, tiny_tokenizer):
    """Children inherit a compressed prefix and then diverge."""
    backend = make_backend(KVBudgetHint(target_ratio=0.75))
    orch = ForkOrchestrator(kv=backend, sandbox=NullSandbox())

    shared = _tokens(tiny_tokenizer, "the quick brown fox jumps")
    orch.create_parent("root", tokens=shared)
    children = orch.fork("root", n=2)

    out_a = orch.generate(
        children[0].branch_id,
        _tokens(tiny_tokenizer, " over"),
        {"max_new_tokens": 2, "temperature": 1.0},
    )
    out_b = orch.generate(
        children[1].branch_id,
        _tokens(tiny_tokenizer, " under"),
        {"max_new_tokens": 2, "temperature": 1.0},
    )

    assert len(out_a["token_ids"]) == 2
    assert len(out_b["token_ids"]) == 2
    assert all(isinstance(t, int) for t in out_a["token_ids"])
    assert all(isinstance(t, int) for t in out_b["token_ids"])


def test_forked_branches_share_compressed_prefix(make_backend, tiny_tokenizer):
    """Total KV memory across N children is bounded by compressed prefix size,
    not by N independent full prefills."""
    backend = make_backend(KVBudgetHint(target_ratio=0.25), n_sink_tokens=2)
    orch = ForkOrchestrator(kv=backend, sandbox=NullSandbox())

    shared = _tokens(
        tiny_tokenizer,
        "the quick brown fox jumps over the lazy dog and runs away quickly",
    )
    n = 4
    orch.create_parent("root", tokens=shared)
    children = orch.fork("root", n=n)

    for child in children:
        orch.extend(child.branch_id, _tokens(tiny_tokenizer, " and"))

    total = backend.total_kv_bytes()
    cfg = backend.compressor.model.config
    bytes_per_token = _bytes_per_token(cfg)
    full_branch_len = len(shared) + 1
    uncompressed_total = (1 + n) * full_branch_len * bytes_per_token
    # The compressed total must be strictly smaller than a naive full-cache fan-out.
    assert total < uncompressed_total
