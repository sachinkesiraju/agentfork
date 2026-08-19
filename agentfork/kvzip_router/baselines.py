"""Uncompressed and naive compression baselines for value comparisons."""

from __future__ import annotations

import torch
from transformers import DynamicCache

from .compressor import CompressedKVCache


def uniform_compress(
    past: DynamicCache,
    target_ratio: float,
    n_sink_tokens: int = 4,
    original_seq_len: int | None = None,
) -> CompressedKVCache:
    """Drop tokens uniformly at random, retaining sinks, for an ablation.

    This is the simplest baseline for "does the learned gate actually help?"
    It preserves the first/last ``n_sink_tokens`` positions and then randomly
    selects from the middle until the target ratio is reached.
    """
    seq_len = past.get_seq_length()
    original_seq_len = original_seq_len or seq_len
    k = max(1, min(seq_len, int(round(target_ratio * original_seq_len))))

    first = list(range(min(n_sink_tokens, seq_len)))
    last = list(range(max(0, seq_len - n_sink_tokens), seq_len))
    middle = list(range(n_sink_tokens, max(n_sink_tokens, seq_len - n_sink_tokens)))

    need = max(0, k - len(first) - len(last))
    if need > 0 and middle:
        step = max(1, len(middle) // need)
        chosen = middle[::step][:need]
    else:
        chosen = []

    retained = sorted(set(first + chosen + last))
    retained_t = torch.tensor(
        [retained], dtype=torch.long, device=past.layers[0].keys.device
    )

    compressed_tuples = []
    for layer in past.layers:
        idx = (
            retained_t.unsqueeze(1)
            .unsqueeze(-1)
            .expand(1, layer.keys.shape[1], -1, layer.keys.shape[-1])
        )
        new_k = torch.gather(layer.keys, dim=2, index=idx)
        new_v = torch.gather(layer.values, dim=2, index=idx)
        compressed_tuples.append((new_k, new_v))

    return CompressedKVCache(
        cache=DynamicCache(ddp_cache_data=compressed_tuples),
        retained_indices=retained_t,
        original_seq_len=original_seq_len,
        target_ratio=target_ratio,
    )


def full_cache_reference(past: DynamicCache) -> CompressedKVCache:
    """Wrap an uncompressed DynamicCache as a CompressedKVCache."""
    return CompressedKVCache(
        cache=past,
        retained_indices=torch.arange(
            past.get_seq_length(),
            dtype=torch.long,
            device=past.layers[0].keys.device,
        ).unsqueeze(0),
        original_seq_len=past.get_seq_length(),
        target_ratio=1.0,
    )
