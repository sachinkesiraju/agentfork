"""Cross-model gate transfer for ladder-tier or small->large model families.

A gate trained on a small/proxy model can warm-start a gate for a larger or
related model.  This module copies the MLP and interpolates the per-layer scale
when the number of layers differs, then lets the new gate be fine-tuned on the
target model's teacher labels.
"""

from __future__ import annotations

import torch

from .gate import KVCacheGate


def warm_start_gate(
    source_gate: KVCacheGate,
    target_hidden_dim: int | None = None,
    target_n_kv_heads: int | None = None,
    target_n_layers: int | None = None,
) -> KVCacheGate:
    """Create a target gate initialized from ``source_gate``.

    The target gate is a fresh ``KVCacheGate`` whose MLP weights are copied from
    the source and whose per-layer scale is interpolated/repeated to match the
    target layer count.  The returned gate is fully trainable, so the target
    model can fine-tune the warm-started importance predictor.

    Args:
        source_gate: gate trained on the source model.
        target_hidden_dim: hidden dimension of the target model.  If ``None``,
            use the source hidden dimension.
        target_n_kv_heads: number of KV heads in the target model.  If ``None``,
            use the source value.
        target_n_layers: number of layers in the target model.  If ``None``,
            use the source value.

    Returns:
        A new ``KVCacheGate`` ready for fine-tuning on the target model.
    """
    hidden_dim = target_hidden_dim or source_gate.hidden_dim
    n_kv_heads = target_n_kv_heads or source_gate.n_kv_heads
    n_layers = target_n_layers or source_gate.n_layers

    target_gate = KVCacheGate(
        hidden_dim=hidden_dim,
        n_kv_heads=n_kv_heads,
        n_layers=n_layers,
        cond_dim=source_gate.cond_dim,
    )

    # If input/output dimensions match, copy the warm MLP weights.  When they
    # differ the MLP is left at its default Xavier init and fine-tuning adapts.
    if hidden_dim == source_gate.hidden_dim and n_kv_heads == source_gate.n_kv_heads:
        target_gate.mlp.load_state_dict(source_gate.mlp.state_dict())

    # Interpolate or repeat the source layer scale to the target layer count.
    src_scale = source_gate.layer_scale.data.detach()
    if n_layers == source_gate.n_layers:
        target_gate.layer_scale.data = src_scale.clone()
    elif n_layers > source_gate.n_layers:
        repeats = n_layers // source_gate.n_layers
        target_gate.layer_scale.data = src_scale.repeat_interleave(repeats).clone()
        if target_gate.layer_scale.shape[0] < n_layers:
            pad = torch.ones(
                n_layers - target_gate.layer_scale.shape[0],
                dtype=src_scale.dtype,
                device=src_scale.device,
            )
            target_gate.layer_scale.data = torch.cat(
                [target_gate.layer_scale.data, pad]
            )
    else:
        # Down-sample: keep the first n_layers source scales.
        target_gate.layer_scale.data = src_scale[:n_layers].clone()

    return target_gate
