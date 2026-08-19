"""Online distillation from production traces into the FastKVzip-style gate.

The gate is the *student*: it predicts KV importance in one forward pass.  The
*teacher* is the expensive but accurate reconstruction pass (KVZip) or the
full-cache attention distribution.  ``OnlineDistiller`` samples router traces,
builds teacher labels offline, and fine-tunes the gate so it tracks the
real task/adapter mix.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class FullCacheTeacher:
    """Teacher that labels each token by its total received attention.

    v1 uses the attention weights from a normal full-cache forward pass.  This
    is still expensive (quadratic), but it moves the cost offline and avoids
    the reconstruction pass entirely.
    """

    def __init__(self, model) -> None:
        self.model = model

    def score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        n_kv_heads: int | None = None,
    ) -> list[torch.Tensor]:
        """Return per-layer per-head token importance.

        Returns:
            A list of ``[batch, seq_len, n_kv_heads]`` score tensors, one per
            layer.
        """
        with torch.no_grad():
            out = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        attentions = out.attentions  # tuple of [B, n_heads, q_len, k_len]
        if not attentions:
            raise RuntimeError(
                "Teacher got no attention weights; load the model with "
                "attn_implementation='eager' so output_attentions works."
            )
        n_heads = attentions[0].shape[1]
        n_kv_heads = n_kv_heads or n_heads
        # Group query heads into key-value heads if GQA is used.
        group_size = n_heads // n_kv_heads if n_heads >= n_kv_heads else 1
        scores = []
        for attn in attentions:
            # Sum over query positions to get per-key importance.
            per_key = attn.sum(dim=2)  # [B, n_heads, k_len]
            if group_size > 1:
                b, nh, s = per_key.shape
                per_key = per_key.view(b, n_kv_heads, group_size, s).mean(dim=2)
            per_key = per_key.transpose(1, 2)  # [B, k_len, n_kv_heads]
            # Normalize to [0, 1] so the sigmoid gate can match it.
            per_key = per_key - per_key.min(dim=1, keepdim=True).values
            max_vals = per_key.max(dim=1, keepdim=True).values.clamp_min(1e-6)
            per_key = per_key / max_vals
            scores.append(per_key)
        return scores


class OnlineDistiller:
    """Fine-tune a KVCacheGate from (hidden_state, teacher_score) pairs."""

    def __init__(self, gate, teacher: FullCacheTeacher | None = None) -> None:
        self.gate = gate
        self.teacher = teacher

    def distill_loss(
        self,
        hidden_states: list[torch.Tensor],
        teacher_scores: list[torch.Tensor],
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MSE between student gate and teacher target scores."""
        total: torch.Tensor = 0.0  # type: ignore[assignment]
        for layer_idx, (hs, target) in enumerate(
            zip(hidden_states, teacher_scores, strict=False)
        ):
            pred = self.gate(hs, layer_idx, condition=condition)
            # Align lengths (teacher may have one extra prefix token from embed).
            min_len = min(pred.shape[1], target.shape[1])
            pred = pred[:, :min_len, :]
            target = target[:, :min_len, :]
            total = total + F.mse_loss(pred, target)
        return total / len(hidden_states)
