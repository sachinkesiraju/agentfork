"""Budget allocation: turn importance scores into per-head, per-layer KV budgets."""

from __future__ import annotations

import math

import torch


def dedupe_retained_indices(indices: torch.Tensor) -> torch.Tensor:
    """Return sorted, deduplicated retained indices per batch row.

    Pads shorter rows by repeating the row's last valid index so the result
    remains a dense tensor.  The padding entry is harmless because it points
    to an already-retained position.
    """
    batch, k = indices.shape
    if k == 0:
        return indices

    sorted_idx = torch.sort(indices, dim=-1).values
    prev = torch.cat([sorted_idx[..., :1], sorted_idx[..., :-1]], dim=-1)
    mask = torch.cat(
        [
            torch.ones(batch, 1, dtype=torch.bool, device=indices.device),
            sorted_idx[..., 1:] != prev[..., 1:],
        ],
        dim=-1,
    )
    lengths = mask.sum(dim=-1)
    max_len = int(lengths.max().item())
    if max_len == 0:
        return torch.zeros(batch, 0, dtype=indices.dtype, device=indices.device)

    padded = torch.zeros(batch, max_len, dtype=indices.dtype, device=indices.device)
    for i in range(batch):
        row = sorted_idx[i][mask[i]]
        if row.numel():
            n = row.numel()
            padded[i, :n] = row
            if n < max_len:
                padded[i, n:] = row[-1]
    return padded


class BudgetAllocator:
    """Allocate a total token budget across layers and heads.

    v1 supports two strategies:

    * ``uniform`` -- every (layer, head) gets the same fixed budget.
    * ``waterfill`` -- importance scores determine a per-head threshold so
      that the total number of retained tokens matches the budget.  This is a
      one-sided water-fill over per-token scores.

    The allocator returns both the per-head target counts and a *global*
    retained-index set.  v1 uses the global set because Hugging Face
    ``past_key_values`` requires a uniform sequence length across all layers.
    A paged-attention backend will instead consume the per-head per-layer
    budgets directly.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        min_per_head: int = 1,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.min_per_head = min_per_head

    def allocate(
        self,
        scores: list[torch.Tensor],
        target_ratio: float | None = None,
        target_tokens: int | None = None,
        seq_len: int | None = None,
    ) -> dict:
        """Return per-head/layer budgets and a global retained-index set.

        Args:
            scores: list of ``[batch, seq_len, num_kv_heads]`` importance
                tensors, one per layer.
            target_ratio: fraction of total tokens to retain globally.
            target_tokens: exact total retained-token budget.  Overrides
                ``target_ratio`` if both are given.
            seq_len: original sequence length; inferred from ``scores`` if
                omitted.

        Returns:
            A dict with keys ``per_head`` (``[L, H]`` ints), ``global_indices``
            (``[batch, k]`` sorted long tensor), and ``summary`` text.
        """
        if not scores:
            raise ValueError("scores must be non-empty")
        batch, s, h = scores[0].shape
        n_layers = len(scores)
        if seq_len is None:
            seq_len = s
        for t in scores:
            if t.shape != (batch, seq_len, h):
                raise ValueError("all score tensors must have the same shape")

        if target_tokens is None:
            if target_ratio is None:
                raise ValueError("provide target_ratio or target_tokens")
            target_tokens = max(1, math.ceil(target_ratio * seq_len))

        # One water-fill threshold across all layer/head/token scores.
        flat = torch.stack(scores, dim=0).view(-1)
        k = min(target_tokens, flat.numel())
        if k <= 0:
            raise ValueError("budget must be positive")
        threshold = torch.kthvalue(flat, flat.numel() - k + 1).values.item()

        per_head = torch.zeros(n_layers, h, dtype=torch.long)
        per_layer_counts = []
        for li, layer_scores in enumerate(scores):
            # layer_scores: [batch, seq, heads]
            head_mask = layer_scores >= threshold
            per_head_counts = head_mask.sum(dim=(0, 1))  # [heads]
            per_head[li] = per_head_counts
            per_layer_counts.append(per_head_counts.max().item())

        # For v1 global pruning, take the union of positions whose max score
        # across any layer/head is above the threshold.  This keeps the
        # sequence length uniform across layers so standard HF caching works.
        max_scores = torch.stack(scores, dim=0).max(dim=0).values  # [B, S, H]
        token_scores = max_scores.max(dim=-1).values  # [B, S]

        # Keep at least min_tokens positions; if the budget is tiny, keep the
        # highest-scoring min_tokens tokens.
        min_tokens = max(1, int(0.05 * seq_len))
        k = max(min_tokens, min(target_tokens, seq_len))

        # Always retain sink tokens: first and last few positions.
        _, topk_idx = torch.topk(token_scores, k, dim=-1, largest=True, sorted=False)
        sink = torch.tensor(
            [0, 1, seq_len - 2, seq_len - 1] if seq_len > 4 else list(range(seq_len)),
            dtype=torch.long,
            device=topk_idx.device,
        )
        global_idx = torch.cat([topk_idx, sink.unsqueeze(0).expand(batch, -1)], dim=-1)
        global_idx = dedupe_retained_indices(global_idx)

        summary = (
            f"target={target_tokens}, retained={global_idx.shape[-1]}, "
            f"per_head_avg={per_head.float().mean():.2f}"
        )

        return {
            "per_head": per_head,
            "per_layer_max": per_layer_counts,
            "threshold": threshold,
            "global_indices": global_idx,
            "summary": summary,
        }
