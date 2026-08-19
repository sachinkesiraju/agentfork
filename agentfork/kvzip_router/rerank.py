"""Query-aware decode rerank over a compressed, query-agnostic KV cache."""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenRelevanceRerank(nn.Module):
    """Cheap query-aware token selector for the decode step.

    v1 uses a learned bilinear score between the query hidden state and the
    hidden states of retained tokens.  In a real backend this would instead
    compute QK attention over retained keys using the model's ``q_proj`` and
    ``k_proj`` weights; the API (``rerank``) stays the same.
    """

    def __init__(self, hidden_dim: int, k_ratio: float = 0.5) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_ratio = k_ratio
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        query_hidden: torch.Tensor,
        token_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-token relevance logits.

        Args:
            query_hidden: ``[batch, 1, hidden_dim]``.
            token_hidden: ``[batch, seq_len, hidden_dim]``.

        Returns:
            ``[batch, seq_len]`` logits.
        """
        batch, seq_len, _ = token_hidden.shape
        q = query_hidden.expand(-1, seq_len, -1)
        pair = torch.cat([q, token_hidden], dim=-1)
        return self.scorer(pair).squeeze(-1)

    def rerank(
        self,
        query_hidden: torch.Tensor,
        token_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Return the indices of the top-k retained tokens for this query."""
        batch, seq_len, _ = token_hidden.shape
        logits = self.forward(query_hidden, token_hidden)
        k = max(1, int(self.k_ratio * seq_len))
        _, topk = torch.topk(logits, k, dim=-1, largest=True, sorted=False)
        return topk.sort(dim=-1).values
