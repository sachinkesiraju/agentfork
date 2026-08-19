"""A small sink-attention-style gate that predicts per-token KV importance."""

from __future__ import annotations

import torch
import torch.nn as nn


class KVCacheGate(nn.Module):
    """Fast, once-trained gate that emits per-token, per-head importance.

    The gate is the ``student`` in the FastKVzip design pattern: at serving
    time it looks at the hidden state and decides which KV blocks are worth
    keeping.  It can be conditioned on a task/LoRA embedding so the same
    compressed cache is safe when different task adapters are active.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_kv_heads: int,
        n_layers: int,
        cond_dim: int = 0,
        hidden: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_kv_heads = n_kv_heads
        self.n_layers = n_layers
        self.cond_dim = cond_dim

        in_dim = hidden_dim + cond_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, n_kv_heads),
        )

        # Per-layer scaling lets a single shared MLP specialize per layer.
        self.layer_scale = nn.Parameter(torch.ones(n_layers))

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-token per-head importance scores.

        Args:
            hidden_states: ``[batch, seq_len, hidden_dim]`` or
                ``[..., hidden_dim]``.
            layer_idx: which transformer layer these hidden states belong to.
            condition: optional ``[batch, cond_dim]`` task/LoRA embedding,
                broadcast across the sequence.

        Returns:
            ``[..., n_kv_heads]`` tensor of sigmoid importance scores.
        """
        x = hidden_states
        if condition is not None:
            if condition.dim() == 2:
                seq_len = hidden_states.shape[-2]
                condition = condition.unsqueeze(-2).expand(-1, seq_len, -1)
            x = torch.cat([x, condition], dim=-1)
        logits = self.mlp(x)
        scale = self.layer_scale[layer_idx]
        return torch.sigmoid(logits * scale)

    def condition_vector(self, task_id: str, adapter_id: str = "") -> torch.Tensor:
        """Return a deterministic but learned-ish conditioning vector.

        v1 uses a simple hash embedding.  In a real deployment this would be
        a LoRA/adapter embedding table loaded from the router.
        """
        # Deterministic vector based on string hash; in v1 this is a placeholder.
        import hashlib

        seed = hashlib.sha256(f"{task_id}:{adapter_id}".encode()).hexdigest()[:8]
        seed_int = int(seed, 16)
        g = torch.Generator().manual_seed(seed_int)
        vec = (
            torch.randn(self.cond_dim, generator=g) if self.cond_dim else torch.empty(0)
        )
        return vec.to(next(self.parameters()).device)
