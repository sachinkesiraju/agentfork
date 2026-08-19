"""Prefill-time KV-cache compression and decode-time reuse for Hugging Face models."""

from __future__ import annotations

import dataclasses
import io
from collections.abc import Iterable

import torch
from transformers import DynamicCache

from .budget import BudgetAllocator, dedupe_retained_indices
from .gate import KVCacheGate
from .router import KVBudgetHint


@dataclasses.dataclass
class CompressedKVCache:
    """A pruned, optionally quantized KV cache together with routing metadata."""

    cache: DynamicCache
    retained_indices: torch.Tensor
    original_seq_len: int
    target_ratio: float
    retained_hidden: torch.Tensor | None = None
    per_head_budgets: torch.Tensor | None = None

    @property
    def seq_len(self) -> int:
        return self.cache.get_seq_length()

    @property
    def effective_ratio(self) -> float:
        return self.seq_len / max(1, self.original_seq_len)

    @property
    def kv_memory_bytes(self) -> int:
        total = 0
        for layer in self.cache.layers:
            for tensor in (layer.keys, layer.values):
                total += tensor.numel() * tensor.element_size()
        return total

    def to_bytes(self) -> bytes:
        """Serialize the compressed KV cache and routing metadata to bytes."""
        buffer = io.BytesIO()
        layers = [(layer.keys, layer.values) for layer in self.cache.layers]
        torch.save(
            {
                "layers": layers,
                "retained_indices": self.retained_indices,
                "original_seq_len": self.original_seq_len,
                "target_ratio": self.target_ratio,
                "retained_hidden": self.retained_hidden,
                "per_head_budgets": self.per_head_budgets,
            },
            buffer,
        )
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> CompressedKVCache:
        """Deserialize a compressed cache produced by ``to_bytes``."""
        buffer = io.BytesIO(data)
        payload = torch.load(buffer, weights_only=True)
        cache = _tuples_to_cache(payload["layers"])
        return cls(
            cache=cache,
            retained_indices=payload["retained_indices"],
            original_seq_len=payload["original_seq_len"],
            target_ratio=payload["target_ratio"],
            retained_hidden=payload.get("retained_hidden"),
            per_head_budgets=payload.get("per_head_budgets"),
        )


def _cache_to_tuples(cache: DynamicCache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return a list of (key, value) tensors from a DynamicCache."""
    return [(layer.keys, layer.values) for layer in cache.layers]


def _tuples_to_cache(
    tuples: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> DynamicCache:
    """Build a DynamicCache from a list of (key, value) tensors."""
    return DynamicCache(ddp_cache_data=list(tuples))


class KVCompressor:
    """Compress a model's prefill KV cache using a learned gate and a budget."""

    def __init__(
        self,
        model,
        gate: KVCacheGate | None = None,
        allocator: BudgetAllocator | None = None,
        n_sink_tokens: int = 4,
    ) -> None:
        self.model = model
        self.config = model.config
        self.n_layers = self._n_layers()
        self.n_kv_heads = self._n_kv_heads()
        self.hidden_size = self._hidden_size()
        self.head_dim = self.hidden_size // self._n_attn_heads()

        if gate is None:
            gate = KVCacheGate(
                hidden_dim=self.hidden_size,
                n_kv_heads=self.n_kv_heads,
                n_layers=self.n_layers,
                cond_dim=16,
            )
        self.gate = gate
        self.allocator = allocator or BudgetAllocator(
            num_layers=self.n_layers,
            num_kv_heads=self.n_kv_heads,
        )
        self.n_sink_tokens = n_sink_tokens

    def _n_layers(self) -> int:
        return getattr(
            self.config,
            "num_hidden_layers",
            getattr(self.config, "n_layer", 1),
        )

    def _n_attn_heads(self) -> int:
        return getattr(
            self.config,
            "num_attention_heads",
            getattr(self.config, "n_head", 1),
        )

    def _n_kv_heads(self) -> int:
        return getattr(
            self.config,
            "num_key_value_heads",
            getattr(
                self.config,
                "num_attention_heads",
                getattr(self.config, "n_head", 1),
            ),
        )

    def _hidden_size(self) -> int:
        return getattr(
            self.config,
            "hidden_size",
            getattr(self.config, "n_embd", 1),
        )

    def _target_ratio(self, budget: KVBudgetHint | float) -> float:
        if isinstance(budget, KVBudgetHint):
            return budget.target_ratio if budget.target_ratio is not None else 1.0
        return float(budget)

    def _condition(
        self, hint: KVBudgetHint, device: torch.device
    ) -> torch.Tensor | None:
        if self.gate.cond_dim == 0:
            return None
        c = self.gate.condition_vector(hint.task_id, hint.adapter_id).to(
            device=device, dtype=next(self.gate.parameters()).dtype
        )
        return c.unsqueeze(0)

    def prefill(
        self,
        input_ids: torch.Tensor,
        budget: KVBudgetHint | float,
    ) -> CompressedKVCache:
        """Run prefill, score every token, allocate budgets, and prune the cache."""
        self.model.eval()
        target_ratio = self._target_ratio(budget)
        hint = (
            budget
            if isinstance(budget, KVBudgetHint)
            else KVBudgetHint(target_ratio=target_ratio)
        )
        seq_len = input_ids.shape[-1]

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )

        # hidden_states[0] is the input to layer 0; hidden_states[l] is the
        # input to layer l (post residual from layer l-1).
        hidden_states = outputs.hidden_states
        past = outputs.past_key_values
        if past is None:
            raise RuntimeError("Model did not return a KV cache; set use_cache=True.")

        condition = self._condition(hint, device=hidden_states[0].device)

        scores: list[torch.Tensor] = []
        for layer_idx in range(self.n_layers):
            # hidden_states has length n_layers + 1
            hs = hidden_states[layer_idx]
            layer_scores = self.gate(hs, layer_idx, condition=condition)
            scores.append(layer_scores)

        allocation = self.allocator.allocate(
            scores,
            target_ratio=target_ratio,
            seq_len=seq_len,
        )
        retained_indices = allocation["global_indices"]

        # Retain sink positions too.
        sinks = self._sink_indices(seq_len, device=retained_indices.device)
        retained_indices = dedupe_retained_indices(
            torch.cat(
                [retained_indices, sinks.unsqueeze(0).expand(input_ids.shape[0], -1)],
                dim=-1,
            )
        )

        compressed_tuples = self._prune_tuples(
            _cache_to_tuples(past), retained_indices, input_ids.shape[0]
        )
        compressed_cache = _tuples_to_cache(compressed_tuples)

        # Keep final hidden states of retained tokens for query-aware rerank.
        last_hidden = outputs.hidden_states[-1]
        retained_hidden = torch.gather(
            last_hidden,
            dim=1,
            index=retained_indices.unsqueeze(-1).expand(-1, -1, last_hidden.shape[-1]),
        )

        return CompressedKVCache(
            cache=compressed_cache,
            retained_indices=retained_indices,
            original_seq_len=seq_len,
            target_ratio=target_ratio,
            retained_hidden=retained_hidden,
            per_head_budgets=allocation["per_head"],
        )

    def _sink_indices(self, seq_len: int, device: torch.device) -> torch.Tensor:
        n = min(self.n_sink_tokens, seq_len)
        first = torch.arange(n, dtype=torch.long, device=device)
        last = torch.arange(
            max(0, seq_len - n), seq_len, dtype=torch.long, device=device
        )
        return torch.cat([first, last]).unique(sorted=True)

    def _prune_tuples(
        self,
        tuples: list[tuple[torch.Tensor, torch.Tensor]],
        retained_indices: torch.Tensor,
        batch_size: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Gather retained positions along the sequence dimension of every KV tensor."""
        out = []
        # retained_indices: [batch, k]
        for key, value in tuples:
            # key/value: [batch, n_kv_heads, seq_len, head_dim]
            b, h, s, d = key.shape
            idx = retained_indices.unsqueeze(1).unsqueeze(-1).expand(b, h, -1, d)
            new_key = torch.gather(key, dim=2, index=idx)
            new_value = torch.gather(value, dim=2, index=idx)
            out.append((new_key, new_value))
        return out

    def decode_step(
        self,
        compressed: CompressedKVCache,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, CompressedKVCache]:
        """Run one decode token against a compressed cache and append its KV."""
        if position_ids is None:
            position_ids = torch.tensor(
                [[compressed.original_seq_len]],
                dtype=torch.long,
                device=input_ids.device,
            )
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=compressed.cache,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        # The returned cache is the updated cache including the new token.
        updated = CompressedKVCache(
            cache=outputs.past_key_values,
            retained_indices=compressed.retained_indices,
            original_seq_len=compressed.original_seq_len,
            target_ratio=compressed.target_ratio,
            per_head_budgets=compressed.per_head_budgets,
        )
        return outputs.logits, updated

    def generate(
        self,
        input_ids: torch.Tensor,
        budget: KVBudgetHint | float,
        max_new_tokens: int = 1,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> tuple[list[int], CompressedKVCache]:
        """Prefill + compressed autoregressive generation for ``max_new_tokens``.

        To guarantee the first generated token is also produced through the
        compressed cache, we prefill all prompt tokens except the last one and
        then decode the last prompt token as the first query.
        """
        if input_ids.shape[-1] < 2:
            raise ValueError("input_ids must contain at least 2 tokens")

        prefix_ids = input_ids[:, :-1]
        first_query = input_ids[:, -1:]

        compressed = self.prefill(prefix_ids, budget)
        tokens: list[int] = []
        position = compressed.original_seq_len

        logits, cache_state = self.decode_step(
            compressed,
            first_query,
            position_ids=torch.tensor(
                [[position]], dtype=torch.long, device=input_ids.device
            ),
        )
        logits = logits[:, -1, :] / temperature
        next_id = self._sample(logits, top_k).item()
        tokens.append(next_id)
        position += 1

        for _ in range(max_new_tokens - 1):
            t = torch.tensor(
                [[next_id]], dtype=input_ids.dtype, device=input_ids.device
            )
            pos = torch.tensor([[position]], dtype=torch.long, device=input_ids.device)
            logits, cache_state = self.decode_step(cache_state, t, position_ids=pos)
            logits = logits[:, -1, :] / temperature
            next_id = self._sample(logits, top_k).item()
            tokens.append(next_id)
            position += 1

        return tokens, cache_state

    @staticmethod
    def _sample(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits = torch.where(logits < v[..., [-1]], -float("inf"), logits)
        return torch.argmax(logits, dim=-1)
