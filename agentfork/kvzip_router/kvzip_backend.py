"""Agentfork KVBackend adapter for ``agentfork.kvzip_router``.

Turns :class:`KVCompressor` into a ``KVBackend`` that
:class:`agentfork.orchestrator.ForkOrchestrator` can drive, making compressed,
router-programmable KV caches a first-class branch resource.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from transformers import DynamicCache

from .budget import dedupe_retained_indices
from .compressor import CompressedKVCache, KVCompressor
from .router import KVBudgetHint


@dataclasses.dataclass
class _BranchState:
    """Mutable branch state kept by KVZipBackend."""

    token_ids: list[int]
    cache: CompressedKVCache | None = None
    last_logits: torch.Tensor | None = None
    next_pos: int = 0
    budget: KVBudgetHint | None = None


class KVZipBackend:
    """KVBackend that compresses prefills with a learned gate and budget.

    Each branch owns a :class:`CompressedKVCache`. Forking deep-copies the
    compressed cache so children start from the same prefix; extensions append
    tokens teacher-forced through the model. Generation samples from the last
    logits produced by the most recent ``extend``/``generate``.

    Parameters
    ----------
    compressor:
        Configured :class:`KVCompressor` (model + gate + allocator).
    default_budget:
        Default :class:`KVBudgetHint` used for every branch that does not have
        a per-branch budget set explicitly via :meth:`set_budget`.
    """

    def __init__(
        self,
        compressor: KVCompressor,
        default_budget: KVBudgetHint | None = None,
    ) -> None:
        self.compressor = compressor
        self.default_budget = default_budget or KVBudgetHint(
            target_ratio=0.5, quality_floor="standard"
        )
        self._branches: dict[str, _BranchState] = {}
        self._device = next(compressor.model.parameters()).device

    # ------------------------------------------------------------------ #
    # KVBackend surface
    # ------------------------------------------------------------------ #

    def create_tree(self, tree_id: str) -> str:
        if tree_id in self._branches:
            raise ValueError(f"tree exists: {tree_id}")
        self._branches[tree_id] = _BranchState(token_ids=[], budget=self.default_budget)
        return tree_id

    def has_tree(self, tree_id: str) -> bool:
        return tree_id in self._branches

    def fork_branch(self, parent_id: str, child_id: str | None = None) -> str:
        if parent_id not in self._branches:
            raise KeyError(f"no such tree: {parent_id}")
        child_id = child_id or f"{parent_id}/kvzip-fork"
        if child_id in self._branches:
            raise ValueError(f"tree exists: {child_id}")
        parent = self._branches[parent_id]
        self._branches[child_id] = _BranchState(
            token_ids=list(parent.token_ids),
            cache=self._clone_cache(parent.cache),
            last_logits=self._clone_logits(parent.last_logits),
            next_pos=parent.next_pos,
            budget=parent.budget,
        )
        return child_id

    def kill(self, tree_id: str) -> int:
        state = self._branches.pop(tree_id, None)
        if state is None:
            return 0
        return len(state.token_ids)

    def extend(self, tree_id: str, tokens: list[int]) -> int:
        if not tokens:
            return 0
        state = self._branches[tree_id]
        budget = state.budget or self.default_budget
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self._device)

        if state.cache is None:
            state.cache, state.last_logits = self._prefill_and_logit(input_ids, budget)
            state.token_ids = list(tokens)
            state.next_pos = len(tokens)
            return len(tokens)

        for tok in tokens:
            ids = torch.tensor([[tok]], dtype=torch.long, device=self._device)
            pos = torch.tensor(
                [[state.next_pos]], dtype=torch.long, device=self._device
            )
            logits, state.cache = self.compressor.decode_step(
                state.cache, ids, position_ids=pos
            )
            state.last_logits = logits
            state.token_ids.append(tok)
            state.next_pos += 1
        return len(tokens)

    def generate(
        self,
        branch_id: str,
        prompt: str | list[int],
        sampling_params: dict[str, Any],
        *,
        branch_end: bool = False,
        reserve_tokens: int | None = None,
    ) -> dict[str, Any]:
        if isinstance(prompt, str):
            raise TypeError(
                "KVZipBackend.generate currently requires list[int] tokens; "
                "pass token IDs or wire a tokenizer into the backend."
            )
        if prompt:
            self.extend(branch_id, prompt)

        state = self._branches[branch_id]
        if state.cache is None or state.last_logits is None:
            raise RuntimeError(
                "no cache; call extend() or generate() with a prompt first"
            )

        max_new = max(1, int(sampling_params.get("max_new_tokens", 1)))
        temperature = float(sampling_params.get("temperature", 1.0))
        top_k = sampling_params.get("top_k")
        if top_k is not None:
            top_k = int(top_k)

        generated: list[int] = []
        logits = state.last_logits[:, -1, :] / temperature
        for _ in range(max_new):
            next_id = int(self._sample(logits, top_k).item())
            generated.append(next_id)
            state.token_ids.append(next_id)
            state.next_pos += 1
            ids = torch.tensor([[next_id]], dtype=torch.long, device=self._device)
            pos = torch.tensor(
                [[state.next_pos - 1]], dtype=torch.long, device=self._device
            )
            logits, state.cache = self.compressor.decode_step(
                state.cache, ids, position_ids=pos
            )
            logits = logits[:, -1, :] / temperature

        state.last_logits = logits.unsqueeze(1)
        # No tokenizer is attached to the backend, so decoded text is empty.
        return {"token_ids": generated, "text": ""}

    # ------------------------------------------------------------------ #
    # Extra controls
    # ------------------------------------------------------------------ #

    def set_budget(self, tree_id: str, budget: KVBudgetHint) -> None:
        """Assign a per-branch KV budget hint before the next extend/prefill."""
        if tree_id not in self._branches:
            raise KeyError(f"no such tree: {tree_id}")
        self._branches[tree_id].budget = budget

    def branch_kv_bytes(self, tree_id: str) -> int:
        """Current KV memory for a branch (including decoded suffix)."""
        state = self._branches[tree_id]
        if state.cache is None:
            return 0
        return state.cache.kv_memory_bytes

    def total_kv_bytes(self) -> int:
        return sum(
            s.cache.kv_memory_bytes
            for s in self._branches.values()
            if s.cache is not None
        )

    def resident_tokens(self) -> int:
        return sum(len(s.token_ids) for s in self._branches.values())

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _prefill_and_logit(
        self, input_ids: torch.Tensor, budget: KVBudgetHint
    ) -> tuple[CompressedKVCache, torch.Tensor]:
        """Prefill all but the last token, then decode the last to get logits."""
        seq_len = input_ids.shape[-1]
        if seq_len >= 2:
            compressed = self.compressor.prefill(input_ids[:, :-1], budget)
            pos = torch.tensor([[seq_len - 1]], dtype=torch.long, device=self._device)
            logits, updated = self.compressor.decode_step(
                compressed, input_ids[:, -1:], position_ids=pos
            )
            return updated, logits

        # Single-token prompt: run a forward pass to get logits and a full
        # cache, then compress using the gate. This path is not compressed in
        # the same way because there is no "prefix" to prune away, but it is
        # kept correct for completeness.
        with torch.no_grad():
            outputs = self.compressor.model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
        cache = self._compress_from_outputs(outputs, budget)
        return cache, outputs.logits[:, -1:, :]

    def _compress_from_outputs(
        self, outputs: Any, budget: KVBudgetHint
    ) -> CompressedKVCache:
        """Build a CompressedKVCache from a model forward pass.

        Mirrors KVCompressor.prefill but returns the logit-producing cache
        for the single-token edge case.
        """
        hidden_states = outputs.hidden_states
        seq_len = hidden_states[0].shape[1]
        condition = self.compressor._condition(budget, hidden_states[0].device)
        scores: list[torch.Tensor] = []
        for layer_idx in range(self.compressor.n_layers):
            scores.append(
                self.compressor.gate(hidden_states[layer_idx], layer_idx, condition)
            )

        allocation = self.compressor.allocator.allocate(
            scores, target_ratio=self.compressor._target_ratio(budget), seq_len=seq_len
        )
        retained = allocation["global_indices"]
        sinks = self.compressor._sink_indices(seq_len, device=retained.device)
        retained = dedupe_retained_indices(
            torch.cat([retained, sinks.unsqueeze(0).expand(1, -1)], dim=-1)
        )

        compressed_tuples = self.compressor._prune_tuples(
            [(layer.keys, layer.values) for layer in outputs.past_key_values.layers],
            retained,
            1,
        )
        return CompressedKVCache(
            cache=DynamicCache(ddp_cache_data=compressed_tuples),
            retained_indices=retained,
            original_seq_len=seq_len,
            target_ratio=self.compressor._target_ratio(budget),
            per_head_budgets=allocation["per_head"],
        )

    def _clone_cache(self, cache: CompressedKVCache | None) -> CompressedKVCache | None:
        if cache is None:
            return None
        new_tuples = [
            (layer.keys.clone(), layer.values.clone()) for layer in cache.cache.layers
        ]
        return CompressedKVCache(
            cache=DynamicCache(ddp_cache_data=new_tuples),
            retained_indices=cache.retained_indices.clone(),
            original_seq_len=cache.original_seq_len,
            target_ratio=cache.target_ratio,
            retained_hidden=(
                cache.retained_hidden.clone()
                if cache.retained_hidden is not None
                else None
            ),
            per_head_budgets=(
                cache.per_head_budgets.clone()
                if cache.per_head_budgets is not None
                else None
            ),
        )

    def _clone_logits(self, logits: torch.Tensor | None) -> torch.Tensor | None:
        if logits is None:
            return None
        return logits.clone()

    @staticmethod
    def _sample(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits = torch.where(logits < v[..., [-1]], float("-inf"), logits)
        return torch.argmax(logits, dim=-1)
