"""Production-trace-driven online distillation loop.

A ``TraceJournal`` records production requests and a ``TraceTrainer`` fine-tunes
``KVCacheGate`` from them using ``FullCacheTeacher`` labels.  This closes the
flywheel: router/gateway traces -> teacher labels -> gate updates.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import torch

from .compressor import KVCompressor
from .distill import FullCacheTeacher, OnlineDistiller
from .router import KVBudgetHint


@dataclasses.dataclass
class TraceRecord:
    """One production request that can be turned into a teacher label."""

    input_ids: torch.Tensor
    budget: KVBudgetHint | None = None


class TraceJournal:
    """Ring buffer of production traces ready for offline/online distillation."""

    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = capacity
        self._records: list[TraceRecord] = []

    def add(self, record: TraceRecord) -> None:
        """Append a record; evict oldest if over capacity."""
        self._records.append(record)
        if len(self._records) > self.capacity:
            self._records.pop(0)

    def sample(self, n: int | None = None) -> list[TraceRecord]:
        """Return up to ``n`` records (or all if ``n`` is None)."""
        if n is None or n >= len(self._records):
            return list(self._records)
        return self._records[-n:]

    def __len__(self) -> int:
        return len(self._records)


class TraceTrainer:
    """Fine-tune a ``KVCompressor`` gate from trace records."""

    def __init__(
        self,
        compressor: KVCompressor,
        teacher: FullCacheTeacher | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        lr: float = 2e-3,
    ) -> None:
        self.compressor = compressor
        self.teacher = teacher or FullCacheTeacher(compressor.model)
        self.distiller = OnlineDistiller(compressor.gate, teacher=self.teacher)
        if optimizer is None:
            optimizer = torch.optim.Adam(compressor.gate.parameters(), lr=lr)
        self.optimizer = optimizer

    def train_step(self, record: TraceRecord) -> float:
        """One gradient step on a single trace record."""
        self.compressor.model.eval()
        self.compressor.gate.train()

        budget = record.budget or KVBudgetHint()
        condition = self.compressor._condition(budget, device=record.input_ids.device)

        with torch.no_grad():
            teacher_scores = self.teacher.score(
                record.input_ids,
                n_kv_heads=self.compressor.n_kv_heads,
            )
            # Compute hidden states once; detach so gradients stay in the gate only.
            model_outputs = self.compressor.model(
                input_ids=record.input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = [h.detach() for h in model_outputs.hidden_states]

        loss = self.distiller.distill_loss(
            hidden_states, teacher_scores, condition=condition
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def fit(
        self,
        records: Iterable[TraceRecord],
        epochs: int = 1,
    ) -> list[float]:
        """Train over an iterable of records, repeating ``epochs`` times."""
        records = list(records)
        losses: list[float] = []
        for _ in range(epochs):
            for record in records:
                losses.append(self.train_step(record))
        return losses
