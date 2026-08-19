"""Control-plane integration: a router emits a per-request KV compression budget."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVBudgetHint:
    """Per-request compression budget from the control plane.

    ``target_ratio`` is the fraction of the original KV cache the backend
    should try to retain.  ``quality_floor`` is a coarse policy class.
    """

    task_id: str = ""
    adapter_id: str = ""
    quality_floor: str = "standard"  # strict | standard | aggressive
    latency_class: str = "balanced"  # latency | balanced | throughput
    target_ratio: float | None = None
    min_tokens: int = 4
    max_drop: float = 0.03

    def __post_init__(self):
        if self.target_ratio is not None and not 0.0 < self.target_ratio <= 1.0:
            raise ValueError(f"target_ratio must be in (0, 1], got {self.target_ratio}")


class KVBudgetRouter:
    """Map route metadata to a concrete KV compression budget.

    This is the control-plane side of the project: a gateway like modelrouter,
    vLLM Semantic Router, or NeMo Switchyard can call ``hint(...)`` with the
    task/adapter/quality/latency it already knows and receive a budget the
    backend can execute.
    """

    DEFAULT_RATIOS = {
        "strict": 0.60,
        "standard": 0.40,
        "aggressive": 0.20,
    }

    LATENCY_MULTIPLIERS = {
        "latency": 1.20,
        "balanced": 1.00,
        "throughput": 0.80,
    }

    def __init__(
        self,
        task_overrides: dict[str, float] | None = None,
        adapter_overrides: dict[str, float] | None = None,
        defaults: dict[str, float] | None = None,
    ) -> None:
        self.task_overrides = task_overrides or {}
        self.adapter_overrides = adapter_overrides or {}
        self.defaults = defaults or dict(self.DEFAULT_RATIOS)

    def hint(
        self,
        task_id: str = "",
        adapter_id: str = "",
        quality_floor: str = "standard",
        latency_class: str = "balanced",
    ) -> KVBudgetHint:
        if adapter_id and adapter_id in self.adapter_overrides:
            base = self.adapter_overrides[adapter_id]
        elif task_id and task_id in self.task_overrides:
            base = self.task_overrides[task_id]
        else:
            base = self.defaults.get(quality_floor, self.DEFAULT_RATIOS["standard"])
        ratio = base * self.LATENCY_MULTIPLIERS.get(latency_class, 1.0)
        ratio = max(0.05, min(0.95, ratio))
        return KVBudgetHint(
            task_id=task_id,
            adapter_id=adapter_id,
            quality_floor=quality_floor,
            latency_class=latency_class,
            target_ratio=ratio,
        )
