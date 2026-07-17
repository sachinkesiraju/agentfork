"""Fanout cost model: tree-keyed KV fork vs composed baselines.

Baselines compared for an N-way sibling fanout over a shared prefix of
``prefix_tokens`` with ``suffix_tokens`` unique work per child:

  A. independent   — every child re-prefills the full prefix (no caching).
  B. provider      — provider prompt caching: cached-input tokens billed at a
                     discount (e.g. 0.1x); cache writes at write_mult.
  C. self_hosted   — stock SGLang/vLLM prefix caching, same-namespace requests:
                     prefix compute and physical residency are amortized when
                     the engine retains and reuses the identical prefix.
  D. agentfork     — tree-keyed lifecycle controls over the same physical
                     prefix sharing; children pay only their unique suffix.

Compute cost is proxied by prefill token-charges; cache residency by resident
tokens. This is token arithmetic, not a latency, dollar-cost, or byte-level
measurement.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass
class Scenario:
    n_children: int
    prefix_tokens: int
    suffix_tokens: int
    provider_cached_discount: float = 0.1
    provider_write_mult: float = 1.25

    def __post_init__(self) -> None:
        if self.n_children < 1:
            raise ValueError("n_children must be at least 1")
        if self.prefix_tokens < 0 or self.suffix_tokens < 0:
            raise ValueError("token counts must be nonnegative")
        if self.prefix_tokens + self.suffix_tokens == 0:
            raise ValueError("prefix_tokens and suffix_tokens cannot both be zero")
        if self.provider_cached_discount < 0 or self.provider_write_mult < 0:
            raise ValueError("provider price multipliers must be nonnegative")


def model(s: Scenario) -> dict:
    n, p, u = s.n_children, s.prefix_tokens, s.suffix_tokens
    independent = {"prefill_charged": n * (p + u), "resident": n * (p + u)}
    provider = {
        "prefill_charged": p * s.provider_write_mult
        + (n - 1) * p * s.provider_cached_discount + n * u,
        "resident": None,  # opaque, provider-side
    }
    self_hosted = {"prefill_charged": p + n * u, "resident": p + n * u}
    agentfork = {"prefill_charged": p + n * u, "resident": p + n * u}
    out = {
        "scenario": vars(s),
        "independent": independent,
        "provider_cached": provider,
        "self_hosted_radix": self_hosted,
        "agentfork": agentfork,
    }
    af = agentfork["prefill_charged"]
    out["compute_gain_vs_independent"] = round(independent["prefill_charged"] / af, 2)
    out["compute_gain_vs_provider"] = round(provider["prefill_charged"] / af, 2)
    out["compute_gain_vs_self_hosted"] = round(self_hosted["prefill_charged"] / af, 2)
    out["cache_residency_gain_vs_self_hosted"] = round(
        self_hosted["resident"] / agentfork["resident"], 2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--children", type=int, default=10)
    ap.add_argument("--prefix", type=int, default=32000)
    ap.add_argument("--suffix", type=int, default=2000)
    a = ap.parse_args()
    print(json.dumps(model(Scenario(a.children, a.prefix, a.suffix)), indent=2))
