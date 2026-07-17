"""Workload-shape census: does the fork primitive's sweet spot exist?

Analyzes a list of agent sessions (id, created, title, prompt) for:
  1. creation-burst fanout width (sessions created within ``burst_window_s``);
  2. shared-prefix fraction ``f`` between sibling prompts within a burst
     (exact leading common prefix, char-level proxy for token prefix);
  3. recurring identical prompts (cross-wake prefix reuse candidates).

The evaluation threshold is fanout >= ~8 and f >= 0.2 (see
report/RESULTS.md). This module makes those two numbers measurable on any
session export.

Input JSON: [{"id": ..., "created": unix_ts, "title": ..., "prompt": ...}]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def cluster_bursts(sessions: list[dict], burst_window_s: int = 120) -> list[list[dict]]:
    if burst_window_s < 0:
        raise ValueError("burst_window_s must be nonnegative")
    ordered = sorted(sessions, key=lambda s: s["created"])
    clusters: list[list[dict]] = []
    for s in ordered:
        if clusters and s["created"] - clusters[-1][-1]["created"] <= burst_window_s:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    return clusters


def analyze(sessions: list[dict], burst_window_s: int = 120) -> dict:
    clusters = cluster_bursts(sessions, burst_window_s)
    if not clusters:
        return {
            "sessions": 0,
            "clusters": 0,
            "width_histogram": {},
            "fanout_p95": 0,
            "fanout_max": 0,
            "bursts": [],
            "recurring_identical_prompt_sessions": 0,
        }
    widths = Counter(len(c) for c in clusters)
    burst_stats = []
    for c in (c for c in clusters if len(c) >= 2):
        prompts = [s.get("prompt") or "" for s in c]
        base = prompts[0]
        if not base:
            continue
        prefs = [common_prefix_len(base, p) / max(len(base), 1) for p in prompts[1:]]
        burst_stats.append({
            "width": len(c),
            "titles": [(s.get("title") or "")[:60] for s in c],
            "shared_prefix_fraction_f": round(sum(prefs) / len(prefs), 3),
        })
    prompt_counts = Counter(s.get("prompt") or "" for s in sessions if s.get("prompt"))
    recurring = sum(n for p, n in prompt_counts.items() if n >= 2)
    ws = sorted(w for c in clusters for w in [len(c)])
    return {
        "sessions": len(sessions),
        "clusters": len(clusters),
        "width_histogram": dict(sorted(widths.items())),
        "fanout_p95": ws[int(0.95 * (len(ws) - 1))],
        "fanout_max": max(ws),
        "bursts": burst_stats,
        "recurring_identical_prompt_sessions": recurring,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions")
    parser.add_argument("--burst-window", type=int, default=120)
    args = parser.parse_args()
    with open(args.sessions) as f:
        sessions = json.load(f)
    print(json.dumps(analyze(sessions, args.burst_window), indent=2))
