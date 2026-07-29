"""End-to-end test of the split-screen race demo (demo/race_demo.py).

The demo is the claim; this test is the check. It runs the real thing --
a real tree-cache HTTP server per arm, the real ``TreeRadixCache`` and KV
pool, real sandboxes, real ``pytest`` candidate checks and a real noisy
neighbour -- at small N/P/U so it fits in a unit-test budget, and asserts the
demo's machine-readable summary says what the demo claims:

* under pressure above the break-even (``U > C - P``) the agentfork arm hits
  the shared prefix on every child and the stock arm hits it on none;
* at or below the break-even (``U <= C - P``) both arms hit on every child --
  the boundary is a property of the pressure, not of the demo's framing;
* killing the losers reclaims their KV, leaving exactly the winner's own
  committed prefix pinned;
* both arms end with a verified winning fix.

Skipped unless the patched SGLang checkout is importable (the demo needs the
real cache); see tools/setup_sglang.sh.
"""

import json
import subprocess
import sys

import pytest

pytest.importorskip(
    "sglang.srt.mem_cache.tree_radix_cache",
    reason="needs the patched SGLang checkout (tools/setup_sglang.sh) on "
           "PYTHONPATH")

race_demo = pytest.importorskip("demo.race_demo")


def run_demo(**overrides) -> dict:
    """Run the demo in --no-ui mode in a subprocess and parse its JSON."""
    args = {
        "--children": "3",
        "--verify-children": "2",
        "--prefix-tokens": "1024",
        "--suffix-tokens": "128",
        "--capacity-tokens": "2560",
        "--noise-request-tokens": "256",
    }
    args.update(overrides)
    argv = [sys.executable, race_demo.__file__, "--no-ui"]
    for key, value in args.items():
        argv += [key, str(value)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    body = proc.stdout.split(race_demo.JSON_BEGIN, 1)[1]
    return json.loads(body.split(race_demo.JSON_END, 1)[0])


@pytest.fixture(scope="module")
def above() -> dict:
    """Pressure above the break-even: C - P = 1536, and the demo's default
    neighbour rate (headroom // request + 1 = 7 requests of 256) puts
    U = 1792 strictly above it."""
    return run_demo()


@pytest.fixture(scope="module")
def below() -> dict:
    """Pressure at or below the break-even: one small neighbour request per
    gap, well inside the cache headroom."""
    return run_demo(**{"--noise-requests-per-gap": "1"})


def test_pressure_is_above_break_even(above):
    cfg = above["config"]
    assert cfg["pressure_regime"] == "above"
    assert cfg["interleaved_tokens_u"] > cfg["break_even_u"]


def test_agentfork_hits_the_shared_prefix_on_every_child(above):
    arm = above["arms"]["agentfork"]
    assert arm["parent_hit_rate"] == 1.0
    assert all(child["parent_hit"] for child in arm["children"])
    assert all(child["parent_hit"] for child in arm["verify_children"])


def test_stock_loses_the_shared_prefix_under_pressure(above):
    arm = above["arms"]["stock"]
    assert arm["parent_hit_rate"] == 0.0
    assert not any(child["parent_hit"] for child in arm["children"])
    # ... and pays for it in real prefill charges from the real cache
    assert (above["claims"]["stock_prefill_charged"]
            > above["claims"]["agentfork_prefill_charged"])


def test_stock_matches_the_cost_model_hit_rate(above):
    """The measured stock hit rate is what the pressure model predicts."""
    assert (above["arms"]["stock"]["parent_hit_rate"]
            == above["pressure_model"]["stock"]["parent_hit_rate"])


def test_both_arms_produce_a_verified_winner(above):
    for name in ("stock", "agentfork"):
        arm = above["arms"][name]
        assert arm["verified"], name
        assert arm["winner"], name
        # exactly one candidate survives the cheap check in round 1
        assert sum(c["check_passed"] for c in arm["children"]) == 1, name


def test_killing_losers_reclaims_their_kv(above):
    claims = above["claims"]
    arm = above["arms"]["agentfork"]
    losers = [c for c in arm["children"]
              if c["branch_id"] != arm["round1_winner"]]
    # the losers' own tokens come back (at least the largest loser's worth;
    # siblings share whatever leading suffix bytes they have in common, so the
    # reclaim is the union of their unique nodes, not the sum of their
    # charges) ...
    assert claims["agentfork_kv_freed_by_kills"] >= max(
        c["charged_tokens"] for c in losers)
    # ... and only theirs: the shared prefix they forked from is still pinned
    # by the winner, so the reclaim never reaches into it
    assert claims["agentfork_kv_freed_by_kills"] < above["config"][
        "prefix_tokens"]
    # what is left pinned is exactly the winner's own committed prefix
    assert (claims["agentfork_pinned_after_kills"]
            == claims["agentfork_expected_pinned_after_kills"])


def test_below_break_even_both_arms_keep_the_prefix(below):
    """The boundary, not the branding: with U <= C - P the stock cache keeps
    the shared prefix too, and the two arms tie on hit rate."""
    cfg = below["config"]
    assert cfg["interleaved_tokens_u"] <= cfg["break_even_u"]
    assert below["arms"]["stock"]["parent_hit_rate"] == 1.0
    assert below["arms"]["agentfork"]["parent_hit_rate"] == 1.0


def test_output_is_labelled_as_stubbed_generation(above):
    """No run of this demo may imply the model output is real."""
    assert above["config"]["model_output"] is False
