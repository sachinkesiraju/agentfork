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
def roomy() -> dict:
    """Low pressure *and* enough capacity for every approach subtree at once:
    the only configuration in which the stock arm keeps whole lineages."""
    return run_demo(**{"--noise-requests-per-gap": "1",
                       "--capacity-tokens": "6144"})


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


def test_tree_is_multi_level_and_leaves_hit_their_own_subtree(above):
    """The tree is root -> approach -> candidate -> verification, and an
    agentfork leaf reuses its *whole lineage*, not just the root context:
    the approach's committed reasoning is cached for it too."""
    for name in ("stock", "agentfork"):
        arm = above["arms"][name]
        approach_ids = {a["branch_id"] for a in arm["approaches"]}
        assert len(approach_ids) == above["config"]["approaches"], name
        assert all(a["depth"] == 1 for a in arm["approaches"]), name
        # every candidate hangs off an approach branch, not off the root
        assert all(c["depth"] == 2 for c in arm["children"]), name
        assert {c["parent_branch"] for c in arm["children"]} <= approach_ids
        assert all(c["depth"] == 3 for c in arm["verify_children"]), name

    agentfork = above["arms"]["agentfork"]
    assert agentfork["subtree_hit_rate"] == 1.0
    assert all(c["hit_level"] == "subtree" for c in agentfork["children"])
    # a leaf's cached tokens exceed the shared context: the extra is its
    # approach's own reasoning, inherited rather than re-prefilled
    assert all(c["cached_tokens"] > above["config"]["prefix_tokens"]
               for c in agentfork["children"])
    # the stock arm cannot even keep the root context, let alone a subtree
    assert above["arms"]["stock"]["subtree_hit_rate"] == 0.0


def test_below_break_even_both_arms_keep_the_prefix(below):
    """The boundary, not the branding: with U <= C - P the stock cache keeps
    the shared prefix too, and the two arms tie on hit rate."""
    cfg = below["config"]
    assert cfg["interleaved_tokens_u"] <= cfg["break_even_u"]
    assert below["arms"]["stock"]["parent_hit_rate"] == 1.0
    assert below["arms"]["agentfork"]["parent_hit_rate"] == 1.0


def test_below_break_even_the_stock_arm_still_loses_subtrees(below):
    """A sharper boundary than `U* = C - P`, which only covers the root.

    Keeping a *whole lineage* unpinned also needs room for every sibling
    subtree's reasoning at once; when it does not fit, the cache's LRU drops
    the older approaches even though the shared root context survives. So the
    stock arm can hit the root on every candidate and still be charged for the
    approach above it -- while the pinned lineage is immune."""
    stock = below["arms"]["stock"]
    assert stock["parent_hit_rate"] == 1.0
    assert stock["subtree_hit_rate"] < 1.0
    assert any(c["hit_level"] == "root" for c in stock["children"])
    assert below["arms"]["agentfork"]["subtree_hit_rate"] == 1.0


def test_with_capacity_for_every_subtree_the_stock_arm_ties(roomy):
    """... and it is capacity, not framing: give the stock cache room for all
    three approach subtrees and it keeps whole lineages too."""
    assert roomy["arms"]["stock"]["subtree_hit_rate"] == 1.0
    assert roomy["arms"]["agentfork"]["subtree_hit_rate"] == 1.0


def test_output_is_labelled_as_stubbed_generation(above):
    """No run of this demo may imply the model output is real."""
    assert above["config"]["model_output"] is False
