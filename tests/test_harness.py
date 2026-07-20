"""Harness tests: fakes only, no network. Cover fan-out, the strict-prefix
invariant, winner selection, loser cleanup (no leaked branches or KV tokens),
re-fork of a winner, and one branch failing without poisoning its siblings."""

import pytest

from agentfork.harness import (
    BranchResult,
    FakeLLM,
    NoWinner,
    PrefixViolation,
    Round,
    TreeAgent,
    utf8_tokens,
)
from agentfork.kv.tree_cache import TreeKVCache
from agentfork.orchestrator import ForkOrchestrator, NullSandbox

CONTEXT = "shared repository context"


def make_agent():
    orch = ForkOrchestrator(kv=TreeKVCache(), sandbox=NullSandbox())
    return TreeAgent(orch), orch


def echo_work(branch_id, prefix):
    return prefix


def score_by_suffix(good_suffix):
    def evaluator(result: BranchResult) -> float:
        text = bytes(result.prefix).decode()
        return 1.0 if text.endswith(good_suffix) else 0.0
    return evaluator


def test_prepare_root_commits_context():
    agent, orch = make_agent()
    try:
        agent.prepare_root("root", CONTEXT)
        assert agent.committed_prefix("root") == utf8_tokens(CONTEXT)
        assert orch.kv.resident_tokens() == len(utf8_tokens(CONTEXT))
    finally:
        orch.close()


def test_fan_out_forks_one_child_per_continuation():
    agent, orch = make_agent()
    try:
        agent.prepare_root("root", CONTEXT)
        conts = [CONTEXT + " A", CONTEXT + " B", CONTEXT + " C"]
        results = agent.fan_out("root", conts, echo_work, lambda r: 0.0)
        assert len(results) == 3
        assert all(r.error is None for r in results)
        # root + three live children
        assert len(orch.branches()) == 4
        for result, cont in zip(results, conts):
            assert result.output == utf8_tokens(cont)
    finally:
        orch.close()


def test_continuation_must_strictly_extend_prefix():
    agent, orch = make_agent()
    try:
        agent.prepare_root("root", CONTEXT)
        # does not extend the committed prefix at all
        with pytest.raises(PrefixViolation):
            agent.fan_out("root", ["totally different"], echo_work,
                          lambda r: 0.0)
        # equal to the prefix is not a *strict* extension
        with pytest.raises(PrefixViolation):
            agent.fan_out("root", [CONTEXT], echo_work, lambda r: 0.0)
        # a rejected continuation must not leak a branch
        assert [b.branch_id for b in orch.branches()] == ["root"]
        assert set(orch.kv.trees) == {"root"}
    finally:
        orch.close()


def test_run_round_keeps_winner_and_reaps_losers():
    agent, orch = make_agent()
    try:
        agent.prepare_root("root", CONTEXT)
        winner = agent.run_round("root", Round(
            continuations=[CONTEXT + " lose", CONTEXT + " WIN",
                           CONTEXT + " lose too"],
            work=echo_work,
            evaluator=score_by_suffix("WIN")))
        assert winner.score == 1.0
        assert bytes(winner.prefix).decode().endswith("WIN")
        # only the winner and root survive, in both halves
        live = {b.branch_id for b in orch.branches()}
        assert live == {"root", winner.branch_id}
        assert set(orch.kv.trees) == {"root", winner.branch_id}
    finally:
        orch.close()


def test_refork_winner_second_round():
    agent, orch = make_agent()
    try:
        r1 = Round(continuations=[CONTEXT + " x", CONTEXT + " KEEP"],
                   work=echo_work, evaluator=score_by_suffix("KEEP"))
        # round two extends the round-one winner's committed prefix
        r2 = Round(continuations=[CONTEXT + " KEEP a", CONTEXT + " KEEP FINAL"],
                   work=echo_work, evaluator=score_by_suffix("FINAL"))
        final = agent.solve("root", CONTEXT, [r1, r2])
        assert bytes(final.prefix).decode().endswith("FINAL")
        branches = {b.branch_id: b.parent_id for b in orch.branches()}
        # root -> round-1 winner -> final winner (a three-generation chain)
        assert len(branches) == 3
        parent = branches[final.branch_id]
        assert parent is not None and parent != "root"
        assert branches[parent] == "root"
        assert set(orch.kv.trees) == set(branches)
    finally:
        orch.close()


def test_second_round_continuation_must_extend_winner_prefix():
    agent, orch = make_agent()
    try:
        agent.prepare_root("root", CONTEXT)
        winner = agent.run_round("root", Round(
            continuations=[CONTEXT + " A", CONTEXT + " B"],
            work=echo_work, evaluator=score_by_suffix("B")))
        # a continuation that extends root but not the winner is rejected
        with pytest.raises(PrefixViolation):
            agent.fan_out(winner.branch_id, [CONTEXT + " C"], echo_work,
                          lambda r: 0.0)
    finally:
        orch.close()


def test_one_branch_failure_does_not_poison_siblings():
    agent, orch = make_agent()
    try:
        agent.prepare_root("root", CONTEXT)

        def flaky_work(branch_id, prefix):
            if bytes(prefix).decode().endswith("BOOM"):
                raise RuntimeError("branch blew up")
            return prefix

        winner = agent.run_round("root", Round(
            continuations=[CONTEXT + " BOOM", CONTEXT + " GOOD"],
            work=flaky_work, evaluator=score_by_suffix("GOOD")))
        assert bytes(winner.prefix).decode().endswith("GOOD")
        # the failed sibling was cleaned up, not left leaking
        assert {b.branch_id for b in orch.branches()} == {
            "root", winner.branch_id}
        assert set(orch.kv.trees) == {"root", winner.branch_id}
    finally:
        orch.close()


def test_all_branches_failing_raises_no_winner():
    agent, orch = make_agent()
    try:
        agent.prepare_root("root", CONTEXT)

        def always_fails(branch_id, prefix):
            raise RuntimeError("nope")

        with pytest.raises(NoWinner):
            agent.run_round("root", Round(
                continuations=[CONTEXT + " a", CONTEXT + " b"],
                work=always_fails, evaluator=lambda r: 1.0))
    finally:
        orch.close()


def test_close_reaps_the_whole_tree():
    agent, orch = make_agent()
    agent.solve("root", CONTEXT, [Round(
        continuations=[CONTEXT + " a", CONTEXT + " WIN"],
        work=echo_work, evaluator=score_by_suffix("WIN"))])
    orch.close()
    assert orch.kv.resident_tokens() == 0
    assert orch.kv.trees == {}


def test_token_list_continuations():
    agent, orch = make_agent()
    try:
        agent.prepare_root("root", [1, 2, 3])
        results = agent.fan_out("root", [[1, 2, 3, 4], [1, 2, 3, 5]],
                                echo_work, lambda r: 0.0)
        assert results[0].output == [1, 2, 3, 4]
        with pytest.raises(PrefixViolation):
            agent.fan_out("root", [[9, 9, 9]], echo_work, lambda r: 0.0)
    finally:
        orch.close()


def test_fake_llm_propose_and_score():
    llm = FakeLLM(["fix-a", "fix-b"], scores={"fix-a": 0.9})
    assert llm.propose("ctx", 3) == ["fix-a", "fix-b", "fix-a"]
    assert llm.score("ctx", "fix-a") == 0.9
    assert llm.score("ctx", "unknown") == 0.0
