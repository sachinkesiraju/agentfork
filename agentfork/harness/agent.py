"""An agent loop over a fork tree.

``ForkOrchestrator`` forks and kills branches — a branch being a KV-cache
branch plus a sandbox — but it forks *state*, not an *agent*: it never decides
what to run on a branch or which branch to keep. ``TreeAgent`` is that missing
layer. It owns an orchestrator and drives the golden-path loop the README's
"tree, not flat best-of-N" example describes:

    prepare a shared context on a root branch
      -> fan out N candidate branches, each with a continuation that
         STRICTLY EXTENDS the shared committed prefix
      -> run per-branch work (sandbox exec and/or generate)
      -> score each branch with a pluggable evaluator
      -> keep the winner, kill the losers
      -> (optionally) fork the winner again for a verification round

The tree shape is one level of ``run_round`` per verification stage: a round's
winner stays live (``kill_losers`` keeps the winner and its ancestors), so the
next round forks *it*, inheriting the root context plus the winning candidate's
committed continuation. ``solve`` chains rounds end to end.

Prefix lineage. Every branch's committed token/text prefix is tracked here,
mirroring how ``SGLangKVBackend`` tracks per-branch lengths, so the harness can
enforce the invariant the SGLang patch (``patches/0003``) enforces engine-side:
a branch's continuation must extend its committed prefix. The harness is
stricter — a continuation must *strictly* extend it (be a proper superset) —
and rejects a violation with ``PrefixViolation`` before anything is forked, so
a bad continuation never leaks a branch. Both units work: token-list
continuations go through ``extend()`` (the ``TreeKVCache`` path); string
continuations go through ``generate()`` (the ``external_data_path`` SGLang
path), and a string given to a token backend is encoded first.

Scope matches the rest of the repository: stdlib only, the orchestrator and its
backends do the heavy lifting, and locking is narrow — only the prefix ledger
is guarded here; every backend call already serializes itself.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

from agentfork.orchestrator import ForkOrchestrator, KillReceipt


class PrefixViolation(ValueError):
    """A continuation does not strictly extend its branch's committed prefix."""


class NoWinner(RuntimeError):
    """Every branch in a round failed, so there is nothing to keep."""


@dataclass
class BranchResult:
    """Outcome of one candidate branch. ``output`` is whatever the round's
    work callable returned; ``error`` is set (and ``score`` left at ``-inf``)
    when the branch's commit, work, or scoring raised — a failed branch never
    poisons its siblings and is dropped at winner selection."""

    branch_id: str
    prefix: str | list[int]
    output: Any = None
    score: float = float("-inf")
    error: BaseException | None = None


# An evaluator scores a finished branch; higher is better.
Evaluator = Callable[[BranchResult], float]
# Work runs a branch's task and returns an arbitrary result object; it is
# handed the branch id and the branch's committed continuation.
Work = Callable[[str, "str | list[int]"], Any]


@dataclass
class Round:
    """One fan-out stage: one continuation per candidate branch, the work to
    run on each, and the evaluator that scores them. Each continuation must
    strictly extend the parent branch's committed prefix."""

    continuations: Sequence[str | list[int]]
    work: Work
    evaluator: Evaluator
    child_ids: list[str] | None = None
    sampling_params: dict | None = None


def utf8_tokens(text: str) -> list[int]:
    """Default text->token encoding for token backends: UTF-8 code units.

    A string prefix maps to a token prefix (``"ab".encode()`` prefixes
    ``"abc".encode()``), so strict text extension implies strict token
    extension — the invariant survives the encoding."""
    return list(text.encode("utf-8"))


class TreeAgent:
    """Runs an agent loop over a ``ForkOrchestrator``'s fork tree."""

    def __init__(self, orch: ForkOrchestrator,
                 encode: Callable[[str], list[int]] = utf8_tokens):
        self.orch = orch
        self._encode = encode
        self._string_path = bool(getattr(orch.kv, "external_data_path", False))
        self._lock = threading.RLock()
        self._prefix: dict[str, str | list[int]] = {}

    # -- lineage helpers -----------------------------------------------------

    def _unit(self, value: str | list[int]) -> str | list[int]:
        """Coerce a continuation into the backend's native prefix unit."""
        if self._string_path:
            if not isinstance(value, str):
                raise TypeError(
                    "this KV backend is text-native (external_data_path); "
                    "continuations must be str")
            return value
        return self._encode(value) if isinstance(value, str) else list(value)

    @staticmethod
    def _strictly_extends(base: str | list[int],
                          candidate: str | list[int]) -> bool:
        if len(candidate) <= len(base):
            return False
        if isinstance(base, str):
            return candidate.startswith(base)
        return candidate[:len(base)] == base

    def _require_extension(self, branch_id: str, base: str | list[int],
                           candidate: str | list[int]) -> None:
        if not self._strictly_extends(base, candidate):
            raise PrefixViolation(
                f"continuation for branch {branch_id} does not strictly "
                "extend its committed prefix")

    def committed_prefix(self, branch_id: str) -> str | list[int]:
        """The branch's committed token/text prefix (a copy for lists)."""
        with self._lock:
            value = self._prefix[branch_id]
        return list(value) if isinstance(value, list) else value

    def _commit(self, branch_id: str, target: str | list[int],
                sampling_params: dict | None) -> None:
        """Charge ``target``'s uncached suffix to the branch and record it as
        the branch's committed prefix. ``target`` is the full new prefix; the
        base it extends is whatever the branch already committed (the inherited
        parent prefix, for a freshly forked child)."""
        with self._lock:
            base = self._prefix[branch_id]
        if self._string_path:
            self.orch.generate(branch_id, target,
                               sampling_params or {"max_new_tokens": 1})
        else:
            self.orch.extend(branch_id, target[len(base):])
        with self._lock:
            self._prefix[branch_id] = target

    def _forget(self, receipts: Sequence[KillReceipt]) -> None:
        with self._lock:
            for receipt in receipts:
                if receipt.reaped:
                    self._prefix.pop(receipt.branch_id, None)

    # -- loop ----------------------------------------------------------------

    def prepare_root(self, root_id: str, context: str | list[int], *,
                     lease_s: float | None = None,
                     sampling_params: dict | None = None) -> str:
        """Create the root branch and commit the shared context every
        candidate will inherit copy-on-write."""
        self.orch.create_parent(root_id, lease_s=lease_s)
        with self._lock:
            self._prefix[root_id] = "" if self._string_path else []
        self._commit(root_id, self._unit(context), sampling_params)
        return root_id

    def fan_out(self, parent_id: str, continuations: Sequence[str | list[int]],
                work: Work, evaluator: Evaluator, *,
                child_ids: list[str] | None = None,
                sampling_params: dict | None = None) -> list[BranchResult]:
        """Fork one child per continuation, commit each continuation, run its
        work, and score it. Continuations are validated against the parent's
        committed prefix *before* any child is forked, so a ``PrefixViolation``
        leaves the tree untouched. A child whose commit/work/scoring raises is
        recorded as a failed ``BranchResult`` and does not stop its siblings."""
        base = self.committed_prefix(parent_id)
        targets = [self._unit(c) for c in continuations]
        for target in targets:
            self._require_extension(parent_id, base, target)

        children = self.orch.fork(parent_id, n=len(targets),
                                  child_ids=child_ids)
        with self._lock:
            for child in children:
                self._prefix[child.branch_id] = (
                    list(base) if isinstance(base, list) else base)

        results = []
        for child, target in zip(children, targets):
            result = BranchResult(child.branch_id, target)
            try:
                self._commit(child.branch_id, target, sampling_params)
                result.output = work(child.branch_id, target)
                result.score = float(evaluator(result))
            except BaseException as exc:  # one branch failing must not poison
                result.error = exc        # its siblings; it is dropped below
            results.append(result)
        return results

    @staticmethod
    def select_winner(results: Sequence[BranchResult]) -> BranchResult:
        live = [r for r in results if r.error is None]
        if not live:
            raise NoWinner("every branch in the round failed")
        return max(live, key=lambda r: r.score)

    def kill_losers(self, winner_id: str) -> list[KillReceipt]:
        """Kill every live branch except the winner and its ancestor chain,
        then drop the reaped branches from the prefix ledger."""
        receipts = self.orch.kill_losers(winner_id)
        self._forget(receipts)
        return receipts

    def run_round(self, parent_id: str, round: Round) -> BranchResult:
        """Fan out, score, keep the winner, kill the losers, return the
        winner (still live, ready to be forked for the next round)."""
        results = self.fan_out(
            parent_id, round.continuations, round.work, round.evaluator,
            child_ids=round.child_ids, sampling_params=round.sampling_params)
        winner = self.select_winner(results)
        self.kill_losers(winner.branch_id)
        return winner

    def solve(self, root_id: str, context: str | list[int],
              rounds: Sequence[Round], *,
              lease_s: float | None = None) -> BranchResult:
        """Golden path: prepare the root, then run each round against the
        previous round's winner. The final winner is the only live leaf; its
        ancestor chain (root + earlier winners) stays resident."""
        if not rounds:
            raise ValueError("solve requires at least one round")
        self.prepare_root(root_id, context, lease_s=lease_s)
        parent_id = root_id
        winner = None
        for round in rounds:
            winner = self.run_round(parent_id, round)
            parent_id = winner.branch_id
        return winner
