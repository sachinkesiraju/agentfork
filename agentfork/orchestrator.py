"""Reference control plane binding sandbox and KV-cache branch lifecycle.

``ForkOrchestrator`` gives one branch ID to both halves of an agent branch:

- ``fork(parent)`` journals intent to a persistent registry, forks the KV
  tree, then spawns the sandbox — and rolls the KV fork back if the sandbox
  fails, so a branch never exists half-created;
- ``kill(branch)`` reaps sandbox then KV and removes the registry record only
  after both succeed, so a crashed or failed kill is retried by
  ``reconcile()`` instead of leaking;
- leases bound every branch's lifetime; ``reap_expired()`` collects branches
  whose lease lapsed, and ``reconcile()`` additionally collects branches a
  previous process left mid-fork.

The registry is a JSON file written atomically (write + ``os.replace``). It
records intent, not live handles: after a supervisor crash a new orchestrator
loads the file and replays ``kill()`` against its backends, which is why
``SandboxBackend.kill`` must be idempotent and tolerate unknown branch IDs.

Scope matches the rest of this repository: the KV half defaults to the CPU
reference ``TreeKVCache`` and the sandbox half to a generic subprocess (via
``BranchReaper``), but either can be any object satisfying ``KVBackend`` or
``SandboxBackend``. Kill remains sequential across the two halves, not atomic.
Adapters for a patched-SGLang engine (``agentfork.kv.sglang_backend``) and for
Firecracker (``agentfork.sandbox.firecracker_backend``) exist and satisfy
those protocols, but are unit-tested against mocks only; neither has been run
against a live SGLang engine or a real Firecracker guest.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from agentfork.kill.reaper import BranchReaper
from agentfork.kv.tree_cache import TreeKVCache

_STATE_FORKING = "forking"
_STATE_LIVE = "live"


class SandboxBackend(Protocol):
    """Sandbox half of a branch. ``kill`` must be idempotent: killing an
    unknown or already-dead branch is a no-op, because ``reconcile()`` replays
    kills recorded by a previous process."""

    def spawn(self, branch_id: str, parent_id: str | None) -> None: ...
    def kill(self, branch_id: str) -> None: ...
    def alive(self, branch_id: str) -> bool: ...


class KVBackend(Protocol):
    """KV half of a branch. Mirrors ``TreeKVCache``'s surface: ``fork_branch``
    performs a zero-copy logical fork, ``kill`` releases a branch's pages and
    returns the number of tokens freed, ``extend`` appends tokens and returns
    the number newly charged (not already cached along this branch's path)."""

    def create_tree(self, tree_id: str) -> object: ...
    def fork_branch(self, parent_id: str, child_id: str | None = None) -> object: ...
    def kill(self, tree_id: str) -> int: ...
    def extend(self, tree_id: str, tokens: list[int]) -> int: ...


class NullSandbox:
    """KV-only orchestration: every branch trivially has a live sandbox."""

    def spawn(self, branch_id: str, parent_id: str | None) -> None:
        pass

    def kill(self, branch_id: str) -> None:
        pass

    def alive(self, branch_id: str) -> bool:
        return True


class ReaperSandbox:
    """Adapts ``BranchReaper`` to ``SandboxBackend`` (Linux pidfd only).

    Every branch runs the same argv template. The reaper is used purely for
    process lifecycle; the orchestrator owns the KV half separately.
    """

    def __init__(self, argv: list[str], reaper: BranchReaper | None = None):
        if not argv:
            raise ValueError("argv must not be empty")
        self.argv = list(argv)
        self.reaper = reaper or BranchReaper()

    def spawn(self, branch_id: str, parent_id: str | None) -> None:
        self.reaper.spawn(branch_id, self.argv)

    def kill(self, branch_id: str) -> None:
        try:
            self.reaper.kill(branch_id)
        except KeyError:
            pass

    def alive(self, branch_id: str) -> bool:
        try:
            return self.reaper.alive(branch_id)
        except KeyError:
            return False


@dataclass
class Branch:
    branch_id: str
    parent_id: str | None
    state: str
    created_at: float
    lease_expires_at: float | None

    def to_dict(self) -> dict:
        return {"branch_id": self.branch_id, "parent_id": self.parent_id,
                "state": self.state, "created_at": self.created_at,
                "lease_expires_at": self.lease_expires_at}

    @staticmethod
    def from_dict(d: dict) -> "Branch":
        return Branch(d["branch_id"], d["parent_id"], d["state"],
                      d["created_at"], d["lease_expires_at"])


@dataclass
class KillReceipt:
    branch_id: str
    kv_freed_tokens: int


class ForkOrchestrator:
    """Owns (KV branch, sandbox) pairs; one ID spans both lifecycles."""

    def __init__(self, kv: KVBackend | None = None,
                 sandbox: SandboxBackend | None = None,
                 registry_path: str | os.PathLike | None = None,
                 default_lease_s: float | None = None,
                 clock: Callable[[], float] = time.time):
        if default_lease_s is not None and default_lease_s <= 0:
            raise ValueError("default_lease_s must be positive")
        self.kv = kv if kv is not None else TreeKVCache()
        self.sandbox: SandboxBackend = sandbox if sandbox is not None else NullSandbox()
        self.registry_path = os.fspath(registry_path) if registry_path else None
        self.default_lease_s = default_lease_s
        self._clock = clock
        self._seq = 0
        self._branches: dict[str, Branch] = {}
        if self.registry_path and os.path.exists(self.registry_path):
            with open(self.registry_path, encoding="utf-8") as f:
                data = json.load(f)
            self._branches = {b["branch_id"]: Branch.from_dict(b)
                              for b in data.get("branches", [])}

    # -- registry ------------------------------------------------------------

    def _persist(self) -> None:
        if not self.registry_path:
            return
        tmp = f"{self.registry_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1,
                       "branches": [b.to_dict() for b in self._branches.values()]},
                      f, indent=1)
        os.replace(tmp, self.registry_path)

    def _record(self, branch: Branch) -> None:
        self._branches[branch.branch_id] = branch
        self._persist()

    def _forget(self, branch_id: str) -> None:
        self._branches.pop(branch_id, None)
        self._persist()

    def _lease_expiry(self, lease_s: float | None) -> float | None:
        lease_s = lease_s if lease_s is not None else self.default_lease_s
        if lease_s is None:
            return None
        if lease_s <= 0:
            raise ValueError("lease_s must be positive")
        return self._clock() + lease_s

    # -- lifecycle -----------------------------------------------------------

    def create_parent(self, branch_id: str, tokens: list[int] | None = None,
                      lease_s: float | None = None) -> Branch:
        """Create a root branch: journal, create the KV tree, spawn sandbox."""
        if branch_id in self._branches:
            raise ValueError(f"branch exists: {branch_id}")
        branch = Branch(branch_id, None, _STATE_FORKING, self._clock(),
                        self._lease_expiry(lease_s))
        self._record(branch)
        try:
            self.kv.create_tree(branch_id)
            self.sandbox.spawn(branch_id, None)
            if tokens:
                self.kv.extend(branch_id, tokens)
        except BaseException:
            self.kv.kill(branch_id)
            self.sandbox.kill(branch_id)
            self._forget(branch_id)
            raise
        branch.state = _STATE_LIVE
        self._persist()
        return branch

    def fork(self, parent_id: str, n: int = 1,
             child_ids: list[str] | None = None,
             lease_s: float | None = None) -> list[Branch]:
        """Fork ``n`` children from a live parent.

        Per child: journal intent, fork the KV branch (CoW, no copy), spawn
        the sandbox. A sandbox failure rolls back that child's KV branch and
        registry record, then re-raises; earlier siblings stay live.
        """
        parent = self._branches.get(parent_id)
        if parent is None or parent.state != _STATE_LIVE:
            raise KeyError(f"no live branch: {parent_id}")
        if child_ids is not None and len(child_ids) != n:
            raise ValueError("child_ids length must equal n")
        children = []
        for k in range(n):
            if child_ids is not None:
                child_id = child_ids[k]
            else:
                self._seq += 1
                child_id = f"{parent_id}/{self._seq}"
            if child_id in self._branches:
                raise ValueError(f"branch exists: {child_id}")
            branch = Branch(child_id, parent_id, _STATE_FORKING, self._clock(),
                            self._lease_expiry(lease_s))
            self._record(branch)
            try:
                self.kv.fork_branch(parent_id, child_id)
                self.sandbox.spawn(child_id, parent_id)
            except BaseException:
                self.kv.kill(child_id)
                self.sandbox.kill(child_id)
                self._forget(child_id)
                raise
            branch.state = _STATE_LIVE
            self._persist()
            children.append(branch)
        return children

    def extend(self, branch_id: str, tokens: list[int]) -> int:
        if branch_id not in self._branches:
            raise KeyError(f"no such branch: {branch_id}")
        return self.kv.extend(branch_id, tokens)

    def kill(self, branch_id: str) -> KillReceipt:
        """Reap sandbox then KV; drop the record only after both succeed.

        The two halves are sequential, not atomic. If either raises, the
        record stays in the registry so ``reconcile()`` retries the kill.
        Killing an unknown branch is a no-op (idempotent).
        """
        if branch_id not in self._branches:
            return KillReceipt(branch_id, 0)
        self.sandbox.kill(branch_id)
        freed = self.kv.kill(branch_id)
        self._forget(branch_id)
        return KillReceipt(branch_id, freed)

    def kill_losers(self, winner_id: str) -> list[KillReceipt]:
        """Kill every live branch except the winner and its ancestor chain."""
        if winner_id not in self._branches:
            raise KeyError(f"no such branch: {winner_id}")
        keep = set()
        cursor: str | None = winner_id
        while cursor is not None and cursor not in keep:
            keep.add(cursor)
            branch = self._branches.get(cursor)
            cursor = branch.parent_id if branch else None
        return [self.kill(bid) for bid in list(self._branches)
                if bid not in keep]

    # -- collection ----------------------------------------------------------

    def renew_lease(self, branch_id: str, lease_s: float) -> Branch:
        branch = self._branches.get(branch_id)
        if branch is None:
            raise KeyError(f"no such branch: {branch_id}")
        if lease_s <= 0:
            raise ValueError("lease_s must be positive")
        branch.lease_expires_at = self._clock() + lease_s
        self._persist()
        return branch

    def reap_expired(self) -> list[KillReceipt]:
        """Kill branches whose lease has lapsed. Children of an expired
        parent are collected too if their own lease lapsed; KV children
        survive a parent kill by design, so per-branch leases are the bound."""
        now = self._clock()
        expired = [b.branch_id for b in self._branches.values()
                   if b.lease_expires_at is not None and b.lease_expires_at <= now]
        return [self.kill(bid) for bid in expired]

    def reconcile(self) -> list[KillReceipt]:
        """Collect leaked branches: mid-fork leftovers from a crashed
        supervisor plus anything past its lease. Safe to call at startup on a
        registry a previous process wrote."""
        stuck = [b.branch_id for b in self._branches.values()
                 if b.state == _STATE_FORKING]
        receipts = [self.kill(bid) for bid in stuck]
        receipts.extend(self.reap_expired())
        return receipts

    # -- introspection / teardown ---------------------------------------------

    def branches(self) -> list[Branch]:
        return list(self._branches.values())

    def alive(self, branch_id: str) -> bool:
        branch = self._branches.get(branch_id)
        return (branch is not None and branch.state == _STATE_LIVE
                and self.sandbox.alive(branch_id))

    def close(self) -> None:
        """Kill every recorded branch; raise the first error after trying all."""
        error = None
        for branch_id in list(self._branches):
            try:
                self.kill(branch_id)
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
