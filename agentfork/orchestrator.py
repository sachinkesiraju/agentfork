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
  previous process left mid-fork or mid-kill (kill intent is journaled
  before either half is reaped);
- ``exec(branch)`` runs a command in a branch's sandbox, for backends that
  expose an exec channel (``FirecrackerSandbox`` over vsock).

The registry is a JSON file written atomically (write + fsync +
``os.replace``) and owned exclusively: a sidecar ``flock`` held for the
orchestrator's lifetime makes a second orchestrator on the same file fail
loudly instead of corrupting it, and the kernel drops the lock if the owner
dies. The file records intent, not live handles: after a supervisor crash a
new orchestrator loads it and replays ``kill()`` against its backends, which
is why ``SandboxBackend.kill`` must be idempotent and tolerate unknown branch
IDs.

Locking is narrow, not coarse: the orchestrator's lock covers only registry
bookkeeping, and every backend call (KV fork/kill, sandbox spawn/kill/exec)
runs outside it, so a slow sandbox operation on one branch never blocks
work on another. Backends serialize themselves internally. Multi-branch
operations (``fork(n)``, ``kill_losers``, ``reap_expired``, ``reconcile``,
``close``) fan out across branches when the sandbox backend declares
``parallel_lifecycle = True``; ``ReaperSandbox`` deliberately does not,
because its ``preexec_fn`` orphan backstop is unsafe to run from threads.
Concurrent callers racing on the *same* branch get no-op kills while a kill
or spawn of that branch is already in flight, and ``reconcile()`` /
``reap_expired()`` skip branches this process is still working on.

Scope matches the rest of this repository: the KV half defaults to the CPU
reference ``TreeKVCache`` and the sandbox half to a generic subprocess (via
``BranchReaper``), but either can be any object satisfying ``KVBackend`` or
``SandboxBackend``. Kill remains sequential across the two halves, not atomic.
Adapters for a patched-SGLang engine (``agentfork.kv.sglang_backend``) and for
Firecracker (``agentfork.sandbox.firecracker_backend``) satisfy those
protocols. The Firecracker adapter has been driven end to end against real
microVMs (``demo/fc_demo.py``, idle guests); the SGLang adapter is unit-tested
against mocks only and has not touched a live engine.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Protocol

from agentfork._locking import locked
from agentfork.kill.reaper import BranchReaper
from agentfork.kv.tree_cache import TreeKVCache

_log = logging.getLogger("agentfork.orchestrator")

_STATE_FORKING = "forking"
_STATE_LIVE = "live"
_STATE_KILLING = "killing"
_MAX_FANOUT_THREADS = 8


class SandboxBackend(Protocol):
    """Sandbox half of a branch. ``kill`` must be idempotent: killing an
    unknown or already-dead branch is a no-op, because ``reconcile()`` replays
    kills recorded by a previous process. Backends whose spawn/kill are safe
    to call from multiple threads at once may set a class attribute
    ``parallel_lifecycle = True`` to let multi-branch operations fan out."""

    def spawn(self, branch_id: str, parent_id: str | None) -> None: ...
    def kill(self, branch_id: str) -> None: ...
    def alive(self, branch_id: str) -> bool: ...


class KVBackend(Protocol):
    """KV half of a branch. Mirrors ``TreeKVCache``'s surface: ``fork_branch``
    performs a zero-copy logical fork, ``kill`` releases a branch's pages and
    returns the number of tokens freed, ``extend`` appends tokens and returns
    the number newly charged (not already cached along this branch's path).
    Must be safe for concurrent callers (all shipped backends lock
    internally)."""

    def create_tree(self, tree_id: str) -> object: ...
    def fork_branch(self, parent_id: str, child_id: str | None = None) -> object: ...
    def kill(self, tree_id: str) -> int: ...
    def extend(self, tree_id: str, tokens: list[int]) -> int: ...


class NullSandbox:
    """KV-only orchestration: every branch trivially has a live sandbox."""

    parallel_lifecycle = True

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
    ``pdeathsig`` is forwarded to the default ``BranchReaper`` (ignored when
    an explicit ``reaper`` is injected); pass ``False`` under threaded
    supervisors. ``parallel_lifecycle`` stays False: fanning spawns out to
    threads would make ``preexec_fn`` unsafe.
    """

    parallel_lifecycle = False

    def __init__(self, argv: list[str], reaper: BranchReaper | None = None,
                 pdeathsig: bool = True):
        if not argv:
            raise ValueError("argv must not be empty")
        self.argv = list(argv)
        self.reaper = reaper or BranchReaper(pdeathsig=pdeathsig)

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
        self._lock = threading.RLock()
        self._seq = 0
        self._closed = False
        self._branches: dict[str, Branch] = {}
        self._spawning: set[str] = set()   # forks in flight in this process
        self._killing: set[str] = set()    # kills in flight in this process
        self._registry_lock_fd: int | None = None
        if self.registry_path:
            self._acquire_registry_lock()
        try:
            if self.registry_path and os.path.exists(self.registry_path):
                with open(self.registry_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._branches = {b["branch_id"]: Branch.from_dict(b)
                                  for b in data.get("branches", [])}
                if self._branches:
                    _log.info("registry %s: loaded %d branch(es) from a "
                              "previous owner", self.registry_path,
                              len(self._branches))
        except BaseException:
            self._release_registry_lock()
            raise

    # -- registry ------------------------------------------------------------

    def _acquire_registry_lock(self) -> None:
        """Take exclusive ownership of the registry for this orchestrator's
        lifetime. The flock lives on a sidecar file because it binds to an
        inode and ``_persist`` swaps the registry's inode on every write; the
        kernel drops the lock automatically if the owning process dies."""
        lock_path = f"{self.registry_path}.lock"
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise RuntimeError(
                f"registry {self.registry_path} is owned by another "
                f"orchestrator (lock: {lock_path})") from None
        except OSError:
            os.close(fd)
            raise
        self._registry_lock_fd = fd

    def _release_registry_lock(self) -> None:
        if self._registry_lock_fd is not None:
            os.close(self._registry_lock_fd)  # closing the fd drops the flock
            self._registry_lock_fd = None

    def _ensure_open(self) -> None:
        """A closed orchestrator gave up registry ownership; letting it keep
        writing would corrupt a successor's registry. Reads stay allowed."""
        if self._closed:
            raise RuntimeError("orchestrator is closed")

    def _persist(self) -> None:
        if not self.registry_path:
            return
        tmp = f"{self.registry_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1,
                       "branches": [b.to_dict() for b in self._branches.values()]},
                      f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.registry_path)
        dir_fd = os.open(os.path.dirname(self.registry_path) or ".", os.O_RDONLY)
        try:
            os.fsync(dir_fd)  # make the rename itself durable
        except OSError:
            pass  # some filesystems reject directory fsync; best effort
        finally:
            os.close(dir_fd)

    @locked
    def _record(self, branch: Branch) -> None:
        self._branches[branch.branch_id] = branch
        self._persist()

    @locked
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

    def _fan_out(self, items: list, fn: Callable) -> list:
        """Apply ``fn`` to every item — threaded when the sandbox declares
        ``parallel_lifecycle``, else in order. Every item is attempted; the
        first exception is re-raised after all complete, and results of
        failed items are dropped."""
        if len(items) > 1 and getattr(self.sandbox, "parallel_lifecycle", False):
            with ThreadPoolExecutor(min(len(items), _MAX_FANOUT_THREADS)) as pool:
                futures = [pool.submit(fn, item) for item in items]
                results, error = [], None
                for future in futures:
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        error = error or exc
        else:
            results, error = [], None
            for item in items:
                try:
                    results.append(fn(item))
                except Exception as exc:
                    error = error or exc
        if error is not None:
            raise error
        return results

    # -- lifecycle -----------------------------------------------------------

    def create_parent(self, branch_id: str, tokens: list[int] | None = None,
                      lease_s: float | None = None) -> Branch:
        """Create a root branch: journal, create the KV tree, spawn sandbox."""
        with self._lock:
            self._ensure_open()
            if branch_id in self._branches:
                raise ValueError(f"branch exists: {branch_id}")
            branch = Branch(branch_id, None, _STATE_FORKING, self._clock(),
                            self._lease_expiry(lease_s))
            self._spawning.add(branch_id)
            self._record(branch)
        try:
            try:
                self.kv.create_tree(branch_id)
                self.sandbox.spawn(branch_id, None)
                if tokens:
                    self.kv.extend(branch_id, tokens)
            except BaseException:
                _log.warning("create_parent %s failed; rolling back",
                             branch_id, exc_info=True)
                self.kv.kill(branch_id)
                self.sandbox.kill(branch_id)
                self._forget(branch_id)
                raise
            with self._lock:
                branch.state = _STATE_LIVE
                self._persist()
        finally:
            with self._lock:
                self._spawning.discard(branch_id)
        _log.info("created root branch %s", branch_id)
        return branch

    def fork(self, parent_id: str, n: int = 1,
             child_ids: list[str] | None = None,
             lease_s: float | None = None) -> list[Branch]:
        """Fork ``n`` children from a live parent.

        Per child: journal intent, fork the KV branch (CoW, no copy), spawn
        the sandbox. Every child is attempted (concurrently, if the sandbox
        backend allows it); a failed child is rolled back — KV branch and
        registry record — without touching its siblings, and the first
        failure is re-raised after all children settle.
        """
        with self._lock:
            self._ensure_open()
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
                if child_id in self._branches or any(
                        c.branch_id == child_id for c in children):
                    raise ValueError(f"branch exists: {child_id}")
                children.append(Branch(child_id, parent_id, _STATE_FORKING,
                                       self._clock(), self._lease_expiry(lease_s)))
            for branch in children:
                self._branches[branch.branch_id] = branch
                self._spawning.add(branch.branch_id)
            self._persist()

        def spawn_one(branch: Branch) -> Branch:
            child_id = branch.branch_id
            try:
                try:
                    self.kv.fork_branch(parent_id, child_id)
                    self.sandbox.spawn(child_id, parent_id)
                except BaseException:
                    _log.warning("fork of %s from %s failed; rolling back",
                                 child_id, parent_id, exc_info=True)
                    self.kv.kill(child_id)
                    self.sandbox.kill(child_id)
                    self._forget(child_id)
                    raise
                with self._lock:
                    branch.state = _STATE_LIVE
                    self._persist()
            finally:
                with self._lock:
                    self._spawning.discard(child_id)
            return branch

        forked = self._fan_out(children, spawn_one)
        _log.info("forked %d child(ren) of %s", len(forked), parent_id)
        return forked

    def extend(self, branch_id: str, tokens: list[int]) -> int:
        with self._lock:
            self._ensure_open()
            if branch_id not in self._branches:
                raise KeyError(f"no such branch: {branch_id}")
        return self.kv.extend(branch_id, tokens)

    def exec(self, branch_id: str, argv: list[str],
             timeout_s: float | None = None):
        """Run a command in the branch's sandbox, for backends that expose
        an ``exec(branch_id, argv, timeout_s)`` channel (``FirecrackerSandbox``
        does; ``NullSandbox``/``ReaperSandbox`` do not).

        Only the branch lookup holds the orchestrator lock; the sandbox I/O
        runs outside it so a long guest command cannot block fork/kill of
        other branches. The cost of that: a kill racing this call surfaces
        as the backend's transport error rather than a tidy KeyError.
        """
        with self._lock:
            self._ensure_open()
            branch = self._branches.get(branch_id)
            if branch is None or branch.state != _STATE_LIVE:
                raise KeyError(f"no live branch: {branch_id}")
            sandbox_exec = getattr(self.sandbox, "exec", None)
            if sandbox_exec is None:
                raise RuntimeError(
                    f"sandbox backend {type(self.sandbox).__name__} does not "
                    "support exec")
        return sandbox_exec(branch_id, argv, timeout_s)

    def kill(self, branch_id: str) -> KillReceipt:
        """Journal kill intent, reap sandbox then KV, then drop the record.

        The two halves are sequential, not atomic. Intent is journaled as
        state ``killing`` before either half runs, so if one raises (or the
        supervisor crashes mid-kill) the record survives and ``reconcile()``
        retries the kill. Killing an unknown branch is a no-op (idempotent),
        as is killing a branch another thread of this process is already
        killing or still forking (the fork wins the race; kill again after
        it settles). A killed parent's children survive by design (their KV
        refs keep the shared prefix resident); their ``parent_id`` then
        names a dead branch, and their own leases bound their lifetime.
        """
        with self._lock:
            self._ensure_open()
            if (branch_id not in self._branches
                    or branch_id in self._killing
                    or branch_id in self._spawning):
                return KillReceipt(branch_id, 0)
            self._killing.add(branch_id)
            self._branches[branch_id].state = _STATE_KILLING
            self._persist()
        try:
            self.sandbox.kill(branch_id)
            freed = self.kv.kill(branch_id)
        except BaseException:
            _log.warning("kill of %s failed; record stays journaled for "
                         "reconcile()", branch_id, exc_info=True)
            with self._lock:
                self._killing.discard(branch_id)
            raise  # record stays in state "killing"; reconcile() retries
        with self._lock:
            self._killing.discard(branch_id)
            self._forget(branch_id)
        _log.debug("killed %s (freed %d KV tokens)", branch_id, freed)
        return KillReceipt(branch_id, freed)

    def kill_losers(self, winner_id: str) -> list[KillReceipt]:
        """Kill every live branch except the winner and its ancestor chain."""
        with self._lock:
            self._ensure_open()
            if winner_id not in self._branches:
                raise KeyError(f"no such branch: {winner_id}")
            keep = set()
            cursor: str | None = winner_id
            while cursor is not None and cursor not in keep:
                keep.add(cursor)
                branch = self._branches.get(cursor)
                cursor = branch.parent_id if branch else None
            losers = [bid for bid in self._branches if bid not in keep]
        return self._fan_out(losers, self.kill)

    # -- collection ----------------------------------------------------------

    @locked
    def renew_lease(self, branch_id: str, lease_s: float) -> Branch:
        self._ensure_open()
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
        survive a parent kill by design, so per-branch leases are the bound.
        Branches this process is still forking are skipped and caught on a
        later pass."""
        with self._lock:
            self._ensure_open()
            now = self._clock()
            expired = [b.branch_id for b in self._branches.values()
                       if b.lease_expires_at is not None
                       and b.lease_expires_at <= now
                       and b.branch_id not in self._spawning]
        if expired:
            _log.info("reaping %d branch(es) with lapsed leases: %s",
                      len(expired), expired)
        return self._fan_out(expired, self.kill)

    def reconcile(self) -> list[KillReceipt]:
        """Collect leaked branches: mid-fork and mid-kill leftovers from a
        crashed or failed supervisor, plus anything past its lease. Not run
        automatically; call it at startup on a registry a previous process
        wrote. Branches with fork or kill work in flight in *this* process
        are not leaks and are left alone."""
        with self._lock:
            self._ensure_open()
            stuck = [b.branch_id for b in self._branches.values()
                     if b.state in (_STATE_FORKING, _STATE_KILLING)
                     and b.branch_id not in self._spawning
                     and b.branch_id not in self._killing]
        if stuck:
            _log.info("reconcile: collecting %d branch(es) left mid-fork or "
                      "mid-kill: %s", len(stuck), stuck)
        receipts = self._fan_out(stuck, self.kill)
        receipts.extend(self.reap_expired())
        return receipts

    # -- introspection / teardown ---------------------------------------------

    @locked
    def branches(self) -> list[Branch]:
        return list(self._branches.values())

    def alive(self, branch_id: str) -> bool:
        with self._lock:
            branch = self._branches.get(branch_id)
            if branch is None or branch.state != _STATE_LIVE:
                return False
        return self.sandbox.alive(branch_id)

    def close(self) -> None:
        """Kill every recorded branch; raise the first error after trying
        all. The orchestrator ends closed either way: registry ownership is
        released so a successor can take over, and every mutating method
        raises from then on. Idempotent. Must not race in-flight lifecycle
        calls on other threads."""
        with self._lock:
            if self._closed:
                return
            branch_ids = list(self._branches)
        error = None
        try:
            try:
                self._fan_out(branch_ids, self.kill)
            except Exception as exc:
                error = exc
        finally:
            with self._lock:
                self._closed = True
                self._release_registry_lock()
        if error is not None:
            raise error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
