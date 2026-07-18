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
protocols. The Firecracker adapter (including its exec/overlay/jailer data
plane) has been driven end to end against real microVMs (``demo/fc_demo.py``),
and the SGLang request path has run against a live engine on a Modal A10G via
``SGLangHTTPBackend``; see report/RESULTS.md for the recorded runs.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
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
    reaped: bool = True   # False for a no-op (unknown branch, or a kill/fork
    #                       of it already in flight on another thread)


@dataclass
class OrchestratorMetrics:
    """Monotonic counters, mutated under the orchestrator lock. Read a
    consistent view via ``ForkOrchestrator.metrics_snapshot()``."""

    forks: int = 0            # children successfully forked
    kills: int = 0            # branches fully reaped (both halves)
    kill_failures: int = 0    # kills that raised (journaled for reconcile)
    execs: int = 0            # guest exec calls dispatched
    reconciles: int = 0       # reconcile() passes
    reaped_expired: int = 0   # branches collected for lapsed leases
    swept_dead: int = 0       # branches collected because their VMM died


class ForkOrchestrator:
    """Owns (KV branch, sandbox) pairs; one ID spans both lifecycles."""

    def __init__(self, kv: KVBackend | None = None,
                 sandbox: SandboxBackend | None = None,
                 registry_path: str | os.PathLike | None = None,
                 default_lease_s: float | None = None,
                 reap_interval_s: float | None = None,
                 clock: Callable[[], float] = time.time):
        if default_lease_s is not None and default_lease_s <= 0:
            raise ValueError("default_lease_s must be positive")
        if reap_interval_s is not None and reap_interval_s <= 0:
            raise ValueError("reap_interval_s must be positive")
        self.kv = kv if kv is not None else TreeKVCache()
        self.sandbox: SandboxBackend = sandbox if sandbox is not None else NullSandbox()
        self.registry_path = os.fspath(registry_path) if registry_path else None
        self.default_lease_s = default_lease_s
        self._clock = clock
        self._lock = threading.RLock()
        self._lifecycle_changed = threading.Condition(self._lock)
        self._seq = 0
        self._closing = False
        self._closed = False
        self._branches: dict[str, Branch] = {}
        self._loaded_branch_ids: set[str] = set()
        self._spawning: set[str] = set()   # forks in flight in this process
        self._killing: set[str] = set()    # kills in flight in this process
        self.metrics = OrchestratorMetrics()
        self._reaper_thread: threading.Thread | None = None
        self._reaper_stop = threading.Event()
        self._registry_lock_fd: int | None = None
        if self.registry_path:
            self._acquire_registry_lock()
        try:
            if self.registry_path and os.path.exists(self.registry_path):
                with open(self.registry_path, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("version") != 1:
                    raise ValueError(
                        f"unsupported registry version: {data.get('version')!r}")
                self._branches = {b["branch_id"]: Branch.from_dict(b)
                                  for b in data.get("branches", [])}
                self._loaded_branch_ids = set(self._branches)
                numeric_suffixes = [
                    int(branch_id.rsplit("/", 1)[1])
                    for branch_id in self._branches
                    if "/" in branch_id
                    and branch_id.rsplit("/", 1)[1].isdigit()
                ]
                self._seq = max(numeric_suffixes, default=0)
                if self._branches:
                    _log.info("registry %s: loaded %d branch(es) from a "
                              "previous owner", self.registry_path,
                              len(self._branches))
        except BaseException:
            self._release_registry_lock()
            raise
        if reap_interval_s is not None:
            self.start_reaper(reap_interval_s)

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

    def _ensure_open(self, *, allow_closing: bool = False) -> None:
        """A closed orchestrator gave up registry ownership; letting it keep
        writing would corrupt a successor's registry. Reads stay allowed."""
        if self._closed:
            raise RuntimeError("orchestrator is closed")
        if self._closing and not allow_closing:
            raise RuntimeError("orchestrator is closing")

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
        previous = self._branches.get(branch.branch_id)
        self._branches[branch.branch_id] = branch
        try:
            self._persist()
        except BaseException:
            if previous is None:
                self._branches.pop(branch.branch_id, None)
            else:
                self._branches[branch.branch_id] = previous
            raise

    @locked
    def _forget(self, branch_id: str) -> None:
        previous = self._branches.pop(branch_id, None)
        try:
            self._persist()
        except BaseException:
            if previous is not None:
                self._branches[branch_id] = previous
            raise
        self._loaded_branch_ids.discard(branch_id)

    def _replace_branch(self, branch_id: str, **changes) -> Branch:
        previous = self._branches[branch_id]
        current = replace(previous, **changes)
        self._branches[branch_id] = current
        try:
            self._persist()
        except BaseException:
            self._branches[branch_id] = previous
            raise
        return current

    def _next_child_id(self, parent_id: str) -> str:
        while True:
            self._seq += 1
            child_id = f"{parent_id}/{self._seq}"
            if child_id not in self._branches:
                return child_id

    def _rollback_branch(self, branch_id: str) -> None:
        """Attempt both backend cleanups and forget only after both succeed."""
        errors = []
        for cleanup in (self.kv.kill, self.sandbox.kill):
            try:
                cleanup(branch_id)
            except BaseException as exc:
                errors.append(exc)
                _log.warning("rollback cleanup failed for %s", branch_id,
                             exc_info=True)
        if not errors:
            with self._lock:
                self._forget(branch_id)

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
            if tokens and getattr(self.kv, "external_data_path", False):
                raise RuntimeError(
                    "this KV backend is populated by inference requests; "
                    "create the branch without tokens and call generate()")
            if branch_id in self._branches:
                raise ValueError(f"branch exists: {branch_id}")
            branch = Branch(branch_id, None, _STATE_FORKING, self._clock(),
                            self._lease_expiry(lease_s))
            self._record(branch)
            self._spawning.add(branch_id)
        try:
            try:
                self.kv.create_tree(branch_id)
                self.sandbox.spawn(branch_id, None)
                sandbox_wait_ready = getattr(self.sandbox, "wait_ready", None)
                if sandbox_wait_ready is not None:
                    sandbox_wait_ready(branch_id)
                if tokens:
                    self.kv.extend(branch_id, tokens)
                with self._lock:
                    branch = self._replace_branch(
                        branch_id, state=_STATE_LIVE)
            except BaseException:
                _log.warning("create_parent %s failed; rolling back",
                             branch_id, exc_info=True)
                self._rollback_branch(branch_id)
                raise
        finally:
            with self._lifecycle_changed:
                self._spawning.discard(branch_id)
                self._lifecycle_changed.notify_all()
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
                    child_id = self._next_child_id(parent_id)
                if child_id in self._branches or any(
                        c.branch_id == child_id for c in children):
                    raise ValueError(f"branch exists: {child_id}")
                children.append(Branch(child_id, parent_id, _STATE_FORKING,
                                       self._clock(), self._lease_expiry(lease_s)))
            for branch in children:
                self._branches[branch.branch_id] = branch
            try:
                self._persist()
            except BaseException:
                for branch in children:
                    self._branches.pop(branch.branch_id, None)
                raise
            for branch in children:
                self._spawning.add(branch.branch_id)

        def spawn_one(branch: Branch) -> Branch:
            child_id = branch.branch_id
            try:
                try:
                    self.kv.fork_branch(parent_id, child_id)
                    self.sandbox.spawn(child_id, parent_id)
                    sandbox_wait_ready = getattr(
                        self.sandbox, "wait_ready", None)
                    if sandbox_wait_ready is not None:
                        sandbox_wait_ready(child_id)
                    with self._lock:
                        parent = self._branches.get(parent_id)
                        if parent is None or parent.state != _STATE_LIVE:
                            raise RuntimeError(
                                f"parent {parent_id} died while forking "
                                f"{child_id}")
                        branch = self._replace_branch(
                            child_id, state=_STATE_LIVE)
                except BaseException:
                    _log.warning("fork of %s from %s failed; rolling back",
                                 child_id, parent_id, exc_info=True)
                    self._rollback_branch(child_id)
                    raise
            finally:
                with self._lifecycle_changed:
                    self._spawning.discard(child_id)
                    self._lifecycle_changed.notify_all()
            return branch

        forked = self._fan_out(children, spawn_one)
        with self._lock:
            self.metrics.forks += len(forked)
        _log.info("forked %d child(ren) of %s", len(forked), parent_id)
        return forked

    def extend(self, branch_id: str, tokens: list[int]) -> int:
        with self._lock:
            self._ensure_open()
            branch = self._branches.get(branch_id)
            if branch is None or branch.state != _STATE_LIVE:
                raise KeyError(f"no live branch: {branch_id}")
        return self.kv.extend(branch_id, tokens)

    def generate(
        self,
        branch_id: str,
        prompt: str | list[int],
        sampling_params: dict,
        *,
        branch_end: bool = False,
        reserve_tokens: int | None = None,
    ) -> dict:
        """Submit inference through a KV backend with an external data path.

        The backend supplies the branch namespace and parent metadata, keeping
        inference and lifecycle operations on the same branch identity.
        """
        with self._lock:
            self._ensure_open()
            branch = self._branches.get(branch_id)
            if branch is None or branch.state != _STATE_LIVE:
                raise KeyError(f"no live branch: {branch_id}")
            backend_generate = getattr(self.kv, "generate", None)
            if backend_generate is None:
                raise RuntimeError(
                    f"KV backend {type(self.kv).__name__} does not support "
                    "inference requests")
        result = backend_generate(
            branch_id,
            prompt,
            sampling_params,
            branch_end=branch_end,
            reserve_tokens=reserve_tokens,
        )
        if branch_end:
            self.kill(branch_id)
        return result

    def exec(self, branch_id: str, argv: list[str],
             timeout_s: float | None = None, stdin: bytes | None = None):
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
            self.metrics.execs += 1
        return sandbox_exec(branch_id, argv, timeout_s, stdin=stdin)

    def exec_detached(self, branch_id: str, argv: list[str]):
        """Start a background process in the branch's sandbox, for
        backends that expose ``exec_detached`` (``FirecrackerSandbox``
        does). Returns the backend's handle (guest pid + log path)."""
        with self._lock:
            self._ensure_open()
            branch = self._branches.get(branch_id)
            if branch is None or branch.state != _STATE_LIVE:
                raise KeyError(f"no live branch: {branch_id}")
            start = getattr(self.sandbox, "exec_detached", None)
            if start is None:
                raise RuntimeError(
                    f"sandbox backend {type(self.sandbox).__name__} does not "
                    "support exec_detached")
            self.metrics.execs += 1
        return start(branch_id, argv)

    def kill(self, branch_id: str) -> KillReceipt:
        return self._kill(branch_id, allow_closing=False)

    def _kill(
        self, branch_id: str, *, allow_closing: bool
    ) -> KillReceipt:
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
            self._ensure_open(allow_closing=allow_closing)
            if (branch_id not in self._branches
                    or branch_id in self._killing
                    or branch_id in self._spawning):
                return KillReceipt(branch_id, 0, reaped=False)
            self._killing.add(branch_id)
            try:
                self._replace_branch(branch_id, state=_STATE_KILLING)
            except BaseException:
                self._killing.discard(branch_id)
                self._lifecycle_changed.notify_all()
                raise
        try:
            self.sandbox.kill(branch_id)
            freed = self.kv.kill(branch_id)
        except BaseException:
            _log.warning("kill of %s failed; record stays journaled for "
                         "reconcile()", branch_id, exc_info=True)
            with self._lifecycle_changed:
                self._killing.discard(branch_id)
                self.metrics.kill_failures += 1
                self._lifecycle_changed.notify_all()
            raise  # record stays in state "killing"; reconcile() retries
        with self._lifecycle_changed:
            self._killing.discard(branch_id)
            self.metrics.kills += 1
            try:
                self._forget(branch_id)
            finally:
                self._lifecycle_changed.notify_all()
        _log.debug("killed %s (freed %d KV tokens)", branch_id, freed)
        return KillReceipt(branch_id, freed)

    def kill_losers(self, winner_id: str) -> list[KillReceipt]:
        """Kill every live branch except the winner and its ancestor chain.

        A loser whose fork is still in flight on another thread is skipped
        by ``kill()`` (the fork wins the race), so after the first sweep any
        loser that survived is waited for and killed again — the sweep does
        not return until every branch recorded at entry is gone. Branches
        forked *after* entry are not this call's problem.
        """
        with self._lock:
            self._ensure_open()
            winner = self._branches.get(winner_id)
            if winner is None or winner.state != _STATE_LIVE:
                raise KeyError(f"no live branch: {winner_id}")
            keep = set()
            cursor: str | None = winner_id
            while cursor is not None and cursor not in keep:
                keep.add(cursor)
                branch = self._branches.get(cursor)
                cursor = branch.parent_id if branch else None
            losers = [bid for bid in self._branches if bid not in keep]
        # keyed by branch so a retry's real receipt replaces the sweep's
        # no-op, never both — exactly one receipt per loser
        receipts = {r.branch_id: r for r in self._fan_out(losers, self.kill)}
        for branch_id in losers:
            while not receipts.get(branch_id,
                                   KillReceipt(branch_id, 0, False)).reaped:
                with self._lock:
                    if branch_id not in self._branches:
                        break  # gone (killed by us or a concurrent killer)
                    settling = (branch_id in self._spawning
                                or branch_id in self._killing)
                if settling:
                    time.sleep(0.01)  # spawn/kill in flight; it will settle
                    continue
                receipts[branch_id] = self.kill(branch_id)
        return list(receipts.values())

    # -- collection ----------------------------------------------------------

    @locked
    def renew_lease(self, branch_id: str, lease_s: float) -> Branch:
        self._ensure_open()
        branch = self._branches.get(branch_id)
        if branch is None or branch.state != _STATE_LIVE:
            raise KeyError(f"no live branch: {branch_id}")
        if lease_s <= 0:
            raise ValueError("lease_s must be positive")
        return self._replace_branch(
            branch_id, lease_expires_at=self._clock() + lease_s)

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
        receipts = self._fan_out(expired, self.kill)
        with self._lock:
            self.metrics.reaped_expired += sum(1 for r in receipts if r.reaped)
        return receipts

    def reconcile(self) -> list[KillReceipt]:
        """Collect leaked branches: mid-fork and mid-kill leftovers from a
        crashed or failed supervisor, every row loaded from a previous owner,
        plus anything past its lease. Loaded rows are cleaned instead of
        adopted because backend handles and process-local KV state cannot be
        reconstructed generically. Not run automatically; call it at startup.
        Branches with work in flight in *this* process are left alone."""
        with self._lock:
            self._ensure_open()
            stuck = [b.branch_id for b in self._branches.values()
                     if b.state in (_STATE_FORKING, _STATE_KILLING)
                     and b.branch_id not in self._spawning
                     and b.branch_id not in self._killing]
            # Registry rows loaded from a previous owner cannot safely be
            # adopted: the reference KV cache is process-local, remote
            # adapters may have lost bookkeeping, and sandbox handles are not
            # generally reconstructable. Reconcile them by replaying kill.
            stuck.extend(
                branch_id for branch_id in self._loaded_branch_ids
                if branch_id not in stuck
                and branch_id not in self._spawning
                and branch_id not in self._killing
            )
            now = self._clock()
            expired = [
                branch.branch_id for branch in self._branches.values()
                if branch.lease_expires_at is not None
                and branch.lease_expires_at <= now
                and branch.branch_id not in self._spawning
                and branch.branch_id not in self._killing
                and branch.branch_id not in stuck
            ]
        if stuck:
            _log.info("reconcile: collecting %d branch(es) left mid-fork or "
                      "mid-kill: %s", len(stuck), stuck)
        if expired:
            _log.info("reaping %d branch(es) with lapsed leases: %s",
                      len(expired), expired)
        candidates = stuck + expired
        try:
            receipts = self._fan_out(candidates, self.kill)
        finally:
            with self._lock:
                self.metrics.reconciles += 1
        with self._lock:
            expired_ids = set(expired)
            self.metrics.reaped_expired += sum(
                1 for r in receipts
                if r.reaped and r.branch_id in expired_ids)
        return receipts

    # -- background collection -------------------------------------------------

    def start_reaper(self, interval_s: float = 5.0) -> None:
        """Run ``reconcile()`` (which includes lease reaping) every
        ``interval_s`` seconds on a daemon thread, and collect branches
        whose sandbox died out from under them (backends may expose
        ``sweep_dead() -> list[branch_id]``, as ``FirecrackerSandbox``
        does). Errors are logged, never fatal to the loop. Idempotent;
        ``close()`` stops it."""
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        with self._lock:
            self._ensure_open()
            if self._reaper_thread is not None and self._reaper_thread.is_alive():
                return
            # the loop captures its own stop event, so a later start_reaper
            # reassigning self._reaper_stop can't strand this thread waiting
            # on an event nobody will set
            stop = self._reaper_stop = threading.Event()
            self._reaper_thread = threading.Thread(
                target=self._reaper_loop, args=(interval_s, stop),
                name="agentfork-reaper", daemon=True)
            self._reaper_thread.start()

    def stop_reaper(self) -> None:
        with self._lock:
            thread, self._reaper_thread = self._reaper_thread, None
            self._reaper_stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def _reaper_loop(self, interval_s: float, stop: threading.Event) -> None:
        while not stop.wait(interval_s):
            try:
                self.reconcile()
                sweep = getattr(self.sandbox, "sweep_dead", None)
                if sweep is not None:
                    for branch_id in sweep():
                        _log.warning("branch %s: sandbox died; collecting",
                                     branch_id)
                        if self.kill(branch_id).reaped:
                            with self._lock:
                                self.metrics.swept_dead += 1
            except RuntimeError:
                # only a "closed" RuntimeError should stop the loop; a
                # transient backend/thread-pool RuntimeError must not kill
                # the reaper silently on a healthy orchestrator
                if self._closed:
                    return
                _log.exception("background reaper pass failed")
            except Exception:
                _log.exception("background reaper pass failed")

    def metrics_snapshot(self) -> dict:
        """A consistent copy of the counters, as a plain dict."""
        with self._lock:
            return dict(vars(self.metrics))

    # -- introspection / teardown ---------------------------------------------

    @locked
    def branches(self) -> list[Branch]:
        return list(self._branches.values())

    def alive(self, branch_id: str) -> bool:
        with self._lock:
            branch = self._branches.get(branch_id)
            if branch is None or branch.state != _STATE_LIVE:
                return False
        if not self.sandbox.alive(branch_id):
            return False
        kv_has_tree = getattr(self.kv, "has_tree", None)
        return kv_has_tree(branch_id) if kv_has_tree is not None else True

    def close(self) -> None:
        """Kill every recorded branch; raise the first error after trying
        all. The orchestrator ends closed either way: registry ownership is
        released so a successor can take over, and every mutating method
        raises from then on. Idempotent. New lifecycle calls are refused while
        closing, and already in-flight spawns/kills are drained before the
        final cleanup sweep."""
        self.stop_reaper()
        with self._lifecycle_changed:
            if self._closed:
                return
            if self._closing:
                while not self._closed:
                    self._lifecycle_changed.wait()
                return
            self._closing = True
            while self._spawning or self._killing:
                self._lifecycle_changed.wait()
            branch_ids = list(self._branches)
        error = None
        try:
            try:
                self._fan_out(
                    branch_ids,
                    lambda branch_id: self._kill(
                        branch_id, allow_closing=True),
                )
            except Exception as exc:
                error = exc
        finally:
            with self._lifecycle_changed:
                self._closed = True
                self._closing = False
                self._release_registry_lock()
                self._lifecycle_changed.notify_all()
        if error is not None:
            raise error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
