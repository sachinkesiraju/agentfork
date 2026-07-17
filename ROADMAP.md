# Roadmap to a production fork-native runtime

Milestone 1 ships in this repository. Milestones 2–4 are the work that cannot
be finished here: they require a maintained SGLang integration, GPU fleets,
and real fanout workloads. Ordering reflects risk: engine integration is the
long pole and gates everything downstream.

## M1 — Control plane and stable API (this repository)

- [x] `ForkOrchestrator`: one branch ID spans the KV branch and sandbox, with
      per-child rollback on partial fork failure.
- [x] Persistent JSON registry with journaled intent, leases,
      `reap_expired()`, and `reconcile()` for crash recovery. Cleanup is
      at-least-once, not atomic.
- [x] Pluggable `SandboxBackend` protocol; `ReaperSandbox` (Linux pidfd
      subprocesses) and `NullSandbox` (KV-only) implementations.
- [x] Stable public API at `agentfork.*`, `__version__`, `py.typed`.
- [x] Hardened reaper and cache error paths with unit tests.

Remaining in scope for M1.x: thread-safety for concurrent callers (the
components are single-controller today), and replacing `preexec_fn` in the
reaper with a mechanism safe under threaded supervisors.

## M2 — Engine integration (requires an SGLang checkout and GPUs)

- [ ] Propagate `tree_id`/`branch_id` from the HTTP server through the
      scheduler and model runner to `TreeRadixCache`; today no request path
      reaches the patch.
- [ ] Enforce quotas and reservations ahead of allocator calls in the
      scheduler; they are accounting-only hooks now.
- [ ] An engine-backed `KVBackend` for `ForkOrchestrator` speaking to the
      patched cache over the server API.
- [ ] Validate end-to-end branch requests on a live engine (the current live
      result is a stock RadixAttention baseline), then beyond one A10:
      70B-class models, tensor parallelism, allocator pressure.
- [ ] Decide upstream-vs-fork: the patch is pinned to `40517b593` and will
      rot without an owner.

## M3 — Sandbox runtime

- [ ] Promote `fc_bench` into an `FCForker` backend: snapshot storage and
      distribution, network/identity regeneration, writable overlays, guest
      readiness probes — or adapt a provider that already ships VM
      snapshot/branch (Modal, E2B, Morph) as a `SandboxBackend`.
- [ ] Supervise VMMs, not just `subprocess.Popen` children.
- [ ] Colocate or proxy sandbox and inference so both halves of a branch are
      reachable from one control plane.

## M4 — Evidence the lifecycle pays

- [ ] Multi-tenant contention benchmarks: eviction pressure, noisy neighbors,
      quota enforcement — the regime where explicit ownership should beat
      best-effort caching (the corrected cost model is ~1.0× without it).
- [ ] A workload census on a real fanout workload; the recorded census of
      organic sessions failed the fanout/prefix thresholds and measured
      visible prompt text only, which lower-bounds the true KV prefix.
- [ ] A measured end-to-end cost study (orchestration, idle capacity,
      snapshot storage, network) instead of token-price arithmetic.
- [ ] Winner merge / durable artifact handoff protocol.
