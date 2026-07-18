# agentfork

agentfork is a runtime for tree-style agent fanout.

It forks a live agent's sandbox and its LLM KV context together, as one
branch. Killing a branch reclaims both halves in <1ms, with no orphan
processes and no leaked KV pages.

![tree-keyed KV: one resident prefix, N logical branches](docs/img/kv-dedup.svg)

**Measured at a glance:** 22 ms for 10 create+extend operations on an A10 ·
9.65× fewer KV slots than an explicitly unshared allocation · 547-line
additive SGLang patch.

## What it does

agentfork implements two runtime operations, both given the same branch ID by
`ForkOrchestrator`:

- **`fork(parent)`** creates a child that shares the parent's cached context and
  runs in its own sandbox.
- **`kill(child)`** stops the sandbox and releases the child's KV state.

It is not an agent framework or a scheduler: it does not decide what an agent
does, only how a branch of it is created and torn down.

By default, `ForkOrchestrator` drives the reference backends: `TreeKVCache`
for KV state and `ReaperSandbox` for the process. Adapters for the SGLang
cache patch and a Firecracker sandbox (`SGLangKVBackend`, `FirecrackerSandbox`)
satisfy the same `KVBackend`/`SandboxBackend` protocols. The Firecracker
adapter runs end to end against real microVMs (`demo/fc_demo.py`); the SGLang
adapter is mock-tested only.

Use it for:

- Cloud agent platforms that
  [map/reduce](https://github.com/sachinkesiraju/agent-mapreduce) one task
  across N parallel attempts: fork from one prepared context, keep the winner.
- Coding agents that try several fixes from one repository context.
- Verification trees that run cheap checks first and kill the failures before
  anything expensive runs.
- Search and planning agents that fork several next steps from the same state.
- Evaluations that reuse one cached context across policies or seeds.

## Example: tree-style agent fanout

A coding agent has read a 32k-token repository, reproduced a bug, and prepared
its build environment. It wants to try 10 fixes.

Forking the agent at that point gives each child the same cached context and
sandbox state, so each child pays only for its own fix. Cheap checks run
first, and branches that fail formatting, compilation, or focused tests are
killed immediately. Without forking, the agent would boot 10 cold sessions and
re-read the repository 10 times.

The strongest version is a tree, not a flat best-of-N batch: if two fixes
survive, fork each again for race tests, performance tests, or independent
review. Those grandchildren inherit the root context plus their candidate's
changes, and the full test suite runs only on the finalists.

![agentfork lifecycle: fork a live agent, race the branches, kill the losers](docs/img/lifecycle.svg)

The fanout cost changes from

```
N × (shared setup + branch work)
```

to

```
shared setup + sum(branch work)
```

Forking pays off when setup is expensive, branches are short, and most of them
are killed early. It pays off less when there are few branches or when most of
the work happens after the fork.

## Quickstart

Requires Python 3.10 or newer. The demo also requires Linux 5.4 or newer.

```bash
git clone https://github.com/sachinkesiraju/agentfork.git
cd agentfork
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python demo/demo.py
```

The CPU-only demo forks a 32k-token parent into ten branches, kills the losing
branches, and verifies that no processes or cache entries leak. It uses token
IDs and sleeping processes, so it does not need a model, GPU, or microVM. A
successful run ends with `CLEAN`.

Minimal Python API:

```python
import sys

from agentfork import ForkOrchestrator, ReaperSandbox

sandbox = ReaperSandbox([sys.executable, "-c", "import time; time.sleep(60)"])

with ForkOrchestrator(sandbox=sandbox, registry_path="branches.json",
                      default_lease_s=600) as orch:
    orch.reconcile()  # collect anything a previous crashed run left behind
    orch.create_parent("parent", tokens=list(range(32_000)))
    children = orch.fork("parent", n=3)

    for i, child in enumerate(children):
        start = 1_000_000 + i * 500
        orch.extend(child.branch_id, list(range(start, start + 500)))

    orch.kill_losers(children[0].branch_id)
```

`kill_losers()` keeps the selected branch and its ancestors and cleans up every
other branch. Run `pytest -q` to execute the test suite.

## How it works

`ForkOrchestrator` gives the sandbox and KV branch one ID, records intent in a
single-owner, fsynced registry, rolls back partial forks, retries interrupted
cleanup, and bounds every branch with a lease.

**The reference path runs today.** `TreeKVCache.fork_branch` walks the
parent's cached path and bumps a reference count at each node; no tokens are
copied. `ReaperSandbox` spawns a fresh subprocess for the child. On kill,
`agentfork/kill/reaper.py` reaps the subprocess through Linux `pidfd` and
drops the matching cache entry; the combined path measured 0.53 ms p50.

Two production backends have adapters behind those same protocols:

1. **KV cache fork**: `patches/0001-sglang-tree-radix-cache.patch` adds
   `TreeRadixCache`; `patches/0002-Wire-branch-lifecycle-through-the-SGLang-request-pat.patch`
   carries branch identity through OpenAI/native requests, adds scheduler-side
   lifecycle and quota admission, and exposes `/tree_cache` control operations.
   `SGLangKVBackend` supports in-process use and `SGLangHTTPBackend` drives a
   remote engine. The live request path was validated on a Modal A10G: ten
   children each reused 2,406 parent tokens and explicit kill released the
   remaining parent pin.
2. **Sandbox fork**: `agentfork/sandbox/fc_bench.py` measures Firecracker
   snapshot and restore, with each child sharing the parent's memory pages
   copy-on-write. The recorded parent pause was 76–83 ms including snapshot
   creation, and snapshot-load API time was 2.1 ms p50 per child.
   `agentfork/sandbox/firecracker_backend.py`'s `FirecrackerSandbox` adapts
   `fc_bench`'s `MicroVM` to `SandboxBackend`. Snapshots are taken lazily at
   fork time, so children inherit the parent's current state and unforked
   branches never pay the snapshot write. Each guest gets a vsock exec
   channel (`orch.exec(branch_id, argv)`, served by
   `agentfork/sandbox/guest_agent.py` baked into the rootfs), a
   `wait_ready()` readiness probe, and, with `overlay_mib` set, a writable
   scratch drive that children inherit as their own copy; `JailerConfig`
   runs every VMM chrooted and deprivileged under Firecracker's jailer.
   All of it is validated on real Firecracker v1.16.1: children answer
   exec over vsock, mount and write their own overlays, and inherit state
   the parent wrote after boot (jailed and unjailed). Networking and
   identity regeneration are not handled.

```
ForkOrchestrator  (registry / leases / rollback / reconcile)
        │
        ▼
   coordinated branch ID
   │
   ├── KV branch
   │    ├── TreeKVCache            CPU reference cache (live)
   │    └── TreeRadixCache patch   via SGLangKVBackend (mock-tested only)
   │
   └── sandbox branch
        ├── ReaperSandbox          pidfd subprocess (live)
        └── Firecracker microVMs   via FirecrackerSandbox (live: exec,
                                   overlays, jailer)
```

"Fork" here is not Linux `fork(2)`: CUDA state cannot be duplicated by forking
a process, so nothing in agentfork relies on that. The KV fork is a logical
reference count on shared KV slots inside the cache, not a copy of GPU memory.
Firecracker's copy-on-write is a separate mechanism that shares a VM's guest
memory pages between snapshot and restore; it does not touch CUDA memory
either.

## Measured results

See [report/RESULTS.md](report/RESULTS.md) for full results, assumptions, and
the checks that fail or remain untested.

| Claim | Measured |
|---|---|
| CPU reference prefix reuse (10-way fanout) | 100% of the parent prefix reused; separate trees stayed isolated |
| Patched cache on a real SGLang GPU pool (A10) | 37k occupied slots vs 357k with sharing disabled; 10 create+extend operations in 22 ms; allocator back to 0 after kill-all |
| Stock SGLang live-engine baseline | 2,402–2,403 of 2,404 prompt tokens cached per sibling; this uses stock RadixAttention, not the patch |
| Subprocess + CPU reference-cache kill | 0.53 ms p50 and 1.46 ms max over 100 cycles |
| Supervisor crash test | 0 surviving Python children across 50 runs with 5 children each |
| Firecracker snapshot load | 2.1 ms p50 API time per child; 25 children loaded in 150 ms; full VMM teardown was 31 ms p50 |
| Firecracker host-page sharing | 117.7 MiB total RSS vs 23.8 MiB total PSS across 25 idle VMMs |
| End-to-end orchestrator + real Firecracker (`demo/fc_demo.py`, aarch64 v1.16.1, idle 256 MiB guests) | Root boot 111–165 ms; 10-way fork at 235–317 ms per child, dominated by the ~125 ms per-branch snapshot write; 9 losers killed in 132–231 ms; zero surviving VMMs across 3 runs |
| Data plane + parallel lifecycle on real Firecracker (v0.3.0, same host) | 5-way fork 28–145 ms per child amortized (lazy fork-time snapshot, parallel restores); exec over vsock answered in every child (0.3–0.7 s steady-state, first exec after restore up to ~10 s); per-child overlay mount+write; fork-after-exec freshness and divergence isolation verified; 4 losers killed in 8–12 ms; identical results under the jailer; zero surviving VMMs |
| SGLang patch size | 547 additive lines: 299 implementation and 248 tests |
| 10,000-branch cache test | 0.95 s to create branches and 0.17 s to bulk-kill them; allocator back to 0; this tests cache metadata, not concurrent inference |
| Tree-native cache controls | Direct API tests cover budgets, reservations, demotion, invalidation, and telemetry; the scheduler does not enforce them |

In the [10-child GPU test](patches/real_pool_validation.py), sharing reduced KV
usage from 357k slots to 37k. Stock SGLang already shares cached prefixes, so
agentfork adds branch tracking and cleanup, not lower memory use.

Provider-cache numbers are estimates based on assumed prices, not real-world
measurements.

## Running benchmarks

```bash
pytest -q
python demo/demo.py
python -m agentfork.bench.kill_bench --cycles 100
python -m agentfork.bench.crash_bench --cycles 50 --children 5
python -m agentfork.bench.cost_model --children 10 --prefix 32000 --suffix 2000

# Direct SGLang cache validation:
export SGLANG_DIR=/path/to/sglang
git -C "$SGLANG_DIR" checkout 40517b593b23870cf351a05a1d53e930cea6a58d
git -C "$SGLANG_DIR" apply "$PWD/patches/0001-sglang-tree-radix-cache.patch"
PYTHONPATH="$SGLANG_DIR/python" python patches/real_pool_validation.py
PYTHONPATH="$SGLANG_DIR/python" python patches/scale_10k_branch_validation.py
PYTHONPATH="$SGLANG_DIR/python" python patches/tree_native_features_validation.py

# Firecracker (requires /dev/kvm, Firecracker, a guest kernel, and a rootfs):
python -m agentfork.sandbox.fc_bench --fc ./firecracker --kernel vmlinux --rootfs rootfs.ext4
python demo/fc_demo.py --fc ./firecracker --kernel vmlinux --rootfs rootfs.ext4  # full lifecycle through ForkOrchestrator

# GPU validation (requires Modal and the patched SGLang checkout):
pip install modal
SGLANG_DIR="$SGLANG_DIR" modal run modal_gpu_validation.py
```

## Limitations

- The SGLang branch request path is live and quota reservations are enforced
  before queue admission, but it has only been measured on one A10G with a
  0.6B model. Cross-worker routing, tensor parallelism, allocator contention,
  and mixed-tenant pressure remain unmeasured.
- The Firecracker adapter has only driven idle guests: no guest networking,
  identity, or readiness probes. Cleanup is retried, not atomic.
- GPU validation used one A10 with a Qwen3-0.6B baseline, and Firecracker
  validation used idle, CPU-only 256 MiB guests. Neither covers production
  scale or GPU-plus-microVM colocation.
- Each component serializes callers behind one coarse lock: concurrent
  threads are safe but parallel forks gain no throughput. The reaper's
  default `PR_SET_PDEATHSIG` backstop uses `preexec_fn`, which CPython
  documents as thread-unsafe; pass `pdeathsig=False` under threaded
  supervisors.
- No winner merge, artifact handoff, hibernation, migration, or resume
  protocol is implemented.

## Why agentfork vs. alternatives

Other projects each branch one piece of this: a sandbox fork, an inference
session, shared-prefix caching, or moving KV caches between tiers. agentfork
gives one identity to both the sandbox and the KV branch, so a single ID
covers ownership and cleanup on both sides.

| Project | What it does | What's missing |
|---|---|---|
| [forkd](https://github.com/deeplethe/forkd) | Forks microVMs from a shared snapshot, copy-on-write | A branch ID that also owns and reclaims the LLM KV cache |
| [SGLang](https://github.com/sgl-project/sglang) RadixAttention, [vLLM](https://github.com/vllm-project/vllm) APC | Automatically reuses KV for requests sharing a prefix | Explicit agent-tree ownership, branch policy, and sandbox coordination |
| [LMCache](https://github.com/LMCache/LMCache), [Mooncake](https://github.com/kvcache-ai/Mooncake), [Dynamo](https://github.com/ai-dynamo/dynamo) | Moves and tiers KV cache across memory and workers | Branch identity and sandbox coordination on top of that movement |
| **agentfork** | Forks a sandbox and its KV cache under one branch ID, and reclaims both on kill | Validating the SGLang adapter against a live engine, giving Firecracker guests networking and readiness, then hosting it as a service |

## License

Apache-2.0. See [LICENSE](LICENSE).
