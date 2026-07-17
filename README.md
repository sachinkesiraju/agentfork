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

The reference backends, `TreeKVCache` and `ReaperSandbox`, run today. The
SGLang cache patch and the Firecracker snapshot benchmark are validated
separately and not yet wired into the orchestrator; see "How it works" below
for the mechanism.

Use it for:

- `map`/`reduce` agent fanout (e.g. [agent-mapreduce](https://github.com/sachinkesiraju/agent-mapreduce)).
- Coding agents that try several fixes from one repository context.
- Verification trees that run cheap checks first and kill branches before the
  expensive ones.
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

Forking pays off when setup is expensive, branches are short, and most branches
die early. It helps little when there are few branches or when most work happens
after the fork.

## Quickstart

```bash
pip install -e ".[dev]"
python demo/demo.py   # Linux, CPU-only reference demo
pytest -q             # non-Linux hosts skip pidfd integration tests
```

The demo does not run a model or a microVM. Integer token IDs stand in for KV
cache entries, and sleeping Python processes stand in for sandboxes. One parent
owns a 32k-token prefix; ten children share that prefix with no re-prefill, add
their own suffixes, and are then killed. The demo ends with zero live trees and
zero resident cache tokens.

The same lifecycle through the Python API (`agentfork.*` is the stable public
surface from 0.2.0; submodule internals are not):

```python
import sys

from agentfork import ForkOrchestrator, ReaperSandbox

prefix_tokens = list(range(32_000))
sandbox = ReaperSandbox([sys.executable, "-c", "import time; time.sleep(60)"])

with ForkOrchestrator(sandbox=sandbox, registry_path="branches.json",
                      default_lease_s=600) as orch:
    orch.create_parent("parent", tokens=prefix_tokens)
    children = orch.fork("parent", n=10)

    for i, child in enumerate(children):
        start = 1_000_000 + i * 500
        orch.extend(child.branch_id, list(range(start, start + 500)))

    orch.kill_losers(children[0].branch_id)
```

`kill()` reaps the sandbox, then the KV branch, then removes the registry
record. The steps are sequential, not atomic; `reconcile()` retries work a
failed or crashed supervisor left behind.

**Compatibility:** Python ≥ 3.10; Linux ≥ 5.4 for the `pidfd` reaper; SGLang @
`40517b593b23870cf351a05a1d53e930cea6a58d` for the patch. Firecracker v1.7 and
an NVIDIA A10 on Modal are the measured environments.

## How it works

`ForkOrchestrator` gives the sandbox and KV branch one ID, records intent in a
registry, rolls back partial forks, retries interrupted cleanup, and bounds
every branch with a lease.

**The reference path runs today.** `TreeKVCache.fork_branch` walks the
parent's cached path and bumps a reference count at each node; no tokens are
copied. `ReaperSandbox` spawns a fresh subprocess for the child. On kill,
`agentfork/kill/reaper.py` reaps the subprocess through Linux `pidfd` and
drops the matching cache entry; the combined path measured 0.53 ms p50.

Two production backends are validated on their own and not yet wired into the
orchestrator:

1. **KV cache fork**: `patches/0001-sglang-tree-radix-cache.patch` adds
   `TreeRadixCache` to SGLang. Children inherit the parent's KV prefix
   copy-on-write and stay pinned through SGLang's existing `lock_ref`
   machinery; `kill_tree()` releases a branch's pins. The patch also adds
   token budgets, reservations, demotion, invalidation, and per-tree
   telemetry. No scheduler, model-runner, server, or router changes are
   included.
2. **Sandbox fork**: `agentfork/sandbox/fc_bench.py` measures Firecracker
   snapshot and restore, with each child sharing the parent's memory pages
   copy-on-write. The recorded parent pause was 76–83 ms including snapshot
   creation, and snapshot-load API time was 2.1 ms p50 per child.

```
ForkOrchestrator  (registry / leases / rollback / reconcile)
        │
        ▼
   coordinated branch ID
   ├── ReaperSandbox           pidfd process + CPU reference-cache backend (live)
   ├── TreeRadixCache patch    fork_branch / kill_tree (validated, not wired in)
   └── Firecracker benchmark   snapshot / load / kill (validated, not wired in)
```

CUDA memory cannot be forked with `fork(2)`. The KV fork is a logical
reference to shared KV slots, not an OS-level copy of GPU state. Firecracker
copy-on-write applies to guest memory, not CUDA memory.

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
| SGLang patch size | 547 additive lines: 299 implementation and 248 tests |
| 10,000-branch cache test | 0.95 s to create branches and 0.17 s to bulk-kill them; allocator back to 0; this tests cache metadata, not concurrent inference |
| Tree-native cache controls | Direct API tests cover budgets, reservations, demotion, invalidation, and telemetry; the scheduler does not enforce them |

**On the 9.65× figure:** it compares shared KV against an allocation that
stores a separate copy of the prefix for every child. Stock SGLang already
avoids that duplication by sharing one cached prefix, so compute and residency
are close to 1.0× against a well-run self-hosted prefix cache. What the patch
adds on top is explicit branch ownership, policy, telemetry, and coordinated
reclaim, not further memory savings.

**On the provider-cache comparison:** it is a pricing model, not a benchmark.
It assumes cached reads cost 0.1× normal input tokens and cache writes cost
1.25×, and it does not measure invoices, latency, or provider memory use.

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

# GPU validation (requires Modal and the patched SGLang checkout):
pip install modal
SGLANG_DIR="$SGLANG_DIR" modal run modal_gpu_validation.py
```

## Limitations

- The orchestrator coordinates the CPU reference cache and fresh subprocesses.
  No production backend spans a Firecracker microVM and the patched SGLang
  cache; cleanup is recorded and retried, but sequential rather than atomic.
- The registry is a JSON file written with `open()` and `os.replace()`; it has
  no `fsync` and no cross-process locking, so only one orchestrator should own
  a given registry file at a time.
- The SGLang patch is not wired into request scheduling, model execution, HTTP
  serving, or a multi-worker router. The live-engine result above is a stock
  RadixAttention baseline, not an end-to-end patch test.
- Cache budgets and reservations are accounting only; the scheduler does not
  enforce them as physical GPU memory reservations.
- GPU validation used one A10 and a Qwen3-0.6B stock-engine baseline; 70B-scale
  models, tensor parallelism, and scheduler contention are unmeasured.
- The Modal validation script pulls `lmsysorg/sglang:latest`, not a
  digest-pinned image, so a GPU run can pick up an unreviewed base-image
  change.
- Firecracker measurements used idle, CPU-only 256 MiB guests; microVMs and GPU
  inference have not been colocated, and guest readiness was not measured.
- The components do not synchronize concurrent callers, and the reaper uses
  `preexec_fn`, which Python warns is unsafe in threaded programs.
- No winner merge, artifact handoff, hibernation, migration, or resume protocol
  is implemented.
- The workload-shape check failed on private organic traces that are not
  included in the repository.
- Provider-cache ratios are modeled rather than measured.

## Why agentfork vs. alternatives

Other projects each branch one piece of this: a sandbox fork, an inference
session, shared-prefix caching, or moving KV caches between tiers. agentfork
gives one identity to both the sandbox and the KV branch, so a single ID
covers ownership and cleanup on both sides.

| Project | What it does | What's missing |
|---|---|---|
| [forkd](https://github.com/deeplethe/forkd), [Mitos](https://github.com/mitos-run/mitos) | Forks microVMs from a shared snapshot, copy-on-write | A branch ID that also owns and reclaims the LLM KV cache |
| [thaw](https://github.com/thaw-ai/thaw), [processfork](https://github.com/manav8498/processfork) | Branches an inference session across generations | An isolated sandbox lifecycle paired with that branch |
| [SGLang](https://github.com/sgl-project/sglang) RadixAttention, [vLLM](https://github.com/vllm-project/vllm) APC | Automatically reuses KV for requests sharing a prefix | Explicit agent-tree ownership, branch policy, and sandbox coordination |
| [LMCache](https://github.com/LMCache/LMCache), [Mooncake](https://github.com/kvcache-ai/Mooncake), [Dynamo](https://github.com/ai-dynamo/dynamo) | Moves and tiers KV cache across memory and workers | Branch identity and sandbox coordination on top of that movement |
| **agentfork** | A tree-keyed SGLang patch, a CPU reference cache and reaper, and a separate Firecracker benchmark | Wiring all three paths into one recoverable fork/kill operation |

## License

Apache-2.0. See [LICENSE](LICENSE).
