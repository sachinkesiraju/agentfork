# agentfork

agentfork is a runtime prototype for tree-style agent fanout.

It forks a live agent's sandbox and its LLM KV context together, as one
branch. Killing a branch reclaims both halves in <1ms, with no orphan
processes and no leaked KV pages.

![tree-keyed KV: one resident prefix, N logical branches](docs/img/kv-dedup.svg)

**Measured at a glance:** 22 ms for 10 create+extend operations on an A10 ·
9.65× fewer KV slots than an explicitly unshared allocation · 547-line
additive SGLang patch.

## What it does

agentfork prototypes two runtime operations:

- **`fork(parent)`** creates a child that shares the parent's cached context and
  runs in its own sandbox.
- **`kill(child)`** stops the sandbox and releases the child's KV state.

`ForkOrchestrator` gives both halves the same branch ID. The reference
implementation uses `TreeKVCache`, a CPU model of the KV cache, and
`ReaperSandbox`, fresh subprocesses supervised through `pidfd`. The SGLang
cache patch and the Firecracker snapshot benchmark are tested separately and
are not connected to the orchestrator yet. agentfork is not an agent framework
or a scheduler.

Use it for:

- `map`/`reduce` agent fanout (e.g. [agent-mapreduce](https://github.com/sachinkesiraju/agent-mapreduce)).
- Trying several coding fixes from one repository context.
- Verification trees that kill most branches early.
- Search and planning agents that explore several next steps.
- Evaluations that reuse one prepared context across policies or seeds.

## Example: tree-style agent fanout

A coding agent has read a 32k-token repository, reproduced a bug, and prepared
its build environment. It wants to try 10 fixes.

Fork the agent at that point. Each child inherits the parent's model context
and sandbox state, so it pays only for its own work. Run cheap checks first and
kill branches that fail formatting, compilation, or focused tests. Without
forking, you boot 10 cold sessions and re-read the repository 10 times.

The strongest version is a tree, not a flat best-of-N batch. If two fixes
survive, fork each again for race tests, performance tests, or independent
review. Those grandchildren inherit the root context plus their candidate's
changes. Run the full test suite only on the finalists.

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
every branch with a lease. The production backends remain separate:

1. **KV cache fork**: `patches/0001-sglang-tree-radix-cache.patch` adds
   `TreeRadixCache` to SGLang. Children inherit the parent's KV prefix
   copy-on-write and keep it pinned through the existing `lock_ref` machinery;
   `kill_tree()` releases the branch path. The patch also adds token budgets,
   reservations, demotion, invalidation, and per-tree telemetry. No scheduler,
   model-runner, server, or router changes are included.
2. **Sandbox fork**: `agentfork/sandbox/fc_bench.py` independently drives
   Firecracker snapshot and restore with page-level copy-on-write. The recorded
   parent pause was 76–83 ms including snapshot creation; snapshot-load API
   time was 2.1 ms p50 per child.
3. **Process + reference-cache kill**: `agentfork/kill/reaper.py` uses Linux
   `pidfd` to reap a subprocess, then drops the matching CPU reference-cache
   entry. The measured combined path was 0.53 ms p50; this is not a
   Firecracker + GPU kill measurement.

```
ForkOrchestrator  (registry / leases / rollback / reconcile)
        │
        ▼
   coordinated branch ID
   ├── TreeRadixCache patch    fork_branch / kill_tree (not yet a backend)
   ├── Firecracker benchmark   snapshot / load / kill (not yet a backend)
   └── ReaperSandbox           pidfd process + CPU reference-cache backend
```

CUDA memory cannot be forked with `fork(2)`. The KV fork is a logical reference
to shared KV slots, not an OS-level copy of GPU state. Firecracker
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

The 9.65× figure compares shared KV against an allocation that stores a
separate copy of the prefix for every child. Stock SGLang already avoids that
duplication by sharing one cached prefix, so compute and residency are close
to 1.0× versus a well-run self-hosted prefix cache. What the patch adds on top
is explicit branch ownership, policy, telemetry, and coordinated reclaim, not
further memory savings.

The provider-cache comparison is a pricing model, not a benchmark. It assumes
cached reads cost 0.1× normal input tokens and cache writes cost 1.25×. It does
not measure invoices, latency, or provider memory use.

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
- The SGLang patch is not wired into request scheduling, model execution, HTTP
  serving, or a multi-worker router. The live-engine result above is a stock
  RadixAttention baseline, not an end-to-end patch test.
- Cache budgets and reservations are accounting only; the scheduler does not
  enforce them as physical GPU memory reservations.
- GPU validation used one A10; 70B-scale models, tensor parallelism, and
  scheduler contention are unmeasured.
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

Other projects branch one piece of this: a fast sandbox fork, an inference
session, shared-prefix caching, or moving KV caches between tiers. agentfork
gives one branch identity to both sandbox state and KV ownership. This
repository validates the pieces of that design but does not yet integrate them
into a production runtime.

| Project | What it covers | What it leaves open |
|---|---|---|
| [forkd](https://github.com/deeplethe/forkd), [Mitos](https://github.com/mitos-run/mitos) | microVM fork with copy-on-write | Binding sandbox identity to inference KV ownership and reclaim |
| [thaw](https://github.com/thaw-ai/thaw), [processfork](https://github.com/manav8498/processfork) | inference/session branching experiments | Pairing the inference branch with an isolated sandbox lifecycle |
| [SGLang](https://github.com/sgl-project/sglang) RadixAttention, [vLLM](https://github.com/vllm-project/vllm) APC | automatic shared-prefix KV reuse | Explicit agent-tree ownership, branch policy, and sandbox coordination |
| [LMCache](https://github.com/LMCache/LMCache), [Mooncake](https://github.com/kvcache-ai/Mooncake), [Dynamo](https://github.com/ai-dynamo/dynamo) | KV transfer and tiering | Composing tree ownership with movement and routing |
| **agentfork (this prototype)** | tree-aware SGLang patch, CPU reference cache and reaper, separate Firecracker benchmark | Integrating all three paths behind one recoverable control-plane operation |

## License

Apache-2.0. See [LICENSE](LICENSE).
