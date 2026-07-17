# agentfork

agentfork is a runtime prototype for tree-style agent fanout.

It gives every branch one ID across its sandbox and LLM KV cache, so the control
plane can fork, track, and clean up both through the same lifecycle.

![tree-keyed KV: one resident prefix, N logical branches](docs/img/kv-dedup.svg)

**Measured at a glance:** A10 direct-cache API: 22 ms for 10 create+extend
operations and 9.65× fewer occupied KV slots vs unshared · Linux subprocess +
CPU reference cache: 0.53 ms p50 kill · 547-line additive SGLang patch.

## What it does

agentfork implements two runtime operations:

- **`fork(parent)`** creates a child KV branch and starts its sandbox.
- **`kill(child)`** stops the sandbox and releases the branch's KV state.

`ForkOrchestrator` gives both sides the same branch ID. The current reference
implementation uses `TreeKVCache` for KV state and `ReaperSandbox` for fresh
subprocesses. The SGLang cache patch and Firecracker snapshot path are tested
separately and are not connected to the orchestrator yet.

Use it for:

- `map`/`reduce` agent fanout (e.g. [agent-mapreduce](https://github.com/sachinkesiraju/agent-mapreduce)).
- Trying several coding fixes from the same repository context.
- Pruning candidate branches with tests or other checks.
- Search and planning agents that explore several next steps.
- Evaluations that reuse one prepared context across policies or seeds.

## Example: tree-style agent fanout

A coding agent has read a repository, reproduced a bug, and prepared its build
environment. It now wants to try 10 fixes.

Fork the agent at that point. Each child starts with the same model context. A
snapshot-capable sandbox backend could also give each child the same filesystem
and process state. Each child then does only the work for its own fix.

Run cheap checks first. Kill branches that fail formatting, compilation, or
focused tests. If two fixes survive, fork them again for race tests, performance
tests, or independent review. Run the full test suite only on the finalists.

![agentfork lifecycle: fork a live agent, race the branches, kill the losers](docs/img/lifecycle.svg)

Without shared state, every branch repeats the setup work:

```
N × (shared setup + branch work)
```

With shared state, the setup runs once:

```
shared setup + sum(branch work)
```

This works best when setup is expensive, branches are short, and most branches
can be rejected early. It helps less when there are few branches or when most
work happens after the fork.

## Quickstart

```bash
pip install -e ".[dev]"
python demo/demo.py   # Linux, CPU-only reference demo
pytest -q             # non-Linux hosts skip pidfd integration tests
```

The demo does not run a model or a microVM. Integer token IDs stand in for KV
cache entries, and sleeping Python processes stand in for sandboxes. One parent
owns a 32k-token prefix. Ten children share that prefix, add separate suffixes,
and are then cleaned up. The demo finishes with no live cache trees or resident
reference-cache tokens.

The same lifecycle through the public Python API:

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

`kill_losers()` keeps the winner and its ancestors until the orchestrator closes.
The registry records branch intent so failed cleanup can be retried. It does not
store process state or KV data.

**Compatibility:** Python ≥ 3.10; Linux ≥ 5.4 for the `pidfd` reaper; SGLang @
`40517b593b23870cf351a05a1d53e930cea6a58d` for the patch. Firecracker v1.7
and an NVIDIA A10 on Modal are the measured environments.

## How it works

The current reference path is:

```
ForkOrchestrator
├── registry and leases
├── KV: TreeKVCache (CPU reference cache)
└── sandbox: ReaperSandbox (fresh subprocesses managed with pidfd)
```

Two target backends are tested separately:

```
GPU KV:  TreeRadixCache patch   fork_branch / kill_tree / cache controls
sandbox: Firecracker benchmark  snapshot / load / kill
```

The pieces work as follows:

1. **Orchestrator** — assigns one ID to the sandbox and KV branch, records branch
   state, rolls back failed forks, and tracks leases.
2. **KV cache** — children share the parent's cached prefix. New KV slots are
   needed only when a child adds different tokens.
3. **Sandbox** — `ReaperSandbox` starts a fresh subprocess today. Firecracker
   snapshot inheritance is a separate benchmark and future backend.
4. **Kill** — the sandbox is stopped first, then the KV branch is released. The
   steps are ordered but not atomic.

CUDA memory is not copied with `fork(2)`. The SGLang patch creates another
logical reference to the parent's KV slots and allocates new slots only for a
child's different suffix. Firecracker copy-on-write applies to guest memory, not
CUDA memory.

## Measured results

See [report/RESULTS.md](report/RESULTS.md) for full results and test details.

| Claim | Measured |
|---|---|
| CPU reference prefix reuse | 10 children reused 100% of the parent prefix; separate trees stayed isolated |
| Patched SGLang cache on an A10 | 37k occupied slots vs 357k with sharing disabled; 10 create+extend operations in 22 ms; allocator returned to 0 after cleanup |
| Stock SGLang prefix caching | 2,402–2,403 of 2,404 prompt tokens were cached; this uses stock RadixAttention, not the agentfork patch |
| Subprocess + CPU cache kill | 0.53 ms p50 and 1.46 ms max over 100 cycles |
| Supervisor crash test | 0 surviving Python children across 50 runs with 5 children each |
| Firecracker snapshot load | 2.1 ms p50 API time per child; 25 children loaded in 150 ms; guest readiness was not measured |
| Firecracker process memory | 117.7 MiB total RSS vs 23.8 MiB total PSS across 25 idle VMMs |
| SGLang patch size | 547 lines: 299 implementation and 248 tests |
| 10,000-branch cache test | 0.95 s to create branches and 0.17 s to remove them; this tests cache metadata, not concurrent model execution |
| Cache controls | Direct API tests cover budgets, reservations, demotion, invalidation, and telemetry |

The 9.65× result compares shared KV slots with a worst-case allocation that
stores a separate 32k-token prefix for every child. Stock SGLang already shares
identical prefixes, so agentfork is not 9.65× smaller than stock SGLang. The
patch instead adds branch IDs, budgets, telemetry, and branch-level cleanup.

The provider-cache estimate is a pricing model, not a benchmark. It assumes
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

- The orchestrator currently uses the CPU cache and fresh subprocesses. The
  SGLang patch and Firecracker snapshot path are not connected to it yet.
- Cleanup is ordered but not atomic. Failed cleanup must be retried by the
  caller.
- The registry has no `fsync` or cross-process locking. One orchestrator should
  own each registry file.
- The reference components are not thread-safe, and `ReaperSandbox` uses
  `preexec_fn`, which Python warns against in threaded programs.
- SGLang budgets and reservations are cache accounting only; the scheduler does
  not enforce them.
- Large models, tensor parallelism, scheduler load, guest readiness, and
  microVM+GPU integration have not been tested.

## Why agentfork vs. alternatives

Different projects branch different kinds of state. Choose based on what needs
to be copied.

| Project | What it does | What is still needed |
|---|---|---|
| [forkd](https://github.com/deeplethe/forkd), [Mitos](https://github.com/mitos-run/mitos) | Start microVMs from a shared snapshot | A matching branch in the LLM server and cleanup for its KV cache |
| [thaw](https://github.com/thaw-ai/thaw), [processfork](https://github.com/manav8498/processfork) | Branch model context for several generations | A separate process/filesystem sandbox for each branch |
| [SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm) | Reuse KV cache for requests with the same prefix | A branch ID that connects KV state to a sandbox and cleanup policy |
| [LMCache](https://github.com/LMCache/LMCache), [Mooncake](https://github.com/kvcache-ai/Mooncake), [Dynamo](https://github.com/ai-dynamo/dynamo) | Store or move KV cache between memory tiers or workers | Sandbox branching and joint process/KV cleanup |
| **agentfork** | Gives a subprocess and CPU KV branch one ID; tests GPU KV and microVM branching separately | Production SGLang and Firecracker backends |

## License

Apache-2.0 — see [LICENSE](LICENSE).
