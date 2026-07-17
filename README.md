# agentfork

Fork a live agent — its sandbox **and** its LLM KV context — as one coordinated
lifecycle.
Kill any branch and reclaim both halves in **sub-millisecond to single-digit
milliseconds**, with zero orphans and zero leaked KV pages.

![tree-keyed KV: one resident prefix, N logical branches](docs/img/kv-dedup.svg)

**Measured at a glance:** 22 ms for 10 patched-cache create+extend operations on
an A10 · 0.53 ms p50 subprocess + CPU reference-cache kill · 9.65× KV-slot
reduction vs an explicitly unshared allocation · 547-line additive SGLang patch.

## What it does

agentfork prototypes two operations for a fork-native agent runtime:

- **`fork(parent)`** gives a child the parent's cached context and sandbox state.
- **`kill(child)`** stops the child and reclaims its process and KV ownership.

The repository validates the lower-level pieces of those operations: an
additive SGLang patch that adds tree identity and lifecycle controls to the
radix cache, a Firecracker snapshot/restore benchmark, a CPU reference cache,
and a `pidfd`-based subprocess reaper. It is not yet a production runtime or an
atomic transaction across Firecracker and a live inference server.

Use it for:

- `map`/`reduce` agent fanout (e.g. [agent-mapreduce](https://github.com/sachinkesiraju/agent-mapreduce)).
- Speculative coding fixes: branch after repository analysis, try independent
  fixes, and retain the best verified result.
- Verification trees: use cheap checks to prune most candidates before running
  expensive tests on finalists.
- Search and planning agents that expand promising branches recursively, score
  them, and reclaim the rest.
- Evaluation matrices that reuse one prepared context and environment across
  policies, seeds, models, or tool configurations.

These are integration targets, not packaged applications in this repository.
`ForkOrchestrator` (`agentfork/orchestrator.py`) is a reference control plane
for the lifecycle half: one branch ID spans the KV branch and the sandbox,
intent is journaled to a registry file before side effects, leases bound every
branch's lifetime, and `reconcile()` collects what a crashed supervisor left
behind. Its backends today are the CPU reference cache and generic
subprocesses; your application still has to submit inference work, score
branches, and persist the winner.

## Example: tree-style agent fanout

Imagine a coding agent that has already read a large repository, loaded its tool
schemas and issue history, reproduced a bug, and prepared a working build
environment. It has reached a good branch point: the shared setup was expensive,
but testing any one plausible fix is comparatively cheap.

Fork the agent into 12 strategy branches from that exact state. Each branch
inherits the warm context and initial filesystem state, then makes only its own
code changes and generates only its own suffix tokens. Run formatting,
compilation, focused tests, and a cheap critic first; kill failures immediately
instead of carrying all 12 into the full suite.

The strongest version is a tree, not a flat best-of-N batch. If two fixes
survive, fork each one again into adversarial-test, race-detection, performance,
and independent-review branches. Those grandchildren inherit the root context
plus their candidate's reasoning and sandbox changes. Run the full suite only
on finalists, export the winning patch and tests, and reclaim everything else.
Winner persistence or merge remains application-specific.

![agentfork lifecycle: fork a live agent, race the branches, kill the losers](docs/img/lifecycle.svg)

The work changes from replaying the full root-to-leaf history for every leaf to
paying once for the root and once for the unique work on each explored edge:

```
root work + sum(unique work on each explored edge)
```

This shape is most attractive when shared setup is long, branches are numerous
and short, cheap verification rejects most candidates, and promising candidates
fan out again. The advantage shrinks when fanout is small, most work happens
after divergence, or ordinary prefix caching already provides all the lifecycle
behavior the application needs.

## Quickstart

```bash
pip install -e ".[dev]"
python demo/demo.py   # Linux, CPU-only reference fork/race/kill demo
pytest -q             # non-Linux hosts skip pidfd integration tests
```

`demo.py` creates a synthetic 32k-token parent in the CPU reference cache,
spawns 10 real Python child processes, forks the cache 10 ways with zero
re-prefill, and ends with 0 resident reference-cache tokens and 0 live trees.
KV residency is 11× deduplicated immediately after fork and 9× after each child
adds an 800-token suffix. It does not run a real LLM or Firecracker guest.

The same reference primitives in Python:

```python
import sys

from agentfork.kv.tree_cache import TreeKVCache
from agentfork.kill.reaper import BranchReaper

cache = TreeKVCache()
parent = cache.create_tree("tree-1")
cache.extend(parent.tree_id, prefix_tokens)

child = cache.fork_branch("tree-1", "tree-1/branch-1")
cache.extend(child.tree_id, unique_suffix_tokens)

reaper = BranchReaper(kv_cache=cache)
reaper.spawn(child.tree_id, [sys.executable, "-c", "import time; time.sleep(60)"])
reaper.kill(child.tree_id)   # sequentially reaps process, then drops cache refs
```

**Compatibility:** Python ≥ 3.10; Linux ≥ 5.4 for the `pidfd` reaper; SGLang @
`40517b593b23870cf351a05a1d53e930cea6a58d` for the patch. Firecracker v1.7
and an NVIDIA A10 on Modal are the measured environments, not a complete
support matrix.

## How it works

The intended runtime coordinates three lifetimes using one branch identity. In
this repository they remain separate components:

1. **KV cache fork** — `patches/0001-sglang-tree-radix-cache.patch` adds
   `TreeRadixCache` to SGLang. Children share the parent's existing KV slots and
   pin the radix path through `lock_ref`; divergent suffixes get new slots.
   `kill_tree()` releases the branch path. The patch also adds logical token
   budgets, suffix reservations, demotion/promotion, invalidation, and
   telemetry. These are direct cache APIs; no scheduler, model-runner, server,
   or router currently calls them.
2. **Sandbox fork** — `agentfork/sandbox/fc_bench.py` is a standalone
   Firecracker full-snapshot/load benchmark. The file memory backend uses
   `MAP_PRIVATE`, so clean pages can be shared and writes become private. The
   recorded run used a ~76–83 ms parent pause window including snapshot creation
   and a 2.1 ms p50 snapshot-load API time per child.
3. **Process + reference-cache kill** — `agentfork/kill/reaper.py` supervises a
   generic subprocess with Linux `pidfd`, waits for confirmed exit, then calls
   the CPU cache's `kill(tree_id)`. The measured 0.53 ms p50 is this sequential
   subprocess + reference-cache path, not Firecracker + GPU reclaim.

```
Your control plane
        │
        ▼
   coordinated branch ID
   ├── TreeRadixCache patch    fork_branch / kill_tree / policy / telemetry
   ├── Firecracker benchmark   snapshot / load / kill (standalone)
   └── BranchReaper            pidfd process + CPU reference-cache kill
```

CUDA state cannot be `fork(2)`-ed. The KV fork is logical CoW over shared paged
KV slots, not an OS-level fork of GPU state. Independent agent trees use separate
cache namespaces; only descendants of the same parent share that tree's prefix.

## Measured results

See [report/RESULTS.md](report/RESULTS.md) for the validation status, captured
outputs, assumptions, and checks that fail or remain untested.

| Claim | Measured |
|---|---|
| CPU reference prefix reuse (10-way fanout) | **100%** of the parent prefix reused; independent trees remain isolated |
| Patched cache on a real SGLang GPU pool (NVIDIA A10) | **9.65× vs an explicitly unshared allocation** (37k slots vs 357k), 10 create+extend operations in 22 ms, allocator back to 0 after kill-all |
| Stock SGLang live-engine prefix-cache baseline | **2,402–2,403 of 2,404 tokens cached** per sibling; 33 ms p50 generation vs 9.07 s first request. This path does not invoke the patch's branch APIs |
| Subprocess + CPU reference-cache kill | **0.53 ms p50 / 1.46 ms max**, 100 cycles on the recorded host |
| Supervisor SIGKILLed (crash injection) | **0 surviving Python children** in 50×5 cycles; 1.5 ms p50 through `PR_SET_PDEATHSIG` |
| Firecracker snapshot load | **2.1 ms p50 load API time/child**, 25-way fanout in 150 ms; full VMM teardown was 31 ms p50 |
| Firecracker host-page sharing | RSS 117.7 MiB vs PSS 23.8 MiB across 25 idle VMMs → **4.95× RSS/PSS ratio** |
| SGLang patch | **547 additive lines**: 299 implementation + 248 tests, with 17 test methods |
| Scale: one prefix into 10,000 logical branches | **0.95 s (10.5k forks/s)** on a CPU-backed SGLang allocator; bulk kill of 10,001 in 0.17 s; allocator back to 0 |
| Tree-native cache controls | Direct API checks cover budgets, reservations, demotion/promotion, invalidation, and telemetry; scheduler enforcement remains unimplemented |

The 9.65× figure compares sharing with a deliberately unshared allocation, not
with stock SGLang RadixAttention. Stock SGLang already stores an identical
cached prefix once. The corrected cost model is therefore ~1.0× compute and
~1.0× cache residency versus a well-run same-namespace self-hosted prefix cache.
The proposed gain over that baseline is explicit ownership, pinning, policy,
telemetry, and coordinated reclaim—not another 9.65× memory reduction.

Provider-cache comparisons are token arithmetic using a 0.1× cached-read price
and 1.25× cache-write price, not measured invoices, latency, or HBM usage.

## Running benchmarks

```bash
pytest -q
python demo/demo.py
python -m agentfork.bench.kill_bench --cycles 100
python -m agentfork.bench.crash_bench --cycles 50 --children 5
python -m agentfork.bench.cost_model --children 10 --prefix 32000 --suffix 2000

# SGLang direct-cache validation (run inside a patched SGLang environment):
python patches/real_pool_validation.py
python patches/scale_10k_branch_validation.py
python patches/tree_native_features_validation.py

# Firecracker (needs Linux, /dev/kvm, binary, guest kernel, and rootfs):
python -m agentfork.sandbox.fc_bench --fc ./firecracker --kernel vmlinux --rootfs rootfs.ext4

# GPU validation (needs Modal and a patched SGLang checkout):
SGLANG_DIR=/path/to/sglang modal run modal_gpu_validation.py
```

CI runs Ruff, the repository tests, the demo, shorter kill/crash benchmarks, and
the cost model on Python 3.10/Linux. It does not run SGLang, Modal, GPU, or
Firecracker validation.

## Limitations

- There is no unified production `fork()` API. Firecracker restore and KV fork
  are separate operations, and `BranchReaper.kill()` is sequential rather than
  transactional.
- The SGLang patch is additive and pinned to one commit. It is not wired into
  request scheduling, model execution, HTTP serving, tensor parallelism, or a
  multi-worker router. The live-engine result above is a stock RadixAttention
  baseline, not an end-to-end test of the patch.
- Cache budgets and reservations are logical direct-API accounting hooks. They
  do not reserve physical allocator slots until a scheduler uses them.
- GPU validation uses one A10, a synthetic 2 GiB fp16 pool, and a Qwen3-0.6B
  stock-engine baseline. 70B-class models, tensor/pipeline parallelism, mixed
  workloads, allocator pressure, and scheduler contention are unmeasured.
- Firecracker measurements use idle, CPU-only 256 MiB guests. Firecracker and GPU
  inference have not been colocated or connected through an API proxy; guest
  networking, identity handling, real agent readiness, and cold page faults can
  change the results.
- `TreeKVCache` and `BranchReaper` are single-controller reference components;
  they do not synchronize concurrent callers. The reaper also uses
  `preexec_fn`, which Python warns is unsafe in threaded programs.
- No winner merge, durable artifact handoff, hibernation, migration, or resume
  protocol is implemented.
- The observed workload-shape check failed on the available organic traces, and
  those private traces are not included in the repository.
- Provider-cache ratios are modeled rather than measured. The Modal image is
  also based on `lmsysorg/sglang:latest`, so GPU reruns are not fully hermetic.

## Why agentfork vs. alternatives

Existing systems cover neighboring parts of the tree: fast sandbox snapshots,
inference-session branching, shared-prefix caching, or KV movement. agentfork's
target is one branch identity spanning sandbox state and explicit KV ownership.
The current repository validates the pieces of that target but does not yet bind
them into a production runtime.

| Project | What it covers | What remains for a fork-native agent runtime |
|---|---|---|
| [forkd](https://github.com/deeplethe/forkd), [Mitos](https://github.com/mitos-run/mitos) | microVM snapshot/fork with CoW memory | Bind sandbox identity to inference KV ownership and reclaim |
| [thaw](https://github.com/thaw-ai/thaw), [processfork](https://github.com/manav8498/processfork) | inference/session branching experiments | Pair the inference branch with an isolated sandbox lifecycle |
| [SGLang](https://github.com/sgl-project/sglang) RadixAttention, [vLLM](https://github.com/vllm-project/vllm) APC | automatic shared-prefix KV reuse | Add explicit agent-tree ownership, branch policy, and sandbox coordination where required |
| [LMCache](https://github.com/LMCache/LMCache), [Mooncake](https://github.com/kvcache-ai/Mooncake), [Dynamo](https://github.com/ai-dynamo/dynamo) | KV movement, storage, and distributed serving | Compose tree ownership with movement and routing |
| **agentfork (this prototype)** | tree-aware SGLang patch, CPU reference cache/reaper, and separate Firecracker benchmark | Integrate all three paths behind one recoverable control-plane operation |

## License

Apache-2.0 — see [LICENSE](LICENSE).
