# agentfork

Fork a live agent — its sandbox **and** its LLM KV context — as one coordinated lifecycle.

Kill any branch and reclaim both halves in **sub-millisecond to single-digit milliseconds**, with zero orphans and zero leaked KV pages.

![tree-keyed KV: one resident prefix, N logical branches](docs/img/kv-dedup.svg)

**Measured at a glance:** A10 direct-cache API: 22 ms for 10 create+extend
operations and 9.65× fewer occupied KV slots vs unshared · Linux subprocess +
CPU reference cache: 0.53 ms p50 kill · 547-line additive SGLang patch.

## What it does

agentfork implements two runtime operations:

- **`fork(parent)`** creates a child KV branch and asks the configured sandbox
  backend to start the corresponding sandbox.
- **`kill(child)`** stops the child and reclaims its process and KV ownership.

The public API exposes these as `ForkOrchestrator.fork(parent_id, n)` and
`ForkOrchestrator.kill(branch_id)`. The included `ReaperSandbox` starts a fresh
subprocess for each branch; inheriting a parent filesystem or process snapshot
remains a Firecracker-backend integration target.

It is built from a reference `ForkOrchestrator`, an additive SGLang patch that
tree-keys the radix KV cache, a Firecracker snapshot/restore benchmark, a CPU
reference cache, and a `pidfd`-based subprocess reaper. It is not a full agent
framework or a scheduler. The orchestrator currently coordinates the CPU cache
and generic subprocesses; Firecracker and live SGLang are not yet integrated as
backends.

Use it for:

- `map`/`reduce` agent fanout (e.g. [agent-mapreduce](https://github.com/sachinkesiraju/agent-mapreduce)).
- Speculative coding fixes: fork multiple candidates from one repo context and pick the best.
- Verification trees: cheap verifiers kill most branches; the application persists the winner.
- Search and planning agents that fan out candidate plans, score them, and prune.
- Evaluation matrices that reuse one prepared context and environment across policies or seeds.

## Example: tree-style agent fanout

agentfork is designed for agent workflows that look like `map`/`reduce` over a
process tree. The target primitive is **agent fanout**: a resident agent reaches
a decision point, `fork`s into N candidate branches that share the parent's
context and, with a snapshot-capable backend, sandbox state; it then `kill`s the
losers while the application persists the winner.

Imagine a coding agent that has already digested a 32k-token repo and prepared a
working build environment. It needs to test 10 possible fixes. In a completed
agentfork integration, you fork the agent 10 times from that exact state. Each
branch reuses the warm context and initial sandbox state, so it pays only for
its unique work. Run cheap verifiers, kill the 9 losers, and persist the winner.
Against a no-sharing baseline, this avoids booting 10 cold sessions and replaying
the repository context 10 times; stock RadixAttention already avoids much of the
KV replay, while agentfork targets explicit tree ownership and sandbox lifecycle.

The strongest version is a tree rather than a flat best-of-N batch. If two fixes
survive, fork each again into adversarial-test, race-detection, performance, and
independent-review branches. Those grandchildren inherit the root context plus
their candidate's reasoning and sandbox changes. Run the full suite only on
finalists, export the winning patch and tests, and reclaim everything else.

![agentfork lifecycle: fork a live agent, race the branches, kill the losers](docs/img/lifecycle.svg)

The fanout cost changes from

```
N × (shared parent work + unique branch work)
```

to

```
shared parent work + N × unique branch work
```

For recursive fanout, the same accounting becomes

```
root work + Σ unique work on each explored edge
```

In the target runtime, the payoff comes from doing the expensive pre-branch
work once. If the parent has already accumulated a large model context and
prepared its environment, every child can start from that state instead of
rebuilding it. Keeping branches short and verifying early limits each
candidate's new token, memory, and compute cost; killing failures quickly also
frees their process and cache state before more expensive checks run. With only
a few branches—or with branches that spend most of their time doing different
work after the fork—the shared setup is a small fraction of total cost. In that
case, the benefit falls and orchestration overhead can dominate.

## Quickstart

```bash
pip install -e ".[dev]"
python demo/demo.py   # Linux, CPU-only reference fork/race/kill demo
pytest -q             # non-Linux hosts skip pidfd integration tests
```

`demo.py` exercises the lifecycle without a model or microVM. Integer token IDs
stand in for KV-cache tensors, and each sandbox is a sleeping Python subprocess.
One parent owns a 32k-token prefix; 10 children reference that same prefix without
copying or re-prefilling it. That is 11 logical copies backed by one resident
prefix (11× dedup). After every child adds a distinct 800-token suffix, the ratio
falls to 9× because those suffixes cannot be shared. The demo picks a winner,
kills every branch and the parent, and verifies that no reference-cache tokens or
tree records remain.

The supported Python API is exported directly from `agentfork` starting in
v0.2.0:

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
    orch.kill_losers(children[0].branch_id)  # keeps winner + ancestors until close
```

`kill()` reaps the sandbox, then the KV branch, then removes the journal record.
The steps are sequential, not atomic; if a cleanup step raises, the record stays
journaled for an explicit `kill()` retry. The registry stores lifecycle
intent—not process handles, sandbox snapshots, or KV contents—so it cannot resume
branch execution. `reconcile()` collects mid-fork records and expired leases;
lease collection is caller-driven and there is no background reaper.

**Compatibility:** Python ≥ 3.10; Linux ≥ 5.4 for the `pidfd` reaper; SGLang @
`40517b593b23870cf351a05a1d53e930cea6a58d` for the patch. Firecracker v1.7
and an NVIDIA A10 on Modal are the measured environments. 70B-scale tensor
parallelism and microVM+GPU colocation are unmeasured.

## How it works

`ForkOrchestrator` gives the sandbox and CPU reference-cache branch one ID,
journals lifecycle intent, and rolls back failures it observes during a fork.
Mid-fork records and expired leases remain until the caller invokes `reconcile()`
or `reap_expired()`; failed kills stay journaled for an explicit retry. The
production backends remain separate:

1. **KV cache fork** — `patches/0001-sglang-tree-radix-cache.patch` adds
   `TreeRadixCache` to SGLang. Children inherit the parent's KV prefix
   copy-on-write and keep it pinned through the existing `lock_ref` machinery.
   `kill_tree()` releases the branch path. The patch also adds logical token
   budgets, suffix reservations, priority demotion/promotion, explicit
   invalidation, and per-tree telemetry. No scheduler, model-runner, server, or
   router changes are included.
2. **Sandbox fork** — `agentfork/sandbox/fc_bench.py` independently drives
   Firecracker snapshot/load with page-level CoW. The recorded parent pause
   window was ~76–83 ms including full snapshot creation; snapshot-load API time
   was 2.1 ms p50 per child.
3. **Process + reference-cache kill** — the standalone kill benchmark gives
   `BranchReaper` both the subprocess and CPU cache, so one call reaps the process
   and drops the matching cache entry. Under `ForkOrchestrator`, `ReaperSandbox`
   reaps the subprocess first and the orchestrator then releases the KV branch.
   The measured combined reference path was 0.53 ms p50; it is not a Firecracker
   + GPU kill measurement.

```
ForkOrchestrator
├── lifecycle state: JSON registry, leases, rollback, reconciliation
└── one branch ID shared by the current reference backends
    ├── KV:      TreeKVCache     CPU model of prefix sharing and reclaim
    └── sandbox: ReaperSandbox   fresh subprocess supervised through pidfd

Validated separately; not yet wired into ForkOrchestrator
├── GPU KV:  TreeRadixCache patch   fork_branch / kill_tree / cache policy
└── sandbox: Firecracker benchmark  snapshot / load / kill
```

`fork(2)` cannot clone CUDA allocations. `TreeRadixCache.fork_branch()` instead
creates another logical owner of the parent's existing KV-slot indices and pins
that shared radix path; only the child's divergent suffix needs new KV slots.
Firecracker's copy-on-write path is separate and applies to the microVM's guest
memory snapshot, not to CUDA state.

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
| Firecracker snapshot load | **2.1 ms p50 load API time/child**, 25-way fanout in 150 ms; guest application readiness was not measured |
| Firecracker idle-VMM RSS/PSS ratio | RSS 117.7 MiB vs PSS 23.8 MiB across 25 idle VMMs → **4.95×**; this includes more than guest snapshot pages |
| SGLang patch | **547 additive lines**: 299 implementation + 248 tests; current patch has 17 test methods, while the captured GPU run used the earlier 7-test revision |
| Scale: one prefix into 10,000 logical branches | **0.95 s (10.5k forks/s)** on a CPU-backed SGLang allocator; bulk kill of 10,001 in 0.17 s; allocator back to 0. This measures metadata scale, not concurrent inference |
| Tree-native cache controls | Direct API checks cover budgets, reservations, demotion/promotion, invalidation, and telemetry; scheduler enforcement remains unimplemented |

**How to read the 9.65× result:** the GPU test compares shared KV slots with a
worst-case allocation that stores a separate copy of the 32k-token prefix for
every child. Normal SGLang RadixAttention already shares identical prefixes, so
agentfork is not 9.65× smaller than stock SGLang. Against that stronger baseline,
the cost model gives both systems roughly the same prefill work and KV residency.
The patch instead adds branch-level ownership primitives—identity, pinning,
budgets, telemetry, and reclaim—that a future SGLang backend can connect to the
sandbox lifecycle.

**Provider-cache estimate:** this is a token-pricing model, not a benchmark. It
assumes cached reads cost 0.1× normal input tokens and cache writes cost 1.25×;
it does not measure invoices, request latency, or provider HBM usage.

## Running benchmarks

```bash
pytest -q
python demo/demo.py
python -m agentfork.bench.kill_bench --cycles 100
python -m agentfork.bench.crash_bench --cycles 50 --children 5
python -m agentfork.bench.cost_model --children 10 --prefix 32000 --suffix 2000

# Direct SGLang cache validation (from this repository):
export SGLANG_DIR=/path/to/sglang
git -C "$SGLANG_DIR" checkout 40517b593b23870cf351a05a1d53e930cea6a58d
git -C "$SGLANG_DIR" apply "$PWD/patches/0001-sglang-tree-radix-cache.patch"
PYTHONPATH="$SGLANG_DIR/python" python patches/real_pool_validation.py
PYTHONPATH="$SGLANG_DIR/python" python patches/scale_10k_branch_validation.py
PYTHONPATH="$SGLANG_DIR/python" python patches/tree_native_features_validation.py

# Firecracker bench (needs /dev/kvm + firecracker binary + guest kernel/rootfs):
python -m agentfork.sandbox.fc_bench --fc ./firecracker --kernel vmlinux --rootfs rootfs.ext4

# GPU validation (needs Modal and the patched SGLang checkout):
pip install modal
SGLANG_DIR="$SGLANG_DIR" modal run modal_gpu_validation.py
```

## Limitations

- `ForkOrchestrator` currently wires only the CPU `TreeKVCache` and fresh-process
  `ReaperSandbox`. The SGLang patch and Firecracker benchmark are not backends,
  and no production path spans a microVM and live inference engine. Cleanup is
  sequential, and winner/artifact handoff remains application-specific.
- The JSON registry uses temporary-file + `os.replace`, but no `fsync` or
  cross-process locking. It assumes one orchestrator owns a registry path;
  cleanup retries and lease collection are caller-driven.
- `ForkOrchestrator`, `TreeKVCache`, and `BranchReaper` do not synchronize
  concurrent callers. `BranchReaper` also uses `preexec_fn`, which Python warns
  is unsafe in threaded code.
- SGLang budgets and reservations are direct cache-API accounting hooks, not
  scheduler-enforced physical HBM reservations.
- Validation is limited to one A10 direct-cache pool, a stock Qwen3-0.6B engine
  baseline, and idle CPU-only Firecracker guests. 70B-scale behavior,
  tensor/pipeline parallelism, scheduler contention, guest readiness, and
  microVM+GPU colocation are unmeasured; the Modal base image is not digest-pinned.

## Why agentfork vs. alternatives

These projects solve different parts of the problem. The practical question is
which state you need to branch: the agent's execution environment, the model's
attention cache, or both. agentfork targets both, but its current end-to-end path
uses the CPU reference cache and fresh subprocesses; the GPU-cache and microVM
paths are still separate.

| Project | What it does | What you still have to build |
|---|---|---|
| [forkd](https://github.com/deeplethe/forkd), [Mitos](https://github.com/mitos-run/mitos) | Start microVMs from a shared memory snapshot, so unchanged guest-memory pages can be reused | A matching branch in the LLM server, plus cleanup that frees its KV-cache slots when the microVM branch is killed |
| [thaw](https://github.com/thaw-ai/thaw), [processfork](https://github.com/manav8498/processfork) | Branch an inference session so multiple generations can continue from shared model context | An isolated filesystem/process sandbox for each branch, and one operation that cleans up both states |
| [SGLang](https://github.com/sgl-project/sglang) RadixAttention, [vLLM](https://github.com/vllm-project/vllm) Automatic Prefix Caching | Reuse KV-cache blocks when requests begin with the same tokens | A persistent agent branch ID that links those cache blocks to a sandbox, lease, and branch-level kill |
| [LMCache](https://github.com/LMCache/LMCache), [Mooncake](https://github.com/kvcache-ai/Mooncake), [Dynamo](https://github.com/ai-dynamo/dynamo) | Store or move KV-cache blocks between GPU memory, CPU memory, storage, or workers | Cloning the agent's executable sandbox and managing process + KV cleanup as one branch |
| **agentfork** | Gives a subprocess and CPU reference-cache branch one ID; separately validates a tree-aware SGLang cache and Firecracker snapshots | Production backends that connect `ForkOrchestrator` to the patched SGLang cache and Firecracker |

## License

Apache-2.0 — see [LICENSE](LICENSE).
