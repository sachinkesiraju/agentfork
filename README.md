# agentfork

agentfork is a research prototype for giving an agent's sandbox process and LLM
KV-cache branch the same lifecycle identity.

**Kill any branch and reclaim both halves in sub-millisecond to single-digit
milliseconds, with zero orphans and zero leaked KV pages.**

The intended primitive is simple: fork a warm agent into several isolated
candidates, let them diverge, then kill the losers and reclaim their process and
KV state. This repository validates the pieces of that design. It is **not yet a
drop-in runtime that atomically forks a Firecracker VM and a live inference
request**.

![tree-keyed KV: one resident prefix, N logical branches](docs/img/kv-dedup.svg)

**Selected experimental results:** 22 ms to create and extend 10 branches on a
patched SGLang allocator backed by an NVIDIA A10 · 0.53 ms p50 to reap a Python
child process and its CPU reference-cache entry · 9.65× KV-slot reduction versus
an explicitly unshared baseline · 547 additive lines in the SGLang patch,
including tests. See [Measured results](#measured-results) for the scope behind
each number.

## What is in this repository

| Component | What it is | What it is not |
|---|---|---|
| `agentfork/kv/tree_cache.py` | CPU reference model for tree identity, prefix sharing, refcounts, quotas, and reclaim | An inference engine or GPU KV cache |
| `patches/0001-sglang-tree-radix-cache.patch` | Additive `TreeRadixCache` implementation and 17 tests for SGLang at `40517b593` | Wired into SGLang's scheduler, model runner, or public server API |
| `agentfork/kill/reaper.py` | Linux `pidfd` supervisor for a generic subprocess plus a cache object | A Firecracker manager or atomic transaction coordinator |
| `agentfork/sandbox/fc_bench.py` | Standalone Firecracker snapshot/load/kill benchmark | A reusable `FCForker` runtime integrated with the cache |
| `demo/demo.py` | Linux demo using real Python child processes and the CPU reference cache | A real LLM or microVM demo |
| `modal_gpu_validation.py` | Patched-cache tests against an A10 HBM pool, plus a stock SGLang prefix-cache baseline | End-to-end use of the patch through `sgl.Engine` |

There is no top-level `fork(parent)` API today. The repository exposes the
lower-level operations an orchestrator would need and records measurements for
them.

## Use it for

Use agentfork to **evaluate or prototype fork-heavy agent runtimes** where:

1. a parent has accumulated a large, expensive context;
2. several children start from exactly that state;
3. each child performs a relatively small amount of unique work; and
4. most children can be cancelled after scoring or verification.

Workloads with that shape include:

- **Best-of-N coding:** branch after repository analysis, try independent fixes,
  run tests, and retain the best result.
- **Map/reduce agents:** fan one prepared context out to many workers, then
  aggregate their outputs (for example,
  [agent-mapreduce](https://github.com/sachinkesiraju/agent-mapreduce)).
- **Verifier trees:** generate candidates, run progressively more expensive
  checks, and reclaim branches as soon as they fail.
- **Search and planning:** expand several next actions from one trajectory,
  score them, and keep only the promising subtree.
- **Evaluation and test matrices:** reuse a common prompt or environment across
  models, policies, seeds, or tool configurations.

Today, those are **integration targets**, not packaged applications in this
repo. Your orchestrator still has to create the sandbox, submit inference work,
associate both with the same branch ID, choose a winner, and merge any durable
artifacts.

agentfork is probably not the right tool when requests do not share a long
prefix, when ordinary SGLang RadixAttention already gives all the cache reuse
you need, when inference is only available through a hosted API, or when you
need a production-ready hibernate/migrate/resume system. No hibernation,
checkpoint migration, winner merge, or durable artifact protocol is implemented
here.

## The target lifecycle

A fork-native agent workflow looks like this:

1. The parent reaches a branch point with a warm sandbox and KV prefix.
2. The control plane restores or creates N sandboxes and calls
   `fork_branch()` N times under one tree namespace.
3. Each branch reuses the parent's cached prefix and allocates only its unique
   suffix.
4. The control plane scores branches and calls `kill()` on the losers.
5. Application-specific code persists or merges the winner's output.

![agentfork lifecycle: fork a live agent, race the branches, kill the losers](docs/img/lifecycle.svg)

For N branches, the idealized work changes from

```text
N × (shared parent work + unique branch work)
```

to

```text
shared parent work + N × unique branch work
```

Stock prefix caches can already achieve much of the compute and physical KV
reuse for identical prefixes. The patch in this repo is about adding explicit
tree/branch identity, pinning, quotas, reservations, demotion, invalidation,
telemetry, and branch-scoped reclaim.

## Quickstart

Requirements for the full demo and test suite:

- Python 3.10+
- Linux 5.4+ (`pidfd_open`, `pidfd_send_signal`, and `waitid(P_PIDFD)`)
- no GPU or Firecracker

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
ruff check agentfork tests
pytest -q
python demo/demo.py
```

The demo creates a synthetic 32k-token parent in the CPU reference cache,
spawns 10 sleeping Python child processes, forks the cache 10 ways, adds an
800-token unique suffix per child, and reaps every process and cache entry. KV
residency is 11× deduplicated immediately after the fork and 9× after all
branches diverge. The final ledger must contain zero resident reference-cache
tokens and zero live trees.

The demo is not a 32k-token LLM prefill: token IDs stand in for KV tensors. On
macOS or another non-Linux host, run the platform-independent cache tests and
cost model instead:

```bash
pytest -q tests/test_tree_cache.py
python -m agentfork.bench.cost_model --children 10 --prefix 32000 --suffix 2000
```

## Reference API example

Imports currently come from submodules; the package does not define a stable
public API at `agentfork.*`.

```python
import sys

from agentfork.kill.reaper import BranchReaper
from agentfork.kv.tree_cache import TreeKVCache

cache = TreeKVCache()
cache.create_tree("parent")
cache.extend("parent", prefix_tokens)

child = cache.fork_branch("parent", "candidate-1")
cache.extend(child.tree_id, unique_suffix_tokens)

reaper = BranchReaper(kv_cache=cache)
reaper.spawn(child.tree_id, [sys.executable, "-c", "import time; time.sleep(60)"])
result = reaper.kill(child.tree_id)
```

`BranchReaper.kill()` first signals and waits for the subprocess, then calls the
cache's `kill(tree_id)`. The two steps share an ID but are sequential, not
atomic. This example uses `TreeKVCache`; the patched SGLang API uses
`create_agent_tree()`, `fork_branch()`, and `kill_tree()` instead.

## How the validated pieces work

### Tree-keyed KV cache

The SGLang patch adds `TreeRadixCache` on top of the existing `RadixCache` API.
Branches in one tree share an `extra_key` namespace. A fork records the parent's
token path and increments existing `lock_ref` pins; it does not copy KV slots.
A kill drops the branch's pin and evicts pages that no live branch protects.

The patch also implements per-tree token accounting, suffix reservations,
branch demotion/promotion, explicit invalidation, and telemetry. These features
have direct unit and allocator-level tests, but no scheduler or router currently
calls them. Reservations and quotas are accounting/admission hooks; they do not
reserve allocator slots by themselves.

CUDA state is never passed through `fork(2)`. “CoW” here means multiple logical
branches refer to the same paged KV slots until their token paths diverge.

### Sandbox snapshot/restore

`agentfork/sandbox/fc_bench.py` uses Firecracker's snapshot APIs directly. A
full parent snapshot is loaded into separate Firecracker processes using the
file memory backend. Firecracker maps that file with `MAP_PRIVATE`, so clean
pages can be shared and writes become private pages.

The measured benchmark used Firecracker v1.7 and 256 MiB CPU-only guests. The
snapshot memory file must remain available for every restored VM's lifetime.
The driver does not configure networking, disks beyond a read-only rootfs,
identity regeneration, GPU access, or an agent workload inside the guest.

### Process and cache reclaim

`BranchReaper` launches a generic subprocess, obtains a pidfd, sends `SIGKILL`
through `pidfd_send_signal`, waits with `waitid(P_PIDFD)`, and then drops the
matching reference-cache entry. `PR_SET_PDEATHSIG` is used as a supervisor-death
backstop. The current code does not use `CLONE_PIDFD_AUTOKILL`.

## Measured results

These are recorded experiment results, not a production service benchmark.
Hardware-dependent numbers are from the hosts described in
[report/RESULTS.md](report/RESULTS.md).

| Result | Measurement | What it proves |
|---|---|---|
| CPU reference prefix reuse | 100% across a 10-way fork | Reference refcount and accounting semantics |
| Patched SGLang cache on an A10 KV pool | 37k slots versus 357k with sharing disabled (**9.65×**); 10 create+extend operations in 22 ms; allocator returned to 0 | Patch can reuse and reclaim real GPU KV slots through SGLang's allocator |
| Stock `sgl.Engine` prefix-cache baseline | 2,402–2,403 of 2,404 prompt tokens cached; 33 ms p50 sibling generation versus 9.07 s first request | Existing SGLang RadixAttention reuses identical prefixes; this path does **not** call the patch's branch APIs |
| Process + CPU reference-cache kill | **0.53 ms p50 / 1.46 ms max**, 100 cycles | Sequential pidfd reap plus Python reference-cache reclaim on the measured host |
| Supervisor crash injection | 0 surviving Python children in 50 × 5 cycles; 1.5 ms p50 | `PR_SET_PDEATHSIG` behavior for the tested subprocess workload |
| Firecracker snapshot load | **2.1 ms p50** load API time per child; 25-way fanout in 150 ms | Standalone restore path on warm, CPU-only 256 MiB guests |
| Firecracker page sharing | 117.7 MiB aggregate RSS versus 23.8 MiB PSS across 25 VMMs (**4.95×**) | Host page sharing in the measured idle-guest setup |
| Patch size and tests | 299 implementation lines + 248 test lines; 17 test methods | Patch scope; test pass records are captured separately from this repo's test suite |
| 10,000 logical branches | 0.95 s to fork; allocator unchanged until suffixes; 0.17 s to kill all | Patched cache metadata scaling on a CPU-backed SGLang allocator |
| Tree controls | quotas, reservations, demotion/promotion, invalidation, telemetry exercised | Engine-local control semantics, not scheduler or multi-worker behavior |

The 9.65× number is versus an intentionally unshared allocation, **not versus
stock SGLang RadixAttention**. Stock SGLang already stores a shared cached prefix
once. The corrected cost model therefore gives agentfork approximately 1.0×
compute and 1.0× cache residency versus a well-run same-namespace self-hosted
prefix cache. The proposed benefit over that baseline is lifecycle control and
predictable ownership, not another 9.65× memory reduction.

The provider comparison is token arithmetic using a 0.1× cached-read price and
1.25× cache-write price. It is not based on measured invoices, latency, or HBM
usage.

## Running benchmarks

### Core Linux checks

```bash
pytest -q
python demo/demo.py
python -m agentfork.bench.kill_bench --cycles 100
python -m agentfork.bench.crash_bench --cycles 50 --children 5
python -m agentfork.bench.cost_model --children 10 --prefix 32000 --suffix 2000
```

CI runs the repository test suite, Ruff, the demo, shorter kill/crash
benchmarks, and the cost model on Python 3.10/Linux. It does not run SGLang,
GPU, Modal, or Firecracker validation.

### SGLang patch

The patch is pinned to public SGLang commit
`40517b593b23870cf351a05a1d53e930cea6a58d`. Both files added by the patch are
absent at that commit, and the imports it uses are present. Apply and test it in
a separate SGLang checkout:

```bash
git checkout 40517b593b23870cf351a05a1d53e930cea6a58d
git apply /path/to/agentfork/patches/0001-sglang-tree-radix-cache.patch
pytest -q test/registered/unit/mem_cache/test_tree_radix_cache.py
python /path/to/agentfork/patches/real_pool_validation.py
python /path/to/agentfork/patches/scale_10k_branch_validation.py
python /path/to/agentfork/patches/tree_native_features_validation.py
```

The three validation scripts require the patched SGLang environment and PyTorch.
They use real SGLang pool/allocator classes on the CPU; “real pool” does not mean
a running model server.

### GPU validation

```bash
pip install modal
SGLANG_DIR=/path/to/patched/sglang modal run modal_gpu_validation.py
```

This requires a Modal account and an SGLang checkout with the patch applied.
The image currently starts from `lmsysorg/sglang:latest`, so reruns are not fully
hermetic even though the mounted source is pinned.

### Firecracker validation

```bash
python -m agentfork.sandbox.fc_bench \
  --fc ./firecracker \
  --kernel vmlinux \
  --rootfs rootfs.ext4 \
  --children 10
```

This requires Linux, `/dev/kvm`, a Firecracker binary, and compatible guest
kernel/rootfs artifacts.

## Limitations and unverified work

### Integration

- There is no unified production `fork()` implementation. Firecracker restore
  and KV fork are separate operations that a control plane would coordinate.
- There is no transaction, rollback, or crash-recovery protocol spanning the
  sandbox and inference server. `BranchReaper.kill()` is sequential.
- `TreeKVCache` and `BranchReaper` are single-controller reference components;
  they do not synchronize concurrent callers.
- There is no winner merge, durable artifact handoff, hibernation, migration,
  or resume protocol.
- The process reaper supervises `subprocess.Popen` children, not Firecracker
  VMMs through the standalone `MicroVM` class.

### Inference engine

- The SGLang patch is additive and pinned to one commit. It is not upstreamed or
  wired into SGLang request scheduling, model execution, HTTP serving, tensor
  parallelism, or a multi-worker router.
- The live `sgl.Engine` measurement exercises stock RadixAttention. End-to-end
  requests carrying `tree_id`/`branch_id` through the patched cache remain
  unvalidated.
- GPU validation used one NVIDIA A10 and a synthetic 2 GiB fp16 KV pool. The
  live baseline used Qwen3-0.6B. 70B-class models, tensor/pipeline parallelism,
  mixed workloads, allocator pressure, and scheduler contention are unmeasured.
- The 10k-branch and tree-control experiments use small CPU tensor shapes. They
  validate metadata and allocator accounting, not 10k concurrent model
  executions.
- Quotas and reservations are cache-level token accounting. Without scheduler
  integration they cannot prevent another caller from allocating first.

### Sandbox and process lifecycle

- Firecracker and GPU inference have not been colocated or connected through an
  API proxy in these experiments.
- Firecracker results use idle, CPU-only 256 MiB guests. A real agent, cold page
  faults, writable disks, networking, and snapshot identity handling can change
  latency and memory use. Full snapshot creation took 75–82 ms on the recorded
  host in addition to the approximately 1 ms pause API call.
- Full Firecracker process teardown was 31 ms p50 in the recorded run; the VMM
  stopped at signal delivery, but teardown completed later.
- `PR_SET_PDEATHSIG` was tested under crash injection, and the child rechecks
  its parent PID after installing the signal. The implementation still uses
  `preexec_fn`, which Python warns is unsafe in threaded programs.
- macOS and Windows cannot run the pidfd demo or reaper tests.

### Evidence and economics

- The observed workload-shape check failed: the available organic traces did not
  show both fanout ≥ 8 and prompt-prefix fraction ≥ 0.2. Those private traces
  are not included, so their exact census is not independently reproducible.
- User-visible prompt text understates the full reusable prefix (system prompt,
  tool schemas, repository context), but the size of that difference has not
  been measured here.
- Provider-cache comparisons use assumed public pricing ratios rather than
  invoices. No end-to-end cost study includes orchestration, idle GPU capacity,
  snapshot storage, or network traffic.
- Captured benchmark outputs are in `report/RESULTS.md`, but not every original
  environment or raw input is checked into the repo.

## Related systems

These projects cover neighboring parts of the design. Their feature sets move
quickly; this table describes the distinction relevant to this prototype, not a
complete product comparison.

| Project | Relevant capability | Remaining question for this prototype |
|---|---|---|
| [forkd](https://github.com/deeplethe/forkd), [Mitos](https://github.com/mitos-run/mitos) | microVM snapshot/fork and CoW memory | How to bind sandbox identity to inference KV ownership and reclaim |
| [thaw](https://github.com/thaw-ai/thaw), [processfork](https://github.com/manav8498/processfork) | inference/session branching experiments | How to pair the inference branch with an isolated sandbox lifecycle |
| [SGLang](https://github.com/sgl-project/sglang) RadixAttention, [vLLM](https://github.com/vllm-project/vllm) APC | automatic shared-prefix KV reuse | How to add explicit agent-tree identity, policy, and branch-scoped lifecycle controls |
| [LMCache](https://github.com/LMCache/LMCache), [Mooncake](https://github.com/kvcache-ai/Mooncake), [Dynamo](https://github.com/ai-dynamo/dynamo) | KV movement, storage, and distributed serving | How tree ownership should compose with movement and routing |
| **agentfork (this prototype)** | tree-aware cache patch, reference reaper, and separate sandbox benchmark | Production integration of all three pieces |

## License

Apache-2.0 — see [LICENSE](LICENSE).
