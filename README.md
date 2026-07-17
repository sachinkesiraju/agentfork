# agentfork

Fork a live agent — its sandbox **and** its LLM KV context — as one coordinated lifecycle.
Kill any branch and reclaim both halves in **sub-millisecond to single-digit
milliseconds**, with zero orphans and zero leaked KV pages.

![tree-keyed KV: one resident prefix, N logical branches](docs/img/kv-dedup.svg)

**Measured at a glance:** 22 ms 10-way patched-cache create+extend · 0.53 ms
subprocess + CPU reference-cache kill · 9.65× KV-slot reduction vs an explicitly
unshared allocation · 547-line additive SGLang patch.

## What it does

agentfork prototypes two runtime operations:

- **`fork(parent)`** creates a child that shares the parent's cached context and
  runs in its own sandbox.
- **`kill(child)`** stops the child and reclaims its process and KV ownership.

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
process tree. The new primitive is **agent fanout**: a resident agent reaches a
decision point, `fork`s into N candidate branches that share the parent's
context and sandbox state, and then `kill`s the losers while the application
persists the winner.

Imagine a coding agent that has already digested a 32k-token repo and prepared a
working build environment. It needs to test 10 possible fixes. In a completed
agentfork integration, you fork the agent 10 times from that exact state. Each
branch reuses the warm context and initial sandbox state, so it pays only for
its unique work. Run cheap verifiers, kill the 9 losers, and persist the winner.
Without it, you boot 10 cold sessions and re-read the repo 10 times.

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

The advantage is largest when shared setup is long, branches are numerous and
short, and cheap verification rejects most candidates early. It shrinks when
fanout is small or most work happens after branches diverge.

## Quickstart

```bash
pip install -e ".[dev]"
python demo/demo.py   # Linux, CPU-only reference fork/race/kill demo
pytest -q             # non-Linux hosts skip pidfd integration tests
```

`demo.py` creates a synthetic 32k-token parent in the CPU reference cache,
forks 10 branches (0 re-prefill, 11× KV dedup before divergence and 9× after),
lets one win, kills the rest, and ends with 0 resident reference-cache tokens
and 0 live trees. It does not run a real LLM or Firecracker guest.

The same reference lifecycle in Python:

```python
import sys

from agentfork import ForkOrchestrator, ReaperSandbox

sandbox = ReaperSandbox([sys.executable, "-c", "import time; time.sleep(60)"])
with ForkOrchestrator(sandbox=sandbox, registry_path="branches.json",
                      default_lease_s=600) as orch:
    orch.create_parent("parent", tokens=prefix_tokens)
    children = orch.fork("parent", n=10)
    for child in children:
        orch.extend(child.branch_id, unique_suffix_tokens)
    orch.kill_losers(children[0].branch_id)
```

`kill()` reaps the sandbox, then the KV branch, then removes the journal record.
The steps are sequential, not atomic; `reconcile()` retries journaled work left
by a failed or crashed supervisor.

**Compatibility:** Python ≥ 3.10; Linux ≥ 5.4 for the `pidfd` reaper; SGLang @
`40517b593b23870cf351a05a1d53e930cea6a58d` for the patch. Firecracker v1.7
and an NVIDIA A10 on Modal are the measured environments. 70B-scale tensor
parallelism and microVM+GPU colocation are unmeasured.

## How it works

`ForkOrchestrator` gives the sandbox and CPU reference-cache branch one ID,
journals lifecycle intent, rolls back partial forks, retries interrupted cleanup,
and bounds branches with leases. The production backends remain separate:

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
3. **Process + reference-cache kill** — `agentfork/kill/reaper.py` uses Linux
   `pidfd` to reap a generic subprocess, then drops the matching CPU
   reference-cache entry. The measured combined path was 0.53 ms p50; it is not
   a Firecracker + GPU kill measurement.

```
ForkOrchestrator  (registry / leases / rollback / reconcile)
        │
        ▼
   coordinated branch ID
   ├── TreeRadixCache patch    fork_branch / kill_tree (not yet a backend)
   ├── Firecracker benchmark   snapshot / load / kill (not yet a backend)
   └── ReaperSandbox           pidfd process + CPU reference-cache backend
```

CUDA state cannot be `fork(2)`-ed. The KV fork is logical CoW over shared paged
KV slots, not an OS-level fork of GPU state.

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

The 9.65× result is versus an explicitly unshared allocation, not stock SGLang
RadixAttention. Stock SGLang already stores an identical cached prefix once.
The corrected cost model is ~1.0× compute and ~1.0× cache residency versus a
well-run same-namespace self-hosted prefix cache. The proposed gain over that
baseline is explicit ownership, branch policy, telemetry, and coordinated
reclaim—not another 9.65× memory reduction.

Provider-cache comparisons use token arithmetic with a 0.1× cached-read price
and 1.25× cache-write price, not measured invoices, latency, or HBM usage.

## Running benchmarks

```bash
pytest -q
python demo/demo.py
python -m agentfork.bench.kill_bench --cycles 100
python -m agentfork.bench.crash_bench --cycles 50 --children 5
python -m agentfork.bench.cost_model --children 10 --prefix 32000 --suffix 2000
# Firecracker bench (needs /dev/kvm + firecracker binary + guest kernel/rootfs):
python -m agentfork.sandbox.fc_bench --fc ./firecracker --kernel vmlinux --rootfs rootfs.ext4
# GPU validation (needs Modal and a patched SGLang checkout):
SGLANG_DIR=/path/to/sglang modal run modal_gpu_validation.py
```

## Limitations

- `ForkOrchestrator` coordinates the CPU reference cache and generic
  subprocesses. There is no production backend spanning a Firecracker microVM
  and the patched SGLang cache; cleanup is journaled and retried, but sequential
  rather than transactional.
- The SGLang patch is not wired into request scheduling, model execution, HTTP
  serving, tensor parallelism, or a multi-worker router. The live-engine result
  above is a stock RadixAttention baseline, not an end-to-end patch test.
- Cache budgets and reservations are logical direct-API accounting hooks, not
  scheduler-enforced physical HBM reservations.
- GPU validation uses one A10 and a Qwen3-0.6B stock-engine baseline; 70B-scale
  behavior, tensor parallelism, and scheduler contention are unmeasured.
- Firecracker measurements use idle, CPU-only 256 MiB guests; Firecracker and GPU
  inference have not been colocated or connected through an API proxy.
- `TreeKVCache` and `BranchReaper` do not synchronize concurrent callers, and
  the reaper uses `preexec_fn`, which Python warns is unsafe in threaded code.
- No winner merge, durable artifact handoff, hibernation, migration, or resume
  protocol is implemented.
- The observed workload-shape check failed on private organic traces that are
  not included in the repository.
- Provider-cache ratios are modeled rather than measured, and the Modal image
  is not digest-pinned.

## Why agentfork vs. alternatives

Existing tools give you one piece of the tree — a fast sandbox fork, a way to
branch an inference session, shared-prefix caching, or a way to move KV caches
around. agentfork explores one branch identity spanning sandbox state and
explicit KV ownership. This repository validates the pieces of that design but
does not yet integrate them into a production runtime.

| Project | What it covers | What remains for a fork-native agent runtime |
|---|---|---|
| [forkd](https://github.com/deeplethe/forkd), [Mitos](https://github.com/mitos-run/mitos) | microVM fork with CoW | Bind sandbox identity to inference KV ownership and reclaim |
| [thaw](https://github.com/thaw-ai/thaw), [processfork](https://github.com/manav8498/processfork) | inference/session branching experiments | Pair the inference branch with an isolated sandbox lifecycle |
| [SGLang](https://github.com/sgl-project/sglang) RadixAttention, [vLLM](https://github.com/vllm-project/vllm) APC | automatic shared-prefix KV reuse | Add explicit agent-tree ownership, branch policy, and sandbox coordination where required |
| [LMCache](https://github.com/LMCache/LMCache), [Mooncake](https://github.com/kvcache-ai/Mooncake), [Dynamo](https://github.com/ai-dynamo/dynamo) | KV transfer / tiering | Compose tree ownership with movement and routing |
| **agentfork (this prototype)** | tree-aware SGLang patch, CPU reference cache/reaper, and separate Firecracker benchmark | Integrate all three paths behind one recoverable control-plane operation |

## License

Apache-2.0 — see [LICENSE](LICENSE).
