# agentfork

Fork a live agent — its sandbox **and** its LLM KV context — as one primitive.
Kill any branch and reclaim both halves in **sub-millisecond to single-digit
milliseconds**, with zero orphans and zero leaked KV pages.

![tree-keyed KV: one resident prefix, N logical branches](docs/img/kv-dedup.svg)

**Measured at a glance:** 22 ms 10-way fork · 0.5 ms kill · 9.65× HBM dedup ·
524-line additive SGLang patch.

## What it does

agentfork gives a runtime two new operations:

- **`fork(parent)`** creates a child that shares the parent's cached context and
  runs in its own sandbox.
- **`kill(child)`** stops the child and reclaims its memory, both in the
  operating-system process and in the GPU KV cache.

It is built from three pieces: a small additive SGLang patch that tree-keys the
radix KV cache, a Firecracker snapshot/restore driver, and a `pidfd`-based
reaper. It is not a full agent framework or a scheduler — it is the runtime
primitive an orchestrator calls to build `map`/`reduce` agent fanout.

Use it for:

- `map`/`reduce` agent fanout (e.g. [agent-mapreduce](https://github.com/sachinkesiraju/agent-mapreduce)).
- Speculative coding fixes: fork multiple candidates from one repo context and pick the best.
- Verification trees: cheap verifiers kill most branches; the winner commits.
- Search and planning agents that fan out candidate plans, score them, and prune.
- Long-lived agents that hibernate and resume without re-prefill.

## Example: tree-style agent fanout

agentfork is designed for agent workflows that look like `map`/`reduce` over a
process tree. The new primitive is **agent fanout**: a resident agent reaches a
decision point, `fork`s into N candidate branches that share the parent's
context and sandbox state, and then `kill`s the losers and folds the winner
back into the parent.

Imagine a coding agent that has already digested a 32k-token repo. It needs to
test 10 possible fixes. With agentfork, you fork the agent 10 times from that
exact state. Each branch reuses the warm context and sandbox, so it only pays
for the tokens it generates. Run cheap verifiers, kill the 9 losers, and merge
the winner. Without it, you boot 10 cold sessions and re-read the repo 10 times.

![agentfork lifecycle: fork a live agent, race the branches, kill the losers](docs/img/lifecycle.svg)

The fanout cost changes from

```
N × (shared parent work + unique branch work)
```

to

```
shared parent work + N × unique branch work
```

## Quickstart

```bash
pip install -e .
python demo/demo.py   # 30-second CPU-only fork/race/kill demo
pytest                # unit tests, no GPU needed
```

`demo.py` prefills a 32k-token parent, forks 10 branches (0 re-prefill,
11× KV dedup), lets one win, kills the rest, and ends with 0 resident KV
tokens and 0 live trees.

The same primitives in Python:

```python
from agentfork.kv.tree_cache import TreeKVCache
from agentfork.kill.reaper import BranchReaper

cache = TreeKVCache()
parent = cache.create_tree("tree-1")
cache.extend(parent.tree_id, prefix_tokens)

child = cache.fork_branch("tree-1", "tree-1/branch-1")
reaper = BranchReaper(kv_cache=cache)
# ... spawn a sandbox bound to child.tree_id ...
reaper.kill("tree-1/branch-1")   # reaps process and drops KV refs
```

**Compatibility:** Linux ≥ 5.7, SGLang @ `40517b593`, Firecracker v1.7. GPU
validation runs on Modal (NVIDIA A10). 70B-scale tensor parallelism and
microVM+GPU colocation are unmeasured.

## How it works

`fork` and `kill` bind three lifetimes into one operation:

1. **KV cache fork** — `patches/0001-sglang-tree-radix-cache.patch` adds
   `TreeRadixCache` to SGLang. Children inherit the parent's KV prefix
   copy-on-write and keep it pinned through the existing `lock_ref` machinery.
   `kill_tree()` unpins pages in O(depth). The patch also adds per-tree HBM
   quotas, fork-time suffix reservations, priority demotion/promotion, explicit
   invalidation, and per-tree telemetry. No scheduler or allocator changes.
2. **Sandbox fork** — `agentfork/sandbox/fc_bench.py` drives Firecracker
   snapshot/restore with page-level CoW. ~1 ms parent pause, ~2 ms restore/child.
3. **Unified kill** — `agentfork/kill/reaper.py` uses Linux `pidfd` tied to the
   KV refcount, so `kill(tree_key)` reaps both the sandbox process and its
   KV pages in ~0.5 ms.

```
Your control plane
        │
        ▼
   agentfork runtime
   ├── TreeRadixCache   (SGLang patch)  fork_branch / kill_tree / quota / telemetry
   ├── FCForker         (Firecracker)  snapshot / restore / kill
   └── BranchReaper     (pidfd + KV)    kill(tree_key)
```

CUDA state cannot be `fork(2)`-ed. The KV fork is logical CoW (refcounted
radix nodes), not an OS-level fork of GPU state.

## Measured results

Every claim is backed by a measured, rerunnable benchmark. See
[report/RESULTS.md](report/RESULTS.md) for the full gate table, including the
gates that fail.

| Claim | Measured |
|---|---|
| Sibling KV prefix reuse (10-way fanout) | 100% reference / **99.9% on a live SGLang engine** (2,402 of 2,404 tokens cached per sibling) |
| GPU HBM dedup on a real KV pool (NVIDIA A10) | **9.65×** (37k slots vs 357k unshared), fork 10-way in 22 ms, allocator back to 0 after kill-all |
| Sibling completion vs cold prefill (live engine) | **33 ms p50 vs 9.07 s** |
| kill → process + KV reclaimed | **0.5 ms p50 / <2 ms max**, 100 cycles, 0 leaks, 0 orphans |
| Supervisor SIGKILLed (crash injection) | **0 orphans** in 50×5 cycles, kernel reaps in 1.5 ms |
| microVM fork | ~1 ms parent pause, **2.1 ms p50 restore/child**, 25-way fanout in 150 ms |
| microVM memory CoW | RSS 118 MiB vs PSS 24 MiB across 25 children → **4.95×** page sharing |
| SGLang patch | **524 additive LOC**, 17/17 new unit tests pass, zero regressions |
| Scale: fork one prefix into 10,000 branches | **0.95 s (10.5k forks/s), zero physical KV copies**; bulk kill of 10,001 in 0.17 s, allocator back to 0 |
| Tree-native control plane: quotas, reservations, demotion, invalidation, telemetry | runaway tree blocked at quota; 100 admission decisions in 3.3 ms (34/66 admit/reject); demote/promote keeps prefix under eviction pressure |

**Economics** (`agentfork/bench/cost_model.py`): 6.5–14.5× vs no-cache and
1.7–2.5× vs provider caching, but ~1.0× compute vs a well-run self-hosted
radix cache. The real wins are HBM residency, branch latency, and leak-free
kill, not prefill amortization.

## Running benchmarks

```bash
pytest                                  # unit tests (CPU only)
python -m agentfork.bench.kill_bench   # kill-path benchmark
python -m agentfork.bench.crash_bench  # crash-injection orphan test
python -m agentfork.bench.cost_model --children 10 --prefix 32000 --suffix 2000
# Firecracker bench (needs /dev/kvm + firecracker binary + guest kernel/rootfs):
python -m agentfork.sandbox.fc_bench --fc ./firecracker --kernel vmlinux --rootfs rootfs.ext4
# GPU validation (needs a Modal account):
modal run modal_gpu_validation.py
```

## Limitations

- GPU validation is at 0.6B-model scale on a single A10; 70B-scale allocator
  pressure, tensor parallelism, and scheduler contention are unmeasured.
- microVM + GPU colocation (vfio passthrough or API-proxied inference) is
  unvalidated — sandbox fork and KV fork are two operations coordinated by a
  control plane: a two-phase transaction, not atomic at the memory level.
- The workload-shape gate fails on organic traces.
- Provider-cache baseline uses public pricing ratios, not measured bills.

## Why agentfork vs. alternatives

Existing tools give you one piece of the tree — a fast sandbox fork, a way to
branch an inference session, or a way to move KV caches around. None of them
bind the three lifetimes together: fork the sandbox without the KV cache and
the GPU context is orphaned; branch the KV cache without the process and you
have no sandbox isolation or clean kill. The table below compares the main
alternatives and what each leaves unaddressed.

| Project | What it covers | Why it is not enough for a fork-native agent runtime |
|---|---|---|
| [forkd](https://github.com/nikita-vanyasin/forkd), [Mitos](https://github.com/mitos-run/mitos) | microVM fork with CoW | No KV fork; the GPU / inference context lives outside the sandbox |
| [thaw](https://github.com/thaw-ai/thaw), [processfork](https://github.com/manav8498/processfork) | vLLM/SGLang inference session branch | No sandbox; no process lifetime; no integrated kill/reclaim |
| [SGLang](https://github.com/sgl-project/sglang) RadixAttention, [vLLM](https://github.com/vllm-project/vllm) APC | prefix caching | Flat, request-level cache; no process-tree identity; no per-branch kill |
| [LMCache](https://github.com/LMCache/LMCache), [Mooncake](https://github.com/kvcache-ai/Mooncake), [Dynamo](https://github.com/ai-dynamo/dynamo) | KV transfer / tiering | Moves KV between workers; does not fork agent state or kill branches atomically |
| **agentfork** | **KV fork + sandbox fork + kill, unified by `tree_key`** | — |

## License

Apache-2.0 — see [LICENSE](LICENSE).
