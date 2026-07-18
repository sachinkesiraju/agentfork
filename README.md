# agentfork

agentfork is a runtime for tree-style agent fanout.

It forks a live agent's sandbox and its LLM KV context together, as one
branch. Killing a branch reclaims both halves in <1ms, with no orphan
processes and no leaked KV pages.

![tree-keyed KV: one resident prefix, N logical branches](docs/img/kv-dedup.svg)

**Measured at a glance:** 22 ms for 10 create+extend operations on an A10 ·
9.65× fewer KV slots than an explicitly unshared allocation · 1,080-line
additive SGLang patch set.

## What it does

agentfork implements two runtime operations, both given the same branch ID by
`ForkOrchestrator`:

- **`fork(parent)`** creates a child that shares the parent's cached context and
  runs in its own sandbox.
- **`kill(child)`** stops the sandbox and releases the child's KV state.

It is not an agent framework or a scheduler: it does not decide what an agent
does, only how a branch of it is created and torn down.

Out of the box, `ForkOrchestrator` uses the CPU reference cache
(`TreeKVCache`) and a no-op sandbox (`NullSandbox`), so you can run the KV
lifecycle with no GPU or VM; `ReaperSandbox` adds a real subprocess per
branch. Two heavier adapters plug into the same `KVBackend`/`SandboxBackend`
protocols:

- **`FirecrackerSandbox`** runs each branch in its own microVM, validated end
  to end on real hardware (`demo/fc_demo.py`): guest exec, writable overlays,
  per-branch networking, and the jailer.
- **`SGLangKVBackend`** (in-process) and **`SGLangHTTPBackend`** (over HTTP)
  fork the KV cache inside a patched SGLang engine. That request path was
  measured live on an A10G; the HTTP client is protocol-tested but has not yet
  run against a live server (see [report/RESULTS.md](report/RESULTS.md)).

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

Requires Python 3.10 or newer. The Firecracker sandbox needs Linux with
`/dev/kvm`; the SGLang KV backend needs a GPU host.

```bash
git clone https://github.com/sachinkesiraju/agentfork.git
cd agentfork
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
tools/setup_sglang.sh   # patches SGLang for the KV backend; prints launch commands
```

The lifecycle is `create_parent` / `fork` / `kill_losers` over two production
backends: a Firecracker microVM per branch (`FirecrackerSandbox`, see
`demo/fc_demo.py`) and a shared KV cache in a patched SGLang engine. With the
server up, fork candidates from a shared prompt and keep the winner, with
inference (`generate`) as the data path:

```python
from agentfork import ForkOrchestrator, SGLangHTTPBackend

kv = SGLangHTTPBackend(
    "https://sglang.example.internal", admin_api_key="admin-secret")
with ForkOrchestrator(kv=kv, registry_path="branches.json") as orch:
    orch.create_parent("parent")
    orch.generate("parent", "Shared context", {"max_new_tokens": 4})

    children = orch.fork("parent", n=3)  # three candidates from one prompt
    for child in children:
        orch.generate(child.branch_id, "Shared context\nCandidate:",
                      {"max_new_tokens": 64}, reserve_tokens=64)

    orch.kill_losers(children[0].branch_id)  # keep the winner, drop the rest
```

Run `pytest -q` to execute the test suite.

## How it works

`ForkOrchestrator` gives the sandbox and KV branch one ID, backed by a
single-owner, fsynced registry: it rolls back partial forks, retries
interrupted cleanup, and bounds every branch with a lease.

**The reference path works today, on CPU.** Forking a KV branch just adds a
reference count along the shared prefix; no tokens are copied. Each branch
runs as an ordinary subprocess. Killing a branch stops that subprocess with a
Linux `pidfd` and releases its cache entry; together that takes 0.53 ms
(median).

Two heavier backends implement the same interfaces:

1. **KV cache fork** (SGLang patches `0001`–`0002`). The patch adds a
   tree-aware KV cache and threads a branch ID through the engine's normal
   request path, so forking, killing, and per-branch quotas all happen inside
   SGLang. `SGLangKVBackend` talks to it in-process; `SGLangHTTPBackend` talks
   to it over HTTP. On a Modal A10G, ten forked children each reused all 2,406
   tokens of the parent's cached prompt, and killing a child freed its share.
   The HTTP client is tested against a stub; running it against a live server
   is still to do.
2. **Sandbox fork** (`FirecrackerSandbox`, over a small microVM wrapper). A
   branch is snapshotted only when it is first forked, so children start from
   the parent's current state and branches that are never forked pay nothing
   (snapshotting pauses the parent 76–83 ms; each child restores in about
   2 ms).
   Inside a guest you can run commands over a vsock channel (`exec`, with
   stdin, plus `exec_detached` for background jobs), wait for it to be ready
   (`wait_ready`), give it a private writable disk (copied cheaply per child),
   lock it down with the jailer, and put it on its own network with outbound
   internet. All of this is checked on real Firecracker v1.16.1: children run
   commands, write to their own disks, keep state the parent set after boot,
   and reach the internet (`HTTP 200`), both jailed and not.

```
ForkOrchestrator  (registry / leases / rollback / reconcile)
        │
        ▼
   coordinated branch ID
   │
   ├── KV branch
   │    ├── TreeKVCache            CPU reference cache (live)
   │    └── TreeRadixCache patch   via SGLangKVBackend / SGLangHTTPBackend
   │
   └── sandbox branch
        ├── ReaperSandbox          pidfd subprocess (live)
        └── Firecracker microVMs   via FirecrackerSandbox (live: exec, stdin,
                                   overlays, networking, jailer)
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

| What we measured | Result |
|---|---|
| KV sharing on a real SGLang GPU allocator (A10) | On a real A10 allocator, 10 branches sharing one 32k-token prefix used 37k slots rather than the 357k a full per-branch copy would take; the 10 create+extend calls ran in 22 ms, and kill-all returned the allocator to exactly 0. Stock SGLang already shares identical prefixes, so the win here is correct branch tracking and clean reclaim, not lower memory. ([code](patches/real_pool_validation.py)) |
| Live request path on a real GPU (A10G) | One parent, 10 children: every child reused all 2,406 cached parent tokens with zero re-prefill; explicit kill released the pin. |
| Sandbox fork on real Firecracker (v0.3.0) | 5-way fork at 28–145 ms per child (lazy fork-time snapshot, parallel restores); exec over vsock in every child; per-child writable overlay; identical results under the jailer; zero leaked VMMs. |
| Loser kill, reference path (subprocess + CPU cache) | 0.53 ms p50, 1.46 ms max over 100 cycles. |
| Guest networking on real Firecracker (v0.4.0) | Two children each reached the internet (GET example.com → HTTP 200) over per-branch netns + NAT; rules torn down with zero leaks; vsock exec 44–73 ms per call. |
| Sibling speedup under cache pressure, vs stock SGLang | When unrelated traffic evicts the shared prefix in stock SGLang but not in agentfork, the children generate 1.60× faster (sustained pressure, 95% CI [1.576×, 1.619×]) and 1.54× on a locked holdout (95% CI [1.518×, 1.554×]). Both synthetic; partner validation still pending. |

Grounding: forking a whole branch, its sandbox microVM plus its KV cache,
runs 28–145 ms per child, and the KV cache is under 1.3% of that. That puts
it in the same class as managed providers that fork only the sandbox
([Morph](https://cloud.morph.so/docs/developers) branches a full VM in under
250 ms). Forking the KV cache too is what lets a fanout reuse the shared
prompt instead of re-prefilling it in every child.

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
git -C "$SGLANG_DIR" apply "$PWD/patches/0002-Wire-branch-lifecycle-through-the-SGLang-request-pat.patch"
git -C "$SGLANG_DIR" apply "$PWD/patches/0003-Harden-tree-request-auth-and-accounting.patch"
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

- SGLang is measured on only one A10G/0.6B; scale, tensor parallelism, and
  multi-tenant pressure need a GPU fleet (a live-server test exists but is
  unrun here).
- Firecracker is single-host: moving migration bundles between hosts is the
  deployer's job, and cleanup is retried, not atomic.
- Nothing is validated at production GPU scale or with GPU-plus-microVM
  colocation.
- `ReaperSandbox` runs spawns serially by default; `pdeathsig="shim"` fans
  them out.
- Single-winner handoff exists (`export_artifact`); multi-winner merge does
  not.

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
| **agentfork** | Forks a sandbox and its KV cache under one branch ID, and reclaims both on kill | Live HTTP/OpenAI validation, multi-worker routing, and hosting it as a service |

## License

Apache-2.0. See [LICENSE](LICENSE).
