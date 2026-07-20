# Integrated end-to-end run: live SGLang HTTP backend + real sandbox

This closes two caveats from `report/RESULTS.md`:

1. **`SGLangHTTPBackend` had never been run against a live SGLang HTTP
   server** (only protocol-tested with fakes).
2. **No recorded run drove the KV half and the sandbox half together** in one
   `ForkOrchestrator`.

Both are now exercised end to end and the exact numbers are captured below.

## Environment (honest infra note)

- Host: Ubuntu, kernel 5.15.200, **2 vCPU, ~7 GB RAM**, `/dev/kvm` present
  (nested KVM, `vmx`), passwordless `sudo`. **No NVIDIA GPU**
  (`nvidia-smi` absent), **no Modal** (`modal` CLI absent, no Modal env vars).
- The available `ANTHROPIC_API_KEY` / `TOGETHER_API_KEY` are hosted-LLM keys
  and cannot host an SGLang server, so they were not used.
- Patched SGLang checkout at `/home/ubuntu/sglang` (commit `40517b593` +
  patches `0001`/`0002`/`0003` via `tools/setup_sglang.sh`), torch CPU build.
- Firecracker **v1.11.0** (x86_64), guest kernel `vmlinux-5.10.225`
  (`CONFIG_VIRTIO_VSOCKETS=y`), agent rootfs built from the Firecracker CI
  `ubuntu-24.04.squashfs` base by `tools/build_rootfs.sh`.

### Why not a real GPU model server

The uncovered claim was specifically **`SGLangHTTPBackend` talking to a LIVE
HTTP server** (auth, `/tree_cache` lifecycle, `/tree_generate`, error paths).
A full `python -m sglang.launch_server` needs a CUDA GPU + model weights, which
this box does not have, and Modal was unavailable. So the live server here
(`demo/sglang_tree_server.py`) mounts the **real** patched pieces over a real
socket and stubs **only** the transformer forward:

| Piece | Real? | Source |
|---|---|---|
| KV pool + allocator (2 GiB fp16 CPU) | **real** | `MHATokenToKVPool` / `TokenToKVPoolAllocator` |
| Tree cache (fork/kill/reserve/telemetry, `lock_ref`, prefix match, charge, evict) | **real** | `TreeRadixCache` |
| `/tree_cache` op handler | **real** | `TreeCacheLifecycle.handle` (patch 0002) |
| `/tree_generate` committed-prefix charge validation | **real** | `TreeRadixCache._prepare_request_update` invariant (patch 0003) |
| Admin auth (`ADMIN_FORCE`, Bearer, constant-time, 401/403) | **real** | `sglang.srt.utils.auth.decide_request_auth` |
| Transformer forward pass (output text) | **stub** | no GPU/model — `meta_info.model_output=false` |

So `/tree_generate` performs the exact KV-cache work a prefill does — match the
shared prefix, allocate real KV slots for the uncached tail, insert, charge and
telemeter the branch — and returns the cache's own `cached_tokens`; only the
generated `text` is a placeholder. This is the same "small CPU-tensor pool"
approach the repo already uses in `patches/real_pool_validation.py`, now behind
a live HTTP surface driven by the real `SGLangHTTPBackend`.

## Reproduce

```bash
# 1. patch SGLang (once)
tools/setup_sglang.sh /home/ubuntu/sglang

# 2. start the live tree-cache server (real patched cache over HTTP)
PYTHONPATH=/home/ubuntu/sglang/python python demo/sglang_tree_server.py \
    --port 30444 --admin-api-key admin-secret &

# 3. integrated demo: live KV backend + 10 real Firecracker microVMs
sudo .venv/bin/python demo/integrated_demo.py \
    --sglang-url http://127.0.0.1:30444 --admin-api-key admin-secret \
    --sandbox firecracker --fc ./firecracker --kernel ./vmlinux-5.10 \
    --rootfs ./agent-rootfs.squashfs --children 10
```

## 1. Live-server auth + error paths (`SGLangHTTPBackend`'s server)

Captured directly against the live socket:

```
no key -> /tree_cache                            -> HTTP 401  {'error': 'Unauthorized'}
wrong key -> /tree_cache                         -> HTTP 401  {'error': 'Unauthorized'}
no key -> /tree_generate                         -> HTTP 401  {'error': 'Unauthorized'}
admin key -> /generate w/ branch_id (guarded)    -> HTTP 403  {'error': 'agent-tree fields require the admin /tree_generate endpoint'}
admin -> /tree_generate unknown branch           -> HTTP 400  {'error': "unknown branch: 'ghost'"}
admin -> /tree_cache negative reserve            -> HTTP 400  {'success': False, 'message': 'num_tokens must be nonnegative'}
health (always open)                             -> HTTP 200  {'status': 'ok'}
```

- 401s come from SGLang's own `decide_request_auth` (`ADMIN_FORCE` +
  constant-time Bearer compare).
- The 403 on `/generate` is patch 0003's guard: agent-tree fields must use the
  admin `/tree_generate` endpoint.
- The `negative reserve` 400 is the **real** cache raising
  `num_tokens must be nonnegative` through `TreeCacheLifecycle.handle`.

## 2. Integrated run — 10 real Firecracker microVMs + live KV backend

One `ForkOrchestrator(kv=SGLangHTTPBackend(...), sandbox=FirecrackerSandbox(...))`.
Verbatim stdout:

```
live server pool baseline: used=0 available=65536/65536
parent 'root': boot+prefill  1251.1 ms  KV cached_tokens=0 prompt_tokens=4392
fork 10 children (KV CoW + sandbox restore)   200.6 ms total,  20.1 ms/branch

    root/1: KV cached=4392 charged=164 | sandbox exit=1  174.8 ms  candidate 0 check: FAIL
    root/2: KV cached=4405 charged=151 | sandbox exit=1   11.2 ms  candidate 1 check: FAIL
    root/3: KV cached=4405 charged=151 | sandbox exit=1    9.8 ms  candidate 2 check: FAIL
    root/4: KV cached=4405 charged=151 | sandbox exit=1    9.2 ms  candidate 3 check: FAIL
    root/5: KV cached=4405 charged=151 | sandbox exit=1   13.4 ms  candidate 4 check: FAIL
    root/6: KV cached=4405 charged=151 | sandbox exit=1    9.0 ms  candidate 5 check: FAIL
    root/7: KV cached=4405 charged=151 | sandbox exit=1   10.7 ms  candidate 6 check: FAIL
    root/8: KV cached=4405 charged=151 | sandbox exit=1    9.1 ms  candidate 7 check: FAIL
    root/9: KV cached=4405 charged=151 | sandbox exit=1   11.3 ms  candidate 8 check: FAIL
   root/10: KV cached=4405 charged=151 | sandbox exit=0    9.2 ms  candidate 9 check: PASS

telemetry(root): charged_tokens=5915 pinned_tokens=5915 saved_tokens=44037 live_branches=11
KV reuse per child (cached_tokens): [4392, 4405, 4405, 4405, 4405, 4405, 4405, 4405, 4405, 4405]
live server pool at peak: used=5915 live_branches=11

winner root/10: exported artifact -> /tmp/.../winner_artifact.tar (10240 bytes)
kill_losers(root/10): 9 branches   85.4 ms, KV freed per loser={'root/1': 151, 'root/2': 151, 'root/3': 151, 'root/4': 151, 'root/5': 151, 'root/6': 151, 'root/7': 151, 'root/8': 151, 'root/9': 151}
total KV tokens freed on kill: 1359
live server pool after kill_losers: used=4556 live_branches=2 (winner still pinned)
live server pool after close: used=0 available=65536 (baseline=65536)

INTEGRATED E2E: PASS
```

### What the numbers show

- **Cached-token reuse per child:** the parent prefilled 4,392 tokens
  (`cached=0`); every child then reused the shared prefix —
  `cached=4,392` for the first child and `4,405` for the rest (the extra 13 are
  the common `"\n# candidate "` bytes the first child already inserted into the
  shared namespace). Each child only **charged 151–164** new tokens for its
  distinct suffix. `saved_tokens=44,037` is the cache's own reuse tally.
- **Real sandbox per branch, same branch id:** each `root/i` ran a real command
  inside its own Firecracker microVM over vsock (exit codes 1 for the 9 losers,
  0 for the winner whose in-guest check PASSed).
- **Freed tokens on kill > 0:** `kill_losers` reaped 9 losing microVMs **and**
  freed each loser's **151** uniquely-pinned KV tokens (**1,359 total**); the
  winner `root/10` and the shared prefix stayed pinned (`used=4,556`).
- **Allocator returns to baseline:** peak `used=5,915`
  (= 4,392 + 164 + 9×151 + 3, the charged totals), and after the orchestrator
  closed (winner + parent killed) the live server's real allocator was back to
  `used=0`, `available=65,536` — **exactly the baseline**.
- **Winner artifact exported:** `export_artifact` tar'd `/tmp/agentfork/fix.diff`
  out of the winner's guest over vsock (10,240-byte tarball).
- **Timings:** parent boot+prefill 1,251 ms; 10-way fork 201 ms
  (20 ms/branch, KV CoW + microVM restore); per-child in-guest exec 9–175 ms
  (first is cold); `kill_losers` 85 ms for 9 branches.

## 3. Client bug the live server exposed (and the fix)

Running `SGLangHTTPBackend` against the live server surfaced a real client bug
the fake-based tests could not: the fakes answered a failed `/tree_cache` op
with **HTTP 200 + `{"success": false}`**, but the **real** patched server
(`http_server.py::tree_cache`) returns **HTTP 400** with that body. So
`_request_json` raised a transport error *before* `_operation` could inspect
`success`, and `has_tree()` — which probes via `telemetry()` and only
recognized `"no such"/"not found"/"http 404"` — re-raised instead of reporting
absence. Against the real server, `has_tree(<killed branch>)` threw.

Minimal fix in `agentfork/kv/sglang_http_backend.py` (no auth/validation
weakened):

- `_request_json(..., expect_op_result=True)` now returns the structured body
  for an **HTTP 400** whose JSON is a `{"success": ...}` object, so `_operation`
  applies its own success/idempotency logic. Auth (401/403) and 5xx stay hard
  errors.
- op failures raise a typed `SGLangTreeCacheError` (a `RuntimeError` subclass),
  and `has_tree()` treats *only* that as "branch absent" — a transport/auth
  error still propagates rather than being misread as absence.

New regression tests (`tests/test_sglang_http_backend.py`) pin the 400-body and
the "401 is not absence" behaviors. Two pre-existing tests in
`tests/test_sglang_live_server.py` were **aspirational and never runnable**
against any server — they called `generate(root, prompt, max_new_tokens=1)`
(the shipped API takes a `sampling_params` dict and returns a dict, not a
string) and asserted the wrong guarded field. They were corrected to the real
API and, gated on `AGENTFORK_SGLANG_URL`, **actually run against this live
server**:

```
$ AGENTFORK_SGLANG_URL=http://127.0.0.1:30444 AGENTFORK_SGLANG_ADMIN_KEY=admin-secret \
    pytest tests/test_sglang_live_server.py -v
tests/test_sglang_live_server.py::test_live_tree_lifecycle_create_fork_generate_kill PASSED
tests/test_sglang_live_server.py::test_live_public_generate_rejects_tree_fields PASSED
2 passed
```

The default suite (no `AGENTFORK_SGLANG_URL`) skips them and stays green on
GPU-less CI: `189 passed, 2 skipped`.

## 4. Fallback path — `ReaperSandbox` (no `/dev/kvm`/root needed)

`--sandbox reaper` runs the same orchestrator with a real per-branch
subprocess via the shipped `ReaperSandbox` pidfd path. Because `ReaperSandbox`
runs one argv template and has no in-guest exec, the per-candidate command and
the winner artifact are run/exported at the host level and labelled
`[host subprocess]` / `[host copy]`. 5-child run also ended `INTEGRATED E2E:
PASS` with the live pool returning to `used=0`. This is the documented fallback
for hosts where Firecracker is not runnable.

## Firecracker: what worked and what would block it elsewhere

Firecracker **was runnable here** and is what the flagship run above used
(10 real microVMs). Requirements that had to be satisfied: `/dev/kvm` +
nested-virt (`vmx`), a `firecracker` binary (v1.11.0), a vsock-enabled guest
kernel, an agent rootfs from `tools/build_rootfs.sh` (needs `squashfs-tools` +
`sudo`), and **root** to open `/dev/kvm` (the `ubuntu` user is not in the `kvm`
group). On a host missing any of these, use `--sandbox reaper`.
