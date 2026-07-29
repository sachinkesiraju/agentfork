# Split-screen race demo: captured run (CPU)

`demo/race_demo.py` -- *10 fixes, 1 inference server, and someone else is using it too.*

Two arms solve the same planted bug against the same live tree-cache HTTP
server, one after the other, each against a freshly started server so neither
inherits the other's cache. An unrelated tenant streams traffic through that
server between candidates.

## What is real and what is not

This box has no GPU and no model weights, so **the transformer forward pass is
stubbed**. Everything the demo measures is real:

| real | stubbed |
| --- | --- |
| KV pool + allocator (`MHATokenToKVPool`, `TokenToKVPoolAllocator`) | the generated text |
| `TreeRadixCache`: prefix matching, charge accounting, `lock_ref` pinning, LRU eviction | generation latency |
| branch lifecycle (`create`/`fork`/`kill`/`demote`) over HTTP with admin auth | |
| the noisy neighbour's requests (real prefills through the same pool) | |
| sandboxes (`ReaperSandbox`, real subprocesses) and candidate checks (real `pytest` runs) | |

So the headline numbers are **prefill tokens charged** and the
**parent-prefix hit rate**, which are the cache's own measurements. Wall clock
is reported but is *not* a model-speed claim: with a stubbed forward pass it
mostly measures HTTP, `pytest` and process spawning.

`/dev/kvm` and a Firecracker binary/kernel/rootfs were not available on this
box, so the sandbox is `ReaperSandbox`. No Firecracker output is reported.

## Environment

* 2 vCPU / 7 GiB Linux 5.15 container, no GPU
* torch 2.13.0+cpu, patched SGLang @ `40517b593` + the three tree-cache patches
* repo at `73f7d53`

## Commands

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
tools/setup_sglang.sh ~/sglang

# live split-screen dashboard
PYTHONPATH=~/sglang/python .venv/bin/python demo/race_demo.py

# plain log + machine-readable JSON summary (what tests and CI use)
PYTHONPATH=~/sglang/python .venv/bin/python demo/race_demo.py --no-ui
```

Defaults: `N=10` candidates, `P=8192` shared-prefix tokens, `S=256` unique
tokens per candidate, `C=24576` KV pool tokens, neighbour requests of 1024
tokens metered so that `U = 17408` tokens land between consecutive candidates.
`U* = C - P = 16384`, so `U > U*`: the break-even model in
`agentfork/bench/cost_model.py` predicts the stock prefix cannot survive and
the pinned one must.

## UI mode: final frame

Captured from a real terminal (ANSI colour stripped); the bars, hit/miss flags
and KV meter update live as the race runs.

```text
== agentfork split-screen race ==
10 fixes, 1 inference server, and someone else is using it too.
CPU-only: KV pool, tree cache, eviction, pinning, auth, sandboxes and candidate checks are REAL;
the transformer forward pass is STUBBED (no GPU/weights), so read prefill tokens and hit rate,
not generation latency.
P=8192  C=24576  U=17408  U*=C-P=16384   U > U*  -> the neighbor evicts an unpinned prefix
N=10 candidates, 3 verification forks; neighbor metered between candidates

STOCK                                                              | AGENTFORK                                                         
------------------------------------------------------------------ | ------------------------------------------------------------------
server http://127.0.0.1:42541                                      | server http://127.0.0.1:47931                                     
shared context: 8192 tok charged (P=8192)                          | shared context: 8192 tok charged (P=8192)                         
KV [#################################...............]  16912/24576 | KV [################################################]  24528/24576
 | 
fix     1/10 [=.........] MISS cached=0      charged=8476   fail   | fix     1/10 [=.........] HIT  cached=8192   charged=284    fail  
fix     2/10 [==........] MISS cached=0      charged=8476   PASS   | fix     2/10 [==........] HIT  cached=8254   charged=222    PASS  
fix     3/10 [===.......] MISS cached=0      charged=8476   fail   | fix     3/10 [===.......] HIT  cached=8257   charged=219    fail  
fix     4/10 [====......] MISS cached=0      charged=8476   fail   | fix     4/10 [====......] HIT  cached=8257   charged=219    fail  
fix     5/10 [=====.....] MISS cached=0      charged=8476   fail   | fix     5/10 [=====.....] HIT  cached=8253   charged=223    fail  
fix     6/10 [======....] MISS cached=0      charged=8476   fail   | fix     6/10 [======....] HIT  cached=8253   charged=223    fail  
fix     7/10 [=======...] MISS cached=0      charged=8476   fail   | fix     7/10 [=======...] HIT  cached=8253   charged=223    fail  
fix     8/10 [========..] MISS cached=0      charged=8476   fail   | fix     8/10 [========..] HIT  cached=8261   charged=215    fail  
fix     9/10 [=========.] MISS cached=0      charged=8476   fail   | fix     9/10 [=========.] HIT  cached=8255   charged=221    fail  
fix    10/10 [==========] MISS cached=0      charged=8476   fail   | fix    10/10 [==========] HIT  cached=8253   charged=223    fail  
verify  1/3 [===.......] MISS cached=0      charged=8760   PASS    | verify  1/3 [===.......] HIT  cached=8476   charged=284    PASS   
verify  2/3 [=======...] MISS cached=0      charged=8760   PASS    | verify  2/3 [=======...] HIT  cached=8515   charged=245    PASS   
verify  3/3 [==========] MISS cached=0      charged=8760   PASS    | verify  3/3 [==========] HIT  cached=8515   charged=245    PASS   
 | 
killed 9 losers: KV freed 16506 tok (pool -16627)                  | killed 9 losers: KV freed 1988 tok (pool -1988)                   
hit 0%  prefill 119232 tok  VERIFIED                               | hit 100%  prefill 11238 tok  VERIFIED
```

## Scoreboard (same run)

```text
metric                                             STOCK           AGENTFORK
----------------------------------------------------------------------------
parent-prefix hit rate                                0%                100%
prefill tokens charged (real)                    119,232              11,238
peak KV pool used                                 16,912              24,528
KV released when losers died                      16,506               1,988
KV still pinned after the kills                        0               8,760
sandbox setup (s, total)                            0.38                0.05
wall clock (s)* -- not a speed claim                 3.6                 4.7
verified winner                          stock/root/2/11  agentfork/root/2/11
neighbor requests served                             221                 221
neighbor requests deferred                             0                   0

* the forward pass is stubbed, so wall clock here is mostly HTTP, pytest and process spawning:
  the gap between the arms is noise, not a speedup. The headline numbers are prefill tokens
  charged and the parent-prefix hit rate, which the cache measures for real.
Note: the stock arm also releases KV when its branches die -- but those were ordinary
evictable pages that the neighbour had already been recycling; it had nothing pinned to
protect, which is exactly why its shared prefix did not survive to the next candidate.
```

**10.6x fewer prefill tokens charged** (119,232 -> 11,238) for the same work,
same server, same neighbour, and the same verified fix. Note the wall clock in
this capture: the agentfork arm was *slower* (4.7 s vs 3.6 s). With the forward
pass stubbed there is no generation time for a cache hit to save, so that
column is noise -- which is exactly why it is not the headline.

## `--no-ui` log (same defaults, separate run)

```text
--- STOCK arm: fresh server at http://127.0.0.1:40251 ---
[stock] shared context prefilled: 8192 tokens charged (P=8192)
[stock/fix 1/10] stock/root/1: cached=0 charged=8476 parent_hit=False sandbox_setup=31ms check=fail
[stock/fix 2/10] stock/root/2: cached=0 charged=8476 parent_hit=False sandbox_setup=31ms check=PASS
[stock/fix 3/10] stock/root/3: cached=0 charged=8476 parent_hit=False sandbox_setup=36ms check=fail
...
[stock/fix 10/10] stock/root/10: cached=0 charged=8476 parent_hit=False sandbox_setup=31ms check=fail
[stock] killed 9 loser branch(es): KV freed=16506 tokens, pool used dropped by 16627
[stock/verify 1/3] stock/root/2/11: cached=0 charged=8760 parent_hit=False sandbox_setup=36ms check=PASS
[stock/verify 2/3] stock/root/2/12: cached=0 charged=8760 parent_hit=False sandbox_setup=37ms check=PASS
[stock/verify 3/3] stock/root/2/13: cached=0 charged=8760 parent_hit=False sandbox_setup=37ms check=PASS
[stock] done in 3.8s: hit_rate=0.00 prefill_charged=119232 peak_kv=16912 verified=True

--- AGENTFORK arm: fresh server at http://127.0.0.1:43041 ---
[agentfork] shared context prefilled: 8192 tokens charged (P=8192)
[agentfork/fix 1/10] agentfork/root/1: cached=8192 charged=284 parent_hit=True sandbox_setup=0ms check=fail
[agentfork/fix 2/10] agentfork/root/2: cached=8254 charged=222 parent_hit=True sandbox_setup=0ms check=PASS
...
[agentfork/fix 10/10] agentfork/root/10: cached=8253 charged=223 parent_hit=True sandbox_setup=0ms check=fail
[agentfork] killed 9 loser branch(es): KV freed=1988 tokens, pool used dropped by 1988
[agentfork/verify 1/3] agentfork/root/2/11: cached=8476 charged=284 parent_hit=True sandbox_setup=0ms check=PASS
[agentfork] done in 3.7s: hit_rate=1.00 prefill_charged=11238 peak_kv=24528 verified=True
```

### Machine-readable summary (excerpt)

The `--no-ui` run ends with a JSON object between
`===RACE_DEMO_JSON_BEGIN===` / `===RACE_DEMO_JSON_END===`:

```json
{
  "claims": {
    "agentfork_parent_hit_rate": 1.0,
    "stock_parent_hit_rate": 0.0,
    "agentfork_prefill_charged": 11238,
    "stock_prefill_charged": 119232,
    "agentfork_verified": true,
    "stock_verified": true,
    "agentfork_kv_freed_by_kills": 1988,
    "agentfork_pinned_after_kills": 8760,
    "agentfork_expected_pinned_after_kills": 8760
  }
}
```

### Measured vs. predicted

The demo also emits the cost model's prediction for the same `(P, N, S, U, C)`:

| | model | measured |
| --- | --- | --- |
| stock parent hit rate | 0.0 | 0.0 |
| pinned parent hit rate | 1.0 | 1.0 |
| stock prefill charged | 92,672 | 119,232 |
| pinned prefill charged | 10,752 | 11,238 |

The model counts only the fan-out round; the measured numbers also include the
parent prefill and the 3-fork verification round, which is why they are higher.
The hit rates -- the thing the break-even math actually predicts -- match
exactly.

## Boundary check (`U <= C - P`)

The advantage is a property of the pressure, not of the framing. Below the
break-even both arms keep the prefix, and `tests/test_race_demo.py` asserts
exactly that (`--noise-requests-per-gap 1`): stock hit rate 1.0, agentfork hit
rate 1.0. Pinning only wins where the model says it wins.

## Tests

```text
$ PYTHONPATH=~/sglang/python .venv/bin/python -m pytest -q tests/test_race_demo.py
........                                                                 [100%]
8 passed in 30.84s

$ PYTHONPATH=~/sglang/python .venv/bin/pytest -q
238 passed, 2 skipped in 46.77s

$ .venv/bin/ruff check agentfork demo tests
All checks passed!
```

Without the patched SGLang checkout on `PYTHONPATH` the race-demo tests skip
(they need the real cache); the rest of the suite is unaffected.

## Reproducibility

Three consecutive `--no-ui` runs gave bit-identical hit rates and prefill
charges (`1.0 / 0.0`, `11,238 / 119,232`); only wall clock and sandbox setup
times varied. The candidates come from the deterministic `FakeLLM` and the
neighbour is metered, so the cache measurements are stable.

## Caveats

* The dashboard wants a 135-column terminal; it shrinks its panels to fit
  narrower ones (dropping the progress bar and `cached=` column) and prints a
  warning below 83 columns. `--no-ui` is always safe.
* Wall clock is noise here, and this capture shows it: the agentfork arm took
  *longer* (4.7 s vs 3.6 s) while charging 10.6x fewer prefill tokens. With a
  stubbed forward pass there is no generation time for a cache hit to save.
* On this 2-vCPU box the whole race takes ~8 s rather than the ~60-90 s a
  human-paced demo wants: with a stubbed forward pass there is no generation
  time to spend. Scale it with `--children`, `--prefix-tokens` and
  `--noise-request-tokens` if you want a longer run; nothing is padded
  artificially.
* The neighbour is *metered* (`gap()` admits exactly `U` tokens between
  candidates) rather than free-running, so the interleaved token count is
  exact and the comparison against `U* = C - P` is meaningful in both
  directions.
* `charged`/`cached` counts are byte-level tokens: the demo server tokenizes
  UTF-8 bytes because there is no tokenizer to load without weights.

## Browser dashboard (`--web`)

```bash
PYTHONPATH=$HOME/sglang/python python3 demo/race_demo.py --web        # port 8765
```

Same race, same events, rendered in a browser instead of the terminal: a
stdlib `ThreadingHTTPServer` serves one self-contained HTML page and streams
the race over server-sent events (`demo/race_web.py`), so there is no build
step and no new dependency. Live KV bars per arm, a HIT/MISS row per candidate
with its cached/charged tokens, the kill event when the losers die, and the
scoreboard at the end. A browser that connects late gets the whole race
replayed, and the server stays up after the race so the result stays readable.

![browser dashboard](race_demo_web.png)

---

# Real GPU, real weights: the same race on an A10 (Modal)

Everything above runs on CPU with the forward pass stubbed. `modal_race_demo.py`
runs the same race against a real `sgl.Engine` serving **Qwen3-0.6B on an
NVIDIA A10**, so prefill, decode and wall clock are model measurements rather
than HTTP overhead.

```bash
SGLANG_DIR=/path/to/patched/sglang python3 -m modal run modal_race_demo.py
```

Captured 2026-07-29 (full JSON: [gpu_race_run.json](gpu_race_run.json)):

```
GPU NVIDIA A10   model Qwen/Qwen3-0.6B
C = 49,152   P = 12,725 (measured with the real tokenizer)
U* = C - P = 36,427      U = 37,327 injected between every candidate
```

| metric | stock | agentfork |
|---|---|---|
| parent-prefix hit rate | 0% | 100% |
| prefill tokens charged | 43,142 | 158 |
| generation time, 13 branches (real decode) | 8.67 s | 6.55 s |
| median per-candidate latency | 0.63 s | 0.50 s |
| KV reclaimed by killing 9 losers | n/a | 399 tok (their suffixes; the parent stays pinned) |
| verified winning fix | yes | yes |

**273x fewer prefill tokens and a real 1.32x end-to-end generation speedup** --
the CPU demo could only measure the first of those. Per-candidate detail shows
what the stock arm is actually doing: it is not missing entirely, it is
*partially* re-prefilling, because the neighbour keeps trimming the shared
prefix from the tail (`child-0` cached 8,192 of 12,758 tokens, `child-1` only
6,144, `child-2` 10,240 -- never the whole prefix). The agentfork children
cached 12,729-12,749 of the same prompts and were charged 9-29 tokens each.

Honest notes:

* The *patch* each branch proposes is still the deterministic candidate list,
  because Qwen3-0.6B does not reliably write a correct fix; the model genuinely
  generates on each branch's prompt, and the script separately checks what it
  wrote and reports it as `model_generated_fix_passed` (0/13 in this run, 1/13
  in an earlier smaller run -- reported, never folded into the race result).
* The speedup depends on `P` relative to decode length: with `MAX_NEW = 32`
  most of each request is prefill, which is exactly the regime a fan-out
  workload lives in. A shorter context shrinks it -- an earlier run at
  `P = 1,385` measured 1.04x while still showing 118x fewer prefill tokens.
* The agentfork arm's parent prefill is slower (3.23 s vs 0.59 s) because it
  also commits and pins the branch; it is paid once and amortised over 13
  children.
