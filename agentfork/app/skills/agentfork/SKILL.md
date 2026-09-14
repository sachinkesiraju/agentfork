---
name: agentfork
description: Drive a running agentfork dashboard from the CLI - create projects, run the map-reduce loop, inspect the experiment tree, read run logs. Use when the user wants to run autoresearch on a repo, fan out experiment candidates, or check on a running agentfork instance. Triggers on "agentfork", "autoresearch", "experiment tree", "fan out ideas".
---

# agentfork

`agentfork up` starts a loopback dashboard (default `http://127.0.0.1:8474`).
State lives in `~/.agentfork` (SQLite + per-project `results.tsv`, worktrees
and run dirs). Every node in the experiment tree is a real agentfork branch:
a git worktree + a KV context slice, cleaned up by `kill_losers`. If the user
binds off loopback (`agentfork up --host 0.0.0.0`), every API call needs the
bearer token printed in the dashboard URL — pass it to the CLI via
`AGENTFORK_TOKEN=...` (or open the printed `?token=` URL in the browser).

## Drive it

```sh
agentfork status                      # is a dashboard up?
agentfork harnesses                   # detected agent CLIs
agentfork new /path/to/repo --baseline main \
    --eval "python train.py" --metric "loss=([0-9.]+)" --harness claude-code \
    --eval-slots 2 --baseline-runs 2 --cost seconds=0.5  # flags all optional
agentfork baseline <project>          # root node + 2 evals → noise margin
agentfork loop <project> start        # autoresearch: propose→fan out→eval→reduce
agentfork tree <project>              # rendered experiment tree
agentfork runs <project>              # all runs
agentfork logs <run> -f               # tail a run's log
agentfork reduce <project> --gen N    # manual reduce (margin = baseline noise)
agentfork metrics <project>           # KV resident/dedup + fork/kill counters
agentfork laws <project>              # written-down constraints
agentfork kill <run>
```

## The contract

- Scores come from `metric_grep` on the eval run's own log — never a worker's claim.
- A candidate survives only by beating *its own parent* by more than the margin.
- `results.tsv` in the project dir is append-only and is the source of truth for resume.
- Champions must revalidate on `holdout_cmd` before shipping — and the loop
  only reports `done` when the champion also beats a fresh *baseline* holdout.
- After a dashboard restart, in-flight worker runs settle to `failed` and
  node worktrees are re-created (and the branch re-adopted) on first use —
  the node's `af/<id>` git branch + commit is the durable evidence; the
  worktree on disk is disposable.

## Map-reduce details

`agentfork amr --tsv <path> log|reduce|tree ...` runs the vendored bookkeeping
tool directly (byte-compatible with the standalone `amr.py` from
sachinkesiraju/agent-mapreduce). The skill `agent-mapreduce` (install with
`agentfork install-skills`) documents the full program contract for agent CLIs.
