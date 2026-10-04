# agentic map-reduce

Karpathy's autoresearch runs one idea at a time: edit, measure, keep or revert.
This program runs the same honest loop in breadth. Each generation fans out K
ideas from the current frontier into isolated worktrees, measures every
candidate with the same deterministic eval, and keeps only the top B branches
as the next frontier. Reasoning is spent proposing and
implementing; selection is arithmetic (`amr.py`). State is `results.tsv` + git,
both append-only, so a crashed or interrupted run resumes cleanly.

You are the agent executing this program. Copy `amr.py` and this file into the
target repo, fill in the parameters, and run the loop. Do not modify `amr.py`
or the loop below — only the parameters and the target they point at.

## Parameters (edit for your domain)

| param        | value |
|--------------|-------|
| eval_cmd     | `<command that runs the full eval and writes a log, e.g. uv run train.py > run.log 2>&1>` |
| metric_grep  | `<how to extract the metric from the artifact, e.g. grep '^val_bpb:' run.log>` |
| metric       | `<name>` — minimize (or maximize) |
| K            | 4 — ideas proposed per frontier node per generation |
| B            | 2 — branches kept per generation (beam width) |
| eval_slots   | 1 — max evals running at once (= GPUs or other contended resource; set to K if evals are cheap) |
| timeout      | `<kill an eval that exceeds this; treat as crash>` |
| cost_guard   | `<optional: name:+tolerance, e.g. peak_vram_mb:+0.15>` |
| holdout_cmd  | `<optional: a second eval never used during the loop>` |
| tag          | `<run tag, e.g. jul6>` |

## Setup

0. **Resume check.** If `results.tsv` exists, this run is already underway.
   Run `python3 amr.py tree`, then `python3 amr.py reduce --gen <latest> --beam B --margin M`
   to reconstruct the frontier, and rejoin the loop from there. Never trust your
   memory of prior generations — the tsv and git are the only state.
1. Confirm branch `amr/<tag>` does not exist; create it from the current
   default branch. Keep `results.tsv` and eval logs untracked — add them to
   the target repo's `.gitignore` if they are not already ignored.
2. **Establish the noise margin.** Run eval_cmd twice on the unmodified
   baseline and log both runs:
   `python3 amr.py log 0 <short-commit> - <score> ran "baseline run N"`.
   margin = |run1 − run2|, floored at one unit in the metric's last meaningful
   digit. Deltas within the margin are not signal, in either direction.
3. Confirm the setup with the user, then begin the loop and do not stop.

## The generation loop (gen = 1, 2, 3, ...)

LOOP FOREVER:

1. **Frontier.** For gen 1 the frontier is simply the baseline commit. For
   every later gen: `python3 amr.py reduce --gen <prev> --beam B --margin M`
   (add `--maximize` if maximizing) — the `FRONTIER:` line lists the parent
   commits for this generation.
2. **Review + propose.** For each frontier node, inspect the artifacts, prior
   descriptions, and laws in `results.tsv`. Then propose K ideas. One variable
   per idea — if you can't name the single thing that changed, split it. Each
   idea gets a one-line rationale naming what it is trying to fix or exploit,
   and may include regions like `optimizer`, `architecture`, `prompt_format`,
   or `retrieval`. Never re-propose a pruned family.
3. **Shard.** For each idea:
   `git worktree add ../<tag>-g<gen>-<slug> -b amr/<tag>/g<gen>-<slug> <parent-commit>`
4. **Map.** One worker per worktree — subagents in parallel if you can spawn
   them, sequentially yourself otherwise (same tree, same verdicts, just
   slower). A worker's entire job: implement the ONE idea, commit, run
   eval_cmd, stop. Give each worker only its worktree path, its idea and
   rationale, eval_cmd, and the constraint that it touches nothing outside its
   worktree. Every candidate in a generation must use the same eval split,
   seed if applicable, budget, and scoring command. **Resource guard:** never
   let more than eval_slots evals run at once — implementation can overlap,
   measurement cannot.
5. **Collect — verify from artifacts.** Workers do not report numbers. For
   each worktree, YOU extract the score with metric_grep from the artifact
   eval_cmd wrote. Empty grep = crash: read the tail of the log; if the fix is
   trivial (typo, missing import) fix and re-run once, otherwise it's a crash.
   Log every idea either way. Include costs and regions when you have them:
   `python3 amr.py log [--cost name=value] [--region name] <gen> <short-commit> <parent-commit> <score|-> <ran|crash> "<what it tried>"`
6. **Reduce.** `python3 amr.py reduce --gen <gen> --beam B --margin M`.
   Add `--cost name:+tolerance` for any cost guard. A candidate survives only
   by beating ITS OWN parent by more than the margin and satisfying the cost
   guards; survivors are ranked globally and the top B become the next
   frontier.
7. **Advance.** `git worktree remove` the losers and delete their branches —
   `results.tsv` is the permanent record. Keep the survivors' worktrees.
8. **Stall rule.** If reduce prints STALL, the frontier is unchanged. Record
   what you learned:
   `python3 amr.py log <gen> - - - note "LAW: <family> doesn't pay because <evidence>"`
   and propose the next generation from NEW hypothesis families. Do not retry
   a pruned family with tweaks.
9. **Combination candidate.** If reduce prints `FUSE_CANDIDATE`, try that
   combination as one extra candidate next generation. It competes like any
   other — combined winners often don't add.

Repeat. NEVER STOP to ask whether to continue. The human may be asleep and
expects the loop to run until manually interrupted. Out of ideas means think
harder: re-read the target, re-read the laws, go more radical.

## Finishing (only when the human stops you)

The best branch is only a champion after it revalidates. Run holdout_cmd on
both the baseline commit and the champion. If no holdout_cmd exists, re-run
eval_cmd on both as a weaker final check. Ship only if the champion still wins
beyond the margin and satisfies the same cost guards. A map-set win that dies
on holdout is a false positive, not a result. If it holds, merge the champion
branch into `amr/<tag>` and report, including `python3 amr.py tree` output. If
it doesn't, say so plainly.

## Rules

- Review artifacts and prior results before proposing.
- One variable per idea.
- Every candidate in a generation uses the same eval split, seed if applicable, budget, and scoring command.
- Scores come from artifacts you grep yourself, never from a worker's claim.
- `results.tsv` is append-only. Losing branches die; their rows never do.
- Deltas within the noise margin are not signal.
- Simplicity criterion (from autoresearch): all else equal, simpler wins; an
  improvement that deletes code is the best outcome.
- Negative results are results — write them down as laws (note rows).
