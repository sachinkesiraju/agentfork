---
name: agent-mapreduce
description: Run a recursive agentic map-reduce experiment - beam search over research branches. Fans out K single-variable ideas per generation into isolated git worktrees, scores each with the same fixed eval command, and keeps the top B as the next frontier. Use when the user wants to systematically optimize a measurable target (training run, prompt, retrieval config, compiler flags) and a command can score each attempt. Triggers on "map-reduce experiment", "beam search my experiments", "fan out ideas and keep the best", "agent mapreduce".
---

# agent-mapreduce

You are about to execute the program in `program.md`, which sits in this skill's directory alongside `amr.py`. The skill is a thin launcher; the program is the contract.

## Launch

0. If `agentfork` is installed, skip the copy: `agentfork amr` is the same tool and the dashboard runs the loop itself (`agentfork loop <project> start`). This skill then documents the contract only.
1. Otherwise, copy `amr.py` and `program.md` from this skill's directory into the root of the target repo, unless they are already there. Make sure `results.tsv` and eval logs are gitignored in the target repo.
2. Fill in the parameter table at the top of the copied `program.md`. Derive what you can from the repo itself (eval command, metric name, GPU count). Ask the user for anything you cannot derive - at minimum `eval_cmd`, `metric_grep`, and whether the metric is minimized or maximized.
3. Confirm the filled-in parameters with the user once, then execute `program.md` exactly: resume check, baseline noise margin, then the generation loop. Do not modify `amr.py` or the loop itself - only the parameters and the target they point at.

## Non-negotiables (from the program)

- State lives in `results.tsv` and git only. If `results.tsv` already exists, this is a resume: rebuild the frontier with `python3 amr.py tree` and `reduce`, never from memory.
- Scores are grepped from eval artifacts by the orchestrator. A worker's claim is not a measurement.
- A candidate survives only by beating its own parent by more than the noise margin and passing any cost guards.
- The final champion must revalidate on holdout before it ships.
