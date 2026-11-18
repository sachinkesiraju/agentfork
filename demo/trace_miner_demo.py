"""Trace-mining demo: HypotheSAEs over agentfork branch transcripts.

Fans out N divergent branch rollouts per task over a real fork tree
(``TreeKVCache`` + ``NullSandbox`` — no GPU/Firecracker), labels each branch
transcript by exact-match outcome, then runs the HypotheSAEs pipeline
(embed -> SAE M=1024 K=32 -> lasso neuron selection -> LLM interpretation ->
holdout annotation -> regression metrics) and emits an early-turn pruning
signal: branches whose first turn activates failure-associated neurons get
killed.

    export TINKER_API_KEY=...
    python demo/trace_miner_demo.py --gen     # Tinker sampling + fork tree
    python demo/trace_miner_demo.py --mine    # HypotheSAEs pipeline
    python demo/trace_miner_demo.py           # both

Outputs land in ``demo/trace_miner_out/``: transcripts.jsonl,
hypotheses.csv, hypotheses_eval.csv, pruning_report.json.

LLMs (Tinker): branches and annotation run ``openai/gpt-oss-20b`` (harmony
channels — final extracted via ``<|channel|>final<|message|>``); neuron
interpretation runs ``moonshotai/Kimi-K2.6`` with a ``- "`` assistant prefill
(gpt-oss-120b's harmony analysis truncates before the final channel and
yields garbled interpretations).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "trace_miner_out")

BRANCH_MODEL = "openai/gpt-oss-20b"
WEAK_MODEL = "Qwen/Qwen3-8B"  # every 4th branch: coherent slips, real fails
INTERPRETER_TINKER = "moonshotai/Kimi-K2.6"
ANNOTATOR_TINKER = "openai/gpt-oss-20b"

N_BRANCHES = 32
TEMPERATURES = [0.4, 0.8, 1.0, 1.3]
PERSONAS = {
    "careful": ("You are a meticulous solver. Reason step by step, check your "
                "arithmetic, then give your final answer on its own line as "
                "'ANSWER: <value>'."),
    "quick": ("You are a fast solver. Go with your first instinct and keep "
              "reasoning brief, then give your final answer on its own line "
              "as 'ANSWER: <value>'."),
}

# (task_id, question, gold, plausible-wrong distractor used as the
# "another solver suggested" anchor on half the branches). All answers
# numeric; the checker normalizes.
TASKS = [
    ("bat-ball", "A bat and a ball cost $1.10 in total. The bat costs $1.00 "
     "more than the ball. How much does the ball cost, in cents?", "5", "10"),
    ("strawberry-r", "How many times does the letter 'r' appear in the word "
     "'strawberry'?", "3", "4"),
    ("sheep", "A farmer has 17 sheep. All but 9 die. How many sheep are "
     "left?", "9", "8"),
    ("widgets", "If 5 machines take 5 minutes to make 5 widgets, how many "
     "minutes would it take 100 machines to make 100 widgets?", "5", "100"),
    ("sally", "Sally has 3 brothers. Each of her brothers has 2 sisters. "
     "How many sisters does Sally have?", "1", "2"),
    ("bigchain", "Compute (23 x 47) + (31 x 59) - (12 x 84) + (66 x 11).",
     "2628", "2754"),
    ("8642x37", "What is 8642 times 37?", "319754", "318754"),
    ("7623x89", "What is 7623 times 89?", "678447", "677447"),
    ("sum3dig", "How many three-digit numbers have digits that sum to 6?",
     "21", "15"),
    ("coins", "A jar has 5 quarters, 4 dimes, 3 nickels, and 2 pennies. "
     "What is the total value in cents?", "182", "177"),
    ("rect2", "A rectangle with area 96 has integer side lengths and "
     "perimeter 40. What is the length of its longer side?", "12", "8"),
    ("zeros40", "How many trailing zeros does 40! have?", "9", "8"),
]

FINAL_MARKER = "<|channel|>final<|message|>"
ANSWER_RE = re.compile(r"ANSWER:\s*(.+)", re.I)
NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


# --------------------------------------------------------------------------
# Tinker sampling helpers
# --------------------------------------------------------------------------

def _tok(repo):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(repo)


def _chat_ids(tok, messages, prefill=None):
    if prefill is not None:
        messages = messages + [{"role": "assistant", "content": prefill}]
        ids = tok.apply_chat_template(
            messages, add_generation_prompt=False,
            continue_final_message=True)
    else:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return [int(t) for t in ids]


def split_channels(text: str) -> tuple[str, str]:
    """Return (reasoning, final) channels: harmony (gpt-oss) or <think>
    style (Qwen)."""
    analysis = ""
    if "<|channel|>analysis<|message|>" in text:
        analysis = text.split("<|channel|>analysis<|message|>", 1)[1]
        analysis = analysis.split("<|channel|>")[0]
    if "<think>" in text and "</think>" in text:
        analysis = text.split("<think>", 1)[1].split("</think>", 1)[0]
    final = text.split(FINAL_MARKER)[-1] if FINAL_MARKER in text else text
    for tok in ("<|end|>", "<|return|>", "<|call|>", "<|start|>"):
        final = final.split(tok)[0]
        analysis = analysis.split(tok)[0]
    # truncated decodes never reach the final channel: drop harmony markup
    final = re.sub(r"<\|[a-z_]+\|>", "", final)
    if "</think>" in final:
        final = final.split("</think>")[-1]
    return analysis.strip(), final.strip()


def extract_answer(final: str) -> str | None:
    """Pull the answer out of the final channel: ANSWER: line, else last
    number."""
    m = ANSWER_RE.search(final)
    raw = m.group(1) if m else None
    if raw is None:
        nums = NUM_RE.findall(final)
        raw = nums[-1] if nums else None
    if raw is None:
        return None
    raw = raw.strip().rstrip(".")
    nums = NUM_RE.findall(raw)
    return nums[-1].replace(",", "") if nums else None


def answers_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-9
    except ValueError:
        return pred.strip().casefold() == gold.strip().casefold()


def _truncate(text: str, n: int = 256) -> str:
    return " ".join(text.split()[:n])


# --------------------------------------------------------------------------
# Stage 1: multi-rollout branch transcripts over a real fork tree
# --------------------------------------------------------------------------

def _sample_all(samplers, jobs):
    """Submit every (model, messages, temperature, max_tokens) job, then
    collect in order. ``samplers`` maps model -> (client, tokenizer).
    Tinker sample() returns a future per call, so submission is async."""
    from tinker.types import ModelInput, SamplingParams
    futs = []
    for model, messages, temp, max_tokens in jobs:
        client, tok = samplers[model]
        ids = _chat_ids(tok, messages)
        futs.append((tok, client.sample(
            prompt=ModelInput.from_ints(ids), num_samples=1,
            sampling_params=SamplingParams(max_tokens=max_tokens,
                                           temperature=temp))))
    out = []
    for tok, fut in futs:
        resp = fut.result()
        out.append(tok.decode(resp.sequences[0].tokens))
    return out


def build_transcript(task, turns):
    """Pack a branch's rollout into ~256 words of transcript text. Sampling
    metadata (persona, temperature) stays out of the text so mining sees
    behavior, not tags."""
    parts = [f"[TASK] {task[1]}"]
    for i, turn in enumerate(turns):
        parts.append(f"[TURN {i + 1}]")
        if turn["reasoning"]:
            parts.append(_truncate(turn["reasoning"], 160))
        parts.append(f"ANSWER: {turn['answer'] or '<none>'}")
        parts.append(f"[VERIFIER] {'correct' if turn['correct'] else 'incorrect'}")
    return "\n".join(parts)


def generate_transcripts(out_dir: str) -> list[dict]:
    """For every task, prepare a root branch, sample N rollouts on Tinker,
    then fan the transcripts out over real KV branches and grade them."""
    import tinker
    from agentfork.harness import TreeAgent
    from agentfork.kv.tree_cache import TreeKVCache
    from agentfork.orchestrator import ForkOrchestrator, NullSandbox

    sc = tinker.ServiceClient()
    samplers = {}
    for model in (BRANCH_MODEL, WEAK_MODEL):
        tok_m = _tok(model)
        cf = sc.create_sampling_client(base_model=model)
        samplers[model] = (cf.result() if hasattr(cf, "result") else cf,
                           tok_m)

    orch = ForkOrchestrator(kv=TreeKVCache(capacity_tokens=8_000_000),
                            sandbox=NullSandbox())
    agent = TreeAgent(orch)
    records = []
    try:
        for task_idx, task in enumerate(TASKS):
            t0 = time.time()
            task_id, question, gold, distractor = task
            context = (f"You are a solver working on a task.\nTASK: {question}\n")
            root_id = f"task-{task_idx}"
            agent.prepare_root(root_id, context)

            # 4 conditions per task: careful/quick x plain/anchored. The
            # anchor is a plausible wrong answer attributed to "another
            # solver" — it lives only in the sampling prompt, never in the
            # packed transcript, so mining sees behavior (deferring to the
            # suggestion vs. re-deriving), not a tag.
            personas = [sorted(PERSONAS)[k % 2] for k in range(N_BRANCHES)]
            anchored = [(k // 2) % 2 == 1 for k in range(N_BRANCHES)]
            temps = [TEMPERATURES[k % len(TEMPERATURES)]
                     for k in range(N_BRANCHES)]
            # every 4th branch runs the weaker model: coherent arithmetic
            # slips are the behavioral failures gpt-oss-20b rarely emits
            models = [WEAK_MODEL if k % 3 == 2 else BRANCH_MODEL
                      for k in range(N_BRANCHES)]
            # quick branches get a tight budget: reasoning truncates and
            # answers slip — a real behavioral failure mode, not noise
            budgets = [300 if p == "quick" else 1400 for p in personas]

            sys_msgs = []
            for p, a in zip(personas, anchored):
                q = question
                if a:
                    q += (f"\n(A verified solver already computed the "
                          f"answer: {distractor}. Build on that result.)")
                sys_msgs.append([{"role": "system", "content": PERSONAS[p]},
                                 {"role": "user", "content": q}])
            raws = _sample_all(samplers,
                               list(zip(models, sys_msgs, temps, budgets)))

            turns_per_branch = []
            retry_jobs, retry_idx = [], []
            for k, raw in enumerate(raws):
                analysis, final = split_channels(raw)
                answer = extract_answer(final)
                correct = answers_match(answer, gold)
                turns = [{"reasoning": analysis, "final": final,
                          "answer": answer, "correct": correct}]
                turns_per_branch.append(turns)
                if not correct:
                    retry_idx.append(k)
                    assistant_msg = ({"role": "assistant", "content": final,
                                      "thinking": analysis}
                                     if models[k] == BRANCH_MODEL else
                                     {"role": "assistant",
                                      "content": final})
                    retry_jobs.append((
                        models[k],
                        sys_msgs[k] + [
                            assistant_msg,
                            {"role": "user", "content":
                             ("Incorrect. Re-check your reasoning and give "
                              "the final answer on its own line as "
                              "'ANSWER: <value>'.")}],
                        temps[k], 1400))

            if retry_jobs:
                raws2 = _sample_all(samplers, retry_jobs)
                for k, raw in zip(retry_idx, raws2):
                    analysis, final = split_channels(raw)
                    answer = extract_answer(final)
                    turns_per_branch[k].append(
                        {"reasoning": analysis, "final": final,
                         "answer": answer,
                         "correct": answers_match(answer, gold)})

            continuations, metas = [], []
            for k, turns in enumerate(turns_per_branch):
                transcript = build_transcript(task, turns)
                label = int(turns[-1]["correct"])
                continuations.append(
                    context + f"\n[ROLLOUT {k}]\n" + transcript)
                metas.append({"task_id": task_id, "branch_idx": k,
                              "persona": personas[k], "model": models[k],
                              "anchored": anchored[k],
                              "temperature": temps[k], "turns": turns,
                              "transcript": transcript, "label": label})

            def eval_(result):
                return 1.0 if result.output and result.output["label"] else 0.0

            # Real fork fanout: each transcript commits as its branch's
            # continuation; work returns the label; losers get killed.
            results = agent.fan_out(
                root_id, continuations,
                lambda bid, pfx: {"label": int(
                    bytes(pfx).decode().rsplit(
                        "[VERIFIER] ", 1)[-1].strip() == "correct")},
                eval_)
            for k, (result, meta) in enumerate(zip(results, metas)):
                meta["branch_id"] = result.branch_id
                records.append(meta)

            winner = agent.select_winner(results)
            agent.kill_losers(winner.branch_id)
            orch.kill(winner.branch_id)
            orch.kill(root_id)

            solved = sum(m["label"] for m in metas)
            print(f"[{task_id}] {solved}/{N_BRANCHES} solved "
                  f"({time.time() - t0:.0f}s; kv resident "
                  f"{orch.kv.resident_tokens()} tokens)", flush=True)
    finally:
        orch.close()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "transcripts.jsonl")
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return records


# --------------------------------------------------------------------------
# Stage 2: HypotheSAEs over branch transcripts
# --------------------------------------------------------------------------

def install_tinker_shim():
    """Point hypothesaes.llm_api at Tinker: 'interp' -> Kimi-K2.6 (with a
    '- "' assistant prefill so the model completes a bullet instead of
    truncating mid-reasoning), everything else -> gpt-oss-20b."""
    import tinker
    import hypothesaes.llm_api as llm_api

    def resolve_model(name):
        if name in ("interp", INTERPRETER_TINKER):
            return INTERPRETER_TINKER, 96
        return ANNOTATOR_TINKER, 192

    class _TinkerResponses:
        def __init__(self):
            self.sc = tinker.ServiceClient()
            self._clients = {}
            self._toks = {}

        def _tok_for(self, tm):
            if tm not in self._toks:
                self._toks[tm] = _tok(tm)
            return self._toks[tm]

        def _client(self, tm):
            if tm not in self._clients:
                c = self.sc.create_sampling_client(base_model=tm)
                self._clients[tm] = (c.result() if hasattr(c, "result")
                                     else c)
            return self._clients[tm]

        @staticmethod
        def _to_messages(inp):
            if isinstance(inp, str):
                return [{"role": "user", "content": inp}]
            msgs = []
            for m in inp:
                content = m.get("content", "")
                if isinstance(content, list):
                    content = "".join(p.get("text", "") for p in content)
                msgs.append({"role": m.get("role", "user"),
                             "content": content})
            return msgs

        def create(self, model=None, input=None, **kwargs):
            from tinker.types import ModelInput, SamplingParams
            tm, default_max = resolve_model(model)
            max_tokens = kwargs.get("max_output_tokens") or default_max
            temperature = kwargs.get("temperature", 0.7)
            messages = self._to_messages(input)
            tok = self._tok_for(tm)
            if tm == INTERPRETER_TINKER:
                # non-harmony model: prefill a "- " bullet so the completion
                # is the feature phrase itself (~20 tokens)
                ids = _chat_ids(tok, messages, prefill='- "')
            else:
                ids = _chat_ids(tok, messages)
            fut = self._client(tm).sample(
                prompt=ModelInput.from_ints(ids), num_samples=1,
                sampling_params=SamplingParams(max_tokens=max_tokens,
                                               temperature=temperature))
            resp = fut.result()
            text = tok.decode(resp.sequences[0].tokens)
            if tm == INTERPRETER_TINKER:
                first = text.split("\n", 1)[0].strip()
                if first.startswith('"'):
                    first = first[1:]
                cut = first.find('"')
                if cut > 0:
                    first = first[:cut]
                out = '- "' + first.strip() + '"'
            else:
                _, out = split_channels(text)
                if not out.lower().startswith(("yes", "no")):
                    hits = re.findall(r"\b(yes|no)\b", text.lower())
                    out = hits[-1] if hits else "no"
            return SimpleNamespace(output_text=out)

    class _ShimClient:
        def __init__(self):
            self.responses = _TinkerResponses()

    _shim = _ShimClient()
    llm_api.get_client = lambda: _shim

    # score paths assume every task was annotated; fill misses with 0
    import hypothesaes.annotate as ann
    import hypothesaes.interpret_neurons as inr
    for _mod in (inr, ann):
        _orig = _mod.annotate

        def _annotate_fill(tasks, _orig=_orig, **kw):
            results = _orig(tasks=tasks, **kw)
            for (text, concept) in tasks:
                results.setdefault(concept, {}).setdefault(text, 0)
            return results

        _mod.annotate = _annotate_fill
    print("Tinker shim installed (interp=Kimi-K2.6, annot=gpt-oss-20b).",
          flush=True)


def balance_tasks(records: list[dict]) -> list[dict]:
    """Downsample each task to 50% solved so task identity carries no label
    signal (otherwise hypotheses collapse to task-specific tokens)."""
    rng = np.random.RandomState(0)
    kept = []
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(r)
    for task_id, rows in by_task.items():
        solved = [r for r in rows if r["label"] == 1]
        failed = [r for r in rows if r["label"] == 0]
        k = min(len(solved), len(failed))
        if k < 4:
            print(f"  dropping {task_id}: {len(solved)} solved / "
                  f"{len(failed)} failed (need >=4 each)", flush=True)
            continue
        kept += rng.choice(solved, k, replace=False).tolist()
        kept += rng.choice(failed, k, replace=False).tolist()
    rng.shuffle(kept)
    return kept


def mine(records: list[dict], out_dir: str) -> None:
    from hypothesaes.embedding import get_local_embeddings
    from hypothesaes.quickstart import (evaluate_hypotheses,
                                        generate_hypotheses, train_sae)

    install_tinker_shim()
    kept = balance_tasks(records)
    texts = [_truncate(r["transcript"], 256) for r in kept]
    labels = np.array([r["label"] for r in kept], dtype=int)
    print(f"mining corpus: {len(texts)} transcripts, "
          f"{labels.mean():.2f} solved", flush=True)

    rng = np.random.RandomState(1)
    n_hold = max(60, int(0.25 * len(texts)))
    hold_idx = rng.choice(len(texts), n_hold, replace=False)
    mask = np.zeros(len(texts), dtype=bool)
    mask[hold_idx] = True
    train_texts = [t for t, m in zip(texts, mask) if not m]
    train_labels = labels[~mask]
    hold_texts = [t for t, m in zip(texts, mask) if m]
    hold_labels = labels[mask]

    print("== embeddings (MiniLM, CPU) ==", flush=True)
    emb = get_local_embeddings(
        train_texts, model="sentence-transformers/all-MiniLM-L6-v2",
        cache_name="trace-miner-train", batch_size=64)
    E = np.stack([emb[t] for t in train_texts])

    print("== train SAE (M=1024 K=32) ==", flush=True)
    sae = train_sae(E, M=1024, K=32,
                    checkpoint_dir=os.path.join(out_dir, "checkpoints"),
                    show_progress=True)

    print("== select neurons (lasso) + interpret + fidelity ==", flush=True)
    df = generate_hypotheses(
        texts=train_texts, labels=train_labels, embeddings=E, sae=sae,
        cache_name="trace-miner", classification=True,
        selection_method="lasso", n_selected_neurons=20,
        interpreter_model="interp", annotator_model="annot",
        n_candidate_interpretations=3, n_scoring_examples=60,
        interpret_temperature=0.7,
        task_specific_instructions=(
            "Each sample is a branch-level transcript of an LLM agent "
            "solving a short math/logic problem, with verifier feedback "
            "between turns. Prefer behavioral features (reasoning style, "
            "answer format, re-checking, hedging) over task-specific "
            "tokens."))
    if isinstance(df, tuple):
        df = df[0]
    df.to_csv(os.path.join(out_dir, "hypotheses.csv"), index=False)
    print(df[["neuron_idx", "interpretation"]].to_string(), flush=True)

    print("== annotate holdout + regression ==", flush=True)
    metrics, eval_df = evaluate_hypotheses(
        df, hold_texts, hold_labels, cache_name="trace-miner",
        annotator_model="annot", classification=True,
        corrected_pval_threshold=0.05)
    eval_df.to_csv(os.path.join(out_dir, "hypotheses_eval.csv"),
                   index=False)
    print(metrics, flush=True)
    print(eval_df.to_string(), flush=True)

    print("== pruning signal ==", flush=True)
    pruning = prune_report(records, kept, df, eval_df, sae, out_dir)
    print(json.dumps(pruning, indent=1), flush=True)


def prune_report(records, kept, df, eval_df, sae, out_dir):
    """Kill rule: a branch's *early* turn (first ~100 words of transcript)
    activates >=1 neuron whose hypothesis predicts failure on holdout
    (significant negative regression coefficient)."""
    from hypothesaes.embedding import get_local_embeddings

    sig = eval_df[eval_df["regression_pval"] < 0.05 / len(eval_df)]
    failure_hyps = set(sig[sig["regression_coef"] < 0]["hypothesis"])
    success_hyps = set(sig[sig["regression_coef"] > 0]["hypothesis"])
    neuron_for_hyp = dict(zip(df["interpretation"], df["neuron_idx"]))
    failure_neurons = [neuron_for_hyp[h] for h in failure_hyps
                       if h in neuron_for_hyp]
    success_neurons = [neuron_for_hyp[h] for h in success_hyps
                       if h in neuron_for_hyp]

    early = [_truncate(r["transcript"], 100) for r in kept]
    emb = get_local_embeddings(
        early, model="sentence-transformers/all-MiniLM-L6-v2",
        cache_name="trace-miner-early", batch_size=64)
    import torch
    E_early = torch.tensor(np.stack([emb[t] for t in early]),
                           dtype=torch.float)
    acts = np.asarray(sae.get_activations(E_early))
    labels = np.array([r["label"] for r in kept])

    recs = []
    kill_mask = np.zeros(len(kept), dtype=bool)
    for i, r in enumerate(kept):
        fired_fail = [j for j in failure_neurons if acts[i, j] > 0]
        fired_succ = [j for j in success_neurons if acts[i, j] > 0]
        kill = bool(fired_fail) and not fired_succ
        kill_mask[i] = kill
        recs.append({"task_id": r["task_id"], "branch_id": r["branch_id"],
                     "label": r["label"], "persona": r["persona"],
                     "temperature": r["temperature"],
                     "kill": kill, "failure_neurons_fired": fired_fail,
                     "success_neurons_fired": fired_succ})

    killed = kill_mask
    report = {
        "n_transcripts": len(kept),
        "significant_hypotheses": int(len(sig)),
        "failure_neurons": failure_neurons,
        "success_neurons": success_neurons,
        "kill_count": int(killed.sum()),
        "precision_failures_killed":
            float((labels[killed] == 0).mean()) if killed.sum() else None,
        "recall_of_failures":
            float(killed[labels == 0].mean()) if (labels == 0).sum() else None,
        "correct_wrongly_killed":
            int(((labels == 1) & killed).sum()),
        "baseline_failure_rate": float((labels == 0).mean()),
    }
    with open(os.path.join(out_dir, "pruning_report.json"), "w") as f:
        json.dump({"report": report, "per_branch": recs}, f, indent=1)
    return report


def main() -> None:
    global N_BRANCHES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen", action="store_true",
                        help="generate transcripts (Tinker + fork tree)")
    parser.add_argument("--mine", action="store_true",
                        help="run HypotheSAEs on existing transcripts")
    parser.add_argument("--branches", type=int, default=N_BRANCHES)
    args = parser.parse_args()
    both = not (args.gen or args.mine)
    N_BRANCHES = args.branches

    if "TINKER_API_KEY" not in os.environ:
        print("BLOCKED: TINKER_API_KEY not set", flush=True)
        raise SystemExit(2)
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.gen or both:
        generate_transcripts(OUT_DIR)
    if args.mine or both:
        path = os.path.join(OUT_DIR, "transcripts.jsonl")
        records = [json.loads(line) for line in open(path)]
        mine(records, OUT_DIR)


if __name__ == "__main__":
    main()
