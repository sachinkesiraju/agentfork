import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, Harnesses } from "../api";
import { Button, ErrorNote, Field, Input, Panel, Select } from "../components/ui";
import { Page, TopBar } from "../components/Shell";

/** Onboarding: a repo plus the map-reduce parameter table. Nothing else is
 *  asked for, because eval_cmd + metric_grep are the whole contract. */
export default function NewProjectPage() {
  const nav = useNavigate();
  const [harnesses, setHarnesses] = useState<Harnesses>({});
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [form, setForm] = useState({
    name: "",
    repo_path: "",
    baseline_branch: "main",
    harness: "claude-code",
    eval_cmd: "",
    metric_grep: "",
    minimize: "true",
    k: 4,
    b: 2,
    max_gens: 3,
    timeout_s: 600,
    baseline_runs: 2,
    holdout_cmd: "",
    eval_slots: 2,
    cost_guards: "",
    model: "",
    api_base: "",
  });
  const [modelList, setModelList] = useState<string[]>([]);
  const [modelNote, setModelNote] = useState("");

  useEffect(() => {
    api.harnesses().then(setHarnesses).catch(() => undefined);
  }, []);

  const set = (k: string, v: string | number) => setForm({ ...form, [k]: v });

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const p = await api.createProject({
        name: form.name || form.repo_path.split("/").filter(Boolean).pop() || "project",
        repo_path: form.repo_path,
        baseline_branch: form.baseline_branch,
        harness: form.harness,
        params: {
          eval_cmd: form.eval_cmd,
          metric_grep: form.metric_grep,
          minimize: form.minimize === "true",
          k: Number(form.k),
          b: Number(form.b),
          max_gens: Number(form.max_gens),
          timeout_s: Number(form.timeout_s),
          baseline_runs: Number(form.baseline_runs),
          eval_slots: Number(form.eval_slots),
          holdout_cmd: form.holdout_cmd,
          model: form.model,
          api_base: form.api_base,
          cost_guards: Object.fromEntries(
            form.cost_guards
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean)
              .map((s) => {
                const [k, v] = s.split("=");
                return [k.trim(), Number(v)];
              })
              .filter(([, v]) => Number.isFinite(v)),
          ),
        },
      });
      nav(`/p/${p.id}`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <TopBar>
        <Link to="/" className="text-sm text-neutral-500 transition-colors hover:text-neutral-300">
          Cancel
        </Link>
      </TopBar>
      <Page>
      <header className="max-w-2xl space-y-2 pt-2">
        <h1 className="text-2xl font-semibold tracking-tight text-neutral-50">New project</h1>
        <p className="text-sm leading-relaxed text-neutral-400">
          A project is a repo plus one fixed eval contract. Both are frozen once it exists —
          that is what makes scores across generations comparable.
        </p>
      </header>

      <form onSubmit={submit} className="max-w-3xl space-y-4">
        <Panel title="Repository" subtitle="what gets branched, and who edits it">
          <div className="grid gap-3 sm:grid-cols-2">
            <Field
              label="repo path"
              hint="absolute path to a git repo on this machine — agentfork appends results.tsv/.agentfork-candidate to its .gitignore"
            >
              <Input
                required
                value={form.repo_path}
                placeholder="/home/you/code/my-model"
                onChange={(e) => set("repo_path", e.target.value)}
              />
            </Field>
            <Field label="baseline branch">
              <Input
                required
                value={form.baseline_branch}
                onChange={(e) => set("baseline_branch", e.target.value)}
              />
            </Field>
            <Field label="name">
              <Input value={form.name} onChange={(e) => set("name", e.target.value)} />
            </Field>
            <Field label="harness" hint="the agent that edits candidate worktrees">
              <Select value={form.harness} onChange={(e) => set("harness", e.target.value)}>
                {Object.entries(harnesses).map(([n, h]) => (
                  <option key={n} value={n}>
                    {n}
                    {h.authed ? "" : h.installed ? " (not authed)" : " (not installed)"}
                  </option>
                ))}
              </Select>
            </Field>
            <Field
              label="model"
              hint={
                form.harness === "api"
                  ? "model id at the endpoint — leave blank to use the server's default"
                  : "optional — passed as the harness's --model flag"
              }
            >
              <div className="flex gap-2">
                <Input
                  value={form.model}
                  list="af-model-options"
                  placeholder={
                    form.harness === "api" ? "e.g. qwen3:14b" : "e.g. claude-sonnet-4"
                  }
                  onChange={(e) => set("model", e.target.value)}
                />
                {form.harness === "api" && (
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      setModelNote("");
                      api
                        .models(form.api_base || "http://127.0.0.1:11434/v1")
                        .then((r) => setModelList(r.models))
                        .catch((err) => setModelNote((err as Error).message));
                    }}
                  >
                    find
                  </Button>
                )}
              </div>
              <datalist id="af-model-options">
                {modelList.map((m) => (
                  <option key={m} value={m} />
                ))}
              </datalist>
              {modelNote && (
                <p className="mt-1 text-[11px] text-red-300">{modelNote}</p>
              )}
            </Field>
            {form.harness === "api" && (
              <Field
                label="endpoint"
                hint="OpenAI-compatible base URL — loopback only (Ollama :11434/v1, LM Studio :1234/v1, vLLM…)"
              >
                <Input
                  value={form.api_base}
                  placeholder="http://127.0.0.1:11434/v1"
                  onChange={(e) => set("api_base", e.target.value)}
                />
              </Field>
            )}
          </div>
        </Panel>

        <Panel
          title="Eval contract"
          subtitle="the orchestrator greps this score from the run log — never from the agent"
        >
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="eval_cmd" hint="fixed for the project's lifetime">
              <Input
                required
                value={form.eval_cmd}
                placeholder="python train.py --steps 2000"
                onChange={(e) => set("eval_cmd", e.target.value)}
              />
            </Field>
            <Field label="metric_grep" hint="regex over the eval log; group 1 is the number">
              <Input
                required
                value={form.metric_grep}
                placeholder="val_loss=([0-9.]+)"
                onChange={(e) => set("metric_grep", e.target.value)}
              />
            </Field>
            <Field label="direction">
              <Select value={form.minimize} onChange={(e) => set("minimize", e.target.value)}>
                <option value="true">minimize</option>
                <option value="false">maximize</option>
              </Select>
            </Field>
            <Field label="holdout_cmd" hint="optional; revalidates the champion">
              <Input
                value={form.holdout_cmd}
                onChange={(e) => set("holdout_cmd", e.target.value)}
              />
            </Field>
          </div>
        </Panel>

        <Panel title="Search" subtitle="how wide each generation fans out, and how much survives">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-6">
            <Field label="K" hint="ideas/gen">
              <Input type="number" min={1} value={form.k} onChange={(e) => set("k", e.target.value)} />
            </Field>
            <Field label="B" hint="beam">
              <Input type="number" min={1} value={form.b} onChange={(e) => set("b", e.target.value)} />
            </Field>
            <Field label="max gens">
              <Input
                type="number"
                min={1}
                value={form.max_gens}
                onChange={(e) => set("max_gens", e.target.value)}
              />
            </Field>
            <Field label="timeout s">
              <Input
                type="number"
                min={1}
                value={form.timeout_s}
                onChange={(e) => set("timeout_s", e.target.value)}
              />
            </Field>
            <Field label="baseline runs" hint="sets noise margin">
              <Input
                type="number"
                min={2}
                value={form.baseline_runs}
                onChange={(e) => set("baseline_runs", e.target.value)}
              />
            </Field>
            <Field label="eval slots" hint="max concurrent evals">
              <Input
                type="number"
                min={0}
                value={form.eval_slots}
                onChange={(e) => set("eval_slots", e.target.value)}
              />
            </Field>
          </div>
          <Field
            label="cost guards"
            hint="optional, comma-separated name=tolerance — a scored run whose cost grows by more than tol relative to its parent fails the reduce (seconds is built in)"
          >
            <Input
              value={form.cost_guards}
              placeholder="seconds=0.5, gpu_hours=0.2"
              onChange={(e) => set("cost_guards", e.target.value)}
            />
          </Field>
        </Panel>

        {error && <ErrorNote>{error}</ErrorNote>}
        <div className="flex items-center gap-3 pb-4">
          <Button variant="primary" type="submit" disabled={busy}>
            {busy ? "Creating…" : "Create project"}
          </Button>
          <span className="text-xs text-neutral-600">
            Nothing runs until you start the baseline.
          </span>
        </div>
      </form>
      </Page>
    </>
  );
}
