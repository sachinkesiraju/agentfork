import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, Harnesses } from "../api";
import { Button, Field, Input, Panel, Select } from "../components/ui";

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
  });

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
          holdout_cmd: form.holdout_cmd,
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
    <div className="mx-auto max-w-3xl space-y-6 p-8">
      <header className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-neutral-100">New project</h1>
        <Link to="/" className="text-sm text-neutral-500 hover:underline">
          cancel
        </Link>
      </header>

      <form onSubmit={submit} className="space-y-4">
        <Panel title="Repository">
          <div className="grid grid-cols-2 gap-3">
            <Field label="repo path" hint="absolute path to a git repo on this machine">
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
          </div>
        </Panel>

        <Panel title="Eval contract">
          <div className="grid grid-cols-2 gap-3">
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

        <Panel title="Search">
          <div className="grid grid-cols-5 gap-3">
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
          </div>
        </Panel>

        {error && (
          <p className="rounded border border-red-900 bg-red-950/50 px-3 py-2 text-sm text-red-200">
            {error}
          </p>
        )}
        <Button variant="primary" type="submit" disabled={busy}>
          {busy ? "creating…" : "Create project"}
        </Button>
      </form>
    </div>
  );
}
