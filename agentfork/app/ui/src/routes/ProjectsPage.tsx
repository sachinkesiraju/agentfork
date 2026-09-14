import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, Harnesses, Project } from "../api";
import { Badge, Button, Empty, Panel } from "../components/ui";

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [harnesses, setHarnesses] = useState<Harnesses>({});
  const [home, setHome] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    api.projects().then(setProjects).catch((e) => setError(String(e.message)));
    api.harnesses().then(setHarnesses).catch(() => undefined);
    api.health().then((h) => setHome(h.home)).catch(() => undefined);
  }, []);

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-8">
      <header className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-neutral-100">agentfork</h1>
          <p className="text-sm text-neutral-500">
            Local autoresearch: fan out candidates, score them with your own eval,
            keep what beats its parent.
          </p>
        </div>
        <Link to="/new">
          <Button variant="primary">New project</Button>
        </Link>
      </header>

      {error && (
        <p className="rounded border border-red-900 bg-red-950/50 px-3 py-2 text-sm text-red-200">
          {error}
        </p>
      )}

      <Panel title="Projects">
        {projects === null ? (
          <Empty>loading…</Empty>
        ) : projects.length === 0 ? (
          <Empty>
            No projects yet. A project is a git repo plus one fixed eval command.
          </Empty>
        ) : (
          <ul className="divide-y divide-neutral-800">
            {projects.map((p) => (
              <li key={p.id} className="flex items-center justify-between py-2">
                <div>
                  <Link
                    to={`/p/${p.id}`}
                    className="text-sm text-neutral-100 hover:underline"
                  >
                    {p.name}
                  </Link>
                  <p className="mono text-xs text-neutral-500">
                    {p.repo_path} · {p.baseline_branch} · {p.params.eval_cmd}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <Badge status={p.harness} />
                  <span className="mono text-xs text-neutral-600">{p.id}</span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </Panel>

      <Panel title="Harnesses">
        <ul className="space-y-1 text-sm">
          {Object.entries(harnesses).map(([name, h]) => (
            <li key={name} className="flex items-center gap-2">
              <span
                className={`h-2 w-2 rounded-full ${h.authed ? "bg-emerald-500" : h.installed ? "bg-amber-500" : "bg-neutral-700"}`}
              />
              <span className="mono text-neutral-300">{name}</span>
              <span className="text-xs text-neutral-500">
                {h.authed
                  ? "ready"
                  : h.installed
                    ? "installed, not authenticated"
                    : "not installed"}
              </span>
            </li>
          ))}
        </ul>
      </Panel>

      {home && (
        <p className="mono text-xs text-neutral-600">
          state: {home} — loopback only, nothing leaves this machine
        </p>
      )}
    </div>
  );
}
