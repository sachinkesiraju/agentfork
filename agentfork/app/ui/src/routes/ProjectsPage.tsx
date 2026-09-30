import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, Harnesses, Project } from "../api";
import { Button, Empty, ErrorNote, Panel } from "../components/ui";
import { LocalFooter, Page, TopBar } from "../components/Shell";

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
    <>
      <TopBar>
        <Link to="/new">
          <Button variant="primary">New project</Button>
        </Link>
      </TopBar>
      <Page>
        <header className="space-y-2 pb-1 pt-2">
          <h1 className="text-2xl font-semibold tracking-tight text-neutral-50">
            Local autoresearch
          </h1>
          <p className="max-w-2xl text-sm leading-relaxed text-neutral-400">
            Fan out candidates from a baseline, score every one with your own eval command,
            and keep only what beats its parent by more than the noise margin.
          </p>
        </header>

        {error && <ErrorNote>{error}</ErrorNote>}

        <Panel
          title="Projects"
          subtitle="a git repo plus one fixed eval command"
          padded={false}
        >
          {projects === null ? (
            <Empty>Loading…</Empty>
          ) : projects.length === 0 ? (
            <Empty title="No projects yet">
              A project pins a repo, a baseline branch and the eval command every candidate
              is judged by. That contract never changes once the project exists.
            </Empty>
          ) : (
            <ul className="divide-y divide-neutral-800/80">
              {projects.map((p) => (
                <li key={p.id} className="group transition-colors hover:bg-neutral-800/30">
                  <Link
                    to={`/p/${p.id}`}
                    className="flex items-center justify-between gap-4 px-4 py-3"
                  >
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium text-neutral-100">{p.name}</p>
                      <p className="mono truncate text-xs text-neutral-500">{p.repo_path}</p>
                    </div>
                    <div className="hidden shrink-0 items-center gap-2 text-[11px] text-neutral-500 sm:flex">
                      <Tag>{p.baseline_branch}</Tag>
                      <Tag>{p.harness}</Tag>
                      <Tag mono>{p.params.eval_cmd}</Tag>
                      <span
                        aria-hidden
                        className="text-neutral-600 transition-transform group-hover:translate-x-0.5"
                      >
                        →
                      </span>
                    </div>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </Panel>

        <Panel title="Harnesses" subtitle="the agent that writes each candidate">
          <ul className="grid gap-2 sm:grid-cols-3">
            {Object.entries(harnesses).map(([name, h]) => (
              <li
                key={name}
                className="flex items-center gap-2.5 rounded-lg border border-neutral-800 bg-neutral-900/60 px-3 py-2"
              >
                <span
                  className={`h-2 w-2 shrink-0 rounded-full ${
                    h.authed ? "bg-emerald-500" : h.installed ? "bg-amber-500" : "bg-neutral-700"
                  }`}
                />
                <div className="min-w-0">
                  <p className="mono truncate text-xs text-neutral-200">{name}</p>
                  <p className="truncate text-[11px] text-neutral-500">
                    {h.authed
                      ? "ready"
                      : h.installed
                        ? "installed, not authenticated"
                        : "not installed"}
                  </p>
                </div>
              </li>
            ))}
          </ul>
        </Panel>

        <LocalFooter home={home} />
      </Page>
    </>
  );
}

function Tag({ children, mono }: { children: React.ReactNode; mono?: boolean }) {
  return (
    <span
      className={`max-w-[14rem] truncate rounded-md border border-neutral-800 bg-neutral-900 px-1.5 py-0.5 ${
        mono ? "mono" : ""
      }`}
    >
      {children}
    </span>
  );
}
