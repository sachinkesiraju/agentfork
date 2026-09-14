import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  api,
  Law,
  LoopState,
  Metrics,
  Node,
  Project,
  Reduction,
  Run,
  subscribe,
} from "../api";
import { Badge, Button, Empty, Panel, fmt } from "../components/ui";
import LogTerminal from "../components/LogTerminal";
import TreeView from "../components/TreeView";

type Tab = "tree" | "runs" | "logs" | "laws" | "ledger";

export default function ProjectPage() {
  const { projectId = "" } = useParams();
  const [project, setProject] = useState<Project | null>(null);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [loop, setLoop] = useState<LoopState | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [laws, setLaws] = useState<Law[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [tab, setTab] = useState<Tab>("tree");
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [reduction, setReduction] = useState<Reduction | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [p, t, r, l, m] = await Promise.all([
        api.project(projectId),
        api.tree(projectId),
        api.runs(projectId),
        api.laws(projectId),
        api.metrics(projectId),
      ]);
      setProject(p);
      setNodes(t.nodes);
      setLoop(t.loop);
      setRuns(r);
      setLaws(l);
      setMetrics(m);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [projectId]);

  useEffect(() => {
    refresh();
    return subscribe((e) => {
      if (e.project_id === projectId) refresh();
    });
  }, [projectId, refresh]);

  const act = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(label);
    setError("");
    try {
      await fn();
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy("");
    }
  };

  const node = useMemo(
    () => nodes.find((n) => n.id === selectedNode) ?? null,
    [nodes, selectedNode],
  );
  const nodeRuns = useMemo(
    () => (node ? runs.filter((r) => r.node_id === node.id) : []),
    [runs, node],
  );
  const hasBaseline = nodes.some((n) => n.gen === 0);
  const looping = loop?.state === "running" || loop?.state === "baseline";

  if (!project) {
    return (
      <div className="p-8 text-sm text-neutral-500">
        {error ? <span className="text-red-300">{error}</span> : "loading…"}
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl space-y-4 p-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <Link to="/" className="text-xs text-neutral-500 hover:underline">
            ← projects
          </Link>
          <h1 className="text-xl font-semibold text-neutral-100">{project.name}</h1>
          <p className="mono text-xs text-neutral-500">
            {project.repo_path} · {project.baseline_branch} · {project.harness} ·{" "}
            {project.params.eval_cmd} · {project.params.minimize ? "minimize" : "maximize"}{" "}
            {project.params.metric_grep}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {!hasBaseline && (
            <Button
              disabled={!!busy}
              onClick={() => act("baseline", () => api.baseline(projectId))}
            >
              {busy === "baseline" ? "starting…" : `Run baseline ×${project.params.baseline_runs}`}
            </Button>
          )}
          {looping ? (
            <Button variant="danger" onClick={() => act("stop", () => api.stopLoop(projectId))}>
              Stop autoresearch
            </Button>
          ) : (
            <Button
              variant="primary"
              disabled={!!busy}
              onClick={() => act("loop", () => api.startLoop(projectId))}
            >
              {busy === "loop" ? "starting…" : "Start autoresearch"}
            </Button>
          )}
        </div>
      </header>

      <Panel>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-xs">
          <span className="text-neutral-500">
            loop <Badge status={loop?.state ?? "idle"} />
          </span>
          <span className="mono text-neutral-400">gen {loop?.gen ?? 0}</span>
          <span className="mono text-neutral-400">
            margin {loop?.margin !== null && loop?.margin !== undefined ? fmt(loop.margin) : "—"}
          </span>
          <span className="mono text-neutral-400">
            K={project.params.k} B={project.params.b} max_gens={project.params.max_gens}
          </span>
          <span className="text-neutral-500">{loop?.message}</span>
        </div>
      </Panel>

      {error && (
        <p className="rounded border border-red-900 bg-red-950/50 px-3 py-2 text-sm text-red-200">
          {error}
        </p>
      )}

      <nav className="flex gap-1 border-b border-neutral-800">
        {(["tree", "runs", "logs", "laws", "ledger"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-3 py-2 text-sm capitalize ${
              tab === t
                ? "border-b-2 border-neutral-200 text-neutral-100"
                : "text-neutral-500 hover:text-neutral-300"
            }`}
          >
            {t}
          </button>
        ))}
      </nav>

      {tab === "tree" && (
        <div className="grid grid-cols-3 gap-4">
          <div className="col-span-2 space-y-3">
            {nodes.length === 0 ? (
              <Empty>
                No nodes yet — run the baseline to create the root and fix the noise margin.
              </Empty>
            ) : (
              <TreeView
                nodes={nodes}
                frontier={loop?.frontier ?? []}
                selected={selectedNode}
                onSelect={setSelectedNode}
              />
            )}
            <div className="flex flex-wrap items-center gap-2">
              <Button
                disabled={!node || !!busy}
                onClick={() =>
                  node && act("descend", () => api.descend(projectId, node.id, project.params.k))
                }
              >
                {busy === "descend" ? "forking…" : `Fan out ×${project.params.k}`}
              </Button>
              <Button
                disabled={!loop || !loop.gen || !!busy}
                onClick={() => loop && act("reduce", async () => setReduction(await api.reduce(projectId, loop.gen)))}
              >
                {busy === "reduce" ? "reducing…" : `Reduce gen ${loop?.gen ?? 0}`}
              </Button>
              <Button
                disabled={!node || !!busy}
                onClick={() => node && act("eval", () => api.evalNode(projectId, node.id))}
              >
                Re-run eval
              </Button>
              {project.params.holdout_cmd && (
                <Button
                  disabled={!node || !!busy}
                  onClick={() => node && act("holdout", () => api.holdout(projectId, node.id))}
                >
                  Holdout
                </Button>
              )}
            </div>
            {reduction && (
              <Panel title={`reduce gen ${reduction.gen}`}>
                <pre className="mono whitespace-pre-wrap text-xs text-neutral-300">
                  {`${reduction.candidates} candidates, ${reduction.beat_parent} beat their parent (margin ${reduction.margin})\n`}
                  {reduction.kept
                    .map(
                      (s) =>
                        `KEEP  ${s.commit}  ${fmt(s.score)}  (parent ${s.parent}, delta ${s.delta > 0 ? "+" : ""}${fmt(s.delta)})`,
                    )
                    .join("\n")}
                  {reduction.cost_failures
                    .map((f) => `\nCOST_FAIL  ${f.commit}  ${f.reason}`)
                    .join("")}
                  {reduction.stall ? "\nSTALL: frontier unchanged." : ""}
                  {`\nFRONTIER: ${reduction.frontier.join(" ")}`}
                  {reduction.fuse.map((f) => `\nFUSE_CANDIDATE  ${f.a}  ${f.b}`).join("")}
                </pre>
              </Panel>
            )}
          </div>
          <Panel title="Node">
            {!node ? (
              <Empty>Click a node.</Empty>
            ) : (
              <div className="space-y-3 text-sm">
                <div>
                  <p className="text-neutral-100">{node.title}</p>
                  <p className="mono text-xs text-neutral-500">{node.id}</p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <Badge status={node.status} />
                  {node.frozen && <Badge status="frozen" />}
                  {node.regions.map((r) => (
                    <span key={r} className="mono text-[11px] text-neutral-500">
                      {r}
                    </span>
                  ))}
                </div>
                <dl className="mono space-y-1 text-xs text-neutral-400">
                  <div>gen {node.gen}</div>
                  <div>score {fmt(node.score)}</div>
                  <div>
                    delta{" "}
                    {node.delta === null ? "—" : `${node.delta > 0 ? "+" : ""}${fmt(node.delta)}`}
                  </div>
                  <div>branch {node.branch_name}</div>
                  <div className="break-all">worktree {node.worktree_path ?? "(reaped)"}</div>
                  <div>commit {node.commit_sha?.slice(0, 12) ?? "—"}</div>
                </dl>
                {node.description && (
                  <p className="whitespace-pre-wrap text-xs text-neutral-400">
                    {node.description}
                  </p>
                )}
                <div className="space-y-1">
                  <p className="text-xs uppercase tracking-wide text-neutral-500">runs</p>
                  {nodeRuns.length === 0 ? (
                    <p className="text-xs text-neutral-600">none</p>
                  ) : (
                    nodeRuns.map((r) => (
                      <button
                        key={r.id}
                        onClick={() => {
                          setSelectedRun(r.id);
                          setTab("logs");
                        }}
                        className="mono block w-full truncate text-left text-xs text-neutral-400 hover:text-neutral-100"
                      >
                        {r.kind} · {r.status} · {fmt(r.score)}
                      </button>
                    ))
                  )}
                </div>
              </div>
            )}
          </Panel>
        </div>
      )}

      {tab === "runs" && (
        <Panel title="Runs">
          {runs.length === 0 ? (
            <Empty>No runs yet.</Empty>
          ) : (
            <table className="w-full text-left text-xs">
              <thead className="text-neutral-500">
                <tr>
                  <th className="py-1">run</th>
                  <th>kind</th>
                  <th>status</th>
                  <th>node</th>
                  <th>score</th>
                  <th>exit</th>
                  <th>command</th>
                  <th />
                </tr>
              </thead>
              <tbody className="mono">
                {runs.map((r) => (
                  <tr key={r.id} className="border-t border-neutral-800">
                    <td className="py-1">
                      <button
                        className="text-neutral-300 hover:underline"
                        onClick={() => {
                          setSelectedRun(r.id);
                          setTab("logs");
                        }}
                      >
                        {r.id}
                      </button>
                    </td>
                    <td>{r.kind}</td>
                    <td>
                      <Badge status={r.status} />
                    </td>
                    <td className="text-neutral-500">{r.node_id}</td>
                    <td>{fmt(r.score)}</td>
                    <td>{r.exit_code ?? "—"}</td>
                    <td className="max-w-[240px] truncate text-neutral-500">{r.command}</td>
                    <td>
                      {(r.status === "running" || r.status === "queued") && (
                        <Button
                          variant="ghost"
                          onClick={() => act("kill", () => api.killRun(r.id))}
                        >
                          kill
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>
      )}

      {tab === "logs" && (
        <Panel title="Log">
          <LogTerminal runId={selectedRun} />
        </Panel>
      )}

      {tab === "laws" && (
        <Panel title="Laws">
          {laws.length === 0 ? (
            <Empty>
              Nothing learned yet. A stall writes down a law so later generations stop
              re-testing it.
            </Empty>
          ) : (
            <ul className="space-y-2 text-sm">
              {laws.map((l) => (
                <li key={l.id} className="border-l-2 border-neutral-700 pl-3">
                  <span className="mono text-xs text-neutral-500">gen {l.gen}</span>
                  <p className="text-neutral-300">{l.text}</p>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      )}

      {tab === "ledger" && (
        <div className="grid grid-cols-2 gap-4">
          <Panel title="KV cache">
            <dl className="mono space-y-1 text-xs text-neutral-300">
              {Object.entries(metrics?.kv ?? {}).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <dt className="text-neutral-500">{k}</dt>
                  <dd>{typeof v === "number" ? v.toLocaleString() : String(v)}</dd>
                </div>
              ))}
              {!metrics?.kv || Object.keys(metrics.kv).length === 0 ? (
                <Empty>No KV context resident yet.</Empty>
              ) : null}
            </dl>
          </Panel>
          <Panel title="Orchestrator">
            <dl className="mono space-y-1 text-xs text-neutral-300">
              <div className="flex justify-between">
                <dt className="text-neutral-500">live branches</dt>
                <dd>{metrics?.branches ?? 0}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-neutral-500">worktrees on disk</dt>
                <dd>{metrics?.worktrees ?? 0}</dd>
              </div>
              {Object.entries(metrics?.orchestrator ?? {}).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <dt className="text-neutral-500">{k}</dt>
                  <dd>{String(v)}</dd>
                </div>
              ))}
            </dl>
          </Panel>
        </div>
      )}
    </div>
  );
}
