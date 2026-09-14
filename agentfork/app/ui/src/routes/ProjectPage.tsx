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
import {
  Badge,
  Button,
  Empty,
  ErrorNote,
  KeyValue,
  Panel,
  Stat,
  fmt,
  signed,
} from "../components/ui";
import { Page, TopBar } from "../components/Shell";
import LogTerminal from "../components/LogTerminal";
import TreeView from "../components/TreeView";

type Tab = "tree" | "runs" | "logs" | "laws" | "ledger";

const TABS: { id: Tab; label: string }[] = [
  { id: "tree", label: "Tree" },
  { id: "runs", label: "Runs" },
  { id: "logs", label: "Logs" },
  { id: "laws", label: "Laws" },
  { id: "ledger", label: "Ledger" },
];

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
  // the loop state is the loop's own word for it: a hand-driven baseline
  // also writes "baseline" — only a live AutoresearchLoop means stop-able
  const looping = loop?.driven === true;
  const activeRuns = runs.filter(
    (r) => r.status === "running" || r.status === "queued",
  ).length;

  if (!project) {
    return (
      <>
        <TopBar />
        <Page>
          {error ? <ErrorNote>{error}</ErrorNote> : <Empty>Loading project…</Empty>}
        </Page>
      </>
    );
  }

  const p = project.params;

  return (
    <>
      <TopBar>
        {activeRuns > 0 && (
          <span className="mono hidden text-xs text-neutral-500 sm:inline">
            {activeRuns} run{activeRuns > 1 ? "s" : ""} in flight
          </span>
        )}
        {!hasBaseline && (
          <Button
            disabled={!!busy}
            onClick={() => act("baseline", () => api.baseline(projectId))}
          >
            {busy === "baseline" ? "Starting…" : `Run baseline ×${p.baseline_runs}`}
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
            {busy === "loop" ? "Starting…" : "Start autoresearch"}
          </Button>
        )}
      </TopBar>

      <Page>
        <header className="space-y-2">
          <Link
            to="/"
            className="inline-flex items-center gap-1 text-xs text-neutral-500 transition-colors hover:text-neutral-300"
          >
            ← All projects
          </Link>
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-xl font-semibold tracking-tight text-neutral-50">
              {project.name}
            </h1>
            <Badge status={loop?.state ?? "idle"} />
          </div>
          <p className="mono flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-neutral-500">
            <span className="truncate">{project.repo_path}</span>
            <Dot />
            <span>{project.baseline_branch}</span>
            <Dot />
            <span>{project.harness}</span>
            <Dot />
            <span>{p.minimize ? "minimize" : "maximize"}</span>
            <span className="text-neutral-400">{p.metric_grep}</span>
          </p>
        </header>

        <section className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-5">
          <Stat label="Generation" value={loop?.gen ?? 0} hint={`of ${p.max_gens} max`} />
          <Stat
            label="Noise margin"
            value={loop?.margin === null || loop?.margin === undefined ? "—" : fmt(loop.margin)}
            hint="from the baseline pair"
          />
          <Stat label="Fan out" value={`K=${p.k}`} hint={`beam B=${p.b}`} />
          <Stat label="Nodes" value={nodes.length} hint={`${runs.length} runs`} />
          <Stat
            label="Eval command"
            value={<span className="truncate">{p.eval_cmd}</span>}
            hint={`timeout ${p.timeout_s}s`}
          />
        </section>

        {loop?.message && (
          <p className="flex items-center gap-2 rounded-lg border border-neutral-800 bg-neutral-900/50 px-3 py-2 text-xs text-neutral-400">
            {looping && (
              <span className="live-dot h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400" />
            )}
            <span className="min-w-0 break-words">{loop.message}</span>
          </p>
        )}

        {error && <ErrorNote>{error}</ErrorNote>}

        <nav className="flex gap-1 border-b border-neutral-800">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`-mb-px border-b-2 px-3 py-2 text-sm font-medium transition-colors ${
                tab === t.id
                  ? "border-emerald-400 text-neutral-50"
                  : "border-transparent text-neutral-500 hover:border-neutral-700 hover:text-neutral-300"
              }`}
            >
              {t.label}
              {t.id === "laws" && laws.length > 0 && (
                <span className="mono ml-1.5 rounded bg-neutral-800 px-1 text-[10px] text-neutral-400">
                  {laws.length}
                </span>
              )}
            </button>
          ))}
        </nav>

        {tab === "tree" && (
          <div className="grid gap-4 lg:grid-cols-3">
            <div className="space-y-3 lg:col-span-2">
              <Panel
                title="Experiment tree"
                subtitle="each node is a branch: a git worktree plus its KV context"
                padded={false}
                right={<Legend />}
              >
                {nodes.length === 0 ? (
                  <Empty title="No nodes yet">
                    Run the baseline to create the root and fix the noise margin — every later
                    candidate is judged against it.
                  </Empty>
                ) : (
                  <TreeView
                    nodes={nodes}
                    frontier={loop?.frontier ?? []}
                    selected={selectedNode}
                    onSelect={setSelectedNode}
                  />
                )}
              </Panel>

              <div className="flex flex-wrap items-center gap-2">
                <Button
                  variant="primary"
                  disabled={!node || !!busy}
                  onClick={() =>
                    node && act("fanout", () => api.fanout(projectId, node.id, p.k))
                  }
                >
                  {busy === "fanout" ? "Forking…" : `Fan out ×${p.k}`}
                </Button>
                <Button
                  disabled={!loop || !loop.gen || !!busy || activeRuns > 0}
                  title={activeRuns > 0 ? "wait for in-flight runs" : undefined}
                  onClick={() =>
                    loop &&
                    act("reduce", async () => setReduction(await api.reduce(projectId, loop.gen)))
                  }
                >
                  {busy === "reduce" ? "Reducing…" : `Reduce gen ${loop?.gen ?? 0}`}
                </Button>
                <Button
                  disabled={!node || !!busy}
                  onClick={() => node && act("eval", () => api.evalNode(projectId, node.id))}
                >
                  Re-run eval
                </Button>
                {p.holdout_cmd && (
                  <Button
                    disabled={!node || !!busy}
                    onClick={() => node && act("holdout", () => api.holdout(projectId, node.id))}
                  >
                    Holdout
                  </Button>
                )}
                {!node && (
                  <span className="text-xs text-neutral-600">
                    Select a node to fan out from it.
                  </span>
                )}
              </div>

              {reduction && (
                <Panel
                  title={`Reduction · gen ${reduction.gen}`}
                  subtitle={`${reduction.candidates} candidates · ${reduction.beat_parent} beat their parent · margin ${reduction.margin}`}
                  right={
                    <Button size="sm" variant="ghost" onClick={() => setReduction(null)}>
                      dismiss
                    </Button>
                  }
                >
                  <div className="space-y-2">
                    {reduction.kept.map((s) => (
                      <div
                        key={s.commit}
                        className="flex flex-wrap items-center gap-2 rounded-lg border border-emerald-800/50 bg-emerald-500/10 px-3 py-2"
                      >
                        <Badge status="kept">KEEP</Badge>
                        <span className="mono truncate text-xs text-neutral-200">{s.commit}</span>
                        <span className="mono nums text-xs text-emerald-200">{fmt(s.score)}</span>
                        <span className="mono nums text-xs text-neutral-500">
                          {signed(s.delta)} vs {s.parent}
                        </span>
                      </div>
                    ))}
                    {reduction.cost_failures.map((f) => (
                      <div
                        key={f.commit}
                        className="flex flex-wrap items-center gap-2 rounded-lg border border-orange-800/50 bg-orange-500/10 px-3 py-2"
                      >
                        <Badge status="cost_fail">COST_FAIL</Badge>
                        <span className="mono truncate text-xs text-neutral-200">{f.commit}</span>
                        <span className="text-xs text-orange-200">{f.reason}</span>
                      </div>
                    ))}
                    {reduction.stall && (
                      <p className="rounded-lg border border-amber-800/50 bg-amber-500/10 px-3 py-2 text-xs text-amber-100">
                        STALL — no candidate beat its parent by more than the margin. The
                        frontier is unchanged and a law was written down.
                      </p>
                    )}
                    {reduction.fuse.map((f) => (
                      <p
                        key={`${f.a}-${f.b}`}
                        className="mono rounded-lg border border-neutral-800 px-3 py-2 text-xs text-neutral-400"
                      >
                        FUSE_CANDIDATE {f.a} + {f.b} — disjoint regions; a combined
                        candidate is worth a slot in the next generation
                      </p>
                    ))}
                    <p className="mono truncate pt-1 text-xs text-neutral-500">
                      frontier: {reduction.frontier.join("  ") || "—"}
                    </p>
                  </div>
                </Panel>
              )}
            </div>

            <Panel title="Node" subtitle={node ? node.id : "nothing selected"}>
              {!node ? (
                <Empty title="No node selected">
                  Click a node in the tree to see its score, branch and runs.
                </Empty>
              ) : (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <p className="text-sm font-medium leading-snug text-neutral-100">
                      {node.title}
                    </p>
                    <div className="flex flex-wrap items-center gap-1.5">
                      <Badge status={node.status} />
                      {node.frozen && <Badge status="frozen" />}
                      {node.regions.map((r) => (
                        <span
                          key={r}
                          className="mono rounded border border-neutral-800 bg-neutral-900 px-1.5 py-0.5 text-[10px] text-neutral-400"
                        >
                          {r}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <Stat label="Score" value={fmt(node.score)} />
                    <Stat
                      label="vs parent"
                      value={
                        <span
                          className={
                            node.delta === null
                              ? ""
                              : (node.delta < 0) === p.minimize
                                ? "text-emerald-300"
                                : "text-red-300"
                          }
                        >
                          {signed(node.delta)}
                        </span>
                      }
                    />
                  </div>

                  <dl>
                    <KeyValue k="generation" v={node.gen} />
                    <KeyValue k="branch" v={node.branch_name} />
                    <KeyValue k="commit" v={node.commit_sha?.slice(0, 12) ?? "—"} />
                    <KeyValue
                      k="worktree"
                      v={
                        <span title={node.worktree_path ?? ""}>
                          {node.worktree_path ?? "(collected)"}
                        </span>
                      }
                    />
                  </dl>

                  {node.description && (
                    <p className="whitespace-pre-wrap rounded-lg border border-neutral-800 bg-neutral-950/50 p-2.5 text-xs leading-relaxed text-neutral-400">
                      {node.description}
                    </p>
                  )}

                  <div className="space-y-1.5">
                    <p className="text-[11px] uppercase tracking-wider text-neutral-500">Runs</p>
                    {nodeRuns.length === 0 ? (
                      <p className="text-xs text-neutral-600">none yet</p>
                    ) : (
                      nodeRuns.map((r) => (
                        <button
                          key={r.id}
                          onClick={() => {
                            setSelectedRun(r.id);
                            setTab("logs");
                          }}
                          className="flex w-full items-center justify-between gap-2 rounded-md border border-neutral-800 bg-neutral-900/60 px-2 py-1.5 text-left transition-colors hover:border-neutral-700 hover:bg-neutral-800/60"
                        >
                          <span className="mono truncate text-xs text-neutral-300">{r.kind}</span>
                          <span className="flex shrink-0 items-center gap-2">
                            <span className="mono nums text-xs text-neutral-400">
                              {fmt(r.score)}
                            </span>
                            <Badge status={r.status} />
                          </span>
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
          <Panel title="Runs" subtitle="every run this project has made" padded={false}>
            {runs.length === 0 ? (
              <Empty title="No runs yet">
                Runs appear when the baseline, a candidate eval or a holdout starts.
              </Empty>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-neutral-800 text-[11px] uppercase tracking-wider text-neutral-500">
                      <th className="px-4 py-2 font-medium">Run</th>
                      <th className="px-2 py-2 font-medium">Kind</th>
                      <th className="px-2 py-2 font-medium">Status</th>
                      <th className="px-2 py-2 font-medium">Node</th>
                      <th className="px-2 py-2 text-right font-medium">Score</th>
                      <th className="px-2 py-2 text-right font-medium">Exit</th>
                      <th className="px-2 py-2 font-medium">Command</th>
                      <th className="px-4 py-2" />
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map((r) => (
                      <tr
                        key={r.id}
                        className="border-b border-neutral-800/60 transition-colors last:border-0 hover:bg-neutral-800/30"
                      >
                        <td className="px-4 py-2">
                          <button
                            className="mono text-neutral-200 hover:text-white hover:underline"
                            onClick={() => {
                              setSelectedRun(r.id);
                              setTab("logs");
                            }}
                          >
                            {r.id}
                          </button>
                        </td>
                        <td className="mono px-2 py-2 text-neutral-400">{r.kind}</td>
                        <td className="px-2 py-2">
                          <Badge status={r.status} />
                        </td>
                        <td className="mono px-2 py-2 text-neutral-500">{r.node_id}</td>
                        <td className="mono nums px-2 py-2 text-right text-neutral-200">
                          {fmt(r.score)}
                        </td>
                        <td className="mono nums px-2 py-2 text-right text-neutral-500">
                          {r.exit_code ?? "—"}
                        </td>
                        <td className="mono max-w-[16rem] truncate px-2 py-2 text-neutral-500">
                          {r.command}
                        </td>
                        <td className="px-4 py-2 text-right">
                          {(r.status === "running" || r.status === "queued") && (
                            <Button
                              size="sm"
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
              </div>
            )}
          </Panel>
        )}

        {tab === "logs" && (
          <Panel
            title="Log"
            subtitle={selectedRun ?? "pick a run from the Runs tab"}
            padded={false}
            right={
              selectedRun && (
                <Button size="sm" variant="ghost" onClick={() => act("kill", () => api.killRun(selectedRun))}>
                  kill run
                </Button>
              )
            }
          >
            <LogTerminal runId={selectedRun} />
          </Panel>
        )}

        {tab === "laws" && (
          <Panel title="Laws" subtitle="what the search has already ruled out">
            {laws.length === 0 ? (
              <Empty title="Nothing learned yet">
                When a generation stalls, agentfork writes the reason down as a law so later
                generations stop re-testing it.
              </Empty>
            ) : (
              <ul className="space-y-2">
                {laws.map((l) => (
                  <li
                    key={l.id}
                    className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-3"
                  >
                    <p className="mono text-[11px] uppercase tracking-wider text-amber-300/80">
                      gen {l.gen}
                    </p>
                    <p className="mt-1 text-sm leading-relaxed text-neutral-200">{l.text}</p>
                  </li>
                ))}
              </ul>
            )}
          </Panel>
        )}

        {tab === "ledger" && (
          <div className="grid gap-4 lg:grid-cols-2">
            <Panel
              title="KV cache"
              subtitle="context reuse across branches — agentfork's own accounting"
            >
              {!metrics?.kv || Object.keys(metrics.kv).length === 0 ? (
                <Empty title="No KV context resident">
                  Context is tokenized into the tree cache when nodes are forked.
                </Empty>
              ) : (
                <dl>
                  {Object.entries(metrics.kv).map(([k, v]) => (
                    <KeyValue
                      key={k}
                      k={k.replace(/_/g, " ")}
                      v={typeof v === "number" ? v.toLocaleString() : String(v)}
                    />
                  ))}
                </dl>
              )}
            </Panel>
            <Panel title="Orchestrator" subtitle="branch lifecycle counters">
              <dl>
                <KeyValue k="live branches" v={metrics?.branches ?? 0} />
                <KeyValue k="worktrees on disk" v={metrics?.worktrees ?? 0} />
                {Object.entries(metrics?.orchestrator ?? {}).map(([k, v]) => (
                  <KeyValue key={k} k={k.replace(/_/g, " ")} v={String(v)} />
                ))}
              </dl>
              <p className="mt-3 text-[11px] leading-relaxed text-neutral-600">
                Branch rows are journaled, so after a crash this count reflects what was
                recorded, not what the previous process left running; the first mutating
                action reconciles it. The tree itself is kept in git.
              </p>
            </Panel>
          </div>
        )}
      </Page>
    </>
  );
}

function Dot() {
  return <span className="text-neutral-700">·</span>;
}

function Legend() {
  const items = [
    ["bg-sky-500/70", "baseline"],
    ["bg-amber-500/70", "implementing"],
    ["bg-blue-500/70", "running"],
    ["bg-emerald-500/70", "kept"],
    ["bg-orange-500/70", "cost_fail"],
    ["bg-red-500/70", "crash"],
    ["bg-neutral-600", "discarded"],
  ];
  return (
    <div className="hidden items-center gap-2.5 md:flex">
      {items.map(([color, label]) => (
        <span key={label} className="flex items-center gap-1 text-[11px] text-neutral-500">
          <span className={`h-2 w-2 rounded-sm ${color}`} />
          {label}
        </span>
      ))}
    </div>
  );
}
