export type Params = {
  eval_cmd: string;
  metric_grep: string;
  minimize: boolean;
  k: number;
  b: number;
  eval_slots: number;
  timeout_s: number;
  max_gens: number;
  baseline_runs: number;
  holdout_cmd: string;
  cost_guards: Record<string, number>;
};

export type Project = {
  id: string;
  name: string;
  repo_path: string;
  baseline_branch: string;
  harness: string;
  params: Params;
  created_at: number;
  loop?: LoopState;
};

export type Node = {
  id: string;
  project_id: string;
  parent_id: string | null;
  gen: number;
  slug: string;
  branch_name: string;
  worktree_path: string | null;
  commit_sha: string | null;
  title: string;
  description: string;
  regions: string[];
  status: string;
  score: number | null;
  delta: number | null;
  costs: Record<string, number>;
  frozen: boolean;
  created_at: number;
};

export type Run = {
  id: string;
  project_id: string;
  node_id: string;
  kind: string;
  status: string;
  command: string;
  pid: number | null;
  exit_code: number | null;
  score: number | null;
  created_at: number;
};

export type LoopState = {
  state: string;
  gen: number;
  margin: number | null;
  frontier: string[];
  message: string;
  driven?: boolean; // a live AutoresearchLoop owns the state right now
};

export type Law = { id: string; gen: number; text: string; created_at: number };

export type Metrics = {
  orchestrator: Record<string, number>;
  kv: Record<string, number>;
  branches?: number;
  worktrees?: number;
};

export type Harnesses = Record<
  string,
  { installed: boolean; authed: boolean; detail?: string }
>;

// When the server is bound off loopback it requires a bearer token, handed
// to the UI via ?token= in the printed URL. Persist it for this tab so a
// refresh keeps working.
const TOKEN = (() => {
  const fromUrl = new URLSearchParams(location.search).get("token");
  if (fromUrl) sessionStorage.setItem("af-token", fromUrl);
  return sessionStorage.getItem("af-token") || "";
})();

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (TOKEN) headers["authorization"] = `Bearer ${TOKEN}`;
  const resp = await fetch(`/api${path}`, {
    ...init,
    headers,
  });
  const text = await resp.text();
  const body = text ? JSON.parse(text) : null;
  if (!resp.ok) throw new Error(body?.error || resp.statusText);
  return body as T;
}

export const api = {
  health: () => call<{ ok: boolean; version: string; home: string }>("/health"),
  harnesses: () => call<Harnesses>("/harnesses"),
  projects: () => call<Project[]>("/projects"),
  project: (id: string) => call<Project>(`/projects/${id}`),
  createProject: (body: {
    name: string;
    repo_path: string;
    baseline_branch: string;
    harness: string;
    params: Partial<Params>;
  }) => call<Project>("/projects", { method: "POST", body: JSON.stringify(body) }),
  deleteProject: (id: string) =>
    call<{ deleted: string }>(`/projects/${id}`, { method: "DELETE" }),
  tree: (id: string) => call<{ nodes: Node[]; loop: LoopState }>(`/projects/${id}/tree`),
  runs: (id: string) => call<Run[]>(`/projects/${id}/runs`),
  laws: (id: string) => call<Law[]>(`/projects/${id}/laws`),
  metrics: (id: string) => call<Metrics>(`/projects/${id}/metrics`),
  results: (id: string) =>
    call<{ path: string; rows: unknown[]; tree: string }>(`/projects/${id}/results`),
  baseline: (id: string) =>
    call<Node>(`/projects/${id}/baseline`, { method: "POST", body: "{}" }),
  fanout: (id: string, parent_id: string, k?: number) =>
    call<{ nodes: Node[] }>(`/projects/${id}/fanout`, {
      method: "POST",
      body: JSON.stringify({ parent_id, k }),
    }),
  reduce: (id: string, gen: number, margin?: number) =>
    call<Reduction>(`/projects/${id}/reduce`, {
      method: "POST",
      body: JSON.stringify({ gen, margin }),
    }),
  startLoop: (id: string) =>
    call<{ state: string }>(`/projects/${id}/loop/start`, { method: "POST", body: "{}" }),
  stopLoop: (id: string) =>
    call<{ state: string }>(`/projects/${id}/loop/stop`, { method: "POST", body: "{}" }),
  holdout: (id: string, node_id: string) =>
    call<Run>(`/projects/${id}/holdout`, {
      method: "POST",
      body: JSON.stringify({ node_id }),
    }),
  evalNode: (id: string, node_id: string) =>
    call<Run>(`/projects/${id}/eval`, {
      method: "POST",
      // the button is the explicit re-answer — answered nodes are frozen
      body: JSON.stringify({ node_id, force: true }),
    }),
  log: (runId: string, offset: number) =>
    call<{
      text: string;
      offset: number;
      alive: boolean;
      exit_code: number | null;
      status: string;
    }>(`/runs/${runId}/log?offset=${offset}`),
  killRun: (runId: string) => call<Run>(`/runs/${runId}/kill`, { method: "POST", body: "{}" }),
};

export type Reduction = {
  gen: number;
  margin: number;
  beam: number;
  candidates: number;
  beat_parent: number;
  kept: { commit: string; score: number; parent: string; delta: number }[];
  cost_failures: { commit: string; reason: string }[];
  fuse: { a: string; b: string; regions: string[] }[];
  frontier: string[];
  stall: boolean;
};

export type Event = {
  seq: number;
  project_id: string | null;
  ts: number;
  kind: string;
  payload: Record<string, unknown>;
};

export function subscribe(onEvent: (e: Event) => void): () => void {
  // EventSource cannot set headers — the server accepts ?token= instead
  const es = new EventSource(`/api/events${TOKEN ? `?token=${TOKEN}` : ""}`);
  es.onmessage = (m) => {
    try {
      onEvent(JSON.parse(m.data) as Event);
    } catch {
      /* ignore malformed frame */
    }
  };
  return () => es.close();
}
