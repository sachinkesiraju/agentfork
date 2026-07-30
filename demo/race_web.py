"""Browser dashboard for the race demo: this is how the race is watched.

``race_demo.py`` streams its events here by default; ``--no-ui`` skips it and
prints the JSON summary instead. A stdlib ``ThreadingHTTPServer`` serves one
self-contained HTML page
and streams the race as server-sent events, so there is no build step, no
framework, and no new dependency: the page is a few hundred bytes of HTML and
an ``EventSource``.

The UI object satisfies the same interface as ``LogUI`` -- the race does not
know which one it is talking to.
"""

from __future__ import annotations

import json
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>agentfork race</title>
<style>
:root { color-scheme: light; --ink:#0f172a; --mute:#64748b; --line:#e2e8f0;
        --bg:#f8fafc; --panel:#ffffff; --blue:#2563eb; --green:#16a34a;
        --red:#dc2626; --amber:#b45309; }
* { box-sizing: border-box; }
html { height: 100%; }
body { margin: 0; padding: 12px 14px; gap: 10px; display: flex;
       flex-direction: column; min-height: 100vh;
       background: var(--bg); color: var(--ink);
       font: 12px/1.4 ui-sans-serif,system-ui,-apple-system,"Segoe UI",
            Roboto,Inter,sans-serif; }
code, .mono { font-family: ui-monospace,SFMono-Regular,Menlo,monospace; }
h1 { font-size: 16px; margin: 0; }
.intro { font-size: 11px; color: #475569; max-width: 720px; }
.intro b { color: var(--blue); }
.top { display: flex; flex-direction: column; gap: 6px; flex: none; }
.sub { color: var(--mute); font-size: 10.5px; min-height: 1.3em; }
.status { font-size: 11px; color: #0f172a; font-weight: 500; min-height: 1.3em; }
.card { background: var(--panel); border: 1px solid var(--line);
        border-radius: 10px; box-shadow: 0 1px 2px rgba(15,23,42,.06); }
.math { padding: 5px 8px; font-size: 10px; display: flex; flex-wrap: wrap;
        gap: 8px 14px; color: #334155; }
.math b { font-family: ui-monospace,Menlo,monospace; color: var(--ink); }
#gate { padding: 6px 10px; display: flex; gap: 10px; align-items: center;
        flex-wrap: wrap; }
#gate.hidden { display: none; }
#startbtn { font: 600 12px/1 ui-sans-serif,system-ui,sans-serif;
            cursor: pointer; color: #fff; background: var(--blue);
            border: none; border-radius: 6px; padding: 8px 14px; }
#startbtn:hover { background: #1d4ed8; }
#startbtn[disabled] { background: #93c5fd; cursor: default; }
.gatehint { color: var(--mute); font-size: 10px; }
.arms { display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
        align-items: start; }
@media (max-width: 900px) { .arms { grid-template-columns: 1fr; } }
.arm { padding: 10px; display: flex; flex-direction: column; gap: 6px;
       min-width: 0; }
.arm h2 { font-size: 12px; margin: 0; letter-spacing: .08em;
          color: var(--blue); }
#arm-stock h2 { color: var(--mute); }
.arm .url { color: var(--mute); font-size: 10px; }
.tree-wrap { min-height: 220px; max-height: 480px; position: relative;
             overflow: auto; border: 1px solid var(--line); border-radius: 8px;
             background: #fff; padding: 8px; }
.tree { display: block; width: 100%; min-width: 320px; height: auto; }
.legend { color: var(--mute); font-size: 9.5px; line-height: 1.3; }
.legend b { font-weight: 600; }
.legend .g { color: var(--green); } .legend .r { color: var(--red); }
.legend .a { color: var(--amber); }
.kv { height: 12px; background: #eef2f7; border-radius: 4px;
      overflow: hidden; position: relative; border: 1px solid var(--line); }
.kv i { display: block; height: 100%; width: 0; background: #bfdbfe;
        transition: width .2s; }
.kv b { position: absolute; inset: 0; text-align: center; font-weight: 600;
        font-size: 9px; line-height: 10px;
        font-family: ui-monospace,Menlo,monospace; }
.kvlabel { color: var(--mute); font-size: 9px; }
.statbar { display: flex; flex-wrap: wrap; gap: 4px; font-size: 9.5px; }
.statbar span { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }
.ticker { max-height: 70px; overflow-y: auto; flex: none;
          border-top: 1px solid var(--line); padding-top: 4px; }
.ticker > div { font-size: 10px; padding: 2px 0;
                display: flex; gap: 8px; color: #334155; }
.ticker .n { color: var(--mute); flex: none; }
.ticker .hl { flex: none; font-weight: 600; }
.ticker .ch { flex: none; font-family: ui-monospace,Menlo,monospace; }
.ev { color: var(--amber); font-size: 10.5px; }
.done { font-size: 10.5px; }
#score { padding: 8px 12px; }
#score.hidden { display: none; }
#score table { width: 100%; border-collapse: collapse; font-size: 10px;
               line-height: 1.3; }
#score td { padding: 2px 4px; border-top: 1px solid var(--line); }
#score tr:first-child td { border-top: none; }
#score td.k { color: var(--mute); width: 48%; }
#score td.v { text-align: right; width: 26%;
              font-family: ui-monospace,Menlo,monospace; }
#score th { text-align: right; font-size: 9px; color: var(--mute);
            padding: 0 4px 2px; font-weight: 600; }
#score th:first-child { text-align: left; }
.foot { color: var(--mute); font-size: 9px; }
.foot details { margin: 0; }
.foot summary { cursor: pointer; color: var(--blue); font-weight: 600; }
@keyframes pop { from { opacity: 0; transform: translateY(-4px); }
                 to { opacity: 1; transform: none; } }
.tree g.pop { animation: pop .3s ease-out both; }
@keyframes reap { from { opacity: 1; } 40% { opacity: .12; } to { opacity: 1; } }
.tree g.reap { animation: reap .8s ease-out both; }
.ticker-row { animation: pop .3s ease-out both; }
</style></head><body>
<div class="top">
<h1>Stock cache vs. pinned branches</h1>
<div class="intro">Two arms fix the same bug. <b>STOCK</b> hopes the shared
context survives; <b>AGENTFORK</b> pins it so the tree cannot be evicted.</div>
<div class="sub" id="header"></div>
<div class="math card" id="math"></div>
<div id="gate" class="card">
  <button id="startbtn">Start race</button>
  <div class="gatehint">Same load, same bug. Watch the trees grow.</div>
</div>
<div class="status" id="status">Press Start.</div>
</div>
<div class="arms">
  <div class="arm card" id="arm-stock">
    <h2>STOCK</h2><div class="url"></div>
    <div class="tree-wrap"><svg class="tree" xmlns="http://www.w3.org/2000/svg"></svg></div>
    <div class="legend"></div>
    <div class="kv"><i></i><b></b></div><div class="kvlabel">KV pool used</div>
    <div class="statbar"><span>root 0%</span><span>lineage 0%</span><span>prefill 0</span></div>
    <div class="ticker"></div>
    <div class="ev"></div><div class="done"></div>
  </div>
  <div class="arm card" id="arm-agentfork">
    <h2>AGENTFORK</h2><div class="url"></div>
    <div class="tree-wrap"><svg class="tree" xmlns="http://www.w3.org/2000/svg"></svg></div>
    <div class="legend"></div>
    <div class="kv"><i></i><b></b></div><div class="kvlabel">KV pool used</div>
    <div class="statbar"><span>root 0%</span><span>lineage 0%</span><span>prefill 0</span></div>
    <div class="ticker"></div>
    <div class="ev"></div><div class="done"></div>
  </div>
</div>
<div id="score" class="card hidden"></div>
<div class="foot" id="foot"></div>
<script>
const $ = (s, r) => (r || document).querySelector(s);
const arm = n => $("#arm-" + n);
const esc = s => String(s).replace(/[&<>]/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const SVG = "http://www.w3.org/2000/svg";
function status(text) { $("#status").textContent = text; }
const WHO = {stock: "STOCK", agentfork: "AGENTFORK"};
const HITSAID = {
  subtree: "reused whole lineage",
  root: "kept root only",
  miss: "re-prefilled from scratch",
};
const trees = {
  stock: {root: null, nodes: [], byId: {}, killed: false,
          seen: new Set(), reaped: new Set()},
  agentfork: {root: null, nodes: [], byId: {}, killed: false,
              seen: new Set(), reaped: new Set()},
};
const FILL = {subtree: "#f0fdf4", root: "#fffbeb", miss: "#fef2f2"};
const STROKE = {subtree: "#86efac", root: "#fcd34d", miss: "#fecaca"};
const EDGE = {subtree: "#22c55e", root: "#f59e0b", miss: "#f87171"};
const INK = {subtree: "#16a34a", root: "#b45309", miss: "#dc2626"};
const NODE_W = {1: 108, 2: 60, 3: 60};
const GAP = {1: 28, 2: 20, 3: 20};
const ROOT_Y = 22;
const LEVEL_H = 56;
const PAD_Y = 14;
const BH = 30;
const M = 30;
function el(tag, attrs, text) {
  const n = document.createElementNS(SVG, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text !== undefined) n.textContent = text;
  return n;
}
function alive(t, n) {
  if (!t.killed) return true;
  for (let cur = n; cur; cur = t.byId[cur.parent]) if (!cur.passed) return false;
  return true;
}
function layout(t, W) {
  // Hierarchical tree layout: each node is centered over its children,
  // and each subtree gets just enough width so siblings never overlap.
  const children = {};
  for (const n of t.nodes) {
    const p = n.parent || "root";
    if (!children[p]) children[p] = [];
    children[p].push(n);
  }
  if (!children["root"]) children["root"] = [];
  // deterministic order
  for (const k in children) children[k].sort((a, b) => a.id.localeCompare(b.id));

  const H = ROOT_Y + 4 * LEVEL_H + PAD_Y;
  const levelY = {
    1: ROOT_Y + LEVEL_H,
    2: ROOT_Y + 2 * LEVEL_H,
    3: ROOT_Y + 3 * LEVEL_H,
  };

  const widths = {};
  function measure(n) {
    const kids = children[n.id] || [];
    if (!kids.length) return widths[n.id] = NODE_W[n.depth];
    let w = -GAP[n.depth + 1];
    for (const c of kids) w += GAP[n.depth + 1] + measure(c);
    widths[n.id] = Math.max(NODE_W[n.depth], w);
    return widths[n.id];
  }
  // virtual root has depth 0; its children are depth 1 nodes
  const rootW = children["root"].length
    ? (function() {
        let w = -GAP[1];
        for (const c of children["root"]) w += GAP[1] + measure(c);
        return Math.max(220, w);
      })()
    : 220;

  const W2 = Math.max(W, rootW + 2 * M);
  const x = {};
  function place(n, left) {
    const kids = children[n.id] || [];
    if (!kids.length) {
      x[n.id] = left + widths[n.id] / 2;
      return;
    }
    let childLeft = left;
    for (const c of kids) {
      place(c, childLeft);
      childLeft += widths[c.id] + GAP[n.depth + 1];
    }
    x[n.id] = (x[kids[0].id] + x[kids[kids.length - 1].id]) / 2;
  }
  const start = Math.max(M, (W2 - rootW) / 2);
  let nextLeft = start;
  for (const c of children["root"]) {
    place(c, nextLeft);
    nextLeft += widths[c.id] + GAP[1];
  }
  const rootX = children["root"].length ? start + rootW / 2 : W2 / 2;
  x["root"] = rootX;

  return {x, W: W2, H, levelY, rootAt: {x: rootX, y: ROOT_Y + 15}};
}
function drawTree(name) {
  const t = trees[name], svg = $(".tree", arm(name));
  svg.textContent = "";
  const wrap = svg.parentElement;
  const W = Math.max(320, (wrap.clientWidth || 336) - 16);
  const {x, W: W2, H, levelY, rootAt} = layout(t, W);
  svg.style.width = "100%";
  svg.style.minWidth = W2 + "px";
  svg.style.height = "auto";
  svg.setAttribute("viewBox", "0 0 " + W2 + " " + H);
  svg.setAttribute("preserveAspectRatio", "xMidYMin meet");
  if (t.root === null) {
    svg.appendChild(el("text", {x: W2 / 2, y: H / 2, fill: "#94a3b8",
      "text-anchor": "middle", "font-size": 12}, "waiting..."));
    return;
  }
  const rootX = rootAt.x;
  const rw = Math.min(220, W2 - 2 * M);
  const rx = rootX - rw / 2;
  svg.appendChild(el("rect", {x: rx, y: 7, width: rw, height: 30, rx: 8,
    fill: "#eff6ff", stroke: "#bfdbfe"}));
  svg.appendChild(el("text", {x: rootX, y: 22, "font-size": 11,
    "font-weight": 600, fill: "#2563eb", "text-anchor": "middle"}, "SHARED CONTEXT"));
  svg.appendChild(el("text", {x: rootX, y: 34, "font-size": 9,
    fill: "#64748b", "text-anchor": "middle"},
    t.root.charged.toLocaleString() + " tok"));

  for (const n of t.nodes) {
    const cx = x[n.id], y = levelY[n.depth];
    if (cx === undefined) continue;
    const dead = !alive(t, n);
    const from = t.byId[n.parent]
      ? {x: x[n.parent], y: levelY[t.byId[n.parent].depth] + BH / 2}
      : rootAt;
    const g = el("g", {});
    if (!t.seen.has(n.id)) { g.setAttribute("class", "pop"); t.seen.add(n.id); }
    else if (dead && !t.reaped.has(n.id)) {
      g.setAttribute("class", "reap"); t.reaped.add(n.id);
    }
    const bw = NODE_W[n.depth];
    const tagFS = n.depth === 1 ? 9 : 10;
    const numFS = 8;
    g.appendChild(el("path", {
      d: `M${from.x},${from.y} C${from.x},${from.y + 18} ${cx},${y - 18} ${cx},${y - BH / 2}`,
      fill: "none", "stroke-width": n.depth === 1 ? 1.6 : 1.2,
      stroke: dead ? "#cbd5e1" : EDGE[n.hit],
      "stroke-dasharray": dead ? "3 2" : "none"}));
    g.appendChild(el("rect", {x: cx - bw / 2, y: y - BH / 2, width: bw,
      height: BH, rx: 6,
      fill: dead ? "#f8fafc" : FILL[n.hit],
      stroke: dead ? "#e2e8f0" : STROKE[n.hit]}));
    const label = n.depth === 1 ? (n.plan || "").slice(0, 14) : ("+" + n.charged);
    const label2 = n.depth === 1 ? ("+" + n.charged) : "";
    g.appendChild(el("text", {x: cx, y: y - BH / 2 + 12, "text-anchor": "middle",
      "font-size": tagFS, "font-weight": 600,
      fill: dead ? "#94a3b8" : INK[n.hit]}, label));
    if (label2) {
      g.appendChild(el("text", {x: cx, y: y + BH / 2 - 4, "text-anchor": "middle",
        "font-size": numFS, fill: dead ? "#cbd5e1" : "#64748b"}, label2));
    }
    if (n.passed && n.depth > 1) {
      g.appendChild(el("circle", {cx: cx + bw / 2 - 5, cy: y - BH / 2 + 5, r: 4,
        fill: dead ? "#cbd5e1" : "#16a34a"}));
    }
    svg.appendChild(g);
  }
  const label = t.killed ? "subtrees reaped" : "fan-out in progress";
  svg.appendChild(el("text", {x: W2 - 10, y: 12, "text-anchor": "end",
    "font-size": 9, fill: "#94a3b8"}, label));
}
function redraw(name) { requestAnimationFrame(() => drawTree(name)); }
function kv(name, used, cap) {
  const a = arm(name), pct = cap ? Math.min(100, 100 * used / cap) : 0;
  $(".kv i", a).style.width = pct + "%";
  $(".kv b", a).textContent = used.toLocaleString() + " / " + cap.toLocaleString();
  $(".kvlabel", a).textContent = "KV pool used";
}
function updateStatbar(name, d) {
  const a = arm(name);
  const hit = Math.round(100 * (d ? d.hit_rate : 0));
  const sub = Math.round(100 * (d ? d.subtree_rate : 0));
  const pre = d ? d.prefill.toLocaleString() : "0";
  $(".statbar", a).innerHTML =
    "<span>root " + hit + "%</span>" +
    "<span>lineage " + sub + "%</span>" +
    "<span>prefill " + pre + "</span>" +
    (d && d.verified ? "<span style='color:var(--green);font-weight:600'>verified</span>" : "");
}
function tick(name, d) {
  const ticker = $(".ticker", arm(name));
  const row = document.createElement("div");
  row.className = "ticker-row";
  const tag = {subtree: "HIT", root: "ROOT", miss: "MISS"}[d.hit_level];
  row.innerHTML =
    "<span class='n'>" + esc(d.label) + " " + (d.idx + 1) + "/" + d.total + 
      "</span>" +
    "<span class='hl' style='color:" + INK[d.hit_level] + "'>" + tag + "</span>" +
    "<span class='ch'>+" + d.charged.toLocaleString() + "</span>" +
    "<span style='color:" + (d.passed ? "var(--green)" : "var(--red)") + "'>" +
      (d.passed ? "PASS" : "FAIL") + "</span>";
  ticker.appendChild(row);
  while (ticker.children.length > 4) ticker.removeChild(ticker.firstChild);
}
const handlers = {
  banner(d) {
    $("#header").textContent = d.header;
    const w = d.regime === "above"
      ? "U > U*: neighbour evicts unpinned context"
      : "U <= U*: context survives in both arms";
    $("#math").innerHTML =
      "<b>P</b> " + d.P + " &nbsp; " +
      "<b>C</b> " + d.C + " &nbsp; " +
      "<b>U</b> " + d.U + " &nbsp; " +
      "<b>U*</b> " + d.Ustar + " &nbsp; " +
      "<span style='color:var(--amber);font-weight:600'>" + w + "</span>";
    document.querySelectorAll(".legend").forEach(n => n.innerHTML =
      "<b>Tree:</b> root &rarr; approach &rarr; candidate &rarr; verify. " +
      "<span class='g'>HIT</span> full lineage; " +
      "<span class='a'>ROOT</span> shared only; " +
      "<span class='r'>MISS</span> cold. Grey = killed; green dot = passed.");
    redraw("stock"); redraw("agentfork");
  },
  arm_start(d) {
    $(".url", arm(d.name)).textContent = d.url;
    status(WHO[d.name] + " started");
  },
  parent(d) {
    $(".url", arm(d.name)).textContent += " · shared " + d.charged + " tok";
    kv(d.name, d.charged, d.capacity);
    trees[d.name].root = {charged: d.charged};
    redraw(d.name);
    status(WHO[d.name] + " loaded shared context");
  },
  child(d) {
    const t = trees[d.name];
    const node = {id: d.branch_id, parent: d.parent_branch, depth: d.depth,
                  hit: d.hit_level, charged: d.charged, passed: d.passed,
                  plan: d.plan || ""};
    t.nodes.push(node);
    t.byId[node.id] = node;
    redraw(d.name);
    tick(d.name, d);
    status(WHO[d.name] + " " + d.label + " " + (d.idx + 1) + "/" + d.total +
           ": " + HITSAID[d.hit_level] + " (+" + d.charged.toLocaleString() + ")");
  },
  kv(d) { kv(d.name, d.used, d.capacity); },
  kills(d) {
    trees[d.name].killed = true;
    redraw(d.name);
    $(".ev", arm(d.name)).textContent = "reaped " + d.n + " losers; freed " + d.freed.toLocaleString() + " tok";
    status(WHO[d.name] + " reaped " + d.n + " losers");
  },
  arm_done(d) {
    updateStatbar(d.name, d);
    $(".done", arm(d.name)).innerHTML =
      (d.verified ? "<span style='color:var(--green);font-weight:600'>VERIFIED</span>" : "UNVERIFIED") +
      " &nbsp; prefill " + d.prefill.toLocaleString();
    redraw(d.name);
    status(WHO[d.name] + " done: root " + Math.round(100 * d.hit_rate) +
      "%, prefill " + d.prefill.toLocaleString());
  },
  scoreboard(d) {
    $("#score").classList.remove("hidden");
    $("#score").innerHTML =
      "<table><thead><tr><th>metric</th><th>STOCK</th><th>AGENTFORK</th></tr></thead><tbody>" +
      d.rows.map(r =>
        "<tr><td class='k'>" + esc(r[0]) + "</td>" +
        "<td class='v'>" + esc(r[1]) + "</td>" +
        "<td class='v'>" + esc(r[2]) + "</td></tr>").join("") +
      "</tbody></table>";
    $("#foot").innerHTML =
      "<details><summary>honest caveats</summary>" +
      esc(d.notes).replace(/\\n/g, "<br>") + "</details>";
    redraw("stock"); redraw("agentfork");
    status("Race finished.");
  },
};
handlers.started = () => { $("#gate").classList.add("hidden"); status("Race running..."); };
$("#startbtn").onclick = () => {
  $("#startbtn").disabled = true;
  $("#startbtn").textContent = "racing...";
  fetch("/start", {method: "POST"});
};
window.addEventListener("resize", () => { redraw("stock"); redraw("agentfork"); });
const src = new EventSource("/events");
const pending = [];
let draining = false;
const PACED = {parent: 420, child: 320, kills: 700, arm_start: 500};
function drain() {
  if (draining) return;
  draining = true;
  (function step() {
    if (!pending.length) { draining = false; return; }
    const msg = pending.shift();
    if (msg.kind === "done") { src.close(); draining = false; return; }
    const h = handlers[msg.kind];
    if (h) h(msg);
    setTimeout(step, PACED[msg.kind] || 0);
  })();
}
src.onmessage = e => { pending.push(JSON.parse(e.data)); drain(); };
</script></body></html>
"""

class _Handler(BaseHTTPRequestHandler):
    server_version = "AgentforkRaceWeb/1"

    def log_message(self, *a):  # keep the terminal clean
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/events":
            self._stream()
            return
        self.send_error(404)

    def do_POST(self):
        if self.path == "/start":
            self.server.ui.start.set()  # type: ignore[attr-defined]
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404)

    def _stream(self):
        ui = self.server.ui  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        q = ui.subscribe()
        try:
            for event in ui.replay():   # a late browser still sees the race
                self._send_event(event)
            while True:
                self._send_event(q.get())
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            ui.unsubscribe(q)

    def _send_event(self, event: dict) -> None:
        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.flush()


class WebUI:
    """Race UI that streams events to a browser (and mirrors them to a log)."""

    def __init__(self, args, inner, port: int = 8765, open_browser: bool = True):
        self.args = args
        self.inner = inner
        self.capacity = args.capacity_tokens
        self._lock = threading.Lock()
        self.start = threading.Event()
        self._history: list[dict] = []
        self._subs: list[queue.Queue] = []
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        self.httpd.ui = self  # type: ignore[attr-defined]
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.inner.say(f"web dashboard: {self.url}")
        if open_browser:
            threading.Thread(target=webbrowser.open, args=(self.url,),
                             daemon=True).start()

    def wait_for_start(self, autostart: bool = False) -> None:
        if autostart:
            self.start.set()
        elif not self.start.is_set():
            self.inner.say("waiting for Start in the browser...")
        self.start.wait()
        self._emit(kind="started")

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def replay(self) -> list[dict]:
        with self._lock:
            return list(self._history)

    def _emit(self, **event) -> None:
        with self._lock:
            self._history.append(event)
            subs = list(self._subs)
        for q in subs:
            q.put(event)

    def say(self, msg: str = "") -> None:
        self.inner.say(msg)

    def banner(self, race, args) -> None:
        from demo.race_demo import HEADER, break_even_u
        self.inner.banner(race, args)
        u = args.noise_request_tokens * args.noise_requests_per_gap
        ustar = break_even_u(race.prefix_tokens, args.capacity_tokens)
        self._emit(kind="banner", header=HEADER, P=race.prefix_tokens,
                   C=args.capacity_tokens, U=u, Ustar=ustar,
                   regime="above" if u > ustar else "at_or_below",
                   N=args.children, verify=args.verify_children,
                   approaches=args.approaches,
                   approach_tokens=args.approach_tokens)

    def arm_start(self, name, url) -> None:
        self.inner.arm_start(name, url)
        self._emit(kind="arm_start", name=name, url=url)

    def parent(self, name, prefix_tokens, charged) -> None:
        self.inner.parent(name, prefix_tokens, charged)
        self._emit(kind="parent", name=name, P=prefix_tokens, charged=charged,
                   capacity=self.capacity)

    def child(self, name, label, idx, rec, total) -> None:
        self.inner.child(name, label, idx, rec, total)
        self._emit(kind="child", name=name, label=label, idx=idx, total=total,
                   branch_id=rec.branch_id, parent_branch=rec.parent_branch,
                   depth=rec.depth, hit_level=rec.hit_level,
                   plan=rec.title or rec.branch_id.rsplit("/", 1)[-1],
                   parent_hit=rec.parent_hit, cached=rec.cached_tokens,
                   charged=rec.charged_tokens, passed=rec.check_passed)

    def kills(self, name, n, freed, pool_delta) -> None:
        self.inner.kills(name, n, freed, pool_delta)
        self._emit(kind="kills", name=name, n=n, freed=freed,
                   pool_delta=pool_delta)

    def event(self, name, kind, detail="") -> None:
        self.inner.event(name, kind, detail)

    def kv(self, name, used) -> None:
        self._emit(kind="kv", name=name, used=used, capacity=self.capacity)

    def arm_done(self, arm) -> None:
        self.inner.arm_done(arm)
        self._emit(kind="arm_done", name=arm.name, hit_rate=arm.parent_hit_rate,
                   subtree_rate=arm.subtree_hit_rate,
                   prefill=arm.prefill_charged, verified=arm.verified)

    def scoreboard(self, rows, notes: str) -> None:
        self._emit(kind="scoreboard", rows=rows, notes=notes)

    def finish(self) -> None:
        self._emit(kind="done")
