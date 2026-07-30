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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>agentfork race</title>
<style>
:root { color-scheme: light; --ink:#0f172a; --mute:#64748b; --line:#e2e8f0;
        --bg:#f8fafc; --panel:#ffffff; --blue:#2563eb; --green:#16a34a;
        --red:#ef4444; --amber:#f59e0b; }
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body { background: var(--bg); color: var(--ink);
       font: 13px/1.4 ui-sans-serif,system-ui,-apple-system,Roboto,Inter,sans-serif;
       display: flex; flex-direction: column; }
header { background: #fff; border-bottom: 1px solid var(--line);
         padding: 10px 16px; display: flex; align-items: center;
         gap: 12px; flex-wrap: wrap; }
header h1 { font-size: 15px; font-weight: 700; margin: 0; letter-spacing: -.2px; }
header p { font-size: 11px; color: var(--mute); margin: 0; flex: 1 1 220px; }
#startbtn { background: var(--blue); color: #fff; border: none;
            border-radius: 8px; padding: 8px 18px;
            font: 600 12px/1 sans-serif; cursor: pointer; }
#startbtn:hover { background: #1d4ed8; }
#startbtn[disabled] { background: #93c5fd; cursor: default; }
#status { font-size: 12px; font-weight: 600; color: var(--ink); min-width: 120px; }
main { flex: 1; display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
       padding: 16px; min-height: 0; overflow: hidden; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } }
.arm { background: var(--panel); border: 1px solid var(--line);
        border-radius: 14px; box-shadow: 0 1px 2px rgba(15,23,42,.05);
        display: flex; flex-direction: column; min-height: 0; overflow: hidden; }
.arm-head { padding: 10px 14px; border-bottom: 1px solid var(--line);
            display: flex; justify-content: space-between; align-items: center; }
.arm h2 { font-size: 13px; margin: 0; letter-spacing: .08em; color: var(--blue); }
#arm-stock h2 { color: var(--mute); }
.arm .url { font-size: 10px; color: var(--mute); }
.metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px;
           background: var(--line); border-bottom: 1px solid var(--line); }
.metric { background: #fff; padding: 8px; text-align: center; }
.metric .num { font: 700 15px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
               color: var(--ink); }
.metric .num.win { color: var(--green); }
.metric .num.lose { color: var(--red); }
.metric .lbl { font-size: 9px; color: var(--mute); text-transform: uppercase;
                letter-spacing: .04em; margin-top: 2px; }
.kvbar { height: 6px; background: #e2e8f0; overflow: hidden; }
.kvbar i { display: block; height: 100%; width: 0; background: var(--blue);
            transition: width .2s; }
.tree-wrap { flex: 1; min-height: 220px; position: relative; overflow: auto;
             padding: 12px; background: #fafafa; }
.tree { display: block; width: 100%; min-width: 320px; height: auto; }
.legend { padding: 8px 14px; font-size: 10px; color: var(--mute);
          border-top: 1px solid var(--line);
          display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.legend b { font-weight: 600; color: var(--ink); }
.legend .dot { display: inline-block; width: 8px; height: 8px;
               border-radius: 50%; margin-right: 3px; }
.legend .g { color: var(--green); } .legend .a { color: var(--amber); }
.legend .r { color: var(--red); }
.ticker { max-height: 58px; overflow-y: auto; border-top: 1px solid var(--line);
          padding: 6px 14px; font-size: 10px; color: #475569; }
.ticker-row { display: flex; gap: 8px; padding: 1px 0; }
.ticker-row .tag { font-weight: 700; }
.ticker-row .dim { color: var(--mute); }
#scoreboard { display: none; background: #fff; border-top: 1px solid var(--line);
              padding: 12px 16px; }
#scoreboard.open { display: block; }
#scoreboard h3 { margin: 0 0 8px; font-size: 13px; }
#scoreboard table { width: 100%; border-collapse: collapse; font-size: 11px; }
#scoreboard th { text-align: right; color: var(--mute); padding: 4px;
                 font-weight: 600; }
#scoreboard th:first-child { text-align: left; }
#scoreboard td { padding: 4px; border-top: 1px solid var(--line);
                 text-align: right; font-family: ui-monospace,monospace; }
#scoreboard td:first-child { text-align: left; color: var(--mute); }
#scoreboard .winner td { font-weight: 700; background: #f0fdf4; }
#foot { padding: 10px 16px; font-size: 10px; color: var(--mute);
        background: #fff; border-top: 1px solid var(--line); }
#foot summary { cursor: pointer; color: var(--blue); font-weight: 600; }
@keyframes pop { from { opacity: 0; transform: translateY(-6px); }
                 to { opacity: 1; transform: none; } }
.tree g.pop { animation: pop .35s ease-out both; }
@keyframes reap { from { opacity: 1; } 40% { opacity: .1; } to { opacity: 1; } }
.tree g.reap { animation: reap .8s ease-out both; }
</style></head><body>
<header>
  <h1>Stock cache vs. pinned branches</h1>
  <p>Two arms race to fix the same bug under identical KV pressure.</p>
  <button id="startbtn">Start race</button>
  <div id="status">Press Start.</div>
</header>
<main>
  <section class="arm" id="arm-stock">
    <div class="arm-head"><h2>STOCK</h2><div class="url"></div></div>
    <div class="metrics">
      <div class="metric"><div class="num" id="m-stock-prefill">0</div><div class="lbl">prefill tok</div></div>
      <div class="metric"><div class="num" id="m-stock-subtree">0%</div><div class="lbl">lineage hit</div></div>
      <div class="metric"><div class="num" id="m-stock-root">0%</div><div class="lbl">root hit</div></div>
    </div>
    <div class="kvbar"><i id="kv-stock"></i></div>
    <div class="tree-wrap"><svg class="tree" xmlns="http://www.w3.org/2000/svg"></svg></div>
    <div class="legend">
      <b>Cache:</b>
      <span class="g"><span class="dot"></span>full lineage</span>
      <span class="a"><span class="dot"></span>root only</span>
      <span class="r"><span class="dot"></span>miss</span>
      <span>grey/dashed = killed</span>
    </div>
    <div class="ticker"></div>
  </section>
  <section class="arm" id="arm-agentfork">
    <div class="arm-head"><h2>AGENTFORK</h2><div class="url"></div></div>
    <div class="metrics">
      <div class="metric"><div class="num" id="m-agentfork-prefill">0</div><div class="lbl">prefill tok</div></div>
      <div class="metric"><div class="num" id="m-agentfork-subtree">0%</div><div class="lbl">lineage hit</div></div>
      <div class="metric"><div class="num" id="m-agentfork-root">0%</div><div class="lbl">root hit</div></div>
    </div>
    <div class="kvbar"><i id="kv-agentfork"></i></div>
    <div class="tree-wrap"><svg class="tree" xmlns="http://www.w3.org/2000/svg"></svg></div>
    <div class="legend">
      <b>Cache:</b>
      <span class="g"><span class="dot"></span>full lineage</span>
      <span class="a"><span class="dot"></span>root only</span>
      <span class="r"><span class="dot"></span>miss</span>
      <span>green dot = tests pass</span>
    </div>
    <div class="ticker"></div>
  </section>
</main>
<div id="scoreboard"></div>
<div id="foot"></div>
<script>
const $ = (s, r) => (r || document).querySelector(s);
const arm = n => $("#arm-" + n);
const esc = s => String(s).replace(/[&<>]/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const SVG = "http://www.w3.org/2000/svg";
function status(text) { $("#status").textContent = text; }
const WHO = {stock: "STOCK", agentfork: "AGENTFORK"};
const HITSAID = {subtree: "reused whole lineage", root: "kept root only",
                 miss: "re-prefilled from scratch"};
const trees = {
  stock: {root: null, nodes: [], byId: {}, killed: false,
          seen: new Set(), reaped: new Set()},
  agentfork: {root: null, nodes: [], byId: {}, killed: false,
              seen: new Set(), reaped: new Set()},
};
const metrics = {
  stock: {prefill: 0, root: 0, subtree: 0, total: 0, verified: false},
  agentfork: {prefill: 0, root: 0, subtree: 0, total: 0, verified: false},
};
const FILL = {subtree: "#f0fdf4", root: "#fffbeb", miss: "#fef2f2"};
const STROKE = {subtree: "#86efac", root: "#fcd34d", miss: "#fecaca"};
const EDGE = {subtree: "#22c55e", root: "#f59e0b", miss: "#f87171"};
const INK = {subtree: "#15803d", root: "#b45309", miss: "#dc2626"};
const NODE_W = {1: 120, 2: 58, 3: 58};
const NODE_H = {1: 32, 2: 26, 3: 26};
const GAP = {1: 24, 2: 16, 3: 16};
const ROOT_Y = 28;
const LEVEL_H = 60;
const PAD_Y = 16;
const M = 24;
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
  const children = {};
  for (const n of t.nodes) {
    const p = n.parent || "root";
    if (!children[p]) children[p] = [];
    children[p].push(n);
  }
  if (!children["root"]) children["root"] = [];
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
  const rootW = children["root"].length
    ? (function() {
        let w = -GAP[1];
        for (const c of children["root"]) w += GAP[1] + measure(c);
        return Math.max(200, w);
      })()
    : 200;

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
  const W = Math.max(320, (wrap.clientWidth || 360) - 24);
  const {x, W: W2, H, levelY, rootAt} = layout(t, W);
  svg.style.width = "100%";
  svg.style.minWidth = W2 + "px";
  svg.style.height = "auto";
  svg.setAttribute("viewBox", "0 0 " + W2 + " " + H);
  svg.setAttribute("preserveAspectRatio", "xMidYMin meet");
  if (t.root === null) {
    svg.appendChild(el("text", {x: W2 / 2, y: H / 2, fill: "#94a3b8",
      "text-anchor": "middle", "font-size": 13}, "waiting..."));
    return;
  }
  const rw = Math.min(200, W2 - 2 * M);
  svg.appendChild(el("rect", {x: rootAt.x - rw / 2, y: 6, width: rw,
    height: 30, rx: 15, fill: "#dbeafe", stroke: "#93c5fd"}));
  svg.appendChild(el("text", {x: rootAt.x, y: 20, "text-anchor": "middle",
    "font-size": 11, "font-weight": 700, fill: "#1d4ed8"}, "SHARED CONTEXT"));
  svg.appendChild(el("text", {x: rootAt.x, y: 30, "text-anchor": "middle",
    "font-size": 9, fill: "#64748b"},
    t.root.charged.toLocaleString() + " tok"));

  for (const n of t.nodes) {
    const cx = x[n.id], y = levelY[n.depth];
    if (cx === undefined) continue;
    const dead = !alive(t, n);
    const from = t.byId[n.parent]
      ? {x: x[n.parent], y: levelY[t.byId[n.parent].depth] +
         NODE_H[t.byId[n.parent].depth] / 2}
      : rootAt;
    const g = el("g", {});
    if (!t.seen.has(n.id)) { g.setAttribute("class", "pop"); t.seen.add(n.id); }
    else if (dead && !t.reaped.has(n.id)) {
      g.setAttribute("class", "reap"); t.reaped.add(n.id);
    }
    const w = NODE_W[n.depth], h = NODE_H[n.depth];
    const r = n.depth === 1 ? 10 : 7;
    g.appendChild(el("path", {
      d: `M${from.x},${from.y} C${from.x},${from.y + 16} ${cx},${y - 16} ${cx},${y - h / 2}`,
      fill: "none", "stroke-width": n.depth === 1 ? 1.8 : 1.2,
      stroke: dead ? "#cbd5e1" : EDGE[n.hit],
      "stroke-dasharray": dead ? "3 2" : "none"}));
    g.appendChild(el("rect", {x: cx - w / 2, y: y - h / 2, width: w,
      height: h, rx: r,
      fill: dead ? "#f8fafc" : FILL[n.hit],
      stroke: dead ? "#e2e8f0" : STROKE[n.hit], "stroke-width": 1.5}));
    const label = n.depth === 1 ? (n.plan || "").slice(0, 14) : ("+" + n.charged);
    const sub = n.depth === 1 ? (n.passed ? "PASS" : "...") : "";
    g.appendChild(el("text", {x: cx, y: y - h / 2 + 13, "text-anchor": "middle",
      "font-size": n.depth === 1 ? 10 : 11, "font-weight": 600,
      fill: dead ? "#94a3b8" : INK[n.hit]}, label));
    if (sub) {
      g.appendChild(el("text", {x: cx, y: y + h / 2 - 5, "text-anchor": "middle",
        "font-size": 8, fill: dead ? "#cbd5e1" : "#64748b"}, sub));
    }
    if (n.depth > 1) {
      const dotFill = dead ? "#cbd5e1" : (n.passed ? "#22c55e" : "#ef4444");
      g.appendChild(el("circle", {cx: cx + w / 2 - 5, cy: y - h / 2 + 5, r: 3.5,
        fill: dotFill}));
    }
    svg.appendChild(g);
  }
  const label = t.killed ? "subtrees reaped" : "fan-out in progress";
  svg.appendChild(el("text", {x: W2 - 10, y: 12, "text-anchor": "end",
    "font-size": 9, fill: "#94a3b8"}, label));
}
function redraw(name) { requestAnimationFrame(() => drawTree(name)); }
function updateMetrics(name) {
  const m = metrics[name];
  const rootPct = m.total ? Math.round(100 * m.root / m.total) : 0;
  const subPct = m.total ? Math.round(100 * m.subtree / m.total) : 0;
  $(`#m-${name}-prefill`).textContent = m.prefill.toLocaleString();
  $(`#m-${name}-subtree`).textContent = subPct + "%";
  $(`#m-${name}-root`).textContent = rootPct + "%";
}
function kv(name, used, cap) {
  const pct = cap ? Math.min(100, 100 * used / cap) : 0;
  $(`#kv-${name}`).style.width = pct + "%";
}
function tick(name, d) {
  const ticker = $(".ticker", arm(name));
  const row = document.createElement("div");
  row.className = "ticker-row";
  const tag = {subtree: "HIT", root: "ROOT", miss: "MISS"}[d.hit_level];
  row.innerHTML =
    `<span class="dim">${esc(d.label)} ${d.idx + 1}/${d.total}</span>` +
    `<span class="tag" style="color:${INK[d.hit_level]}">${tag}</span>` +
    `<span class="dim">+${d.charged.toLocaleString()}</span>` +
    (d.passed ? `<span style="color:var(--green);font-weight:700">PASS</span>` : "");
  ticker.appendChild(row);
  ticker.scrollTop = ticker.scrollHeight;
  while (ticker.children.length > 5) ticker.removeChild(ticker.firstChild);
}
function setArmStatus(name, verified) {
  const url = $(".url", arm(name));
  if (!url.textContent.includes("verified"))
    url.textContent += verified ? " · verified" : " · unverified";
}
const handlers = {
  banner(d) {},
  arm_start(d) {
    $(".url", arm(d.name)).textContent = d.url;
    status(WHO[d.name] + " started");
  },
  parent(d) {
    $(".url", arm(d.name)).textContent += ` · ${d.charged.toLocaleString()} tok`;
    kv(d.name, d.charged, d.capacity);
    trees[d.name].root = {charged: d.charged};
    redraw(d.name);
    status(WHO[d.name] + " loaded shared context");
  },
  child(d) {
    const t = trees[d.name], m = metrics[d.name];
    const node = {id: d.branch_id, parent: d.parent_branch, depth: d.depth,
                  hit: d.hit_level, charged: d.charged, passed: d.passed,
                  plan: d.plan || ""};
    t.nodes.push(node);
    t.byId[node.id] = node;
    m.prefill += d.charged;
    m.total++;
    if (d.hit_level === "subtree") { m.subtree++; m.root++; }
    else if (d.hit_level === "root") m.root++;
    updateMetrics(d.name);
    redraw(d.name);
    tick(d.name, d);
    status(WHO[d.name] + " " + d.label + " " + (d.idx + 1) + "/" + d.total +
           ": " + HITSAID[d.hit_level]);
  },
  kv(d) { kv(d.name, d.used, d.capacity); },
  kills(d) {
    trees[d.name].killed = true;
    redraw(d.name);
    status(WHO[d.name] + " reaped " + d.n + " losers");
  },
  arm_done(d) {
    const m = metrics[d.name];
    m.verified = d.verified;
    updateMetrics(d.name);
    setArmStatus(d.name, d.verified);
    $(`#m-${d.name}-prefill`).classList.add(d.verified ? "win" : "lose");
    redraw(d.name);
    status(WHO[d.name] + " done — prefill " + d.prefill.toLocaleString() +
           (d.verified ? " (verified)" : ""));
  },
  scoreboard(d) {
    const sb = $("#scoreboard");
    sb.classList.add("open");
    const winner = metrics.agentfork.prefill < metrics.stock.prefill
      ? "AGENTFORK" : "STOCK";
    const rows = d.rows.map(r =>
      "<tr" + (r[0].toLowerCase().includes("prefill") ? " class='winner'" : "") +
      "><td>" + esc(r[0]) + "</td><td>" + esc(r[1]) + "</td><td>" + esc(r[2]) + "</td></tr>"
    ).join("");
    sb.innerHTML =
      "<h3>" + winner + " wins on prefill tokens</h3>" +
      "<table><thead><tr><th>metric</th><th>STOCK</th><th>AGENTFORK</th></tr></thead><tbody>" +
      rows + "</tbody></table>";
    $("#foot").innerHTML =
      "<details><summary>honest caveats</summary>" +
      esc(d.notes).replace(/\\n/g, "<br>") + "</details>";
    redraw("stock"); redraw("agentfork");
    status(winner + " wins — see scoreboard below");
  },
};
handlers.started = () => {
  $("#startbtn").disabled = true;
  $("#startbtn").textContent = "racing...";
  status("Race running...");
};
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
