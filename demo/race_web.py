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
<title>agentfork: split-screen race</title>
<style>
 :root { color-scheme: light; --ink:#0f172a; --mute:#64748b; --line:#e2e8f0;
         --bg:#f8fafc; --panel:#ffffff; --blue:#2563eb; --green:#16a34a;
         --red:#dc2626; --amber:#b45309; }
 * { box-sizing: border-box; }
 html, body { height: 100vh; margin: 0; overflow: hidden; }
 body { display: flex; flex-direction: column; padding: 10px 12px; gap: 6px;
        background: var(--bg); color: var(--ink);
        font: 12px/1.4 ui-sans-serif,system-ui,-apple-system,"Segoe UI",
             Roboto,Inter,sans-serif; }
 code, .mono, td.num { font-family: ui-monospace,SFMono-Regular,Menlo,monospace; }
 h1 { font-size: 16px; margin: 0 0 2px; letter-spacing: -.3px; }
 .intro { font-size: 11.5px; max-width: 980px; color: #334155;
          line-height: 1.45; margin-bottom: 4px; }
 .intro b { color: var(--blue); }
 .top { flex: none; }
 .sub { color: var(--mute); white-space: pre-wrap; font-size: 10.5px;
        margin-bottom: 3px; }
 .status { font-size: 11px; color: #0f172a; min-height: 1.3em;
           font-weight: 500; }
 .card { background: var(--panel); border: 1px solid var(--line);
         border-radius: 10px; box-shadow: 0 1px 2px rgba(15,23,42,.06); }
 .math { padding: 6px 8px; font-size: 11px; }
 .math b { font-family: ui-monospace,SFMono-Regular,Menlo,monospace; }
 .math span { color: var(--mute); }
 .warn { color: var(--amber); font-weight: 600; }
 #gate { margin: 0 0 4px; padding: 6px 8px; display: flex; gap: 8px;
         align-items: center; }
 #gate.hidden { display: none; }
 #startbtn { font: 600 12px/1 ui-sans-serif,system-ui,sans-serif;
             cursor: pointer; color: #fff; background: var(--blue);
             border: none; border-radius: 6px; padding: 8px 12px; }
 #startbtn:hover { background: #1d4ed8; }
 #startbtn[disabled] { background: #93c5fd; cursor: default; }
 .gatehint { color: var(--mute); font-size: 10px; max-width: 640px; }
 .arms { flex: 1 1 0; display: grid; grid-template-columns: 1fr 1fr;
         grid-template-rows: minmax(0, 1fr);
         gap: 10px; min-height: 0; overflow: hidden;
         margin-bottom: 4px; }
 @media (max-width: 1000px) { .arms { grid-template-columns: 1fr; } }
 .arm { padding: 8px; display: flex; flex-direction: column;
        gap: 4px; min-width: 0; min-height: 0; height: 100%; }
 .arm h2 { font-size: 11px; margin: 0; letter-spacing: .08em;
           color: var(--blue); }
 #arm-stock h2 { color: var(--mute); }
 .arm .url { color: var(--mute); font-size: 10px; margin-bottom: 1px; }
 .tree-wrap { flex: 1 1 0; min-height: 50px; position: relative; }
 .tree { position: absolute; inset: 0; width: 100%; height: 100%;
         display: block; }
 .legend { color: var(--mute); font-size: 9.5px; line-height: 1.35; }
 .legend b { font-weight: 600; }
 .legend .g { color: var(--green); } .legend .r { color: var(--red); }
 .legend .a { color: var(--amber); }
 .kv { height: 14px; background: #eef2f7; border-radius: 4px;
       overflow: hidden; position: relative; border: 1px solid var(--line); }
 .kv i { display: block; height: 100%; width: 0; background: #bfdbfe;
         transition: width .2s; }
 .kv b { position: absolute; inset: 0; text-align: center; font-weight: 600;
         font-size: 9.5px; line-height: 12px;
         font-family: ui-monospace,SFMono-Regular,Menlo,monospace; }
 .kvlabel { color: var(--mute); font-size: 10px; }
 .statbar { display: flex; flex-wrap: wrap; gap: 4px; font-size: 10px;
            margin-top: 1px; }
 .statbar span { background: #f1f5f9; padding: 1px 5px; border-radius: 4px; }
 .ticker { height: 44px; overflow: hidden; flex: none;
           border-top: 1px solid var(--line); padding-top: 2px; }
 .ticker > div { font-size: 10.5px; padding: 1px 0;
                 display: flex; gap: 8px; color: #334155; }
 .ticker .n { color: var(--mute); flex: none; }
 .ticker .hl { flex: none; font-weight: 600; }
 .ticker .ch { flex: none; font-family: ui-monospace,Menlo,monospace; }
 .ev { color: var(--amber); font-size: 10.5px; margin-top: 1px; }
 .done { font-size: 10.5px; margin-top: 1px; }
 #score { flex: none; max-height: 150px; padding: 5px 10px; overflow-y: auto; overflow-x: hidden; }
 #score.hidden { display: none; }
 #score table { width: 100%; border-collapse: collapse; font-size: 9.5px;
                line-height: 1.25; }
 #score td { padding: 1px 3px; border-top: 1px solid var(--line); }
 #score tr:first-child td { border-top: none; }
 #score td.k { color: var(--mute); width: 48%; }
 #score td.v { text-align: right; width: 26%;
               font-family: ui-monospace,SFMono-Regular,Menlo,monospace; }
 #score th { text-align: right; font-size: 9px; color: var(--mute);
             padding: 0 3px 1px; font-weight: 600; }
 #score th:first-child { text-align: left; }
 .foot { color: var(--mute); font-size: 9px; line-height: 1.35;
         max-width: 980px; }
 .foot details { margin: 0; }
 .foot summary { cursor: pointer; color: var(--blue); font-weight: 600; }
 @keyframes pop { from { opacity: 0; transform: translateY(-6px); }
                  to { opacity: 1; transform: none; } }
 .tree g.pop { animation: pop .4s ease-out both; }
 @keyframes reap { from { opacity: 1; } 40% { opacity: .12; } to { opacity: 1; } }
 .tree g.reap { animation: reap .8s ease-out both; }
 .ticker-row { animation: pop .35s ease-out both; }
</style></head><body>
<div class="top">
<h1>agentfork: 10 fixes, 2 simultaneous arms, and someone else is using it too</h1>
<div class="intro">Two identical agents race side by side to fix the same bug
by trying <b>10 candidate patches at once</b>. Each arm runs on its own fresh
LLM server while an unrelated tenant hammers that server's KV cache.
<b>STOCK</b> (left) just hopes its context survives; <b>AGENTFORK</b> (right)
pins its branch tree so it cannot be evicted. Same bug, same load, same noise.</div>
<div class="sub" id="header"></div>
<div class="math card" id="math"></div>
<div id="gate" class="card">
  <button id="startbtn">&#9654;&nbsp; Start the race</button>
  <div class="gatehint">Both arms race side by side against identical fresh
  servers, each under the same noisy-neighbour load. Watch the trees grow:
  every branch animates in as the cache resolves it.</div>
</div>
<div class="status" id="status">Press Start to begin the race.</div>
</div>
<div class="arms">
  <div class="arm card" id="arm-stock">
    <h2>STOCK</h2><div class="url"></div>
    <div class="tree-wrap"><svg class="tree" viewBox="0 0 460 220" preserveAspectRatio="xMidYMid meet"></svg></div>
    <div class="legend"></div>
    <div class="kv"><i></i><b></b></div><div class="kvlabel">KV pool used</div>
    <div class="statbar"><span>root hit 0%</span><span>lineage hit 0%</span><span>prefill 0</span></div>
    <div class="ticker"></div>
    <div class="ev"></div><div class="done"></div>
  </div>
  <div class="arm card" id="arm-agentfork">
    <h2>AGENTFORK</h2><div class="url"></div>
    <div class="tree-wrap"><svg class="tree" viewBox="0 0 460 220" preserveAspectRatio="xMidYMid meet"></svg></div>
    <div class="legend"></div>
    <div class="kv"><i></i><b></b></div><div class="kvlabel">KV pool used</div>
    <div class="statbar"><span>root hit 0%</span><span>lineage hit 0%</span><span>prefill 0</span></div>
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
  subtree: "reused its whole lineage, paid only the suffix",
  root: "kept the shared context but re-paid its approach",
  miss: "found nothing in cache, re-prefilled from scratch",
};
const trees = {
  stock: {root: null, nodes: [], byId: {}, killed: false,
          seen: new Set(), reaped: new Set()},
  agentfork: {root: null, nodes: [], byId: {}, killed: false,
              seen: new Set(), reaped: new Set()},
};
const LEVEL_Y = {1: 84, 2: 142, 3: 198};
const FILL = {subtree: "#f0fdf4", root: "#fffbeb", miss: "#fef2f2"};
const STROKE = {subtree: "#bbf7d0", root: "#fde68a", miss: "#fecaca"};
const EDGE = {subtree: "#86efac", root: "#fcd34d", miss: "#fecaca"};
const INK = {subtree: "#16a34a", root: "#b45309", miss: "#dc2626"};
const TAG = {subtree: "HIT", root: "ROOT", miss: "MISS"};
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
  const depths = [3, 2, 1];
  const x = {};
  for (const d of depths) {
    const row = t.nodes.filter(n => n.depth === d);
    if (!row.length) continue;
    const placed = row.filter(n => x[n.id] === undefined);
    const parents = [];
    for (const n of placed) if (!parents.includes(n.parent)) parents.push(n.parent);
    const gap = 0.5 * (parents.length - 1);
    const step = Math.min(46, (W - 44) / Math.max(placed.length + gap, 8));
    const span = step * (placed.length + gap);
    const anchor = placed.length && x[placed[0].parent] !== undefined
      ? x[placed[0].parent] : W / 2;
    const start = d === 3
      ? Math.max(24, Math.min(W - 24 - span, anchor - span / 2))
      : 22 + (W - 44 - span) / 2;
    let slot = 0, prev = null;
    for (const n of placed) {
      if (prev !== null && n.parent !== prev) slot += 0.5;
      x[n.id] = start + step * slot + step / 2;
      prev = n.parent;
      slot += 1;
    }
    for (const n of t.nodes.filter(m => m.depth === d - 1)) {
      const kids = t.nodes.filter(m => m.parent === n.id).map(m => x[m.id])
        .filter(v => v !== undefined);
      if (kids.length) x[n.id] = kids.reduce((a, b) => a + b, 0) / kids.length;
    }
  }
  return {x: x, step: Math.min(46, (W - 44) / 8)};
}
function drawTree(name) {
  const t = trees[name], svg = $(".tree", arm(name));
  svg.textContent = "";
  const W = 460;
  if (t.root === null) {
    svg.appendChild(el("text", {x: 14, y: 20, fill: "#94a3b8",
      "font-size": 12}, "waiting for the shared context..."));
    return;
  }
  const {x} = layout(t, W);
  const rw = 168, rx = (W - rw) / 2, RY = 22;
  svg.appendChild(el("rect", {x: rx, y: RY - 18, width: rw, height: 36, rx: 10,
    fill: "#eff6ff", stroke: "#bfdbfe"}));
  svg.appendChild(el("text", {x: rx + 12, y: RY - 4, "font-size": 10,
    "font-weight": 600, fill: "#2563eb"}, "ROOT CONTEXT"));
  svg.appendChild(el("text", {x: rx + 12, y: RY + 10, "font-size": 11,
    fill: "#0f172a", "font-family": "ui-monospace,Menlo,monospace"},
    t.root.charged.toLocaleString() + " tok shared prefix"));
  const rootAt = {x: W / 2, y: RY + 18};
  for (const n of t.nodes) {
    const cx = x[n.id], y = LEVEL_Y[n.depth];
    if (cx === undefined) continue;
    const dead = !alive(t, n);
    const from = t.byId[n.parent]
      ? {x: x[n.parent], y: LEVEL_Y[t.byId[n.parent].depth] + 14} : rootAt;
    const g = el("g", {});
    if (!t.seen.has(n.id)) { g.setAttribute("class", "pop"); t.seen.add(n.id); }
    else if (dead && !t.reaped.has(n.id)) {
      g.setAttribute("class", "reap"); t.reaped.add(n.id);
    }
    g.appendChild(el("path", {
      d: `M${from.x},${from.y} C${from.x},${from.y + 24} ${cx},${y - 30} ${cx},${y - 14}`,
      fill: "none", "stroke-width": n.depth === 1 ? 1.8 : 1.4,
      stroke: dead ? "#cbd5e1" : EDGE[n.hit],
      "stroke-dasharray": dead ? "3 3" : "none"}));
    const bw = n.depth === 1 ? 76 : 38;
    g.appendChild(el("rect", {x: cx - bw / 2, y: y - 14, width: bw,
      height: 28, rx: 8,
      fill: dead ? "#f1f5f9" : FILL[n.hit],
      stroke: dead ? "#e2e8f0" : STROKE[n.hit]}));
    g.appendChild(el("text", {x: cx, y: y - 1, "text-anchor": "middle",
      "font-size": n.depth === 1 ? 9 : 10, "font-weight": 600,
      fill: dead ? "#94a3b8" : INK[n.hit]},
      n.depth === 1 ? n.plan : TAG[n.hit]));
    g.appendChild(el("text", {x: cx, y: y + 10, "text-anchor": "middle",
      "font-size": 8, fill: dead ? "#cbd5e1" : "#64748b"},
      (n.depth === 1 ? TAG[n.hit] + " " : "") + "+" + n.charged));
    if (n.passed && n.depth > 1) {
      g.appendChild(el("circle", {cx: cx + bw / 2 - 4, cy: y - 14, r: 4,
        fill: dead ? "#cbd5e1" : "#16a34a"}));
    }
    svg.appendChild(g);
  }
  const label = t.killed
    ? (t.nodes.some(n => n.depth === 3)
        ? "losing subtrees reaped; winner re-forked to verify"
        : "losing candidates and their approach subtrees reaped")
    : "fan-out in progress";
  svg.appendChild(el("text", {x: W - 14, y: 14, "text-anchor": "end",
    "font-size": 10, fill: "#94a3b8"}, label));
}
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
    "<span>root hit " + hit + "%</span>" +
    "<span>lineage hit " + sub + "%</span>" +
    "<span>prefill " + pre + " tok</span>" +
    (d && d.verified ? "<span class='pass'>verified</span>" : "");
}
function tick(name, d) {
  const ticker = $(".ticker", arm(name));
  const row = document.createElement("div");
  row.className = "ticker-row";
  const tag = {subtree: "HIT", root: "ROOT", miss: "MISS"}[d.hit_level];
  row.innerHTML =
    "<span class='n'>" + esc(d.label) + " " + (d.idx + 1) + "/" + d.total + 
      " L" + d.depth + "</span>" +
    "<span class='hl " + d.hit_level + "'>cache " + tag + "</span>" +
    "<span class='ch'>+" + d.charged.toLocaleString() + " tok</span>" +
    "<span class='" + (d.passed ? "pass" : "fail") + "'>tests " +
      (d.passed ? "PASS" : "FAIL") + "</span>";
  ticker.appendChild(row);
  while (ticker.children.length > 3) ticker.removeChild(ticker.firstChild);
}
const handlers = {
  banner(d) {
    $("#header").textContent = d.header;
    const w = d.regime === "above"
      ? "U > U*  &rarr; the neighbour evicts an unpinned prefix"
      : "U &le; U*  &rarr; the prefix survives in both arms";
    $("#math").innerHTML =
      "<b>P</b> <span>shared prefix</span> " + d.P + " &nbsp; " +
      "<b>C</b> <span>KV capacity</span> " + d.C + " &nbsp; " +
      "<b>U</b> <span>neighbour per gap</span> " + d.U + " &nbsp; " +
      "<b>U*</b> <span>= C - P</span> " + d.Ustar +
      "&nbsp;&nbsp;<span class='warn'>" + w + "</span><br>" +
      "<span>" + d.N + " candidates over " + d.approaches +
      " approach branches of " + d.approach_tokens + " tok, " + d.verify +
      " verification forks; the neighbour is metered between branches</span>";
    document.querySelectorAll(".legend").forEach(n => n.innerHTML =
      "tree: <b>root context</b> &rarr; <b>approach</b> (committed reasoning, L1) &rarr; " +
      "<b>candidate</b> (L2) &rarr; <b>verification</b> (L3). " +
      "<b>cache</b> <span class='g'>HIT</span> = reused whole lineage; " +
      "<span class='a'>ROOT</span> = kept shared context but re-paid approach; " +
      "<span class='r'>MISS</span> = re-prefilled everything. " +
      "Grey + dashed = killed. Green dot = tests pass.");
    drawTree("stock"); drawTree("agentfork");
  },
  arm_start(d) {
    $(".url", arm(d.name)).textContent = d.url;
    status(WHO[d.name] + " starts on a fresh server" +
      (d.name === "stock" ? " with ordinary cache reuse" : " with pinned branches"));
  },
  parent(d) {
    $(".url", arm(d.name)).textContent +=
      " &middot; shared context: " + d.charged + " tok";
    kv(d.name, d.charged, d.capacity);
    trees[d.name].root = {charged: d.charged};
    drawTree(d.name);
    status(WHO[d.name] + " prefilled the shared repo context: " +
           d.charged.toLocaleString() + " tokens");
  },
  child(d) {
    const t = trees[d.name];
    const node = {id: d.branch_id, parent: d.parent_branch, depth: d.depth,
                  hit: d.hit_level, charged: d.charged, passed: d.passed,
                  plan: d.plan || ""};
    t.nodes.push(node);
    t.byId[node.id] = node;
    drawTree(d.name);
    tick(d.name, d);
    const what = d.label === "approach" ? "approach "
      : d.label === "verify" ? "verify " : "candidate ";
    status(WHO[d.name] + " " + what + (d.idx + 1) + "/" + d.total + " " +
           HITSAID[d.hit_level] + " (+" + d.charged.toLocaleString() + " tok)");
  },
  kv(d) { kv(d.name, d.used, d.capacity); },
  kills(d) {
    trees[d.name].killed = true;
    drawTree(d.name);
    $(".ev", arm(d.name)).textContent =
      "killed " + d.n + " losers: KV freed " + d.freed.toLocaleString() +
      " tok";
    status(WHO[d.name] + " killed " + d.n + " losers" +
      (d.freed > 0 ? ", reclaiming " + d.freed.toLocaleString() + " KV tokens" : ""));
  },
  arm_done(d) {
    updateStatbar(d.name, d);
    $(".done", arm(d.name)).innerHTML =
      (d.verified ? "<span class='pass'>VERIFIED</span>" : "UNVERIFIED") +
      " &nbsp; prefill " + d.prefill.toLocaleString() + " tok";
    status(WHO[d.name] + " done: root hit " + Math.round(100 * d.hit_rate) +
      "%, prefill " + d.prefill.toLocaleString() + " tokens");
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
    status("Race finished -- same verified fix, very different token bills.");
  },
};
handlers.started = () => { $("#gate").classList.add("hidden"); status("Race running..."); };
$("#startbtn").onclick = () => {
  $("#startbtn").disabled = true;
  $("#startbtn").textContent = "racing...";
  fetch("/start", {method: "POST"});
};
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

    def do_GET(self):  # noqa: N802 (stdlib naming)
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

    def do_POST(self):  # noqa: N802 (stdlib naming)
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
