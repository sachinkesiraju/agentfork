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
 body { margin:0; padding:28px; background:var(--bg); color:var(--ink);
        font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",
             Roboto,Inter,sans-serif; }
 code, .mono, td.num { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
 h1 { font-size:20px; margin:0 0 4px; letter-spacing:-.3px; }
 .sub { color:var(--mute); white-space:pre-wrap; margin-bottom:10px;
        font-size:12.5px; }
 .card { background:var(--panel); border:1px solid var(--line);
         border-radius:12px; box-shadow:0 1px 2px rgba(15,23,42,.06); }
 .math { margin:14px 0 20px; padding:12px 14px; font-size:13px; }
 .math b { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
 .math span { color:var(--mute); }
 .warn { color:var(--amber); }
 #gate { margin:0 0 18px; padding:14px 16px; display:flex; gap:14px;
         align-items:center; }
 #gate.hidden { display:none; }
 #startbtn { font:600 14px/1 ui-sans-serif,system-ui,sans-serif; cursor:pointer;
             color:#fff; background:var(--blue); border:none; border-radius:8px;
             padding:11px 18px; }
 #startbtn:hover { background:#1d4ed8; }
 #startbtn[disabled] { background:#93c5fd; cursor:default; }
 .gatehint { color:var(--mute); font-size:12.5px; max-width:640px; }
 @keyframes pop { from { opacity:0; transform:translateY(-10px); }
                  to { opacity:1; transform:none; } }
 .tree g.pop { animation:pop .5s ease-out both; }
 @keyframes reap { from { opacity:1; } 40% { opacity:.15; } to { opacity:1; } }
 .tree g.reap { animation:reap .9s ease-out both; }
 tr.pop { animation:pop .4s ease-out both; }
 .arms { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
 @media (max-width:1000px) { .arms { grid-template-columns:1fr; } }
 .arm { padding:16px; }
 .arm h2 { font-size:14px; margin:0 0 2px; letter-spacing:.08em;
           color:var(--blue); }
 #arm-stock h2 { color:var(--mute); }
 .arm .url { color:var(--mute); font-size:12px; margin-bottom:12px; }
 .tree { width:100%; height:290px; display:block; margin-bottom:6px; }
 .kv { height:22px; background:#eef2f7; border-radius:5px; overflow:hidden;
       position:relative; margin-bottom:4px; border:1px solid var(--line); }
 .kv i { display:block; height:100%; width:0; background:#bfdbfe;
         transition:width .2s; }
 .kv b { position:absolute; inset:0; text-align:center; font-weight:600;
         font-size:11.5px; line-height:20px;
         font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
 .kvlabel { color:var(--mute); font-size:12px; margin-bottom:12px; }
 table { width:100%; border-collapse:collapse; font-size:12.5px; }
 td { padding:2px 4px; white-space:nowrap; }
 td.n { color:var(--mute); width:1%; }
 td.bar { width:78px; }
 .p { display:block; height:7px; background:#eef2f7; border-radius:4px; }
 .p i { display:block; height:100%; background:#c7b2f5; border-radius:4px; }
 .hit, .subtree { color:var(--green); font-weight:600; }
 .root { color:var(--amber); font-weight:600; }
 .miss { color:var(--red); font-weight:600; }
 .pass { color:var(--green); font-weight:600; }
.fail { color:var(--mute); }
.lbl { color:var(--mute); font-weight:400; }
.legend { color:var(--mute); font-size:11.5px; line-height:1.5;
          margin-bottom:12px; }
.legend b { font-weight:600; }
.legend .g { color:var(--green); } .legend .r { color:var(--red); }
.legend .a { color:var(--amber); }
 .ev { margin-top:10px; color:var(--amber); font-size:12.5px; }
 .done { margin-top:6px; font-size:13px; }
 #score { margin-top:24px; padding:14px 16px; max-width:820px; }
 #score td { border-top:1px solid var(--line); padding:5px 4px;
             font-size:13px; }
 #score tr:first-child td { border-top:none; }
 #score td.k { color:var(--mute); }
 #score td.v { text-align:right;
               font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
 .foot { color:var(--mute); margin-top:12px; white-space:pre-wrap;
         font-size:12px; max-width:860px; }
</style></head><body>
<h1>agentfork: 10 fixes, 1 inference server, and someone else is using it too</h1>
<div class="sub" id="header"></div>
<div class="math card" id="math"></div>
<div id="gate" class="card">
  <button id="startbtn">&#9654;&nbsp; Start the race</button>
  <div class="gatehint">Both arms will run the same fan-out against their own
  fresh server while the noisy neighbour streams through it. Watch the trees
  grow: every branch animates in as the cache resolves it.</div>
</div>
<div class="arms">
  <div class="arm card" id="arm-stock"><h2>STOCK</h2><div class="url"></div>
    <svg class="tree" viewBox="0 0 460 290" preserveAspectRatio="xMidYMid meet"></svg>
    <div class="legend"></div>
    <div class="kv"><i></i><b></b></div><div class="kvlabel"></div>
    <table></table><div class="ev"></div><div class="done"></div></div>
  <div class="arm card" id="arm-agentfork"><h2>AGENTFORK</h2><div class="url"></div>
    <svg class="tree" viewBox="0 0 460 290" preserveAspectRatio="xMidYMid meet"></svg>
    <div class="legend"></div>
    <div class="kv"><i></i><b></b></div><div class="kvlabel"></div>
    <table></table><div class="ev"></div><div class="done"></div></div>
</div>
<div id="score" class="card"></div>
<div class="foot" id="foot"></div>
<script>
const $ = (s, r) => (r || document).querySelector(s);
const arm = n => $("#arm-" + n);
const esc = s => String(s).replace(/[&<>]/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const SVG = "http://www.w3.org/2000/svg";

// ---- live branch tree (same visual language as docs/img/lifecycle.svg) ----
// The tree is root context -> approach -> candidate -> verification, so a node
// inherits everything its whole lineage committed, not just the root prefix:
//   green  HIT   the branch reused its parent's full prefix (root + lineage)
//   amber  ROOT  it kept the shared root context but had to re-prefill the
//                approach reasoning above it -- a partial subtree hit
//   red    MISS  nothing was left; it re-prefilled from scratch
// Killed losers grey out with dashed edges; the surviving lineage stays lit.
const trees = {
  stock: {root: null, nodes: [], byId: {}, killed: false,
          seen: new Set(), reaped: new Set()},
  agentfork: {root: null, nodes: [], byId: {}, killed: false,
              seen: new Set(), reaped: new Set()},
};

const LEVEL_Y = {1: 108, 2: 186, 3: 258};
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

// A node is dead once the kills happened and it is not on the winning lineage.
function alive(t, n) {
  if (!t.killed) return true;
  for (let cur = n; cur; cur = t.byId[cur.parent]) if (!cur.passed) return false;
  return true;
}

function layout(t, W) {
  // deepest level first: spread it evenly, then centre each parent over its
  // own children so the subtree structure is what you actually see.
  const depths = [3, 2, 1];
  const x = {};
  for (const d of depths) {
    const row = t.nodes.filter(n => n.depth === d);
    if (!row.length) continue;
    const placed = row.filter(n => x[n.id] === undefined);
    // siblings sit together: half a slot of air between subtrees, so the
    // grouping under each approach is visible and not just implied by edges
    const parents = [];
    for (const n of placed) if (!parents.includes(n.parent)) parents.push(n.parent);
    const gap = 0.5 * (parents.length - 1);
    const step = Math.min(46, (W - 44) / Math.max(placed.length + gap, 8));
    const span = step * (placed.length + gap);
    // verification forks hang under their own parent; a full fan-out is centred
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
    // ... and every parent recentres over the children it actually has
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
  const W = 460, RY = 34;
  if (t.root === null) {
    svg.appendChild(el("text", {x: 14, y: 22, fill: "#94a3b8",
      "font-size": 12}, "waiting for the shared context..."));
    return;
  }
  const {x} = layout(t, W);
  // root: the shared repo context every branch inherits
  const rw = 168, rx = (W - rw) / 2;
  svg.appendChild(el("rect", {x: rx, y: RY - 20, width: rw, height: 40, rx: 10,
    fill: "#eff6ff", stroke: "#bfdbfe"}));
  svg.appendChild(el("text", {x: rx + 12, y: RY - 4, "font-size": 10,
    "font-weight": 600, fill: "#2563eb"}, "ROOT CONTEXT"));
  svg.appendChild(el("text", {x: rx + 12, y: RY + 12, "font-size": 11,
    fill: "#0f172a", "font-family": "ui-monospace,Menlo,monospace"},
    t.root.charged.toLocaleString() + " tok shared prefix"));
  const rootAt = {x: W / 2, y: RY + 20};

  for (const n of t.nodes) {
    const cx = x[n.id], y = LEVEL_Y[n.depth];
    if (cx === undefined) continue;
    const dead = !alive(t, n);
    const from = t.byId[n.parent]
      ? {x: x[n.parent], y: LEVEL_Y[t.byId[n.parent].depth] + 14} : rootAt;
    const g = el("g", {});
    // the full redraw would restart CSS animations, so each node animates
    // exactly once: a "pop" when it first appears, a "reap" when it dies
    if (!t.seen.has(n.id)) { g.setAttribute("class", "pop"); t.seen.add(n.id); }
    else if (dead && !t.reaped.has(n.id)) {
      g.setAttribute("class", "reap"); t.reaped.add(n.id);
    }
    g.appendChild(el("path", {
      d: `M${from.x},${from.y} C${from.x},${from.y + 26} ${cx},${y - 34} ${cx},${y - 14}`,
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
  svg.appendChild(el("text", {x: W - 14, y: 16, "text-anchor": "end",
    "font-size": 10, fill: "#94a3b8"}, label));
}

function kv(name, used, cap) {
  const a = arm(name), pct = cap ? Math.min(100, 100 * used / cap) : 0;
  $(".kv i", a).style.width = pct + "%";
  $(".kv b", a).textContent = used.toLocaleString() + " / " + cap.toLocaleString();
  $(".kvlabel", a).textContent = "KV pool used";
}

const handlers = {
  banner(d) {
    $("#header").textContent = d.header;
    const w = d.regime === "above"
      ? "U &gt; U*  &rarr; the neighbour evicts an unpinned prefix"
      : "U &le; U*  &rarr; the prefix survives in both arms";
    $("#math").innerHTML =
      "<b>P</b> <span>shared prefix</span> " + d.P + " &nbsp; " +
      "<b>C</b> <span>KV capacity</span> " + d.C + " &nbsp; " +
      "<b>U</b> <span>neighbour per gap</span> " + d.U + " &nbsp; " +
      "<b>U*</b> <span>= C - P</span> " + d.Ustar +
      "<br><span class='warn'>" + w + "</span><br>" +
      "<span>" + d.N + " candidates over " + d.approaches +
      " approach branches of " + d.approach_tokens + " tok, " + d.verify +
      " verification forks; the neighbour is metered between branches</span>";
    document.querySelectorAll(".legend").forEach(n => n.innerHTML =
      "tree: <b>root context</b> \u2192 <b>approach</b> (its own committed " +
      "reasoning, L1) \u2192 <b>candidate</b> (L2) \u2192 " +
      "<b>verification</b> (L3)<br>" +
      "<b>cache</b> <span class='g'>HIT</span> = reused its parent's whole " +
      "lineage &nbsp;\u00b7&nbsp; <span class='a'>ROOT</span> = kept the " +
      "shared context but re-prefilled the approach above it " +
      "&nbsp;\u00b7&nbsp; <span class='r'>MISS</span> = re-prefilled " +
      "everything (<b>+n</b> = tokens charged)<br>" +
      "<b>tests</b> <span class='g'>PASS</span>/FAIL = did that candidate's " +
      "patch pass its pytest check (green dot on the node) &nbsp;\u00b7&nbsp; " +
      "grey + dashed = branch killed, whole subtree reaped");
    drawTree("stock"); drawTree("agentfork");
  },
  arm_start(d) { $(".url", arm(d.name)).textContent = d.url; },
  parent(d) {
    $(".url", arm(d.name)).textContent +=
      "  \u00b7  shared context: " + d.charged + " tok charged (P=" + d.P + ")";
    kv(d.name, d.charged, d.capacity);
    trees[d.name].root = {charged: d.charged};
    drawTree(d.name);
  },
  child(d) {
    const t = trees[d.name];
    const node = {id: d.branch_id, parent: d.parent_branch, depth: d.depth,
                  hit: d.hit_level, charged: d.charged, passed: d.passed,
                  plan: d.plan || ""};
    t.nodes.push(node);
    t.byId[node.id] = node;
    drawTree(d.name);
    const row = document.createElement("tr");
    row.className = "pop";
    row.innerHTML =
      "<td class='n'>" + esc(d.label) + " " + (d.idx + 1) + "/" + d.total +
        "<span class='lbl'> L" + d.depth + "</span></td>" +
      "<td class='bar'><span class='p'><i style='width:" +
        (100 * (d.idx + 1) / d.total) + "%'></i></span></td>" +
      "<td class='" + d.hit_level + "'>" +
        "<span class='lbl'>cache </span>" +
        ({subtree: "HIT", root: "ROOT", miss: "MISS"}[d.hit_level]) + "</td>" +
      "<td class='n num'>cached " + d.cached.toLocaleString() + "</td>" +
      "<td class='num'>charged " + d.charged.toLocaleString() + "</td>" +
      "<td class='" + (d.passed ? "pass" : "fail") + "'>" +
        "<span class='lbl'>tests </span>" +
        (d.passed ? "PASS" : "FAIL") + "</td>";
    $("table", arm(d.name)).appendChild(row);
  },
  kv(d) { kv(d.name, d.used, d.capacity); },
  kills(d) {
    trees[d.name].killed = true;
    drawTree(d.name);
    $(".ev", arm(d.name)).textContent =
      "killed " + d.n + " losers: KV freed " + d.freed.toLocaleString() +
      " tok (pool " + (d.pool_delta > 0 ? "-" : "") + d.pool_delta.toLocaleString() + ")";
  },
  arm_done(d) {
    $(".done", arm(d.name)).innerHTML =
      "<span class='" + (d.hit_rate >= 1 ? "hit" : "miss") + "'>root hit " +
      Math.round(100 * d.hit_rate) + "%</span> &nbsp; " +
      "<span class='" + (d.subtree_rate >= 1 ? "hit" : "miss") +
      "'>lineage hit " + Math.round(100 * d.subtree_rate) + "%</span>" +
      " &nbsp; prefill " +
      d.prefill.toLocaleString() + " tok &nbsp; " +
      (d.verified ? "<span class='pass'>VERIFIED</span>" : "UNVERIFIED");
  },
  scoreboard(d) {
    $("#score").innerHTML = "<table>" + d.rows.map(r =>
      "<tr><td class='k'>" + esc(r[0]) + "</td><td class='v'>" + esc(r[1]) +
      "</td><td class='v'>" + esc(r[2]) + "</td></tr>").join("") + "</table>";
    $("#foot").textContent = d.notes;
  },
};

handlers.started = () => { $("#gate").classList.add("hidden"); };

$("#startbtn").onclick = () => {
  $("#startbtn").disabled = true;
  $("#startbtn").textContent = "racing...";
  fetch("/start", {method: "POST"});
};

// The server streams events as fast as the race produces them (bursts);
// pacing the visual ones a beat apart is what makes the trees animate --
// including for a browser that connects late and gets the replay.
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
        self.inner = inner          # a LogUI, so the terminal still shows progress
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
        """Block until the browser presses Start (or ``--autostart``)."""
        if autostart:
            self.start.set()
        elif not self.start.is_set():
            self.inner.say("waiting for Start in the browser...")
        self.start.wait()
        self._emit(kind="started")

    # -- pub/sub -----------------------------------------------------------

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

    # -- race UI interface -------------------------------------------------

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
