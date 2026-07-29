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
 .arms { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
 @media (max-width:1000px) { .arms { grid-template-columns:1fr; } }
 .arm { padding:16px; }
 .arm h2 { font-size:14px; margin:0 0 2px; letter-spacing:.08em;
           color:var(--blue); }
 #arm-stock h2 { color:var(--mute); }
 .arm .url { color:var(--mute); font-size:12px; margin-bottom:12px; }
 .tree { width:100%; height:210px; display:block; margin-bottom:6px; }
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
 .hit { color:var(--green); font-weight:600; }
 .miss { color:var(--red); font-weight:600; }
 .pass { color:var(--green); font-weight:600; }
.fail { color:var(--mute); }
.lbl { color:var(--mute); font-weight:400; }
.legend { color:var(--mute); font-size:11.5px; line-height:1.5;
          margin-bottom:12px; }
.legend b { font-weight:600; }
.legend .g { color:var(--green); } .legend .r { color:var(--red); }
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
<div class="arms">
  <div class="arm card" id="arm-stock"><h2>STOCK</h2><div class="url"></div>
    <svg class="tree" viewBox="0 0 460 210" preserveAspectRatio="xMidYMid meet"></svg>
    <div class="legend"></div>
    <div class="kv"><i></i><b></b></div><div class="kvlabel"></div>
    <table></table><div class="ev"></div><div class="done"></div></div>
  <div class="arm card" id="arm-agentfork"><h2>AGENTFORK</h2><div class="url"></div>
    <svg class="tree" viewBox="0 0 460 210" preserveAspectRatio="xMidYMid meet"></svg>
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
// Parent holds the shared prefix; children fan out from it. A child is green
// when it hit that prefix, red when it had to re-prefill. Losers grey out and
// their edge goes dashed when they are killed; the winner keeps its
// verification forks.
const trees = {
  stock: {parent: null, children: [], verify: [], killed: false, winner: null},
  agentfork: {parent: null, children: [], verify: [], killed: false, winner: null},
};

function el(tag, attrs, text) {
  const n = document.createElementNS(SVG, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text !== undefined) n.textContent = text;
  return n;
}

function drawTree(name) {
  const t = trees[name], svg = $(".tree", arm(name));
  svg.textContent = "";
  const W = 460, PX = 60, PY = 34, CY = 112, VY = 182;
  if (t.parent === null) {
    svg.appendChild(el("text", {x: 14, y: 22, fill: "#94a3b8",
      "font-size": 12}, "waiting for the shared context..."));
    return;
  }
  // parent
  const pw = 150, px = PX - 12;
  svg.appendChild(el("rect", {x: px, y: PY - 20, width: pw, height: 40, rx: 10,
    fill: "#eff6ff", stroke: "#bfdbfe"}));
  svg.appendChild(el("text", {x: px + 12, y: PY - 4, "font-size": 10,
    "font-weight": 600, fill: "#2563eb"}, "PARENT"));
  svg.appendChild(el("text", {x: px + 12, y: PY + 12, "font-size": 11,
    fill: "#0f172a", "font-family": "ui-monospace,Menlo,monospace"},
    t.parent.charged.toLocaleString() + " tok prefix"));
  const anchorX = px + pw / 2, anchorY = PY + 20;

  const rows = [[t.children, CY, false], [t.verify, VY, true]];
  for (const [nodes, y, underWinner] of rows) {
    if (!nodes.length) continue;
    const step = Math.min(40, (W - 40) / Math.max(nodes.length, 10));
    const bw = Math.max(20, step - 8);
    // verification forks hang under the winner, not under the whole fan-out
    const startX = underWinner
      ? Math.max(20, Math.min(W - 20 - step * nodes.length,
                              winnerX(t, W) - step * nodes.length / 2))
      : 20 + (W - 40 - step * nodes.length) / 2;
    nodes.forEach((c, i) => {
      const x = startX + i * step, cx = x + bw / 2;
      const dead = t.killed && !c.passed;
      const from = y === VY ? {x: winnerX(t, W), y: CY + 14} :
                              {x: anchorX, y: anchorY};
      svg.appendChild(el("path", {
        d: `M${from.x},${from.y} C${from.x},${from.y + 24} ${cx},${y - 36} ${cx},${y - 14}`,
        fill: "none", "stroke-width": 1.4,
        stroke: dead ? "#cbd5e1" : (c.hit ? "#86efac" : "#fecaca"),
        "stroke-dasharray": dead ? "3 3" : "none"}));
      svg.appendChild(el("rect", {x: x, y: y - 14, width: bw, height: 28,
        rx: 8, fill: dead ? "#f1f5f9" : (c.hit ? "#f0fdf4" : "#fef2f2"),
        stroke: dead ? "#e2e8f0" : (c.hit ? "#bbf7d0" : "#fecaca")}));
      svg.appendChild(el("text", {x: cx, y: y + 1, "text-anchor": "middle",
        "font-size": 10, "font-weight": 600,
        fill: dead ? "#94a3b8" : (c.hit ? "#16a34a" : "#dc2626")},
        c.hit ? "HIT" : "MISS"));
      svg.appendChild(el("text", {x: cx, y: y + 11, "text-anchor": "middle",
        "font-size": 8, fill: dead ? "#cbd5e1" : "#64748b"},
        "+" + c.charged));
      if (c.passed) {
        svg.appendChild(el("circle", {cx: x + bw - 4, cy: y - 14, r: 4,
          fill: "#16a34a"}));
      }
    });
  }
  const label = t.killed
    ? (t.verify.length ? "winner re-forked for verification"
                       : "9 losers killed, winner survives")
    : "fan-out in progress";
  svg.appendChild(el("text", {x: W - 14, y: 16, "text-anchor": "end",
    "font-size": 10, fill: "#94a3b8"}, label));
}

function winnerX(t, W) {
  const i = t.children.findIndex(c => c.passed);
  if (i < 0) return W / 2;
  const step = Math.min(40, (W - 40) / Math.max(t.children.length, 10));
  const bw = Math.max(20, step - 8);
  const startX = 20 + (W - 40 - step * t.children.length) / 2;
  return startX + i * step + bw / 2;
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
      "<span>" + d.N + " candidates, " + d.verify +
      " verification forks; the neighbour is metered between candidates</span>";
    document.querySelectorAll(".legend").forEach(n => n.innerHTML =
      "<b>cache</b> <span class='g'>HIT</span> = this branch reused the " +
      "shared prefix already in the KV cache &nbsp;\u00b7&nbsp; " +
      "<span class='r'>MISS</span> = it was evicted, so the branch paid to " +
      "re-prefill it (<b>+n</b> = tokens charged)<br>" +
      "<b>tests</b> <span class='g'>PASS</span>/FAIL = did that candidate's " +
      "patch pass its pytest check (green dot on the node) &nbsp;\u00b7&nbsp; " +
      "grey + dashed = branch killed as a loser");
    drawTree("stock"); drawTree("agentfork");
  },
  arm_start(d) { $(".url", arm(d.name)).textContent = d.url; },
  parent(d) {
    $(".url", arm(d.name)).textContent +=
      "  \u00b7  shared context: " + d.charged + " tok charged (P=" + d.P + ")";
    kv(d.name, d.charged, d.capacity);
    trees[d.name].parent = {charged: d.charged};
    drawTree(d.name);
  },
  child(d) {
    const node = {hit: d.parent_hit, charged: d.charged, passed: d.passed};
    (d.label === "verify" ? trees[d.name].verify : trees[d.name].children)
      .push(node);
    drawTree(d.name);
    const row = document.createElement("tr");
    row.innerHTML =
      "<td class='n'>" + esc(d.label) + " " + (d.idx + 1) + "/" + d.total + "</td>" +
      "<td class='bar'><span class='p'><i style='width:" +
        (100 * (d.idx + 1) / d.total) + "%'></i></span></td>" +
      "<td class='" + (d.parent_hit ? "hit" : "miss") + "'>" +
        "<span class='lbl'>cache </span>" +
        (d.parent_hit ? "HIT" : "MISS") + "</td>" +
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
      "<span class='" + (d.hit_rate >= 1 ? "hit" : "miss") + "'>hit " +
      Math.round(100 * d.hit_rate) + "%</span> &nbsp; prefill " +
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

const src = new EventSource("/events");
src.onmessage = e => {
  const msg = JSON.parse(e.data);
  if (msg.kind === "done") { src.close(); return; }
  const h = handlers[msg.kind];
  if (h) h(msg);
};
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
                   N=args.children, verify=args.verify_children)

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
                   prefill=arm.prefill_charged, verified=arm.verified)

    def scoreboard(self, rows, notes: str) -> None:
        self._emit(kind="scoreboard", rows=rows, notes=notes)

    def finish(self) -> None:
        self._emit(kind="done")
