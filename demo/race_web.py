"""Browser dashboard for the split-screen race (``race_demo.py --web``).

Same race, same events, same JSON -- rendered in a browser instead of a
terminal. A stdlib ``ThreadingHTTPServer`` serves one self-contained HTML page
and streams the race as server-sent events, so there is no build step, no
framework, and no new dependency: the page is a few hundred bytes of HTML and
an ``EventSource``.

The UI object satisfies the same interface as ``LogUI``/``DashboardUI`` -- the
race does not know which one it is talking to.
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
 :root { color-scheme: dark; --hit:#39d353; --miss:#f85149; --dim:#8b949e;
         --line:#30363d; --bg:#0d1117; --panel:#161b22; }
 * { box-sizing: border-box; }
 body { margin:0; padding:24px; background:var(--bg); color:#e6edf3;
        font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
 h1 { font-size:18px; margin:0 0 4px; }
 .sub { color:var(--dim); white-space:pre-wrap; margin-bottom:8px; }
 .warn { color:#d29922; }
 .math { margin:12px 0 20px; padding:10px 12px; background:var(--panel);
         border:1px solid var(--line); border-radius:6px; }
 .math b { color:#e6edf3; } .math span { color:var(--dim); }
 .arms { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
 @media (max-width:900px) { .arms { grid-template-columns:1fr; } }
 .arm { background:var(--panel); border:1px solid var(--line);
        border-radius:6px; padding:14px; }
 .arm h2 { font-size:15px; margin:0 0 2px; letter-spacing:.06em; }
 .arm .url { color:var(--dim); font-size:12px; margin-bottom:10px; }
 .kv { height:20px; background:#21262d; border-radius:3px; overflow:hidden;
       position:relative; margin-bottom:4px; }
 .kv i { display:block; height:100%; width:0; background:#1f6feb;
         transition:width .2s; }
 .kv b { position:absolute; inset:0; text-align:center; font-weight:400;
         font-size:12px; line-height:20px; }
 .kvlabel { color:var(--dim); font-size:12px; margin-bottom:10px; }
 table { width:100%; border-collapse:collapse; font-size:13px; }
 td { padding:2px 4px; white-space:nowrap; }
 td.n { color:var(--dim); width:1%; }
 td.bar { width:90px; }
 .p { display:block; height:8px; background:#21262d; border-radius:2px; }
 .p i { display:block; height:100%; background:#8957e5; border-radius:2px; }
 .hit { color:var(--hit); } .miss { color:var(--miss); }
 .pass { color:var(--hit); } .fail { color:var(--dim); }
 .ev { margin-top:10px; color:#d29922; font-size:13px; }
 .done { margin-top:6px; font-size:13px; }
 #score { margin-top:22px; }
 #score table { max-width:760px; }
 #score td { border-top:1px solid var(--line); padding:4px; }
 #score td.k { color:var(--dim); }
 #score td.v { text-align:right; }
 .foot { color:var(--dim); margin-top:12px; white-space:pre-wrap;
         font-size:12px; max-width:820px; }
</style></head><body>
<h1>agentfork: 10 fixes, 1 inference server, and someone else is using it too</h1>
<div class="sub" id="header"></div>
<div class="math" id="math"></div>
<div class="arms">
  <div class="arm" id="arm-stock"><h2>STOCK</h2><div class="url"></div>
    <div class="kv"><i></i><b></b></div><div class="kvlabel"></div>
    <table></table><div class="ev"></div><div class="done"></div></div>
  <div class="arm" id="arm-agentfork"><h2>AGENTFORK</h2><div class="url"></div>
    <div class="kv"><i></i><b></b></div><div class="kvlabel"></div>
    <table></table><div class="ev"></div><div class="done"></div></div>
</div>
<div id="score"></div>
<div class="foot" id="foot"></div>
<script>
const $ = (s, r) => (r || document).querySelector(s);
const arm = n => $("#arm-" + n);
const esc = s => String(s).replace(/[&<>]/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

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
  },
  arm_start(d) { $(".url", arm(d.name)).textContent = d.url; },
  parent(d) {
    $(".url", arm(d.name)).textContent +=
      "  \\u00b7  shared context: " + d.charged + " tok charged (P=" + d.P + ")";
    kv(d.name, d.charged, d.capacity);
  },
  child(d) {
    const row = document.createElement("tr");
    row.innerHTML =
      "<td class='n'>" + esc(d.label) + " " + (d.idx + 1) + "/" + d.total + "</td>" +
      "<td class='bar'><span class='p'><i style='width:" +
        (100 * (d.idx + 1) / d.total) + "%'></i></span></td>" +
      "<td class='" + (d.parent_hit ? "hit" : "miss") + "'>" +
        (d.parent_hit ? "HIT" : "MISS") + "</td>" +
      "<td class='n'>cached " + d.cached.toLocaleString() + "</td>" +
      "<td>charged " + d.charged.toLocaleString() + "</td>" +
      "<td class='" + (d.passed ? "pass" : "fail") + "'>" +
        (d.passed ? "PASS" : "fail") + "</td>";
    $("table", arm(d.name)).appendChild(row);
  },
  kv(d) { kv(d.name, d.used, d.capacity); },
  kills(d) {
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
