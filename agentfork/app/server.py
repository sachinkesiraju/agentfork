"""Loopback-only HTTP API + dashboard statics, stdlib ``http.server``.

Routes (JSON unless noted):

    GET  /api/health                 {ok, version}
    GET  /api/harnesses              detection probe for onboarding
    GET  /api/projects               list
    POST /api/projects               {name, repo_path, baseline_branch, harness, params}
    GET  /api/projects/{id}          project + loop state + params
    DEL  /api/projects/{id}
    GET  /api/projects/{id}/tree     experiment tree (nodes + loop state)
    GET  /api/projects/{id}/metrics  orchestrator + KV counters (Ledger)
    GET  /api/projects/{id}/runs     all runs
    GET  /api/projects/{id}/laws
    POST /api/projects/{id}/baseline create root node + queue baseline evals
    POST /api/projects/{id}/descend  {parent_id, k} propose+fan out
    POST /api/projects/{id}/reduce   {gen, margin?}
    POST /api/projects/{id}/holdout  {node_id}
    POST /api/projects/{id}/loop/start   run the whole map-reduce loop
    POST /api/projects/{id}/loop/stop
    POST /api/projects/{id}/law      {gen, text} record a law/note
    GET  /api/runs/{id}/log?offset=  incremental log tail
    POST /api/runs/{id}/kill
    GET  /api/events?after=&project= SSE stream
    GET  /*                          dashboard statics (package ``ui/dist``)

``ThreadingHTTPServer`` — detached runs + in-process workers keep working
while SSE clients hang.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from agentfork.app import amr, runner
from agentfork.app.engine import Engine, EngineError
from agentfork.app.harness import detect_harnesses
from agentfork.app.loop import AutoresearchLoop
from agentfork.app.store import Store

_log = logging.getLogger("agentfork.app.server")

VERSION = "0.4.0"


class App:
    def __init__(self, home: Path | None = None):
        self.store = Store(home / "agentfork.db" if home else None)
        self.engine = Engine(self.store, home=self.store.path.parent)
        self.loops: dict[str, AutoresearchLoop] = {}
        self._subs: list[queue.Queue] = []
        self._last_seq = self.store.last_seq()
        self._closed = threading.Event()
        threading.Thread(target=self._event_feed, daemon=True).start()

    def close(self) -> None:
        self._closed.set()
        for loop in self.loops.values():
            loop.stop()
        self.engine.close()
        self.store.close()

    # -- SSE fanout ---------------------------------------------------------

    def _event_feed(self) -> None:
        while not self._closed.is_set():
            events = self.store.events(after=self._last_seq)
            if events:
                self._last_seq = events[-1]["seq"]
                dead = []
                for q in self._subs:
                    for e in events:
                        try:
                            q.put_nowait(e)
                        except queue.Full:
                            dead.append(q)
                for q in dead:
                    if q in self._subs:
                        self._subs.remove(q)
            self._closed.wait(0.25)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2000)
        self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        if q in self._subs:
            self._subs.remove(q)


def make_handler(app: App, ui_root: Path | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "agentfork/" + VERSION
        protocol_version = "HTTP/1.1"

        # -- plumbing -------------------------------------------------------

        def log_message(self, fmt, *args):
            _log.debug(fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str,
                  extra: dict | None = None):
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, default=str).encode(),
                           "application/json")

        def _error(self, code: int, msg: str) -> None:
            self._json(code, {"error": msg})

        def _body(self) -> dict:
            length = int(self.headers.get("content-length") or 0)
            if not length:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw or b"{}")

        def _dispatch(self, method: str) -> None:
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            query = parse_qs(url.query)
            try:
                out = self._route(method, parts, query)
            except (EngineError, amr.AmrError, ValueError, KeyError) as exc:
                self._error(400, str(exc))
                return
            except BrokenPipeError:
                return
            except Exception as exc:  # noqa: BLE001
                _log.exception("request failed: %s %s", method, self.path)
                self._error(500, str(exc))
                return
            if out is not None:
                self._json(200, out)

        def do_GET(self):    self._dispatch("GET")
        def do_POST(self):   self._dispatch("POST")
        def do_DELETE(self): self._dispatch("DELETE")

        # -- routes -----------------------------------------------------------

        def _route(self, method, parts, q):
            if parts[:1] != ["api"]:
                return self._static(parts)
            rest = parts[1:]
            if rest == ["health"]:
                return {"ok": True, "version": VERSION,
                        "home": str(app.store.path.parent)}
            if rest == ["harnesses"]:
                return detect_harnesses()
            if rest == ["events"]:
                return self._sse(q)
            if rest == ["projects"]:
                if method == "GET":
                    return app.store.projects()
                if method == "POST":
                    b = self._body()
                    p = app.engine.create_project(
                        name=b.get("name") or Path(b["repo_path"]).name,
                        repo_path=b["repo_path"],
                        baseline_branch=b.get("baseline_branch") or "main",
                        harness_name=b.get("harness", "claude-code"),
                        params=b.get("params") or {})
                    return p
            if len(rest) >= 2 and rest[0] == "projects":
                return self._project(method, rest[1], rest[2:], q)
            if len(rest) == 3 and rest[0] == "runs":
                run_id, action = rest[1], rest[2]
                if action == "kill" and method == "POST":
                    return app.engine.kill_run(run_id)
            if len(rest) == 3 and rest[0] == "runs" and rest[2] == "log":
                run = app.store.run(rest[1])
                if run is None:
                    raise KeyError("no such run")
                off = int(q.get("offset", ["0"])[0])
                text, new_off = runner.tail(run["run_dir"], offset=off)
                return {"text": text, "offset": new_off,
                        "alive": runner.alive(run["run_dir"]),
                        "exit_code": runner.exit_code(run["run_dir"]),
                        "status": run["status"]}
            raise KeyError(f"no route: {method} /{'/'.join(parts)}")

        def _project(self, method, pid, rest, q):
            store, engine = app.store, app.engine
            if not rest:
                if method == "GET":
                    p = store.project(pid)
                    if p is None:
                        raise KeyError("no such project")
                    p["loop"] = store.loop_state(pid)
                    return p
                if method == "DELETE":
                    for loop in list(app.loops.values()):
                        if loop.project_id == pid:
                            loop.stop()
                    for run in store.active_runs():
                        if run["project_id"] == pid:
                            engine.kill_run(run["id"])
                    store.delete_project(pid)
                    return {"deleted": pid}
            action = rest[0]
            if action == "tree":
                return engine.tree(pid)
            if action == "metrics":
                return engine.metrics(pid)
            if action == "runs":
                return store.runs(pid)
            if action == "laws":
                return store.laws(pid)
            if action == "results":
                r = engine.results(pid)
                rows = [row.to_dict() for row in r.rows()]
                return {"path": str(r.path), "rows": rows,
                        "tree": r.tree(minimize=store.project(pid)
                                       ["params"]["minimize"]).render()}
            if action == "baseline" and method == "POST":
                return engine.baseline(pid)
            if action == "descend" and method == "POST":
                b = self._body()
                nodes = engine.descend(pid, b["parent_id"],
                                       int(b.get("k") or
                                           store.project(pid)["params"]["k"]))
                return {"nodes": nodes}
            if action == "reduce" and method == "POST":
                b = self._body()
                red = engine.reduce(pid, int(b["gen"]),
                                    margin=b.get("margin"))
                return red.to_dict()
            if action == "holdout" and method == "POST":
                b = self._body()
                return engine.start_eval(pid, b["node_id"], kind="holdout")
            if action == "law" and method == "POST":
                b = self._body()
                return store.add_law(pid, int(b.get("gen", 0)), b["text"])
            if action == "eval" and method == "POST":
                b = self._body()
                return engine.start_eval(pid, b["node_id"],
                                         force=bool(b.get("force")))
            if action == "loop":
                return self._loop(method, pid, rest[1:])
            raise KeyError(f"no route: {method} /api/projects/{pid}/{'/'.join(rest)}")

        def _loop(self, method, pid, rest):
            if not rest:
                return app.store.loop_state(pid)
            if rest[0] == "start" and method == "POST":
                loop = app.loops.get(pid)
                if loop is None or not loop.alive:
                    loop = AutoresearchLoop(app.engine, pid)
                    app.loops[pid] = loop
                loop.start()
                return {"state": "running"}
            if rest[0] == "stop" and method == "POST":
                loop = app.loops.get(pid)
                if loop:
                    loop.stop()
                return {"state": "stopping"}
            raise KeyError("bad loop route")

        # -- SSE --------------------------------------------------------------

        def _sse(self, q):
            after = int(q.get("after", ["0"])[0])
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "keep-alive")
            self.end_headers()
            sub = app.subscribe()
            try:
                for e in app.store.events(after=after):
                    self.wfile.write(_sse_frame(e))
                while True:
                    try:
                        e = sub.get(timeout=15)
                        self.wfile.write(_sse_frame(e))
                    except queue.Empty:
                        self.wfile.write(b":\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                app.unsubscribe(sub)
            return None

        # -- statics -----------------------------------------------------------

        def _static(self, parts):
            if ui_root is None:
                self._send(200, b"agentfork is running; UI bundle not installed",
                           "text/plain")
                return None
            rel = "/".join(parts) or "index.html"
            path = (ui_root / rel).resolve()
            if not str(path).startswith(str(ui_root.resolve())) \
                    or not path.is_file():
                path = ui_root / "index.html"  # SPA fallback
            if not path.is_file():
                self._send(404, b"not found", "text/plain")
                return None
            ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self._send(200, path.read_bytes(), ctype)
            return None

    return Handler


def _sse_frame(event: dict) -> bytes:
    return (f"id: {event['seq']}\ndata: {json.dumps(event, default=str)}\n\n"
            .encode())


def serve(home: Path | None = None, host: str = "127.0.0.1", port: int = 8474,
          open_browser: bool = True) -> None:
    app = App(home)
    ui_root = Path(__file__).parent / "ui" / "dist"
    if not ui_root.is_dir():
        ui_root = None
    httpd = ThreadingHTTPServer((host, port), make_handler(app, ui_root))
    url = f"http://{host}:{port}"
    print(f"agentfork dashboard: {url}")
    print(f"data: {app.store.path.parent}")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
