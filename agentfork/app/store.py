"""Local SQLite store for the agentfork app.

Everything the dashboard shows lives here: projects, the experiment tree
(nodes), eval/worker runs, laws, and an event log the UI tails over SSE. The
canonical scientific record is still the project's append-only
``results.tsv`` (see :mod:`agentfork.app.amr`); this database is the index the
UI reads and is rebuilt from the tsv + git on import.

Loopback-only product, one process: a single connection guarded by a lock is
plenty and keeps the stdlib-only footprint.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    repo_path       TEXT NOT NULL,
    baseline_branch TEXT NOT NULL,
    tag             TEXT NOT NULL,
    harness         TEXT NOT NULL,
    params_json     TEXT NOT NULL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS nodes (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL,
    parent_id     TEXT,
    gen           INTEGER NOT NULL,
    slug          TEXT NOT NULL,
    branch_name   TEXT NOT NULL,
    worktree_path TEXT,
    commit_sha    TEXT,
    title         TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    regions       TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL,
    score         REAL,
    delta         REAL,
    costs_json    TEXT NOT NULL DEFAULT '{}',
    frozen        INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nodes_project ON nodes(project_id, gen);
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    node_id     TEXT NOT NULL,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    command     TEXT NOT NULL,
    run_dir     TEXT NOT NULL,
    pid         INTEGER,
    exit_code   INTEGER,
    score       REAL,
    created_at  REAL NOT NULL,
    started_at  REAL,
    ended_at    REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id, created_at);
CREATE TABLE IF NOT EXISTS laws (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    gen         INTEGER NOT NULL,
    text        TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT,
    ts          REAL NOT NULL,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS loop_state (
    project_id  TEXT PRIMARY KEY,
    state       TEXT NOT NULL,
    gen         INTEGER NOT NULL DEFAULT 0,
    margin      REAL,
    frontier    TEXT NOT NULL DEFAULT '[]',
    message     TEXT NOT NULL DEFAULT '',
    updated_at  REAL NOT NULL
);
"""

NODE_STATUSES = ("baseline", "proposed", "implementing", "ready", "running",
                 "ran", "crash", "kept", "discarded", "cost_fail")
RUN_STATUSES = ("queued", "running", "done", "failed", "killed", "timeout")


def data_dir() -> Path:
    """``$AGENTFORK_HOME`` or ``~/.agentfork``."""
    return Path(os.environ.get("AGENTFORK_HOME", Path.home() / ".agentfork"))


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _row(cur: sqlite3.Cursor, row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


class Store:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else data_dir() / "agentfork.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False,
                                     isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- helpers ------------------------------------------------------------

    def _exec(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, tuple(params))

    def _one(self, sql: str, params: Iterable[Any] = ()) -> dict | None:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            return _row(cur, cur.fetchone())

    def _all(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, tuple(params))]

    @staticmethod
    def _update(table: str, key: str, ident: str, **changes) -> tuple[str, list]:
        changes["updated_at"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in changes)
        return (f"UPDATE {table} SET {cols} WHERE {key} = ?",
                [*changes.values(), ident])

    # -- projects -----------------------------------------------------------

    def create_project(self, *, name: str, repo_path: str, baseline_branch: str,
                       tag: str, harness: str, params: dict) -> dict:
        now = time.time()
        pid = new_id("proj")
        self._exec(
            "INSERT INTO projects VALUES (?,?,?,?,?,?,?,?,?)",
            (pid, name, repo_path, baseline_branch, tag, harness,
             json.dumps(params), now, now))
        return self.project(pid)

    def project(self, project_id: str) -> dict | None:
        p = self._one("SELECT * FROM projects WHERE id = ?", (project_id,))
        return self._inflate_project(p) if p else None

    def projects(self) -> list[dict]:
        return [self._inflate_project(p) for p in
                self._all("SELECT * FROM projects ORDER BY created_at DESC")]

    def update_project(self, project_id: str, **changes) -> dict:
        if "params" in changes:
            changes["params_json"] = json.dumps(changes.pop("params"))
        sql, params = self._update("projects", "id", project_id, **changes)
        self._exec(sql, params)
        return self.project(project_id)

    def delete_project(self, project_id: str) -> None:
        for table in ("nodes", "runs", "laws", "events", "loop_state"):
            self._exec(f"DELETE FROM {table} WHERE project_id = ?", (project_id,))
        self._exec("DELETE FROM projects WHERE id = ?", (project_id,))

    @staticmethod
    def _inflate_project(p: dict) -> dict:
        p = dict(p)
        p["params"] = json.loads(p.pop("params_json"))
        return p

    # -- nodes --------------------------------------------------------------

    def create_node(self, *, project_id: str, parent_id: str | None, gen: int,
                    slug: str, branch_name: str, title: str,
                    description: str = "", regions: Iterable[str] = (),
                    status: str = "proposed", worktree_path: str | None = None,
                    commit_sha: str | None = None, node_id: str | None = None) -> dict:
        if status not in NODE_STATUSES:
            raise ValueError(f"bad node status: {status}")
        now = time.time()
        nid = node_id or new_id("node")
        self._exec(
            "INSERT INTO nodes (id, project_id, parent_id, gen, slug, branch_name,"
            " worktree_path, commit_sha, title, description, regions, status,"
            " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (nid, project_id, parent_id, gen, slug, branch_name, worktree_path,
             commit_sha, title, description, ",".join(sorted(set(regions))),
             status, now, now))
        return self.node(nid)

    def node(self, node_id: str) -> dict | None:
        n = self._one("SELECT * FROM nodes WHERE id = ?", (node_id,))
        return self._inflate_node(n) if n else None

    def nodes(self, project_id: str, gen: int | None = None) -> list[dict]:
        if gen is None:
            rows = self._all("SELECT * FROM nodes WHERE project_id = ? "
                             "ORDER BY gen, created_at", (project_id,))
        else:
            rows = self._all("SELECT * FROM nodes WHERE project_id = ? AND gen = ? "
                             "ORDER BY created_at", (project_id, gen))
        return [self._inflate_node(n) for n in rows]

    def update_node(self, node_id: str, **changes) -> dict:
        if "costs" in changes:
            changes["costs_json"] = json.dumps(changes.pop("costs"))
        if "regions" in changes and not isinstance(changes["regions"], str):
            changes["regions"] = ",".join(sorted(set(changes["regions"])))
        if "status" in changes and changes["status"] not in NODE_STATUSES:
            raise ValueError(f"bad node status: {changes['status']}")
        sql, params = self._update("nodes", "id", node_id, **changes)
        self._exec(sql, params)
        return self.node(node_id)

    @staticmethod
    def _inflate_node(n: dict) -> dict:
        n = dict(n)
        n["costs"] = json.loads(n.pop("costs_json") or "{}")
        n["regions"] = [r for r in (n.get("regions") or "").split(",") if r]
        n["frozen"] = bool(n["frozen"])
        return n

    # -- runs ---------------------------------------------------------------

    def create_run(self, *, project_id: str, node_id: str, kind: str,
                   command: str, run_dir: str, status: str = "queued") -> dict:
        now = time.time()
        rid = new_id("run")
        self._exec(
            "INSERT INTO runs (id, project_id, node_id, kind, status, command,"
            " run_dir, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (rid, project_id, node_id, kind, status, command, run_dir, now))
        return self.run(rid)

    def run(self, run_id: str) -> dict | None:
        return self._one("SELECT * FROM runs WHERE id = ?", (run_id,))

    def runs(self, project_id: str, node_id: str | None = None) -> list[dict]:
        if node_id is None:
            return self._all("SELECT * FROM runs WHERE project_id = ? "
                             "ORDER BY created_at DESC", (project_id,))
        return self._all("SELECT * FROM runs WHERE project_id = ? AND node_id = ? "
                         "ORDER BY created_at DESC", (project_id, node_id))

    def active_runs(self) -> list[dict]:
        return self._all("SELECT * FROM runs WHERE status IN ('queued','running')")

    def settle_run(self, run_id: str, **changes) -> bool:
        """Terminal write for a run, applied only while it is still in
        flight. A kill racing the poller must not let the later finisher
        resurrect the row (double run.finished events, duplicate tsv
        rows). Returns False when the run had already settled."""
        status = changes.get("status")
        if status not in RUN_STATUSES:
            raise ValueError(f"bad run status: {status}")
        changes.pop("updated_at", None)
        cols = ", ".join(f"{k} = ?" for k in changes)
        cur = self._exec(
            f"UPDATE runs SET {cols} WHERE id = ? "
            "AND status IN ('queued','running')",
            [*changes.values(), run_id])
        return cur.rowcount > 0

    def update_run(self, run_id: str, **changes) -> dict:
        if "status" in changes and changes["status"] not in RUN_STATUSES:
            raise ValueError(f"bad run status: {changes['status']}")
        changes.pop("updated_at", None)
        cols = ", ".join(f"{k} = ?" for k in changes)
        self._exec(f"UPDATE runs SET {cols} WHERE id = ?",
                   [*changes.values(), run_id])
        return self.run(run_id)

    # -- laws ---------------------------------------------------------------

    def add_law(self, project_id: str, gen: int, text: str) -> dict:
        lid = new_id("law")
        self._exec("INSERT INTO laws VALUES (?,?,?,?,?)",
                   (lid, project_id, gen, text, time.time()))
        return self._one("SELECT * FROM laws WHERE id = ?", (lid,))

    def laws(self, project_id: str) -> list[dict]:
        return self._all("SELECT * FROM laws WHERE project_id = ? ORDER BY created_at",
                         (project_id,))

    # -- loop state ---------------------------------------------------------

    def loop_state(self, project_id: str) -> dict:
        s = self._one("SELECT * FROM loop_state WHERE project_id = ?", (project_id,))
        if s is None:
            return {"project_id": project_id, "state": "idle", "gen": 0,
                    "margin": None, "frontier": [], "message": "",
                    "updated_at": None}
        s["frontier"] = json.loads(s["frontier"])
        return s

    def set_loop_state(self, project_id: str, **changes) -> dict:
        cur = self.loop_state(project_id)
        cur.update(changes)
        self._exec(
            "INSERT OR REPLACE INTO loop_state VALUES (?,?,?,?,?,?,?)",
            (project_id, cur["state"], cur["gen"], cur["margin"],
             json.dumps(cur["frontier"]), cur["message"], time.time()))
        return self.loop_state(project_id)

    # -- events -------------------------------------------------------------

    def add_event(self, project_id: str | None, kind: str, payload: dict) -> dict:
        ts = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (project_id, ts, kind, payload) VALUES (?,?,?,?)",
                (project_id, ts, kind, json.dumps(payload)))
            seq = cur.lastrowid
        return {"seq": seq, "project_id": project_id, "ts": ts, "kind": kind,
                "payload": payload}

    def events(self, project_id: str | None = None, after: int = 0,
               limit: int = 200) -> list[dict]:
        if project_id is None:
            rows = self._all("SELECT * FROM events WHERE seq > ? "
                             "ORDER BY seq LIMIT ?", (after, limit))
        else:
            rows = self._all("SELECT * FROM events WHERE seq > ? AND "
                             "(project_id = ? OR project_id IS NULL) "
                             "ORDER BY seq LIMIT ?", (after, project_id, limit))
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows

    def last_seq(self) -> int:
        row = self._one("SELECT MAX(seq) AS s FROM events")
        return int(row["s"] or 0) if row else 0
