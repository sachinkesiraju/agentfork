"""The autonomous map-reduce driver: the baked-in agent-mapreduce program.

``AutoresearchLoop`` executes program.md's generation loop end to end —
baseline margin, then per generation: propose K on every frontier node, fan
out, wait for evals, reduce parent-relative, keep top B, write a law on stall.
It is deliberately a state machine over the engine (not inline orchestration)
so the dashboard can pause/stop it and the CLI can watch the same loop state.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from agentfork.app.engine import Engine

_log = logging.getLogger("agentfork.app.loop")


class AutoresearchLoop:
    def __init__(self, engine: Engine, project_id: str):
        self.engine = engine
        self.project_id = project_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.alive:
            return
        self._stop.clear()
        self.engine.driven.add(self.project_id)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _set(self, **kw):
        self.engine.store.set_loop_state(self.project_id, **kw)

    def _release(self) -> None:
        self.engine.driven.discard(self.project_id)

    def _wait_runs(self, node_ids: list[str]) -> None:
        """Block until every eval for the given nodes has finished."""
        pending = set(node_ids)
        while pending and not self._stop.is_set():
            for node_id in list(pending):
                node = self.engine.store.node(node_id)
                if node["status"] not in ("proposed", "implementing", "ready",
                                          "running"):
                    pending.discard(node_id)
            time.sleep(0.5)

    def _run(self) -> None:
        store, engine = self.engine.store, self.engine
        project = store.project(self.project_id)
        params = project["params"]
        try:
            self._set(state="baseline",
                      message="running baseline evals to fix the noise margin")
            if not store.nodes(self.project_id, gen=0):
                engine.baseline(self.project_id)
                self._wait_runs([n["id"] for n in
                                 store.nodes(self.project_id, gen=0)])
            margin = engine._margin(self.project_id)
            self._set(state="running", margin=margin, gen=0,
                      message=f"margin fixed at {margin:g}")
            engine.store.add_event(self.project_id, "loop.margin",
                                   {"margin": margin})
            for gen in range(1, int(params["max_gens"]) + 1):
                if self._stop.is_set():
                    break
                frontier = store.loop_state(self.project_id)["frontier"]
                if not frontier:
                    break
                self._set(state="running", gen=gen,
                          message=f"gen {gen}: proposing {params['k']} ideas "
                                  f"per frontier node")
                children = []
                for node_id in frontier:
                    ideas = engine.propose(self.project_id, node_id,
                                           int(params["k"]))
                    children += engine.fan_out(self.project_id, node_id, ideas)
                self._set(message=f"gen {gen}: {len(children)} candidates "
                                  "implementing + evaluating")
                self._wait_runs([n["id"] for n in children])
                if self._stop.is_set():
                    break
                self._set(message=f"gen {gen}: reducing (parent-relative, "
                                  f"margin {margin:g}, beam {params['b']})")
                red = engine.reduce(self.project_id, gen, margin=margin)
                if red.stall:
                    self._set(state="stalled", gen=gen,
                              message=f"gen {gen}: STALL — frontier unchanged")
                    break
            self._finish()
        except Exception as exc:  # noqa: BLE001
            self._set(state="error", message=str(exc))
            _log.exception("loop failed")
        finally:
            self._release()

    def _finish(self) -> None:
        store = self.engine.store
        if self._stop.is_set():
            self._set(state="stopped", message="stopped by user")
            return
        project = store.project(self.project_id)
        params = project["params"]
        state = store.loop_state(self.project_id)
        if state["state"] == "error":
            return
        champion = self._champion(params["minimize"])
        if champion and params.get("holdout_cmd"):
            self._set(state="holdout", gen=state["gen"],
                      message=f"champion {champion['id']}: holdout revalidation")
            self.engine.start_eval(self.project_id, champion["id"],
                                   kind="holdout")
            self._wait_runs([champion["id"]])
        self._set(state="done",
                  message="search complete" +
                  (f"; champion {champion['id']} ({champion['score']})"
                   if champion else ""))

    def _champion(self, minimize: bool) -> dict | None:
        nodes = [n for n in self.engine.store.nodes(self.project_id)
                 if n["status"] in ("kept", "baseline") and n["score"] is not None]
        if not nodes:
            return None
        return min(nodes, key=lambda n: n["score"]) if minimize \
            else max(nodes, key=lambda n: n["score"])
