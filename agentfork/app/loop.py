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
        """Block until every eval for the given nodes has finished — a node
        can look settled while a later run of it is still queued (eval_slots
        throttling), so the run rows are the ground truth."""
        pending = set(node_ids)
        while pending and not self._stop.is_set():
            for node_id in list(pending):
                node = self.engine.store.node(node_id)
                runs = self.engine.store.runs(self.project_id,
                                              node_id=node_id)
                in_flight = any(r["status"] in ("queued", "running")
                                for r in runs)
                if not in_flight and node["status"] not in (
                        "proposed", "implementing", "ready", "running"):
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
            for _ in range(int(params["max_gens"])):
                if self._stop.is_set():
                    break
                frontier = store.loop_state(self.project_id)["frontier"]
                if not frontier:
                    break
                # the gen is derived from the frontier, not the loop counter —
                # a loop restarted after done/stalled must fan out at the
                # frontier's gen + 1, not re-run stale gen numbers
                gen = max(store.node(nid)["gen"] for nid in frontier) + 1
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
                              message=f"gen {gen}: STALL — no candidate beat "
                                      "its parent beyond the margin. The "
                                      "frontier is unchanged and still "
                                      "expandable — try a new hypothesis "
                                      "family.")
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
            baseline = next((n for n in store.nodes(self.project_id, gen=0)
                             if n["id"] != champion["id"]), None)
            verdict = self._holdout(champion, baseline, params)
            if verdict is not None:
                self._set(**verdict)
                return
        self._set(state="done",
                  message="search complete" +
                  (f"; champion {champion['id']} ({champion['score']})"
                   if champion else ""))

    def _run_holdout(self, node_id: str) -> dict:
        """Start a holdout run and wait on the run row — the node is already
        settled, so node status would slip past any node-level wait."""
        store = self.engine.store
        run = self.engine.start_eval(self.project_id, node_id, kind="holdout")
        while not self._stop.is_set():
            run = store.run(run["id"])
            if run["status"] not in ("queued", "running"):
                break
            time.sleep(0.5)
        return store.run(run["id"])

    def _holdout(self, champion: dict, baseline: dict | None,
                 params: dict) -> dict | None:
        """program.md's finish line: holdout on the champion and the baseline,
        ship only if the champion still wins beyond the margin. Returns the
        loop-state update to apply, or None to fall through to 'done'."""
        store = self.engine.store
        margin = store.loop_state(self.project_id)["margin"] or 0.0
        self._set(state="holdout",
                  message=f"champion {champion['id']}: holdout revalidation"
                          + (" (baseline too)" if baseline else ""))
        champ_run = self._run_holdout(champion["id"])
        base_run = self._run_holdout(baseline["id"]) if baseline else None
        if champ_run["status"] != "done" or champ_run["score"] is None:
            return {"state": "error",
                    "message": f"holdout for champion {champion['id']} ended "
                               f"{champ_run['status']} with no score — the "
                               "search ends unvalidated, not done"}
        if base_run is None:
            # the champion IS the baseline — nothing to compare against
            return None
        if base_run["status"] != "done" or base_run["score"] is None:
            return {"state": "error",
                    "message": "baseline holdout did not score — cannot "
                               "validate the champion"}
        if params["minimize"]:
            wins = champ_run["score"] < base_run["score"] - margin
        else:
            wins = champ_run["score"] > base_run["score"] + margin
        if not wins:
            return {"state": "error",
                    "message": f"champion {champion['id']} did not beat the "
                               f"baseline on the holdout "
                               f"({champ_run['score']:g} vs "
                               f"{base_run['score']:g}, margin {margin:g})"}
        return {"state": "done",
                "message": f"champion {champion['id']} validated on the "
                           f"holdout ({champ_run['score']:g} vs baseline "
                           f"{base_run['score']:g})"}

    def _champion(self, minimize: bool) -> dict | None:
        # the baseline node ends up "ran", not "baseline" — include it so a
        # search where nothing beat the baseline still has a champion
        nodes = [n for n in self.engine.store.nodes(self.project_id)
                 if n["score"] is not None
                 and (n["status"] in ("kept", "baseline")
                      or (n["gen"] == 0 and n["status"] == "ran"))]
        if not nodes:
            return None
        return min(nodes, key=lambda n: n["score"]) if minimize \
            else max(nodes, key=lambda n: n["score"])
