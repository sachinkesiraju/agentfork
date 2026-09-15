"""Detached worker process: apply a node's idea inside its worktree.

Spawned by ``Engine.start_worker`` as a real process (not a thread) so the
run is killable mid-edit — a ``claude -p`` worker can take minutes — and
streams its log like any other run. Because the worktree survives a
dashboard shutdown (``close`` no longer collects branches), a worker that
outlives the dashboard still commits and queues its eval; the restarted
engine adopts the eval through the normal queue.

Reads ``payload.json`` in its run dir — written by the engine — then:
implement → commit → mark the node ready → queue its eval run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agentfork.app import git
from agentfork.app.harness import Idea, build_harness
from agentfork.app.store import Store


def main(run_dir: str) -> int:
    run_dir = Path(run_dir)
    payload = json.loads((run_dir / "payload.json").read_text())
    store = Store(payload["db"])
    project = store.project(payload["project_id"])
    node = store.node(payload["node_id"])
    if project is None or node is None:
        print("project or node deleted; worker exiting", file=sys.stderr)
        return 3
    params = project["params"]
    harness = build_harness(project["harness"],
                            model=params.get("model") or None,
                            api_base=params.get("api_base") or None)
    idea = Idea(title=payload["title"], description=payload["description"],
                regions=payload["regions"])
    summary = harness.implement(node["worktree_path"], idea,
                                payload["context"])
    print(summary)
    sha = git.commit_all(node["worktree_path"], f"agentfork: {idea.title}")
    store.update_node(node["id"], status="ready",
                      commit_sha=sha or node["commit_sha"])
    store.add_event(project["id"], "node.ready", {"node_id": node["id"]})
    # queue the eval; the engine's queue drain starts it when a slot frees
    command = project["params"]["eval_cmd"]
    run = store.create_run(project_id=project["id"], node_id=node["id"],
                           kind="eval", command=command, run_dir="")
    eval_dir = Path(payload["project_dir"]) / "runs" / run["id"]
    eval_dir.mkdir(parents=True, exist_ok=True)
    store.update_run(run["id"], run_dir=str(eval_dir))
    store.update_node(node["id"], status="running")
    store.add_event(project["id"], "run.queued",
                    {"node_id": node["id"], "run_id": run["id"]})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
