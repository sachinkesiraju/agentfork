"""The dashboard's SQLite index: projects, nodes, runs, laws, events."""

import pytest

from agentfork.app.store import Store


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "agentfork.db")
    yield s
    s.close()


def _project(store, **kw):
    return store.create_project(name="p", repo_path="/repo",
                                baseline_branch="main", tag="p",
                                harness="fake",
                                params={"eval_cmd": "true", "k": 4}, **kw)


def test_project_params_roundtrip(store):
    p = _project(store)
    assert p["params"]["eval_cmd"] == "true"
    assert store.project(p["id"])["params"]["k"] == 4
    assert [x["id"] for x in store.projects()] == [p["id"]]


def test_update_project_params(store):
    p = _project(store)
    updated = store.update_project(p["id"], params={"eval_cmd": "pytest"})
    assert updated["params"] == {"eval_cmd": "pytest"}


def test_delete_project_cascades(store):
    p = _project(store)
    n = store.create_node(project_id=p["id"], parent_id=None, gen=0,
                          slug="base", branch_name="main", title="baseline")
    store.create_run(project_id=p["id"], node_id=n["id"], kind="eval",
                     command="true", run_dir="/tmp/x")
    store.add_law(p["id"], 1, "law")
    store.delete_project(p["id"])
    assert store.project(p["id"]) is None
    assert store.nodes(p["id"]) == []
    assert store.runs(p["id"]) == []
    assert store.laws(p["id"]) == []


def test_node_regions_and_costs_are_structured(store):
    p = _project(store)
    n = store.create_node(project_id=p["id"], parent_id=None, gen=1,
                          slug="c", branch_name="af/c", title="cand",
                          regions=["data", "model", "data"])
    assert n["regions"] == ["data", "model"]
    n = store.update_node(n["id"], costs={"seconds": 12.5}, score=0.4,
                          status="ran", frozen=1)
    assert n["costs"] == {"seconds": 12.5}
    assert n["frozen"] is True


def test_node_status_is_validated(store):
    p = _project(store)
    with pytest.raises(ValueError):
        store.create_node(project_id=p["id"], parent_id=None, gen=0,
                          slug="s", branch_name="b", title="t",
                          status="nonsense")


def test_nodes_filtered_by_generation(store):
    p = _project(store)
    store.create_node(project_id=p["id"], parent_id=None, gen=0, slug="b",
                      branch_name="main", title="baseline")
    store.create_node(project_id=p["id"], parent_id=None, gen=1, slug="c",
                      branch_name="af/c", title="cand")
    assert len(store.nodes(p["id"])) == 2
    assert [n["gen"] for n in store.nodes(p["id"], gen=1)] == [1]


def test_active_runs_tracks_status(store):
    p = _project(store)
    n = store.create_node(project_id=p["id"], parent_id=None, gen=0, slug="b",
                          branch_name="main", title="baseline")
    r = store.create_run(project_id=p["id"], node_id=n["id"], kind="eval",
                         command="true", run_dir="/tmp/x", status="running")
    assert [x["id"] for x in store.active_runs()] == [r["id"]]
    store.update_run(r["id"], status="done", exit_code=0, score=1.0)
    assert store.active_runs() == []
    assert store.run(r["id"])["score"] == 1.0


def test_run_status_is_validated(store):
    p = _project(store)
    n = store.create_node(project_id=p["id"], parent_id=None, gen=0, slug="b",
                          branch_name="main", title="baseline")
    r = store.create_run(project_id=p["id"], node_id=n["id"], kind="eval",
                         command="true", run_dir="/tmp/x")
    with pytest.raises(ValueError):
        store.update_run(r["id"], status="wat")


def test_loop_state_defaults_and_merges(store):
    p = _project(store)
    assert store.loop_state(p["id"])["state"] == "idle"
    store.set_loop_state(p["id"], state="running", gen=2, frontier=["a", "b"])
    s = store.set_loop_state(p["id"], message="hi")
    assert (s["state"], s["gen"], s["frontier"], s["message"]) == \
        ("running", 2, ["a", "b"], "hi")


def test_events_are_ordered_and_filtered(store):
    p = _project(store)
    other = store.create_project(name="o", repo_path="/o",
                                 baseline_branch="main", tag="o",
                                 harness="fake", params={})
    a = store.add_event(p["id"], "one", {"x": 1})
    store.add_event(other["id"], "two", {})
    assert a["seq"] < store.last_seq()
    mine = store.events(project_id=p["id"])
    assert [e["kind"] for e in mine] == ["one"]
    assert mine[0]["payload"] == {"x": 1}
    assert len(store.events(after=a["seq"])) == 1
