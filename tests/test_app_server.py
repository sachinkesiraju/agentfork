"""The loopback API end to end: project → baseline → descend → reduce over
HTTP, with the fake harness and a real git repo."""

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from agentfork.app import git
from agentfork.app.server import App, make_handler
from test_app_engine import EVAL


@pytest.fixture()
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    git.git(path, "init", "-q", "-b", "main")
    git.git(path, "config", "user.email", "t@example.com")
    git.git(path, "config", "user.name", "t")
    (path / "eval.sh").write_text(EVAL)
    (path / "eval.sh").chmod(0o755)
    git.git(path, "add", "-A")
    git.git(path, "commit", "-qm", "init")
    return path


@pytest.fixture()
def server(tmp_path):
    app = App(home=tmp_path / "home")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app, None))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield port
    httpd.shutdown()
    app.close()


def _req(port, method, path, body=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as exc:
        raise AssertionError(
            f"{method} {path}: {exc.code} {exc.read().decode()}") from None


def _new_project(port, repo):
    return _req(port, "POST", "/api/projects", {
        "name": "t", "repo_path": str(repo), "baseline_branch": "main",
        "harness": "fake",
        "params": {"eval_cmd": "./eval.sh", "metric_grep": "val_loss=([0-9.]+)",
                   "minimize": True, "k": 2, "b": 1, "baseline_runs": 2,
                   "timeout_s": 60}})


def _wait_state(port, pid, statuses, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        loop = _req(port, "GET", f"/api/projects/{pid}/loop")
        if loop["state"] in statuses:
            return loop
        time.sleep(0.2)
    raise AssertionError(f"loop never reached {statuses}")


def test_health(server):
    out = _req(server, "GET", "/api/health")
    assert out["ok"] and out["version"]


def test_harnesses(server):
    out = _req(server, "GET", "/api/harnesses")
    assert out["fake"]["authed"]
    assert "claude-code" in out


def test_project_crud(server, repo):
    p = _new_project(server, repo)
    assert p["id"] in [x["id"] for x in _req(server, "GET", "/api/projects")]
    assert _req(server, "GET", f"/api/projects/{p['id']}")["params"]["k"] == 2
    _req(server, "DELETE", f"/api/projects/{p['id']}")
    assert _req(server, "GET", "/api/projects") == []


def test_bad_project_body_is_a_400(server):
    with pytest.raises(AssertionError) as e:
        _req(server, "POST", "/api/projects", {"repo_path": "/nope"})
    assert "not a git repository" in str(e.value)


def test_full_manual_generation_over_http(server, repo):
    p = _new_project(server, repo)
    root = _req(server, "POST", f"/api/projects/{p['id']}/baseline", {})
    _wait_state(server, p["id"], {"ready"})
    tree = _req(server, "GET", f"/api/projects/{p['id']}/tree")
    assert tree["nodes"][0]["score"] == 1.0

    out = _req(server, "POST", f"/api/projects/{p['id']}/descend",
               {"parent_id": root["id"], "k": 2})
    assert len(out["nodes"]) == 2
    # wait for both children to be scored
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        tree = _req(server, "GET", f"/api/projects/{p['id']}/tree")
        if all(n["status"] in ("ran", "crash") for n in tree["nodes"]
               if n["gen"] == 1):
            break
        time.sleep(0.2)
    red = _req(server, "POST", f"/api/projects/{p['id']}/reduce", {"gen": 1})
    assert red["frontier"] and red["candidates"] == 2


def test_run_log_endpoint_streams(server, repo):
    p = _new_project(server, repo)
    _req(server, "POST", f"/api/projects/{p['id']}/baseline", {})
    _wait_state(server, p["id"], {"ready"})
    runs = _req(server, "GET", f"/api/projects/{p['id']}/runs")
    out = _req(server, "GET", f"/api/runs/{runs[0]['id']}/log")
    assert "val_loss" in out["text"]
    again = _req(server, "GET",
                 f"/api/runs/{runs[0]['id']}/log?offset={out['offset']}")
    assert again["text"] == ""


def test_metrics_endpoint_reports_kv_and_orchestrator(server, repo):
    p = _new_project(server, repo)
    _req(server, "POST", f"/api/projects/{p['id']}/baseline", {})
    _wait_state(server, p["id"], {"ready"})
    m = _req(server, "GET", f"/api/projects/{p['id']}/metrics")
    assert "orchestrator" in m and "kv" in m
    assert m["orchestrator"].get("forks", 0) == 0


def test_loop_runs_a_full_generation_autonomously(server, repo):
    p = _new_project(server, repo)
    out = _req(server, "POST", f"/api/projects/{p['id']}/loop/start", {})
    assert out["state"] == "running"
    loop = _wait_state(server, p["id"], {"done", "stalled", "error"},
                       timeout=60)
    assert loop["state"] in ("done", "stalled"), loop["message"]


def test_loop_stop(server, repo):
    p = _new_project(server, repo)
    _req(server, "POST", f"/api/projects/{p['id']}/loop/start", {})
    out = _req(server, "POST", f"/api/projects/{p['id']}/loop/stop", {})
    assert out["state"] == "stopping"


def test_unknown_route_is_a_404(server):
    with pytest.raises(AssertionError) as e:
        _req(server, "GET", "/api/nothing")
    assert "404" in str(e.value)


def test_models_endpoint_is_loopback_only(server):
    """GET /api/models fetches {base}/models for the picker's "find" button —
    a remote base would turn the loopback API into an HTTP relay."""
    with pytest.raises(AssertionError) as e:
        _req(server, "GET", "/api/models?base=https://evil.example/v1")
    assert "400" in str(e.value)


def test_cross_origin_posts_are_rejected(server, repo):
    """A malicious web page can POST to a loopback port (no preflight needed
    for plain requests) — every browser sends Origin, so a foreign one is
    refused before the route runs."""
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", server, timeout=30)
    try:
        conn.request("POST", "/api/projects", body=b"{}",
                     headers={"content-type": "application/json",
                              "origin": "https://evil.example"})
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 403
        # same-origin browser POST carries the dashboard's own origin
        conn.request("POST", "/api/projects", body=b"{}",
                     headers={"content-type": "application/json",
                              "origin": f"http://127.0.0.1:{server}"})
        resp = conn.getresponse()
        resp.read()
        assert resp.status != 403
    finally:
        conn.close()


def test_a_token_is_enforced_when_the_server_asks_for_one(tmp_path):
    """Binding off loopback turns on bearer auth; the app only checks when
    a token is configured."""
    app = App(home=tmp_path / "home2", token="sekret")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app, None))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    try:
        with pytest.raises(AssertionError) as e:
            _req(port, "GET", "/api/health")
        assert "403" in str(e.value)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/health",
            headers={"authorization": "Bearer sekret"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert json.loads(resp.read())["ok"]
        # EventSource-style query auth works too
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/health?token=sekret",
                timeout=30) as resp:
            assert json.loads(resp.read())["ok"]
    finally:
        httpd.shutdown()
        app.close()


def test_keepalive_survives_a_post_whose_route_ignores_the_body(server, repo):
    """A body left unread desyncs the connection and the next request on it
    is parsed as ``{}GET /...`` (HTTP 501)."""
    import http.client

    p = _new_project(server, repo)
    conn = http.client.HTTPConnection("127.0.0.1", server, timeout=30)
    try:
        for method, path in [("POST", f"/api/projects/{p['id']}/baseline"),
                             ("GET", f"/api/projects/{p['id']}"),
                             ("POST", f"/api/projects/{p['id']}/loop/stop"),
                             ("GET", f"/api/projects/{p['id']}/runs")]:
            conn.request(method, path, body=b"{}",
                         headers={"content-type": "application/json"})
            resp = conn.getresponse()
            body = resp.read()
            assert resp.status == 200, (method, path, resp.status, body[:200])
            assert resp.getheader("content-type") == "application/json"
    finally:
        conn.close()
