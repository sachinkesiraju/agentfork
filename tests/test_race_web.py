"""The browser dashboard: does it serve, and does it stream the race?"""

import argparse
import json
import threading
import urllib.request

import pytest

race_web = pytest.importorskip("demo.race_web")


class FakeInner:
    def __init__(self):
        self.lines = []

    def say(self, msg=""):
        self.lines.append(msg)

    def arm_done(self, arm):
        self.lines.append(arm)


class FakeArm:
    name = "agentfork"
    parent_hit_rate = 1.0
    prefill_charged = 11238
    verified = True


@pytest.fixture
def ui():
    args = argparse.Namespace(capacity_tokens=24576)
    web = race_web.WebUI(args, FakeInner(), port=0, open_browser=False)
    yield web
    web.httpd.shutdown()
    web.httpd.server_close()


def get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=10) as r:
        assert r.status == 200
        return r.read().decode()


def test_page_serves_the_dashboard(ui):
    page = get(ui.url + "/")
    assert "EventSource(\"/events\")" in page
    assert "AGENTFORK" in page and "STOCK" in page


def test_unknown_path_is_404(ui):
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(ui.url + "/nope")
    assert exc.value.code == 404


def test_events_stream_replays_then_follows_live(ui):
    ui.kv("stock", 1234)                      # happens before the browser opens
    received = []
    ready = threading.Event()

    def read_stream():
        with urllib.request.urlopen(ui.url + "/events", timeout=15) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                received.append(event)
                ready.set()
                if event["kind"] == "done":
                    return

    t = threading.Thread(target=read_stream, daemon=True)
    t.start()
    assert ready.wait(10), "no replayed event"

    ui.arm_done(FakeArm())
    ui.finish()
    t.join(timeout=10)

    kinds = [e["kind"] for e in received]
    assert kinds == ["kv", "arm_done", "done"]
    assert received[0] == {"kind": "kv", "name": "stock", "used": 1234,
                           "capacity": 24576}
    assert received[1]["hit_rate"] == 1.0
    assert received[1]["prefill"] == 11238
