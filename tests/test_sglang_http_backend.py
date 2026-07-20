import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agentfork import ForkOrchestrator, SGLangHTTPBackend


class Stub:
    def __init__(self):
        self.operations = []
        self.paths = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                size = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(size))
                stub.paths.append(self.path)
                stub.operations.append(body)
                if self.path == "/tree_generate":
                    payload = {"text": "ok", "meta_info": {"cached_tokens": 3}}
                else:
                    payload = {
                        "success": True,
                        "value": (
                            {"live_branches": 2}
                            if body["operation"] == "telemetry"
                            else 7 if body["operation"] == "kill" else None
                        ),
                    }
                encoded = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()


def test_sglang_http_backend_drives_tree_lifecycle():
    stub = Stub()
    try:
        backend = SGLangHTTPBackend(stub.url)
        backend.create_tree("root")
        assert backend.fork_branch("root", "child").branch_id == "child"
        backend.reserve("child", 128)
        assert backend.telemetry("root") == {"live_branches": 2}
        assert backend.kill("child") == 7
        assert [operation["operation"] for operation in stub.operations] == [
            "create",
            "fork",
            "reserve",
            "telemetry",
            "kill",
        ]
    finally:
        stub.close()


class ErrorStub(Stub):
    """Same endpoint, but every operation is answered with a fixed
    (status, payload) response instead of success."""

    def __init__(self, status, payload):
        self.status_payload = (status, payload)
        super().__init__()
        stub = self
        parent_handler = self.server.RequestHandlerClass

        class Handler(parent_handler):
            def do_POST(self):
                size = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(size))
                stub.operations.append(body)
                stub.auth = self.headers.get("Authorization")
                status, payload = stub.status_payload
                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server.RequestHandlerClass = Handler


def test_api_key_is_sent_as_bearer_token():
    stub = ErrorStub(200, {"success": True})
    try:
        SGLangHTTPBackend(stub.url, api_key="sk-test").create_tree("root")
        assert stub.auth == "Bearer sk-test"
    finally:
        stub.close()


def test_tree_lifecycle_and_generation_use_admin_key():
    stub = ErrorStub(200, {"success": True})
    try:
        backend = SGLangHTTPBackend(
            stub.url, api_key="user-key", admin_api_key="admin-key")
        backend.create_tree("root")
        assert stub.auth == "Bearer admin-key"
        backend.generate("root", "prompt", {"max_new_tokens": 1})
        assert stub.auth == "Bearer admin-key"
    finally:
        stub.close()


def test_unsuccessful_payload_raises_with_server_message():
    stub = ErrorStub(200, {"success": False, "message": "capacity exceeded"})
    try:
        with pytest.raises(RuntimeError, match="capacity exceeded"):
            SGLangHTTPBackend(stub.url).kill("ghost")
    finally:
        stub.close()


def test_kill_of_already_gone_branch_is_idempotent():
    # kill is retried; a retry after a lost success response hits an
    # already-deleted branch. "no such" on a kill is treated as success, not
    # a spurious failure, so the retry does not report the kill as failed.
    stub = ErrorStub(200, {"success": False, "message": "no such tree"})
    try:
        assert SGLangHTTPBackend(stub.url).kill("ghost") == 0
    finally:
        stub.close()


def test_http_error_raises_with_status_and_detail():
    stub = ErrorStub(503, {"error": "scheduler overloaded"})
    try:
        with pytest.raises(RuntimeError, match="HTTP 503"):
            SGLangHTTPBackend(stub.url).create_tree("root")
    finally:
        stub.close()


def test_fork_requires_explicit_child_id():
    with pytest.raises(ValueError, match="child_id"):
        SGLangHTTPBackend("http://127.0.0.1:1").fork_branch("root")


def test_extend_is_a_local_stub_that_never_calls_the_server():
    with pytest.raises(RuntimeError, match="generate"):
        SGLangHTTPBackend("http://127.0.0.1:1").extend("root", [1, 2])


def test_empty_base_url_rejected():
    with pytest.raises(ValueError):
        SGLangHTTPBackend("")


def test_orchestrator_coordinates_remote_lifecycle_and_generate():
    stub = Stub()
    try:
        backend = SGLangHTTPBackend(stub.url)
        with ForkOrchestrator(kv=backend) as orch:
            orch.create_parent("root")
            child = orch.fork(
                "root", child_ids=["root/child"])[0]
            result = orch.generate(
                child.branch_id,
                "prompt",
                {"max_new_tokens": 2},
                reserve_tokens=2,
            )

        assert result["text"] == "ok"
        generate = stub.operations[2]
        assert stub.paths[2] == "/tree_generate"
        assert generate["tree_id"] == "root"
        assert generate["branch_id"] == "root/child"
        assert generate["parent_id"] == "root"
    finally:
        stub.close()


def test_remote_backend_rejects_local_token_extend_path():
    backend = SGLangHTTPBackend("http://127.0.0.1:1")
    orch = ForkOrchestrator(kv=backend)

    with pytest.raises(RuntimeError, match="generate"):
        orch.create_parent("root", tokens=[1, 2])

    assert orch.branches() == []


def test_remote_control_requires_tls_and_key():
    with pytest.raises(ValueError, match="HTTPS"):
        SGLangHTTPBackend("http://engine.example.com")
    with pytest.raises(ValueError, match="API key"):
        SGLangHTTPBackend("https://engine.example.com")


def test_kill_retries_transient_http_errors():
    stub = Stub()
    original = stub.server.RequestHandlerClass
    attempts = [0]

    class Handler(original):
        def do_POST(self):
            size = int(self.headers.get("Content-Length", 0))
            json.loads(self.rfile.read(size))
            attempts[0] += 1
            if attempts[0] == 1:
                encoded = b'{"error":"busy"}'
                self.send_response(503)
            else:
                encoded = b'{"success":true,"value":0}'
                self.send_response(200)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    stub.server.RequestHandlerClass = Handler
    try:
        backend = SGLangHTTPBackend(
            stub.url, retry_delay=0, max_retries=1)
        assert backend.kill("ghost") == 0
        assert attempts[0] == 2
    finally:
        stub.close()


def test_kill_waits_for_in_flight_generate():
    stub = Stub()
    backend = SGLangHTTPBackend(stub.url)
    backend.create_tree("root")
    backend.fork_branch("root", "child")
    started = threading.Event()
    release = threading.Event()
    kill_received = threading.Event()
    original = stub.server.RequestHandlerClass

    class Handler(original):
        def do_POST(self):
            size = int(self.headers.get("Content-Length", 0))
            json.loads(self.rfile.read(size))
            if self.path == "/tree_generate":
                started.set()
                release.wait(2)
                payload = {"text": "ok"}
            else:
                kill_received.set()
                payload = {"success": True, "value": 0}
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    stub.server.RequestHandlerClass = Handler
    generate = threading.Thread(
        target=backend.generate,
        args=("child", "prompt", {"max_new_tokens": 1}),
    )
    kill = threading.Thread(target=backend.kill, args=("child",))
    try:
        generate.start()
        assert started.wait(1)
        kill.start()
        time.sleep(0.05)
        assert not kill_received.is_set()
        with pytest.raises(RuntimeError, match="being killed"):
            backend.generate("child", "late", {"max_new_tokens": 1})
        release.set()
        generate.join(2)
        kill.join(2)
        assert kill_received.is_set()
    finally:
        release.set()
        generate.join(2)
        kill.join(2)
        stub.close()
