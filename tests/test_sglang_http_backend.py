import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agentfork import SGLangHTTPBackend


class Stub:
    def __init__(self):
        self.operations = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                size = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(size))
                stub.operations.append(body)
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


def test_unsuccessful_payload_raises_with_server_message():
    stub = ErrorStub(200, {"success": False, "message": "no such tree"})
    try:
        with pytest.raises(RuntimeError, match="no such tree"):
            SGLangHTTPBackend(stub.url).kill("ghost")
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
    # remote extend/accounting is undesigned; the stub must stay off the wire
    assert SGLangHTTPBackend("http://127.0.0.1:1").extend("root", [1, 2]) == 0


def test_empty_base_url_rejected():
    with pytest.raises(ValueError):
        SGLangHTTPBackend("")
