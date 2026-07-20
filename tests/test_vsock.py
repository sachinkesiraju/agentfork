"""End-to-end tests for the exec channel: the real ``VsockExecClient``
against the real ``guest_agent`` request handler, joined by a fake
Firecracker vsock muxer over an AF_UNIX socket.

Only the muxer is fake: it speaks Firecracker's ``CONNECT <port>`` /
``OK <hostport>`` handshake and then hands the byte stream to
``guest_agent.handle_connection``, which runs real subprocesses. What is NOT
covered here: AF_VSOCK itself and a real VMM's muxer — those need a guest.
"""

import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time

import pytest

from agentfork.sandbox import guest_agent
from agentfork.sandbox.vsock import VsockError, VsockExecClient


@pytest.fixture
def short_dir():
    """pytest's tmp_path exceeds AF_UNIX's ~104-byte path limit on macOS;
    sockets need a short-lived short path instead."""
    d = tempfile.mkdtemp(prefix="afv-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


class FakeMuxer:
    """Firecracker-style vsock muxer over AF_UNIX. Connections asking for
    ``guest_port`` reach the real guest agent; any other port is refused the
    way Firecracker refuses a port with no guest listener (connection
    closed, no OK)."""

    def __init__(self, uds_path: str, guest_port: int = 52):
        self.guest_port = guest_port
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(uds_path)
        self.listener.listen(8)
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return  # closed
            line = b""
            while not line.endswith(b"\n") and len(line) < 64:
                chunk = conn.recv(64)
                if not chunk:
                    break
                line += chunk
            if line.strip() == f"CONNECT {self.guest_port}".encode():
                conn.sendall(b"OK 1024\n")
                threading.Thread(target=guest_agent.handle_connection,
                                 args=(conn,), daemon=True).start()
            else:
                conn.close()

    def close(self):
        self.listener.close()


@pytest.fixture
def channel(short_dir):
    uds = os.path.join(short_dir, "v.sock")
    muxer = FakeMuxer(uds)
    yield VsockExecClient(uds, port=52, handshake_timeout_s=5.0)
    muxer.close()


def test_exec_runs_command_and_returns_stdout(channel):
    result = channel.exec([sys.executable, "-c", "print('hi')"])
    assert result.exit_code == 0
    assert result.stdout == b"hi\n"
    assert result.stderr == b""
    assert result.timed_out is False


def test_exec_reports_exit_code_and_stderr(channel):
    result = channel.exec([sys.executable, "-c",
                           "import sys; sys.stderr.write('err'); sys.exit(3)"])
    assert result.exit_code == 3
    assert result.stderr == b"err"


def test_exec_binary_safe_output(channel):
    result = channel.exec([sys.executable, "-c",
                           "import sys; sys.stdout.buffer.write(bytes(range(256)))"])
    assert result.stdout == bytes(range(256))


def test_exec_timeout_kills_guest_command(channel):
    result = channel.exec([sys.executable, "-c", "import time; time.sleep(60)"],
                          timeout_s=0.2)
    assert result.timed_out is True
    assert result.exit_code != 0


def test_exec_spawn_failure_raises_vsock_error(channel):
    with pytest.raises(VsockError, match="guest agent error"):
        channel.exec(["/definitely/not/a/binary"])


def test_malformed_agent_reply_raises_vsock_error(short_dir):
    # A valid-JSON reply missing the exec keys must surface as VsockError (the
    # contract callers catch), not a bare KeyError leaking out.
    uds = os.path.join(short_dir, "v.sock")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(uds)
    listener.listen(8)

    def serve():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            for _ in range(2):  # drain CONNECT, then the request line
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = conn.recv(256)
                    if not chunk:
                        break
                    buf += chunk
                if _ == 0:
                    conn.sendall(b"OK 1024\n")
            conn.sendall(b'{"unexpected": true}\n')
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    try:
        client = VsockExecClient(uds, port=52, handshake_timeout_s=5.0)
        with pytest.raises(VsockError, match="incomplete agent exec reply"):
            client.exec(["true"])
    finally:
        listener.close()


def test_exec_empty_argv_rejected_client_side(channel):
    with pytest.raises(ValueError):
        channel.exec([])


def test_port_with_no_guest_listener_fails_handshake(short_dir):
    uds = os.path.join(short_dir, "v.sock")
    muxer = FakeMuxer(uds, guest_port=52)
    try:
        client = VsockExecClient(uds, port=99, handshake_timeout_s=0.3)
        with pytest.raises(VsockError, match="handshake"):
            client.exec(["true"])
    finally:
        muxer.close()


def test_missing_uds_fails_after_retry_deadline(short_dir):
    client = VsockExecClient(os.path.join(short_dir, "nope.sock"),
                             handshake_timeout_s=0.2)
    with pytest.raises(VsockError, match="handshake"):
        client.exec(["true"])


def test_exec_feeds_stdin_to_the_command(channel):
    result = channel.exec([sys.executable, "-c",
                           "import sys; sys.stdout.write(sys.stdin.read().upper())"],
                          stdin=b"fed via vsock")
    assert result.exit_code == 0
    assert result.stdout == b"FED VIA VSOCK"


def test_exec_detached_starts_a_background_process(channel):
    import time

    marker = os.path.join(tempfile.gettempdir(), f"afv-{os.getpid()}.marker")
    detached = None
    try:
        detached = channel.exec_detached(
            [sys.executable, "-c",
             "import time; time.sleep(0.1); print('bg-done'); "
             f"open({marker!r}, 'w').write('x')"])

        assert detached.pid > 0
        assert detached.log_path  # guest-side path for tail-follow
        # returns before the process finishes...
        assert not os.path.exists(marker)
        # ...and the process runs to completion, its stdout landing in the
        # log. Poll the log *content*: a child's block-buffered stdout only
        # flushes at exit, which can trail the marker write, so waiting on
        # the marker and reading once races (empty read on a slow runner).
        deadline = time.monotonic() + 5
        contents = b""
        while time.monotonic() < deadline:
            try:
                with open(detached.log_path, "rb") as f:
                    contents = f.read()
            except FileNotFoundError:
                contents = b""
            if contents == b"bg-done\n":
                break
            time.sleep(0.02)
        assert contents == b"bg-done\n"
        assert os.path.exists(marker)
    finally:
        for path in (marker, detached.log_path if detached else None):
            if path and os.path.exists(path):
                os.unlink(path)


def test_detach_with_stdin_is_rejected(channel):
    with pytest.raises(VsockError, match="mutually exclusive"):
        channel._roundtrip({"argv": ["true"], "detach": True,
                            "stdin": "aGk="}, 5.0)


def test_concurrent_execs_are_served_concurrently(channel):
    results = [None] * 4
    def run(i):
        results[i] = channel.exec(
            [sys.executable, "-c", f"import time; time.sleep(0.3); print({i})"])
    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    assert [r.stdout for r in results] == [b"0\n", b"1\n", b"2\n", b"3\n"]
    # four 0.3s sleeps served serially take >=1.2s; generous margin for CI
    assert elapsed < 1.0


def test_guest_agent_rejects_non_string_argv():
    client, server = socket.socketpair()
    thread = threading.Thread(
        target=guest_agent.handle_connection, args=(server,))
    thread.start()
    client.sendall(
        json.dumps({"argv": [123], "timeout_s": None}).encode() + b"\n")
    reply = json.loads(client.makefile().readline())
    thread.join(1)
    client.close()

    assert "non-empty strings" in reply["error"]
