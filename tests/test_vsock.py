"""End-to-end tests for the exec channel: the real ``VsockExecClient``
against the real ``guest_agent`` request handler, joined by a fake
Firecracker vsock muxer over an AF_UNIX socket.

Only the muxer is fake: it speaks Firecracker's ``CONNECT <port>`` /
``OK <hostport>`` handshake and then hands the byte stream to
``guest_agent.handle_connection``, which runs real subprocesses. What is NOT
covered here: AF_VSOCK itself and a real VMM's muxer — those need a guest.
"""

import os
import shutil
import socket
import sys
import tempfile
import threading

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


def test_concurrent_execs_are_served_concurrently(channel):
    results = [None] * 4
    def run(i):
        results[i] = channel.exec(
            [sys.executable, "-c", f"import time; time.sleep(0.2); print({i})"])
    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    import time
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    assert [r.stdout for r in results] == [b"0\n", b"1\n", b"2\n", b"3\n"]
    # four 0.2s sleeps served serially would take >=0.8s
    assert elapsed < 0.7
