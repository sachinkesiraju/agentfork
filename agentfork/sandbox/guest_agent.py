"""Guest side of the exec channel. Copy this single file into the rootfs.

Listens on a vsock port (default 52) inside the guest and serves the
protocol ``agentfork.sandbox.vsock`` speaks from the host: one JSON request
line ``{"argv": [...], "timeout_s": <float|null>}`` per connection, answered
with one JSON line carrying exit code and base64 stdout/stderr, or
``{"error": ...}`` if the command could not be started. Each connection is
one command; concurrent connections run concurrently.

Deliberately standalone: stdlib only, no agentfork imports, one file, so
baking it into a guest image is ``cp guest_agent.py`` plus an init line such
as ``python3 /opt/guest_agent.py &``. A listening socket bound before a
snapshot keeps listening in every VM restored from that snapshot; active
connections do not survive a snapshot, which is why the host side holds
connections only for the duration of one command.

Run: ``python3 guest_agent.py [port]``.

``serve()`` takes any listening socket, so tests drive the full protocol
over AF_UNIX without a guest (tests/test_vsock.py); AF_VSOCK binding is
exercised only inside a real guest.
"""

from __future__ import annotations

import base64
import json
import socket
import subprocess
import sys
import threading

DEFAULT_PORT = 52
_MAX_REQUEST = 1024 * 1024  # 1 MiB of argv is already absurd


def _read_line(conn: socket.socket, limit: int = _MAX_REQUEST) -> bytes | None:
    """Read until ``\\n`` (exclusive); None on clean EOF before any data."""
    chunks = []
    size = 0
    while True:
        b = conn.recv(65536)
        if not b:
            return None if not chunks else b"".join(chunks)
        nl = b.find(b"\n")
        if nl != -1:
            chunks.append(b[:nl])
            return b"".join(chunks)
        chunks.append(b)
        size += len(b)
        if size > limit:
            raise ValueError(f"request line exceeds {limit} bytes")


def handle_connection(conn: socket.socket) -> None:
    """Serve one command on one connection, then close it."""
    try:
        try:
            line = _read_line(conn)
            if not line:
                return
            request = json.loads(line)
            argv = request["argv"]
            timeout_s = request.get("timeout_s")
            if not isinstance(argv, list) or not argv:
                raise ValueError("argv must be a non-empty list")
        except (ValueError, KeyError, TypeError) as exc:
            conn.sendall(json.dumps({"error": f"bad request: {exc}"}).encode() + b"\n")
            return
        try:
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            conn.sendall(json.dumps({"error": f"spawn failed: {exc}"}).encode() + b"\n")
            return
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            stdout, stderr = proc.communicate()
        reply = {
            "exit_code": proc.returncode,
            "stdout": base64.b64encode(stdout).decode(),
            "stderr": base64.b64encode(stderr).decode(),
            "timed_out": timed_out,
        }
        conn.sendall(json.dumps(reply).encode() + b"\n")
    except OSError:
        pass  # peer vanished mid-command; nothing useful left to do
    finally:
        conn.close()


def serve(listener: socket.socket) -> None:
    """Accept forever on an already-listening socket, one thread per
    connection. The socket's family is the caller's business: AF_VSOCK in a
    guest, AF_UNIX in tests."""
    while True:
        conn, _ = listener.accept()
        threading.Thread(target=handle_connection, args=(conn,),
                         daemon=True).start()


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    listener = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)  # type: ignore[attr-defined]
    listener.bind((socket.VMADDR_CID_ANY, port))  # type: ignore[attr-defined]
    listener.listen(16)
    serve(listener)
    return 0


if __name__ == "__main__":
    sys.exit(main())
