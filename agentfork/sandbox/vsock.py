"""Host side of the guest exec channel: Firecracker vsock + one-line JSON.

Firecracker exposes guest vsock ports through a host Unix socket (the
"muxer"): a host client connects to the UDS, writes ``CONNECT <port>\\n``,
and, if a guest listener is bound to that port, receives ``OK <hostport>\\n``
after which the stream is a plain byte pipe to the guest. This module speaks
that handshake and the exec protocol ``guest_agent.py`` serves over it:

- request: one JSON line ``{"argv": [...], "timeout_s": <float|null>}``;
- response: one JSON line ``{"exit_code": <int>, "stdout": <b64>,
  "stderr": <b64>, "timed_out": <bool>}``, or ``{"error": <str>}`` if the
  agent could not run the command at all.

The guest command's timeout is enforced by the agent inside the guest; the
host socket timeout is set slightly above it so a wedged guest still cannot
hang the caller forever. A handshake retry loop absorbs the window right
after boot when the VMM is up but the agent has not bound its port yet.

Tested against a fake muxer+agent over a real Unix socket
(tests/test_vsock.py); the handshake grammar follows Firecracker's vsock
documentation and has not itself been validated against a live VMM.
"""

from __future__ import annotations

import base64
import json
import socket
import time
from dataclasses import dataclass

DEFAULT_PORT = 52
_MAX_LINE = 64 * 1024 * 1024  # bound response size: 64 MiB of JSON+base64
_HOST_GRACE_S = 10.0          # host waits this much past the guest timeout


class VsockError(RuntimeError):
    """Handshake or protocol failure on the host-guest exec channel."""


@dataclass
class ExecResult:
    exit_code: int
    stdout: bytes
    stderr: bytes
    timed_out: bool = False


def _read_line(sock: socket.socket, limit: int = _MAX_LINE) -> bytes:
    """Read until ``\\n`` (exclusive). Raises on EOF or an oversized line."""
    chunks = []
    size = 0
    while True:
        b = sock.recv(65536)
        if not b:
            raise VsockError("peer closed the connection mid-line")
        nl = b.find(b"\n")
        if nl != -1:
            if b[nl + 1:]:
                raise VsockError("unexpected data after response line")
            chunks.append(b[:nl])
            return b"".join(chunks)
        chunks.append(b)
        size += len(b)
        if size > limit:
            raise VsockError(f"response line exceeds {limit} bytes")


class VsockExecClient:
    """Runs commands in one guest, addressed by its vsock muxer UDS path."""

    def __init__(self, uds_path: str, port: int = DEFAULT_PORT,
                 handshake_timeout_s: float = 10.0):
        self.uds_path = uds_path
        self.port = port
        self.handshake_timeout_s = handshake_timeout_s

    def _connect(self) -> socket.socket:
        """Connect and complete the muxer handshake, retrying while the VMM
        or the guest agent is still coming up."""
        deadline = time.monotonic() + self.handshake_timeout_s
        last: Exception = VsockError("handshake never attempted")
        while True:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(max(deadline - time.monotonic(), 0.001))
            try:
                sock.connect(self.uds_path)
                sock.sendall(f"CONNECT {self.port}\n".encode())
                reply = _read_line(sock, limit=256)
                if not reply.startswith(b"OK "):
                    raise VsockError(
                        f"vsock handshake refused: {reply[:80]!r}")
                return sock
            except (OSError, VsockError) as exc:
                sock.close()
                last = exc
                if time.monotonic() >= deadline:
                    raise VsockError(
                        f"vsock handshake to {self.uds_path}:{self.port} "
                        f"failed: {last}") from last
                time.sleep(0.1)

    def exec(self, argv: list[str], timeout_s: float | None = None) -> ExecResult:
        if not argv:
            raise ValueError("argv must not be empty")
        sock = self._connect()
        try:
            if timeout_s is not None:
                sock.settimeout(timeout_s + _HOST_GRACE_S)
            else:
                sock.settimeout(None)
            request = json.dumps({"argv": argv, "timeout_s": timeout_s})
            sock.sendall(request.encode() + b"\n")
            try:
                line = _read_line(sock)
            except socket.timeout:
                raise VsockError(
                    f"guest did not answer within {timeout_s}s + grace; "
                    "agent dead or guest wedged") from None
            try:
                reply = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VsockError(f"malformed agent reply: {line[:80]!r}") from exc
            if "error" in reply:
                raise VsockError(f"guest agent error: {reply['error']}")
            return ExecResult(
                exit_code=reply["exit_code"],
                stdout=base64.b64decode(reply["stdout"]),
                stderr=base64.b64decode(reply["stderr"]),
                timed_out=reply.get("timed_out", False),
            )
        finally:
            sock.close()
