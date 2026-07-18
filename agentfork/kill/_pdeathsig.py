"""Thread-safe ``PR_SET_PDEATHSIG`` launcher.

``BranchReaper``'s orphan backstop arms ``PR_SET_PDEATHSIG`` so a branch
process dies with the supervisor. The obvious way to set it — a
``subprocess`` ``preexec_fn`` — runs Python between ``fork()`` and
``exec()`` in the child, which CPython documents as unsafe when other
threads exist in the parent (they may hold locks the fork child can never
release). That is exactly the case under a threaded supervisor, and it is
why ``ReaperSandbox`` otherwise refuses to fan out spawns in parallel.

This module is that launcher instead. The reaper spawns

    python -m agentfork.kill._pdeathsig <expected_ppid> <cmd> [args...]

with no ``preexec_fn``: the ``fork`` child immediately ``exec``s the Python
interpreter (no parent-thread state is touched), and only then — in a
normal, single-threaded process — does this code set the parent-death
signal and ``exec`` the real command. ``PR_SET_PDEATHSIG`` survives that
``execve`` for ordinary (non-SUID/SGID/file-capability) binaries, so the
backstop stays armed.

The parent may have died during the brief window before the signal was
armed; the ``getppid()`` check catches that and exits rather than
orphaning. Cost is one interpreter startup per spawn (tens of ms), traded
for spawn thread-safety.
"""

from __future__ import annotations

import ctypes
import os
import signal
import sys

PR_SET_PDEATHSIG = 1


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 2:
        print("usage: _pdeathsig <expected_ppid> <cmd> [args...]",
              file=sys.stderr)
        return 2
    expected_ppid = int(argv[0])
    cmd = argv[1:]

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        return 127
    # the supervisor may have exited between its fork and this point; the
    # death signal would then never fire, so bail instead of orphaning
    if os.getppid() != expected_ppid:
        return 128
    try:
        os.execvp(cmd[0], cmd)
    except OSError as exc:
        print(f"_pdeathsig: exec {cmd[0]!r} failed: {exc}", file=sys.stderr)
        return 126
    return 0  # unreachable after a successful execvp


if __name__ == "__main__":
    sys.exit(main())
