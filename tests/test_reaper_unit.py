import os
import signal
import subprocess

import pytest

from agentfork.kill.reaper import BranchReaper


class FakeProcess:
    def __init__(self, pid=123, returncode=None):
        self.pid = pid
        self.returncode = returncode
        self.killed = False
        self.waited = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -signal.SIGKILL

    def wait(self):
        self.waited = True
        return self.returncode


def test_spawn_cleans_up_when_pidfd_open_fails(monkeypatch):
    proc = FakeProcess()
    monkeypatch.setattr(BranchReaper, "supported", staticmethod(lambda: True))
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(
        os, "pidfd_open", lambda pid: (_ for _ in ()).throw(OSError("boom")),
        raising=False)

    with pytest.raises(OSError, match="boom"):
        BranchReaper().spawn("branch", ["command"])

    assert proc.killed
    assert proc.waited


def test_spawn_rejects_duplicate_branch_before_starting_process(monkeypatch):
    monkeypatch.setattr(BranchReaper, "supported", staticmethod(lambda: True))
    reaper = BranchReaper()
    reaper._branches["branch"] = (FakeProcess(), 1)
    called = False

    def popen(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(subprocess, "Popen", popen)

    with pytest.raises(ValueError, match="branch exists"):
        reaper.spawn("branch", ["command"])

    assert not called


def test_kill_closes_pidfd_and_reaps_on_waitid_failure(monkeypatch):
    proc = FakeProcess()
    reaper = BranchReaper()
    reaper._branches["branch"] = (proc, 7)
    closed = []
    monkeypatch.setattr(signal, "pidfd_send_signal", lambda *args: None, raising=False)
    monkeypatch.setattr(os, "P_PIDFD", 3, raising=False)
    monkeypatch.setattr(os, "WEXITED", 4, raising=False)
    monkeypatch.setattr(os, "waitid", lambda *args: (_ for _ in ()).throw(OSError("boom")))
    monkeypatch.setattr(os, "close", closed.append)

    with pytest.raises(OSError, match="boom"):
        reaper.kill("branch")

    assert proc.killed
    assert proc.waited
    assert closed == [7]
    assert "branch" not in reaper._branches
