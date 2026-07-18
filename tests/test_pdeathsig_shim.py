"""Thread-safe ``pdeathsig='shim'`` reaper mode and the launcher itself.

The shim arms ``PR_SET_PDEATHSIG`` without a ``preexec_fn``, so spawns are
safe to run from many threads at once and ``ReaperSandbox`` may fan out.
These tests need real Linux process/pidfd support."""

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from agentfork.kill.reaper import BranchReaper
from agentfork.orchestrator import ForkOrchestrator, ReaperSandbox

pytestmark = pytest.mark.skipif(
    not BranchReaper.supported() or not os.path.exists("/proc"),
    reason="needs Linux pidfd + /proc")


def test_thread_safe_flag_by_mode():
    assert BranchReaper(pdeathsig=True).thread_safe is False
    assert BranchReaper(pdeathsig="shim").thread_safe is True
    assert BranchReaper(pdeathsig=False).thread_safe is True


def test_invalid_pdeathsig_mode_rejected():
    with pytest.raises(ValueError, match="pdeathsig"):
        BranchReaper(pdeathsig="nonsense")


def test_reaper_sandbox_parallel_lifecycle_follows_mode():
    assert ReaperSandbox(["true"], pdeathsig=True).parallel_lifecycle is False
    assert ReaperSandbox(["true"], pdeathsig="shim").parallel_lifecycle is True
    assert ReaperSandbox(["true"], pdeathsig=False).parallel_lifecycle is True


def test_shim_launcher_execs_the_real_command():
    # the launcher must replace itself with the target command (pid is stable
    # across execvp), so the recorded pid is the real process
    marker = f"/tmp/afpd-{os.getpid()}-{time.time_ns()}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "agentfork.kill._pdeathsig", str(os.getpid()),
         sys.executable, "-c", f"open({marker!r}, 'w').write('ran')"])
    try:
        proc.wait(timeout=10)
        assert proc.returncode == 0
        with open(marker) as f:
            assert f.read() == "ran"
    finally:
        if os.path.exists(marker):
            os.unlink(marker)


def test_shim_launcher_bails_when_parent_already_gone():
    # a wrong expected_ppid stands in for "supervisor died during handoff":
    # the launcher must refuse to exec the command and exit non-zero
    proc = subprocess.Popen(
        [sys.executable, "-m", "agentfork.kill._pdeathsig", "999999999",
         sys.executable, "-c", "import sys; sys.exit(0)"],
        stderr=subprocess.DEVNULL)
    proc.wait(timeout=10)
    assert proc.returncode == 128


def test_shim_reaper_spawns_kills_and_reaps():
    reaper = BranchReaper(pdeathsig="shim")
    reaper.spawn("b", [sys.executable, "-c", "import time; time.sleep(60)"])
    assert reaper.alive("b") is True
    reaper.kill("b")
    assert reaper._branches == {}


def test_shim_reaper_spawns_are_thread_safe_and_concurrent():
    reaper = BranchReaper(pdeathsig="shim")
    argv = [sys.executable, "-c", "import time; time.sleep(30)"]
    try:
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda i: reaper.spawn(f"b{i}", argv), range(16)))
        assert len(reaper._branches) == 16
        assert all(reaper.alive(f"b{i}") for i in range(16))
    finally:
        reaper.close()


def test_orchestrator_fans_out_reaper_sandbox_in_shim_mode():
    sandbox = ReaperSandbox(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        pdeathsig="shim")
    with ForkOrchestrator(sandbox=sandbox) as orch:
        orch.create_parent("root", tokens=list(range(64)))
        children = orch.fork("root", n=8)
        assert len(children) == 8
        assert all(orch.alive(c.branch_id) for c in children)


def test_shim_pdeathsig_kills_orphan_when_supervisor_dies():
    # a grandchild spawned via the shim must die when its supervisor (the
    # child process) is SIGKILLed, proving the death signal survived execve
    supervisor = subprocess.Popen([sys.executable, "-c", """
import subprocess, sys, time, os
p = subprocess.Popen([sys.executable, "-m", "agentfork.kill._pdeathsig",
                      str(os.getpid()),
                      sys.executable, "-c", "import time; time.sleep(60)"])
print(p.pid, flush=True)
time.sleep(60)
"""], stdout=subprocess.PIPE, env={**os.environ,
      "PYTHONPATH": os.getcwd()})
    try:
        grandchild_pid = int(supervisor.stdout.readline())
        # let the shim arm pdeathsig and exec the sleeper
        time.sleep(1.5)
        assert os.path.exists(f"/proc/{grandchild_pid}")  # alive
        supervisor.kill()
        supervisor.wait(timeout=5)
        # the kernel delivers SIGKILL to the grandchild on supervisor death
        deadline = time.monotonic() + 5
        while os.path.exists(f"/proc/{grandchild_pid}") and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not os.path.exists(f"/proc/{grandchild_pid}")
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()
