"""Unit tests for agentfork.sandbox.firecracker_backend.FirecrackerSandbox.

Exercises the adapter's spawn/kill/alive/exec translation against fakes
standing in for ``agentfork.sandbox.fc_bench.MicroVM`` and the vsock exec
client. ``MicroVM.__init__`` launches a real Firecracker subprocess, so a
real ``MicroVM`` is never constructed here -- every test injects a fake
factory via the ``microvm_factory`` constructor argument, and a fake exec
client via ``exec_client_factory``.
"""

import os

import pytest

from agentfork.sandbox.firecracker_backend import FirecrackerSandbox
from agentfork.sandbox.vsock import ExecResult


class FakeProc:
    """Stands in for subprocess.Popen: poll() toggles running vs exited."""

    def __init__(self):
        self._poll_value = None  # None == still running, like a real Popen
        self.pid = 54321

    def poll(self):
        return self._poll_value


class FakeMicroVM:
    """Same constructor signature as MicroVM(fc_bin, vm_dir); no subprocess."""

    def __init__(self, fc_bin: str, vm_dir: str):
        self.fc_bin = fc_bin
        self.vm_dir = vm_dir
        self.events = []
        self.proc = FakeProc()

    def boot(self, kernel, rootfs, overlay=None, vsock_uds=None):
        self.events.append(("boot", kernel, rootfs, overlay, vsock_uds))
        return 0.001

    def pause(self):
        self.events.append(("pause",))

    def resume(self):
        self.events.append(("resume",))

    def snapshot(self, mem_path, state_path):
        self.events.append(("snapshot", mem_path, state_path))
        # the real VMM writes these files; forks-from-disk depend on them
        for path in (mem_path, state_path):
            with open(path, "wb") as f:
                f.write(b"snap")
        return 0.001

    def restore(self, mem_path, state_path):
        self.events.append(("restore", mem_path, state_path))
        return 0.001

    def kill(self):
        self.events.append(("kill",))
        self.proc._poll_value = 0
        return 0.001, 0.001


class FakeMicroVMFactory:
    """Records every fake VM it creates, in creation order, keyed by vm_dir."""

    def __init__(self):
        self.instances = []
        self.by_dir = {}

    def __call__(self, fc_bin, vm_dir):
        vm = FakeMicroVM(fc_bin, vm_dir)
        self.instances.append(vm)
        self.by_dir[vm_dir] = vm
        return vm


class FailingBootMicroVM(FakeMicroVM):
    """Fails boot(), the way a real MicroVM would on a bad kernel/rootfs."""

    def boot(self, kernel, rootfs, overlay=None, vsock_uds=None):
        self.events.append(("boot", kernel, rootfs, overlay, vsock_uds))
        raise RuntimeError("Firecracker PUT /actions returned HTTP 400")


class FailingBootMicroVMFactory(FakeMicroVMFactory):
    def __call__(self, fc_bin, vm_dir):
        vm = FailingBootMicroVM(fc_bin, vm_dir)
        self.instances.append(vm)
        self.by_dir[vm_dir] = vm
        return vm


class FakeExecClient:
    def __init__(self, uds_path, port, calls):
        self.uds_path = uds_path
        self.port = port
        self.calls = calls

    def exec(self, argv, timeout_s=None):
        self.calls.append((self.uds_path, self.port, tuple(argv), timeout_s))
        return ExecResult(exit_code=0, stdout=b"ok\n", stderr=b"")


class FakeExecClientFactory:
    def __init__(self):
        self.calls = []

    def __call__(self, uds_path, port):
        return FakeExecClient(uds_path, port, self.calls)


def _make_sandbox(tmp_path, factory=None, **kwargs):
    factory = factory or FakeMicroVMFactory()
    exec_factory = FakeExecClientFactory()
    sandbox = FirecrackerSandbox(
        fc_bin="fc-bin", kernel="kernel", rootfs="rootfs.ext4",
        work_dir=str(tmp_path), microvm_factory=factory,
        exec_client_factory=exec_factory, **kwargs)
    return sandbox, factory, exec_factory


def _events(vm):
    return [e[0] for e in vm.events]


def test_spawn_root_boots_with_vsock_and_no_eager_snapshot(tmp_path):
    sandbox, factory, _ = _make_sandbox(tmp_path)

    sandbox.spawn("root", None)

    vm = factory.instances[-1]
    # snapshots are taken lazily at fork time, not at spawn
    assert vm.events == [("boot", "kernel", "rootfs.ext4", None, "v.sock")]
    assert sandbox.alive("root") is True


def test_spawn_root_without_vsock(tmp_path):
    sandbox, factory, _ = _make_sandbox(tmp_path, vsock=False)
    sandbox.spawn("root", None)
    assert factory.instances[-1].events[0] == \
        ("boot", "kernel", "rootfs.ext4", None, None)


def test_first_fork_snapshots_parent_then_restores_child(tmp_path):
    sandbox, factory, _ = _make_sandbox(tmp_path)
    sandbox.spawn("root", None)
    root_vm = factory.instances[-1]

    sandbox.spawn("child", "root")

    assert _events(root_vm) == ["boot", "pause", "snapshot", "resume"]
    child_vm = factory.instances[-1]
    mem = os.path.join(str(tmp_path), "root", "mem")
    state = os.path.join(str(tmp_path), "root", "state")
    assert child_vm.events == [("restore", mem, state)]
    assert sandbox.alive("child") is True


def test_second_fork_reuses_clean_snapshot(tmp_path):
    sandbox, factory, _ = _make_sandbox(tmp_path)
    sandbox.spawn("root", None)
    root_vm = factory.instances[-1]
    sandbox.spawn("c1", "root")
    sandbox.spawn("c2", "root")

    # parent paused/snapshotted exactly once: nothing changed between forks
    assert _events(root_vm).count("snapshot") == 1


def test_exec_marks_parent_dirty_so_next_fork_resnapshots(tmp_path):
    sandbox, factory, exec_factory = _make_sandbox(tmp_path)
    sandbox.spawn("root", None)
    root_vm = factory.instances[-1]
    sandbox.spawn("c1", "root")

    sandbox.exec("root", ["touch", "/scratch/x"])
    sandbox.spawn("c2", "root")

    # exec changed guest state; the second fork must not reuse the old image
    assert _events(root_vm).count("snapshot") == 2


def test_exec_routes_to_branch_uds_and_returns_result(tmp_path):
    sandbox, factory, exec_factory = _make_sandbox(tmp_path)
    sandbox.spawn("root", None)

    result = sandbox.exec("root", ["echo", "hi"], timeout_s=5.0)

    assert result.exit_code == 0 and result.stdout == b"ok\n"
    uds, port, argv, timeout_s = exec_factory.calls[-1]
    assert uds == os.path.join(str(tmp_path), "root", "v.sock")
    assert port == 52
    assert argv == ("echo", "hi") and timeout_s == 5.0


def test_wait_ready_retries_until_agent_answers(tmp_path):
    from agentfork.sandbox.vsock import VsockError

    class SlowBootExecClient(FakeExecClient):
        failures = [3]  # class-level countdown shared across instances

        def exec(self, argv, timeout_s=None):
            if self.failures[0] > 0:
                self.failures[0] -= 1
                raise VsockError("guest still booting")
            return super().exec(argv, timeout_s)

    class SlowBootFactory(FakeExecClientFactory):
        def __call__(self, uds_path, port):
            return SlowBootExecClient(uds_path, port, self.calls)

    factory = FakeMicroVMFactory()
    sandbox = FirecrackerSandbox(
        fc_bin="fc-bin", kernel="kernel", rootfs="rootfs.ext4",
        work_dir=str(tmp_path), microvm_factory=factory,
        exec_client_factory=SlowBootFactory())
    sandbox.spawn("root", None)

    sandbox.wait_ready("root", timeout_s=5.0)  # must not raise

    assert SlowBootExecClient.failures[0] == 0


def test_wait_ready_gives_up_after_deadline(tmp_path):
    from agentfork.sandbox.vsock import VsockError

    class NeverReadyClient(FakeExecClient):
        def exec(self, argv, timeout_s=None):
            raise VsockError("nobody home")

    class NeverReadyFactory(FakeExecClientFactory):
        def __call__(self, uds_path, port):
            return NeverReadyClient(uds_path, port, self.calls)

    sandbox = FirecrackerSandbox(
        fc_bin="fc-bin", kernel="kernel", rootfs="rootfs.ext4",
        work_dir=str(tmp_path), microvm_factory=FakeMicroVMFactory(),
        exec_client_factory=NeverReadyFactory())
    sandbox.spawn("root", None)

    with pytest.raises(VsockError, match="not ready"):
        sandbox.wait_ready("root", timeout_s=0.3)


def test_exec_unknown_branch_raises_key_error(tmp_path):
    sandbox, _, _ = _make_sandbox(tmp_path)
    with pytest.raises(KeyError):
        sandbox.exec("ghost", ["true"])


def test_exec_with_vsock_disabled_raises(tmp_path):
    sandbox, _, _ = _make_sandbox(tmp_path, vsock=False)
    sandbox.spawn("root", None)
    with pytest.raises(RuntimeError, match="vsock"):
        sandbox.exec("root", ["true"])


def test_overlay_created_for_root_and_copied_per_child(tmp_path):
    # mkfs="true" satisfies the subprocess call without formatting anything
    sandbox, factory, exec_factory = _make_sandbox(
        tmp_path, overlay_mib=4, mkfs="true")
    sandbox.spawn("root", None)

    root_overlay = tmp_path / "root" / "overlay.ext4"
    assert root_overlay.exists()
    assert root_overlay.stat().st_size == 4 * 1024 * 1024
    assert factory.instances[0].events[0] == \
        ("boot", "kernel", "rootfs.ext4", "overlay.ext4", "v.sock")

    sandbox.spawn("child", "root")

    assert (tmp_path / "child" / "overlay.ext4").exists()
    # the copy is taken while the parent is paused, after a best-effort sync
    assert [c[2] for c in exec_factory.calls] == [("sync",)]


def test_spawn_duplicate_branch_id_raises_value_error(tmp_path):
    sandbox, factory, _ = _make_sandbox(tmp_path)
    sandbox.spawn("root", None)

    with pytest.raises(ValueError):
        sandbox.spawn("root", None)


def test_fork_from_unknown_parent_with_no_snapshot_raises(tmp_path):
    sandbox, _, _ = _make_sandbox(tmp_path)
    with pytest.raises(KeyError, match="no snapshot"):
        sandbox.spawn("child", "root")


def test_restarted_adapter_forks_from_snapshot_files_on_disk(tmp_path):
    first, factory, _ = _make_sandbox(tmp_path)
    first.spawn("root", None)
    first.spawn("c1", "root")  # writes root's mem/state via the fake VMM

    # a restarted adapter has no in-memory handles, only the files
    second, factory2, _ = _make_sandbox(tmp_path)
    second.spawn("c2", "root")

    mem = os.path.join(str(tmp_path), "root", "mem")
    state = os.path.join(str(tmp_path), "root", "state")
    assert factory2.instances[-1].events == [("restore", mem, state)]


def test_kill_calls_vm_kill_and_alive_becomes_false(tmp_path):
    sandbox, factory, _ = _make_sandbox(tmp_path)
    sandbox.spawn("root", None)
    vm = factory.instances[-1]

    sandbox.kill("root")

    assert vm.events[-1][0] == "kill"
    assert sandbox.alive("root") is False


def test_kill_unknown_branch_is_a_noop(tmp_path):
    sandbox, factory, _ = _make_sandbox(tmp_path)

    sandbox.kill("does-not-exist")  # must not raise


def test_alive_is_false_when_process_has_exited_but_still_tracked(tmp_path):
    sandbox, factory, _ = _make_sandbox(tmp_path)
    sandbox.spawn("root", None)
    vm = factory.instances[-1]

    # simulate the guest process exiting on its own, without sandbox.kill()
    vm.proc._poll_value = 0

    assert sandbox.alive("root") is False
    assert "root" in sandbox._vms  # still tracked, just not alive


def test_spawn_records_pid_and_kill_removes_it(tmp_path):
    sandbox, factory, _ = _make_sandbox(tmp_path)
    sandbox.spawn("root", None)

    pid_path = tmp_path / "root" / "fc.pid"
    assert pid_path.read_text() == "54321"

    sandbox.kill("root")
    assert not pid_path.exists()


def test_kill_reclaims_orphan_from_pid_file_after_restart(tmp_path):
    import subprocess
    import sys

    orphan = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        # simulate a restarted supervisor: fresh adapter, no in-memory
        # handles, only the pid file a previous life recorded
        (tmp_path / "ghost").mkdir()
        (tmp_path / "ghost" / "fc.pid").write_text(str(orphan.pid))
        sandbox, factory, _ = _make_sandbox(tmp_path)

        sandbox.kill("ghost")

        assert orphan.wait(timeout=5) != 0  # SIGKILLed
        assert not (tmp_path / "ghost" / "fc.pid").exists()
    finally:
        if orphan.poll() is None:
            orphan.kill()
            orphan.wait()


def test_spawn_failure_kills_the_vm_and_leaves_no_bookkeeping(tmp_path):
    sandbox, factory, _ = _make_sandbox(
        tmp_path, factory=FailingBootMicroVMFactory())

    with pytest.raises(RuntimeError):
        sandbox.spawn("root", None)

    vm = factory.instances[-1]
    assert vm.events[-1][0] == "kill"  # best-effort cleanup ran
    assert "root" not in sandbox._vms
    assert "root" not in sandbox._snapshots
    assert sandbox.alive("root") is False
