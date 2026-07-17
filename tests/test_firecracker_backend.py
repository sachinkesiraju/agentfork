"""Unit tests for agentfork.sandbox.firecracker_backend.FirecrackerSandbox.

Exercises the adapter's spawn/kill/alive translation against a fake standing
in for ``agentfork.sandbox.fc_bench.MicroVM``. ``MicroVM.__init__`` launches a
real Firecracker subprocess, so a real ``MicroVM`` is never constructed here
-- every test injects a fake factory via the ``microvm_factory`` constructor
argument instead of letting it default to the real ``MicroVM``.
"""

import pytest

from agentfork.sandbox.firecracker_backend import FirecrackerSandbox


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

    def boot(self, kernel, rootfs):
        self.events.append(("boot", kernel, rootfs))
        return 0.001

    def pause(self):
        self.events.append(("pause",))

    def resume(self):
        self.events.append(("resume",))

    def snapshot(self, mem_path, state_path):
        self.events.append(("snapshot", mem_path, state_path))
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

    def boot(self, kernel, rootfs):
        self.events.append(("boot", kernel, rootfs))
        raise RuntimeError("Firecracker PUT /actions returned HTTP 400")


class FailingBootMicroVMFactory(FakeMicroVMFactory):
    def __call__(self, fc_bin, vm_dir):
        vm = FailingBootMicroVM(fc_bin, vm_dir)
        self.instances.append(vm)
        self.by_dir[vm_dir] = vm
        return vm


def _make_sandbox(tmp_path, factory=None):
    factory = factory or FakeMicroVMFactory()
    sandbox = FirecrackerSandbox(
        fc_bin="fc-bin", kernel="kernel", rootfs="rootfs.ext4",
        work_dir=str(tmp_path), microvm_factory=factory)
    return sandbox, factory


def test_spawn_root_boots_pauses_snapshots_and_becomes_alive(tmp_path):
    sandbox, factory = _make_sandbox(tmp_path)

    sandbox.spawn("root", None)

    vm = factory.instances[-1]
    assert [e[0] for e in vm.events] == ["boot", "pause", "snapshot", "resume"]
    assert vm.events[0] == ("boot", "kernel", "rootfs.ext4")
    assert sandbox.alive("root") is True


def test_spawn_child_restores_from_parent_snapshot_not_boot(tmp_path):
    sandbox, factory = _make_sandbox(tmp_path)
    sandbox.spawn("root", None)
    parent_mem, parent_state = sandbox._snapshots["root"]

    sandbox.spawn("child", "root")

    child_vm = factory.instances[-1]
    assert [e[0] for e in child_vm.events] == ["restore", "pause", "snapshot", "resume"]
    assert child_vm.events[0] == ("restore", parent_mem, parent_state)
    assert sandbox.alive("child") is True


def test_spawn_duplicate_branch_id_raises_value_error(tmp_path):
    sandbox, factory = _make_sandbox(tmp_path)
    sandbox.spawn("root", None)

    with pytest.raises(ValueError):
        sandbox.spawn("root", None)


def test_kill_calls_vm_kill_and_alive_becomes_false(tmp_path):
    sandbox, factory = _make_sandbox(tmp_path)
    sandbox.spawn("root", None)
    vm = factory.instances[-1]

    sandbox.kill("root")

    assert vm.events[-1][0] == "kill"
    assert sandbox.alive("root") is False


def test_kill_unknown_branch_is_a_noop(tmp_path):
    sandbox, factory = _make_sandbox(tmp_path)

    sandbox.kill("does-not-exist")  # must not raise


def test_alive_is_false_when_process_has_exited_but_still_tracked(tmp_path):
    sandbox, factory = _make_sandbox(tmp_path)
    sandbox.spawn("root", None)
    vm = factory.instances[-1]

    # simulate the guest process exiting on its own, without sandbox.kill()
    vm.proc._poll_value = 0

    assert sandbox.alive("root") is False
    assert "root" in sandbox._vms  # still tracked, just not alive


def test_spawn_records_pid_and_kill_removes_it(tmp_path):
    sandbox, factory = _make_sandbox(tmp_path)
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
        sandbox, factory = _make_sandbox(tmp_path)

        sandbox.kill("ghost")

        assert orphan.wait(timeout=5) != 0  # SIGKILLed
        assert not (tmp_path / "ghost" / "fc.pid").exists()
    finally:
        if orphan.poll() is None:
            orphan.kill()
            orphan.wait()


def test_spawn_failure_kills_the_vm_and_leaves_no_bookkeeping(tmp_path):
    sandbox, factory = _make_sandbox(tmp_path, factory=FailingBootMicroVMFactory())

    with pytest.raises(RuntimeError):
        sandbox.spawn("root", None)

    vm = factory.instances[-1]
    assert vm.events[-1][0] == "kill"  # best-effort cleanup ran
    assert "root" not in sandbox._vms
    assert "root" not in sandbox._snapshots
    assert sandbox.alive("root") is False
