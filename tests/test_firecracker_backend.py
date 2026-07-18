"""Unit tests for agentfork.sandbox.firecracker_backend.FirecrackerSandbox.

Exercises the adapter's spawn/kill/alive/exec translation against fakes
standing in for ``agentfork.sandbox.fc_bench.MicroVM`` and the vsock exec
client. ``MicroVM.__init__`` launches a real Firecracker subprocess, so a
real ``MicroVM`` is never constructed here -- every test injects a fake
factory via the ``microvm_factory`` constructor argument, and a fake exec
client via ``exec_client_factory``.
"""

import json
import os
import subprocess
import sys
import threading
import time

import pytest

from agentfork import ForkOrchestrator
from agentfork.sandbox.firecracker_backend import FirecrackerSandbox
from agentfork.sandbox.fc_bench import JailerConfig
from agentfork.sandbox.vsock import ExecResult, VsockError


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

    def boot(self, kernel, rootfs, overlay=None, vsock_uds=None, tap=None):
        self.events.append(("boot", kernel, rootfs, overlay, vsock_uds))
        self.tap = tap
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

    def __call__(self, fc_bin, vm_dir, jailer=None, netns=None):
        vm = FakeMicroVM(fc_bin, vm_dir)
        vm.jailer = jailer
        vm.netns = netns
        self.instances.append(vm)
        self.by_dir[vm_dir] = vm
        return vm


class FailingBootMicroVM(FakeMicroVM):
    """Fails boot(), the way a real MicroVM would on a bad kernel/rootfs."""

    def boot(self, kernel, rootfs, overlay=None, vsock_uds=None, tap=None):
        self.events.append(("boot", kernel, rootfs, overlay, vsock_uds))
        raise RuntimeError("Firecracker PUT /actions returned HTTP 400")


class FailingBootMicroVMFactory(FakeMicroVMFactory):
    def __call__(self, fc_bin, vm_dir, jailer=None, netns=None):
        vm = FailingBootMicroVM(fc_bin, vm_dir)
        self.instances.append(vm)
        self.by_dir[vm_dir] = vm
        return vm


class FakeExecClient:
    def __init__(self, uds_path, port, calls):
        self.uds_path = uds_path
        self.port = port
        self.calls = calls

    def exec(self, argv, timeout_s=None, stdin=None):
        self.calls.append((self.uds_path, self.port, tuple(argv), timeout_s))
        return ExecResult(exit_code=0, stdout=b"ok\n", stderr=b"")

    def exec_detached(self, argv, timeout_s=30.0):
        from agentfork.sandbox.vsock import DetachedExec
        self.calls.append((self.uds_path, self.port, tuple(argv), "detach"))
        return DetachedExec(pid=4242, log_path="/tmp/fake.log")


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
    class SlowBootExecClient(FakeExecClient):
        failures = [3]  # class-level countdown shared across instances

        def exec(self, argv, timeout_s=None, stdin=None):
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
    class NeverReadyClient(FakeExecClient):
        def exec(self, argv, timeout_s=None, stdin=None):
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


def test_orchestrator_waits_for_guest_readiness_before_forking(tmp_path):
    sandbox, _, exec_factory = _make_sandbox(tmp_path)
    with ForkOrchestrator(sandbox=sandbox) as orch:
        orch.create_parent("root")
        orch.fork("root", child_ids=["root/child"])
        sandbox.await_reseed("root/child")  # reseed runs off the fork path now

    argvs = [call[2] for call in exec_factory.calls]
    assert argvs.count(("true",)) == 2  # readiness probes bracket the fork
    assert ("tee", "/dev/urandom") in argvs  # child RNG pool reseeded


def test_reseed_runs_off_the_fork_path(tmp_path):
    import threading
    import time

    started, release = threading.Event(), threading.Event()

    class SlowReseedClient(FakeExecClient):
        def exec(self, argv, timeout_s=None, stdin=None):
            if argv and argv[0] == "tee":
                started.set()
                release.wait(5)  # hold the reseed open
            return super().exec(argv, timeout_s, stdin=stdin)

    class SlowFactory(FakeExecClientFactory):
        def __call__(self, uds_path, port):
            return SlowReseedClient(uds_path, port, self.calls)

    sandbox = FirecrackerSandbox(
        fc_bin="fc-bin", kernel="kernel", rootfs="rootfs.ext4",
        work_dir=str(tmp_path), microvm_factory=FakeMicroVMFactory(),
        exec_client_factory=SlowFactory())
    sandbox.spawn("root", None)

    t0 = time.perf_counter()
    sandbox.spawn("child", "root")  # must not block on the (held) reseed
    spawn_ms = (time.perf_counter() - t0) * 1000

    try:
        assert started.wait(2)      # the reseed did start, in the background
        assert spawn_ms < 500       # ...but spawn returned without waiting
    finally:
        release.set()
        sandbox.await_reseed("child")


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
    sandbox.await_reseed("child")  # reseed runs off the fork path now

    assert (tmp_path / "child" / "overlay.ext4").exists()
    # the copy is taken while the parent is paused after a best-effort sync,
    # then the restored child's RNG pool is reseeded to de-correlate siblings
    assert [c[2] for c in exec_factory.calls] == [("sync",), ("tee", "/dev/urandom")]


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
    assert json.loads(pid_path.read_text()) == {"pid": 54321}

    sandbox.kill("root")
    assert not pid_path.exists()


@pytest.mark.skipif(not os.path.exists("/proc"),
                    reason="needs /proc for process identity")
def test_kill_reclaims_orphan_from_pid_file_after_restart(tmp_path):
    orphan = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        # simulate a restarted supervisor: fresh adapter, no in-memory
        # handles, only the pid record a previous life wrote.
        (tmp_path / "ghost").mkdir()
        sandbox, factory, _ = _make_sandbox(tmp_path)
        record = sandbox._process_identity(orphan.pid)
        assert record is not None
        (tmp_path / "ghost" / "fc.pid").write_text(json.dumps(record))

        assert sandbox.alive("ghost") is (os.path.exists("/proc"))
        sandbox.kill("ghost")

        assert orphan.wait(timeout=5) != 0  # SIGKILLed
        assert not (tmp_path / "ghost").exists()
    finally:
        if orphan.poll() is None:
            orphan.kill()
            orphan.wait()


@pytest.mark.skipif(not os.path.exists("/proc"), reason="needs /proc comm")
def test_recycled_pid_with_wrong_comm_is_spared(tmp_path):
    bystander = subprocess.Popen([sys.executable, "-c",
                                  "import time; time.sleep(60)"])
    try:
        # legacy plain-pid file whose pid now belongs to a non-VMM process
        (tmp_path / "ghost").mkdir()
        (tmp_path / "ghost" / "fc.pid").write_text(str(bystander.pid))
        sandbox, _, _ = _make_sandbox(tmp_path)  # fc_bin="fc-bin"

        assert sandbox.alive("ghost") is False  # not one of our VMMs
        sandbox.kill("ghost")

        assert bystander.poll() is None  # still running, unharmed
        assert not (tmp_path / "ghost").exists()  # artifacts cleared
    finally:
        bystander.kill()
        bystander.wait()


def test_copy_overlay_preserves_content_and_sparseness(tmp_path):
    src, dst = tmp_path / "src.img", tmp_path / "dst.img"
    logical = 8 * 1024 * 1024
    with open(src, "wb") as f:
        f.truncate(logical)  # sparse: no data blocks yet
        f.seek(1024 * 1024)
        f.write(b"A" * 4096)
        f.seek(5 * 1024 * 1024)
        f.write(b"B" * 4096)

    FirecrackerSandbox._copy_overlay(str(src), str(dst))

    assert dst.stat().st_size == logical
    assert src.read_bytes() == dst.read_bytes()
    # the copy must not materialize the holes: a reflink or sparse-aware
    # copy allocates far fewer blocks than the 8 MiB logical size
    if src.stat().st_blocks * 512 < logical // 2:  # fs tracks sparseness
        assert dst.stat().st_blocks * 512 < logical // 2


def test_sparse_copy_reraises_real_io_errors_instead_of_truncating(tmp_path):
    import errno

    src, dst = tmp_path / "s.img", tmp_path / "d.img"
    with open(src, "wb") as f:
        f.truncate(1 << 20)
        f.write(b"data")

    real_lseek = os.lseek

    def exploding_lseek(fd, pos, how):
        if how == os.SEEK_DATA:
            raise OSError(errno.EIO, "injected I/O error")
        return real_lseek(fd, pos, how)

    # a genuine I/O error mid-copy must propagate (so _copy_overlay's outer
    # fallback fires), not be mistaken for ENXIO end-of-data and truncate
    orig = os.lseek
    os.lseek = exploding_lseek
    try:
        with open(src, "rb") as fs, open(dst, "wb") as fd_:
            with pytest.raises(OSError) as ei:
                FirecrackerSandbox._sparse_copy(fs.fileno(), fd_.fileno())
            assert ei.value.errno == errno.EIO
    finally:
        os.lseek = orig


def test_sweep_dead_lists_branches_whose_vmm_exited(tmp_path):
    sandbox, factory, _ = _make_sandbox(tmp_path)
    sandbox.spawn("root", None)
    sandbox.spawn("child", "root")

    assert sandbox.sweep_dead() == []
    factory.by_dir[os.path.join(str(tmp_path), "child")].proc._poll_value = 137

    assert sandbox.sweep_dead() == ["child"]
    assert sandbox.alive("root") is True


def test_branch_paths_cannot_escape_or_collide(tmp_path):
    sandbox, _, _ = _make_sandbox(tmp_path)

    with pytest.raises(ValueError, match="branch_id"):
        sandbox._vm_dir("..")
    with pytest.raises(ValueError, match="branch_id"):
        sandbox._vm_dir("")
    assert sandbox._vm_dir("a/b") != sandbox._vm_dir("a_b")
    encoded_name = os.path.basename(sandbox._vm_dir("a/b"))
    with pytest.raises(ValueError, match="branch_id"):
        sandbox._vm_dir(encoded_name)


@pytest.mark.skipif(not os.path.exists("/proc"),
                    reason="needs /proc for process identity")
def test_pid_reuse_identity_mismatch_never_kills_process(tmp_path):
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        sandbox, _, _ = _make_sandbox(tmp_path)
        vm_dir = tmp_path / "ghost"
        vm_dir.mkdir()
        record = sandbox._process_identity(process.pid)
        assert record is not None
        record["start_time"] += 1
        (vm_dir / "fc.pid").write_text(json.dumps(record))

        sandbox.kill("ghost")

        assert process.poll() is None
        assert not vm_dir.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_kill_removes_snapshot_and_overlay_artifacts(tmp_path):
    sandbox, _, _ = _make_sandbox(
        tmp_path, overlay_mib=1, mkfs="true")
    sandbox.spawn("root", None)
    sandbox.spawn("child", "root")
    assert (tmp_path / "root" / "mem").exists()
    assert (tmp_path / "root" / "overlay.ext4").exists()

    sandbox.kill("root")

    assert not (tmp_path / "root").exists()


def test_parent_kill_waits_until_child_restore_finishes(tmp_path):
    restore_started = threading.Event()
    release_restore = threading.Event()

    class BlockingRestoreVM(FakeMicroVM):
        def restore(self, mem_path, state_path):
            restore_started.set()
            release_restore.wait(2)
            return super().restore(mem_path, state_path)

    class Factory(FakeMicroVMFactory):
        def __call__(self, fc_bin, vm_dir):
            vm = BlockingRestoreVM(fc_bin, vm_dir)
            self.instances.append(vm)
            self.by_dir[vm_dir] = vm
            return vm

    sandbox, _, _ = _make_sandbox(tmp_path, factory=Factory())
    sandbox.spawn("root", None)
    child = threading.Thread(
        target=sandbox.spawn, args=("child", "root"))
    child.start()
    assert restore_started.wait(1)
    killed = threading.Event()
    kill = threading.Thread(
        target=lambda: (sandbox.kill("root"), killed.set()))
    kill.start()

    time.sleep(0.05)
    assert not killed.is_set()
    release_restore.set()
    child.join(2)
    kill.join(2)

    assert killed.is_set()
    assert sandbox.alive("child")


def test_fork_waits_for_in_flight_exec_before_snapshot(tmp_path):
    exec_started = threading.Event()
    release_exec = threading.Event()

    class BlockingClient(FakeExecClient):
        def exec(self, argv, timeout_s=None, stdin=None):
            exec_started.set()
            release_exec.wait(2)
            return super().exec(argv, timeout_s, stdin=stdin)

    class ExecFactory(FakeExecClientFactory):
        def __call__(self, uds_path, port):
            return BlockingClient(uds_path, port, self.calls)

    factory = FakeMicroVMFactory()
    sandbox = FirecrackerSandbox(
        fc_bin="fc-bin", kernel="kernel", rootfs="rootfs.ext4",
        work_dir=str(tmp_path), microvm_factory=factory,
        exec_client_factory=ExecFactory())
    sandbox.spawn("root", None)
    root = factory.instances[0]
    execute = threading.Thread(
        target=sandbox.exec, args=("root", ["touch", "/tmp/x"]))
    execute.start()
    assert exec_started.wait(1)
    fork = threading.Thread(
        target=sandbox.spawn, args=("child", "root"))
    fork.start()

    time.sleep(0.05)
    assert "snapshot" not in _events(root)
    release_exec.set()
    execute.join(2)
    fork.join(2)

    assert "snapshot" in _events(root)


class JailAwareFakeFactory(FakeMicroVMFactory):
    """Fake factory matching the 3-arg signature the adapter uses when a
    jailer is configured."""

    def __call__(self, fc_bin, vm_dir, jailer=None, netns=None):
        vm = FakeMicroVM(fc_bin, vm_dir)
        vm.jailer = jailer
        vm.netns = netns
        self.instances.append(vm)
        self.by_dir[vm_dir] = vm
        return vm


def test_jailed_child_gets_snapshot_pair_and_rootfs_staged(tmp_path):
    rootfs = tmp_path / "rootfs.squashfs"
    rootfs.write_bytes(b"rootfs-bytes")
    # uid/gid = our own so the _chown_into_jail call is permitted in tests
    jailer = JailerConfig(jailer_bin="/usr/bin/jailer", uid=os.getuid(),
                          gid=os.getgid(), chroot_base=str(tmp_path / "jail"))
    factory = JailAwareFakeFactory()
    sandbox = FirecrackerSandbox(
        fc_bin="fc-bin", kernel="kernel", rootfs=str(rootfs),
        work_dir=str(tmp_path / "work"), microvm_factory=factory,
        overlay_mib=1, mkfs="true",
        exec_client_factory=FakeExecClientFactory(), jailer=jailer)

    sandbox.spawn("root", None)
    root_chroot = tmp_path / "jail" / "fc-bin" / "root" / "root"
    assert (root_chroot / "overlay.ext4").exists()  # created in the jail
    assert factory.instances[0].jailer is jailer    # factory got the config

    sandbox.spawn("child", "root")

    child_chroot = tmp_path / "jail" / "fc-bin" / "child" / "root"
    # a jailed child cannot see its parent's chroot: the snapshot pair and
    # the shared rootfs must be staged into the child's own jail, and the
    # restore must reference the staged copies
    assert (child_chroot / "pmem").read_bytes() == b"snap"
    assert (child_chroot / "pstate").read_bytes() == b"snap"
    assert (child_chroot / "rootfs.img").read_bytes() == b"rootfs-bytes"
    assert (child_chroot / "overlay.ext4").exists()
    child_vm = factory.instances[-1]
    assert child_vm.events[0] == ("restore", str(child_chroot / "pmem"),
                                  str(child_chroot / "pstate"))


def test_networking_sets_up_and_tears_down_a_netns_per_branch(tmp_path):
    from agentfork.sandbox.netns import NetworkConfig

    net_calls = []
    sandbox, factory, _ = _make_sandbox(
        tmp_path, network=NetworkConfig(uplink="eth0"))
    sandbox._netns._run = lambda argv, *, check=True: net_calls.append(argv)

    sandbox.spawn("root", None)

    # the VMM was launched into a namespace and the tap was passed to boot
    root_vm = factory.instances[-1]
    assert root_vm.netns == "af-root"
    assert root_vm.tap == "tap0"
    assert any(c[:3] == ["ip", "netns", "add"] for c in net_calls)

    net_calls.clear()
    sandbox.kill("root")

    # kill tears the namespace back down
    assert any(c[:3] == ["ip", "netns", "del"] for c in net_calls)


def test_networking_index_journaled_and_recovered_on_restart(tmp_path):
    from agentfork.sandbox.netns import NetworkConfig

    # first adapter spawns a branch with networking, then "crashes" (we drop
    # it without killing) leaving the netns up and its index only on disk
    first, _, _ = _make_sandbox(tmp_path, network=NetworkConfig(uplink="eth0"))
    first._netns._run = lambda argv, *, check=True: None
    first.spawn("root", None)
    assert (tmp_path / "root" / "netns.idx").exists()

    # restarted adapter: no in-memory _netns_index; kill must still recover
    # the index from disk and tear the namespace down
    second, _, _ = _make_sandbox(tmp_path, network=NetworkConfig(uplink="eth0"))
    torn = []
    second._netns._run = lambda argv, *, check=True: torn.append(argv)
    second.kill("root")

    assert any(c[:3] == ["ip", "netns", "del"] for c in torn)
    assert not (tmp_path / "root" / "netns.idx").exists()


def test_networking_teardown_runs_on_spawn_failure(tmp_path):
    from agentfork.sandbox.netns import NetworkConfig

    net_calls = []
    sandbox, _, _ = _make_sandbox(
        tmp_path, factory=FailingBootMicroVMFactory(),
        network=NetworkConfig(uplink="eth0"))
    sandbox._netns._run = lambda argv, *, check=True: net_calls.append(argv)

    with pytest.raises(RuntimeError):
        sandbox.spawn("root", None)

    # the half-built namespace must not leak
    assert any(c[:3] == ["ip", "netns", "del"] for c in net_calls)
    assert "root" not in sandbox._netns_index


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
