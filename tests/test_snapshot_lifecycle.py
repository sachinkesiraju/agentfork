"""checkpoint / restart / export_bundle / import_bundle on FirecrackerSandbox,
exercised against the fake MicroVM (no real VMM). These close the
'single-host, reaper collects rather than restarts, no hibernation/migration'
limitations at the adapter level; a live checkpoint->kill->restart run is in
the Firecracker demo."""

import os

import pytest

from agentfork.sandbox.firecracker_backend import FirecrackerSandbox
# pytest puts the tests/ dir on sys.path, so the sibling module is imported by
# its bare name (the `tests.` package prefix isn't importable under the pytest
# console script, only under `python -m pytest`)
from test_firecracker_backend import FakeExecClientFactory, FakeMicroVMFactory


def _sandbox(tmp_path, **kwargs):
    return FirecrackerSandbox(
        fc_bin="fc-bin", kernel="kernel", rootfs="rootfs.ext4",
        work_dir=str(tmp_path), microvm_factory=FakeMicroVMFactory(),
        exec_client_factory=FakeExecClientFactory(), **kwargs)


def test_checkpoint_pauses_snapshots_and_resumes(tmp_path):
    sandbox = _sandbox(tmp_path)
    sandbox.spawn("root", None)
    vm = sandbox._vms["root"]

    sandbox.checkpoint("root")

    assert [e[0] for e in vm.events][-3:] == ["pause", "snapshot", "resume"]
    assert os.path.exists(os.path.join(str(tmp_path), "root", "mem"))
    # a checkpoint marks the branch's snapshot current, so a following fork
    # restores from it without re-snapshotting the parent
    sandbox.spawn("child", "root")
    assert [e[0] for e in vm.events].count("snapshot") == 1


def test_checkpoint_unknown_branch_raises(tmp_path):
    sandbox = _sandbox(tmp_path)
    with pytest.raises(KeyError):
        sandbox.checkpoint("ghost")


def test_restart_respawns_a_dead_branch_from_checkpoint(tmp_path):
    sandbox = _sandbox(tmp_path)
    sandbox.spawn("root", None)
    sandbox.checkpoint("root")
    # simulate the VMM dying out from under us
    sandbox._vms["root"].proc._poll_value = 137
    assert sandbox.alive("root") is False

    assert sandbox.restart("root") is True
    assert sandbox.alive("root") is True
    # the new VM restored from the checkpoint files
    assert sandbox._vms["root"].events[0][0] == "restore"


def test_restart_without_checkpoint_returns_false(tmp_path):
    sandbox = _sandbox(tmp_path)
    sandbox.spawn("root", None)  # never checkpointed -> no mem/state on disk
    sandbox._vms["root"].proc._poll_value = 1

    assert sandbox.restart("root") is False


def test_restart_is_noop_when_branch_still_alive(tmp_path):
    sandbox = _sandbox(tmp_path)
    sandbox.spawn("root", None)
    sandbox.checkpoint("root")
    n_before = len(sandbox._vms["root"].events)

    assert sandbox.restart("root") is True  # already alive
    assert len(sandbox._vms["root"].events) == n_before  # untouched


def test_export_then_import_into_fresh_sandbox_migrates_the_branch(tmp_path):
    # simulate migration to another host: export from sandbox A, import into
    # a fresh sandbox B rooted at a different work dir, restart it there
    src = _sandbox(tmp_path / "hostA", overlay_mib=4, mkfs="true")
    src.spawn("root", None)
    src.checkpoint("root")

    bundle = str(tmp_path / "bundle")
    src.export_bundle("root", bundle)
    assert os.path.exists(os.path.join(bundle, "mem"))
    assert os.path.exists(os.path.join(bundle, "state"))
    assert os.path.exists(os.path.join(bundle, "overlay.ext4"))
    assert os.path.exists(os.path.join(bundle, "manifest.json"))

    dst = _sandbox(tmp_path / "hostB", overlay_mib=4, mkfs="true")
    dst.import_bundle("root", bundle)
    assert dst.restart("root") is True
    assert dst.alive("root") is True
    assert dst._vms["root"].events[0][0] == "restore"


def test_export_requires_a_checkpoint(tmp_path):
    sandbox = _sandbox(tmp_path)
    sandbox.spawn("root", None)  # no checkpoint
    with pytest.raises(KeyError, match="no checkpoint"):
        sandbox.export_bundle("root", str(tmp_path / "b"))


def test_import_rejects_incomplete_bundle(tmp_path):
    sandbox = _sandbox(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        sandbox.import_bundle("root", str(empty))


class ArtifactExecClient:
    """Fake exec client returning a canned tar for the artifact export."""

    payload = b"TAR-BYTES"

    def __init__(self, uds_path, port, calls):
        self.calls = calls

    def exec(self, argv, timeout_s=None, stdin=None):
        from agentfork.sandbox.vsock import ExecResult
        self.calls.append(tuple(argv))
        return ExecResult(exit_code=0, stdout=self.payload, stderr=b"")


class ArtifactExecFactory:
    def __init__(self):
        self.calls = []

    def __call__(self, uds_path, port):
        return ArtifactExecClient(uds_path, port, self.calls)


def test_export_artifact_writes_guest_tar_to_host(tmp_path):
    exec_factory = ArtifactExecFactory()
    sandbox = FirecrackerSandbox(
        fc_bin="fc-bin", kernel="kernel", rootfs="rootfs.ext4",
        work_dir=str(tmp_path), microvm_factory=FakeMicroVMFactory(),
        exec_client_factory=exec_factory)
    sandbox.spawn("winner", None)

    dest = tmp_path / "out" / "winner.tar"
    n = sandbox.export_artifact("winner", "/workspace/build", str(dest))

    assert n == len(ArtifactExecClient.payload)
    assert dest.read_bytes() == ArtifactExecClient.payload
    # tars the requested path from its parent dir
    argv = exec_factory.calls[-1]
    assert argv[0] == "tar" and "build" in argv and "/workspace" in argv


def test_export_artifact_requires_a_live_branch(tmp_path):
    sandbox = _sandbox(tmp_path)
    with pytest.raises(KeyError):
        sandbox.export_artifact("ghost", "/x", str(tmp_path / "a.tar"))
