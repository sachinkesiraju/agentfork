import pytest

from agentfork.sandbox.fc_bench import (
    JailerConfig,
    MicroVM,
    jail_id_for,
    jail_root,
    jailer_argv,
    run,
)


class FakeAPI:
    def __init__(self, status):
        self.status = status

    def request(self, method, path, body):
        return self.status


def test_firecracker_api_errors_are_not_assertions():
    vm = object.__new__(MicroVM)
    vm.api = FakeAPI(400)

    with pytest.raises(RuntimeError, match="HTTP 400"):
        vm._request("PUT", "/actions", {})


def test_firecracker_benchmark_requires_children():
    with pytest.raises(ValueError, match="n_children"):
        run("firecracker", "vmlinux", "rootfs.ext4", 0)


JAILER = JailerConfig(jailer_bin="/usr/bin/jailer", uid=1234, gid=1234,
                      chroot_base="/srv/jailer")


def test_jail_id_sanitizes_to_jailer_charset():
    assert jail_id_for("/work/root_1") == "root-1"
    assert jail_id_for("/work/" + "x" * 100) == "x" * 64


def test_jail_root_is_under_chroot_base_and_exec_file_name():
    root = jail_root(JAILER, "/opt/fc/firecracker", "/work/branch_a")
    assert root == "/srv/jailer/firecracker/branch-a/root"


def test_jailer_argv_shape():
    argv = jailer_argv(JAILER, "/opt/fc/firecracker", "/work/branch_a")
    assert argv[0] == "/usr/bin/jailer"
    assert argv[argv.index("--id") + 1] == "branch-a"
    assert argv[argv.index("--exec-file") + 1] == "/opt/fc/firecracker"
    assert argv[argv.index("--uid") + 1] == "1234"
    assert argv[-3:] == ["--", "--api-sock", "fc.sock"]


def _bare_vm(jailer=None, host_dir="/srv/jailer/firecracker/b/root"):
    vm = object.__new__(MicroVM)
    vm.jailer = jailer
    vm.host_dir = host_dir
    return vm


def test_vm_path_translation_unjailed_absolutizes():
    vm = _bare_vm(jailer=None, host_dir="/work/b")
    assert vm._vm_path("/work/b/mem") == "/work/b/mem"


def test_vm_path_translation_jailed_is_chroot_relative():
    vm = _bare_vm(jailer=JAILER)
    assert vm._vm_path("/srv/jailer/firecracker/b/root/mem") == "mem"
    with pytest.raises(ValueError, match="outside jail"):
        vm._vm_path("/work/elsewhere/mem")


def test_shared_input_jailed_links_into_chroot(tmp_path):
    kernel = tmp_path / "vmlinux"
    kernel.write_bytes(b"kernel")
    chroot = tmp_path / "jail" / "root"
    chroot.mkdir(parents=True)
    vm = _bare_vm(jailer=JAILER, host_dir=str(chroot))

    assert vm._shared_input(str(kernel), "vmlinux") == "vmlinux"
    assert (chroot / "vmlinux").read_bytes() == b"kernel"
