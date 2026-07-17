import pytest

from agentfork.sandbox.fc_bench import MicroVM, run


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
