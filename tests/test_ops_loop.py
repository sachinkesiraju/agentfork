"""Background reaper thread and metrics counters."""

import time

from agentfork.orchestrator import ForkOrchestrator


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class DyingSandbox:
    """Sandbox whose branches can be marked dead externally, with the
    ``sweep_dead`` supervision hook the background reaper consumes."""

    parallel_lifecycle = True

    def __init__(self):
        self.live = set()
        self.dead = set()

    def spawn(self, branch_id, parent_id):
        self.live.add(branch_id)

    def kill(self, branch_id):
        self.live.discard(branch_id)
        self.dead.discard(branch_id)

    def alive(self, branch_id):
        return branch_id in self.live and branch_id not in self.dead

    def sweep_dead(self):
        return sorted(self.dead)


def _wait_until(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_background_reaper_collects_lapsed_leases_unprompted():
    clock = FakeClock()
    orch = ForkOrchestrator(clock=clock, reap_interval_s=0.05)
    try:
        orch.create_parent("root", tokens=[1, 2, 3])
        orch.fork("root", n=2, lease_s=30)
        clock.now += 60

        assert _wait_until(lambda: len(orch.branches()) == 1)
        assert orch.branches()[0].branch_id == "root"
        assert orch.metrics_snapshot()["reaped_expired"] >= 2
    finally:
        orch.close()


def test_background_reaper_collects_branches_whose_sandbox_died():
    sandbox = DyingSandbox()
    orch = ForkOrchestrator(sandbox=sandbox)
    try:
        orch.create_parent("root")
        child = orch.fork("root", n=1)[0]
        orch.start_reaper(interval_s=0.05)

        sandbox.dead.add(child.branch_id)  # the "VMM" crashed

        assert _wait_until(
            lambda: child.branch_id not in {b.branch_id for b in orch.branches()})
        assert orch.metrics_snapshot()["swept_dead"] == 1
    finally:
        orch.close()


def test_close_stops_the_reaper_thread():
    orch = ForkOrchestrator()
    orch.start_reaper(interval_s=0.05)
    thread = orch._reaper_thread
    assert thread is not None and thread.is_alive()

    orch.close()

    assert not thread.is_alive()
    assert orch._reaper_thread is None


def test_start_reaper_is_idempotent():
    orch = ForkOrchestrator()
    try:
        orch.start_reaper(interval_s=10)
        first = orch._reaper_thread
        orch.start_reaper(interval_s=10)
        assert orch._reaper_thread is first
    finally:
        orch.close()


def test_reaper_survives_a_transient_runtime_error_and_keeps_running():
    # a non-"closed" RuntimeError from a pass must not silently kill the
    # reaper on a healthy orchestrator
    class FlakySweepSandbox(DyingSandbox):
        def __init__(self):
            super().__init__()
            self.sweeps = 0

        def sweep_dead(self):
            self.sweeps += 1
            if self.sweeps == 1:
                raise RuntimeError("transient transport blip")
            return []

    sandbox = FlakySweepSandbox()
    orch = ForkOrchestrator(sandbox=sandbox)
    try:
        orch.create_parent("root")
        orch.start_reaper(interval_s=0.03)
        # the reaper must run several passes despite the first one raising
        assert _wait_until(lambda: sandbox.sweeps >= 3)
        assert orch._reaper_thread.is_alive()
    finally:
        orch.close()


def test_start_then_stop_reaper_does_not_hang_with_interleaving():
    # exercise repeated start/stop cycles: the loop captures its own stop
    # event, so a restart can't strand the previous thread
    orch = ForkOrchestrator()
    try:
        for _ in range(5):
            orch.start_reaper(interval_s=0.02)
            t = orch._reaper_thread
            orch.stop_reaper()
            assert not t.is_alive()
    finally:
        orch.close()


def test_metrics_count_lifecycle_operations():
    class ExecSandbox(DyingSandbox):
        def exec(self, branch_id, argv, timeout_s=None, stdin=None):
            return "ok"

    orch = ForkOrchestrator(sandbox=ExecSandbox())
    orch.create_parent("root", tokens=[1])
    orch.fork("root", n=3)
    orch.exec("root", ["true"])
    orch.kill("root/1")
    orch.reconcile()

    m = orch.metrics_snapshot()
    assert m["forks"] == 3
    assert m["execs"] == 1
    assert m["kills"] == 1
    assert m["reconciles"] == 1
    assert m["kill_failures"] == 0

    orch.close()
    assert orch.metrics_snapshot()["kills"] == 4  # root + 2 remaining children
