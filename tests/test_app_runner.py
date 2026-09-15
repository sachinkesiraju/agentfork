"""Detached runs: on-disk layout, liveness, kill, log tail, score extraction."""

import time

from agentfork.app import runner


def _wait_done(run_dir, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not runner.alive(run_dir):
            return True
        time.sleep(0.02)
    return False


def test_run_writes_the_expected_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    handle = runner.start(run_dir, "echo hello; exit 0", tmp_path)
    assert _wait_done(run_dir)
    assert (run_dir / runner.RUN_SH).exists()
    assert runner.pid(run_dir) == handle.pid
    assert runner.exit_code(run_dir) == 0
    assert "hello" in (run_dir / runner.LOG).read_text()


def test_nonzero_exit_is_recorded(tmp_path):
    run_dir = tmp_path / "run"
    runner.start(run_dir, "exit 3", tmp_path)
    assert _wait_done(run_dir)
    assert runner.exit_code(run_dir) == 3


def test_command_runs_in_the_given_worktree(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "marker").write_text("x")
    run_dir = tmp_path / "run"
    runner.start(run_dir, "ls", work)
    assert _wait_done(run_dir)
    assert "marker" in (run_dir / runner.LOG).read_text()


def test_stderr_is_captured_with_stdout(tmp_path):
    run_dir = tmp_path / "run"
    runner.start(run_dir, "echo out; echo err 1>&2", tmp_path)
    assert _wait_done(run_dir)
    log = (run_dir / runner.LOG).read_text()
    assert "out" in log and "err" in log


def test_alive_then_kill_stops_the_process_group(tmp_path):
    run_dir = tmp_path / "run"
    runner.start(run_dir, "sleep 30", tmp_path)
    time.sleep(0.2)
    assert runner.alive(run_dir)
    assert runner.kill(run_dir)
    assert _wait_done(run_dir, timeout=8)
    assert not runner.alive(run_dir)


def test_kill_of_finished_run_is_harmless(tmp_path):
    run_dir = tmp_path / "run"
    runner.start(run_dir, "true", tmp_path)
    assert _wait_done(run_dir)
    runner.kill(run_dir)
    assert runner.exit_code(run_dir) == 0


def test_tail_is_incremental(tmp_path):
    run_dir = tmp_path / "run"
    runner.start(run_dir, "echo one; echo two", tmp_path)
    assert _wait_done(run_dir)
    first, offset = runner.tail(run_dir, limit=4)
    assert len(first) == 4
    rest, new_offset = runner.tail(run_dir, offset=offset)
    assert (first + rest).splitlines() == ["one", "two"]
    assert runner.tail(run_dir, offset=new_offset)[0] == ""


def test_tail_of_missing_log_is_empty(tmp_path):
    assert runner.tail(tmp_path / "nope") == ("", 0)


def test_extract_score_takes_the_last_match(tmp_path):
    run_dir = tmp_path / "run"
    runner.start(run_dir, "echo 'val_loss=0.9'; echo 'val_loss=0.4'", tmp_path)
    assert _wait_done(run_dir)
    assert runner.extract_score(run_dir, r"val_loss=([0-9.]+)") == 0.4
    assert runner.extract_score(run_dir, r"val_loss=([0-9.]+)",
                                last=False) == 0.9


def test_extract_score_handles_scientific_notation(tmp_path):
    run_dir = tmp_path / "run"
    runner.start(run_dir, "echo 'loss 1.5e-3'", tmp_path)
    assert _wait_done(run_dir)
    assert runner.extract_score(run_dir, r"loss (\S+)") == 0.0015


def test_extract_score_returns_none_when_metric_absent(tmp_path):
    run_dir = tmp_path / "run"
    runner.start(run_dir, "echo nothing useful", tmp_path)
    assert _wait_done(run_dir)
    assert runner.extract_score(run_dir, r"val_loss=([0-9.]+)") is None


def test_paths_with_spaces_are_quoted(tmp_path):
    work = tmp_path / "a dir"
    work.mkdir()
    run_dir = tmp_path / "run dir"
    runner.start(run_dir, "pwd", work)
    assert _wait_done(run_dir)
    assert "a dir" in (run_dir / runner.LOG).read_text()
    assert runner.exit_code(run_dir) == 0
