"""CLI behaviour that doesn't need a running dashboard."""

import pytest

from agentfork.app import cli


def test_status_without_server_errors_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--port", "1", "status"])
    assert "agentfork up" in str(exc.value)
    assert exc.value.code != 0


def test_install_skills_copies_both_skills(tmp_path, capsys):
    dest = tmp_path / "skills"
    assert cli.main(["install-skills", str(dest)]) == 0
    amr = dest / "agent-mapreduce"
    af = dest / "agentfork"
    assert (amr / "SKILL.md").exists()
    assert (amr / "amr.py").exists()
    assert (amr / "program.md").exists()
    assert (af / "SKILL.md").exists()
    out = capsys.readouterr().out
    assert "agent-mapreduce" in out and "agentfork" in out


def test_install_skills_is_idempotent(tmp_path):
    dest = tmp_path / "skills"
    cli.main(["install-skills", str(dest)])
    assert cli.main(["install-skills", str(dest)]) == 0


def test_skill_prints_the_agentfork_skill(capsys):
    assert cli.main(["skill"]) == 0
    assert "agentfork up" in capsys.readouterr().out


def test_amr_subcommand_passthrough(tmp_path, capsys):
    tsv = tmp_path / "results.tsv"
    assert cli.main(["amr", "--tsv", str(tsv), "log", "0", "base", "-", "1.0",
                     "ran", "baseline"]) == 0
    assert cli.main(["amr", "--tsv", str(tsv), "tree"]) == 0
    assert "base" in capsys.readouterr().out


def test_install_skills_default_dir_prefers_dot_claude(tmp_path, monkeypatch,
                                                     capsys):
    monkeypatch.chdir(tmp_path)
    cli.main(["install-skills"])
    assert (tmp_path / ".agents" / "skills" / "agentfork").exists()
    (tmp_path / ".claude").mkdir()
    cli.main(["install-skills"])
    assert (tmp_path / ".claude" / "skills" / "agentfork").exists()
