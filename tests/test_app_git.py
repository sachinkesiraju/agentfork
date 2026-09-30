"""git plumbing: commit identity fallback for machines with none configured."""

from __future__ import annotations

import pytest

from agentfork.app import git


@pytest.fixture()
def bare_repo(tmp_path):
    """A repo whose author identity is deliberately empty — what a fresh
    machine without `git config --global user.*` looks like to the worker."""
    path = tmp_path / "repo"
    path.mkdir()
    git.git(path, "init", "-q", "-b", "main")
    git.git(path, "config", "user.name", "")
    git.git(path, "config", "user.email", "")
    (path / "f.txt").write_text("x")
    git.git(path, "add", "-A")
    git.git(path, "-c", "user.name=t", "-c", "user.email=t@t",
            "commit", "-qm", "init")
    return path


def test_commit_all_falls_back_to_service_identity(bare_repo):
    (bare_repo / "f.txt").write_text("y")
    sha = git.commit_all(bare_repo, "agentfork: idea-0")
    assert sha
    assert git.git(bare_repo, "log", "-1", "--format=%an") == "agentfork"


def test_commit_all_keeps_configured_identity(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    git.git(path, "init", "-q", "-b", "main")
    git.git(path, "config", "user.name", "someone")
    git.git(path, "config", "user.email", "s@example.com")
    (path / "f.txt").write_text("x")
    sha = git.commit_all(path, "agentfork: idea-0")
    assert sha
    assert git.git(path, "log", "-1", "--format=%an") == "someone"


def test_commit_all_clean_returns_none(bare_repo):
    assert git.commit_all(bare_repo, "agentfork: noop") is None
