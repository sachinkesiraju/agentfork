"""Git plumbing for experiment nodes: one worktree per node, branch identity.

An experiment node *is* a git branch plus its worktree, so a node's evidence is
recoverable from the repository alone, independent of the SQLite index. Thin
wrappers over the ``git`` CLI (no dependency on a git library).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def git(repo: str | Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def is_repo(path: str | Path) -> bool:
    try:
        return git(path, "rev-parse", "--is-inside-work-tree") == "true"
    except GitError:
        return False


def toplevel(path: str | Path) -> str:
    return git(path, "rev-parse", "--show-toplevel")


def current_branch(repo: str | Path) -> str:
    return git(repo, "rev-parse", "--abbrev-ref", "HEAD")


def head_sha(repo: str | Path, ref: str = "HEAD") -> str:
    return git(repo, "rev-parse", ref)


def short_sha(repo: str | Path, ref: str = "HEAD") -> str:
    return git(repo, "rev-parse", "--short", ref)


def branch_exists(repo: str | Path, branch: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(repo), "show-ref", "--verify", "--quiet",
         f"refs/heads/{branch}"], capture_output=True)
    return proc.returncode == 0


def dirty(repo: str | Path) -> bool:
    return bool(git(repo, "status", "--porcelain"))


def slugify(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:limit].strip("-") or "node")


def add_worktree(repo: str | Path, path: str | Path, branch: str,
                 start_point: str) -> str:
    """Check ``branch`` out at ``path``, creating it at ``start_point`` first if
    it does not exist.

    An existing branch is checked out as-is — ``worktree add <path> <branch>``
    — never reset to ``start_point``, because a node's branch is its evidence
    and resetting it would discard the commits a run already made on it.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if branch_exists(repo, branch):
        git(repo, "worktree", "add", str(path), branch)
    else:
        git(repo, "worktree", "add", "-b", branch, str(path), start_point)
    return str(path)


def remove_worktree(repo: str | Path, path: str | Path) -> None:
    git(repo, "worktree", "remove", "--force", str(path), check=False)
    git(repo, "worktree", "prune", check=False)


def delete_branch(repo: str | Path, branch: str) -> None:
    git(repo, "branch", "-D", branch, check=False)


def commit_all(repo: str | Path, message: str) -> str | None:
    """Commit every change in a worktree. Returns the sha, or None if clean."""
    git(repo, "add", "-A")
    if not git(repo, "diff", "--cached", "--name-only"):
        return None
    git(repo, "commit", "-m", message, "--no-verify")
    return head_sha(repo)


def gitignore(repo: str | Path, patterns: list[str]) -> None:
    """Append any missing ``patterns`` to the repo's .gitignore."""
    path = Path(repo) / ".gitignore"
    existing = path.read_text().splitlines() if path.exists() else []
    missing = [p for p in patterns if p not in existing]
    if not missing:
        return
    body = "\n".join(existing + ["", "# agentfork", *missing]) + "\n"
    path.write_text(body.lstrip("\n"))
