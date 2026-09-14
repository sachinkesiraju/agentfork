"""Agent harnesses: what actually edits a candidate worktree.

A harness is the "agent" half of map-reduce — given a node's idea and its
worktree, it makes the change and explains it. The eval/scoring path in
``runner`` is deliberately separate, so scores always come from artifacts.

Harnesses, in detection order:

* ``claude-code`` — the ``claude`` CLI in headless ``-p`` mode, detected via
  ``claude auth status --json`` (the check OpenResearch's claude.rs uses).
* ``api`` — the existing ``agentfork.harness`` LLM adapters: proposes ideas and
  applies a change by asking the model to emit complete replacement files in
  ``## path`` sections (no tool use, but works offline of any CLI and needs
  only an API key).
* ``fake`` — deterministic harness for tests and dashboards demos; makes a
  real (harmless) change so the tree has evidence.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

_IDEA_FENCE = re.compile(r"```[ \t]*[\w+-]*[ \t]*\n(.*?)```", re.DOTALL)


class HarnessError(RuntimeError):
    pass


@dataclass
class Idea:
    title: str
    description: str
    regions: list[str] = field(default_factory=list)


class Harness(Protocol):
    name: str

    def propose(self, context: str, n: int,
                cwd: str | None = None) -> list[Idea]:
        """Propose ``n`` candidate ideas for the next generation; ``cwd`` is
        the project's repo so CLI harnesses can read the code."""
        ...

    def implement(self, worktree: str, idea: Idea, context: str) -> str:
        """Apply the idea inside ``worktree``; return a one-line summary."""
        ...


# -- Claude Code --------------------------------------------------------------

class ClaudeCodeHarness:
    name = "claude-code"

    def __init__(self, bin: str = "claude"):
        self.bin = bin

    def probe(self) -> dict:
        """``claude auth status --json`` — same check orx runs."""
        try:
            proc = subprocess.run(
                [self.bin, "auth", "status", "--json"],
                capture_output=True, text=True, timeout=15)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return {"installed": False, "authed": False}
        if proc.returncode != 0:
            return {"installed": True, "authed": False,
                    "detail": proc.stderr.strip()[:200]}
        try:
            out = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            out = {}
        return {"installed": True,
                "authed": bool(out.get("loggedIn", out.get("authed", False))),
                "detail": proc.stdout.strip()[:200]}

    def _run(self, prompt: str, cwd: str, timeout_s: float) -> str:
        try:
            proc = subprocess.run(
                [self.bin, "-p", prompt, "--output-format", "text",
                 "--dangerously-skip-permissions"],
                cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
        except FileNotFoundError:
            raise HarnessError("claude CLI not found on PATH") from None
        except subprocess.TimeoutExpired:
            raise HarnessError(f"claude timed out after {timeout_s}s") from None
        if proc.returncode != 0:
            raise HarnessError(
                f"claude exited {proc.returncode}: {proc.stderr.strip()[:400]}")
        return proc.stdout.strip()

    def propose(self, context: str, n: int, cwd: str | None = None) -> list[Idea]:
        prompt = (
            f"{context}\n\n"
            f"Propose exactly {n} distinct single-variable ideas to try next. "
            "Return them as a JSON array of objects with keys "
            "\"title\", \"description\", \"regions\" (a list of area names). "
            "Output only the JSON.")
        # run inside the project's repo so the proposing agent can read the
        # code it is proposing changes to
        return _parse_ideas(self._run(prompt, cwd or os.getcwd(), 300), n)

    def implement(self, worktree: str, idea: Idea, context: str) -> str:
        prompt = (
            "You are working inside the git worktree in your current directory. "
            "Implement exactly this change — nothing else:\n\n"
            f"Title: {idea.title}\n\n{idea.description}\n\n"
            "Edit the files, then finish. Do not run the eval command; the "
            "orchestrator scores this worktree after you return.")
        self._run(prompt, worktree, 900)
        return f"claude: {idea.title}"


# -- API adapter harness --------------------------------------------------------

class ApiHarness:
    """Changes via plain chat completions (``_HTTPChatLLM`` adapters).

    ``implement`` asks for complete replacement files — the model has no tool
    access, so the prompt returns ``## <path>\\n<full file>`` sections. This is
    deliberately simple; Claude Code is the expected worker in real use."""

    name = "api"

    def __init__(self, llm):
        self.llm = llm

    def probe(self) -> dict:
        return {"installed": True, "authed": self._has_key()}

    def _has_key(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or
                    os.environ.get("TOGETHER_API_KEY"))

    def propose(self, context: str, n: int, cwd: str | None = None) -> list[Idea]:
        prompt = (
            f"{context}\n\n"
            f"Propose exactly {n} distinct single-variable ideas to try next. "
            "Return them as a JSON array of objects with keys "
            "\"title\", \"description\", \"regions\" (a list of area names). "
            "Output only the JSON.")
        # _HTTPChatLLM._message returns the reply text.
        return _parse_ideas(self.llm._message(prompt), n)

    def implement(self, worktree: str, idea: Idea, context: str) -> str:
        listing = "\n".join(
            str(p.relative_to(worktree))
            for p in sorted(Path(worktree).rglob("*"))
            if p.is_file() and ".git" not in p.parts and p.stat().st_size < 20_000)
        prompt = (
            f"{context}\n\nImplement this change in the repository:\n\n"
            f"Title: {idea.title}\n{idea.description}\n\n"
            f"Repo files:\n{listing}\n\n"
            "Output the complete new contents of every file you need to "
            "change as sections of the form:\n\n"
            "## path/relative/to/repo.ext\n<file contents>\n\n"
            "Nothing else.")
        reply = self.llm._message(prompt)
        written = _apply_file_sections(reply, Path(worktree))
        if not written:
            raise HarnessError("model returned no '## path' file sections")
        return ", ".join(written)


def _apply_file_sections(reply: str, root: Path) -> list[str]:
    written: list[str] = []
    for match in re.finditer(r"^## (.+?)\s*$", reply, re.MULTILINE):
        rel = match.group(1).strip().strip("`")
        start = match.end() + 1
        nxt = reply.find("\n## ", start)
        body = reply[start: nxt if nxt != -1 else len(reply)]
        body = body.strip("\n")
        fence = _IDEA_FENCE.search(body)
        if fence:
            body = fence.group(1)
        root_r = root.resolve()
        path = (root_r / rel).resolve()
        try:
            path.relative_to(root_r)
        except ValueError:
            continue  # absolute paths, .. traversal, and sibling prefixes
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body + ("\n" if body else ""))
        written.append(rel)
    return written


# -- fake harness ---------------------------------------------------------------

class FakeHarness:
    """Deterministic worker: appends a tagged comment to a scratch file so
    each node has a real diff without any network."""

    name = "fake"

    def __init__(self, ideas: list[Idea] | None = None):
        self.scripted = list(ideas or [])

    def probe(self) -> dict:
        return {"installed": True, "authed": True}

    def propose(self, context: str, n: int, cwd: str | None = None) -> list[Idea]:
        ideas = self.scripted[:n]
        while len(ideas) < n:
            i = len(ideas)
            ideas.append(Idea(title=f"idea-{i}",
                              description=f"synthetic candidate {i}"))
        return ideas

    def implement(self, worktree: str, idea: Idea, context: str) -> str:
        # scripted failure for tests: the title "fail" always crashes
        if idea.title == "fail":
            raise HarnessError("scripted failure")
        marker = Path(worktree) / ".agentfork-candidate"
        marker.write_text(f"{idea.title}\n{idea.description}\n")
        return f"wrote .agentfork-candidate ({idea.title})"


def _parse_ideas(text: str, n: int) -> list[Idea]:
    """Parse a JSON array of ideas out of a model reply."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            ideas = []
            for item in data[:n]:
                if isinstance(item, dict):
                    ideas.append(Idea(
                        title=str(item.get("title", "idea"))[:120],
                        description=str(item.get("description", "")),
                        regions=[str(r) for r in item.get("regions", []) or []]))
            if ideas:
                while len(ideas) < n:
                    ideas.append(ideas[len(ideas) % len(ideas)])
                return ideas
        except json.JSONDecodeError:
            pass
    # fall back to fenced blocks / paragraphs as idea titles
    parts = [b.strip() for b in _IDEA_FENCE.findall(text) if b.strip()]
    if not parts:
        parts = [p.strip().splitlines()[0] for p in text.split("\n\n") if p.strip()]
    if not parts:
        raise HarnessError("harness returned no parseable ideas")
    ideas = [Idea(title=p[:120], description=p) for p in parts[:n]]
    while len(ideas) < n:
        ideas.append(ideas[len(ideas) % len(ideas)])
    return ideas


def detect_harnesses() -> dict[str, dict]:
    """Dashboard readiness probe for the onboarding screen. The api harness
    reports which provider key it would actually use — that's what
    ``build_harness`` selects."""
    api_key = ("ANTHROPIC_API_KEY" if os.environ.get("ANTHROPIC_API_KEY")
               else "TOGETHER_API_KEY" if os.environ.get("TOGETHER_API_KEY")
               else None)
    return {
        "claude-code": ClaudeCodeHarness().probe(),
        "api": {"installed": True, "authed": api_key is not None,
                "detail": f"via {api_key}" if api_key else "needs "
                "ANTHROPIC_API_KEY or TOGETHER_API_KEY"},
        "fake": {"installed": True, "authed": True},
    }


def build_harness(name: str, llm=None) -> Harness:
    if name == "claude-code":
        return ClaudeCodeHarness()
    if name == "api":
        if llm is None:
            # pick the adapter by whichever provider key is present —
            # matching what detect_harnesses reports as authed
            if os.environ.get("ANTHROPIC_API_KEY"):
                from agentfork.harness.adapter import AnthropicLLM
                llm = AnthropicLLM()
            else:
                from agentfork.harness.adapter import OpenAICompatLLM
                llm = OpenAICompatLLM()
        return ApiHarness(llm)
    if name == "fake":
        return FakeHarness()
    raise HarnessError(f"unknown harness: {name}")
