"""Agent harnesses: what actually edits a candidate worktree.

A harness is the "agent" half of map-reduce — given a node's idea and its
worktree, it makes the change and explains it. The eval/scoring path in
``runner`` is deliberately separate, so scores always come from artifacts.

Harnesses, in detection order:

* ``claude-code`` — the ``claude`` CLI in headless ``-p`` mode, detected via
  ``claude auth status --json`` (the check OpenResearch's claude.rs uses).
* ``codex`` / ``opencode`` / ``cursor-agent`` — the other vendor CLIs in
  their headless print modes (``codex exec``, ``opencode run``,
  ``cursor-agent -p``), detected via each CLI's own status command.
* ``api`` — plain chat completions over the ``agentfork.harness`` adapters:
  proposes ideas and applies a change by asking the model to emit complete
  replacement files in ``## path`` sections (no tool use, but works without
  any vendor CLI). Provider selection: ``AGENTFORK_API_BASE`` (+ optional
  ``AGENTFORK_API_KEY``/``AGENTFORK_API_MODEL``) points it at any
  OpenAI-compatible endpoint — LM Studio, Ollama, vLLM — else
  ``ANTHROPIC_API_KEY`` or ``TOGETHER_API_KEY`` pick the hosted default.
* ``fake`` — deterministic harness for tests and dashboards demos; makes a
  real (harmless) change so the tree has evidence.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.request
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


# -- vendor CLI harnesses -------------------------------------------------------

_PROPOSE_TEMPLATE = (
    "{context}\n\n"
    "Propose exactly {n} distinct single-variable ideas to try next. "
    "Return them as a JSON array of objects with keys "
    "\"title\", \"description\", \"regions\" (a list of area names). "
    "Output only the JSON.")

_IMPLEMENT_TEMPLATE = (
    "You are working inside the git worktree in your current directory. "
    "Implement exactly this change — nothing else:\n\n"
    "Title: {title}\n\n{description}\n\n"
    "Edit the files, then finish. Do not run the eval command; the "
    "orchestrator scores this worktree after you return.")


class CliHarness:
    """A vendor agent CLI driven in headless print mode: propose and
    implement are both one non-interactive invocation inside the repo/
    worktree, so the agent can read the code it proposes changes to and edit
    it in place. Subclasses supply the argv shape and the status probe."""

    name = "cli"
    bin = ""
    model_flag: str | None = None

    def __init__(self, model: str | None = None):
        self.model = model

    # per-CLI command shapes -------------------------------------------------
    def _argv(self, prompt: str) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def _probe_cmd(self) -> list[str] | None:  # pragma: no cover
        return None

    def probe(self) -> dict:
        if not shutil.which(self.bin):
            return {"installed": False, "authed": False}
        cmd = self._probe_cmd()
        if cmd is None:
            return {"installed": True, "authed": None,
                    "detail": "no status command"}
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=15)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return {"installed": True, "authed": None,
                    "detail": "status probe failed"}
        detail = (proc.stdout or proc.stderr).strip()[:200]
        return {"installed": True, "authed": proc.returncode == 0,
                "detail": detail}

    def _run(self, prompt: str, cwd: str, timeout_s: float) -> str:
        argv = self._argv(prompt)
        if self.model and self.model_flag:
            argv += [self.model_flag, self.model]
        try:
            proc = subprocess.run(argv, cwd=cwd, capture_output=True,
                                  text=True, timeout=timeout_s)
        except FileNotFoundError:
            raise HarnessError(f"{self.bin} CLI not found on PATH") from None
        except subprocess.TimeoutExpired:
            raise HarnessError(
                f"{self.bin} timed out after {timeout_s}s") from None
        if proc.returncode != 0:
            raise HarnessError(
                f"{self.bin} exited {proc.returncode}: "
                f"{proc.stderr.strip()[:400]}")
        return proc.stdout.strip()

    def propose(self, context: str, n: int, cwd: str | None = None) -> list[Idea]:
        prompt = _PROPOSE_TEMPLATE.format(context=context, n=n)
        # run inside the project's repo so the proposing agent can read the
        # code it is proposing changes to
        return _parse_ideas(self._run(prompt, cwd or os.getcwd(), 300), n)

    def implement(self, worktree: str, idea: Idea, context: str) -> str:
        prompt = _IMPLEMENT_TEMPLATE.format(title=idea.title,
                                            description=idea.description)
        self._run(prompt, worktree, 900)
        return f"{self.bin}: {idea.title}"


class ClaudeCodeHarness(CliHarness):
    name = "claude-code"
    bin = "claude"
    model_flag = "--model"

    def _argv(self, prompt: str) -> list[str]:
        return [self.bin, "-p", prompt, "--output-format", "text",
                "--dangerously-skip-permissions"]

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


class CodexHarness(CliHarness):
    """OpenAI Codex CLI: ``codex exec`` runs one headless task; login is
    ChatGPT-subscription or API key, probed via ``codex login status``."""

    name = "codex"
    bin = "codex"
    model_flag = "--model"

    def _argv(self, prompt: str) -> list[str]:
        return [self.bin, "exec", "--full-auto", prompt]

    def _probe_cmd(self) -> list[str]:
        return [self.bin, "login", "status"]


class OpenCodeHarness(CliHarness):
    """OpenCode CLI: ``opencode run`` is the headless mode; its own provider
    config (incl. any loopback model servers) carries auth."""

    name = "opencode"
    bin = "opencode"
    model_flag = "-m"

    def _argv(self, prompt: str) -> list[str]:
        return [self.bin, "run", prompt]

    def _probe_cmd(self) -> list[str]:
        return [self.bin, "auth", "list"]


class CursorAgentHarness(CliHarness):
    """Cursor's ``cursor-agent`` CLI in ``-p`` print mode."""

    name = "cursor-agent"
    bin = "cursor-agent"
    model_flag = "--model"

    def _argv(self, prompt: str) -> list[str]:
        return [self.bin, "-p", prompt, "--output-format", "text",
                "--force"]

    def _probe_cmd(self) -> list[str]:
        return [self.bin, "status"]


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
        return {"installed": True, "authed": _api_credentials() is not None}

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


def _api_credentials() -> str | None:
    """Which API route ``api`` would take — custom endpoint beats keys,
    matching ``build_harness``'s selection order."""
    if os.environ.get("AGENTFORK_API_BASE"):
        return "AGENTFORK_API_BASE"
    for var in ("ANTHROPIC_API_KEY", "TOGETHER_API_KEY"):
        if os.environ.get(var):
            return var
    return None


def list_models(base_url: str, api_key: str | None = None,
                timeout: float = 10.0) -> list[str]:
    """``GET {base}/models`` on an OpenAI-compatible server — the "find
    models" call orx's model picker makes against LM Studio/Ollama/vLLM."""
    headers = {"user-agent": "agentfork-harness/0.4"}
    key = api_key if api_key is not None \
        else os.environ.get("AGENTFORK_API_KEY", "")
    if key:
        headers["authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base_url.rstrip("/") + "/models",
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode())
    return [m["id"] for m in payload.get("data", []) if "id" in m]


def detect_harnesses() -> dict[str, dict]:
    """Dashboard readiness probe for the onboarding screen. The api harness
    reports which provider route it would actually use — that's what
    ``build_harness`` selects."""
    api = _api_credentials()
    return {
        "claude-code": ClaudeCodeHarness().probe(),
        "codex": CodexHarness().probe(),
        "opencode": OpenCodeHarness().probe(),
        "cursor-agent": CursorAgentHarness().probe(),
        "api": {"installed": True, "authed": api is not None,
                "detail": f"via {api}" if api else "needs "
                "AGENTFORK_API_BASE, ANTHROPIC_API_KEY or TOGETHER_API_KEY"},
        "fake": {"installed": True, "authed": True},
    }


def build_harness(name: str, llm=None, model: str | None = None,
                  api_base: str | None = None) -> Harness:
    if name == "claude-code":
        return ClaudeCodeHarness(model=model)
    if name == "codex":
        return CodexHarness(model=model)
    if name == "opencode":
        return OpenCodeHarness(model=model)
    if name == "cursor-agent":
        return CursorAgentHarness(model=model)
    if name == "api":
        if llm is None:
            # selection order mirrors _api_credentials(): an explicit
            # endpoint first, then whichever provider key is present
            base = api_base or os.environ.get("AGENTFORK_API_BASE")
            if base:
                from agentfork.harness.adapter import OpenAICompatLLM
                llm = OpenAICompatLLM(
                    api_key=os.environ.get("AGENTFORK_API_KEY", ""),
                    base_url=base,
                    model=model or os.environ.get("AGENTFORK_API_MODEL")
                          or "default")
            elif os.environ.get("ANTHROPIC_API_KEY"):
                from agentfork.harness.adapter import AnthropicLLM
                kw = {"model": model} if model else {}
                llm = AnthropicLLM(**kw)
            else:
                from agentfork.harness.adapter import OpenAICompatLLM
                kw = {"model": model} if model else {}
                llm = OpenAICompatLLM(**kw)
        return ApiHarness(llm)
    if name == "fake":
        return FakeHarness()
    raise HarnessError(f"unknown harness: {name}")
