"""The LLM seam for ``TreeAgent``: propose candidate continuations from a
shared context, and (optionally) score an outcome.

The harness never talks to a model directly; it goes through the ``LLM``
Protocol so tests run a deterministic fake with no network and the demo runs a
real model. ``FakeLLM`` replays a fixed script. ``AnthropicLLM`` calls the
Anthropic Messages API over stdlib ``urllib`` (no SDK dependency, matching
``SGLangHTTPBackend``); ``OpenAICompatLLM`` covers Together and any other
OpenAI ``/chat/completions`` endpoint as a fallback. Both real adapters share
the same prompting and parsing (``_HTTPChatLLM``); they differ only in wire
format.

The evaluator a caller passes to ``TreeAgent`` is separate from ``LLM.score``:
the golden path scores branches with a *cheap deterministic check* (does the
test suite pass?) and reserves ``LLM.score`` for judgments no check can make.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Protocol

_FENCE = re.compile(r"```[ \t]*[\w+-]*[ \t]*\n(.*?)```", re.DOTALL)


class LLM(Protocol):
    """Proposes candidate continuations and scores outcomes."""

    def propose(self, context: str, n: int) -> list[str]:
        """Return ``n`` candidate continuations for ``context``."""
        ...

    def score(self, context: str, outcome: str) -> float:
        """Rate an outcome in ``[0, 1]`` (higher is better)."""
        ...


def _propose_prompt(context: str, n: int) -> str:
    return (f"{context}\n\n"
            f"Provide exactly {n} distinct candidate solutions. Put each "
            "candidate in its own fenced ``` code block and output nothing "
            "else — no numbering, preamble, or commentary.")


def _score_prompt(context: str, outcome: str) -> str:
    return (f"{context}\n\nCandidate outcome:\n{outcome}\n\n"
            "Rate this outcome from 0.0 (worst) to 1.0 (best). Reply with only "
            "the number.")


def _parse_candidates(reply: str, n: int) -> list[str]:
    """Pull ``n`` candidates out of a reply: fenced code blocks if present
    (the reliable path for code), else the whole reply as one candidate.
    Pads by cycling and truncates so exactly ``n`` come back."""
    parts = [block.strip() for block in _FENCE.findall(reply) if block.strip()]
    if not parts:
        parts = [reply.strip()]
    if len(parts) < n:
        parts += [parts[i % len(parts)] for i in range(n - len(parts))]
    return parts[:n]


def _parse_score(reply: str) -> float:
    match = re.search(r"[-+]?\d*\.?\d+", reply)
    return max(0.0, min(1.0, float(match.group()))) if match else 0.0


class FakeLLM:
    """Deterministic ``LLM`` for tests: replays a fixed candidate list and
    looks scores up in a table (missing outcomes get ``default_score``)."""

    def __init__(self, candidates: list[str],
                 scores: dict[str, float] | None = None,
                 default_score: float = 0.0):
        if not candidates:
            raise ValueError("candidates must not be empty")
        self.candidates = list(candidates)
        self.scores = dict(scores or {})
        self.default_score = default_score

    def propose(self, context: str, n: int) -> list[str]:
        if n <= 0:
            raise ValueError("n must be positive")
        # cycle the script so a caller may ask for any n
        return [self.candidates[i % len(self.candidates)] for i in range(n)]

    def score(self, context: str, outcome: str) -> float:
        return self.scores.get(outcome, self.default_score)


class _HTTPChatLLM:
    """Shared prompting/parsing for HTTP chat models. Subclasses implement
    ``_message(prompt, max_tokens) -> str``."""

    max_tokens: int

    def _message(self, prompt: str, *, max_tokens: int | None = None) -> str:
        raise NotImplementedError

    def propose(self, context: str, n: int) -> list[str]:
        if n <= 0:
            raise ValueError("n must be positive")
        return _parse_candidates(self._message(_propose_prompt(context, n)), n)

    def score(self, context: str, outcome: str) -> float:
        reply = self._message(_score_prompt(context, outcome), max_tokens=16)
        return _parse_score(reply)


class AnthropicLLM(_HTTPChatLLM):
    """Calls the Anthropic Messages API over ``urllib`` (no SDK dependency).
    The API key defaults to ``ANTHROPIC_API_KEY``."""

    _URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, *, api_key: str | None = None,
                 model: str = "claude-3-5-haiku-20241022",
                 max_tokens: int = 1024, timeout: float = 60.0):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _message(self, prompt: str, *, max_tokens: int | None = None) -> str:
        body = {"model": self.model, "max_tokens": max_tokens or self.max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        request = urllib.request.Request(
            self._URL, data=json.dumps(body).encode(),
            headers={"content-type": "application/json",
                     "x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01",
                     "user-agent": "agentfork-harness/0.4"},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"Anthropic API failed: HTTP {exc.code}: {detail}") from exc
        return "".join(block.get("text", "")
                       for block in payload.get("content", []))


class OpenAICompatLLM(_HTTPChatLLM):
    """Calls any OpenAI-compatible ``/chat/completions`` endpoint (Together,
    etc.) over ``urllib``. Together's key defaults to ``TOGETHER_API_KEY``."""

    def __init__(self, *, api_key: str | None = None,
                 base_url: str = "https://api.together.xyz/v1",
                 model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                 max_tokens: int = 1024, timeout: float = 60.0):
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("an API key is required")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _message(self, prompt: str, *, max_tokens: int | None = None) -> str:
        body = {"model": self.model, "max_tokens": max_tokens or self.max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json",
                     "authorization": f"Bearer {self.api_key}",
                     "user-agent": "agentfork-harness/0.4"},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"OpenAI-compatible API failed: HTTP {exc.code}: "
                f"{detail}") from exc
        return payload["choices"][0]["message"]["content"]
