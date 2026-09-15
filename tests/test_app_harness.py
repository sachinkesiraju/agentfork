"""Harness selection and CLI adapters: model flag plumbing, OpenAI-compatible
endpoint routing, and the /api/models loopback guard."""

import json

import pytest

import agentfork.app.harness as h
from agentfork.app.harness import (CodexHarness, CursorAgentHarness,
                                   HarnessError, OpenCodeHarness,
                                   build_harness)


def test_cli_harnesses_are_registered():
    for name, cls in (("codex", CodexHarness),
                      ("opencode", OpenCodeHarness),
                      ("cursor-agent", CursorAgentHarness)):
        harness = build_harness(name)
        assert isinstance(harness, cls)
        assert harness.model is None


def test_model_flag_is_appended(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)

        class P:
            returncode = 0
            stdout = "ok"
            stderr = ""
        return P()

    monkeypatch.setattr(h.subprocess, "run", fake_run)
    build_harness("codex", model="o4-mini").implement("/tmp", idea=_idea(),
                                                    context="")
    assert calls[0][:3] == ["codex", "exec", "--full-auto"]
    assert calls[0][-2:] == ["--model", "o4-mini"]


def test_model_flag_omitted_when_unset(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)

        class P:
            returncode = 0
            stdout = "ok"
            stderr = ""
        return P()

    monkeypatch.setattr(h.subprocess, "run", fake_run)
    build_harness("opencode").implement("/tmp", idea=_idea(), context="")
    assert calls[0] == ["opencode", "run",
                        calls[0][2]]  # prompt only, no -m


def test_cli_probe_reports_uninstalled(monkeypatch):
    monkeypatch.setattr(h.shutil, "which", lambda b: None)
    probe = CodexHarness().probe()
    assert probe == {"installed": False, "authed": False}


def test_api_custom_endpoint(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.setenv("AGENTFORK_API_KEY", "sekret")
    harness = build_harness("api", model="qwen3:14b",
                          api_base="http://127.0.0.1:11434/v1")
    llm = harness.llm
    assert llm.base_url == "http://127.0.0.1:11434/v1"
    assert llm.model == "qwen3:14b"
    assert llm.api_key == "sekret"


def test_api_keyless_local_endpoint_omits_auth(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["headers"] = dict(req.header_items())

        class R:
            def read(self):
                return json.dumps({"choices": [{"message":
                                                {"content": "hi"}}]
                                   }).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        return R()

    import agentfork.harness.adapter as adapter
    monkeypatch.setattr(adapter.urllib.request, "urlopen", fake_urlopen)
    llm = build_harness(
        "api", api_base="http://127.0.0.1:1234/v1").llm
    llm._message("ping")
    assert "authorization" not in captured["headers"]


def test_api_models_lists_ids(monkeypatch):
    def fake_urlopen(req, timeout=None):
        class R:
            def read(self):
                return json.dumps(
                    {"data": [{"id": "m1"}, {"id": "m2"}]}).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        return R()

    monkeypatch.setattr(h.urllib.request, "urlopen", fake_urlopen)
    assert h.list_models("http://127.0.0.1:11434/v1/") == ["m1", "m2"]


def test_detect_includes_all_harnesses(monkeypatch):
    monkeypatch.setattr(h.shutil, "which", lambda b: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv("AGENTFORK_API_BASE", raising=False)
    found = h.detect_harnesses()
    assert set(found) == {"claude-code", "codex", "opencode",
                          "cursor-agent", "api", "fake"}
    assert found["api"]["authed"] is False


def test_unknown_harness_raises():
    with pytest.raises(HarnessError):
        build_harness("nope")


def _idea():
    return h.Idea(title="x", description="do x")
