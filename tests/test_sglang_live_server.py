"""Live SGLang HTTP server integration test.

The rest of the SGLang HTTP coverage (test_sglang_http_backend.py) drives a
fake server, so it validates the client's protocol but not that the patched
SGLang server actually honors it. This test drives a *real* patched SGLang
HTTP server end to end — but only when one is provided, because standing it
up needs SGLang, a model, and (realistically) a GPU.

Run it against a server you have already started with patches 0001–0003
applied:

    AGENTFORK_SGLANG_URL=http://host:30000 \
    AGENTFORK_SGLANG_ADMIN_KEY=... \
    pytest tests/test_sglang_live_server.py -v

Without ``AGENTFORK_SGLANG_URL`` set it skips, so CI on GPU-less runners
stays green while the path is exercisable wherever a server exists.
"""

import os

import pytest

from agentfork import SGLangHTTPBackend

_URL = os.environ.get("AGENTFORK_SGLANG_URL")
pytestmark = pytest.mark.skipif(
    not _URL, reason="set AGENTFORK_SGLANG_URL to a patched SGLang server")


@pytest.fixture
def backend():
    return SGLangHTTPBackend(
        _URL, api_key=os.environ.get("AGENTFORK_SGLANG_ADMIN_KEY"))


def test_live_tree_lifecycle_create_fork_generate_kill(backend):
    root = "aftest-root"
    child = "aftest-root/1"
    prompt = "The capital of France is"
    try:
        backend.create_tree(root)
        assert backend.has_tree(root)

        # extend the root's cached prefix via a real generate on the tree
        backend.generate(root, prompt, max_new_tokens=1)

        branch = backend.fork_branch(root, child)
        assert branch.branch_id == child
        assert backend.has_tree(child)

        # the child continues from the parent's cached prefix
        out = backend.generate(child, prompt, max_new_tokens=4)
        assert isinstance(out, str) and out

        tel = backend.telemetry(root)
        assert isinstance(tel, dict)

        freed = backend.kill(child)
        assert isinstance(freed, int) and freed >= 0
        assert not backend.has_tree(child)
    finally:
        for tid in (child, root):
            try:
                backend.kill(tid)
            except Exception:
                pass


def test_live_public_generate_rejects_tree_fields(backend):
    # patch 0003 blocks tree_id/branch_id on the public generate path; only
    # the admin /tree_generate may carry them
    import urllib.error
    import urllib.request

    body = b'{"text": "hi", "tree_id": "x", "sampling_params": {"max_new_tokens": 1}}'
    req = urllib.request.Request(
        _URL.rstrip("/") + "/generate", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req, timeout=30)
    assert ei.value.code in (400, 403)
