"""A live HTTP server exposing agentfork's patched-SGLang tree-cache endpoints.

This stands up ``/tree_cache`` and ``/tree_generate`` (plus a guarded
``/generate`` and ``/health``) over a real socket so that
``agentfork.kv.SGLangHTTPBackend`` can be exercised against a *live* server
instead of an in-process protocol stub. It is the missing half of the
"SGLangHTTPBackend has never run against a live SGLang server" gap.

What is REAL here (imported from the patched SGLang checkout, no
re-implementation):

* the KV pool + allocator (``MHATokenToKVPool`` / ``TokenToKVPoolAllocator``
  on ``device="cpu"`` -- the same "small CPU-tensor pool" the repo's
  ``patches/real_pool_validation.py`` uses to measure real slot dedup);
* the tree-cache primitive (``TreeRadixCache``): fork/kill/reserve/telemetry,
  ``lock_ref`` pinning, prefix matching, charge accounting, eviction;
* the request-lifecycle handler (``TreeCacheLifecycle.handle``) -- the exact
  code patch 0002 wires into SGLang's scheduler for ``/tree_cache`` ops;
* the committed-prefix charge validation from patch 0003
  (``TreeRadixCache._prepare_request_update``), invoked on every
  ``/tree_generate`` request;
* the admin auth decision (``decide_request_auth`` / ``AuthLevel``) that patch
  0003 marks ``/tree_cache`` and ``/tree_generate`` with (``ADMIN_FORCE``).

What is NOT real (requires a GPU + model weights, unreachable on a CPU box):
the transformer forward pass. ``/tree_generate`` therefore does the cache work
a real prefill does -- match the shared prefix, allocate real KV slots for the
uncached suffix, insert them, charge/telemeter the branch -- and returns
``meta_info.cached_tokens`` computed by the real cache, but the ``text`` field
is a deterministic stub, not model output. Every response carries
``meta_info.model_output = false`` to make that explicit.

Run (needs the patched SGLang checkout on PYTHONPATH):

    PYTHONPATH=/path/to/sglang/python python demo/sglang_tree_server.py \
        --port 30000 --admin-api-key admin-secret

Tokenization is byte-level (``prompt.encode("utf-8")``) so a child's prompt
that is a string-extension of its parent's committed prompt is guaranteed to
be a token-level extension too -- which is exactly the shared-repo-context
fanout shape and what lets the real cache report a nonzero cached-token reuse.
"""

from __future__ import annotations

import argparse
import json
import threading
from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from sglang.srt.managers.io_struct import TreeCacheOpReqInput
from sglang.srt.managers.tree_cache_lifecycle import TreeCacheLifecycle
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.tree_radix_cache import TreeRadixCache
from sglang.srt.utils.auth import AuthLevel, decide_request_auth

# Llama-1B-like KV shape, matching patches/real_pool_validation.py.
POOL_TOKENS = 65536
KV_SHAPE = dict(head_num=8, head_dim=64, layer_num=16)

# Per-path auth level, mirroring the @auth_level decorators patch 0003 adds to
# SGLang's http_server: the two admin control paths are ADMIN_FORCE.
_AUTH_LEVELS = {
    "/tree_cache": AuthLevel.ADMIN_FORCE,
    "/tree_generate": AuthLevel.ADMIN_FORCE,
    "/generate": AuthLevel.NORMAL,
}


def tokenize(prompt) -> list[int]:
    """Deterministic byte-level tokenizer (no model tokenizer available).

    String-prefix relationships are preserved at the token level, which is
    what makes the shared-context fanout produce real cached-token reuse."""
    if isinstance(prompt, list):
        return [int(t) for t in prompt]
    return list(prompt.encode("utf-8"))


class TreeCacheServer:
    """Owns the real KV pool + TreeRadixCache and serializes access to it.

    The real SGLang scheduler serializes tree ops through a single loop; a
    lock around the (not thread-safe) cache is the faithful equivalent for a
    threaded HTTP server.
    """

    def __init__(self, *, admin_api_key: str | None, api_key: str | None,
                 quota_tokens: int | None):
        self.admin_api_key = admin_api_key
        self.api_key = api_key
        self._lock = threading.Lock()

        self.kvcache = MHATokenToKVPool(
            size=POOL_TOKENS, page_size=1, dtype=torch.float16,
            device="cpu", enable_memory_saver=False, **KV_SHAPE)
        self.alloc = TokenToKVPoolAllocator(
            size=POOL_TOKENS, dtype=torch.float16, device="cpu",
            kvcache=self.kvcache, need_sort=False)
        params = CacheInitParams(
            disable=False, req_to_token_pool=None,
            token_to_kv_pool_allocator=self.alloc, page_size=1)
        self.cache = TreeRadixCache(params, tree_quota_tokens=quota_tokens)
        self.lifecycle = TreeCacheLifecycle(self.cache)
        self.baseline_available = self.alloc.available_size()

    def used(self) -> int:
        return POOL_TOKENS - self.alloc.available_size()

    # -- auth --------------------------------------------------------------

    def authorize(self, method: str, path: str, authorization: str | None):
        level = _AUTH_LEVELS.get(path, AuthLevel.NORMAL)
        return decide_request_auth(
            method=method, path=path, authorization_header=authorization,
            api_key=self.api_key, admin_api_key=self.admin_api_key,
            auth_level=level)

    # -- /tree_cache -------------------------------------------------------

    def tree_cache_op(self, body: dict) -> tuple[int, dict]:
        """Dispatch a lifecycle op through the REAL TreeCacheLifecycle.handle."""
        try:
            req = TreeCacheOpReqInput(
                operation=body["operation"], branch_id=body["branch_id"],
                tree_id=body.get("tree_id"), parent_id=body.get("parent_id"),
                tokens=body.get("tokens"))
        except (KeyError, TypeError) as exc:
            return 400, {"success": False, "message": f"bad request: {exc}"}

        with self._lock:
            result = self.lifecycle.handle(req)
        if not result.success:
            # Same shape as patch 0002's http_server.tree_cache handler.
            return 400, {"success": False, "message": result.message}
        return 200, {"success": True, "value": result.value}

    # -- /tree_generate ----------------------------------------------------

    def tree_generate(self, body: dict) -> tuple[int, dict]:
        branch_id = body.get("branch_id")
        if branch_id is None:
            return 400, {"error": "branch_id is required for /tree_generate"}
        reserve = body.get("branch_reserve_tokens")
        if reserve is not None and reserve < 0:
            return 400, {"error": "branch_reserve_tokens must be nonnegative"}
        if "input_ids" in body:
            token_ids = tokenize(body["input_ids"])
        elif "text" in body:
            token_ids = tokenize(body["text"])
        else:
            return 400, {"error": "text or input_ids is required"}
        namespace = body.get("tree_id") or branch_id
        parent_id = body.get("parent_id")
        max_new = int(body.get("sampling_params", {}).get("max_new_tokens", 0))

        with self._lock:
            try:
                hit, charged, prompt_tokens = self._prefill(
                    branch_id, namespace, parent_id, token_ids, reserve)
            except KeyError as exc:
                return 400, {"error": f"unknown branch: {exc}"}
            except (ValueError, MemoryError) as exc:
                return 400, {"error": str(exc)}

        return 200, {
            "text": f"[no-model stub output; max_new_tokens={max_new}]",
            "meta_info": {
                "cached_tokens": hit,
                "charged_tokens": charged,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": max_new,
                "finish_reason": {"type": "length", "length": max_new},
                "model_output": False,
            },
            "index": 0,
        }

    def _prefill(self, branch_id, namespace, parent_id, token_ids, reserve):
        """Reproduce the cache effect of a real prefill on the branch.

        Mirrors the scheduler request path: validate the branch/namespace as
        TreeCacheLifecycle.prepare_request does, apply the fork-time
        reservation, then allocate real KV slots for the uncached suffix and
        insert them via the real cache -- so cached_tokens/charge/telemetry are
        the cache's own numbers, not this server's."""
        cache = self.cache
        if not cache.has_branch(branch_id):
            raise KeyError(branch_id)
        br = cache.branch(branch_id)
        if br.namespace != namespace:
            raise ValueError(
                f"branch {branch_id} belongs to tree {br.namespace}, "
                f"not {namespace}")
        if parent_id is not None and br.parent_id != parent_id:
            raise ValueError(
                f"branch {branch_id} has parent {br.parent_id}, "
                f"not {parent_id}")
        if reserve:
            cache.reserve(branch_id, reserve)  # real admission control

        committed = list(br.token_ids)
        # Same invariant patch 0003 enforces in _prepare_request_update.
        if len(token_ids) < len(committed) or token_ids[:len(committed)] != committed:
            raise ValueError(
                f"request for branch {branch_id} does not extend its "
                "committed token prefix")
        suffix = token_ids[len(committed):]

        # Match the full prompt first: any token already resident (this
        # branch's committed prefix, plus prefix pages a sibling already
        # inserted into the shared namespace) is reused, not re-allocated.
        # This mirrors the scheduler prefill -- allocate slots ONLY for the
        # uncached tail (`charged`), and reuse the matched device slots for
        # everything else -- so the pool never leaks slots for shared prefixes.
        key = RadixKey(array("q", token_ids), extra_key=br.namespace)
        resident = self.cache.match_prefix(MatchPrefixParams(key=key)).device_indices
        hit_pre = len(resident)
        charged = len(token_ids) - max(hit_pre, len(committed))
        if charged < 0:
            raise ValueError(f"branch {branch_id}: negative charge")
        if charged:
            new_slots = self.alloc.alloc(charged)
            if new_slots is None:
                raise MemoryError(f"KV pool exhausted allocating {charged} slots")
            full_value = torch.cat([resident, new_slots])
        else:
            full_value = resident
        hit = cache.extend_tree(branch_id, suffix, value=full_value)
        return hit, charged, len(token_ids)


class _Handler(BaseHTTPRequestHandler):
    server_version = "AgentforkTreeCache/1"

    @property
    def app(self) -> TreeCacheServer:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, *args):  # quieter logs
        pass

    def _send(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path.startswith("/health"):
            self._send(200, {"status": "ok"})
            return
        if path == "/pool_stats":
            # Diagnostic view of the REAL allocator, so a client can verify
            # KV slots return to baseline after kills. Read-only, no secrets.
            app = self.app
            self._send(200, {
                "pool_tokens": POOL_TOKENS,
                "used": app.used(),
                "available": app.alloc.available_size(),
                "baseline_available": app.baseline_available,
                "live_branches": app.cache.live_branches(),
            })
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        decision = self.app.authorize(
            "POST", path, self.headers.get("Authorization"))
        if not decision.allowed:
            code = decision.error_status_code
            self._send(code, {"error": "Unauthorized" if code == 401
                              else "Forbidden"})
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON"})
            return
        if path == "/tree_cache":
            status, payload = self.app.tree_cache_op(body)
        elif path == "/tree_generate":
            status, payload = self.app.tree_generate(body)
        elif path == "/generate":
            # Patch 0003's guard: agent-tree fields must go to /tree_generate.
            if body.get("branch_id") is not None:
                self._send(403, {
                    "error": "agent-tree fields require the admin "
                             "/tree_generate endpoint"})
                return
            status, payload = 200, {"text": "[no-model stub]",
                                    "meta_info": {"model_output": False}}
        else:
            status, payload = 404, {"error": "not found"}
        self._send(status, payload)


def build_server(host: str, port: int, *, admin_api_key: str | None,
                 api_key: str | None, quota_tokens: int | None):
    app = TreeCacheServer(admin_api_key=admin_api_key, api_key=api_key,
                          quota_tokens=quota_tokens)
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.app = app  # type: ignore[attr-defined]
    return httpd, app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--admin-api-key", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--quota-tokens", type=int, default=None,
                        help="per-tree HBM quota (TreeRadixCache quota)")
    args = parser.parse_args()

    httpd, app = build_server(
        args.host, args.port, admin_api_key=args.admin_api_key,
        api_key=args.api_key, quota_tokens=args.quota_tokens)
    pool_gib = POOL_TOKENS * (2 * KV_SHAPE["layer_num"] * KV_SHAPE["head_num"]
                              * KV_SHAPE["head_dim"] * 2) / 2**30
    print(f"live tree-cache server on http://{args.host}:{args.port} "
          f"(real {pool_gib:.2f} GiB CPU KV pool, "
          f"admin_api_key={'set' if args.admin_api_key else 'unset'})",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
