"""Remote SGLang tree-cache lifecycle over its administrative HTTP endpoint.

Covers lifecycle and inference: create/fork/kill/reserve/telemetry map to
``/tree_cache`` operations, while ``generate()`` sends native SGLang requests
carrying the same tree and branch identity. ``extend()`` is deliberately
unsupported because remote token accounting only occurs on the inference
request path."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


class SGLangHTTPBackend:
    external_data_path = True

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        admin_api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        retry_delay: float = 0.1,
        allow_insecure: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        if not self.base_url:
            raise ValueError("base_url is required")
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("base_url must be an http(s) URL")
        local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
        self.admin_api_key = admin_api_key or api_key
        if not local and not allow_insecure:
            if parsed.scheme != "https":
                raise ValueError(
                    "remote SGLang control requires HTTPS; pass "
                    "allow_insecure=True only on a trusted private network")
            if not self.admin_api_key:
                raise ValueError(
                    "remote SGLang control requires an admin API key")
        if max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._condition = threading.Condition(threading.RLock())
        self._active: dict[str, int] = {}
        self._closing_branches: set[str] = set()
        self._namespaces: dict[str, str] = {}
        self._parents: dict[str, str | None] = {}

    def create_tree(self, tree_id: str):
        value = self._operation("create", tree_id, tree_id=tree_id)
        with self._condition:
            self._namespaces[tree_id] = tree_id
            self._parents[tree_id] = None
        return value

    def fork_branch(self, parent_id: str, child_id: str | None = None):
        if child_id is None:
            raise ValueError("child_id is required for remote SGLang forks")
        with self._condition:
            if parent_id in self._closing_branches:
                raise RuntimeError(f"parent branch is being killed: {parent_id}")
            try:
                namespace = self._namespaces[parent_id]
            except KeyError:
                raise KeyError(f"untracked parent branch: {parent_id}") from None
            # Register as in-flight against the parent so a concurrent kill of
            # the parent waits for this fork instead of deleting it mid-request
            # (which would leave the child pointing at a gone parent).
            self._active[parent_id] = self._active.get(parent_id, 0) + 1
        try:
            self._operation("fork", child_id, parent_id=parent_id)
        finally:
            with self._condition:
                remaining = self._active[parent_id] - 1
                if remaining:
                    self._active[parent_id] = remaining
                else:
                    self._active.pop(parent_id, None)
                self._condition.notify_all()
        with self._condition:
            self._namespaces[child_id] = namespace
            self._parents[child_id] = parent_id
        return _Branch(child_id)

    def kill(self, tree_id: str) -> int:
        deadline = time.monotonic() + self.timeout
        with self._condition:
            while tree_id in self._closing_branches:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timed out waiting for concurrent kill of {tree_id}")
                self._condition.wait(remaining)
            self._closing_branches.add(tree_id)
            while self._active.get(tree_id, 0):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._closing_branches.discard(tree_id)
                    self._condition.notify_all()
                    raise TimeoutError(
                        f"timed out waiting for inference on {tree_id}")
                self._condition.wait(remaining)
        try:
            value = self._operation("kill", tree_id).get("value")
            with self._condition:
                self._namespaces.pop(tree_id, None)
                self._parents.pop(tree_id, None)
            return int(value or 0)
        finally:
            with self._condition:
                self._closing_branches.discard(tree_id)
                self._condition.notify_all()

    def extend(self, tree_id: str, tokens: list[int]) -> int:
        raise RuntimeError(
            "remote SGLang branches are populated through generate(), not "
            "extend(); use ForkOrchestrator.generate()")

    def reserve(self, tree_id: str, tokens: int) -> None:
        if tokens < 0:
            raise ValueError(f"reserve tokens must be non-negative: {tokens}")
        self._operation("reserve", tree_id, tokens=tokens)

    def telemetry(self, tree_id: str) -> dict:
        return self._operation("telemetry", tree_id).get("value") or {}

    def has_tree(self, tree_id: str) -> bool:
        try:
            self.telemetry(tree_id)
            return True
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "no such" in msg or "not found" in msg or "http 404" in msg:
                return False
            raise

    def generate(
        self,
        branch_id: str,
        prompt: str | list[int],
        sampling_params: dict,
        *,
        branch_end: bool = False,
        reserve_tokens: int | None = None,
    ) -> dict:
        if reserve_tokens is not None and reserve_tokens < 0:
            raise ValueError(
                f"reserve_tokens must be non-negative: {reserve_tokens}")
        with self._condition:
            if branch_id in self._closing_branches:
                raise RuntimeError(f"branch is being killed: {branch_id}")
            try:
                namespace = self._namespaces[branch_id]
            except KeyError:
                raise KeyError(f"untracked branch: {branch_id}") from None
            parent_id = self._parents[branch_id]
            self._active[branch_id] = self._active.get(branch_id, 0) + 1
        body = {
            "sampling_params": dict(sampling_params),
            "tree_id": namespace,
            "branch_id": branch_id,
            "branch_end": branch_end,
        }
        if parent_id is not None:
            body["parent_id"] = parent_id
        if reserve_tokens is not None:
            body["branch_reserve_tokens"] = reserve_tokens
        if isinstance(prompt, str):
            body["text"] = prompt
        else:
            body["input_ids"] = list(prompt)
        try:
            return self._request_json("/tree_generate", body, retry=False)
        finally:
            with self._condition:
                remaining = self._active[branch_id] - 1
                if remaining:
                    self._active[branch_id] = remaining
                else:
                    self._active.pop(branch_id, None)
                self._condition.notify_all()

    def _operation(self, operation: str, branch_id: str, **fields) -> dict:
        body = {"operation": operation, "branch_id": branch_id, **fields}
        payload = self._request_json(
            "/tree_cache",
            body,
            retry=operation in ("kill", "telemetry", "invalidate"),
        )
        if not payload.get("success"):
            message = payload.get("message", "unknown error")
            # kill is retried, and it is not idempotent server-side: if the
            # first attempt succeeded but its response was lost, the retry hits
            # an already-deleted branch. The delete did happen, so treat "no
            # such branch" on a kill as success rather than a spurious failure.
            if operation == "kill" and "no such" in message.lower():
                return payload
            raise RuntimeError(
                f"SGLang tree-cache operation {operation} failed: {message}"
            )
        return payload

    def _request_json(self, path: str, body: dict, *, retry: bool) -> dict:
        headers = {"Content-Type": "application/json"}
        key = (
            self.admin_api_key
            if path in ("/tree_cache", "/tree_generate")
            else self.api_key
        )
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        attempts = self.max_retries if retry else 0
        for attempt in range(attempts + 1):
            try:
                with urllib.request.urlopen(
                        request, timeout=self.timeout) as response:
                    raw = response.read().decode()
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"SGLang {path} returned invalid JSON") from exc
                if not isinstance(payload, dict):
                    raise RuntimeError(
                        f"SGLang {path} returned a non-object response")
                return payload
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                if exc.code not in (429, 502, 503, 504) or attempt >= attempts:
                    raise RuntimeError(
                        f"SGLang {path} failed: HTTP {exc.code}: {detail}"
                    ) from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                if attempt >= attempts:
                    raise RuntimeError(
                        f"SGLang {path} request failed: {exc}") from exc
            time.sleep(self.retry_delay * (2 ** attempt))
        raise AssertionError("unreachable")


class _Branch:
    def __init__(self, branch_id: str):
        self.branch_id = branch_id
