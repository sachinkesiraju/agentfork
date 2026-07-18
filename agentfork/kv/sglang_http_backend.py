"""Remote SGLang tree-cache lifecycle over its administrative HTTP endpoint."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class SGLangHTTPBackend:
    external_data_path = True

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        if not self.base_url:
            raise ValueError("base_url is required")
        self.api_key = api_key
        self.timeout = timeout

    def create_tree(self, tree_id: str):
        return self._operation("create", tree_id, tree_id=tree_id)

    def fork_branch(self, parent_id: str, child_id: str | None = None):
        if child_id is None:
            raise ValueError("child_id is required for remote SGLang forks")
        self._operation("fork", child_id, parent_id=parent_id)
        return _Branch(child_id)

    def kill(self, tree_id: str) -> int:
        value = self._operation("kill", tree_id).get("value")
        return int(value or 0)

    def extend(self, tree_id: str, tokens: list[int]) -> int:
        return 0

    def reserve(self, tree_id: str, tokens: int) -> None:
        self._operation("reserve", tree_id, tokens=tokens)

    def telemetry(self, tree_id: str) -> dict:
        return self._operation("telemetry", tree_id).get("value") or {}

    def _operation(self, operation: str, branch_id: str, **fields) -> dict:
        body = {"operation": operation, "branch_id": branch_id, **fields}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + "/tree_cache",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"SGLang tree-cache operation {operation} failed: "
                f"HTTP {exc.code}: {detail}"
            ) from exc
        if not payload.get("success"):
            raise RuntimeError(
                f"SGLang tree-cache operation {operation} failed: "
                f"{payload.get('message', 'unknown error')}"
            )
        return payload


class _Branch:
    def __init__(self, branch_id: str):
        self.branch_id = branch_id
