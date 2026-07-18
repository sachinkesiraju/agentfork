"""agentfork: coordinated sandbox and LLM KV-cache branch lifecycle.

Stable public surface. Everything importable from ``agentfork`` directly is
covered by semantic versioning from 0.2.0 onward; submodule internals are not.
"""

from agentfork.kill.reaper import BranchReaper, KillResult
from agentfork.kv.sglang_http_backend import SGLangHTTPBackend
from agentfork.kv.tree_cache import CacheStats, TreeId, TreeKVCache
from agentfork.orchestrator import (
    Branch,
    ForkOrchestrator,
    KillReceipt,
    KVBackend,
    NullSandbox,
    ReaperSandbox,
    SandboxBackend,
)
from agentfork.sandbox.fc_bench import JailerConfig
from agentfork.sandbox.netns import NetworkConfig
from agentfork.sandbox.vsock import DetachedExec, ExecResult, VsockError

__version__ = "0.3.0"

__all__ = [
    "Branch",
    "BranchReaper",
    "CacheStats",
    "DetachedExec",
    "ExecResult",
    "ForkOrchestrator",
    "KVBackend",
    "KillReceipt",
    "JailerConfig",
    "KillResult",
    "NetworkConfig",
    "NullSandbox",
    "ReaperSandbox",
    "SandboxBackend",
    "SGLangHTTPBackend",
    "TreeId",
    "TreeKVCache",
    "VsockError",
    "__version__",
]
