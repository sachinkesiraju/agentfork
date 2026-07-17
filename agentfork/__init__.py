"""agentfork: coordinated sandbox and LLM KV-cache branch lifecycle.

Stable public surface. Everything importable from ``agentfork`` directly is
covered by semantic versioning from 0.2.0 onward; submodule internals are not.
"""

from agentfork.kill.reaper import BranchReaper, KillResult
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

__version__ = "0.2.0"

__all__ = [
    "Branch",
    "BranchReaper",
    "CacheStats",
    "ForkOrchestrator",
    "KVBackend",
    "KillReceipt",
    "KillResult",
    "NullSandbox",
    "ReaperSandbox",
    "SandboxBackend",
    "TreeId",
    "TreeKVCache",
    "__version__",
]
