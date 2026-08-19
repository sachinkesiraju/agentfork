"""kvzip_router: a router-programmable KV-cache compression primitive."""

from .baselines import full_cache_reference, uniform_compress
from .budget import BudgetAllocator
from .compressor import CompressedKVCache, KVCompressor
from .distill import FullCacheTeacher, OnlineDistiller
from .gate import KVCacheGate
from .rerank import TokenRelevanceRerank
from .router import KVBudgetHint, KVBudgetRouter
from .trace import TraceJournal, TraceTrainer
from .transfer import warm_start_gate

try:
    from .kvzip_backend import KVZipBackend
except ImportError:  # agentfork not installed
    KVZipBackend = None  # type: ignore[assignment]

__all__ = [
    "BudgetAllocator",
    "CompressedKVCache",
    "FullCacheTeacher",
    "KVCompressor",
    "KVCacheGate",
    "KVBudgetHint",
    "KVBudgetRouter",
    "KVZipBackend",
    "OnlineDistiller",
    "TokenRelevanceRerank",
    "TraceJournal",
    "TraceTrainer",
    "full_cache_reference",
    "uniform_compress",
    "warm_start_gate",
]
