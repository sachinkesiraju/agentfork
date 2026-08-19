"""vLLM V1 attention backend with block-granular KVZip pruning.

This module registers a custom ``AttentionBackendEnum.CUSTOM`` backend that
automatically selects vLLM's FlashAttention backend when a GPU is available and
falls back to the CPU backend otherwise.  At every attention forward it scores each
physical KV block by the current query's average attention energy to the block,
keeps the top ``target_ratio`` fraction of used blocks plus the current-token
block, and zeroes the rest before calling the underlying paged-attention kernel.

The zeroing is a GPU/CPU-friendly proxy for true block eviction; a full paged
block-manager integration would free the physical blocks instead.  The module is
optional and degrades gracefully if vLLM is not installed.

Runtime budget is controlled via environment variables so it is visible to vLLM
worker processes spawned after startup:

* ``KVZIP_TARGET_RATIO`` -- fraction of used blocks to retain (default 1.0)
* ``KVZIP_MIN_BLOCKS``   -- minimum blocks to keep per sequence (default 1)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch

try:
    from vllm.platforms import current_platform
    from vllm.v1.attention.backend import AttentionImpl
    from vllm.v1.attention.backends.cpu_attn import (
        CPUAttentionBackend,
        CPUAttentionBackendImpl,
    )
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionBackend,
        FlashAttentionImpl,
    )
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    HAS_VLLM = True
except Exception as _err:  # pragma: no cover - vllm may not be installed
    HAS_VLLM = False
    AttentionImpl = object  # type: ignore[misc,assignment]
    CPUAttentionBackend = object  # type: ignore[misc,assignment]
    CPUAttentionBackendImpl = object  # type: ignore[misc,assignment]
    FlashAttentionBackend = object  # type: ignore[misc,assignment]
    FlashAttentionImpl = object  # type: ignore[misc,assignment]
    AttentionBackendEnum = None  # type: ignore[misc,assignment]
    register_backend = None  # type: ignore[misc,assignment]
    current_platform = None  # type: ignore[misc,assignment]
    logging.getLogger(__name__).debug("vllm backend not available: %s", _err)

# Triton backend may be unavailable in CPU-only vLLM wheels.
if HAS_VLLM:
    try:
        from vllm.v1.attention.backends.triton_attn import (
            TritonAttentionBackend,
            TritonAttentionImpl,
        )
    except Exception:
        TritonAttentionBackend = object  # type: ignore[misc,assignment]
        TritonAttentionImpl = object  # type: ignore[misc,assignment]
else:
    TritonAttentionBackend = object  # type: ignore[misc,assignment]
    TritonAttentionImpl = object  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    from vllm.model.layers.attention import AttentionLayer
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata


def _runtime_backend() -> type:
    """Return the underlying vLLM backend for the current device environment."""
    if not HAS_VLLM or not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return CPUAttentionBackend
    try:
        cap = current_platform.get_device_capability()
    except Exception:
        return CPUAttentionBackend
    if cap is None:
        return CPUAttentionBackend
    for backend in (FlashAttentionBackend, TritonAttentionBackend):
        if backend is object:
            continue
        try:
            if backend.supports_compute_capability(cap):
                return backend
        except Exception:
            continue
    return CPUAttentionBackend


def _runtime_impl_cls() -> type:
    """Return the underlying vLLM attention implementation for this process."""
    backend = _runtime_backend()
    if backend is TritonAttentionBackend and TritonAttentionImpl is not object:
        return TritonAttentionImpl
    if backend is FlashAttentionBackend:
        return FlashAttentionImpl
    return CPUAttentionBackendImpl


def _apply_kvzip_block_mask(
    kv_cache: torch.Tensor,
    query: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    target_ratio: float,
    min_blocks: int = 1,
    num_actual_tokens: int | None = None,
    num_heads: int | None = None,
    head_size: int | None = None,
) -> None:
    """Zero low-importance KV cache blocks before the attention kernel runs.

    Args:
        kv_cache: ``[num_blocks, num_kv_heads, block_size, 2*head_size]`` tensor.
        query: ``[num_tokens, num_heads, head_size]`` query tensor.
        block_table: ``[num_seqs, max_num_blocks]`` physical block indices.
        seq_lens: ``[num_seqs]`` total sequence lengths.
        target_ratio: Fraction of used blocks to retain (0, 1].
        min_blocks: Always keep at least this many blocks per sequence.
        num_actual_tokens: Number of real query tokens (ignore padding).
        num_heads: Number of query heads (defaults to ``query.shape[1]``).
    """
    if kv_cache.numel() == 0 or query.numel() == 0:
        return

    # Avoid mutating integer/quantized cache layouts; the current proof supports
    # floating-point (auto/float16/bfloat16/float32) caches only.
    if not kv_cache.is_floating_point():
        return

    num_blocks, num_kv_heads, block_size, _ = kv_cache.shape
    if num_blocks == 0 or block_size == 0:
        return

    head_dim = head_size if head_size is not None else kv_cache.shape[-1] // 2
    if head_dim == 0:
        return

    n_tokens = num_actual_tokens if num_actual_tokens is not None else query.shape[0]
    if n_tokens == 0:
        return

    n_heads = num_heads if num_heads is not None else query.shape[1]
    if n_heads == 0 or num_kv_heads == 0 or n_heads % num_kv_heads != 0:
        return
    n_query_per_kv = n_heads // num_kv_heads

    used = block_table.unique()
    used = used[(used >= 0) & (used < num_blocks)]
    if used.numel() == 0:
        return

    # Score each physical block by the current query's average dot-product
    # energy to every key position in the block.  This is a cheap, causal-safe,
    # query-aware block importance signal in the FastKVzip spirit.
    q = query[:n_tokens].to(torch.float32)
    q = q.reshape(n_tokens, num_kv_heads, n_query_per_kv, head_dim)
    k = kv_cache[..., :head_dim].to(torch.float32)  # [B, K, S, D]

    # [n_tokens, num_blocks, num_kv_heads, n_query_per_kv, block_size]
    attn = torch.einsum("n k q d, b k s d -> n b k q s", q, k)
    block_scores = attn.abs().mean(dim=(0, 2, 3, 4))  # [num_blocks]

    # Always retain the block containing the current token for every sequence.
    forced: set[int] = set()
    for seq_len, blocks in zip(seq_lens.tolist(), block_table.tolist(), strict=False):
        if seq_len <= 0:
            continue
        last_block_idx = (seq_len - 1) // block_size
        if last_block_idx < len(blocks):
            forced.add(int(blocks[last_block_idx]))

    forced_t = torch.tensor(list(forced), dtype=used.dtype, device=used.device)
    # Keep the highest-scoring used blocks (including forced ones).
    keep_count = max(min_blocks, int(used.numel() * target_ratio))
    keep_count = min(keep_count, used.numel())
    used_scores = block_scores[used]
    topk_indices = torch.topk(used_scores, keep_count).indices
    retained = used[topk_indices]

    if forced_t.numel() > 0:
        retained = torch.cat([retained, forced_t])
        retained = torch.unique(retained)

    drop_mask = torch.ones(num_blocks, dtype=torch.bool, device=kv_cache.device)
    drop_mask[retained] = False
    if drop_mask.any():
        kv_cache[drop_mask] = 0


_DEFAULT_TARGET_RATIO = float(os.environ.get("KVZIP_TARGET_RATIO", "1.0"))
_DEFAULT_MIN_BLOCKS = int(os.environ.get("KVZIP_MIN_BLOCKS", "1"))


def set_kvzip_budget(target_ratio: float, min_blocks: int = 1) -> None:
    """Set the global KVZip block-retention budget.

    Also updates the current process's environment so subsequently spawned
    vLLM worker processes inherit the values.
    """
    global _DEFAULT_TARGET_RATIO, _DEFAULT_MIN_BLOCKS
    _DEFAULT_TARGET_RATIO = float(target_ratio)
    _DEFAULT_MIN_BLOCKS = int(min_blocks)
    os.environ["KVZIP_TARGET_RATIO"] = str(_DEFAULT_TARGET_RATIO)
    os.environ["KVZIP_MIN_BLOCKS"] = str(_DEFAULT_MIN_BLOCKS)
    if HAS_VLLM:
        FastKVzipAttentionImpl.target_ratio = _DEFAULT_TARGET_RATIO
        FastKVzipAttentionImpl.min_blocks = _DEFAULT_MIN_BLOCKS


class _FastKVzipBackendMeta(type):
    """Forward all unhandled class attributes to the runtime vLLM backend."""

    def __getattr__(cls, name: str) -> object:
        if name.startswith("__"):
            raise AttributeError(name)
        backend = _runtime_backend()
        return getattr(backend, name)


if HAS_VLLM:

    class FastKVzipAttentionImpl(AttentionImpl):  # type: ignore[misc]
        """Attention impl that prunes low-scoring KV blocks per forward."""

        # Global compression budget knobs.  A production integration would thread
        # these through the vLLM request metadata instead.
        target_ratio: float = _DEFAULT_TARGET_RATIO
        min_blocks: int = _DEFAULT_MIN_BLOCKS

        def __init__(self, *args: object, **kwargs: object) -> None:
            # ``AttentionImpl.__init__`` is abstract and raises; do not call it.
            # Instead instantiate the real vLLM impl (Flash or CPU) that this
            # wrapper delegates to.
            impl_cls = _runtime_impl_cls()
            self._impl = impl_cls(*args, **kwargs)
            # Mirror a few attributes the caller may inspect.
            self.num_heads = self._impl.num_heads
            self.head_size = self._impl.head_size
            self.num_kv_heads = getattr(self._impl, "num_kv_heads", self.num_heads)
            self.attn_type = self._impl.attn_type

        def forward(  # type: ignore[override]
            self,
            layer: AttentionLayer,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: FlashAttentionMetadata | None,
            output: torch.Tensor,
            output_scale: torch.Tensor | None = None,
            output_block_scale: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if (
                attn_metadata is not None
                and kv_cache.numel() > 0
                and self.attn_type == "decoder"
                and self.target_ratio < 1.0
            ):
                _apply_kvzip_block_mask(
                    kv_cache,
                    query,
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                    target_ratio=self.target_ratio,
                    min_blocks=self.min_blocks,
                    num_actual_tokens=attn_metadata.num_actual_tokens,
                    num_heads=self.num_heads,
                    head_size=self.head_size,
                )
            return self._impl.forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )

    class FastKVzipBackend(metaclass=_FastKVzipBackendMeta):  # type: ignore[misc]
        """Custom vLLM backend that zeroes low-scoring KV blocks at runtime."""

        @staticmethod
        def get_name() -> str:
            return "KVZIP"

        @staticmethod
        def get_impl_cls() -> type[FastKVzipAttentionImpl]:
            return FastKVzipAttentionImpl

    # Register as the vLLM V1 custom attention backend.
    register_backend(  # type: ignore[misc]
        AttentionBackendEnum.CUSTOM,
        "agentfork.kvzip_router.vllm_backend.FastKVzipBackend",
    )

else:
    FastKVzipBackend = None  # type: ignore[assignment]
    FastKVzipAttentionImpl = None  # type: ignore[assignment]
