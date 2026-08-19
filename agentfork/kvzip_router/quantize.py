"""Optional KV quantization helpers for v1.

This is intentionally minimal: a real backend will use fused 4-bit KV kernels.
v1 provides a symmetric 8-bit round-trip so tests can prove the prune+quantize
memory reduction is multiplicative without requiring a GPU kernel.
"""

from __future__ import annotations

import torch


def quantize_tensor(
    x: torch.Tensor, bits: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel min/max quantization.

    Args:
        x: tensor of any shape.  The last dimension is treated as the channel.
        bits: target bit width.

    Returns:
        (quantized integer tensor, scale tensor).
    """
    if bits not in (8, 4):
        raise ValueError("v1 only supports 8- or 4-bit symmetric quantization")
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / qmax
    q = torch.clamp(torch.round(x / scale), qmin, qmax).to(torch.int8)
    return q, scale.squeeze(-1)


def dequantize_tensor(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize an integer tensor back to the original dtype."""
    return q.to(torch.float32) * scale.view(
        *scale.shape, *([1] * (q.dim() - scale.dim()))
    )
