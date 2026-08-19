"""Fused GPU kernels for KV cache compression.

The main proof point is a Triton 4-bit packing/unpacking kernel that stores two
4-bit quantized values per ``uint8`` and recovers them with a single fused
dequantization step.  It is optional: the module degrades to a CPU PyTorch
fallback when Triton or a CUDA device is unavailable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception as _err:  # pragma: no cover - Triton may not be installed
    HAS_TRITON = False
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    logging.getLogger(__name__).debug("Triton not available: %s", _err)

if TYPE_CHECKING:  # pragma: no cover
    pass


_BITS = 4
_QMAX = 2 ** (_BITS - 1) - 1  # 7 for signed 4-bit
_QMIN = -(2 ** (_BITS - 1))  # -8


def _compute_scale(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor scale that maps the max absolute value to the signed 4-bit range."""
    max_abs = x.abs().max()
    return max_abs.clamp_min(1e-6) / _QMAX


if HAS_TRITON:
    # Triton kernels can only read globals that are constexpr.
    _TL_QMIN = tl.constexpr(float(_QMIN))
    _TL_QMAX = tl.constexpr(float(_QMAX))

    @triton.jit  # type: ignore[misc]
    def _pack_4bit_kernel(
        x_ptr,
        out_ptr,
        scale_ptr,
        n,
    ):
        pid = tl.program_id(0)
        i0 = 2 * pid
        i1 = i0 + 1

        x0 = tl.load(x_ptr + i0, mask=i0 < n, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + i1, mask=i1 < n, other=0.0).to(tl.float32)

        inv_scale = 1.0 / tl.load(scale_ptr)
        q0_f = tl.clamp(x0 * inv_scale, _TL_QMIN, _TL_QMAX)
        q1_f = tl.clamp(x1 * inv_scale, _TL_QMIN, _TL_QMAX)
        q0 = tl.extra.cuda.libdevice.llrint(q0_f).to(tl.int8)
        q1 = tl.extra.cuda.libdevice.llrint(q1_f).to(tl.int8)

        # Store only the low 4 bits of each quantized value in a single byte.
        packed = (q0.to(tl.uint8) & 0xF) | ((q1.to(tl.uint8) & 0xF) << 4)
        tl.store(out_ptr + pid, packed)

    @triton.jit  # type: ignore[misc]
    def _unpack_4bit_kernel(
        packed_ptr,
        out_ptr,
        scale_ptr,
        n,
    ):
        pid = tl.program_id(0)
        i0 = 2 * pid
        i1 = i0 + 1

        p = tl.load(packed_ptr + pid).to(tl.uint8)
        q0_u = (p & 0xF).to(tl.int32)
        q1_u = ((p >> 4) & 0xF).to(tl.int32)

        # Convert unsigned 4-bit nibble to signed 4-bit value.
        q0 = tl.where(q0_u > 7, q0_u - 16, q0_u).to(tl.float32)
        q1 = tl.where(q1_u > 7, q1_u - 16, q1_u).to(tl.float32)

        scale = tl.load(scale_ptr)
        tl.store(out_ptr + i0, q0 * scale, mask=i0 < n)
        tl.store(out_ptr + i1, q1 * scale, mask=i1 < n)


def pack_4bit_fused(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused 4-bit quantization: returns ``(packed_uint8, scale)``.

    ``packed_uint8`` has shape ``x.shape[:-1] + (x.shape[-1] // 2 + x.shape[-1] % 2,)``
    and stores two 4-bit signed values per byte.  The scale is a per-tensor scalar
    that maps the max absolute input value into the signed 4-bit grid.

    Falls back to a non-fused PyTorch implementation on CPU.
    """
    if x.numel() == 0:
        return x, torch.tensor(1.0, device=x.device, dtype=x.dtype)

    scale = _compute_scale(x)
    flat = x.reshape(-1)
    n = flat.numel()
    out_len = (n + 1) // 2

    if x.is_cuda and HAS_TRITON:
        packed = torch.empty(out_len, device=x.device, dtype=torch.uint8)
        grid = (out_len,)
        _pack_4bit_kernel[grid](  # type: ignore[index]
            flat,
            packed,
            scale,
            n,
        )
        return packed, scale

    # CPU fallback: the same math, but not fused into a single kernel.
    inv_scale = 1.0 / scale
    q = torch.clamp(torch.round(flat * inv_scale), _QMIN, _QMAX).to(torch.int8)
    q0 = q[0::2] & 0xF
    q1 = (q[1::2] & 0xF) << 4
    if n % 2 == 1:
        q1 = torch.nn.functional.pad(q1, (0, 1), value=0)
    packed = (q0 | q1).to(torch.uint8)
    return packed, scale


def unpack_4bit_fused(
    packed: torch.Tensor,
    scale: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Fused dequantization of ``pack_4bit_fused`` output back to ``shape``."""
    if packed.numel() == 0:
        return torch.zeros(shape, device=packed.device, dtype=scale.dtype)

    n = int(torch.prod(torch.tensor(shape, dtype=torch.int64)))

    if packed.is_cuda and HAS_TRITON:
        out = torch.empty(n, device=packed.device, dtype=scale.dtype)
        grid = (packed.numel(),)
        _unpack_4bit_kernel[grid](  # type: ignore[index]
            packed,
            out,
            scale,
            n,
        )
        return out.reshape(shape)

    # CPU fallback.
    p = packed.to(torch.int32)
    q0_u = p & 0xF
    q1_u = (p >> 4) & 0xF
    q = torch.empty(2 * p.numel(), dtype=torch.float32, device=packed.device)
    q[0::2] = q0_u.to(torch.float32)
    q[1::2] = q1_u.to(torch.float32)
    q = q[:n]
    q = torch.where(q > 7, q - 16, q)
    return (q * scale).reshape(shape)
