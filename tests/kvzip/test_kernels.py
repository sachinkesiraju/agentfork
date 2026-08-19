"""Tests for fused KV-cache kernels."""

import pytest
import torch

from agentfork.kvzip_router.kernels import pack_4bit_fused, unpack_4bit_fused


def test_4bit_roundtrip_shape_and_error():
    """4-bit pack+unpack roundtrip preserves shape and has bounded max error."""
    x = torch.randn(4, 8, 32, 64)
    packed, scale = pack_4bit_fused(x)
    # Two 4-bit values packed per uint8 -> half the element count (rounded up).
    assert packed.numel() == (x.numel() + 1) // 2
    assert packed.dtype == torch.uint8

    recovered = unpack_4bit_fused(packed, scale, x.shape)
    assert recovered.shape == x.shape
    max_abs_error = (x - recovered).abs().max().item()
    # 4-bit signed grid gives a worst-case quantization bin of scale/7. Two values
    # may round in opposite directions, so the error is bounded by ~2*scale.
    assert max_abs_error <= 2 * scale.item() + 1e-5


def test_4bit_odd_numel():
    """Packing handles odd total element counts."""
    x = torch.randn(2, 3, 7)
    packed, scale = pack_4bit_fused(x)
    recovered = unpack_4bit_fused(packed, scale, x.shape)
    assert recovered.shape == x.shape
    assert (recovered - x).abs().max().item() <= 2 * scale.item() + 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_4bit_fused_kernel_matches_fallback():
    """On GPU the Triton kernel and CPU fallback produce the same packed bytes."""
    x = torch.randn(2, 8, 64, device="cuda")
    packed_cuda, scale_cuda = pack_4bit_fused(x)
    packed_cpu, scale_cpu = pack_4bit_fused(x.cpu())
    assert scale_cuda.allclose(scale_cpu.cuda())
    assert torch.equal(packed_cuda, packed_cpu.cuda())
