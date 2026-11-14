# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Gemma FP32 residual -> BF16 norm -> row INT8, without framework dependencies.

The fixed RMSNorm tolerance is one BF16 ULP because FP32 reductions may use
different trees. INT8 bytes and FP32 scales must exactly match the existing
LightOp provider observing the returned BF16 norm. Inputs and old residuals
must be preserved. Performance is exercised explicitly by the external paired
benchmark, never by test import or default pytest collection.
"""

import math
import pytest
import torch


@pytest.fixture
def operation():
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    from lightop.norm import gemma_rms_norm_per_token_int8_quant

    return gemma_rms_norm_per_token_int8_quant


def assert_contract(actual, x, weight, residual, epsilon):
    from lightop.quant import per_token_quant_int8

    norm, quant, scales, summed = actual
    expected_sum = x.float() if residual is None else x.float() + residual
    expected = expected_sum * torch.rsqrt(
        expected_sum.square().mean(-1, keepdim=True) + epsilon
    )
    expected = (expected * (weight + 1.0)).to(torch.bfloat16)
    if norm.numel():
        ulp = (norm.view(torch.int16).int() - expected.view(torch.int16).int()).abs()
        ulp = torch.where(norm == expected, 0, ulp)
        assert int(ulp.max().item()) <= 1
    expected_quant, expected_scales = per_token_quant_int8(norm)
    assert norm.dtype == torch.bfloat16 and norm.shape == x.shape
    assert quant.dtype == torch.int8 and quant.shape == x.shape
    assert scales.dtype == torch.float32 and scales.shape == (x.shape[0], 1)
    assert summed.dtype == torch.float32
    assert torch.equal(summed, expected_sum)
    assert torch.equal(quant, expected_quant)
    assert torch.equal(scales, expected_scales)


@pytest.mark.parametrize("m", [0, 1, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 96, 128])
@pytest.mark.parametrize("kind", ["fp32", "bf16", "bf16_residual"])
def test_decode_shapes(operation, m, kind):
    torch.manual_seed(31 + m)
    x = torch.randn(
        (m, 4096),
        device="cuda",
        dtype=torch.float32 if kind == "fp32" else torch.bfloat16,
    )
    weight = torch.randn((4096,), device="cuda", dtype=torch.float32) * 0.1
    residual = (
        torch.randn_like(x, dtype=torch.float32) if kind.endswith("residual") else None
    )
    x_original = x.clone()
    residual_original = residual.clone() if residual is not None else None
    outputs = operation(x, weight, 1e-5, residual)
    if m:
        assert_contract(outputs, x, weight, residual, 1e-5)
    else:
        assert all(t.numel() == 0 for t in outputs)
    assert torch.equal(x, x_original)
    if residual is not None:
        assert torch.equal(residual, residual_original)
        assert outputs[3] is not residual
        if m:
            assert outputs[3].data_ptr() != residual.data_ptr()
    elif x.dtype == torch.float32:
        assert outputs[3].data_ptr() == x.data_ptr()


@pytest.mark.parametrize("k", [128, 512, 2048, 4088, 4104, 8192, 16384])
def test_supported_widths(operation, k):
    torch.manual_seed(k)
    x = torch.randn((7, k), device="cuda", dtype=torch.float32)
    weight = torch.randn((k,), device="cuda", dtype=torch.float32) * 0.1
    assert_contract(operation(x, weight), x, weight, None, 1e-5)


def test_zero_tiny_and_halfway(operation):
    # Norm values remain zero/tiny in these rows. Quantization must preserve
    # the provider's exact 1e-10 floor and nearest-even conversion.
    x = torch.zeros((7, 4096), device="cuda", dtype=torch.float32)
    x[1].fill_(1e-15)
    x[2].fill_(-1e-15)
    x[3].fill_(1e-12)
    x[4].fill_(-1e-12)
    x[5, :4] = torch.tensor([127.0, 0.5, -0.5, 0.0], device="cuda")
    x[6, :4] = torch.tensor([1.984375, 0.0078125, -0.0078125, 0.0], device="cuda")
    weight = torch.zeros((4096,), device="cuda", dtype=torch.float32)
    assert_contract(operation(x, weight), x, weight, None, 1e-5)
    # Choose a Gemma weight that cancels RMS scaling, so quantization receives
    # literal BF16 halfway values and nearest-even has an independent oracle.
    half = torch.zeros((1, 4096), device="cuda", dtype=torch.float32)
    half[0, :7] = torch.tensor([127.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5], device="cuda")
    gamma_minus_one = math.sqrt(16146.5 / 4096 + 1e-5) - 1.0
    half_weight = torch.full((4096,), gamma_minus_one, device="cuda")
    outputs = operation(half, half_weight)
    assert_contract(outputs, half, half_weight, None, 1e-5)
    assert torch.equal(outputs[0], half.bfloat16())
    assert outputs[1][0, :7].tolist() == [127, 0, 0, 2, -2, 2, -2]
    assert outputs[2].item() == 1.0


def test_changed_graph_inputs_and_nondefault_stream(operation):
    torch.manual_seed(41)
    x = torch.randn((64, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((4096,), device="cuda", dtype=torch.float32) * 0.1
    residual = torch.randn_like(x, dtype=torch.float32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            operation(x, weight, 1e-5, residual)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = operation(x, weight, 1e-5, residual)
    stream.synchronize()
    for seed in range(3):
        torch.manual_seed(seed)
        x.copy_(torch.randn_like(x))
        residual.copy_(torch.randn_like(residual))
        x_before, residual_before = x.clone(), residual.clone()
        graph.replay()
        torch.cuda.synchronize()
        assert_contract(outputs, x, weight, residual, 1e-5)
        assert torch.equal(x, x_before)
        assert torch.equal(residual, residual_before)


def test_rejects_unsupported_metadata(operation):
    x = torch.randn((4, 4096), device="cuda", dtype=torch.float32)
    weight = torch.zeros((4096,), device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="epsilon"):
        operation(x, weight, -1.0)
    with pytest.raises(RuntimeError, match="contiguous"):
        operation(torch.empty((4, 8192), device="cuda")[:, ::2], weight)
    with pytest.raises(RuntimeError, match="FP32 vector"):
        operation(x, weight.bfloat16())
    with pytest.raises(RuntimeError, match="FP32 and match"):
        operation(x, weight, residual=torch.zeros_like(x, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="alignment"):
        operation(torch.empty((4 * 4096 + 1,), device="cuda")[1:].view(4, 4096), weight)
    with pytest.raises(ValueError, match="2D"):
        operation(x.flatten(), weight)
