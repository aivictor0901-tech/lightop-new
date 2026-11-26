# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Exact-shape Smooth GEMM dispatch, graph replay, and fallback coverage."""

import pytest
import torch

from lightop import gemm_ops, gemmopt, op
from lightop.config_smooth_gemm import _W8A8_SMOOTH_GFX936_TUNED


@pytest.fixture
def tuned(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("An HCU device is required")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if props.gcnArchName.split(":")[0] != "gfx936" or props.multi_processor_count != 80:
        pytest.skip("Measured gfx936/80CU configurations")
    monkeypatch.setenv("LIGHTOP_GEMM_W8A8_SMOOTH", "1")


def operands(m, k, n):
    a = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8).T
    sa = torch.rand(m, 1, device="cuda", dtype=torch.float32) * 0.02
    sb = torch.rand(n, 1, device="cuda", dtype=torch.float32) * 0.02
    sa[0] = 0
    return [a, b, sa, sb, None, torch.bfloat16]


def reference(shape, args):
    """The previously measured explicit call, without framework dispatch."""
    m, k, n = shape
    kind, setting = _W8A8_SMOOTH_GFX936_TUNED[shape]
    a, b, sa, sb, bias, dtype = args
    if kind == "blaslt":
        return op._lmslimquant_blaslt_w8a8_bf16_gemm_cached(a, b, sa, sb, setting)
    if kind == "asm":
        return op.gemm_w8a8_smooth_asm(
            a, b, sa, sb, bias, dtype, m, n, k, True, setting
        )
    return op.gemm_w8_a8_smooth_colmajor(a, b, sa, sb, bias, dtype, *setting)


def check_integer_reference(actual, args):
    a, b, sa, sb, _, _ = args
    m, n = actual.shape
    for row, col in ((0, 0), (m - 1, n - 1), (m // 2, n // 2)):
        integer = (a[row].cpu().long() * b[:, col].cpu().long()).sum()
        scale = sa[row, 0].cpu() * sb[col, 0].cpu()
        expected = (scale * integer.float()).bfloat16()
        assert actual[row, col].cpu() == expected


@pytest.mark.parametrize("shape", sorted(_W8A8_SMOOTH_GFX936_TUNED))
def test_tuned_smooth_eager_and_graph(tuned, shape, monkeypatch):
    torch.manual_seed(897)
    args = operands(*shape)
    assert gemm_ops.gemm_w8a8_smooth is gemmopt.gemm_w8a8_smooth
    expected = reference(shape, args)
    assert expected is not None
    selected = []
    original_dispatch = gemmopt._try_tuned_w8a8_smooth

    def check_dispatch(*operands):
        output = original_dispatch(*operands)
        selected.append(output is not None)
        return output

    monkeypatch.setattr(gemmopt, "_try_tuned_w8a8_smooth", check_dispatch)
    supported, actual = gemm_ops.gemm_w8a8_smooth(*args)
    assert supported
    assert selected == [True]
    assert torch.equal(actual, expected)
    check_integer_reference(actual, args)

    # Warm plans on a side stream, then replay with two different sets of inputs.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            gemm_ops.gemm_w8a8_smooth(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        supported, graph_out = gemm_ops.gemm_w8a8_smooth(*args)
    assert supported
    for _ in range(2):
        args[0].random_(-128, 128)
        args[1].random_(-128, 128)
        args[2].uniform_(0.001, 0.02)
        args[3].uniform_(0.001, 0.02)
        graph.replay()
        assert torch.equal(graph_out, reference(shape, args))
        check_integer_reference(graph_out, args)


@pytest.mark.parametrize(
    "change", ["bias", "dtype", "scale", "stride", "offset", "shape"]
)
def test_tuned_smooth_rejects_incompatible_descriptor(tuned, change):
    args = operands(48, 5120, 512)
    if change == "bias":
        args[4] = torch.zeros(512, device="cuda", dtype=torch.bfloat16)
    elif change == "dtype":
        args[5] = torch.float16
    elif change == "scale":
        args[2] = args[2].flatten()
    elif change == "stride":
        args[1] = args[1].contiguous()
    elif change == "offset":
        storage = torch.empty(48 * 5120 + 16, device="cuda", dtype=torch.int8)
        args[0] = storage[16:].view(48, 5120)
    else:
        args = operands(7, 5120, 512)
    assert gemmopt._try_tuned_w8a8_smooth(*args) is None


def test_tuned_smooth_preserves_global_switch_and_scale_order(tuned, monkeypatch):
    def forbidden(*args):
        raise AssertionError("Measured dispatch must not run in this mode")

    monkeypatch.setattr(gemmopt, "_try_tuned_w8a8_smooth", forbidden)
    args = operands(48, 5120, 512)
    gemmopt.gemm_w8a8_smooth(*args, input_scale_first=True)
    monkeypatch.setenv("LIGHTOP_GEMM_W8A8_SMOOTH", "0")
    assert gemmopt.gemm_w8a8_smooth(*args) == (False, None)


def test_tuned_smooth_rejects_other_hardware(tuned, monkeypatch):
    monkeypatch.setattr(gemmopt, "_is_smooth_gfx936_80cu", lambda device: False)
    assert gemmopt._try_tuned_w8a8_smooth(*operands(48, 5120, 512)) is None
