# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise, boundary, ownership and stream checks for mHC repeat+CP."""

import pytest
import torch


@pytest.fixture(scope="module")
def api():
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("Requires ROCm/gfx936")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx936":
        pytest.skip("Requires gfx936")
    from lightop.attention import mhc

    return mhc


def assert_same_bits(actual, expected):
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("hidden", [4096, 5120])
@pytest.mark.parametrize("rows", [0, 8, 120, 128, 136, 32760, 32768])
@pytest.mark.parametrize("rank", [0, 3, 7])
def test_slice_repeat_bitwise(api, hidden, rows, rank):
    source = torch.randn((rows, hidden), device="cuda", dtype=torch.bfloat16)
    count = min(source.numel(), 65536)
    source.view(torch.int16).flatten()[:count] = torch.arange(
        -32768, -32768 + count, device="cuda", dtype=torch.int32
    ).to(torch.int16)
    before = source.clone()
    expected = source[rank::8].unsqueeze(1).repeat(1, 4, 1)
    assert api.supports_mhc_repeat_cp_sglang(source, 4, 8, rank)
    actual = api.mhc_repeat_cp_sglang(source, 4, 8, rank)
    assert actual.shape == expected.shape and actual.is_contiguous()
    assert_same_bits(actual, expected)
    assert_same_bits(source, before)
    out = torch.empty_like(expected)
    returned = api.mhc_repeat_cp_sglang_out(source, 4, 8, rank, out)
    assert returned.data_ptr() == out.data_ptr()
    assert_same_bits(out, expected)


def test_metadata_rejections(api):
    source = torch.empty((128, 5120), device="cuda", dtype=torch.bfloat16)
    cases = [
        (source[0], 4, 8, 0),
        (source.detach().requires_grad_(True), 4, 8, 0),
        (source, 0, 8, 0),
        (source, 4, 0, 0),
        (source, 4, 8, -1),
        (source, 4, 8, 8),
        (torch.empty((128, 5120), dtype=torch.bfloat16), 4, 8, 0),
    ]
    for args in cases:
        assert api.supports_mhc_repeat_cp_sglang(*args) is False
    with pytest.raises(RuntimeError, match="inference-only"):
        api.mhc_repeat_cp_sglang(source.detach().requires_grad_(True), 4, 8, 0)


def test_output_alias_and_metadata_rejected(api):
    source = torch.empty((128, 5120), device="cuda", dtype=torch.bfloat16)
    alias = source.flatten()[: 16 * 4 * 5120].view(16, 4, 5120)
    with pytest.raises(RuntimeError, match="single memory location|overlap"):
        api.mhc_repeat_cp_sglang_out(source, 4, 8, 0, alias)
    with pytest.raises(RuntimeError, match="output metadata"):
        api.mhc_repeat_cp_sglang_out(source, 4, 8, 0, source)


def test_current_stream_and_guard(api):
    source = torch.empty((128, 5120), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        source.fill_(3.5)
        output = api.mhc_repeat_cp_sglang(source, 4, 8, 7)
        output.add_(1)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(output, torch.full_like(output, 4.5))


@pytest.mark.parametrize(
    "shape,hc,cp,rank",
    [
        ((13, 7), 3, 5, 0),
        ((13, 7), 3, 5, 4),
        ((1, 1), 1, 1, 0),
        ((3, 17), 5, 8, 7),
        ((0, 11), 2, 3, 1),
        ((7, 0), 3, 2, 1),
        ((32777, 23), 2, 3, 2),
        ((137, 513), 7, 1, 0),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.uint8,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int64,
        torch.complex128,
    ],
)
def test_general_shapes_dtypes_and_ragged_rows(api, shape, hc, cp, rank, dtype):
    source = torch.arange(shape[0] * shape[1], device="cuda", dtype=torch.int64)
    source = source.to(dtype).reshape(shape)
    expected = source[rank::cp].unsqueeze(1).repeat(1, hc, 1)
    assert api.supports_mhc_repeat_cp_sglang(source, hc, cp, rank)
    actual = api.mhc_repeat_cp_sglang(source, hc, cp, rank)
    assert actual.shape == expected.shape and actual.dtype == dtype
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("layout", ["transpose", "column_stride", "offset", "expand"])
def test_strides_and_unaligned_storage(api, layout):
    base = torch.randn((19, 34), device="cuda", dtype=torch.bfloat16)
    source = {
        "transpose": base.T,
        "column_stride": base[:, ::2],
        "offset": base[:, 1:18],
        "expand": base[:1, :17].expand(19, -1),
    }[layout]
    expected = source[2::5].unsqueeze(1).repeat(1, 3, 1)
    assert api.supports_mhc_repeat_cp_sglang(source, 3, 5, 2)
    actual = api.mhc_repeat_cp_sglang(source, 3, 5, 2)
    assert_same_bits(actual, expected)


def test_input_lifetime_on_another_stream(api):
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        source = torch.full((4096, 5120), 3.5, device="cuda", dtype=torch.bfloat16)
    consumer.wait_stream(producer)
    with torch.cuda.stream(consumer):
        result = api.mhc_repeat_cp_sglang(source, 4, 8, 0)
    del source
    with torch.cuda.stream(producer):
        replacement = torch.full(
            (4096, 5120), -7.0, device="cuda", dtype=torch.bfloat16
        )
    torch.cuda.current_stream().wait_stream(consumer)
    assert torch.equal(result, torch.full_like(result, 3.5))
    del replacement
