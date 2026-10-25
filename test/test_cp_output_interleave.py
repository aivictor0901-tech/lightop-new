# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the inference-only in-place CP output permutation."""

import pytest
import torch

from lightop.attention.cp_output import interleave_cp_output_

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize("cp,local", [(1, 0), (1, 7), (8, 1), (2, 31), (3, 137),
                                    (5, 11), (8, 4096), (16, 8)])
@pytest.mark.parametrize("shape", [(0,), (1,), (17,), (3, 513)])
def test_matches_out_of_place_reference(cp, local, shape):
    x = torch.randint(0, 256, (cp * local, *shape), dtype=torch.uint8, device="cuda")
    expected = x.reshape(cp, local, *shape).transpose(0, 1).reshape(x.shape).clone()
    pointer = x.data_ptr()
    result = interleave_cp_output_(x, cp)
    assert result.data_ptr() == pointer
    assert torch.equal(result, expected)


def test_byte_preservation_and_storage_offset():
    raw = torch.randint(0, 256, (24 * 19 * 8 + 8,), dtype=torch.uint8, device="cuda")
    x = raw[8:].view(torch.float64).reshape(24, 19)
    expected = x.reshape(3, 8, 19).transpose(0, 1).reshape_as(x).clone()
    interleave_cp_output_(x, 3)
    assert torch.equal(x.view(torch.uint8), expected.view(torch.uint8))


def test_unaligned_byte_storage():
    x = torch.arange(24 * 32 + 1, dtype=torch.int32, device="cuda").to(torch.uint8)[1:].reshape(24, 32)
    expected = x.reshape(3, 8, 32).transpose(0, 1).reshape_as(x).clone()
    interleave_cp_output_(x, 3)
    assert torch.equal(x, expected)


def test_stream_dependency_and_graph_replay():
    source = torch.arange(24 * 512, device="cuda").reshape(24, 512)
    expected = source.reshape(3, 8, 512).transpose(0, 1).reshape_as(source)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = source.clone()
        interleave_cp_output_(x, 3)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(x, expected)
    x.copy_(source)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        interleave_cp_output_(x, 3)
    graph.replay()
    assert torch.equal(x, expected)


@pytest.mark.parametrize("cp", [0, -1, 7])
def test_rejects_invalid_cp(cp):
    with pytest.raises(ValueError):
        interleave_cp_output_(torch.empty((24, 19), device="cuda"), cp)


def test_rejects_noncontiguous_and_training_inputs():
    with pytest.raises(ValueError):
        interleave_cp_output_(torch.empty((24, 19), device="cuda").t(), 1)
    with pytest.raises(ValueError):
        interleave_cp_output_(torch.empty((24, 19), device="cuda", requires_grad=True), 3)
