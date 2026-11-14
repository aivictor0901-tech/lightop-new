"""Independent full split-K/rounding/activation/quantization oracle.

The producer's four exact INT32 partial sums are used without executing or
depending on a GEMM implementation. End-to-end producer timing is separate.
"""

import pytest
import torch
from lightop.moe import shared_gateup_int8_epilogue

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize(
    "m", [0, 1, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 129]
)
def test_shared_epilogue(m):
    torch.manual_seed(98213)
    n_m = (m + 31) // 32
    tiles = 12 * n_m
    raw = torch.randint(
        -10000, 10001, (4, tiles, 2, 32, 64), device="cuda", dtype=torch.int32
    )
    xs = torch.rand((m, 1), device="cuda") * 0.02 + 0.001
    ws = torch.rand((768, 1), device="cuda") * 0.004 + 0.0001
    summed = torch.empty((m, 768), device="cuda", dtype=torch.int32)
    for tile in range(6):
        for block in range(n_m):
            lo = block * 32
            end = min(lo + 32, m)
            p = (6 + tile) * n_m + block
            summed[lo:end, tile * 64 : (tile + 1) * 64] = (
                raw[:, p, 0, : end - lo].sum(0).to(torch.int32)
            )
            summed[lo:end, 384 + tile * 64 : 384 + (tile + 1) * 64] = (
                raw[:, p, 1, : end - lo].sum(0).to(torch.int32)
            )
    bf = (summed.float() * (xs * ws.view(1, 768))).to(torch.bfloat16).float()
    gate, up = bf[:, :384], bf[:, 384:]
    act = (
        ((gate * torch.sigmoid(gate)).clamp_max(7) * up.clamp(-7, 7))
        .to(torch.bfloat16)
        .float()
    )
    scale = act.abs().amax(1, keepdim=True).clamp_min(1.0e-10) / 127
    expected = torch.round(act / scale).clamp(-127, 127).to(torch.int8)
    q, s = shared_gateup_int8_epilogue(raw, xs, ws)
    torch.cuda.synchronize()
    assert torch.equal(s, scale)
    assert torch.equal(q, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, s = shared_gateup_int8_epilogue(raw, xs, ws)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(q, expected)
    assert torch.equal(s, scale)


@pytest.mark.parametrize(
    "m", [0, 1, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 129]
)
def test_bf16_gateup_epilogue(m):
    torch.manual_seed(98217)
    gateup = (torch.randn((m, 768), device="cuda") * 9).to(torch.bfloat16)
    if m:
        gateup[0].zero_()
        if m > 1:
            gateup[1, :384] = 8
            gateup[1, 384:] = (
                torch.arange(384, device="cuda").remainder(4).to(torch.bfloat16) * 0.5
            )
    gate, up = gateup.float().chunk(2, dim=1)
    act = (
        ((gate * torch.sigmoid(gate)).clamp_max(7) * up.clamp(-7, 7))
        .to(torch.bfloat16)
        .float()
    )
    expected_scale = act.abs().amax(1, keepdim=True).clamp_min(1.0e-10) / 127
    expected = torch.round(act / expected_scale).clamp(-127, 127).to(torch.int8)
    q, s = shared_gateup_int8_epilogue(gateup)
    torch.cuda.synchronize()
    assert torch.equal(q, expected)
    assert torch.equal(s, expected_scale)
    oq = torch.empty_like(q)
    os = torch.empty_like(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, s = shared_gateup_int8_epilogue(gateup, output=oq, output_scales=os)
    graph.replay()
    torch.cuda.synchronize()
    assert q.data_ptr() == oq.data_ptr() and s.data_ptr() == os.data_ptr()
    assert torch.equal(q, expected) and torch.equal(s, expected_scale)


@pytest.mark.parametrize(
    "bad", ["activation_alias", "scale_alias", "unaligned_input", "unaligned_output"]
)
def test_reject_unsafe_epilogue_buffers(bad):
    m = 4
    x = torch.ones((m, 768), device="cuda", dtype=torch.bfloat16)
    q = torch.empty((m, 384), device="cuda", dtype=torch.int8)
    s = torch.empty((m, 1), device="cuda")
    if bad == "activation_alias":
        q = x.view(torch.int8).view(-1)[: m * 384].view(m, 384)
    elif bad == "scale_alias":
        s = x.view(torch.float32).view(-1)[:m].view(m, 1)
    elif bad == "unaligned_input":
        x = torch.empty(m * 768 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(
            m, 768
        )
    else:
        q = torch.empty(m * 384 + 1, device="cuda", dtype=torch.int8)[1:].view(m, 384)
    with pytest.raises(RuntimeError, match="aligned|overlap|single memory"):
        shared_gateup_int8_epilogue(x, output=q, output_scales=s)
