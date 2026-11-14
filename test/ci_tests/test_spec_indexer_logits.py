"""Page8 indexer oracle; no model, vLLM or external attention dependency.

Run pytest normally for correctness. Performance against the existing serving
kernel is measured separately with frozen real paged inputs and CUDA Graphs.
"""

import pytest
import torch
from lightop.attention import spec_indexer_logits

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx938"),
    reason="FP8 native indexer requires gfx938",
)


@pytest.mark.parametrize("width", [0, 1, 7, 8, 63, 64, 65, 127, 128, 129, 257])
@pytest.mark.parametrize("lengths", [[4, 4], [0, 1, 2, 3, 4]])
def test_spec_indexer_oracle(width, lengths):
    torch.manual_seed(90273)
    t = sum(lengths)
    r = len(lengths)
    pages = (width + 7) // 8
    pool = (torch.randn((r * pages * 8, 1, 256), device="cuda") * 0.25).to(
        torch.float8_e4m3fn
    )
    table = (
        torch.randperm(r * pages, device="cuda", dtype=torch.int64)
        .to(torch.int32)
        .view(r, pages)
    )
    q = (
        (torch.randn((t, 1, 4, 256), device="cuda") * 0.25)
        .to(torch.float8_e4m3fn)
        .to(torch.bfloat16)
    )
    w = torch.randn((t, 5024), device="cuda", dtype=torch.bfloat16)[:, :4].view(t, 1, 4)
    ns = torch.randint(0, width + 1, (t,), device="cuda", dtype=torch.int32)
    starts = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    output = torch.full((t, width), -12345.25, device="cuda")
    expected = output.clone()
    for req, length in enumerate(lengths):
        keys = torch.arange(width, device="cuda")
        rows = table[req, keys // 8].long() * 8 + keys % 8
        k = pool[rows, 0].float()
        for token in range(sum(lengths[:req]), sum(lengths[:req]) + length):
            dot = q[token, 0].double() @ k.double().T
            heads = dot.clamp_min(0).float() * (w[token, 0].float() * 0.5)[:, None]
            value = ((heads[0] + heads[1]) + heads[2]) + heads[3]
            n = int(ns[token])
            expected[token, :n] = value[:n]
    spec_indexer_logits(q, pool, table, w, ns, starts, output, 0.5)
    torch.cuda.synchronize()
    mask = torch.arange(width, device="cuda")[None, :] < ns[:, None]
    torch.testing.assert_close(output[mask], expected[mask], atol=2.0e-4, rtol=2.0e-4)
    assert torch.equal(output[~mask], expected[~mask])
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        spec_indexer_logits(q, pool, table, w, ns, starts, output, 0.5)
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output[mask], expected[mask], atol=2.0e-4, rtol=2.0e-4)


def test_spec_indexer_rejects_unrounded_dtype():
    q = torch.zeros((1, 1, 4, 256), device="cuda", dtype=torch.float32)
    pool = torch.zeros((8, 1, 256), device="cuda", dtype=torch.float8_e4m3fn)
    with pytest.raises(RuntimeError, match="BF16 Q"):
        spec_indexer_logits(
            q,
            pool,
            torch.zeros((1, 1), device="cuda", dtype=torch.int32),
            torch.zeros((1, 1, 4), device="cuda", dtype=torch.bfloat16),
            torch.ones(1, device="cuda", dtype=torch.int32),
            torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            torch.empty((1, 8), device="cuda"),
        )


@pytest.mark.parametrize("value", [0.0, 2.0**-9, 448.0])
def test_exact_e4m3_special_values(value):
    q = torch.full((4, 1, 4, 256), value, device="cuda", dtype=torch.bfloat16)
    q[:, :, 1::2].neg_()
    pool = torch.full((8, 1, 256), value, device="cuda").to(torch.float8_e4m3fn)
    table = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
    w = (
        torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=torch.bfloat16)
        .view(1, 1, 4)
        .expand(4, 1, 4)
        .contiguous()
    )
    ns = torch.full((4,), 8, device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    out = torch.empty((4, 8), device="cuda")
    dots = q[:, 0].double() @ pool[:, 0].double().T
    heads = dots.clamp_min(0).float() * (w[:, 0, :, None].float() * 0.5)
    expected = (((0.0 + heads[:, 0]) + heads[:, 1]) + heads[:, 2]) + heads[:, 3]
    spec_indexer_logits(q, pool, table, w, ns, starts, out, 0.5)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "bad", ["weight_stride", "unaligned_summary", "output_alias", "output_overlap"]
)
def test_reject_unsafe_indexer_views(bad):
    q = torch.zeros((4, 1, 4, 256), device="cuda", dtype=torch.bfloat16)
    pool = torch.zeros((8, 1, 256), device="cuda", dtype=torch.float8_e4m3fn)
    w = torch.zeros((4, 1, 4), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((4, 8), device="cuda")
    if bad == "weight_stride":
        w = torch.empty((4, 5), device="cuda", dtype=torch.bfloat16)[:, :4].view(
            4, 1, 4
        )
    elif bad == "unaligned_summary":
        pool = torch.empty(8 * 256 + 1, device="cuda", dtype=torch.float8_e4m3fn)[
            1:
        ].view(8, 1, 256)
    elif bad == "output_alias":
        out = pool.view(torch.float32).view(-1)[:32].view(4, 8)
    else:
        out = torch.empty((1, 8), device="cuda").expand(4, 8)
    with pytest.raises(RuntimeError, match="aligned|overlap|single memory"):
        spec_indexer_logits(
            q,
            pool,
            torch.zeros((1, 1), device="cuda", dtype=torch.int32),
            w,
            torch.full((4,), 8, device="cuda", dtype=torch.int32),
            torch.tensor([0, 4], device="cuda", dtype=torch.int32),
            out,
            0.5,
        )
