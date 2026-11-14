"""Stable expert-major routing oracle, including duplicate and invalid IDs."""

import pytest
import torch
from lightop.moe import ep_permute_int8_small

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize(
    "m", [0, 1, 7, 8, 16, 24, 32, 40, 48, 56, 64, 96, 128, 160, 192, 224, 256]
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("alignment", [64, 128])
def test_stable_int8_routes(m, dtype, alignment, e=44):
    torch.manual_seed(49381)
    ge, k, topk = 352, 4096, 8
    x = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8)
    scales = torch.rand((m, 1), device="cuda") * 0.03
    ids = torch.randint(-3, ge + 3, (m, topk), device="cuda", dtype=dtype)
    mapping = torch.full((ge,), -1, device="cuda", dtype=torch.int32)
    mapping[:e] = torch.randperm(e, device="cuda", dtype=torch.int32)
    payload, ss, expert_ids, inverse = ep_permute_int8_small(
        x, scales, ids, mapping, e, alignment
    )
    expected_rows = (
        (m * topk + min(m * topk, e) * (alignment - 1) + alignment - 1) // alignment
    ) * alignment
    assert payload.shape == (expected_rows, k)
    cpu_ids = ids.cpu().view(-1)
    mp = mapping.cpu()
    expected = torch.full_like(cpu_ids, -1)
    expected_e = torch.full_like(expert_ids.cpu(), -1)
    dest = []
    src = []
    offset = 0
    for local in range(e):
        hits = [
            r
            for r, g in enumerate(cpu_ids.tolist())
            if 0 <= g < ge and int(mp[g]) == local
        ]
        for j, r in enumerate(hits):
            p = offset + j
            expected[r] = p
            expected_e[p] = local
            dest.append(p)
            src.append(r // topk)
        offset += (len(hits) + alignment - 1) // alignment * alignment
    assert torch.equal(inverse.cpu().view(-1), expected)
    assert torch.equal(expert_ids.cpu(), expected_e)
    di = torch.tensor(dest, device="cuda", dtype=torch.long)
    si = torch.tensor(src, device="cuda", dtype=torch.long)
    assert torch.equal(payload[di], x[si])
    assert torch.equal(ss[di], scales[si])
    output = torch.empty_like(payload)
    result = ep_permute_int8_small(x, scales, ids, mapping, e, alignment, output)
    assert result[0].data_ptr() == output.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = ep_permute_int8_small(x, scales, ids, mapping, e, alignment, output)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(result[3].cpu().view(-1), expected)
    assert torch.equal(result[0][di], x[si])


@pytest.mark.parametrize("e,m", [(1, 129), (64, 256)])
@pytest.mark.parametrize("alignment", [64, 128])
def test_expert_count_boundaries(e, m, alignment):
    test_stable_int8_routes(m, torch.int64, alignment, e=e)


@pytest.mark.parametrize("bad", ["unaligned_input", "unaligned_output", "overlap"])
def test_reject_unsafe_vector_buffers(bad):
    m, k, e = 4, 32, 1
    rows = ((m + min(m, e) * 127 + 127) // 128) * 128
    x = torch.zeros((m, k), device="cuda", dtype=torch.int8)
    output = torch.empty((rows, k), device="cuda", dtype=torch.int8)
    if bad == "unaligned_input":
        x = torch.empty(m * k + 1, device="cuda", dtype=torch.int8)[1:].view(m, k)
    elif bad == "unaligned_output":
        output = torch.empty(rows * k + 1, device="cuda", dtype=torch.int8)[1:].view(
            rows, k
        )
    else:
        x = output[:m]
    s = torch.ones((m, 1), device="cuda")
    ids = torch.zeros((m, 1), device="cuda", dtype=torch.int32)
    mapping = torch.zeros((1,), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="aligned|overlap|single memory"):
        ep_permute_int8_small(x, s, ids, mapping, e, 128, output)
