# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Sparse workspace contracts, independent of compression ratio and KV layout."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("n,queries,topk", [(1, 1, 1), (257, 13, 7), (65537, 33, 512)])
@pytest.mark.parametrize("strided", [False, True])
def test_logical_union_and_swa_order(dtype, n, queries, topk, strided):
    from lightop.kvcache import dsv4_pack_sparse_topk_workspace

    width = topk + 11
    owner = torch.full(
        (queries, width + (3 if strided else 0)), -1, device="cuda", dtype=torch.int32
    )
    indices = owner[:, :width]
    lengths = torch.empty(queries, device="cuda", dtype=torch.int32)
    expected_rows = []
    for q in range(queries):
        chosen = [(q * 13 + j * 17) % n for j in range(topk)]
        chosen[1::3] = chosen[::3][: len(chosen[1::3])]
        row = chosen + [n + (q + j) % 11 for j in range(11)]
        length = 0 if q % 5 == 3 else len(row) - q % 3
        indices[q].copy_(torch.tensor(row, device="cuda", dtype=torch.int32))
        lengths[q] = length
        expected_rows.append(row[:length])
    # Two logical rows may intentionally refer to the same physical page row.
    full_ids = (torch.arange(n, device="cuda", dtype=dtype) * 19) % (n // 2 + 1)
    packed, remapped, count = dsv4_pack_sparse_topk_workspace(
        full_ids, indices, lengths, topk, swa_rows=11
    )
    union = sorted({v for row in expected_rows for v in row if 0 <= v < n})
    mapping = {v: i for i, v in enumerate(union)}
    capacity = min(n, queries * topk)
    assert count.item() == len(union)
    assert count.untyped_storage().nbytes() == 4
    assert torch.equal(packed[: len(union)], full_ids[union].int())
    assert bool((packed[len(union) :] == -1).all())
    for q, row in enumerate(expected_rows):
        expected = [mapping[v] if v < n else capacity + v - n for v in row]
        assert remapped[q, : len(row)].cpu().tolist() == expected
        assert bool((remapped[q, len(row) :] == -1).all())


def test_all_padding_has_no_selected_rows():
    from lightop.kvcache import dsv4_pack_sparse_topk_workspace

    ids = torch.arange(65537, dtype=torch.int32, device="cuda")
    indices = torch.full((7, 640), 2**30, dtype=torch.int32, device="cuda")
    lengths = torch.zeros(7, dtype=torch.int32, device="cuda")
    packed, remapped, count = dsv4_pack_sparse_topk_workspace(
        ids, indices, lengths, 512, swa_rows=0
    )
    assert count.item() == 0
    assert count.untyped_storage().nbytes() == 4
    assert bool((packed == -1).all()) and bool((remapped == -1).all())


def test_c4_api_keeps_the_existing_contract():
    from lightop.kvcache import (
        dsv4_pack_c4_topk_workspace,
        dsv4_pack_sparse_topk_workspace,
    )

    ids = torch.arange(1024, dtype=torch.int64, device="cuda")
    indices = torch.tensor([[5, 5, 7, 1024, 1025]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([5], dtype=torch.int32, device="cuda")
    legacy = dsv4_pack_c4_topk_workspace(ids, indices, lengths, 3)
    generic = dsv4_pack_sparse_topk_workspace(ids, indices, lengths, 3, swa_rows=2)
    assert all(torch.equal(a, b) for a, b in zip(legacy, generic))


@pytest.mark.parametrize("page", [1, 2, 7, 64, 128, 256])
@pytest.mark.parametrize("active", [0, 1, 17, 37])
@pytest.mark.parametrize("fnuz", [False, True])
def test_counted_dequant_ignores_unused_ids_and_preserves_output(page, active, fnuz):
    from lightop.kvcache import dequantize_k_cache_paged_hcu_counted

    pages = 40 // page + 2
    stride = ((page * 584 + 575) // 576) * 576
    cache = torch.zeros((pages, stride), dtype=torch.uint8, device="cuda")
    rows = torch.arange(pages * page, device="cuda").reshape(pages, page)
    data = cache[:, : page * 576].view(pages, page, 576)
    data[:, :, :448].copy_(((rows % 60) + 16).byte()[:, :, None])
    data[:, :, 448:].view(torch.bfloat16).copy_(
        (rows.float() / 32).bfloat16()[:, :, None]
    )
    cache[:, page * 576 : page * 584].fill_(127)
    ids = torch.full((37,), -(2**30), dtype=torch.int64, device="cuda")
    ids[:active] = torch.arange(active, device="cuda")
    owner = torch.full((37, 1, 520), 19.0, dtype=torch.bfloat16, device="cuda")
    out = owner[:, :, 4:516]
    count = torch.tensor([active], dtype=torch.int32, device="cuda")
    dequantize_k_cache_paged_hcu_counted(cache, ids, out, page, fnuz, count)
    encoding = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    for i in range(active):
        physical = data[i // page, i % page]
        expected = torch.cat(
            (
                physical[:448].view(encoding).float().bfloat16(),
                physical[448:].view(torch.bfloat16),
            )
        )
        assert torch.equal(out[i, 0].view(torch.uint8), expected.view(torch.uint8))
    assert bool((out[active:] == 19).all())
    assert bool((owner[:, :, :4] == 19).all()) and bool((owner[:, :, 516:] == 19).all())
