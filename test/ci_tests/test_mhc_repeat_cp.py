# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Keep arbitrary mHC/CP shapes covered by the normal ROCm CI job."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm"
)


@pytest.mark.parametrize(
    "rows,hidden,hc,cp,rank",
    [(13, 7, 3, 5, 4), (3, 17, 5, 8, 7), (137, 513, 7, 1, 0)],
)
def test_arbitrary_cp_row_ownership(rows, hidden, hc, cp, rank):
    from lightop.attention import mhc

    source = torch.randn((rows, hidden), device="cuda", dtype=torch.bfloat16)
    expected = source[rank::cp].unsqueeze(1).repeat(1, hc, 1)
    assert mhc.supports_mhc_repeat_cp_sglang(source, hc, cp, rank)
    actual = mhc.mhc_repeat_cp_sglang(source, hc, cp, rank)
    assert actual.shape == expected.shape
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
