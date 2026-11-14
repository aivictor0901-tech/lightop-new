# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Native expert-parallel data movement APIs."""

from .. import op as lightop_native

ep_build_m_indices = lightop_native.ep_build_m_indices
ep_gather = lightop_native.ep_gather
ep_scatter = lightop_native.ep_scatter

__all__ = [
    "ep_permute_int8_small",
    "ep_build_m_indices",
    "ep_gather",
    "ep_scatter",
]


def ep_permute_int8_small(
    input, scales, topk_ids, expert_map, local_experts, alignment=128, output=None
):
    """Stable expert-major INT8 payload/scale gather for at most2048 routes.

    Return (payload, scales, expert_ids, inverse). Negative or unmapped routes
    have inverse=-1; padding expert_ids=-1. Padding payload/scales are unused.
    All sizes use the same metadata and gather kernels. Caller-provided output
    must own the exact conservative padded shape used by the W4 workspace.
    Input/output addresses must be16-byte aligned and must not overlap inputs.
    The native implementation requires a wave64 device.
    """
    if not hasattr(lightop_native, "ep_permute_int8_small"):
        raise RuntimeError("ep_permute_int8_small requires a rebuilt LightOp extension")
    return tuple(
        lightop_native.ep_permute_int8_small(
            input, scales, topk_ids, expert_map, local_experts, alignment, output
        )
    )
