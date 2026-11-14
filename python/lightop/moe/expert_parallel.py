# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Native expert-parallel data movement APIs."""

from .. import op as lightop_native

ep_build_m_indices = lightop_native.ep_build_m_indices
ep_gather = lightop_native.ep_gather
ep_scatter = lightop_native.ep_scatter

__all__ = [
    "shared_gateup_int8_epilogue",
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


def shared_gateup_int8_epilogue(
    partials,
    xscale=None,
    wscale=None,
    router_columns=352,
    limit=7.0,
    output=None,
    output_scales=None,
):
    """Reduce INT32 split-K shared gate/up directly to INT8 and row scales.

    BF16 input[M,768] consumes no scales. INT32 input requires xscale/wscale.
    Both input formats use the same two-wave kernel for every row count.
    Workspace uses four splits, BM32/BN64 tiles and six shared N tiles (I384).
    Router tiles precede shared tiles in each split. Preserve W8A8 scale order,
    BF16 gate/up rounding, clamp(SiLU(g),max=limit)*clamp(up,-limit,limit),
    BF16 activation rounding and full384-wide dynamic symmetric quantization.
    Caller-provided outputs must not overlap the inputs or each other. Input
    and channel-scale addresses require16-byte alignment, INT8 output4 bytes.
    The native implementation requires a wave64 device.
    """
    if not hasattr(lightop_native, "shared_gateup_int8_epilogue"):
        raise RuntimeError(
            "shared_gateup_int8_epilogue requires a rebuilt LightOp extension"
        )
    return tuple(
        lightop_native.shared_gateup_int8_epilogue(
            partials, xscale, wscale, router_columns, limit, output, output_scales
        )
    )
