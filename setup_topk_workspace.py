# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Build the optional TopK workspace extension against an existing LightOp."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent

setup(
    name="lightop-topk-workspace",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            "lightop._dsv4_topk_workspace",
            sources=[
                str(ROOT / "csrc/dsv4_topk_workspace.cu"),
                str(ROOT / "csrc/dequantize_k_cache_paged.cu"),
                str(ROOT / "csrc/export.cpp"),
            ],
            define_macros=[("LIGHTOP_TOPK_WORKSPACE_ONLY", "1")],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    options={"build_ext": {"build_temp": str(ROOT / "build/temp_topk_workspace")}},
    packages=[],
)
