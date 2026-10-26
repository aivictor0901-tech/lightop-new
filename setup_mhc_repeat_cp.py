# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Build only the optional mHC extension against an existing LightOp install."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent

setup(
    name="lightop-mhc-repeat-cp",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            "lightop._mhc_repeat_cp",
            sources=[
                str(ROOT / "csrc/mhc_repeat_cp_sglang.cu"),
                str(ROOT / "csrc/export.cpp"),
            ],
            define_macros=[("LIGHTOP_MHC_REPEAT_CP_ONLY", "1")],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    options={"build_ext": {"build_temp": str(ROOT / "build/temp_mhc_repeat_cp")}},
    packages=[],
)
