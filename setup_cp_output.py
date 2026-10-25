# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Build the optional CP output extension against an installed LightOp wheel."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent

setup(
    name="lightop-cp-output",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            "lightop._cp_output",
            sources=[str(ROOT / "csrc/cp_output/interleave.cu"),
                     str(ROOT / "csrc/export.cpp")],
            define_macros=[("LIGHTOP_CP_OUTPUT_ONLY", "1")],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    options={"build_ext": {"build_temp": str(ROOT / "build/temp_cp_output")}},
    packages=[],
)
