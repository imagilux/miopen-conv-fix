import os
import sys

from setuptools import setup

# Only build the C extension on ROCm
try:
    import torch

    _IS_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None
except ImportError:
    _IS_ROCM = False

ext_modules = []

if _IS_ROCM:
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension

    rocm_home = os.environ.get("ROCM_HOME", "/opt/rocm")

    ext_modules = [
        CUDAExtension(
            name="miopen_conv_fix._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/miopen_conv.hip",
            ],
            include_dirs=[
                os.path.join(rocm_home, "include"),
                os.path.join(rocm_home, "include", "rocthrust"),
                os.path.join(rocm_home, "include", "rocprim"),
            ],
            libraries=["MIOpen"],
            library_dirs=[
                os.path.join(rocm_home, "lib"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3", "-std=c++17",
                    f"-I{os.path.join(rocm_home, 'include', 'rocthrust')}",
                    f"-I{os.path.join(rocm_home, 'include', 'rocprim')}",
                ],
            },
        )
    ]

    setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
    )
else:
    # Pure-Python install (no-op on CUDA/CPU)
    setup()
