from pathlib import Path
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

if sys.platform != "win32" or sys.maxsize <= 2**32:
    raise RuntimeError("torch-iris requires 64-bit Windows")

root = Path(__file__).parent.resolve()
setup(
    package_dir={"": "src"},
    packages=["torch_iris"],
    ext_modules=[
        CppExtension(
            "torch_iris._C",
            sources=["csrc/backend.cpp", "csrc/opencl_runtime.cpp", "csrc/spatial.cpp", "csrc/ops_extra.cpp", "csrc/ops_phase3.cpp", "csrc/autocast.cpp", "csrc/fallback.cpp"],
            depends=["csrc/opencl_runtime.h", "csrc/rng_internal.h", "csrc/fallback.h", "csrc/kernels.h", "csrc/spatial_kernels.h", "csrc/ops_kernels.h", "csrc/ops_phase3_kernels.h"],
            include_dirs=[str(root / "third_party" / "opencl")],
            define_macros=[("CL_TARGET_OPENCL_VERSION", "120")],
            extra_compile_args=["/std:c++20", "/O2"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
