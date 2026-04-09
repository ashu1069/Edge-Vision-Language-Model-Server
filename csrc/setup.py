"""
Build script for edge_nms C++ extension.

Usage:
    cd csrc
    python setup.py install     # Install into current environment
    python setup.py build_ext --inplace  # Build .so in-place for testing

The extension is JIT-compilable too:
    import torch.utils.cpp_extension
    edge_nms = torch.utils.cpp_extension.load(name="edge_nms", sources=["csrc/nms.cpp"])
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="edge_nms",
    version="0.1.0",
    ext_modules=[
        CppExtension(
            name="edge_nms",
            sources=["nms.cpp"],
            extra_compile_args=["-O3", "-std=c++17"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
