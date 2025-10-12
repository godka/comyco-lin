from __future__ import annotations

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


ext_modules = [
    Pybind11Extension(
        "libcore",
        ["core.cc"],
        extra_compile_args=["-O3", "-Wall", "-std=c++11"],
    )
]


setup(
    name="libcore",
    version="0.1.0",
    description="Pybind11 bindings for the ABR environment core",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    package_dir={"": ".."},
    zip_safe=False,
)
