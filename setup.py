"""Legacy build script kept for editable installs.

All metadata has moved to pyproject.toml (PEP 621). This file only defines the C++ Extension
for environments that still invoke setup.py directly (e.g. some older tooling or manual builds).
"""

from setuptools import setup, Extension

EDSDK_PATH = "dependencies"

extension = Extension(
    "edsdk.api",
    libraries=["EDSDK"],
    include_dirs=[f"{EDSDK_PATH}/EDSDK/Header"],
    library_dirs=[f"{EDSDK_PATH}/EDSDK_64/Library"],
    depends=["edsdk/edsdk_python.h", "edsdk/edsdk_utils.h"],
    sources=["edsdk/edsdk_python.cpp", "edsdk/edsdk_utils.cpp"],
    extra_compile_args=["/W4", "/DDEBUG=0"],
)


# Delegate metadata to pyproject.toml; build extension here.
setup(ext_modules=[extension])
