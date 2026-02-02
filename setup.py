"""Legacy build script kept for editable installs.

All metadata has moved to pyproject.toml (PEP 621). This file only defines the C++ Extension
for environments that still invoke setup.py directly (e.g. some older tooling or manual builds).
"""

from setuptools import setup, Extension
from pathlib import Path
import os
import platform


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


def _detect_edsdk_root() -> Path:
    env_path = os.getenv("EDSDK_PATH") or os.getenv("EDSDK_ROOT")
    if env_path:
        return Path(env_path)
    if platform.system() == "Linux":
        candidate = PROJECT_ROOT / "EDSDKv132010L" / "Linux" / "EDSDK"
        if candidate.exists():
            return candidate
    # Default to local dependencies layout (Windows-focused)
    return ROOT / "dependencies" / "EDSDK"


def _detect_library_dir(edsdk_root: Path) -> Path:
    if platform.system() != "Linux":
        return ROOT / "dependencies" / "EDSDK_64" / "Library"
    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64"):
        return edsdk_root / "Library" / "x86_64"
    if arch in ("aarch64", "arm64"):
        return edsdk_root / "Library" / "ARM64"
    if arch in ("armv7l", "armv7", "arm"):
        return edsdk_root / "Library" / "ARM32"
    return edsdk_root / "Library" / "x86_64"


edsdk_root = _detect_edsdk_root()
library_dir = _detect_library_dir(edsdk_root)
include_dir = edsdk_root / "Header"

extra_compile_args = ["/W4", "/DDEBUG=0"]
extra_link_args = []
if platform.system() == "Linux":
    extra_compile_args = ["-Wall", "-Wextra", "-Wno-unused-parameter"]
    extra_link_args = [f"-Wl,-rpath,{library_dir}"]

extension = Extension(
    "edsdk.api",
    libraries=["EDSDK"],
    include_dirs=[str(include_dir)],
    library_dirs=[str(library_dir)],
    depends=["edsdk/edsdk_python.h", "edsdk/edsdk_utils.h"],
    sources=["edsdk/edsdk_python.cpp", "edsdk/edsdk_utils.cpp"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)


# Delegate metadata to pyproject.toml; build extension here.
setup(ext_modules=[extension])
