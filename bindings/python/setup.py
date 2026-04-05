"""
quantcpp build script.

Compiles quant.h into a shared library at install time using the system
C compiler. No build dependencies beyond a working C toolchain.

    pip install .           # build + install
    pip install -e .        # editable / development install
    python setup.py build   # just compile the shared library
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent          # quant.cpp repo root
QUANT_H = PROJECT_ROOT / "quant.h"

# The tiny C file that triggers the implementation
_IMPL_C = """
#define QUANT_IMPLEMENTATION
#include "quant.h"
"""


def _lib_name() -> str:
    """Return the platform-appropriate shared library filename."""
    if sys.platform == "darwin":
        return "libquant.dylib"
    elif sys.platform == "win32":
        return "quant.dll"
    else:
        return "libquant.so"


def _find_cc() -> str:
    """Find a C compiler."""
    for cc in [os.environ.get("CC"), "cc", "gcc", "clang"]:
        if cc and shutil.which(cc):
            return cc
    raise RuntimeError(
        "No C compiler found. Install gcc or clang, or set the CC "
        "environment variable."
    )


def _compile_shared_lib(output_dir: Path) -> Path:
    """Compile quant.h into a shared library and return the output path."""
    if not QUANT_H.is_file():
        raise FileNotFoundError(
            f"quant.h not found at {QUANT_H}. "
            "Make sure you are building from the quant.cpp repository."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    lib_name = _lib_name()
    lib_path = output_dir / lib_name

    # Write the implementation .c file
    impl_c = output_dir / "_quant_impl.c"
    impl_c.write_text(_IMPL_C)

    cc = _find_cc()
    cmd = [cc]

    # Shared library flags
    if sys.platform == "darwin":
        cmd += ["-dynamiclib", "-install_name", f"@rpath/{lib_name}"]
    elif sys.platform == "win32":
        cmd += ["-shared"]
    else:
        cmd += ["-shared", "-fPIC"]

    cmd += [
        "-O2",
        "-fPIC",
        "-I", str(PROJECT_ROOT),
        str(impl_c),
        "-o", str(lib_path),
        "-lm",
    ]

    # pthreads (not needed on Windows)
    if sys.platform != "win32":
        cmd.append("-lpthread")

    # Suppress common warnings from single-header builds
    cmd += ["-w"]

    print(f"[quantcpp] Compiling {QUANT_H.name} -> {lib_name}")
    print(f"[quantcpp] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[quantcpp] STDOUT: {result.stdout}", file=sys.stderr)
        print(f"[quantcpp] STDERR: {result.stderr}", file=sys.stderr)
        raise RuntimeError(
            f"Compilation failed (exit code {result.returncode}).\n"
            f"stderr: {result.stderr}"
        )

    print(f"[quantcpp] Built {lib_path} ({lib_path.stat().st_size:,} bytes)")
    return lib_path


# ---------------------------------------------------------------------------
# Custom build command
# ---------------------------------------------------------------------------

class BuildWithCompile(build_py):
    """Extend build_py to compile the shared library into the package."""

    def run(self):
        super().run()

        # Destination inside the built package
        pkg_dir = Path(self.build_lib) / "quantcpp"
        pkg_dir.mkdir(parents=True, exist_ok=True)

        _compile_shared_lib(pkg_dir)


# ---------------------------------------------------------------------------
# Also compile for editable installs (pip install -e .)
# ---------------------------------------------------------------------------

class BuildInPlace(build_py):
    """For editable installs, compile into the source tree."""

    def run(self):
        super().run()

        # Build into the source quantcpp/ directory
        pkg_dir = HERE / "quantcpp"
        lib_path = pkg_dir / _lib_name()
        if not lib_path.exists():
            _compile_shared_lib(pkg_dir)


def _get_build_class():
    """Choose build class based on whether this is an editable install."""
    # pip install -e . sets this; regular install does not
    if "develop" in sys.argv or "editable_wheel" in sys.argv:
        return BuildInPlace
    return BuildWithCompile


setup(
    cmdclass={"build_py": _get_build_class()},
)
