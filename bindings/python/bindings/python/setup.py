"""
quantcpp Python package setup.

Install in development mode:
    pip install -e .

Install from source:
    pip install .

The package uses ctypes to load the pre-built quant.cpp shared library.
You must build the C library first:

    cd /path/to/quant.cpp
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
    cmake --build build -j$(nproc)

Then set TURBOQUANT_LIB if the library is not in a standard location:
    export TURBOQUANT_LIB=/path/to/build/libturboquant.dylib
"""

from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent

long_description = ""
readme_path = this_dir / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="quantcpp",
    version="0.5.0",
    description="Python bindings for quant.cpp -- LLM inference with 7x KV compression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="quant.cpp contributors",
    license="MIT",
    url="https://github.com/hunscompany/quant.cpp",
    packages=find_packages(),
    python_requires=">=3.8",
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords="llm inference quantization kv-cache compression",
    project_urls={
        "Documentation": "https://github.com/hunscompany/quant.cpp/tree/main/docs",
        "Source": "https://github.com/hunscompany/quant.cpp",
        "Tracker": "https://github.com/hunscompany/quant.cpp/issues",
    },
)
