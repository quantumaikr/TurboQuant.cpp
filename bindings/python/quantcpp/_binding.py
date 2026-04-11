"""
quantcpp._binding -- ctypes wrapper for the quant.h C API.

Loads the compiled shared library (libquant.so / libquant.dylib) and
exposes thin Python wrappers around the 7 public C functions:

    quant_load(path)             -> quant_model*
    quant_new(model, config)     -> quant_ctx*
    quant_generate(ctx, prompt, on_token, user_data) -> int
    quant_ask(ctx, prompt)       -> char*
    quant_free_ctx(ctx)
    quant_free_model(model)
    quant_version()              -> char*
"""

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

def _lib_name() -> str:
    if sys.platform == "darwin":
        return "libquant.dylib"
    elif sys.platform == "win32":
        return "quant.dll"
    else:
        return "libquant.so"


def _find_library() -> str:
    """Locate the libquant shared library.

    Search order:
      1. QUANTCPP_LIB environment variable (explicit path)
      2. Same directory as this Python file (installed alongside package)
      3. build/ relative to project root (development layout)
      4. System library path
    """
    # 1. Explicit environment variable
    env_path = os.environ.get("QUANTCPP_LIB")
    if env_path and os.path.isfile(env_path):
        return env_path

    lib = _lib_name()
    pkg_dir = Path(__file__).resolve().parent

    candidates = [
        # 2. Installed alongside the package
        pkg_dir / lib,
        # 3. Development: bindings/python/quantcpp -> project_root/build/
        pkg_dir.parent.parent.parent / "build" / lib,
        pkg_dir.parent.parent.parent / "build" / "lib" / lib,
    ]

    for c in candidates:
        if c.is_file():
            return str(c)

    # 4. System search
    system_path = ctypes.util.find_library("quant")
    if system_path:
        return system_path

    raise OSError(
        f"Cannot find {lib}. Either:\n"
        "  - pip install . (compiles automatically)\n"
        "  - Set QUANTCPP_LIB=/path/to/libquant.so\n"
        "  - Place the library in the package directory"
    )


# ---------------------------------------------------------------------------
# Load and configure the C library
# ---------------------------------------------------------------------------

_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    path = _find_library()
    _lib = ctypes.CDLL(path)
    _setup_signatures(_lib)
    return _lib


# Callback type: void (*on_token)(const char* text, void* user_data)
ON_TOKEN_CB = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


class QuantConfig(ctypes.Structure):
    """Mirror of quant_config from quant.h."""
    _fields_ = [
        ("temperature", ctypes.c_float),     # default: 0.7
        ("top_p", ctypes.c_float),           # default: 0.9
        ("max_tokens", ctypes.c_int),        # default: 256
        ("n_threads", ctypes.c_int),         # default: 4
        ("kv_compress", ctypes.c_int),       # 0=off, 1=4-bit, 2=delta+3-bit
        ("context_length", ctypes.c_int),    # 0=auto(4096), or user override
        ("k_highres_window", ctypes.c_int),  # 0=off, 128=sweet spot for progressive
    ]


def _setup_signatures(lib: ctypes.CDLL) -> None:
    """Declare C function signatures for type safety."""

    # quant_model* quant_load(const char* path)
    lib.quant_load.argtypes = [ctypes.c_char_p]
    lib.quant_load.restype = ctypes.c_void_p

    # quant_ctx* quant_new(quant_model* model, const quant_config* config)
    lib.quant_new.argtypes = [ctypes.c_void_p, ctypes.POINTER(QuantConfig)]
    lib.quant_new.restype = ctypes.c_void_p

    # int quant_generate(quant_ctx* ctx, const char* prompt,
    #                    void (*on_token)(const char*, void*), void* user_data)
    lib.quant_generate.argtypes = [
        ctypes.c_void_p,    # ctx
        ctypes.c_char_p,    # prompt
        ON_TOKEN_CB,        # on_token callback
        ctypes.c_void_p,    # user_data
    ]
    lib.quant_generate.restype = ctypes.c_int

    # int quant_chat(quant_ctx* ctx, const char* prompt,
    #                void (*on_token)(const char*, void*), void* user_data)
    # Multi-turn chat with KV cache reuse — avoids the O(n^2) prefill cost
    # of quant_generate when the user re-sends conversation history.
    # Optional: only present in single-header builds (>= v0.13).
    if hasattr(lib, "quant_chat"):
        lib.quant_chat.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ON_TOKEN_CB,
            ctypes.c_void_p,
        ]
        lib.quant_chat.restype = ctypes.c_int

    # char* quant_ask(quant_ctx* ctx, const char* prompt)
    lib.quant_ask.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.quant_ask.restype = ctypes.c_void_p  # We use c_void_p so we can free()

    # void quant_free_string(char*) — added in v0.8.2 to free quant_ask
    # results without cross-heap libc.free() crashes on macOS arm64.
    # Optional: older single-headers may not export this symbol.
    if hasattr(lib, "quant_free_string"):
        lib.quant_free_string.argtypes = [ctypes.c_void_p]
        lib.quant_free_string.restype = None

    # int quant_save_context(quant_ctx* ctx, const char* path)
    if hasattr(lib, "quant_save_context"):
        lib.quant_save_context.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.quant_save_context.restype = ctypes.c_int

    # int quant_load_context(quant_ctx* ctx, const char* path)
    if hasattr(lib, "quant_load_context"):
        lib.quant_load_context.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.quant_load_context.restype = ctypes.c_int

    # void quant_free_ctx(quant_ctx* ctx)
    lib.quant_free_ctx.argtypes = [ctypes.c_void_p]
    lib.quant_free_ctx.restype = None

    # void quant_free_model(quant_model* model)
    lib.quant_free_model.argtypes = [ctypes.c_void_p]
    lib.quant_free_model.restype = None

    # const char* quant_version(void)
    lib.quant_version.argtypes = []
    lib.quant_version.restype = ctypes.c_char_p

    # libc free() for quant_ask return value
    if sys.platform == "win32":
        lib_c = ctypes.cdll.msvcrt
    else:
        lib_c = ctypes.CDLL(None)  # default libc
    lib_c.free.argtypes = [ctypes.c_void_p]
    lib_c.free.restype = None


def get_lib() -> ctypes.CDLL:
    """Return the loaded C library handle (for advanced use)."""
    return _get_lib()


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def load_model(path: str) -> ctypes.c_void_p:
    """Load a GGUF model file. Returns an opaque model handle."""
    lib = _get_lib()
    model = lib.quant_load(path.encode("utf-8"))
    if not model:
        raise RuntimeError(f"Failed to load model: {path}")
    return model


def new_context(
    model,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
    n_threads: int = 4,
    kv_compress: int = 1,
    context_length: int = 0,
    k_highres_window: int = 0,
) -> ctypes.c_void_p:
    """Create an inference context with the given config."""
    lib = _get_lib()
    cfg = QuantConfig(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n_threads=n_threads,
        kv_compress=kv_compress,
        context_length=context_length,
        k_highres_window=k_highres_window,
    )
    ctx = lib.quant_new(model, ctypes.byref(cfg))
    if not ctx:
        raise RuntimeError("Failed to create inference context")
    return ctx


def generate(ctx, prompt: str, callback, user_data=None) -> int:
    """Generate tokens with a callback. Returns token count."""
    lib = _get_lib()
    cb = ON_TOKEN_CB(callback)
    # Must prevent cb from being garbage collected during the call
    n = lib.quant_generate(ctx, prompt.encode("utf-8"), cb, user_data)
    return n


def ask(ctx, prompt: str) -> str:
    """Generate a full response string. Handles memory cleanup."""
    lib = _get_lib()
    ptr = lib.quant_ask(ctx, prompt.encode("utf-8"))
    if not ptr:
        return ""
    # Read the string before freeing
    result = ctypes.cast(ptr, ctypes.c_char_p).value
    text = result.decode("utf-8", errors="replace") if result else ""
    # Free the C-allocated string
    libc = ctypes.CDLL(None) if sys.platform != "win32" else ctypes.cdll.msvcrt
    libc.free(ptr)
    return text


def free_ctx(ctx) -> None:
    """Free an inference context."""
    lib = _get_lib()
    lib.quant_free_ctx(ctx)


def free_model(model) -> None:
    """Free a model."""
    lib = _get_lib()
    lib.quant_free_model(model)


def version() -> str:
    """Return the quant.cpp version string."""
    lib = _get_lib()
    v = lib.quant_version()
    return v.decode("utf-8") if v else "unknown"
