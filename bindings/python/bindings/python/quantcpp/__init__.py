"""
quantcpp — Python bindings for quant.cpp LLM inference engine.

Usage:
    from quantcpp import Model

    m = Model("model.gguf")
    text = m.ask("What is 2+2?")

    for token in m.generate("Hello"):
        print(token, end="", flush=True)
"""

__version__ = "0.5.0"

import ctypes
import ctypes.util
import os
import sys
import threading
from pathlib import Path
from typing import Iterator, Optional


def _find_library() -> ctypes.CDLL:
    """Locate and load libturboquant shared library.

    Search order:
      1. TURBOQUANT_LIB environment variable (explicit path)
      2. Same directory as this Python file
      3. build/ relative to project root (development layout)
      4. System library path (ldconfig / DYLD)
    """
    env_path = os.environ.get("TURBOQUANT_LIB")
    if env_path and os.path.isfile(env_path):
        return ctypes.CDLL(env_path)

    if sys.platform == "darwin":
        lib_name = "libturboquant.dylib"
    elif sys.platform == "win32":
        lib_name = "turboquant.dll"
    else:
        lib_name = "libturboquant.so"

    # Directory containing this __init__.py
    pkg_dir = Path(__file__).resolve().parent

    candidates = [
        pkg_dir / lib_name,
        pkg_dir.parent / lib_name,
        # Development layout: bindings/python/quantcpp -> project_root/build/
        pkg_dir.parent.parent.parent / "build" / lib_name,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return ctypes.CDLL(str(candidate))

    # Fall back to system search
    system_path = ctypes.util.find_library("turboquant")
    if system_path:
        return ctypes.CDLL(system_path)

    raise OSError(
        f"Cannot find {lib_name}. Set TURBOQUANT_LIB environment variable "
        "or place the library in the build/ directory. "
        "Build with: cmake -B build && cmake --build build"
    )


_lib = _find_library()

# ── C function signatures ────────────────────────────────────────────

# tq_model_t* tq_load_model(const char* path)
_lib.tq_load_model.argtypes = [ctypes.c_char_p]
_lib.tq_load_model.restype = ctypes.c_void_p

# void tq_free_model(tq_model_t* model)
_lib.tq_free_model.argtypes = [ctypes.c_void_p]
_lib.tq_free_model.restype = None

# tq_tokenizer_t* tq_load_tokenizer_from_gguf(const void* gguf_ctx)
_lib.tq_load_tokenizer_from_gguf.argtypes = [ctypes.c_void_p]
_lib.tq_load_tokenizer_from_gguf.restype = ctypes.c_void_p

# tq_tokenizer_t* tq_load_tokenizer(const char* path)
_lib.tq_load_tokenizer.argtypes = [ctypes.c_char_p]
_lib.tq_load_tokenizer.restype = ctypes.c_void_p

# void tq_free_tokenizer(tq_tokenizer_t* tok)
_lib.tq_free_tokenizer.argtypes = [ctypes.c_void_p]
_lib.tq_free_tokenizer.restype = None

# int tq_encode(const tq_tokenizer_t* tok, const char* text,
#               int* tokens, int max_tokens, int add_bos)
_lib.tq_encode.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
]
_lib.tq_encode.restype = ctypes.c_int

# const char* tq_decode(const tq_tokenizer_t* tok, int prev_token, int token)
_lib.tq_decode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.tq_decode.restype = ctypes.c_char_p

# tq_gen_config_t tq_default_gen_config(void)
# We build the struct in Python instead of calling this, since returning
# structs by value via ctypes is fragile across platforms.

# tq_state_t* tq_create_state(const tq_model_config_t* config, tq_type kv_type)
_lib.tq_create_state.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.tq_create_state.restype = ctypes.c_void_p

# tq_state_t* tq_create_state_ex(const tq_model_config_t* config, tq_type kv_type, int value_quant_bits)
_lib.tq_create_state_ex.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.tq_create_state_ex.restype = ctypes.c_void_p

# void tq_free_state(tq_state_t* state)
_lib.tq_free_state.argtypes = [ctypes.c_void_p]
_lib.tq_free_state.restype = None

# float* tq_forward(tq_model_t* model, tq_state_t* state, int token, int pos)
_lib.tq_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.tq_forward.restype = ctypes.POINTER(ctypes.c_float)

# int tq_generate(tq_model_t* model, tq_tokenizer_t* tokenizer,
#                 const char* prompt, tq_gen_config_t* config,
#                 char* output, int output_size)
_lib.tq_generate.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_char_p, ctypes.c_void_p,
    ctypes.c_char_p, ctypes.c_int,
]
_lib.tq_generate.restype = ctypes.c_int

# int tq_sample_argmax(const float* logits, int vocab_size)
_lib.tq_sample_argmax.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.tq_sample_argmax.restype = ctypes.c_int

# int tq_sample_topp(const float* logits, int vocab_size,
#                    float temperature, float top_p, unsigned long long* rng)
_lib.tq_sample_topp.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_ulonglong),
]
_lib.tq_sample_topp.restype = ctypes.c_int

# const char* tq_type_name(tq_type type)
_lib.tq_type_name.argtypes = [ctypes.c_int]
_lib.tq_type_name.restype = ctypes.c_char_p

# const char* tq_status_string(tq_status status)
_lib.tq_status_string.argtypes = [ctypes.c_int]
_lib.tq_status_string.restype = ctypes.c_char_p


# ── KV compression type mapping ─────────────────────────────────────

# tq_type enum values (must match tq_types.h)
TQ_TYPE_POLAR_3B = 0
TQ_TYPE_POLAR_4B = 1
TQ_TYPE_QJL_1B = 2
TQ_TYPE_TURBO_3B = 3
TQ_TYPE_TURBO_4B = 4
TQ_TYPE_UNIFORM_4B = 5
TQ_TYPE_UNIFORM_2B = 6
TQ_TYPE_MIXED_4B8 = 7
TQ_TYPE_TURBO_KV_3B = 8
TQ_TYPE_TURBO_KV_4B = 9
TQ_TYPE_TURBO_KV_1B = 10
TQ_TYPE_TURBO_KV_2B = 11
TQ_TYPE_UNIFORM_3B = 12

_KV_COMPRESS_MAP = {
    None: TQ_TYPE_TURBO_KV_3B,  # sensible default
    "none": TQ_TYPE_POLAR_3B,   # placeholder for "no special KV quant"
    "1bit": TQ_TYPE_TURBO_KV_1B,
    "2bit": TQ_TYPE_TURBO_KV_2B,
    "3bit": TQ_TYPE_TURBO_KV_3B,
    "4bit": TQ_TYPE_TURBO_KV_4B,
    "polar3": TQ_TYPE_POLAR_3B,
    "polar4": TQ_TYPE_POLAR_4B,
    "qjl": TQ_TYPE_QJL_1B,
    "turbo3": TQ_TYPE_TURBO_3B,
    "turbo4": TQ_TYPE_TURBO_4B,
    "uniform4": TQ_TYPE_UNIFORM_4B,
    "uniform2": TQ_TYPE_UNIFORM_2B,
    "uniform3": TQ_TYPE_UNIFORM_3B,
}


# ── tq_gen_config_t ctypes struct ────────────────────────────────────

# Callback type: void (*on_token)(const char* text, void* user_data)
_ON_TOKEN_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


class _GenConfig(ctypes.Structure):
    """Mirror of tq_gen_config_t from tq_engine.h."""
    _fields_ = [
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("max_tokens", ctypes.c_int),
        ("kv_type", ctypes.c_int),
        ("value_quant_bits", ctypes.c_int),
        ("v_highres_window", ctypes.c_int),
        ("delta_kv", ctypes.c_int),
        ("delta_iframe_interval", ctypes.c_int),
        ("k_highres_window", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("rep_penalty", ctypes.c_float),
        ("rep_window", ctypes.c_int),
        ("on_token", _ON_TOKEN_FUNC),
        ("user_data", ctypes.c_void_p),
    ]


def _make_gen_config(
    *,
    kv_type: int = TQ_TYPE_TURBO_KV_3B,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 512,
    n_threads: int = 0,
    rep_penalty: float = 1.1,
    rep_window: int = 32,
    on_token=None,
    user_data=None,
) -> _GenConfig:
    cfg = _GenConfig()
    cfg.temperature = temperature
    cfg.top_p = top_p
    cfg.max_tokens = max_tokens
    cfg.kv_type = kv_type
    cfg.value_quant_bits = 0
    cfg.v_highres_window = 0
    cfg.delta_kv = 0
    cfg.delta_iframe_interval = 0
    cfg.k_highres_window = 0
    cfg.n_threads = n_threads
    cfg.rep_penalty = rep_penalty
    cfg.rep_window = rep_window
    if on_token is not None:
        cfg.on_token = _ON_TOKEN_FUNC(on_token)
    else:
        cfg.on_token = _ON_TOKEN_FUNC(0)
    cfg.user_data = user_data
    return cfg


# ── Chat template helpers ────────────────────────────────────────────

def _chat_wrap(prompt: str, model_type: int = 0) -> str:
    """Wrap a user message in a chat template based on model architecture.

    model_type values (from tq_model_config_t):
      0 = qwen35  (Qwen/ChatML style)
      1 = gemma3   (Gemma style)
      2 = qwen2moe (Qwen/ChatML style)
    """
    if model_type == 1:
        # Gemma chat template
        return (
            "<start_of_turn>user\n"
            f"{prompt}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
    else:
        # Qwen / ChatML style (default)
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


# ── Public API ───────────────────────────────────────────────────────

class Model:
    """High-level Python interface to quant.cpp inference.

    Parameters
    ----------
    path : str
        Path to a GGUF or TQM model file.
    kv_compress : str, optional
        KV cache compression mode. One of:
        "1bit", "2bit", "3bit" (default), "4bit",
        "polar3", "polar4", "qjl", "turbo3", "turbo4",
        "uniform2", "uniform3", "uniform4", "none".
    n_threads : int, optional
        Number of CPU threads (0 = auto-detect).
    """

    def __init__(
        self,
        path: str,
        kv_compress: Optional[str] = None,
        n_threads: int = 0,
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        self._path = path
        self._n_threads = n_threads

        # Resolve KV compression type
        kv_key = kv_compress.lower() if kv_compress else None
        if kv_key is not None and kv_key not in _KV_COMPRESS_MAP:
            valid = ", ".join(k for k in _KV_COMPRESS_MAP if k is not None)
            raise ValueError(
                f"Unknown kv_compress={kv_compress!r}. Valid: {valid}"
            )
        self._kv_type = _KV_COMPRESS_MAP.get(kv_key, TQ_TYPE_TURBO_KV_3B)

        # Load model
        self._model = _lib.tq_load_model(path.encode("utf-8"))
        if not self._model:
            raise RuntimeError(f"Failed to load model: {path}")

        # Load tokenizer from the GGUF context embedded in the model.
        # tq_model_t.gguf_ctx is at a known offset; we access it by
        # loading the tokenizer from the same file path as fallback.
        self._tokenizer = None
        if path.lower().endswith(".gguf"):
            self._tokenizer = _lib.tq_load_tokenizer(path.encode("utf-8"))
        if not self._tokenizer:
            # Try loading tokenizer from TQM or standalone file
            self._tokenizer = _lib.tq_load_tokenizer(path.encode("utf-8"))

        # Read model_type from config (first field after config start).
        # We use a simplified approach: default to 0 (qwen/chatml).
        self._model_type = 0

        self._lock = threading.Lock()

    def __del__(self):
        self.close()

    def close(self):
        """Release model and tokenizer resources."""
        if hasattr(self, "_tokenizer") and self._tokenizer:
            _lib.tq_free_tokenizer(self._tokenizer)
            self._tokenizer = None
        if hasattr(self, "_model") and self._model:
            _lib.tq_free_model(self._model)
            self._model = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _ensure_loaded(self):
        if not self._model:
            raise RuntimeError("Model has been closed")
        if not self._tokenizer:
            raise RuntimeError("Tokenizer not available for this model")

    def ask(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> str:
        """Send a chat-formatted prompt and return the full response.

        Automatically wraps the prompt in a chat template based on the
        model architecture.
        """
        wrapped = _chat_wrap(prompt, self._model_type)
        return self._generate_text(
            wrapped,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def chat(
        self,
        message: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> str:
        """Convenience alias for ask() with chat template."""
        return self.ask(
            message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
        chat: bool = False,
    ) -> Iterator[str]:
        """Stream tokens from a prompt. Yields token strings one at a time.

        Parameters
        ----------
        prompt : str
            Raw prompt text (or user message if chat=True).
        max_tokens : int
            Maximum tokens to generate.
        temperature : float
            Sampling temperature (0.0 = greedy).
        top_p : float
            Nucleus sampling threshold.
        chat : bool
            If True, wrap prompt in chat template first.
        """
        if chat:
            prompt = _chat_wrap(prompt, self._model_type)

        self._ensure_loaded()

        # We use a thread + callback to stream tokens via a shared list
        tokens_queue = []
        done_event = threading.Event()
        error_holder = [None]

        def _on_token_cb(text_ptr, _user_data):
            if text_ptr:
                tokens_queue.append(text_ptr.decode("utf-8", errors="replace"))

        # Must keep a reference to prevent GC of the callback
        cb = _ON_TOKEN_FUNC(_on_token_cb)

        cfg = _make_gen_config(
            kv_type=self._kv_type,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n_threads=self._n_threads,
        )
        cfg.on_token = cb
        cfg.user_data = None

        output_buf = ctypes.create_string_buffer(max_tokens * 16)

        def _run():
            try:
                with self._lock:
                    _lib.tq_generate(
                        self._model,
                        self._tokenizer,
                        prompt.encode("utf-8"),
                        ctypes.byref(cfg),
                        output_buf,
                        len(output_buf),
                    )
            except Exception as e:
                error_holder[0] = e
            finally:
                done_event.set()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        yielded = 0
        while not done_event.is_set() or yielded < len(tokens_queue):
            if yielded < len(tokens_queue):
                yield tokens_queue[yielded]
                yielded += 1
            else:
                done_event.wait(timeout=0.01)

        # Yield any remaining tokens
        while yielded < len(tokens_queue):
            yield tokens_queue[yielded]
            yielded += 1

        if error_holder[0] is not None:
            raise error_holder[0]

    def _generate_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> str:
        """Generate text synchronously (blocking). Returns full output."""
        self._ensure_loaded()

        cfg = _make_gen_config(
            kv_type=self._kv_type,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n_threads=self._n_threads,
        )

        buf_size = max_tokens * 16  # generous buffer
        output_buf = ctypes.create_string_buffer(buf_size)

        with self._lock:
            n = _lib.tq_generate(
                self._model,
                self._tokenizer,
                prompt.encode("utf-8"),
                ctypes.byref(cfg),
                output_buf,
                buf_size,
            )

        if n < 0:
            raise RuntimeError(f"tq_generate failed with code {n}")

        return output_buf.value.decode("utf-8", errors="replace")

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> str:
        """Raw completion without chat template wrapping."""
        return self._generate_text(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    @property
    def path(self) -> str:
        """Path to the loaded model file."""
        return self._path

    @property
    def kv_type(self) -> str:
        """Name of the KV compression type in use."""
        name = _lib.tq_type_name(self._kv_type)
        return name.decode("utf-8") if name else "unknown"


# ── Module-level convenience ─────────────────────────────────────────

def load(path: str, **kwargs) -> Model:
    """Shorthand for Model(path, **kwargs)."""
    return Model(path, **kwargs)
