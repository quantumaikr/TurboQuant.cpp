"""
quantcpp -- Compress AI's memory 3x. It gets faster.

Quick start:

    from quantcpp import Model
    m = Model.from_pretrained("Llama-3.2-1B")
    print(m.ask("What is gravity?"))

Note: SmolLM2-135M downloads faster but produces low-quality output.
Use Llama-3.2-1B (~750 MB, one-time download) for good results.
"""

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("quantcpp")
except Exception:
    __version__ = "0.12.1"  # fallback for editable / source-tree imports

import os
import sys
import threading
from pathlib import Path
from typing import Iterator, Optional

from quantcpp._binding import (
    QuantConfig,
    ON_TOKEN_CB,
    get_lib,
    load_model,
    new_context,
    free_ctx,
    free_model,
    version as _c_version,
)


# -----------------------------------------------------------------------
# Model registry — small GGUF models auto-downloaded from HuggingFace
# -----------------------------------------------------------------------

_CACHE_DIR = Path(os.environ.get("QUANTCPP_CACHE",
                                  Path.home() / ".cache" / "quantcpp"))

# name → (HuggingFace repo, filename, approx size in MB)
_MODEL_REGISTRY = {
    "SmolLM2-135M": (
        "Felladrin/gguf-Q8_0-SmolLM2-135M-Instruct",
        "smollm2-135m-instruct-q8_0.gguf",
        135,
    ),
    "Qwen3.5-0.8B": (
        "unsloth/Qwen3.5-0.8B-GGUF",
        "Qwen3.5-0.8B-Q4_K_M.gguf",
        508,
    ),
    "Llama-3.2-1B": (
        "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF",
        "llama-3.2-1b-instruct-q4_k_m.gguf",
        750,
    ),
}

def available_models():
    """List available model names for ``from_pretrained``."""
    return sorted(_MODEL_REGISTRY.keys())


def _download_with_progress(url: str, dest: Path, desc: str) -> None:
    """Download a file with a tqdm-free progress bar (stdlib only)."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")

    req = urllib.request.Request(url, headers={"User-Agent": f"quantcpp/{__version__}"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        block = 1024 * 256  # 256 KB chunks

        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(block)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    mb = downloaded / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    bar_len = 30
                    filled = bar_len * downloaded // total
                    bar = "#" * filled + "-" * (bar_len - filled)
                    print(f"\r  [{bar}] {pct:3d}% ({mb:.0f}/{total_mb:.0f} MB) {desc}",
                          end="", flush=True, file=sys.stderr)
        print(file=sys.stderr)

    tmp.rename(dest)


def download(name: str) -> str:
    """Download a model from HuggingFace Hub and return its local path.

    Parameters
    ----------
    name : str
        Model name from the registry. Currently available:
        ``"SmolLM2-135M"`` (~135 MB, good for testing).

    Returns
    -------
    str
        Path to the downloaded ``.gguf`` file.

    Examples
    --------
    >>> path = quantcpp.download("SmolLM2-135M")
    >>> m = quantcpp.Model(path)
    """
    if name not in _MODEL_REGISTRY:
        avail = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model {name!r}. Available: {avail}. "
            "Or pass a local .gguf path to Model() directly."
        )

    repo, filename, _mb = _MODEL_REGISTRY[name]
    dest = _CACHE_DIR / filename

    if dest.is_file():
        print(f"  Using cached {dest}", file=sys.stderr)
        return str(dest)

    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    print(f"  Downloading {name} (~{_mb} MB) ...", file=sys.stderr)
    _download_with_progress(url, dest, name)
    return str(dest)


class Model:
    """High-level Python interface to quant.cpp inference.

    Parameters
    ----------
    path : str
        Path to a GGUF model file. Use ``Model.from_pretrained("SmolLM2-135M")``
        to auto-download a small model for quick testing.
    temperature : float
        Sampling temperature (default 0.7). Use 0.0 for greedy.
    top_p : float
        Nucleus sampling threshold (default 0.9).
    max_tokens : int
        Maximum tokens per generation (default 256).
    n_threads : int
        CPU thread count (default 4).
    kv_compress : int
        KV cache compression: 0=off (default in v0.8.x).

    Examples
    --------
    >>> m = Model.from_pretrained("SmolLM2-135M")
    >>> m.ask("What is gravity?")
    'Gravity is a force that attracts ...'

    >>> with Model("model.gguf") as m:
    ...     for tok in m.generate("Once upon a time"):
    ...         print(tok, end="")
    """

    @classmethod
    def from_pretrained(cls, name: str, **kwargs) -> "Model":
        """Download a model and create a Model instance in one call.

        Parameters
        ----------
        name : str
            Model name (e.g. ``"SmolLM2-135M"``). See ``quantcpp.download()``.
        **kwargs
            Forwarded to ``Model.__init__`` (temperature, max_tokens, etc.).
        """
        path = download(name)
        return cls(path, **kwargs)

    def __init__(
        self,
        path: str,
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        n_threads: int = 4,
        kv_compress: int = 1,
        context_length: int = 0,
        progressive: bool = True,
        aggressive: bool = False,
    ):
        """
        Parameters
        ----------
        progressive : bool
            Progressive KV compression (default True). Keeps last 128
            tokens' keys at FP32 while compressing the rest. Verified
            on 3 models: +0% to +3% PPL improvement at 1.75 MB cost.
            No reason to disable — it's strictly better.
        aggressive : bool
            Maximum memory savings (default False). Uses 4-bit KV with
            last 512 tokens at FP32. Ideal for very long context.
            At 128K context: 4.6 GB instead of 9.2 GB KV cache.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        self._path = path
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._n_threads = n_threads
        self._context_length = context_length
        self._progressive = progressive
        self._aggressive = aggressive

        if aggressive:
            # 4-bit KV + 512-token FP32 window: best memory/quality ratio.
            # Measured: same PPL as flat 4-bit, attention-aware precision.
            # TODO: add uniform_2b (kv_compress=3) for 48% more savings.
            k_win = 512
        elif progressive:
            k_win = 128
        else:
            k_win = 0

        self._kv_compress = kv_compress

        self._model = load_model(path)
        self._ctx = new_context(
            self._model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n_threads=n_threads,
            kv_compress=kv_compress,
            context_length=context_length,
            k_highres_window=k_win,
        )
        self._chat = True  # auto-wrap with chat template for instruct models
        self._lock = threading.Lock()

    # -- Chat template -----------------------------------------------------

    @staticmethod
    def _apply_chat_template(prompt: str) -> str:
        """Wrap a user prompt with a generic ChatML-style template.

        Works with SmolLM2, Llama 3.x Instruct, and most HuggingFace
        instruct models that use the ``<|im_start|>`` / ``<|begin_of_text|>``
        token convention. Simpler models may ignore the template tokens and
        still generate correctly.
        """
        return (
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    # -- Context manager ---------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    # -- Core API ----------------------------------------------------------

    def ask(self, prompt: str) -> str:
        """Send a prompt and return the full response as a string.

        Parameters
        ----------
        prompt : str
            The input prompt / question.

        Returns
        -------
        str
            The model's complete response.
        """
        self._ensure_open()
        lib = get_lib()
        import ctypes
        import sys

        if self._chat:
            prompt = self._apply_chat_template(prompt)

        with self._lock:
            ptr = lib.quant_ask(self._ctx, prompt.encode("utf-8"))

        if not ptr:
            return ""

        result = ctypes.cast(ptr, ctypes.c_char_p).value
        text = result.decode("utf-8", errors="replace") if result else ""

        # Free via the dylib's own free wrapper (added in v0.8.2). Falls back
        # to a leak if the loaded library is an older single-header that
        # doesn't export quant_free_string — preserves binary compat.
        if hasattr(lib, "quant_free_string"):
            lib.quant_free_string(ptr)

        return text

    def generate(self, prompt: str) -> Iterator[str]:
        """Stream tokens from a prompt. Yields token strings one at a time.

        Parameters
        ----------
        prompt : str
            The input prompt.

        Yields
        ------
        str
            Individual token strings as they are generated.

        Examples
        --------
        >>> for token in m.generate("Hello"):
        ...     print(token, end="", flush=True)
        """
        self._ensure_open()
        lib = get_lib()

        if self._chat:
            prompt = self._apply_chat_template(prompt)

        tokens = []
        done = threading.Event()
        error_box = [None]

        def _on_token(text_ptr, _user_data):
            if text_ptr:
                tokens.append(text_ptr.decode("utf-8", errors="replace"))

        # prevent GC of the callback during generation
        cb = ON_TOKEN_CB(_on_token)

        def _run():
            try:
                with self._lock:
                    lib.quant_generate(
                        self._ctx,
                        prompt.encode("utf-8"),
                        cb,
                        None,
                    )
            except Exception as e:
                error_box[0] = e
            finally:
                done.set()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        yielded = 0
        while not done.is_set() or yielded < len(tokens):
            if yielded < len(tokens):
                yield tokens[yielded]
                yielded += 1
            else:
                done.wait(timeout=0.01)

        # Drain remaining tokens
        while yielded < len(tokens):
            yield tokens[yielded]
            yielded += 1

        if error_box[0] is not None:
            raise error_box[0]

    def save_context(self, path: str) -> None:
        """Save the current KV cache to disk.

        Enables "read once, query forever": process a long document
        once (slow prefill), save the context, then reload instantly
        for follow-up questions without re-processing.

        Parameters
        ----------
        path : str
            File path to write the context to (.kv extension recommended).
        """
        self._ensure_open()
        lib = get_lib()
        rc = lib.quant_save_context(self._ctx, path.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"Failed to save context to {path}")

    def load_context(self, path: str) -> None:
        """Load a previously saved KV cache from disk.

        Restores the exact conversation state — the model can immediately
        answer follow-up questions about a previously processed document
        without re-reading it.

        Parameters
        ----------
        path : str
            Path to a context file saved by ``save_context``.
        """
        self._ensure_open()
        lib = get_lib()
        rc = lib.quant_load_context(self._ctx, path.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"Failed to load context from {path}")

    def close(self) -> None:
        """Release model and context resources.

        Safe to call multiple times. Called automatically when used
        as a context manager or when garbage collected.
        """
        if hasattr(self, "_ctx") and self._ctx:
            free_ctx(self._ctx)
            self._ctx = None
        if hasattr(self, "_model") and self._model:
            free_model(self._model)
            self._model = None

    # -- Properties --------------------------------------------------------

    @property
    def path(self) -> str:
        """Path to the loaded model file."""
        return self._path

    # -- Internals ---------------------------------------------------------

    def _ensure_open(self):
        if not getattr(self, "_ctx", None) or not getattr(self, "_model", None):
            raise RuntimeError("Model has been closed")

    def __repr__(self) -> str:
        state = "open" if getattr(self, "_ctx", None) else "closed"
        return f"quantcpp.Model({self._path!r}, state={state})"


def load(path: str, **kwargs) -> Model:
    """Shorthand for Model(path, **kwargs)."""
    return Model(path, **kwargs)


__all__ = ["Model", "load", "download", "__version__"]
