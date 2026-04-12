# quantcpp

Python bindings for [quant.cpp](https://github.com/quantumaikr/quant.cpp) -- a minimal C inference engine for local LLMs with KV cache compression.

## Installation

```bash
pip install quantcpp
```

Pre-built wheels are published for Linux (x86_64, aarch64), macOS (Intel + Apple Silicon), and Windows (x64). On other platforms pip falls back to the source distribution and compiles `quant.h` automatically using your system C compiler — no external dependencies.

### From source (dev tree)

```bash
cd quant.cpp/bindings/python
pip install .          # build + install
pip install -e .       # editable / development install
```

To point at a pre-built library instead:

```bash
export QUANTCPP_LIB=/path/to/libquant.dylib
pip install .
```

## Requirements

- Python >= 3.8
- A C compiler (cc, gcc, or clang)
- The quant.cpp repository (for `quant.h`)

## Usage

### Quick start (auto-download)

```python
from quantcpp import Model

m = Model.from_pretrained("Phi-3.5-mini")  # ~2.4 GB, downloaded once and cached
print(m.ask("What is 2+2?"))
```

`from_pretrained` accepts any name from `quantcpp.available_models()`.
**Phi-3.5-mini** is the recommended default — 3.8B params with the smallest
vocab (32K) in the registry, which makes the per-token `lm_head` matmul
the fastest of any model we ship. Other ready-to-use names:

- `SmolLM2-1.7B` — lightweight all-rounder (1.7 GB, vocab 49K)
- `Llama-3.2-1B` — smallest download (750 MB) but slower at inference
- `SmolLM2-135M` — 138 MB demo model, low quality
- `Qwen3.5-0.8B`

You can also load any local GGUF file directly:

```python
m = Model("model.gguf")
print(m.ask("What is 2+2?"))
```

### Streaming generation

```python
for token in m.generate("Once upon a time"):
    print(token, end="", flush=True)
```

### Multi-turn chat with KV cache reuse

```python
m = Model.from_pretrained("Phi-3.5-mini")
history = ""
while True:
    user = input("\nYou: ")
    history += f"<|user|>\n{user}<|end|>\n<|assistant|>\n"
    print("AI: ", end="", flush=True)
    reply = ""
    for tok in m.chat(history):
        print(tok, end="", flush=True)
        reply += tok
    history += reply + "<|end|>\n"
```

`m.chat()` reuses the KV cache across turns — turn N's prefill cost is
O(new tokens), not O(history). Catch `quantcpp.ChatContextOverflow` if
the conversation exceeds the model's context window.

### Context manager

```python
with Model.from_pretrained("Phi-3.5-mini") as m:
    print(m.ask("Explain gravity in one sentence"))
```

### Configuration

```python
m = Model(
    "model.gguf",
    temperature=0.5,      # Lower = more deterministic
    top_p=0.9,            # Nucleus sampling
    max_tokens=512,       # Max tokens per generation
    n_threads=8,          # CPU threads
    kv_compress=2,        # 0=off, 1=4-bit K+V, 2=delta+3-bit
)
```

### Convenience loader

```python
from quantcpp import load

m = load("model.gguf", kv_compress=2)
print(m.ask("Hello!"))
```

## API Reference

### `Model(path, *, temperature=0.7, top_p=0.9, max_tokens=256, n_threads=4, kv_compress=1)`

Load a GGUF model file and create an inference context.

- `path` -- Path to a `.gguf` model file.
- `temperature` -- Sampling temperature (0.0 = greedy).
- `top_p` -- Nucleus sampling threshold.
- `max_tokens` -- Maximum tokens per generation.
- `n_threads` -- CPU thread count.
- `kv_compress` -- KV cache compression mode (0=off, 1=4-bit, 2=delta+3-bit).

### `Model.from_pretrained(name) -> Model`

Download a registered model from HuggingFace (cached at
`~/.cache/quantcpp/`) and return an open Model. See
`quantcpp.available_models()` for the registry.

### `Model.ask(prompt) -> str`

Generate a complete response. Returns the full text.

### `Model.generate(prompt) -> Iterator[str]`

Stream tokens one at a time. Yields individual token strings.

### `Model.chat(prompt) -> Iterator[str]`

Stream tokens with KV cache reuse across calls — turn N pays only for
the new bytes since turn N-1. Pass `prompt=None` (or call
`Model.reset_chat()`) to start a fresh session. Raises
`quantcpp.ChatContextOverflow` when the history exceeds the model's
context window (the C side has already auto-reset by then).

### `Model.close()`

Release resources. Called automatically via `with` or garbage collection.

### `Model.path -> str`

The path to the loaded model file (read-only property).

## Library search order

The package looks for the compiled shared library in this order:

1. `QUANTCPP_LIB` environment variable
2. Installed alongside the Python package (normal `pip install`)
3. `build/` relative to the project root (development)
4. System library path

## Running tests

```bash
cd bindings/python
python -m pytest tests/
```
