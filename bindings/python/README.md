# quantcpp

Python bindings for [quant.cpp](https://github.com/quantumaikr/quant.cpp) -- a minimal C inference engine for local LLMs with KV cache compression.

## Installation

```bash
cd quant.cpp/bindings/python
pip install .
```

This compiles `quant.h` into a shared library automatically using your system C compiler (`cc`, `gcc`, or `clang`). No external dependencies required.

For development (editable install):

```bash
pip install -e .
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

### Basic question answering

```python
from quantcpp import Model

m = Model("model.gguf")
answer = m.ask("What is 2+2?")
print(answer)
```

### Streaming generation

```python
for token in m.generate("Once upon a time"):
    print(token, end="", flush=True)
```

### Context manager

```python
with Model("model.gguf") as m:
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

### `Model.ask(prompt) -> str`

Generate a complete response. Returns the full text.

### `Model.generate(prompt) -> Iterator[str]`

Stream tokens one at a time. Yields individual token strings.

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
