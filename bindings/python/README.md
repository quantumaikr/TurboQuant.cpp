# quantcpp

Python bindings for [quant.cpp](https://github.com/hunscompany/quant.cpp) -- a minimal C inference engine for local LLMs with KV cache compression.

## Installation

```bash
# Build the shared library first
cd /path/to/quant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build -j$(nproc)

# Install the Python package
cd bindings/python
pip install .
```

Or set the library path explicitly:

```bash
export TURBOQUANT_LIB=/path/to/build/libturboquant.dylib
pip install .
```

## Usage

### Basic generation

```python
from quantcpp import Model

m = Model("model.gguf")
text = m.ask("What is 2+2?")
print(text)
```

### KV cache compression

```python
m = Model("model.gguf", kv_compress="4bit")
text = m.ask("Explain quantum computing")
```

Available compression modes: `"1bit"`, `"2bit"`, `"3bit"` (default), `"4bit"`, `"polar3"`, `"polar4"`, `"qjl"`, `"turbo3"`, `"turbo4"`, `"uniform2"`, `"uniform3"`, `"uniform4"`, `"none"`.

### Streaming

```python
for token in m.generate("Once upon a time"):
    print(token, end="", flush=True)
```

### Chat mode

```python
text = m.chat("What is the capital of France?")
print(text)
```

### Raw completion (no chat template)

```python
text = m.complete("The quick brown fox", max_tokens=64)
print(text)
```

### Context manager

```python
with Model("model.gguf") as m:
    print(m.ask("Hello!"))
```

## API Reference

### `Model(path, kv_compress=None, n_threads=0)`

Load a GGUF or TQM model file.

- `path` -- Path to model file.
- `kv_compress` -- KV cache compression mode (see above).
- `n_threads` -- CPU thread count (0 = auto).

### `Model.ask(prompt, *, max_tokens=512, temperature=0.6, top_p=0.9)`

Chat-formatted generation. Returns the full response string.

### `Model.chat(message, **kwargs)`

Alias for `ask()`.

### `Model.generate(prompt, *, max_tokens=512, temperature=0.6, top_p=0.9, chat=False)`

Streaming generation. Yields token strings. Set `chat=True` to apply chat template.

### `Model.complete(prompt, **kwargs)`

Raw text completion without chat template.

### `Model.close()`

Release model resources. Called automatically on garbage collection or context exit.

## Library search order

1. `TURBOQUANT_LIB` environment variable
2. Same directory as the Python package
3. `build/` relative to the project root
4. System library path
