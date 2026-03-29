# TurboQuant.cpp — LLM Inference Engine with Extreme KV Cache Compression

We built an LLM inference engine from scratch in pure C. It runs Qwen3.5-0.8B at **14 tok/s on CPU** — 17x faster than PyTorch, and faster than PyTorch on Apple GPU.

## The Numbers

```
PyTorch (CPU):     0.8 tok/s
PyTorch (MPS GPU): 10  tok/s
TurboQuant (CPU):  14  tok/s  ← faster than GPU, no dependencies
```

Weight memory: 1.7 GB → **533 MB** with Q8 quantization.
KV cache: **7.5x compressed** with 99.4% attention accuracy.

## What It Does

One binary. Zero Python. Load a model, generate text:

```bash
./tq_run model.safetensors -t tokenizer.json -p "What is AI?" -j 4 -q
```

Output:
```
Artificial intelligence is a field of computer science...
100 tokens in 7.2s (13.9 tok/s, 4 threads, kv=uniform_4b)
```

## How

- Safetensors model loader with mmap (zero-copy)
- DeltaNet + Self-Attention hybrid forward pass (Qwen3.5 architecture)
- NEON-optimized matmul (4-accumulator, multi-threaded)
- Integer Q4×Q8 attention (2.9x faster than FP32)
- Q8 weight quantization (4x memory savings)
- HuggingFace BPE tokenizer (248K vocab)
- Streaming token output

Built on TurboQuant (ICLR 2026), QJL (AAAI 2025), PolarQuant (AISTATS 2026).
Architecture patterns from llama.cpp, vLLM, and ONNX.

8,500 lines of C. 70+ tests. Apache 2.0.

https://github.com/quantumaikr/TurboQuant.cpp
