# Reddit r/LocalLLaMA — quantcpp v0.9.2 + `pip install` (EN)

**Suggested title:** `[Project] quantcpp — "The SQLite of LLMs". Add AI to any C project with one 16K-line file. Now on PyPI.`

**Suggested flair:** `Resources` or `Other`

---

## Body

We just shipped **quantcpp 0.9.2** — a single-header C inference engine that you can `pip install` and use in 3 lines:

```bash
pip install quantcpp
```

```python
from quantcpp import Model

m = Model.from_pretrained("Llama-3.2-1B")  # auto-downloads ~750MB GGUF
print(m.ask("What is gravity?"))
```

No API key, no GPU, no configuration. Model downloads once, cached locally. KV cache compression is on by default (4-bit, ~4x memory reduction). Pre-built wheels for Linux x86_64/aarch64, macOS arm64 (Python 3.9–3.13).

### What it is

- **Single header (`quant.h`, ~16K LOC, ~646 KB)** — drop one file into any C project, no CMake, no submodule.
- **7 KV cache quantization types** in one engine, all reproducible from public papers (TurboQuant, PolarQuant, QJL).
- **Pure C, zero deps** — runs everywhere a C compiler runs (iOS, Android, WASM, microcontrollers, MSVC).
- **Multi-channel distribution**: PyPI, GGUF integration, llama.cpp PR draft (filed separately), single-header drop-in.

### Headline result (Llama 3.2 3B, M-series, CPU-only, 957-token PPL eval, 3-run avg)

| KV type | tok/s | vs FP32 | PPL | ΔPPL | Compression |
|---|---:|---:|---:|---:|---:|
| FP32 | 17.93 | baseline | 13.56 | — | 1× |
| **turbo_kv_4b** | 18.13 | **+1.1%** ✅ | 14.08 | +3.8% | **7.1×** |
| turbo_kv_5b_fast 🆕 | 17.53 | −2.2% | 13.65 | +0.7% | 3.76× |
| turbo_kv_5b | 16.93 | −5.6% | 13.65 | +0.7% | 5.8× |

The **`turbo_kv_4b`** path achieves fp32 *speed parity* at 7.1× KV compression on Apple Silicon. The kernel that gets us there is a single NEON instruction (`vqtbl1q_s8`) doing a 16-entry codebook lookup — Round 10 of our public Karpathy-loop log. v0.8.0 ports the same pattern to AVX2 (`_mm_shuffle_epi8`) for Linux/Windows x86-64.

### What we are NOT claiming

- We are **not faster than llama.cpp on GPU**. llama.cpp + Metal/CUDA wins production throughput by 5–10×. Our value is on CPU/embedded where dispatch overhead dominates GPU compute, and on the **research velocity** of porting new quant methods quickly.
- We are **not a llama.cpp replacement**. llama.cpp supports 100+ archs, we support 7 (the ones we benchmark).
- The Python bindings in 0.8.x default to **`kv_compress=0`** (no KV compression). The CLI binary works with all KV types; bringing them to the bindings is tracked for v0.8.2 (regenerated single-header). The `pip install` package will load + generate; KV compression comes next release.

### Honesty track record

This is the project's **6th self-correction** in the 0.6.x → 0.8.x series. We caught both v0.8.0 Python binding bugs (a default-path abort and a cross-heap `libc.free()`) within hours of publishing by running an end-user simulation in a clean venv. v0.8.1 is the hotfix. PyPI 0.8.0 is being yanked.

We treat retractions as the project's primary trust asset and log them in the [CHANGELOG](https://github.com/quantumaikr/quant.cpp/blob/main/CHANGELOG.md) the same way we log features.

### Links

- **PyPI**: https://pypi.org/project/quantcpp/
- **GitHub**: https://github.com/quantumaikr/quant.cpp
- **Reproduction harness**: 11 Karpathy-loop rounds documented at `bench/results/turboquant_reproduction.md`
- **Karpathy loop scoring**: 6 dimensions including a 10-year-position guard (single-header LOC, zero-deps, papers ported, honest correction count) — failures break CI

### What we'd love feedback on

1. If you `pip install quantcpp` and `Model("your.gguf").ask("hi")` doesn't return cleanly on **your** OS / Python version, please open an issue with the trace. The wheel matrix is Linux x86_64/aarch64 + macOS arm64 today; everything else uses sdist (source compile).
2. The llama.cpp PR draft for `TQ_TURBO_KV_4B` is at `docs/pr/2026-04-09-llama-cpp-pr-draft.md`. If anyone with llama.cpp ggml internals experience wants to co-author the actual port, we'd welcome the help.

---

## Pre-post checklist (for the user posting)

- [ ] Yank PyPI 0.8.0 first (https://pypi.org/manage/project/quantcpp/release/0.8.0/) → Options → Yank → reason text
- [ ] Confirm `pip install quantcpp` resolves to **0.8.1** in a fresh venv
- [ ] Test the code snippet (`Model("file.gguf").ask("...")`) one more time on the target platform you'll mention
- [ ] Decide on the title flair — `Resources` is least likely to be auto-removed by mods
- [ ] Pin a comment with the link to the v0.8.1 release notes
- [ ] Be ready to respond to "how does this compare to llama.cpp" — the answer is in the "What we are NOT claiming" section above

## Notes for response strategy

- If anyone says "you're slower than llama.cpp" → agree, point to "What we are NOT claiming"
- If anyone says "this is just llama.cpp + a Python binding" → point to single-header (drop into any C project, no submodule, no CMake), 7 KV quant types, research velocity (KIVI/HIGGS/etc.)
- If Tim Dettmers, Amir Zandieh, or other quant-research authors comment → engage thoughtfully, they're the actual target audience for the research-velocity pillar
- If a downvote brigade hits → leave the post up, do not delete. The honesty track record is the moat; deletion erodes it.
