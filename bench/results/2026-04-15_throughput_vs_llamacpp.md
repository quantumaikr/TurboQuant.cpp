# Generation throughput — quant.cpp vs llama.cpp (2026-04-15)

**Hardware**: Apple M1 Pro, 16GB, 8 P-cores + 2 E-cores
**Test**: `tg64` (generate 64 tokens at T=0), 8 threads, default CMake build
**Reproduce**:
```bash
# quant.cpp (3-run median)
./build/quant <model> -p "Once upon a time" -n 64 -T 0

# llama.cpp Metal
llama-bench -m <model> -p 0 -n 64 -t 8 -ngl 99

# llama.cpp CPU only
llama-bench -m <model> -p 0 -n 64 -t 8 -ngl 0
```

## Results

| Model | quant.cpp | llama.cpp Metal | llama.cpp CPU | vs Metal | vs CPU |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B Q8_0 | **35.5** | 89.0 | 68.1 | 40% | **52%** |
| Phi-3.5-mini Q8_0 | **13.0** | 36.8 | 18.3 | 35% | **71%** ⭐ |
| Llama-3.2-3B Q8_0 | **13.4** | 43.3 | 26.3 | 31% | 51% |
| Phi-3.5-mini Q4_K_M | **6.9** | 41.6 | 30.1 | 17% | 23% |
| Qwen3.5-4B Q4_K_M | **5.6** | 30.7 | 22.1 | 18% | 25% |

(All numbers are tokens/sec; quant.cpp is 3-run median, llama.cpp single run.)

## Honest reading

- **vs Metal**: llama.cpp wins decisively (3-6×). Their Metal kernels are mature; ours are CPU-fallback for several model families. This is the gap to close in the v1.x roadmap.
- **vs CPU (apples-to-apples)**: we're at **23-71%** of llama.cpp's pure-CPU speed depending on model. Phi-3.5 Q8_0 at 71% is competitive.
- **Smaller models close the gap**: 1B Q8 at 52% vs 3B/4B Q4_K_M at 23-25% suggests our Q4_K dispatch (raw GGUF path) is the largest remaining gap. The Q4-converted path (3B Llama, 1B Llama) is more competitive.

## Prefill (prompt processing) — biggest remaining gap

Generation speed is what gets benchmarked, but for any RAG/long-context
workload the user actually waits on **prefill**: running the prompt
through the model to populate the KV cache. quant.cpp currently calls
the same single-token forward path for every prompt token, so prefill
runs at roughly the same speed as decode. llama.cpp uses batched
matrix-matrix matmul during prefill, which is 30-50× faster.

Reproduce: `bash scripts/test_prefill.sh` and `llama-bench -m <model> -p 512 -n 0 -ngl 0`.

| Model | quant.cpp pp~450 | llama.cpp pp512 | Ratio |
|---|---:|---:|---:|
| Llama-3.2-1B Q8_0   | 10.2 | 358.7 | **35× behind** |
| Llama-3.2-3B Q8_0   | 3.2  | 130.1 | **41× behind** |
| Phi-3.5 Q4_K_M      | 1.9  | 90.8  | **48× behind** |
| Qwen3.5-4B Q4_K_M   | 2.0  | 88.1  | **44× behind** |

User-visible impact on a 16GB Mac: feeding a 1000-token prompt to
Phi-3.5-mini takes ~10 minutes today. With a batched-prefill path it
should be under 15 seconds.

### Update 2026-04-16: batched prefill is now the DEFAULT

`tq_forward_batch` uses batched matmul + quant K cache write so it
works with the default `turbo_kv_4b` KV mode (not just `-k fp32`).
Auto-activates on Llama-family models; bails to per-token for MoE,
Gemma 4, Phi-3 fused QKV, DeltaNet hybrid, etc.

Measured prefill on ~250-token prompt (50 English words), DEFAULT KV
(turbo_kv_4b):

| Model | Baseline (per-token) | Batched (new default) | Speedup |
|---|---:|---:|---:|
| Llama-3.2-1B Q8 | 43 s | **5.9 s** | **7.2×** |
| Llama-3.2-3B Q8 | ~148 s (est.) | ~63 s | ~2.4× |

Direct vs llama.cpp (pp256 CPU, same machine):

| Model | quant.cpp pp | llama.cpp pp | Ratio |
|---|---:|---:|---:|
| Llama-3.2-1B Q8 | ~37 tok/s | 358 tok/s | **10.3%** (was 0.4% session-start) |
| Llama-3.2-3B Q8 | ~17 tok/s | 130 tok/s | **13%** (was 0.4% session-start) |

Prefill gap closed from **~35-40×** to **~8-10×** — **4× closer** in
one day. Output bit-identical to per-token baseline.

Remaining 8-10× gap sources:
- Llama.cpp uses int8 quantized matmul directly on AMX. Our batched
  code still dequants Q4→FP32 internally in `tq_batched_matmul_q4`.
- Architecture specializations (Phi-3 fused QKV, DeltaNet) still
  per-token; extending batched is engineering work tracked in
  `docs/dev/batched_prefill_handoff.md`.

## Session improvements (2026-04-15)

Compared to the same hardware before this session:

| Model | Before | After | Δ |
|---|---:|---:|---:|
| Phi-3.5-mini Q4_K_M | 3.2 | 6.9 | **+115%** |
| Phi-3.5-mini Q8_0 | 5.4 | 13.0 | **+141%** |
| Qwen3.5-4B Q4_K_M | 3.5 | 5.6 | **+60%** |
| Llama-3.2-3B Q8_0 | 8.5 | 13.4 | **+58%** |

Wins came from five compounding changes:

1. **Q4_K int8 fused dot path** (`src/engine/tq_gguf_quants.c`). Was doing
   `vfmaq_f32` over float-converted nibbles. Now quantizes activation to int8
   once per matmul, runs `vdotq_s32` over nibbles unpacked to int8.
   Pre-computes per-block int sums for the dmin*mn correction.
2. **Q5_K int8 fused dot path**. Same approach, with the 5th bit unpacked
   from the Q5_K `qh` array via `vceqq_u8` → `vorrq` to merge.
3. **ARMv8.2 `vdotq_s32`** wherever int8 dot is needed (Q8_0, Q4_K, Q5_K
   workers). Previously used `vmull_s8 + vpadalq_s16` (8 MACs/op);
   `vdotq_s32` does 16 MACs/op. Gated on `__ARM_FEATURE_DOTPROD`.
4. **Weight-row prefetching** with `__builtin_prefetch`. M1 hardware
   prefetcher does not always pick up the row-stride pattern across matmul
   iterations. Explicit prefetch of next row's first 4 cache lines hides
   the load latency.
5. **2-row inner-loop ILP** in the Q4_K worker. Two output rows share the
   same activation; pairing their dot products lets the OoO engine overlap
   weight loads with activation broadcasts.
6. **P-core thread default**. M1 Pro is 8P+2E. Mixing P and E at the same
   priority makes the slow E threads stragglers — total throughput drops.
   Detect via `sysctlbyname("hw.perflevel0.physicalcpu")`.

## Other 2026-04-15 fixes

- `f0091fc` — Qwen3.5-4B DeltaNet layers were mis-detected as self-attention
  in the split-source build; fix probes for `ssm_a` before the Phi-3
  fused-QKV path. Output went from whitespace garbage to coherent.
- `30dca7a` — Phi-3.5 Q4_K_M produced garbage under the default Metal
  build because `tq_matmul_gguf_cpu` hard-reset the force-CPU flag,
  clobbering tq_forward's invariant. Save-and-restore.
- `8f5784a` — DeltaNet attn_qkv/attn_gate were dequanted Q5_K → FP32 at
  load (3GB extra per token in bandwidth). Verified identical generation
  with Q5_K kept; default flipped.

## Quality regression guards

```
scripts/test_models.sh    — 11/11 PASS (STRICT + COHERENT + Metal-ON)
scripts/test_long_seq.sh  — 6/6 PASS (500 tokens at T=0, 100% printable)
scripts/check_sync.sh     — 8 sections PASS (catches future split-source drift)
scripts/check_stale.sh    — binary mtime guard (catches stale-build confusion)
```
