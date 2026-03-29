# TurboQuant.cpp — Product Requirements Document v0.7

**Version**: 0.7
**Date**: 2026-03-29
**Focus**: CPU/Metal 속도 가속 — 정수 도메인 Attention ("Never Dequantize")

---

## 1. Problem

현재 양자화 attention이 FP32보다 **2배 느림** (0.49x):
```
FP32:      query × key = dot (1-pass, 217μs @seq=2048)
Quantized: load → dequantize → query × deq_key (2-pass, 445μs)
```

Google은 H100에서 8x 빠르다고 주장. GPU는 메모리 대역폭 병목이라 양자화가 자동으로 빨라짐.
CPU에서도 빠르게 만들려면 **dequantize를 제거**해야 함.

## 2. Solution: llama.cpp의 Integer-Domain Dot Product

핵심 인사이트 (llama.cpp `ggml_vec_dot_q4_0_q8_0`):

```
기존 (느림):
  FP32 query × [dequantize(Q4 key) → FP32] = FP32 dot

개선 (빠름):
  [quantize(FP32 query) → Q8] × Q4 key = INT32 dot × scale

  쿼리는 1번만 양자화 (비용 무시)
  seq_len개 키와의 내적은 모두 정수 도메인
```

**왜 빠른가:**
1. **데이터 4x 작음**: Q4 key = 0.5B/element vs FP32 = 4B/element → L1 캐시 적재율 8x
2. **정수 내적이 FP 내적만큼 빠름**: ARM `vdotq_s32` = 16 int8×int8/cycle = `vfmaq_f32`와 동일 throughput
3. **dequantize 제거**: 중간 FP32 버퍼 할당/기록 없음

**예상 속도**: FP32 대비 **3-5x 빠름** (seq_len ≥ 512)

## 3. Requirements

### FR-V7-1: Query Q8 Quantization

```c
void tq_quantize_query_q8(const float* query, int8_t* q8_out,
                           float* scale_out, float* sum_out, int n);
```
- `scale = max(|query[d]|) / 127`
- `q8[d] = round(query[d] / scale)`
- `sum = Σ query[d]` (zero-point 보정용)
- NEON 최적화: `vmaxvq_f32` → `vcvtnq_s32_f32` → `vqmovn_s32/s16`

### FR-V7-2: Integer-Domain Attention (Uniform 4-bit)

```c
void tq_uniform_4b_attention_int(const int8_t* q8, float q_scale, float q_sum,
                                  const void* kv, float* scores,
                                  int seq_len, int head_dim);
```
- Q4 nibble unpack: `vand(packed, 0x0F)`, `vshr(packed, 4)`
- Integer dot: `vdotq_s32(acc, q4_vals, q8_vals)` (16 products/instruction)
- Scale once: `scores[s] = (float)isum * k_scale * q_scale + k_zp * q_sum`
- NEON: `vdotq_s32` (ARM v8.2+ DOTPROD, all Apple M-series)

### FR-V7-3: NEON Optimized Integer Attention

`src/backend/cpu/tq_neon.c`에 구현:
- `vld1q_u8` → 16 packed bytes = 32 nibbles 로드
- `vandq_u8` + `vshrq_n_u8` → nibble unpack to int8
- `vdotq_s32` → 16 int8×int8 products per cycle
- `vaddvq_s32` → horizontal reduction
- 기존 `tq_uniform_4b_attention_neon()`을 교체

### FR-V7-4: A/B Speed Benchmark

`bench/speed_int_vs_float.cpp`:
- FP32 dot product vs Integer Q4×Q8 dot product
- seq_len = 128, 512, 2048, 8192
- head_dim = 128, 256
- 측정: μs/query, speedup ratio

## 4. Success Criteria

| 지표 | v0.6 현재 | v0.7 목표 |
|------|----------|----------|
| seq=2048 attention | 445μs (0.49x) | < 100μs (**2x faster than FP32**) |
| seq=8192 attention | 1785μs | < 400μs (**2x faster**) |
| 정확도 (cosine) | 0.994 | > 0.99 유지 |
| 모든 테스트 통과 | 16 | 17+ |
