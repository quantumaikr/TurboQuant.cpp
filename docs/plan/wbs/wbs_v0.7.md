# TurboQuant.cpp — Work Breakdown Structure v0.7

**Version**: 0.7
**Date**: 2026-03-29
**Focus**: Integer-domain attention — FP32보다 빠른 양자화 attention

---

## Phase 1: Query Q8 Quantization

- [ ] `src/core/tq_uniform.c` — `tq_quantize_query_q8()` 구현
  - [ ] float query → int8 query 변환 (scale = amax/127)
  - [ ] query_sum 계산 (zero-point 보정용)
  - [ ] head_dim=128, 256 지원
- [ ] `src/backend/cpu/tq_neon.c` — `tq_quantize_query_q8_neon()` NEON 최적화
  - [ ] `vld1q_f32` → `vabsq_f32` → `vmaxvq_f32` 로 amax
  - [ ] `vcvtnq_s32_f32` → `vqmovn_s32` → `vqmovn_s16` 로 int8 변환
- [ ] `include/turboquant/turboquant.h` — API 선언

---

## Phase 2: Integer-Domain Attention (핵심)

- [ ] `src/core/tq_uniform.c` — `tq_uniform_4b_attention_int_ref()` 레퍼런스 구현
  - [ ] Q4 nibble unpack: `(byte & 0x0F)`, `(byte >> 4)`
  - [ ] Q4 × Q8 정수 내적: `isum += q4_val * q8_val`
  - [ ] 최종 스케일링: `score = isum * k_scale * q_scale + k_zp * q_sum`
- [ ] `src/backend/cpu/tq_neon.c` — `tq_uniform_4b_attention_int_neon()` NEON 최적화
  - [ ] `vld1q_u8` → 16 packed bytes 로드
  - [ ] `vandq_u8(packed, 0x0F)` + `vshrq_n_u8(packed, 4)` nibble unpack
  - [ ] `vdotq_s32(acc, q4, q8)` — 16 int8×int8 per instruction
  - [ ] `vaddvq_s32(acc)` — horizontal sum
  - [ ] 기존 `tq_uniform_4b_attention_neon()` (dequant+dot) 교체
- [ ] `src/core/tq_traits.c` — attention 함수 포인터 업데이트
  - [ ] Wrapper: query Q8 양자화 → integer attention 호출

---

## Phase 3: A/B Speed Benchmark

- [ ] `bench/speed_int_vs_float.c` — 속도 비교 벤치마크
  - [ ] FP32 dot product attention (baseline)
  - [ ] Dequant+FP32 dot (현재 방식, v0.6)
  - [ ] Integer Q4×Q8 dot (v0.7 신규)
  - [ ] seq_len = 128, 512, 2048, 8192
  - [ ] head_dim = 128, 256
  - [ ] 출력: μs/query, speedup ratio
- [ ] 정확도 검증: integer attention vs FP32 cosine > 0.99

---

## Phase 4: 테스트 + 통합

- [ ] `tests/test_int_attention.cpp` — 정수 attention 테스트
  - [ ] Q8 양자화 왕복: float → Q8 → float, 오차 < 0.01
  - [ ] Integer attention vs FP32 attention cosine > 0.99
  - [ ] Integer attention vs dequant attention 결과 일치
  - [ ] Edge case: head_dim=2, seq_len=1
- [ ] 기존 16개 테스트 전체 통과 확인
- [ ] score.sh ≥ 0.99 유지

---

## 완료 기준

- [ ] seq=2048 attention: < 100μs (현재 445μs, FP32=217μs)
- [ ] FP32 대비 **2x+ 빠름** (목표: 3-5x)
- [ ] 정확도: cosine > 0.99 유지
- [ ] 17+ 테스트 전체 통과
