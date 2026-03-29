# TurboQuant.cpp — Work Breakdown Structure v0.6

**Version**: 0.6
**Date**: 2026-03-29
**Focus**: RHT, K/V 비대칭, Mixed Precision

---

## Phase 1: K/V 비대칭 API (FR-V6-1)

- [x] `include/turboquant/turboquant.h` — 새 API 선언
  - [x] `tq_quantize_kv()` — key_type + value_type 별도 지정
  - [x] `tq_quantize_kv_size()` — key/value 각각 크기 반환
- [x] `src/core/tq_context.c` — `tq_quantize_kv()` 구현
  - [x] keys를 key_type으로, values를 value_type으로 독립 양자화
- [x] `tests/test_kv_asymmetric.cpp` — K/V 비대칭 테스트
  - [x] K=uniform_4b, V=uniform_2b 조합 동작 검증
  - [x] K 품질(cosine > 0.99) + V 품질(cosine > 0.85) 각각 확인
  - [x] 평균 비트 계산: (4.25 + 2.25) / 2 = 3.25 bit
- [x] A/B 측정: K4V2 vs K4V4 vs K2V2 비교

---

## Phase 2: Random Hadamard Transform (FR-V6-2)

- [x] `src/core/tq_rht.c` — Walsh-Hadamard 변환 구현
  - [x] `tq_rht_transform(float* data, int n, uint32_t seed)` — in-place O(n log n)
    - [x] Walsh-Hadamard: butterfly 패턴 (for log2(n) stages)
    - [x] Random sign flip: seed 기반 의사 랜덤 부호 곱
    - [x] 스케일링: 1/sqrt(n)
  - [x] `tq_rht_inverse(float* data, int n, uint32_t seed)` — 역변환
    - [x] 동일 부호 플립 + 동일 Hadamard (자기 역원) + 스케일링
- [x] `include/turboquant/turboquant.h` — RHT API 선언
  - [x] `tq_quantize_keys_rht()` — RHT + 양자화 파이프라인
  - [x] `tq_dequantize_keys_rht()` — 역양자화 + 역RHT
- [x] `src/core/tq_context.c` — RHT 파이프라인 통합
  - [x] `tq_quantize_keys_rht()`: RHT → quantize → store
  - [x] `tq_dequantize_keys_rht()`: load → dequantize → inverse RHT
- [x] `tests/test_rht.cpp` — RHT 테스트
  - [x] RHT → inverse RHT = identity (왕복 오차 < 1e-6)
  - [x] RHT + uniform_4b MSE < raw uniform_4b MSE
  - [x] RHT + uniform_4b cosine > raw uniform_4b cosine
- [x] A/B 측정: with RHT vs without RHT on real model data

---

## Phase 3: Mixed Precision Outlier (FR-V6-3)

- [ ] `include/turboquant/tq_types.h` — `TQ_TYPE_MIXED_3B8` 타입 추가
  - [ ] `block_tq_mixed_3b8` 구조체 정의
  - [ ] TQ_CHECK_SIZE 검증
- [ ] `src/core/tq_mixed.c` — Mixed precision 구현
  - [ ] `tq_mixed_3b8_quantize_ref()` — 아웃라이어 탐지 + 분리 양자화
  - [ ] `tq_mixed_3b8_dequantize_ref()` — 아웃라이어 복원 + base 복원
  - [ ] `tq_mixed_3b8_attention_ref()` — 직접 attention
- [ ] `src/core/tq_traits.c` — TQ_TRAITS 테이블에 등록
- [ ] `tests/test_mixed.cpp` — Mixed precision 테스트
  - [ ] 아웃라이어 탐지 정확성
  - [ ] Roundtrip MSE (target: < 0.005)
  - [ ] Attention cosine (target: > 0.97)
- [ ] A/B 측정: mixed_3b8 vs uniform_4b vs uniform_2b

---

## Phase 4: 통합 벤치마크 + 문서

- [ ] `bench/ab_v06_comparison.cpp` — v0.5 vs v0.6 A/B 비교
  - [ ] 모든 타입 + 새 타입 (mixed_3b8, K4V2+RHT) 비교
  - [ ] 실제 모델 데이터 기반
- [ ] README 업데이트 — 커뮤니티 검증 결과 반영
  - [ ] "커뮤니티 추천: uniform_4b (MSE-only)" 명시
  - [ ] K/V 비대칭 사용법
  - [ ] RHT 활성화 방법
- [ ] `docs/real_model_results.md` — v0.6 결과 추가

---

## 완료 기준

- [ ] K/V 비대칭 API 동작 + 테스트 통과
- [ ] RHT 변환 왕복 오차 < 1e-6
- [ ] RHT + uniform_4b MSE 감소 (A/B 증거)
- [ ] Mixed precision: cosine > 0.97 at ~3.6 bit
- [ ] 14+ 테스트 전체 통과
- [ ] score.sh ≥ 0.99 유지
