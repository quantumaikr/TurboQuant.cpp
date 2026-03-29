# TurboQuant.cpp — Work Breakdown Structure v0.5

**Version**: 0.5
**Date**: 2026-03-29
**Focus**: 극한 품질 최적화 — 모든 변경을 A/B 테스트로 측정

---

## Phase 1: 1줄 수정 고임팩트 (ALG-1 + ALG-2)

### 1.1 PolarQuant 각도 범위 [0, 2π]

- [ ] `src/core/tq_polar.c` — quantize 함수에서 atan2 결과 보정
  - [ ] `if (t < 0.0f) t += 2.0f * TQ_PI;` 추가
- [ ] A/B 측정: `build/real_model_validation` 실행
  - [ ] Before polar_4b MSE: ____
  - [ ] After polar_4b MSE: ____
  - [ ] 개선율: ____%

### 1.2 중심 보정 (+0.5 offset)

- [ ] `src/core/tq_polar.c` — dequantize 함수에서 +0.5 추가
  - [ ] theta: `tscale * ((float)tq + 0.5f) + tmn`
  - [ ] radius: `rscale * ((float)rq + 0.5f) + rmn`
- [ ] `src/core/tq_polar.c` — attention 함수의 LUT에도 +0.5 반영
  - [ ] `cos_lut[q] = cosf(tscale * ((float)q + 0.5f) + tmn)`
- [ ] `src/core/tq_uniform.c` — uniform dequantize에도 +0.5 추가
  - [ ] `val = scale * ((float)q + 0.5f) + zero_point`
- [ ] A/B 측정
  - [ ] Before uniform_4b MSE: ____
  - [ ] After uniform_4b MSE: ____

### 1.3 검증

- [ ] 모든 기존 테스트 통과 (테스트 tolerance 조정 필요할 수 있음)
- [ ] `build/tq_quality` 결과 개선 확인
- [ ] `build/ab_test` 결과 개선 확인

---

## Phase 2: QJL 이중 스트림 (ALG-3)

### 2.1 양자화 분리

- [ ] `src/core/tq_qjl.c` — `tq_qjl_quantize_ref()` 수정
  - [ ] 아웃라이어 차원에 대해 별도 hash 계산 (outlier_hash)
  - [ ] 인라이어 = 전체 투영에서 아웃라이어 투영을 뺀 것
  - [ ] `block_tq_qjl`에 outlier hash 추가 (또는 기존 hash와 분리)

### 2.2 어텐션 이중 스트림

- [ ] `src/core/tq_qjl.c` — `tq_qjl_attention_ref()` 수정
  - [ ] Query sketch 계산
  - [ ] Query outlier sketch 계산 (outlier 차원만)
  - [ ] Inlier sketch = query_sketch - query_outlier_sketch
  - [ ] Inlier score = sqrt(π/2)/S × norm_inlier × (S - 2×hamming_inlier)
  - [ ] Outlier score = sqrt(π/2)/S_out × norm_outlier × (S_out - 2×hamming_outlier)
  - [ ] Total = inlier_score + outlier_score

### 2.3 검증

- [ ] A/B 측정: QJL cosine on real model data
  - [ ] Before: ____
  - [ ] After: ____
- [ ] 모든 QJL 테스트 통과

---

## Phase 3: Value Per-Tile Scaling (ALG-5)

### 3.1 타일 구조 추가

- [ ] `include/turboquant/tq_types.h` — `block_tq_value_tiled` 구조체 정의
  - [ ] 4개 타일 × (scale + zero_point) = 16 bytes metadata
  - [ ] qs[TQ_BK/2] = 64 bytes data
  - [ ] 총 80 bytes (기존 68 bytes 대비 12 bytes 증가)

### 3.2 타일 양자화 구현

- [ ] `src/core/tq_value_quant.c` — `tq_value_quantize_tiled()` 구현
  - [ ] 128 요소를 32개씩 4개 타일로 분할
  - [ ] 각 타일마다 독립 min/max → scale/zero_point
  - [ ] 4-bit 양자화 + 패킹

### 3.3 검증

- [ ] A/B 측정: Value roundtrip MSE
  - [ ] Per-block MSE: ____
  - [ ] Per-tile MSE: ____
  - [ ] 개선율: ____%

---

## Phase 4: 통합 + 최종 벤치마크

### 4.1 Key/Value 분리 설정

- [ ] `include/turboquant/tq_types.h` — progressive config 확장
  - [ ] `key_warm_type`, `value_warm_type` 분리
  - [ ] `tq_progressive_default_config()` 업데이트

### 4.2 최종 A/B 비교 벤치마크

- [ ] `bench/ab_comparison_v05.cpp` — v0.4 vs v0.5 비교 벤치마크
  - [ ] 모든 타입 × 실제 모델 데이터
  - [ ] Before/After MSE, Cosine 비교 테이블
  - [ ] 개선율 요약

### 4.3 문서 업데이트

- [ ] `docs/real_model_results.md` — v0.5 결과 추가
- [ ] README — 최신 수치 반영

---

## 완료 기준

- [ ] polar_4b real cosine > 0.85 (현재 0.786)
- [ ] QJL real cosine > 0.90 (현재 0.857)
- [ ] turbo_3b real cosine > 0.95 (현재 0.939)
- [ ] uniform_4b real MSE < 0.002 (현재 0.0025)
- [ ] 모든 개선에 A/B 측정 수치 기록
- [ ] 13+ C++ 테스트 전체 통과
- [ ] score.sh ≥ 0.99 유지
