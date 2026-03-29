# TurboQuant.cpp — Work Breakdown Structure v0.3

**Version**: 0.3
**Date**: 2026-03-29
**Focus**: Bug fixes + measurable improvements only

---

## Phase 1: BUG-1 — Uniform Attention 구현

- [x] `src/core/tq_uniform.c` — `tq_uniform_4b_attention_ref()` 추가
  - [x] dequantize + dot product (reference path)
  - [x] query[d] × dequantized_key[d] 합산
- [x] `src/core/tq_uniform.c` — `tq_uniform_2b_attention_ref()` 추가
- [x] `src/core/tq_traits.c` — TQ_TRAITS 테이블에 attention 함수 등록
  - [x] `[TQ_TYPE_UNIFORM_4B].attention = tq_uniform_4b_attention_ref`
  - [x] `[TQ_TYPE_UNIFORM_2B].attention = tq_uniform_2b_attention_ref`
- [x] `tests/test_uniform.cpp` — attention 테스트 추가
  - [x] Uniform 4B attention vs FP32 dot product 정확도
  - [x] Uniform 2B attention 동작 검증

---

## Phase 2: BUG-2 — Progressive Recompression 수정

- [x] `src/cache/tq_progressive.c` — `compress_slot()` 수정
  - [x] warm_type 하드코딩 제거
  - [x] `TQ_TRAITS[stored_type].dequantize` 로 역양자화
  - [x] `TQ_TRAITS[cold_type].quantize` 로 재양자화
- [x] `tests/test_progressive.cpp` — Tier 1→2 재압축 테스트
  - [x] warm_type=POLAR_4B, cold_type=POLAR_3B 조합 테스트
  - [x] 재압축 후 역양자화 → MSE 검증

---

## Phase 3: BUG-3 — Value Cache 저장

- [x] `src/cache/tq_paged_cache.c` — value 저장 구현
  - [x] `tq_head_cache_t`에 value blocks 배열 추가
  - [x] `tq_cache_append()`에서 value를 uniform_4b로 양자화하여 저장
  - [x] `tq_cache_free()`에서 value blocks 해제
- [x] `include/turboquant/turboquant.h` — `tq_cache_get_value()` API 추가
- [x] `tests/test_paged_cache.cpp` — value 저장/조회 테스트
  - [x] append key+value → get value → dequantize → compare

---

## Phase 4: All-Type Attention Test

- [x] `tests/test_attention_all_types.cpp` — 모든 타입 통합 테스트
  - [x] 7개 타입 모두 `tq_attention()` 호출 → TQ_OK 반환
  - [x] 7개 타입 모두 FP32 대비 cosine > 0.8 (abs for QJL sign-flip)
  - [x] edge case: seq_len=1, head_dim=128

---

## 완료 기준

- [x] BUG-1 수정: Uniform attention 동작 + 테스트 통과
- [x] BUG-2 수정: Progressive recompression 동작 + 테스트 통과
- [x] BUG-3 수정: Value cache 저장 동작 + 테스트 통과
- [x] 모든 타입 attention 테스트 통과
- [x] 기존 10개 + 신규 테스트 전체 통과
- [x] score.sh ≥ 0.99 유지
