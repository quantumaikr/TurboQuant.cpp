# TurboQuant.cpp — Product Requirements Document v0.3

**Version**: 0.3
**Date**: 2026-03-29
**Focus**: Correctness fixes + measurable performance gains

---

## 1. v0.3 Goal

v0.2 감사에서 발견된 **3개 실질적 버그**를 수정하고, **측정 가능한 성능 개선**을 달성한다.

### 발견된 문제

| # | 문제 | 심각도 | 영향 |
|---|------|--------|------|
| BUG-1 | Uniform 타입에 attention 함수 없음 (NULL) | **Critical** | `tq_attention(TQ_TYPE_UNIFORM_4B)` 호출 시 NOT_IMPL 반환 |
| BUG-2 | Progressive 재압축이 UNIFORM_4B 하드코딩 | **High** | 다른 warm_type 사용 시 잘못된 역양자화 |
| BUG-3 | Value 양자화가 paged cache에서 무시됨 | **High** | `tq_cache_append(key, value)` 시 value가 저장 안 됨 |
| PERF-1 | AVX2에 attention 구현 없음 | **Medium** | x86 시스템에서 2-4x 느림 |

### v0.3 Target

| 지표 | v0.2 현재 | v0.3 목표 |
|------|----------|----------|
| Uniform attention | NULL (미구현) | 동작 + 테스트 통과 |
| Progressive recompression | 하드코딩 버그 | 모든 타입에서 동작 |
| Value cache | 무시됨 | 저장 + 조회 동작 |
| 모든 타입 attention 테스트 | 3/7 타입 | 7/7 타입 |

---

## 2. Functional Requirements

### FR-V3-1: Uniform Attention 구현

- `tq_uniform_4b_attention_ref()` — dequantize+dot reference
- `tq_uniform_2b_attention_ref()` — 동일
- `TQ_TRAITS[TQ_TYPE_UNIFORM_4B].attention` 등록
- NEON 디스패치에서 기존 `tq_uniform_4b_attention_neon()` 연결

### FR-V3-2: Progressive Recompression 수정

- `compress_slot()`에서 warm_type을 하드코딩 대신 `tq_progressive_t.config.warm_type` 참조
- 역양자화 함수를 `TQ_TRAITS[warm_type].dequantize`로 조회
- 재양자화 함수를 `TQ_TRAITS[cold_type].quantize`로 조회
- Tier 1→2 전환 검증 테스트 추가

### FR-V3-3: Value Cache 저장

- `tq_cache_append()` 수정: value를 uniform 4bit으로 양자화하여 저장
- Value 저장용 별도 블록 배열 추가
- `tq_cache_get_value()` API 추가
- Value roundtrip 테스트 추가

### FR-V3-4: Attention 전체 타입 테스트

- 모든 7개 타입에 대해 `tq_attention()` 호출 → 정상 점수 반환 검증
- FP32 dot product 대비 cosine similarity > 0.9 검증

---

## 3. Success Criteria

- BUG-1, BUG-2, BUG-3 모두 수정 + 테스트 통과
- `tq_attention()` 7개 타입 모두 동작
- `tq_cache_append()` value 저장 동작
- Progressive tier 1→2 전환 테스트 통과
- 기존 10개 테스트 + 신규 테스트 모두 통과
- score.sh ≥ 0.99 유지
