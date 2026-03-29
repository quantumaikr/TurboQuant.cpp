# TurboQuant.cpp — Work Breakdown Structure v0.4

**Version**: 0.4
**Date**: 2026-03-29
**Focus**: Production readiness — every item is a real bug fix or measurable DX improvement

---

## Phase 1: Critical Bug Fixes

### 1.1 Integer Overflow Protection (BUG-4)

- [x] `src/core/tq_context.c` — `tq_quantize_keys_size()` 오버플로 방어
  - [x] `#define TQ_MAX_SEQ_LEN (1 << 20)` 상수 추가
  - [x] `n <= 0 || head_dim <= 0` → return 0
  - [x] `n > TQ_MAX_SEQ_LEN` → return 0
  - [x] 곱셈 오버플로 체크: `result / type_size != blocks_per_key * n` → return 0
- [x] `src/core/tq_context.c` — `tq_quantize_keys()` 버퍼 크기 검증
  - [x] `out_size < tq_quantize_keys_size(...)` → return TQ_ERR_BUFFER_TOO_SMALL
- [x] `tests/test_edge_cases.cpp` — 오버플로 테스트
  - [x] n=INT_MAX → size 반환 0
  - [x] out_size 부족 → TQ_ERR_BUFFER_TOO_SMALL

### 1.2 CoW Reference Count Fix (BUG-5)

- [x] `src/cache/tq_paged_cache.c` — CoW 순서 수정
  - [x] new_block = malloc() 먼저 시도
  - [x] malloc 실패 → ref_count 변경 없이 TQ_ERR_OUT_OF_MEM 반환
  - [x] malloc 성공 → 복사 → 그 다음에만 ref_count 감소
- [x] `tests/test_paged_cache.cpp` — malloc 실패 시나리오 테스트

### 1.3 Progressive O(1) Append (BUG-6)

- [x] `src/cache/tq_progressive.c` — O(n²) → O(1) 최적화
  - [x] `tq_progressive_t`에 `oldest_hot` 인덱스 필드 추가
  - [x] `append()` 시 `oldest_hot`만 검사하여 tier 전환 결정
  - [x] 전체 순회 제거 — oldest_hot++로 포인터 이동
- [x] `tests/test_progressive.cpp` — 대량 append 성능 테스트
  - [x] 10,000 토큰 append 시간이 선형(O(n))인지 검증

### 1.4 Edge Case 방어 (BUG-7)

- [x] `src/core/tq_context.c` — 입력 검증 강화
  - [x] `seq_len == 0` → TQ_OK 즉시 반환 (no-op)
  - [x] `head_dim < 2` → TQ_ERR_INVALID_DIM
  - [x] `head_dim % 2 != 0` (PolarQuant/TurboQuant 타입) → TQ_ERR_INVALID_DIM
  - [x] `keys == NULL || out == NULL` → TQ_ERR_NULL_PTR
- [x] `src/core/tq_context.c` — `tq_attention()` 입력 검증
  - [x] `query == NULL || kv_cache == NULL || scores == NULL` → TQ_ERR_NULL_PTR
  - [x] `seq_len == 0` → TQ_OK (scores 배열 건드리지 않음)
- [x] `tests/test_edge_cases.cpp` — 전체 edge case 스위트
  - [x] 7개 타입 × (seq_len=0, head_dim=2, NULL input) = 21개 테스트
  - [x] PolarQuant/Turbo + 홀수 head_dim → 적절한 에러

---

## Phase 2: Developer Experience

### 2.1 에러 코드 세분화 (DX-2)

- [x] `include/turboquant/turboquant.h` — 에러 코드 추가
  - [x] `TQ_ERR_BUFFER_TOO_SMALL = -7`
  - [ ] `TQ_ERR_INVALID_SEQ_LEN = -8`
  - [ ] `TQ_ERR_INVALID_HEAD_DIM = -9`
- [x] `src/core/tq_traits.c` — `tq_status_string()` 업데이트

### 2.2 크로스 플랫폼 상수 (DX-5)

- [x] `include/turboquant/tq_types.h` — 자체 수학 상수
  - [x] `#define TQ_PI   3.14159265358979323846f`
  - [x] `#define TQ_PI_2 1.5707963267948966f`
- [x] `src/core/tq_qjl.c` — `M_PI`, `M_PI_2` → `TQ_PI`, `TQ_PI_2`로 교체
- [x] `src/core/tq_polar.c` — 동일 교체 (no M_PI usage found)

### 2.3 Progressive API 공개 (DX-3)

- [x] `include/turboquant/turboquant.h` — Progressive API 선언 추가
  ```
  tq_status tq_progressive_create(...)
  tq_status tq_progressive_append(...)
  tq_status tq_progressive_attention(...)
  void      tq_progressive_free(...)
  ```
- [x] `tq_progressive_config_t`에 대한 기본값 생성 함수: `tq_progressive_default_config()`

### 2.4 최소 예제 (DX-4)

- [x] `examples/minimal.c` — 15줄 hello world
  ```c
  #include "turboquant/turboquant.h"
  int main() {
      tq_context_t* ctx; tq_init(&ctx, TQ_BACKEND_CPU);
      float key[128] = {/*...*/}, query[128] = {/*...*/}, score;
      block_tq_uniform_4b block;
      tq_quantize_keys(ctx, key, 1, 128, TQ_TYPE_UNIFORM_4B, &block, sizeof(block));
      tq_attention(ctx, query, &block, 1, 128, TQ_TYPE_UNIFORM_4B, &score);
      printf("score = %f\n", score);
      tq_free(ctx); return 0;
  }
  ```

### 2.5 편의 함수 추가

- [x] `tq_type_count()` — 사용 가능한 타입 수 반환
- [x] `tq_type_from_name(const char* name)` — 문자열 → tq_type 변환
  - [x] "uniform_4b" → TQ_TYPE_UNIFORM_4B
  - [x] 잘못된 이름 → TQ_TYPE_COUNT (에러)

---

## Phase 3: Code Robustness

### 3.1 Defensive malloc

- [ ] `src/cache/tq_paged_cache.c` — 모든 malloc 후 NULL 체크 통일
- [ ] `src/cache/tq_progressive.c` — 동일
- [ ] `src/core/tq_context.c` — 동일

### 3.2 BPE 값 정확성 검증

- [x] `src/core/tq_traits.c` — BPE 값을 실제 블록 크기에서 계산
  - [x] `bpe = (float)type_size * 8.0f / block_size`
  - [x] 하드코딩 제거, 컴파일타임 계산

---

## 완료 기준

- [x] BUG-4~7 전체 수정 + 테스트 통과
- [x] 새 에러 코드 (BUFFER_TOO_SMALL 등) 동작 검증
- [ ] `M_PI` / `M_PI_2` 제거 → 자체 상수
- [ ] `examples/minimal.c` 15줄 이내 컴파일+실행
- [ ] `tq_type_from_name()` / `tq_type_count()` 동작
- [ ] Progressive API가 turboquant.h에 선언
- [x] 12개 이상 테스트 스위트 전체 통과
- [ ] ASan/UBSan 클린
- [ ] score.sh ≥ 0.99 유지
