# TurboQuant.cpp — Product Requirements Document v0.4

**Version**: 0.4
**Date**: 2026-03-29
**Focus**: Production readiness — bugs, DX, robustness

---

## 1. v0.4 Goal

v0.3까지는 기능 구현에 집중했다. v0.4는 **실제 개발자가 30분 안에 통합할 수 있는 프로덕션급 라이브러리**로 만드는 것이 목표다.

### 발견된 문제 (v0.3 감사 결과)

| # | 문제 | 심각도 | 영향 |
|---|------|--------|------|
| BUG-4 | `tq_quantize_keys_size()` 정수 오버플로 | **Critical** | 큰 n에서 잘못된 버퍼 크기 → 메모리 손상 |
| BUG-5 | CoW ref_count: malloc 실패 시 ref_count 꼬임 | **High** | 메모리 누수 또는 use-after-free 가능성 |
| BUG-6 | Progressive append O(n²) — 매 토큰마다 전체 재검사 | **High** | 64K 컨텍스트에서 실용 불가 |
| BUG-7 | edge case: seq_len=0, head_dim 미정렬 미처리 | **High** | 크래시 가능 |
| DX-1 | API 파라미터 순서 비일관적 | **Medium** | 개발자 혼란 |
| DX-2 | 에러 메시지가 어느 파라미터 문제인지 모름 | **Medium** | 디버깅 어려움 |
| DX-3 | Progressive API가 public 헤더에 없음 | **Medium** | 사용 불가 |
| DX-4 | 10줄 hello world 예제 없음 | **Medium** | 첫인상 나쁨 |
| DX-5 | `M_PI_2` 가정 — Windows MSVC 빌드 실패 | **Medium** | 크로스 플랫폼 깨짐 |

---

## 2. Functional Requirements

### FR-V4-1: Critical Bug Fixes

**정수 오버플로 방어** (BUG-4)
- `tq_quantize_keys_size()`에 오버플로 체크 추가
- `n < 0 || n > TQ_MAX_SEQ_LEN` 검증 (TQ_MAX_SEQ_LEN = 1M)
- `tq_quantize_keys()`에 out_size vs 필요 크기 비교 검증

**CoW ref_count 순서 수정** (BUG-5)
- malloc 성공 확인 후에만 ref_count 감소
- 실패 시 원본 블록 유지, 에러 반환

**Progressive O(n²) → O(1) 개선** (BUG-6)
- 매 append마다 전체 순회 대신, `oldest_uncompressed` 인덱스 유지
- 새 토큰 추가 시 해당 인덱스만 검사 → O(1) amortized

**Edge case 방어** (BUG-7)
- `seq_len=0`: 즉시 TQ_OK 반환 (no-op)
- `head_dim < 2`: TQ_ERR_INVALID_DIM 반환
- `head_dim % 2 != 0` (PolarQuant): TQ_ERR_INVALID_DIM
- NULL 포인터: 모든 public API에서 검증

### FR-V4-2: Developer Experience

**API 일관성** (DX-1)
- 모든 함수: `(context_or_handle, inputs..., config..., outputs...)` 순서 통일

**에러 상세화** (DX-2)
- `tq_status` 코드 세분화: `TQ_ERR_INVALID_SEQ_LEN`, `TQ_ERR_INVALID_HEAD_DIM`, `TQ_ERR_BUFFER_TOO_SMALL`
- `tq_last_error_detail(ctx)` — 마지막 에러의 상세 문자열 반환

**Progressive API 공개** (DX-3)
- `turboquant.h`에 progressive 관련 함수 선언 추가
- `tq_progressive_create/append/attention/free` 공식 API화

**최소 예제** (DX-4)
- `examples/minimal.c` — 15줄 이내, 핵심만

**크로스 플랫폼 수정** (DX-5)
- `M_PI_2` → `TQ_PI_2 (1.5707963267948966f)` 자체 상수
- `M_PI` → `TQ_PI (3.14159265358979323846f)` 자체 상수

### FR-V4-3: Robustness

**Edge case 테스트 추가**
- `tests/test_edge_cases.cpp` — seq_len=0, head_dim=2, NULL input, overflow size
- 모든 7개 타입에 대해 edge case 검증

**코드 방어 강화**
- 모든 `malloc` 호출 후 NULL 체크
- 모든 배열 접근 전 범위 체크

---

## 3. Non-Functional

- 기존 11개 테스트 + 신규 edge case 테스트 전체 통과
- ASan/UBSan 클린 유지
- score.sh ≥ 0.99 유지
- Linux GCC + macOS Clang + Windows MSVC 빌드 가능
