---
name: qa
description: "TurboQuant.cpp의 통합 정합성을 검증한다. 모듈 간 경계면 불일치, 타입 시스템 정합성, quantize→attention 파이프라인, cache 무결성을 교차 비교한다. '검증', 'QA', '테스트', '정합성 확인', 'validate' 요청 시, 또는 코드 변경 후 머지 전에 자동으로 사용."
---

# QA — Integration Coherence Verification

"존재 확인"이 아닌 **"경계면 교차 비교"**가 핵심이다.

## 검증 대상: 5대 경계면

### 1. 타입 시스템 ↔ 블록 구조
```
TQ_TRAITS[type].type_size == sizeof(block_type)
TQ_TRAITS[type].attention != NULL  (모든 7개 타입)
tq_type_bpe(type) == sizeof(block) * 8.0 / block_size
```

### 2. Quantize → Attention 파이프라인
```
for each type:
  quantize(key) → attention(query, quantized) → finite score
  quantize(key) → dequantize(quantized) → MSE < threshold
```

### 3. Cache → Block
```
cache_append(key, value) → cache_get_block() → valid pointer
cache_append(key, value) → cache_get_value() → valid pointer
cache_share_block() → ref_count == 2
```

### 4. Progressive → Tier
```
append N tokens → tier(0) for recent, tier(1) for warm, tier(2) for cold
recompression: warm_type → cold_type 정확한 타입 사용 확인
```

### 5. SIMD → Generic
```
neon_quantize(input) == generic_quantize(input)  (bit-exact)
neon_attention(q, k) ≈ generic_attention(q, k)  (tolerance)
```

## 실행 방법

1. 변경된 파일 목록 확인 (`git diff --name-only`)
2. 해당 파일이 속한 모듈 식별
3. 해당 모듈의 경계면 체크리스트 실행
4. 빌드 + 테스트: `cmake --build build && ctest --test-dir build`
5. 발견 사항 보고

## 버그 패턴 (과거 발견 사례)

| 패턴 | 사례 | 교훈 |
|------|------|------|
| 함수 포인터 NULL | Uniform attention 미등록 (BUG-1) | traits 테이블 완전성 항상 검증 |
| 하드코딩 가정 | Progressive가 UNIFORM_4B 가정 (BUG-2) | 동적 타입 조회 사용 |
| 파라미터 무시 | Value cache 미저장 (BUG-3) | 모든 파라미터 사용 여부 확인 |
| 정수 오버플로 | size 계산 (BUG-4) | 곱셈 전 오버플로 체크 |
