# TurboQuant.cpp — Product Requirements Document v0.6

**Version**: 0.6
**Date**: 2026-03-29
**Focus**: 커뮤니티 검증된 실전 최적화 — RHT, K/V 비대칭, Mixed Precision

---

## 1. Background

Reddit r/LocalLLaMA 커뮤니티와 llama.cpp Discussion #20969에서 다음이 검증되었다:

1. **QJL 불필요**: MSE-only(PolarQuant 단독)가 QJL보다 실전에서 우수
2. **K/V 비대칭 필수**: Key와 Value의 norm 차이 최대 182x → 별도 비트 할당 필요
3. **RHT가 핵심**: Random Hadamard Transform 회전이 양자화 품질의 핵심
4. **Mixed Precision**: 3-bit base + 8-bit outlier → 평균 3.6bit에서 PPL +2.1%

우리 프로젝트는 이 중 1번만 부분적으로 반영(uniform_4b 추천). 나머지를 v0.6에서 구현한다.

---

## 2. Requirements (임팩트 순)

### FR-V6-1: K/V 비대칭 양자화 API

**문제**: 현재 `tq_quantize_keys()`와 `tq_quantize_values()`가 동일 비트를 사용.
**해법**: Key와 Value에 독립 타입 지정 가능한 API 추가.

```c
// 새 API
tq_status tq_quantize_kv(tq_context_t* ctx,
                          const float* keys, const float* values,
                          int n, int head_dim,
                          tq_type key_type,    // e.g. TQ_TYPE_UNIFORM_4B
                          tq_type value_type,   // e.g. TQ_TYPE_UNIFORM_2B
                          void* key_out, size_t key_size,
                          void* val_out, size_t val_size);
```

**실전 조합 (llama.cpp 패턴)**:
- `--cache-type-k uniform_4b --cache-type-v uniform_2b` → Key 4bit + Value 2bit
- 평균 3bit, FP16 대비 5.3x 압축, Key 품질 유지

### FR-V6-2: Random Hadamard Transform (RHT) 전처리

**문제**: 직접 양자화하면 채널별 분산 차이가 크고 아웃라이어에 취약.
**해법**: 양자화 전 RHT 회전 → 좌표 간 상관관계 제거 → 스칼라 양자화 최적화.

구현:
```c
// Walsh-Hadamard 변환: O(n log n), in-place
void tq_rht_transform(float* data, int n, uint32_t seed);
void tq_rht_inverse(float* data, int n, uint32_t seed);

// 양자화 파이프라인:
// 1. key → RHT(key) → quantize(rotated) → store
// 2. load → dequantize(rotated) → inverse_RHT → reconstructed_key
```

핵심: Hadamard 행렬은 O(n log n)으로 곱셈 가능 (저장 불필요). 랜덤 부호 플립으로 각 블록마다 다른 회전.

### FR-V6-3: Mixed Precision Outlier Channels

**문제**: 몇 개 채널이 극단적으로 큰 값을 가지면 min-max 양자화의 동적 범위를 낭비.
**해법**: 아웃라이어 채널은 8-bit로 유지, 나머지 3-bit → 평균 ~3.6bit.

```c
typedef struct {
    uint16_t scale;
    uint16_t zero_point;
    uint8_t  qs_3bit[TQ_BK * 3 / 8];     // 3-bit: 8 values per 3 bytes
    uint8_t  outlier_channels;             // 아웃라이어 채널 수
    uint8_t  outlier_idx[4];               // 아웃라이어 채널 인덱스
    uint8_t  outlier_qs[4];                // 8-bit 아웃라이어 값
} block_tq_mixed_3b8;
```

---

## 3. Success Criteria

모든 측정은 `build/real_model_validation` + `build/ab_test`:

| 지표 | v0.5 현재 | v0.6 목표 |
|------|----------|----------|
| K4V2 평균 비트 | N/A | ~3.0 bit |
| K4V2 cosine | N/A | > 0.98 |
| RHT + uniform_4b MSE | 0.00247 | < 0.002 |
| RHT + uniform_4b cosine | 0.991 | > 0.995 |
| 모든 테스트 통과 | 13 | 14+ |
| Score | 99.7% | ≥ 99.7% |
