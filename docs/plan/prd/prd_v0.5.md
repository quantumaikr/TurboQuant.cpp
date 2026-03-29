# TurboQuant.cpp — Product Requirements Document v0.5

**Version**: 0.5
**Date**: 2026-03-29
**Focus**: 극한 품질 최적화 — refs 코드와의 차이를 0으로 만든다

---

## 1. v0.5 Goal

v0.4까지 프로덕션 안정성을 확보했다. v0.5는 **논문 구현과의 알고리즘 차이를 좁혀 품질을 극한까지 끌어올린다**. 모든 개선은 A/B 테스트로 측정하고, 측정 불가능한 변경은 하지 않는다.

### 발견된 알고리즘 차이 (refs 코드 대비)

| # | 차이 | 파일 | 예상 임팩트 | 난이도 |
|---|------|------|-----------|--------|
| ALG-1 | PolarQuant 각도 범위 [-π,π] → [0,2π] 미적용 | tq_polar.c | MSE -5% | 1줄 |
| ALG-2 | 역양자화 시 +0.5 중심 보정 미적용 | tq_polar.c | MSE -15~20% | 1줄 |
| ALG-3 | QJL inlier/outlier 이중 스트림 스코어 미적용 | tq_qjl.c | Cosine +10~15% | 중 |
| ALG-4 | QJL 그룹 기반 아웃라이어 (개별→그룹) | tq_qjl.c | Cosine +5% | 중 |
| ALG-5 | Value 양자화 per-tile 스케일링 (vLLM 패턴) | tq_uniform.c | Value MSE -10% | 중 |
| ALG-6 | Key/Value 분리 양자화 전략 | tq_types.h | 아키텍처 정확성 | 소 |

**핵심 원칙**: ALG-1, ALG-2는 각각 1줄 수정으로 즉시 품질 개선이 가능. 먼저 적용하고 측정한다.

---

## 2. Functional Requirements

### FR-V5-1: PolarQuant 정밀도 개선 (ALG-1 + ALG-2)

**각도 범위 수정** (ALG-1):
```c
// Before (tq_polar.c):
float t = atan2f(y, x);  // [-π, π]

// After:
float t = atan2f(y, x);
if (t < 0.0f) t += 2.0f * TQ_PI;  // [0, 2π]
```
이유: [-π,π]에서 min-max 양자화하면 π 근처에서 wrap-around 오차 발생. [0,2π]로 이동하면 연속적.

**중심 보정** (ALG-2):
```c
// Before (dequantize):
float theta_f = tscale * (float)theta_idx + tmn;

// After:
float theta_f = tscale * ((float)theta_idx + 0.5f) + tmn;
float radius_f = rscale * ((float)rho_idx + 0.5f) + rmn;
```
이유: 양자화 bin의 가장자리가 아닌 중심으로 복원하면 평균 오차가 절반.

**검증**: MSE 측정 — before vs after on 실제 모델 데이터.

### FR-V5-2: QJL 이중 스트림 스코어 (ALG-3)

현재: `score = sqrt(π/2)/S × norm × (S - 2×hamming)`
목표: `score = sqrt(π/2)/S × norm_inlier × hamming_inlier + sqrt(π/2)/S_out × norm_outlier × hamming_outlier`

핵심 변경:
1. 양자화 시: inlier와 outlier를 분리하여 각각 별도 hash 생성
2. 쿼리 투영 시: 전체 sketch에서 outlier 기여분을 빼서 inlier sketch 계산
3. 점수 계산: 두 스트림의 점수를 각각 norm-weighted하여 합산

참조: `refs/QJL/qjl_kernel/csrc/qjl_score_kernel.cu` line 130

**검증**: Cosine similarity — QJL attention on 실제 모델 데이터.

### FR-V5-3: Value Per-Tile Scaling (ALG-5)

현재: 128개 요소에 단일 scale
목표: 128개 요소를 32개씩 4개 타일로 나눠 각각 scale

```c
typedef struct {
    uint16_t scale[4];        // 4 tiles × fp16 scale
    uint16_t zero_point[4];   // 4 tiles × fp16 zero
    uint8_t  qs[TQ_BK / 2];  // 4-bit packed values
} block_tq_value_tiled;
```

참조: `refs/vllm/csrc/cache_kernels.cu` line 400-418

**검증**: Value roundtrip MSE — per-tile vs per-block on 실제 모델 데이터.

### FR-V5-4: Key/Value 분리 전략 (ALG-6)

Progressive config에 별도 key_type, value_type 설정:
```c
typedef struct {
    tq_type  key_warm_type;    // Keys: PolarQuant or TurboQuant
    tq_type  value_warm_type;  // Values: Uniform (amplitude preservation)
    tq_type  key_cold_type;
    tq_type  value_cold_type;
    ...
} tq_progressive_config_t;
```

이유: 키는 방향 보존이 중요 (PolarQuant/QJL), 값은 진폭 보존이 중요 (Uniform).

---

## 3. Success Criteria

모든 측정은 `build/real_model_validation` 기준:

| 지표 | v0.4 현재 | v0.5 목표 |
|------|----------|----------|
| uniform_4b real cosine | 0.991 | > 0.993 |
| polar_4b real cosine | 0.786 | > 0.85 (ALG-1+2 적용) |
| turbo_3b real cosine | 0.939 | > 0.95 |
| QJL real cosine | 0.857 | > 0.90 (ALG-3 적용) |
| uniform_4b real MSE | 0.0025 | < 0.002 |
| polar_4b real MSE | 0.053 | < 0.03 (ALG-1+2 적용) |

**모든 개선은 A/B 테스트 증거가 있어야 머지한다.**
