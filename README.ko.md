# quant.cpp

![quant.cpp Hero](docs/assets/hero.png)

로컬 LLM을 위한 미니멀 C 추론 엔진. 33K LOC. 외부 의존성 없음.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![CI](https://img.shields.io/github/actions/workflow/status/quantumaikr/quant.cpp/ci.yml?label=CI)]()
[![Tests](https://img.shields.io/badge/tests-34%20pass-brightgreen)]()

---

## 같은 하드웨어에서 4배 긴 컨텍스트

Delta KV 압축으로 품질 손실 없이 4배 더 많은 컨텍스트를 처리합니다.

| 하드웨어 | 모델 | Before | After | 배율 |
|----------|------|--------|-------|------|
| 8GB 노트북 | Llama 8B (Q4) | 16K 토큰 | 61K 토큰 | 3.8x |
| 16GB Mac Air | SmolLM2 1.7B | 78K 토큰 | 298K 토큰 | 3.8x |
| 24GB RTX 3090 | Llama 8B (Q4) | 147K 토큰 | 559K 토큰 | 3.8x |

```bash
./quant model.gguf -p "hello" -k uniform_4b -v q4
```

---

## 왜 quant.cpp인가

|  | quant.cpp | llama.cpp |
|--|-----------|-----------|
| 코드베이스 | 33K LOC, Pure C | 250K+ LOC, C++ |
| KV 압축 품질 | PPL -3.2% (FP32보다 좋음) | PPL +10.6% |
| 의존성 | zero (libc/libm only) | - |
| 설계 목표 | 읽고, 이해하고, 수정 가능 | 기능 완성도 |

같은 모델 (SmolLM2 1.7B), 같은 벤치마크. llama.cpp의 Q4_0 KV는 품질을 떨어뜨립니다. quant.cpp는 오히려 개선합니다.

---

## 빠른 시작

```bash
git clone https://github.com/quantumaikr/quant.cpp && cd quant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 추론 실행
./build/quant model.gguf -p "hello"

# KV 압축 (4-bit K + Q4 V, 3.8x)
./build/quant model.gguf -p "hello" -k uniform_4b -v q4

# Delta 압축 (3-bit K + Q4 V, 4.3x)
./build/quant model.gguf -p "hello" -k uniform_3b -v q4 --delta

# PPL 측정
./build/quant model.gguf --ppl input.txt -k uniform_4b -v q4
```

---

## KV 캐시 압축

### 압축 모드

| 구성 | 압축률 | PPL vs FP32 | 용도 |
|------|--------|-------------|------|
| delta + 3b K + Q4 V | ~4.3x | -3.2% | 최대 압축 |
| delta + 4b K + Q4 V | ~3.8x | -12.2% | 최고 품질 |
| uniform 4b K + Q4 V | 3.8x | -7.8% | 심플, delta 오버헤드 없음 |
| uniform 4b K + FP16 V | 1.6x | +0.0% | 무손실 |

### Delta 압축 원리

표준 KV 캐시는 각 key를 그대로 저장합니다. Delta 압축은 인접 key의 *차이*를 저장합니다 — 비디오의 P-frame과 I-frame처럼.

트랜스포머의 인접 key는 절대값 범위의 ~30%만 차이납니다. 이 작은 범위 덕분에 3-bit 양자화로 충분합니다. Delta 없이 3-bit는 PPL +62%. Delta와 함께라면 PPL -3.2%.

64 토큰마다 FP32 I-frame을 저장하여 누적 드리프트를 방지합니다.

### 전체 PPL 결과 (SmolLM2 1.7B, 999 토큰)

| 구성 | PPL | vs FP32 | 비고 |
|------|-----|---------|------|
| FP32 baseline | 14.58 | -- | 기준 |
| delta + 4b K + Q4 V | 12.80 | -12.2% | 최고 품질 |
| delta + 3b K + Q4 V | 14.11 | -3.2% | 최고 압축 |
| uniform 4b K + Q4 V | 13.44 | -7.8% | 검증됨 |
| uniform 3b K + Q4 V (no delta) | 23.62 | +62% | delta 필수 |

### 모델별 검증 (4b K + Q4 V)

| 모델 | PPL 변화 |
|------|----------|
| SmolLM2 1.7B | -1.6% |
| Qwen3.5 0.8B | +0.9% |
| Qwen3.5 4B | +0.6% |

---

## 지원 모델

| 모델 | 아키텍처 | 파라미터 | KV 검증 |
|------|----------|----------|---------|
| SmolLM2-1.7B | Llama | 1.7B | PPL -1.6% |
| Qwen3.5-0.8B | Qwen3.5 (DeltaNet) | 752M | PPL +0.9% |
| Qwen3.5-4B | Qwen3.5 (DeltaNet) | 4B | PPL +0.6% |
| Qwen3.5-35B-A3B | Qwen2-MoE | 35B (3B active) | 4-bit K verified |
| Gemma 3 270M | Gemma 3 | 270M | 4-bit K verified |
| Gemma 4 E2B | Gemma 4 | 2B | WIP |

5개 아키텍처: Llama, Gemma 3, Gemma 4, Qwen3.5 (DeltaNet), Qwen2-MoE.

---

## FAQ

**Delta 압축은 어떻게 작동하나요?**

각 key를 직접 저장하는 대신 `key[t] - reconstruct(key[t-1])`을 저장합니다. 트랜스포머의 인접 key는 높은 상관관계를 가지므로 delta의 범위가 절대값의 ~30%입니다. 64 토큰마다 full-precision I-frame으로 드리프트를 방지합니다.

**llama.cpp와 뭐가 다른가요?**

quant.cpp는 독립 추론 엔진입니다 (33K LOC, Pure C) — 포크나 래퍼가 아닙니다. KV 압축에서 핵심 차이: llama.cpp Q4_0은 SmolLM2 1.7B에서 PPL +10.6%. quant.cpp의 4-bit K는 같은 모델에서 PPL +0.0%.

**3-bit 이하는요?**

광범위하게 테스트했습니다: 2-bit delta, sub-block scaling, multi-hash, error feedback, NF2, online SVD 등. 어떤 접근도 허용 가능한 품질을 달성하지 못했습니다. 근본 장벽: step당 코사인 유사도 0.997이 200 step 후 0.885로 누적됩니다. 3-bit + delta가 실용적 최소입니다.

---

**[QuantumAI](https://quantumai.kr)** | [GitHub](https://github.com/quantumaikr/quant.cpp)
