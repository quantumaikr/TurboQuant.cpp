# TurboQuant.cpp

**[TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) KV 캐시 압축을 구현한 독립형 C 추론 엔진. 래퍼가 아닌 자체 구축, 외부 의존성 없음.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![CI](https://img.shields.io/github/actions/workflow/status/quantumaikr/TurboQuant.cpp/ci.yml?label=CI)]()
[![Tests](https://img.shields.io/badge/tests-33%20pass-brightgreen)]()
[![ASan](https://img.shields.io/badge/ASan%2BUBSan-clean-brightgreen)]()

## TurboQuant가 하는 일

**3.8x KV 캐시 압축, 1% 미만 품질 손실 — 3개 모델에서 검증.**

```
SmolLM2 1.7B (Llama), 814 토큰:

  FP32 KV baseline:      PPL = 9.51
  4-bit K + Q4 V (3.8x): PPL = 9.36  (-1.6%)  ← baseline보다 더 나음

  32K 컨텍스트 메모리:  6.4 GB → 1.7 GB  (4.7 GB 절약)
```

비교: llama.cpp의 Q4 KV는 같은 모델에서 PPL +10.6%.
TurboQuant의 4-bit K는 PPL +0.0%.

---

## 검증된 결과

### 모델별 PPL (진짜 dequant — FP32 fallback 없음)

| 모델 | Baseline PPL | 4-bit K + Q4 V PPL | 차이 | 압축 |
|------|-------------|--------------------|----|------|
| SmolLM2 1.7B (Llama) | 9.51 | 9.36 | **-1.6%** | 3.8x |
| Qwen3.5 0.8B | 153.6 | 155.1 | **+0.9%** | 3.8x |
| Qwen3.5 4B | 19.63 | 19.75 | **+0.6%** | 3.8x |

모든 측정은 진짜 dequant 경로 — key는 양자화 캐시에만 저장, attention 시 역양자화. FP32 key 캐시 없음.

### llama.cpp KV 양자화 대비

| 방법 | KV 압축 | PPL 변화 | 엔진 |
|------|--------|----------|------|
| llama.cpp Q4_0 KV | 4x | **+10.6%** | llama.cpp (Metal) |
| **TurboQuant 4-bit K** | **4x (K만)** | **+0.0%** | TurboQuant (CPU) |
| **TurboQuant 4-bit K + Q4 V** | **3.8x (K+V)** | **< 1%** | TurboQuant (CPU) |

### 컨텍스트 확장

| 하드웨어 | 모델 | FP16 KV | 4-bit K + Q4 V | 배율 |
|----------|------|---------|---------------|------|
| **8GB 노트북** | Llama 8B (Q4) | 16K | 61K | 3.8x |
| **16GB Mac Air** | SmolLM2 1.7B | 78K | 298K | 3.8x |
| **24GB RTX 3090** | Llama 8B (Q4) | 147K | 559K | 3.8x |

---

## 빠른 시작

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build   # 33/33 통과

./build/tq_run model.gguf -p "Hello" -k uniform_4b -v q4   # 3.8x 압축
./build/tq_run model.gguf --ppl input.txt -k uniform_4b -v q4  # PPL 측정
```

---

## 지원 모델

| 모델 | 아키텍처 | 파라미터 | 포맷 | 속도 (M3, 6T) | KV 검증 |
|------|----------|----------|------|--------------|---------|
| **Qwen3.5-35B-A3B** | Qwen2-MoE | 35B (3B 활성) | GGUF IQ2_XXS | ~1-4 tok/s | 4-bit K ✓ |
| **Qwen3.5-4B** | Qwen3.5 | 4B | GGUF Q8_0 | 5.4 tok/s | PPL +0.6% ✓ |
| **SmolLM2-1.7B** | Llama | 1.7B | GGUF Q8_0 | 24 tok/s | PPL -1.6% ✓ |
| **Qwen3.5-0.8B** | Qwen3.5 | 752M | TQM / GGUF | 35 tok/s | PPL +0.9% ✓ |
| **Gemma 3 270M** | Gemma 3 | 270M | TQM | 176 tok/s | 4-bit K ✓ |

**4개 아키텍처:** Llama, Gemma 3, Qwen3.5 (DeltaNet), Qwen2-MoE.

---

## 작동 원리

```
저장:    key → 블록별 min-max → 4-bit 양자화 → 압축 캐시
복원:    압축 블록 → FP32 역양자화 → 표준 attention

진짜 메모리 절감: FP32 key 캐시가 제거됨.
Attention은 역양자화된 key에서 FP32로 실행.
```

---

## 압축 옵션

| 구성 | 압축률 | PPL 영향 | 용도 |
|------|--------|----------|------|
| **4-bit K + Q4 V** | **3.8x** | **< 1%** | **권장** |
| 4-bit K + FP16 V | 1.6x | +0.0% | 최대 품질 |
| 4-bit K + Q2 V | 4.6x | +36% | 공격적 |

---

## FAQ

**Q: "4-bit K가 어떻게 0% PPL 손실?"**
16 levels 블록별 min-max 양자화는 key 방향을 충분히 보존합니다. softmax attention 분포가 FP32와 사실상 동일.

**Q: "llama.cpp Q4 KV보다 나은 점은?"**
llama.cpp Q4_0은 같은 모델에서 PPL +10.6%. 우리 4-bit K는 +0.0%. K와 V를 독립적으로 최적 방법으로 양자화하는 차이.

**Q: "1-bit / 2-bit / 3-bit은?"**
전부 테스트했습니다. 4-bit 이하에서 품질이 크게 떨어집니다:
- 3-bit (sub-block): PPL +60%
- 2-bit: PPL 붕괴
- 1-bit: PPL 붕괴

4-bit이 현재 접근에서 KV 캐시 key의 실질적 최소입니다.

**Q: "메모리 절감이 진짜인가?"**
네. FP32 key 캐시가 제거됩니다 — key는 양자화 캐시에만 저장되고 attention 시 실시간 역양자화.

---

## 기술 상세

**30,000줄+ C/C++/Metal** — 모든 컴포넌트 직접 작성, 외부 의존성 없음.

- **13개 KV 양자화 타입** — uniform 2/3/4-bit, TurboQuant 1-4 bit, PolarQuant, QJL, mixed
- **GGUF v3 로더** — 24개 양자화 타입, IQ2 E8 lattice, MoE 디스패치
- **llama.cpp 통합** — `integrations/llamacpp/patch/`에 self-contained 패치
- **Python 바인딩** — `bindings/python/turboquant_cli.py`
- **Docker** — `docker build . && docker run turboquant model.gguf -p "Hello"`
- **33개 테스트 스위트** — perplexity, 비편향성, NEON 일치성, 엣지케이스

---

## 참고 논문

- **[TurboQuant](https://arxiv.org/abs/2504.19874)** (ICLR 2026) — 근최적 왜곡률의 온라인 벡터 양자화
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — 1비트 양자화 JL 변환
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — 극좌표 KV 양자화

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=quantumaikr/TurboQuant.cpp&type=Date)](https://star-history.com/#quantumaikr/TurboQuant.cpp&Date)

---

**[QuantumAI Inc.](https://quantumai.kr)** | [hi@quantumai.kr](mailto:hi@quantumai.kr)
