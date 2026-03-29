# TurboQuant.cpp 오픈소스 공개 — LLM KV 캐시 7.5배 압축

**TurboQuant.cpp**를 오픈소스로 공개합니다. 외부 의존성 없는 순수 C/C++ 라이브러리로, LLM의 KV 캐시를 16비트에서 2~4비트로 압축합니다. **같은 GPU에서 3배 긴 컨텍스트**를 처리할 수 있습니다.

## 문제

KV 캐시는 LLM 추론의 최대 메모리 병목입니다. Llama-3.2-3B로 64K 컨텍스트를 돌리면 KV 캐시만 **7GB** — 모델 가중치보다 많습니다.

## TurboQuant이 하는 일

옵션 하나 바꾸면 됩니다. 모델 동일. GPU 동일. 컨텍스트 3배.

```
적용 전:  Llama-3.2-3B @ 64K → KV 캐시 7.00 GB
적용 후:  Llama-3.2-3B @ 64K → KV 캐시 0.93 GB (87% 절약)
```

## A/B 테스트: 품질은 유지되나?

실제 LLM 분포를 시뮬레이션한 200개 쿼리 × 512개 캐시 키로 직접 비교했습니다:

| 방식 | 압축률 | FP16 대비 코사인 | 등급 |
|------|--------|-----------------|------|
| FP16 (기준) | 1x | 1.000 | — |
| **uniform_4b** | **7.5x** | **0.995** | **A+** |
| turbo_3b | 4.6x | 0.917 | B+ |
| uniform_2b | 14.2x | 0.897 | B |

**uniform_4b는 7.5배 압축에서 99.5% 정확도. 사실상 무손실입니다.**

## 핵심 수치

- 양자화 처리량 **2.87M 요소/ms**
- 어텐션 처리량 **331K 쿼리/초**
- SIMD 가속 **5.74배** (ARM NEON)
- 테스트 **11개 스위트**, ASan/UBSan/TSan 클린
- 외부 의존성 **없음** — 순수 C11, libc/libm만 사용

## 특징

- 7개 양자화 타입 (PolarQuant, QJL, TurboQuant, Uniform)
- 직접 어텐션 커널 — 역양자화 없이 바로 계산 (QJL: 해밍 거리, PolarQuant: cos/sin 룩업)
- 점진적 압축 — 최근 토큰은 고정밀, 오래된 토큰은 자동 압축
- 빔 서치용 Copy-on-Write 페이지 캐시
- CPU (Generic + NEON + AVX2), CUDA, Metal 백엔드
- llama.cpp / vLLM 통합 인터페이스

## 직접 실행해보세요

```bash
git clone https://github.com/anthropics/TurboQuant.cpp
cd TurboQuant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_TESTS=ON -DTQ_BUILD_BENCH=ON
cmake --build build -j$(nproc)
./build/ab_test           # A/B 비교 직접 확인
./build/demo_real_model   # Llama, Qwen, Phi 모델별 메모리 절약
```

TurboQuant (ICLR 2026), QJL (AAAI 2025), PolarQuant (AISTATS 2026) 논문 기반. llama.cpp, vLLM, ONNX의 아키텍처 패턴을 흡수하여 설계했습니다.

Apache 2.0 라이선스. 기여를 환영합니다.

---

**개발사: [QuantumAI Inc.](https://quantumai.kr)** | hi@quantumai.kr
