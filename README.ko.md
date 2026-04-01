# TurboQuant.cpp

**[TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) KV 캐시 압축을 구현한 독립형 C 추론 엔진. 래퍼가 아닌 자체 구축, 외부 의존성 없음.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![Tests](https://img.shields.io/badge/tests-31%20pass-brightgreen)]()
[![ASan](https://img.shields.io/badge/ASan%2BUBSan-clean-brightgreen)]()

```
Qwen3.5-35B-A3B MoE (IQ2_XXS, GGUF):
  baseline:        "The capital of France is Paris."     ✓
  1-bit K:         "The capital of France is Paris."     ✓  ← 동일 출력

Gemma 3 4B perplexity (101 토큰):
  FP16 KV:         PPL = 35.99
  1-bit K + Q4 V:  PPL = 36.00  (+0.03%)

GPU 백엔드: CUDA | Metal | Vulkan (AMD) | ROCm/HIP (AMD) | NEON | AVX2
```

---

## 빠른 시작

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build   # 31/31 통과해야 합니다

./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q4
```

> 이것은 llama.cpp 포크나 래퍼가 아닌, 처음부터 자체 구축한 독립 추론 엔진입니다.
> 모델은 TQM 포맷(사전 양자화) 또는 GGUF Q8_0(실험적)으로 로딩합니다.

---

## 지원 모델

| 모델 | 파라미터 | 포맷 | 속도 (6T) | KV 압축 |
|------|----------|------|-----------|---------|
| **Qwen3.5-35B-A3B** | 35B (3B 활성) | GGUF IQ2_XXS | 3.0 tok/s | 1-bit K ✓ (byte-identical) |
| **Gemma 3 4B** | 4B | TQM | 20.2 tok/s | PPL +0.03%, 모든 KV 타입 ✓ |
| **Qwen3.5-0.8B** | 752M | TQM/GGUF | 80.1 tok/s | 모든 KV 타입 ✓ |
| **Gemma 3 270M** | 270M | TQM | 176 tok/s | 모든 KV 타입 ✓ |

아키텍처: Gemma 3 (슬라이딩 윈도우, GeGLU), Qwen3.5 (DeltaNet 하이브리드), Qwen2-MoE (top-K 라우팅, 공유 전문가).

GGUF: Q8_0 검증 완료. IQ2_XXS/IQ2_S 역양자화 구현 (E8 lattice codebook). 35B MoE 로딩 + 추론 검증 (RSS 4.7GB on 16GB Mac).

---

## KV 압축

Key는 RHT + 부호 해싱(1비트) 또는 Lloyd-Max 코드북(3/4비트)으로 압축.
Value는 독립적으로 Q4 또는 Q2로 양자화.

```bash
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q4   # 4.9x 총 K+V
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q2   # 7.1x 총 K+V
./build/tq_run model.tqm -p "Hello" -k turbo_kv_3b          # 3-bit keys, FP16 values
./build/tq_run model.tqm -p "Hello" -M                       # 메모리 통계
```

| 구성 | K+V/토큰 (Gemma 4B) | 압축률 | PPL 영향 |
|------|---------------------|--------|----------|
| FP16 K+V | 136.00 KB | 1.0x | 기준 |
| 1-bit K + FP16 V | 74.38 KB | 1.8x | +0.00% |
| 1-bit K + Q4 V | 27.62 KB | 4.9x | +0.03% |
| 1-bit K + Q2 V | 19.12 KB | 7.1x | +17.3% |

> K-only 양자화(V는 FP16)는 perplexity 무손실.
> Q4 V는 +0.03% PPL — 사실상 무손실. Q2 V는 눈에 띄게 저하.

---

## 알고리즘

```
Key:   key → L2 정규화 → RHT → Lloyd-Max 코드북 (b-1 bits) → QJL 부호 (1 bit)
       1-bit: 부호만 → XOR + popcount attention

Value: value → 블록별 Q4/Q2 양자화 → packed nibble에서 직접 fused 누적
```

[TurboQuant 논문](https://arxiv.org/abs/2504.19874) (ICLR 2026)은 일반 양자화기가 내적 추정에 체계적 편향을 도입함을 증명. RHT + QJL 보정으로 추정기가 증명 가능하게 비편향.

---

## 분석 도구

```bash
./build/tq_run model --ppl input.txt -k turbo_kv_1b -v q4   # perplexity
./build/tq_run model --profile-kv -k turbo_kv_1b -p "text"  # 활성값 분포
./build/tq_run model --recommend -k turbo_kv_1b -p "text"   # 레이어별 비트 할당
./build/tq_run model --calibrate -k turbo_kv_1b -p "text"   # 코드북 캘리브레이션
./build/tq_run model --attn-entropy -k turbo_kv_1b -p "text" # attention 엔트로피
bash bench/auto_profile.sh model                              # 전체 파이프라인
```

---

## 검증

| 항목 | 결과 | 재현 방법 |
|------|------|----------|
| Perplexity (1b K + Q4 V) | PPL +0.03% vs FP16 | Gemma 4B `--ppl` |
| 비편향성 | 상대 bias < 0.2%, 10만 샘플 | `test_unbiased` |
| Attention 코사인 (1-bit) | 0.634 = 이론 한계 2/pi | `test_attention_distribution` |
| Lloyd-Max 코드북 | MSE가 정보이론 최적의 1.18배 이내 | `test_codebook_theory` |
| 코드북 캘리브레이션 | 실제 활성값에서 MSE 49.7% 개선 | `--calibrate` |
| 누적 오차 (16 레이어) | 코사인 0.998 (Q4), 준선형 성장 | `test_cumulative_error` |
| NEON/스칼라 일치성 | 14개 경로 검증 | `test_neon_scalar` |
| 엣지케이스 | 29개 (NaN, Inf, n=1, dim=0) | `test_edge_cases` |
| ASan + UBSan | 31/31 클린 | `scripts/sanitize.sh` |
| Rate-distortion gap | Q4: 하한 대비 2.41배 | `test_rate_distortion` |

벤치마크: `bench/ablation_test.sh`, `bench/kv_quality_bench.sh`, `bench/long_quality_test.sh`, `bench/sampling_test.sh`

---

## FAQ

**Q: "1-bit attention 코사인 0.634는 너무 낮지 않나?"**
2/pi = 0.637이 부호 양자화의 정보이론적 최대값. 우리 0.634가 이 한계에 도달. 더 높은 코사인이 필요하면 3-bit(0.918) 사용.

**Q: "llama.cpp KV 양자화와 뭐가 다른가?"**
llama.cpp는 uniform min-max. TurboQuant는 RHT + Lloyd-Max + QJL 잔차 보정으로 증명 가능한 비편향 내적 추정. 코드북 centroid 이론 검증 완료 (`test_codebook_theory`).

**Q: "Perplexity는?"**
측정 완료. Gemma 4B 1-bit K + Q4 V: PPL = 36.00 vs 35.99 기준 (+0.03%). K-only 양자화는 정확히 무손실 (PPL 동일). `--ppl` 플래그 참조.

**Q: "NEON 코드가 정확한가?"**
모든 NEON 경로를 스칼라 참조와 비교 검증 (`test_neon_scalar`). Q4 dequant nibble 인터리빙 버그를 검증 과정에서 발견 후 수정. ASan + UBSan 31개 전체 스위트 클린.

**Q: "RHT 오버헤드는?"**
128차원 벡터당 147 ns (NEON 벡터화). 1-bit attention: 1.2 ns/key. matmul (~1ms/레이어) 대비 무시 가능. `bench/bench_kv_overhead.cpp` 참조.

**Q: "소형 모델만 지원?"**
아니요. 270M~35B까지 검증. Qwen3.5-35B-A3B MoE (IQ2_XXS, 9.9GB)가 16GB Mac Air M3에서 RSS ~4.7GB로 mmap 기반 실행. KV 압축은 아키텍처 독립적이며 수정 없이 스케일.

**Q: "AMD GPU 지원?"**
지원. Vulkan 컴퓨트 셰이더 (크로스플랫폼, AMD/NVIDIA/Intel) 또는 ROCm/HIP (네이티브 AMD, CUDA 호환 API). 빌드 시 `-DTQ_BUILD_VULKAN=ON` 또는 `-DTQ_BUILD_ROCM=ON`.

**Q: "어떤 GGUF 포맷이 작동하나?"**
Q8_0은 coherent output 검증 완료. Q5_K/Q6_K는 비순환 레이어에서 작동. IQ2_XXS/IQ2_S 역양자화 구현 완료 (E8 lattice codebook). DeltaNet 레이어는 순환 상태 민감도로 Q8_0 이상 필요.

---

## GPU 백엔드

AMD를 포함한 모든 주요 GPU 플랫폼에서 실행 가능.

| 백엔드 | 대상 | 상태 | 코드량 |
|--------|------|------|--------|
| **CUDA** | NVIDIA GPU | 프로덕션 | 1,919줄 |
| **Metal** | Apple Silicon | 프로덕션 | 1,494줄 |
| **Vulkan** | **AMD + 크로스플랫폼** | 신규 | 2,317줄 |
| **ROCm/HIP** | **AMD ROCm** | 신규 | 2,174줄 |
| **NEON** | ARM CPU | 프로덕션 | 980줄 |
| **AVX2** | x86 CPU | 확장 | 638줄 |

```bash
cmake -B build -DTQ_BUILD_VULKAN=ON  # AMD / 크로스플랫폼
cmake -B build -DTQ_BUILD_ROCM=ON    # AMD ROCm (CUDA 호환 API)
cmake -B build -DTQ_BUILD_CUDA=ON    # NVIDIA
cmake -B build -DTQ_BUILD_METAL=ON   # Apple Silicon
```

> AMD 사용자: Vulkan (크로스플랫폼) 또는 ROCm/HIP (네이티브) 선택 가능.

---

## GGUF 모델 로딩

커뮤니티 GGUF 모델을 직접 로딩 — 변환 불필요.

```bash
./build/tq_run model.gguf -p "Hello" -k turbo_kv_1b
# 지원: Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_S, BF16, F16, F32
# MoE: top-K 라우팅 + 공유 전문가 + SwiGLU
```

| 기능 | 상태 |
|------|------|
| GGUF v3 파서 (mmap) | 24개 양자화 타입 지원 |
| IQ2_XXS (E8 lattice) | 전체 codebook 역양자화 |
| IQ2_S (10-bit grid) | 전체 codebook 역양자화 |
| MoE 라우팅 | 256 전문가, top-8, 공유 전문가 |
| DeltaNet 하이브리드 | Qwen3.5 DeltaNet + self_attn |
| On-the-fly 가중치 역양자화 | FP32 변환 없이 ~5GB 절감 |

---

## 기술 상세

**자체 구축 추론 엔진** — 포크도 래퍼도 아닌, 모든 컴포넌트를 직접 작성.

- **20,000줄+ C/C++** — transformer, tokenizer, matmul, attention, sampling, GPU 커널 — 외부 의존성 없음
- **12개 KV 양자화 타입** — 핵심 차별점: RHT + Lloyd-Max + QJL로 비편향 내적
- **6개 컴퓨트 백엔드** — CUDA, Metal, Vulkan, ROCm/HIP, NEON, AVX2
- **Fused Q4 attention** — packed nibble에서 직접 가중합, dequant 버퍼 없음
- **적응적 압축** — 레이어별 비트 추천, 온라인 코드북 캘리브레이션 (MSE 49.7% 개선)
- **GGUF v3 로더** — 24개 양자화 타입, IQ2 E8 lattice, MoE 전문가 디스패치, on-the-fly 역양자화
- **31개 테스트 스위트** — perplexity, 비편향성, attention 분포, 코드북 이론, NEON 일치성, 엣지케이스, rate-distortion, 누적 오차

---

## 참고 논문

- **[TurboQuant](https://arxiv.org/abs/2504.19874)** (ICLR 2026) — 근최적 왜곡률의 온라인 벡터 양자화
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — 1비트 양자화 JL 변환
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — 극좌표 KV 양자화

전체 변경 이력: [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md)

---

**[QuantumAI Inc.](https://quantumai.kr)** | [hi@quantumai.kr](mailto:hi@quantumai.kr)
