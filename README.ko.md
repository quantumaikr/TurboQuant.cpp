# TurboQuant.cpp

![TurboQuant Hero](docs/assets/hero.png)

**극한 KV 캐시 압축을 내장한 LLM 추론 엔진. 외부 의존성 없음. 순수 C.**

모델 로드, 텍스트 생성, KV 캐시 압축 — 하나의 바이너리, Python 불필요.

[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Tests](https://img.shields.io/badge/tests-70%2B%20pass-brightgreen)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![Qwen3.5](https://img.shields.io/badge/Qwen3.5--0.8B-14%20tok%2Fs-blue)]()

---

## 한눈에 보기

| | PyTorch | TurboQuant.cpp |
|---|---|---|
| **CPU 속도** | 0.8 tok/s | **14 tok/s** (17배) |
| **GPU 속도** | 10 tok/s (MPS) | **14 tok/s (CPU만으로!)** |
| **가중치 메모리** | 1.7 GB (BF16) | **533 MB** (Q8, `-q` 플래그) |
| **KV 캐시** | FP16 (전체 크기) | **7.5배 압축** (4-bit) |
| **의존성** | PyTorch + transformers | **0개** (순수 C) |

> Qwen3.5-0.8B, Apple Silicon 기준. CPU 엔진이 PyTorch GPU보다 빠름.

---

## 실행하기

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

# 텍스트 생성 (14 tok/s, Q8 가중치, 4 스레드)
./build/tq_run model.safetensors -t tokenizer.json -p "What is AI?" -j 4 -q
```

```
Prompt: What is AI?
---
Artificial intelligence (AI) is a field of computer science that focuses
on creating systems capable of performing tasks that typically require
human intelligence...
---
100 tokens in 7.2s (13.9 tok/s, 4 threads, kv=uniform_4b)
```

### Python

```python
from turboquant import TurboQuant
tq = TurboQuant("cpu")
compressed = tq.quantize_keys(keys, TurboQuant.UNIFORM_4B)  # 7.5배 압축
scores = tq.attention(query, compressed, seq_len, dim, TurboQuant.UNIFORM_4B)
```

---

## 왜 빠른가

### 1. 자체 추론 엔진

래퍼가 아닌 순수 C 추론 엔진:

```
모델 로딩       safetensors (mmap, BF16→FP32 스트리밍)
토크나이저     HuggingFace BPE (248K 어휘)
Forward Pass   DeltaNet + Self-Attention (Qwen3.5 하이브리드)
KV 캐시        TurboQuant 양자화 (4-bit, 자동 압축)
Attention      정수 Q4×Q8 (FP32 대비 2.9배 빠름)
가중치          Q8 양자화 (-q 플래그, 메모리 4배 절약)
생성            Top-p 샘플링, 스트리밍 출력
```

### 2. 정수 도메인 Attention

양자화 데이터에서 직접 attention 계산 — 역양자화 없음:

```
FP32 attention:  22.8 μs (기준)
Q4×Q8 정수:       7.8 μs (2.9배 빠름, ARM vdotq_s32)
```

### 3. Q8 가중치 양자화

가중치 4배 압축, 품질 손실 무시:

```
./build/tq_run model.safetensors -p "1+1=" -q
→ "2" (정확, 2.1 GB 대신 533 MB)
```

---

## 실제 모델 검증

[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)로 검증 — 합성이 아닌 실제 추론:

| 테스트 | 결과 |
|--------|------|
| "1+1=" | **2** ✓ |
| "The capital of France is" | **Paris** ✓ |
| "The capital of Japan is" | **Tokyo** ✓ |
| "What is deep learning?" | 정확한 문단 ✓ |
| PyTorch 대비 logits 코사인 | **0.999** |

### KV 캐시 품질

| 타입 | 압축률 | 품질 (코사인) | 등급 |
|------|--------|-------------|------|
| **uniform_4b** | 7.5x | 0.994 | **A+** |
| **mixed_4b8** | 6.4x | 0.994 | **A+** |
| uniform_2b | 14.2x | 0.953 | A |

---

## CLI 사용법

```bash
# 기본 추론
./build/tq_run MODEL -t TOKENIZER -p "프롬프트" -n 100

# 옵션
-j 4          # 스레드 수 (기본: 4)
-q            # Q8 가중치 양자화 (메모리 4배 절약)
-k uniform_4b # KV 캐시 타입
-T 0.7        # temperature
-P 0.9        # top-p
--info         # 모델 정보 표시
```

### Python CLI

```bash
python3 tools/tq info                          # 양자화 타입 정보
python3 tools/tq bench                         # 성능 벤치마크
python3 tools/tq +memory llama-3.2-3b 65536    # 메모리 계산
python3 tools/tq +memory qwen3.5-0.8b 131072 --json  # JSON 출력
```

---

## 문서

| 문서 | 설명 |
|------|------|
| **[시작 가이드](docs/getting-started.md)** | 빌드, 실행, 통합 |
| [아키텍처](docs/architecture.md) | 엔진 설계, 타입 시스템 |
| [Qwen3.5 검증](docs/qwen35_validation_results.md) | 실제 모델 A/B 결과 |
| [통합 가이드](docs/integration_guide.md) | llama.cpp, vLLM, Python |
| [변경 이력](CHANGELOG.md) | 릴리즈 노트 |

---

## 기술 요약

- **자체 추론 엔진** — 모델 로드, 토큰화, forward, 생성을 순수 C로
- **8개 양자화 타입** — Uniform, Mixed Precision, PolarQuant, QJL, TurboQuant
- **Q8 가중치** — 메모리 4배 절약, NEON 최적화 matmul
- **정수 attention** — Q4×Q8, ARM `vdotq_s32`
- **멀티스레드** — pthread matmul, 설정 가능한 스레드 수
- **하이브리드 모델** — DeltaNet (순환) + Self-Attention (Qwen3.5)
- **RHT** — Random Hadamard Transform, MSE 3.9배 감소
- **K/V 비대칭** — 키/값 독립 비트 할당
- **외부 의존성 제로** — 순수 C11, libc/libm만
- **70+ 테스트** — 19 C++ 스위트 + 22 Python, ASan/UBSan/TSan 클린

---

## 참고 논문

- **TurboQuant** — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- **QJL** — [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) (AAAI 2025)
- **PolarQuant** — [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) (AISTATS 2026)

---

**개발사: [QuantumAI Inc.](https://quantumai.kr)** | [hi@quantumai.kr](mailto:hi@quantumai.kr)
