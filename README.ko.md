# TurboQuant.cpp

![TurboQuant Hero](docs/assets/hero.png)

**순수 C LLM 추론 엔진. 47 tok/s. 외부 의존성 없음.**

로드 → 생성 → 끝. Python 없이. GPU 없이. 바이너리 하나로.

[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Tests](https://img.shields.io/badge/tests-70%2B%20pass-brightgreen)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![Speed](https://img.shields.io/badge/47%20tok%2Fs-Qwen3.5--0.8B-blue)]()

```
PyTorch CPU:     0.8 tok/s
PyTorch GPU:      10 tok/s
TurboQuant CPU:   47 tok/s  ← 59배 빠름, GPU 불필요
```

---

## 30초 빠른 시작

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

# 모델 변환 (1회, HuggingFace 캐시 자동 감지)
./build/tq_convert

# 실행
./build/tq_run model.tqm -p "What is deep learning?" -j 4
```

```
Prompt: What is deep learning?
---
Deep learning is a field of artificial intelligence and machine learning
that uses artificial neural networks to learn complex patterns...
---
100 tokens in 2.1s (46.9 tok/s, 4 threads, weights=Q4, kv=uniform_4b)
```

---

## 왜 TurboQuant인가?

|  | PyTorch | TurboQuant.cpp |
|---|---|---|
| **속도** | 0.8 tok/s | **47 tok/s** (59배) |
| **로딩** | 3초 | **0.3초** (mmap) |
| **가중치 메모리** | 1.7 GB | **270 MB** (Q4) |
| **KV 캐시** | 전체 크기 | **7.5배 압축** |
| **의존성** | PyTorch, transformers | **없음** |
| **바이너리** | ~2 GB 설치 | **~1 MB** |
| **품질** | 기준 | **코사인 0.999** (PyTorch 대비) |

---

## 구성 요소

```
┌─────────────────────────────────────────────────────┐
│  tq_convert                                          │
│    safetensors → TQM (사전 양자화, mmap 가능)         │
├─────────────────────────────────────────────────────┤
│  tq_run                                              │
│    TQM → mmap 로드 → forward → 토큰 스트리밍         │
│                                                      │
│    ┌─── Forward Pass ────────────────────────────┐  │
│    │  DeltaNet (18 레이어, 순환)                  │  │
│    │  Self-Attention (6 레이어, GQA + RoPE)      │  │
│    │  SwiGLU FFN (전체 24 레이어)                 │  │
│    │  KV 캐시: TurboQuant Q4 양자화              │  │
│    │  Attention: 정수 Q4×Q8 (FP32 대비 2.9배)    │  │
│    └─────────────────────────────────────────────┘  │
│                                                      │
│    Q4 가중치 ── NEON matmul ── 멀티스레드            │
└─────────────────────────────────────────────────────┘
```

### 5가지 최적화

| # | 기법 | 효과 |
|---|------|------|
| 1 | **Q4 가중치** — 4-bit, 8배 작음 | 2배 빠름 |
| 2 | **TQM 포맷** — 사전 양자화 mmap | 10배 빠른 로딩 |
| 3 | **정수 attention** — Q4×Q8, ARM vdotq_s32 | 2.9배 빠름 |
| 4 | **멀티스레드 matmul** — pthread, NEON | 1.6배 빠름 |
| 5 | **스트리밍 BF16** — 임베딩 온디맨드 | 메모리 6배 절약 |

### 실제 모델 검증

[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) — 실제 추론, 합성 아님:

```
"1+1="                      → "2"                    ✓
"The capital of France is"  → "Paris"                ✓
"What is deep learning?"    → 정확한 문단             ✓
PyTorch 대비 logits 코사인  → 0.999                  ✓
```

---

## 시퀀스 길이별 속도

```
토큰 수   속도        비고
──────    ─────────   ──────────────────
10        12 tok/s    첫 토큰 지연 포함
30        41 tok/s    ← 40 tok/s 돌파
50        44 tok/s
100       47 tok/s    ← 정상 속도
200       48 tok/s    ← 최대
```

---

## CLI

```bash
# 변환 (1회)
./build/tq_convert                     # 자동 감지

# 추론
./build/tq_run model.tqm -p "Hello"    # 토크나이저 내장
./build/tq_run model.tqm -p "Hello" -j 4 -n 200 -T 0.7

# Python CLI
python3 tools/tq info                  # 양자화 타입
python3 tools/tq +memory llama-3.2-3b 65536
python3 tools/tq_chat.py "What is AI?" # 네이티브 엔진 + KV 분석
```

### Python API

```python
from turboquant import TurboQuant
tq = TurboQuant("cpu")
compressed = tq.quantize_keys(keys, TurboQuant.UNIFORM_4B)  # 7.5배 압축
scores = tq.attention(query, compressed, seq_len, dim, TurboQuant.UNIFORM_4B)
```

---

## 문서

| 문서 | 내용 |
|------|------|
| **[시작 가이드](docs/getting-started.md)** | 빌드, 변환, 실행, 통합 |
| [아키텍처](docs/architecture.md) | 엔진 설계, 4-layer 스택 |
| [Qwen3.5 결과](docs/qwen35_validation_results.md) | 실제 모델 A/B 테스트 |
| [변경 이력](CHANGELOG.md) | 전체 버전 히스토리 |
| [통합 가이드](docs/integration_guide.md) | llama.cpp, vLLM, Python |

---

## 기술 상세

- **8,500줄 이상의 C** — 완전한 추론 엔진, 래퍼 아님
- **8개 양자화 타입** — Uniform, Mixed Precision, PolarQuant, QJL, TurboQuant
- **TQM 포맷** — 사전 양자화 바이너리, mmap 즉시 로딩
- **DeltaNet + Self-Attention** — Qwen3.5 하이브리드 아키텍처 순수 C
- **BPE 토크나이저** — HuggingFace 호환 (248K 어휘, TQM 내장)
- **Q4×Q8 정수 attention** — ARM vdotq_s32, float 역양자화 없음
- **멀티스레드** — pthread matmul + NEON, 설정 가능
- **반복 방지** — repetition penalty로 퇴화 방지
- **20 테스트 스위트, 70+ 테스트** — ASan + UBSan + TSan 클린

---

## 여정

```
1일차 오전:   빈 디렉토리
1일차 오후:   KV 캐시 압축 라이브러리 (8개 타입, A/B 테스트)
1일차 저녁:   완전한 추론 엔진 (모델 로드 → 텍스트 생성)
1일차 밤:    47 tok/s, Q4 가중치, TQM 즉시 로딩

C 코드:       8,500줄 이상
테스트:       20개 스위트 (70+ 테스트)
커밋:         52개
속도:         0.8 → 47 tok/s (59배 개선)
```

---

## 참고 논문

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — KV 캐시 압축
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — 1비트 양자화 JL 변환
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — 극좌표 양자화

아키텍처: [llama.cpp](https://github.com/ggerganov/llama.cpp), [vLLM](https://github.com/vllm-project/vllm), [ONNX](https://github.com/onnx/onnx) 참조.

---

**[QuantumAI Inc.](https://quantumai.kr)** | [hi@quantumai.kr](mailto:hi@quantumai.kr)
