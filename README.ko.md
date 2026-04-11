<p align="center">
  <img src="docs/assets/hero.png" alt="quant.cpp" width="600">
</p>

<h3 align="center">quant.cpp</h3>
<p align="center"><b>Beyond RAG: 문서 전체를 노트북에 통째로 로드하세요.</b></p>

<p align="center">
  Chunking은 작은 컨텍스트 윈도우를 위한 임시방편이었습니다. 우리는 그것을 불필요하게 만들었습니다.<br>
  6.4× KV 압축으로 16GB Mac에서 전체 문서 이해가 가능합니다.<br>
  C 파일 하나(16K줄), 외부 의존성 0.
</p>

<p align="center">
  <a href="https://pypi.org/project/quantcpp/"><img src="https://img.shields.io/pypi/v/quantcpp.svg?label=PyPI&color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/quantcpp/"><img src="https://img.shields.io/pypi/pyversions/quantcpp.svg" alt="Python"></a>
  <a href="https://github.com/quantumaikr/quant.cpp/releases/latest"><img src="https://img.shields.io/github/v/release/quantumaikr/quant.cpp?label=release" alt="Release"></a>
  <a href="#"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-35%20pass-brightgreen" alt="Tests"></a>
  <a href="https://quantumaikr.github.io/quant.cpp/guide/"><img src="https://img.shields.io/badge/guide-작동_원리-blueviolet" alt="Guide"></a>
</p>

---

## 빠른 시작

**Ollama 스타일 CLI (v0.12.0+):**
```bash
pip install quantcpp

quantcpp pull llama3.2:1b               # HuggingFace에서 다운로드
quantcpp run llama3.2:1b                # 대화형 채팅
quantcpp serve llama3.2:1b -p 8080      # OpenAI 호환 HTTP 서버 (SSE 스트리밍)
quantcpp client "안녕"                   # 스트리밍 클라이언트 → :8080 서버
quantcpp list                           # 캐시된 모델 목록
```

짧은 별칭: `smollm2:135m`, `qwen3.5:0.8b`, `llama3.2:1b`. `run`/`serve` 첫 실행 시 자동 다운로드. `serve`는 OpenAI 호환 `POST /v1/chat/completions` 엔드포인트를 8080 포트에 제공합니다 — 클라이언트가 `"stream": true`를 보내면 SSE 토큰 단위 스트리밍, 생략하면 단일 JSON 응답. 내장 `quantcpp client`는 두 모드 모두 지원 (기본: 스트리밍, `--no-stream`: 단일 응답).

**한 줄 질문:**
```bash
quantcpp run llama3.2:1b "중력이란 무엇인가요?"
```

**Python API (3줄):**
```python
from quantcpp import Model
m = Model.from_pretrained("Llama-3.2-1B")
print(m.ask("중력이란 무엇인가요?"))
```

API 키 없음. GPU 없음. 설정 없음. 모델은 `~/.cache/quantcpp/`에 캐시됩니다. [브라우저에서 바로 체험 →](https://quantumaikr.github.io/quant.cpp/) · [**작동 원리 가이드 →**](https://quantumaikr.github.io/quant.cpp/guide/)

---

## 핵심 발견: Document-QA 7/7 vs Chunk-RAG 0/7

Llama 3.2 3B Q8_0로 측정한 직접 비교 결과:

| 방법 | 정확도 | 할루시네이션 |
|---|---:|---|
| Chunk-RAG (잘못된 청크 검색) | **0/7** | 7개 모두 |
| 전체 문서 (FP32 KV) | **7/7** | 없음 |
| **전체 문서 (6.4x 압축 KV)** | **7/7** | **없음 — 품질 손실 0** |

Chunk-RAG가 잘못된 섹션을 검색하면, 모델은 **"모른다"고 하지 않고 그럴듯한 거짓을 생성**합니다:
- "CTO는 누구인가?" → **"John Smith"** (실제: Maria Santos)
- "매출은?" → **"$1,000,000"** (실제: 8억 4700만)
- "R&D 비율은?" → **"순이익의 15%"** (실제: 매출의 14%)

6.4x KV 압축으로 전체 문서를 한 번에 로드하면, 모델은 **multi-hop 추론**까지 정확히 답합니다 (예: "성장 지역에 영향을 주는 위험은?" → 환율 변동, Section 3 + Section 5 정보 연결 필요).

**핵심**: KV 압축은 단순한 메모리 절감이 아니라 **근본적으로 다른 RAG 접근**을 가능하게 합니다. RAG는 "어떤 문서를 볼지" 결정하고, long-context는 "그 문서를 얼마나 깊이 이해할지" 결정합니다. 전체 결과: [bench/results/document_level_rag_breakthrough.md](bench/results/document_level_rag_breakthrough.md)

> **v2 후속 — Working Memory Cliff (2026-04-11)**: v1 결과를 더 큰 grid로 확장 측정했습니다 (1B/3B 모델, ctx 256-2048, 204 NIAH trials + FP32-weights 통제 실험). 두 모델 모두 명목 128K context window의 **1% 미만**에서 sharp cliff가 존재합니다 (1B Q8 cliff 512-1024, 3B Q4 cliff 1024-1280을 **step function**으로). 6.4× KV 압축은 20개 cell 중 18개에서 fp32 baseline과 bit-for-bit 일치 — cliff는 model property이지 KV/weight quantization artifact가 아닙니다. 정직한 재해석: Beyond RAG는 *유효* working memory 안에 들어가는 문서에 대해서만 동작하며, 그 크기는 명목 context window의 100분의 1에서 1000분의 1입니다. 전체 tech report: [`docs/paper/working-memory-cliff.md`](docs/paper/working-memory-cliff.md). HuggingFace blog post draft: [`docs/paper/hf-blog-draft.md`](docs/paper/hf-blog-draft.md).

---

## 왜 quant.cpp인가?

AI 모델이 대화를 기억하려면 **KV 캐시**라는 메모리가 필요합니다. 대화가 길어질수록 이 메모리가 빠르게 커져서, 모델 자체보다 더 많은 메모리를 차지합니다.

```
일반 엔진:  모델(4GB) + KV 캐시(8GB) = 12GB 필요 → 8GB Mac에서 OOM
quant.cpp:  모델(4GB) + KV 캐시(2.3GB) = 6.3GB → 8GB Mac에서 OK ✅
```

quant.cpp는 이 KV 캐시를 **3배 압축**합니다. 같은 컴퓨터에서 **3배 더 긴 대화**가 가능합니다.

놀라운 점: 압축해도 **품질이 떨어지지 않고**, 오히려 **13% 더 빨라집니다**.

---

## 핵심 성과

Llama 3.2 3B 모델, 3970 토큰 평가:

| 설정 | 품질 (PPL) | 메모리 (32K) | 속도 |
|---|---:|---:|---:|
| 압축 없음 (FP32) | 19.41 | 7.17 GB | 기준 |
| **압축 + progressive** | **19.39 (-0.1%)** | **2.33 GB** | **+13%** |
| 압축 (일반) | 20.02 (+3.1%) | 2.30 GB | +13% |

`progressive=True` 한 줄로, **메모리 3배 절감 + 속도 13% 향상 + 품질 동등**을 달성합니다.

```python
m = Model("model.gguf", progressive=True)
```

---

## 어디에 쓸 수 있나요?

| 용도 | 코드 |
|---|---|
| **챗봇 만들기** | `m.ask("안녕하세요!")` |
| **긴 문서 질문** | `m = Model("model.gguf", context_length=32768)` |
| **대화 저장/복원** | `m.save_context("대화.kv")` → `m.load_context("대화.kv")` |
| **끝없는 대화** | 자동 — context 초과 시 오래된 대화를 압축, 삭제하지 않음 |
| **C 앱에 AI 추가** | `#include "quant.h"` → `cc app.c -lm` |
| **브라우저에서 실행** | [WASM 데모](https://quantumaikr.github.io/quant.cpp/) (193 KB) |

---

## 더 많은 기능

**내 모델 사용:**
```python
m = Model("path/to/any-model.gguf")  # 모든 GGUF 파일 지원
for tok in m.generate("옛날 옛적에"):
    print(tok, end="", flush=True)
```

**대화 저장 & 복원:**
```python
m.ask("이 긴 문서를 읽어줘: ...")
m.save_context("document.kv")      # 압축 상태로 디스크에 저장

m2 = Model("model.gguf")
m2.load_context("document.kv")     # 즉시 복원 — 다시 읽을 필요 없음
m2.ask("37페이지에 뭐라고 써있었어?")
```

**설치 방법:**

| 방법 | 명령어 | 설명 |
|---|---|---|
| **Python** | `pip install quantcpp` | 사전 빌드 wheel (Linux, macOS) |
| **C (단일 헤더)** | `#include "quant.h"` | 654 KB 파일 하나, 의존성 0 |
| **브라우저** | [데모 링크](https://quantumaikr.github.io/quant.cpp/) | 193 KB WASM |
| **소스 빌드** | `cmake -B build && cmake --build build` | 72K LOC, 테스트 35개 |

---

## 기술적 배경

<details>
<summary><b>왜 "최근 128 토큰만 FP32"로 충분한가?</b></summary>

Transformer의 attention 메커니즘은 최근 토큰에 자연스럽게 집중합니다 (causal masking + positional encoding). 측정 결과, attention 가중치의 ~70%가 최근 128 토큰에 집중됩니다.

양자화 오류는 `attention_weight × MSE`로 전파됩니다. 가중치가 높은 영역(최근 128 토큰)만 FP32로 유지하면, 전체 가중 오류가 최소화됩니다.

이것은 **context 길이와 무관**합니다 — 128K context에서도 128 토큰(0.1%)의 FP32이면 충분합니다.
</details>

<details>
<summary><b>벤치마크 전체 데이터</b></summary>

### Llama 3.2 3B — KV 압축 비교 (957-token PPL eval, M-series CPU)

| KV 설정 | 블록 크기 | 압축비 | PPL | Δ vs FP32 | tok/s | vs FP32 속도 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | — | 1× | 13.56 | — | 18.43 | baseline |
| **turbo_kv_4b** ⭐ | 72 | 7.1× | 14.08 | +3.8% | 18.17 | -1.4% (패리티) |
| turbo_kv_5b | 88 | 5.8× | 13.65 | +0.7% | 16.80 | -8.8% |
| turbo_kv_3b | 56 | 9.1× | 15.36 | +13.3% | 16.57 | -10.1% |
| uniform_4b | 68 | 7.5× | 14.60 | +7.7% | 13.27 | -26.8% |

### Progressive KV (3970-token eval, 정직한 조건)

| 설정 | PPL | vs FP32 | k FP32 비율 |
|---|---:|---:|---:|
| FP32 | 19.41 | — | 100% |
| **4-bit + k128** | **19.39** | **-0.1%** | **3.2%** |
| 4-bit flat | 20.02 | +3.1% | 0% |
| 2-bit + k512 | 26.53 | +36.7% | 12.9% |

### Context 확장 (Llama 3.2 3B + turbo_kv_4b, 32K context)

- FP32 KV: 10.4 GB (8GB Mac에서 OOM)
- turbo_kv_4b: 5.5 GB (8GB Mac에서 OK)
- 속도: 7.8 tok/s (+13% vs FP32)
</details>

<details>
<summary><b>"llama.cpp로도 임베딩 되는데, 왜 quant.cpp?"</b></summary>

맞습니다. llama.cpp는 훌륭하고 임베딩도 가능합니다. 차이는 **통합 방식**입니다:

**llama.cpp = 컴파일된 라이브러리** (250K+ LOC). `libllama`를 링크하면 GGML 텐서 그래프, Metal/CUDA 백엔드, 샘플러, 토크나이저가 따라옵니다. 빌드 시스템이 이를 감당할 수 있다면 훌륭합니다 — 하지만 빌드 단계가 필요한 _라이브러리_입니다.

**quant.cpp = 파일 하나** (16K LOC). `#include "quant.h"`, `cc app.c -lm`으로 컴파일. CMake 없음, 링커 플래그는 libc뿐. 하나의 번역 단위.

```
# quant.cpp — C 프로젝트에 AI 추가: 2줄
cc -O2 my_app.c -lm -lpthread -o my_app    # 끝

# llama.cpp — 먼저 라이브러리를 빌드해야 합니다
cmake -B build && cmake --build build
cc my_app.c -Ibuild/include -Lbuild -lllama -lm -lstdc++ -o my_app
```

| 시나리오 | quant.cpp | llama.cpp |
|:---------|:---------:|:---------:|
| **WASM 브라우저** | 192 KB 바이너리 | GGML 텐서 그래프가 너무 큼 |
| **마이크로컨트롤러 / RTOS** | `#include`만 가능 (FS/링커 없음) | 빌드 시스템 필요 |
| **게임 엔진** (Unity/Unreal/Godot) | `.h` 파일 하나 드롭 | 250K LOC 빌드 통합 |
| **교육 / 연구** | 하루만에 전체 코드 읽기 가능 | 훌륭하지만 코드가 방대 |
| **GPU 속도** | 기본 | **Metal/CUDA 최적화** |
| **모델 지원** | 7개 아키텍처 | **100+** |

> **llama.cpp** — 워크스테이션에서 최고 속도가 필요할 때.
> **vLLM** — 배치 서빙이 필요할 때.
> **quant.cpp** — AI를 앱/게임/브라우저/디바이스 _안에_ 넣어야 할 때, 통합 단순성이 GPU 처리량보다 중요할 때.

</details>

<details>
<summary><b>아키텍처 상세</b></summary>

```
include/turboquant/   — 공개 C API
src/core/             — 양자화 알고리즘 (polar, qjl, turbo, uniform)
src/cache/            — 페이지드 캐시 + 점진적 압축
src/backend/cpu/      — CPU 커널 (NEON, AVX2)
src/engine/           — GGUF 로더, Transformer, 토크나이저
tests/                — Google Test (35개 테스트)
wasm/                 — 브라우저 데모 (193 KB)
bindings/python/      — PyPI 패키지
```

지원 모델: Llama 3.x, SmolLM2, Gemma 3/4 (MoE), Qwen 3.5, Phi-3, Mistral, DeltaNet
</details>

---

## 프로젝트 히스토리 & 신뢰성

### 정직한 정정 기록 (10건, 100% 자체 발견)

우리는 잘못된 주장을 외부 신고 전에 스스로 발견하고 공개적으로 정정합니다.

<details>
<summary><b>전체 정정 기록 보기</b></summary>

| # | 시기 | 내용 |
|---|---|---|
| 1 | v0.6.3 | "lossless 7×" 주장 → 재측정 후 정정 |
| 2 | v0.6.x | "beats fp32" → FP32 baseline이 scalar(비최적화)였음 |
| 3 | v0.7.x | "with Metal default" → CMake 기본은 Metal=OFF |
| 4 | v0.7.x | Tim Dettmers HIGGS 댓글 해석 오류 |
| 5 | v0.8.0 | Python kv_compress=1 default abort |
| 6 | v0.8.0 | Cross-heap libc.free abort |
| 7 | v0.8.1 | 65KB/call ask() 메모리 누수 |
| 8 | v0.9.0 | 작동하는 기능을 잘못된 분석으로 비활성화 |
| 9 | v0.10 | 957-token eval에서 k512=53% FP32로 측정 (과대 주장) |
| 10 | v0.10 | 2-bit Pareto 주장 철회 (long context에서 PPL +36.7%) |
</details>

### 연구 기반

- **TurboQuant** (ICLR 2026) — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **PolarQuant** — [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)
- **QJL** — [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)
- **HIGGS** — [arXiv:2411.17525](https://arxiv.org/abs/2411.17525)

### 벤치마크 아티팩트

모든 주장은 재현 가능한 벤치마크 데이터로 뒷받침됩니다:

- [`bench/results/progressive_kv_compression.md`](bench/results/progressive_kv_compression.md) — Progressive KV 발견
- [`bench/results/attention_aware_quantization.md`](bench/results/attention_aware_quantization.md) — Attention-aware 양자화
- [`bench/results/long_context_kv_compression.md`](bench/results/long_context_kv_compression.md) — Long context 메모리 측정
- [`bench/results/layer_adaptive_analysis.md`](bench/results/layer_adaptive_analysis.md) — 레이어 적응 분석 (부정적 결과)

### 라이선스

Apache 2.0 — [LICENSE](LICENSE)

---

<p align="center">
  <a href="https://pypi.org/project/quantcpp/">PyPI</a> ·
  <a href="https://quantumaikr.github.io/quant.cpp/">WASM Demo</a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="https://github.com/quantumaikr/quant.cpp/issues">Issues</a>
</p>
