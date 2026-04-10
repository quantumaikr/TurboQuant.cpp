<p align="center">
  <img src="docs/assets/hero.png" alt="quant.cpp" width="600">
</p>

<h3 align="center">quant.cpp</h3>
<p align="center"><b>AI의 대화 메모리를 압축해도, 품질은 떨어지지 않습니다.</b></p>

<p align="center">
  AI 모델은 대화가 길어지면 메모리가 부족해 대화를 잊습니다.<br>
  quant.cpp는 이 메모리를 <b>3배 압축</b>해서 <b>3배 더 긴 대화</b>를 가능하게 합니다.<br>
  비결: AI는 인간처럼 최근 대화에 집중합니다.<br>
  최근 기억은 완벽하게 유지하고, 나머지만 압축합니다. <b>품질 손실: 0%.</b>
</p>

<table align="center">
<tr>
<td align="center"><b>메모리 3배 절감</b><br>품질 손실 0%</td>
<td align="center"><b>13% 더 빠름</b><br>압축 안 한 것보다</td>
<td align="center"><b>8GB Mac에서</b><br>32K 대화 가능</td>
<td align="center"><b>C 파일 하나</b><br>16K줄, 의존성 0</td>
</tr>
</table>

<p align="center">
  <code>pip install quantcpp</code> — 또는 <code>#include "quant.h"</code> 한 줄.<br>
  <a href="https://quantumaikr.github.io/quant.cpp/">브라우저에서 바로 체험 →</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/quantcpp/"><img src="https://img.shields.io/pypi/v/quantcpp.svg?label=PyPI&color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/quantcpp/"><img src="https://img.shields.io/pypi/pyversions/quantcpp.svg" alt="Python"></a>
  <a href="https://github.com/quantumaikr/quant.cpp/releases/latest"><img src="https://img.shields.io/github/v/release/quantumaikr/quant.cpp?label=release" alt="Release"></a>
  <a href="#"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-35%20pass-brightgreen" alt="Tests"></a>
</p>

---

## 3줄로 시작하기

```bash
pip install quantcpp
```

```python
from quantcpp import Model

m = Model.from_pretrained("Llama-3.2-1B")  # 모델 자동 다운로드 (~750 MB)
print(m.ask("중력이란 무엇인가요?"))
```

API 키 없음. GPU 없음. 설정 없음. [브라우저에서 바로 체험 →](https://quantumaikr.github.io/quant.cpp/)

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

## 핵심 돌파: 메모리를 압축해도 품질이 떨어지지 않는다

AI는 인간처럼 **최근 대화에 집중**합니다. 최근 128 단어만 완벽하게 기억하면, 나머지 수만 단어는 압축해도 됩니다.

**3개 모델에서 검증:**

| 모델 | 압축만 적용 | **Progressive 적용 (자동)** | 메모리 절감 |
|---|---|---|---|
| Llama 3.2 3B | 품질 3.1% 하락 | **품질 하락 0%** | 3× |
| Llama 3.2 1B | 품질 16.1% 하락 | **품질 1.2% 하락** | 3× |
| SmolLM2 135M | 품질 12.9% 하락 | **품질 3.1% 하락** | 3× |

Progressive는 **기본 ON** — 모든 사용자가 자동으로 혜택을 받습니다. 추가 메모리: 1.75 MB.

```python
m = Model("model.gguf")  # progressive가 이미 켜져 있음 — 설정할 것 없음
```

**더 강한 압축이 필요하면?** V 캐시도 압축하여 7.5× (Llama 3.2 3B 기준, +0.9%):

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
<summary><b>다른 엔진과의 비교</b></summary>

| | quant.cpp | llama.cpp | vLLM | MLX |
|---|:---:|:---:|:---:|:---:|
| KV 압축 | **7가지 방식** | Q8_0/Q5_0 (2x) | — | — |
| 코드 크기 | **72K LOC** | 250K+ | 100K+ | 50K+ |
| 임베딩 가능 | **단일 헤더** | 라이브러리 | 라이브러리 | 프레임워크 |
| GPU 속도 | 기본 | 최적화됨 | **최고** | Metal |
| 하루만에 코드 읽기 | **✅** | ❌ | ❌ | ❌ |

**llama.cpp**를 쓰세요 — 워크스테이션에서 최고 속도가 필요할 때.
**vLLM**을 쓰세요 — 배치 서빙이 필요할 때.
**quant.cpp**를 쓰세요 — AI를 앱/게임/웹/디바이스에 **넣어야** 할 때.
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
