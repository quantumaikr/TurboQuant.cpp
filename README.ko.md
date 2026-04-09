<p align="center">
  <img src="docs/assets/hero.png" alt="quant.cpp" width="600">
</p>

<h3 align="center">KV 캐시 압축 LLM 추론을 위한 단일 헤더 C 엔진</h3>

<p align="center">
  프로덕션: <code>uniform_4b</code> KV 캐시 (Llama 3.2 3B에서 4–7x 압축, +6% PPL).<br>
  연구: <a href="https://arxiv.org/abs/2504.19874">TurboQuant</a>, <a href="https://arxiv.org/abs/2502.02617">PolarQuant</a>, <a href="https://arxiv.org/abs/2406.03482">QJL</a> 빌딩 블록 — 한 엔진에 7개 KV 양자화 기법.<br>
  72K LOC 순수 C, 의존성 제로. <a href="#-단일-헤더-모드"><b>quant.h</b></a> 단일 헤더 라이브러리.<br>
  C 컴파일러가 있는 모든 곳에서 동작: <b>iOS · Android · WASM · MSVC · 마이크로컨트롤러</b>.
</p>

<p align="center">
  <a href="https://github.com/quantumaikr/quant.cpp/releases/tag/v0.5.0"><img src="https://img.shields.io/badge/release-v0.5.0-blue" alt="Release"></a>
  <a href="#"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-34%20pass-brightgreen" alt="Tests"></a>
  <a href="#"><img src="https://img.shields.io/badge/score-99.2%25-brightgreen" alt="Score"></a>
  <br>
  <a href="#"><img src="https://img.shields.io/badge/models-7%20verified-blue" alt="Models"></a>
  <a href="https://quantumaikr.github.io/quant.cpp/"><img src="https://img.shields.io/badge/WASM_데모-192KB-purple" alt="WASM"></a>
  <a href="#"><img src="https://img.shields.io/badge/platforms-macOS%20%7C%20Linux%20%7C%20Windows%20%7C%20WASM-orange" alt="Platforms"></a>
</p>

---

## 문제

LLM 메모리의 병목은 모델 가중치가 아니라 **KV 캐시**입니다. 32K 컨텍스트에서 8B 모델의 KV 캐시는 **4GB** — 모델 자체보다 큽니다. 기존 엔진은 모두 KV를 FP16으로 저장합니다. 우리는 이것을 압축합니다.

```
  +------------+-------------------------------+
  |            | KV Cache (FP16)               |
  | Model(4GB) | ██████████████   8K  <-- OOM  |
  +------------+-------------------------------+
  |            | KV (4-bit)                    |
  | Model(4GB) | ██ -------------> 350K ctx    |
  |            |      6.9x smaller             |
  +------------+-------------------------------+
```

## 결과

> **같은 하드웨어. 4–7배 긴 컨텍스트. PPL 측정 + 공개.**

### Llama 3.2 3B Instruct — PPL × 속도 (FP32 KV = 13.56 PPL @ 17.9 tok/s)

> **🏆 Round 10 (NEON `vqtbl1q_s8`) — `turbo_kv_4b`가 fp32 KV 속도와 동등 (7.1× 압축).** 10 라운드 Karpathy 루프로 −45% (literal port) → PARITY까지. Profile 기반 분석으로 진짜 bottleneck이 scalar inner loop임을 발견. 16 Lloyd-Max-Gaussian centroids를 int8로 양자화하고 `vqtbl1q_s8`로 SIMD table lookup하여 fp32와 동등한 SIMD 처리량 확보.

| KV 설정 | 블록 바이트 | 압축 | PPL | Δ vs FP32 | tok/s | vs FP32 속도 |
|:--------|----:|----:|----:|----:|----:|----:|
| FP32 reference | — | 1× | 13.56 | — | 17.9 | baseline |
| **`turbo_kv_4b`** ⭐ 기본 | **72** | **7.1×** | **14.08** | **+3.8%** | **18.7** | **+4.5%** ⬆ |
| `turbo_kv_5b` 🏆 quality | 88 | 5.8× | **13.65** | **+0.7%** | 15.3 | −14.5% |
| `turbo_kv_3b` | 56 | 9.1× | 15.36 | +13.3% | 15.7 | −12.3% |
| `uniform_4b` | 68 | 7.5× | 14.60 | +7.7% | 13.27 | −26.8% |
| llama.cpp `q4_0` KV (lit.) | ~70 | ~7.3× | ~14.99 | +10.6% | — | — |

`turbo_kv_4b` (기본)는 이제 모든 면에서 `uniform_4b`를 Pareto-dominate: 더 나은 PPL, 빠른 속도, 같은 압축. 동시에 fp32 KV 속도와 parity에서 7× 적은 메모리 + 3.8% PPL trade-off만.

5b/3b는 아직 Round 10 NEON 미적용 (v0.7.1 예정).

**빌드 노트**: 모든 측정은 CMake 기본값 `TQ_BUILD_METAL=OFF`. Metal은 batch-1 추론에서 net negative. [Issue #16](https://github.com/quantumaikr/quant.cpp/issues/16).

> **이 비교에 대해**: v0.6.3 릴리스 노트에서 처음 "turbo_kv가 fp32 KV 속도를 능가"라고 주장했습니다. 그건 fp32 attention path가 scalar였기 때문에 발생한 artifact였고, fp32 path에 NEON을 추가한 후(commit `4490c83`) 정직한 격차는 `+5~10%`가 아닌 `−7~−12%`입니다. README와 v0.6.3 릴리스 노트를 정정했습니다.

### 컨텍스트 길이 증가 (`turbo_kv_4b` + `q4` value cache)

| 하드웨어 | 모델 | FP16 KV 컨텍스트 | quant.cpp 컨텍스트 | KV 배율 |
|:---------|:------|------------:|--------------------:|------:|
| 16GB Mac | Llama 3.2 3B | 50K 토큰 | **350K 토큰** | **6.9x** |
| 16GB Mac | Gemma 4 26B MoE | 4K 토큰 | **14K 토큰** | **3.5x** |
| 8GB 노트북 | Llama 8B (Q4) | 16K 토큰 | **61K 토큰** | **3.8x** |
| 24GB RTX 3090 | Llama 8B (Q4) | 147K 토큰 | **559K 토큰** | **3.8x** |

## 왜 quant.cpp인가?

LLM 메모리는 KV 캐시가 지배합니다. quant.cpp는 **실제로 동작하는 KV 캐시 양자화를 다른 누구도 제공하지 않는 폼팩터로 제공하는 미니멀 C 엔진**입니다: 단일 헤더 한 파일, 의존성 제로, iOS/Android/WASM/MSVC/마이크로컨트롤러에서 동작.

**사용 이유 두 가지:**

1. **무언가에 LLM 추론을 임베딩해야 합니다.** 앱, 게임, 웹페이지, 디바이스. quant.cpp는 한 파일(`quant.h`, 628KB) + libc. C 컴파일러가 동작하는 곳은 어디든.

2. **KV 캐시 압축을 연구하고 싶습니다.** quant.cpp는 7개의 KV 양자화 기법을 나란히 구현: `uniform_4b/2b/3b`, `polar_3b/4b`, `qjl_1b`, `turbo_kv_*`. 각각을 한 C 파일에서 읽고 3개 함수로 새 타입을 추가할 수 있습니다.

**정직한 공개**: 2026년 4월 Google이 [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874)을 발표했습니다. quant.cpp의 `turbo_kv_*` 타입은 처음엔 그 알고리즘 구조(Random Hadamard Transform → Lloyd-Max codebook → 1-bit QJL 잔차)를 그대로 포팅했습니다. Karpathy 루프 ablation으로 QJL 잔차 단계가 점수에 **byte-identical 0** 기여한다는 것을 발견했고, 그 단계를 제거한 뒤 freed 16바이트를 더 큰 codebook에 재투자했습니다. 결과 (`turbo_kv_4b` at Llama 3.2 3B PPL 14.28)는 **이전 production 챔피언 `uniform_4b`와 llama.cpp `q4_0` KV를 같은 4-bit 예산에서 모두 능가**합니다. 전체 최적화 이력은 [bench/results/turboquant_reproduction.md](bench/results/turboquant_reproduction.md).

> **논문 수치 그대로의 TurboQuant가 필요하면?** [Google 레퍼런스](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) 사용.
> **휴대폰, 브라우저, 마이크로컨트롤러, 게임 엔진에서 동작하는 KV 압축 + 작고 읽을 수 있는 C 엔진이 필요하면?** quant.cpp.

## 60초 시작 가이드

```bash
# 1. 빌드
git clone https://github.com/quantumaikr/quant.cpp && cd quant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

# 2. 모델 다운로드 (135MB 입문용)
pip install huggingface_hub
hf download bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf --local-dir models/

# 3. 실행
./build/quant models/SmolLM2-135M-Instruct-Q8_0.gguf --chat -p "안녕!" -j 4

# 4. KV 압축 (7배 긴 컨텍스트)
./build/quant models/SmolLM2-135M-Instruct-Q8_0.gguf --chat -p "안녕!" -k uniform_4b -v q4
```

> **[API 레퍼런스](docs/api.md)** · **[WASM 데모](https://quantumaikr.github.io/quant.cpp/)** · **[커스텀 양자화 가이드](docs/custom-quantization.md)** · **[Python: `pip install quantcpp`](#python)**

---

## 실제 동작: Book-in-a-Chat

소설 한 권을 컨텍스트에 넣고 질문합니다. llama.cpp는 메모리 부족, quant.cpp는 전체를 기억합니다.

```bash
# 이상한 나라의 앨리스 (~27K 토큰) KV 압축으로 로드
bash bench/demo/book_chat.sh models/Llama-3.2-3B-Instruct-Q8_0.gguf

# Q: "모자 장수가 앨리스에게 낸 수수께끼는?"
# A: "왜 까마귀가 책상과 같을까?" — 7장, 미친 다과회에서...
```

16GB Mac + Llama 3.2 3B: llama.cpp는 ~50K 토큰에서 OOM. quant.cpp는 KV 6.9x 압축 → **350K 토큰** — 소설 12권 분량.

---

## 비교

### vs llama.cpp: 같은 비트 수에서의 품질

```
                    KV 양자화 품질 (SmolLM2 1.7B, WikiText-2)
                    
  llama.cpp Q4_0 KV │██████████████████████████████████████ PPL +10.6%
                    │
  llama.cpp Q8K+Q5V │▎ PPL ~+1%  ← 추천 설정 (1.6x 압축)
                    │
   quant.cpp 4-bit  │▏ PPL +0.0%  ← 무손실 (3.8x 압축)
                    │
   quant.cpp 3-bit  │█ PPL +1.3%  ← delta 압축 (4.3x)
                    └────────────────────────────────────────────────
                     0%                                         +12%
                              Perplexity 저하 →
```

둘 다 per-block 방식입니다. 품질 차이는 블록 크기(128 vs 32), min-max 범위 인코딩, 독립적 K/V 처리, delta 압축에서 옵니다. ~1.6x 압축이면 llama.cpp Q8+Q5가 우수합니다. quant.cpp는 차이가 큰 **4-7x 범위**를 타겟합니다.

### vs 다른 TurboQuant 구현들

|  | quant.cpp | turbo-quant (Rust) | turboquant-pytorch | scos-lab/turboquant |
|:--|:---------:|:------------------:|:------------------:|:-------------------:|
| 언어 | **순수 C11** | Rust | Python | Python |
| 단일 헤더 | **✅ quant.h (628KB)** | ❌ Cargo crate | ❌ pip install | ❌ |
| 의존성 | **libc + libm** | Rust 툴체인 | PyTorch + CUDA | PyTorch |
| iOS / Android | **✅** | ❌ | ❌ | ❌ |
| WASM (브라우저) | **✅ 192KB** | ❌ | ❌ | ❌ |
| MCU / 임베디드 | **✅** | ❌ | ❌ | ❌ |
| Windows MSVC | **✅** | ✅ | (Python) | (Python) |
| GGUF 모델 로딩 | **✅ 7개 아키텍처** | ❌ | ❌ | research only |
| End-to-end 추론 | **✅** | 커널만 | 커널만 | 커널만 |

### vs 프로덕션 추론 엔진

|  | quant.cpp | llama.cpp | vLLM | MLX |
|:--|:---------:|:---------:|:----:|:---:|
| KV 양자화 | **TurboQuant + 6개 기법** | Q8_0/Q5_0 (2x) | -- | -- |
| 코드 크기 | **72K LOC** | 250K+ | 100K+ | 50K+ |
| 임베더블 | **단일 헤더** | 라이브러리 | 라이브러리 | 프레임워크 |
| 오후 한나절에 읽기 | **✅** | ❌ | ❌ | ❌ |
| GPU 처리량 | 기본 | 풀 | **최고** | Metal |

> 워크스테이션 **속도**가 필요하면 llama.cpp. **배치 서빙**이 필요하면 vLLM.
> **무언가에 LLM 추론을 넣어 배포**해야 한다면 — 앱, 게임, 웹사이트, 디바이스 — quant.cpp.

---

## 지원 모델

| 모델 | 파라미터 | 아키텍처 | 속도 (M1 Pro, 8T) | KV 압축 |
|:------|-------:|:-------------|-------------------:|:---------:|
| SmolLM2 135M | 135M | Llama | **103 tok/s** | 2.4x |
| Llama 3.2 3B Instruct | 3B | Llama 3 (GQA) | **10 tok/s** | 6.9x |
| Gemma 4 26B-A4B-it | 26B (4B active) | MoE 128 experts | **3.9 tok/s** | 3.5x |
| Qwen3.5 0.8B | 752M | DeltaNet 하이브리드 | 80 tok/s | 3.8x |
| Qwen3.5 4B | 4B | DeltaNet 하이브리드 | 20 tok/s | 3.8x |
| SmolLM2 1.7B | 1.7B | Llama | 25 tok/s | 3.8x |
| Gemma 3 270M | 270M | Gemma 3 | 176 tok/s | 3.8x |

GGUF 포맷. llama.cpp 호환 모델을 그대로 사용합니다.

<details>
<summary><b>Gemma 4 26B-A4B 아키텍처 상세</b></summary>

Gemma 4의 하이브리드 MoE 아키텍처를 완전 지원합니다:

- **Dual-FFN**: Dense MLP + 128-expert MoE 병렬 실행 (레이어당)
- **하이브리드 어텐션**: 25 sliding (head_dim=256) + 5 full (head_dim=512) 레이어
- **QK-norm 인식 KV 압축**: K는 FP32 자동 유지, V만 Q4 양자화 (3.5x 절약)
- **Learned RoPE** — 레이어별 주파수 팩터
- **IQ3_XXS/IQ4_NL** NEON 최적화 fused dot (MoE expert 가속)
- **GeGLU** 활성화 (NEON fast tanh 근사)

```bash
./build/quant gemma-4-26B-A4B-it-UD-Q3_K_M.gguf \
  -p "<start_of_turn>user\n대한민국의 수도는?\n<end_of_turn>\n<start_of_turn>model\n" \
  -n 50 -j 8 -T 0.0 -k uniform_4b -v q4
# 출력: "대한민국의 수도는 **서울**입니다."
```

</details>

---

## KV 캐시 압축

### 아이디어

```
표준 방식:  key를 그대로 저장               → 16 bits/원소 → FP16

quant.cpp: key를 4-bit으로 양자화           → 4 bits/원소  → 3.8x
           + value를 Q4로 양자화            → 4 bits/원소  → 6.9x
           + 인접 key의 차이만 delta 인코딩  → 3 bits/원소  → 8.5x

비디오 압축과 같은 원리: 64 토큰마다 I-frame (FP32), 그 사이는 P-frame (3-bit delta).
```

### 품질 vs 압축

```
                    WikiText-2 PPL (SmolLM2 1.7B)

  FP32 baseline      14.63 │ ●
  4b K + FP16 V       14.63 │ ● 동일
  4b K + Q4 V         14.57 │ ● 약간 더 좋음 (!)
  delta 3b K + Q4 V   14.82 │  ●  +1.3%
  llama.cpp Q8K+Q5V   ~14.8 │  ●  ~+1% (1.6x 압축)
  llama.cpp Q4_0 KV   16.18 │          ● +10.6% (3.8x 압축)
  3b K (delta 없음)     ——  │                              ● +62%
                            └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──
                              14  15  16  17  18  19  20  21+
```

### 모드

| 구성 | 압축률 | PPL vs FP32 | 용도 |
|:-----|:------:|:-----------:|:-----|
| `delta + 3b K + Q4 V` | **~8.5x** | +1.3% | 최대 컨텍스트 |
| `delta + 4b K + Q4 V` | ~6.9x | ~0% | 품질 + 압축 |
| `uniform_4b K + Q4 V` | 6.9x | ~0% | 심플, delta 오버헤드 없음 |
| `uniform_4b K + FP16 V` | 1.6x | +0.0% | 무손실 베이스라인 |

### QK-norm 인식 (Gemma 4)

QK-norm이 적용된 모델은 key 벡터를 단위 구체로 정규화하여 극단적으로 sparse한 분포를 만듭니다. quant.cpp는 이를 자동 감지하여 K는 FP32로, V만 양자화합니다 — 완벽한 정밀도 + **3.5x V 메모리 절약**.

---

## 고급 사용법

> **참고:** 아래 `MODEL`은 **본인이 가진** GGUF 파일 경로의 자리표시자입니다. 위 Quick Start에서 다운로드한 `models/SmolLM2-135M-Instruct-Q8_0.gguf`를 그대로 써도 되고, 다른 GGUF 파일 경로로 바꿔도 됩니다. `model.gguf`라는 이름의 실제 파일은 존재하지 않습니다.

```bash
# 디스크에 있는 GGUF 파일 경로를 지정 (Quick Start에서 받은 파일 예시):
MODEL=models/SmolLM2-135M-Instruct-Q8_0.gguf

# Delta 압축 (최대 컨텍스트, 8.5x)
./build/quant $MODEL --chat -p "hello" -k uniform_3b -v q4 --delta

# PPL 벤치마크
./build/quant $MODEL --ppl input.txt -k uniform_4b -v q4

# 모델 정보
./build/quant $MODEL --info

# 성능 프로파일링
./build/quant $MODEL --chat -p "hello" -n 50 --profile
```

---

## 단일 헤더 모드

> 파일 1개 복사. 어떤 C 프로젝트에든 LLM 추가.

```c
#define QUANT_IMPLEMENTATION
#include "quant.h"

int main() {
    quant_model* m = quant_load("path/to/your.gguf");  // 본인의 GGUF 파일 경로
    quant_ctx*   c = quant_new(m, NULL);
    
    // 스트리밍
    quant_generate(c, "농담 하나 해줘", print_token, NULL);
    
    // 또는 일괄 생성
    char* answer = quant_ask(c, "2+2는?");
    printf("%s\n", answer);
    free(answer);
    
    quant_free_ctx(c);
    quant_free_model(m);
}
```

```bash
cc app.c -o app -lm -lpthread    # 끝 — cmake 없음, 프레임워크 없음
```

**15.7K LOC, 643KB, 컴파일 ~2초.** 전체 API:

| 함수 | 설명 |
|:-----|:-----|
| `quant_load(path)` | GGUF 모델 로드 |
| `quant_new(model, config)` | 추론 컨텍스트 생성 |
| `quant_generate(ctx, prompt, cb, ud)` | 콜백으로 토큰 스트리밍 |
| `quant_ask(ctx, prompt)` | 전체 응답 문자열 반환 |
| `quant_free_ctx(ctx)` | 컨텍스트 해제 |
| `quant_free_model(model)` | 모델 해제 |

---

## 브라우저 데모 (WASM)

> **192KB.** 전체 추론 엔진이 대부분의 JPEG 이미지보다 작은 WASM 바이너리로 컴파일됩니다.

```bash
cd wasm && bash build.sh          # 필요: Emscripten
python3 -m http.server 8080       # 로컬 서버 시작
# http://localhost:8080 열고 GGUF 모델 드래그 앤 드롭
```

완전 클라이언트 실행. 서버 전송 없음. KV 압축 기본 활성화.

---

## Docker & 서버

**Docker** (의존성 제로, ~10MB 이미지):
```bash
docker build -t quant.cpp .
docker run -v ./models:/models quant.cpp /models/SmolLM2-135M-Instruct-Q8_0.gguf -p "hello" -k uniform_4b -v q4
# ./models 안에 둔 본인 GGUF 파일명으로 바꿔주세요
```

**OpenAI 호환 서버** (`/v1/chat/completions`):
```bash
cmake -B build -DTQ_BUILD_SERVER=ON && cmake --build build
./build/quant-server models/SmolLM2-135M-Instruct-Q8_0.gguf -p 8080 -k uniform_4b
# 본인의 GGUF 파일 경로로 바꿔서 사용

# OpenAI Python SDK와 호환
curl http://localhost:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

`-DTQ_BUILD_SERVER=ON`으로 빌드. SSE 스트리밍 지원. 요청별 KV 압축 설정 가능.

---

## Python

```bash
cd bindings/python && pip install .
```

```python
from quantcpp import Model

with Model("models/SmolLM2-135M-Instruct-Q8_0.gguf", kv_compress=1) as m:  # 본인의 GGUF 경로로 변경
    print(m.ask("프랑스의 수도는?"))

    # 스트리밍
    for token in m.generate("옛날 옛적에"):
        print(token, end="", flush=True)
```

C 컴파일러 외 빌드 의존성 없음. 설치 시 `quant.h`를 자동 컴파일.

---

## 백엔드 & 성능

| 백엔드 | 플랫폼 | 상태 | 비고 |
|:--------|:-------|:----:|:-----|
| **NEON** | ARM (Apple Silicon) | Production | 5.8x SIMD 가속 |
| **AVX2** | x86 | Production | |
| **Metal** | Apple GPU | Verified | 배치 matmul 디스패치 |
| **CUDA** | NVIDIA GPU | Compiles | |
| **Vulkan** | 크로스 플랫폼 | Compiles | |
| **WASM** | 브라우저 | **NEW** | 192KB 바이너리 |
| **MSVC** | Windows | **NEW** | VS 2019/2022 |

<details>
<summary><b>성능 상세 (Gemma 4 26B, M1 Pro)</b></summary>

| 컴포넌트 | ms/토큰 | 비율 |
|:---------|--------:|------:|
| Attention matmul (Q8_0 NEON) | 168 | 65% |
| MoE experts (IQ3_XXS/IQ4_NL NEON) | 72 | 28% |
| Attention scores | 3 | 1% |
| 기타 | 14 | 6% |
| **합계** | **257** | **3.9 tok/s** |

</details>

---

## FAQ

<details>
<summary><b>llama.cpp와 뭐가 다른가요?</b></summary>

llama.cpp는 전체 기능을 갖춘 추론 프레임워크 (250K+ LOC). quant.cpp는 읽고, 수정하고, 임베딩할 수 있는 미니멀 엔진 (72K LOC). 다른 문제를 위한 다른 도구입니다: llama.cpp는 속도를, quant.cpp는 메모리(KV 압축)와 임베더빌리티(단일 헤더)를 최적화합니다.

</details>

<details>
<summary><b>llama.cpp에도 KV 양자화가 있는데, 뭐가 다른가요?</b></summary>

llama.cpp도 KV 캐시 양자화를 지원합니다 (Q8_0 K + Q5_0 V 추천, ~1.6x 압축에 품질 손실 거의 없음). quant.cpp는 더 높은 압축을 목표로 합니다: 4-bit K + Q4 V로 3.8x에서 PPL +0.0%, delta 압축으로 4.3x에서 +1.3%. 품질 우위는 128-element min-max 블록(vs 32-element), 독립적 K/V 양자화, 인접 키의 delta 인코딩에서 옵니다. 1.6x면 llama.cpp KV 양자화로 충분합니다. 4-7x가 필요하면 quant.cpp를 쓰세요.

</details>

<details>
<summary><b>Karpathy의 llm.c와 비교하면?</b></summary>

비슷한 철학: 미니멀 C, 교육적. 핵심 차이: quant.cpp는 양자화 가중치(Q4_K_M, Q8_0, IQ2), 다중 아키텍처(Llama, Qwen, Gemma, MoE), GGUF 로딩, KV 캐시 압축을 지원합니다. llm.c가 교과서라면 quant.cpp는 프로덕션 버전.

</details>

<details>
<summary><b>왜 llama.cpp보다 느린가요?</b></summary>

세 가지 이유: (1) llama.cpp는 모든 양자화 포맷에 수년간 최적화한 NEON/AVX2 어셈블리가 있고, (2) Metal/CUDA GPU로 전체 forward pass를 오프로드하며, (3) 250K+ LOC vs 72K LOC로 더 많은 마이크로 최적화가 있습니다. quant.cpp는 메모리와 임베더빌리티를 먼저 최적화했습니다. 속도 개선(Metal GPU 전체 오프로드, SIMD 커널 추가)이 진행 중입니다 — [v1.3 계획](docs/plan/prd/prd_v1.3.md) 참고.

</details>

<details>
<summary><b>내 앱에 임베딩할 수 있나요?</b></summary>

네. 두 가지 방법:
1. **단일 헤더**: `quant.h` 복사, `.c` 파일 하나에 `#define QUANT_IMPLEMENTATION`. 끝.
2. **전체 라이브러리**: `libturboquant.a`에 링크.

Linux, macOS, Windows (MSVC/MinGW), iOS, Android, WASM에서 동작합니다.

</details>

<details>
<summary><b>GPU 없으면 쓸모없는 거 아닌가요?</b></summary>

100+ tok/s가 필요하면 llama.cpp + Metal/CUDA를 쓰세요. iOS 앱, WASM 모듈, 게임 엔진, IoT에 추론을 임베딩해야 하면 quant.cpp가 적합합니다. Apple Silicon CPU: 25 tok/s (1.7B), 11.6 tok/s (3B), 3.9 tok/s (26B MoE).

</details>

<details>
<summary><b>브라우저에서 돌아가나요?</b></summary>

네. `cd wasm && bash build.sh`. WASM 바이너리는 192KB. GGUF 모델을 드래그하면 채팅 가능. 모든 것이 클라이언트에서 실행됩니다.

</details>

<details>
<summary><b>3-bit 이하는요?</b></summary>

광범위하게 테스트했습니다 (2-bit delta, NF2, online SVD, multi-hash). 어떤 접근도 허용 가능한 품질을 달성하지 못했습니다. step당 코사인 0.997이 200 step 후 0.885로 누적됩니다. 3-bit + delta가 실용적 최소입니다.

</details>

---

## 문서

| 문서 | 설명 |
|:-----|:-----|
| **[API 레퍼런스](docs/api.md)** | quant.h + libturboquant 전체 C API (730줄) |
| **[커스텀 양자화 가이드](docs/custom-quantization.md)** | 함수 3개로 새 KV 양자화 타입 추가 |
| **[H2H 벤치마크](bench/head_to_head/)** | 재현 가능한 quant.cpp vs llama.cpp 비교 |
| **[KV 압축 랜드스케이프](docs/blog/kv-cache-landscape.md)** | Eviction vs Architecture vs Compression 가이드 |
| **[로드맵](ROADMAP.md)** | 프로젝트 방향과 계획 |
| **[변경 이력](CHANGELOG.md)** | 버전별 릴리스 노트 |
| **[기술 리포트](docs/papers/quant_cpp_tech_report.md)** | 아키텍처와 벤치마크 (Arxiv 초안) |
| **[WASM 데모](https://quantumaikr.github.io/quant.cpp/)** | 브라우저에서 바로 체험 — 설치 불필요 |

---

## 참고 논문 및 인용

quant.cpp는 발표된 연구의 독립 구현체입니다. Variant F 아키텍처 (RHT 전처리 + scalar Lloyd-Max codebook, QJL stage 없음)는 두 prior work의 계보에 위치합니다:

- **HIGGS** — Malinovskii, Panferov, Ilin, Guo, Richtárik, Alistarh. *Pushing the Limits of Large Language Model Quantization via the Linearity Theorem*. Nov 2024. [arXiv:2411.17525](https://arxiv.org/abs/2411.17525). HIGGS가 **Random Hadamard Transform + MSE-optimal grid quantization** 패턴을 weight 양자화에 도입. 우리 `tq_rht.c` (Walsh-Hadamard + Rademacher)가 이 패턴을 따름. *Tim Dettmers가 [llama.cpp discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)에서 thread 참여자들에게 (6+개 fork가 "TurboQuant" 이름을 느슨하게 사용 중) HIGGS credit을 요청한 일반 코멘트를 본 후 우리가 이 attribution을 추가했습니다. 그의 코멘트는 우리에게 직접 향한 것이 아니었지만, substance가 우리 naming에도 적용되어 자발적으로 정정했습니다.*
- **TurboQuant** — Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874). TurboQuant는 rotation 패턴을 **KV cache**에 적용 + 1-bit QJL residual + per-channel outlier handling. 우리 작업은 TurboQuant 직접 포팅으로 시작했으나, 9 라운드 Karpathy 루프로 단순화 (QJL 제거, outlier channel 제거)하여 현재 Variant F가 됨. 우리는 shipped variant가 TurboQuant 알고리즘이라고 주장하지 않습니다 — 경험적으로 도출된 단순화입니다.
- **PolarQuant** — *Quantizing KV Caches with Polar Transformation*. AISTATS 2026. [arXiv:2502.02617](https://arxiv.org/abs/2502.02617). 우리 `tq_polar.c` baseline의 polar-coordinate KV quantization.
- **QJL** — *Quantized Johnson-Lindenstrauss Transform for KV Cache Compression*. AAAI 2025. [arXiv:2406.03482](https://arxiv.org/abs/2406.03482). 1-bit sketch building block. `tq_qjl.c` baseline에 사용; Variant F regime에서 attention 점수에 ~0 기여한다는 것을 발견하고 제거.
- [TurboQuant — Google Research 블로그](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

**정직한 attribution**: Variant F의 구조 (RHT + scalar grid quantization)는 정신적으로 HIGGS에 가장 가깝고, TurboQuant처럼 KV cache에 적용되며, QJL residual과 outlier channel split을 ablation으로 제거. 학술 작업에서 quant.cpp를 사용하면 세 논문 (HIGGS, TurboQuant, PolarQuant)과 본 저장소를 함께 인용해주세요.

---

<p align="center">
  <b><a href="https://quantumai.kr">QuantumAI</a></b> · <a href="https://github.com/quantumaikr/quant.cpp">GitHub</a>
</p>

<p align="center">
  <a href="https://star-history.com/#quantumaikr/quant.cpp&Date">
    <img src="https://api.star-history.com/svg?repos=quantumaikr/quant.cpp&type=Date" alt="Star History" width="600">
  </a>
</p>
