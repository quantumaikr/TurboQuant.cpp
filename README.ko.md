<p align="center">
  <img src="docs/assets/hero.png" alt="quant.cpp" width="600">
</p>

<h3 align="center">7배 긴 컨텍스트를 만드는 LLM 추론 엔진 — 순수 C, 의존성 제로</h3>

<p align="center">
  무손실 KV 캐시 압축. <a href="#-단일-헤더-모드"><b>quant.h</b></a> 단일 헤더 라이브러리로도 제공됩니다.<br>
  72K LOC. 임베딩 가능. 오후 한나절이면 전체 코드를 읽을 수 있습니다.
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

> **같은 하드웨어. 7배 긴 컨텍스트. 품질 손실 제로.**

| 하드웨어 | 모델 | FP16 KV | quant.cpp KV | 배율 |
|:---------|:------|--------:|-------------:|-----:|
| 16GB Mac | Llama 3.2 3B | 50K 토큰 | **350K 토큰** | **6.9x** |
| 16GB Mac | Gemma 4 26B MoE | 4K 토큰 | **30K 토큰** | **6.9x** |
| 8GB 노트북 | Llama 8B (Q4) | 16K 토큰 | **61K 토큰** | **3.8x** |
| 24GB RTX 3090 | Llama 8B (Q4) | 147K 토큰 | **559K 토큰** | **3.8x** |

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

> **[API 레퍼런스](docs/api.md)** · **[WASM 데모](https://quantumaikr.github.io/quant.cpp/)** · **[커스텀 양자화 가이드](docs/custom-quantization.md)**

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

### vs 다른 엔진들

|  | quant.cpp | llama.cpp | vLLM | MLX | ONNX RT |
|:--|:---------:|:---------:|:----:|:---:|:-------:|
| KV 압축 | **3.8-6.9x, +0% PPL** | 1.6x ~+1% PPL | -- | -- | -- |
| 코드 크기 | **72K LOC** | 250K+ | 100K+ | 50K+ | 500K+ |
| 의존성 | **제로** | ggml | PyTorch | Apple fw | 런타임 |
| 임베더블 | **단일 헤더** | -- | -- | -- | 복잡 |
| WASM | **192KB** | -- | -- | -- | -- |
| GPU 서빙 | 기본 | 풀 | **최고** | Metal | 다양 |

> **속도**가 필요하면 llama.cpp. **처리량**이 필요하면 vLLM.
> **같은 메모리에서 더 긴 컨텍스트**가 필요하거나, **앱에 LLM을 임베딩**하려면 quant.cpp.

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

```bash
# Delta 압축 (최대 컨텍스트, 8.5x)
./build/quant model.gguf --chat -p "hello" -k uniform_3b -v q4 --delta

# PPL 벤치마크
./build/quant model.gguf --ppl input.txt -k uniform_4b -v q4

# 모델 정보
./build/quant model.gguf --info

# 성능 프로파일링
./build/quant model.gguf --chat -p "hello" -n 50 --profile
```

---

## 단일 헤더 모드

> 파일 1개 복사. 어떤 C 프로젝트에든 LLM 추가.

```c
#define QUANT_IMPLEMENTATION
#include "quant.h"

int main() {
    quant_model* m = quant_load("model.gguf");
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
docker run -v ./models:/models quant.cpp /models/model.gguf -p "hello" -k uniform_4b -v q4
```

**OpenAI 호환 서버** (`/v1/chat/completions`):
```bash
cmake -B build -DTQ_BUILD_SERVER=ON && cmake --build build
./build/quant-server model.gguf -p 8080 -k uniform_4b

# OpenAI Python SDK와 호환
curl http://localhost:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

`-DTQ_BUILD_SERVER=ON`으로 빌드. SSE 스트리밍 지원. 요청별 KV 압축 설정 가능.

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
| **[로드맵](ROADMAP.md)** | 프로젝트 방향과 계획 |
| **[변경 이력](CHANGELOG.md)** | 버전별 릴리스 노트 |
| **[기술 리포트](docs/papers/quant_cpp_tech_report.md)** | 아키텍처와 벤치마크 (Arxiv 초안) |
| **[WASM 데모](https://quantumaikr.github.io/quant.cpp/)** | 브라우저에서 바로 체험 — 설치 불필요 |

---

## 참고 논문

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — KV 캐시 압축 이론
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — 양자화 JL 변환
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — 극좌표 양자화

---

<p align="center">
  <b><a href="https://quantumai.kr">QuantumAI</a></b> · <a href="https://github.com/quantumaikr/quant.cpp">GitHub</a>
</p>

<p align="center">
  <a href="https://star-history.com/#quantumaikr/quant.cpp&Date">
    <img src="https://api.star-history.com/svg?repos=quantumaikr/quant.cpp&type=Date" alt="Star History" width="600">
  </a>
</p>
