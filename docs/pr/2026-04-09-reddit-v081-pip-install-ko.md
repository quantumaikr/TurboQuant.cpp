# Reddit r/LocalLLaMA — quantcpp v0.8.1 + `pip install` (KO)

**제안 제목:** `[Project] quantcpp 0.8.1 — 단일 헤더 KV 압축 LLM 엔진, 이제 PyPI에서`

**제안 flair:** `Resources` 또는 `Other`

---

## 본문

**quantcpp 0.8.1**을 발행했습니다 — KV 캐시 압축 연구에 집중한 단일 헤더 C 추론 엔진입니다. 이제 PyPI에서 설치할 수 있습니다:

```bash
pip install quantcpp
```

```python
from quantcpp import Model
m = Model("model.gguf")
print(m.ask("What is 2+2?"))
```

Linux x86_64, Linux aarch64, macOS arm64에 사전 빌드 휠 (CPython 3.9–3.13). 다른 플랫폼은 sdist로 fallback해서 `quant.h`를 자동 컴파일합니다 — 런타임 의존성 0.

### 무엇인가

- **단일 헤더 (`quant.h`, ~16K LOC, ~646 KB)** — C 프로젝트 어디든 파일 하나만 떨어뜨리면 동작. CMake 불필요, 서브모듈 불필요.
- **7가지 KV 캐시 양자화 타입** 한 엔진에. 모두 공개 논문(TurboQuant, PolarQuant, QJL)에서 재현 가능.
- **순수 C, 의존성 0** — C 컴파일러가 도는 곳이면 어디든: iOS, Android, WASM, 마이크로컨트롤러, MSVC.
- **다채널 배포**: PyPI, GGUF integration, llama.cpp PR 드래프트, single-header drop-in.

### 핵심 결과 (Llama 3.2 3B, M-series, CPU-only, 957 토큰 PPL eval, 3-run 평균)

| KV 타입 | tok/s | vs FP32 | PPL | ΔPPL | 압축비 |
|---|---:|---:|---:|---:|---:|
| FP32 | 17.93 | baseline | 13.56 | — | 1× |
| **turbo_kv_4b** | 18.13 | **+1.1%** ✅ | 14.08 | +3.8% | **7.1×** |
| turbo_kv_5b_fast 🆕 | 17.53 | −2.2% | 13.65 | +0.7% | 3.76× |
| turbo_kv_5b | 16.93 | −5.6% | 13.65 | +0.7% | 5.8× |

**`turbo_kv_4b`** 경로는 Apple Silicon에서 7.1× KV 압축에 fp32 *속도 패리티*를 달성합니다. 이를 가능케 한 커널은 NEON 명령어 하나 (`vqtbl1q_s8`) — 16-entry 코드북 룩업. 공개 Karpathy 루프 로그의 Round 10입니다. v0.8.0은 같은 패턴을 AVX2 (`_mm_shuffle_epi8`)로 포팅해서 Linux/Windows x86-64에도 적용했습니다.

### 우리가 주장하지 *않는* 것

- llama.cpp보다 GPU에서 빠르다고 주장하지 않습니다. llama.cpp + Metal/CUDA는 production throughput에서 5–10배 우위. 우리 가치는 dispatch overhead가 GPU compute를 넘어서는 **CPU/임베디드** 환경, 그리고 새 quant 방법을 빠르게 포팅하는 **연구 속도**입니다.
- llama.cpp 대체재가 아닙니다. llama.cpp 100+ archs vs 우리 7개.
- Python 바인딩의 0.8.x 기본값은 **`kv_compress=0`** (KV 압축 없음)입니다. CLI 바이너리는 모든 KV 타입에서 작동하고, 바인딩에 가져오는 것은 v0.8.2 (단일 헤더 재생성)로 트래킹됩니다. `pip install` 패키지는 로드 + 생성을 합니다. KV 압축은 다음 릴리스.

### 정직성 트랙 레코드

이는 0.6.x → 0.8.x 시리즈에서 프로젝트의 **6번째 자가 정정**입니다. v0.8.0 Python 바인딩의 두 버그(default 경로 abort, cross-heap `libc.free()`)를 발행 후 몇 시간 안에 클린 venv에서 end-user 시뮬레이션으로 직접 잡았습니다. v0.8.1이 핫픽스. PyPI 0.8.0은 yank 처리 중.

우리는 정정을 프로젝트의 1차 신뢰 자산으로 다루고, 기능과 동일하게 [CHANGELOG](https://github.com/quantumaikr/quant.cpp/blob/main/CHANGELOG.md)에 기록합니다.

### 링크

- **PyPI**: https://pypi.org/project/quantcpp/
- **GitHub**: https://github.com/quantumaikr/quant.cpp
- **재현 하네스**: 11라운드 Karpathy 루프 문서 (`bench/results/turboquant_reproduction.md`)
- **Karpathy 루프 스코어링**: 10년 포지션 가드를 포함한 6차원 (단일 헤더 LOC, zero-deps, 포팅된 논문, honest correction count) — 실패 시 CI 차단

### 피드백을 환영합니다

1. `pip install quantcpp` 후 **여러분의** OS/Python 버전에서 `Model("your.gguf").ask("hi")` 가 정상 종료하지 않으면 트레이스와 함께 이슈 등록해주세요. 휠 매트릭스는 오늘 기준 Linux x86_64/aarch64 + macOS arm64. 그 외는 sdist (소스 컴파일).
2. `TQ_TURBO_KV_4B`을 위한 llama.cpp PR 드래프트는 `docs/pr/2026-04-09-llama-cpp-pr-draft.md`. llama.cpp ggml 내부 경험이 있어서 실제 포팅을 공동 저자로 하실 분 환영합니다.

---

## 게시 전 체크리스트 (사용자)

- [ ] PyPI 0.8.0 yank 먼저 (https://pypi.org/manage/project/quantcpp/release/0.8.0/) → Options → Yank → 사유 입력
- [ ] 새 venv에서 `pip install quantcpp` 가 **0.8.1** 로 resolve되는지 확인
- [ ] 게시할 플랫폼에서 코드 스니펫 (`Model("file.gguf").ask("...")`) 한 번 더 테스트
- [ ] 제목 flair 결정 — `Resources` 가 mod 자동 삭제될 가능성 가장 낮음
- [ ] v0.8.1 릴리스 노트 링크 댓글로 핀
- [ ] "llama.cpp랑 어떻게 다른가" 답변 준비 — "주장하지 않는 것" 섹션 사용

## 응답 전략 메모

- "llama.cpp보다 느리다" → 동의, "주장하지 않는 것" 섹션 가리키기
- "그냥 llama.cpp + Python binding 아닌가" → 단일 헤더 (어떤 C 프로젝트에든 drop-in, 서브모듈/CMake 불필요), 7가지 KV quant 타입, 연구 속도 (KIVI/HIGGS 등) 설명
- Tim Dettmers, Amir Zandieh 등 quant 연구 저자가 댓글 → 정중하게 engage. 연구 속도 기둥의 실제 타깃 청중
- downvote brigade → 글 삭제 금지. 정직성 트랙 레코드가 해자, 삭제는 그것을 침식
