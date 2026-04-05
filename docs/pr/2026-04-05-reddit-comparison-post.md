# Reddit Post — quant.cpp vs every other engine (2026-04-05)

---

## English

**Title:** quant.cpp vs llama.cpp vs vLLM vs MLX vs ONNX RT — honest comparison table

**Body:**

I keep getting asked "why not just use llama.cpp?" so I made a comparison table.

|  | quant.cpp | llama.cpp | vLLM | MLX | ONNX RT |
|--|-----------|-----------|------|-----|---------|
| KV compression | **7x, +0% PPL** | +10.6% PPL | -- | -- | -- |
| Code size | **67K LOC** | 250K+ | 100K+ | 50K+ | 500K+ |
| Dependencies | **zero** | ggml | PyTorch | Apple fw | runtime |
| Embeddable | **single header** | -- | -- | -- | complex |
| WASM | **192KB** | -- | -- | -- | -- |
| GPU serving | basic | full | **best** | Metal | multi |

The short version:

- **llama.cpp** when you need speed
- **vLLM** when you need throughput
- **quant.cpp** when you need to fit more context in less memory — or embed LLM in your own app

quant.cpp is not trying to replace llama.cpp. It solves a different problem. If your bottleneck is context length on limited hardware, 7x KV compression at +0% PPL is something no other engine offers. If your bottleneck is tok/s, use llama.cpp — it's faster.

The other unique thing: `quant.h` is a single 15K-line C header. `#include` it, compile with `cc app.c -lm -lpthread`, done. Try doing that with any other engine on this list.

Source: [github.com/quantumaikr/quant.cpp](https://github.com/quantumaikr/quant.cpp)

---

## 한글

**Title:** quant.cpp vs llama.cpp vs vLLM vs MLX vs ONNX RT — 비교표

**Body:**

"llama.cpp 쓰면 되지 왜 새로 만들었냐"는 질문을 계속 받아서 비교표를 만들었습니다.

|  | quant.cpp | llama.cpp | vLLM | MLX | ONNX RT |
|--|-----------|-----------|------|-----|---------|
| KV 압축 | **7x, +0% PPL** | +10.6% PPL | -- | -- | -- |
| 코드 규모 | **67K LOC** | 250K+ | 100K+ | 50K+ | 500K+ |
| 외부 의존성 | **없음** | ggml | PyTorch | Apple fw | runtime |
| 임베딩 가능 | **싱글 헤더** | -- | -- | -- | 복잡 |
| WASM | **192KB** | -- | -- | -- | -- |
| GPU 서빙 | 기본 | 풀 | **최고** | Metal | 멀티 |

한 줄 요약:

- 속도가 필요하면 **llama.cpp**
- 처리량이 필요하면 **vLLM**
- 같은 메모리에서 더 긴 컨텍스트가 필요하거나, LLM을 내 앱에 넣고 싶으면 **quant.cpp**

quant.cpp는 llama.cpp를 대체하려는 게 아닙니다. 다른 문제를 풉니다. 제한된 하드웨어에서 컨텍스트 길이가 병목이라면, 품질 손실 없이 KV 7배 압축은 다른 엔진에 없는 기능입니다. tok/s가 병목이면 llama.cpp를 쓰세요 — 더 빠릅니다.

또 하나 독특한 점: `quant.h`는 15K줄짜리 C 헤더 파일 하나입니다. `#include`하고 `cc app.c -lm -lpthread`로 컴파일하면 끝. 이 표의 다른 엔진으로는 불가능한 일입니다.

소스: [github.com/quantumaikr/quant.cpp](https://github.com/quantumaikr/quant.cpp)
