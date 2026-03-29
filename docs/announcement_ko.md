# TurboQuant.cpp — 극한 KV 캐시 압축을 내장한 LLM 추론 엔진

순수 C로 LLM 추론 엔진을 직접 구현했습니다. Qwen3.5-0.8B를 CPU에서 **14 tok/s**로 실행합니다 — PyTorch보다 17배 빠르고, PyTorch+GPU보다도 빠릅니다.

## 숫자로 보기

```
PyTorch (CPU):     0.8 tok/s
PyTorch (MPS GPU): 10  tok/s
TurboQuant (CPU):  14  tok/s  ← GPU보다 빠름, 의존성 없음
```

가중치 메모리: 1.7 GB → **533 MB** (Q8 양자화).
KV 캐시: **7.5배 압축**, 99.4% attention 정확도.

## 하는 일

하나의 바이너리. Python 없음. 모델 로드, 텍스트 생성:

```bash
./tq_run model.safetensors -t tokenizer.json -p "What is AI?" -j 4 -q
```

출력:
```
Artificial intelligence is a field of computer science...
100 tokens in 7.2s (13.9 tok/s, 4 threads, kv=uniform_4b)
```

## 어떻게

- safetensors 모델 로더 (mmap, zero-copy)
- DeltaNet + Self-Attention 하이브리드 forward pass (Qwen3.5 아키텍처)
- NEON 최적화 matmul (4-accumulator, 멀티스레드)
- 정수 Q4×Q8 attention (FP32 대비 2.9배 빠름)
- Q8 가중치 양자화 (메모리 4배 절약)
- HuggingFace BPE 토크나이저 (248K 어휘)
- 스트리밍 토큰 출력

TurboQuant (ICLR 2026), QJL (AAAI 2025), PolarQuant (AISTATS 2026) 논문 기반.
llama.cpp, vLLM, ONNX의 아키텍처 패턴 흡수.

8,500줄 C 코드. 70+ 테스트. Apache 2.0.

https://github.com/quantumaikr/TurboQuant.cpp

---

**개발사: [QuantumAI Inc.](https://quantumai.kr)** | hi@quantumai.kr
