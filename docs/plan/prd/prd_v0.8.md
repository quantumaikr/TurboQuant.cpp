# TurboQuant.cpp — Product Requirements Document v0.8

**Version**: 0.8
**Date**: 2026-03-29
**Focus**: 자체 추론 엔진 — 모델 로드부터 토큰 생성까지, 외부 의존성 없이

---

## 1. Problem

현재 TurboQuant.cpp는 KV 캐시 압축 **라이브러리**일 뿐이다. 실제 추론은 PyTorch에 의존하며, CPU에서 0.8 tok/s로 매우 느리다.

refs/ 프로젝트들의 강점을 융합한 **자체 추론 엔진**이 필요하다:

| refs/ 프로젝트 | 가져올 강점 | 현재 구현 |
|---------------|-----------|----------|
| **llama.cpp** | 순수 C 추론, GGUF 로더, NEON matmul | ❌ 없음 |
| **vLLM** | PagedAttention, 퓨전 커널 | ⚠️ 캐시만 |
| **ONNX** | 표준 연산자, 포맷 호환 | ⚠️ 비트패킹만 |

## 2. Goal

**순수 C로 구현된 최소 추론 엔진** — Qwen3.5-0.8B를 외부 의존성 없이 CPU에서 **10+ tok/s**로 실행.

참조: Karpathy의 llama2.c (순수 C, ~700줄, 외부 의존성 없음)

## 3. Architecture

```
┌─────────────────────────────────────────────┐
│ tq_generate()  — Autoregressive decode loop │
├─────────────────────────────────────────────┤
│ tq_forward()   — Single forward pass        │
│   ├── RMSNorm                               │
│   ├── QKV Projection (matmul)               │
│   ├── RoPE (rotary position embedding)      │
│   ├── KV Cache (TurboQuant quantized!)      │
│   ├── Attention (integer Q4×Q8!)            │
│   ├── Output Projection (matmul)            │
│   ├── FFN: gate/up → SiLU → down           │
│   └── Residual connections                  │
├─────────────────────────────────────────────┤
│ tq_load_model() — Weight loader             │
│   ├── safetensors / GGUF / custom format    │
│   └── Weight quantization (Q4/Q8)           │
├─────────────────────────────────────────────┤
│ tq_tokenize()  — BPE tokenizer              │
├─────────────────────────────────────────────┤
│ tq_sample()    — Top-p, temperature         │
└─────────────────────────────────────────────┘
```

핵심: **KV 캐시에 TurboQuant 양자화가 내장**된 추론 엔진. 기존 라이브러리의 모든 기능(정수 attention, RHT, mixed precision, progressive compression)이 추론 파이프라인 안에서 동작.

## 4. Requirements

### FR-V8-1: 텐서 연산 (src/engine/tq_ops.c)

llama.cpp GGML 패턴 참조:
- `tq_matmul()` — 행렬-벡터 곱 (가중치 × 활성화), NEON 최적화
- `tq_rmsnorm()` — RMS normalization
- `tq_rope()` — Rotary Position Embedding
- `tq_silu()` — SiLU activation (x * sigmoid(x))
- `tq_softmax()` — Softmax (attention scores)
- `tq_add()` — 잔차 연결

### FR-V8-2: 모델 로더 (src/engine/tq_model.c)

- safetensors 포맷 읽기 (Qwen3.5-0.8B 호환)
- 가중치를 FP32 또는 Q8로 로드
- 모델 구조 자동 감지 (config.json 파싱)
- mmap 지원 (대용량 모델 메모리 효율)

### FR-V8-3: Transformer 블록 (src/engine/tq_transformer.c)

Qwen3.5-0.8B의 Gated Attention 레이어:
```
input → RMSNorm → QKV_proj → RoPE → Attention → O_proj → residual
      → RMSNorm → gate_proj + up_proj → SiLU → down_proj → residual
```

KV 캐시에 TurboQuant 양자화 자동 적용:
- 새 키 생성 → `tq_quantize_keys()` → 양자화 캐시에 저장
- Attention 계산 → `tq_attention_int()` → 정수 도메인에서 직접 계산

### FR-V8-4: 토크나이저 (src/engine/tq_tokenizer.c)

- BPE 토크나이저 (tokenizer.json 로드)
- encode: 문자열 → 토큰 ID
- decode: 토큰 ID → 문자열

### FR-V8-5: 생성 루프 (src/engine/tq_generate.c)

- Autoregressive decode: 한 토큰씩 생성
- Prefill: 프롬프트 전체를 한번에 처리
- Sampling: temperature, top-p, top-k
- 스트리밍 출력 (토큰 생성 즉시 출력)

### FR-V8-6: CLI (tools/tq_run)

```bash
# 모델 실행
tq_run --model qwen3.5-0.8b.safetensors --prompt "What is deep learning?"

# 옵션
tq_run --model MODEL --prompt TEXT \
       --kv-type uniform_4b \       # KV 캐시 양자화 타입
       --max-tokens 100 \
       --temperature 0.7 \
       --threads 4
```

## 5. Success Criteria

| 지표 | 목표 |
|------|------|
| CPU 추론 속도 | **10+ tok/s** (현재 PyTorch: 0.8 tok/s) |
| MPS 추론 속도 | **30+ tok/s** |
| 메모리 사용량 | **< 2 GB** (Qwen3.5-0.8B Q8 가중치 + KV 캐시) |
| 외부 의존성 | **0개** (libc/libm만) |
| KV 캐시 압축 | 기존 TurboQuant 전체 기능 내장 |
| 정확도 | PyTorch FP32 대비 동일 텍스트 생성 |

## 6. Scope

### v0.8.0 (최소 동작)
- Qwen3.5-0.8B Gated Attention 레이어만 지원
- safetensors 로더
- FP32 가중치 + TurboQuant KV 캐시
- CPU 추론 (NEON 최적화)
- 기본 BPE 토크나이저

### v0.8.1 (최적화)
- Q8 가중치 양자화 (메모리 절반)
- NEON matmul 최적화
- 멀티스레드 prefill

### v0.9+ (확장)
- DeltaNet 레이어 지원 (Qwen3.5 전체)
- Metal GPU 추론
- GGUF 호환
- 다른 모델 아키텍처 (Llama, Phi)
