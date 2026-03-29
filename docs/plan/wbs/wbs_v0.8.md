# TurboQuant.cpp — Work Breakdown Structure v0.8

**Version**: 0.8
**Date**: 2026-03-29
**Focus**: 자체 추론 엔진 구현

---

## Phase 1: 텐서 연산 커널

- [ ] `src/engine/tq_ops.c` — 핵심 연산
  - [ ] `tq_matmul(out, x, w, n, d)` — 행렬-벡터 곱 (w[n,d] × x[d] → out[n])
  - [ ] `tq_matmul_neon()` — ARM NEON 최적화 (vfmaq_f32, 4-wide)
  - [ ] `tq_rmsnorm(out, x, weight, n, eps)` — RMS normalization
  - [ ] `tq_rope(q, k, pos, head_dim, freq_base)` — Rotary Position Embedding
  - [ ] `tq_silu(x, n)` — SiLU activation (in-place)
  - [ ] `tq_softmax(x, n)` — Softmax
  - [ ] `tq_add(out, a, b, n)` — 벡터 덧셈
  - [ ] `tq_mul(out, a, b, n)` — 벡터 곱셈 (element-wise)
- [ ] `include/turboquant/tq_engine.h` — 추론 엔진 헤더
- [ ] `tests/test_ops.cpp` — 연산 단위 테스트

---

## Phase 2: 모델 로더

- [ ] `src/engine/tq_model.c` — safetensors 로더
  - [ ] safetensors 헤더 파싱 (JSON 메타데이터)
  - [ ] 텐서 데이터 mmap 로드
  - [ ] 모델 구조 정의 (tq_model_t)
    ```c
    typedef struct {
        int n_layers, n_heads, n_kv_heads, head_dim;
        int hidden_dim, intermediate_dim, vocab_size;
        float rope_freq_base;
        float* token_embedding;  // [vocab_size, hidden_dim]
        struct { // per layer
            float* attn_norm, *ffn_norm;
            float* wq, *wk, *wv, *wo;
            float* w_gate, *w_up, *w_down;
        } layers[];
    } tq_model_t;
    ```
  - [ ] `tq_load_model(path)` → `tq_model_t*`
  - [ ] `tq_free_model(model)`
- [ ] `tests/test_model_load.cpp` — 로더 테스트

---

## Phase 3: Transformer Forward Pass

- [ ] `src/engine/tq_transformer.c` — forward pass
  - [ ] `tq_forward(model, token, pos, kv_cache)` → logits
  - [ ] Attention 블록:
    ```
    x → RMSNorm → Q,K,V projection (matmul)
    Q,K → RoPE
    K,V → TurboQuant KV cache (quantize + store)
    Q × KV_cache → attention scores (integer Q4×Q8!)
    scores → softmax → weighted sum of V
    → output projection → residual add
    ```
  - [ ] FFN 블록:
    ```
    x → RMSNorm → gate_proj + up_proj (matmul)
    gate → SiLU
    gate × up → down_proj (matmul) → residual add
    ```
  - [ ] KV 캐시 통합: `tq_quantize_keys()` 자동 호출
- [ ] `tests/test_forward.cpp` — forward pass 정확도 테스트

---

## Phase 4: 토크나이저

- [ ] `src/engine/tq_tokenizer.c` — BPE 토크나이저
  - [ ] tokenizer.json 파싱 (vocab + merges)
  - [ ] `tq_encode(text, tokens, max_tokens)` → token count
  - [ ] `tq_decode(token_id)` → string
  - [ ] 특수 토큰 처리 (BOS, EOS, PAD)
- [ ] `tests/test_tokenizer.cpp` — 토크나이저 테스트

---

## Phase 5: 생성 루프 + CLI

- [ ] `src/engine/tq_generate.c` — autoregressive 생성
  - [ ] `tq_generate(model, prompt, config)` → generated text
  - [ ] Prefill: 프롬프트 전체 forward
  - [ ] Decode: 한 토큰씩 생성
  - [ ] Sampling: temperature, top-p
  - [ ] 스트리밍: 토큰 생성 즉시 콜백
- [ ] `tools/tq_run.c` — CLI 실행 파일
  ```bash
  tq_run --model model.safetensors --prompt "Hello" --kv-type uniform_4b
  ```
- [ ] 벤치마크: tok/s 측정

---

## Phase 6: 검증

- [ ] PyTorch 대비 출력 비교 (동일 프롬프트 → 유사 로짓)
- [ ] 속도: CPU 10+ tok/s 달성 확인
- [ ] 메모리: < 2 GB 확인
- [ ] KV 캐시: TurboQuant 양자화 동작 확인
- [ ] 정수 attention: 실제 추론에서 사용 확인

---

## 완료 기준

- [ ] `tq_run --model qwen3.5-0.8b --prompt "What is AI?"` 실행 → 텍스트 생성
- [ ] CPU 10+ tok/s
- [ ] 외부 의존성 0 (libc/libm만)
- [ ] KV 캐시에 TurboQuant uniform_4b 자동 적용
- [ ] 정수 Q4×Q8 attention이 실제 추론에서 동작
