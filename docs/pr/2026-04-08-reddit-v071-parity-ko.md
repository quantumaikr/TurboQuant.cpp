# r/LocalLLaMA — quant.cpp v0.7.1 — KV 캐시 압축 + fp32 KV 속도 (단일 헤더 C, 11 카파시 라운드)

## 제목 (≤ 300자)

quant.cpp v0.7.1: 단일 헤더 C KV 캐시 양자화기를 4 세션 동안 최적화했습니다. Round 10에서 마침내 Llama 3.2 3B에서 fp32 KV 속도 parity + 7.1× 압축을 달성했습니다. publishing 전에 catch한 4번의 정직한 정정을 포함한 솔직한 리뷰.

## 본문

**TL;DR**: 단일 헤더 (628 KB) C KV 캐시 양자화 reference 엔진. 11 라운드 Karpathy 루프 후 `turbo_kv_4b`가 압축 없는 FP32 KV 속도와 동등 (−1.4% within noise) **7.1× 메모리 압축** + Llama 3.2 3B에서 **+3.8% PPL** trade-off. CPU 전용 빌드, iOS/Android/WASM/MSVC/마이크로컨트롤러에서 동작. Apache 2.0. https://github.com/quantumaikr/quant.cpp

---

### 이게 뭔가요

quant.cpp는 제가 작업해 온 작은 C 추론 엔진으로 **KV 캐시 양자화 연구**에 집중합니다. [TurboQuant 논문 (Zandieh et al., ICLR 2026)](https://arxiv.org/abs/2504.19874)의 literal port로 시작해서, 11 라운드의 measurement-driven iteration을 거쳐 더 단순한 무언가로 수렴했고, 공유하고 싶었습니다.

차별점은 **단일 헤더 portability**입니다. 전체 엔진이 한 개 628 KB `quant.h` 파일이고, 어느 C/C++ 프로젝트에든 드롭할 수 있습니다 (Cargo 없음, Python 없음, PyTorch 없음, 프레임워크 없음). `cc app.c -lm -lpthread`로 빌드하면 7× 압축된 KV 캐시로 LLM이 동작합니다. iOS, Android, WASM (192 KB 바이너리), MSVC, 마이크로컨트롤러에서 동작.

### 핵심 결과 (Llama 3.2 3B Instruct, CPU-only 빌드, 3-run 평균)

| KV 타입 | 블록 바이트 | 압축 | PPL | Δ vs FP32 | tok/s | vs FP32 속도 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 KV | — | 1× | 13.56 | — | 18.43 | baseline |
| **`turbo_kv_4b`** ⭐ 기본 | **72** | **7.1×** | 14.08 | **+3.8%** | **18.17** | **−1.4%** ✅ |
| `turbo_kv_5b` 🏆 quality | 88 | 5.8× | 13.65 | **+0.7%** | 16.80 | −8.8% |
| `turbo_kv_3b` | 56 | 9.1× | 15.36 | +13.3% | 16.57 | −10.1% |
| `uniform_4b` (legacy) | 68 | 7.5× | 14.60 | +7.7% | 13.27 | −26.8% |

`turbo_kv_4b`는 이제 모든 면에서 `uniform_4b`를 Pareto-dominate (더 나은 PPL, 더 빠른 속도, 비슷한 압축). 그리고 7.1× 압축하면서 **fp32 KV 속도와 동등**.

### 여정 (11 라운드, 4 세션, 4번의 정직한 정정)

이건 "짠, 만들었어요" 글이 아닙니다. **measurement discipline의 기록**입니다.

**Round 0** — TurboQuant literal port: PPL 16.03, `uniform_4b`보다 훨씬 느림. 부끄럽습니다.

**Round 6 (Variant F)** — Karpathy ablation으로 QJL 잔차 단계가 attention 점수에 *byte-identical 0* 기여한다는 것을 발견. 그것을 제거하고, 블록당 16 바이트를 더 큰 Lloyd-Max 코드북에 재투자 (3-bit → 4-bit, 8 → 16 levels). PPL 16.03 → 14.28. tuning이 아닌 구조적 단순화.

**Rounds 7–9** — Local fusion, NEON unroll, LUT hoisting, prefetch. 각각 최대 +5%만. fp32 대비 −7%에 막힘.

**Round 10 — 돌파**. 세 세션 동안 추측한 후, 마침내 기존 `--profile` 플래그를 실행했습니다. 데이터는 분명했습니다: matmul은 fp32와 quant 사이에서 동일했습니다 (38.6 vs 38.9 ms, 둘 다 같은 NEON tbl matmul 커널 공유). 전체 8% 속도 격차는 attention dot-product 루프 안에 있었습니다. fp32 path는 4-way NEON SIMD였고, 제 것은 scalar였습니다. 요소당 ~2× 더 많은 instructions. **Memory-bound가 아닌 compute-bound** — 16-entry LUT으로는 예상 밖.

해법: Apple Silicon의 `vqtbl1q_s8`, 16 byte-table lookups를 16 lanes에 걸쳐 하나의 명령으로 실행. 16 Lloyd-Max-Gaussian 센트로이드를 시작 시점에 int8으로 양자화 (~1% 정밀도 손실, regression test cosine ≥ 0.99 임계치보다 훨씬 낮음), 16-byte 레지스터에 저장하면 inner loop가:

```c
uint8x16_t bytes = vld1q_u8(mi);                    // 16B = 32 nibbles
uint8x16_t low_nib  = vandq_u8(bytes, vdupq_n_u8(0x0F));
uint8x16_t high_nib = vshrq_n_u8(bytes, 4);
int8x16_t low_vals  = vqtbl1q_s8(cb_vec, low_nib);  // 1 instr, 16 gathers
int8x16_t high_vals = vqtbl1q_s8(cb_vec, high_nib);
// ... interleave + int8→fp32 + per-block scale + vfmaq_f32
```

inner loop iteration당 32 elements (이전 scalar 버전의 8 elements와 비교). 결과: **fp32 parity**, single representative run에서 +4.5%, 3-run 평균에서 +0.8%. PPL도 약간 개선 (int8 코드북 discretization이 우연히 favorably align).

**Round 11 (v0.7.1)**은 같은 패턴을 5b/3b에 적용. lookup side는 잘 scale 합니다 (어떤 작은 codebook이든 16 lanes당 1 instruction) 하지만 **bit-unpack side**가 새로운 bottleneck: 5-bit과 3-bit 인덱스가 byte 경계를 불규칙하게 걸쳐서 16 indices의 unpack은 scalar shifts가 필요. 5b는 −14.5%에서 −8.8%로 (+9% speed jump), 3b는 −13%에서 −10%로. Full parity는 아니지만 의미 있음.

### 정직한 정정 기록 (4개 사건)

저는 인플레된 "lossless 7×" claim으로 시작해서 widely publishing 전에 4번 walk back 했습니다. 각 정정은 영구 메모리에 기록된 교훈을 가르쳤습니다:

1. **v0.6.0** "lossless 7× compression" → 측정 후 "+6.3% PPL on Llama 3.2 3B"
2. **v0.6.4** "turbo_kv beats fp32 KV speed" → fp32 attention path가 unoptimized scalar임을 발견; 양쪽 모두 NEON 추가 후 정직한 격차는 −7%
3. **v0.6.5** "with Metal" → 기존 Metal 백엔드가 SmolLM 135M부터 Gemma 4 26B까지 모든 모델 사이즈에서 *net negative* (13–40% 더 느림)임을 발견. CMake 기본값이 OFF지만 우리 내부 벤치마크가 5 릴리스 동안 14–22% 잘못되었습니다. [Issue #16 작성](https://github.com/quantumaikr/quant.cpp/issues/16).
4. **v0.6.5 post**: [@TimDettmers](https://github.com/TimDettmers) (HIGGS / QLoRA / bitsandbytes)가 [llama.cpp discussion thread](https://github.com/ggml-org/llama.cpp/discussions/20969)에서 코멘트 — 우리에게 직접 한 게 아니지만 substance가 적용됨 — 우리가 "TurboQuant"라고 부르던 RHT + scalar grid 패턴이 실제로는 HIGGS (Malinovskii et al., Nov 2024)에서 origin. 24시간 안에 모든 docs에 HIGGS credit을 추가했고, 사용자가 우리가 관계를 overstate 했다고 지적한 후 "Tim gave us feedback"을 "Tim's general comment we observed"로 reframe.

위 어떤 숫자에 회의적이라면, **모든 측정값은 재현 가능**합니다: `cmake -B build && cmake --build build && ./build/quant model.gguf --ppl bench/data/ppl_1k.txt -k turbo_kv_4b`.

### 정직한 framing (이게 아닌 것)

- **TurboQuant 구현이 아닙니다.** Ablation으로 published 논문이 사용하는 QJL residual과 per-channel outlier handling을 모두 제거했습니다. 우리가 ship하는 것은 TurboQuant보다 HIGGS (RHT + scalar grid quantization)에 구조적으로 더 가깝습니다. 둘 다 우리 docs에 credit 됨.
- **가장 빠른 GPU 추론이 아닙니다.** llama.cpp가 그 자리를 full Metal/CUDA tensor graphs로 차지. 우리는 CPU 전용이고 그것에 자부심.
- **가장 feature-complete가 아닙니다.** 7개 아키텍처 검증, 100+ 아님. 단일 헤더 제약이 많은 features를 배제.
- **아직 Llama 3.1 8B (paper baseline)에서 검증 안 됨.** 시도했으나 — Q8_0가 16 GB RAM에서 swap, Q4_K_M이 prohibitively 느림. TODO로 추적 중.
- **5b/3b는 아직 parity 아님.** Round 11이 격차를 크게 close했지만 −9% / −10%에 있습니다. Future work.

### Cross-size 검증 (3개 Llama 패밀리 모델, 모두 CPU 전용)

| 모델 | turbo_kv_4b PPL Δ | turbo_kv_5b PPL Δ |
|---|---|---|
| SmolLM2 135M | +5.8% | +1.7% |
| Llama 3.2 1B | +7.3% | **+0.7%** |
| Llama 3.2 3B | +5.7% | **+0.7%** |

`turbo_kv_5b`는 모든 모델 사이즈에서 일관되게 near-lossless (~1% PPL Δ).

### 사용해 보세요

```bash
git clone https://github.com/quantumaikr/quant.cpp
cd quant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release   # 기본값: TQ_BUILD_METAL=OFF
cmake --build build -j

# 작은 모델 다운로드
hf download bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf --local-dir models/

./build/quant models/SmolLM2-135M-Instruct-Q8_0.gguf --chat -p "안녕!" -j 8
```

`turbo_kv_4b`가 기본값. near-lossless 품질에는 `-k turbo_kv_5b`, 최대 압축에는 `-k turbo_kv_3b`.

### 가치가 어디 있는가

솔직히, fp32 parity에서 7.1× 압축이 헤드라인 숫자입니다. 하지만 4 세션 후, 더 가치 있다고 생각하는 것은 **measurement transparency**입니다. 모든 claim이 reproduction script로 링크됩니다. 모든 release notes가 이전 release의 정정을 언급합니다. commit hashes와 함께 11-라운드 Karpathy history는 [`bench/results/turboquant_reproduction.md`](https://github.com/quantumaikr/quant.cpp/blob/main/bench/results/turboquant_reproduction.md)에 있습니다. 미래 paper가 "single-header C reference implementation of HIGGS-style KV quantization"을 cite하고 싶다면, 이게 그것입니다.

### 로드맵 (다음 세션들)

- v0.7.2: 5b 1-byte-per-index variant for full parity (compression을 speed로 trade)
- v0.8.0: NEON tbl 패턴의 AVX2 + WASM SIMD 포팅
- v0.9.0: fp32 능가 가능성을 위한 `vusdotq` 탐색 (ARMv8.6+)
- v1.0.0: arXiv 제출 + spec compliance test suite + llama.cpp PR

### 링크

- 저장소: https://github.com/quantumaikr/quant.cpp
- v0.7.1 릴리스 노트: https://github.com/quantumaikr/quant.cpp/releases/tag/v0.7.1
- Round 10 commit: https://github.com/quantumaikr/quant.cpp/commit/2537a12
- 우리가 참여 중인 llama.cpp discussion thread: https://github.com/ggml-org/llama.cpp/discussions/20969
- Reproduction history: https://github.com/quantumaikr/quant.cpp/blob/main/bench/results/turboquant_reproduction.md

비판적 피드백 환영. 특히:
- 동일 하드웨어에서 Cross-implementation 비교 (MLX, Rust forks, llama.cpp turboquant forks)
- 32+ GB 박스에서 quant.cpp로 Llama 3.1 8B를 돌려본 분
- 같은 패턴의 AVX2 / SIMD128 구현
- 5b/3b unpack bottleneck 제안 (SIMD bit-extraction tricks?)
