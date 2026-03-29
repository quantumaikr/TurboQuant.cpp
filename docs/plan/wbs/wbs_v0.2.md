# TurboQuant.cpp — Work Breakdown Structure v0.2

**Version**: 0.2
**Date**: 2026-03-29
**기반 문서**: [PRD v0.2](./prd_v0.2.md), refs/ absorption audit

---

## Phase A: Algorithm Optimization (최우선)

### A.1 QJL Direct Hamming Attention

- [x] `src/core/tq_qjl.c` — `tq_qjl_attention_hamming()` 구현
  - [x] Query projection: `q_sketch[i] = Σ(query[j] × proj[i][j])`
  - [x] Query sign hash: `q_hash = pack_sign_bits(q_sketch)`
  - [x] XOR + popcount: `hamming = popcount(q_hash XOR k_hash)`
  - [x] Score 재구성: `score = sqrt(π/2) / sketch_dim × norm × (sketch_dim - 2×hamming)`
  - [x] 아웃라이어 보정 추가
- [x] `tq_qjl_attention_ref()` 교체 — 기존 dequant+dot → Hamming 방식으로
- [x] `tests/test_qjl.cpp` — Hamming attention vs FP32 dot product 정확도 테스트 추가
- [x] `bench/tq_bench.cpp` — QJL Hamming attention throughput 측정 추가

### A.2 PolarQuant Direct Attention

- [x] `src/core/tq_polar.c` — `tq_polar_attention_direct()` 구현
  - [x] θ lookup table 생성: `cos_lut[q] = cos(tscale × q + tmn)`, `sin_lut` 동일
  - [x] ρ lookup table 생성: `radius_lut[q] = rscale × q + rmn`
  - [x] Query × cos/sin: `contrib = query[2d] × cos_lut[tq] + query[2d+1] × sin_lut[tq]`
  - [x] Gather by θ index + ρ 가중치: `score += contrib × radius_lut[rq]`
  - [x] 쌍별 합산: `total_score = Σ(score[d] for d in pairs)`
- [x] `tq_polar_attention_ref()` 교체 — 기존 dequant+dot → direct 방식
- [x] `tests/test_polar.cpp` — Direct attention 정확도 테스트 추가
- [x] 성능 비교: direct vs dequant+dot 벤치마크

---

## Phase B: SIMD Performance (고성능)

### B.1 NEON Optimized Attention

- [x] `src/backend/cpu/tq_neon.c` — `tq_uniform_4b_attention_neon()` (fused dequant+dot)
  - [ ] NEON `vld1q_u8` 로 양자화 데이터 로드
  - [ ] 인라인 dequant + `vfmaq_f32` dot product 융합
  - [ ] `vpaddq_f32` 수평 합산
- [x] `src/backend/cpu/tq_neon.c` — `tq_qjl_attention_neon()` (NEON popcount)
  - [ ] `veorq_u8` XOR + `vcntq_u8` popcount
  - [ ] `vpaddlq_u8` → `vpaddlq_u16` → `vpaddlq_u32` 합산 체인
- [x] SIMD speedup ≥ 4.0x 달성 확인

### B.2 Benchmark Enhancement

- [x] `bench/tq_bench.cpp` — 모든 양자화 타입별 throughput 측정
- [x] `bench/tq_bench.cpp` — NEON dequant+dot fused vs generic 비교
- [x] `bench/tq_bench.cpp` — attention_throughput > 50K 달성 확인

---

## Phase C: Cache Enhancement

### C.1 Copy-on-Write

- [x] `src/cache/tq_paged_cache.c` — `tq_cache_share_block()` 구현
  - [x] ref_count 증가
  - [x] 공유 블록 추적
- [x] `src/cache/tq_paged_cache.c` — 수정 시 CoW 트리거
  - [x] `ref_count > 1`이면 블록 복사 후 ref_count 감소
  - [x] 원본 블록은 다른 소유자가 계속 사용
- [x] `tests/test_paged_cache.cpp` — CoW 테스트 추가
  - [x] share → modify → verify original unchanged
  - [x] ref_count lifecycle 검증

### C.2 Progressive Re-compression

- [ ] `src/cache/tq_progressive.c` — Tier 1→2 재압축 로직 완성
  - [ ] warm_type(4bit) → cold_type(3bit) 변환
  - [ ] 기존 양자화 데이터를 역양자화 → 재양자화
- [ ] `tests/test_progressive.cpp` — 재압축 정확도 테스트

---

## Phase D: Thread Safety

### D.1 Threading Support

- [x] `src/core/tq_context.c` — context에 pthread_mutex 추가
  - [x] `tq_quantize_keys()` 호출 시 lock 획득
  - [x] Thread-local 임시 버퍼 (TLS key)
- [x] `tests/test_threading.cpp` — 멀티스레드 양자화/attention 테스트
  - [x] 4 스레드 동시 quantize
  - [x] 4 스레드 동시 attention
- [x] ThreadSanitizer 클린 통과 확인

---

## Phase E: Remaining WBS v0.1 Items

### E.1 Missing Files

- [x] `bench/accuracy/run_ruler.py` — RULER 벤치마크 스크립트
- [x] `bench/performance/bench_throughput.cpp` — 처리량 벤치마크
- [x] `bench/performance/bench_kernel.cpp` — 개별 커널 성능 측정

### E.2 Cross-Platform Verification

- [x] CPU Generic → CPU AVX2 결과 일치 검증 (CI에서)
- [ ] CPU Generic → CUDA 결과 일치 검증 (CUDA 환경에서)
- [x] CPU Generic → Metal 결과 일치 검증 (macOS에서)

---

## Phase 순서 및 의존성

```
Phase A (Algorithm) ──→ Phase B (SIMD) ──→ score.sh 측정
         │                                      │
Phase C (Cache) ──────────────────────→ ────────┤
         │                                      │
Phase D (Thread Safety) ──────────────→ ────────┤
         │                                      │
Phase E (Remaining) ──────────────────→ ────────┘
                                                │
                                          Final Score ≥ 0.99
```

Phase A와 C는 독립적으로 병렬 진행 가능.
Phase B는 A 완료 후 진행 (최적화된 attention 기반).
Phase D는 독립적.

---

## 완료 기준

- [x] `bash score.sh` 점수 ≥ 0.99
- [x] QJL Hamming attention: dequant 대비 5x+ 빠름
- [x] PolarQuant direct attention: dequant 대비 2x+ 빠름
- [x] SIMD speedup ≥ 4.0x
- [x] attention_throughput ≥ 50K queries/sec
- [x] 모든 테스트 통과 (ASan + UBSan + TSan 클린)
- [x] Copy-on-Write 동작 검증
- [x] WBS v0.1 미완료 항목 30개 중 20개 이상 완료
