# TurboQuant.cpp — Product Requirements Document v0.2

**Version**: 0.2
**Date**: 2026-03-29
**Status**: Active Development
**Based on**: v0.1 + refs/ absorption audit + gap analysis

---

## 1. v0.2 Goal

v0.1에서 핵심 알고리즘과 구조를 구축했다. v0.2의 목표는 **성능 최적화와 누락 기능 구현**이다.

### v0.2 Target Metrics

| 지표 | v0.1 현재 | v0.2 목표 |
|------|----------|----------|
| Score (harness) | 0.9775 | > 0.99 |
| QJL attention | dequant+dot (느림) | Direct Hamming (10x faster) |
| PolarQuant attention | dequant+dot | Direct cos/sin lookup |
| SIMD speedup | 3.97x | > 4x (NEON fine-tune) |
| attention_throughput | 14.5K/s | > 50K/s |
| Copy-on-Write | Not impl | Working |
| Metal backend | Stub only | Compilable shaders |
| Thread safety | Not verified | TSan-clean |

---

## 2. Critical Gaps from v0.1 Audit

### 2.1 Algorithm Optimization (Highest Impact)

**GAP-1: QJL Direct Hamming Attention**
- 현재: 전체 벡터 복원 후 dot product (O(head_dim) per key)
- 목표: XOR + popcount로 직접 score 계산 (O(sketch_dim/8) per key)
- 참조: `refs/QJL/qjl_kernel/csrc/qjl_score_kernel.cu` lines 130-157
- 공식: `score = sqrt(π/2) / sketch_dim × norm × (sketch_dim - 2 × hamming_distance)`
- 예상 성능 개선: attention throughput 5-10x

**GAP-2: PolarQuant Direct Attention**
- 현재: dequantize → dot product (2-pass)
- 목표: cos/sin lookup table → gather by θ index → ρ weighting (1-pass)
- 참조: `refs/PolarQuant/models/kernel4group.py` lines 44-77
- 예상 성능 개선: attention throughput 2-3x

### 2.2 Cache Management

**GAP-3: Copy-on-Write for Beam Search**
- 현재: `ref_count` 필드 존재하지만 로직 없음
- 목표: `tq_cache_copy_block()` → ref_count 증가, 수정 시 실제 복사
- 참조: `refs/vllm/csrc/cache_kernels.cu` copy_blocks_kernel

**GAP-4: Progressive Re-compression (Tier 1→2)**
- 현재: Tier 0→1 전환만 구현
- 목표: Tier 1(4bit) → Tier 2(3bit) 재압축 완전 구현

### 2.3 Thread Safety

**GAP-5: Thread-Safe API**
- 현재: 단일 스레드 가정
- 목표: `tq_context_t` 내부 mutex, thread-local 임시 버퍼
- 검증: ThreadSanitizer 클린

### 2.4 Performance Fine-tuning

**GAP-6: NEON Attention Kernel**
- 현재: NEON은 quantize/dequantize만
- 목표: NEON attention kernel (dot product + dequant 융합)

**GAP-7: SIMD 4x+ Speedup**
- 현재: 3.97x (4x 미달)
- 목표: NEON dequant+dot 최적화로 4x 이상 달성

---

## 3. Functional Requirements (v0.2)

### FR-V2-1: Direct Hamming Attention for QJL

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-V2-1.1 | Query를 projection matrix로 투영하여 sketch 생성 | P0 |
| FR-V2-1.2 | XOR(query_hash, key_hash) + popcount로 hamming distance 계산 | P0 |
| FR-V2-1.3 | `score = sqrt(π/2)/d × norm × (d - 2×hamming)` 공식 적용 | P0 |
| FR-V2-1.4 | 아웃라이어 보정: `+ outlier_norm × Σ(query[idx] × proj[idx])` | P1 |
| FR-V2-1.5 | CPU 레퍼런스 + NEON popcount 최적화 | P0 |

### FR-V2-2: Direct PolarQuant Attention

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-V2-2.1 | θ 양자화 레벨별 cos/sin lookup table 사전 생성 | P0 |
| FR-V2-2.2 | query × interleave(cos(θ), sin(θ)) → gather by θ index | P0 |
| FR-V2-2.3 | ρ 양자화 레벨별 radius table → gather by ρ index → 가중치 적용 | P0 |
| FR-V2-2.4 | 쌍별 합산으로 최종 attention score 도출 | P0 |

### FR-V2-3: Copy-on-Write Cache

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-V2-3.1 | `tq_cache_share_block()` — ref_count 증가 | P1 |
| FR-V2-3.2 | 수정 시 `ref_count > 1`이면 실제 복사 후 수정 | P1 |
| FR-V2-3.3 | `tq_cache_free_block()` — ref_count 감소, 0이면 해제 | P1 |

### FR-V2-4: Thread Safety

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-V2-4.1 | `tq_context_t`에 mutex 추가 | P1 |
| FR-V2-4.2 | Thread-local 임시 버퍼 관리 | P1 |
| FR-V2-4.3 | ThreadSanitizer 클린 통과 | P1 |

### FR-V2-5: Performance Optimization

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-V2-5.1 | NEON optimized attention (dequant+dot fused) | P0 |
| FR-V2-5.2 | SIMD speedup ≥ 4.0x | P0 |
| FR-V2-5.3 | attention_throughput ≥ 50K queries/sec | P0 |

---

## 4. Success Criteria

v0.2는 `score.sh` 기준 **0.99 이상** 달성 시 완료.

추가 검증:
- QJL Hamming attention이 dequant 방식 대비 **5x 이상 빠름**
- PolarQuant direct attention이 dequant 방식 대비 **2x 이상 빠름**
- 9/9+ 테스트 통과, ASan/UBSan/TSan 클린
- SIMD speedup ≥ 4.0x
