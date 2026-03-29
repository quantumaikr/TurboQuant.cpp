# TurboQuant.cpp — Product Requirements Document v0.1

**Version**: 0.1
**Date**: 2026-03-29
**Status**: Draft

---

## 1. Executive Summary

TurboQuant.cpp는 LLM 추론 시 KV 캐시를 3비트 수준으로 압축하여, 동일한 하드웨어에서 5배 이상 긴 컨텍스트를 처리할 수 있게 하는 **프레임워크 무관 C/C++ 라이브러리**이다.

**핵심 가치**: 논문으로 증명된 극한 KV 캐시 압축(TurboQuant, PolarQuant, QJL)을 GPU(CUDA), CPU(AVX2/NEON/SVE), MPS(Apple Metal) 어디서든 즉시 사용 가능한 프로덕션급 라이브러리로 제공한다.

---

## 2. Problem Statement

### 2.1 현재 상황

LLM의 KV 캐시는 컨텍스트 길이에 비례하여 VRAM을 소비한다.

| 모델 | KV 캐시 (FP16, 100K 토큰) | GPU 요구량 |
|------|--------------------------|-----------|
| Llama-3-8B | ~12.5 GB | 24GB GPU 1대 거의 전부 |
| Llama-3-70B | ~100 GB | 80GB GPU 2대 이상 |
| Mixtral-8x7B | ~25 GB | 48GB GPU 1대 이상 |

이로 인해:
- 컨텍스트 길이가 GPU 메모리에 의해 제한됨
- 동시 서빙 사용자 수가 KV 캐시 메모리에 의해 제한됨
- 긴 컨텍스트가 필요한 작업(문서 분석, 코드 리뷰, RAG)의 품질이 제한됨

### 2.2 기존 솔루션의 한계

| 솔루션 | 압축률 | 한계 |
|--------|--------|------|
| llama.cpp Q4_0 KV | 4x | 단순 uniform 양자화 → 긴 컨텍스트에서 품질 저하 |
| vLLM FP8 | 2x | 8비트 한계 → 절약 부족 |
| QJL (논문) | 5x | Python/CUDA 전용, HuggingFace 종속 |
| PolarQuant (논문) | 4x | Triton 전용, A100 필수 |
| 커뮤니티 포크 | 다양 | 파편화, 단일 플랫폼, 유지보수 불확실 |

### 2.3 Gap

**3비트 품질 무손실 KV 캐시 압축이 이론적으로 검증되었으나, 크로스 플랫폼에서 프로덕션급으로 사용할 수 있는 구현체가 존재하지 않는다.**

---

## 3. Target Users

### 3.1 Primary Users

| 사용자 | 니즈 | 사용 방식 |
|--------|------|----------|
| **LLM 추론 엔진 개발자** (llama.cpp, vLLM 등) | KV 캐시 압축 라이브러리를 자신의 엔진에 통합 | C API 호출, 빌드 시스템에 의존성 추가 |
| **LLM 서빙 인프라 엔지니어** | GPU 비용 절감, 동시 서빙 수 증가 | vLLM/TensorRT-LLM에 플러그인 |
| **로컬 LLM 사용자** | 제한된 VRAM에서 더 긴 컨텍스트 | llama.cpp `--kv-cache-type turbo3` |

### 3.2 Secondary Users

| 사용자 | 니즈 |
|--------|------|
| **연구자** | 양자화 알고리즘 실험, 벤치마크 비교 |
| **엣지/모바일 개발자** | 제한된 메모리에서 LLM 실행 |
| **프레임워크 개발자** | ONNX Runtime 등에 커스텀 양자화 연산자 통합 |

---

## 4. Product Goals & Success Metrics

### 4.1 Goals

| 목표 | 측정 기준 | 목표값 |
|------|----------|--------|
| **G1: 메모리 절약** | KV 캐시 메모리 압축률 | 5x 이상 (3-bit) |
| **G2: 품질 유지** | LongBench F1 score 대비 FP16 | 99% 이상 |
| **G3: 크로스 플랫폼** | 지원 백엔드 수 | CPU(x86+ARM) + CUDA + Metal |
| **G4: 프레임워크 무관** | 통합 가능한 엔진 수 | llama.cpp + vLLM 최소 2개 |
| **G5: 성능 오버헤드** | 양자화로 인한 추론 지연 증가 | Decode < 5%, Prefill < 15% |

### 4.2 Non-Goals (v0.1)

- 가중치 양자화 (llama.cpp의 GGML이 이미 해결)
- 모델 학습/파인튜닝 지원
- 분산 추론 (multi-node)
- Android/iOS 네이티브 빌드 (향후 확장)

---

## 5. Technical Architecture

### 5.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Integration                                        │
│  llama.cpp plugin │ vLLM binding │ ONNX Runtime CustomOp     │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Cache Management                                   │
│  PagedQuantCache │ Progressive Compression │ Fused Ops       │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Compute Kernels                                    │
│  TypeTraits │ BlockFormat │ SIMD/GPU Dispatch                │
├─────────────────────────────────────────────────────────────┤
│  Layer 0: Specification                                      │
│  FormatSpec │ OpSchema │ TestVectors │ Versioning            │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Design Principles

1. **Spec-First**: ONNX 철학 — 사양을 먼저 정의하고 구현은 사양을 따른다
2. **Zero-Overhead Dispatch**: llama.cpp 철학 — 타입 트레이트 테이블 + 함수 포인터로 런타임 분기 비용 제거
3. **Fused Kernels**: vLLM 철학 — 양자화+캐시저장+어텐션을 최소 커널로 퓨전
4. **Self-Contained Blocks**: llama.cpp 철학 — 각 양자화 블록이 메타데이터를 내장, 외부 룩업 불필요
5. **Progressive Compression**: 새로운 철학 — 최근 토큰은 고정밀도, 오래된 토큰은 저정밀도로 자동 전환

---

## 6. Functional Requirements

### 6.1 Core: 양자화 알고리즘

#### FR-1: PolarQuant Engine

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-1.1 | 벡터를 D/2개 2D 쌍으로 분할하여 극좌표(ρ, θ) 변환 | P0 |
| FR-1.2 | 그룹별(block_size=128) min-max 스케일 계산 | P0 |
| FR-1.3 | θ를 n비트(2~4), ρ를 m비트(1~4)로 양자화 | P0 |
| FR-1.4 | `rho << tbits | theta` 패킹으로 uint8 인덱스 생성 | P0 |
| FR-1.5 | 역양자화 없이 극좌표 기반 직접 attention score 계산 | P0 |
| FR-1.6 | 설정 가능한 비트 할당 (θ:ρ 비율 조정) | P1 |

#### FR-2: QJL Engine

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-2.1 | Gaussian 랜덤 투영 행렬 생성 (d_key × d_sketch) | P0 |
| FR-2.2 | 선택적 QR 직교화 및 Hadamard 변환 | P1 |
| FR-2.3 | 1비트 부호 양자화 (sign → +1/-1) + uint8 비트패킹 | P0 |
| FR-2.4 | L2 norm 기반 top-k 아웃라이어 탐지 및 분리 저장 | P0 |
| FR-2.5 | Hamming distance 기반 attention score 계산 | P0 |
| FR-2.6 | Norm-weighted score 재구성: `scl * norm_k * inner_prod` | P0 |

#### FR-3: TurboQuant Composite

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-3.1 | PolarQuant(1단계) + QJL(2단계) 결합 양자화 | P0 |
| FR-3.2 | 1단계 잔여 오차에 대해서만 QJL 적용 (1비트) | P0 |
| FR-3.3 | 총 비트 = PolarQuant 비트 + QJL 1비트로 구성 | P0 |
| FR-3.4 | PolarQuant-only 모드 (QJL 없이) 지원 | P0 |
| FR-3.5 | QJL-only 모드 지원 | P1 |
| FR-3.6 | Uniform baseline (단순 min-max) 모드 지원 | P0 |

#### FR-4: Value Cache 양자화

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-4.1 | Value 캐시 2~4비트 그룹별 양자화 (min-max 스케일) | P0 |
| FR-4.2 | Value 비트패킹: 2-bit → 4 values/byte, 4-bit → 2 values/byte | P0 |
| FR-4.3 | 양자화된 Value × attention weight의 fused matmul | P1 |
| FR-4.4 | Value 캐시 양자화 비활성화 옵션 (Key만 양자화) | P0 |

### 6.2 Cache Management

#### FR-5: Paged Quantized Cache

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-5.1 | 고정 크기 블록 단위 KV 캐시 할당/해제 | P0 |
| FR-5.2 | 논리 시퀀스 → 물리 블록 매핑 (block table) | P0 |
| FR-5.3 | Copy-on-Write: beam search 시 블록 공유 | P1 |
| FR-5.4 | 블록별 양자화 타입 독립 설정 (혼합 정밀도) | P0 |
| FR-5.5 | 블록 헤더에 양자화 메타데이터 내장 | P0 |

#### FR-6: Progressive Compression

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-6.1 | 최근 N 토큰은 FP16 유지 (residual window) | P0 |
| FR-6.2 | residual window 초과 시 자동 양자화 트리거 | P0 |
| FR-6.3 | 시간 경과에 따른 점진적 재압축 (4bit → 3bit) | P1 |
| FR-6.4 | 레이어별 독립 압축 정책 설정 | P1 |
| FR-6.5 | 설정 가능한 residual window 크기 | P0 |

#### FR-7: Fused Operations

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-7.1 | Fused Quantize+Store: 양자화와 캐시 기록을 단일 커널 | P0 |
| FR-7.2 | Fused Load+Attend: 양자화 캐시 로드와 attention 계산을 단일 커널 | P1 |
| FR-7.3 | 중간 버퍼 제거로 메모리 대역폭 절약 | P0 |

### 6.3 Compute Backends

#### FR-8: CPU Backend

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-8.1 | Generic C 레퍼런스 구현 (모든 플랫폼) | P0 |
| FR-8.2 | x86 AVX2 최적화: 비트 연산 + dot product | P0 |
| FR-8.3 | x86 AVX-512 최적화 | P2 |
| FR-8.4 | ARM NEON 최적화 (Apple M-series, Snapdragon) | P0 |
| FR-8.5 | ARM SVE 최적화 (AWS Graviton) | P2 |
| FR-8.6 | 초기화 시 CPUID 감지 → 최적 구현 자동 선택 | P0 |
| FR-8.7 | 멀티스레드 양자화/역양자화 (OpenMP 또는 pthreads) | P1 |

#### FR-9: CUDA Backend

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-9.1 | PolarQuant CUDA 커널: 극좌표 변환 + 양자화 | P0 |
| FR-9.2 | QJL CUDA 커널: JL 투영 + 1비트 해시 + 아웃라이어 | P0 |
| FR-9.3 | Attention score CUDA 커널: 양자화 KV → score | P0 |
| FR-9.4 | Fused quantize+cache CUDA 커널 (vLLM 패턴) | P1 |
| FR-9.5 | GQA (Grouped-Query Attention) 전용 커널 | P0 |
| FR-9.6 | MQA (Multi-Query Attention) 전용 커널 | P1 |
| FR-9.7 | Warp-level reduction + shared memory 최적화 | P0 |
| FR-9.8 | CUDA Compute Capability 7.0+ (Volta 이상) 지원 | P0 |
| FR-9.9 | L2 캐시 persistence hint 적용 | P1 |

#### FR-10: Metal Backend (MPS)

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-10.1 | Metal Compute Shader: PolarQuant 양자화 | P0 |
| FR-10.2 | Metal Compute Shader: QJL 1비트 해시 | P0 |
| FR-10.3 | Metal Compute Shader: Attention score 계산 | P0 |
| FR-10.4 | Threadgroup memory (shared memory) 최적화 | P0 |
| FR-10.5 | SIMD-group (warp) 수준 reduction | P0 |
| FR-10.6 | Apple M1/M2/M3/M4 Unified Memory 활용 | P1 |
| FR-10.7 | Metal 3 mesh shader 활용 (가능한 경우) | P2 |

### 6.4 Type System & Format

#### FR-11: Block Format

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-11.1 | `block_tq_polar` 구조체: 스케일(4×fp16) + 패킹된 인덱스 | P0 |
| FR-11.2 | `block_tq_qjl` 구조체: norm(2×fp16) + 해시(uint8[]) + 아웃라이어 인덱스 | P0 |
| FR-11.3 | `block_tq_turbo` 구조체: polar + qjl 합성 | P0 |
| FR-11.4 | `block_tq_uniform` 구조체: 단순 min-max + 패킹된 값 | P0 |
| FR-11.5 | 모든 블록에 `static_assert` 크기 검증 (llama.cpp 패턴) | P0 |
| FR-11.6 | 비트 패킹: ONNX int2/int4 LSB-first 표준 준수 | P0 |

#### FR-12: Type Traits

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-12.1 | `tq_type` 열거형: 모든 양자화 타입 정의 | P0 |
| FR-12.2 | `tq_type_traits` 배열: block_size, type_size, quantize, attention 함수 포인터 | P0 |
| FR-12.3 | 타입 페어링: turbo → polar + qjl residual 관계 정의 | P0 |
| FR-12.4 | 런타임 타입 조회 O(1) (인덱스 기반) | P0 |
| FR-12.5 | 새 타입 추가 시 기존 코드 변경 불필요 (append-only enum) | P0 |

### 6.5 Specification & Testing

#### FR-13: Format Specification

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-13.1 | 포맷 사양서 v1: 블록 구조, 비트 레이아웃, 패킹 규칙 문서화 | P0 |
| FR-13.2 | 연산자 사양서 v1: 양자화, 역양자화, attention 연산 시맨틱 정의 | P0 |
| FR-13.3 | 버전 관리: spec_version 필드로 하위 호환 보장 | P0 |
| FR-13.4 | 결정론적 테스트 벡터: 고정 입력 → 기대 출력 쌍 | P0 |

#### FR-14: Quality Validation

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-14.1 | 단위 테스트: 각 양자화 타입의 quantize/dequantize 왕복 오차 검증 | P0 |
| FR-14.2 | 크로스 플랫폼 일관성 테스트: CPU/CUDA/Metal 동일 결과 검증 | P0 |
| FR-14.3 | LongBench 벤치마크 통합 테스트 | P1 |
| FR-14.4 | Needle-in-a-Haystack 벤치마크 | P1 |
| FR-14.5 | 메모리 사용량 프로파일링 테스트 | P0 |
| FR-14.6 | 성능 회귀 테스트 (지연시간, 처리량) | P1 |

### 6.6 Integration

#### FR-15: Public API

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-15.1 | C API (`turboquant.h`): 순수 C 헤더, C++ 종속성 없음 | P0 |
| FR-15.2 | 컨텍스트 생성/해제: `tq_init()`, `tq_free()` | P0 |
| FR-15.3 | KV 양자화: `tq_quantize_keys()`, `tq_quantize_values()` | P0 |
| FR-15.4 | Attention: `tq_attention()` — 양자화 KV 기반 attention 계산 | P0 |
| FR-15.5 | 캐시 관리: `tq_cache_create()`, `tq_cache_append()`, `tq_cache_get_block()` | P0 |
| FR-15.6 | 전략 추천: `tq_recommend_strategy()` | P2 |
| FR-15.7 | 에러 코드 및 메시지 시스템 | P0 |
| FR-15.8 | Thread-safe API (멀티스레드 호출 안전) | P1 |

#### FR-16: Framework Integration

| ID | 요구사항 | 우선순위 |
|----|---------|---------|
| FR-16.1 | llama.cpp 통합: GGML 타입 등록 + KV 캐시 백엔드 플러그인 | P0 |
| FR-16.2 | Python 바인딩: pybind11 또는 ctypes 래퍼 | P1 |
| FR-16.3 | vLLM 통합: Python 바인딩 + reshape_and_cache 오버라이드 | P1 |
| FR-16.4 | ONNX Runtime CustomOp 등록 | P2 |
| FR-16.5 | Rust FFI 래퍼 | P2 |

---

## 7. Non-Functional Requirements

### 7.1 Performance

| ID | 요구사항 | 목표값 |
|----|---------|--------|
| NFR-1 | Decode 지연 오버헤드 (토큰당) | < 5% |
| NFR-2 | Prefill 지연 오버헤드 | < 15% |
| NFR-3 | KV 캐시 메모리 절약 (3-bit 모드) | > 5x |
| NFR-4 | 양자화 처리량 (GPU) | > 1M elements/ms |
| NFR-5 | 양자화 처리량 (CPU AVX2) | > 100K elements/ms |

### 7.2 Compatibility

| ID | 요구사항 |
|----|---------|
| NFR-6 | C11 / C++17 표준 준수 |
| NFR-7 | Linux (x86_64, aarch64), macOS (arm64), Windows (x86_64) 빌드 |
| NFR-8 | CUDA 11.8+ / 12.x 지원 |
| NFR-9 | Metal 2.0+ (macOS 13+, iOS 16+) 지원 |
| NFR-10 | CMake 3.20+ 빌드 시스템 |

### 7.3 Quality

| ID | 요구사항 |
|----|---------|
| NFR-11 | 코어 라이브러리 외부 종속성 0개 (self-contained) |
| NFR-12 | 모든 public API에 대한 단위 테스트 커버리지 > 90% |
| NFR-13 | CI/CD: Linux + macOS + Windows 매 커밋 빌드 검증 |
| NFR-14 | AddressSanitizer / UBSanitizer 클린 |
| NFR-15 | Valgrind 메모리 누수 제로 |

### 7.4 Usability

| ID | 요구사항 |
|----|---------|
| NFR-16 | 단일 CMake `add_subdirectory()` 또는 `find_package()`로 통합 |
| NFR-17 | 헤더 1개 include로 전체 API 사용 가능 |
| NFR-18 | 예제 코드: 최소 3개 (standalone, llama.cpp 통합, Python) |
| NFR-19 | API 문서: Doxygen 기반 자동 생성 |

---

## 8. Quantization Type Specifications

### 8.1 Type Registry

| Type ID | Name | Key Bits | Value Bits | BPE (Key) | Algorithm | Block Size |
|---------|------|----------|-----------|-----------|-----------|------------|
| 0 | `TQ_POLAR_3B` | 3 (θ:2+ρ:1) | - | 3.25 | PolarQuant | 128 |
| 1 | `TQ_POLAR_4B` | 4 (θ:2+ρ:2) | - | 4.50 | PolarQuant | 128 |
| 2 | `TQ_QJL_1B` | 1 | - | 1.xx | QJL sign hash | 256 |
| 3 | `TQ_TURBO_3B` | 3 (Polar 2b + QJL 1b) | - | 3.xx | TurboQuant | 128 |
| 4 | `TQ_TURBO_4B` | 4 (Polar 3b + QJL 1b) | - | 4.xx | TurboQuant | 128 |
| 5 | `TQ_UNIFORM_4B` | 4 | - | 4.50 | Min-Max Uniform | 128 |
| 6 | `TQ_UNIFORM_2B` | 2 | - | 2.25 | Min-Max Uniform | 128 |

*BPE = Bits Per Element (메타데이터 포함)*

### 8.2 Block Structure Summary

```
block_tq_polar (TQ_BK=128, θ:2bit, ρ:2bit):
┌──────────┬──────────┬──────────┬──────────┬─────────────────────┐
│ rscale   │ rmn      │ tscale   │ tmn      │ indices[64]         │
│ (fp16,2B)│ (fp16,2B)│ (fp16,2B)│ (fp16,2B)│ (rho<<2|theta, 64B)│
└──────────┴──────────┴──────────┴──────────┴─────────────────────┘
Total: 72 bytes / 128 elements = 4.5 bits per element

block_tq_qjl (sketch_dim=256):
┌──────────┬──────────┬───────────────┬──────────────────┐
│ norm     │ out_norm │ hash[32]      │ out_idx[4]       │
│ (fp16,2B)│ (fp16,2B)│ (1bit×256, 32B)│ (uint8×4, 4B)   │
└──────────┴──────────┴───────────────┴──────────────────┘
Total: 40 bytes per group element

block_tq_turbo:
┌──────────────────────┬──────────────────────┐
│ block_tq_polar       │ block_tq_qjl         │
│ (72 bytes)           │ (40 bytes)           │
└──────────────────────┴──────────────────────┘
Total: 112 bytes / 128 elements = 7.0 bits per element (key+residual)
```

---

## 9. Supported Attention Variants

| Attention Type | 설명 | 우선순위 |
|---------------|------|---------|
| MHA (Multi-Head) | 표준 Transformer attention | P0 |
| GQA (Grouped-Query) | Llama-3, Gemma 등 | P0 |
| MQA (Multi-Query) | Falcon, StarCoder 등 | P1 |
| MLA (Multi-Head Latent) | DeepSeek-V2 등 | P2 |

---

## 10. Progressive Compression Specification

### 10.1 Compression Tiers

```
Tier 0 (Hot):    FP16          — 최근 residual_window 토큰
Tier 1 (Warm):   POLAR_4B      — residual_window ~ 2×residual_window
Tier 2 (Cold):   TURBO_3B      — 2×residual_window ~ 이전 전체
```

### 10.2 Configuration

```c
typedef struct {
    int      residual_window;    // Tier 0 크기 (기본: 128)
    int      warm_window;        // Tier 1 크기 (기본: 256)
    tq_type  warm_type;          // Tier 1 양자화 (기본: TQ_POLAR_4B)
    tq_type  cold_type;          // Tier 2 양자화 (기본: TQ_TURBO_3B)
    bool     enable_recompression; // Tier 1→2 재압축 활성화
} tq_progressive_config_t;
```

### 10.3 Transition Rules

1. 새 토큰 생성 시 Tier 0에 FP16으로 추가
2. Tier 0이 `residual_window` 초과 시, 가장 오래된 블록을 `warm_type`으로 양자화 → Tier 1
3. `enable_recompression=true`이고 Tier 1이 `warm_window` 초과 시, 가장 오래된 블록을 `cold_type`으로 재압축 → Tier 2
4. Attention 계산 시 각 블록의 타입에 맞는 커널을 타입 트레이트에서 O(1) 조회

---

## 11. Build & Distribution

### 11.1 Build System

```cmake
# 사용자 측 통합 (두 가지 방식)

# 방식 1: 서브디렉토리
add_subdirectory(turboquant)
target_link_libraries(my_app turboquant::turboquant)

# 방식 2: find_package
find_package(TurboQuant REQUIRED)
target_link_libraries(my_app TurboQuant::TurboQuant)
```

### 11.2 Build Options

| Option | 기본값 | 설명 |
|--------|--------|------|
| `TQ_BUILD_CUDA` | OFF | CUDA 커널 빌드 |
| `TQ_BUILD_METAL` | OFF (macOS: ON) | Metal 셰이더 빌드 |
| `TQ_BUILD_TESTS` | OFF | 테스트 빌드 |
| `TQ_BUILD_BENCH` | OFF | 벤치마크 빌드 |
| `TQ_BUILD_PYTHON` | OFF | Python 바인딩 빌드 |
| `TQ_BUILD_EXAMPLES` | OFF | 예제 빌드 |

### 11.3 Distribution

- GitHub 릴리즈: 소스 tarball + 빌드 가이드
- vcpkg / Conan 패키지 (향후)
- PyPI: `pip install turboquant` (Python 바인딩, 향후)

---

## 12. Risk Assessment

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| GPU 커널 성능이 연구 구현 대비 부족 | 중 | 고 | QJL/PolarQuant CUDA 코드를 최대한 직접 포팅. 프로파일링 기반 반복 최적화 |
| 특정 모델 아키텍처에서 품질 저하 | 중 | 중 | Adaptive 전략으로 fallback. 모델별 최적 설정 프리셋 제공 |
| llama.cpp 메인 브랜치가 자체 TurboQuant 통합 | 낮 | 고 | 독립 라이브러리이므로 llama.cpp의 의존성이 되는 것이 이상적 시나리오 |
| Metal 셰이더 최적화 난이도 | 중 | 중 | llama.cpp Metal 코드 패턴 참조. 초기에는 CPU NEON이 Apple Silicon 대안 |
| API 설계 변경으로 하위 호환 깨짐 | 중 | 중 | v0.x 동안은 불안정 명시. v1.0에서 API 동결 |

---

## 13. Dependencies

### 13.1 빌드 도구

| 도구 | 버전 | 용도 |
|------|------|------|
| CMake | 3.20+ | 빌드 시스템 |
| C Compiler | C11 지원 | 코어 라이브러리 |
| C++ Compiler | C++17 지원 | CUDA, Metal 래퍼 |
| CUDA Toolkit | 11.8+ | CUDA 백엔드 (선택) |
| Xcode Command Line Tools | 15+ | Metal 백엔드 (선택) |

### 13.2 런타임 종속성

**없음.** 코어 라이브러리는 표준 C 라이브러리(libc, libm)만 사용한다.

### 13.3 테스트 종속성

| 도구 | 용도 |
|------|------|
| Google Test | C++ 단위 테스트 |
| Python 3.10+ | 레퍼런스 구현 비교 테스트 |
| NumPy | 테스트 벡터 생성 |

---

## 14. Milestones

| 마일스톤 | 내용 | 산출물 |
|---------|------|--------|
| **M0: Foundation** | 빌드 시스템, C API 헤더, 포맷 사양, 테스트 프레임워크 | 빌드 가능한 빈 라이브러리 + 사양서 |
| **M1: CPU Core** | PolarQuant + QJL + Turbo 레퍼런스 C 구현 + 테스트 벡터 검증 | CPU에서 동작하는 양자화/역양자화 |
| **M2: CPU SIMD** | AVX2 + NEON 최적화 + Paged Cache + Progressive Compression | 성능 최적화된 CPU 백엔드 |
| **M3: CUDA** | CUDA 커널 포팅 + Fused 커널 + GQA 지원 | GPU 가속 동작 |
| **M4: Metal** | Metal Compute Shader + Apple Silicon 최적화 | macOS MPS 지원 |
| **M5: Integration** | llama.cpp 플러그인 + Python 바인딩 | 프레임워크 통합 |
| **M6: Validation** | LongBench/NIAH 벤치마크 + 성능 프로파일링 | 품질/성능 검증 보고서 |

---

## 15. Glossary

| 용어 | 정의 |
|------|------|
| **KV 캐시** | Transformer attention의 Key, Value 텐서를 저장하는 메모리 영역 |
| **PolarQuant** | 벡터를 극좌표(ρ, θ)로 변환하여 양자화하는 기법 |
| **QJL** | Johnson-Lindenstrauss 변환 후 1비트 부호 양자화하는 기법 |
| **TurboQuant** | PolarQuant(1단계) + QJL(2단계 잔여 보정)의 결합 기법 |
| **Progressive Compression** | 토큰의 나이에 따라 점진적으로 압축률을 높이는 전략 |
| **Type Traits** | 양자화 타입별 함수 포인터와 메타데이터를 담는 조회 테이블 |
| **Fused Kernel** | 여러 연산(양자화, 캐시 기록, attention)을 하나의 GPU 커널로 합친 것 |
| **GQA** | Grouped-Query Attention — 여러 Query head가 하나의 KV head를 공유 |
| **BPE** | Bits Per Element — 양자화 후 원소당 평균 비트 수 (메타데이터 포함) |

---

## Appendix A: Reference Materials

| 자료 | 위치 |
|------|------|
| TurboQuant 논문 | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) |
| QJL 논문 | [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) |
| PolarQuant 논문 (Zandieh) | [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) |
| QJL 공식 구현 | `refs/QJL/` |
| PolarQuant 공식 구현 | `refs/PolarQuant/` |
| llama.cpp (TQ 포크) | `refs/llama.cpp/` |
| vLLM (KV 캐시 참조) | `refs/vllm/` |
| ONNX (양자화 사양 참조) | `refs/onnx/` |
