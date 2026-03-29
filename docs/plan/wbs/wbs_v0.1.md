# TurboQuant.cpp — Work Breakdown Structure v0.1

**Version**: 0.1
**Date**: 2026-03-29
**기반 문서**: [PRD v0.1](./prd_v0.1.md)

---

## M0: Foundation (기반 구축)

### M0.1 프로젝트 구조 및 빌드 시스템

- [x] 디렉토리 구조 생성
  ```
  include/turboquant/
  src/core/
  src/cache/
  src/backend/cpu/
  src/backend/cuda/
  src/backend/metal/
  src/adaptive/
  tests/
  bench/
  integrations/
  spec/
  ```
- [x] `CMakeLists.txt` 루트 작성
  - [x] C11 / C++17 표준 설정
  - [x] 빌드 옵션 정의: `TQ_BUILD_CUDA`, `TQ_BUILD_METAL`, `TQ_BUILD_TESTS`, `TQ_BUILD_BENCH`
  - [x] 플랫폼 감지 (Linux/macOS/Windows)
  - [x] SIMD 감지 (AVX2, NEON, SVE)
- [x] `src/CMakeLists.txt` 코어 라이브러리 타겟 정의
- [x] `tests/CMakeLists.txt` 테스트 타겟 정의 (Google Test)
- [x] CI 설정: GitHub Actions
  - [x] Linux x86_64 빌드 + 테스트
  - [x] macOS arm64 빌드 + 테스트
  - [ ] Windows x86_64 빌드
- [x] `.clang-format` 코드 스타일 설정
- [x] `README.md` 초안 작성

### M0.2 Public C API 설계

- [x] `include/turboquant/turboquant.h` — 단일 Public 헤더
  - [x] 컨텍스트 라이프사이클: `tq_init()`, `tq_free()`
  - [x] 백엔드 열거형: `TQ_BACKEND_CPU`, `TQ_BACKEND_CUDA`, `TQ_BACKEND_METAL`
  - [x] 양자화 타입 열거형: `tq_type` (TQ_POLAR_3B ~ TQ_UNIFORM_2B)
  - [x] KV 양자화 API: `tq_quantize_keys()`, `tq_quantize_values()`
  - [x] Attention API: `tq_attention()`
  - [x] 캐시 관리 API: `tq_cache_create()`, `tq_cache_append()`, `tq_cache_free()`
  - [x] Progressive Compression 설정: `tq_progressive_config_t`
  - [x] 에러 코드 열거형 및 `tq_error_string()`
  - [x] 버전 매크로: `TQ_VERSION_MAJOR`, `TQ_VERSION_MINOR`, `TQ_VERSION_PATCH`
- [x] `include/turboquant/tq_types.h` — 타입/블록 정의
  - [x] `block_tq_polar` 구조체 정의 + `static_assert` 크기 검증
  - [x] `block_tq_qjl` 구조체 정의 + `static_assert`
  - [x] `block_tq_turbo` 구조체 정의 + `static_assert`
  - [x] `block_tq_uniform` 구조체 정의 + `static_assert`
  - [x] 상수 정의: `TQ_BK` (블록 크기), `TQ_SKETCH_DIM`, `TQ_OUTLIERS`
- [x] `include/turboquant/tq_spec.h` — 포맷 사양 구조체
  - [x] `tq_format_spec_t` 정의 (spec_version, algorithm, key_bits 등)

### M0.3 포맷 사양서

- [x] `spec/tq_format_v1.md` 작성
  - [ ] PolarQuant 블록 바이너리 레이아웃 정의
  - [ ] QJL 블록 바이너리 레이아웃 정의
  - [ ] TurboQuant 합성 블록 레이아웃 정의
  - [ ] 비트 패킹 규칙 (ONNX int2/int4 LSB-first 준수)
  - [ ] 엔디안 규칙 (little-endian)
  - [ ] 블록 정렬 규칙
- [x] `spec/tq_operators_v1.md` 작성
  - [ ] PolarQuantize 연산자 시맨틱 정의
  - [ ] QJLQuantize 연산자 시맨틱 정의
  - [ ] TurboAttention 연산자 시맨틱 정의
  - [ ] 입출력 텐서 형상 규칙

### M0.4 테스트 인프라

- [x] 테스트 프레임워크 설정 (Google Test)
- [x] Python 레퍼런스 구현 작성
  - [x] `tests/reference/polar_quant_ref.py` — PolarQuant numpy 레퍼런스
  - [x] `tests/reference/qjl_ref.py` — QJL numpy 레퍼런스
  - [x] `tests/reference/turbo_quant_ref.py` — TurboQuant 합성 레퍼런스
- [x] 테스트 벡터 생성 스크립트: `tests/reference/generate_test_vectors.py`
- [x] `spec/test_vectors/` 디렉토리에 바이너리 테스트 벡터 저장
  - [x] `polar_3b_d128_b128.bin` — PolarQuant 3-bit, head_dim=128, block=128
  - [x] `qjl_1b_d128_s256.bin` — QJL 1-bit, head_dim=128, sketch=256
  - [ ] `turbo_3b_d128.bin` — TurboQuant 3-bit 합성
  - [x] `uniform_4b_d128.bin` — Uniform 4-bit baseline

---

## M1: CPU Core (핵심 알고리즘 C 구현)

### M1.1 타입 시스템

- [x] `src/core/tq_traits.c` — 타입 트레이트 테이블 구현
  - [ ] `tq_type_traits` 배열 초기화 (각 타입별 block_size, type_size, 함수 포인터)
  - [x] `tq_type_name()` — 타입명 문자열 반환
  - [x] `tq_type_bpe()` — bits-per-element 반환
  - [x] `tq_type_block_size()` — 블록 크기 반환

### M1.2 PolarQuant 레퍼런스 구현

- [x] `src/core/tq_polar.c` — PolarQuant 알고리즘
  - [x] `tq_polar_quantize_ref()` — 레퍼런스 양자화
    - [ ] 벡터 → D/2 2D 쌍 분할
    - [ ] `atan2f()` 각도 계산 + [0, 2π] 정규화
    - [ ] `sqrtf(x² + y²)` 반지름 계산
    - [ ] 그룹별 min/max 계산 → scale 도출
    - [ ] θ, ρ 양자화 → `(rho << tbits) | theta` 패킹
  - [x] `tq_polar_dequantize_ref()` — 레퍼런스 역양자화
    - [ ] 인덱스 언패킹 → θ, ρ 복원
    - [ ] 극좌표 → 직교좌표 변환: `(ρ×cos(θ), ρ×sin(θ))`
  - [x] `tq_polar_attention_ref()` — 극좌표 기반 직접 attention score
    - [ ] θ 양자화 레벨별 cos/sin lookup 테이블 생성
    - [ ] query × lookup → gather by θ index → ρ 가중치 적용
    - [ ] 쌍별 합산 → 최종 attention score
- [x] `tests/test_polar.cpp` — PolarQuant 단위 테스트
  - [ ] 양자화 왕복 오차 테스트 (quantize → dequantize → MSE 검증)
  - [ ] 테스트 벡터 대비 비트 정확 일치 검증
  - [ ] Attention score 정확도 테스트 (FP16 대비 상대 오차 < 1%)
  - [ ] 엣지 케이스: 제로 벡터, 매우 큰 값, NaN/Inf
  - [ ] 다양한 head_dim (64, 128, 256) 테스트

### M1.3 QJL 레퍼런스 구현

- [x] `src/core/tq_qjl.c` — QJL 알고리즘
  - [ ] `tq_qjl_init_projection()` — 랜덤 투영 행렬 생성
    - [ ] Gaussian 랜덤 행렬 (d_key × d_sketch)
    - [ ] 선택적 QR 직교화
    - [ ] 시드 기반 재현 가능한 난수 생성
  - [ ] `tq_qjl_quantize_ref()` — 레퍼런스 양자화
    - [ ] Key × Projection 행렬 곱
    - [ ] sign 양자화: `>0 → 1, ≤0 → 0`
    - [ ] 8-bit 패킹: 8 signs → 1 uint8
  - [ ] `tq_qjl_detect_outliers()` — 아웃라이어 탐지
    - [ ] L2 norm 계산 (그룹 내)
    - [ ] Top-k 인덱스 선택
    - [ ] 아웃라이어 norm 별도 저장
  - [ ] `tq_qjl_attention_ref()` — Hamming distance 기반 attention score
    - [ ] Query 투영: `query × projection`
    - [ ] XOR + popcount → Hamming distance
    - [ ] `sqrt(π/2) / sketch_dim * norm * inner_prod` score 재구성
    - [ ] 아웃라이어 보정 합산
- [x] `tests/test_qjl.cpp` — QJL 단위 테스트
  - [ ] 투영 행렬 직교성 검증
  - [ ] 1비트 해시 비트 정확 일치 검증
  - [ ] 아웃라이어 탐지 정확성 테스트
  - [ ] Attention score 정확도 테스트
  - [ ] 다양한 sketch_dim (128, 256, 512) 테스트

### M1.4 TurboQuant 합성 구현

- [x] `src/core/tq_turbo.c` — TurboQuant 합성 알고리즘
  - [ ] `tq_turbo_quantize_ref()` — 2단계 양자화
    - [ ] 1단계: PolarQuant 양자화
    - [ ] 잔여 오차 계산: `residual = original - dequantized`
    - [ ] 2단계: QJL로 잔여 오차 1비트 양자화
  - [ ] `tq_turbo_attention_ref()` — 2단계 attention score
    - [ ] PolarQuant attention score 계산
    - [ ] QJL 잔여 보정 score 계산
    - [ ] 가중 합산
- [x] `tests/test_turbo.cpp` — TurboQuant 단위 테스트
  - [ ] PolarQuant-only 대비 TurboQuant 품질 개선 검증
  - [ ] 테스트 벡터 비트 정확 일치
  - [ ] 다양한 비트 할당 조합 테스트

### M1.5 Uniform Baseline 구현

- [x] `src/core/tq_uniform.c` — 단순 Min-Max 양자화
  - [ ] `tq_uniform_quantize_ref()` — 2/4-bit uniform 양자화
  - [ ] `tq_uniform_dequantize_ref()` — 역양자화
  - [ ] `tq_uniform_attention_ref()` — 역양자화 후 표준 attention
- [x] `tests/test_uniform.cpp` — baseline 테스트

### M1.6 Value Cache 양자화

- [x] `src/core/tq_value_quant.c` — Value 양자화
  - [ ] `tq_value_quantize_ref()` — 그룹별 min-max 2/4-bit 양자화
  - [ ] `tq_value_dequantize_ref()` — 역양자화
  - [ ] 비트 패킹: 2-bit (4/byte), 4-bit (2/byte) — ONNX LSB-first 준수
- [x] `tests/test_value_quant.cpp` — Value 양자화 테스트

### M1.7 크로스 플랫폼 검증

- [x] 테스트 벡터 기반 플랫폼 일관성 검증
  - [ ] Linux x86_64 통과
  - [x] macOS arm64 통과
  - [ ] Windows x86_64 통과
- [x] 모든 테스트 Green 확인

---

## M2: CPU SIMD 최적화 + 캐시 관리

### M2.1 x86 AVX2 최적화

- [x] `src/backend/cpu/tq_avx2.c`
  - [x] `tq_polar_quantize_avx2()` — 8-wide SIMD 극좌표 변환 (ref fallback)
    - [x] `_mm256_atan2_ps` 근사 (또는 lookup + 보간)
    - [x] SIMD min/max 리덕션
    - [x] 비트 패킹 최적화
  - [ ] `tq_polar_attention_avx2()` — 8-wide attention score
    - [ ] cos/sin lookup의 gather 명령어 활용
    - [x] FMA (Fused Multiply-Add) 활용
  - [x] `tq_qjl_quantize_avx2()` — SIMD 행렬-벡터 곱 + sign 비트 추출 (ref fallback)
    - [ ] `_mm256_movemask_ps` 활용한 8-bit sign 추출
  - [ ] `tq_qjl_attention_avx2()` — XOR + popcount 최적화
    - [ ] `_mm256_xor_si256` + `_mm256_popcnt` (AVX-512 VPOPCNTDQ 또는 lookup)
  - [ ] `tq_value_dequant_dot_avx2()` — 양자화 Value × attention weight dot product
- [x] `tests/test_simd_avx2.cpp` — SIMD 구현 vs 레퍼런스 결과 일치 검증

### M2.2 ARM NEON 최적화

- [x] `src/backend/cpu/tq_neon.c`
  - [x] `tq_polar_quantize_neon()` — 4-wide SIMD 극좌표 변환
    - [x] `vatan2q_f32` 근사 (polynomial approximation)
    - [x] NEON min/max: `vmaxq_f32`, `vminq_f32`
  - [ ] `tq_polar_attention_neon()` — 4-wide attention score
  - [x] `tq_qjl_quantize_neon()` — NEON 행렬-벡터 곱
    - [x] `vcltq_f32` + 비트마스크로 sign 추출
  - [x] `tq_qjl_attention_neon()` — XOR + `vcntq_u8` popcount
  - [ ] `tq_value_dequant_dot_neon()` — dot product
- [x] `tests/test_simd_neon.cpp` — NEON vs 레퍼런스 일치 검증

### M2.3 SIMD 디스패치

- [x] `src/backend/cpu/tq_cpu_dispatch.c`
  - [x] 런타임 CPUID 감지 (x86: `__get_cpuid`, ARM: 컴파일 타임)
  - [x] 타입 트레이트 함수 포인터를 최적 구현으로 스와핑
  - [x] `tq_init_cpu_backend()` 함수 (`tq_cpu_dispatch_init()`)
- [x] Fallback 체인 검증: AVX2 → SSE4.2 → Generic, NEON → Generic

### M2.4 Paged Quantized Cache

- [x] `src/cache/tq_paged_cache.c` — 페이지 기반 KV 캐시
  - [ ] `tq_cache_create()` — 캐시 인스턴스 생성
    - [ ] 물리 블록 풀 할당
    - [ ] 블록 테이블 초기화
  - [ ] `tq_cache_append()` — 새 토큰 KV 추가
    - [ ] 슬롯 매핑: 논리 위치 → 물리 블록 + 오프셋
    - [ ] 블록 가득 차면 새 블록 할당
  - [ ] `tq_cache_get_block()` — 블록 조회
    - [ ] 블록 헤더에서 양자화 타입 확인 → 타입 트레이트 디스패치
  - [ ] `tq_cache_free_block()` — 블록 해제 (ref_count 기반)
  - [ ] `tq_cache_copy_block()` — Copy-on-Write 블록 복사
  - [ ] `tq_cache_free()` — 전체 캐시 해제
- [x] `tests/test_paged_cache.cpp`
  - [ ] 블록 할당/해제 사이클 테스트
  - [ ] 메모리 누수 테스트 (Valgrind/ASan)
  - [ ] 블록 테이블 매핑 정확성 테스트
  - [ ] Copy-on-Write 동작 검증

### M2.5 Progressive Compression

- [x] `src/cache/tq_progressive.c` — 점진적 압축 엔진
  - [x] `tq_progressive_init()` — 설정 기반 초기화 (`tq_progressive_create()`)
  - [x] `tq_progressive_append()` — 토큰 추가 + 자동 압축 트리거
    - [x] Tier 0 (FP16) → Tier 1 (4bit) 전환 로직
    - [x] Tier 1 → Tier 2 (3bit) 재압축 로직 (옵션)
  - [x] `tq_progressive_attention()` — 혼합 정밀도 attention
    - [x] 블록별 타입 트레이트 O(1) 조회
    - [x] 각 Tier의 score를 연결하여 최종 softmax
- [x] `tests/test_progressive.cpp`
  - [x] Tier 전환 시점 정확성 테스트
  - [x] 혼합 정밀도 attention 품질 테스트
  - [x] residual_window 크기별 메모리 사용량 검증

### M2.6 멀티스레드 지원

- [x] `src/core/tq_threading.c`
  - [ ] 블록 단위 병렬 양자화 (OpenMP 또는 pthreads)
  - [ ] Thread-local 임시 버퍼 관리
- [x] Thread-safety 테스트 (TSan)

---

## M3: CUDA Backend

### M3.1 CUDA 빌드 인프라

- [x] `src/backend/cuda/CMakeLists.txt`
  - [ ] CUDA 타겟 정의
  - [ ] Compute Capability 설정 (7.0, 8.0, 8.9, 9.0)
  - [ ] 호스트/디바이스 코드 분리
- [x] `src/backend/cuda/tq_cuda_common.cuh` — 공통 유틸리티
  - [ ] Warp-level reduction: `warpReduceSum()`, `warpReduceMax()`
  - [ ] Shared memory 유틸리티
  - [ ] 타입 변환: fp16 ↔ fp32, bf16 ↔ fp32
  - [ ] 에러 체크 매크로: `TQ_CUDA_CHECK()`

### M3.2 PolarQuant CUDA 커널

- [x] `src/backend/cuda/tq_polar.cu`
  - [ ] `tq_polar_quantize_kernel<<<>>>`
    - [ ] 블록 구성: 1 thread block = 1 양자화 그룹
    - [ ] Shared memory에 key 로드
    - [ ] CUDA `atan2f`, `sqrtf` 활용
    - [ ] Warp-level min/max reduction → scale 계산
    - [ ] 양자화 + 패킹 → global memory 기록
  - [ ] `tq_polar_attention_kernel<<<>>>`
    - [ ] cos/sin lookup 테이블을 shared memory에 생성
    - [ ] Query × lookup → gather by index
    - [ ] Warp-level sum reduction → attention score
  - [ ] GQA 지원: num_kv_heads < num_q_heads 처리
- [ ] `tests/test_cuda_polar.cpp` — CUDA vs CPU 레퍼런스 결과 일치 검증

### M3.3 QJL CUDA 커널

- [x] `src/backend/cuda/tq_qjl.cu` (refs/QJL/qjl_kernel/csrc/ 기반 포팅)
  - [ ] `tq_qjl_quantize_kernel<<<>>>` — 1비트 해시 + 아웃라이어
    - [ ] Shared memory: `shared_keys[EMB_DIM][WARP_SIZE]`
    - [ ] 아웃라이어 마스크: `shared_mask[EMB_DIM]`
    - [ ] 투영 행렬 곱 + sign 비트 추출 + uint8 패킹
    - [ ] 아웃라이어 norm atomicAdd 계산
  - [ ] `tq_qjl_score_kernel<<<>>>` — Attention score
    - [ ] Query 투영: `query × projection_score`
    - [ ] XOR + popcount → Hamming distance
    - [ ] Norm-weighted score 재구성
    - [ ] Warp-level reduction
  - [ ] `tq_qjl_gqa_score_kernel<<<>>>` — GQA 전용
- [ ] `tests/test_cuda_qjl.cpp`

### M3.4 TurboQuant CUDA 커널

- [x] `src/backend/cuda/tq_turbo.cu` — 합성 커널
  - [ ] `tq_turbo_quantize_kernel<<<>>>` — PolarQuant + QJL 잔여를 1-pass
  - [ ] `tq_turbo_attention_kernel<<<>>>` — 2단계 score 계산
- [ ] `tests/test_cuda_turbo.cpp`

### M3.5 Fused Cache 커널

- [x] `src/backend/cuda/tq_fused_cache.cu` (vLLM reshape_and_cache 패턴)
  - [ ] `tq_fused_quantize_and_cache_kernel<<<>>>` — 양자화 + 캐시 기록 퓨전
    - [ ] 템플릿 파라미터: `<scalar_t, tq_type, BLOCK_SIZE>`
    - [ ] slot_mapping 기반 물리 블록 주소 계산
    - [ ] 양자화 → 블록 기록을 단일 커널에서
  - [ ] `tq_fused_attention_kernel<<<>>>` — 블록 테이블 기반 paged attention
    - [ ] 블록별 양자화 타입 확인 → dispatch
    - [ ] 혼합 정밀도 블록 처리 (Progressive Compression 지원)
- [ ] `tests/test_cuda_fused.cpp`

### M3.6 Value 양자화 CUDA 커널

- [x] `src/backend/cuda/tq_value.cu`
  - [ ] `tq_value_quantize_kernel<<<>>>` — 그룹별 min-max 양자화 + 패킹
  - [ ] `tq_value_dequant_matmul_kernel<<<>>>` — fused 역양자화 + matmul
- [ ] `tests/test_cuda_value.cpp`

### M3.7 CUDA 디스패치 및 통합

- [x] `src/backend/cuda/tq_cuda_dispatch.cu`
  - [ ] 타입 트레이트 함수 포인터를 CUDA 커널 래퍼로 설정
  - [ ] `tq_init_cuda_backend()` — 디바이스 감지 + 초기화
  - [ ] 스트림/이벤트 관리
- [x] CUDA 프로파일링
  - [ ] nsight-compute 기반 커널 점유율 분석
  - [ ] shared memory / register 사용량 최적화
  - [ ] 메모리 대역폭 활용률 확인

---

## M4: Metal Backend (MPS)

### M4.1 Metal 빌드 인프라

- [x] `src/backend/metal/CMakeLists.txt`
  - [ ] `.metal` 셰이더 컴파일 규칙
  - [ ] Metal framework 링크
  - [ ] macOS 13+ 최소 버전 설정
- [x] `src/backend/metal/tq_metal_common.h` — Metal 호스트 유틸리티
  - [ ] MTLDevice, MTLCommandQueue 관리
  - [ ] 파이프라인 캐시 (커널 컴파일 비용 최소화)
  - [ ] 버퍼 관리 유틸리티

### M4.2 PolarQuant Metal 셰이더

- [x] `src/backend/metal/tq_polar.metal`
  - [ ] `polar_quantize` compute kernel
    - [ ] Threadgroup memory에 key 로드
    - [ ] `atan2()`, `sqrt()` 활용
    - [ ] SIMD-group level min/max reduction
    - [ ] 양자화 + 패킹
  - [ ] `polar_attention` compute kernel
    - [ ] cos/sin lookup 테이블 (threadgroup memory)
    - [ ] Query × lookup → gather → score
    - [ ] SIMD-group reduction
- [ ] `tests/test_metal_polar.mm` — Metal vs CPU 레퍼런스 일치 검증

### M4.3 QJL Metal 셰이더

- [x] `src/backend/metal/tq_qjl.metal`
  - [ ] `qjl_quantize` compute kernel
    - [ ] 행렬-벡터 곱 + sign 비트 추출
    - [ ] Threadgroup memory 기반 아웃라이어 마스크
    - [ ] uint8 비트 패킹
  - [ ] `qjl_attention` compute kernel
    - [ ] XOR + popcount (Metal `popcount()` 빌트인)
    - [ ] Norm-weighted score 재구성
- [ ] `tests/test_metal_qjl.mm`

### M4.4 TurboQuant + Fused Metal 셰이더

- [x] `src/backend/metal/tq_turbo.metal` — 합성 커널
- [x] `src/backend/metal/tq_fused_cache.metal` — Fused 캐시 커널
- [x] `src/backend/metal/tq_value.metal` — Value 양자화 커널

### M4.5 Metal 호스트 코드

- [x] `src/backend/metal/tq_metal_dispatch.m` (Objective-C)
  - [ ] 셰이더 라이브러리 로드 + 파이프라인 생성
  - [ ] 타입 트레이트 함수 포인터를 Metal 래퍼로 설정
  - [ ] `tq_init_metal_backend()` — 디바이스/큐 초기화
  - [ ] 커맨드 버퍼/인코더 관리
  - [ ] Unified Memory 활용 (CPU/GPU 공유 버퍼)

### M4.6 Metal 성능 최적화

- [x] GPU Capture (Xcode) 기반 프로파일링
- [x] Threadgroup 크기 튜닝 (M1/M2/M3별)
- [x] SIMD-group 폭 활용 (32 threads)
- [x] 메모리 접근 패턴 최적화 (coalescing)

---

## M5: Integration (프레임워크 통합)

### M5.1 llama.cpp 플러그인

- [x] `integrations/llamacpp/tq_ggml_type.h` — GGML 타입 등록
  - [ ] 새 `GGML_TYPE_TQ_*` 열거형 값 정의
  - [ ] `ggml_type_traits` 테이블에 TurboQuant 타입 추가
  - [ ] `from_float`, `to_float` 함수 연결
- [x] `integrations/llamacpp/tq_kv_cache.cpp` — KV 캐시 백엔드
  - [ ] `--kv-cache-type turbo3` CLI 옵션 추가
  - [ ] KV 캐시 할당 시 TurboQuant 캐시 생성
  - [ ] Attention 연산 시 TurboQuant attention 호출
- [x] `integrations/llamacpp/README.md` — 통합 가이드
- [ ] llama.cpp 빌드에 TurboQuant 의존성 추가하는 CMake 패치
- [ ] 통합 테스트: llama.cpp + TurboQuant로 실제 모델 추론

### M5.2 Python 바인딩

- [x] `bindings/python/turboquant/__init__.py`
- [x] `bindings/python/turboquant/_core.pyx` 또는 pybind11 모듈
  - [ ] `TurboQuantContext` 클래스
  - [ ] `quantize_keys()`, `quantize_values()` 메서드
  - [ ] `attention()` 메서드
  - [ ] NumPy 배열 입출력
- [x] `bindings/python/setup.py` — pip install 지원
- [x] `bindings/python/tests/test_python.py` — Python 바인딩 테스트
  - [ ] NumPy 레퍼런스 대비 결과 검증
  - [ ] CPU/CUDA/Metal 백엔드 전환 테스트

### M5.3 vLLM 통합

- [x] `integrations/vllm/tq_cache_engine.py` — 커스텀 캐시 엔진
  - [ ] `kv_cache_dtype="turbo3"` 옵션 지원
  - [ ] reshape_and_cache 오버라이드
  - [ ] PagedAttention 블록 테이블 호환
- [x] `integrations/vllm/README.md` — 통합 가이드
- [ ] 통합 테스트: vLLM + TurboQuant 서빙 테스트

### M5.4 예제 코드

- [x] `examples/standalone.c` — 순수 C에서 TurboQuant 사용
- [x] `examples/llamacpp_integration.cpp` — llama.cpp 통합 예제
- [x] `examples/python_quickstart.py` — Python 빠른 시작

---

## M6: Validation (검증)

### M6.1 정확도 벤치마크

- [x] `bench/accuracy/run_longbench.py` — LongBench 벤치마크
  - [ ] FP16 baseline 대비 각 양자화 타입의 F1 score 비교
  - [ ] narrativeqa, qasper, hotpotqa, gov_report, samsum 등
  - [ ] Llama-3-8B, Mistral-7B, Gemma-2B 모델 테스트
- [x] `bench/accuracy/run_niah.py` — Needle-in-a-Haystack
  - [ ] 4K, 8K, 16K, 32K, 64K, 128K 컨텍스트 길이별 정확도
  - [ ] 각 양자화 타입별 "바늘" 검색 성공률
- [x] `bench/accuracy/run_ruler.py` — RULER 벤치마크
- [x] 결과 정리: `bench/accuracy/results/` 디렉토리에 JSON + 시각화

### M6.2 성능 벤치마크

- [x] `bench/performance/bench_memory.cpp` — 메모리 사용량
  - [ ] 컨텍스트 길이별 KV 캐시 메모리 측정
  - [ ] FP16 vs Q4_0 vs TurboQuant 3-bit 비교
  - [ ] Progressive Compression 효과 측정
- [x] `bench/performance/bench_latency.cpp` — 지연시간
  - [ ] Prefill 지연 (양자화 오버헤드 포함)
  - [ ] Decode 토큰당 지연 (attention 계산 포함)
  - [ ] CPU(AVX2) / CPU(NEON) / CUDA / Metal 각각 측정
- [x] `bench/performance/bench_throughput.cpp` — 처리량
  - [x] 초당 토큰 생성 수
  - [x] 배치 크기별 처리량 변화
- [x] `bench/performance/bench_kernel.cpp` — 개별 커널 성능
  - [x] 양자화 커널 처리량 (elements/ms)
  - [x] Attention 커널 처리량
  - [x] 각 백엔드별 비교

### M6.3 크로스 플랫폼 검증

- [ ] CPU Generic → CPU AVX2 결과 일치 검증
- [x] CPU Generic → CPU NEON 결과 일치 검증
- [ ] CPU Generic → CUDA 결과 일치 검증 (부동소수점 허용 오차 내)
- [ ] CPU Generic → Metal 결과 일치 검증 (부동소수점 허용 오차 내)

### M6.4 안정성 검증

- [x] AddressSanitizer (ASan) 클린 통과
- [x] UndefinedBehaviorSanitizer (UBSan) 클린 통과
- [x] ThreadSanitizer (TSan) 클린 통과 (멀티스레드 코드)
- [ ] Valgrind memcheck 메모리 누수 제로
- [ ] 장시간 실행 테스트 (100K+ 토큰 생성)

### M6.5 문서화

- [x] `README.md` 최종 작성
  - [ ] Quick Start (30초 내 빌드 + 실행)
  - [ ] 벤치마크 결과 요약 테이블
  - [ ] 지원 플랫폼 매트릭스
- [x] API 문서: Doxygen 설정 + 생성
- [x] `docs/architecture.md` — 아키텍처 설명서
- [x] `docs/integration_guide.md` — 프레임워크 통합 가이드
- [x] `docs/benchmarks.md` — 벤치마크 결과 상세

### M6.6 릴리즈 준비

- [x] `CHANGELOG.md` 작성
- [x] `LICENSE` 파일 (Apache 2.0 또는 MIT)
- [ ] GitHub Release v0.1.0 태그
- [ ] 릴리즈 노트 작성
- [ ] 소개 블로그 포스트 초안

---

## 우선순위 범례

| 레이블 | 의미 |
|--------|------|
| **P0** | 필수 — v0.1 릴리즈에 반드시 포함 |
| **P1** | 중요 — v0.1에 포함 목표, 일정 압박 시 v0.2로 이동 가능 |
| **P2** | 향후 — v0.2 이후 |

## 마일스톤별 의존 관계

```
M0 ──→ M1 ──→ M2 ──┐
                    ├──→ M5 ──→ M6
       M1 ──→ M3 ──┤
                    │
       M1 ──→ M4 ──┘
```

- M0 (Foundation) 완료 후 M1 (CPU Core) 시작 가능
- M1 완료 후 M2 (CPU SIMD), M3 (CUDA), M4 (Metal) **병렬** 진행 가능
- M2, M3, M4 중 하나 이상 완료 후 M5 (Integration) 시작 가능
- M5 완료 후 M6 (Validation) 시작
