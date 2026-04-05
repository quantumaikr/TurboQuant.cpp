# quant.cpp — Agent Development Guide

## Project Vision

**"LLM의 SQLite"** — 가장 작고, 가장 읽기 쉽고, 가장 쉽게 임베딩할 수 있는 LLM 엔진.

Two directions:
1. **Embedding Engine**: quant.h 단일 헤더(15K LOC)로 어디든 LLM 추가
2. **KV Compression Research**: 함수 3개로 새 양자화 타입 추가 가능한 연구 플랫폼

**Non-goals**: GPU 속도 경쟁 (llama.cpp 영역), 배치 서빙 (vLLM 영역), 학습

## Project Overview

quant.cpp is a minimal C inference engine for local LLM with KV cache compression.
67K LOC, pure C, zero dependencies. Supports 7 architectures via GGUF.
Killer feature: KV cache compression — 7x compression with PPL +0.0% vs FP32.
Ships as quant.h (15K LOC single header) and WASM (192KB).

## Architecture

```
include/turboquant/   — Public C API (turboquant.h, tq_types.h, tq_spec.h)
src/core/             — Algorithms (tq_polar.c, tq_qjl.c, tq_turbo.c, tq_uniform.c, tq_traits.c)
src/cache/            — Paged cache + progressive compression
src/backend/cpu/      — CPU kernels (generic, AVX2, NEON)
src/backend/cuda/     — CUDA kernels
src/backend/metal/    — Metal compute shaders (7 kernels: matmul, rmsnorm, rope, attention, etc.)
src/engine/           — GGUF loader, transformer forward, tokenizer, generate
tests/                — Google Test unit tests (34 tests)
wasm/                 — Browser demo (quant.wasm 192KB + index.html)
docs/                 — API reference, custom quantization guide, tech report
examples/             — Embedding examples (minimal, chat, kv_compare)
bench/                — Performance benchmarks
spec/                 — Format specification + test vectors
integrations/         — llama.cpp, vLLM, Python bindings
harness/              — Autonomous development harness (run.sh, team.toml)
```

## Key Documents

- `docs/prd_v0.1.md` — Full product requirements
- `docs/wbs_v0.1.md` — Work breakdown structure with checklists
- `program.md` — Current agent task specification (READ THIS FIRST)
- `score.sh` — Automated 5-dimension scoring (0.0 ~ 1.0)

## Reference Implementations (DO NOT MODIFY)

- `refs/QJL/` — QJL Python/CUDA implementation (Amir Zandieh)
- `refs/PolarQuant/` — PolarQuant Python/Triton implementation
- `refs/llama.cpp/` — llama.cpp fork with TQ1/TQ2 weight quantization
- `refs/vllm/` — vLLM KV cache infrastructure
- `refs/onnx/` — ONNX quantization operator specification

## Build & Test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_TESTS=ON -DTQ_BUILD_BENCH=ON
cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
ctest --test-dir build --output-on-failure
```

## Scoring (5-Dimension Harness)

```bash
bash score.sh              # Full 5-dimension evaluation
bash score.sh --quick      # Build + correctness only (fast iteration)
bash score.sh --bench      # Performance only
bash score.sh --quality    # Quantization quality only
```

The score measures 5 dimensions:
1. **Structure** (10w) — Headers, sources, tests, specs exist
2. **Correctness** (11w) — Builds, tests pass, zero warnings
3. **Quality** (11w) — Roundtrip MSE < 0.01, attention cosine > 0.99
4. **Performance** (14w) — Throughput, compression ratio 5x, SIMD speedup 4x
5. **Integration** (6w) — llama.cpp/vLLM/Python plugins, docs, examples

## Coding Standards

- C11 for core library, C++17 for tests/CUDA/Metal wrappers
- No external dependencies in core (libc/libm only)
- Every block struct must have static_assert size verification
- Every public function must have a unit test
- ONNX LSB-first bit-packing convention for all quantized formats
- Use refs/ code as algorithm reference — port to C, don't wrap Python

## Quantization Types

| Type | Key Bits | Algorithm | Block Size |
|------|----------|-----------|------------|
| TQ_POLAR_3B | 3 | PolarQuant | 128 |
| TQ_POLAR_4B | 4 | PolarQuant | 128 |
| TQ_QJL_1B | 1 | QJL sign hash | 256 |
| TQ_TURBO_3B | 3 | Polar 2b + QJL 1b | 128 |
| TQ_TURBO_4B | 4 | Polar 3b + QJL 1b | 128 |
| TQ_UNIFORM_4B | 4 | Min-Max | 128 |
| TQ_UNIFORM_2B | 2 | Min-Max | 128 |

## Module Ownership (Conflict-Free Zones)

Each module has exclusive file ownership. When working in parallel, agents must only modify their assigned files:

| Module | Owned Files | Dependencies |
|--------|-------------|-------------|
| `foundation` | `CMakeLists.txt`, `include/**`, `src/core/tq_traits.c` | None |
| `polar` | `src/core/tq_polar.*`, `tests/test_polar.*` | foundation |
| `qjl` | `src/core/tq_qjl.*`, `tests/test_qjl.*` | foundation |
| `turbo` | `src/core/tq_turbo.*`, `tests/test_turbo.*` | polar, qjl |
| `uniform` | `src/core/tq_uniform.*`, `src/core/tq_value_quant.*`, `tests/test_uniform.*`, `tests/test_value.*` | foundation |
| `cache` | `src/cache/**`, `tests/test_paged_cache.*`, `tests/test_progressive.*` | foundation |
| `simd-neon` | `src/backend/cpu/**` | polar, qjl |
| `gpu-cuda` | `src/backend/cuda/**` | polar, qjl |
| `gpu-metal` | `src/backend/metal/**` | polar, qjl |
| `bench` | `bench/**`, `spec/**`, `tests/reference/**` | all core |
| `integration` | `integrations/**`, `bindings/**`, `examples/**` | all |

## Agent Team & Skills (Harness Architecture)

이 프로젝트는 [Harness](https://github.com/revfactory/harness) 패턴을 적용한 에이전트 팀 구조를 사용한다.

### Agents (.claude/agents/)

| Agent | Role | Subagent Type |
|-------|------|---------------|
| `architect` | 기술 리더, 작업 분해/위임, Merge Gate | Leader |
| `core-dev` | 알고리즘 구현, 테스트 작성 | general-purpose |
| `perf-dev` | SIMD/GPU 최적화, 벤치마크 | general-purpose |
| `qa` | 경계면 교차 비교, 통합 정합성 검증 | general-purpose |

### Skills (.claude/skills/)

| Skill | Trigger | Description |
|-------|---------|-------------|
| `orchestrate` | "개발 시작", "하네스 실행" | 팀 구성 → 위임 → Merge Gate → 루프 |
| `develop` | "다음 항목", 모듈명 | Karpathy 루프: score → implement → verify |
| `score` | "점수", "현재 상태" | 5차원 자동 평가 + 병목 분석 |
| `qa` | "검증", 머지 전 | 경계면 교차 비교 (5대 경계면) |

### QA 원칙 (from refs/harness)

- **경계면 집중**: 단일 함수가 아니라 함수 간/모듈 간 데이터 흐름 검증
- **점진적 QA**: 전체 완성 후 1회가 아닌, 모듈 완성 직후 즉시 검증
- **교차 비교**: API 출력과 실제 블록 데이터를 동시에 읽고 비교
- **회귀 방지**: 발견된 버그는 반드시 테스트로 영구 방어

## Development Methodology: Hierarchical Harness

This project uses a **Hierarchical Harness** that combines two methodologies:

### Methodology A: Karpathy AutoResearch Loop
Single agent, sequential, safe. Score → modify → score → revert-if-worse → repeat.

### Methodology B: ClawTeam Multi-Agent
Multiple agents in isolated git worktrees, parallel, fast. Leader delegates, workers execute.

### Combined: Hierarchical Harness

```
Outer Loop (Leader — Karpathy pattern):
  1. bash score.sh → measure full project score
  2. Identify lowest-scoring dimension
  3. Delegate independent modules to Workers (ClawTeam)
  4. Wait for workers to complete
  5. Merge Gate: merge each worker's results one-by-one
     - If score improves or stays: keep merge
     - If score drops: revert that merge
  6. Repeat from 1

Inner Loop (Workers — each runs own Karpathy loop):
  - Work in isolated git worktree
  - Only modify files in their module ownership
  - score → modify → score → revert-if-worse
  - Report completion via clawteam task update
```

### Phase Transitions (Score-Based)

| Score Range | Phase | Strategy |
|-------------|-------|----------|
| 0.00 ~ 0.05 | Foundation | Single agent, sequential (CMake, headers, types) |
| 0.05 ~ 0.30 | Core Algorithms | 3 parallel workers (polar, qjl, uniform) |
| 0.30 ~ 0.60 | Advanced | 4 parallel workers (turbo, cache, simd, bench) |
| 0.60 ~ 1.00 | Fine-tuning | Single agent, precision optimization |

### Merge Gate Protocol

When merging worker results back to main:
1. Save current HEAD
2. Merge worker branch
3. Run `bash score.sh --quick`
4. If score dropped → `git reset --hard` to saved HEAD
5. If score OK → proceed to next worker

### Slash Commands

- `/score` — Run the scoring harness and show results
- `/develop` — Start autonomous development (one WBS item per round)
- `/harness` — Launch the full hierarchical harness
- `/spawn-team` — Spawn ClawTeam parallel workers for current phase

### Running the Harness

```bash
# Recommended: Hierarchical hybrid mode
./harness/run.sh --target 0.9

# Single agent Karpathy loop only
./harness/run.sh --single --rounds 50

# ClawTeam parallel workers only
./harness/run.sh --parallel-only

# Manual team spawn
clawteam launch harness/team.toml --goal "Build quant.cpp" --workspace
```
