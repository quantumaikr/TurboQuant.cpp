# Module Ownership Table

각 에이전트/모듈은 아래 파일만 수정할 수 있다. 이 규칙은 병렬 작업 시 머지 충돌을 구조적으로 방지한다.

| Module | Owned Files | Dependencies |
|--------|-------------|-------------|
| `foundation` | `CMakeLists.txt`, `include/**`, `src/core/tq_traits.c`, `src/core/tq_context.c` | None |
| `polar` | `src/core/tq_polar.*`, `tests/test_polar.*` | foundation |
| `qjl` | `src/core/tq_qjl.*`, `tests/test_qjl.*` | foundation |
| `turbo` | `src/core/tq_turbo.*`, `tests/test_turbo.*` | polar, qjl |
| `uniform` | `src/core/tq_uniform.*`, `src/core/tq_value_quant.*`, `tests/test_uniform.*`, `tests/test_value.*` | foundation |
| `cache` | `src/cache/**`, `tests/test_paged_cache.*`, `tests/test_progressive.*` | foundation |
| `simd-neon` | `src/backend/cpu/**`, `tests/test_simd_*` | polar, qjl, uniform |
| `gpu-cuda` | `src/backend/cuda/**` | polar, qjl |
| `gpu-metal` | `src/backend/metal/**` | polar, qjl |
| `bench` | `bench/**`, `spec/**`, `tests/reference/**` | all core |
| `integration` | `integrations/**`, `bindings/**`, `examples/**` | all |

## 의존성 규칙

- 의존하는 모듈이 아직 미완이면, 해당 모듈 작업을 먼저 수행한다
- `turbo`는 `polar`과 `qjl`이 완료된 후에만 작업 가능
- `simd-neon`은 generic 구현이 완료된 후에만 작업 가능
