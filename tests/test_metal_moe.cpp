/**
 * test_metal_moe.cpp — Minimal Metal IQ2_XXS matmul kernel test
 *
 * Isolates the Metal matmul_iq2_xxs shader from the fused MoE dispatch
 * to determine whether a hang originates in the shader itself or in
 * the MoE dispatch logic.
 *
 * If this test hangs: the IQ2_XXS Metal shader is broken.
 * If this test passes: the fused MoE dispatch has the bug.
 */
#include <gtest/gtest.h>

#ifndef TQ_HAS_METAL

TEST(MetalMatmul, SkipNoMetal) {
    GTEST_SKIP() << "Metal backend not compiled (TQ_HAS_METAL not defined)";
}

#else /* TQ_HAS_METAL */

#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>

/* Relative tolerance: allow 1e-5 relative error or 0.01 absolute (whichever is larger) */
static void expect_near_rel(float cpu, float gpu, int row) {
    float tol = std::max(0.01f, std::fabs(cpu) * 1e-5f);
    EXPECT_NEAR(cpu, gpu, tol)
        << "Row " << row << ": CPU=" << cpu << " GPU=" << gpu
        << " tol=" << tol;
}

extern "C" {
#include "turboquant/tq_gguf.h"

/* Internal Metal dispatch functions (not in public header) */
int tq_metal_available(void);
int tq_metal_matmul_gguf(float* out, const float* x, const void* weight,
                          tq_ggml_dtype weight_type, int out_dim, int in_dim);
}

/**
 * Zero-weight smoke test: all-zero IQ2_XXS blocks should produce zero output.
 *
 * IQ2_XXS format: 66 bytes per 256-element block
 *   - 2 bytes: FP16 scale (d)
 *   - 64 bytes: 8 sub-blocks of 8 bytes each (2B grid index + 6B signs/scale)
 *
 * With d=0 (zero scale), any grid values * 0 = 0, so output must be zero.
 */
TEST(MetalMatmul, IQ2_XXS_ZeroWeights) {
    if (!tq_metal_available()) {
        GTEST_SKIP() << "Metal device not available on this machine";
    }

    /* Use 8 rows (not 4) to avoid weight cache collision with SmallMatrix test.
     * The Metal weight cache keys on (pointer, size); malloc may reuse the
     * same address after free, returning a stale GPU buffer. */
    const int out_dim = 8;
    const int in_dim  = 256;

    /* IQ2_XXS: 66 bytes per 256 elements */
    const size_t block_size = 66;
    const size_t weight_bytes = (size_t)out_dim * block_size;

    uint8_t* weight = (uint8_t*)calloc(weight_bytes, 1);
    ASSERT_NE(weight, nullptr);
    memset(weight, 0, weight_bytes);

    /* Uniform input vector */
    float input[256];
    for (int i = 0; i < in_dim; i++) input[i] = 1.0f;

    std::vector<float> output_gpu(out_dim, 999.0f);
    std::vector<float> output_cpu(out_dim, 999.0f);

    /* GPU path */
    int rc = tq_metal_matmul_gguf(output_gpu.data(), input, weight,
                                   TQ_GGML_TYPE_IQ2_XXS,
                                   out_dim, in_dim);
    ASSERT_EQ(0, rc) << "Metal matmul dispatch failed (returned " << rc << ")";

    /* CPU reference */
    tq_matmul_gguf(output_cpu.data(), input, weight,
                   TQ_GGML_TYPE_IQ2_XXS, out_dim, in_dim);

    /* Both should be zero (or at least match) */
    for (int i = 0; i < out_dim; i++) {
        expect_near_rel(output_cpu[i], output_gpu[i], i);
    }

    free(weight);
}

/**
 * Small non-trivial test: 4 rows x 256 cols with known-pattern weights.
 *
 * We fill IQ2_XXS blocks with a simple repeating pattern and verify
 * CPU and GPU produce the same results. We do not need exact numerical
 * correctness — just GPU/CPU agreement.
 */
TEST(MetalMatmul, IQ2_XXS_SmallMatrix) {
    if (!tq_metal_available()) {
        GTEST_SKIP() << "Metal device not available on this machine";
    }

    const int out_dim = 4;
    const int in_dim  = 256;
    const size_t block_size = 66;
    const size_t weight_bytes = (size_t)out_dim * block_size;

    /* Fill with deterministic non-zero pattern */
    uint8_t* weight = (uint8_t*)malloc(weight_bytes);
    ASSERT_NE(weight, nullptr);
    for (size_t i = 0; i < weight_bytes; i++) {
        weight[i] = (uint8_t)((i * 37 + 13) & 0xFF);
    }

    /* Random-ish input vector */
    float input[256];
    for (int i = 0; i < in_dim; i++) {
        input[i] = sinf((float)i * 0.1f);
    }

    float output_gpu[4] = {0};
    float output_cpu[4] = {0};

    /* CPU reference first (known to work) */
    tq_matmul_gguf(output_cpu, input, weight,
                   TQ_GGML_TYPE_IQ2_XXS, out_dim, in_dim);

    /* GPU path */
    int rc = tq_metal_matmul_gguf(output_gpu, input, weight,
                                   TQ_GGML_TYPE_IQ2_XXS,
                                   out_dim, in_dim);
    ASSERT_EQ(0, rc) << "Metal matmul dispatch failed";

    /* Compare GPU vs CPU */
    for (int i = 0; i < out_dim; i++) {
        expect_near_rel(output_cpu[i], output_gpu[i], i);
    }

    free(weight);
}

/**
 * Larger matrix: 16 rows x 256 cols.
 * Tests that multi-row dispatch works correctly.
 * Uses 16 rows (not 8) to avoid weight cache collision with the zero test.
 */
TEST(MetalMatmul, IQ2_XXS_16Rows) {
    if (!tq_metal_available()) {
        GTEST_SKIP() << "Metal device not available on this machine";
    }

    const int out_dim = 16;
    const int in_dim  = 256;
    const size_t block_size = 66;
    const size_t weight_bytes = (size_t)out_dim * block_size;

    uint8_t* weight = (uint8_t*)malloc(weight_bytes);
    ASSERT_NE(weight, nullptr);
    for (size_t i = 0; i < weight_bytes; i++) {
        weight[i] = (uint8_t)((i * 53 + 7) & 0xFF);
    }

    float input[256];
    for (int i = 0; i < in_dim; i++) {
        input[i] = cosf((float)i * 0.05f) * 0.5f;
    }

    std::vector<float> output_gpu(out_dim, 0.0f);
    std::vector<float> output_cpu(out_dim, 0.0f);

    tq_matmul_gguf(output_cpu.data(), input, weight,
                   TQ_GGML_TYPE_IQ2_XXS, out_dim, in_dim);

    int rc = tq_metal_matmul_gguf(output_gpu.data(), input, weight,
                                   TQ_GGML_TYPE_IQ2_XXS,
                                   out_dim, in_dim);
    ASSERT_EQ(0, rc) << "Metal matmul dispatch failed";

    for (int i = 0; i < out_dim; i++) {
        expect_near_rel(output_cpu[i], output_gpu[i], i);
    }

    free(weight);
}

/**
 * IQ2_S zero-weight smoke test: all-zero IQ2_S blocks should produce zero output.
 *
 * IQ2_S format: 82 bytes per 256-element block
 *   - 2 bytes: FP16 scale (d)
 *   - 32 bytes: qs (grid index low 8 bits)
 *   - 32 bytes: signs (sign bitmasks)
 *   - 8 bytes: qh (grid index high 2 bits)
 *   - 8 bytes: scales (4-bit sub-block scales)
 *
 * With d=0 (zero scale), output must be zero regardless of other fields.
 */
TEST(MetalMatmul, IQ2_S_ZeroWeights) {
    if (!tq_metal_available()) {
        GTEST_SKIP() << "Metal device not available on this machine";
    }

    /* Use 8 rows (not 4) for zero test to avoid weight cache collision with
     * the SmallMatrix test — see tq_get_weight_buffer cache keyed on (ptr, size) */
    const int out_dim = 8;
    const int in_dim  = 256;

    /* IQ2_S: 82 bytes per 256 elements */
    const size_t block_size = 82;
    const size_t weight_bytes = (size_t)out_dim * block_size;

    uint8_t* weight = (uint8_t*)calloc(weight_bytes, 1);
    ASSERT_NE(weight, nullptr);

    /* Uniform input vector */
    float input[256];
    for (int i = 0; i < in_dim; i++) input[i] = 1.0f;

    float output_gpu[8] = {999, 999, 999, 999, 999, 999, 999, 999};
    float output_cpu[8] = {999, 999, 999, 999, 999, 999, 999, 999};

    /* GPU path */
    int rc = tq_metal_matmul_gguf(output_gpu, input, weight,
                                   TQ_GGML_TYPE_IQ2_S,
                                   out_dim, in_dim);
    ASSERT_EQ(0, rc) << "Metal IQ2_S matmul dispatch failed (returned " << rc << ")";

    /* CPU reference */
    tq_matmul_gguf(output_cpu, input, weight,
                   TQ_GGML_TYPE_IQ2_S, out_dim, in_dim);

    /* Both should be zero (or at least match) */
    for (int i = 0; i < out_dim; i++) {
        expect_near_rel(output_cpu[i], output_gpu[i], i);
    }

    free(weight);
}

/**
 * IQ2_S small non-trivial test: 4 rows x 256 cols with known-pattern weights.
 *
 * Unlike IQ2_XXS, IQ2_S uses a 10-bit grid index. We must ensure qh values
 * don't push grid_idx beyond 1023.
 *
 * Block layout (82 bytes):
 *   [0..1]   d (fp16)
 *   [2..33]  qs[32] (grid index low 8 bits, 4 per sub-block)
 *   [34..65] signs[32] (sign bitmasks)
 *   [66..73] qh[8] (grid index high 2 bits per sub-block)
 *   [74..81] scales[8] (4-bit packed sub-block scales)
 */
TEST(MetalMatmul, IQ2_S_SmallMatrix) {
    if (!tq_metal_available()) {
        GTEST_SKIP() << "Metal device not available on this machine";
    }

    const int out_dim = 4;
    const int in_dim  = 256;
    const size_t block_size = 82;
    const size_t weight_bytes = (size_t)out_dim * block_size;

    uint8_t* weight = (uint8_t*)calloc(weight_bytes, 1);
    ASSERT_NE(weight, nullptr);

    /* Fill with a controlled pattern that produces valid grid indices.
     * Grid index = qs[l] | ((qh[ib32] << (8 - 2*l)) & 0x300)
     * Max valid index is 1023 = 0x3FF, so we keep qh values small. */
    for (int row = 0; row < out_dim; row++) {
        uint8_t* blk = weight + row * block_size;

        /* Set scale to a non-zero FP16 value (0x3C00 = 1.0f in fp16) */
        blk[0] = 0x00;
        blk[1] = 0x3C;

        /* Fill qs with small values (grid indices 0..127) */
        for (int i = 0; i < 32; i++) {
            blk[2 + i] = (uint8_t)((row * 37 + i * 13) & 0x7F);
        }

        /* Fill signs with alternating pattern */
        for (int i = 0; i < 32; i++) {
            blk[34 + i] = (uint8_t)((i & 1) ? 0xAA : 0x55);
        }

        /* Keep qh values small to avoid index overflow.
         * Each qh byte provides 2 bits per sub-block for 4 groups:
         * l=0 uses bits 0-1, l=1 uses bits 2-3, l=2 uses bits 4-5, l=3 uses bits 6-7
         * Max contribution: 3 << 8 = 768, plus max qs = 127, total = 895 < 1024 */
        for (int i = 0; i < 8; i++) {
            blk[66 + i] = (uint8_t)((row + i) & 0xFF);
        }

        /* Sub-block scales: each byte has two 4-bit scales */
        for (int i = 0; i < 8; i++) {
            blk[74 + i] = 0x55; /* both scales = 5 */
        }
    }

    /* Input vector */
    float input[256];
    for (int i = 0; i < in_dim; i++) {
        input[i] = sinf((float)i * 0.1f);
    }

    float output_gpu[4] = {0};
    float output_cpu[4] = {0};

    /* CPU reference first */
    tq_matmul_gguf(output_cpu, input, weight,
                   TQ_GGML_TYPE_IQ2_S, out_dim, in_dim);

    /* GPU path */
    int rc = tq_metal_matmul_gguf(output_gpu, input, weight,
                                   TQ_GGML_TYPE_IQ2_S,
                                   out_dim, in_dim);
    ASSERT_EQ(0, rc) << "Metal IQ2_S matmul dispatch failed";

    /* Compare GPU vs CPU */
    for (int i = 0; i < out_dim; i++) {
        expect_near_rel(output_cpu[i], output_gpu[i], i);
    }

    free(weight);
}

/**
 * IQ2_S 8-row test: exercises multi-row dispatch.
 */
TEST(MetalMatmul, IQ2_S_16Rows) {
    if (!tq_metal_available()) {
        GTEST_SKIP() << "Metal device not available on this machine";
    }

    /* Use 16 rows to avoid weight cache collision with 8-row zero test */
    const int out_dim = 16;
    const int in_dim  = 256;
    const size_t block_size = 82;
    const size_t weight_bytes = (size_t)out_dim * block_size;

    uint8_t* weight = (uint8_t*)calloc(weight_bytes, 1);
    ASSERT_NE(weight, nullptr);

    /* Fill with valid IQ2_S data */
    for (int row = 0; row < out_dim; row++) {
        uint8_t* blk = weight + row * block_size;

        /* FP16 scale: 0x3800 = 0.5f */
        blk[0] = 0x00;
        blk[1] = 0x38;

        /* qs: small grid indices */
        for (int i = 0; i < 32; i++) {
            blk[2 + i] = (uint8_t)((row * 53 + i * 7) & 0x7F);
        }

        /* signs */
        for (int i = 0; i < 32; i++) {
            blk[34 + i] = (uint8_t)((row * 11 + i * 3) & 0xFF);
        }

        /* qh: keep grid indices in range */
        for (int i = 0; i < 8; i++) {
            blk[66 + i] = (uint8_t)((row + i * 5) & 0xFF);
        }

        /* scales */
        for (int i = 0; i < 8; i++) {
            blk[74 + i] = 0x33; /* scales = 3 and 3 */
        }
    }

    float input[256];
    for (int i = 0; i < in_dim; i++) {
        input[i] = cosf((float)i * 0.05f) * 0.5f;
    }

    std::vector<float> output_gpu(out_dim, 0.0f);
    std::vector<float> output_cpu(out_dim, 0.0f);

    tq_matmul_gguf(output_cpu.data(), input, weight,
                   TQ_GGML_TYPE_IQ2_S, out_dim, in_dim);

    int rc = tq_metal_matmul_gguf(output_gpu.data(), input, weight,
                                   TQ_GGML_TYPE_IQ2_S,
                                   out_dim, in_dim);
    ASSERT_EQ(0, rc) << "Metal IQ2_S matmul dispatch failed";

    for (int i = 0; i < out_dim; i++) {
        expect_near_rel(output_cpu[i], output_gpu[i], i);
    }

    free(weight);
}

#endif /* TQ_HAS_METAL */
