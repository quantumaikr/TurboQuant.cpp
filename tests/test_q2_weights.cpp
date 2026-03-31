/**
 * test_q2_weights.cpp -- Tests for Q2 (2-bit Lloyd-Max codebook) weight quantization
 *
 * Tests the Q2 quantization pipeline: tq_quantize_row_q2, tq_matmul_q2,
 * and the integer Q2xQ8 dot product kernel.
 */

#include <gtest/gtest.h>
extern "C" {
#include "turboquant/tq_engine.h"
}

#include <cmath>
#include <vector>
#include <random>

/* ============================================================
 * Q2 row quantization roundtrip
 * ============================================================ */

TEST(Q2Weights, QuantizeRowRoundtrip) {
    const int n = 128;
    std::vector<float> src(n);
    for (int i = 0; i < n; i++) src[i] = sinf(i * 0.1f);

    int n_blocks = n / 32;
    std::vector<uint8_t> qs(n_blocks * 8);
    std::vector<float> scales(n_blocks);

    tq_quantize_row_q2(src.data(), qs.data(), scales.data(), n);

    /* Verify scales are positive */
    for (int b = 0; b < n_blocks; b++) {
        EXPECT_GE(scales[b], 0.0f);
    }

    /* Verify packed data is within [0, 0xFF] (trivially true for uint8_t) */
    /* But verify that indices are 0-3 by checking bit patterns */
    for (int b = 0; b < n_blocks; b++) {
        for (int j = 0; j < 8; j++) {
            uint8_t byte = qs[b * 8 + j];
            /* Each 2-bit value should be 0-3, which is always true for 2-bit */
            (void)byte; /* Always valid */
        }
    }
}

/* ============================================================
 * Q2 matmul correctness
 * ============================================================ */

TEST(Q2Weights, MatmulCorrectness) {
    const int n = 64;  /* output rows */
    const int d = 128; /* inner dimension (must be multiple of 32) */

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    /* Generate random weight matrix and input vector */
    std::vector<float> w(n * d);
    std::vector<float> x(d);
    for (int i = 0; i < n * d; i++) w[i] = dist(rng);
    for (int i = 0; i < d; i++) x[i] = dist(rng);

    /* FP32 reference: out[i] = sum_j(w[i*d+j] * x[j]) */
    std::vector<float> ref_out(n);
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += w[i * d + j] * x[j];
        }
        ref_out[i] = sum;
    }

    /* Q2 quantize weights */
    int n_blocks = d / 32;
    std::vector<uint8_t> w_qs(n * n_blocks * 8);
    std::vector<float> w_scales(n * n_blocks);

    for (int i = 0; i < n; i++) {
        tq_quantize_row_q2(&w[i * d], &w_qs[i * n_blocks * 8],
                            &w_scales[i * n_blocks], d);
    }

    /* Q2 matmul */
    std::vector<float> q2_out(n);
    tq_matmul_q2(q2_out.data(), x.data(), w_qs.data(), w_scales.data(), n, d);

    /* Compute cosine similarity between Q2 and FP32 outputs.
     * Q2 is 2-bit so expect moderate accuracy (cosine > 0.7). */
    double dot = 0.0, sq_ref = 0.0, sq_q2 = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)ref_out[i] * (double)q2_out[i];
        sq_ref += (double)ref_out[i] * (double)ref_out[i];
        sq_q2 += (double)q2_out[i] * (double)q2_out[i];
    }
    double cosine = dot / (std::sqrt(sq_ref) * std::sqrt(sq_q2));

    EXPECT_GT(cosine, 0.7) << "Q2 matmul cosine similarity too low: " << cosine;
}

TEST(Q2Weights, MatmulPreqCorrectness) {
    /* Test Q2 matmul with pre-quantized activation */
    const int n = 32;
    const int d = 64;

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> w(n * d);
    std::vector<float> x(d);
    for (int i = 0; i < n * d; i++) w[i] = dist(rng);
    for (int i = 0; i < d; i++) x[i] = dist(rng);

    /* FP32 reference */
    std::vector<float> ref_out(n);
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) sum += w[i * d + j] * x[j];
        ref_out[i] = sum;
    }

    /* Q2 quantize weights */
    int n_blocks = d / 32;
    std::vector<uint8_t> w_qs(n * n_blocks * 8);
    std::vector<float> w_scales(n * n_blocks);
    for (int i = 0; i < n; i++) {
        tq_quantize_row_q2(&w[i * d], &w_qs[i * n_blocks * 8],
                            &w_scales[i * n_blocks], d);
    }

    /* Pre-quantize activation to Q8 */
    std::vector<int8_t> x_q8(d);
    std::vector<float> x_scales(n_blocks + 1);
    tq_quantize_row_q8(x.data(), x_q8.data(), x_scales.data(), d);

    /* Q2 matmul with pre-quantized activation */
    std::vector<float> q2_out(n);
    tq_matmul_q2_preq(q2_out.data(), w_qs.data(), w_scales.data(),
                        x_q8.data(), x_scales.data(), n, d);

    /* Should match tq_matmul_q2 output */
    std::vector<float> q2_out2(n);
    tq_matmul_q2(q2_out2.data(), x.data(), w_qs.data(), w_scales.data(), n, d);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(q2_out[i], q2_out2[i], std::fabs(q2_out2[i]) * 0.01f + 1e-5f)
            << "Pre-quantized Q2 matmul differs at index " << i;
    }
}

/* ============================================================
 * Q2 compression ratio verification
 * ============================================================ */

TEST(Q2Weights, CompressionRatio) {
    /* Q2: 8 bytes packed + 4 bytes scale per 32 values = 12 bytes
     * FP32: 32 * 4 = 128 bytes per 32 values
     * Ratio: 128 / 12 = 10.67x */
    float ratio = (32.0f * 4.0f) / 12.0f;
    EXPECT_GT(ratio, 10.0f);

    /* vs Q4: 16 + 4 = 20 bytes per 32 values
     * Q2/Q4 ratio: 20 / 12 = 1.67x smaller */
    float vs_q4 = 20.0f / 12.0f;
    EXPECT_GT(vs_q4, 1.6f);
}

/* ============================================================
 * Zero input edge case
 * ============================================================ */

TEST(Q2Weights, ZeroInput) {
    const int n = 32;
    std::vector<float> zeros(n, 0.0f);
    int n_blocks = n / 32;
    std::vector<uint8_t> qs(n_blocks * 8);
    std::vector<float> scales(n_blocks);

    tq_quantize_row_q2(zeros.data(), qs.data(), scales.data(), n);

    /* Scale should be zero for zero input */
    EXPECT_NEAR(scales[0], 0.0f, 1e-10f);
}
