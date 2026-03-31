#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"
#include "turboquant/tq_engine.h"
}
#include <vector>
#include <cmath>
#include <cstdlib>
#include <numeric>

TEST(ValueQuant, SizeCalculation4B) {
    size_t size = tq_quantize_values_size(1, 128, 4);
    // 128 elements, block size 128, one block of uniform_4b
    EXPECT_EQ(size, sizeof(block_tq_uniform_4b));
}

TEST(ValueQuant, SizeCalculation2B) {
    size_t size = tq_quantize_values_size(1, 128, 2);
    EXPECT_EQ(size, sizeof(block_tq_uniform_2b));
}

TEST(ValueQuant, SizeCalculationMultipleKeys) {
    size_t size = tq_quantize_values_size(4, 128, 4);
    EXPECT_EQ(size, 4 * sizeof(block_tq_uniform_4b));
}

TEST(ValueQuant, InvalidBits) {
    size_t size = tq_quantize_values_size(1, 128, 3);
    EXPECT_EQ(size, 0u);
}

/* ============================================================
 * Q4 value dequantize roundtrip tests
 * ============================================================ */

TEST(ValueQuant, Q4RoundtripBasic) {
    const int n = 256;
    std::vector<float> src(n), dst(n);
    // Generate values in [-1, 1]
    for (int i = 0; i < n; i++) {
        src[i] = 2.0f * ((float)i / (float)n) - 1.0f;
    }

    int n_blocks = n / 32;
    std::vector<uint8_t> qs(n_blocks * 16);
    std::vector<float> scales(n_blocks);

    tq_quantize_row_q4(src.data(), qs.data(), scales.data(), n);
    tq_dequantize_row_q4(qs.data(), scales.data(), dst.data(), n);

    // Q4 should have reasonable MSE for a [-1,1] signal
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = (double)src[i] - (double)dst[i];
        mse += diff * diff;
    }
    mse /= n;
    EXPECT_LT(mse, 0.01) << "Q4 roundtrip MSE too high: " << mse;
}

TEST(ValueQuant, Q4RoundtripZero) {
    const int n = 64;
    std::vector<float> src(n, 0.0f), dst(n);
    int n_blocks = n / 32;
    std::vector<uint8_t> qs(n_blocks * 16);
    std::vector<float> scales(n_blocks);

    tq_quantize_row_q4(src.data(), qs.data(), scales.data(), n);
    tq_dequantize_row_q4(qs.data(), scales.data(), dst.data(), n);

    for (int i = 0; i < n; i++) {
        EXPECT_FLOAT_EQ(dst[i], 0.0f);
    }
}

/* ============================================================
 * Q2 value dequantize roundtrip tests
 * ============================================================ */

TEST(ValueQuant, Q2RoundtripBasic) {
    const int n = 256;
    std::vector<float> src(n), dst(n);
    // Generate Gaussian-like values
    srand(42);
    for (int i = 0; i < n; i++) {
        // Simple Box-Muller
        float u1 = ((float)(rand() % 10000) + 1.0f) / 10001.0f;
        float u2 = ((float)(rand() % 10000) + 1.0f) / 10001.0f;
        src[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
    }

    int n_blocks = n / 32;
    std::vector<uint8_t> qs(n_blocks * 8);
    std::vector<float> scales(n_blocks);

    tq_quantize_row_q2(src.data(), qs.data(), scales.data(), n);
    tq_dequantize_row_q2(qs.data(), scales.data(), dst.data(), n);

    // Q2 has higher error than Q4, but MSE should be bounded
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = (double)src[i] - (double)dst[i];
        mse += diff * diff;
    }
    mse /= n;
    // Q2 with Lloyd-Max on Gaussian should have SQNR ~9.3 dB
    // For unit-variance Gaussian, MSE ~ 0.12
    EXPECT_LT(mse, 0.5) << "Q2 roundtrip MSE too high: " << mse;
}

TEST(ValueQuant, Q2RoundtripZero) {
    const int n = 64;
    std::vector<float> src(n, 0.0f), dst(n);
    int n_blocks = n / 32;
    std::vector<uint8_t> qs(n_blocks * 8);
    std::vector<float> scales(n_blocks);

    tq_quantize_row_q2(src.data(), qs.data(), scales.data(), n);
    tq_dequantize_row_q2(qs.data(), scales.data(), dst.data(), n);

    for (int i = 0; i < n; i++) {
        EXPECT_FLOAT_EQ(dst[i], 0.0f);
    }
}

/* ============================================================
 * State creation with value quantization
 * ============================================================ */

TEST(ValueQuant, StateCreateQ4) {
    tq_model_config_t cfg = {};
    cfg.n_layers = 2;
    cfg.hidden_dim = 128;
    cfg.intermediate_dim = 256;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 64;
    cfg.vocab_size = 100;
    cfg.max_seq_len = 32;
    cfg.rms_norm_eps = 1e-5f;

    tq_state_t* s = tq_create_state_ex(&cfg, TQ_TYPE_UNIFORM_4B, 4);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->value_quant_bits, 4);
    EXPECT_NE(s->value_cache_qs, nullptr);
    EXPECT_NE(s->value_cache_scales, nullptr);
    EXPECT_EQ(s->value_cache_fp16, nullptr);
    EXPECT_EQ(s->value_cache, nullptr);
    tq_free_state(s);
}

TEST(ValueQuant, StateCreateQ2) {
    tq_model_config_t cfg = {};
    cfg.n_layers = 2;
    cfg.hidden_dim = 128;
    cfg.intermediate_dim = 256;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 64;
    cfg.vocab_size = 100;
    cfg.max_seq_len = 32;
    cfg.rms_norm_eps = 1e-5f;

    tq_state_t* s = tq_create_state_ex(&cfg, TQ_TYPE_UNIFORM_4B, 2);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->value_quant_bits, 2);
    EXPECT_NE(s->value_cache_qs, nullptr);
    EXPECT_NE(s->value_cache_scales, nullptr);
    tq_free_state(s);
}

TEST(ValueQuant, StateCreateDefault) {
    // value_quant_bits=0 should use FP16 when KV quant is enabled
    tq_model_config_t cfg = {};
    cfg.n_layers = 2;
    cfg.hidden_dim = 128;
    cfg.intermediate_dim = 256;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 64;
    cfg.vocab_size = 100;
    cfg.max_seq_len = 32;
    cfg.rms_norm_eps = 1e-5f;

    tq_state_t* s = tq_create_state_ex(&cfg, TQ_TYPE_UNIFORM_4B, 0);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->value_quant_bits, 0);
    EXPECT_EQ(s->use_fp16_values, 1);
    EXPECT_NE(s->value_cache_fp16, nullptr);
    EXPECT_EQ(s->value_cache_qs, nullptr);
    tq_free_state(s);
}
