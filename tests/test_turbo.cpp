#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"
void tq_turbo_quantize_ref(const float* src, void* dst, int n);
void tq_turbo_dequantize_ref(const void* src, float* dst, int n);
void tq_turbo_attention_ref(const float* query, const void* kv,
                            float* scores, int seq_len, int head_dim);
}
#include <cmath>
#include <vector>

TEST(TurboQuant, RoundtripBasic) {
    std::vector<float> input(TQ_BK);
    for (int i = 0; i < TQ_BK; i++) input[i] = sinf(i * 0.1f);

    block_tq_turbo block;
    tq_turbo_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_turbo_dequantize_ref(&block, output.data(), TQ_BK);

    // Turbo should be at least as good as polar alone
    double mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - output[i];
        mse += d * d;
    }
    mse /= TQ_BK;
    EXPECT_LT(mse, 1.0); // Bounded MSE
}

TEST(TurboQuant, AttentionAccuracy) {
    std::vector<float> key(128), query(128);
    for (int i = 0; i < 128; i++) {
        key[i] = sinf(i * 0.05f);
        query[i] = cosf(i * 0.05f);
    }

    // FP32 dot product
    float fp32_score = 0;
    for (int i = 0; i < 128; i++) fp32_score += query[i] * key[i];

    // Quantized attention
    block_tq_turbo block;
    tq_turbo_quantize_ref(key.data(), &block, 128);
    float quant_score = 0;
    tq_turbo_attention_ref(query.data(), &block, &quant_score, 1, 128);

    // Turbo composite should have reasonable accuracy
    EXPECT_NEAR(quant_score, fp32_score, fabsf(fp32_score) * 0.5f + 1.0f);
}

TEST(TurboQuant, BlockSize) {
    EXPECT_EQ(sizeof(block_tq_turbo),
              sizeof(block_tq_polar) + sizeof(block_tq_qjl));
}

TEST(TurboQuant, CompositeStructure) {
    // Verify the turbo block contains both polar and QJL parts
    block_tq_turbo block;
    memset(&block, 0, sizeof(block));

    std::vector<float> input(TQ_BK);
    for (int i = 0; i < TQ_BK; i++) input[i] = sinf(i * 0.3f);

    tq_turbo_quantize_ref(input.data(), &block, TQ_BK);

    // Polar part should have non-zero scales
    EXPECT_NE(block.polar.rscale, 0);
    EXPECT_NE(block.polar.tscale, 0);

    // QJL residual should have non-zero norm (unless residual is tiny)
    // Just check hash is not all zeros
    bool has_nonzero = false;
    for (int i = 0; i < TQ_SKETCH_DIM / 8; i++) {
        if (block.residual.hash[i] != 0) has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero);
}
