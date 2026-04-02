#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"
void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
void tq_uniform_4b_attention_ref(const float* query, const void* kv,
                                  float* scores, int seq_len, int head_dim);
void tq_uniform_2b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_2b_dequantize_ref(const void* src, float* dst, int n);
void tq_uniform_2b_attention_ref(const float* query, const void* kv,
                                  float* scores, int seq_len, int head_dim);
void tq_uniform_3b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_3b_dequantize_ref(const void* src, float* dst, int n);
void tq_uniform_3b_attention_ref(const float* query, const void* kv,
                                  float* scores, int seq_len, int head_dim);
}
#include <cmath>
#include <vector>

TEST(Uniform4B, RoundtripBasic) {
    std::vector<float> input(TQ_BK);
    for (int i = 0; i < TQ_BK; i++) input[i] = sinf(i * 0.1f);

    block_tq_uniform_4b block;
    tq_uniform_4b_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_uniform_4b_dequantize_ref(&block, output.data(), TQ_BK);

    // 4-bit uniform should have low MSE
    double mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - output[i];
        mse += d * d;
    }
    mse /= TQ_BK;
    EXPECT_LT(mse, 0.01); // Very low MSE for 4-bit on [-1, 1] range
}

TEST(Uniform4B, ExtremeValues) {
    std::vector<float> input(TQ_BK);
    for (int i = 0; i < TQ_BK; i++) input[i] = (float)i / TQ_BK * 100.0f - 50.0f;

    block_tq_uniform_4b block;
    tq_uniform_4b_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_uniform_4b_dequantize_ref(&block, output.data(), TQ_BK);

    // MSE scales with range^2 / (16^2); range=100 -> step~6.67 -> MSE~3.7
    double mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - output[i];
        mse += d * d;
    }
    mse /= TQ_BK;
    EXPECT_LT(mse, 5.0); // Wider range = higher MSE but still bounded
}

TEST(Uniform4B, BlockSize) {
    EXPECT_EQ(sizeof(block_tq_uniform_4b), 4u + TQ_BK / 2);
}

TEST(Uniform2B, RoundtripBasic) {
    std::vector<float> input(TQ_BK);
    for (int i = 0; i < TQ_BK; i++) input[i] = sinf(i * 0.1f);

    block_tq_uniform_2b block;
    tq_uniform_2b_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_uniform_2b_dequantize_ref(&block, output.data(), TQ_BK);

    // 2-bit is more lossy than 4-bit
    double mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - output[i];
        mse += d * d;
    }
    mse /= TQ_BK;
    EXPECT_LT(mse, 0.15); // Higher MSE for 2-bit, but still bounded
}

TEST(Uniform2B, BlockSize) {
    EXPECT_EQ(sizeof(block_tq_uniform_2b), 4u + TQ_BK / 4);
}

TEST(Uniform4B, ConstantInput) {
    // All same value should roundtrip perfectly
    std::vector<float> input(TQ_BK, 3.14f);

    block_tq_uniform_4b block;
    tq_uniform_4b_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_uniform_4b_dequantize_ref(&block, output.data(), TQ_BK);

    for (int i = 0; i < TQ_BK; i++) {
        EXPECT_NEAR(output[i], 3.14f, 0.1f);
    }
}

TEST(Uniform4B, Attention) {
    const int head_dim = TQ_BK;
    const int seq_len = 4;

    // Create query vector
    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = cosf(i * 0.05f);

    // Create key vectors and quantize them
    std::vector<block_tq_uniform_4b> blocks(seq_len);
    std::vector<std::vector<float>> keys(seq_len, std::vector<float>(head_dim));
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < head_dim; d++)
            keys[s][d] = sinf(s * 1.0f + d * 0.1f);
        tq_uniform_4b_quantize_ref(keys[s].data(), &blocks[s], head_dim);
    }

    // Compute attention scores via the new function
    std::vector<float> scores(seq_len);
    tq_uniform_4b_attention_ref(query.data(), blocks.data(), scores.data(),
                                 seq_len, head_dim);

    // Compare with FP32 dot product on dequantized keys
    for (int s = 0; s < seq_len; s++) {
        std::vector<float> deq(head_dim);
        tq_uniform_4b_dequantize_ref(&blocks[s], deq.data(), head_dim);
        float fp32_dot = 0;
        for (int d = 0; d < head_dim; d++) fp32_dot += query[d] * deq[d];
        EXPECT_NEAR(scores[s], fp32_dot, 1e-4f);
    }
}

TEST(Uniform2B, Attention) {
    const int head_dim = TQ_BK;
    const int seq_len = 4;

    // Create query vector
    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = cosf(i * 0.05f);

    // Create key vectors and quantize them
    std::vector<block_tq_uniform_2b> blocks(seq_len);
    std::vector<std::vector<float>> keys(seq_len, std::vector<float>(head_dim));
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < head_dim; d++)
            keys[s][d] = sinf(s * 1.0f + d * 0.1f);
        tq_uniform_2b_quantize_ref(keys[s].data(), &blocks[s], head_dim);
    }

    // Compute attention scores via the new function
    std::vector<float> scores(seq_len);
    tq_uniform_2b_attention_ref(query.data(), blocks.data(), scores.data(),
                                 seq_len, head_dim);

    // Compare with FP32 dot product on dequantized keys
    for (int s = 0; s < seq_len; s++) {
        std::vector<float> deq(head_dim);
        tq_uniform_2b_dequantize_ref(&blocks[s], deq.data(), head_dim);
        float fp32_dot = 0;
        for (int d = 0; d < head_dim; d++) fp32_dot += query[d] * deq[d];
        EXPECT_NEAR(scores[s], fp32_dot, 1e-4f);
    }
}

/* ====================================================================
 * Uniform 3-bit with sub-block scales tests
 * ==================================================================== */

TEST(Uniform3B, RoundtripBasic) {
    std::vector<float> input(TQ_BK);
    for (int i = 0; i < TQ_BK; i++) input[i] = sinf(i * 0.1f);

    block_tq_uniform_3b block;
    tq_uniform_3b_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_uniform_3b_dequantize_ref(&block, output.data(), TQ_BK);

    // 3-bit with sub-block scales should have moderate MSE
    double mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - output[i];
        mse += d * d;
    }
    mse /= TQ_BK;
    // Sub-block 3-bit should be better than global 3-bit, targeting < 0.05
    EXPECT_LT(mse, 0.05);
}

TEST(Uniform3B, ExtremeValues) {
    std::vector<float> input(TQ_BK);
    for (int i = 0; i < TQ_BK; i++) input[i] = (float)i / TQ_BK * 100.0f - 50.0f;

    block_tq_uniform_3b block;
    tq_uniform_3b_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_uniform_3b_dequantize_ref(&block, output.data(), TQ_BK);

    // With wide range, MSE will be higher but sub-blocks help
    double mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - output[i];
        mse += d * d;
    }
    mse /= TQ_BK;
    // 8 levels over 25-unit sub-range: step~3.6, MSE~1.1
    EXPECT_LT(mse, 20.0);
}

TEST(Uniform3B, BlockSize) {
    // 8 (sub_scale) + 8 (sub_min) + 48 (qs) = 64 bytes
    EXPECT_EQ(sizeof(block_tq_uniform_3b), 64u);
}

TEST(Uniform3B, BitsPerElement) {
    float bpe = tq_type_bpe(TQ_TYPE_UNIFORM_3B);
    // 64 * 8 / 128 = 4.0
    EXPECT_NEAR(bpe, 4.0f, 0.01f);
}

TEST(Uniform3B, Attention) {
    const int head_dim = TQ_BK;
    const int seq_len = 4;

    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = cosf(i * 0.05f);

    std::vector<block_tq_uniform_3b> blocks(seq_len);
    std::vector<std::vector<float>> keys(seq_len, std::vector<float>(head_dim));
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < head_dim; d++)
            keys[s][d] = sinf(s * 1.0f + d * 0.1f);
        tq_uniform_3b_quantize_ref(keys[s].data(), &blocks[s], head_dim);
    }

    std::vector<float> scores(seq_len);
    tq_uniform_3b_attention_ref(query.data(), blocks.data(), scores.data(),
                                 seq_len, head_dim);

    // Compare with FP32 dot product on dequantized keys
    for (int s = 0; s < seq_len; s++) {
        std::vector<float> deq(head_dim);
        tq_uniform_3b_dequantize_ref(&blocks[s], deq.data(), head_dim);
        float fp32_dot = 0;
        for (int d = 0; d < head_dim; d++) fp32_dot += query[d] * deq[d];
        EXPECT_NEAR(scores[s], fp32_dot, 1e-4f);
    }
}

TEST(Uniform3B, SubBlockBenefit) {
    // Verify sub-block scales handle heterogeneous distributions well
    // First 32 elements in [-1, 1], next 32 in [-100, 100], etc.
    std::vector<float> input(TQ_BK);
    for (int i = 0; i < 32; i++) input[i] = sinf(i * 0.1f);           // [-1, 1]
    for (int i = 32; i < 64; i++) input[i] = (float)(i - 48) * 3.0f;  // [-48, 48]
    for (int i = 64; i < 96; i++) input[i] = sinf(i * 0.05f) * 0.1f;  // [-0.1, 0.1]
    for (int i = 96; i < 128; i++) input[i] = (float)(i - 112) * 0.5f; // [-8, 8]

    block_tq_uniform_3b block;
    tq_uniform_3b_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_uniform_3b_dequantize_ref(&block, output.data(), TQ_BK);

    // Check per-sub-block MSE: each should be reasonable for its own range
    for (int sb = 0; sb < 4; sb++) {
        double mse = 0;
        for (int i = sb * 32; i < (sb + 1) * 32; i++) {
            double d = input[i] - output[i];
            mse += d * d;
        }
        mse /= 32;
        // Each sub-block's MSE should be proportional to its own range, not the global range
        float range = 0;
        for (int i = sb * 32; i < (sb + 1) * 32; i++) {
            float a = fabsf(input[i]);
            if (a > range) range = a;
        }
        // MSE should be small relative to the sub-block's range^2
        // 3-bit (8 levels): step = range/3.5, MSE ~ step^2/12 ~ range^2/147
        float expected_max_mse = (range * range) / 10.0f; // generous bound
        if (expected_max_mse < 0.01f) expected_max_mse = 0.01f;
        EXPECT_LT(mse, expected_max_mse)
            << "Sub-block " << sb << " MSE too high relative to its range";
    }
}
