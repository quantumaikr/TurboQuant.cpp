/**
 * test_turbo_kv_1b.cpp -- Tests for TurboQuant 1-bit KV cache (Hamming attention)
 *
 * Tests the RHT + sign extraction pipeline for extreme 1-bit KV compression.
 * Validates block structure, attention accuracy via XOR+popcount, and
 * integration with the type system.
 */

#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"

void tq_turbo_kv_1b_quantize_ref(const float* src, void* dst, int n);
void tq_turbo_kv_1b_dequantize_ref(const void* src, float* dst, int n);
void tq_turbo_kv_1b_attention_ref(const float* query, const void* kv,
                                    float* scores, int seq_len, int head_dim);
}

#include <cmath>
#include <vector>
#include <random>
#include <cstring>

static double compute_cosine(const float* a, const float* b, int n) {
    double dot = 0.0, sq_a = 0.0, sq_b = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        sq_a += (double)a[i] * (double)a[i];
        sq_b += (double)b[i] * (double)b[i];
    }
    double denom = std::sqrt(sq_a) * std::sqrt(sq_b);
    return (denom > 1e-15) ? dot / denom : 0.0;
}

/* ============================================================
 * Block structure tests
 * ============================================================ */

TEST(TurboKV1B, BlockStructureSize) {
    /* 8 bytes header (norm + pad + seed) + 16 bytes signs (128/8) = 24 bytes */
    EXPECT_EQ(sizeof(block_tq_turbo_kv_1b), 24u);
    /* BPE = 24 * 8 / 128 = 1.5 */
    float bpe = (float)sizeof(block_tq_turbo_kv_1b) * 8.0f / TQ_BK;
    EXPECT_NEAR(bpe, 1.5f, 0.01f);
}

TEST(TurboKV1B, CompressionRatio) {
    /* 128 dims * 4 bytes FP32 = 512 bytes original
     * 24 bytes compressed = 21.3x compression */
    float ratio = (128.0f * 4.0f) / sizeof(block_tq_turbo_kv_1b);
    EXPECT_GT(ratio, 20.0f);
}

/* ============================================================
 * Quantize/dequantize roundtrip
 * ============================================================ */

TEST(TurboKV1B, RoundtripBasic) {
    const int dim = TQ_BK;
    std::vector<float> input(dim);
    for (int i = 0; i < dim; i++) input[i] = sinf(i * 0.1f);

    block_tq_turbo_kv_1b block;
    memset(&block, 0, sizeof(block));
    tq_turbo_kv_1b_quantize_ref(input.data(), &block, dim);

    std::vector<float> output(dim);
    tq_turbo_kv_1b_dequantize_ref(&block, output.data(), dim);

    /* 1-bit is very rough for point-wise reconstruction.
     * Cosine should still be positive (better than random). */
    double cosine = compute_cosine(input.data(), output.data(), dim);
    EXPECT_GT(cosine, 0.1) << "TurboKV 1B cosine too low: " << cosine;
}

TEST(TurboKV1B, NormPreserved) {
    const int dim = TQ_BK;
    std::vector<float> input(dim);
    for (int i = 0; i < dim; i++) input[i] = 2.0f * sinf(i * 0.2f);

    block_tq_turbo_kv_1b block;
    tq_turbo_kv_1b_quantize_ref(input.data(), &block, dim);

    /* Decode stored FP16 norm */
    uint16_t stored = block.norm;
    uint32_t sign = (stored & 0x8000) << 16;
    uint32_t exp = (stored >> 10) & 0x1F;
    uint32_t mant = stored & 0x03FF;
    union { float f; uint32_t u; } fp16;
    if (exp == 0) fp16.u = sign;
    else { exp = exp - 15 + 127; fp16.u = sign | (exp << 23) | (mant << 13); }

    float expected_norm = 0.0f;
    for (int i = 0; i < dim; i++) expected_norm += input[i] * input[i];
    expected_norm = sqrtf(expected_norm);

    EXPECT_NEAR(fp16.f, expected_norm, expected_norm * 0.01f);
}

TEST(TurboKV1B, SignsNonZero) {
    const int dim = TQ_BK;
    std::vector<float> input(dim);
    for (int i = 0; i < dim; i++) input[i] = sinf(i * 0.3f);

    block_tq_turbo_kv_1b block;
    memset(&block, 0, sizeof(block));
    tq_turbo_kv_1b_quantize_ref(input.data(), &block, dim);

    bool has_nonzero = false;
    for (int i = 0; i < TQ_BK / 8; i++) {
        if (block.signs[i] != 0) has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero);
}

/* ============================================================
 * Attention accuracy tests (the real value of 1-bit)
 * ============================================================ */

TEST(TurboKV1B, AttentionAccuracy) {
    const int dim = 128;
    const int seq_len = 16;

    std::vector<float> keys(seq_len * dim);
    std::vector<float> query(dim);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < seq_len * dim; i++) keys[i] = dist(rng);
    for (int i = 0; i < dim; i++) query[i] = dist(rng);

    /* FP32 reference scores */
    std::vector<float> fp32_scores(seq_len);
    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int d = 0; d < dim; d++)
            dot += query[d] * keys[s * dim + d];
        fp32_scores[s] = dot;
    }

    /* Quantize keys to 1-bit */
    std::vector<block_tq_turbo_kv_1b> blocks(seq_len);
    for (int s = 0; s < seq_len; s++) {
        tq_turbo_kv_1b_quantize_ref(&keys[s * dim], &blocks[s], dim);
    }

    /* Compute quantized attention */
    std::vector<float> quant_scores(seq_len);
    tq_turbo_kv_1b_attention_ref(query.data(), blocks.data(),
                                   quant_scores.data(), seq_len, dim);

    /* Cosine similarity between score vectors.
     * 1-bit Hamming attention preserves ranking reasonably well.
     * With only 128 bits per key, cosine ~0.4-0.6 is typical for random data. */
    double cosine = compute_cosine(fp32_scores.data(), quant_scores.data(), seq_len);
    EXPECT_GT(std::fabs(cosine), 0.4)
        << "TurboKV 1B attention cosine too low: " << cosine;
}

TEST(TurboKV1B, AttentionOrdering) {
    /* Test that the highest-scoring key remains highest after 1-bit quantization */
    const int dim = 128;
    const int seq_len = 8;

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> query(dim);
    for (int i = 0; i < dim; i++) query[i] = dist(rng);

    /* Create keys where key[0] is aligned with query (high score) */
    std::vector<float> keys(seq_len * dim);
    for (int i = 0; i < seq_len * dim; i++) keys[i] = dist(rng);
    /* Make key[0] = query + small noise (guaranteed highest score) */
    for (int i = 0; i < dim; i++) keys[i] = query[i] + 0.1f * dist(rng);

    /* FP32 reference: find argmax */
    float best_fp32 = -1e30f;
    int best_idx_fp32 = -1;
    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int d = 0; d < dim; d++)
            dot += query[d] * keys[s * dim + d];
        if (dot > best_fp32) { best_fp32 = dot; best_idx_fp32 = s; }
    }
    EXPECT_EQ(best_idx_fp32, 0); /* Sanity check: key[0] should have highest score */

    /* 1-bit quantized: find argmax */
    std::vector<block_tq_turbo_kv_1b> blocks(seq_len);
    for (int s = 0; s < seq_len; s++) {
        tq_turbo_kv_1b_quantize_ref(&keys[s * dim], &blocks[s], dim);
    }

    std::vector<float> quant_scores(seq_len);
    tq_turbo_kv_1b_attention_ref(query.data(), blocks.data(),
                                   quant_scores.data(), seq_len, dim);

    int best_idx_quant = 0;
    for (int s = 1; s < seq_len; s++) {
        if (quant_scores[s] > quant_scores[best_idx_quant]) best_idx_quant = s;
    }

    /* The aligned key should remain at position 0 (or at least top-2) */
    EXPECT_LE(best_idx_quant, 1)
        << "1-bit attention lost the highest-scoring key: best=" << best_idx_quant;
}

/* ============================================================
 * Zero input edge case
 * ============================================================ */

TEST(TurboKV1B, ZeroInput) {
    const int dim = TQ_BK;
    std::vector<float> input(dim, 0.0f);

    block_tq_turbo_kv_1b block;
    memset(&block, 0, sizeof(block));
    tq_turbo_kv_1b_quantize_ref(input.data(), &block, dim);
    std::vector<float> output(dim);
    tq_turbo_kv_1b_dequantize_ref(&block, output.data(), dim);

    for (int i = 0; i < dim; i++) {
        EXPECT_NEAR(output[i], 0.0f, 1e-4f);
    }
}

/* ============================================================
 * Traits table integration
 * ============================================================ */

TEST(TurboKV1B, TraitsTable) {
    EXPECT_STREQ(tq_type_name(TQ_TYPE_TURBO_KV_1B), "turbo_kv_1b");
    EXPECT_EQ(tq_type_block_size(TQ_TYPE_TURBO_KV_1B), (size_t)TQ_BK);
    EXPECT_EQ(tq_type_type_size(TQ_TYPE_TURBO_KV_1B), sizeof(block_tq_turbo_kv_1b));
    EXPECT_GT(tq_type_bpe(TQ_TYPE_TURBO_KV_1B), 0.0f);
    EXPECT_LT(tq_type_bpe(TQ_TYPE_TURBO_KV_1B), 2.0f); /* Should be 1.5 bpe */
    EXPECT_EQ(tq_type_from_name("turbo_kv_1b"), TQ_TYPE_TURBO_KV_1B);
}

TEST(TurboKV1B, FormatSpec) {
    tq_format_spec_t spec = tq_get_format_spec(TQ_TYPE_TURBO_KV_1B);
    EXPECT_EQ(spec.algorithm, TQ_ALG_TURBO);
    EXPECT_EQ(spec.key_bits, 1);
}

/* ============================================================
 * Context API integration
 * ============================================================ */

TEST(TurboKV1B, ContextAPIRoundtrip) {
    tq_context_t* ctx;
    ASSERT_EQ(tq_init(&ctx, TQ_BACKEND_CPU), TQ_OK);

    const int dim = 128;
    std::vector<float> key(dim);
    for (int i = 0; i < dim; i++) key[i] = sinf(i * 0.1f);

    size_t buf_size = tq_quantize_keys_size(1, dim, TQ_TYPE_TURBO_KV_1B);
    ASSERT_GT(buf_size, 0u);
    std::vector<uint8_t> buf(buf_size);

    tq_status st = tq_quantize_keys(ctx, key.data(), 1, dim,
                                     TQ_TYPE_TURBO_KV_1B,
                                     buf.data(), buf_size);
    ASSERT_EQ(st, TQ_OK);

    std::vector<float> output(dim);
    st = tq_dequantize_keys(ctx, buf.data(), 1, dim,
                             TQ_TYPE_TURBO_KV_1B, output.data());
    ASSERT_EQ(st, TQ_OK);

    /* 1-bit: cosine just needs to be positive (direction preserved) */
    double cosine = compute_cosine(key.data(), output.data(), dim);
    EXPECT_GT(cosine, 0.1);

    tq_free(ctx);
}
