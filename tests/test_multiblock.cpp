/**
 * Tests for multi-block attention (head_dim > TQ_BK).
 *
 * Validates that attention functions correctly handle key vectors
 * spanning multiple quantization blocks.
 */

#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"

void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
void tq_uniform_4b_attention_ref(const float* query, const void* kv,
                                  float* scores, int seq_len, int head_dim);
void tq_uniform_4b_attention_int_ref(const float* query, const void* kv,
                                      float* scores, int seq_len, int head_dim);
void tq_uniform_2b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_2b_attention_ref(const float* query, const void* kv,
                                  float* scores, int seq_len, int head_dim);
}

#include <cmath>
#include <vector>

static float cosine_similarity(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na < 1e-20 || nb < 1e-20) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

/* Helper: quantize keys using per-block quantization (matching multi-block layout) */
static void quantize_multiblock_4b(const float* key, block_tq_uniform_4b* blocks,
                                    int head_dim) {
    int blocks_per_key = (head_dim + TQ_BK - 1) / TQ_BK;
    for (int b = 0; b < blocks_per_key; b++) {
        int offset = b * TQ_BK;
        int chunk = (head_dim - offset > TQ_BK) ? TQ_BK : (head_dim - offset);
        tq_uniform_4b_quantize_ref(key + offset, &blocks[b], chunk);
    }
}

static void quantize_multiblock_2b(const float* key, block_tq_uniform_2b* blocks,
                                    int head_dim) {
    int blocks_per_key = (head_dim + TQ_BK - 1) / TQ_BK;
    for (int b = 0; b < blocks_per_key; b++) {
        int offset = b * TQ_BK;
        int chunk = (head_dim - offset > TQ_BK) ? TQ_BK : (head_dim - offset);
        tq_uniform_2b_quantize_ref(key + offset, &blocks[b], chunk);
    }
}

/* ================================================================
 * Uniform 4-bit multi-block tests
 * ================================================================ */

TEST(MultiBlock, Attention256) {
    const int head_dim = 256;  /* 2 blocks */
    const int seq_len = 16;
    const int blocks_per_key = (head_dim + TQ_BK - 1) / TQ_BK;

    /* Create query */
    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = cosf(i * 0.05f);

    /* Create and quantize keys (multi-block layout) */
    std::vector<block_tq_uniform_4b> blocks(seq_len * blocks_per_key);
    std::vector<std::vector<float>> keys(seq_len);
    for (int s = 0; s < seq_len; s++) {
        keys[s].resize(head_dim);
        for (int d = 0; d < head_dim; d++)
            keys[s][d] = sinf(s * 1.0f + d * 0.1f);
        quantize_multiblock_4b(keys[s].data(), &blocks[s * blocks_per_key], head_dim);
    }

    /* Compute FP32 reference (dequantize per block, dot product) */
    std::vector<float> ref_scores(seq_len);
    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int b = 0; b < blocks_per_key; b++) {
            int offset = b * TQ_BK;
            int chunk = (head_dim - offset > TQ_BK) ? TQ_BK : (head_dim - offset);
            float deq[TQ_BK];
            tq_uniform_4b_dequantize_ref(&blocks[s * blocks_per_key + b], deq, chunk);
            for (int d = 0; d < chunk; d++)
                dot += query[offset + d] * deq[d];
        }
        ref_scores[s] = dot;
    }

    /* Compute via attention function */
    std::vector<float> attn_scores(seq_len);
    tq_uniform_4b_attention_ref(query.data(), blocks.data(),
                                 attn_scores.data(), seq_len, head_dim);

    /* Should match reference exactly (same code path) */
    for (int s = 0; s < seq_len; s++) {
        EXPECT_FLOAT_EQ(attn_scores[s], ref_scores[s])
            << "Score mismatch at position " << s;
    }

    /* Cosine similarity with original FP32 dot products should be high */
    std::vector<float> fp32_scores(seq_len);
    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int d = 0; d < head_dim; d++)
            dot += query[d] * keys[s][d];
        fp32_scores[s] = dot;
    }
    float cos_sim = cosine_similarity(fp32_scores.data(), attn_scores.data(), seq_len);
    EXPECT_GT(cos_sim, 0.95f) << "Cosine similarity too low: " << cos_sim;
}

TEST(MultiBlock, Attention384) {
    const int head_dim = 384;  /* 3 blocks */
    const int seq_len = 8;
    const int blocks_per_key = (head_dim + TQ_BK - 1) / TQ_BK;

    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = cosf(i * 0.03f);

    std::vector<block_tq_uniform_4b> blocks(seq_len * blocks_per_key);
    std::vector<std::vector<float>> keys(seq_len);
    for (int s = 0; s < seq_len; s++) {
        keys[s].resize(head_dim);
        for (int d = 0; d < head_dim; d++)
            keys[s][d] = sinf(s * 0.7f + d * 0.08f) * (1.0f + s * 0.01f);
        quantize_multiblock_4b(keys[s].data(), &blocks[s * blocks_per_key], head_dim);
    }

    /* FP32 reference */
    std::vector<float> fp32_scores(seq_len);
    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int d = 0; d < head_dim; d++)
            dot += query[d] * keys[s][d];
        fp32_scores[s] = dot;
    }

    std::vector<float> attn_scores(seq_len);
    tq_uniform_4b_attention_ref(query.data(), blocks.data(),
                                 attn_scores.data(), seq_len, head_dim);

    float cos_sim = cosine_similarity(fp32_scores.data(), attn_scores.data(), seq_len);
    EXPECT_GT(cos_sim, 0.95f) << "Cosine similarity too low: " << cos_sim;
}

TEST(MultiBlock, IntegerAttention256) {
    const int head_dim = 256;  /* 2 blocks */
    const int seq_len = 16;
    const int blocks_per_key = (head_dim + TQ_BK - 1) / TQ_BK;

    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = cosf(i * 0.05f);

    std::vector<block_tq_uniform_4b> blocks(seq_len * blocks_per_key);
    std::vector<std::vector<float>> keys(seq_len);
    for (int s = 0; s < seq_len; s++) {
        keys[s].resize(head_dim);
        for (int d = 0; d < head_dim; d++)
            keys[s][d] = sinf(s * 1.0f + d * 0.1f);
        quantize_multiblock_4b(keys[s].data(), &blocks[s * blocks_per_key], head_dim);
    }

    /* FP32 dequant+dot reference */
    std::vector<float> fp32_scores(seq_len);
    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int b = 0; b < blocks_per_key; b++) {
            int offset = b * TQ_BK;
            int chunk = (head_dim - offset > TQ_BK) ? TQ_BK : (head_dim - offset);
            float deq[TQ_BK];
            tq_uniform_4b_dequantize_ref(&blocks[s * blocks_per_key + b], deq, chunk);
            for (int d = 0; d < chunk; d++)
                dot += query[offset + d] * deq[d];
        }
        fp32_scores[s] = dot;
    }

    /* Integer-domain attention */
    std::vector<float> int_scores(seq_len);
    tq_uniform_4b_attention_int_ref(query.data(), blocks.data(),
                                     int_scores.data(), seq_len, head_dim);

    float cos_sim = cosine_similarity(fp32_scores.data(), int_scores.data(), seq_len);
    EXPECT_GT(cos_sim, 0.99f) << "Integer attention cosine similarity too low: " << cos_sim;

    /* Absolute tolerance check */
    float max_mag = 0;
    for (int s = 0; s < seq_len; s++) {
        float m = fabsf(fp32_scores[s]);
        if (m > max_mag) max_mag = m;
    }
    float tol = max_mag * 0.05f;
    for (int s = 0; s < seq_len; s++) {
        EXPECT_LT(fabsf(int_scores[s] - fp32_scores[s]), tol)
            << "Score " << s << ": fp32=" << fp32_scores[s]
            << " int=" << int_scores[s];
    }
}

TEST(MultiBlock, Uniform2b256) {
    const int head_dim = 256;
    const int seq_len = 8;
    const int blocks_per_key = (head_dim + TQ_BK - 1) / TQ_BK;

    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = cosf(i * 0.04f);

    std::vector<block_tq_uniform_2b> blocks(seq_len * blocks_per_key);
    std::vector<std::vector<float>> keys(seq_len);
    for (int s = 0; s < seq_len; s++) {
        keys[s].resize(head_dim);
        for (int d = 0; d < head_dim; d++)
            keys[s][d] = sinf(s * 0.5f + d * 0.07f);
        quantize_multiblock_2b(keys[s].data(), &blocks[s * blocks_per_key], head_dim);
    }

    /* FP32 reference */
    std::vector<float> fp32_scores(seq_len);
    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int d = 0; d < head_dim; d++)
            dot += query[d] * keys[s][d];
        fp32_scores[s] = dot;
    }

    std::vector<float> attn_scores(seq_len);
    tq_uniform_2b_attention_ref(query.data(), blocks.data(),
                                 attn_scores.data(), seq_len, head_dim);

    float cos_sim = cosine_similarity(fp32_scores.data(), attn_scores.data(), seq_len);
    EXPECT_GT(cos_sim, 0.90f) << "2-bit attention cosine similarity too low: " << cos_sim;
}

/* ================================================================
 * Context API multi-block test (end-to-end)
 * ================================================================ */

TEST(MultiBlock, ContextAPI256) {
    const int head_dim = 256;
    const int seq_len = 4;

    tq_context_t* ctx = nullptr;
    ASSERT_EQ(tq_init(&ctx, TQ_BACKEND_CPU), TQ_OK);

    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = cosf(i * 0.05f);

    /* Create keys */
    std::vector<float> keys(seq_len * head_dim);
    for (int s = 0; s < seq_len; s++)
        for (int d = 0; d < head_dim; d++)
            keys[s * head_dim + d] = sinf(s * 1.0f + d * 0.1f);

    /* Quantize via context API */
    size_t kv_size = tq_quantize_keys_size(seq_len, head_dim, TQ_TYPE_UNIFORM_4B);
    ASSERT_GT(kv_size, 0u);
    std::vector<uint8_t> kv_buf(kv_size);
    ASSERT_EQ(tq_quantize_keys(ctx, keys.data(), seq_len, head_dim,
                                TQ_TYPE_UNIFORM_4B, kv_buf.data(), kv_size), TQ_OK);

    /* Attention via context API */
    std::vector<float> scores(seq_len);
    ASSERT_EQ(tq_attention(ctx, query.data(), kv_buf.data(), seq_len, head_dim,
                            TQ_TYPE_UNIFORM_4B, scores.data()), TQ_OK);

    /* FP32 reference */
    std::vector<float> fp32_scores(seq_len);
    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int d = 0; d < head_dim; d++)
            dot += query[d] * keys[s * head_dim + d];
        fp32_scores[s] = dot;
    }

    float cos_sim = cosine_similarity(fp32_scores.data(), scores.data(), seq_len);
    EXPECT_GT(cos_sim, 0.95f) << "Context API multi-block cosine similarity: " << cos_sim;

    tq_free(ctx);
}
