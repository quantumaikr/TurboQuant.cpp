/**
 * Tests for integer-domain Q4xQ8 attention.
 *
 * Validates that the integer-domain attention path produces results
 * equivalent to the FP32 dequantize+dot path, within floating-point
 * tolerance introduced by Q8 query quantization.
 */

#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"

/* Reference functions */
void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);

/* Integer-domain functions */
void tq_quantize_query_q8(const float* query, int8_t* q8_out,
                           float* scale_out, float* sum_out, int n);
void tq_uniform_4b_attention_int_ref(const float* query, const void* kv,
                                      float* scores, int seq_len, int head_dim);
void tq_uniform_4b_attention_ref(const float* query, const void* kv,
                                  float* scores, int seq_len, int head_dim);

#ifdef __ARM_NEON
void tq_quantize_query_q8_neon(const float* query, int8_t* q8_out,
                                float* scale_out, float* sum_out, int n);
void tq_uniform_4b_attention_int_neon(const float* query, const void* kv,
                                       float* scores, int seq_len, int head_dim);
#endif
}

#include <cmath>
#include <vector>
#include <numeric>

/* ================================================================
 * Q8 quantization tests
 * ================================================================ */

TEST(Q8Quantize, ScaleAndClamp) {
    const int n = 128;
    std::vector<float> query(n);
    for (int i = 0; i < n; i++) query[i] = sinf(i * 0.1f) * 2.0f;

    int8_t q8[128];
    float scale, sum;
    tq_quantize_query_q8(query.data(), q8, &scale, &sum, n);

    /* Scale should be positive */
    EXPECT_GT(scale, 0.0f);

    /* All Q8 values should be in [-128, 127] */
    for (int i = 0; i < n; i++) {
        EXPECT_GE(q8[i], -128);
        EXPECT_LE(q8[i], 127);
    }

    /* Dequantized Q8 should approximate original */
    double mse = 0;
    for (int i = 0; i < n; i++) {
        double diff = query[i] - (double)q8[i] * scale;
        mse += diff * diff;
    }
    mse /= n;
    /* Q8 quantization of [-2,2] range: step ~ 4/255 ~ 0.016, MSE ~ step^2/12 ~ 2e-5 */
    EXPECT_LT(mse, 0.001);
}

TEST(Q8Quantize, SumAccuracy) {
    const int n = 128;
    std::vector<float> query(n);
    float expected_sum = 0;
    for (int i = 0; i < n; i++) {
        query[i] = cosf(i * 0.05f);
        expected_sum += query[i];
    }

    int8_t q8[128];
    float scale, sum;
    tq_quantize_query_q8(query.data(), q8, &scale, &sum, n);

    EXPECT_NEAR(sum, expected_sum, 1e-4f);
}

TEST(Q8Quantize, ZeroInput) {
    const int n = 64;
    std::vector<float> query(n, 0.0f);

    int8_t q8[64];
    float scale, sum;
    tq_quantize_query_q8(query.data(), q8, &scale, &sum, n);

    /* Scale should be clamped to minimum */
    EXPECT_GT(scale, 0.0f);
    EXPECT_NEAR(sum, 0.0f, 1e-10f);

    for (int i = 0; i < n; i++) {
        EXPECT_EQ(q8[i], 0);
    }
}

TEST(Q8Quantize, MaxSaturation) {
    const int n = 4;
    float query[4] = {1000.0f, -1000.0f, 500.0f, -500.0f};

    int8_t q8[4];
    float scale, sum;
    tq_quantize_query_q8(query, q8, &scale, &sum, n);

    /* Max values should saturate to +/-127 */
    EXPECT_EQ(q8[0], 127);
    EXPECT_EQ(q8[1], -127);
}

/* ================================================================
 * Integer-domain attention tests
 * ================================================================ */

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

TEST(IntAttention, VsFP32Dequant) {
    const int head_dim = TQ_BK;
    const int seq_len = 8;

    /* Create query */
    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = cosf(i * 0.05f);

    /* Create and quantize keys */
    std::vector<block_tq_uniform_4b> blocks(seq_len);
    for (int s = 0; s < seq_len; s++) {
        std::vector<float> key(head_dim);
        for (int d = 0; d < head_dim; d++)
            key[d] = sinf(s * 1.0f + d * 0.1f);
        tq_uniform_4b_quantize_ref(key.data(), &blocks[s], head_dim);
    }

    /* Compute FP32 dequant+dot reference */
    std::vector<float> fp32_scores(seq_len);
    for (int s = 0; s < seq_len; s++) {
        std::vector<float> deq(head_dim);
        tq_uniform_4b_dequantize_ref(&blocks[s], deq.data(), head_dim);
        float dot = 0;
        for (int d = 0; d < head_dim; d++) dot += query[d] * deq[d];
        fp32_scores[s] = dot;
    }

    /* Compute integer-domain attention */
    std::vector<float> int_scores(seq_len);
    tq_uniform_4b_attention_int_ref(query.data(), blocks.data(),
                                     int_scores.data(), seq_len, head_dim);

    /* Results should have high cosine similarity (> 0.99) */
    float cos_sim = cosine_similarity(fp32_scores.data(), int_scores.data(), seq_len);
    EXPECT_GT(cos_sim, 0.99f) << "Cosine similarity too low: " << cos_sim;

    /* Individual scores: use absolute tolerance for near-zero values,
     * relative tolerance otherwise. Q8 quantization introduces ~1/127
     * relative error compounded over head_dim elements. */
    float max_magnitude = 0;
    for (int s = 0; s < seq_len; s++) {
        float m = fabsf(fp32_scores[s]);
        if (m > max_magnitude) max_magnitude = m;
    }
    float abs_tol = max_magnitude * 0.05f; /* 5% of max score as absolute tolerance */

    for (int s = 0; s < seq_len; s++) {
        float abs_error = fabsf(int_scores[s] - fp32_scores[s]);
        EXPECT_LT(abs_error, abs_tol)
            << "Score " << s << ": fp32=" << fp32_scores[s]
            << " int=" << int_scores[s] << " abs_err=" << abs_error
            << " tol=" << abs_tol;
    }
}

TEST(IntAttention, IntRefDeterministic) {
    /* Verify the integer path produces deterministic results across calls. */
    const int head_dim = TQ_BK;
    const int seq_len = 4;

    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = cosf(i * 0.03f);

    std::vector<block_tq_uniform_4b> blocks(seq_len);
    for (int s = 0; s < seq_len; s++) {
        std::vector<float> key(head_dim);
        for (int d = 0; d < head_dim; d++)
            key[d] = sinf(s * 0.7f + d * 0.05f);
        tq_uniform_4b_quantize_ref(key.data(), &blocks[s], head_dim);
    }

    std::vector<float> scores_a(seq_len);
    std::vector<float> scores_b(seq_len);

    tq_uniform_4b_attention_int_ref(query.data(), blocks.data(),
                                     scores_a.data(), seq_len, head_dim);
    tq_uniform_4b_attention_int_ref(query.data(), blocks.data(),
                                     scores_b.data(), seq_len, head_dim);

    for (int s = 0; s < seq_len; s++) {
        EXPECT_FLOAT_EQ(scores_a[s], scores_b[s])
            << "Integer attention should be deterministic";
    }
}

TEST(IntAttention, LargeSeqLen) {
    const int head_dim = TQ_BK;
    const int seq_len = 256;

    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = cosf(i * 0.02f);

    std::vector<block_tq_uniform_4b> blocks(seq_len);
    for (int s = 0; s < seq_len; s++) {
        std::vector<float> key(head_dim);
        for (int d = 0; d < head_dim; d++)
            key[d] = sinf(s * 0.1f + d * 0.05f) * (1.0f + s * 0.01f);
        tq_uniform_4b_quantize_ref(key.data(), &blocks[s], head_dim);
    }

    /* Compute FP32 reference */
    std::vector<float> fp32_scores(seq_len);
    for (int s = 0; s < seq_len; s++) {
        std::vector<float> deq(head_dim);
        tq_uniform_4b_dequantize_ref(&blocks[s], deq.data(), head_dim);
        float dot = 0;
        for (int d = 0; d < head_dim; d++) dot += query[d] * deq[d];
        fp32_scores[s] = dot;
    }

    /* Integer attention */
    std::vector<float> int_scores(seq_len);
    tq_uniform_4b_attention_int_ref(query.data(), blocks.data(),
                                     int_scores.data(), seq_len, head_dim);

    float cos_sim = cosine_similarity(fp32_scores.data(), int_scores.data(), seq_len);
    EXPECT_GT(cos_sim, 0.99f);
}

TEST(IntAttention, ConstantKeys) {
    /* Edge case: all keys are constant vectors */
    const int head_dim = TQ_BK;
    const int seq_len = 4;

    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = 1.0f;

    std::vector<block_tq_uniform_4b> blocks(seq_len);
    for (int s = 0; s < seq_len; s++) {
        std::vector<float> key(head_dim, 1.0f);
        tq_uniform_4b_quantize_ref(key.data(), &blocks[s], head_dim);
    }

    std::vector<float> scores(seq_len);
    tq_uniform_4b_attention_int_ref(query.data(), blocks.data(),
                                     scores.data(), seq_len, head_dim);

    /* All scores should be approximately the same and close to head_dim */
    for (int s = 1; s < seq_len; s++) {
        EXPECT_NEAR(scores[s], scores[0], 1e-4f);
    }
    EXPECT_NEAR(scores[0], (float)head_dim, 2.0f);
}

#ifdef __ARM_NEON
/* ================================================================
 * NEON-specific tests (only compiled on ARM)
 * ================================================================ */

TEST(Q8QuantizeNEON, MatchesRef) {
    const int n = 128;
    std::vector<float> query(n);
    for (int i = 0; i < n; i++) query[i] = sinf(i * 0.1f) * 3.0f;

    int8_t q8_ref[128], q8_neon[128];
    float scale_ref, sum_ref, scale_neon, sum_neon;

    tq_quantize_query_q8(query.data(), q8_ref, &scale_ref, &sum_ref, n);
    tq_quantize_query_q8_neon(query.data(), q8_neon, &scale_neon, &sum_neon, n);

    EXPECT_NEAR(scale_ref, scale_neon, 1e-6f);
    EXPECT_NEAR(sum_ref, sum_neon, 1e-3f);

    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        if (abs(q8_ref[i] - q8_neon[i]) > 1) mismatches++;
    }
    /* Allow up to 5% mismatches due to rounding differences */
    EXPECT_LT(mismatches, n / 20);
}

TEST(IntAttentionNEON, MatchesRef) {
    const int head_dim = TQ_BK;
    const int seq_len = 8;

    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = cosf(i * 0.05f);

    std::vector<block_tq_uniform_4b> blocks(seq_len);
    for (int s = 0; s < seq_len; s++) {
        std::vector<float> key(head_dim);
        for (int d = 0; d < head_dim; d++)
            key[d] = sinf(s * 1.0f + d * 0.1f);
        tq_uniform_4b_quantize_ref(key.data(), &blocks[s], head_dim);
    }

    std::vector<float> ref_scores(seq_len);
    std::vector<float> neon_scores(seq_len);

    tq_uniform_4b_attention_int_ref(query.data(), blocks.data(),
                                     ref_scores.data(), seq_len, head_dim);
    tq_uniform_4b_attention_int_neon(query.data(), blocks.data(),
                                      neon_scores.data(), seq_len, head_dim);

    float cos_sim = cosine_similarity(ref_scores.data(), neon_scores.data(), seq_len);
    EXPECT_GT(cos_sim, 0.999f);

    for (int s = 0; s < seq_len; s++) {
        float magnitude = std::max(fabsf(ref_scores[s]), 1e-6f);
        float rel_error = fabsf(neon_scores[s] - ref_scores[s]) / magnitude;
        EXPECT_LT(rel_error, 0.02f)
            << "Score " << s << ": ref=" << ref_scores[s]
            << " neon=" << neon_scores[s];
    }
}
#endif /* __ARM_NEON */
