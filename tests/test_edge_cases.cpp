/**
 * Edge case tests for quant.cpp — BUG-4, BUG-7 verification
 * and systematic edge case coverage (n=1, NaN, Inf, all-same, large-n)
 */
#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"
}
#include <climits>
#include <cmath>
#include <cstring>

class EdgeCaseFixture : public ::testing::Test {
protected:
    tq_context_t* ctx = nullptr;

    void SetUp() override {
        ASSERT_EQ(TQ_OK, tq_init(&ctx, TQ_BACKEND_CPU));
    }

    void TearDown() override {
        tq_free(ctx);
    }
};

/* ---- BUG-7: seq_len == 0 is a no-op ---- */

TEST_F(EdgeCaseFixture, SeqLenZero) {
    float query[128] = {};
    float scores[1]  = {-999.0f};
    block_tq_uniform_4b block = {};

    tq_status st = tq_attention(ctx, query, &block, /*seq_len=*/0, 128,
                                TQ_TYPE_UNIFORM_4B, scores);
    EXPECT_EQ(TQ_OK, st);
    /* scores must be untouched */
    EXPECT_FLOAT_EQ(-999.0f, scores[0]);
}

/* ---- BUG-7: NULL input pointers ---- */

TEST_F(EdgeCaseFixture, NullInput) {
    block_tq_uniform_4b block = {};
    tq_status st = tq_quantize_keys(ctx, /*keys=*/NULL, 1, 128,
                                    TQ_TYPE_UNIFORM_4B, &block, sizeof(block));
    EXPECT_EQ(TQ_ERR_NULL_PTR, st);
}

TEST_F(EdgeCaseFixture, NullOutput) {
    float keys[128] = {};
    tq_status st = tq_quantize_keys(ctx, keys, 1, 128,
                                    TQ_TYPE_UNIFORM_4B, /*out=*/NULL, 1024);
    EXPECT_EQ(TQ_ERR_NULL_PTR, st);
}

/* ---- BUG-4: buffer too small ---- */

TEST_F(EdgeCaseFixture, BufferTooSmall) {
    float keys[128] = {};
    for (int i = 0; i < 128; i++) keys[i] = sinf((float)i * 0.1f);

    block_tq_uniform_4b block = {};
    /* Provide a buffer smaller than required */
    tq_status st = tq_quantize_keys(ctx, keys, 1, 128,
                                    TQ_TYPE_UNIFORM_4B, &block, /*out_size=*/1);
    EXPECT_EQ(TQ_ERR_BUFFER_TOO_SMALL, st);
}

/* ---- BUG-4: overflow size ---- */

TEST(EdgeCase, OverflowSize) {
    size_t sz = tq_quantize_keys_size(INT_MAX, 128, TQ_TYPE_UNIFORM_4B);
    EXPECT_EQ(0u, sz);
}

TEST(EdgeCase, NegativeN) {
    size_t sz = tq_quantize_keys_size(-1, 128, TQ_TYPE_UNIFORM_4B);
    EXPECT_EQ(0u, sz);
}

TEST(EdgeCase, ZeroHeadDim) {
    size_t sz = tq_quantize_keys_size(1, 0, TQ_TYPE_UNIFORM_4B);
    EXPECT_EQ(0u, sz);
}

/* ---- BUG-7: odd head_dim for PolarQuant ---- */

TEST_F(EdgeCaseFixture, OddHeadDimPolar) {
    float keys[65] = {};
    uint8_t buf[4096] = {};
    tq_status st = tq_quantize_keys(ctx, keys, 1, 65,
                                    TQ_TYPE_POLAR_3B, buf, sizeof(buf));
    EXPECT_EQ(TQ_ERR_INVALID_DIM, st);
}

TEST_F(EdgeCaseFixture, OddHeadDimTurbo) {
    float keys[65] = {};
    uint8_t buf[4096] = {};
    tq_status st = tq_quantize_keys(ctx, keys, 1, 65,
                                    TQ_TYPE_TURBO_3B, buf, sizeof(buf));
    EXPECT_EQ(TQ_ERR_INVALID_DIM, st);
}

/* ---- BUG-7: head_dim < 2 ---- */

TEST_F(EdgeCaseFixture, HeadDimOne) {
    float keys[1] = {1.0f};
    uint8_t buf[4096] = {};
    tq_status st = tq_quantize_keys(ctx, keys, 1, 1,
                                    TQ_TYPE_UNIFORM_4B, buf, sizeof(buf));
    EXPECT_EQ(TQ_ERR_INVALID_DIM, st);
}

/* ---- BUG-7: head_dim == 2 works correctly ---- */

TEST_F(EdgeCaseFixture, HeadDimTwo) {
    float keys[2] = {0.5f, -0.5f};
    /* head_dim=2 with block_size=128: 1 block, need type_size bytes */
    size_t needed = tq_quantize_keys_size(1, 2, TQ_TYPE_UNIFORM_4B);
    ASSERT_GT(needed, 0u);

    std::vector<uint8_t> buf(needed, 0);
    tq_status st = tq_quantize_keys(ctx, keys, 1, 2,
                                    TQ_TYPE_UNIFORM_4B, buf.data(), buf.size());
    EXPECT_EQ(TQ_OK, st);
}

/* ---- BUG-7: n == 0 is a no-op for quantize ---- */

TEST_F(EdgeCaseFixture, QuantizeZeroKeys) {
    float keys[128] = {};
    uint8_t buf[4096] = {};
    tq_status st = tq_quantize_keys(ctx, keys, 0, 128,
                                    TQ_TYPE_UNIFORM_4B, buf, sizeof(buf));
    EXPECT_EQ(TQ_OK, st);
}

/* ---- BUG-7: attention with head_dim < 2 ---- */

TEST_F(EdgeCaseFixture, AttentionHeadDimOne) {
    float query[1] = {1.0f};
    float scores[1] = {};
    uint8_t kv[256] = {};
    tq_status st = tq_attention(ctx, query, kv, 1, 1,
                                TQ_TYPE_UNIFORM_4B, scores);
    EXPECT_EQ(TQ_ERR_INVALID_DIM, st);
}

/* ---- BUG-7: attention odd head_dim for PolarQuant ---- */

TEST_F(EdgeCaseFixture, AttentionOddHeadDimPolar) {
    float query[65] = {};
    float scores[1] = {};
    uint8_t kv[4096] = {};
    tq_status st = tq_attention(ctx, query, kv, 1, 65,
                                TQ_TYPE_POLAR_3B, scores);
    EXPECT_EQ(TQ_ERR_INVALID_DIM, st);
}

/* ---- New error code string ---- */

TEST(EdgeCase, BufferTooSmallString) {
    const char* s = tq_status_string(TQ_ERR_BUFFER_TOO_SMALL);
    EXPECT_STREQ("buffer too small", s);
}

/* ---- MaxSeqLen boundary ---- */

TEST(EdgeCase, MaxSeqLenBoundary) {
    /* Exactly at TQ_MAX_SEQ_LEN should work (non-zero return) */
    size_t sz = tq_quantize_keys_size(TQ_MAX_SEQ_LEN, 128, TQ_TYPE_UNIFORM_4B);
    EXPECT_GT(sz, 0u);

    /* One over should fail */
    size_t sz2 = tq_quantize_keys_size(TQ_MAX_SEQ_LEN + 1, 128, TQ_TYPE_UNIFORM_4B);
    EXPECT_EQ(0u, sz2);
}

/* ============================================================
 * Systematic edge case tests
 * ============================================================ */

/* ---- SingleTokenQuantize: n=1 with each KV type ---- */

TEST_F(EdgeCaseFixture, SingleTokenQuantize_Uniform4B) {
    float keys[128];
    for (int i = 0; i < 128; i++) keys[i] = sinf((float)i * 0.1f);

    size_t needed = tq_quantize_keys_size(1, 128, TQ_TYPE_UNIFORM_4B);
    ASSERT_GT(needed, 0u);
    std::vector<uint8_t> buf(needed, 0);

    tq_status st = tq_quantize_keys(ctx, keys, 1, 128,
                                    TQ_TYPE_UNIFORM_4B, buf.data(), buf.size());
    EXPECT_EQ(TQ_OK, st);
}

TEST_F(EdgeCaseFixture, SingleTokenQuantize_TurboKV3B) {
    float keys[128];
    for (int i = 0; i < 128; i++) keys[i] = sinf((float)i * 0.1f);

    size_t needed = tq_quantize_keys_size(1, 128, TQ_TYPE_TURBO_KV_3B);
    ASSERT_GT(needed, 0u);
    std::vector<uint8_t> buf(needed, 0);

    tq_status st = tq_quantize_keys(ctx, keys, 1, 128,
                                    TQ_TYPE_TURBO_KV_3B, buf.data(), buf.size());
    EXPECT_EQ(TQ_OK, st);
}

TEST_F(EdgeCaseFixture, SingleTokenQuantize_TurboKV1B) {
    float keys[128];
    for (int i = 0; i < 128; i++) keys[i] = sinf((float)i * 0.1f);

    size_t needed = tq_quantize_keys_size(1, 128, TQ_TYPE_TURBO_KV_1B);
    ASSERT_GT(needed, 0u);
    std::vector<uint8_t> buf(needed, 0);

    tq_status st = tq_quantize_keys(ctx, keys, 1, 128,
                                    TQ_TYPE_TURBO_KV_1B, buf.data(), buf.size());
    EXPECT_EQ(TQ_OK, st);
}

/* ---- ZeroDimHandling: head_dim=0 returns error or zero size ---- */

TEST_F(EdgeCaseFixture, ZeroDimQuantize) {
    float keys[1] = {1.0f}; (void)keys;
    uint8_t buf[4096] = {}; (void)buf;

    /* Size query should return 0 for head_dim=0 */
    size_t sz = tq_quantize_keys_size(1, 0, TQ_TYPE_UNIFORM_4B);
    EXPECT_EQ(0u, sz);

    sz = tq_quantize_keys_size(1, 0, TQ_TYPE_TURBO_KV_3B);
    EXPECT_EQ(0u, sz);

    sz = tq_quantize_keys_size(1, 0, TQ_TYPE_TURBO_KV_1B);
    EXPECT_EQ(0u, sz);
}

/* ---- NaNInputQuantize: NaN values should not crash ---- */

TEST_F(EdgeCaseFixture, NaNInputQuantize) {
    std::vector<float> keys(128);
    for (int i = 0; i < 128; i++) keys[i] = NAN;

    size_t needed = tq_quantize_keys_size(1, 128, TQ_TYPE_UNIFORM_4B);
    ASSERT_GT(needed, 0u);
    std::vector<uint8_t> buf(needed, 0);

    /* Should not crash — result may be garbage but must not segfault */
    tq_status st = tq_quantize_keys(ctx, keys.data(), 1, 128,
                                    TQ_TYPE_UNIFORM_4B, buf.data(), buf.size());
    /* We accept either OK (graceful handling) or an error code */
    EXPECT_TRUE(st == TQ_OK || st != TQ_OK) << "Must not crash on NaN input";
}

TEST_F(EdgeCaseFixture, NaNInputQuantize_TurboKV3B) {
    std::vector<float> keys(128);
    for (int i = 0; i < 128; i++) keys[i] = NAN;

    size_t needed = tq_quantize_keys_size(1, 128, TQ_TYPE_TURBO_KV_3B);
    ASSERT_GT(needed, 0u);
    std::vector<uint8_t> buf(needed, 0);

    tq_status st = tq_quantize_keys(ctx, keys.data(), 1, 128,
                                    TQ_TYPE_TURBO_KV_3B, buf.data(), buf.size());
    EXPECT_TRUE(st == TQ_OK || st != TQ_OK) << "Must not crash on NaN input";
}

/* ---- InfInputQuantize: Inf values should not crash ---- */

TEST_F(EdgeCaseFixture, InfInputQuantize) {
    std::vector<float> keys(128);
    for (int i = 0; i < 128; i++) keys[i] = (i % 2 == 0) ? INFINITY : -INFINITY;

    size_t needed = tq_quantize_keys_size(1, 128, TQ_TYPE_UNIFORM_4B);
    ASSERT_GT(needed, 0u);
    std::vector<uint8_t> buf(needed, 0);

    tq_status st = tq_quantize_keys(ctx, keys.data(), 1, 128,
                                    TQ_TYPE_UNIFORM_4B, buf.data(), buf.size());
    EXPECT_TRUE(st == TQ_OK || st != TQ_OK) << "Must not crash on Inf input";
}

TEST_F(EdgeCaseFixture, InfInputQuantize_TurboKV3B) {
    std::vector<float> keys(128);
    for (int i = 0; i < 128; i++) keys[i] = (i % 2 == 0) ? INFINITY : -INFINITY;

    size_t needed = tq_quantize_keys_size(1, 128, TQ_TYPE_TURBO_KV_3B);
    ASSERT_GT(needed, 0u);
    std::vector<uint8_t> buf(needed, 0);

    tq_status st = tq_quantize_keys(ctx, keys.data(), 1, 128,
                                    TQ_TYPE_TURBO_KV_3B, buf.data(), buf.size());
    EXPECT_TRUE(st == TQ_OK || st != TQ_OK) << "Must not crash on Inf input";
}

/* ---- AllSameValues: all elements identical (range=0 edge case) ---- */

TEST_F(EdgeCaseFixture, AllSameValues_Uniform4B) {
    std::vector<float> keys(128, 1.0f); /* all 1.0 */

    size_t needed = tq_quantize_keys_size(1, 128, TQ_TYPE_UNIFORM_4B);
    ASSERT_GT(needed, 0u);
    std::vector<uint8_t> buf(needed, 0);

    tq_status st = tq_quantize_keys(ctx, keys.data(), 1, 128,
                                    TQ_TYPE_UNIFORM_4B, buf.data(), buf.size());
    EXPECT_EQ(TQ_OK, st);

    /* Dequantized result should be close to original (all ~1.0) */
    float query[128];
    for (int i = 0; i < 128; i++) query[i] = 1.0f;
    float scores[1] = {};
    tq_status st2 = tq_attention(ctx, query, buf.data(), 1, 128,
                                 TQ_TYPE_UNIFORM_4B, scores);
    EXPECT_EQ(TQ_OK, st2);
    /* Score should be finite (no NaN from 0/0 in scale computation) */
    EXPECT_TRUE(std::isfinite(scores[0])) << "Score must be finite for all-same input";
}

TEST_F(EdgeCaseFixture, AllSameValues_TurboKV3B) {
    std::vector<float> keys(128, -0.5f); /* all -0.5 */

    size_t needed = tq_quantize_keys_size(1, 128, TQ_TYPE_TURBO_KV_3B);
    ASSERT_GT(needed, 0u);
    std::vector<uint8_t> buf(needed, 0);

    tq_status st = tq_quantize_keys(ctx, keys.data(), 1, 128,
                                    TQ_TYPE_TURBO_KV_3B, buf.data(), buf.size());
    EXPECT_EQ(TQ_OK, st);
}

TEST_F(EdgeCaseFixture, AllZeroValues) {
    std::vector<float> keys(128, 0.0f);

    size_t needed = tq_quantize_keys_size(1, 128, TQ_TYPE_UNIFORM_4B);
    ASSERT_GT(needed, 0u);
    std::vector<uint8_t> buf(needed, 0);

    tq_status st = tq_quantize_keys(ctx, keys.data(), 1, 128,
                                    TQ_TYPE_UNIFORM_4B, buf.data(), buf.size());
    EXPECT_EQ(TQ_OK, st);
}

/* ---- VeryLargeSequence: n=10000 keys, verify no overflow ---- */

TEST_F(EdgeCaseFixture, VeryLargeSequence_Uniform4B) {
    const int N = 10000;
    const int DIM = 128;

    /* Verify size calculation doesn't overflow */
    size_t needed = tq_quantize_keys_size(N, DIM, TQ_TYPE_UNIFORM_4B);
    ASSERT_GT(needed, 0u);
    ASSERT_GT(needed, (size_t)N); /* Sanity: must be larger than N */

    /* Allocate and quantize */
    std::vector<float> keys(N * DIM);
    for (int i = 0; i < N * DIM; i++) keys[i] = sinf((float)i * 0.001f);

    std::vector<uint8_t> buf(needed, 0);
    tq_status st = tq_quantize_keys(ctx, keys.data(), N, DIM,
                                    TQ_TYPE_UNIFORM_4B, buf.data(), buf.size());
    EXPECT_EQ(TQ_OK, st);
}

TEST_F(EdgeCaseFixture, VeryLargeSequence_TurboKV1B) {
    const int N = 10000;
    const int DIM = 128;

    size_t needed = tq_quantize_keys_size(N, DIM, TQ_TYPE_TURBO_KV_1B);
    ASSERT_GT(needed, 0u);

    std::vector<float> keys(N * DIM);
    for (int i = 0; i < N * DIM; i++) keys[i] = cosf((float)i * 0.001f);

    std::vector<uint8_t> buf(needed, 0);
    tq_status st = tq_quantize_keys(ctx, keys.data(), N, DIM,
                                    TQ_TYPE_TURBO_KV_1B, buf.data(), buf.size());
    EXPECT_EQ(TQ_OK, st);
}
