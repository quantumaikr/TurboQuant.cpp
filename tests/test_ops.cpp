/**
 * test_ops.cpp — Unit tests for tensor operations (tq_ops.c)
 *
 * Tests matmul, rmsnorm, rope, silu, softmax, add, mul.
 * Verifies correctness against reference scalar implementations.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>
#include <random>

extern "C" {
#include "turboquant/tq_engine.h"
}

/* ============================================================
 * Helper: reference implementations for comparison
 * ============================================================ */

static void ref_matmul(float* out, const float* x, const float* w, int n, int d) {
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += w[i * d + j] * x[j];
        }
        out[i] = sum;
    }
}

static void ref_rmsnorm(float* out, const float* x, const float* weight, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = ss / n + eps;
    float rsqrt = 1.0f / std::sqrt(ss);
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * rsqrt * weight[i];
    }
}

static void ref_silu(float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

static void ref_softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

/* Fill array with deterministic pseudo-random values */
static void fill_random(float* data, int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; i++) {
        data[i] = dist(rng);
    }
}

/* ============================================================
 * MatMul tests
 * ============================================================ */

TEST(TqOps, MatMulSmall) {
    /* 2x3 matrix times 3-vector */
    float w[] = {1, 2, 3,
                 4, 5, 6};
    float x[] = {1, 1, 1};
    float out[2], ref[2];

    tq_matmul(out, x, w, 2, 3);
    ref_matmul(ref, x, w, 2, 3);

    EXPECT_NEAR(out[0], ref[0], 1e-5f);
    EXPECT_NEAR(out[1], ref[1], 1e-5f);
    EXPECT_NEAR(out[0], 6.0f, 1e-5f);
    EXPECT_NEAR(out[1], 15.0f, 1e-5f);
}

TEST(TqOps, MatMulIdentity) {
    /* Identity-like: diagonal matrix */
    const int n = 64;
    std::vector<float> w(n * n, 0.0f);
    for (int i = 0; i < n; i++) w[i * n + i] = 1.0f;

    std::vector<float> x(n), out(n);
    fill_random(x.data(), n, 42);

    tq_matmul(out.data(), x.data(), w.data(), n, n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(out[i], x[i], 1e-5f);
    }
}

TEST(TqOps, MatMulLarge) {
    /* Realistic size: 128 x 256 */
    const int n = 128, d = 256;
    std::vector<float> w(n * d), x(d), out(n), ref(n);

    fill_random(w.data(), n * d, 100);
    fill_random(x.data(), d, 200);

    tq_matmul(out.data(), x.data(), w.data(), n, d);
    ref_matmul(ref.data(), x.data(), w.data(), n, d);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(out[i], ref[i], std::abs(ref[i]) * 1e-4f + 1e-4f)
            << "Mismatch at index " << i;
    }
}

TEST(TqOps, MatMulNEONAligned) {
    /* Test with sizes that exercise NEON 16-element unroll */
    const int n = 32, d = 128;  /* d is multiple of 16 */
    std::vector<float> w(n * d), x(d), out(n), ref(n);

    fill_random(w.data(), n * d, 300);
    fill_random(x.data(), d, 400);

    tq_matmul(out.data(), x.data(), w.data(), n, d);
    ref_matmul(ref.data(), x.data(), w.data(), n, d);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(out[i], ref[i], std::abs(ref[i]) * 1e-4f + 1e-4f);
    }
}

TEST(TqOps, MatMulNEONUnaligned) {
    /* Test with size NOT multiple of 4/16 to exercise scalar tail */
    const int n = 7, d = 13;
    std::vector<float> w(n * d), x(d), out(n), ref(n);

    fill_random(w.data(), n * d, 500);
    fill_random(x.data(), d, 600);

    tq_matmul(out.data(), x.data(), w.data(), n, d);
    ref_matmul(ref.data(), x.data(), w.data(), n, d);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(out[i], ref[i], std::abs(ref[i]) * 1e-4f + 1e-4f);
    }
}

TEST(TqOps, MatMulMultiThreaded) {
    /* Large n to trigger multi-threaded path (n >= 256) */
    const int n = 1024, d = 512;
    std::vector<float> w(n * d), x(d), out(n), ref(n);

    fill_random(w.data(), n * d, 700);
    fill_random(x.data(), d, 800);

    /* Enable 4 threads */
    tq_set_threads(4);

    tq_matmul(out.data(), x.data(), w.data(), n, d);
    ref_matmul(ref.data(), x.data(), w.data(), n, d);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(out[i], ref[i], std::abs(ref[i]) * 1e-4f + 1e-4f)
            << "Mismatch at row " << i;
    }

    /* Restore single-threaded */
    tq_set_threads(1);
}

TEST(TqOps, MatMulMultiThreadedVocab) {
    /* Simulate vocab projection: very large n, moderate d */
    const int n = 4096, d = 256;
    std::vector<float> w(n * d), x(d), out(n), ref(n);

    fill_random(w.data(), n * d, 900);
    fill_random(x.data(), d, 1000);

    tq_set_threads(4);
    tq_matmul(out.data(), x.data(), w.data(), n, d);

    ref_matmul(ref.data(), x.data(), w.data(), n, d);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(out[i], ref[i], std::abs(ref[i]) * 1e-4f + 1e-4f)
            << "Mismatch at row " << i;
    }

    tq_set_threads(1);
}

TEST(TqOps, SetGetThreads) {
    tq_set_threads(8);
    EXPECT_EQ(tq_get_threads(), 8);
    tq_set_threads(1);
    EXPECT_EQ(tq_get_threads(), 1);
    /* Clamp to valid range */
    tq_set_threads(0);
    EXPECT_EQ(tq_get_threads(), 1);
    tq_set_threads(100);
    EXPECT_EQ(tq_get_threads(), 16);
    tq_set_threads(1);
}

/* ============================================================
 * RMSNorm tests
 * ============================================================ */

TEST(TqOps, RMSNormBasic) {
    const int n = 64;
    std::vector<float> x(n), weight(n, 1.0f), out(n), ref(n);

    fill_random(x.data(), n, 42);

    tq_rmsnorm(out.data(), x.data(), weight.data(), n, 1e-5f);
    ref_rmsnorm(ref.data(), x.data(), weight.data(), n, 1e-5f);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(out[i], ref[i], 1e-5f);
    }
}

TEST(TqOps, RMSNormWithWeights) {
    const int n = 128;
    std::vector<float> x(n), weight(n), out(n), ref(n);

    fill_random(x.data(), n, 10);
    fill_random(weight.data(), n, 20);

    tq_rmsnorm(out.data(), x.data(), weight.data(), n, 1e-5f);
    ref_rmsnorm(ref.data(), x.data(), weight.data(), n, 1e-5f);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(out[i], ref[i], std::abs(ref[i]) * 1e-4f + 1e-5f);
    }
}

TEST(TqOps, RMSNormOutputNorm) {
    /* Verify that output has unit RMS when weight=1 */
    const int n = 256;
    std::vector<float> x(n), weight(n, 1.0f), out(n);

    fill_random(x.data(), n, 77);

    tq_rmsnorm(out.data(), x.data(), weight.data(), n, 1e-5f);

    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += out[i] * out[i];
    float rms = std::sqrt(ss / n);

    EXPECT_NEAR(rms, 1.0f, 1e-3f);
}

/* ============================================================
 * RoPE tests
 * ============================================================ */

TEST(TqOps, RoPEPosition0) {
    /* At position 0, RoPE should not change anything
     * (cos(0)=1, sin(0)=0) */
    const int head_dim = 64;
    const int n_heads = 2, n_kv_heads = 2;
    std::vector<float> q(n_heads * head_dim), k(n_kv_heads * head_dim);
    std::vector<float> q_orig, k_orig;

    fill_random(q.data(), n_heads * head_dim, 42);
    fill_random(k.data(), n_kv_heads * head_dim, 43);
    q_orig = q;
    k_orig = k;

    tq_rope(q.data(), k.data(), 0, head_dim, n_heads, n_kv_heads, 10000.0f);

    for (int i = 0; i < n_heads * head_dim; i++) {
        EXPECT_NEAR(q[i], q_orig[i], 1e-5f);
    }
    for (int i = 0; i < n_kv_heads * head_dim; i++) {
        EXPECT_NEAR(k[i], k_orig[i], 1e-5f);
    }
}

TEST(TqOps, RoPEPreservesNorm) {
    /* RoPE is a rotation, so it preserves L2 norm */
    const int head_dim = 128;
    const int n_heads = 4, n_kv_heads = 2;
    std::vector<float> q(n_heads * head_dim), k(n_kv_heads * head_dim);

    fill_random(q.data(), n_heads * head_dim, 100);
    fill_random(k.data(), n_kv_heads * head_dim, 200);

    /* Compute norms before */
    auto norm = [](const float* v, int n) {
        float s = 0;
        for (int i = 0; i < n; i++) s += v[i] * v[i];
        return std::sqrt(s);
    };

    std::vector<float> q_norms_before(n_heads), k_norms_before(n_kv_heads);
    for (int h = 0; h < n_heads; h++) {
        q_norms_before[h] = norm(q.data() + h * head_dim, head_dim);
    }
    for (int h = 0; h < n_kv_heads; h++) {
        k_norms_before[h] = norm(k.data() + h * head_dim, head_dim);
    }

    tq_rope(q.data(), k.data(), 42, head_dim, n_heads, n_kv_heads, 10000.0f);

    /* Verify norms are preserved */
    for (int h = 0; h < n_heads; h++) {
        float n_after = norm(q.data() + h * head_dim, head_dim);
        EXPECT_NEAR(n_after, q_norms_before[h], q_norms_before[h] * 1e-4f);
    }
    for (int h = 0; h < n_kv_heads; h++) {
        float n_after = norm(k.data() + h * head_dim, head_dim);
        EXPECT_NEAR(n_after, k_norms_before[h], k_norms_before[h] * 1e-4f);
    }
}

TEST(TqOps, RoPEGQA) {
    /* GQA: different number of Q and KV heads */
    const int head_dim = 64;
    const int n_heads = 8, n_kv_heads = 2;
    std::vector<float> q(n_heads * head_dim), k(n_kv_heads * head_dim);

    fill_random(q.data(), n_heads * head_dim, 55);
    fill_random(k.data(), n_kv_heads * head_dim, 66);

    /* Should not crash */
    tq_rope(q.data(), k.data(), 10, head_dim, n_heads, n_kv_heads, 10000.0f);

    /* Values should have changed (non-zero position) */
    std::vector<float> q_zero(n_heads * head_dim);
    fill_random(q_zero.data(), n_heads * head_dim, 55);
    bool changed = false;
    for (int i = 0; i < n_heads * head_dim; i++) {
        if (std::abs(q[i] - q_zero[i]) > 1e-6f) { changed = true; break; }
    }
    EXPECT_TRUE(changed);
}

/* ============================================================
 * SiLU tests
 * ============================================================ */

TEST(TqOps, SiLUBasic) {
    const int n = 128;
    std::vector<float> x(n), ref(n);

    fill_random(x.data(), n, 42);
    ref = x;

    tq_silu(x.data(), n);
    ref_silu(ref.data(), n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], ref[i], std::abs(ref[i]) * 1e-4f + 1e-6f);
    }
}

TEST(TqOps, SiLUZero) {
    /* SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0 */
    float x[] = {0.0f};
    tq_silu(x, 1);
    EXPECT_NEAR(x[0], 0.0f, 1e-7f);
}

TEST(TqOps, SiLUMonotonic) {
    /* SiLU is roughly monotonic for x > 0 */
    const int n = 100;
    std::vector<float> x(n);
    for (int i = 0; i < n; i++) x[i] = 0.01f * (i + 1);

    tq_silu(x.data(), n);

    for (int i = 1; i < n; i++) {
        EXPECT_GE(x[i], x[i-1] - 1e-6f);
    }
}

TEST(TqOps, SiLUUnaligned) {
    /* Size not multiple of 4 to exercise scalar tail */
    const int n = 13;
    std::vector<float> x(n), ref(n);

    fill_random(x.data(), n, 99);
    ref = x;

    tq_silu(x.data(), n);
    ref_silu(ref.data(), n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], ref[i], std::abs(ref[i]) * 1e-4f + 1e-6f);
    }
}

/* ============================================================
 * Softmax tests
 * ============================================================ */

TEST(TqOps, SoftmaxSumsToOne) {
    const int n = 64;
    std::vector<float> x(n);
    fill_random(x.data(), n, 42);

    tq_softmax(x.data(), n);

    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(TqOps, SoftmaxAllPositive) {
    const int n = 32;
    std::vector<float> x(n);
    fill_random(x.data(), n, 42);

    tq_softmax(x.data(), n);

    for (int i = 0; i < n; i++) {
        EXPECT_GE(x[i], 0.0f);
        EXPECT_LE(x[i], 1.0f);
    }
}

TEST(TqOps, SoftmaxMatchesReference) {
    const int n = 128;
    std::vector<float> x(n), ref(n);
    fill_random(x.data(), n, 42);
    ref = x;

    tq_softmax(x.data(), n);
    ref_softmax(ref.data(), n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], ref[i], 1e-5f);
    }
}

TEST(TqOps, SoftmaxLargeValues) {
    /* Numerical stability: large values should not cause overflow */
    float x[] = {1000.0f, 999.0f, 998.0f, 997.0f};
    tq_softmax(x, 4);

    float sum = x[0] + x[1] + x[2] + x[3];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
    EXPECT_GT(x[0], x[1]); /* Largest input -> largest probability */
}

TEST(TqOps, SoftmaxSingleElement) {
    float x[] = {42.0f};
    tq_softmax(x, 1);
    EXPECT_NEAR(x[0], 1.0f, 1e-6f);
}

/* ============================================================
 * Add and Mul tests
 * ============================================================ */

TEST(TqOps, AddBasic) {
    const int n = 128;
    std::vector<float> a(n), b(n), out(n);
    fill_random(a.data(), n, 10);
    fill_random(b.data(), n, 20);

    tq_add(out.data(), a.data(), b.data(), n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(out[i], a[i] + b[i], 1e-6f);
    }
}

TEST(TqOps, AddInPlace) {
    /* out == a: in-place addition */
    const int n = 64;
    std::vector<float> a(n), b(n), orig(n);
    fill_random(a.data(), n, 30);
    fill_random(b.data(), n, 40);
    orig = a;

    tq_add(a.data(), a.data(), b.data(), n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(a[i], orig[i] + b[i], 1e-6f);
    }
}

TEST(TqOps, MulBasic) {
    const int n = 128;
    std::vector<float> a(n), b(n), out(n);
    fill_random(a.data(), n, 50);
    fill_random(b.data(), n, 60);

    tq_mul(out.data(), a.data(), b.data(), n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(out[i], a[i] * b[i], 1e-6f);
    }
}

TEST(TqOps, MulUnaligned) {
    /* Test with non-multiple-of-4 size */
    const int n = 11;
    std::vector<float> a(n), b(n), out(n);
    fill_random(a.data(), n, 70);
    fill_random(b.data(), n, 80);

    tq_mul(out.data(), a.data(), b.data(), n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(out[i], a[i] * b[i], 1e-6f);
    }
}

/* ============================================================
 * Sampling tests
 * ============================================================ */

TEST(TqOps, SampleArgmax) {
    float logits[] = {1.0f, 5.0f, 3.0f, 2.0f, 4.0f};
    int result = tq_sample_argmax(logits, 5);
    EXPECT_EQ(result, 1);
}

TEST(TqOps, SampleArgmaxFirst) {
    float logits[] = {10.0f, 1.0f, 2.0f};
    EXPECT_EQ(tq_sample_argmax(logits, 3), 0);
}

TEST(TqOps, SampleArgmaxLast) {
    float logits[] = {1.0f, 2.0f, 100.0f};
    EXPECT_EQ(tq_sample_argmax(logits, 3), 2);
}

TEST(TqOps, SampleTopPGreedy) {
    /* Temperature ~0 should behave like argmax */
    float logits[] = {1.0f, 5.0f, 3.0f, 2.0f, 4.0f};
    unsigned long long rng = 42;
    int result = tq_sample_topp(logits, 5, 0.001f, 0.9f, &rng);
    /* With very low temperature, should pick max */
    EXPECT_EQ(result, 1);
}

TEST(TqOps, SampleTopPDistribution) {
    /* With temperature=1 and top_p=1, all tokens possible */
    float logits[] = {0.0f, 0.0f, 0.0f, 0.0f};
    int counts[4] = {0};
    unsigned long long rng = 12345;

    for (int i = 0; i < 1000; i++) {
        int tok = tq_sample_topp(logits, 4, 1.0f, 1.0f, &rng);
        ASSERT_GE(tok, 0);
        ASSERT_LT(tok, 4);
        counts[tok]++;
    }

    /* With uniform logits, each should get ~25% of samples */
    for (int i = 0; i < 4; i++) {
        EXPECT_GT(counts[i], 100); /* At least 10% */
        EXPECT_LT(counts[i], 500); /* At most 50% */
    }
}

/* ============================================================
 * State management tests
 * ============================================================ */

TEST(TqOps, CreateFreeState) {
    tq_model_config_t config;
    memset(&config, 0, sizeof(config));
    config.n_layers = 2;
    config.hidden_dim = 64;
    config.intermediate_dim = 128;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.head_dim = 16;
    config.vocab_size = 100;
    config.max_seq_len = 32;
    config.rope_freq_base = 10000.0f;
    config.rms_norm_eps = 1e-5f;

    tq_state_t* state = tq_create_state(&config, TQ_TYPE_UNIFORM_4B);
    ASSERT_NE(state, nullptr);
    EXPECT_NE(state->x, nullptr);
    EXPECT_NE(state->logits, nullptr);
    EXPECT_NE(state->key_cache, nullptr);
    EXPECT_NE(state->value_cache, nullptr);

    tq_free_state(state);
}

TEST(TqOps, CreateStateNull) {
    tq_state_t* state = tq_create_state(NULL, TQ_TYPE_UNIFORM_4B);
    EXPECT_EQ(state, nullptr);
}

/* ============================================================
 * Default config test
 * ============================================================ */

TEST(TqOps, DefaultGenConfig) {
    tq_gen_config_t config = tq_default_gen_config();
    EXPECT_GT(config.temperature, 0.0f);
    EXPECT_GT(config.top_p, 0.0f);
    EXPECT_GT(config.max_tokens, 0);
    EXPECT_EQ(config.on_token, nullptr);
}

/* ============================================================
 * Integration: mini forward pass test
 *
 * Creates a tiny model (1 layer, dim=8) and runs a forward pass.
 * This verifies the full pipeline works end-to-end.
 * ============================================================ */

TEST(TqOps, MiniForwardPass) {
    /* Build a tiny model in memory */
    tq_model_t model;
    memset(&model, 0, sizeof(model));

    model.config.n_layers = 1;
    model.config.hidden_dim = 16;
    model.config.intermediate_dim = 32;
    model.config.n_heads = 2;
    model.config.n_kv_heads = 2;
    model.config.head_dim = 8;
    model.config.vocab_size = 10;
    model.config.max_seq_len = 16;
    model.config.rope_freq_base = 10000.0f;
    model.config.rms_norm_eps = 1e-5f;

    int dim = model.config.hidden_dim;
    int kv_dim = model.config.n_kv_heads * model.config.head_dim;
    int q_dim = model.config.n_heads * model.config.head_dim;
    int inter = model.config.intermediate_dim;
    int vocab = model.config.vocab_size;

    /* Allocate weights with small random values */
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    auto fill = [&](int size) -> float* {
        float* p = new float[size];
        for (int i = 0; i < size; i++) p[i] = dist(rng);
        return p;
    };

    model.token_embedding = fill(vocab * dim);
    model.layers = new tq_layer_weights_t[1];
    memset(model.layers, 0, sizeof(tq_layer_weights_t));

    tq_layer_weights_t* layer = &model.layers[0];
    layer->attn_norm = new float[dim];
    layer->ffn_norm  = new float[dim];
    for (int i = 0; i < dim; i++) {
        layer->attn_norm[i] = 1.0f;
        layer->ffn_norm[i]  = 1.0f;
    }
    layer->wq     = fill(q_dim * dim);
    layer->wk     = fill(kv_dim * dim);
    layer->wv     = fill(kv_dim * dim);
    layer->wo     = fill(dim * q_dim);
    layer->w_gate = fill(inter * dim);
    layer->w_up   = fill(inter * dim);
    layer->w_down = fill(dim * inter);

    model.output_norm = new float[dim];
    for (int i = 0; i < dim; i++) model.output_norm[i] = 1.0f;
    model.output_weight = fill(vocab * dim);

    /* Create state */
    tq_state_t* state = tq_create_state(&model.config, TQ_TYPE_UNIFORM_4B);
    ASSERT_NE(state, nullptr);

    /* Forward pass for token 3 at position 0 */
    float* logits = tq_forward(&model, state, 3, 0);
    ASSERT_NE(logits, nullptr);

    /* Logits should be finite */
    for (int i = 0; i < vocab; i++) {
        EXPECT_TRUE(std::isfinite(logits[i]))
            << "logit[" << i << "] = " << logits[i];
    }

    /* Should be able to sample */
    int next = tq_sample_argmax(logits, vocab);
    EXPECT_GE(next, 0);
    EXPECT_LT(next, vocab);

    /* Forward another token at position 1 */
    float* logits2 = tq_forward(&model, state, next, 1);
    ASSERT_NE(logits2, nullptr);
    for (int i = 0; i < vocab; i++) {
        EXPECT_TRUE(std::isfinite(logits2[i]));
    }

    /* Cleanup */
    tq_free_state(state);
    delete[] model.token_embedding;
    delete[] layer->attn_norm;
    delete[] layer->ffn_norm;
    delete[] layer->wq;
    delete[] layer->wk;
    delete[] layer->wv;
    delete[] layer->wo;
    delete[] layer->w_gate;
    delete[] layer->w_up;
    delete[] layer->w_down;
    delete[] model.output_norm;
    delete[] model.output_weight;
    delete[] model.layers;
}
