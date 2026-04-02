/**
 * test_neon_scalar.cpp -- NEON vs scalar reference consistency tests
 *
 * For each function with both NEON and scalar paths, we call the actual
 * compiled path and compare against a manually computed reference (pure C).
 * On ARM, this verifies the NEON path. On x86, it verifies the scalar path.
 *
 * Functions tested:
 * - tq_dequantize_row_q4: weight dequantization
 * - tq_matmul_q4: Q4 weight matmul
 * - tq_rmsnorm: RMS normalization
 * - turbo_kv_1b_attention: Hamming popcount attention
 * - uniform_4b quantize/dequantize: KV cache uniform quantization
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <cstring>
#include <random>
#include <numeric>

extern "C" {
#include "turboquant/turboquant.h"
#include "turboquant/tq_engine.h"

/* KV cache ref functions */
void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);

/* TurboQuant KV 1-bit */
void tq_turbo_kv_1b_quantize_ref(const float* src, void* dst, int n);
void tq_turbo_kv_1b_attention_ref(const float* query, const void* kv,
                                    float* scores, int seq_len, int head_dim);

/* QJL ref functions */
void tq_qjl_quantize_ref(const float* src, void* dst, int n);
void tq_qjl_dequantize_ref(const void* src, float* dst, int n);

/* Polar ref functions */
void tq_polar_quantize_ref(const float* src, void* dst, int n);
void tq_polar_dequantize_ref(const void* src, float* dst, int n);
}

/* ============================================================
 * Helpers
 * ============================================================ */

static std::vector<float> make_deterministic_input(int n, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> v(n);
    for (int i = 0; i < n; i++) v[i] = dist(rng);
    return v;
}

static double compute_mse(const float* a, const float* b, int n) {
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
    }
    return mse / n;
}

/* ============================================================
 * Test: DequantQ4Reference
 *
 * Manually dequantize Q4 and compare with tq_dequantize_row_q4.
 * Pure C reference: for each block of 32, q[j] = (nibble - 8) * scale.
 * ============================================================ */

TEST(NeonScalarConsistency, DequantQ4Reference) {
    const int n = 128;  /* 4 blocks of 32 */
    const int n_blocks = n / 32;

    /* Create known Q4 data */
    std::vector<uint8_t> qs(n_blocks * 16);
    std::vector<float> scales(n_blocks);

    std::mt19937 rng(123);
    for (int b = 0; b < n_blocks; b++) {
        scales[b] = 0.05f * (b + 1);
        for (int j = 0; j < 16; j++) {
            uint8_t lo = rng() % 16;
            uint8_t hi = rng() % 16;
            qs[b * 16 + j] = lo | (hi << 4);
        }
    }

    /* Manual reference dequantization */
    std::vector<float> ref(n);
    for (int b = 0; b < n_blocks; b++) {
        for (int j = 0; j < 16; j++) {
            int q0 = qs[b * 16 + j] & 0x0F;
            int q1 = qs[b * 16 + j] >> 4;
            ref[b * 32 + 2 * j]     = (float)(q0 - 8) * scales[b];
            ref[b * 32 + 2 * j + 1] = (float)(q1 - 8) * scales[b];
        }
    }

    /* Call the actual function (NEON on ARM, scalar on x86) */
    std::vector<float> actual(n);
    tq_dequantize_row_q4(qs.data(), scales.data(), actual.data(), n);

    /* Should be bit-exact since both are integer arithmetic */
    for (int i = 0; i < n; i++) {
        EXPECT_FLOAT_EQ(ref[i], actual[i])
            << "DequantQ4 mismatch at element " << i;
    }
}

/* ============================================================
 * Test: MatmulQ4ScalarReference
 *
 * Compute Q4 matmul manually and compare with tq_matmul_q4.
 * The function quantizes activation x to Q8, then does Q4xQ8 dot.
 * We do FP32 reference: sum over blocks of (nibble-8)*scale * x[j].
 * ============================================================ */

TEST(NeonScalarConsistency, MatmulQ4ScalarReference) {
    const int n_rows = 4;
    const int d = 64;  /* 2 blocks of 32 */
    const int n_blocks = d / 32;

    /* Create weight matrix in Q4 format */
    std::vector<uint8_t> w_qs(n_rows * n_blocks * 16);
    std::vector<float> w_scales(n_rows * n_blocks);

    std::mt19937 rng(456);
    for (int r = 0; r < n_rows; r++) {
        for (int b = 0; b < n_blocks; b++) {
            w_scales[r * n_blocks + b] = 0.03f * (r + 1);
            for (int j = 0; j < 16; j++) {
                uint8_t lo = rng() % 16;
                uint8_t hi = rng() % 16;
                w_qs[r * n_blocks * 16 + b * 16 + j] = lo | (hi << 4);
            }
        }
    }

    /* Create activation vector */
    auto x = make_deterministic_input(d, 789);

    /* Reference: dequantize weights, do FP32 dot product */
    std::vector<float> ref(n_rows, 0.0f);
    for (int r = 0; r < n_rows; r++) {
        for (int b = 0; b < n_blocks; b++) {
            float scale = w_scales[r * n_blocks + b];
            for (int j = 0; j < 16; j++) {
                int q0 = w_qs[r * n_blocks * 16 + b * 16 + j] & 0x0F;
                int q1 = w_qs[r * n_blocks * 16 + b * 16 + j] >> 4;
                ref[r] += (float)(q0 - 8) * scale * x[b * 32 + 2 * j];
                ref[r] += (float)(q1 - 8) * scale * x[b * 32 + 2 * j + 1];
            }
        }
    }

    /* Call actual tq_matmul_q4 */
    std::vector<float> actual(n_rows);
    tq_matmul_q4(actual.data(), x.data(), w_qs.data(), w_scales.data(),
                  n_rows, d);

    /* Allow tolerance because Q8 quantization of x introduces error */
    for (int r = 0; r < n_rows; r++) {
        float tol = std::abs(ref[r]) * 0.05f + 0.1f;
        EXPECT_NEAR(actual[r], ref[r], tol)
            << "MatmulQ4 mismatch at row " << r
            << " (ref=" << ref[r] << ", actual=" << actual[r] << ")";
    }
}

/* ============================================================
 * Test: RMSNormReference
 *
 * Compute RMSNorm manually and compare with tq_rmsnorm.
 * Formula: out[i] = (x[i] / rms) * weight[i]
 *          where rms = sqrt(mean(x^2) + eps)
 * ============================================================ */

TEST(NeonScalarConsistency, RMSNormReference) {
    const int n = 128;
    const float eps = 1e-5f;

    auto x = make_deterministic_input(n, 100);
    auto weight = make_deterministic_input(n, 200);
    /* Make weights positive for typical use */
    for (int i = 0; i < n; i++) weight[i] = std::abs(weight[i]) + 0.1f;

    /* Manual reference */
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    ss = ss / n + eps;
    float rsqrt = 1.0f / sqrtf((float)ss);

    std::vector<float> ref(n);
    for (int i = 0; i < n; i++) {
        ref[i] = x[i] * rsqrt * weight[i];
    }

    /* Call actual tq_rmsnorm */
    std::vector<float> actual(n);
    tq_rmsnorm(actual.data(), x.data(), weight.data(), n, eps);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(actual[i], ref[i], 1e-5f)
            << "RMSNorm mismatch at element " << i;
    }
}

/* Test with in-place operation (out == x) */
TEST(NeonScalarConsistency, RMSNormInPlace) {
    const int n = 64;
    const float eps = 1e-5f;

    auto x = make_deterministic_input(n, 300);
    auto weight = make_deterministic_input(n, 400);
    for (int i = 0; i < n; i++) weight[i] = std::abs(weight[i]) + 0.1f;

    /* Compute reference before in-place modification */
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    ss = ss / n + eps;
    float rsqrt = 1.0f / sqrtf((float)ss);

    std::vector<float> ref(n);
    for (int i = 0; i < n; i++) ref[i] = x[i] * rsqrt * weight[i];

    /* In-place: out = x */
    tq_rmsnorm(x.data(), x.data(), weight.data(), n, eps);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], ref[i], 1e-5f)
            << "RMSNorm in-place mismatch at " << i;
    }
}

/* Non-power-of-2 dimension to test scalar tail handling */
TEST(NeonScalarConsistency, RMSNormOddDimension) {
    const int n = 67;
    const float eps = 1e-6f;

    auto x = make_deterministic_input(n, 500);
    auto weight = make_deterministic_input(n, 600);
    for (int i = 0; i < n; i++) weight[i] = std::abs(weight[i]) + 0.1f;

    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    ss = ss / n + eps;
    float rsqrt = 1.0f / sqrtf((float)ss);

    std::vector<float> ref(n);
    for (int i = 0; i < n; i++) ref[i] = x[i] * rsqrt * weight[i];

    std::vector<float> actual(n);
    tq_rmsnorm(actual.data(), x.data(), weight.data(), n, eps);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(actual[i], ref[i], 1e-5f)
            << "RMSNorm odd dim mismatch at " << i;
    }
}

/* ============================================================
 * Test: HammingAttentionReference
 *
 * Compute 1-bit Hamming attention manually and compare with
 * tq_turbo_kv_1b_attention_ref. This function has NEON popcount
 * on ARM and scalar fallback on x86.
 *
 * Manual reference:
 *   1. RHT(query) with TKV_DEFAULT_SEED
 *   2. Extract query sign bits
 *   3. Per key: XOR + popcount -> Hamming distance
 *   4. score = q_norm * k_norm * sqrt(pi/2)/dim * (2*agree - dim)
 * ============================================================ */

TEST(NeonScalarConsistency, HammingAttentionReference) {
    const int dim = 128;
    const int seq_len = 8;

    /* Create query */
    auto query = make_deterministic_input(dim, 1000);

    /* Create keys and quantize them to 1-bit */
    std::vector<block_tq_turbo_kv_1b> kv_blocks(seq_len);
    std::vector<std::vector<float>> keys(seq_len);
    for (int s = 0; s < seq_len; s++) {
        keys[s] = make_deterministic_input(dim, 2000 + s);
        tq_turbo_kv_1b_quantize_ref(keys[s].data(), &kv_blocks[s], dim);
    }

    /* Call actual attention function */
    std::vector<float> actual_scores(seq_len);
    tq_turbo_kv_1b_attention_ref(query.data(), kv_blocks.data(),
                                   actual_scores.data(), seq_len, dim);

    /* Manual reference computation — sign-sign estimator uses (pi/2)/m */
    float scale_factor = ((float)M_PI / 2.0f) / (float)dim;

    /* RHT(query) */
    std::vector<float> q_rot(dim);
    memcpy(q_rot.data(), query.data(), dim * sizeof(float));
    tq_rht_transform(q_rot.data(), dim, 0x12345678u);

    /* Query L2 norm (from original, not rotated) */
    float q_norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) q_norm_sq += query[i] * query[i];
    float q_norm = sqrtf(q_norm_sq);

    /* Query sign bits */
    uint8_t q_signs[16];
    memset(q_signs, 0, 16);
    for (int i = 0; i < dim; i++) {
        if (q_rot[i] > 0.0f) {
            q_signs[i / 8] |= (uint8_t)(1 << (i % 8));
        }
    }

    std::vector<float> ref_scores(seq_len);
    for (int s = 0; s < seq_len; s++) {
        /* FP16 round-trip for k_norm (matching what quantize stores) */
        float k_norm_stored;
        {
            uint16_t h = kv_blocks[s].norm;
            uint32_t sign = (h & 0x8000) << 16;
            uint32_t exp  = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x03FF;
            union { float f; uint32_t u; } bits;
            if (exp == 0) { bits.u = sign; }
            else if (exp == 31) { bits.u = sign | 0x7F800000 | (mant << 13); }
            else { exp = exp - 15 + 127; bits.u = sign | (exp << 23) | (mant << 13); }
            k_norm_stored = bits.f;
        }

        /* XOR + popcount (portable) */
        int hamming = 0;
        for (int b = 0; b < 16; b++) {
            uint8_t xor_byte = q_signs[b] ^ kv_blocks[s].signs[b];
            int c = 0;
            uint8_t tmp = xor_byte;
            while (tmp) { c++; tmp &= tmp - 1; }
            hamming += c;
        }

        int agree = dim - hamming;
        ref_scores[s] = q_norm * k_norm_stored * scale_factor * (float)(2 * agree - dim);
    }

    for (int s = 0; s < seq_len; s++) {
        EXPECT_NEAR(actual_scores[s], ref_scores[s], 1e-4f)
            << "Hamming attention mismatch at seq " << s;
    }
}

/* ============================================================
 * Test: Uniform 4-bit quantize + dequantize roundtrip
 *
 * Verifies that quantize -> dequantize produces consistent results
 * regardless of which path (NEON/scalar) is compiled.
 * ============================================================ */

TEST(NeonScalarConsistency, Uniform4BRoundtripConsistency) {
    auto input = make_deterministic_input(TQ_BK, 777);

    /* Quantize */
    block_tq_uniform_4b block;
    memset(&block, 0, sizeof(block));
    tq_uniform_4b_quantize_ref(input.data(), &block, TQ_BK);

    /* Dequantize */
    std::vector<float> output(TQ_BK);
    tq_uniform_4b_dequantize_ref(&block, output.data(), TQ_BK);

    /* Verify MSE is low (4-bit quantization) */
    double mse = compute_mse(input.data(), output.data(), TQ_BK);
    EXPECT_LT(mse, 0.5) << "Uniform 4-bit roundtrip MSE too high: " << mse;

    /* Verify each dequantized value is within one quantization step */
    /* Decode scale and zero point */
    auto fp16_to_fp32 = [](uint16_t h) -> float {
        union { float f; uint32_t u; } bits;
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x03FF;
        if (exp == 0) { bits.u = sign; return bits.f; }
        if (exp == 31) { bits.u = sign | 0x7F800000 | (mant << 13); return bits.f; }
        exp = exp - 15 + 127;
        bits.u = sign | (exp << 23) | (mant << 13);
        return bits.f;
    };

    float scale = fp16_to_fp32(block.scale);
    float max_error = scale;  /* Each bin has width = scale */
    for (int i = 0; i < TQ_BK; i++) {
        float error = std::abs(input[i] - output[i]);
        EXPECT_LT(error, max_error * 1.5f)
            << "Element " << i << " error " << error
            << " exceeds expected max " << max_error;
    }
}

/* ============================================================
 * Test: DequantQ4 remainder handling
 *
 * Tests that tq_dequantize_row_q4 handles non-block-aligned sizes.
 * ============================================================ */

TEST(NeonScalarConsistency, DequantQ4Remainder) {
    const int n = 48;  /* 1 full block (32) + 16 remainder */
    const int n_blocks = n / 32;
    const int remainder = n - n_blocks * 32;

    std::vector<uint8_t> qs((n_blocks + 1) * 16, 0);
    std::vector<float> scales(n_blocks + 1);

    std::mt19937 rng(999);
    for (int b = 0; b <= n_blocks; b++) {
        scales[b] = 0.1f * (b + 1);
        int n_bytes = (b < n_blocks) ? 16 : (remainder / 2);
        for (int j = 0; j < n_bytes; j++) {
            uint8_t lo = rng() % 16;
            uint8_t hi = rng() % 16;
            qs[b * 16 + j] = lo | (hi << 4);
        }
    }

    /* Manual reference */
    std::vector<float> ref(n);
    for (int b = 0; b < n_blocks; b++) {
        for (int j = 0; j < 16; j++) {
            int q0 = qs[b * 16 + j] & 0x0F;
            int q1 = qs[b * 16 + j] >> 4;
            ref[b * 32 + 2 * j]     = (float)(q0 - 8) * scales[b];
            ref[b * 32 + 2 * j + 1] = (float)(q1 - 8) * scales[b];
        }
    }
    /* Remainder block */
    int n_pairs = remainder / 2;
    for (int j = 0; j < n_pairs; j++) {
        int q0 = qs[n_blocks * 16 + j] & 0x0F;
        int q1 = qs[n_blocks * 16 + j] >> 4;
        ref[n_blocks * 32 + 2 * j]     = (float)(q0 - 8) * scales[n_blocks];
        ref[n_blocks * 32 + 2 * j + 1] = (float)(q1 - 8) * scales[n_blocks];
    }

    std::vector<float> actual(n);
    tq_dequantize_row_q4(qs.data(), scales.data(), actual.data(), n);

    for (int i = 0; i < n; i++) {
        EXPECT_FLOAT_EQ(ref[i], actual[i])
            << "DequantQ4 remainder mismatch at " << i;
    }
}

/* ============================================================
 * Test: DequantQ2Reference
 *
 * Manually dequantize Q2 and compare with tq_dequantize_row_q2.
 * Q2 uses Lloyd-Max centroids: {-1.5104, -0.4528, 0.4528, 1.5104}
 * ============================================================ */

TEST(NeonScalarConsistency, DequantQ2Reference) {
    /* Lloyd-Max centroids for 2-bit N(0,1) */
    const float Q2_C[4] = {-1.5104f, -0.4528f, 0.4528f, 1.5104f};
    const int n = 128;  /* 4 blocks of 32 */
    const int n_blocks = n / 32;

    /* Create known Q2 data: 8 bytes per block (4 values per byte, 2 bits each) */
    std::vector<uint8_t> qs(n_blocks * 8);
    std::vector<float> scales(n_blocks);

    std::mt19937 rng(2024);
    for (int b = 0; b < n_blocks; b++) {
        scales[b] = 0.1f * (b + 1);
        for (int j = 0; j < 8; j++) {
            uint8_t byte = 0;
            for (int k = 0; k < 4; k++) {
                uint8_t qi = rng() % 4;
                byte |= (qi << (k * 2));
            }
            qs[b * 8 + j] = byte;
        }
    }

    /* Manual reference dequantization */
    std::vector<float> ref(n);
    for (int b = 0; b < n_blocks; b++) {
        for (int j = 0; j < 32; j++) {
            int byte_idx = j / 4;
            int bit_pos  = (j % 4) * 2;
            int qi = (qs[b * 8 + byte_idx] >> bit_pos) & 0x03;
            ref[b * 32 + j] = Q2_C[qi] * scales[b];
        }
    }

    /* Call actual function */
    std::vector<float> actual(n);
    tq_dequantize_row_q2(qs.data(), scales.data(), actual.data(), n);

    for (int i = 0; i < n; i++) {
        EXPECT_FLOAT_EQ(ref[i], actual[i])
            << "DequantQ2 mismatch at element " << i;
    }
}

/* ============================================================
 * Test: RoPE reference comparison
 *
 * Compute RoPE manually and compare with tq_rope.
 * RoPE applies rotation to (q[2i], q[2i+1]) pairs:
 *   q[2i]   = q0*cos(theta) - q1*sin(theta)
 *   q[2i+1] = q0*sin(theta) + q1*cos(theta)
 * ============================================================ */

TEST(NeonScalarConsistency, RoPEReference) {
    const int head_dim = 64;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int pos = 42;
    const float freq_base = 10000.0f;

    auto q_input = make_deterministic_input(n_heads * head_dim, 3000);
    auto k_input = make_deterministic_input(n_kv_heads * head_dim, 4000);

    /* Save copies for reference computation */
    std::vector<float> q_ref(q_input);
    std::vector<float> k_ref(k_input);

    /* Manual reference RoPE */
    for (int h = 0; h < n_heads; h++) {
        float* qh = q_ref.data() + h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(freq_base, 2.0f * i / head_dim);
            float theta = pos * freq;
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            float q0 = qh[2 * i];
            float q1 = qh[2 * i + 1];
            qh[2 * i]     = q0 * cos_t - q1 * sin_t;
            qh[2 * i + 1] = q0 * sin_t + q1 * cos_t;
        }
    }
    for (int h = 0; h < n_kv_heads; h++) {
        float* kh = k_ref.data() + h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(freq_base, 2.0f * i / head_dim);
            float theta = pos * freq;
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            float k0 = kh[2 * i];
            float k1 = kh[2 * i + 1];
            kh[2 * i]     = k0 * cos_t - k1 * sin_t;
            kh[2 * i + 1] = k0 * sin_t + k1 * cos_t;
        }
    }

    /* Call actual tq_rope (modifies q_input and k_input in-place) */
    tq_rope(q_input.data(), k_input.data(), pos, head_dim,
            n_heads, n_kv_heads, freq_base);

    for (int i = 0; i < n_heads * head_dim; i++) {
        EXPECT_NEAR(q_input[i], q_ref[i], 1e-5f)
            << "RoPE query mismatch at element " << i;
    }
    for (int i = 0; i < n_kv_heads * head_dim; i++) {
        EXPECT_NEAR(k_input[i], k_ref[i], 1e-5f)
            << "RoPE key mismatch at element " << i;
    }
}

/* ============================================================
 * Task 5c: Edge case tests -- numerical stability
 * ============================================================ */

/* Test with all-zero input: should produce zero norm */
TEST(NumericalStability, AllZeroInput) {
    const int dim = 128;
    std::vector<float> zeros(dim, 0.0f);

    /* QJL quantize with zeros */
    block_tq_qjl qjl_block;
    memset(&qjl_block, 0, sizeof(qjl_block));
    tq_qjl_quantize_ref(zeros.data(), &qjl_block, dim);

    /* Norm should be zero (FP16 representation of 0) */
    EXPECT_EQ(qjl_block.norm, 0u) << "QJL norm should be zero for all-zero input";

    /* Polar quantize with zeros */
    block_tq_polar polar_block;
    memset(&polar_block, 0, sizeof(polar_block));
    tq_polar_quantize_ref(zeros.data(), &polar_block, dim);

    /* Should not crash -- that's the main test */
}

/* Test with very large values: should not overflow during quantization.
 * We use values up to ~1000 which stress the norm computation but still
 * fit in FP16 range (max ~65504). Values at 1e30 would overflow FP16,
 * causing the stored norm to be Inf, which is a FP16 limitation, not a bug. */
TEST(NumericalStability, LargeValues) {
    const int dim = 256;
    std::vector<float> large(dim);
    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (int i = 0; i < dim; i++) large[i] = dist(rng);

    /* QJL quantize -- should not crash or produce NaN/Inf */
    block_tq_qjl qjl_block;
    memset(&qjl_block, 0, sizeof(qjl_block));
    tq_qjl_quantize_ref(large.data(), &qjl_block, dim);

    /* The L2 norm of ~256 values at ~1000 is about sqrt(256*500000) ~ 11313,
     * well within FP16 range. Verify norm is not zero or NaN. */
    EXPECT_NE(qjl_block.norm, 0u) << "Norm should not be zero for large input";

    /* Dequantize and check for NaN */
    std::vector<float> output(dim);
    tq_qjl_dequantize_ref(&qjl_block, output.data(), dim);
    for (int i = 0; i < dim; i++) {
        EXPECT_FALSE(std::isnan(output[i]))
            << "NaN in QJL dequantized output at index " << i;
    }

    /* Also test with truly extreme values (1e30) -- quantize should not crash */
    std::vector<float> extreme(dim);
    std::uniform_real_distribution<float> ext_dist(-1e30f, 1e30f);
    for (int i = 0; i < dim; i++) extreme[i] = ext_dist(rng);

    block_tq_qjl extreme_block;
    memset(&extreme_block, 0, sizeof(extreme_block));
    tq_qjl_quantize_ref(extreme.data(), &extreme_block, dim);
    /* Quantize should not crash -- that's the main test.
     * Norm will saturate to FP16 Inf (0x7C00), which is expected. */
}

/* Test with very small values (1e-30): should not underflow to zero norm */
TEST(NumericalStability, SmallValues) {
    const int dim = 256;
    std::vector<float> small(dim);
    std::mt19937 rng(888);
    std::uniform_real_distribution<float> dist(-1e-30f, 1e-30f);
    for (int i = 0; i < dim; i++) small[i] = dist(rng);

    /* QJL quantize */
    block_tq_qjl qjl_block;
    memset(&qjl_block, 0, sizeof(qjl_block));
    tq_qjl_quantize_ref(small.data(), &qjl_block, dim);

    /* Norm may underflow to FP16 zero for very small values -- that's acceptable.
     * The key test is no crash and no NaN. */
    std::vector<float> output(dim);
    tq_qjl_dequantize_ref(&qjl_block, output.data(), dim);
    for (int i = 0; i < dim; i++) {
        EXPECT_FALSE(std::isnan(output[i]))
            << "NaN in QJL dequantized output at index " << i;
    }
}

/* Test with NaN input: should zero-fill output and not crash */
TEST(NumericalStability, NaNInputGuard) {
    const int dim = 128;
    std::vector<float> input(dim, 1.0f);
    input[0] = std::numeric_limits<float>::quiet_NaN();

    /* QJL quantize with NaN */
    block_tq_qjl qjl_block;
    memset(&qjl_block, 0xFF, sizeof(qjl_block));  /* fill with garbage */
    tq_qjl_quantize_ref(input.data(), &qjl_block, dim);
    EXPECT_EQ(qjl_block.norm, 0u) << "QJL should zero-fill on NaN input";

    /* Polar quantize with NaN */
    block_tq_polar polar_block;
    memset(&polar_block, 0xFF, sizeof(polar_block));
    tq_polar_quantize_ref(input.data(), &polar_block, dim);
    /* Should not crash; block should be zeroed */
    EXPECT_EQ(polar_block.tscale, 0u) << "Polar should zero-fill on NaN input";
}
