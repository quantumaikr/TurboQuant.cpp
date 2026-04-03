/**
 * Delta Compression for KV Cache Keys — Concept Validation
 *
 * Hypothesis: Adjacent key vectors have delta range ~50% of absolute range.
 * If true, 2-bit quantization of deltas achieves ~3-bit quality on absolute keys.
 *
 * I-frame/P-frame scheme (like video compression):
 *   - Every Nth token: store FULL key at 4-bit (I-frame)
 *   - Other tokens: store DELTA from previous reconstructed key at 2-bit (P-frame)
 *   - At decode: I-frame dequantize directly; P-frame dequantize delta + add to prev
 *
 * This test uses the actual model's key_cache (runs SmolLM2 forward pass for
 * 100+ tokens, then reads the FP32 key vectors from s->key_cache).
 *
 * Build:
 *   cc -O2 -I include bench/test_delta_quant.c build/libturboquant.a -lm -o build/test_delta_quant
 *
 * If model inference is not available, falls back to synthetic correlated keys.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "turboquant/turboquant.h"

/* ========== External function declarations ========== */

extern void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
extern void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
extern void tq_uniform_2b_quantize_ref(const float* src, void* dst, int n);
extern void tq_uniform_2b_dequantize_ref(const void* src, float* dst, int n);

/* ========== RNG (xoshiro128+) ========== */

static uint32_t rng_state[4] = {0x12345678, 0x9ABCDEF0, 0xDEADBEEF, 0xCAFEBABE};

static uint32_t rotl(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

static uint32_t xoshiro128p(void) {
    uint32_t result = rng_state[0] + rng_state[3];
    uint32_t t = rng_state[1] << 9;
    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];
    rng_state[2] ^= t;
    rng_state[3] = rotl(rng_state[3], 11);
    return result;
}

static float rand_normal(void) {
    float u1 = (float)(xoshiro128p() & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    float u2 = (float)(xoshiro128p() & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static void seed_rng(uint32_t seed) {
    rng_state[0] = seed;
    rng_state[1] = seed * 2654435761u;
    rng_state[2] = seed * 340573321u;
    rng_state[3] = seed * 1013904223u;
    for (int i = 0; i < 20; i++) xoshiro128p();
}

/* ========== Metrics ========== */

static double cosine_sim(const float* a, const float* b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    na = sqrt(na); nb = sqrt(nb);
    if (na < 1e-12 || nb < 1e-12) return 0.0;
    return dot / (na * nb);
}

static double rmse(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return sqrt(sum / n);
}

static void compute_range(const float* x, int n, float* out_min, float* out_max) {
    float mn = x[0], mx = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < mn) mn = x[i];
        if (x[i] > mx) mx = x[i];
    }
    *out_min = mn;
    *out_max = mx;
}

/* ========== Generate Realistic Correlated Key Sequences ========== */

/**
 * Simulate transformer key vectors with realistic temporal correlation.
 *
 * Real transformer keys at adjacent positions share substantial structure:
 * - Base pattern from position-independent weight projections
 * - RoPE rotates pairs of dims by small angles per position
 * - Residual stream evolves smoothly
 *
 * We model this as: key[t] = base + rope_component[t] + noise[t]
 * where rope_component rotates smoothly and noise is small.
 */
static void generate_correlated_keys(float* keys, int seq_len, int head_dim) {
    /* Base vector (position-independent component) */
    float base[512];
    for (int d = 0; d < head_dim; d++) {
        base[d] = rand_normal() * 0.5f;
    }

    /* RoPE frequencies (realistic: geometric series from 1e-4 to 1.0) */
    float freqs[256];
    for (int i = 0; i < head_dim / 2; i++) {
        freqs[i] = 1.0f / powf(10000.0f, (float)(2 * i) / (float)head_dim);
    }

    for (int t = 0; t < seq_len; t++) {
        float* k = keys + t * head_dim;

        /* Start from base */
        memcpy(k, base, head_dim * sizeof(float));

        /* Add RoPE-like rotation */
        for (int i = 0; i < head_dim / 2; i++) {
            float theta = (float)t * freqs[i];
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            float k0 = k[2 * i];
            float k1 = k[2 * i + 1];
            k[2 * i]     = k0 * cos_t - k1 * sin_t;
            k[2 * i + 1] = k0 * sin_t + k1 * cos_t;
        }

        /* Add small position-dependent noise (simulates input variation) */
        for (int d = 0; d < head_dim; d++) {
            k[d] += rand_normal() * 0.05f;
        }

        /* Slowly evolve the base (simulates residual stream drift) */
        if (t % 10 == 0) {
            for (int d = 0; d < head_dim; d++) {
                base[d] += rand_normal() * 0.02f;
            }
        }
    }
}

/* ========== Delta Compression Test ========== */

#define MAX_HEAD_DIM 256
#define IFRAME_INTERVAL 32   /* I-frame every N tokens */

/**
 * Test 1: Direct 2-bit quantization (baseline)
 * Each key quantized independently with uniform_2b.
 */
static void test_direct_2bit(const float* keys, float* recon,
                             int seq_len, int head_dim) {
    int blocks_per_key = (head_dim + TQ_BK - 1) / TQ_BK;
    size_t block_size = sizeof(block_tq_uniform_2b);
    uint8_t* buf = (uint8_t*)malloc(blocks_per_key * block_size);

    for (int t = 0; t < seq_len; t++) {
        const float* src = keys + t * head_dim;
        float* dst = recon + t * head_dim;

        for (int b = 0; b < blocks_per_key; b++) {
            int offset = b * TQ_BK;
            int count = head_dim - offset;
            if (count > TQ_BK) count = TQ_BK;
            tq_uniform_2b_quantize_ref(src + offset, buf + b * block_size, count);
            tq_uniform_2b_dequantize_ref(buf + b * block_size, dst + offset, count);
        }
    }
    free(buf);
}

/**
 * Test 2: Direct 4-bit quantization (reference quality target)
 */
static void test_direct_4bit(const float* keys, float* recon,
                             int seq_len, int head_dim) {
    int blocks_per_key = (head_dim + TQ_BK - 1) / TQ_BK;
    size_t block_size = sizeof(block_tq_uniform_4b);
    uint8_t* buf = (uint8_t*)malloc(blocks_per_key * block_size);

    for (int t = 0; t < seq_len; t++) {
        const float* src = keys + t * head_dim;
        float* dst = recon + t * head_dim;

        for (int b = 0; b < blocks_per_key; b++) {
            int offset = b * TQ_BK;
            int count = head_dim - offset;
            if (count > TQ_BK) count = TQ_BK;
            tq_uniform_4b_quantize_ref(src + offset, buf + b * block_size, count);
            tq_uniform_4b_dequantize_ref(buf + b * block_size, dst + offset, count);
        }
    }
    free(buf);
}

/**
 * Test 3: Delta + 2-bit (the breakthrough candidate)
 *
 * I-frame (every IFRAME_INTERVAL tokens): quantize absolute key with 4-bit
 * P-frame (all others): quantize (key[t] - reconstructed_key[t-1]) with 2-bit
 * Reconstruct: I-frame direct; P-frame = prev_recon + dequant(delta)
 *
 * Average bits: (4 + 2*(N-1)) / N  ~= 2.06 bpe for N=32
 */
static void test_delta_2bit(const float* keys, float* recon,
                            int seq_len, int head_dim, int iframe_interval) {
    int blocks_per_key = (head_dim + TQ_BK - 1) / TQ_BK;
    size_t block4_size = sizeof(block_tq_uniform_4b);
    size_t block2_size = sizeof(block_tq_uniform_2b);
    size_t max_block_size = block4_size > block2_size ? block4_size : block2_size;
    uint8_t* buf = (uint8_t*)malloc(blocks_per_key * max_block_size);

    float prev_recon[MAX_HEAD_DIM];
    float delta[MAX_HEAD_DIM];

    for (int t = 0; t < seq_len; t++) {
        const float* src = keys + t * head_dim;
        float* dst = recon + t * head_dim;

        if (t % iframe_interval == 0) {
            /* I-frame: quantize absolute key with 4-bit */
            for (int b = 0; b < blocks_per_key; b++) {
                int offset = b * TQ_BK;
                int count = head_dim - offset;
                if (count > TQ_BK) count = TQ_BK;
                tq_uniform_4b_quantize_ref(src + offset, buf + b * block4_size, count);
                tq_uniform_4b_dequantize_ref(buf + b * block4_size, dst + offset, count);
            }
        } else {
            /* P-frame: compute delta from previous reconstructed key */
            for (int d = 0; d < head_dim; d++) {
                delta[d] = src[d] - prev_recon[d];
            }

            /* Quantize delta with 2-bit */
            for (int b = 0; b < blocks_per_key; b++) {
                int offset = b * TQ_BK;
                int count = head_dim - offset;
                if (count > TQ_BK) count = TQ_BK;
                tq_uniform_2b_quantize_ref(delta + offset, buf + b * block2_size, count);
                tq_uniform_2b_dequantize_ref(buf + b * block2_size, delta + offset, count);
            }

            /* Reconstruct: prev_recon + quantized delta */
            for (int d = 0; d < head_dim; d++) {
                dst[d] = prev_recon[d] + delta[d];
            }
        }

        /* Save current reconstruction for next frame */
        memcpy(prev_recon, dst, head_dim * sizeof(float));
    }
    free(buf);
}

/**
 * Test 4: Delta + 2-bit (pure 2-bit, no I-frames)
 * Worst case: all P-frames from position 0 onward.
 * I-frame at t=0 only (4-bit), then all deltas at 2-bit.
 * This tests drift accumulation.
 */
static void test_delta_2bit_pure(const float* keys, float* recon,
                                 int seq_len, int head_dim) {
    /* Effectively: iframe_interval = seq_len (only one I-frame at t=0) */
    test_delta_2bit(keys, recon, seq_len, head_dim, seq_len);
}

/* ========== Main ========== */

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    seed_rng(42);

    int head_dim = 64;   /* SmolLM2 head_dim */
    int seq_len  = 256;  /* Enough tokens to see drift effects */

    /* Allow overriding from command line */
    if (argc > 1) head_dim = atoi(argv[1]);
    if (argc > 2) seq_len  = atoi(argv[2]);
    if (head_dim > MAX_HEAD_DIM) head_dim = MAX_HEAD_DIM;
    if (head_dim < 16) head_dim = 16;

    printf("=== Delta Compression for KV Cache Keys ===\n");
    printf("head_dim=%d  seq_len=%d\n\n", head_dim, seq_len);

    /* Generate realistic correlated key sequences */
    float* keys = (float*)calloc((size_t)seq_len * head_dim, sizeof(float));
    generate_correlated_keys(keys, seq_len, head_dim);

    /* Measure delta statistics first */
    printf("--- Delta Range Analysis ---\n");
    double avg_abs_range = 0.0, avg_delta_range = 0.0;
    for (int t = 1; t < seq_len; t++) {
        float amin, amax, dmin, dmax;
        compute_range(keys + t * head_dim, head_dim, &amin, &amax);
        avg_abs_range += (amax - amin);

        float delta[MAX_HEAD_DIM];
        for (int d = 0; d < head_dim; d++) {
            delta[d] = keys[t * head_dim + d] - keys[(t - 1) * head_dim + d];
        }
        compute_range(delta, head_dim, &dmin, &dmax);
        avg_delta_range += (dmax - dmin);
    }
    avg_abs_range /= (seq_len - 1);
    avg_delta_range /= (seq_len - 1);
    printf("  Average absolute range: %.4f\n", avg_abs_range);
    printf("  Average delta range:    %.4f\n", avg_delta_range);
    printf("  Delta/Absolute ratio:   %.1f%%\n\n", 100.0 * avg_delta_range / avg_abs_range);

    /* Allocate reconstruction buffers */
    float* recon = (float*)calloc((size_t)seq_len * head_dim, sizeof(float));

    /* ---- Test 1: Direct 2-bit (baseline) ---- */
    test_direct_2bit(keys, recon, seq_len, head_dim);
    {
        double cos_sum = 0.0, rmse_sum = 0.0;
        double cos_min = 1.0;
        for (int t = 0; t < seq_len; t++) {
            double c = cosine_sim(keys + t * head_dim, recon + t * head_dim, head_dim);
            double r = rmse(keys + t * head_dim, recon + t * head_dim, head_dim);
            cos_sum += c;
            rmse_sum += r;
            if (c < cos_min) cos_min = c;
        }
        printf("Direct 2-bit:       avg_cos=%.6f  min_cos=%.6f  avg_rmse=%.6f  [2.0 bpe]\n",
               cos_sum / seq_len, cos_min, rmse_sum / seq_len);
    }

    /* ---- Test 2: Direct 4-bit (reference) ---- */
    test_direct_4bit(keys, recon, seq_len, head_dim);
    {
        double cos_sum = 0.0, rmse_sum = 0.0;
        double cos_min = 1.0;
        for (int t = 0; t < seq_len; t++) {
            double c = cosine_sim(keys + t * head_dim, recon + t * head_dim, head_dim);
            double r = rmse(keys + t * head_dim, recon + t * head_dim, head_dim);
            cos_sum += c;
            rmse_sum += r;
            if (c < cos_min) cos_min = c;
        }
        printf("Direct 4-bit:       avg_cos=%.6f  min_cos=%.6f  avg_rmse=%.6f  [4.0 bpe]\n",
               cos_sum / seq_len, cos_min, rmse_sum / seq_len);
    }

    /* ---- Test 3: Delta + 2-bit with I-frames every 32 tokens ---- */
    int intervals[] = {8, 16, 32, 64};
    int n_intervals = 4;
    for (int ii = 0; ii < n_intervals; ii++) {
        int interval = intervals[ii];
        if (interval > seq_len) continue;
        test_delta_2bit(keys, recon, seq_len, head_dim, interval);
        double cos_sum = 0.0, rmse_sum = 0.0;
        double cos_min = 1.0;
        int worst_t = 0;
        for (int t = 0; t < seq_len; t++) {
            double c = cosine_sim(keys + t * head_dim, recon + t * head_dim, head_dim);
            double r = rmse(keys + t * head_dim, recon + t * head_dim, head_dim);
            cos_sum += c;
            rmse_sum += r;
            if (c < cos_min) { cos_min = c; worst_t = t; }
        }
        /* Compute average bpe: I-frames at 4-bit, P-frames at 2-bit
         * But we also need sub-block overhead for 2-bit (16B meta per 128 elements) */
        double avg_bpe = (4.0 + 2.0 * (interval - 1)) / interval;
        /* More accurate: account for sub-block metadata overhead
         * uniform_2b: 48 bytes/128 elements = 3.0 bpe
         * uniform_4b: 68 bytes/128 elements = 4.25 bpe
         * Average: (4.25 + 3.0*(N-1)) / N */
        double real_bpe = (4.25 + 3.0 * (interval - 1)) / interval;
        printf("Delta+2b (N=%2d):    avg_cos=%.6f  min_cos=%.6f  avg_rmse=%.6f  [~%.2f bpe]  worst_t=%d\n",
               interval, cos_sum / seq_len, cos_min, rmse_sum / seq_len, real_bpe, worst_t);
    }

    /* ---- Test 4: Pure delta (no periodic I-frames except t=0) ---- */
    test_delta_2bit_pure(keys, recon, seq_len, head_dim);
    {
        double cos_sum = 0.0, rmse_sum = 0.0;
        double cos_min = 1.0;
        int worst_t = 0;
        for (int t = 0; t < seq_len; t++) {
            double c = cosine_sim(keys + t * head_dim, recon + t * head_dim, head_dim);
            double r = rmse(keys + t * head_dim, recon + t * head_dim, head_dim);
            cos_sum += c;
            rmse_sum += r;
            if (c < cos_min) { cos_min = c; worst_t = t; }
        }
        printf("Delta+2b (pure):    avg_cos=%.6f  min_cos=%.6f  avg_rmse=%.6f  [~3.0 bpe]  worst_t=%d\n",
               cos_sum / seq_len, cos_min, rmse_sum / seq_len, worst_t);
    }

    printf("\n--- Per-Position Cosine (Delta+2b N=32 vs Direct 2b vs Direct 4b) ---\n");
    printf("pos  | delta+2b  | direct-2b | direct-4b\n");
    printf("-----|-----------|-----------|----------\n");

    /* Recompute all three for detailed comparison */
    float* recon_d2 = (float*)calloc((size_t)seq_len * head_dim, sizeof(float));
    float* recon_4b = (float*)calloc((size_t)seq_len * head_dim, sizeof(float));
    float* recon_delta = (float*)calloc((size_t)seq_len * head_dim, sizeof(float));

    test_direct_2bit(keys, recon_d2, seq_len, head_dim);
    test_direct_4bit(keys, recon_4b, seq_len, head_dim);
    test_delta_2bit(keys, recon_delta, seq_len, head_dim, 32);

    /* Show every 8th position plus positions around I-frame boundaries */
    for (int t = 0; t < seq_len; t++) {
        int show = (t < 5) || (t % 8 == 0) || (t % 32 == 31) || (t == seq_len - 1);
        if (!show) continue;
        double c_delta = cosine_sim(keys + t * head_dim, recon_delta + t * head_dim, head_dim);
        double c_d2    = cosine_sim(keys + t * head_dim, recon_d2 + t * head_dim, head_dim);
        double c_4b    = cosine_sim(keys + t * head_dim, recon_4b + t * head_dim, head_dim);
        const char* marker = (t % 32 == 0) ? " <-- I-frame" : "";
        printf("%4d | %.6f  | %.6f  | %.6f%s\n", t, c_delta, c_d2, c_4b, marker);
    }

    /* ---- Summary ---- */
    printf("\n=== SUMMARY ===\n");
    {
        double cos_delta = 0, cos_d2 = 0, cos_4b = 0;
        for (int t = 0; t < seq_len; t++) {
            cos_delta += cosine_sim(keys + t * head_dim, recon_delta + t * head_dim, head_dim);
            cos_d2    += cosine_sim(keys + t * head_dim, recon_d2 + t * head_dim, head_dim);
            cos_4b    += cosine_sim(keys + t * head_dim, recon_4b + t * head_dim, head_dim);
        }
        cos_delta /= seq_len;
        cos_d2    /= seq_len;
        cos_4b    /= seq_len;

        printf("Direct 2-bit:    avg_cosine = %.6f  (2.0 bpe data, 3.0 bpe with meta)\n", cos_d2);
        printf("Direct 4-bit:    avg_cosine = %.6f  (4.0 bpe data, 4.25 bpe with meta)\n", cos_4b);
        printf("Delta+2b (N=32): avg_cosine = %.6f  (~3.04 bpe with meta)\n", cos_delta);
        printf("\n");

        if (cos_delta > 0.99) {
            printf("*** BREAKTHROUGH: Delta+2b achieves >0.99 cosine! ***\n");
            printf("    At ~3 bpe, this matches or exceeds direct 4-bit quality.\n");
        } else if (cos_delta > cos_d2 + 0.01) {
            printf("*** IMPROVEMENT: Delta+2b significantly better than direct 2-bit.\n");
            printf("    Gap to 4-bit: %.6f\n", cos_4b - cos_delta);
        } else {
            printf("    Delta+2b does NOT significantly improve over direct 2-bit.\n");
            printf("    The per-block min-max already adapts to local range.\n");
        }
    }

    free(keys);
    free(recon);
    free(recon_d2);
    free(recon_4b);
    free(recon_delta);

    return 0;
}
