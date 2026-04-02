/**
 * Multi-hash dequant quality benchmark
 *
 * Tests the DEQUANTIZATION quality of multi-hash sign quantization
 * vs uniform quantization at equivalent bit budgets.
 *
 * Multi-hash dequant: for each hash h, reconstruct ±norm/sqrt(dim) vector
 * from sign bits via inverse permute+RHT, then average K reconstructions.
 *
 * Hypothesis: sign-based multi-hash preserves DIRECTION (cosine) better
 * than uniform quantization at same total bits, because sign hashing is
 * optimized for inner products while uniform is optimized for MSE.
 *
 * Build:
 *   cc -O2 -I include bench/test_multihash_dequant.c build/libturboquant.a -lm -o build/test_multihash_dequant
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "turboquant/turboquant.h"

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

/* ========== Deterministic permutation (Fisher-Yates) ========== */

static void seeded_permutation(int* perm, int n, uint32_t seed) {
    for (int i = 0; i < n; i++) perm[i] = i;
    uint32_t h = seed;
    for (int i = n - 1; i > 0; i--) {
        h = h * 2654435761u + 1;
        int j = (int)(h % (uint32_t)(i + 1));
        int tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
}

/* ========== Helpers ========== */

static float dot_product(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)a[i] * (double)b[i];
    return (float)sum;
}

static float l2_norm(const float* x, int n) {
    return sqrtf(dot_product(x, x, n));
}

static float cosine_sim(const float* a, const float* b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    na = sqrt(na); nb = sqrt(nb);
    if (na < 1e-12 || nb < 1e-12) return 0.0f;
    return (float)(dot / (na * nb));
}

static float mse(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return (float)(sum / n);
}

/* ========== Multi-hash sign quantize + dequantize (Permute+RHT) ========== */

/**
 * Quantize: for each hash h:
 *   1. Permute vector dims with seed-specific permutation
 *   2. Apply RHT with fixed seed
 *   3. Extract sign bits
 *   4. Store: norm (fp32 for prototype), sign bits
 *
 * Dequantize: for each hash h:
 *   1. Reconstruct ±1 vector from sign bits
 *   2. Scale by norm / sqrt(dim)
 *   3. Apply inverse RHT with fixed seed
 *   4. Inverse permute
 * Average K reconstructions.
 */

#define MAX_DIM 512
#define MAX_HASHES 16

typedef struct {
    float norm;
    uint32_t perm_seed;
    uint8_t signs[MAX_DIM / 8]; /* 1 bit per dim */
} hash_block_t;

static void multihash_quantize(const float* src, int dim, int n_hashes,
                                hash_block_t* blocks) {
    float norm = l2_norm(src, dim);
    int perm[MAX_DIM];
    float tmp[MAX_DIM];

    for (int h = 0; h < n_hashes; h++) {
        uint32_t perm_seed = 0x12345678u + (uint32_t)h * 0x9ABCDEF0u;
        blocks[h].norm = norm;
        blocks[h].perm_seed = perm_seed;

        /* Permute */
        seeded_permutation(perm, dim, perm_seed);
        for (int i = 0; i < dim; i++) tmp[i] = src[perm[i]];

        /* RHT with fixed seed */
        tq_rht_transform(tmp, dim, 0xDEADBEEFu);

        /* Extract signs */
        memset(blocks[h].signs, 0, (dim + 7) / 8);
        for (int i = 0; i < dim; i++) {
            if (tmp[i] >= 0.0f)
                blocks[h].signs[i / 8] |= (1u << (i % 8));
        }
    }
}

static void multihash_dequantize(const hash_block_t* blocks, int dim,
                                  int n_hashes, float* dst) {
    int perm[MAX_DIM];
    int inv_perm[MAX_DIM];
    float tmp[MAX_DIM];
    float recon[MAX_DIM];

    memset(dst, 0, dim * sizeof(float));

    float scale = 1.0f / sqrtf((float)dim);

    for (int h = 0; h < n_hashes; h++) {
        float norm = blocks[h].norm;

        /* Reconstruct ±1 vector from signs */
        for (int i = 0; i < dim; i++) {
            int bit = (blocks[h].signs[i / 8] >> (i % 8)) & 1;
            tmp[i] = bit ? (norm * scale) : (-norm * scale);
        }

        /* Inverse RHT */
        tq_rht_inverse(tmp, dim, 0xDEADBEEFu);

        /* Inverse permute */
        seeded_permutation(perm, dim, blocks[h].perm_seed);
        for (int i = 0; i < dim; i++) inv_perm[perm[i]] = i;
        for (int i = 0; i < dim; i++) recon[i] = tmp[inv_perm[i]];

        /* Accumulate */
        for (int i = 0; i < dim; i++) dst[i] += recon[i];
    }

    /* Average */
    float inv_k = 1.0f / (float)n_hashes;
    for (int i = 0; i < dim; i++) dst[i] *= inv_k;
}

/* ========== FP16 helpers (for uniform quant) ========== */

static uint16_t fp32_to_fp16(float v) {
    union { float f; uint32_t u; } bits;
    bits.f = v;
    uint32_t sign = (bits.u >> 16) & 0x8000;
    int32_t  exp  = ((bits.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits.u >> 13) & 0x03FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static float fp16_to_fp32(uint16_t h) {
    union { float f; uint32_t u; } bits;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0) { bits.u = sign; return bits.f; }
    if (exp == 31) { bits.u = sign | 0x7F800000 | (mant << 13); return bits.f; }
    exp = exp - 15 + 127;
    bits.u = sign | (exp << 23) | (mant << 13);
    return bits.f;
}

/* ========== Manual uniform dequant (standalone, any bits) ========== */

static void uniform_quantize_dequantize(const float* src, float* dst, int dim, int bits) {
    int levels = 1 << bits;
    float mn = src[0], mx = src[0];
    for (int i = 1; i < dim; i++) {
        if (src[i] < mn) mn = src[i];
        if (src[i] > mx) mx = src[i];
    }
    float range = mx - mn;
    if (range < 1e-8f) range = 1e-8f;
    float scale = range / (float)levels;

    /* Store scale/min as fp16 (realistic overhead) */
    float s16 = fp16_to_fp32(fp32_to_fp16(scale));
    float m16 = fp16_to_fp32(fp32_to_fp16(mn));

    for (int i = 0; i < dim; i++) {
        int q = (int)floorf((src[i] - m16) / s16);
        if (q < 0) q = 0;
        if (q >= levels) q = levels - 1;
        dst[i] = m16 + ((float)q + 0.5f) * s16;
    }
}

/* Uniform with RHT preprocessing (better quality) */
static void uniform_rht_quantize_dequantize(const float* src, float* dst, int dim,
                                              int bits, uint32_t rht_seed) {
    float tmp[MAX_DIM];
    memcpy(tmp, src, dim * sizeof(float));

    /* Forward RHT */
    tq_rht_transform(tmp, dim, rht_seed);

    /* Uniform quantize+dequant in RHT domain */
    float deq[MAX_DIM];
    uniform_quantize_dequantize(tmp, deq, dim, bits);

    /* Inverse RHT */
    memcpy(dst, deq, dim * sizeof(float));
    tq_rht_inverse(dst, dim, rht_seed);
}

/* ========== Uniform 3-bit with sub-block scales (matching library impl) ========== */

static void uniform_3b_subblock_qd(const float* src, float* dst, int dim) {
    /* 4 sub-blocks of 32 elements each (for dim=128) */
    int nsub = 4;
    int subk = dim / nsub;
    if (subk < 1) subk = dim; /* fallback for tiny dims */
    if (subk < 1) { memcpy(dst, src, dim * sizeof(float)); return; }

    for (int sb = 0; sb < nsub && sb * subk < dim; sb++) {
        int start = sb * subk;
        int end = start + subk;
        if (end > dim) end = dim;
        int cnt = end - start;

        float mn = src[start], mx = src[start];
        for (int i = start + 1; i < end; i++) {
            if (src[i] < mn) mn = src[i];
            if (src[i] > mx) mx = src[i];
        }
        float range = mx - mn;
        if (range < 1e-8f) range = 1e-8f;
        float scale = range / 8.0f; /* 3-bit: 8 levels */

        float s16 = fp16_to_fp32(fp32_to_fp16(scale));
        float m16 = fp16_to_fp32(fp32_to_fp16(mn));

        for (int i = start; i < end; i++) {
            int q = (int)floorf((src[i] - m16) / s16);
            if (q < 0) q = 0;
            if (q > 7) q = 7;
            dst[i] = m16 + ((float)q + 0.5f) * s16;
        }
    }
}

/* ========== Main ========== */

int main(void) {
    const int N_TRIALS = 10000;

    printf("================================================================\n");
    printf("  Multi-Hash Dequant Quality Benchmark\n");
    printf("  N_TRIALS = %d\n", N_TRIALS);
    printf("  Tests VECTOR RECONSTRUCTION quality (cosine + MSE)\n");
    printf("================================================================\n\n");

    int dims[] = {64, 128};
    int n_dims = 2;

    for (int di = 0; di < n_dims; di++) {
        int dim = dims[di];

        printf("================================================================\n");
        printf("  dim = %d\n", dim);
        printf("================================================================\n\n");

        /* Accumulators */
        double mh_cos[MAX_HASHES+1];  /* index = n_hashes */
        double mh_mse_acc[MAX_HASHES+1];
        double uni_cos[5];   /* index 0=2b, 1=3b, 2=4b, 3=3b-subblock, 4=unused */
        double uni_mse_acc[5];
        double uni_rht_cos[5]; /* uniform with RHT preprocessing */
        double uni_rht_mse_acc[5];

        memset(mh_cos, 0, sizeof(mh_cos));
        memset(mh_mse_acc, 0, sizeof(mh_mse_acc));
        memset(uni_cos, 0, sizeof(uni_cos));
        memset(uni_mse_acc, 0, sizeof(uni_mse_acc));
        memset(uni_rht_cos, 0, sizeof(uni_rht_cos));
        memset(uni_rht_mse_acc, 0, sizeof(uni_rht_mse_acc));

        int hash_counts[] = {1, 2, 3, 4, 8};
        int n_hash_counts = 5;

        seed_rng(42 + dim);

        for (int t = 0; t < N_TRIALS; t++) {
            float src[MAX_DIM];
            float dst[MAX_DIM];
            hash_block_t blocks[MAX_HASHES];

            /* Generate random vector */
            for (int d = 0; d < dim; d++) src[d] = rand_normal();

            /* --- Multi-hash sign dequant --- */
            for (int hi = 0; hi < n_hash_counts; hi++) {
                int K = hash_counts[hi];
                multihash_quantize(src, dim, K, blocks);
                multihash_dequantize(blocks, dim, K, dst);
                mh_cos[K] += cosine_sim(src, dst, dim);
                mh_mse_acc[K] += mse(src, dst, dim);
            }

            /* --- Uniform dequant (plain) --- */
            /* 2-bit */
            uniform_quantize_dequantize(src, dst, dim, 2);
            uni_cos[0] += cosine_sim(src, dst, dim);
            uni_mse_acc[0] += mse(src, dst, dim);

            /* 3-bit (flat, single scale) */
            uniform_quantize_dequantize(src, dst, dim, 3);
            uni_cos[1] += cosine_sim(src, dst, dim);
            uni_mse_acc[1] += mse(src, dst, dim);

            /* 4-bit */
            uniform_quantize_dequantize(src, dst, dim, 4);
            uni_cos[2] += cosine_sim(src, dst, dim);
            uni_mse_acc[2] += mse(src, dst, dim);

            /* 3-bit sub-block (Q3_K style) */
            uniform_3b_subblock_qd(src, dst, dim);
            uni_cos[3] += cosine_sim(src, dst, dim);
            uni_mse_acc[3] += mse(src, dst, dim);

            /* --- Uniform with RHT preprocessing --- */
            /* 2-bit + RHT */
            uniform_rht_quantize_dequantize(src, dst, dim, 2, 0xAABBCCDD);
            uni_rht_cos[0] += cosine_sim(src, dst, dim);
            uni_rht_mse_acc[0] += mse(src, dst, dim);

            /* 3-bit + RHT */
            uniform_rht_quantize_dequantize(src, dst, dim, 3, 0xAABBCCDD);
            uni_rht_cos[1] += cosine_sim(src, dst, dim);
            uni_rht_mse_acc[1] += mse(src, dst, dim);

            /* 4-bit + RHT */
            uniform_rht_quantize_dequantize(src, dst, dim, 4, 0xAABBCCDD);
            uni_rht_cos[2] += cosine_sim(src, dst, dim);
            uni_rht_mse_acc[2] += mse(src, dst, dim);
        }

        /* Print results */
        printf("  %-35s  %10s  %10s  %8s\n",
               "Method", "cosine", "MSE", "bpe*");
        printf("  %-35s  %10s  %10s  %8s\n",
               "-----------------------------------", "------", "---", "---");

        for (int hi = 0; hi < n_hash_counts; hi++) {
            int K = hash_counts[hi];
            /* bpe for multi-hash: K * (norm_fp16 + perm_seed_u32 + dim/8 sign bytes)
             * = K * (2 + 4 + dim/8) bytes per dim elements
             * = K * (2 + 4 + dim/8) * 8 / dim bits per element
             * But for fair comparison: the sign bits are 1 bpe, plus metadata overhead.
             * Total bits = K * (dim + 48) per block.
             * bpe = K * (dim + 48) / dim = K * (1 + 48/dim)
             * For dim=64: K * 1.75;  For dim=128: K * 1.375 */
            float bpe = (float)K * (1.0f + 48.0f / (float)dim);
            char label[64];
            snprintf(label, sizeof(label), "multi-hash sign (K=%d)", K);
            printf("  %-35s  %10.6f  %10.6f  %7.2f\n",
                   label,
                   mh_cos[K] / N_TRIALS,
                   mh_mse_acc[K] / N_TRIALS,
                   bpe);
        }

        printf("\n");

        const char* uni_names[] = {"uniform_2b", "uniform_3b (flat)",
                                    "uniform_4b", "uniform_3b (sub-block)"};
        float uni_bpe[] = {2.25f, 3.25f, 4.25f, 4.0f}; /* including fp16 scale+zp overhead */
        /* For dim=128: uniform_2b = (4 meta + 32 data) * 8 / 128 = 2.25 bpe
         * uniform_3b flat = (4 meta + 48 data) * 8 / 128 = 3.25 bpe
         * uniform_4b = (4 meta + 64 data) * 8 / 128 = 4.25 bpe
         * uniform_3b sub = (16 meta + 48 data) * 8 / 128 = 4.0 bpe */

        for (int u = 0; u < 4; u++) {
            printf("  %-35s  %10.6f  %10.6f  %7.2f\n",
                   uni_names[u],
                   uni_cos[u] / N_TRIALS,
                   uni_mse_acc[u] / N_TRIALS,
                   uni_bpe[u]);
        }

        printf("\n");

        const char* rht_names[] = {"uniform_2b + RHT", "uniform_3b + RHT",
                                    "uniform_4b + RHT"};
        float rht_bpe[] = {2.5f, 3.5f, 4.5f}; /* +4 byte seed overhead */

        for (int u = 0; u < 3; u++) {
            printf("  %-35s  %10.6f  %10.6f  %7.2f\n",
                   rht_names[u],
                   uni_rht_cos[u] / N_TRIALS,
                   uni_rht_mse_acc[u] / N_TRIALS,
                   rht_bpe[u]);
        }

        printf("\n");

        /* ---- Attention score comparison ---- */
        printf("  --- Attention Score Estimation (Q dot K_dequant) ---\n");
        printf("  %-35s  %10s  %10s\n", "Method", "attn_cos", "attn_MSE");
        printf("  %-35s  %10s  %10s\n",
               "-----------------------------------", "------", "---");

        /* For each trial: generate Q, K. Compute true Q.K.
         * Dequant K with each method. Compute Q.dequant(K). Compare. */
        double attn_mh_cos[MAX_HASHES+1];
        double attn_mh_mse[MAX_HASHES+1];
        double attn_uni_cos[5], attn_uni_mse[5];
        double attn_rht_cos[5], attn_rht_mse[5];
        memset(attn_mh_cos, 0, sizeof(attn_mh_cos));
        memset(attn_mh_mse, 0, sizeof(attn_mh_mse));
        memset(attn_uni_cos, 0, sizeof(attn_uni_cos));
        memset(attn_uni_mse, 0, sizeof(attn_uni_mse));
        memset(attn_rht_cos, 0, sizeof(attn_rht_cos));
        memset(attn_rht_mse, 0, sizeof(attn_rht_mse));

        seed_rng(999 + dim);
        for (int t = 0; t < N_TRIALS; t++) {
            float query[MAX_DIM], key[MAX_DIM], dst[MAX_DIM];
            hash_block_t blocks[MAX_HASHES];

            for (int d = 0; d < dim; d++) {
                query[d] = rand_normal();
                key[d] = rand_normal();
            }
            float true_dot = dot_product(query, key, dim);

            /* Multi-hash */
            for (int hi = 0; hi < n_hash_counts; hi++) {
                int K = hash_counts[hi];
                multihash_quantize(key, dim, K, blocks);
                multihash_dequantize(blocks, dim, K, dst);
                float est_dot = dot_product(query, dst, dim);
                /* Treat each (true, est) pair as a 1D "vector" for cosine-like stat */
                double err = (double)est_dot - (double)true_dot;
                attn_mh_mse[K] += err * err;
            }

            /* Uniform */
            {
                uniform_quantize_dequantize(key, dst, dim, 2);
                double err = (double)dot_product(query, dst, dim) - (double)true_dot;
                attn_uni_mse[0] += err * err;
            }
            {
                uniform_quantize_dequantize(key, dst, dim, 3);
                double err = (double)dot_product(query, dst, dim) - (double)true_dot;
                attn_uni_mse[1] += err * err;
            }
            {
                uniform_quantize_dequantize(key, dst, dim, 4);
                double err = (double)dot_product(query, dst, dim) - (double)true_dot;
                attn_uni_mse[2] += err * err;
            }
            {
                uniform_3b_subblock_qd(key, dst, dim);
                double err = (double)dot_product(query, dst, dim) - (double)true_dot;
                attn_uni_mse[3] += err * err;
            }

            /* Uniform + RHT */
            {
                uniform_rht_quantize_dequantize(key, dst, dim, 2, 0xAABBCCDD);
                double err = (double)dot_product(query, dst, dim) - (double)true_dot;
                attn_rht_mse[0] += err * err;
            }
            {
                uniform_rht_quantize_dequantize(key, dst, dim, 3, 0xAABBCCDD);
                double err = (double)dot_product(query, dst, dim) - (double)true_dot;
                attn_rht_mse[1] += err * err;
            }
            {
                uniform_rht_quantize_dequantize(key, dst, dim, 4, 0xAABBCCDD);
                double err = (double)dot_product(query, dst, dim) - (double)true_dot;
                attn_rht_mse[2] += err * err;
            }
        }

        /* Compute attention cosine: over N_TRIALS pairs of (true_dot, est_dot) */
        /* Re-run to collect both true and est vectors for cosine computation */
        float* true_dots = (float*)malloc(N_TRIALS * sizeof(float));
        float* est_dots  = (float*)malloc(N_TRIALS * sizeof(float));

        for (int hi = 0; hi < n_hash_counts; hi++) {
            int K = hash_counts[hi];
            seed_rng(999 + dim);
            for (int t = 0; t < N_TRIALS; t++) {
                float query[MAX_DIM], key[MAX_DIM], dst[MAX_DIM];
                hash_block_t blocks[MAX_HASHES];
                for (int d = 0; d < dim; d++) { query[d] = rand_normal(); key[d] = rand_normal(); }
                true_dots[t] = dot_product(query, key, dim);
                multihash_quantize(key, dim, K, blocks);
                multihash_dequantize(blocks, dim, K, dst);
                est_dots[t] = dot_product(query, dst, dim);
            }
            float bpe = (float)K * (1.0f + 48.0f / (float)dim);
            char label[64];
            snprintf(label, sizeof(label), "multi-hash sign (K=%d)", K);
            printf("  %-35s  %10.6f  %10.4f  %7.2f bpe\n",
                   label,
                   cosine_sim(true_dots, est_dots, N_TRIALS),
                   attn_mh_mse[K] / N_TRIALS,
                   bpe);
        }
        printf("\n");

        /* Uniform attention */
        const char* all_uni_names[] = {"uniform_2b", "uniform_3b (flat)",
                                        "uniform_4b", "uniform_3b (sub-block)",
                                        "uniform_2b + RHT", "uniform_3b + RHT",
                                        "uniform_4b + RHT"};
        float all_uni_bpe[] = {2.25f, 3.25f, 4.25f, 4.0f, 2.5f, 3.5f, 4.5f};

        for (int u = 0; u < 7; u++) {
            seed_rng(999 + dim);
            for (int t = 0; t < N_TRIALS; t++) {
                float query[MAX_DIM], key[MAX_DIM], dst[MAX_DIM];
                for (int d = 0; d < dim; d++) { query[d] = rand_normal(); key[d] = rand_normal(); }
                true_dots[t] = dot_product(query, key, dim);

                if (u == 0) uniform_quantize_dequantize(key, dst, dim, 2);
                else if (u == 1) uniform_quantize_dequantize(key, dst, dim, 3);
                else if (u == 2) uniform_quantize_dequantize(key, dst, dim, 4);
                else if (u == 3) uniform_3b_subblock_qd(key, dst, dim);
                else if (u == 4) uniform_rht_quantize_dequantize(key, dst, dim, 2, 0xAABBCCDD);
                else if (u == 5) uniform_rht_quantize_dequantize(key, dst, dim, 3, 0xAABBCCDD);
                else uniform_rht_quantize_dequantize(key, dst, dim, 4, 0xAABBCCDD);

                est_dots[t] = dot_product(query, dst, dim);
            }
            double amse = 0;
            if (u < 4) amse = attn_uni_mse[u];
            else amse = attn_rht_mse[u - 4];
            printf("  %-35s  %10.6f  %10.4f  %7.2f bpe\n",
                   all_uni_names[u],
                   cosine_sim(true_dots, est_dots, N_TRIALS),
                   amse / N_TRIALS,
                   all_uni_bpe[u]);
        }

        free(true_dots);
        free(est_dots);
        printf("\n");
    }

    /* ========== Summary ========== */
    printf("================================================================\n");
    printf("  INTERPRETATION GUIDE\n");
    printf("================================================================\n");
    printf("\n");
    printf("  bpe* = bits per element INCLUDING metadata overhead\n");
    printf("    multi-hash K=2, dim=64:  2 * (1 + 48/64) = 3.50 bpe\n");
    printf("    multi-hash K=2, dim=128: 2 * (1 + 48/128) = 2.75 bpe\n");
    printf("    multi-hash K=3, dim=128: 3 * (1 + 48/128) = 4.13 bpe\n");
    printf("    uniform_2b, dim=128:     (4+32)*8/128 = 2.25 bpe\n");
    printf("    uniform_3b flat:         (4+48)*8/128 = 3.25 bpe\n");
    printf("    uniform_4b:              (4+64)*8/128 = 4.25 bpe\n");
    printf("\n");
    printf("  KEY QUESTION: At similar bpe, does multi-hash sign beat uniform?\n");
    printf("  Compare:\n");
    printf("    multi-hash K=2 (~2.75 bpe) vs uniform_2b (2.25) / uniform_3b (3.25)\n");
    printf("    multi-hash K=3 (~4.13 bpe) vs uniform_4b (4.25)\n");
    printf("  Look at BOTH cosine (direction) AND attn_cos (dot product preservation)\n");
    printf("\n");

    return 0;
}
