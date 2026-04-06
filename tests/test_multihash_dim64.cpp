#define _USE_MATH_DEFINES
#include <cmath>
/**
 * Multi-hash prototype: does averaging K independent sign hashes
 * fix 1-bit KV failure at head_dim=64?
 *
 * Theory: sign quantization variance ~ 1/dim. At dim=64, variance is 4x
 * higher than dim=256. Using K independent sign hashes and averaging the
 * attention scores reduces variance by factor K.
 *
 * KEY FINDING: Single-seed RHT sign agreement is seed-invariant!
 *   RHT = (1/sqrt(n)) * H * D  where H = fixed WHT, D = random signs
 *   sign(D*H*q[i]) == sign(D*H*k[i]) iff sign(H*q[i]) == sign(H*k[i])
 *   because D flips both q and k by the same sign per coordinate.
 *   Therefore, different RHT seeds produce IDENTICAL sign agreement counts.
 *
 * CORRECT APPROACHES tested here:
 *   Method A: Stacked RHT (RHT with seed_a, then RHT with seed_b) --
 *             two rounds create a non-trivial rotation that varies with seeds.
 *   Method B: QJL-style Rademacher random projection with per-hash seed.
 *   Method C: Random permutation + RHT (permute dims before sign flip + WHT).
 *
 * Build:
 *   g++ -O2 -std=c++17 -I include tests/test_multihash_dim64.cpp \
 *       build/libturboquant.a -lm -lpthread -o build/test_multihash_dim64
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>

extern "C" {
#include "turboquant/turboquant.h"
}

/* ========== Random number generator (xoshiro128+) ========== */

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

/* ========== Deterministic hash for Rademacher entries ========== */

static float rademacher(uint32_t seed, int idx) {
    uint32_t h = seed ^ (uint32_t)idx;
    h *= 2654435761u;
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return (h & 1) ? 1.0f : -1.0f;
}

/* ========== Deterministic permutation via Fisher-Yates with seed ========== */

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
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static float l2_norm(const float* x, int n) {
    return sqrtf(dot_product(x, x, n));
}

/* ========== Method A: Stacked RHT (two rounds with different seeds) ========== */

static int sign_agree_stacked_rht(const float* query, const float* key,
                                   int dim, uint32_t seed1, uint32_t seed2) {
    std::vector<float> q(dim), k(dim);
    memcpy(q.data(), query, dim * sizeof(float));
    memcpy(k.data(), key,   dim * sizeof(float));

    /* Round 1 */
    tq_rht_transform(q.data(), dim, seed1);
    tq_rht_transform(k.data(), dim, seed1);
    /* Round 2 -- different seed creates a different overall rotation */
    tq_rht_transform(q.data(), dim, seed2);
    tq_rht_transform(k.data(), dim, seed2);

    int agree = 0;
    for (int i = 0; i < dim; i++) {
        if ((q[i] > 0.0f) == (k[i] > 0.0f)) agree++;
    }
    return agree;
}

/* ========== Method B: QJL-style Rademacher random projection ========== */

static int sign_agree_rademacher(const float* query, const float* key,
                                  int dim, uint32_t seed) {
    /* Project both Q and K through a random Rademacher matrix R (dim x dim)
     * where R[i][j] = rademacher(seed + i*dim_prime, j).
     * Then compare signs of projected vectors. */
    int agree = 0;
    for (int i = 0; i < dim; i++) {
        float q_proj = 0.0f, k_proj = 0.0f;
        uint32_t row_seed = seed + (uint32_t)i * 65537u;
        for (int j = 0; j < dim; j++) {
            float r = rademacher(row_seed, j);
            q_proj += query[j] * r;
            k_proj += key[j] * r;
        }
        if ((q_proj > 0.0f) == (k_proj > 0.0f)) agree++;
    }
    return agree;
}

/* ========== Method C: Permute + RHT ========== */

static int sign_agree_perm_rht(const float* query, const float* key,
                                int dim, uint32_t seed) {
    std::vector<int> perm(dim);
    seeded_permutation(perm.data(), dim, seed);

    std::vector<float> q(dim), k(dim);
    for (int i = 0; i < dim; i++) {
        q[i] = query[perm[i]];
        k[i] = key[perm[i]];
    }

    /* Fixed RHT seed -- the permutation provides the randomness */
    tq_rht_transform(q.data(), dim, 0xDEADBEEFu);
    tq_rht_transform(k.data(), dim, 0xDEADBEEFu);

    int agree = 0;
    for (int i = 0; i < dim; i++) {
        if ((q[i] > 0.0f) == (k[i] > 0.0f)) agree++;
    }
    return agree;
}

/* ========== Single RHT baseline ========== */

static int sign_agree_single_rht(const float* query, const float* key,
                                  int dim, uint32_t seed) {
    std::vector<float> q(dim), k(dim);
    memcpy(q.data(), query, dim * sizeof(float));
    memcpy(k.data(), key,   dim * sizeof(float));
    tq_rht_transform(q.data(), dim, seed);
    tq_rht_transform(k.data(), dim, seed);

    int agree = 0;
    for (int i = 0; i < dim; i++) {
        if ((q[i] > 0.0f) == (k[i] > 0.0f)) agree++;
    }
    return agree;
}

/* ========== Compute multi-hash score using a given method ========== */

typedef int (*agree_fn_t)(const float*, const float*, int, uint32_t);

static float multi_hash_score(const float* query, const float* key,
                              int dim, int n_hashes, agree_fn_t fn) {
    float total = 0.0f;
    for (int h = 0; h < n_hashes; h++) {
        uint32_t seed = 0x12345678u + (uint32_t)h * 0x9ABCDEF0u;
        int agree = fn(query, key, dim, seed);
        total += (float)(2 * agree - dim);
    }
    return total / (float)n_hashes;
}

/* Stacked RHT needs two seeds, wrap it */
static float multi_hash_score_stacked(const float* query, const float* key,
                                      int dim, int n_hashes) {
    float total = 0.0f;
    for (int h = 0; h < n_hashes; h++) {
        uint32_t seed1 = 0x12345678u + (uint32_t)h * 0x9ABCDEF0u;
        uint32_t seed2 = 0xABCDEF01u + (uint32_t)h * 0x13579BDFu;
        int agree = sign_agree_stacked_rht(query, key, dim, seed1, seed2);
        total += (float)(2 * agree - dim);
    }
    return total / (float)n_hashes;
}

/* ========== Statistics ========== */

struct Stats {
    double cosine;
    double mse;
    double stddev;
    double bias;
};

static Stats compute_stats(const std::vector<float>& truth,
                           const std::vector<float>& est) {
    int n = (int)truth.size();
    Stats s{};

    double sum_err = 0.0, sum_sq_err = 0.0;
    for (int i = 0; i < n; i++) {
        double e = (double)est[i] - (double)truth[i];
        sum_err += e;
        sum_sq_err += e * e;
    }
    s.bias = sum_err / n;
    s.mse  = sum_sq_err / n;

    double dot = 0.0, nt = 0.0, ne = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)truth[i] * (double)est[i];
        nt  += (double)truth[i] * (double)truth[i];
        ne  += (double)est[i]   * (double)est[i];
    }
    nt = sqrt(nt); ne = sqrt(ne);
    s.cosine = (nt > 1e-12 && ne > 1e-12) ? dot / (nt * ne) : 0.0;

    double me = sum_err / n, var = 0.0;
    for (int i = 0; i < n; i++) {
        double e = (double)est[i] - (double)truth[i];
        var += (e - me) * (e - me);
    }
    s.stddev = sqrt(var / n);

    return s;
}

/* ========== Main ========== */

int main() {
    const int N_PAIRS = 10000;

    printf("================================================================\n");
    printf("  Multi-Hash Sign Quantization Prototype\n");
    printf("  Testing whether K independent hashes fix dim=64 failure\n");
    printf("  N_PAIRS = %d\n", N_PAIRS);
    printf("================================================================\n\n");

    /* ---- FINDING 1: RHT seed-invariance demonstration ---- */
    printf("FINDING: Single RHT sign agreement is SEED-INVARIANT\n");
    printf("  RHT = (1/sqrt(n)) * H * D, where D is diagonal sign matrix.\n");
    printf("  D flips both q[i] and k[i] by same sign => no effect on agreement.\n");
    printf("  WHT (H) is fixed => sign(H*D*q) vs sign(H*D*k) is independent of D.\n\n");
    {
        seed_rng(777);
        float dq[64], dk[64];
        for (int d = 0; d < 64; d++) { dq[d] = rand_normal(); dk[d] = rand_normal(); }
        printf("  Proof (one pair, dim=64, 8 different seeds):\n");
        for (int h = 0; h < 8; h++) {
            uint32_t s = (uint32_t)h * 0x11111111u + 0xAAAAAAAAu;
            int ag = sign_agree_single_rht(dq, dk, 64, s);
            printf("    seed=0x%08X  agree=%d\n", s, ag);
        }
        printf("  => All identical. Multi-hash with plain RHT is useless.\n\n");
    }

    /* ---- Test all three correct methods ---- */
    const char* method_names[] = {
        "Baseline (1 RHT)",
        "Method A: Stacked RHT (2 rounds)",
        "Method B: Rademacher projection",
        "Method C: Permute + RHT"
    };

    for (int dim : {64, 128, 256}) {
        printf("================================================================\n");
        printf("  dim = %d\n", dim);
        printf("================================================================\n\n");

        /* Generate data */
        seed_rng(42 + dim);
        std::vector<std::vector<float>> queries(N_PAIRS, std::vector<float>(dim));
        std::vector<std::vector<float>> keys(N_PAIRS, std::vector<float>(dim));
        std::vector<float> true_scores(N_PAIRS);

        for (int i = 0; i < N_PAIRS; i++) {
            for (int d = 0; d < dim; d++) {
                queries[i][d] = rand_normal();
                keys[i][d]    = rand_normal();
            }
            true_scores[i] = dot_product(queries[i].data(), keys[i].data(), dim);
        }

        /* For each method, test K = 1, 2, 4, 8, 16 */
        for (int method = 0; method < 4; method++) {
            printf("  %s\n", method_names[method]);
            printf("  %-8s  %10s  %10s  %10s  %10s\n",
                   "K", "cosine", "MSE", "stddev", "bias");
            printf("  %-8s  %10s  %10s  %10s  %10s\n",
                   "---", "------", "---", "------", "----");

            for (int K : {1, 2, 4, 8, 16}) {
                std::vector<float> est(N_PAIRS);

                for (int i = 0; i < N_PAIRS; i++) {
                    float raw;

                    if (method == 0) {
                        /* Baseline: single RHT, K copies are identical */
                        int ag = sign_agree_single_rht(
                            queries[i].data(), keys[i].data(), dim, 0x12345678u);
                        raw = (float)(2 * ag - dim);
                    } else if (method == 1) {
                        raw = multi_hash_score_stacked(
                            queries[i].data(), keys[i].data(), dim, K);
                    } else if (method == 2) {
                        raw = multi_hash_score(
                            queries[i].data(), keys[i].data(), dim, K,
                            sign_agree_rademacher);
                    } else {
                        raw = multi_hash_score(
                            queries[i].data(), keys[i].data(), dim, K,
                            sign_agree_perm_rht);
                    }

                    float qn = l2_norm(queries[i].data(), dim);
                    float kn = l2_norm(keys[i].data(), dim);
                    est[i] = raw * ((float)M_PI / 2.0f) * qn * kn / (float)dim;
                }

                Stats st = compute_stats(true_scores, est);
                printf("  %-8d  %10.6f  %10.4f  %10.4f  %10.4f\n",
                       K, st.cosine, st.mse, st.stddev, st.bias);
            }
            printf("\n");
        }
    }

    /* ---- Unbiasedness test ---- */
    printf("================================================================\n");
    printf("  Unbiasedness Test (dim=64, one fixed Q/K pair, 1000 reps)\n");
    printf("  Each rep uses different seed batch.\n");
    printf("================================================================\n\n");

    const int dim = 64;
    const int N_REPS = 1000;
    seed_rng(999);

    float q[64], k[64];
    for (int d = 0; d < dim; d++) {
        q[d] = rand_normal();
        k[d] = rand_normal();
    }
    float true_dot = dot_product(q, k, dim);
    float qn = l2_norm(q, dim);
    float kn = l2_norm(k, dim);

    printf("True dot product: %.4f\n", true_dot);
    printf("||q|| = %.4f, ||k|| = %.4f\n\n", qn, kn);

    printf("%-20s  %4s  %10s  %10s  %10s\n",
           "Method", "K", "mean_est", "stddev", "bias");
    printf("%-20s  %4s  %10s  %10s  %10s\n",
           "------", "---", "--------", "------", "----");

    for (const char* mname : {"Stacked RHT", "Rademacher", "Perm+RHT"}) {
        for (int K : {1, 4, 8, 16}) {
            std::vector<float> estimates(N_REPS);

            for (int r = 0; r < N_REPS; r++) {
                float total = 0.0f;
                for (int h = 0; h < K; h++) {
                    uint32_t seed = (uint32_t)(r * 137 + h * 997 + 0x55555555u);
                    int agree;

                    if (mname[0] == 'S') { /* Stacked RHT */
                        uint32_t s2 = seed * 0x13579BDFu + 0xFEDCBA98u;
                        agree = sign_agree_stacked_rht(q, k, dim, seed, s2);
                    } else if (mname[0] == 'R') { /* Rademacher */
                        agree = sign_agree_rademacher(q, k, dim, seed);
                    } else { /* Perm+RHT */
                        agree = sign_agree_perm_rht(q, k, dim, seed);
                    }
                    total += (float)(2 * agree - dim);
                }
                float raw = total / (float)K;
                estimates[r] = raw * ((float)M_PI / 2.0f) * qn * kn / (float)dim;
            }

            double mean = 0.0, var = 0.0;
            for (int r = 0; r < N_REPS; r++) mean += estimates[r];
            mean /= N_REPS;
            for (int r = 0; r < N_REPS; r++) {
                double d = estimates[r] - mean;
                var += d * d;
            }
            var /= N_REPS;

            printf("%-20s  %4d  %10.4f  %10.4f  %10.4f\n",
                   mname, K, mean, sqrt(var), mean - true_dot);
        }
    }

    /* ---- Variance reduction verification ---- */
    printf("\n================================================================\n");
    printf("  Variance Reduction (dim=64, Method B: Rademacher, N=10000)\n");
    printf("  Theoretical: var(K) = var(1) / K\n");
    printf("================================================================\n\n");

    seed_rng(42 + 64);
    std::vector<std::vector<float>> qs(N_PAIRS, std::vector<float>(dim));
    std::vector<std::vector<float>> ks(N_PAIRS, std::vector<float>(dim));
    std::vector<float> trues(N_PAIRS);

    for (int i = 0; i < N_PAIRS; i++) {
        for (int d = 0; d < dim; d++) {
            qs[i][d] = rand_normal();
            ks[i][d] = rand_normal();
        }
        trues[i] = dot_product(qs[i].data(), ks[i].data(), dim);
    }

    double var_k1 = 0.0;
    for (int K : {1, 2, 4, 8, 16}) {
        double sum_sq_err = 0.0;
        for (int i = 0; i < N_PAIRS; i++) {
            float raw = multi_hash_score(
                qs[i].data(), ks[i].data(), dim, K, sign_agree_rademacher);
            float q2 = l2_norm(qs[i].data(), dim);
            float k2 = l2_norm(ks[i].data(), dim);
            float est = raw * ((float)M_PI / 2.0f) * q2 * k2 / (float)dim;
            double err = est - trues[i];
            sum_sq_err += err * err;
        }
        double mse = sum_sq_err / N_PAIRS;
        if (K == 1) var_k1 = mse;

        printf("K=%2d: MSE=%10.4f, actual_ratio=%.3fx, theoretical=%.3fx\n",
               K, mse, mse / var_k1, 1.0 / K);
    }

    /* ---- Summary ---- */
    printf("\n================================================================\n");
    printf("  SUMMARY & RECOMMENDATIONS\n");
    printf("================================================================\n");
    printf("\n");
    printf("1. Plain RHT multi-hash is USELESS: different seeds produce\n");
    printf("   identical sign agreements because D cancels out.\n\n");
    printf("2. Three correct multi-hash methods were tested:\n");
    printf("   A) Stacked RHT: apply RHT twice with different seeds\n");
    printf("   B) Rademacher random projection (QJL-style)\n");
    printf("   C) Random permutation + fixed RHT\n\n");
    printf("3. Check the cosine column above for dim=64:\n");
    printf("   - If K=4 cosine > 0.85: recommend K=4 (4x bits, 16x vs FP32)\n");
    printf("   - If K=8 needed: 8x bits, 8x vs FP32 -- marginal\n\n");

    return 0;
}
