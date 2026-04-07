/**
 * QJL (Quantized Johnson-Lindenstrauss) — reference C implementation
 *
 * 1-bit sign hash: stores sign(key dot random_projection) as packed bits.
 * Uses a deterministic pseudo-random projection seeded by dimension index.
 */

#include "turboquant/turboquant.h"
#include <math.h>
#include <string.h>
#include <float.h>

/* ---------- FP16 helpers ---------- */

static uint16_t qjl_fp32_to_fp16(float v)
{
    union
    {
        float f;
        uint32_t u;
    } bits;
    bits.f = v;
    uint32_t sign = (bits.u >> 16) & 0x8000;
    int32_t exp = ((bits.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits.u >> 13) & 0x03FF;
    if (exp <= 0)
        return (uint16_t)sign;
    if (exp >= 31)
        return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static float qjl_fp16_to_fp32(uint16_t h)
{
    union
    {
        float f;
        uint32_t u;
    } bits;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0)
    {
        bits.u = sign;
        return bits.f;
    }
    if (exp == 31)
    {
        bits.u = sign | 0x7F800000 | (mant << 13);
        return bits.f;
    }
    exp = exp - 15 + 127;
    bits.u = sign | (exp << 23) | (mant << 13);
    return bits.f;
}

/* ---------- Deterministic pseudo-random projection ---------- */

static float qjl_random_entry(int dim_idx, int sketch_idx)
{
    /* Simple hash-based Rademacher random variable (+1/-1) */
    uint32_t h = (uint32_t)(dim_idx * 2654435761u + sketch_idx * 340573321u);
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    return (h & 1) ? 1.0f : -1.0f;
}

/* ---------- QJL quantize (reference) ---------- */

void tq_qjl_quantize_ref(const float *src, void *dst, int n)
{
    block_tq_qjl *block = (block_tq_qjl *)dst;
    int dim = n;
    if (dim > TQ_BK_QJL)
        dim = TQ_BK_QJL;

    /* Quick NaN check on first and last element */
    if (dim > 0 && (src[0] != src[0] || src[dim - 1] != src[dim - 1]))
    {
        memset(block, 0, sizeof(*block));
        return;
    }

    /* Compute L2 norm with max-abs rescaling for overflow protection */
    float max_abs = 0.0f;
    for (int d = 0; d < dim; d++)
    {
        float a = fabsf(src[d]);
        if (a > max_abs)
            max_abs = a;
    }
    if (max_abs == 0.0f)
    {
        memset(block, 0, sizeof(*block));
        return;
    }
    float inv_max = 1.0f / max_abs;
    float norm_sq = 0.0f;
    for (int d = 0; d < dim; d++)
    {
        float v = src[d] * inv_max;
        norm_sq += v * v;
    }
    block->norm = qjl_fp32_to_fp16(max_abs * sqrtf(norm_sq));

    /* Find outlier dimensions (largest absolute values) */
    float abs_vals[TQ_BK_QJL];
    for (int d = 0; d < dim; d++)
        abs_vals[d] = fabsf(src[d]);
    for (int d = dim; d < TQ_BK_QJL; d++)
        abs_vals[d] = 0.0f;

    float outlier_norm_sq = 0.0f;
    for (int o = 0; o < TQ_OUTLIERS; o++)
    {
        int best = 0;
        float best_val = -1.0f;
        for (int d = 0; d < dim; d++)
        {
            if (abs_vals[d] > best_val)
            {
                best_val = abs_vals[d];
                best = d;
            }
        }
        block->outlier_idx[o] = (uint8_t)(best < 256 ? best : 255);
        outlier_norm_sq += src[best] * src[best];
        abs_vals[best] = -1.0f; /* mark as used */
    }
    block->outlier_norm = qjl_fp32_to_fp16(sqrtf(outlier_norm_sq));

    /* Compute sign hash: for each sketch dimension, compute
       sign(sum_d key[d] * random(d, sketch_idx)) */
    memset(block->hash, 0, TQ_SKETCH_DIM / 8);
    for (int s = 0; s < TQ_SKETCH_DIM; s++)
    {
        float proj = 0.0f;
        for (int d = 0; d < dim; d++)
        {
            proj += src[d] * qjl_random_entry(d, s);
        }
        if (proj > 0.0f)
        {
            block->hash[s / 8] |= (1 << (s % 8));
        }
    }
}

/* ---------- QJL dequantize (reference, approximate) ---------- */

void tq_qjl_dequantize_ref(const void *src, float *dst, int n)
{
    const block_tq_qjl *block = (const block_tq_qjl *)src;
    int dim = n;
    if (dim > TQ_BK_QJL)
        dim = TQ_BK_QJL;

    /* QJL is a lossy 1-bit hash; exact dequantization is not possible.
       We reconstruct an approximation by summing projections weighted
       by their sign bits, then rescaling to match the stored norm. */
    for (int d = 0; d < dim; d++)
    {
        float val = 0.0f;
        for (int s = 0; s < TQ_SKETCH_DIM; s++)
        {
            int bit = (block->hash[s / 8] >> (s % 8)) & 1;
            float sign = bit ? 1.0f : -1.0f;
            val += sign * qjl_random_entry(d, s);
        }
        dst[d] = val;
    }

    /* Normalize to match original L2 norm */
    float recon_norm_sq = 0.0f;
    for (int d = 0; d < dim; d++)
    {
        recon_norm_sq += dst[d] * dst[d];
    }
    float recon_norm = sqrtf(recon_norm_sq);
    float target_norm = qjl_fp16_to_fp32(block->norm);
    if (recon_norm > 1e-8f)
    {
        float scale = target_norm / recon_norm;
        for (int d = 0; d < dim; d++)
        {
            dst[d] *= scale;
        }
    }
}

/* ---------- Popcount helpers ---------- */

/* Byte-level popcount lookup table */
static const uint8_t popcount_table[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

static int qjl_popcount_bytes(const uint8_t *data, int nbytes)
{
#if defined(__GNUC__) || defined(__clang__)
    /* Use compiler built-in for 32-bit chunks when available */
    int count = 0;
    int i = 0;
    for (; i + 4 <= nbytes; i += 4)
    {
        uint32_t word;
        memcpy(&word, data + i, 4);
        count += __builtin_popcount(word);
    }
    for (; i < nbytes; i++)
    {
        count += popcount_table[data[i]];
    }
    return count;
#else
    int count = 0;
    for (int i = 0; i < nbytes; i++)
    {
        count += popcount_table[data[i]];
    }
    return count;
#endif
}

/* ---------- QJL attention (direct Hamming distance) ---------- */

void tq_qjl_attention_ref(const float *query, const void *kv_cache,
                          float *scores, int seq_len, int head_dim)
{
    const block_tq_qjl *blocks = (const block_tq_qjl *)kv_cache;
    int dim = head_dim;
    if (dim > TQ_BK_QJL)
        dim = TQ_BK_QJL;

    const int sketch_dim = TQ_SKETCH_DIM;
    const int hash_bytes = sketch_dim / 8;

    /* Step 1: Project query into sketch space */
    float q_sketch[TQ_SKETCH_DIM];
    for (int s = 0; s < sketch_dim; s++)
    {
        float proj = 0.0f;
        for (int d = 0; d < dim; d++)
        {
            proj += query[d] * qjl_random_entry(d, s);
        }
        q_sketch[s] = proj;
    }

    /* Step 2: Sign-quantize query sketch into packed bits */
    uint8_t q_hash[TQ_SKETCH_DIM / 8];
    memset(q_hash, 0, hash_bytes);
    for (int s = 0; s < sketch_dim; s++)
    {
        if (q_sketch[s] > 0.0f)
        {
            q_hash[s / 8] |= (1 << (s % 8));
        }
    }

    /* Precompute outlier query contributions:
     * For each cached key we need query[outlier_idx[i]].
     * Since outlier indices vary per key, we precompute the full query
     * and index per key below. */

    /* Scaling factor from QJL theory: sqrt(PI/2) / sketch_dim
     * This converts Hamming-based inner product estimate to dot product estimate.
     * See refs/QJL/qjl_kernel/csrc/qjl_score_kernel.cu line 153. */
    const float scale = sqrtf(TQ_PI_2) / (float)sketch_dim;

    for (int s = 0; s < seq_len; s++)
    {
        const block_tq_qjl *block = &blocks[s];
        float key_norm = qjl_fp16_to_fp32(block->norm);
        float outlier_norm = qjl_fp16_to_fp32(block->outlier_norm);

        /* Step 3: XOR hash bytes and popcount for Hamming distance */
        uint8_t diff[TQ_SKETCH_DIM / 8];
        for (int b = 0; b < hash_bytes; b++)
        {
            diff[b] = q_hash[b] ^ block->hash[b];
        }
        int hamming = qjl_popcount_bytes(diff, hash_bytes);

        /* The inner product of sign vectors: each matching bit contributes +1,
         * each mismatched bit contributes -1.
         * inner_prod = (sketch_dim - hamming) - hamming = sketch_dim - 2*hamming
         * But we need to account for the continuous query sketch values.
         *
         * Actually, following the CUDA reference more precisely:
         * The score is computed as sum over bits: if key_bit matches q_sketch sign,
         * add +|q_sketch_val|, else add -|q_sketch_val|.
         * This equals sum(q_sketch[i] * key_sign[i]) where key_sign is +1/-1.
         *
         * For the fast path, we use the Hamming approximation:
         * inner_prod ~= sketch_dim - 2*hamming (treating all sketch magnitudes as equal)
         */

        /* Compute the inlier norm: norm_inlier = sqrt(key_norm^2 - outlier_norm^2) */
        float norm_sq_diff = key_norm * key_norm - outlier_norm * outlier_norm;
        float inlier_norm = (norm_sq_diff > 0.0f) ? sqrtf(norm_sq_diff) : 0.0f;

        /* Inlier score via Hamming distance */
        float inlier_score = scale * inlier_norm * (float)(sketch_dim - 2 * hamming);

        /* Step 4: Outlier correction
         * For outlier dimensions, compute exact contribution using:
         * query[outlier_idx] projected through random matrix, compared with
         * the key's hash bits. This is the outlier sketch inner product.
         *
         * For simplicity in the reference implementation, we compute outlier
         * contribution as a direct dot product estimate from the sketch bits,
         * weighted by the outlier norm. */
        float outlier_inner = 0.0f;
        for (int s_idx = 0; s_idx < TQ_SKETCH_DIM; s_idx++)
        {
            /* Compute outlier projection for this sketch dimension */
            float outlier_proj = 0.0f;
            for (int o = 0; o < TQ_OUTLIERS; o++)
            {
                int idx = block->outlier_idx[o];
                outlier_proj += query[idx] * qjl_random_entry(idx, s_idx);
            }
            /* Key sign for this sketch bit */
            int key_bit = (block->hash[s_idx / 8] >> (s_idx % 8)) & 1;
            float key_sign = key_bit ? 1.0f : -1.0f;
            outlier_inner += outlier_proj * key_sign;
        }
        float outlier_score = scale * outlier_norm * outlier_inner;

        scores[s] = inlier_score + outlier_score;
    }
}
