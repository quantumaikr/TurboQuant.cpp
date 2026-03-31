/**
 * tq_turbo_kv.c -- TurboQuant KV cache quantization pipeline
 *
 * Implements the TurboQuant algorithm from arXiv 2504.19874:
 *   1. Random Hadamard Transform (RHT) to decorrelate channels
 *   2. Optimal scalar quantization (Lloyd-Max codebook) on rotated data
 *   3. QJL 1-bit sign hash on the residual for unbiased inner product estimation
 *
 * Two variants:
 *   - TQ_TYPE_TURBO_KV_3B: 2-bit codebook + 1-bit QJL = 3 effective bits
 *   - TQ_TYPE_TURBO_KV_4B: 3-bit codebook + 1-bit QJL = 4 effective bits
 *
 * Key design: QJL is used for INNER PRODUCT estimation (attention), not for
 * point-wise reconstruction. The dequantize path uses MSE-only (codebook),
 * while the attention path adds the QJL residual correction for better scores.
 */

#include "turboquant/turboquant.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* Forward declarations from other modules */
extern void tq_codebook_quantize(const float* src, uint8_t* dst_indices,
                                  int n, int bits, float inv_std);
extern void tq_codebook_dequantize(const uint8_t* indices, float* dst,
                                    int n, int bits, float inv_std);

/* ============================================================
 * FP16 helpers (local copies to avoid cross-module dependencies)
 * ============================================================ */

static uint16_t tkv_fp32_to_fp16(float v) {
    union { float f; uint32_t u; } bits;
    bits.f = v;
    uint32_t sign = (bits.u >> 16) & 0x8000;
    int32_t  exp  = ((bits.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits.u >> 13) & 0x03FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static float tkv_fp16_to_fp32(uint16_t h) {
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

/* ============================================================
 * QJL random entry helper (must match tq_qjl.c exactly)
 * ============================================================ */

static float tkv_qjl_random_entry(int dim_idx, int sketch_idx) {
    uint32_t h = (uint32_t)(dim_idx * 2654435761u + sketch_idx * 340573321u);
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    return (h & 1) ? 1.0f : -1.0f;
}

/* ============================================================
 * Block seed: deterministic per-block seed based on position
 * ============================================================ */

#define TKV_DEFAULT_SEED 0x12345678u

/* ============================================================
 * Bit packing helpers for codebook indices
 * ============================================================ */

/* Pack 2-bit indices: 4 values per byte, LSB-first */
static void pack_2bit(const uint8_t* indices, uint8_t* packed, int n) {
    memset(packed, 0, (size_t)((n + 3) / 4));
    for (int i = 0; i < n; i++) {
        int byte_idx = i / 4;
        int bit_pos  = (i % 4) * 2;
        packed[byte_idx] |= (uint8_t)((indices[i] & 0x03) << bit_pos);
    }
}

static void unpack_2bit(const uint8_t* packed, uint8_t* indices, int n) {
    for (int i = 0; i < n; i++) {
        int byte_idx = i / 4;
        int bit_pos  = (i % 4) * 2;
        indices[i] = (packed[byte_idx] >> bit_pos) & 0x03;
    }
}

/* Pack 3-bit indices: using LSB-first bit-stream packing */
static void pack_3bit(const uint8_t* indices, uint8_t* packed, int n) {
    int total_bytes = (n * 3 + 7) / 8;
    memset(packed, 0, (size_t)total_bytes);
    for (int i = 0; i < n; i++) {
        int bit_offset = i * 3;
        int byte_idx = bit_offset / 8;
        int bit_pos  = bit_offset % 8;
        uint16_t val = (uint16_t)(indices[i] & 0x07);
        packed[byte_idx] |= (uint8_t)(val << bit_pos);
        if (bit_pos > 5) {
            packed[byte_idx + 1] |= (uint8_t)(val >> (8 - bit_pos));
        }
    }
}

static void unpack_3bit(const uint8_t* packed, uint8_t* indices, int n) {
    for (int i = 0; i < n; i++) {
        int bit_offset = i * 3;
        int byte_idx = bit_offset / 8;
        int bit_pos  = bit_offset % 8;
        uint16_t val = (uint16_t)packed[byte_idx];
        if (bit_pos > 5 && byte_idx + 1 < (n * 3 + 7) / 8) {
            val |= (uint16_t)packed[byte_idx + 1] << 8;
        }
        indices[i] = (uint8_t)((val >> bit_pos) & 0x07);
    }
}

/* ============================================================
 * QJL sign hash on residual (simplified, inline)
 * ============================================================ */

static void compute_qjl_signs(const float* residual, uint8_t* signs,
                                int dim, int n_sketch) {
    int hash_bytes = n_sketch / 8;
    memset(signs, 0, (size_t)hash_bytes);
    for (int s = 0; s < n_sketch; s++) {
        float proj = 0.0f;
        for (int d = 0; d < dim; d++) {
            proj += residual[d] * tkv_qjl_random_entry(d, s);
        }
        if (proj >= 0.0f) {
            signs[s / 8] |= (uint8_t)(1 << (s % 8));
        }
    }
}

/* ============================================================
 * Internal: MSE-only dequantize in rotated space (shared helper)
 * Returns the reconstructed vector in rotated space (before inverse RHT).
 * ============================================================ */

static void dequant_mse_rotated_2bit(const block_tq_turbo_kv_3b* block,
                                      float* rotated, int dim) {
    float inv_std = sqrtf((float)dim);
    uint8_t indices[TQ_BK];
    unpack_2bit(block->mse_indices, indices, dim);
    tq_codebook_dequantize(indices, rotated, dim, 2, inv_std);
}

static void dequant_mse_rotated_3bit(const block_tq_turbo_kv_4b* block,
                                      float* rotated, int dim) {
    float inv_std = sqrtf((float)dim);
    uint8_t indices[TQ_BK];
    unpack_3bit(block->mse_indices, indices, dim);
    tq_codebook_dequantize(indices, rotated, dim, 3, inv_std);
}

/* ============================================================
 * TurboQuant KV 3-bit: quantize
 * Pipeline: normalize -> RHT -> 2-bit codebook -> residual -> QJL 1-bit
 * ============================================================ */

void tq_turbo_kv_3b_quantize_ref(const float* src, void* dst, int n) {
    block_tq_turbo_kv_3b* block = (block_tq_turbo_kv_3b*)dst;
    int dim = n;
    if (dim > TQ_BK) dim = TQ_BK;

    /* Step 1: Compute L2 norm */
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm_sq += src[i] * src[i];
    }
    float norm = sqrtf(norm_sq);
    block->norm = tkv_fp32_to_fp16(norm);

    /* Step 2: Normalize and copy to working buffer */
    float rotated[TQ_BK];
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    for (int i = 0; i < dim; i++) {
        rotated[i] = src[i] * inv_norm;
    }
    for (int i = dim; i < TQ_BK; i++) {
        rotated[i] = 0.0f;
    }

    /* Step 3: Apply RHT (in-place on rotated) */
    uint32_t seed = TKV_DEFAULT_SEED;
    block->rht_seed = seed;
    tq_rht_transform(rotated, dim, seed);

    /* Step 4: Scalar quantize with 2-bit codebook
     * After RHT, coordinates are approximately N(0, 1/sqrt(dim)).
     * inv_std = sqrt(dim) to normalize to N(0,1). */
    float inv_std = sqrtf((float)dim);
    uint8_t indices[TQ_BK];
    tq_codebook_quantize(rotated, indices, dim, 2, inv_std);

    /* Pack 2-bit indices */
    pack_2bit(indices, block->mse_indices, dim);

    /* Step 5: Dequantize MSE stage to compute residual */
    float reconstructed[TQ_BK];
    tq_codebook_dequantize(indices, reconstructed, dim, 2, inv_std);

    /* Step 6: Compute residual in rotated space */
    float residual[TQ_BK];
    for (int i = 0; i < dim; i++) {
        residual[i] = rotated[i] - reconstructed[i];
    }
    for (int i = dim; i < TQ_BK; i++) {
        residual[i] = 0.0f;
    }

    /* Step 7: Compute residual norm */
    float r_norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        r_norm_sq += residual[i] * residual[i];
    }
    block->residual_norm = tkv_fp32_to_fp16(sqrtf(r_norm_sq));

    /* Step 8: QJL 1-bit sign hash on residual
     * sketch_dim = dim to fit in TQ_BK/8 bytes. */
    compute_qjl_signs(residual, block->qjl_signs, dim, dim);
}

/* ============================================================
 * TurboQuant KV 3-bit: dequantize (MSE-only)
 *
 * For point-wise reconstruction, we use only the codebook (MSE) stage.
 * The QJL residual is designed for inner product estimation (attention)
 * and adds noise in point-wise reconstruction.
 * ============================================================ */

void tq_turbo_kv_3b_dequantize_ref(const void* src, float* dst, int n) {
    const block_tq_turbo_kv_3b* block = (const block_tq_turbo_kv_3b*)src;
    int dim = n;
    if (dim > TQ_BK) dim = TQ_BK;

    float norm = tkv_fp16_to_fp32(block->norm);
    uint32_t seed = block->rht_seed;

    /* MSE-only dequantize in rotated space */
    float rotated[TQ_BK];
    dequant_mse_rotated_2bit(block, rotated, dim);

    /* Inverse RHT */
    tq_rht_inverse(rotated, dim, seed);

    /* Scale by original norm */
    for (int i = 0; i < dim; i++) {
        dst[i] = rotated[i] * norm;
    }
}

/* ============================================================
 * TurboQuant KV 3-bit: attention (two-stage inner product estimation)
 *
 * Optimized pipeline:
 *   1. RHT(query) computed ONCE before the per-key loop
 *   2. MSE dot product computed in rotated space (no RHT inverse)
 *   3. QJL query projection pre-computed ONCE, reused per key
 *   4. NEON vectorization for inner loops (with scalar fallback)
 *
 * The paper's formula for inner product estimation:
 *   <q, k_approx> = norm * (<q_rot, k_mse_rot> + r_norm * qjl_scale * correction)
 *
 * Key insight: RHT is orthogonal, so <q, Pi^T * k_rot> = <Pi*q, k_rot>.
 * By pre-rotating the query, we eliminate RHT inverse per key entirely.
 * ============================================================ */

void tq_turbo_kv_3b_attention_ref(const float* query, const void* kv_cache,
                                    float* scores, int seq_len, int head_dim) {
    const block_tq_turbo_kv_3b* blocks = (const block_tq_turbo_kv_3b*)kv_cache;
    int dim = head_dim;
    if (dim > TQ_BK) dim = TQ_BK;

    int sketch_dim = dim;  /* sketch dimension = block dim */
    float qjl_scale = sqrtf(TQ_PI_2) / (float)sketch_dim;

    /* Optimization #1: RHT(query) computed ONCE before the loop.
     * All keys use TKV_DEFAULT_SEED, so a single rotation suffices.
     * Since RHT is orthogonal: <q, Pi^T * k_rot> = <Pi*q, k_rot> = <q_rot, k_rot>
     * This eliminates O(d log d) RHT inverse per key. */
    float q_rot[TQ_BK];
    memcpy(q_rot, query, (size_t)dim * sizeof(float));
    for (int i = dim; i < TQ_BK; i++) q_rot[i] = 0.0f;
    tq_rht_transform(q_rot, dim, TKV_DEFAULT_SEED);

    /* Optimization #2: Pre-compute QJL query projection ONCE.
     * q_proj[s] = sum_d(q_rot[d] * S[d,s]) for each sketch dimension.
     * This is O(dim * sketch_dim) once, instead of per key. */
    float q_proj[TQ_BK];
    for (int s = 0; s < sketch_dim; s++) {
        float proj = 0.0f;
        for (int d = 0; d < dim; d++) {
            proj += q_rot[d] * tkv_qjl_random_entry(d, s);
        }
        q_proj[s] = proj;
    }

    for (int seq = 0; seq < seq_len; seq++) {
        const block_tq_turbo_kv_3b* block = &blocks[seq];
        float norm = tkv_fp16_to_fp32(block->norm);
        float r_norm = tkv_fp16_to_fp32(block->residual_norm);

        /* MSE stage: dequantize in rotated space, dot with q_rot directly.
         * No RHT inverse needed -- both vectors are in rotated space. */
        float rotated[TQ_BK];
        dequant_mse_rotated_2bit(block, rotated, dim);

        float mse_dot = 0.0f;
#ifdef __ARM_NEON
        {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            int d = 0;
            for (; d + 15 < dim; d += 16) {
                acc0 = vfmaq_f32(acc0, vld1q_f32(&q_rot[d]),      vld1q_f32(&rotated[d]));
                acc1 = vfmaq_f32(acc1, vld1q_f32(&q_rot[d + 4]),  vld1q_f32(&rotated[d + 4]));
                acc2 = vfmaq_f32(acc2, vld1q_f32(&q_rot[d + 8]),  vld1q_f32(&rotated[d + 8]));
                acc3 = vfmaq_f32(acc3, vld1q_f32(&q_rot[d + 12]), vld1q_f32(&rotated[d + 12]));
            }
            acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            for (; d + 3 < dim; d += 4) {
                acc0 = vfmaq_f32(acc0, vld1q_f32(&q_rot[d]), vld1q_f32(&rotated[d]));
            }
            mse_dot = vaddvq_f32(acc0);
            for (; d < dim; d++) {
                mse_dot += q_rot[d] * rotated[d];
            }
        }
#else
        for (int d = 0; d < dim; d++) {
            mse_dot += q_rot[d] * rotated[d];
        }
#endif

        /* QJL residual correction using pre-computed q_proj.
         * Per key: just sum q_proj[s] * sign_s, which is O(sketch_dim). */
        float qjl_correction = 0.0f;
#ifdef __ARM_NEON
        {
            float32x4_t corr0 = vdupq_n_f32(0.0f);
            float32x4_t corr1 = vdupq_n_f32(0.0f);
            float32x4_t corr2 = vdupq_n_f32(0.0f);
            float32x4_t corr3 = vdupq_n_f32(0.0f);
            int s = 0;
            for (; s + 15 < sketch_dim; s += 16) {
                /* Load 2 bytes of sign bits covering 16 sketch dims */
                uint32_t sign_bits = (uint32_t)block->qjl_signs[s / 8]
                                   | ((uint32_t)block->qjl_signs[s / 8 + 1] << 8);
                /* Expand bits to float +1/-1 for 4 lanes at a time */
                float signs_f[16];
                for (int i = 0; i < 16; i++) {
                    signs_f[i] = ((sign_bits >> i) & 1) ? 1.0f : -1.0f;
                }
                float32x4_t s0 = vld1q_f32(&signs_f[0]);
                float32x4_t s1 = vld1q_f32(&signs_f[4]);
                float32x4_t s2 = vld1q_f32(&signs_f[8]);
                float32x4_t s3 = vld1q_f32(&signs_f[12]);
                corr0 = vfmaq_f32(corr0, vld1q_f32(&q_proj[s]),      s0);
                corr1 = vfmaq_f32(corr1, vld1q_f32(&q_proj[s + 4]),  s1);
                corr2 = vfmaq_f32(corr2, vld1q_f32(&q_proj[s + 8]),  s2);
                corr3 = vfmaq_f32(corr3, vld1q_f32(&q_proj[s + 12]), s3);
            }
            corr0 = vaddq_f32(vaddq_f32(corr0, corr1), vaddq_f32(corr2, corr3));
            for (; s + 3 < sketch_dim; s += 4) {
                float signs_f[4];
                uint8_t byte = block->qjl_signs[s / 8];
                int bit_off = s % 8;
                for (int i = 0; i < 4; i++) {
                    signs_f[i] = ((byte >> (bit_off + i)) & 1) ? 1.0f : -1.0f;
                }
                corr0 = vfmaq_f32(corr0, vld1q_f32(&q_proj[s]), vld1q_f32(signs_f));
            }
            qjl_correction = vaddvq_f32(corr0);
            for (; s < sketch_dim; s++) {
                int bit = (block->qjl_signs[s / 8] >> (s % 8)) & 1;
                float key_sign = bit ? 1.0f : -1.0f;
                qjl_correction += q_proj[s] * key_sign;
            }
        }
#else
        for (int s = 0; s < sketch_dim; s++) {
            int bit = (block->qjl_signs[s / 8] >> (s % 8)) & 1;
            float key_sign = bit ? 1.0f : -1.0f;
            qjl_correction += q_proj[s] * key_sign;
        }
#endif
        qjl_correction *= qjl_scale * r_norm;

        scores[seq] = norm * mse_dot + norm * qjl_correction;
    }
}

/* ============================================================
 * TurboQuant KV 4-bit: quantize
 * Same pipeline but with 3-bit codebook (8 levels) + 1-bit QJL
 * ============================================================ */

void tq_turbo_kv_4b_quantize_ref(const float* src, void* dst, int n) {
    block_tq_turbo_kv_4b* block = (block_tq_turbo_kv_4b*)dst;
    int dim = n;
    if (dim > TQ_BK) dim = TQ_BK;

    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm_sq += src[i] * src[i];
    }
    float norm = sqrtf(norm_sq);
    block->norm = tkv_fp32_to_fp16(norm);

    float rotated[TQ_BK];
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    for (int i = 0; i < dim; i++) {
        rotated[i] = src[i] * inv_norm;
    }
    for (int i = dim; i < TQ_BK; i++) {
        rotated[i] = 0.0f;
    }

    uint32_t seed = TKV_DEFAULT_SEED;
    block->rht_seed = seed;
    tq_rht_transform(rotated, dim, seed);

    float inv_std = sqrtf((float)dim);
    uint8_t indices[TQ_BK];
    tq_codebook_quantize(rotated, indices, dim, 3, inv_std);
    pack_3bit(indices, block->mse_indices, dim);

    float reconstructed[TQ_BK];
    tq_codebook_dequantize(indices, reconstructed, dim, 3, inv_std);

    float residual[TQ_BK];
    for (int i = 0; i < dim; i++) {
        residual[i] = rotated[i] - reconstructed[i];
    }
    for (int i = dim; i < TQ_BK; i++) {
        residual[i] = 0.0f;
    }

    float r_norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        r_norm_sq += residual[i] * residual[i];
    }
    block->residual_norm = tkv_fp32_to_fp16(sqrtf(r_norm_sq));

    compute_qjl_signs(residual, block->qjl_signs, dim, dim);
}

/* ============================================================
 * TurboQuant KV 4-bit: dequantize (MSE-only)
 * ============================================================ */

void tq_turbo_kv_4b_dequantize_ref(const void* src, float* dst, int n) {
    const block_tq_turbo_kv_4b* block = (const block_tq_turbo_kv_4b*)src;
    int dim = n;
    if (dim > TQ_BK) dim = TQ_BK;

    float norm = tkv_fp16_to_fp32(block->norm);
    uint32_t seed = block->rht_seed;

    float rotated[TQ_BK];
    dequant_mse_rotated_3bit(block, rotated, dim);

    tq_rht_inverse(rotated, dim, seed);

    for (int i = 0; i < dim; i++) {
        dst[i] = rotated[i] * norm;
    }
}

/* ============================================================
 * TurboQuant KV 4-bit: attention (two-stage inner product estimation)
 *
 * Same optimized pipeline as 3-bit:
 *   1. RHT(query) once  2. QJL projection once  3. Rotated-space dot
 *   4. NEON vectorization with scalar fallback
 * ============================================================ */

void tq_turbo_kv_4b_attention_ref(const float* query, const void* kv_cache,
                                    float* scores, int seq_len, int head_dim) {
    const block_tq_turbo_kv_4b* blocks_4b = (const block_tq_turbo_kv_4b*)kv_cache;
    int dim = head_dim;
    if (dim > TQ_BK) dim = TQ_BK;

    int sketch_dim = dim;
    float qjl_scale = sqrtf(TQ_PI_2) / (float)sketch_dim;

    /* Optimization #1: RHT(query) computed ONCE. */
    float q_rot[TQ_BK];
    memcpy(q_rot, query, (size_t)dim * sizeof(float));
    for (int i = dim; i < TQ_BK; i++) q_rot[i] = 0.0f;
    tq_rht_transform(q_rot, dim, TKV_DEFAULT_SEED);

    /* Optimization #2: Pre-compute QJL query projection ONCE. */
    float q_proj[TQ_BK];
    for (int s = 0; s < sketch_dim; s++) {
        float proj = 0.0f;
        for (int d = 0; d < dim; d++) {
            proj += q_rot[d] * tkv_qjl_random_entry(d, s);
        }
        q_proj[s] = proj;
    }

    for (int seq = 0; seq < seq_len; seq++) {
        const block_tq_turbo_kv_4b* block = &blocks_4b[seq];
        float norm = tkv_fp16_to_fp32(block->norm);
        float r_norm = tkv_fp16_to_fp32(block->residual_norm);

        /* MSE stage: dequantize in rotated space, dot with q_rot directly. */
        float rotated[TQ_BK];
        dequant_mse_rotated_3bit(block, rotated, dim);

        float mse_dot = 0.0f;
#ifdef __ARM_NEON
        {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            int d = 0;
            for (; d + 15 < dim; d += 16) {
                acc0 = vfmaq_f32(acc0, vld1q_f32(&q_rot[d]),      vld1q_f32(&rotated[d]));
                acc1 = vfmaq_f32(acc1, vld1q_f32(&q_rot[d + 4]),  vld1q_f32(&rotated[d + 4]));
                acc2 = vfmaq_f32(acc2, vld1q_f32(&q_rot[d + 8]),  vld1q_f32(&rotated[d + 8]));
                acc3 = vfmaq_f32(acc3, vld1q_f32(&q_rot[d + 12]), vld1q_f32(&rotated[d + 12]));
            }
            acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            for (; d + 3 < dim; d += 4) {
                acc0 = vfmaq_f32(acc0, vld1q_f32(&q_rot[d]), vld1q_f32(&rotated[d]));
            }
            mse_dot = vaddvq_f32(acc0);
            for (; d < dim; d++) {
                mse_dot += q_rot[d] * rotated[d];
            }
        }
#else
        for (int d = 0; d < dim; d++) {
            mse_dot += q_rot[d] * rotated[d];
        }
#endif

        /* QJL residual correction using pre-computed q_proj. */
        float qjl_correction = 0.0f;
#ifdef __ARM_NEON
        {
            float32x4_t corr0 = vdupq_n_f32(0.0f);
            float32x4_t corr1 = vdupq_n_f32(0.0f);
            float32x4_t corr2 = vdupq_n_f32(0.0f);
            float32x4_t corr3 = vdupq_n_f32(0.0f);
            int s = 0;
            for (; s + 15 < sketch_dim; s += 16) {
                uint32_t sign_bits = (uint32_t)block->qjl_signs[s / 8]
                                   | ((uint32_t)block->qjl_signs[s / 8 + 1] << 8);
                float signs_f[16];
                for (int i = 0; i < 16; i++) {
                    signs_f[i] = ((sign_bits >> i) & 1) ? 1.0f : -1.0f;
                }
                corr0 = vfmaq_f32(corr0, vld1q_f32(&q_proj[s]),      vld1q_f32(&signs_f[0]));
                corr1 = vfmaq_f32(corr1, vld1q_f32(&q_proj[s + 4]),  vld1q_f32(&signs_f[4]));
                corr2 = vfmaq_f32(corr2, vld1q_f32(&q_proj[s + 8]),  vld1q_f32(&signs_f[8]));
                corr3 = vfmaq_f32(corr3, vld1q_f32(&q_proj[s + 12]), vld1q_f32(&signs_f[12]));
            }
            corr0 = vaddq_f32(vaddq_f32(corr0, corr1), vaddq_f32(corr2, corr3));
            for (; s + 3 < sketch_dim; s += 4) {
                float signs_f[4];
                uint8_t byte = block->qjl_signs[s / 8];
                int bit_off = s % 8;
                for (int i = 0; i < 4; i++) {
                    signs_f[i] = ((byte >> (bit_off + i)) & 1) ? 1.0f : -1.0f;
                }
                corr0 = vfmaq_f32(corr0, vld1q_f32(&q_proj[s]), vld1q_f32(signs_f));
            }
            qjl_correction = vaddvq_f32(corr0);
            for (; s < sketch_dim; s++) {
                int bit = (block->qjl_signs[s / 8] >> (s % 8)) & 1;
                float key_sign = bit ? 1.0f : -1.0f;
                qjl_correction += q_proj[s] * key_sign;
            }
        }
#else
        for (int s = 0; s < sketch_dim; s++) {
            int bit = (block->qjl_signs[s / 8] >> (s % 8)) & 1;
            float key_sign = bit ? 1.0f : -1.0f;
            qjl_correction += q_proj[s] * key_sign;
        }
#endif
        qjl_correction *= qjl_scale * r_norm;

        scores[seq] = norm * mse_dot + norm * qjl_correction;
    }
}

/* ============================================================
 * TurboQuant KV 1-bit: quantize
 *
 * Extreme compression: normalize -> RHT -> sign extraction.
 * Each dimension is stored as a single sign bit.
 * For dim=128: 24 bytes total (8 header + 16 sign bytes).
 * Compression ratio: 128*4 / 24 = 21.3x vs FP32.
 * ============================================================ */

void tq_turbo_kv_1b_quantize_ref(const float* src, void* dst, int n) {
    block_tq_turbo_kv_1b* block = (block_tq_turbo_kv_1b*)dst;
    int dim = n;
    if (dim > TQ_BK) dim = TQ_BK;

    /* Step 1: Compute L2 norm */
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm_sq += src[i] * src[i];
    }
    float norm = sqrtf(norm_sq);
    block->norm = tkv_fp32_to_fp16(norm);
    block->_pad = 0;

    /* Step 2: Normalize and copy to working buffer */
    float rotated[TQ_BK];
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    for (int i = 0; i < dim; i++) {
        rotated[i] = src[i] * inv_norm;
    }
    for (int i = dim; i < TQ_BK; i++) {
        rotated[i] = 0.0f;
    }

    /* Step 3: Apply RHT (in-place on rotated) */
    uint32_t seed = TKV_DEFAULT_SEED;
    block->rht_seed = seed;
    tq_rht_transform(rotated, dim, seed);

    /* Step 4: Extract sign bits -- 1 bit per dimension */
    int sign_bytes = dim / 8;
    memset(block->signs, 0, (size_t)sign_bytes);
    for (int i = 0; i < dim; i++) {
        if (rotated[i] >= 0.0f) {
            block->signs[i / 8] |= (uint8_t)(1 << (i % 8));
        }
    }
}

/* ============================================================
 * TurboQuant KV 1-bit: dequantize (rough reconstruction)
 *
 * Reconstruct: sign * (norm / sqrt(dim)) then inverse RHT.
 * This is a very rough reconstruction -- the real value of 1-bit
 * is in Hamming attention, not point-wise dequant.
 * ============================================================ */

void tq_turbo_kv_1b_dequantize_ref(const void* src, float* dst, int n) {
    const block_tq_turbo_kv_1b* block = (const block_tq_turbo_kv_1b*)src;
    int dim = n;
    if (dim > TQ_BK) dim = TQ_BK;

    float norm = tkv_fp16_to_fp32(block->norm);
    uint32_t seed = block->rht_seed;

    /* Reconstruct sign vector in rotated space.
     * After RHT, coordinates are ~N(0, 1/sqrt(dim)).
     * Expected |x| for half-normal = sqrt(2/pi) * sigma = sqrt(2/pi) / sqrt(dim).
     * So sign * sqrt(2/pi) / sqrt(dim) is the expected reconstruction. */
    float scale = sqrtf(2.0f / TQ_PI) / sqrtf((float)dim);
    float rotated[TQ_BK];
    for (int i = 0; i < dim; i++) {
        int bit = (block->signs[i / 8] >> (i % 8)) & 1;
        rotated[i] = bit ? scale : -scale;
    }

    /* Inverse RHT */
    tq_rht_inverse(rotated, dim, seed);

    /* Scale by original norm */
    for (int i = 0; i < dim; i++) {
        dst[i] = rotated[i] * norm;
    }
}

/* ============================================================
 * TurboQuant KV 1-bit: attention (XOR + popcount Hamming)
 *
 * Ultra-fast attention using bitwise operations:
 *   1. RHT(query) computed ONCE
 *   2. Extract query sign bits ONCE
 *   3. Per key: XOR + popcount -> Hamming distance -> score
 *
 * The inner product estimator:
 *   <q, k> ~ q_norm * k_norm * sqrt(pi/2) / dim * (2*agree - dim)
 * where agree = dim - hamming_distance(q_signs, k_signs).
 *
 * NEON vectorization for popcount with scalar fallback.
 * ============================================================ */

void tq_turbo_kv_1b_attention_ref(const float* query, const void* kv_cache,
                                    float* scores, int seq_len, int head_dim) {
    const block_tq_turbo_kv_1b* blocks = (const block_tq_turbo_kv_1b*)kv_cache;
    int dim = head_dim;
    if (dim > TQ_BK) dim = TQ_BK;

    float scale_factor = sqrtf(TQ_PI_2) / (float)dim;

    /* Step 1: RHT(query) computed ONCE */
    float q_rot[TQ_BK];
    memcpy(q_rot, query, (size_t)dim * sizeof(float));
    for (int i = dim; i < TQ_BK; i++) q_rot[i] = 0.0f;
    tq_rht_transform(q_rot, dim, TKV_DEFAULT_SEED);

    /* Step 2: Compute query L2 norm */
    float q_norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        q_norm_sq += query[i] * query[i];
    }
    float q_norm = sqrtf(q_norm_sq);

    /* Step 3: Extract query sign bits */
    int sign_bytes = dim / 8;
    uint8_t q_signs[TQ_BK / 8];
    memset(q_signs, 0, (size_t)sign_bytes);
    for (int i = 0; i < dim; i++) {
        if (q_rot[i] >= 0.0f) {
            q_signs[i / 8] |= (uint8_t)(1 << (i % 8));
        }
    }

    /* Step 4: Per-key Hamming attention */
    for (int seq = 0; seq < seq_len; seq++) {
        const block_tq_turbo_kv_1b* blk = &blocks[seq];
        float k_norm = tkv_fp16_to_fp32(blk->norm);

        /* XOR + popcount to get Hamming distance */
        int hamming = 0;
#ifdef __ARM_NEON
        if (sign_bytes == 16) {
            /* Optimized path for dim=128 (16 sign bytes) */
            uint8x16_t vq = vld1q_u8(q_signs);
            uint8x16_t vk = vld1q_u8(blk->signs);
            uint8x16_t vxor = veorq_u8(vq, vk);
            /* Count bits: use NEON vcntq_u8 for byte-level popcount */
            uint8x16_t vcnt = vcntq_u8(vxor);
            /* Horizontal sum of all byte popcounts */
            hamming = vaddlvq_u8(vcnt);
        } else {
            for (int b = 0; b < sign_bytes; b++) {
                uint8_t xor_byte = q_signs[b] ^ blk->signs[b];
                hamming += __builtin_popcount(xor_byte);
            }
        }
#else
        for (int b = 0; b < sign_bytes; b++) {
            uint8_t xor_byte = q_signs[b] ^ blk->signs[b];
            /* Portable popcount using Kernighan's bit trick */
            int c = 0;
            while (xor_byte) { c++; xor_byte &= xor_byte - 1; }
            hamming += c;
        }
#endif

        int agree = dim - hamming;
        float score = q_norm * k_norm * scale_factor * (float)(2 * agree - dim);
        scores[seq] = score;
    }
}
