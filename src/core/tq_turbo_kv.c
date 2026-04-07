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
        if (proj > 0.0f) {
            signs[s / 8] |= (uint8_t)(1 << (s % 8));
        }
    }
}

/* ============================================================
 * Internal: MSE-only dequantize in rotated space (shared helper)
 * Returns the reconstructed vector in rotated space (before inverse RHT).
 * ============================================================ */

static void dequant_mse_rotated_3bit_v2(const block_tq_turbo_kv_3b* block,
                                         float* rotated, int dim) {
    /* Variant F (3b): 3-bit codebook (8 levels) + max-abs scaling, no QJL */
    float inv_std = tkv_fp16_to_fp32(block->inv_std_fp16);
    if (inv_std < 1e-10f) inv_std = sqrtf((float)dim);
    uint8_t indices[TQ_BK] = {0};
    unpack_3bit(block->mse_indices, indices, dim);
    tq_codebook_dequantize(indices, rotated, dim, 3, inv_std);
}

static void dequant_mse_rotated_4bit_v2(const block_tq_turbo_kv_4b* block,
                                         float* rotated, int dim) {
    /* Variant F: 4-bit codebook (16 levels) + max-abs scaling */
    float inv_std = tkv_fp16_to_fp32(block->inv_std_fp16);
    if (inv_std < 1e-10f) inv_std = sqrtf((float)dim);
    /* Unpack 4-bit indices: 2 per byte, LSB-first */
    uint8_t indices[TQ_BK];
    for (int i = 0; i < dim; i++) {
        uint8_t b = block->mse_indices[i / 2];
        indices[i] = (i & 1) ? (b >> 4) : (b & 0x0F);
    }
    tq_codebook_dequantize(indices, rotated, dim, 4, inv_std);
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
    tq_rht_transform(rotated, dim, TKV_DEFAULT_SEED);

    /* Variant F (3b): 3-bit codebook (8 levels) + max-abs scaling, no QJL */
    float max_abs = 0.0f;
    for (int i = 0; i < dim; i++) {
        float a = fabsf(rotated[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-10f) max_abs = 1.0f;
    const float CENT_3BIT_MAX = 2.1520f;
    float inv_std = CENT_3BIT_MAX / max_abs;
    block->inv_std_fp16 = tkv_fp32_to_fp16(inv_std);
    block->residual_norm = 0;
    block->_pad = 0;

    uint8_t indices[TQ_BK];
    tq_codebook_quantize(rotated, indices, dim, 3, inv_std);
    pack_3bit(indices, block->mse_indices, dim);
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

    float rotated[TQ_BK];
    dequant_mse_rotated_3bit_v2(block, rotated, dim);
    tq_rht_inverse(rotated, dim, TKV_DEFAULT_SEED);

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

    /* Pre-rotate query once */
    float q_rot[TQ_BK];
    memcpy(q_rot, query, (size_t)dim * sizeof(float));
    for (int i = dim; i < TQ_BK; i++) q_rot[i] = 0.0f;
    tq_rht_transform(q_rot, dim, TKV_DEFAULT_SEED);

    for (int seq = 0; seq < seq_len; seq++) {
        const block_tq_turbo_kv_3b* block = &blocks[seq];
        float norm = tkv_fp16_to_fp32(block->norm);

        float rotated[TQ_BK];
        dequant_mse_rotated_3bit_v2(block, rotated, dim);

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

        scores[seq] = norm * mse_dot;
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

    tq_rht_transform(rotated, dim, TKV_DEFAULT_SEED);

    /* Variant F: 4-bit (16-level) Lloyd-Max codebook with max-abs scaling.
     * QJL residual is dropped — ablation showed it contributed ~0 to scores.
     * The 16 freed bytes pay for 16 levels (vs the previous 8). */
    float max_abs = 0.0f;
    for (int i = 0; i < dim; i++) {
        float a = fabsf(rotated[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-10f) max_abs = 1.0f;
    const float CENT_4BIT_MAX = 2.7326f;
    float inv_std = CENT_4BIT_MAX / max_abs;
    block->inv_std_fp16 = tkv_fp32_to_fp16(inv_std);
    block->residual_norm = 0;
    block->_pad = 0;

    uint8_t indices[TQ_BK];
    tq_codebook_quantize(rotated, indices, dim, 4, inv_std);
    /* Pack 4-bit: 2 indices per byte, LSB-first */
    memset(block->mse_indices, 0, TQ_BK / 2);
    for (int i = 0; i < dim; i++) {
        int byte_idx = i / 2;
        int bit_pos  = (i & 1) * 4;
        block->mse_indices[byte_idx] |= (uint8_t)((indices[i] & 0x0F) << bit_pos);
    }
}

/* ============================================================
 * TurboQuant KV 4-bit: dequantize (MSE-only)
 * ============================================================ */

void tq_turbo_kv_4b_dequantize_ref(const void* src, float* dst, int n) {
    const block_tq_turbo_kv_4b* block = (const block_tq_turbo_kv_4b*)src;
    int dim = n;
    if (dim > TQ_BK) dim = TQ_BK;

    float norm = tkv_fp16_to_fp32(block->norm);

    float rotated[TQ_BK];
    dequant_mse_rotated_4bit_v2(block, rotated, dim);
    tq_rht_inverse(rotated, dim, TKV_DEFAULT_SEED);

    for (int i = 0; i < dim; i++) {
        dst[i] = rotated[i] * norm;
    }
}

/* ============================================================
 * TurboQuant KV 4-bit: attention (Variant F: codebook-only, no QJL)
 *
 * QJL contributed ~0 in our ablation, so the 4b path is now a clean
 * single-stage 4-bit codebook estimator. Pre-rotated query, dot product
 * in rotated space, no per-key RHT inverse.
 * ============================================================ */

void tq_turbo_kv_4b_attention_ref(const float* query, const void* kv_cache,
                                    float* scores, int seq_len, int head_dim) {
    const block_tq_turbo_kv_4b* blocks_4b = (const block_tq_turbo_kv_4b*)kv_cache;
    int dim = head_dim;
    if (dim > TQ_BK) dim = TQ_BK;

    /* Pre-rotate query once */
    float q_rot[TQ_BK];
    memcpy(q_rot, query, (size_t)dim * sizeof(float));
    for (int i = dim; i < TQ_BK; i++) q_rot[i] = 0.0f;
    tq_rht_transform(q_rot, dim, TKV_DEFAULT_SEED);

    for (int seq = 0; seq < seq_len; seq++) {
        const block_tq_turbo_kv_4b* block = &blocks_4b[seq];
        float norm = tkv_fp16_to_fp32(block->norm);

        float rotated[TQ_BK];
        dequant_mse_rotated_4bit_v2(block, rotated, dim);

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

        scores[seq] = norm * mse_dot;
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

    /* QJL paper: sketch_dim / dim >= 2 for acceptable distortion.
     * For dim < 128, expand sketch to 128 bits (block has signs[16] = 128 bits).
     * We replicate the vector and apply RHT with different seeds per chunk. */
    int sketch_dim = dim;
    if (sketch_dim < TQ_BK) sketch_dim = TQ_BK;  /* at least 128 */

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

    /* Step 3: Apply RHT.  When sketch_dim > dim we replicate the normalized
     * vector into multiple dim-sized chunks and apply RHT with different seeds
     * to each chunk.  This is a structured random projection (QJL paper). */
    uint32_t seed = TKV_DEFAULT_SEED;
    block->rht_seed = seed;

    if (sketch_dim > dim) {
        /* Replicate normalized vector into each chunk */
        int n_chunks = sketch_dim / dim;
        for (int c = 1; c < n_chunks; c++) {
            for (int i = 0; i < dim; i++) {
                rotated[c * dim + i] = rotated[i];
            }
        }
        /* Apply RHT with different seed per chunk */
        for (int c = 0; c < n_chunks; c++) {
            uint32_t chunk_seed = seed + (uint32_t)c * 0x9ABCDEF0u;
            tq_rht_transform(rotated + c * dim, dim, chunk_seed);
        }
    } else {
        tq_rht_transform(rotated, dim, seed);
    }

    /* Step 4: Extract sign bits -- 1 bit per sketch dimension */
    int sign_bytes = sketch_dim / 8;
    memset(block->signs, 0, (size_t)sign_bytes);
    for (int i = 0; i < sketch_dim; i++) {
        if (rotated[i] > 0.0f) {
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

    int sketch_dim = dim;
    if (sketch_dim < TQ_BK) sketch_dim = TQ_BK;

    float norm = tkv_fp16_to_fp32(block->norm);
    uint32_t seed = TKV_DEFAULT_SEED;

    /* Reconstruct sign vector in rotated space.
     * After RHT, coordinates are ~N(0, 1/sqrt(dim)).
     * Expected |x| for half-normal = sqrt(2/pi) * sigma = sqrt(2/pi) / sqrt(dim).
     * So sign * sqrt(2/pi) / sqrt(dim) is the expected reconstruction. */
    float scale = sqrtf(2.0f / TQ_PI) / sqrtf((float)dim);

    if (sketch_dim > dim) {
        /* When sketch was expanded, reconstruct each chunk and average */
        int n_chunks = sketch_dim / dim;
        float accum[TQ_BK];
        memset(accum, 0, (size_t)dim * sizeof(float));

        for (int c = 0; c < n_chunks; c++) {
            float chunk[TQ_BK];
            for (int i = 0; i < dim; i++) {
                int si = c * dim + i;
                int bit = (block->signs[si / 8] >> (si % 8)) & 1;
                chunk[i] = bit ? scale : -scale;
            }
            uint32_t chunk_seed = seed + (uint32_t)c * 0x9ABCDEF0u;
            tq_rht_inverse(chunk, dim, chunk_seed);
            for (int i = 0; i < dim; i++) {
                accum[i] += chunk[i];
            }
        }

        float inv_chunks = 1.0f / (float)n_chunks;
        for (int i = 0; i < dim; i++) {
            dst[i] = accum[i] * inv_chunks * norm;
        }
    } else {
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

    /* Match quantize: expand sketch_dim for small dimensions */
    int sketch_dim = dim;
    if (sketch_dim < TQ_BK) sketch_dim = TQ_BK;

    /* Scale factor for sign-sign agreement estimator: (pi/2) / m.
     * Note: sqrt(pi/2)/m is for random-projection-then-sign (QJL).
     * sign-sign (Hamming) uses pi/2 per the arcsin law.
     * Currently int_attn is disabled, but fix for future use. */
    float scale_factor = TQ_PI_2 / (float)sketch_dim;

    /* Step 1: RHT(query) with expansion matching quantize */
    float q_rot[TQ_BK];
    memcpy(q_rot, query, (size_t)dim * sizeof(float));
    for (int i = dim; i < TQ_BK; i++) q_rot[i] = 0.0f;

    if (sketch_dim > dim) {
        int n_chunks = sketch_dim / dim;
        for (int c = 1; c < n_chunks; c++) {
            for (int i = 0; i < dim; i++) {
                q_rot[c * dim + i] = q_rot[i];
            }
        }
        for (int c = 0; c < n_chunks; c++) {
            uint32_t chunk_seed = TKV_DEFAULT_SEED + (uint32_t)c * 0x9ABCDEF0u;
            tq_rht_transform(q_rot + c * dim, dim, chunk_seed);
        }
    } else {
        tq_rht_transform(q_rot, dim, TKV_DEFAULT_SEED);
    }

    /* Step 2: Compute query L2 norm */
    float q_norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        q_norm_sq += query[i] * query[i];
    }
    float q_norm = sqrtf(q_norm_sq);

    /* Step 3: Extract query sign bits over sketch_dim */
    int sign_bytes = sketch_dim / 8;
    uint8_t q_signs[TQ_BK / 8];
    if (sign_bytes > 0) memset(q_signs, 0, (size_t)sign_bytes);
    for (int i = 0; i < sketch_dim; i++) {
        if (q_rot[i] > 0.0f) {
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
            /* Optimized path for sketch_dim=128 (16 sign bytes) */
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

        int agree = sketch_dim - hamming;
        float score = q_norm * k_norm * scale_factor * (float)(2 * agree - sketch_dim);
        scores[seq] = score;
    }
}

/* ============================================================
 * TurboQuant KV 2-bit: 1-bit codebook + 1-bit QJL residual
 *
 * Lightweight 2-bit variant that stores the sign bit from
 * 1-bit MSE quantization plus a 1-bit QJL hash on the residual.
 * Total: 2 bits per element.
 * ============================================================ */

void tq_turbo_kv_2b_quantize_ref(const float* src, void* dst, int n) {
    const int dim = (n < TQ_BK) ? n : TQ_BK;
    block_tq_turbo_kv_2b* blk = (block_tq_turbo_kv_2b*)dst;
    memset(blk, 0, sizeof(*blk));

    /* Expand sketch_dim for small dimensions (QJL paper: m/d >= 2) */
    int sketch_dim = dim;
    if (sketch_dim < TQ_BK) sketch_dim = TQ_BK;

    /* Step 1: Compute L2 norm */
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) norm_sq += src[i] * src[i];
    float norm = sqrtf(norm_sq);
    blk->norm = tkv_fp32_to_fp16(norm);

    if (norm < 1e-12f) return;

    /* Step 2: Normalize and copy to working buffer */
    float inv_norm = 1.0f / norm;
    float rotated[TQ_BK];
    memset(rotated, 0, sizeof(rotated));
    for (int i = 0; i < dim; i++) rotated[i] = src[i] * inv_norm;

    /* Step 3: Apply RHT with expansion for small dim */
    uint32_t seed = TKV_DEFAULT_SEED;
    blk->rht_seed = seed;

    if (sketch_dim > dim) {
        int n_chunks = sketch_dim / dim;
        for (int c = 1; c < n_chunks; c++) {
            for (int i = 0; i < dim; i++) {
                rotated[c * dim + i] = rotated[i];
            }
        }
        for (int c = 0; c < n_chunks; c++) {
            uint32_t chunk_seed = seed + (uint32_t)c * 0x9ABCDEF0u;
            tq_rht_transform(rotated + c * dim, dim, chunk_seed);
        }
    } else {
        tq_rht_transform(rotated, dim, seed);
    }

    /* Step 4: 1-bit MSE quantization (sign only) over sketch_dim
     * After RHT, coordinates are ~N(0, 1/sqrt(dim)).
     * Expected |x| for half-normal = sqrt(2/pi) / sqrt(dim). */
    float scale = sqrtf(2.0f / TQ_PI) / sqrtf((float)dim);
    float residual[TQ_BK];
    memset(residual, 0, sizeof(residual));
    for (int i = 0; i < sketch_dim; i++) {
        int sign_bit = (rotated[i] > 0.0f) ? 1 : 0;
        if (sign_bit)
            blk->mse_indices[i / 8] |= (uint8_t)(1 << (i % 8));
        float recon = sign_bit ? scale : -scale;
        residual[i] = rotated[i] - recon;
    }

    /* Step 5: QJL 1-bit sign hash on residual */
    float res_norm_sq = 0.0f;
    for (int i = 0; i < sketch_dim; i++) res_norm_sq += residual[i] * residual[i];
    blk->residual_norm = tkv_fp32_to_fp16(sqrtf(res_norm_sq));

    for (int i = 0; i < sketch_dim; i++) {
        if (residual[i] > 0.0f) {
            blk->qjl_signs[i / 8] |= (uint8_t)(1 << (i % 8));
        }
    }
}

void tq_turbo_kv_2b_dequantize_ref(const void* src, float* dst, int n) {
    const int dim = (n < TQ_BK) ? n : TQ_BK;
    const block_tq_turbo_kv_2b* blk = (const block_tq_turbo_kv_2b*)src;

    int sketch_dim = dim;
    if (sketch_dim < TQ_BK) sketch_dim = TQ_BK;

    float norm = tkv_fp16_to_fp32(blk->norm);
    if (norm < 1e-12f) {
        memset(dst, 0, (size_t)dim * sizeof(float));
        for (int i = dim; i < n; i++) dst[i] = 0.0f;
        return;
    }

    /* Reconstruct from 1-bit MSE codebook (sign only) in rotated space.
     * After RHT, coordinates are ~N(0, 1/sqrt(dim)).
     * Expected |x| for half-normal = sqrt(2/pi) / sqrt(dim). */
    float scale = sqrtf(2.0f / TQ_PI) / sqrtf((float)dim);

    if (sketch_dim > dim) {
        int n_chunks = sketch_dim / dim;
        float accum[TQ_BK];
        memset(accum, 0, (size_t)dim * sizeof(float));

        for (int c = 0; c < n_chunks; c++) {
            float chunk[TQ_BK];
            for (int i = 0; i < dim; i++) {
                int si = c * dim + i;
                int sign_bit = (blk->mse_indices[si / 8] >> (si % 8)) & 1;
                chunk[i] = sign_bit ? scale : -scale;
            }
            uint32_t chunk_seed = blk->rht_seed + (uint32_t)c * 0x9ABCDEF0u;
            tq_rht_inverse(chunk, dim, chunk_seed);
            for (int i = 0; i < dim; i++) {
                accum[i] += chunk[i];
            }
        }

        float inv_chunks = 1.0f / (float)n_chunks;
        for (int i = 0; i < dim; i++) {
            dst[i] = accum[i] * inv_chunks * norm;
        }
    } else {
        float rotated[TQ_BK];
        memset(rotated, 0, sizeof(rotated));
        for (int i = 0; i < dim; i++) {
            int sign_bit = (blk->mse_indices[i / 8] >> (i % 8)) & 1;
            rotated[i] = sign_bit ? scale : -scale;
        }

        /* Inverse RHT */
        tq_rht_inverse(rotated, dim, blk->rht_seed);

        /* Scale by original norm */
        for (int i = 0; i < dim; i++) {
            dst[i] = rotated[i] * norm;
        }
    }
    for (int i = dim; i < n; i++) dst[i] = 0.0f;
}

void tq_turbo_kv_2b_attention_ref(const float* query, const void* kv,
                                   float* scores, int seq_len, int head_dim) {
    const int dim = (head_dim < TQ_BK) ? head_dim : TQ_BK;
    const block_tq_turbo_kv_2b* blocks = (const block_tq_turbo_kv_2b*)kv;

    /* Expand sketch_dim for small dimensions */
    int sketch_dim = dim;
    if (sketch_dim < TQ_BK) sketch_dim = TQ_BK;

    /* Compute query norm */
    float q_norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) q_norm_sq += query[i] * query[i];
    float q_norm = sqrtf(q_norm_sq);
    float scale_factor = 1.0f / (float)sketch_dim;

    /* RHT(query) with expansion matching quantize */
    float q_rot[TQ_BK];
    memset(q_rot, 0, sizeof(q_rot));
    float q_inv_norm = (q_norm > 1e-12f) ? (1.0f / q_norm) : 0.0f;
    for (int i = 0; i < dim; i++) q_rot[i] = query[i] * q_inv_norm;

    if (sketch_dim > dim) {
        int n_chunks = sketch_dim / dim;
        for (int c = 1; c < n_chunks; c++) {
            for (int i = 0; i < dim; i++) {
                q_rot[c * dim + i] = q_rot[i];
            }
        }
        for (int c = 0; c < n_chunks; c++) {
            uint32_t chunk_seed = TKV_DEFAULT_SEED + (uint32_t)c * 0x9ABCDEF0u;
            tq_rht_transform(q_rot + c * dim, dim, chunk_seed);
        }
    } else {
        tq_rht_transform(q_rot, dim, TKV_DEFAULT_SEED);
    }

    /* Extract query sign bits over sketch_dim */
    uint8_t q_signs[TQ_BK / 8];
    int sign_bytes = sketch_dim / 8;
    if (sign_bytes > 0) memset(q_signs, 0, (size_t)sign_bytes);
    for (int i = 0; i < sketch_dim; i++) {
        if (q_rot[i] > 0.0f) {
            q_signs[i / 8] |= (uint8_t)(1 << (i % 8));
        }
    }

    for (int seq = 0; seq < seq_len; seq++) {
        const block_tq_turbo_kv_2b* blk = &blocks[seq];
        float k_norm = tkv_fp16_to_fp32(blk->norm);

        /* 1-bit MSE attention: sign agreement over sketch_dim */
        int mse_hamming = 0;
        for (int b = 0; b < sign_bytes; b++) {
            uint8_t xor_byte = q_signs[b] ^ blk->mse_indices[b];
            int c = 0;
            uint8_t tmp = xor_byte;
            while (tmp) { c++; tmp &= tmp - 1; }
            mse_hamming += c;
        }
        int mse_agree = sketch_dim - mse_hamming;
        float mse_score = q_norm * k_norm * scale_factor * (float)(2 * mse_agree - sketch_dim);

        /* QJL residual correction */
        float res_norm = tkv_fp16_to_fp32(blk->residual_norm);
        int qjl_hamming = 0;
        for (int b = 0; b < sign_bytes; b++) {
            uint8_t xor_byte = q_signs[b] ^ blk->qjl_signs[b];
            int c = 0;
            uint8_t tmp = xor_byte;
            while (tmp) { c++; tmp &= tmp - 1; }
            qjl_hamming += c;
        }
        int qjl_agree = sketch_dim - qjl_hamming;
        float qjl_correction = q_norm * res_norm * scale_factor * (float)(2 * qjl_agree - sketch_dim);

        scores[seq] = mse_score + qjl_correction;
    }
}

/* ============================================================
 * TurboQuant KV 5-bit (Variant F architecture):
 *   normalize -> RHT -> 5-bit (32-level) Lloyd-Max codebook on rotated values
 * Single-stage estimator, no QJL residual.
 * ============================================================ */

/* Pack 5-bit indices into a bit-stream, LSB-first.
 * 128 elems × 5 bits = 640 bits = 80 bytes. */
static void pack_5bit(const uint8_t* indices, uint8_t* packed, int n) {
    int total_bytes = (n * 5 + 7) / 8;
    memset(packed, 0, (size_t)total_bytes);
    for (int i = 0; i < n; i++) {
        int bit_offset = i * 5;
        int byte_idx = bit_offset / 8;
        int bit_pos  = bit_offset % 8;
        uint16_t val = (uint16_t)(indices[i] & 0x1F);
        packed[byte_idx] |= (uint8_t)(val << bit_pos);
        if (bit_pos > 3) {
            packed[byte_idx + 1] |= (uint8_t)(val >> (8 - bit_pos));
        }
    }
}

static void unpack_5bit(const uint8_t* packed, uint8_t* indices, int n) {
    int total_bytes = (n * 5 + 7) / 8;
    for (int i = 0; i < n; i++) {
        int bit_offset = i * 5;
        int byte_idx = bit_offset / 8;
        int bit_pos  = bit_offset % 8;
        uint16_t val = (uint16_t)packed[byte_idx];
        if (bit_pos > 3 && byte_idx + 1 < total_bytes) {
            val |= (uint16_t)packed[byte_idx + 1] << 8;
        }
        indices[i] = (uint8_t)((val >> bit_pos) & 0x1F);
    }
}

void tq_turbo_kv_5b_quantize_ref(const float* src, void* dst, int n) {
    block_tq_turbo_kv_5b* block = (block_tq_turbo_kv_5b*)dst;
    int dim = n;
    if (dim > TQ_BK) dim = TQ_BK;

    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) norm_sq += src[i] * src[i];
    float norm = sqrtf(norm_sq);
    block->norm = tkv_fp32_to_fp16(norm);
    block->residual_norm = 0;
    block->_pad = 0;

    float rotated[TQ_BK];
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    for (int i = 0; i < dim; i++) rotated[i] = src[i] * inv_norm;
    for (int i = dim; i < TQ_BK; i++) rotated[i] = 0.0f;

    tq_rht_transform(rotated, dim, TKV_DEFAULT_SEED);

    /* Variant F: max-abs scaling, no clipping */
    float max_abs = 0.0f;
    for (int i = 0; i < dim; i++) {
        float a = fabsf(rotated[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-10f) max_abs = 1.0f;
    const float CENT_5BIT_MAX = 1.9956f;
    float inv_std = CENT_5BIT_MAX / max_abs;
    block->inv_std_fp16 = tkv_fp32_to_fp16(inv_std);

    uint8_t indices[TQ_BK];
    tq_codebook_quantize(rotated, indices, dim, 5, inv_std);
    pack_5bit(indices, block->mse_indices, dim);
}

static void dequant_mse_rotated_5bit(const block_tq_turbo_kv_5b* block,
                                      float* rotated, int dim) {
    float inv_std = tkv_fp16_to_fp32(block->inv_std_fp16);
    if (inv_std < 1e-10f) inv_std = sqrtf((float)dim);
    uint8_t indices[TQ_BK] = {0};
    unpack_5bit(block->mse_indices, indices, dim);
    tq_codebook_dequantize(indices, rotated, dim, 5, inv_std);
}

void tq_turbo_kv_5b_dequantize_ref(const void* src, float* dst, int n) {
    const block_tq_turbo_kv_5b* block = (const block_tq_turbo_kv_5b*)src;
    int dim = n;
    if (dim > TQ_BK) dim = TQ_BK;

    float norm = tkv_fp16_to_fp32(block->norm);

    float rotated[TQ_BK];
    dequant_mse_rotated_5bit(block, rotated, dim);
    tq_rht_inverse(rotated, dim, TKV_DEFAULT_SEED);

    for (int i = 0; i < dim; i++) dst[i] = rotated[i] * norm;
}

void tq_turbo_kv_5b_attention_ref(const float* query, const void* kv_cache,
                                    float* scores, int seq_len, int head_dim) {
    const block_tq_turbo_kv_5b* blocks_5b = (const block_tq_turbo_kv_5b*)kv_cache;
    int dim = head_dim;
    if (dim > TQ_BK) dim = TQ_BK;

    /* Pre-rotate query once */
    float q_rot[TQ_BK];
    memcpy(q_rot, query, (size_t)dim * sizeof(float));
    for (int i = dim; i < TQ_BK; i++) q_rot[i] = 0.0f;
    tq_rht_transform(q_rot, dim, TKV_DEFAULT_SEED);

    for (int seq = 0; seq < seq_len; seq++) {
        const block_tq_turbo_kv_5b* block = &blocks_5b[seq];
        float norm = tkv_fp16_to_fp32(block->norm);

        float rotated[TQ_BK];
        dequant_mse_rotated_5bit(block, rotated, dim);

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
            for (; d < dim; d++) mse_dot += q_rot[d] * rotated[d];
        }
#else
        for (int d = 0; d < dim; d++) mse_dot += q_rot[d] * rotated[d];
#endif
        scores[seq] = norm * mse_dot;
    }
}

/* ============================================================
 * TurboQuant KV 4-bit + outliers (Variant G):
 *   normalize -> RHT -> 4-bit (16-level) Lloyd-Max codebook
 *   + top-K outliers stored verbatim as FP16 with channel index
 *
 * Same Variant F base + per-block outlier list. The K largest |rotated|
 * channels are stored exactly and overwrite the codebook reconstruction
 * at dequant time. Closes more PPL gap than 4b-only without going as
 * heavy as 5b on memory.
 * ============================================================ */

void tq_turbo_kv_4bo_quantize_ref(const float* src, void* dst, int n) {
    block_tq_turbo_kv_4bo* block = (block_tq_turbo_kv_4bo*)dst;
    int dim = n;
    if (dim > TQ_BK) dim = TQ_BK;

    /* Step 1: L2 norm */
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) norm_sq += src[i] * src[i];
    float norm = sqrtf(norm_sq);
    block->norm = tkv_fp32_to_fp16(norm);
    block->residual_norm = 0;
    block->_pad = 0;

    /* Step 2: Normalize + RHT */
    float rotated[TQ_BK];
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    for (int i = 0; i < dim; i++) rotated[i] = src[i] * inv_norm;
    for (int i = dim; i < TQ_BK; i++) rotated[i] = 0.0f;
    tq_rht_transform(rotated, dim, TKV_DEFAULT_SEED);

    /* Step 3: Find top-K outliers by |rotated| (selection sort, K is small) */
    int K = TQ_KV_4BO_OUTLIERS;
    int out_idx[TQ_KV_4BO_OUTLIERS];
    float out_abs[TQ_KV_4BO_OUTLIERS];
    for (int k = 0; k < K; k++) { out_idx[k] = -1; out_abs[k] = -1.0f; }

    for (int i = 0; i < dim; i++) {
        float a = fabsf(rotated[i]);
        /* Find smallest in current top-K and replace if larger */
        int min_pos = 0;
        for (int k = 1; k < K; k++) {
            if (out_abs[k] < out_abs[min_pos]) min_pos = k;
        }
        if (a > out_abs[min_pos]) {
            out_abs[min_pos] = a;
            out_idx[min_pos] = i;
        }
    }

    /* Store outlier indices and FP16 values */
    for (int k = 0; k < K; k++) {
        int idx = out_idx[k];
        if (idx < 0) {
            block->out_indices[k] = 0;
            block->out_values[k] = 0;
        } else {
            block->out_indices[k] = (uint8_t)idx;
            block->out_values[k] = tkv_fp32_to_fp16(rotated[idx]);
        }
    }

    /* Step 4: max-abs scaling on the NON-outlier values for the codebook.
     * Outliers are stored exact, so the codebook only needs to fit the body.
     * Mask outliers out for max-abs computation. */
    char is_outlier[TQ_BK];
    memset(is_outlier, 0, sizeof(is_outlier));
    for (int k = 0; k < K; k++) {
        if (out_idx[k] >= 0) is_outlier[out_idx[k]] = 1;
    }

    float body_max_abs = 0.0f;
    for (int i = 0; i < dim; i++) {
        if (is_outlier[i]) continue;
        float a = fabsf(rotated[i]);
        if (a > body_max_abs) body_max_abs = a;
    }
    if (body_max_abs < 1e-10f) body_max_abs = 1.0f;
    const float CENT_4BIT_MAX = 2.7326f;
    float inv_std = CENT_4BIT_MAX / body_max_abs;
    block->inv_std_fp16 = tkv_fp32_to_fp16(inv_std);

    /* Step 5: Quantize all 128 with 4-bit codebook (outlier values get
     * overwritten at dequant time, so their codebook indices don't matter
     * for accuracy — but we still write something so the bytes are defined). */
    uint8_t indices[TQ_BK];
    tq_codebook_quantize(rotated, indices, dim, 4, inv_std);
    memset(block->mse_indices, 0, TQ_BK / 2);
    for (int i = 0; i < dim; i++) {
        int byte_idx = i / 2;
        int bit_pos  = (i & 1) * 4;
        block->mse_indices[byte_idx] |= (uint8_t)((indices[i] & 0x0F) << bit_pos);
    }
}

static void dequant_mse_rotated_4bo(const block_tq_turbo_kv_4bo* block,
                                     float* rotated, int dim) {
    /* 4-bit codebook lookup */
    float inv_std = tkv_fp16_to_fp32(block->inv_std_fp16);
    if (inv_std < 1e-10f) inv_std = sqrtf((float)dim);
    uint8_t indices[TQ_BK];
    for (int i = 0; i < dim; i++) {
        uint8_t b = block->mse_indices[i / 2];
        indices[i] = (i & 1) ? (b >> 4) : (b & 0x0F);
    }
    tq_codebook_dequantize(indices, rotated, dim, 4, inv_std);

    /* Overwrite outlier positions with stored exact FP16 values */
    int K = TQ_KV_4BO_OUTLIERS;
    for (int k = 0; k < K; k++) {
        int idx = block->out_indices[k];
        if (idx < dim) {
            rotated[idx] = tkv_fp16_to_fp32(block->out_values[k]);
        }
    }
}

void tq_turbo_kv_4bo_dequantize_ref(const void* src, float* dst, int n) {
    const block_tq_turbo_kv_4bo* block = (const block_tq_turbo_kv_4bo*)src;
    int dim = n;
    if (dim > TQ_BK) dim = TQ_BK;

    float norm = tkv_fp16_to_fp32(block->norm);
    float rotated[TQ_BK];
    dequant_mse_rotated_4bo(block, rotated, dim);
    tq_rht_inverse(rotated, dim, TKV_DEFAULT_SEED);
    for (int i = 0; i < dim; i++) dst[i] = rotated[i] * norm;
}

void tq_turbo_kv_4bo_attention_ref(const float* query, const void* kv_cache,
                                     float* scores, int seq_len, int head_dim) {
    const block_tq_turbo_kv_4bo* blocks_4bo = (const block_tq_turbo_kv_4bo*)kv_cache;
    int dim = head_dim;
    if (dim > TQ_BK) dim = TQ_BK;

    /* Pre-rotate query once */
    float q_rot[TQ_BK];
    memcpy(q_rot, query, (size_t)dim * sizeof(float));
    for (int i = dim; i < TQ_BK; i++) q_rot[i] = 0.0f;
    tq_rht_transform(q_rot, dim, TKV_DEFAULT_SEED);

    for (int seq = 0; seq < seq_len; seq++) {
        const block_tq_turbo_kv_4bo* block = &blocks_4bo[seq];
        float norm = tkv_fp16_to_fp32(block->norm);

        float rotated[TQ_BK];
        dequant_mse_rotated_4bo(block, rotated, dim);

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
            for (; d < dim; d++) mse_dot += q_rot[d] * rotated[d];
        }
#else
        for (int d = 0; d < dim; d++) mse_dot += q_rot[d] * rotated[d];
#endif
        scores[seq] = norm * mse_dot;
    }
}
