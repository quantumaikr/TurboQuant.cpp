/**
 * Uniform min-max quantization — reference C implementation
 *
 * Simple baseline quantizer: find min/max, linearly map to 2^bits levels.
 * NOTE: This is the GENERIC reference. Compiler auto-vectorization is disabled
 * so that SIMD speedup measurement is meaningful.
 */
/* Generic reference — no compiler-specific pragmas */

#include "turboquant/turboquant.h"
#include <math.h>
#include <string.h>
#include <float.h>

/* ---------- FP16 helpers ---------- */

static uint16_t uni_fp32_to_fp16(float v) {
    union { float f; uint32_t u; } bits;
    bits.f = v;
    uint32_t sign = (bits.u >> 16) & 0x8000;
    int32_t  exp  = ((bits.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits.u >> 13) & 0x03FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static float uni_fp16_to_fp32(uint16_t h) {
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

/* ---------- Uniform 4-bit quantize ---------- */

void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n) {
    block_tq_uniform_4b* block = (block_tq_uniform_4b*)dst;
    int count = n;
    if (count > TQ_BK) count = TQ_BK;

    float mn = FLT_MAX, mx = -FLT_MAX;
    for (int i = 0; i < count; i++) {
        if (src[i] < mn) mn = src[i];
        if (src[i] > mx) mx = src[i];
    }

    float range = mx - mn;
    if (range < 1e-8f) range = 1e-8f;
    float scale = range / 16.0f; /* 4-bit: 16 bins of width range/16 */

    block->scale      = uni_fp32_to_fp16(scale);
    block->zero_point = uni_fp32_to_fp16(mn);

    memset(block->qs, 0, TQ_BK / 2);
    for (int i = 0; i < count; i++) {
        int q = (int)floorf((src[i] - mn) / scale);
        if (q < 0)  q = 0;
        if (q > 15) q = 15;
        /* LSB-first packing: two 4-bit values per byte */
        if (i % 2 == 0) {
            block->qs[i / 2] = (uint8_t)q;
        } else {
            block->qs[i / 2] |= (uint8_t)(q << 4);
        }
    }
}

/* ---------- Uniform 4-bit dequantize ---------- */

void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n) {
    const block_tq_uniform_4b* block = (const block_tq_uniform_4b*)src;
    int count = n;
    if (count > TQ_BK) count = TQ_BK;

    float scale = uni_fp16_to_fp32(block->scale);
    float mn    = uni_fp16_to_fp32(block->zero_point);

    for (int i = 0; i < count; i++) {
        uint8_t byte = block->qs[i / 2];
        int q = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        dst[i] = mn + ((float)q + 0.5f) * scale;
    }
}

/* ---------- Uniform 2-bit quantize (sub-block scales) ---------- */

void tq_uniform_2b_quantize_ref(const float* src, void* dst, int n) {
    block_tq_uniform_2b* block = (block_tq_uniform_2b*)dst;
    int count = n;
    if (count > TQ_BK) count = TQ_BK;

    /* Compute per-sub-block min/max and store FP16 scale/min */
    for (int sb = 0; sb < TQ_2B_NSUB; sb++) {
        int start = sb * TQ_2B_SUBK;
        int end = start + TQ_2B_SUBK;
        if (end > count) end = count;
        float mn = FLT_MAX, mx = -FLT_MAX;
        for (int i = start; i < end; i++) {
            if (src[i] < mn) mn = src[i];
            if (src[i] > mx) mx = src[i];
        }
        if (end <= start) { mn = 0; mx = 0; }

        float range = mx - mn;
        if (range < 1e-8f) range = 1e-8f;
        float scale = range / 4.0f; /* 2-bit: 4 bins of width range/4 */

        block->sub_scale[sb] = uni_fp32_to_fp16(scale);
        block->sub_min[sb]   = uni_fp32_to_fp16(mn);
    }

    /* Pack 2-bit quantized values using FP16-reconstructed scale/min */
    memset(block->qs, 0, TQ_BK / 4);
    for (int i = 0; i < count; i++) {
        int sb = i / TQ_2B_SUBK;
        float scale = uni_fp16_to_fp32(block->sub_scale[sb]);
        float mn    = uni_fp16_to_fp32(block->sub_min[sb]);
        if (scale < 1e-10f) scale = 1e-10f;

        int q = (int)floorf((src[i] - mn) / scale);
        if (q < 0) q = 0;
        if (q > 3) q = 3;
        /* LSB-first packing: four 2-bit values per byte */
        int pos = i % 4;
        block->qs[i / 4] |= (uint8_t)(q << (pos * 2));
    }
}

/* ---------- Uniform 2-bit dequantize (sub-block scales) ---------- */

void tq_uniform_2b_dequantize_ref(const void* src, float* dst, int n) {
    const block_tq_uniform_2b* block = (const block_tq_uniform_2b*)src;
    int count = n;
    if (count > TQ_BK) count = TQ_BK;

    for (int i = 0; i < count; i++) {
        int sb = i / TQ_2B_SUBK;
        float scale = uni_fp16_to_fp32(block->sub_scale[sb]);
        float mn    = uni_fp16_to_fp32(block->sub_min[sb]);

        uint8_t byte = block->qs[i / 4];
        int pos = i % 4;
        int q = (byte >> (pos * 2)) & 0x03;
        dst[i] = mn + ((float)q + 0.5f) * scale;
    }
}

/* ---------- Q8 query quantization for integer-domain attention ---------- */

void tq_quantize_query_q8(const float* query, int8_t* q8_out,
                           float* scale_out, float* sum_out, int n) {
    /* Find absolute max */
    float amax = 0;
    float qsum = 0;
    for (int i = 0; i < n; i++) {
        float a = fabsf(query[i]);
        if (a > amax) amax = a;
        qsum += query[i];
    }

    float scale = amax / 127.0f;
    if (scale < 1e-10f) scale = 1e-10f;
    float inv_scale = 1.0f / scale;

    for (int i = 0; i < n; i++) {
        int v = (int)roundf(query[i] * inv_scale);
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        q8_out[i] = (int8_t)v;
    }

    *scale_out = scale;
    *sum_out = qsum;
}

/* ---------- Integer-domain Q4xQ8 attention (no dequantization!) ---------- */

/* The key insight: query is quantized ONCE to Q8, then reused for all seq_len keys.
 * Original dequantized value = mn + (q4 + 0.5) * k_scale
 * So: dot = sum(query[i] * (mn + (q4+0.5)*k_scale))
 *         = k_scale * sum(query[i] * q4) + (mn + 0.5*k_scale) * sum(query[i])
 * With Q8 query: query[i] ~ q8[i] * q_scale
 *   dot ~ q_scale * k_scale * isum + (mn + 0.5*k_scale) * q_sum
 * where isum = sum(q8[i] * q4[i]) computed in integer domain.
 */
void tq_uniform_4b_attention_int_ref(const float* query, const void* kv,
                                      float* scores, int seq_len, int head_dim) {
    /* Step 1: Quantize query to Q8 (once, amortized over seq_len) */
    int8_t q8[512]; /* max head_dim supported */
    float q_scale, q_sum;
    tq_quantize_query_q8(query, q8, &q_scale, &q_sum, head_dim);

    int blocks_per_key = (head_dim + TQ_BK - 1) / TQ_BK;
    const block_tq_uniform_4b* all_blocks = (const block_tq_uniform_4b*)kv;

    for (int s = 0; s < seq_len; s++) {
        float score = 0;
        for (int b = 0; b < blocks_per_key; b++) {
            int offset = b * TQ_BK;
            int chunk = (head_dim - offset > TQ_BK) ? TQ_BK : (head_dim - offset);
            const block_tq_uniform_4b* block = &all_blocks[s * blocks_per_key + b];

            float k_scale = uni_fp16_to_fp32(block->scale);
            float k_zp    = uni_fp16_to_fp32(block->zero_point);

            /* Integer dot product (no dequantize!) */
            int32_t isum = 0;
            for (int i = 0; i < chunk / 2; i++) {
                uint8_t packed = block->qs[i];
                isum += (int32_t)(packed & 0x0F) * (int32_t)q8[offset + 2*i];
                isum += (int32_t)(packed >> 4)   * (int32_t)q8[offset + 2*i + 1];
            }

            /* Partial query sum for this block's zero-point correction */
            float block_q_sum = 0;
            for (int d = 0; d < chunk; d++) block_q_sum += query[offset + d];

            score += (float)isum * k_scale * q_scale + (k_zp + 0.5f * k_scale) * block_q_sum;
        }
        scores[s] = score;
    }
}

/* ---------- Uniform 4-bit attention (dequantize + dot product) ---------- */

void tq_uniform_4b_attention_ref(const float* query, const void* kv,
                                  float* scores, int seq_len, int head_dim) {
    int blocks_per_key = (head_dim + TQ_BK - 1) / TQ_BK;
    const block_tq_uniform_4b* all_blocks = (const block_tq_uniform_4b*)kv;

    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int b = 0; b < blocks_per_key; b++) {
            int offset = b * TQ_BK;
            int chunk = (head_dim - offset > TQ_BK) ? TQ_BK : (head_dim - offset);

            float deq[TQ_BK];
            tq_uniform_4b_dequantize_ref(&all_blocks[s * blocks_per_key + b], deq, chunk);

            for (int d = 0; d < chunk; d++)
                dot += query[offset + d] * deq[d];
        }
        scores[s] = dot;
    }
}

/* ---------- Uniform 2-bit attention (dequantize + dot product) ---------- */

void tq_uniform_2b_attention_ref(const float* query, const void* kv,
                                  float* scores, int seq_len, int head_dim) {
    int blocks_per_key = (head_dim + TQ_BK - 1) / TQ_BK;
    const block_tq_uniform_2b* all_blocks = (const block_tq_uniform_2b*)kv;

    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int b = 0; b < blocks_per_key; b++) {
            int offset = b * TQ_BK;
            int chunk = (head_dim - offset > TQ_BK) ? TQ_BK : (head_dim - offset);

            float deq[TQ_BK];
            tq_uniform_2b_dequantize_ref(&all_blocks[s * blocks_per_key + b], deq, chunk);

            for (int d = 0; d < chunk; d++)
                dot += query[offset + d] * deq[d];
        }
        scores[s] = dot;
    }
}

/* ====================================================================
 * Uniform 3-bit with per-sub-block FP16 scales (Q3_K-style)
 *
 * Each 128-element block is split into 4 sub-blocks of 32 elements.
 * Each sub-block has independent FP16 scale and minimum, giving
 * excellent adaptation to local value distributions.
 *
 * 8 quantization levels (3-bit) per value.
 * 64 bytes / 128 elements = 4.0 bpe.
 *
 * Compared to uniform_4b (4.0 bpe, 16 levels, 1 global scale):
 * - Fewer levels (8 vs 16) but finer per-sub-block adaptation
 * - Better for heterogeneous distributions within a head dimension
 * ==================================================================== */

/* ---------- Uniform 3-bit sub-block quantize ---------- */

void tq_uniform_3b_quantize_ref(const float* src, void* dst, int n) {
    block_tq_uniform_3b* block = (block_tq_uniform_3b*)dst;
    int count = n;
    if (count > TQ_BK) count = TQ_BK;

    /* Compute per-sub-block min/max and store FP16 scale/min */
    for (int sb = 0; sb < TQ_3B_NSUB; sb++) {
        int start = sb * TQ_3B_SUBK;
        int end = start + TQ_3B_SUBK;
        if (end > count) end = count;
        float mn = FLT_MAX, mx = -FLT_MAX;
        for (int i = start; i < end; i++) {
            if (src[i] < mn) mn = src[i];
            if (src[i] > mx) mx = src[i];
        }
        if (end <= start) { mn = 0; mx = 0; }

        float range = mx - mn;
        if (range < 1e-8f) range = 1e-8f;
        float scale = range / 8.0f; /* 3-bit: 8 bins of width range/8 */

        block->sub_scale[sb] = uni_fp32_to_fp16(scale);
        block->sub_min[sb]   = uni_fp32_to_fp16(mn);
    }

    /* Pack 3-bit quantized values into qs (LSB-first).
     * Use the FP16-reconstructed scale/min for quantization
     * to minimize encode/decode mismatch.
     */
    memset(block->qs, 0, TQ_BK * 3 / 8);
    for (int i = 0; i < count; i++) {
        int sb = i / TQ_3B_SUBK;
        float scale = uni_fp16_to_fp32(block->sub_scale[sb]);
        float mn    = uni_fp16_to_fp32(block->sub_min[sb]);
        if (scale < 1e-10f) scale = 1e-10f;

        int q = (int)floorf((src[i] - mn) / scale);
        if (q < 0) q = 0;
        if (q > 7) q = 7;

        /* 3-bit packing: element i uses bits [i*3 .. i*3+2] across qs bytes */
        int bit_pos = i * 3;
        int byte_idx = bit_pos / 8;
        int bit_off  = bit_pos % 8;
        block->qs[byte_idx] |= (uint8_t)(q << bit_off);
        /* Handle cross-byte boundary (when bit_off > 5, bits spill into next byte) */
        if (bit_off > 5 && byte_idx + 1 < TQ_BK * 3 / 8) {
            block->qs[byte_idx + 1] |= (uint8_t)(q >> (8 - bit_off));
        }
    }
}

/* ---------- Uniform 3-bit sub-block dequantize ---------- */

void tq_uniform_3b_dequantize_ref(const void* src, float* dst, int n) {
    const block_tq_uniform_3b* block = (const block_tq_uniform_3b*)src;
    int count = n;
    if (count > TQ_BK) count = TQ_BK;

    for (int i = 0; i < count; i++) {
        int sb = i / TQ_3B_SUBK;
        float scale = uni_fp16_to_fp32(block->sub_scale[sb]);
        float mn    = uni_fp16_to_fp32(block->sub_min[sb]);

        /* Extract 3-bit value */
        int bit_pos = i * 3;
        int byte_idx = bit_pos / 8;
        int bit_off  = bit_pos % 8;
        int q = (block->qs[byte_idx] >> bit_off) & 0x07;
        if (bit_off > 5 && byte_idx + 1 < TQ_BK * 3 / 8) {
            q |= (block->qs[byte_idx + 1] << (8 - bit_off)) & 0x07;
        }

        dst[i] = mn + ((float)q + 0.5f) * scale;
    }
}

/* ---------- Uniform 3-bit attention (dequantize + dot product) ---------- */

void tq_uniform_3b_attention_ref(const float* query, const void* kv,
                                  float* scores, int seq_len, int head_dim) {
    int blocks_per_key = (head_dim + TQ_BK - 1) / TQ_BK;
    const block_tq_uniform_3b* all_blocks = (const block_tq_uniform_3b*)kv;

    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int b = 0; b < blocks_per_key; b++) {
            int offset = b * TQ_BK;
            int chunk = (head_dim - offset > TQ_BK) ? TQ_BK : (head_dim - offset);

            float deq[TQ_BK];
            tq_uniform_3b_dequantize_ref(&all_blocks[s * blocks_per_key + b], deq, chunk);

            for (int dd = 0; dd < chunk; dd++)
                dot += query[offset + dd] * deq[dd];
        }
        scores[s] = dot;
    }
}
