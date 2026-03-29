/**
 * Mixed precision outlier quantization — reference C implementation
 *
 * Strategy: detect top-k channels by absolute value (outliers), store them
 * at fp16 precision. Remaining channels use 4-bit uniform quantization
 * with a tighter min-max range (computed WITHOUT outliers). This yields
 * much lower MSE than standard uniform_4b on data with heavy-tailed
 * distributions (which is typical for real LLM KV caches).
 */

#include "turboquant/turboquant.h"
#include <math.h>
#include <string.h>
#include <float.h>

/* ---------- FP16 helpers (same pattern as tq_uniform.c) ---------- */

static uint16_t mixed_fp32_to_fp16(float v) {
    union { float f; uint32_t u; } bits;
    bits.f = v;
    uint32_t sign = (bits.u >> 16) & 0x8000;
    int32_t  exp  = ((bits.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits.u >> 13) & 0x03FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static float mixed_fp16_to_fp32(uint16_t h) {
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

/* ---------- Helper: check if index is in outlier set ---------- */

static int is_outlier_idx(const uint8_t* outlier_idx, int num_outliers, int i) {
    for (int o = 0; o < num_outliers; o++) {
        if (outlier_idx[o] == (uint8_t)i) return 1;
    }
    return 0;
}

/* ---------- Mixed 4b8 quantize ---------- */

void tq_mixed_4b8_quantize_ref(const float* src, void* dst, int n) {
    block_tq_mixed_4b8* block = (block_tq_mixed_4b8*)dst;
    int count = n;
    if (count > TQ_BK) count = TQ_BK;

    /* Step 1: Find top-4 outlier channels by absolute value */
    uint8_t outlier_idx[TQ_MIXED_OUTLIERS];
    memset(outlier_idx, 0, sizeof(outlier_idx));

    for (int o = 0; o < TQ_MIXED_OUTLIERS; o++) {
        float max_abs = -1.0f;
        int   max_i   = 0;
        for (int i = 0; i < count; i++) {
            /* Skip indices already picked */
            int already = 0;
            for (int k = 0; k < o; k++) {
                if (outlier_idx[k] == (uint8_t)i) { already = 1; break; }
            }
            if (already) continue;
            float a = fabsf(src[i]);
            if (a > max_abs) { max_abs = a; max_i = i; }
        }
        outlier_idx[o] = (uint8_t)max_i;
        block->outlier_idx[o] = (uint8_t)max_i;
        block->outlier_vals[o] = (int16_t)mixed_fp32_to_fp16(src[max_i]);
    }

    /* Step 2: Compute min-max EXCLUDING outlier channels */
    float mn = FLT_MAX, mx = -FLT_MAX;
    int non_outlier_count = 0;
    for (int i = 0; i < count; i++) {
        if (is_outlier_idx(outlier_idx, TQ_MIXED_OUTLIERS, i)) continue;
        if (src[i] < mn) mn = src[i];
        if (src[i] > mx) mx = src[i];
        non_outlier_count++;
    }

    /* Handle edge case: all values are outliers or very small block */
    if (non_outlier_count == 0 || mn > mx) {
        mn = 0.0f;
        mx = 1e-8f;
    }

    float range = mx - mn;
    if (range < 1e-8f) range = 1e-8f;
    float scale = range / 16.0f; /* 4-bit: 16 bins */

    block->scale      = mixed_fp32_to_fp16(scale);
    block->zero_point = mixed_fp32_to_fp16(mn);

    /* Step 3: Quantize to 4-bit. Outlier positions get 0 in the packed array. */
    memset(block->qs, 0, TQ_BK / 2);
    for (int i = 0; i < count; i++) {
        int q;
        if (is_outlier_idx(outlier_idx, TQ_MIXED_OUTLIERS, i)) {
            q = 0; /* placeholder — outlier is reconstructed from outlier_vals */
        } else {
            q = (int)floorf((src[i] - mn) / scale);
            if (q < 0)  q = 0;
            if (q > 15) q = 15;
        }
        /* LSB-first packing: two 4-bit values per byte */
        if (i % 2 == 0) {
            block->qs[i / 2] = (uint8_t)q;
        } else {
            block->qs[i / 2] |= (uint8_t)(q << 4);
        }
    }
}

/* ---------- Mixed 4b8 dequantize ---------- */

void tq_mixed_4b8_dequantize_ref(const void* src, float* dst, int n) {
    const block_tq_mixed_4b8* block = (const block_tq_mixed_4b8*)src;
    int count = n;
    if (count > TQ_BK) count = TQ_BK;

    float scale = mixed_fp16_to_fp32(block->scale);
    float mn    = mixed_fp16_to_fp32(block->zero_point);

    /* First dequantize all as 4-bit uniform */
    for (int i = 0; i < count; i++) {
        uint8_t byte = block->qs[i / 2];
        int q = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        dst[i] = mn + ((float)q + 0.5f) * scale;
    }

    /* Override outlier positions with fp16 values */
    for (int o = 0; o < TQ_MIXED_OUTLIERS; o++) {
        int idx = (int)block->outlier_idx[o];
        if (idx < count) {
            dst[idx] = mixed_fp16_to_fp32((uint16_t)block->outlier_vals[o]);
        }
    }
}

/* ---------- Mixed 4b8 attention (dequantize + dot product) ---------- */

void tq_mixed_4b8_attention_ref(const float* query, const void* kv,
                                 float* scores, int seq_len, int head_dim) {
    int blocks_per_key = (head_dim + TQ_BK - 1) / TQ_BK;
    const block_tq_mixed_4b8* all_blocks = (const block_tq_mixed_4b8*)kv;

    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int b = 0; b < blocks_per_key; b++) {
            int offset = b * TQ_BK;
            int chunk = (head_dim - offset > TQ_BK) ? TQ_BK : (head_dim - offset);

            float deq[TQ_BK];
            tq_mixed_4b8_dequantize_ref(&all_blocks[s * blocks_per_key + b], deq, chunk);

            for (int d = 0; d < chunk; d++)
                dot += query[offset + d] * deq[d];
        }
        scores[s] = dot;
    }
}
