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

/* ---------- Uniform 2-bit quantize ---------- */

void tq_uniform_2b_quantize_ref(const float* src, void* dst, int n) {
    block_tq_uniform_2b* block = (block_tq_uniform_2b*)dst;
    int count = n;
    if (count > TQ_BK) count = TQ_BK;

    float mn = FLT_MAX, mx = -FLT_MAX;
    for (int i = 0; i < count; i++) {
        if (src[i] < mn) mn = src[i];
        if (src[i] > mx) mx = src[i];
    }

    float range = mx - mn;
    if (range < 1e-8f) range = 1e-8f;
    float scale = range / 4.0f; /* 2-bit: 4 bins of width range/4 */

    block->scale      = uni_fp32_to_fp16(scale);
    block->zero_point = uni_fp32_to_fp16(mn);

    memset(block->qs, 0, TQ_BK / 4);
    for (int i = 0; i < count; i++) {
        int q = (int)floorf((src[i] - mn) / scale);
        if (q < 0) q = 0;
        if (q > 3) q = 3;
        /* LSB-first packing: four 2-bit values per byte */
        int pos = i % 4;
        block->qs[i / 4] |= (uint8_t)(q << (pos * 2));
    }
}

/* ---------- Uniform 2-bit dequantize ---------- */

void tq_uniform_2b_dequantize_ref(const void* src, float* dst, int n) {
    const block_tq_uniform_2b* block = (const block_tq_uniform_2b*)src;
    int count = n;
    if (count > TQ_BK) count = TQ_BK;

    float scale = uni_fp16_to_fp32(block->scale);
    float mn    = uni_fp16_to_fp32(block->zero_point);

    for (int i = 0; i < count; i++) {
        uint8_t byte = block->qs[i / 4];
        int pos = i % 4;
        int q = (byte >> (pos * 2)) & 0x03;
        dst[i] = mn + ((float)q + 0.5f) * scale;
    }
}

/* ---------- Uniform 4-bit attention (dequantize + dot product) ---------- */

void tq_uniform_4b_attention_ref(const float* query, const void* kv,
                                  float* scores, int seq_len, int head_dim) {
    const block_tq_uniform_4b* blocks = (const block_tq_uniform_4b*)kv;
    for (int s = 0; s < seq_len; s++) {
        float deq[256]; /* max head_dim */
        tq_uniform_4b_dequantize_ref(&blocks[s], deq, head_dim);
        float dot = 0;
        for (int d = 0; d < head_dim; d++) dot += query[d] * deq[d];
        scores[s] = dot;
    }
}

/* ---------- Uniform 2-bit attention (dequantize + dot product) ---------- */

void tq_uniform_2b_attention_ref(const float* query, const void* kv,
                                  float* scores, int seq_len, int head_dim) {
    const block_tq_uniform_2b* blocks = (const block_tq_uniform_2b*)kv;
    for (int s = 0; s < seq_len; s++) {
        float deq[256]; /* max head_dim */
        tq_uniform_2b_dequantize_ref(&blocks[s], deq, head_dim);
        float dot = 0;
        for (int d = 0; d < head_dim; d++) dot += query[d] * deq[d];
        scores[s] = dot;
    }
}
