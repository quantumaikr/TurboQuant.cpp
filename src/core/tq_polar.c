/**
 * PolarQuant quantization — reference C implementation
 *
 * Quantizes key vectors using polar coordinates (theta, rho).
 * Each pair of adjacent values is converted to (angle, radius),
 * then quantized independently.
 */

#include "turboquant/turboquant.h"
#include <math.h>
#include <string.h>
#include <float.h>

/* ---------- FP16 helpers (storage only, compute in FP32) ---------- */

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
    if (exp == 0) {
        bits.u = sign;
        return bits.f;
    }
    if (exp == 31) {
        bits.u = sign | 0x7F800000 | (mant << 13);
        return bits.f;
    }
    exp = exp - 15 + 127;
    bits.u = sign | (exp << 23) | (mant << 13);
    return bits.f;
}

/* ---------- PolarQuant quantize (reference) ---------- */

void tq_polar_quantize_ref(const float* src, void* dst, int n) {
    block_tq_polar* block = (block_tq_polar*)dst;
    int pairs = n / 2;
    if (pairs > TQ_BK / 2) pairs = TQ_BK / 2;

    /* Compute polar coordinates for each pair */
    float thetas[TQ_BK / 2];
    float radii[TQ_BK / 2];
    float tmin = FLT_MAX, tmax = -FLT_MAX;
    float rmin = FLT_MAX, rmax = -FLT_MAX;

    for (int i = 0; i < pairs; i++) {
        float x = src[2 * i];
        float y = src[2 * i + 1];
        float r = sqrtf(x * x + y * y);
        float t = atan2f(y, x); /* range: [-pi, pi] */
        if (t < 0.0f) t += 2.0f * TQ_PI; /* shift to [0, 2pi] */

        thetas[i] = t;
        radii[i]  = r;

        if (t < tmin) tmin = t;
        if (t > tmax) tmax = t;
        if (r < rmin) rmin = r;
        if (r > rmax) rmax = r;
    }

    /* Quantize theta to 2 bits (4 levels), rho to 2 bits (4 levels) */
    float trange = tmax - tmin;
    float rrange = rmax - rmin;
    if (trange < 1e-8f) trange = 1e-8f;
    if (rrange < 1e-8f) rrange = 1e-8f;

    float tscale = trange / 4.0f; /* 4 bins of width range/4 */
    float rscale = rrange / 4.0f;

    block->tscale = fp32_to_fp16(tscale);
    block->tmn    = fp32_to_fp16(tmin);
    block->rscale = fp32_to_fp16(rscale);
    block->rmn    = fp32_to_fp16(rmin);

    memset(block->indices, 0, TQ_BK / 2);

    for (int i = 0; i < pairs; i++) {
        int tq = (int)floorf((thetas[i] - tmin) / tscale);
        int rq = (int)floorf((radii[i] - rmin) / rscale);
        if (tq < 0) { tq = 0; }
        if (tq > 3) { tq = 3; }
        if (rq < 0) { rq = 0; }
        if (rq > 3) { rq = 3; }

        /* Pack: rho in upper 2 bits, theta in lower 2 bits = 4 bits per pair */
        uint8_t packed = (uint8_t)((rq << 2) | tq);
        /* Two pairs per byte, LSB-first */
        if (i % 2 == 0) {
            block->indices[i / 2] = packed;
        } else {
            block->indices[i / 2] |= (packed << 4);
        }
    }
}

/* ---------- PolarQuant dequantize (reference) ---------- */

void tq_polar_dequantize_ref(const void* src, float* dst, int n) {
    const block_tq_polar* block = (const block_tq_polar*)src;
    int pairs = n / 2;
    if (pairs > TQ_BK / 2) pairs = TQ_BK / 2;

    float tscale = fp16_to_fp32(block->tscale);
    float tmin   = fp16_to_fp32(block->tmn);
    float rscale = fp16_to_fp32(block->rscale);
    float rmin   = fp16_to_fp32(block->rmn);

    for (int i = 0; i < pairs; i++) {
        uint8_t byte = block->indices[i / 2];
        uint8_t packed = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        int tq = packed & 0x03;
        int rq = (packed >> 2) & 0x03;

        float theta  = tmin + ((float)tq + 0.5f) * tscale;
        float radius = rmin + ((float)rq + 0.5f) * rscale;

        dst[2 * i]     = radius * cosf(theta);
        dst[2 * i + 1] = radius * sinf(theta);
    }
}

/* ---------- PolarQuant attention (direct LUT-based) ---------- */

void tq_polar_attention_ref(const float* query, const void* kv_cache,
                            float* scores, int seq_len, int head_dim) {
    /* Each key may span multiple blocks when head_dim > TQ_BK.
     * We precompute cos/sin/radius lookup tables per block and gather by index. */
    int blocks_per_key = (head_dim + TQ_BK - 1) / TQ_BK;
    const block_tq_polar* all_blocks = (const block_tq_polar*)kv_cache;

    /* Theta uses 2 bits (4 levels), rho uses 2 bits (4 levels) */
    const int theta_levels = 4;
    const int rho_levels = 4;

    for (int s = 0; s < seq_len; s++) {
        float score = 0.0f;

        for (int blk = 0; blk < blocks_per_key; blk++) {
            int offset = blk * TQ_BK;
            int chunk = (head_dim - offset > TQ_BK) ? TQ_BK : (head_dim - offset);
            int pairs = chunk / 2;

            const block_tq_polar* block = &all_blocks[s * blocks_per_key + blk];

            /* Decode block parameters from FP16 */
            float tscale = fp16_to_fp32(block->tscale);
            float tmin   = fp16_to_fp32(block->tmn);
            float rscale = fp16_to_fp32(block->rscale);
            float rmin   = fp16_to_fp32(block->rmn);

            /* Precompute theta lookup tables */
            float cos_lut[4], sin_lut[4];
            for (int q = 0; q < theta_levels; q++) {
                float theta = tmin + ((float)q + 0.5f) * tscale;
                cos_lut[q] = cosf(theta);
                sin_lut[q] = sinf(theta);
            }

            /* Precompute radius lookup table */
            float radius_lut[4];
            for (int q = 0; q < rho_levels; q++) {
                radius_lut[q] = rmin + ((float)q + 0.5f) * rscale;
            }

            /* For each pair, gather from LUT by index and accumulate */
            for (int i = 0; i < pairs; i++) {
                uint8_t byte = block->indices[i / 2];
                uint8_t packed = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
                int tq = packed & 0x03;
                int rq = (packed >> 2) & 0x03;

                float contrib = query[offset + 2 * i] * cos_lut[tq]
                              + query[offset + 2 * i + 1] * sin_lut[tq];
                contrib *= radius_lut[rq];
                score += contrib;
            }
        }

        scores[s] = score;
    }
}
