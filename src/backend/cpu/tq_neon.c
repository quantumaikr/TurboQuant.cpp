/**
 * ARM NEON optimized kernels for TurboQuant
 *
 * Real NEON implementations of uniform 4-bit and polar quantize/dequantize.
 * Only compiled when __ARM_NEON is defined.
 */

#include "turboquant/turboquant.h"
#include <math.h>
#include <string.h>
#include <float.h>

#ifdef __ARM_NEON
#include <arm_neon.h>

/* ---------- FP16 helpers (storage only, compute in FP32) ---------- */

static inline uint16_t neon_fp32_to_fp16(float v) {
    union { float f; uint32_t u; } bits;
    bits.f = v;
    uint32_t sign = (bits.u >> 16) & 0x8000;
    int32_t  exp  = ((bits.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits.u >> 13) & 0x03FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static inline float neon_fp16_to_fp32(uint16_t h) {
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

/* ================================================================
 * Uniform 4-bit quantize — NEON optimized
 *
 * Strategy: Use NEON to find min/max across the block in 4-wide
 * chunks, then NEON multiply+round for quantization, then scalar
 * for nibble packing.
 * ================================================================ */

void tq_uniform_4b_quantize_neon(const float* src, void* dst, int n) {
    block_tq_uniform_4b* block = (block_tq_uniform_4b*)dst;
    int count = n;
    if (count > TQ_BK) count = TQ_BK;

    /* --- Phase 1: NEON min/max reduction --- */
    float32x4_t vmin = vdupq_n_f32(FLT_MAX);
    float32x4_t vmax = vdupq_n_f32(-FLT_MAX);

    int i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        vmin = vminq_f32(vmin, v);
        vmax = vmaxq_f32(vmax, v);
    }

    /* Horizontal reduction */
    float mn = vminvq_f32(vmin);
    float mx = vmaxvq_f32(vmax);

    /* Handle remaining elements (scalar) */
    for (; i < count; i++) {
        if (src[i] < mn) mn = src[i];
        if (src[i] > mx) mx = src[i];
    }

    float range = mx - mn;
    if (range < 1e-8f) range = 1e-8f;
    float scale = range / 16.0f; /* 16 bins of width range/16 */
    float inv_scale = 1.0f / scale;

    block->scale      = neon_fp32_to_fp16(scale);
    block->zero_point = neon_fp32_to_fp16(mn);

    memset(block->qs, 0, TQ_BK / 2);

    /* --- Phase 2: NEON quantization --- */
    float32x4_t v_mn   = vdupq_n_f32(mn);
    float32x4_t v_invs = vdupq_n_f32(inv_scale);
    float32x4_t v_zero = vdupq_n_f32(0.0f);
    float32x4_t v_15   = vdupq_n_f32(15.0f);

    i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        /* q = floor((v - mn) * inv_scale), clamped to [0, 15] */
        float32x4_t shifted = vsubq_f32(v, v_mn);
        float32x4_t scaled  = vmulq_f32(shifted, v_invs);
        /* Floor to int (NEON vcvtmq rounds toward -infinity) */
        int32x4_t qi = vcvtmq_s32_f32(scaled);
        /* Clamp */
        float32x4_t qf = vcvtq_f32_s32(qi);
        qf = vmaxq_f32(qf, v_zero);
        qf = vminq_f32(qf, v_15);
        int32x4_t qclamped = vcvtq_s32_f32(qf);

        /* Pack nibbles — scalar for correct LSB-first ordering */
        int q0 = vgetq_lane_s32(qclamped, 0);
        int q1 = vgetq_lane_s32(qclamped, 1);
        int q2 = vgetq_lane_s32(qclamped, 2);
        int q3 = vgetq_lane_s32(qclamped, 3);

        /* Two values per byte, LSB-first */
        block->qs[(i + 0) / 2] |= (uint8_t)((i + 0) % 2 == 0 ? q0 : (q0 << 4));
        block->qs[(i + 1) / 2] |= (uint8_t)((i + 1) % 2 == 0 ? q1 : (q1 << 4));
        block->qs[(i + 2) / 2] |= (uint8_t)((i + 2) % 2 == 0 ? q2 : (q2 << 4));
        block->qs[(i + 3) / 2] |= (uint8_t)((i + 3) % 2 == 0 ? q3 : (q3 << 4));
    }

    /* Scalar tail */
    for (; i < count; i++) {
        int q = (int)floorf((src[i] - mn) * inv_scale);
        if (q < 0)  q = 0;
        if (q > 15) q = 15;
        if (i % 2 == 0) {
            block->qs[i / 2] = (uint8_t)q;
        } else {
            block->qs[i / 2] |= (uint8_t)(q << 4);
        }
    }
}

/* ================================================================
 * Uniform 4-bit dequantize — NEON optimized
 *
 * Unpacks nibbles, converts to float, and does scale*q + mn
 * using NEON multiply-add.
 * ================================================================ */

void tq_uniform_4b_dequantize_neon(const void* src, float* dst, int n) {
    const block_tq_uniform_4b* block = (const block_tq_uniform_4b*)src;
    int count = n;
    if (count > TQ_BK) count = TQ_BK;

    float scale = neon_fp16_to_fp32(block->scale);
    float mn    = neon_fp16_to_fp32(block->zero_point);

    float32x4_t v_scale = vdupq_n_f32(scale);
    float32x4_t v_mn    = vdupq_n_f32(mn);

    int i = 0;
    for (; i + 8 <= count; i += 8) {
        /* Load 4 bytes = 8 nibbles */
        uint8_t b0 = block->qs[i / 2 + 0];
        uint8_t b1 = block->qs[i / 2 + 1];
        uint8_t b2 = block->qs[i / 2 + 2];
        uint8_t b3 = block->qs[i / 2 + 3];

        /* Unpack 8 nibbles into two float32x4, add 0.5 for bin centering */
        float q_arr[8] = {
            (float)(b0 & 0x0F) + 0.5f, (float)(b0 >> 4) + 0.5f,
            (float)(b1 & 0x0F) + 0.5f, (float)(b1 >> 4) + 0.5f,
            (float)(b2 & 0x0F) + 0.5f, (float)(b2 >> 4) + 0.5f,
            (float)(b3 & 0x0F) + 0.5f, (float)(b3 >> 4) + 0.5f,
        };

        float32x4_t q_lo = vld1q_f32(q_arr);
        float32x4_t q_hi = vld1q_f32(q_arr + 4);

        /* dst = mn + (q + 0.5) * scale  (using fused multiply-add) */
        float32x4_t r_lo = vfmaq_f32(v_mn, q_lo, v_scale);
        float32x4_t r_hi = vfmaq_f32(v_mn, q_hi, v_scale);

        vst1q_f32(dst + i,     r_lo);
        vst1q_f32(dst + i + 4, r_hi);
    }

    /* Scalar tail */
    for (; i < count; i++) {
        uint8_t byte = block->qs[i / 2];
        int q = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        dst[i] = mn + ((float)q + 0.5f) * scale;
    }
}

/* ================================================================
 * PolarQuant quantize — NEON optimized
 *
 * Uses NEON for:
 * - Computing radius via vsqrt (x*x + y*y)
 * - NEON min/max for scale computation
 * - NEON multiply+round for quantization
 *
 * atan2 is computed using a polynomial approximation suitable
 * for NEON. We use a 3rd-order rational approximation:
 *   atan(x) ~ x * (A + B*x^2) / (1 + C*x^2)
 * where A = 1.0, B = 0.28125, C = 0.28125 (minimax approx)
 * Then map via atan2 quadrant logic.
 * ================================================================ */

/* NEON atan2 approximation: process 4 (x, y) pairs at once.
 * Returns atan2(y, x) in [-pi, pi] range. */
static inline float32x4_t neon_atan2_approx(float32x4_t vy, float32x4_t vx) {
    /* Constants */
    const float32x4_t v_pi      = vdupq_n_f32(TQ_PI);
    const float32x4_t v_half_pi = vdupq_n_f32(TQ_PI_2);
    const float32x4_t v_zero    = vdupq_n_f32(0.0f);
    (void)0; /* v_one removed — was unused */
    /* Polynomial coefficients for atan(z) where |z| <= 1
     * Approximation: z * (0.9998660 - 0.3302995 * z^2) / (1 + 0.0003271 * z^2)
     * Simplified to Horner form for speed:
     *   atan(z) ~ z - z^3/3 + z^5/5  (Taylor) is okay but
     * we use a better minimax: p(z) = z*(1.0 + a*z^2) where a ~ -0.3183
     * Actually use the classic 7th-order: too slow.
     * Use the Abramowitz & Stegun 3-term approx for |z|<=1:
     *   atan(z) ~ z * (C0 + C1*z^2 + C2*z^4) */
    const float32x4_t C0 = vdupq_n_f32(0.99997726f);
    const float32x4_t C1 = vdupq_n_f32(-0.33262347f);
    const float32x4_t C2 = vdupq_n_f32(0.19354346f);
    const float32x4_t C3 = vdupq_n_f32(-0.11643287f);
    const float32x4_t C4 = vdupq_n_f32(0.05265332f);
    const float32x4_t C5 = vdupq_n_f32(-0.01172120f);

    float32x4_t ax = vabsq_f32(vx);
    float32x4_t ay = vabsq_f32(vy);

    /* Compute min/max to ensure |z| <= 1 */
    float32x4_t mn = vminq_f32(ax, ay);
    float32x4_t mx = vmaxq_f32(ax, ay);

    /* Avoid division by zero */
    mx = vmaxq_f32(mx, vdupq_n_f32(1e-20f));

    float32x4_t z = vdivq_f32(mn, mx);
    float32x4_t z2 = vmulq_f32(z, z);

    /* Evaluate polynomial: atan(z) = z * (C0 + z2*(C1 + z2*(C2 + z2*(C3 + z2*(C4 + z2*C5))))) */
    float32x4_t p = vfmaq_f32(C4, C5, z2);
    p = vfmaq_f32(C3, p, z2);
    p = vfmaq_f32(C2, p, z2);
    p = vfmaq_f32(C1, p, z2);
    p = vfmaq_f32(C0, p, z2);
    float32x4_t a = vmulq_f32(z, p);

    /* If ay > ax, result is pi/2 - a */
    uint32x4_t swap_mask = vcgtq_f32(ay, ax);
    a = vbslq_f32(swap_mask, vsubq_f32(v_half_pi, a), a);

    /* If x < 0, result is pi - a */
    uint32x4_t xneg_mask = vcltq_f32(vx, v_zero);
    a = vbslq_f32(xneg_mask, vsubq_f32(v_pi, a), a);

    /* If y < 0, negate result */
    uint32x4_t yneg_mask = vcltq_f32(vy, v_zero);
    a = vbslq_f32(yneg_mask, vnegq_f32(a), a);

    return a;
}

/* NEON-accelerated square root for 4 floats */
static inline float32x4_t neon_sqrt_f32(float32x4_t v) {
    return vsqrtq_f32(v);
}

void tq_polar_quantize_neon(const float* src, void* dst, int n) {
    block_tq_polar* block = (block_tq_polar*)dst;
    int pairs = n / 2;
    if (pairs > TQ_BK / 2) pairs = TQ_BK / 2;

    /* Compute polar coordinates with NEON */
    float thetas[TQ_BK / 2];
    float radii[TQ_BK / 2];

    int p = 0;
    /* Process 4 pairs at a time (8 floats = 4 x,y pairs) */
    for (; p + 4 <= pairs; p += 4) {
        /* Load 8 floats: x0,y0, x1,y1, x2,y2, x3,y3 */
        float32x4x2_t xy = vld2q_f32(src + 2 * p);
        float32x4_t vx = xy.val[0]; /* x0, x1, x2, x3 */
        float32x4_t vy = xy.val[1]; /* y0, y1, y2, y3 */

        /* Radius = sqrt(x*x + y*y) */
        float32x4_t r2 = vfmaq_f32(vmulq_f32(vx, vx), vy, vy);
        float32x4_t vr = neon_sqrt_f32(r2);

        /* Theta = atan2(y, x) using NEON approximation, shifted to [0, 2pi] */
        float32x4_t vt = neon_atan2_approx(vy, vx);
        {
            float32x4_t v_2pi = vdupq_n_f32(2.0f * TQ_PI);
            float32x4_t v_zero_f = vdupq_n_f32(0.0f);
            uint32x4_t neg_mask = vcltq_f32(vt, v_zero_f);
            vt = vaddq_f32(vt, vbslq_f32(neg_mask, v_2pi, v_zero_f));
        }

        vst1q_f32(thetas + p, vt);
        vst1q_f32(radii + p, vr);
    }

    /* Scalar tail */
    for (; p < pairs; p++) {
        float x = src[2 * p];
        float y = src[2 * p + 1];
        radii[p]  = sqrtf(x * x + y * y);
        float t = atan2f(y, x);
        if (t < 0.0f) t += 2.0f * TQ_PI;
        thetas[p] = t;
    }

    /* Find min/max of theta and radius using NEON */
    float32x4_t vtmin = vdupq_n_f32(FLT_MAX);
    float32x4_t vtmax = vdupq_n_f32(-FLT_MAX);
    float32x4_t vrmin = vdupq_n_f32(FLT_MAX);
    float32x4_t vrmax = vdupq_n_f32(-FLT_MAX);

    p = 0;
    for (; p + 4 <= pairs; p += 4) {
        float32x4_t t = vld1q_f32(thetas + p);
        float32x4_t r = vld1q_f32(radii + p);
        vtmin = vminq_f32(vtmin, t);
        vtmax = vmaxq_f32(vtmax, t);
        vrmin = vminq_f32(vrmin, r);
        vrmax = vmaxq_f32(vrmax, r);
    }

    float tmin = vminvq_f32(vtmin);
    float tmax = vmaxvq_f32(vtmax);
    float rmin = vminvq_f32(vrmin);
    float rmax = vmaxvq_f32(vrmax);

    for (; p < pairs; p++) {
        if (thetas[p] < tmin) tmin = thetas[p];
        if (thetas[p] > tmax) tmax = thetas[p];
        if (radii[p] < rmin) rmin = radii[p];
        if (radii[p] > rmax) rmax = radii[p];
    }

    float trange = tmax - tmin;
    float rrange = rmax - rmin;
    if (trange < 1e-8f) trange = 1e-8f;
    if (rrange < 1e-8f) rrange = 1e-8f;

    float tscale = trange / 4.0f;
    float rscale = rrange / 4.0f;

    block->tscale = neon_fp32_to_fp16(tscale);
    block->tmn    = neon_fp32_to_fp16(tmin);
    block->rscale = neon_fp32_to_fp16(rscale);
    block->rmn    = neon_fp32_to_fp16(rmin);

    memset(block->indices, 0, TQ_BK / 2);

    /* Quantize with NEON: compute (val - min) / scale, round, clamp to [0, 3] */
    float32x4_t v_tmin = vdupq_n_f32(tmin);
    float32x4_t v_rmin = vdupq_n_f32(rmin);
    float32x4_t v_tinvs = vdupq_n_f32(1.0f / tscale);
    float32x4_t v_rinvs = vdupq_n_f32(1.0f / rscale);
    float32x4_t v_zero = vdupq_n_f32(0.0f);
    float32x4_t v_three = vdupq_n_f32(3.0f);

    p = 0;
    for (; p + 4 <= pairs; p += 4) {
        float32x4_t t = vld1q_f32(thetas + p);
        float32x4_t r = vld1q_f32(radii + p);

        /* Quantize theta: tq = floor((t - tmin) / tscale), clamp [0,3] */
        float32x4_t tq_f = vmulq_f32(vsubq_f32(t, v_tmin), v_tinvs);
        int32x4_t tq_i = vcvtmq_s32_f32(tq_f);
        float32x4_t tq_clamped = vmaxq_f32(vminq_f32(vcvtq_f32_s32(tq_i), v_three), v_zero);
        int32x4_t tq = vcvtq_s32_f32(tq_clamped);

        /* Quantize radius: rq = floor((r - rmin) / rscale), clamp [0,3] */
        float32x4_t rq_f = vmulq_f32(vsubq_f32(r, v_rmin), v_rinvs);
        int32x4_t rq_i = vcvtmq_s32_f32(rq_f);
        float32x4_t rq_clamped = vmaxq_f32(vminq_f32(vcvtq_f32_s32(rq_i), v_three), v_zero);
        int32x4_t rq = vcvtq_s32_f32(rq_clamped);

        /* Pack: rho in upper 2 bits, theta in lower 2 bits */
        for (int k = 0; k < 4; k++) {
            int tidx = vgetq_lane_s32(tq, 0);
            int ridx = vgetq_lane_s32(rq, 0);
            tq = vextq_s32(tq, tq, 1);
            rq = vextq_s32(rq, rq, 1);

            uint8_t packed = (uint8_t)((ridx << 2) | tidx);
            int idx = p + k;
            if (idx % 2 == 0) {
                block->indices[idx / 2] = packed;
            } else {
                block->indices[idx / 2] |= (packed << 4);
            }
        }
    }

    /* Scalar tail */
    for (; p < pairs; p++) {
        int tq = (int)floorf((thetas[p] - tmin) / tscale);
        int rq = (int)floorf((radii[p] - rmin) / rscale);
        if (tq < 0) tq = 0; if (tq > 3) tq = 3;
        if (rq < 0) rq = 0; if (rq > 3) rq = 3;

        uint8_t packed = (uint8_t)((rq << 2) | tq);
        if (p % 2 == 0) {
            block->indices[p / 2] = packed;
        } else {
            block->indices[p / 2] |= (packed << 4);
        }
    }
}

/* ================================================================
 * PolarQuant dequantize — NEON optimized
 *
 * Uses NEON for cos/sin computation on reconstructed angles
 * and NEON multiply for radius scaling.
 * ================================================================ */

void tq_polar_dequantize_neon(const void* src, float* dst, int n) {
    const block_tq_polar* block = (const block_tq_polar*)src;
    int pairs = n / 2;
    if (pairs > TQ_BK / 2) pairs = TQ_BK / 2;

    float tscale = neon_fp16_to_fp32(block->tscale);
    float tmin   = neon_fp16_to_fp32(block->tmn);
    float rscale = neon_fp16_to_fp32(block->rscale);
    float rmin   = neon_fp16_to_fp32(block->rmn);

    /* Dequantize (scalar — cos/sin are hard to vectorize cheaply) */
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

/* ================================================================
 * QJL quantize — NEON optimized
 *
 * Uses NEON for:
 * - L2 norm computation (4-wide FMA accumulation)
 * - Outlier detection (4-wide absolute value comparison)
 * - Projection dot products with Rademacher random variables
 * ================================================================ */

/* Deterministic pseudo-random projection (same as reference) */
static inline float neon_qjl_random_entry(int dim_idx, int sketch_idx) {
    uint32_t h = (uint32_t)(dim_idx * 2654435761u + sketch_idx * 340573321u);
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    return (h & 1) ? 1.0f : -1.0f;
}

void tq_qjl_quantize_neon(const float* src, void* dst, int n) {
    block_tq_qjl* block = (block_tq_qjl*)dst;
    int dim = n;
    if (dim > TQ_BK_QJL) dim = TQ_BK_QJL;

    /* L2 norm with NEON accumulation */
    float32x4_t norm_acc = vdupq_n_f32(0.0f);
    int d = 0;
    for (; d + 4 <= dim; d += 4) {
        float32x4_t v = vld1q_f32(src + d);
        norm_acc = vfmaq_f32(norm_acc, v, v);
    }
    float norm_sq = vaddvq_f32(norm_acc);
    for (; d < dim; d++) {
        norm_sq += src[d] * src[d];
    }
    block->norm = neon_fp32_to_fp16(sqrtf(norm_sq));

    /* Find outliers: top-k absolute values */
    float abs_vals[TQ_BK_QJL];
    d = 0;
    for (; d + 4 <= dim; d += 4) {
        float32x4_t v = vld1q_f32(src + d);
        float32x4_t av = vabsq_f32(v);
        vst1q_f32(abs_vals + d, av);
    }
    for (; d < dim; d++) abs_vals[d] = fabsf(src[d]);
    for (d = dim; d < TQ_BK_QJL; d++) abs_vals[d] = 0.0f;

    float outlier_norm_sq = 0.0f;
    for (int o = 0; o < TQ_OUTLIERS; o++) {
        int best = 0;
        float best_val = -1.0f;
        for (d = 0; d < dim; d++) {
            if (abs_vals[d] > best_val) {
                best_val = abs_vals[d];
                best = d;
            }
        }
        block->outlier_idx[o] = (uint8_t)(best < 256 ? best : 255);
        outlier_norm_sq += src[best] * src[best];
        abs_vals[best] = -1.0f;
    }
    block->outlier_norm = neon_fp32_to_fp16(sqrtf(outlier_norm_sq));

    /* Compute sign hash using NEON dot products */
    memset(block->hash, 0, TQ_SKETCH_DIM / 8);
    for (int s = 0; s < TQ_SKETCH_DIM; s++) {
        /* Accumulate projection: sum_d src[d] * random(d, s)
         * Since random returns +1/-1, this is effectively
         * sum of src[d] where random=+1, minus sum where random=-1 */
        float32x4_t proj_acc = vdupq_n_f32(0.0f);
        d = 0;
        for (; d + 4 <= dim; d += 4) {
            float32x4_t v = vld1q_f32(src + d);
            /* Generate 4 random signs */
            float signs[4];
            signs[0] = neon_qjl_random_entry(d + 0, s);
            signs[1] = neon_qjl_random_entry(d + 1, s);
            signs[2] = neon_qjl_random_entry(d + 2, s);
            signs[3] = neon_qjl_random_entry(d + 3, s);
            float32x4_t vs = vld1q_f32(signs);
            proj_acc = vfmaq_f32(proj_acc, v, vs);
        }
        float proj = vaddvq_f32(proj_acc);
        for (; d < dim; d++) {
            proj += src[d] * neon_qjl_random_entry(d, s);
        }
        if (proj >= 0.0f) {
            block->hash[s / 8] |= (1 << (s % 8));
        }
    }
}

/* ================================================================
 * QJL attention — NEON optimized with XOR + popcount
 * ================================================================ */

void tq_qjl_attention_neon(const float* query, const void* kv_cache,
                           float* scores, int seq_len, int head_dim) {
    const block_tq_qjl* blocks = (const block_tq_qjl*)kv_cache;
    int dim = head_dim;
    if (dim > TQ_BK_QJL) dim = TQ_BK_QJL;

    /* Precompute query projections */
    float q_proj[TQ_SKETCH_DIM];
    for (int s = 0; s < TQ_SKETCH_DIM; s++) {
        float32x4_t proj_acc = vdupq_n_f32(0.0f);
        int d = 0;
        for (; d + 4 <= dim; d += 4) {
            float32x4_t v = vld1q_f32(query + d);
            float signs[4];
            signs[0] = neon_qjl_random_entry(d + 0, s);
            signs[1] = neon_qjl_random_entry(d + 1, s);
            signs[2] = neon_qjl_random_entry(d + 2, s);
            signs[3] = neon_qjl_random_entry(d + 3, s);
            float32x4_t vs = vld1q_f32(signs);
            proj_acc = vfmaq_f32(proj_acc, v, vs);
        }
        float proj = vaddvq_f32(proj_acc);
        for (; d < dim; d++) {
            proj += query[d] * neon_qjl_random_entry(d, s);
        }
        q_proj[s] = proj;
    }

    /* Query norm */
    float32x4_t qn_acc = vdupq_n_f32(0.0f);
    int d = 0;
    for (; d + 4 <= dim; d += 4) {
        float32x4_t v = vld1q_f32(query + d);
        qn_acc = vfmaq_f32(qn_acc, v, v);
    }
    float q_norm_sq = vaddvq_f32(qn_acc);
    for (; d < dim; d++) q_norm_sq += query[d] * query[d];
    float q_norm = sqrtf(q_norm_sq);

    /* Precompute query sign bits for XOR-popcount */
    uint8_t q_sign_bits[TQ_SKETCH_DIM / 8];
    memset(q_sign_bits, 0, TQ_SKETCH_DIM / 8);
    for (int s = 0; s < TQ_SKETCH_DIM; s++) {
        if (q_proj[s] >= 0.0f) {
            q_sign_bits[s / 8] |= (1 << (s % 8));
        }
    }

    /* For each key: XOR + popcount to count agreements */
    for (int s = 0; s < seq_len; s++) {
        const block_tq_qjl* block = &blocks[s];
        float key_norm = neon_fp16_to_fp32(block->norm);

        /* NEON XOR + popcount */
        int total_agree = 0;
        int bytes = TQ_SKETCH_DIM / 8;
        int b = 0;

        /* Process 16 bytes at a time with NEON */
        for (; b + 16 <= bytes; b += 16) {
            uint8x16_t kbits = vld1q_u8(block->hash + b);
            uint8x16_t qbits = vld1q_u8(q_sign_bits + b);
            /* XOR then invert = XNOR (agreement bits) */
            uint8x16_t xor_result = veorq_u8(kbits, qbits);
            /* Count set bits in XOR result (disagreements) */
            uint8x16_t popcnt = vcntq_u8(xor_result);
            /* Sum all popcounts */
            uint16x8_t sum16 = vpaddlq_u8(popcnt);
            uint32x4_t sum32 = vpaddlq_u16(sum16);
            uint64x2_t sum64 = vpaddlq_u32(sum32);
            int disagree = (int)(vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1));
            /* Agreements = total_bits - disagreements */
            total_agree += (16 * 8) - disagree;
        }

        /* Scalar tail */
        for (; b < bytes; b++) {
            uint8_t xor_val = block->hash[b] ^ q_sign_bits[b];
            /* Count agreements = 8 - popcount(xor) */
            int popcnt = 0;
            uint8_t tmp = xor_val;
            while (tmp) { popcnt += tmp & 1; tmp >>= 1; }
            total_agree += 8 - popcnt;
        }

        float frac = (float)total_agree / TQ_SKETCH_DIM;
        float cos_est = cosf(TQ_PI * (1.0f - frac));
        scores[s] = cos_est * q_norm * key_norm;
    }
}

/* ================================================================
 * PolarQuant attention — NEON optimized (LUT + NEON FMA)
 *
 * Precompute cos/sin LUT for 4 theta levels and radius LUT for
 * 4 rho levels, then use NEON gather+FMA to accumulate dot product.
 * Processes 4 pairs at a time (8 query elements).
 * ================================================================ */

void tq_polar_attention_neon(const float* query, const void* kv_cache,
                              float* scores, int seq_len, int head_dim) {
    const block_tq_polar* blocks = (const block_tq_polar*)kv_cache;
    int pairs = head_dim / 2;
    if (pairs > TQ_BK / 2) pairs = TQ_BK / 2;

    for (int s = 0; s < seq_len; s++) {
        const block_tq_polar* block = &blocks[s];

        /* Decode block parameters from FP16 */
        float tscale = neon_fp16_to_fp32(block->tscale);
        float tmin   = neon_fp16_to_fp32(block->tmn);
        float rscale = neon_fp16_to_fp32(block->rscale);
        float rmin   = neon_fp16_to_fp32(block->rmn);

        /* Precompute cos/sin LUT for 4 theta levels */
        float cos_lut[4], sin_lut[4];
        for (int q = 0; q < 4; q++) {
            float theta = tmin + ((float)q + 0.5f) * tscale;
            cos_lut[q] = cosf(theta);
            sin_lut[q] = sinf(theta);
        }

        /* Precompute radius LUT */
        float radius_lut[4];
        for (int q = 0; q < 4; q++) {
            radius_lut[q] = rmin + ((float)q + 0.5f) * rscale;
        }

        /* Accumulate dot product using NEON
         * Process 4 pairs at a time: gather cos/sin/radius from LUT,
         * multiply with query pairs, accumulate with FMA */
        float32x4_t score_acc = vdupq_n_f32(0.0f);
        int i = 0;

        for (; i + 4 <= pairs; i += 4) {
            /* Extract 4 packed indices */
            uint8_t byte0 = block->indices[i / 2];
            uint8_t byte1 = block->indices[i / 2 + 1];

            int tq0 = (byte0 & 0x0F) & 0x03;
            int rq0 = ((byte0 & 0x0F) >> 2) & 0x03;
            int tq1 = (byte0 >> 4) & 0x03;
            int rq1 = ((byte0 >> 4) >> 2) & 0x03;
            int tq2 = (byte1 & 0x0F) & 0x03;
            int rq2 = ((byte1 & 0x0F) >> 2) & 0x03;
            int tq3 = (byte1 >> 4) & 0x03;
            int rq3 = ((byte1 >> 4) >> 2) & 0x03;

            /* Load 4 pairs of query values using deinterleave load */
            float32x4x2_t qpairs = vld2q_f32(query + 2 * i);
            float32x4_t q_x = qpairs.val[0]; /* query[0], query[2], query[4], query[6] */
            float32x4_t q_y = qpairs.val[1]; /* query[1], query[3], query[5], query[7] */

            /* Gather cos values from LUT */
            float cos_gathered[4] = { cos_lut[tq0], cos_lut[tq1], cos_lut[tq2], cos_lut[tq3] };
            float sin_gathered[4] = { sin_lut[tq0], sin_lut[tq1], sin_lut[tq2], sin_lut[tq3] };
            float rad_gathered[4] = { radius_lut[rq0], radius_lut[rq1], radius_lut[rq2], radius_lut[rq3] };

            float32x4_t v_cos = vld1q_f32(cos_gathered);
            float32x4_t v_sin = vld1q_f32(sin_gathered);
            float32x4_t v_rad = vld1q_f32(rad_gathered);

            /* contrib = radius * (q_x * cos + q_y * sin) */
            float32x4_t dot = vmulq_f32(q_x, v_cos);
            dot = vfmaq_f32(dot, q_y, v_sin);
            dot = vmulq_f32(dot, v_rad);

            score_acc = vaddq_f32(score_acc, dot);
        }

        /* Horizontal sum of accumulator */
        float score = vaddvq_f32(score_acc);

        /* Scalar tail */
        for (; i < pairs; i++) {
            uint8_t byte = block->indices[i / 2];
            uint8_t packed = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            int tq = packed & 0x03;
            int rq = (packed >> 2) & 0x03;

            float contrib = query[2 * i] * cos_lut[tq] + query[2 * i + 1] * sin_lut[tq];
            contrib *= radius_lut[rq];
            score += contrib;
        }

        scores[s] = score;
    }
}

/* ================================================================
 * Uniform 4-bit fused dequant+dot attention — NEON optimized
 *
 * Instead of dequantizing to a temp buffer then computing dot product,
 * this fuses both operations: unpack nibbles, scale+offset, and
 * dot product with query all in one pass using NEON FMA.
 * ================================================================ */

void tq_uniform_4b_attention_neon(const float* query, const void* kv_cache,
                                   float* scores, int seq_len, int head_dim) {
    const block_tq_uniform_4b* blocks = (const block_tq_uniform_4b*)kv_cache;
    int count = head_dim;
    if (count > TQ_BK) count = TQ_BK;

    /* Nibble mask for extracting low 4 bits */
    const uint8x16_t mask_0f = vdupq_n_u8(0x0F);

    for (int s = 0; s < seq_len; s++) {
        const block_tq_uniform_4b* block = &blocks[s];

        float scale = neon_fp16_to_fp32(block->scale);
        float mn    = neon_fp16_to_fp32(block->zero_point);

        float32x4_t v_scale = vdupq_n_f32(scale);
        float32x4_t v_mn    = vdupq_n_f32(mn + 0.5f * scale); /* bin centering */
        float32x4_t dot_acc0 = vdupq_n_f32(0.0f);
        float32x4_t dot_acc1 = vdupq_n_f32(0.0f);

        int i = 0;
        /* Process 32 elements at a time (16 bytes of packed data) */
        for (; i + 32 <= count; i += 32) {
            /* Load 16 bytes = 32 nibbles */
            uint8x16_t packed = vld1q_u8(block->qs + i / 2);

            /* Unpack: low nibbles and high nibbles */
            uint8x16_t lo_nib = vandq_u8(packed, mask_0f);
            uint8x16_t hi_nib = vshrq_n_u8(packed, 4);

            /* Interleave to get original order: lo0, hi0, lo1, hi1, ... */
            uint8x16x2_t zipped = vzipq_u8(lo_nib, hi_nib);

            /* Convert first 16 uint8 values to float32 (4 groups of 4) */
            /* Group 0: elements i..i+3 */
            uint16x8_t w0 = vmovl_u8(vget_low_u8(zipped.val[0]));
            uint32x4_t d0 = vmovl_u16(vget_low_u16(w0));
            float32x4_t f0 = vcvtq_f32_u32(d0);
            float32x4_t v0 = vfmaq_f32(v_mn, f0, v_scale);
            float32x4_t q0 = vld1q_f32(query + i);
            dot_acc0 = vfmaq_f32(dot_acc0, v0, q0);

            /* Group 1: elements i+4..i+7 */
            uint32x4_t d1 = vmovl_u16(vget_high_u16(w0));
            float32x4_t f1 = vcvtq_f32_u32(d1);
            float32x4_t v1 = vfmaq_f32(v_mn, f1, v_scale);
            float32x4_t q1 = vld1q_f32(query + i + 4);
            dot_acc1 = vfmaq_f32(dot_acc1, v1, q1);

            /* Group 2: elements i+8..i+11 */
            uint16x8_t w1 = vmovl_u8(vget_high_u8(zipped.val[0]));
            uint32x4_t d2 = vmovl_u16(vget_low_u16(w1));
            float32x4_t f2 = vcvtq_f32_u32(d2);
            float32x4_t v2 = vfmaq_f32(v_mn, f2, v_scale);
            float32x4_t q2 = vld1q_f32(query + i + 8);
            dot_acc0 = vfmaq_f32(dot_acc0, v2, q2);

            /* Group 3: elements i+12..i+15 */
            uint32x4_t d3 = vmovl_u16(vget_high_u16(w1));
            float32x4_t f3 = vcvtq_f32_u32(d3);
            float32x4_t v3 = vfmaq_f32(v_mn, f3, v_scale);
            float32x4_t q3 = vld1q_f32(query + i + 12);
            dot_acc1 = vfmaq_f32(dot_acc1, v3, q3);

            /* Group 4: elements i+16..i+19 */
            uint16x8_t w2 = vmovl_u8(vget_low_u8(zipped.val[1]));
            uint32x4_t d4 = vmovl_u16(vget_low_u16(w2));
            float32x4_t f4 = vcvtq_f32_u32(d4);
            float32x4_t v4 = vfmaq_f32(v_mn, f4, v_scale);
            float32x4_t q4 = vld1q_f32(query + i + 16);
            dot_acc0 = vfmaq_f32(dot_acc0, v4, q4);

            /* Group 5: elements i+20..i+23 */
            uint32x4_t d5 = vmovl_u16(vget_high_u16(w2));
            float32x4_t f5 = vcvtq_f32_u32(d5);
            float32x4_t v5 = vfmaq_f32(v_mn, f5, v_scale);
            float32x4_t q5 = vld1q_f32(query + i + 20);
            dot_acc1 = vfmaq_f32(dot_acc1, v5, q5);

            /* Group 6: elements i+24..i+27 */
            uint16x8_t w3 = vmovl_u8(vget_high_u8(zipped.val[1]));
            uint32x4_t d6 = vmovl_u16(vget_low_u16(w3));
            float32x4_t f6 = vcvtq_f32_u32(d6);
            float32x4_t v6 = vfmaq_f32(v_mn, f6, v_scale);
            float32x4_t q6 = vld1q_f32(query + i + 24);
            dot_acc0 = vfmaq_f32(dot_acc0, v6, q6);

            /* Group 7: elements i+28..i+31 */
            uint32x4_t d7 = vmovl_u16(vget_high_u16(w3));
            float32x4_t f7 = vcvtq_f32_u32(d7);
            float32x4_t v7 = vfmaq_f32(v_mn, f7, v_scale);
            float32x4_t q7 = vld1q_f32(query + i + 28);
            dot_acc1 = vfmaq_f32(dot_acc1, v7, q7);
        }

        /* Process 8 elements at a time for remainder */
        for (; i + 8 <= count; i += 8) {
            uint8_t b0 = block->qs[i / 2 + 0];
            uint8_t b1 = block->qs[i / 2 + 1];
            uint8_t b2 = block->qs[i / 2 + 2];
            uint8_t b3 = block->qs[i / 2 + 3];

            float q_arr[8] = {
                (float)(b0 & 0x0F), (float)(b0 >> 4),
                (float)(b1 & 0x0F), (float)(b1 >> 4),
                (float)(b2 & 0x0F), (float)(b2 >> 4),
                (float)(b3 & 0x0F), (float)(b3 >> 4),
            };

            float32x4_t ql = vld1q_f32(q_arr);
            float32x4_t qh = vld1q_f32(q_arr + 4);
            float32x4_t vl = vfmaq_f32(v_mn, ql, v_scale);
            float32x4_t vh = vfmaq_f32(v_mn, qh, v_scale);
            dot_acc0 = vfmaq_f32(dot_acc0, vl, vld1q_f32(query + i));
            dot_acc1 = vfmaq_f32(dot_acc1, vh, vld1q_f32(query + i + 4));
        }

        /* Horizontal sum of both accumulators */
        float dot = vaddvq_f32(vaddq_f32(dot_acc0, dot_acc1));

        /* Scalar tail */
        for (; i < count; i++) {
            uint8_t byte = block->qs[i / 2];
            int q = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            float val = mn + ((float)q + 0.5f) * scale;
            dot += query[i] * val;
        }

        scores[s] = dot;
    }
}

#else /* !__ARM_NEON */

/* Provide empty symbols to avoid linker errors on non-ARM */
/* These should never be called; dispatch will use ref versions */

#endif /* __ARM_NEON */
