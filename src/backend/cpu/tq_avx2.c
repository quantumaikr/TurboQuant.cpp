/**
 * x86 AVX2 optimized kernels for TurboQuant
 *
 * AVX2 implementations of quantize/dequantize.
 * Only compiled when __AVX2__ is defined.
 */

#include "turboquant/turboquant.h"
#include <math.h>
#include <string.h>
#include <float.h>

#ifdef __AVX2__
#include <immintrin.h>

/* ---------- FP16 helpers ---------- */

static inline uint16_t avx_fp32_to_fp16(float v) {
    union { float f; uint32_t u; } bits;
    bits.f = v;
    uint32_t sign = (bits.u >> 16) & 0x8000;
    int32_t  exp  = ((bits.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits.u >> 13) & 0x03FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static inline float avx_fp16_to_fp32(uint16_t h) {
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

/* Helper: horizontal min of __m256 */
static inline float hmin_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 mn = _mm_min_ps(lo, hi);
    mn = _mm_min_ps(mn, _mm_shuffle_ps(mn, mn, _MM_SHUFFLE(2, 3, 0, 1)));
    mn = _mm_min_ps(mn, _mm_shuffle_ps(mn, mn, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(mn);
}

/* Helper: horizontal max of __m256 */
static inline float hmax_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 mx = _mm_max_ps(lo, hi);
    mx = _mm_max_ps(mx, _mm_shuffle_ps(mx, mx, _MM_SHUFFLE(2, 3, 0, 1)));
    mx = _mm_max_ps(mx, _mm_shuffle_ps(mx, mx, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(mx);
}

/* Helper: horizontal sum of __m256 */
static inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(2, 3, 0, 1)));
    s = _mm_add_ps(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(s);
}

/* ================================================================
 * Uniform 4-bit quantize — AVX2 optimized (8-wide)
 * ================================================================ */

void tq_uniform_4b_quantize_avx2(const float* src, void* dst, int n) {
    block_tq_uniform_4b* block = (block_tq_uniform_4b*)dst;
    int count = n;
    if (count > TQ_BK) count = TQ_BK;

    /* Phase 1: 8-wide min/max */
    __m256 vmin = _mm256_set1_ps(FLT_MAX);
    __m256 vmax = _mm256_set1_ps(-FLT_MAX);

    int i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        vmin = _mm256_min_ps(vmin, v);
        vmax = _mm256_max_ps(vmax, v);
    }

    float mn = hmin_ps(vmin);
    float mx = hmax_ps(vmax);

    for (; i < count; i++) {
        if (src[i] < mn) mn = src[i];
        if (src[i] > mx) mx = src[i];
    }

    float range = mx - mn;
    if (range < 1e-8f) range = 1e-8f;
    float scale = range / 16.0f; /* 16 bins of width range/16 */
    float inv_scale = 1.0f / scale;

    block->scale      = avx_fp32_to_fp16(scale);
    block->zero_point = avx_fp32_to_fp16(mn);
    memset(block->qs, 0, TQ_BK / 2);

    /* Phase 2: 8-wide quantization */
    __m256 v_mn   = _mm256_set1_ps(mn);
    __m256 v_invs = _mm256_set1_ps(inv_scale);
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_15   = _mm256_set1_ps(15.0f);

    i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 shifted = _mm256_sub_ps(v, v_mn);
        __m256 scaled  = _mm256_mul_ps(shifted, v_invs);
        /* Floor: _mm256_round_ps with _MM_FROUND_TO_NEG_INF */
        __m256 rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        /* Clamp to [0, 15] */
        rounded = _mm256_max_ps(rounded, v_zero);
        rounded = _mm256_min_ps(rounded, v_15);
        __m256i qi = _mm256_cvtps_epi32(rounded);

        /* Extract and pack nibbles */
        int q_vals[8];
        _mm256_storeu_si256((__m256i*)q_vals, qi);

        for (int k = 0; k < 8; k++) {
            int idx = i + k;
            if (idx % 2 == 0) {
                block->qs[idx / 2] = (uint8_t)q_vals[k];
            } else {
                block->qs[idx / 2] |= (uint8_t)(q_vals[k] << 4);
            }
        }
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
 * Uniform 4-bit dequantize — AVX2 optimized
 * ================================================================ */

void tq_uniform_4b_dequantize_avx2(const void* src, float* dst, int n) {
    const block_tq_uniform_4b* block = (const block_tq_uniform_4b*)src;
    int count = n;
    if (count > TQ_BK) count = TQ_BK;

    float scale = avx_fp16_to_fp32(block->scale);
    float mn    = avx_fp16_to_fp32(block->zero_point);

    __m256 v_scale = _mm256_set1_ps(scale);
    __m256 v_mn    = _mm256_set1_ps(mn);

    int i = 0;
    for (; i + 8 <= count; i += 8) {
        /* Unpack 4 bytes = 8 nibbles */
        float q_arr[8];
        for (int k = 0; k < 8; k++) {
            int idx = i + k;
            uint8_t byte = block->qs[idx / 2];
            q_arr[k] = (float)((idx % 2 == 0) ? (byte & 0x0F) : (byte >> 4)) + 0.5f;
        }

        __m256 q = _mm256_loadu_ps(q_arr);
        /* dst = mn + (q + 0.5) * scale using FMA (0.5 already added above) */
        __m256 result = _mm256_fmadd_ps(q, v_scale, v_mn);
        _mm256_storeu_ps(dst + i, result);
    }

    /* Scalar tail */
    for (; i < count; i++) {
        uint8_t byte = block->qs[i / 2];
        int q = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        dst[i] = mn + ((float)q + 0.5f) * scale;
    }
}

/* ================================================================
 * AVX2 atan2 approximation — 8-wide
 *
 * Uses 5-coefficient polynomial (same as NEON version):
 *   atan(z) ~ z * (C0 + z2*(C1 + z2*(C2 + z2*(C3 + z2*(C4 + z2*C5)))))
 * with quadrant correction for full atan2 range.
 * ================================================================ */

static inline __m256 avx2_atan2_approx(__m256 vy, __m256 vx) {
    const __m256 v_pi      = _mm256_set1_ps(TQ_PI);
    const __m256 v_half_pi = _mm256_set1_ps(TQ_PI_2);
    const __m256 v_zero    = _mm256_setzero_ps();

    /* Polynomial coefficients (Abramowitz & Stegun, same as NEON) */
    const __m256 C0 = _mm256_set1_ps(0.99997726f);
    const __m256 C1 = _mm256_set1_ps(-0.33262347f);
    const __m256 C2 = _mm256_set1_ps(0.19354346f);
    const __m256 C3 = _mm256_set1_ps(-0.11643287f);
    const __m256 C4 = _mm256_set1_ps(0.05265332f);
    const __m256 C5 = _mm256_set1_ps(-0.01172120f);

    /* abs(x), abs(y) */
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 ax = _mm256_andnot_ps(sign_mask, vx);
    __m256 ay = _mm256_andnot_ps(sign_mask, vy);

    /* z = min(ax,ay) / max(ax,ay), ensuring |z| <= 1 */
    __m256 mn = _mm256_min_ps(ax, ay);
    __m256 mx = _mm256_max_ps(ax, ay);
    mx = _mm256_max_ps(mx, _mm256_set1_ps(1e-20f));
    __m256 z = _mm256_div_ps(mn, mx);
    __m256 z2 = _mm256_mul_ps(z, z);

    /* Horner evaluation: p = C0 + z2*(C1 + z2*(C2 + z2*(C3 + z2*(C4 + z2*C5)))) */
    __m256 p = _mm256_fmadd_ps(C5, z2, C4);
    p = _mm256_fmadd_ps(p, z2, C3);
    p = _mm256_fmadd_ps(p, z2, C2);
    p = _mm256_fmadd_ps(p, z2, C1);
    p = _mm256_fmadd_ps(p, z2, C0);
    __m256 a = _mm256_mul_ps(z, p);

    /* If ay > ax: a = pi/2 - a */
    __m256 swap_mask = _mm256_cmp_ps(ay, ax, _CMP_GT_OQ);
    a = _mm256_blendv_ps(a, _mm256_sub_ps(v_half_pi, a), swap_mask);

    /* If x < 0: a = pi - a */
    __m256 xneg_mask = _mm256_cmp_ps(vx, v_zero, _CMP_LT_OQ);
    a = _mm256_blendv_ps(a, _mm256_sub_ps(v_pi, a), xneg_mask);

    /* If y < 0: a = -a */
    __m256 yneg_mask = _mm256_cmp_ps(vy, v_zero, _CMP_LT_OQ);
    a = _mm256_blendv_ps(a, _mm256_sub_ps(v_zero, a), yneg_mask);

    return a;
}

/* ================================================================
 * Polar quantize — AVX2 optimized (8-wide atan2 + radius)
 * ================================================================ */

void tq_polar_quantize_avx2(const float* src, void* dst, int n) {
    block_tq_polar* block = (block_tq_polar*)dst;
    int pairs = n / 2;
    if (pairs > TQ_BK / 2) pairs = TQ_BK / 2;

    /* Compute polar coordinates */
    float thetas[TQ_BK / 2];
    float radii[TQ_BK / 2];

    int p = 0;
    /* Process 8 pairs at a time (16 floats -> 8 x,y pairs) */
    for (; p + 8 <= pairs; p += 8) {
        /* Load 16 floats: x0,y0,x1,y1,...,x7,y7 */
        __m256 xy0 = _mm256_loadu_ps(src + 2 * p);      /* x0,y0,x1,y1,x2,y2,x3,y3 */
        __m256 xy1 = _mm256_loadu_ps(src + 2 * p + 8);  /* x4,y4,x5,y5,x6,y6,x7,y7 */

        /* Deinterleave x and y using shuffles */
        /* From xy0: x0,y0,x1,y1,x2,y2,x3,y3
         * We want: vx = x0,x1,x2,x3,x4,x5,x6,x7
         *          vy = y0,y1,y2,y3,y4,y5,y6,y7
         */
        __m256 t0 = _mm256_shuffle_ps(xy0, xy0, _MM_SHUFFLE(2, 0, 2, 0)); /* x0,x1,x0,x1, x2,x3,x2,x3 */
        __m256 t1 = _mm256_shuffle_ps(xy0, xy0, _MM_SHUFFLE(3, 1, 3, 1)); /* y0,y1,y0,y1, y2,y3,y2,y3 */
        __m256 t2 = _mm256_shuffle_ps(xy1, xy1, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 t3 = _mm256_shuffle_ps(xy1, xy1, _MM_SHUFFLE(3, 1, 3, 1));

        /* Use permute to gather into correct positions */
        __m128 x_lo = _mm_shuffle_ps(_mm256_castps256_ps128(t0), _mm256_extractf128_ps(t0, 1), _MM_SHUFFLE(2, 0, 2, 0));
        __m128 x_hi = _mm_shuffle_ps(_mm256_castps256_ps128(t2), _mm256_extractf128_ps(t2, 1), _MM_SHUFFLE(2, 0, 2, 0));
        __m128 y_lo = _mm_shuffle_ps(_mm256_castps256_ps128(t1), _mm256_extractf128_ps(t1, 1), _MM_SHUFFLE(2, 0, 2, 0));
        __m128 y_hi = _mm_shuffle_ps(_mm256_castps256_ps128(t3), _mm256_extractf128_ps(t3, 1), _MM_SHUFFLE(2, 0, 2, 0));

        __m256 vx = _mm256_set_m128(x_hi, x_lo);
        __m256 vy = _mm256_set_m128(y_hi, y_lo);

        /* Radius = sqrt(x*x + y*y) */
        __m256 r2 = _mm256_fmadd_ps(vy, vy, _mm256_mul_ps(vx, vx));
        __m256 vr = _mm256_sqrt_ps(r2);

        /* Theta = atan2(y, x), shifted to [0, 2pi] */
        __m256 vt = avx2_atan2_approx(vy, vx);
        __m256 v_2pi = _mm256_set1_ps(2.0f * TQ_PI);
        __m256 neg_mask = _mm256_cmp_ps(vt, _mm256_setzero_ps(), _CMP_LT_OQ);
        vt = _mm256_add_ps(vt, _mm256_and_ps(neg_mask, v_2pi));

        _mm256_storeu_ps(thetas + p, vt);
        _mm256_storeu_ps(radii + p, vr);
    }

    /* Scalar tail for remaining pairs */
    for (; p < pairs; p++) {
        float x = src[2 * p];
        float y = src[2 * p + 1];
        radii[p] = sqrtf(x * x + y * y);
        float t = atan2f(y, x);
        if (t < 0.0f) t += 2.0f * TQ_PI;
        thetas[p] = t;
    }

    /* Find min/max with AVX2 */
    __m256 vtmin = _mm256_set1_ps(FLT_MAX);
    __m256 vtmax = _mm256_set1_ps(-FLT_MAX);
    __m256 vrmin = _mm256_set1_ps(FLT_MAX);
    __m256 vrmax = _mm256_set1_ps(-FLT_MAX);

    p = 0;
    for (; p + 8 <= pairs; p += 8) {
        __m256 t = _mm256_loadu_ps(thetas + p);
        __m256 r = _mm256_loadu_ps(radii + p);
        vtmin = _mm256_min_ps(vtmin, t);
        vtmax = _mm256_max_ps(vtmax, t);
        vrmin = _mm256_min_ps(vrmin, r);
        vrmax = _mm256_max_ps(vrmax, r);
    }

    float tmin = hmin_ps(vtmin);
    float tmax = hmax_ps(vtmax);
    float rmin = hmin_ps(vrmin);
    float rmax = hmax_ps(vrmax);

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

    block->tscale = avx_fp32_to_fp16(tscale);
    block->tmn    = avx_fp32_to_fp16(tmin);
    block->rscale = avx_fp32_to_fp16(rscale);
    block->rmn    = avx_fp32_to_fp16(rmin);

    memset(block->indices, 0, TQ_BK / 2);

    /* Quantize with AVX2 */
    __m256 v_tmin  = _mm256_set1_ps(tmin);
    __m256 v_rmin  = _mm256_set1_ps(rmin);
    __m256 v_tinvs = _mm256_set1_ps(1.0f / tscale);
    __m256 v_rinvs = _mm256_set1_ps(1.0f / rscale);
    __m256 v_zero  = _mm256_setzero_ps();
    __m256 v_three = _mm256_set1_ps(3.0f);

    p = 0;
    for (; p + 8 <= pairs; p += 8) {
        __m256 t = _mm256_loadu_ps(thetas + p);
        __m256 r = _mm256_loadu_ps(radii + p);

        /* tq = floor((t - tmin) / tscale), clamp [0,3] */
        __m256 tq_f = _mm256_mul_ps(_mm256_sub_ps(t, v_tmin), v_tinvs);
        __m256 tq_r = _mm256_round_ps(tq_f, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        tq_r = _mm256_max_ps(_mm256_min_ps(tq_r, v_three), v_zero);
        __m256i tq = _mm256_cvtps_epi32(tq_r);

        /* rq = floor((r - rmin) / rscale), clamp [0,3] */
        __m256 rq_f = _mm256_mul_ps(_mm256_sub_ps(r, v_rmin), v_rinvs);
        __m256 rq_r = _mm256_round_ps(rq_f, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        rq_r = _mm256_max_ps(_mm256_min_ps(rq_r, v_three), v_zero);
        __m256i rq = _mm256_cvtps_epi32(rq_r);

        /* Extract and pack: rho in upper 2 bits, theta in lower 2 */
        int tq_vals[8], rq_vals[8];
        _mm256_storeu_si256((__m256i*)tq_vals, tq);
        _mm256_storeu_si256((__m256i*)rq_vals, rq);

        for (int k = 0; k < 8; k++) {
            uint8_t packed = (uint8_t)((rq_vals[k] << 2) | tq_vals[k]);
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

void tq_polar_dequantize_avx2(const void* src, float* dst, int n) {
    /* Dequantize is dominated by cos/sin — keep scalar like NEON version */
    const block_tq_polar* block = (const block_tq_polar*)src;
    int pairs = n / 2;
    if (pairs > TQ_BK / 2) pairs = TQ_BK / 2;

    float tscale = avx_fp16_to_fp32(block->tscale);
    float tmin   = avx_fp16_to_fp32(block->tmn);
    float rscale = avx_fp16_to_fp32(block->rscale);
    float rmin   = avx_fp16_to_fp32(block->rmn);

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
 * QJL quantize — AVX2 with 8-wide FMA dot products
 *
 * Ports the NEON implementation: L2 norm, outlier detection,
 * and Rademacher sign hash projection.
 * ================================================================ */

/* Deterministic pseudo-random projection (same hash as NEON/reference) */
static inline float avx_qjl_random_entry(int dim_idx, int sketch_idx) {
    uint32_t h = (uint32_t)(dim_idx * 2654435761u + sketch_idx * 340573321u);
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    return (h & 1) ? 1.0f : -1.0f;
}

void tq_qjl_quantize_avx2(const float* src, void* dst, int n) {
    block_tq_qjl* block = (block_tq_qjl*)dst;
    int dim = n;
    if (dim > TQ_BK_QJL) dim = TQ_BK_QJL;

    /* L2 norm with AVX2 8-wide accumulation */
    __m256 norm_acc = _mm256_setzero_ps();
    int d = 0;
    for (; d + 8 <= dim; d += 8) {
        __m256 v = _mm256_loadu_ps(src + d);
        norm_acc = _mm256_fmadd_ps(v, v, norm_acc);
    }
    float norm_sq = hsum_ps(norm_acc);
    for (; d < dim; d++) {
        norm_sq += src[d] * src[d];
    }
    block->norm = avx_fp32_to_fp16(sqrtf(norm_sq));

    /* Find outliers: top-k by absolute value */
    float abs_vals[TQ_BK_QJL];
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    d = 0;
    for (; d + 8 <= dim; d += 8) {
        __m256 v = _mm256_loadu_ps(src + d);
        __m256 av = _mm256_andnot_ps(sign_mask, v);
        _mm256_storeu_ps(abs_vals + d, av);
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
    block->outlier_norm = avx_fp32_to_fp16(sqrtf(outlier_norm_sq));

    /* Compute sign hash using AVX2 8-wide dot products */
    memset(block->hash, 0, TQ_SKETCH_DIM / 8);
    for (int s = 0; s < TQ_SKETCH_DIM; s++) {
        __m256 proj_acc = _mm256_setzero_ps();
        d = 0;
        for (; d + 8 <= dim; d += 8) {
            __m256 v = _mm256_loadu_ps(src + d);
            float signs[8];
            signs[0] = avx_qjl_random_entry(d + 0, s);
            signs[1] = avx_qjl_random_entry(d + 1, s);
            signs[2] = avx_qjl_random_entry(d + 2, s);
            signs[3] = avx_qjl_random_entry(d + 3, s);
            signs[4] = avx_qjl_random_entry(d + 4, s);
            signs[5] = avx_qjl_random_entry(d + 5, s);
            signs[6] = avx_qjl_random_entry(d + 6, s);
            signs[7] = avx_qjl_random_entry(d + 7, s);
            __m256 vs = _mm256_loadu_ps(signs);
            proj_acc = _mm256_fmadd_ps(v, vs, proj_acc);
        }
        float proj = hsum_ps(proj_acc);
        for (; d < dim; d++) {
            proj += src[d] * avx_qjl_random_entry(d, s);
        }
        if (proj > 0.0f) {
            block->hash[s / 8] |= (1 << (s % 8));
        }
    }
}

/* ================================================================
 * QJL attention — AVX2 with XOR + popcount
 *
 * For each key in the cache: XOR sign bits, popcount disagreements,
 * convert to cosine similarity estimate.
 * ================================================================ */

void tq_qjl_attention_avx2(const float* query, const void* kv_cache,
                            float* scores, int seq_len, int head_dim) {
    const block_tq_qjl* blocks = (const block_tq_qjl*)kv_cache;
    int dim = head_dim;
    if (dim > TQ_BK_QJL) dim = TQ_BK_QJL;

    /* Precompute query projections */
    float q_proj[TQ_SKETCH_DIM];
    for (int s = 0; s < TQ_SKETCH_DIM; s++) {
        __m256 proj_acc = _mm256_setzero_ps();
        int d = 0;
        for (; d + 8 <= dim; d += 8) {
            __m256 v = _mm256_loadu_ps(query + d);
            float signs[8];
            signs[0] = avx_qjl_random_entry(d + 0, s);
            signs[1] = avx_qjl_random_entry(d + 1, s);
            signs[2] = avx_qjl_random_entry(d + 2, s);
            signs[3] = avx_qjl_random_entry(d + 3, s);
            signs[4] = avx_qjl_random_entry(d + 4, s);
            signs[5] = avx_qjl_random_entry(d + 5, s);
            signs[6] = avx_qjl_random_entry(d + 6, s);
            signs[7] = avx_qjl_random_entry(d + 7, s);
            __m256 vs = _mm256_loadu_ps(signs);
            proj_acc = _mm256_fmadd_ps(v, vs, proj_acc);
        }
        float proj = hsum_ps(proj_acc);
        for (; d < dim; d++) {
            proj += query[d] * avx_qjl_random_entry(d, s);
        }
        q_proj[s] = proj;
    }

    /* Query norm */
    __m256 qn_acc = _mm256_setzero_ps();
    int d = 0;
    for (; d + 8 <= dim; d += 8) {
        __m256 v = _mm256_loadu_ps(query + d);
        qn_acc = _mm256_fmadd_ps(v, v, qn_acc);
    }
    float q_norm_sq = hsum_ps(qn_acc);
    for (; d < dim; d++) q_norm_sq += query[d] * query[d];
    float q_norm = sqrtf(q_norm_sq);

    /* Precompute query sign bits */
    uint8_t q_sign_bits[TQ_SKETCH_DIM / 8];
    memset(q_sign_bits, 0, TQ_SKETCH_DIM / 8);
    for (int s = 0; s < TQ_SKETCH_DIM; s++) {
        if (q_proj[s] > 0.0f) {
            q_sign_bits[s / 8] |= (1 << (s % 8));
        }
    }

    /* For each key: XOR + popcount */
    for (int s = 0; s < seq_len; s++) {
        const block_tq_qjl* block = &blocks[s];
        float key_norm = avx_fp16_to_fp32(block->norm);

        int total_agree = 0;
        int bytes = TQ_SKETCH_DIM / 8;
        int b = 0;

        /* Process 32 bytes at a time with AVX2 (256 bits) */
        for (; b + 32 <= bytes; b += 32) {
            __m256i kbits = _mm256_loadu_si256((const __m256i*)(block->hash + b));
            __m256i qbits = _mm256_loadu_si256((const __m256i*)(q_sign_bits + b));
            /* XOR to find disagreements */
            __m256i xor_result = _mm256_xor_si256(kbits, qbits);

            /* Popcount via lookup table (vpshufb trick) */
            const __m256i lo_mask = _mm256_set1_epi8(0x0F);
            const __m256i lut = _mm256_setr_epi8(
                0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
                0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
            __m256i lo_nib = _mm256_and_si256(xor_result, lo_mask);
            __m256i hi_nib = _mm256_and_si256(_mm256_srli_epi16(xor_result, 4), lo_mask);
            __m256i pop_lo = _mm256_shuffle_epi8(lut, lo_nib);
            __m256i pop_hi = _mm256_shuffle_epi8(lut, hi_nib);
            __m256i popcnt = _mm256_add_epi8(pop_lo, pop_hi);

            /* Sum all byte popcounts */
            __m256i sum16 = _mm256_sad_epu8(popcnt, _mm256_setzero_si256());
            /* sum16 has 4 x 64-bit sums */
            int disagree = (int)(_mm256_extract_epi64(sum16, 0) +
                                 _mm256_extract_epi64(sum16, 1) +
                                 _mm256_extract_epi64(sum16, 2) +
                                 _mm256_extract_epi64(sum16, 3));
            total_agree += (32 * 8) - disagree;
        }

        /* Scalar tail */
        for (; b < bytes; b++) {
            uint8_t xor_val = block->hash[b] ^ q_sign_bits[b];
            int pc = 0;
            uint8_t tmp = xor_val;
            while (tmp) { pc += tmp & 1; tmp >>= 1; }
            total_agree += 8 - pc;
        }

        float frac = (float)total_agree / TQ_SKETCH_DIM;
        float cos_est = cosf(TQ_PI * (1.0f - frac));
        scores[s] = cos_est * q_norm * key_norm;
    }
}

#endif /* __AVX2__ */
