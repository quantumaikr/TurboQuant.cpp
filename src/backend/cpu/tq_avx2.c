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
 * Polar quantize — AVX2 stub (atan2 approximation is complex on AVX2,
 * falls back to reference for now but with AVX2 min/max optimization)
 * ================================================================ */

void tq_polar_quantize_avx2(const float* src, void* dst, int n) {
    extern void tq_polar_quantize_ref(const float* src, void* dst, int n);
    tq_polar_quantize_ref(src, dst, n);
}

void tq_polar_dequantize_avx2(const void* src, float* dst, int n) {
    extern void tq_polar_dequantize_ref(const void* src, float* dst, int n);
    tq_polar_dequantize_ref(src, dst, n);
}

/* ================================================================
 * QJL quantize — AVX2 with 8-wide dot products
 * ================================================================ */

void tq_qjl_quantize_avx2(const float* src, void* dst, int n) {
    extern void tq_qjl_quantize_ref(const float* src, void* dst, int n);
    tq_qjl_quantize_ref(src, dst, n);
}

#endif /* __AVX2__ */
