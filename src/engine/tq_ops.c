/**
 * tq_ops.c — Core tensor operations for transformer inference
 *
 * Implements matmul, RMSNorm, RoPE, SiLU, softmax, and element-wise ops.
 * NEON-optimized where available (Apple Silicon / ARM64).
 * No external dependencies — libc/libm only.
 */

#include "turboquant/tq_engine.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <pthread.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ============================================================
 * Global thread count for matmul parallelism
 * ============================================================ */
static int g_n_threads = 1;

void tq_set_threads(int n_threads) {
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 16) n_threads = 16;
    g_n_threads = n_threads;
}

int tq_get_threads(void) {
    return g_n_threads;
}

/* ============================================================
 * Multi-threaded matmul worker
 * ============================================================ */
typedef struct {
    float* out;
    const float* x;
    const float* w;
    int start_row;
    int end_row;
    int d;
} matmul_task_t;

static void matmul_rows(float* out, const float* x, const float* w,
                        int start_row, int end_row, int d) {
#ifdef __ARM_NEON
    for (int i = start_row; i < end_row; i++) {
        const float* wi = w + (size_t)i * d;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 15 < d; j += 16) {
            float32x4_t vx0 = vld1q_f32(x + j);
            float32x4_t vx1 = vld1q_f32(x + j + 4);
            float32x4_t vx2 = vld1q_f32(x + j + 8);
            float32x4_t vx3 = vld1q_f32(x + j + 12);
            float32x4_t vw0 = vld1q_f32(wi + j);
            float32x4_t vw1 = vld1q_f32(wi + j + 4);
            float32x4_t vw2 = vld1q_f32(wi + j + 8);
            float32x4_t vw3 = vld1q_f32(wi + j + 12);
            acc0 = vfmaq_f32(acc0, vx0, vw0);
            acc1 = vfmaq_f32(acc1, vx1, vw1);
            acc2 = vfmaq_f32(acc2, vx2, vw2);
            acc3 = vfmaq_f32(acc3, vx3, vw3);
        }
        for (; j + 3 < d; j += 4) {
            float32x4_t vx = vld1q_f32(x + j);
            float32x4_t vw = vld1q_f32(wi + j);
            acc0 = vfmaq_f32(acc0, vx, vw);
        }
        acc0 = vaddq_f32(acc0, acc1);
        acc2 = vaddq_f32(acc2, acc3);
        acc0 = vaddq_f32(acc0, acc2);
        float sum = vaddvq_f32(acc0);
        for (; j < d; j++) {
            sum += wi[j] * x[j];
        }
        out[i] = sum;
    }
#else
    for (int i = start_row; i < end_row; i++) {
        const float* wi = w + (size_t)i * d;
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += wi[j] * x[j];
        }
        out[i] = sum;
    }
#endif
}

static void* matmul_worker(void* arg) {
    matmul_task_t* t = (matmul_task_t*)arg;
    matmul_rows(t->out, t->x, t->w, t->start_row, t->end_row, t->d);
    return NULL;
}

/* ============================================================
 * Matrix-vector multiply: out[i] = sum_j(w[i*d + j] * x[j])
 *
 * This is THE dominant cost in LLM inference (~90% of compute).
 * w is [n, d] row-major, x is [d], out is [n].
 * ============================================================ */
void tq_matmul(float* out, const float* x, const float* w, int n, int d) {
    int n_threads = g_n_threads;

    /* For small matrices or single-thread config, skip thread overhead */
    if (n < 256 || n_threads <= 1) {
        matmul_rows(out, x, w, 0, n, d);
        return;
    }

    /* Cap threads to available rows */
    if (n_threads > n) n_threads = n;
    if (n_threads > 16) n_threads = 16;

    pthread_t threads[16];
    matmul_task_t tasks[16];

    int rows_per_thread = n / n_threads;
    for (int t = 0; t < n_threads; t++) {
        tasks[t].out = out;
        tasks[t].x = x;
        tasks[t].w = w;
        tasks[t].d = d;
        tasks[t].start_row = t * rows_per_thread;
        tasks[t].end_row = (t == n_threads - 1) ? n : (t + 1) * rows_per_thread;
        pthread_create(&threads[t], NULL, matmul_worker, &tasks[t]);
    }
    for (int t = 0; t < n_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

/* ============================================================
 * Q8 quantization: float -> int8 + per-block scale (block_size=32)
 *
 * For each block of 32 values:
 *   scale = max(|x_i|) / 127
 *   q_i = round(x_i / scale)
 * ============================================================ */
void tq_quantize_row_q8(const float* src, int8_t* dst_qs, float* dst_scales, int n) {
    int n_blocks = n / 32;
    for (int b = 0; b < n_blocks; b++) {
        const float* block = src + b * 32;
        float amax = 0.0f;
#ifdef __ARM_NEON
        float32x4_t vmax = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 4) {
            float32x4_t v = vld1q_f32(block + j);
            vmax = vmaxq_f32(vmax, vabsq_f32(v));
        }
        amax = vmaxvq_f32(vmax);
#else
        for (int j = 0; j < 32; j++) {
            float a = fabsf(block[j]);
            if (a > amax) amax = a;
        }
#endif
        float scale = amax / 127.0f;
        dst_scales[b] = scale;
        float inv = (scale > 1e-10f) ? 1.0f / scale : 0.0f;
        int8_t* qb = dst_qs + b * 32;
#ifdef __ARM_NEON
        float32x4_t vinv = vdupq_n_f32(inv);
        for (int j = 0; j < 32; j += 4) {
            float32x4_t v = vld1q_f32(block + j);
            float32x4_t scaled = vmulq_f32(v, vinv);
            /* Round to nearest and convert to int32 */
            int32x4_t vi = vcvtnq_s32_f32(scaled);
            /* Narrow to int16 then int8 */
            int16x4_t v16 = vmovn_s32(vi);
            int16x8_t v16_wide = vcombine_s16(v16, v16);
            int8x8_t v8 = vmovn_s16(v16_wide);
            /* Store only 4 bytes */
            qb[j]   = vget_lane_s8(v8, 0);
            qb[j+1] = vget_lane_s8(v8, 1);
            qb[j+2] = vget_lane_s8(v8, 2);
            qb[j+3] = vget_lane_s8(v8, 3);
        }
#else
        for (int j = 0; j < 32; j++) {
            qb[j] = (int8_t)roundf(block[j] * inv);
        }
#endif
    }
    /* Handle remainder (if n is not multiple of 32) */
    int remainder = n - n_blocks * 32;
    if (remainder > 0) {
        const float* block = src + n_blocks * 32;
        float amax = 0.0f;
        for (int j = 0; j < remainder; j++) {
            float a = fabsf(block[j]);
            if (a > amax) amax = a;
        }
        float scale = amax / 127.0f;
        dst_scales[n_blocks] = scale;
        float inv = (scale > 1e-10f) ? 1.0f / scale : 0.0f;
        int8_t* qb = dst_qs + n_blocks * 32;
        for (int j = 0; j < remainder; j++) {
            qb[j] = (int8_t)roundf(block[j] * inv);
        }
    }
}

/* ============================================================
 * Q8 matmul: w is Q8 [n, d], x is FP32 [d], out is FP32 [n]
 *
 * For each output row i:
 *   out[i] = sum over blocks { scale[b] * sum_j(w_q8[j] * x[j]) }
 *
 * Block size = 32, so n_blocks = d / 32.
 * ============================================================ */

typedef struct {
    float* out;
    const float* x;
    const int8_t* w_qs;
    const float* w_scales;
    int start_row;
    int end_row;
    int d;
} matmul_q8_task_t;

static void matmul_q8_rows(float* out, const float* x,
                            const int8_t* w_qs, const float* w_scales,
                            int start_row, int end_row, int d) {
    int n_blocks = d / 32;
    for (int i = start_row; i < end_row; i++) {
        const int8_t* wi = w_qs + (size_t)i * d;
        const float* si = w_scales + (size_t)i * n_blocks;
        float sum = 0.0f;
#ifdef __ARM_NEON
        for (int b = 0; b < n_blocks; b++) {
            const int8_t* qb = wi + b * 32;
            const float* xb = x + b * 32;
            float block_sum = 0.0f;
            /* Process 16 elements at a time using NEON int8 dot product:
             * Load 16 int8 weights, convert to float, multiply with x, accumulate */
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            /* First 16: convert int8 -> int16 -> int32 -> float, then fma */
            int8x16_t vq0 = vld1q_s8(qb);
            int8x16_t vq1 = vld1q_s8(qb + 16);

            /* Expand first 16 int8 to 4x float32x4 */
            int16x8_t v16_lo = vmovl_s8(vget_low_s8(vq0));
            int16x8_t v16_hi = vmovl_s8(vget_high_s8(vq0));
            float32x4_t fq0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_lo)));
            float32x4_t fq1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_lo)));
            float32x4_t fq2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_hi)));
            float32x4_t fq3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_hi)));
            acc0 = vfmaq_f32(acc0, fq0, vld1q_f32(xb));
            acc1 = vfmaq_f32(acc1, fq1, vld1q_f32(xb + 4));
            acc2 = vfmaq_f32(acc2, fq2, vld1q_f32(xb + 8));
            acc3 = vfmaq_f32(acc3, fq3, vld1q_f32(xb + 12));

            /* Expand next 16 int8 to 4x float32x4 */
            v16_lo = vmovl_s8(vget_low_s8(vq1));
            v16_hi = vmovl_s8(vget_high_s8(vq1));
            fq0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_lo)));
            fq1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_lo)));
            fq2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_hi)));
            fq3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_hi)));
            acc0 = vfmaq_f32(acc0, fq0, vld1q_f32(xb + 16));
            acc1 = vfmaq_f32(acc1, fq1, vld1q_f32(xb + 20));
            acc2 = vfmaq_f32(acc2, fq2, vld1q_f32(xb + 24));
            acc3 = vfmaq_f32(acc3, fq3, vld1q_f32(xb + 28));

            acc0 = vaddq_f32(acc0, acc1);
            acc2 = vaddq_f32(acc2, acc3);
            acc0 = vaddq_f32(acc0, acc2);
            block_sum = vaddvq_f32(acc0);
            sum += block_sum * si[b];
        }
#else
        for (int b = 0; b < n_blocks; b++) {
            const int8_t* qb = wi + b * 32;
            const float* xb = x + b * 32;
            float block_sum = 0.0f;
            for (int j = 0; j < 32; j++) {
                block_sum += (float)qb[j] * xb[j];
            }
            sum += block_sum * si[b];
        }
#endif
        out[i] = sum;
    }
}

static void* matmul_q8_worker(void* arg) {
    matmul_q8_task_t* t = (matmul_q8_task_t*)arg;
    matmul_q8_rows(t->out, t->x, t->w_qs, t->w_scales,
                    t->start_row, t->end_row, t->d);
    return NULL;
}

/* Q8 matmul with multi-threading support */
void tq_matmul_q8(float* out, const float* x, const int8_t* w_qs, const float* w_scales,
                   int n, int d) {
    int n_threads = g_n_threads;

    if (n < 256 || n_threads <= 1) {
        matmul_q8_rows(out, x, w_qs, w_scales, 0, n, d);
        return;
    }

    if (n_threads > n) n_threads = n;
    if (n_threads > 16) n_threads = 16;

    pthread_t threads[16];
    matmul_q8_task_t tasks[16];

    int rows_per_thread = n / n_threads;
    for (int t = 0; t < n_threads; t++) {
        tasks[t].out = out;
        tasks[t].x = x;
        tasks[t].w_qs = w_qs;
        tasks[t].w_scales = w_scales;
        tasks[t].d = d;
        tasks[t].start_row = t * rows_per_thread;
        tasks[t].end_row = (t == n_threads - 1) ? n : (t + 1) * rows_per_thread;
        pthread_create(&threads[t], NULL, matmul_q8_worker, &tasks[t]);
    }
    for (int t = 0; t < n_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

/* ============================================================
 * Q4_0 quantization: float -> packed 4-bit + per-block scale (block_size=32)
 *
 * For each block of 32 values:
 *   scale = max(|x_i|) / 7.0  (symmetric 4-bit: [-7, 7] maps to [1,15])
 *   q_i = round(x_i / scale) + 8, clamped to [0, 15]
 * Packed: two 4-bit values per byte, low nibble first.
 * ============================================================ */
void tq_quantize_row_q4(const float* src, uint8_t* dst_qs, float* dst_scales, int n) {
    int n_blocks = n / 32;
    for (int b = 0; b < n_blocks; b++) {
        const float* block = src + b * 32;
        float amax = 0.0f;
#ifdef __ARM_NEON
        float32x4_t vmax = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 4) {
            float32x4_t v = vld1q_f32(block + j);
            vmax = vmaxq_f32(vmax, vabsq_f32(v));
        }
        amax = vmaxvq_f32(vmax);
#else
        for (int j = 0; j < 32; j++) {
            float a = fabsf(block[j]);
            if (a > amax) amax = a;
        }
#endif
        float d = amax / 7.0f;
        dst_scales[b] = d;
        float id = (d > 1e-10f) ? 1.0f / d : 0.0f;

        uint8_t* qb = dst_qs + b * 16;
        for (int j = 0; j < 16; j++) {
            int q0 = (int)roundf(block[2 * j] * id) + 8;
            int q1 = (int)roundf(block[2 * j + 1] * id) + 8;
            if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
            if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
            qb[j] = (uint8_t)((q1 << 4) | q0);
        }
    }
    /* Handle remainder (if n is not multiple of 32) */
    int remainder = n - n_blocks * 32;
    if (remainder > 0) {
        const float* block = src + n_blocks * 32;
        float amax = 0.0f;
        for (int j = 0; j < remainder; j++) {
            float a = fabsf(block[j]);
            if (a > amax) amax = a;
        }
        float d = amax / 7.0f;
        dst_scales[n_blocks] = d;
        float id = (d > 1e-10f) ? 1.0f / d : 0.0f;
        uint8_t* qb = dst_qs + n_blocks * 16;
        int n_pairs = remainder / 2;
        for (int j = 0; j < n_pairs; j++) {
            int q0 = (int)roundf(block[2 * j] * id) + 8;
            int q1 = (int)roundf(block[2 * j + 1] * id) + 8;
            if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
            if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
            qb[j] = (uint8_t)((q1 << 4) | q0);
        }
        if (remainder & 1) {
            int q0 = (int)roundf(block[remainder - 1] * id) + 8;
            if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
            qb[n_pairs] = (uint8_t)(q0);
        }
    }
}

/* ============================================================
 * Q4 matmul: w is Q4_0 [n, d], x is FP32 [d], out is FP32 [n]
 *
 * Strategy: quantize activation x to Q8 once, then compute
 * Q4 x Q8 integer dot product per block for maximum throughput.
 * ============================================================ */

typedef struct {
    float* out;
    const float* x;
    const uint8_t* w_qs;
    const float* w_scales;
    const int8_t* x_q8;
    const float* x_scales;
    int start_row;
    int end_row;
    int d;
} matmul_q4_task_t;

static void matmul_q4_rows(float* out, const float* x,
                            const uint8_t* w_qs, const float* w_scales,
                            const int8_t* x_q8, const float* x_scales,
                            int start_row, int end_row, int d) {
    int n_blocks = d / 32;
    for (int i = start_row; i < end_row; i++) {
        const uint8_t* wi = w_qs + (size_t)i * n_blocks * 16;
        const float* si = w_scales + (size_t)i * n_blocks;
        float sum = 0.0f;
#ifdef __ARM_NEON
        for (int b = 0; b < n_blocks; b++) {
            const uint8_t* qb = wi + b * 16;
            const int8_t* xb = x_q8 + b * 32;

            /* Load 16 packed bytes = 32 Q4 values */
            uint8x16_t packed = vld1q_u8(qb);
            /* Unpack low nibbles: values 0..15 */
            uint8x16_t lo_u = vandq_u8(packed, vdupq_n_u8(0x0F));
            /* Unpack high nibbles: values 0..15 */
            uint8x16_t hi_u = vshrq_n_u8(packed, 4);
            /* Subtract 8 to center: [0,15] -> [-8,7], reinterpret as signed */
            int8x16_t lo = vreinterpretq_s8_u8(vsubq_u8(lo_u, vdupq_n_u8(8)));
            int8x16_t hi = vreinterpretq_s8_u8(vsubq_u8(hi_u, vdupq_n_u8(8)));

            /* Load Q8 activation: 32 int8 values, split as lo[0..15] hi[16..31]
             * But Q4 packing interleaves: lo nibble = even indices, hi nibble = odd indices
             * So we need to deinterleave x_q8 to match */

            /* x_q8[b*32+0], x_q8[b*32+2], ..., x_q8[b*32+30] -> even indices
             * x_q8[b*32+1], x_q8[b*32+3], ..., x_q8[b*32+31] -> odd indices */
            int8x16x2_t x_deint = vld2q_s8(xb);
            /* x_deint.val[0] = even indices (matches lo nibble)
             * x_deint.val[1] = odd indices (matches hi nibble) */

            int32x4_t acc = vdupq_n_s32(0);
#if defined(__ARM_FEATURE_DOTPROD)
            acc = vdotq_s32(acc, lo, x_deint.val[0]);
            acc = vdotq_s32(acc, hi, x_deint.val[1]);
#else
            /* Fallback: widen multiply and accumulate */
            int16x8_t prod_lo_0 = vmull_s8(vget_low_s8(lo), vget_low_s8(x_deint.val[0]));
            int16x8_t prod_lo_1 = vmull_s8(vget_high_s8(lo), vget_high_s8(x_deint.val[0]));
            int16x8_t prod_hi_0 = vmull_s8(vget_low_s8(hi), vget_low_s8(x_deint.val[1]));
            int16x8_t prod_hi_1 = vmull_s8(vget_high_s8(hi), vget_high_s8(x_deint.val[1]));
            acc = vaddq_s32(acc, vpaddlq_s16(prod_lo_0));
            acc = vaddq_s32(acc, vpaddlq_s16(prod_lo_1));
            acc = vaddq_s32(acc, vpaddlq_s16(prod_hi_0));
            acc = vaddq_s32(acc, vpaddlq_s16(prod_hi_1));
#endif
            int32_t isum = vaddvq_s32(acc);
            sum += (float)isum * si[b] * x_scales[b];
        }
#else
        for (int b = 0; b < n_blocks; b++) {
            const uint8_t* qb = wi + b * 16;
            const int8_t* xb = x_q8 + b * 32;
            int32_t isum = 0;
            for (int j = 0; j < 16; j++) {
                int q0 = (qb[j] & 0x0F) - 8;
                int q1 = (qb[j] >> 4) - 8;
                isum += q0 * (int)xb[2 * j] + q1 * (int)xb[2 * j + 1];
            }
            sum += (float)isum * si[b] * x_scales[b];
        }
#endif
        out[i] = sum;
    }
}

static void* matmul_q4_worker(void* arg) {
    matmul_q4_task_t* t = (matmul_q4_task_t*)arg;
    matmul_q4_rows(t->out, t->x, t->w_qs, t->w_scales,
                    t->x_q8, t->x_scales,
                    t->start_row, t->end_row, t->d);
    return NULL;
}

/* Q4 matmul with multi-threading support.
 * Quantizes activation x to Q8 once, then does Q4xQ8 integer dot products. */
void tq_matmul_q4(float* out, const float* x, const uint8_t* w_qs, const float* w_scales,
                   int n, int d) {
    /* Quantize activation x to Q8 (amortized across all rows) */
    int n_blocks = d / 32;
    int8_t* x_q8 = (int8_t*)malloc((size_t)d * sizeof(int8_t));
    float* x_scales = (float*)malloc((size_t)(n_blocks + 1) * sizeof(float));
    if (!x_q8 || !x_scales) {
        free(x_q8);
        free(x_scales);
        return;
    }
    tq_quantize_row_q8(x, x_q8, x_scales, d);

    int n_threads = g_n_threads;

    if (n < 256 || n_threads <= 1) {
        matmul_q4_rows(out, x, w_qs, w_scales, x_q8, x_scales, 0, n, d);
        free(x_q8);
        free(x_scales);
        return;
    }

    if (n_threads > n) n_threads = n;
    if (n_threads > 16) n_threads = 16;

    pthread_t threads[16];
    matmul_q4_task_t tasks[16];

    int rows_per_thread = n / n_threads;
    for (int t = 0; t < n_threads; t++) {
        tasks[t].out = out;
        tasks[t].x = x;
        tasks[t].w_qs = w_qs;
        tasks[t].w_scales = w_scales;
        tasks[t].x_q8 = x_q8;
        tasks[t].x_scales = x_scales;
        tasks[t].d = d;
        tasks[t].start_row = t * rows_per_thread;
        tasks[t].end_row = (t == n_threads - 1) ? n : (t + 1) * rows_per_thread;
        pthread_create(&threads[t], NULL, matmul_q4_worker, &tasks[t]);
    }
    for (int t = 0; t < n_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    free(x_q8);
    free(x_scales);
}

/* ============================================================
 * Q4 matmul with pre-quantized activation (no redundant quantization).
 *
 * When the same activation vector x is multiplied by multiple weight
 * matrices (e.g., QKV, Z, A, B projections in DeltaNet), we quantize
 * x to Q8 once and reuse across all calls.
 * ============================================================ */
void tq_matmul_q4_preq(float* out, const uint8_t* w_qs, const float* w_scales,
                        const int8_t* x_q8, const float* x_scales,
                        int n, int d) {
    int n_threads = g_n_threads;

    if (n < 256 || n_threads <= 1) {
        matmul_q4_rows(out, NULL, w_qs, w_scales, x_q8, x_scales, 0, n, d);
        return;
    }

    if (n_threads > n) n_threads = n;
    if (n_threads > 16) n_threads = 16;

    pthread_t threads[16];
    matmul_q4_task_t tasks[16];

    int rows_per_thread = n / n_threads;
    for (int t = 0; t < n_threads; t++) {
        tasks[t].out = out;
        tasks[t].x = NULL;
        tasks[t].w_qs = w_qs;
        tasks[t].w_scales = w_scales;
        tasks[t].x_q8 = x_q8;
        tasks[t].x_scales = x_scales;
        tasks[t].d = d;
        tasks[t].start_row = t * rows_per_thread;
        tasks[t].end_row = (t == n_threads - 1) ? n : (t + 1) * rows_per_thread;
        pthread_create(&threads[t], NULL, matmul_q4_worker, &tasks[t]);
    }
    for (int t = 0; t < n_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

/* ============================================================
 * BF16 matmul worker helpers
 * ============================================================ */
typedef struct {
    float* out;
    const float* x;
    const uint16_t* w_bf16;
    int start_row;
    int end_row;
    int d;
} matmul_bf16_task_t;

static void matmul_bf16_rows(float* out, const float* x,
                              const uint16_t* w_bf16,
                              int start_row, int end_row, int d) {
#ifdef __ARM_NEON
    for (int i = start_row; i < end_row; i++) {
        const uint16_t* wi = w_bf16 + (size_t)i * d;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 15 < d; j += 16) {
            /* Convert 4 BF16 values to FP32 via shift-left-16 */
            uint16x4_t b0 = vld1_u16(wi + j);
            uint16x4_t b1 = vld1_u16(wi + j + 4);
            uint16x4_t b2 = vld1_u16(wi + j + 8);
            uint16x4_t b3 = vld1_u16(wi + j + 12);
            float32x4_t vw0 = vreinterpretq_f32_u32(vshll_n_u16(b0, 16));
            float32x4_t vw1 = vreinterpretq_f32_u32(vshll_n_u16(b1, 16));
            float32x4_t vw2 = vreinterpretq_f32_u32(vshll_n_u16(b2, 16));
            float32x4_t vw3 = vreinterpretq_f32_u32(vshll_n_u16(b3, 16));
            float32x4_t vx0 = vld1q_f32(x + j);
            float32x4_t vx1 = vld1q_f32(x + j + 4);
            float32x4_t vx2 = vld1q_f32(x + j + 8);
            float32x4_t vx3 = vld1q_f32(x + j + 12);
            acc0 = vfmaq_f32(acc0, vx0, vw0);
            acc1 = vfmaq_f32(acc1, vx1, vw1);
            acc2 = vfmaq_f32(acc2, vx2, vw2);
            acc3 = vfmaq_f32(acc3, vx3, vw3);
        }
        for (; j + 3 < d; j += 4) {
            uint16x4_t b = vld1_u16(wi + j);
            float32x4_t vw = vreinterpretq_f32_u32(vshll_n_u16(b, 16));
            float32x4_t vx = vld1q_f32(x + j);
            acc0 = vfmaq_f32(acc0, vx, vw);
        }
        acc0 = vaddq_f32(acc0, acc1);
        acc2 = vaddq_f32(acc2, acc3);
        acc0 = vaddq_f32(acc0, acc2);
        float sum = vaddvq_f32(acc0);
        for (; j < d; j++) {
            uint32_t bits = ((uint32_t)wi[j]) << 16;
            float wf;
            memcpy(&wf, &bits, 4);
            sum += wf * x[j];
        }
        out[i] = sum;
    }
#else
    for (int i = start_row; i < end_row; i++) {
        const uint16_t* wi = w_bf16 + (size_t)i * d;
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            uint32_t bits = ((uint32_t)wi[j]) << 16;
            float wf;
            memcpy(&wf, &bits, 4);
            sum += wf * x[j];
        }
        out[i] = sum;
    }
#endif
}

static void* matmul_bf16_worker(void* arg) {
    matmul_bf16_task_t* t = (matmul_bf16_task_t*)arg;
    matmul_bf16_rows(t->out, t->x, t->w_bf16, t->start_row, t->end_row, t->d);
    return NULL;
}

/* ============================================================
 * Matrix-vector multiply with BF16 weights (streaming conversion)
 *
 * Same as tq_matmul but weights are BF16 (uint16_t*), converted
 * to FP32 on-the-fly during dot product. Saves ~2x memory vs
 * pre-converting all weights to FP32.
 *
 * w_bf16 is [n, d] row-major BF16, x is [d] FP32, out is [n] FP32.
 * ============================================================ */
void tq_matmul_bf16(float* out, const float* x, const uint16_t* w_bf16, int n, int d) {
    int n_threads = g_n_threads;

    if (n < 256 || n_threads <= 1) {
        matmul_bf16_rows(out, x, w_bf16, 0, n, d);
        return;
    }

    if (n_threads > n) n_threads = n;
    if (n_threads > 16) n_threads = 16;

    pthread_t threads[16];
    matmul_bf16_task_t tasks[16];

    int rows_per_thread = n / n_threads;
    for (int t = 0; t < n_threads; t++) {
        tasks[t].out = out;
        tasks[t].x = x;
        tasks[t].w_bf16 = w_bf16;
        tasks[t].d = d;
        tasks[t].start_row = t * rows_per_thread;
        tasks[t].end_row = (t == n_threads - 1) ? n : (t + 1) * rows_per_thread;
        pthread_create(&threads[t], NULL, matmul_bf16_worker, &tasks[t]);
    }
    for (int t = 0; t < n_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

/* ============================================================
 * RMS Normalization: out[i] = (x[i] / rms) * weight[i]
 * where rms = sqrt(mean(x^2) + eps)
 * ============================================================ */
void tq_rmsnorm(float* out, const float* x, const float* weight, int n, float eps) {
#ifdef __ARM_NEON
    float32x4_t sum_sq = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        sum_sq = vfmaq_f32(sum_sq, vx, vx);
    }
    float ss = vaddvq_f32(sum_sq);
    for (; i < n; i++) {
        ss += x[i] * x[i];
    }
    ss = ss / n + eps;
    float rsqrt = 1.0f / sqrtf(ss);

    float32x4_t vrs = vdupq_n_f32(rsqrt);
    i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vw = vld1q_f32(weight + i);
        float32x4_t vo = vmulq_f32(vmulq_f32(vx, vrs), vw);
        vst1q_f32(out + i, vo);
    }
    for (; i < n; i++) {
        out[i] = x[i] * rsqrt * weight[i];
    }
#else
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    ss = ss / n + eps;
    float rsqrt = 1.0f / sqrtf(ss);
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * rsqrt * weight[i];
    }
#endif
}

/* ============================================================
 * Rotary Positional Embedding (RoPE)
 *
 * Applies rotation to pairs (q[2i], q[2i+1]) based on position.
 * Compatible with LLaMA / Qwen RoPE convention.
 * ============================================================ */
void tq_rope(float* q, float* k, int pos, int head_dim,
             int n_heads, int n_kv_heads, float freq_base) {
    /* Apply RoPE to query heads */
    for (int h = 0; h < n_heads; h++) {
        float* qh = q + h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(freq_base, 2.0f * i / head_dim);
            float theta = pos * freq;
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            float q0 = qh[2 * i];
            float q1 = qh[2 * i + 1];
            qh[2 * i]     = q0 * cos_t - q1 * sin_t;
            qh[2 * i + 1] = q0 * sin_t + q1 * cos_t;
        }
    }
    /* Apply RoPE to key heads */
    for (int h = 0; h < n_kv_heads; h++) {
        float* kh = k + h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(freq_base, 2.0f * i / head_dim);
            float theta = pos * freq;
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            float k0 = kh[2 * i];
            float k1 = kh[2 * i + 1];
            kh[2 * i]     = k0 * cos_t - k1 * sin_t;
            kh[2 * i + 1] = k0 * sin_t + k1 * cos_t;
        }
    }
}

/* ============================================================
 * SiLU activation: x[i] = x[i] * sigmoid(x[i])
 * Also known as swish activation.
 * ============================================================ */
void tq_silu(float* x, int n) {
#ifdef __ARM_NEON
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        /* sigmoid(x) = 1/(1+exp(-x)) — compute per-lane */
        float vals[4];
        vst1q_f32(vals, vx);
        float sig[4];
        for (int j = 0; j < 4; j++) {
            sig[j] = 1.0f / (1.0f + expf(-vals[j]));
        }
        float32x4_t vs = vld1q_f32(sig);
        float32x4_t vo = vmulq_f32(vx, vs);
        vst1q_f32(x + i, vo);
    }
    for (; i < n; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
#else
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
#endif
}

/* ============================================================
 * Softmax: numerically stable with max subtraction
 * ============================================================ */
void tq_softmax(float* x, int n) {
    if (n <= 0) return;

    /* Find max for numerical stability */
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    /* normalize */
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < n; i++) {
            x[i] *= inv_sum;
        }
    }
}

/* ============================================================
 * Element-wise add: out[i] = a[i] + b[i]
 * ============================================================ */
void tq_add(float* out, const float* a, const float* b, int n) {
#ifdef __ARM_NEON
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(va, vb));
    }
    for (; i < n; i++) {
        out[i] = a[i] + b[i];
    }
#else
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
#endif
}

/* ============================================================
 * Element-wise multiply: out[i] = a[i] * b[i]
 * ============================================================ */
void tq_mul(float* out, const float* a, const float* b, int n) {
#ifdef __ARM_NEON
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vmulq_f32(va, vb));
    }
    for (; i < n; i++) {
        out[i] = a[i] * b[i];
    }
#else
    for (int i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
#endif
}

/* ============================================================
 * Default generation config
 * ============================================================ */
tq_gen_config_t tq_default_gen_config(void) {
    tq_gen_config_t config;
    memset(&config, 0, sizeof(config));
    config.temperature = 0.7f;
    config.top_p = 0.9f;
    config.max_tokens = 256;
    config.kv_type = TQ_TYPE_UNIFORM_4B;
    config.n_threads = 1;
    config.on_token = NULL;
    config.user_data = NULL;
    return config;
}
