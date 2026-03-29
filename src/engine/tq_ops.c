/**
 * tq_ops.c — Core tensor operations for transformer inference
 *
 * Implements matmul, RMSNorm, RoPE, SiLU, softmax, and element-wise ops.
 * NEON-optimized where available (Apple Silicon / ARM64).
 * No external dependencies — libc/libm only.
 */

#include "turboquant/tq_engine.h"
#include <math.h>
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
    float32x4_t one = vdupq_n_f32(1.0f);
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
