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

#ifdef _WIN32
#include <windows.h>
/* pthread compat for MSVC (using Windows threads) */
typedef HANDLE pthread_t;
typedef CRITICAL_SECTION pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;
#define pthread_mutex_init(m, a) InitializeCriticalSection(m)
#define pthread_mutex_lock(m)    EnterCriticalSection(m)
#define pthread_mutex_unlock(m)  LeaveCriticalSection(m)
#define pthread_mutex_destroy(m) DeleteCriticalSection(m)
#define pthread_cond_init(c, a)  InitializeConditionVariable(c)
/* Note: cond_wait uses SRW variant because mutex is SRWLOCK (see below) */
#define pthread_cond_wait(c, m)  SleepConditionVariableSRW(c, m, INFINITE, 0)
#define pthread_cond_broadcast(c) WakeAllConditionVariable(c)
#define pthread_cond_destroy(c)  ((void)0)
/* __thread → __declspec(thread) */
#define __thread __declspec(thread)
/* PTHREAD_MUTEX_INITIALIZER: SRWLOCK can be zero-initialized */
typedef SRWLOCK pthread_mutex_srw_t;
#undef pthread_mutex_t
#define pthread_mutex_t SRWLOCK
#undef pthread_mutex_init
#define pthread_mutex_init(m, a) InitializeSRWLock(m)
#undef pthread_mutex_lock
#define pthread_mutex_lock(m)    AcquireSRWLockExclusive(m)
#undef pthread_mutex_unlock
#define pthread_mutex_unlock(m)  ReleaseSRWLockExclusive(m)
#undef pthread_mutex_destroy
#define pthread_mutex_destroy(m) ((void)0)
#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
#define pthread_cond_signal(c) WakeConditionVariable(c)
#include <process.h>
static inline int pthread_create(pthread_t* t, const void* a, void*(*fn)(void*), void* arg) {
    (void)a; *t = (HANDLE)_beginthreadex(NULL,0,(unsigned(__stdcall*)(void*))fn,arg,0,NULL);
    return *t ? 0 : -1;
}
static inline int pthread_join(pthread_t t, void** r) {
    (void)r; WaitForSingleObject(t, INFINITE); CloseHandle(t); return 0;
}
#else
#include <pthread.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __APPLE__
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#endif

/* ============================================================
 * Thread pool — condition variable based, minimal overhead
 * Workers sleep between dispatches, wake via cond_broadcast.
 * Main thread does task[0], workers do task[1..n-1].
 * ============================================================ */
#if defined(_MSC_VER)
    /* MSVC: use interlocked intrinsics instead of C11 atomics */
    #include <intrin.h>
    typedef volatile long atomic_int;
    #define atomic_store(p, v)     _InterlockedExchange((p), (v))
    #define atomic_load(p)         _InterlockedCompareExchange((p), 0, 0)
    #define atomic_fetch_add(p, v) _InterlockedExchangeAdd((p), (v))
#else
    #include <stdatomic.h>
#endif

/* Forward declaration for 1-bit matmul (defined at end of file) */
void tq_matmul_1bit(float* out, const float* x, const uint8_t* sign_data, const float* norms,
                     int n_rows, int dim);

#define TP_MAX 16

typedef void* (*tp_fn)(void*);

static struct {
    pthread_t       thr[TP_MAX];
    pthread_mutex_t mtx;
    pthread_cond_t  wake;          /* signal workers to start */
    pthread_cond_t  done_cv;       /* signal main when all done */
    tp_fn           fn;
    void*           args[TP_MAX];
    int             n_workers;     /* total including main = n_workers+1 */
    int             generation;    /* incremented each dispatch */
    atomic_int      done;
    int             active;
} g_tp;

static int g_n_threads = 1;


static void* tp_worker(void* arg) {
    int id = (int)(intptr_t)arg;
    int my_gen = 0;
    for (;;) {
        pthread_mutex_lock(&g_tp.mtx);
        while (g_tp.generation == my_gen && g_tp.active)
            pthread_cond_wait(&g_tp.wake, &g_tp.mtx);
        if (!g_tp.active) { pthread_mutex_unlock(&g_tp.mtx); return NULL; }
        my_gen = g_tp.generation;
        tp_fn fn = g_tp.fn;
        void* a = g_tp.args[id];
        pthread_mutex_unlock(&g_tp.mtx);

        if (a) fn(a);
        if (atomic_fetch_add(&g_tp.done, 1) + 1 >= g_tp.n_workers) {
            pthread_mutex_lock(&g_tp.mtx);
            pthread_cond_signal(&g_tp.done_cv);
            pthread_mutex_unlock(&g_tp.mtx);
        }
    }
    return NULL;
}

static void tp_init(int n) {
    /* n = total threads including main. Workers = n-1 */
    int n_workers = n - 1;
    if (n_workers < 1) return;
    if (g_tp.active && g_tp.n_workers == n) return;
    if (g_tp.active) {
        pthread_mutex_lock(&g_tp.mtx);
        g_tp.active = 0;
        pthread_cond_broadcast(&g_tp.wake);
        pthread_mutex_unlock(&g_tp.mtx);
        for (int i = 0; i < g_tp.n_workers - 1; i++)
            pthread_join(g_tp.thr[i], NULL);
        pthread_mutex_destroy(&g_tp.mtx);
        pthread_cond_destroy(&g_tp.wake);
        pthread_cond_destroy(&g_tp.done_cv);
    }
    memset(&g_tp, 0, sizeof(g_tp));
    pthread_mutex_init(&g_tp.mtx, NULL);
    pthread_cond_init(&g_tp.wake, NULL);
    pthread_cond_init(&g_tp.done_cv, NULL);
    g_tp.active = 1;
    g_tp.n_workers = n;  /* total threads including main */
    g_tp.generation = 0;
    atomic_store(&g_tp.done, 0);
    for (int i = 0; i < n_workers; i++)
        pthread_create(&g_tp.thr[i], NULL, tp_worker, (void*)(intptr_t)(i + 1));
}

/* Dispatch: main does task[0], workers do task[1..n-1] */
static void tp_run(tp_fn fn, void** args, int n_tasks) {
    if (n_tasks <= 1 || !g_tp.active) {
        if (n_tasks >= 1 && args[0]) fn(args[0]);
        return;
    }
    /* Set up and wake workers */
    pthread_mutex_lock(&g_tp.mtx);
    g_tp.fn = fn;
    for (int i = 0; i < g_tp.n_workers; i++)
        g_tp.args[i] = (i < n_tasks) ? args[i] : NULL;
    atomic_store(&g_tp.done, 0);
    g_tp.generation++;
    pthread_cond_broadcast(&g_tp.wake);
    pthread_mutex_unlock(&g_tp.mtx);

    /* Main thread does task[0] */
    if (args[0]) fn(args[0]);
    if (atomic_fetch_add(&g_tp.done, 1) + 1 >= g_tp.n_workers) {
        /* All done already */
        return;
    }

    /* Wait for stragglers */
    pthread_mutex_lock(&g_tp.mtx);
    while (atomic_load(&g_tp.done) < g_tp.n_workers)
        pthread_cond_wait(&g_tp.done_cv, &g_tp.mtx);
    pthread_mutex_unlock(&g_tp.mtx);
}

void tq_set_threads(int n_threads) {
    if (n_threads < 1) n_threads = 1;
    if (n_threads > TP_MAX) n_threads = TP_MAX;
    g_n_threads = n_threads;
    if (n_threads > 1) tp_init(n_threads);
}

int tq_get_threads(void) {
    return g_n_threads;
}

/* Public thread pool dispatch — allows other translation units to use the pool */
void tq_tp_run(void* (*fn)(void*), void** args, int n_tasks) {
    if (g_tp.active && n_tasks == g_tp.n_workers) {
        tp_run(fn, args, n_tasks);
    } else {
        /* Fallback: create/join pthreads */
        if (n_tasks <= 1) {
            if (n_tasks == 1 && args[0]) fn(args[0]);
            return;
        }
        pthread_t threads[TP_MAX];
        for (int t = 0; t < n_tasks; t++)
            pthread_create(&threads[t], NULL, fn, args[t]);
        for (int t = 0; t < n_tasks; t++)
            pthread_join(threads[t], NULL);
    }
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
 *
 * On Apple Silicon: uses Accelerate cblas_sgemv which automatically
 * dispatches to AMX coprocessor (2-5x faster than NEON).
 * ============================================================ */
void tq_matmul(float* out, const float* x, const float* w, int n, int d) {
#ifdef __APPLE__
    /* Apple Accelerate → AMX coprocessor for large FP32 matmuls.
     * cblas_sgemv is faster than NEON for large dimensions.
     * For small n (< 64), NEON is faster due to lower overhead. */
    if (n >= 64 && d >= 256) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, n, d,
                    1.0f, w, d, x, 1, 0.0f, out, 1);
        return;
    }
#endif

    int n_threads = g_n_threads;

    /* For small matrices or single-thread config, skip thread overhead */
    if (n < 256 || n_threads <= 1) {
        matmul_rows(out, x, w, 0, n, d);
        return;
    }

    /* Cap threads to available rows */
    if (n_threads > n) n_threads = n;
    if (n_threads > TP_MAX) n_threads = TP_MAX;

    matmul_task_t tasks[TP_MAX];
    void* ptrs[TP_MAX];

    int rows_per_thread = n / n_threads;
    for (int t = 0; t < n_threads; t++) {
        tasks[t].out = out;
        tasks[t].x = x;
        tasks[t].w = w;
        tasks[t].d = d;
        tasks[t].start_row = t * rows_per_thread;
        tasks[t].end_row = (t == n_threads - 1) ? n : (t + 1) * rows_per_thread;
        ptrs[t] = &tasks[t];
    }

    if (g_tp.active && n_threads == g_tp.n_workers) {
        tp_run(matmul_worker, ptrs, n_threads);
    } else {
        pthread_t threads[TP_MAX];
        for (int t = 0; t < n_threads; t++)
            pthread_create(&threads[t], NULL, matmul_worker, &tasks[t]);
        for (int t = 0; t < n_threads; t++)
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
            if (q0 < 0) { q0 = 0; } if (q0 > 15) { q0 = 15; }
            if (q1 < 0) { q1 = 0; } if (q1 > 15) { q1 = 15; }
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
            if (q0 < 0) { q0 = 0; } if (q0 > 15) { q0 = 15; }
            if (q1 < 0) { q1 = 0; } if (q1 > 15) { q1 = 15; }
            qb[j] = (uint8_t)((q1 << 4) | q0);
        }
        if (remainder & 1) {
            int q0 = (int)roundf(block[remainder - 1] * id) + 8;
            if (q0 < 0) { q0 = 0; } if (q0 > 15) { q0 = 15; }
            qb[n_pairs] = (uint8_t)(q0);
        }
    }
}

/* ============================================================
 * Q4 dequantize: packed 4-bit + per-block scale -> float
 *
 * Inverse of tq_quantize_row_q4. For each block of 32 values:
 *   x_i = (q_i - 8) * scale
 * where q_i is a 4-bit unsigned value [0,15].
 * ============================================================ */
void tq_dequantize_row_q4(const uint8_t* qs, const float* scales, float* dst, int n) {
    int n_blocks = n / 32;
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t* qb = qs + b * 16;
        float d = scales[b];
        float* out = dst + b * 32;
#ifdef __ARM_NEON
        /* Process 16 packed bytes → 32 float values using NEON.
         * Each byte packs two 4-bit values: lo nibble at even index,
         * hi nibble at odd index. vzip interleaves them correctly. */
        {
            uint8x16_t packed = vld1q_u8(qb);
            /* Extract lo nibbles (even-indexed output values) */
            uint8x8_t lo_lo = vand_u8(vget_low_u8(packed), vdup_n_u8(0x0F));
            uint8x8_t lo_hi = vand_u8(vget_high_u8(packed), vdup_n_u8(0x0F));
            /* Extract hi nibbles (odd-indexed output values) */
            uint8x8_t hi_lo = vshr_n_u8(vget_low_u8(packed), 4);
            uint8x8_t hi_hi = vshr_n_u8(vget_high_u8(packed), 4);
            /* Interleave: lo[0],hi[0],lo[1],hi[1],... */
            uint8x8x2_t zip0 = vzip_u8(lo_lo, hi_lo);
            uint8x8x2_t zip1 = vzip_u8(lo_hi, hi_hi);
            /* zip0.val[0] = first 8 interleaved, zip0.val[1] = next 8, etc. */
            float32x4_t vd_vec = vdupq_n_f32(d);
            float32x4_t v8f = vdupq_n_f32(8.0f);

            /* Process zip0.val[0]: output[0..7] */
            uint16x8_t w0 = vmovl_u8(zip0.val[0]);
            float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w0)));
            float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w0)));
            vst1q_f32(out + 0,  vmulq_f32(vsubq_f32(f0, v8f), vd_vec));
            vst1q_f32(out + 4,  vmulq_f32(vsubq_f32(f1, v8f), vd_vec));

            /* Process zip0.val[1]: output[8..15] */
            uint16x8_t w1 = vmovl_u8(zip0.val[1]);
            float32x4_t f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w1)));
            float32x4_t f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w1)));
            vst1q_f32(out + 8,  vmulq_f32(vsubq_f32(f2, v8f), vd_vec));
            vst1q_f32(out + 12, vmulq_f32(vsubq_f32(f3, v8f), vd_vec));

            /* Process zip1.val[0]: output[16..23] */
            uint16x8_t w2 = vmovl_u8(zip1.val[0]);
            float32x4_t f4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w2)));
            float32x4_t f5 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w2)));
            vst1q_f32(out + 16, vmulq_f32(vsubq_f32(f4, v8f), vd_vec));
            vst1q_f32(out + 20, vmulq_f32(vsubq_f32(f5, v8f), vd_vec));

            /* Process zip1.val[1]: output[24..31] */
            uint16x8_t w3 = vmovl_u8(zip1.val[1]);
            float32x4_t f6 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w3)));
            float32x4_t f7 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w3)));
            vst1q_f32(out + 24, vmulq_f32(vsubq_f32(f6, v8f), vd_vec));
            vst1q_f32(out + 28, vmulq_f32(vsubq_f32(f7, v8f), vd_vec));
        }
#else
        for (int j = 0; j < 16; j++) {
            int q0 = qb[j] & 0x0F;
            int q1 = qb[j] >> 4;
            out[2*j]     = (float)(q0 - 8) * d;
            out[2*j + 1] = (float)(q1 - 8) * d;
        }
#endif
    }
    /* Handle remainder */
    int remainder = n - n_blocks * 32;
    if (remainder > 0) {
        const uint8_t* qb = qs + n_blocks * 16;
        float d = scales[n_blocks];
        float* out = dst + n_blocks * 32;
        int n_pairs = remainder / 2;
        for (int j = 0; j < n_pairs; j++) {
            int q0 = qb[j] & 0x0F;
            int q1 = qb[j] >> 4;
            out[2*j]     = (float)(q0 - 8) * d;
            out[2*j + 1] = (float)(q1 - 8) * d;
        }
        if (remainder & 1) {
            int q0 = qb[n_pairs] & 0x0F;
            out[remainder - 1] = (float)(q0 - 8) * d;
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
    (void)x; /* activation already in x_q8 */
#ifdef __ARM_NEON
    const uint8x16_t mask_0f = vdupq_n_u8(0x0F);
    const uint8x16_t v8 = vdupq_n_u8(8);
#endif

    for (int i = start_row; i < end_row - 1; i += 2) {
        /* Process 2 rows simultaneously for better ILP */
        const uint8_t* wi0 = w_qs + (size_t)i * n_blocks * 16;
        const uint8_t* wi1 = w_qs + (size_t)(i + 1) * n_blocks * 16;
        const float* si0 = w_scales + (size_t)i * n_blocks;
        const float* si1 = w_scales + (size_t)(i + 1) * n_blocks;

#ifdef __ARM_NEON
        float32x4_t sumv0 = vdupq_n_f32(0.0f);
        float32x4_t sumv1 = vdupq_n_f32(0.0f);

        /* Process 2 blocks per iteration for reduced loop overhead */
        int b = 0;
        for (; b + 1 < n_blocks; b += 2) {
            /* Block b */
            uint8x16_t pk0_0 = vld1q_u8(wi0 + b * 16);
            uint8x16_t pk1_0 = vld1q_u8(wi1 + b * 16);
            int8x16x2_t xd0 = vld2q_s8(x_q8 + b * 32);

            int8x16_t lo0_0 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(pk0_0, mask_0f), v8));
            int8x16_t hi0_0 = vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(pk0_0, 4), v8));
            int8x16_t lo1_0 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(pk1_0, mask_0f), v8));
            int8x16_t hi1_0 = vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(pk1_0, 4), v8));

            int32x4_t a0_0 = vdupq_n_s32(0);
            int32x4_t a1_0 = vdupq_n_s32(0);
#if defined(__ARM_FEATURE_DOTPROD)
            a0_0 = vdotq_s32(vdotq_s32(a0_0, lo0_0, xd0.val[0]), hi0_0, xd0.val[1]);
            a1_0 = vdotq_s32(vdotq_s32(a1_0, lo1_0, xd0.val[0]), hi1_0, xd0.val[1]);
#else
            a0_0 = vaddq_s32(a0_0, vpaddlq_s16(vmull_s8(vget_low_s8(lo0_0), vget_low_s8(xd0.val[0]))));
            a0_0 = vaddq_s32(a0_0, vpaddlq_s16(vmull_s8(vget_high_s8(lo0_0), vget_high_s8(xd0.val[0]))));
            a0_0 = vaddq_s32(a0_0, vpaddlq_s16(vmull_s8(vget_low_s8(hi0_0), vget_low_s8(xd0.val[1]))));
            a0_0 = vaddq_s32(a0_0, vpaddlq_s16(vmull_s8(vget_high_s8(hi0_0), vget_high_s8(xd0.val[1]))));
            a1_0 = vaddq_s32(a1_0, vpaddlq_s16(vmull_s8(vget_low_s8(lo1_0), vget_low_s8(xd0.val[0]))));
            a1_0 = vaddq_s32(a1_0, vpaddlq_s16(vmull_s8(vget_high_s8(lo1_0), vget_high_s8(xd0.val[0]))));
            a1_0 = vaddq_s32(a1_0, vpaddlq_s16(vmull_s8(vget_low_s8(hi1_0), vget_low_s8(xd0.val[1]))));
            a1_0 = vaddq_s32(a1_0, vpaddlq_s16(vmull_s8(vget_high_s8(hi1_0), vget_high_s8(xd0.val[1]))));
#endif
            float s0 = x_scales[b];
            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(a0_0), si0[b] * s0);
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(a1_0), si1[b] * s0);

            /* Block b+1 */
            uint8x16_t pk0_1 = vld1q_u8(wi0 + (b + 1) * 16);
            uint8x16_t pk1_1 = vld1q_u8(wi1 + (b + 1) * 16);
            int8x16x2_t xd1 = vld2q_s8(x_q8 + (b + 1) * 32);

            int8x16_t lo0_1 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(pk0_1, mask_0f), v8));
            int8x16_t hi0_1 = vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(pk0_1, 4), v8));
            int8x16_t lo1_1 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(pk1_1, mask_0f), v8));
            int8x16_t hi1_1 = vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(pk1_1, 4), v8));

            int32x4_t a0_1 = vdupq_n_s32(0);
            int32x4_t a1_1 = vdupq_n_s32(0);
#if defined(__ARM_FEATURE_DOTPROD)
            a0_1 = vdotq_s32(vdotq_s32(a0_1, lo0_1, xd1.val[0]), hi0_1, xd1.val[1]);
            a1_1 = vdotq_s32(vdotq_s32(a1_1, lo1_1, xd1.val[0]), hi1_1, xd1.val[1]);
#else
            a0_1 = vaddq_s32(a0_1, vpaddlq_s16(vmull_s8(vget_low_s8(lo0_1), vget_low_s8(xd1.val[0]))));
            a0_1 = vaddq_s32(a0_1, vpaddlq_s16(vmull_s8(vget_high_s8(lo0_1), vget_high_s8(xd1.val[0]))));
            a0_1 = vaddq_s32(a0_1, vpaddlq_s16(vmull_s8(vget_low_s8(hi0_1), vget_low_s8(xd1.val[1]))));
            a0_1 = vaddq_s32(a0_1, vpaddlq_s16(vmull_s8(vget_high_s8(hi0_1), vget_high_s8(xd1.val[1]))));
            a1_1 = vaddq_s32(a1_1, vpaddlq_s16(vmull_s8(vget_low_s8(lo1_1), vget_low_s8(xd1.val[0]))));
            a1_1 = vaddq_s32(a1_1, vpaddlq_s16(vmull_s8(vget_high_s8(lo1_1), vget_high_s8(xd1.val[0]))));
            a1_1 = vaddq_s32(a1_1, vpaddlq_s16(vmull_s8(vget_low_s8(hi1_1), vget_low_s8(xd1.val[1]))));
            a1_1 = vaddq_s32(a1_1, vpaddlq_s16(vmull_s8(vget_high_s8(hi1_1), vget_high_s8(xd1.val[1]))));
#endif
            float s1 = x_scales[b + 1];
            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(a0_1), si0[b + 1] * s1);
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(a1_1), si1[b + 1] * s1);
        }
        /* Handle odd remaining block */
        for (; b < n_blocks; b++) {
            uint8x16_t pk0 = vld1q_u8(wi0 + b * 16);
            uint8x16_t pk1 = vld1q_u8(wi1 + b * 16);
            int8x16x2_t xd = vld2q_s8(x_q8 + b * 32);

            int8x16_t lo0 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(pk0, mask_0f), v8));
            int8x16_t hi0 = vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(pk0, 4), v8));
            int8x16_t lo1 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(pk1, mask_0f), v8));
            int8x16_t hi1 = vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(pk1, 4), v8));

            int32x4_t a0 = vdupq_n_s32(0);
            int32x4_t a1 = vdupq_n_s32(0);
#if defined(__ARM_FEATURE_DOTPROD)
            a0 = vdotq_s32(vdotq_s32(a0, lo0, xd.val[0]), hi0, xd.val[1]);
            a1 = vdotq_s32(vdotq_s32(a1, lo1, xd.val[0]), hi1, xd.val[1]);
#else
            a0 = vaddq_s32(a0, vpaddlq_s16(vmull_s8(vget_low_s8(lo0), vget_low_s8(xd.val[0]))));
            a0 = vaddq_s32(a0, vpaddlq_s16(vmull_s8(vget_high_s8(lo0), vget_high_s8(xd.val[0]))));
            a0 = vaddq_s32(a0, vpaddlq_s16(vmull_s8(vget_low_s8(hi0), vget_low_s8(xd.val[1]))));
            a0 = vaddq_s32(a0, vpaddlq_s16(vmull_s8(vget_high_s8(hi0), vget_high_s8(xd.val[1]))));
            a1 = vaddq_s32(a1, vpaddlq_s16(vmull_s8(vget_low_s8(lo1), vget_low_s8(xd.val[0]))));
            a1 = vaddq_s32(a1, vpaddlq_s16(vmull_s8(vget_high_s8(lo1), vget_high_s8(xd.val[0]))));
            a1 = vaddq_s32(a1, vpaddlq_s16(vmull_s8(vget_low_s8(hi1), vget_low_s8(xd.val[1]))));
            a1 = vaddq_s32(a1, vpaddlq_s16(vmull_s8(vget_high_s8(hi1), vget_high_s8(xd.val[1]))));
#endif
            float s = x_scales[b];
            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(a0), si0[b] * s);
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(a1), si1[b] * s);
        }
        out[i]     = vaddvq_f32(sumv0);
        out[i + 1] = vaddvq_f32(sumv1);
#else
        float sum0 = 0.0f, sum1 = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const int8_t* xb = x_q8 + b * 32;
            const uint8_t* qb0 = wi0 + b * 16;
            const uint8_t* qb1 = wi1 + b * 16;
            int32_t isum0 = 0, isum1 = 0;
            for (int j = 0; j < 16; j++) {
                int x0 = (int)xb[2 * j], x1 = (int)xb[2 * j + 1];
                isum0 += ((qb0[j] & 0x0F) - 8) * x0 + ((qb0[j] >> 4) - 8) * x1;
                isum1 += ((qb1[j] & 0x0F) - 8) * x0 + ((qb1[j] >> 4) - 8) * x1;
            }
            float s = x_scales[b];
            sum0 += (float)isum0 * si0[b] * s;
            sum1 += (float)isum1 * si1[b] * s;
        }
        out[i]     = sum0;
        out[i + 1] = sum1;
#endif
    }
    /* Handle odd remaining row */
    if ((end_row - start_row) & 1) {
        int i = end_row - 1;
        const uint8_t* wi = w_qs + (size_t)i * n_blocks * 16;
        const float* si = w_scales + (size_t)i * n_blocks;
#ifdef __ARM_NEON
        float32x4_t sumv = vdupq_n_f32(0.0f);
        for (int b = 0; b < n_blocks; b++) {
            uint8x16_t pk = vld1q_u8(wi + b * 16);
            int8x16x2_t xd = vld2q_s8(x_q8 + b * 32);
            int8x16_t lo = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(pk, mask_0f), v8));
            int8x16_t hi = vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(pk, 4), v8));
            int32x4_t a = vdupq_n_s32(0);
#if defined(__ARM_FEATURE_DOTPROD)
            a = vdotq_s32(vdotq_s32(a, lo, xd.val[0]), hi, xd.val[1]);
#else
            a = vaddq_s32(a, vpaddlq_s16(vmull_s8(vget_low_s8(lo), vget_low_s8(xd.val[0]))));
            a = vaddq_s32(a, vpaddlq_s16(vmull_s8(vget_high_s8(lo), vget_high_s8(xd.val[0]))));
            a = vaddq_s32(a, vpaddlq_s16(vmull_s8(vget_low_s8(hi), vget_low_s8(xd.val[1]))));
            a = vaddq_s32(a, vpaddlq_s16(vmull_s8(vget_high_s8(hi), vget_high_s8(xd.val[1]))));
#endif
            sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(a), si[b] * x_scales[b]);
        }
        out[i] = vaddvq_f32(sumv);
#else
        float sum = 0.0f;
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
        out[i] = sum;
#endif
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
/* Persistent Q8 workspace to avoid per-call malloc.
 * Protected by mutex: concurrent calls to tq_matmul_q4/q2 from different
 * threads could race on realloc. The workspace itself is read-only during
 * the parallel matmul phase (workers read different rows), so locking is
 * only needed around the resize + quantize step. */
static int8_t*  g_q8_buf = NULL;
static float*   g_q8_scales = NULL;
static int      g_q8_cap = 0;
static pthread_mutex_t g_q8_mutex = PTHREAD_MUTEX_INITIALIZER;

void tq_matmul_q4(float* out, const float* x, const uint8_t* w_qs, const float* w_scales,
                   int n, int d) {
#ifdef TQ_HAS_METAL
    {
        extern int tq_metal_batch_active(void);
        extern int tq_metal_matmul_q4(float*, const float*, const uint8_t*, const float*, int, int);
        /* GPU: batch mode (batched independent matmuls), or immediate for
         * very large matmuls where GPU throughput overcomes per-dispatch
         * overhead (~0.15ms). For batch-1 inference on Apple Silicon unified
         * memory, matmul is bandwidth-bound — Metal helps most when the
         * output dimension is very large (e.g., classifier/logits). Smaller
         * matmuls (attention, FFN) are faster on CPU via NEON Q4xQ8 path. */
        /* Only check Metal for very large output dims (vocab projection).
         * Batch mode is only active for GGUF layers, not Q4-converted. */
        if (n >= 8192 && tq_metal_batch_active()) {
            int rc = tq_metal_matmul_q4(out, x, w_qs, w_scales, n, d);
            if (rc == 0) return;
        }
    }
#endif
    /* Quantize activation x to Q8 (amortized across all rows) */
    pthread_mutex_lock(&g_q8_mutex);
    if (d > g_q8_cap) {
        free(g_q8_buf); free(g_q8_scales);
        g_q8_buf = (int8_t*)malloc((size_t)d * sizeof(int8_t));
        g_q8_scales = (float*)malloc((size_t)(d / 32 + 2) * sizeof(float));
        g_q8_cap = d;
    }
    int8_t* x_q8 = g_q8_buf;
    float* x_scales = g_q8_scales;
    if (!x_q8 || !x_scales) { pthread_mutex_unlock(&g_q8_mutex); return; }
    tq_quantize_row_q8(x, x_q8, x_scales, d);
    pthread_mutex_unlock(&g_q8_mutex);

    int n_threads = g_n_threads;

    if (n < 256 || n_threads <= 1) {
        matmul_q4_rows(out, x, w_qs, w_scales, x_q8, x_scales, 0, n, d);
        return;
    }

    if (n_threads > n) n_threads = n;
    if (n_threads > TP_MAX) n_threads = TP_MAX;

    matmul_q4_task_t tasks[TP_MAX];
    void* ptrs[TP_MAX];

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
        ptrs[t] = &tasks[t];
    }

    if (g_tp.active && n_threads == g_tp.n_workers) {
        tp_run(matmul_q4_worker, ptrs, n_threads);
    } else {
        pthread_t threads[TP_MAX];
        for (int t = 0; t < n_threads; t++)
            pthread_create(&threads[t], NULL, matmul_q4_worker, &tasks[t]);
        for (int t = 0; t < n_threads; t++)
            pthread_join(threads[t], NULL);
    }
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
    if (n_threads > TP_MAX) n_threads = TP_MAX;

    matmul_q4_task_t tasks[TP_MAX];
    void* ptrs[TP_MAX];

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
        ptrs[t] = &tasks[t];
    }

    if (g_tp.active && n_threads == g_tp.n_workers) {
        tp_run(matmul_q4_worker, ptrs, n_threads);
    } else {
        pthread_t threads[TP_MAX];
        for (int t = 0; t < n_threads; t++)
            pthread_create(&threads[t], NULL, matmul_q4_worker, &tasks[t]);
        for (int t = 0; t < n_threads; t++)
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
    if (n_threads > TP_MAX) n_threads = TP_MAX;

    matmul_bf16_task_t tasks[TP_MAX];
    void* ptrs[TP_MAX];

    int rows_per_thread = n / n_threads;
    for (int t = 0; t < n_threads; t++) {
        tasks[t].out = out;
        tasks[t].x = x;
        tasks[t].w_bf16 = w_bf16;
        tasks[t].d = d;
        tasks[t].start_row = t * rows_per_thread;
        tasks[t].end_row = (t == n_threads - 1) ? n : (t + 1) * rows_per_thread;
        ptrs[t] = &tasks[t];
    }

    if (g_tp.active && n_threads == g_tp.n_workers) {
        tp_run(matmul_bf16_worker, ptrs, n_threads);
    } else {
        pthread_t threads[TP_MAX];
        for (int t = 0; t < n_threads; t++)
            pthread_create(&threads[t], NULL, matmul_bf16_worker, &tasks[t]);
        for (int t = 0; t < n_threads; t++)
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
 * GELU with tanh approximation (Gemma3 GeGLU activation)
 * gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * ============================================================ */
void tq_gelu_tanh(float* x, int n) {
    for (int i = 0; i < n; i++) {
        float xi = x[i];
        float x3 = xi * xi * xi;
        float inner = 0.7978845608f * (xi + 0.044715f * x3);
        x[i] = 0.5f * xi * (1.0f + tanhf(inner));
    }
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
 * Q2_0 quantization: float -> packed 2-bit + per-block scale (block_size=32)
 *
 * Uses Lloyd-Max optimal codebook for Gaussian data:
 *   4 centroids: {-1.5104, -0.4528, 0.4528, 1.5104} (indices 0,1,2,3)
 * For each block of 32 values:
 *   scale = amax / 1.5104  (normalize so max maps to outermost centroid)
 *   q_i = nearest centroid index (0..3)
 * Packed: four 2-bit values per byte, LSB-first.
 * Block layout: 8 bytes packed + 4 bytes float scale = 12 bytes per 32 values.
 * This is ~1.7x more compact than Q4_0 (20 bytes per 32 values).
 * ============================================================ */

/* Lloyd-Max centroids for N(0,1) at 2 bits */
static const float Q2_CENTROIDS[4] = {-1.5104f, -0.4528f, 0.4528f, 1.5104f};

void tq_quantize_row_q2(const float* src, uint8_t* dst_qs, float* dst_scales, int n) {
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
        /* Scale: normalize so amax maps to centroid 1.5104 */
        float d = amax / 1.5104f;
        dst_scales[b] = d;
        float id = (d > 1e-10f) ? 1.0f / d : 0.0f;

        /* Quantize and pack 4 values per byte */
        uint8_t* qb = dst_qs + b * 8;
        memset(qb, 0, 8);
        for (int j = 0; j < 32; j++) {
            float x = block[j] * id;
            /* Find nearest centroid (linear search, only 4 entries) */
            int best = 0;
            float best_dist = fabsf(x - Q2_CENTROIDS[0]);
            for (int c = 1; c < 4; c++) {
                float dist = fabsf(x - Q2_CENTROIDS[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }
            /* Pack: 4 values per byte, 2 bits each, LSB-first */
            int byte_idx = j / 4;
            int bit_pos  = (j % 4) * 2;
            qb[byte_idx] |= (uint8_t)((best & 0x03) << bit_pos);
        }
    }
    /* Handle remainder */
    int remainder = n - n_blocks * 32;
    if (remainder > 0) {
        const float* block = src + n_blocks * 32;
        float amax = 0.0f;
        for (int j = 0; j < remainder; j++) {
            float a = fabsf(block[j]);
            if (a > amax) amax = a;
        }
        float d = amax / 1.5104f;
        dst_scales[n_blocks] = d;
        float id = (d > 1e-10f) ? 1.0f / d : 0.0f;
        uint8_t* qb = dst_qs + n_blocks * 8;
        int rem_bytes = (remainder + 3) / 4;
        memset(qb, 0, (size_t)rem_bytes);
        for (int j = 0; j < remainder; j++) {
            float x = block[j] * id;
            int best = 0;
            float best_dist = fabsf(x - Q2_CENTROIDS[0]);
            for (int c = 1; c < 4; c++) {
                float dist = fabsf(x - Q2_CENTROIDS[c]);
                if (dist < best_dist) { best_dist = dist; best = c; }
            }
            int byte_idx = j / 4;
            int bit_pos  = (j % 4) * 2;
            qb[byte_idx] |= (uint8_t)((best & 0x03) << bit_pos);
        }
    }
}

/* ============================================================
 * Q2 dequantize: packed 2-bit + per-block scale -> float
 *
 * Inverse of tq_quantize_row_q2. For each block of 32 values:
 *   x_i = Q2_CENTROIDS[q_i] * scale
 * where q_i is a 2-bit index [0,3].
 * ============================================================ */
void tq_dequantize_row_q2(const uint8_t* qs, const float* scales, float* dst, int n) {
    int n_blocks = n / 32;
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t* qb = qs + b * 8;
        float d = scales[b];
        float* out = dst + b * 32;
        for (int j = 0; j < 32; j++) {
            int byte_idx = j / 4;
            int bit_pos  = (j % 4) * 2;
            int qi = (qb[byte_idx] >> bit_pos) & 0x03;
            out[j] = Q2_CENTROIDS[qi] * d;
        }
    }
    /* Handle remainder */
    int remainder = n - n_blocks * 32;
    if (remainder > 0) {
        const uint8_t* qb = qs + n_blocks * 8;
        float d = scales[n_blocks];
        float* out = dst + n_blocks * 32;
        for (int j = 0; j < remainder; j++) {
            int byte_idx = j / 4;
            int bit_pos  = (j % 4) * 2;
            int qi = (qb[byte_idx] >> bit_pos) & 0x03;
            out[j] = Q2_CENTROIDS[qi] * d;
        }
    }
}

/* ============================================================
 * Q2 matmul: w is Q2_0 [n, d], x is Q8 [d], out is FP32 [n]
 *
 * For each row, unpack 2-bit indices, dequantize via centroid lookup,
 * then dot with Q8-quantized activation.
 *
 * Block layout: 8 bytes Q2 packed + float scale per 32 values.
 * To compute dot product efficiently we convert Q2 indices to signed
 * integer representatives and compute integer dot product with Q8 values:
 *   centroid_int[4] = {-3, -1, 1, 3} (scaled centroids * 2)
 *   dot = sum(centroid_int[qi] * x_q8[i]) * w_scale * x_scale * 0.5
 * This avoids float conversion in the inner loop.
 * ============================================================ */

/* Integer representatives for Q2 centroids: round(centroid * 2) */
static const int8_t Q2_INT_MAP[4] = {-3, -1, 1, 3};

typedef struct {
    float* out;
    const uint8_t* w_qs;
    const float* w_scales;
    const int8_t* x_q8;
    const float* x_scales;
    int start_row;
    int end_row;
    int d;
} matmul_q2_task_t;

static void matmul_q2_rows(float* out,
                            const uint8_t* w_qs, const float* w_scales,
                            const int8_t* x_q8, const float* x_scales,
                            int start_row, int end_row, int d) {
    int n_blocks = d / 32;
    for (int i = start_row; i < end_row; i++) {
        const uint8_t* wi = w_qs + (size_t)i * n_blocks * 8;
        const float* si = w_scales + (size_t)i * n_blocks;
        float sum = 0.0f;
#ifdef __ARM_NEON
        for (int b = 0; b < n_blocks; b++) {
            const uint8_t* qb = wi + b * 8;
            const int8_t* xb = x_q8 + b * 32;
            /* Unpack 8 bytes of Q2 into 32 int8 centroid values.
             * For each byte, extract 4 x 2-bit indices, map to {-3,-1,1,3}. */
            int8_t q2_vals[32];
            for (int j = 0; j < 8; j++) {
                uint8_t packed = qb[j];
                q2_vals[j * 4 + 0] = Q2_INT_MAP[(packed >> 0) & 0x03];
                q2_vals[j * 4 + 1] = Q2_INT_MAP[(packed >> 2) & 0x03];
                q2_vals[j * 4 + 2] = Q2_INT_MAP[(packed >> 4) & 0x03];
                q2_vals[j * 4 + 3] = Q2_INT_MAP[(packed >> 6) & 0x03];
            }
            /* Integer dot product using NEON sdot or widening multiply */
            int8x16_t vq0 = vld1q_s8(q2_vals);
            int8x16_t vq1 = vld1q_s8(q2_vals + 16);
            int8x16_t vx0 = vld1q_s8(xb);
            int8x16_t vx1 = vld1q_s8(xb + 16);
            int32x4_t acc = vdupq_n_s32(0);
#if defined(__ARM_FEATURE_DOTPROD)
            acc = vdotq_s32(acc, vq0, vx0);
            acc = vdotq_s32(acc, vq1, vx1);
#else
            acc = vaddq_s32(acc, vpaddlq_s16(vmull_s8(vget_low_s8(vq0), vget_low_s8(vx0))));
            acc = vaddq_s32(acc, vpaddlq_s16(vmull_s8(vget_high_s8(vq0), vget_high_s8(vx0))));
            acc = vaddq_s32(acc, vpaddlq_s16(vmull_s8(vget_low_s8(vq1), vget_low_s8(vx1))));
            acc = vaddq_s32(acc, vpaddlq_s16(vmull_s8(vget_high_s8(vq1), vget_high_s8(vx1))));
#endif
            int32_t isum = vaddvq_s32(acc);
            /* Scale: centroid_int = centroid * 2 / 1.5104, so multiply by 0.5 * 1.5104 = 0.7552 */
            sum += (float)isum * si[b] * x_scales[b] * 0.7552f;
        }
#else
        for (int b = 0; b < n_blocks; b++) {
            const uint8_t* qb = wi + b * 8;
            const int8_t* xb = x_q8 + b * 32;
            int32_t isum = 0;
            for (int j = 0; j < 8; j++) {
                uint8_t packed = qb[j];
                isum += Q2_INT_MAP[(packed >> 0) & 0x03] * (int)xb[j * 4 + 0];
                isum += Q2_INT_MAP[(packed >> 2) & 0x03] * (int)xb[j * 4 + 1];
                isum += Q2_INT_MAP[(packed >> 4) & 0x03] * (int)xb[j * 4 + 2];
                isum += Q2_INT_MAP[(packed >> 6) & 0x03] * (int)xb[j * 4 + 3];
            }
            sum += (float)isum * si[b] * x_scales[b] * 0.7552f;
        }
#endif
        out[i] = sum;
    }
}

static void* matmul_q2_worker(void* arg) {
    matmul_q2_task_t* t = (matmul_q2_task_t*)arg;
    matmul_q2_rows(t->out, t->w_qs, t->w_scales,
                    t->x_q8, t->x_scales,
                    t->start_row, t->end_row, t->d);
    return NULL;
}

/* Q2 matmul: quantize activation x to Q8 once, then Q2xQ8 integer dot products */
void tq_matmul_q2(float* out, const float* x, const uint8_t* w_qs, const float* w_scales,
                   int n, int d) {
    /* Quantize activation x to Q8 (reuse global buffer, mutex-protected) */
    pthread_mutex_lock(&g_q8_mutex);
    if (d > g_q8_cap) {
        free(g_q8_buf); free(g_q8_scales);
        g_q8_buf = (int8_t*)malloc((size_t)d * sizeof(int8_t));
        g_q8_scales = (float*)malloc((size_t)(d / 32 + 2) * sizeof(float));
        g_q8_cap = d;
    }
    int8_t* x_q8 = g_q8_buf;
    float* x_scales = g_q8_scales;
    if (!x_q8 || !x_scales) { pthread_mutex_unlock(&g_q8_mutex); return; }
    tq_quantize_row_q8(x, x_q8, x_scales, d);
    pthread_mutex_unlock(&g_q8_mutex);

    int n_threads = g_n_threads;

    if (n < 256 || n_threads <= 1) {
        matmul_q2_rows(out, w_qs, w_scales, x_q8, x_scales, 0, n, d);
        return;
    }

    if (n_threads > n) n_threads = n;
    if (n_threads > TP_MAX) n_threads = TP_MAX;

    matmul_q2_task_t tasks[TP_MAX];
    void* ptrs[TP_MAX];

    int rows_per_thread = n / n_threads;
    for (int t = 0; t < n_threads; t++) {
        tasks[t].out = out;
        tasks[t].w_qs = w_qs;
        tasks[t].w_scales = w_scales;
        tasks[t].x_q8 = x_q8;
        tasks[t].x_scales = x_scales;
        tasks[t].d = d;
        tasks[t].start_row = t * rows_per_thread;
        tasks[t].end_row = (t == n_threads - 1) ? n : (t + 1) * rows_per_thread;
        ptrs[t] = &tasks[t];
    }

    if (g_tp.active && n_threads == g_tp.n_workers) {
        tp_run(matmul_q2_worker, ptrs, n_threads);
    } else {
        pthread_t threads[TP_MAX];
        for (int t = 0; t < n_threads; t++)
            pthread_create(&threads[t], NULL, matmul_q2_worker, &tasks[t]);
        for (int t = 0; t < n_threads; t++)
            pthread_join(threads[t], NULL);
    }
}

/* Q2 matmul with pre-quantized activation (no redundant Q8 quantization) */
void tq_matmul_q2_preq(float* out, const uint8_t* w_qs, const float* w_scales,
                        const int8_t* x_q8, const float* x_scales,
                        int n, int d) {
    int n_threads = g_n_threads;

    if (n < 256 || n_threads <= 1) {
        matmul_q2_rows(out, w_qs, w_scales, x_q8, x_scales, 0, n, d);
        return;
    }

    if (n_threads > n) n_threads = n;
    if (n_threads > TP_MAX) n_threads = TP_MAX;

    matmul_q2_task_t tasks[TP_MAX];
    void* ptrs[TP_MAX];

    int rows_per_thread = n / n_threads;
    for (int t = 0; t < n_threads; t++) {
        tasks[t].out = out;
        tasks[t].w_qs = w_qs;
        tasks[t].w_scales = w_scales;
        tasks[t].x_q8 = x_q8;
        tasks[t].x_scales = x_scales;
        tasks[t].d = d;
        tasks[t].start_row = t * rows_per_thread;
        tasks[t].end_row = (t == n_threads - 1) ? n : (t + 1) * rows_per_thread;
        ptrs[t] = &tasks[t];
    }

    if (g_tp.active && n_threads == g_tp.n_workers) {
        tp_run(matmul_q2_worker, ptrs, n_threads);
    } else {
        pthread_t threads[TP_MAX];
        for (int t = 0; t < n_threads; t++)
            pthread_create(&threads[t], NULL, matmul_q2_worker, &tasks[t]);
        for (int t = 0; t < n_threads; t++)
            pthread_join(threads[t], NULL);
    }
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
    config.rep_penalty = 1.1f;
    config.rep_window = 32;
    config.rng_seed = 42ULL;
    config.on_token = NULL;
    config.user_data = NULL;
    return config;
}

/* ============================================================
 * RHT + Q4 + Q2 Residual Weight Quantization
 *
 * TurboQuant's novel approach: apply KV cache insights to weights.
 * 1. RHT (Walsh-Hadamard) → spreads outliers, uniformizes distribution
 * 2. Q4 quantize in RHT space → captures main signal
 * 3. Compute residual → Q2 quantize → captures correction
 * 4. At matmul: dequant(Q4) + dequant(Q2) in RHT space, dot with RHT(x)
 *
 * Achieves Q8 quality (cosine 0.9998) at 6-bit effective (~25% smaller than Q8).
 * ============================================================ */

/* Simplified Walsh-Hadamard butterfly (in-place) */
static void rht_transform(float* data, int n) {
    for (int step = 1; step < n; step *= 2) {
        for (int i = 0; i < n; i += step * 2) {
            for (int j = i; j < i + step && j + step < n; j++) {
                float a = data[j], b = data[j + step];
                data[j]        = (a + b) * 0.7071067811865475f;
                data[j + step] = (a - b) * 0.7071067811865475f;
            }
        }
    }
}

/* Quantize a single row: RHT → Q4 + Q2 residual
 * Stores Q4 in (qs4, sc4) and Q2 in (qs2, sc2).
 * Both use block_size=32. */
void tq_quantize_row_rht_q4q2(const float* src, 
                                uint8_t* qs4, float* sc4,
                                uint8_t* qs2, float* sc2,
                                float* rht_buf, int n) {
    /* Step 1: RHT */
    memcpy(rht_buf, src, (size_t)n * sizeof(float));
    rht_transform(rht_buf, n);
    
    /* Step 2: Q4 quantize */
    tq_quantize_row_q4(rht_buf, qs4, sc4, n);
    
    /* Step 3: Compute residual = RHT(src) - dequant(Q4) */
    float dequant_buf[32];
    int n_blocks = n / 32;
    for (int b = 0; b < n_blocks; b++) {
        float scale = sc4[b];
        for (int j = 0; j < 16; j++) {
            uint8_t packed = qs4[b * 16 + j];
            int lo = packed & 0xF;
            int hi = packed >> 4;
            dequant_buf[j]      = (float)(lo - 8) * scale;
            dequant_buf[j + 16] = (float)(hi - 8) * scale;
        }
        for (int j = 0; j < 32; j++) {
            rht_buf[b * 32 + j] -= dequant_buf[j];
        }
    }
    
    /* Step 4: Q2 quantize residual */
    tq_quantize_row_q2(rht_buf, qs2, sc2, n);
}

/* Matmul with RHT+Q4+Q2 weights: y[row] = (dequant_q4 + dequant_q2)(row) · RHT(x)
 * Uses existing tq_dequantize_row_q4/q2 for correctness. */
void tq_matmul_rht_q4q2(float* out, const float* x,
                          const uint8_t* w_qs4, const float* w_sc4,
                          const uint8_t* w_qs2, const float* w_sc2,
                          float* x_rht, int n, int d) {
    /* RHT the input once */
    memcpy(x_rht, x, (size_t)d * sizeof(float));
    rht_transform(x_rht, d);

    int nb = d / 32;
    size_t q4_row_bytes = (size_t)nb * 16;
    size_t q2_row_bytes = (size_t)nb * 8;
    /* Thread-local buffers to avoid per-call malloc */
    static __thread float* row_q4 = NULL;
    static __thread float* row_q2 = NULL;
    static __thread int row_cap = 0;
    if (d > row_cap) {
        free(row_q4); free(row_q2);
        row_q4 = (float*)malloc((size_t)d * sizeof(float));
        row_q2 = (float*)malloc((size_t)d * sizeof(float));
        row_cap = d;
    }

    for (int row = 0; row < n; row++) {
        /* Dequant Q4 component */
        tq_dequantize_row_q4(w_qs4 + row * q4_row_bytes,
                              w_sc4 + row * nb, row_q4, d);
        /* Dequant Q2 residual component */
        tq_dequantize_row_q2(w_qs2 + row * q2_row_bytes,
                              w_sc2 + row * nb, row_q2, d);
        /* Sum and dot with RHT(x) */
        float sum = 0;
        for (int j = 0; j < d; j++)
            sum += (row_q4[j] + row_q2[j]) * x_rht[j];
        out[row] = sum;
    }
    /* row_q4/row_q2 are thread-local, kept for reuse */
}

/* Q4+Q2 fused matmul: Q4 primary + Q2 residual correction.
 * out[row] = (dequant_q4(row) + dequant_q2(row)) · x
 * Uses tq_matmul_q4_preq for Q4, then adds Q2 correction. */
void tq_matmul_q4q2_preq(float* out,
                           const uint8_t* w_q4, const float* w_q4s,
                           const uint8_t* w_q2, const float* w_q2s,
                           const int8_t* x_q8, const float* x_scales,
                           int n, int d) {
    /* Q4 matmul */
    tq_matmul_q4_preq(out, w_q4, w_q4s, x_q8, x_scales, n, d);
    
    /* Q2 residual correction — uses thread-local static buffer to avoid hot-path malloc */
    if (w_q2 && w_q2s) {
        static __thread float* t_corr = NULL;
        static __thread int t_corr_cap = 0;
        if (n > t_corr_cap) {
            free(t_corr);
            t_corr = (float*)malloc((size_t)n * sizeof(float));
            t_corr_cap = n;
        }
        if (t_corr) {
            tq_matmul_q2_preq(t_corr, w_q2, w_q2s, x_q8, x_scales, n, d);
            for (int i = 0; i < n; i++) out[i] += t_corr[i];
        }
    }
}

/* ============================================================
 * 1-bit Weight Quantization (TurboQuant QJL method)
 *
 * Each weight row: FP32 → sign bits + L2 norm
 * matmul: y[r] = norm[r] / sqrt(dim) * sum(sign[j] * x[j])
 *
 * Uses per-row L2 norm as scale factor.
 * Compression: FP32 → 1 bit + 1 float/row ≈ 1.03 bpw
 * ============================================================ */

/* Per-row 1-bit quantize: store sign bits + L2 norm */
void tq_quantize_row_1bit(const float* src, uint8_t* sign_bits, float* norm_out, int n) {
    if (n <= 0) { *norm_out = 0; return; }
    float norm_sq = 0;
    for (int j = 0; j < n; j++) norm_sq += src[j] * src[j];
    *norm_out = sqrtf(norm_sq);

    int n_bytes = (n + 7) / 8;
    memset(sign_bits, 0, (size_t)n_bytes);
    for (int j = 0; j < n; j++) {
        if (src[j] > 0) sign_bits[j / 8] |= (1 << (j % 8));
    }
}

/* 1-bit matmul: y[r] = norm[r]/sqrt(dim) * sum(sign_match * x) */
void tq_matmul_1bit(float* out, const float* x,
                     const uint8_t* sign_data, const float* norms,
                     int n_rows, int dim) {
    float scale = 1.0f / sqrtf((float)dim);
    int n_bytes = (dim + 7) / 8;
    
    for (int r = 0; r < n_rows; r++) {
        const uint8_t* signs = sign_data + (size_t)r * n_bytes;
        float sum = 0;
        
#ifdef __ARM_NEON
        /* NEON: process 16 bytes (128 bits) at a time */
        int b = 0;
        float32x4_t vsum = vdupq_n_f32(0); (void)vsum; /* TODO: vectorize */
        for (; b + 15 < n_bytes; b += 16) {
            for (int k = 0; k < 16; k++) {
                uint8_t s = signs[b + k];
                int base = (b + k) * 8;
                for (int bit = 0; bit < 8 && base + bit < dim; bit++) {
                    float v = x[base + bit];
                    sum += (s & (1 << bit)) ? v : -v;
                }
            }
        }
        for (; b < n_bytes; b++) {
            uint8_t s = signs[b];
            int base = b * 8;
            for (int bit = 0; bit < 8 && base + bit < dim; bit++) {
                sum += (s & (1 << bit)) ? x[base + bit] : -x[base + bit];
            }
        }
#else
        for (int b = 0; b < n_bytes; b++) {
            uint8_t s = signs[b];
            int base = b * 8;
            for (int bit = 0; bit < 8 && base + bit < dim; bit++) {
                sum += (s & (1 << bit)) ? x[base + bit] : -x[base + bit];
            }
        }
#endif
        
        out[r] = norms[r] * scale * sum;
    }
}
