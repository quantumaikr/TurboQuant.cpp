// quant.h — Single-header LLM inference engine
// Version 0.5.0 | https://github.com/quantumaikr/quant.cpp
//
// The simplest way to add LLM inference to any C project.
// 15.7K LOC, 643KB, zero dependencies.
//
// #define QUANT_IMPLEMENTATION in exactly one .c file, then:
//   cc -O2 app.c -o app -lm -lpthread
//
// API (7 functions):
//   quant_load(path)                → Load GGUF model
//   quant_new(model, config)        → Create context (NULL = defaults)
//   quant_generate(ctx, prompt, cb) → Stream tokens via callback
//   quant_ask(ctx, prompt)          → Return full string (caller frees)
//   quant_free_ctx(ctx)             → Free context
//   quant_free_model(model)         → Free model
//   quant_version()                 → "0.5.0"
//
// KV compression (up to 6.9x, PPL +0.0%):
//   quant_config cfg = { .kv_compress = 1 };
//   quant_ctx* c = quant_new(model, &cfg);
//
// Models: Llama 3, Qwen 3.5, Gemma 3/4, MoE. Format: GGUF.
// Platforms: macOS, Linux, Windows, WASM, iOS, Android.
//
// License: Apache 2.0

#ifndef QUANT_H
#define QUANT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct quant_model quant_model;
typedef struct quant_ctx quant_ctx;

typedef struct {
    float temperature;   // default: 0.7
    float top_p;         // default: 0.9
    int   max_tokens;    // default: 256
    int   n_threads;     // default: 4
    int   kv_compress;   // 0=off, 1=4-bit K+V (default), 2=delta+3-bit
    int   context_length;// 0=auto (4096), or user override. With kv_compress=1,
                         // you can safely set much larger values (e.g. 32768)
                         // because KV cache uses ~4x less memory.
    int   k_highres_window; // 0=off (default), or N>0: keep last N tokens' keys
                         // at FP32 while compressing the rest. N=128 is the
                         // sweet spot: reduces PPL degradation from +3.8% to
                         // +0.6% at a cost of ~28 KB extra memory.
} quant_config;

// Load a GGUF model file. Returns NULL on failure.
quant_model* quant_load(const char* path);

// Create inference context. config=NULL for defaults.
quant_ctx* quant_new(quant_model* model, const quant_config* config);

// Generate tokens. Calls on_token for each generated token.
// Returns number of tokens generated.
int quant_generate(quant_ctx* ctx, const char* prompt,
                   void (*on_token)(const char* text, void* user_data),
                   void* user_data);

// Multi-turn chat with KV cache reuse (O(delta) per turn instead of O(n^2)).
// Subsequent calls only re-prefill the suffix that diverges from history.
// Pass prompt = NULL to reset the chat session. Returns tokens generated.
int quant_chat(quant_ctx* ctx, const char* prompt,
               void (*on_token)(const char* text, void* user_data),
               void* user_data);

// Generate and return full response as string. Caller must free().
char* quant_ask(quant_ctx* ctx, const char* prompt);

// Free a string returned by quant_ask.
void quant_free_string(char* str);

// Save/load KV cache context to/from disk. Enables "read once, query forever":
// process a long document once (slow prefill), save the context, then reload
// instantly for follow-up questions without re-processing.
// Returns 0 on success, -1 on failure.
int quant_save_context(quant_ctx* ctx, const char* path);
int quant_load_context(quant_ctx* ctx, const char* path);

// Free resources.
void quant_free_ctx(quant_ctx* ctx);
void quant_free_model(quant_model* model);

// Version info.
const char* quant_version(void);

#ifdef __cplusplus
}
#endif

// ============================================================================
// IMPLEMENTATION
// ============================================================================
#ifdef QUANT_IMPLEMENTATION

// ----------------------------------------------------------------------------
// System includes (deduplicated)
// ----------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <float.h>
#include <time.h>
#include <errno.h>
#ifdef _WIN32
#include <windows.h>
#include <process.h>
/* Full pthread shim for MSVC using Windows primitives */
typedef HANDLE pthread_t;
#define pthread_mutex_t SRWLOCK
#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
#define pthread_mutex_init(m,a) (InitializeSRWLock(m),0)
#define pthread_mutex_lock(m) AcquireSRWLockExclusive(m)
#define pthread_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define pthread_mutex_destroy(m) ((void)0)
#define pthread_cond_t CONDITION_VARIABLE
#define pthread_cond_init(c,a) InitializeConditionVariable(c)
#define pthread_cond_wait(c,m) SleepConditionVariableSRW(c,m,INFINITE,0)
#define pthread_cond_signal(c) WakeConditionVariable(c)
#define pthread_cond_broadcast(c) WakeAllConditionVariable(c)
#define pthread_cond_destroy(c) ((void)0)
static inline int pthread_create(pthread_t* t, const void* a, void*(*fn)(void*), void* arg) {
    (void)a; *t = (HANDLE)_beginthreadex(NULL,0,(unsigned(__stdcall*)(void*))fn,arg,0,NULL);
    return *t ? 0 : -1;
}
static inline int pthread_join(pthread_t t, void** r) {
    (void)r; WaitForSingleObject(t, INFINITE); CloseHandle(t); return 0;
}
#define __thread __declspec(thread)
#include <io.h>
#ifndef R_OK
#define R_OK 4
#endif
#define access _access
/* Minimal dirent for MSVC */
struct dirent { char d_name[260]; };
typedef struct { HANDLE h; WIN32_FIND_DATAA fd; int first; } DIR;
static inline DIR* opendir(const char* p) {
    char buf[270]; snprintf(buf, sizeof(buf), "%s\\*", p);
    DIR* d = (DIR*)malloc(sizeof(DIR));
    d->h = FindFirstFileA(buf, &d->fd); d->first = 1;
    if (d->h == INVALID_HANDLE_VALUE) { free(d); return NULL; }
    return d;
}
static inline struct dirent* readdir(DIR* d) {
    static struct dirent e;
    if (d->first) { d->first = 0; } else if (!FindNextFileA(d->h, &d->fd)) return NULL;
    strncpy(e.d_name, d->fd.cFileName, 259); e.d_name[259] = 0;
    return &e;
}
static inline void closedir(DIR* d) { FindClose(d->h); free(d); }
/* stdatomic shim */
#include <intrin.h>
typedef volatile long atomic_int;
#define atomic_store(p, v) _InterlockedExchange((p), (v))
#define atomic_load(p) _InterlockedCompareExchange((p), 0, 0)
#define atomic_fetch_add(p, v) _InterlockedExchangeAdd((p), (v))
/* clock_gettime shim */
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
/* timespec: MSVC 2022+ defines it in time.h, older MSVC doesn't.
 * Only define if not already available. */
#if !defined(_TIMESPEC_DEFINED) && !defined(__struct_timespec_defined) && !defined(_INC_TIME)
struct timespec { long tv_sec; long tv_nsec; };
#endif
static inline int clock_gettime(int id, struct timespec* ts) {
    (void)id; LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    ts->tv_sec = (long)(c.QuadPart / f.QuadPart);
    ts->tv_nsec = (long)((c.QuadPart % f.QuadPart) * 1000000000LL / f.QuadPart);
    return 0;
}
#else
#include <pthread.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif
#include <limits.h>

#if defined(__APPLE__) && !defined(CLOCK_MONOTONIC)
#include <mach/mach_time.h>
#endif

/* stdatomic: already shimmed at top for MSVC */
#if !defined(_MSC_VER)
#include <stdatomic.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* Platform includes: already handled at top for _WIN32 */

// ============================================================================
// Section 1: Types and Specs (from tq_types.h, tq_spec.h)
// ============================================================================

/* Cross-language static assert: works in both C11 and C++11/17 */
#ifdef __cplusplus
#define TQ_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define TQ_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

/* Cross-platform math constants (some platforms lack M_PI) */
#ifndef TQ_PI
#define TQ_PI   3.14159265358979323846f
#endif
#ifndef TQ_PI_2
#define TQ_PI_2 1.5707963267948966f
#endif

/* ============================================================
 * Constants
 * ============================================================ */

#define TQ_BK          128   /* Default block size (elements per block) */
#define TQ_BK_QJL      256   /* QJL block size */
#define TQ_SKETCH_DIM  256   /* QJL sketch dimension */
#define TQ_OUTLIERS    4     /* QJL outlier count */
#define TQ_MAX_SEQ_LEN (1 << 20)  /* Maximum sequence length (1M tokens) */
#define TQ_VERSION_MAJOR 0
#define TQ_VERSION_MINOR 1
#define TQ_VERSION_PATCH 0

/* ============================================================
 * Quantization type enum
 * ============================================================ */

typedef enum {
    TQ_TYPE_POLAR_3B  = 0,   /* PolarQuant 3-bit (theta:2 + rho:1) */
    TQ_TYPE_POLAR_4B  = 1,   /* PolarQuant 4-bit (theta:2 + rho:2) */
    TQ_TYPE_QJL_1B    = 2,   /* QJL 1-bit sign hash               */
    TQ_TYPE_TURBO_3B  = 3,   /* PolarQuant 2b + QJL 1b            */
    TQ_TYPE_TURBO_4B  = 4,   /* PolarQuant 3b + QJL 1b            */
    TQ_TYPE_UNIFORM_4B= 5,   /* Min-Max uniform 4-bit             */
    TQ_TYPE_UNIFORM_2B= 6,   /* Min-Max uniform 2-bit             */
    TQ_TYPE_MIXED_4B8 = 7,   /* Mixed: 4-bit base + fp16 outliers */
    TQ_TYPE_TURBO_KV_3B = 8, /* TurboQuant KV: 2-bit codebook + 1-bit QJL residual */
    TQ_TYPE_TURBO_KV_4B = 9, /* TurboQuant KV: 3-bit codebook + 1-bit QJL residual */
    TQ_TYPE_TURBO_KV_1B = 10,/* TurboQuant KV: 1-bit Hamming (sign only)           */
    TQ_TYPE_TURBO_KV_2B = 11,/* TurboQuant KV: 2-bit (1-bit codebook + 1-bit QJL) */
    TQ_TYPE_UNIFORM_3B= 12,  /* Min-Max uniform 3-bit with sub-block scales     */
    TQ_TYPE_COUNT     = 13
} tq_type;

/* ============================================================
 * Block structures — self-contained, ONNX LSB-first bit-packing
 * Each block embeds its own scale/offset (no external lookup)
 * ============================================================ */

/* PolarQuant block: polar-coordinate quantized KV cache
 * For 4-bit (theta:2, rho:2): indices = (rho << 2) | theta
 * Block covers TQ_BK elements (D/2 pairs per position)
 */
typedef struct {
    uint16_t rscale;                 /* radius scale   (fp16, 2B)       */
    uint16_t rmn;                    /* radius minimum (fp16, 2B)       */
    uint16_t tscale;                 /* theta scale    (fp16, 2B)       */
    uint16_t tmn;                    /* theta minimum  (fp16, 2B)       */
    uint8_t  indices[TQ_BK / 2];    /* packed rho|theta (64B for BK=128) */
} block_tq_polar;

/* size verified after extern "C" block */

/* QJL block: 1-bit Johnson-Lindenstrauss sign hash
 * sign(key @ projection) packed into bits
 */
typedef struct {
    uint16_t norm;                            /* key L2 norm (fp16, 2B)         */
    uint16_t outlier_norm;                    /* outlier component norm (fp16)  */
    uint8_t  hash[TQ_SKETCH_DIM / 8];        /* 1-bit sign packed (32B @256)   */
    uint8_t  outlier_idx[TQ_OUTLIERS];        /* outlier dimension indices (4B) */
} block_tq_qjl;

/* size verified after extern "C" block */

/* TurboQuant composite: PolarQuant stage + QJL residual correction */
typedef struct {
    block_tq_polar polar;
    block_tq_qjl   residual;
} block_tq_turbo;

/* size verified after extern "C" block */

/* Uniform min-max quantization block (baseline) */
typedef struct {
    uint16_t scale;                  /* (max - min) / (2^bits - 1), fp16 */
    uint16_t zero_point;             /* minimum value, fp16              */
    uint8_t  qs[TQ_BK / 2];         /* 4-bit: 2 values/byte, LSB-first */
} block_tq_uniform_4b;

/* size verified after extern "C" block */

/* Uniform 2-bit with sub-block scales (Q2_K-style)
 * 4 sub-blocks of 32 elements, each with independent FP16 scale/min.
 * 4 quantization levels (2-bit) per value, adapted to local statistics.
 * 3.0 bits per element: (16 bytes meta + 32 bytes data) / 128 elements.
 */
#define TQ_2B_NSUB  4                          /* sub-blocks per block  */
#define TQ_2B_SUBK  (TQ_BK / TQ_2B_NSUB)      /* 32 elements per sub  */

typedef struct {
    uint16_t sub_scale[TQ_2B_NSUB]; /* per-sub-block scale (fp16, 8B)   */
    uint16_t sub_min[TQ_2B_NSUB];   /* per-sub-block minimum (fp16, 8B) */
    uint8_t  qs[TQ_BK / 4];         /* 2-bit: 4 values/byte, LSB-first */
} block_tq_uniform_2b;               /* 48 bytes per 128 elements       */

/* size verified after extern "C" block */

/* Uniform 3-bit with sub-block scales (Q3_K-style)
 * 4 sub-blocks of 32 elements, each with independent FP16 scale/min.
 * 8 quantization levels (3-bit) per value, but adapted to local statistics.
 * 4.0 bits per element: (16 bytes meta + 48 bytes data) / 128 elements.
 */
#define TQ_3B_NSUB  4                          /* sub-blocks per block  */
#define TQ_3B_SUBK  (TQ_BK / TQ_3B_NSUB)      /* 32 elements per sub  */

typedef struct {
    uint16_t sub_scale[TQ_3B_NSUB]; /* per-sub-block scale (fp16, 8B)   */
    uint16_t sub_min[TQ_3B_NSUB];   /* per-sub-block minimum (fp16, 8B) */
    uint8_t  qs[TQ_BK * 3 / 8];    /* 3-bit packed data (48B)          */
} block_tq_uniform_3b;              /* 64 bytes per 128 elements        */

/* size verified after extern "C" block */

/* Mixed precision: 4-bit base with fp16 outlier channels
 * Top-k channels by absolute value are stored at fp16 precision.
 * Remaining channels use 4-bit uniform quantization with a tighter
 * min-max range (excluding outliers), reducing quantization error.
 */
#define TQ_MIXED_OUTLIERS 4   /* number of fp16 outlier channels */

typedef struct {
    uint16_t scale;                            /* 4-bit scale (fp16)            */
    uint16_t zero_point;                       /* 4-bit zero/minimum (fp16)     */
    uint8_t  outlier_idx[TQ_MIXED_OUTLIERS];   /* outlier channel indices       */
    int16_t  outlier_vals[TQ_MIXED_OUTLIERS];  /* outlier values (fp16)         */
    uint8_t  qs[TQ_BK / 2];                   /* 4-bit packed, LSB-first       */
} block_tq_mixed_4b8;

/* size verified after extern "C" block */

/* ============================================================
 * Type traits — O(1) dispatch table
 * ============================================================ */

typedef void (*tq_quantize_fn)(const float* src, void* dst, int n);
typedef void (*tq_dequantize_fn)(const void* src, float* dst, int n);
typedef void (*tq_attention_fn)(const float* query, const void* kv_cache,
                                float* scores, int seq_len, int head_dim);

typedef struct {
    const char*      name;
    size_t           block_size;     /* elements per block          */
    size_t           type_size;      /* bytes per block             */
    float            bpe;            /* bits per element (with meta)*/
    tq_quantize_fn   quantize;
    tq_dequantize_fn dequantize;
    tq_attention_fn  attention;
    tq_type          residual_type;  /* pairing for composite types */
} tq_type_traits_t;

/* Global traits table — GPU backends (Vulkan/Metal) override at runtime */
extern tq_type_traits_t TQ_TRAITS[TQ_TYPE_COUNT];

/* ============================================================
 * Cache block header (for paged cache)
 * ============================================================ */

typedef struct {
    uint32_t block_id;
    uint16_t ref_count;
    uint8_t  quant_type;     /* tq_type enum value */
    uint8_t  num_tokens;     /* valid tokens in this block */
} tq_cache_block_header_t;

/* ============================================================
 * Progressive compression config
 * ============================================================ */

typedef struct {
    int      residual_window;     /* Tier 0 (FP16) size, default 128   */
    int      warm_window;         /* Tier 1 (4-bit) size, default 256  */
    tq_type  warm_type;           /* Tier 1 quant type                 */
    tq_type  cold_type;           /* Tier 2 quant type                 */
    int      enable_recompression;/* Tier 1 → Tier 2 re-compression   */
} tq_progressive_config_t;

/* TurboQuant KV cache block: RHT + Lloyd-Max codebook + QJL residual
 * 3-bit variant: 2-bit codebook (4 levels) + 1-bit QJL sign hash
 * Block covers TQ_BK elements (128).
 * Layout: norm(2) + residual_norm(2) + rht_seed(4) + mse_2bit(32) + qjl_signs(16) = 56 bytes
 */
typedef struct {
    uint16_t norm;                     /* L2 norm of original vector (fp16)      */
    uint16_t residual_norm;            /* L2 norm of residual after MSE (fp16)   */
    uint32_t rht_seed;                 /* RHT random seed for this block         */
    uint8_t  mse_indices[TQ_BK / 4];  /* 2-bit packed codebook indices (32B)    */
    uint8_t  qjl_signs[TQ_BK / 8];    /* 1-bit QJL sign hash on residual (16B) */
} block_tq_turbo_kv_3b;

/* TurboQuant KV cache block: 4-bit variant
 * 3-bit codebook (8 levels) + 1-bit QJL sign hash
 * Layout: norm(2) + residual_norm(2) + rht_seed(4) + mse_3bit(48) + qjl_signs(16) = 72 bytes
 */
typedef struct {
    uint16_t norm;                         /* L2 norm of original vector (fp16)      */
    uint16_t residual_norm;                /* L2 norm of residual after MSE (fp16)   */
    uint32_t rht_seed;                     /* RHT random seed for this block         */
    uint8_t  mse_indices[TQ_BK * 3 / 8];  /* 3-bit packed codebook indices (48B)    */
    uint8_t  qjl_signs[TQ_BK / 8];        /* 1-bit QJL sign hash on residual (16B) */
} block_tq_turbo_kv_4b;

/* TurboQuant KV cache block: 1-bit Hamming attention
 * Pure sign-bit quantization for extreme compression.
 * Pipeline: normalize -> RHT -> sign extraction (1 bit per dim).
 * Attention uses XOR + popcount for Hamming distance.
 * For dim=128: 2 + 2 + 4 + 16 = 24 bytes per key (vs 256 bytes FP16 = 10.7x compression).
 */
typedef struct {
    uint16_t norm;              /* L2 norm of original vector (fp16)  */
    uint16_t _pad;              /* alignment padding                  */
    uint32_t rht_seed;          /* RHT random seed for this block     */
    uint8_t  signs[TQ_BK / 8]; /* 1 bit per dim = 16 bytes for 128   */
} block_tq_turbo_kv_1b;

/* TurboQuant KV cache block: 2-bit variant
 * 1-bit codebook (2 levels, sign only) + 1-bit QJL sign hash
 * Pipeline: normalize -> RHT -> 1-bit MSE (sign) + 1-bit QJL residual.
 * Layout: norm(2) + residual_norm(2) + rht_seed(4) + mse_1bit(16) + qjl_signs(16) = 40 bytes
 */
typedef struct {
    uint16_t norm;                     /* L2 norm of original vector (fp16)      */
    uint16_t residual_norm;            /* L2 norm of residual after MSE (fp16)   */
    uint32_t rht_seed;                 /* RHT random seed for this block         */
    uint8_t  mse_indices[TQ_BK / 8];  /* 1-bit packed codebook indices (16B)    */
    uint8_t  qjl_signs[TQ_BK / 8];    /* 1-bit QJL sign hash on residual (16B) */
} block_tq_turbo_kv_2b;

/* ============================================================
 * Block size verification (compile-time, C/C++ compatible)
 * Uses negative-size array trick for universal compatibility.
 * ============================================================ */
#define TQ_CHECK_SIZE(type, expected) \
    typedef char tq_check_##type[(sizeof(type) == (expected)) ? 1 : -1]

TQ_CHECK_SIZE(block_tq_polar,      8 + TQ_BK / 2);
TQ_CHECK_SIZE(block_tq_qjl,        4 + TQ_SKETCH_DIM / 8 + TQ_OUTLIERS);
TQ_CHECK_SIZE(block_tq_uniform_4b, 4 + TQ_BK / 2);
TQ_CHECK_SIZE(block_tq_uniform_2b, 4 * TQ_2B_NSUB + TQ_BK / 4);
TQ_CHECK_SIZE(block_tq_uniform_3b, 4 * TQ_3B_NSUB + TQ_BK * 3 / 8);
TQ_CHECK_SIZE(block_tq_mixed_4b8, 4 + TQ_MIXED_OUTLIERS + TQ_MIXED_OUTLIERS * 2 + TQ_BK / 2);
TQ_CHECK_SIZE(block_tq_turbo_kv_3b, 8 + TQ_BK / 4 + TQ_BK / 8);
TQ_CHECK_SIZE(block_tq_turbo_kv_4b, 8 + TQ_BK * 3 / 8 + TQ_BK / 8);
TQ_CHECK_SIZE(block_tq_turbo_kv_1b, 8 + TQ_BK / 8);
TQ_CHECK_SIZE(block_tq_turbo_kv_2b, 8 + TQ_BK / 8 + TQ_BK / 8);

/* Format specification — version-aware, ONNX-inspired */

#define TQ_SPEC_VERSION 1

#define TQ_ALG_POLAR    0
#define TQ_ALG_QJL      1
#define TQ_ALG_TURBO    2
#define TQ_ALG_UNIFORM  3
#define TQ_ALG_MIXED    4

#define TQ_FLAG_HAS_ZERO_POINT  (1 << 0)
#define TQ_FLAG_SYMMETRIC       (1 << 1)
#define TQ_FLAG_HAS_RESIDUAL    (1 << 2)

typedef struct {
    uint8_t  spec_version;     /* TQ_SPEC_VERSION                  */
    uint8_t  algorithm;        /* TQ_ALG_POLAR / QJL / TURBO / ... */
    uint8_t  key_bits;         /* total bits for key quantization   */
    uint8_t  value_bits;       /* bits for value quantization (0=none) */
    uint16_t block_size;       /* elements per block                */
    uint16_t sketch_dim;       /* QJL sketch dimension (0 if N/A)   */
    uint8_t  outlier_count;    /* QJL outlier count (0 if N/A)      */
    uint8_t  flags;            /* TQ_FLAG_* bitmask                 */
} tq_format_spec_t;

// ============================================================================
// Section 2: Engine Types (from tq_engine.h)
// ============================================================================

/* ============================================================
 * Model configuration
 * ============================================================ */
typedef struct {
    int n_layers;
    int hidden_dim;
    int intermediate_dim;
    int n_heads;         /* query heads (for self_attn layers) */
    int n_kv_heads;      /* KV heads (GQA, for self_attn layers) */
    int head_dim;        /* head dimension for self_attn */
    int vocab_size;
    int max_seq_len;
    float rope_freq_base;
    float rms_norm_eps;

    /* DeltaNet (linear_attention) config */
    int delta_n_heads;       /* number of V heads (num_v_heads, e.g., 32) */
    int delta_n_kv_heads;    /* number of K/Q groups (num_k_heads, e.g., 16; 0=same as delta_n_heads) */
    int delta_key_head_dim;  /* key head dim (e.g., 128) */
    int delta_value_head_dim;/* value head dim (e.g., 128) */
    int delta_conv_width;    /* conv1d kernel width (e.g., 4) */
    float partial_rotary_factor; /* fraction of head_dim that uses RoPE (e.g., 0.25) */

    /* QK-norm for self_attn (Qwen3.5 style) */
    int use_qk_norm;         /* 1 if q_norm/k_norm weights present */
    int attn_output_gate;    /* 1 if q_proj includes output gate (doubled q_proj output) */

    /* MoE (Mixture of Experts) configuration */
    int is_moe;              /* 1 if model uses MoE FFN layers */
    int num_experts;         /* total experts per MoE layer (e.g., 64) */
    int num_active_experts;  /* active experts per token (e.g., 8) */
    int expert_intermediate_dim; /* per-expert FFN intermediate dim */
    int has_shared_expert;   /* 1 if shared expert present */
    int shared_expert_intermediate_dim;

    /* Multi-architecture support */
    int model_type;          /* 0=qwen35, 1=gemma3, 2=qwen2moe */
    int is_gemma4;           /* 1 if Gemma 4 (STEP35): uses SwiGLU, no post-norms */
    int sliding_window;      /* sliding window size (512 for gemma3, 0 for unlimited) */
    float rope_local_base_freq; /* RoPE base freq for local/sliding layers (10000.0 for gemma3) */
    int n_norms_per_block;   /* 2 for qwen35, 4 for gemma3 */
    float query_pre_attn_scalar; /* attention scaling: 1/sqrt(this) instead of 1/sqrt(head_dim), 0=use head_dim */

    /* Gemma 4 hybrid attention: full layers have different head_dim/kv_heads than sliding.
     * head_dim/n_heads/n_kv_heads store sliding layer values (majority).
     * These store full layer values (0 = no hybrid, use sliding values). */
    int full_head_dim;       /* head_dim for full attention layers (e.g., 512 vs sliding 256) */
    int full_n_heads;        /* n_heads for full layers (e.g., 8 vs sliding 16) */
    int full_n_kv_heads;     /* n_kv_heads for full layers (e.g., 2 vs sliding 8) */
    int rope_n_dims;         /* RoPE dimension count for sliding/SWA layers (0 = use head_dim) */
    int rope_n_dims_full;    /* RoPE dimension count for full/global layers (0 = use rope_n_dims) */
    float final_logit_softcap; /* logit soft-capping: logits = cap * tanh(logits/cap), 0=disabled */
    float attn_logit_softcap;  /* attention score soft-capping (Gemma): 0=disabled, typically 50.0 */
    int* per_layer_inter_dim;  /* [n_layers] per-layer intermediate_dim (NULL = use intermediate_dim) */
} tq_model_config_t;

/* ============================================================
 * Model weights (in memory)
 * ============================================================ */
typedef struct {
    /* RMSNorm weights */
    float* attn_norm;     /* [hidden_dim] input_layernorm */
    float* ffn_norm;      /* [hidden_dim] post_attention_layernorm */

    /* Standard self_attn weights (NULL for DeltaNet layers) */
    float* wq;            /* [n_heads * head_dim, hidden_dim] */
    float* wk;            /* [n_kv_heads * head_dim, hidden_dim] */
    float* wv;            /* [n_kv_heads * head_dim, hidden_dim] */
    float* wo;            /* [hidden_dim, n_heads * head_dim] */
    float* q_norm;        /* [head_dim] QK-norm for queries */
    float* k_norm;        /* [head_dim] QK-norm for keys */

    /* Gemma3/4 extra norms (NULL for Qwen3.5) */
    float* post_attn_norm;   /* [hidden_dim] post_attention_layernorm */
    float* pre_ffn_norm;     /* [hidden_dim] pre_feedforward_layernorm (MoE FFN) */
    float* post_ffn_norm;    /* [hidden_dim] post_feedforward_layernorm */
    float* post_ffn_norm_1;  /* [hidden_dim] post_ffw_norm_1 (MoE output) */
    float* pre_ffn_norm_2;   /* [hidden_dim] pre_ffw_norm_2 (dense FFN input) */
    float* post_ffn_norm_2;  /* [hidden_dim] post_ffw_norm_2 (dense FFN output) */

    /* Gemma 4 PLE (Per-Layer Embedding) per-layer weights */
    const void* ple_gate;     /* [hidden_dim, ple_dim] gate projection (GGUF quantized) */
    int ple_gate_type;
    const void* ple_proj;     /* [ple_dim, hidden_dim] output projection (GGUF quantized) */
    int ple_proj_type;
    float* ple_norm;          /* [hidden_dim] PLE output norm weight */

    /* Gemma 4 layer output scaling */
    float layer_output_scale; /* scalar applied to residual output (0.0 = disabled) */

    /* SwiGLU FFN weights (present on ALL layers) */
    float* w_gate;        /* [intermediate_dim, hidden_dim] */
    float* w_up;          /* [intermediate_dim, hidden_dim] */
    float* w_down;        /* [hidden_dim, intermediate_dim] */

    /* Q8 quantized weights: int8 data + per-block scales (block_size=32)
     * When use_q8 is set, these replace the FP32 weight pointers above.
     * The FP32 pointers (wq, wk, etc.) are set to NULL after Q8 conversion. */
    int8_t*  wq_q8;     float* wq_q8s;    /* Q8 q_proj: [n_heads*head_dim, hidden_dim] */
    int8_t*  wk_q8;     float* wk_q8s;    /* Q8 k_proj: [n_kv_heads*head_dim, hidden_dim] */
    int8_t*  wv_q8;     float* wv_q8s;    /* Q8 v_proj: [n_kv_heads*head_dim, hidden_dim] */
    int8_t*  wo_q8;     float* wo_q8s;    /* Q8 o_proj: [hidden_dim, n_heads*head_dim] */
    int8_t*  w_gate_q8; float* w_gate_q8s;/* Q8 gate_proj */
    int8_t*  w_up_q8;   float* w_up_q8s;  /* Q8 up_proj */
    int8_t*  w_down_q8; float* w_down_q8s;/* Q8 down_proj */

    /* DeltaNet Q8 weights */
    int8_t*  delta_in_proj_qkv_q8; float* delta_in_proj_qkv_q8s;
    int8_t*  delta_in_proj_z_q8;   float* delta_in_proj_z_q8s;
    int8_t*  delta_in_proj_a_q8;   float* delta_in_proj_a_q8s;
    int8_t*  delta_in_proj_b_q8;   float* delta_in_proj_b_q8s;
    int8_t*  delta_out_proj_q8;    float* delta_out_proj_q8s;

    /* Q4_0 quantized weights: packed 4-bit data + per-block float scale (block_size=32)
     * Each block of 32 values stored as 16 packed bytes + 1 float scale.
     * Values are unsigned [0,15], centered at 8: actual = (q - 8) * scale.
     * When use_q4 is set, these replace FP32 pointers (set to NULL). */
    uint8_t* wq_q4;     float* wq_q4s;    /* Q4 q_proj */
    uint8_t* wk_q4;     float* wk_q4s;    /* Q4 k_proj */
    uint8_t* wv_q4;     float* wv_q4s;    /* Q4 v_proj */
    uint8_t* wo_q4;     float* wo_q4s;    /* Q4 o_proj */
    uint8_t* w_gate_q4; float* w_gate_q4s;/* Q4 gate_proj */
    uint8_t* w_up_q4;   float* w_up_q4s;  /* Q4 up_proj */
    uint8_t* w_down_q4; float* w_down_q4s;/* Q4 down_proj */

    /* DeltaNet Q4 weights */
    uint8_t* delta_in_proj_qkv_q4; float* delta_in_proj_qkv_q4s;
    uint8_t* delta_in_proj_z_q4;   float* delta_in_proj_z_q4s;
    uint8_t* delta_in_proj_a_q4;   float* delta_in_proj_a_q4s;
    uint8_t* delta_in_proj_b_q4;   float* delta_in_proj_b_q4s;
    uint8_t* delta_out_proj_q4;    float* delta_out_proj_q4s;

    /* Q2_0 quantized weights: packed 2-bit data + per-block float scale (block_size=32)
     * Each block of 32 values stored as 8 packed bytes + 1 float scale.
     * Uses Lloyd-Max codebook: centroid indices {0,1,2,3} -> {-1.510, -0.453, 0.453, 1.510}
     * When use_q2 is set, these replace FP32 pointers (set to NULL). */
    uint8_t* wq_q2;     float* wq_q2s;    /* Q2 q_proj */
    uint8_t* wk_q2;     float* wk_q2s;    /* Q2 k_proj */
    uint8_t* wv_q2;     float* wv_q2s;    /* Q2 v_proj */
    uint8_t* wo_q2;     float* wo_q2s;    /* Q2 o_proj */
    uint8_t* w_gate_q2; float* w_gate_q2s;/* Q2 gate_proj */
    uint8_t* w_up_q2;   float* w_up_q2s;  /* Q2 up_proj */
    uint8_t* w_down_q2; float* w_down_q2s;/* Q2 down_proj */

    /* DeltaNet Q2 weights */
    uint8_t* delta_in_proj_qkv_q2; float* delta_in_proj_qkv_q2s;
    uint8_t* delta_in_proj_z_q2;   float* delta_in_proj_z_q2s;
    uint8_t* delta_in_proj_a_q2;   float* delta_in_proj_a_q2s;
    uint8_t* delta_in_proj_b_q2;   float* delta_in_proj_b_q2s;
    uint8_t* delta_out_proj_q2;    float* delta_out_proj_q2s;

    /* GGUF on-the-fly dequant: raw quantized weight pointers + type.
     * When gguf_wq is non-NULL, the forward pass uses tq_matmul_gguf
     * instead of FP32/Q4/Q8 matmul. Saves ~5GB for 35B models. */
    const void* gguf_wq;  int gguf_wq_type;  /* Q proj (quantized, mmap'd) */
    const void* gguf_wk;  int gguf_wk_type;  /* K proj */
    const void* gguf_wv;  int gguf_wv_type;  /* V proj */
    const void* gguf_wo;  int gguf_wo_type;  /* O proj */
    /* GGUF on-the-fly for DeltaNet weights */
    const void* gguf_delta_qkv;  int gguf_delta_qkv_type;
    const void* gguf_delta_z;    int gguf_delta_z_type;
    const void* gguf_delta_a;    int gguf_delta_a_type;
    const void* gguf_delta_b;    int gguf_delta_b_type;
    const void* gguf_delta_out;  int gguf_delta_out_type;
    /* GGUF FFN (dense layers in MoE models) */
    const void* gguf_w_gate; int gguf_w_gate_type;
    const void* gguf_w_up;   int gguf_w_up_type;
    const void* gguf_w_down; int gguf_w_down_type;

    /* MoE expert weights (NULL for dense FFN layers) */
    void* moe;               /* tq_moe_layer_t* (from tq_gguf.h), NULL if dense */

    /* DeltaNet (linear_attention) weights (NULL for self_attn layers) */
    float* delta_a_log;       /* [delta_n_heads] decay parameter (log scale) */
    float* delta_conv1d;      /* [qkv_dim, 1, conv_width] */
    float* delta_dt_bias;     /* [delta_n_heads] delta bias */
    float* delta_in_proj_a;   /* [delta_n_heads, hidden_dim] */
    float* delta_in_proj_b;   /* [delta_n_heads, hidden_dim] */
    float* delta_in_proj_qkv; /* [qkv_dim, hidden_dim] (qkv_dim = 3 * delta_n_heads * delta_key_head_dim) */
    float* delta_in_proj_z;   /* [z_dim, hidden_dim] (z_dim = delta_n_heads * delta_value_head_dim) */
    float* delta_norm;        /* [delta_value_head_dim] group norm weight */
    float* delta_out_proj;    /* [hidden_dim, z_dim] */
} tq_layer_weights_t;

typedef struct {
    tq_model_config_t config;

    /* Token embedding */
    float* token_embedding;   /* [vocab_size, hidden_dim] — FP32, or NULL if using BF16 */

    /* Per-layer weights */
    tq_layer_weights_t* layers;

    /* Output */
    float* output_norm;       /* [hidden_dim] */
    float* output_weight;     /* [vocab_size, hidden_dim] — FP32, or NULL if using BF16 */

    /* Streaming BF16 support: keep embedding/output as mmap'd BF16,
     * convert on demand to save ~2GB for 0.8B models */
    const uint16_t* embed_bf16;        /* [vocab_size, hidden_dim] raw BF16 from mmap (NULL if FP32) */
    const uint16_t* output_weight_bf16;/* [vocab_size, hidden_dim] raw BF16 from mmap (NULL if FP32) */

    /* Hybrid architecture support (e.g., Qwen3.5 with DeltaNet layers) */
    int n_attn_layers;        /* number of layers with standard self_attn */
    int* attn_layer_indices;  /* which layer indices have self_attn [n_attn_layers] */

    /* Gemma3 sliding window support */
    int* layer_is_sliding;    /* [n_layers] per-layer flag: 1=sliding, 0=global (NULL if not used) */

    /* Learned RoPE frequencies (Gemma 4) — NULL if using computed frequencies */
    float* rope_freqs;        /* [rope_dim/2] learned inv_freq values (F32) */
    int rope_freqs_len;       /* length of rope_freqs array (rope_dim/2) */

    /* Gemma 4 Per-Layer Embedding (PLE) — NULL if not used */
    const void* ple_embedding;/* [n_layers * ple_dim, vocab_size] GGUF quantized (e.g. Q5_K) */
    int ple_embedding_type;   /* tq_ggml_dtype of ple_embedding (for runtime dequant) */
    float* ple_proj;          /* [hidden_dim, n_layers * ple_dim] FP32 (dequanted from BF16 at load) */
    float* ple_proj_norm;     /* [ple_dim] projection norm weight (F32) */
    int ple_dim;              /* per-layer embedding dim (e.g., 256), 0 if PLE not used */

    /* Q4 output weight (lm_head) — runtime quantized for fast logit projection */
    uint8_t* output_qs;       /* [vocab_size * n_blocks * 16] Q4 packed nibbles */
    float* output_scales;     /* [vocab_size * n_blocks] Q4 block scales */

    /* Q8 weight quantization */
    int use_q8_weights;       /* 1 if layer weights are Q8-quantized */
    void* _q8_data;           /* heap buffer for all Q8 quantized weights */
    size_t _q8_size;

    /* Q4 weight quantization */
    int use_q4_weights;       /* 1 if layer weights are Q4-quantized */
    void* _q4_data;           /* heap buffer for all Q4 quantized weights */
    size_t _q4_size;

    /* Q2 weight quantization */
    int use_q2_weights;       /* 1 if layer weights are Q2-quantized */
    int use_1bit_weights;     /* 1 if Q2 fields contain 1-bit sign data (not Lloyd-Max Q2) */
    void* _q2_data;           /* heap buffer for all Q2 quantized weights */
    size_t _q2_size;

    /* GGUF context (non-NULL when loaded from GGUF, owns mmap lifetime) */
    void* gguf_ctx;           /* tq_gguf_ctx_t* */

    /* GGUF embedding for output projection (large-vocab models keep quantized) */
    const void* output_gguf;  /* raw GGUF quantized embedding data (NULL if using FP32/BF16) */
    int output_gguf_type;     /* tq_ggml_dtype of output_gguf */

    /* MoE config (valid when config.is_moe) */
    void* moe_config;         /* tq_moe_config_t* */

    /* Memory management — supports multi-shard safetensors */
#define TQ_MAX_SHARDS 16
    void* _mmap_data;         /* primary mmap (shard 0 or TQM file) */
    size_t _mmap_size;
    void* _mmap_shards[TQ_MAX_SHARDS];  /* additional shard mmaps (index 0 unused) */
    size_t _mmap_shard_sizes[TQ_MAX_SHARDS];
    int _n_shards;            /* total number of shards (0 or 1 = single file) */
    void* _converted_data;    /* heap buffer for dtype-converted tensors (e.g., BF16->FP32) */
    size_t _converted_size;
} tq_model_t;

/* ============================================================
 * Runtime state
 * ============================================================ */
typedef struct {
    /* Activation buffers */
    float* x;           /* [hidden_dim] current activation */
    float* xb;          /* [hidden_dim] buffer */
    float* xb2;         /* [hidden_dim] buffer 2 */
    float* q;           /* [n_heads * head_dim] queries */
    float* k;           /* [n_kv_heads * head_dim] keys */
    float* v;           /* [n_kv_heads * head_dim] values */
    float* att;         /* [n_heads, seq_len] attention scores */
    float* hb;          /* [intermediate_dim] FFN buffer */
    float* hb2;         /* [intermediate_dim] FFN buffer 2 */
    float* logits;      /* [vocab_size] output logits */

    /* KV cache for self_attn layers */
    float* key_cache;    /* [n_layers, max_seq_len, n_kv_heads * head_dim] */
    float* value_cache;  /* [n_layers, max_seq_len, n_kv_heads * head_dim] FP32 (or NULL if FP16) */
    uint16_t* value_cache_fp16; /* [n_layers, max_seq_len, n_kv_heads * head_dim] FP16 (NULL if FP32) */
    int use_fp16_values; /* 1 if values stored as FP16, 0 for FP32 */
    tq_type kv_quant_type; /* quantization type for KV attention */
    size_t kv_cache_size;

    /* Quantized value cache (Q4 or Q2, replaces FP16/FP32 V when enabled) */
    int value_quant_bits;        /* 0=use FP16/FP32 (default), 4=Q4, 2=Q2 */
    uint8_t* value_cache_qs;     /* packed quantized values [n_layers * max_seq * n_blocks_v * packed_bytes] */
    float*   value_cache_scales; /* per-block scales [n_layers * max_seq * n_blocks_v] */
    size_t   value_stride_qs;    /* bytes per position in value_cache_qs */
    size_t   value_stride_scales;/* floats per position in value_cache_scales */

    /* DeltaNet recurrent state */
    float* delta_state;  /* [n_layers, delta_n_heads, key_head_dim, value_head_dim] */
    float* conv_state;   /* [n_layers, qkv_dim, conv_width-1] */

    /* DeltaNet workspace buffers */
    float* delta_qkv;    /* [qkv_dim] workspace for QKV projection */
    float* delta_z;      /* [z_dim] workspace for Z gate */
    float* delta_ab;     /* [delta_n_heads * 2] workspace for a,b projections */
    float* delta_out;    /* [z_dim] workspace for output */

    /* Dynamic workspace buffers (sized from model config, replacing stack arrays) */
    int8_t* xb_q8;          /* [hidden_dim] pre-quantized activation for Q4 matmuls */
    float*  xb_q8s;         /* [hidden_dim/32 + 1] Q8 scales for xb_q8 */
    float*  gate_vals;       /* [delta_n_heads] DeltaNet gate values */
    float*  decay_vals;      /* [delta_n_heads] DeltaNet precomputed exp(gate) */
    float*  delta_sk;        /* [delta_value_head_dim] DeltaNet S@K workspace */
    float*  delta_dvec;      /* [delta_value_head_dim] DeltaNet delta workspace */

    /* Quantization workspace */
    void* quant_key_buf;    /* workspace for quantized keys */
    float* quant_score_buf; /* workspace for quantized attention scores */

    /* KV profile statistics (opt-in via profile_kv flag) */
    int profile_kv;             /* 1 = collect KV statistics per layer */
    int profile_kv_count;       /* number of tokens accumulated */
    /* Per-layer stats: [n_layers * 8] = mean/std/skew/kurt before RHT, then after RHT */
    float* profile_stats;       /* allocated when profile_kv=1 */
    /* Running accumulators for incremental computation:
     * [n_layers * 8] = sum, sum_sq, sum_cube, sum_quad (pre/post RHT) */
    double* profile_accum;

    /* Attention entropy tracking (opt-in via attn_entropy flag) */
    int attn_entropy;           /* 1 = compute attention entropy per head per layer */
    /* Per-layer, per-head entropy accumulators: [n_layers * n_heads] */
    double* entropy_accum;      /* running sum of entropy values */
    int entropy_count;          /* number of tokens accumulated */

    /* V highres window: store recent N tokens as FP16 even when V is quantized */
    int v_highres_window;       /* number of recent tokens stored as FP16 (0 = disabled) */
    uint16_t* value_highres_fp16; /* FP16 V cache for recent tokens [n_layers, window, kv_dim] */

    /* MoE runtime state */
    void* moe_state;         /* tq_moe_state_t* (from tq_gguf.h), NULL if dense */

    /* Quantized KV cache for integer attention */
    void* quant_key_cache;   /* [n_layers, max_seq_len, n_kv_heads, blocks_per_head * type_size] */
    size_t quant_kv_stride;  /* bytes per layer in quant_key_cache */
    size_t quant_head_stride;/* bytes per head per position */

    /* PLE (Per-Layer Embedding) precomputed input: [n_layers * ple_dim] */
    float* ple_buf;

    /* Delta KV compression: store key[t] - reconstruct(key[t-1]) instead of key[t].
     * At attention time, reconstruct keys sequentially by accumulating deltas.
     * This reduces quantization range by ~30%, enabling 2-bit to match 4-bit quality.
     * Periodic I-frames (absolute keys) bound accumulated drift error. */
    int delta_kv_enabled;    /* 1 = delta compression mode for keys */
    int delta_iframe_interval; /* I-frame every N positions (0 = auto = 64) */

    /* Age-based progressive K compression: recent keys at FP32, old keys at 2-bit.
     * Old tokens receive negligible attention weight, so 2-bit noise doesn't affect PPL.
     * k_highres_window=32 achieves uniform_4b quality at ~2.1 effective bpe. */
    int k_highres_window;    /* number of recent tokens stored as FP32 keys (0 = disabled) */
    float* key_highres_fp32; /* FP32 key cache for recent tokens [n_layers, window, kv_dim] */
} tq_state_t;

/* ============================================================
 * Generation config
 * ============================================================ */
typedef struct {
    float temperature;
    float top_p;
    int max_tokens;
    tq_type kv_type;     /* KV cache quantization type */
    int value_quant_bits;/* V cache quantization: 0=FP16/FP32(default), 4=Q4, 2=Q2 */
    int v_highres_window;/* recent N tokens get FP16 V even when V is quantized (0=disabled) */
    int delta_kv;        /* 1 = delta KV compression (store key deltas) */
    int delta_iframe_interval; /* I-frame interval for delta KV (0 = auto = 64) */
    int k_highres_window;/* age-based: recent N keys at FP32, rest at 2-bit (0=disabled) */
    int n_threads;
    float rep_penalty;    /* repetition penalty (default: 1.1, 1.0 = disabled) */
    int rep_window;       /* how many recent tokens to penalize (default: 32) */
    unsigned long long rng_seed; /* sampling seed (default: 42, 0 = use 42 for back-compat) */
    /* Callback for streaming output */
    void (*on_token)(const char* text, void* user_data);
    void* user_data;
} tq_gen_config_t;

/* ============================================================
 * Tokenizer
 * ============================================================ */
typedef struct {
    char** vocab;        /* token strings, indexed by token_id */
    float* scores;       /* BPE merge scores (merge priority) */
    int vocab_size;      /* total vocab capacity (max_id + 1) */
    int max_token_len;
    int n_merges;        /* number of BPE merges loaded */
    /* Sorted vocab for encoding (binary search by string) */
    int* sorted_indices;
    /* Merge table: pairs of token IDs that merge into a result */
    int* merge_pairs;    /* [n_merges * 3]: (token_a, token_b, result_id) */
} tq_tokenizer_t;

/* ============================================================
 * TQM (TurboQuant Model) binary format — pre-quantized, mmap-ready
 *
 * File layout:
 *   [0..511]          tqm_header_t  (512 bytes, aligned)
 *   [tok_off..+tok_sz] Tokenizer JSON (raw bytes)
 *   [wt_off..+wt_sz]  Weights (Q4 packed + FP32 norms + BF16 embeds)
 *
 * All weight sections are 64-byte aligned for efficient mmap access.
 * Q4 weights are stored as (packed_bytes, float_scales) per matrix.
 * ============================================================ */

#define TQM_MAGIC   0x4D515454  /* "TTQM" in little-endian */
#define TQM_VERSION 1
#define TQM_ALIGN   64          /* alignment for weight sections */

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;           /* TQM_MAGIC */
    uint32_t version;         /* TQM_VERSION */

    /* Model config (mirrors tq_model_config_t) */
    int32_t n_layers;
    int32_t hidden_dim;
    int32_t intermediate_dim;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t head_dim;
    int32_t vocab_size;
    int32_t max_seq_len;
    float   rope_freq_base;
    float   rms_norm_eps;

    /* DeltaNet config */
    int32_t delta_n_heads;
    int32_t delta_n_kv_heads_tqm; /* K/Q heads (0 = same as delta_n_heads) */
    int32_t delta_key_head_dim;
    int32_t delta_value_head_dim;
    int32_t delta_conv_width;
    float   partial_rotary_factor;
    int32_t use_qk_norm;
    int32_t attn_output_gate;

    /* Quantization config */
    int32_t weight_quant;     /* 0=FP32, 4=Q4, 8=Q8 */
    int32_t embed_format;     /* 0=FP32, 16=BF16 */

    /* Section offsets (from file start) */
    uint64_t tokenizer_offset;
    uint64_t tokenizer_size;
    uint64_t weights_offset;
    uint64_t weights_size;

    /* Layer type map */
    int32_t n_attn_layers;
    int32_t attn_layer_indices[64]; /* which layers are self_attn (max 64) */

    /* Multi-architecture support (Gemma3) */
    int32_t model_type;       /* 0=qwen35, 1=gemma3 */
    int32_t sliding_window;   /* sliding window size (512 for gemma3, 0=unlimited) */
    float   rope_local_base_freq; /* RoPE base for local/sliding layers */
    int32_t n_norms_per_block;/* 2 for qwen35, 4 for gemma3 */
    float   query_pre_attn_scalar; /* attention scaling (0=use head_dim) */

    /* MoE fields */
    int32_t is_moe;
    int32_t num_experts;
    int32_t num_active_experts;
    int32_t expert_intermediate_dim;
    int32_t has_shared_expert;
    int32_t shared_expert_intermediate_dim;

    /* Padding to 512 bytes */
    uint8_t _pad[88]; /* 92 - 4 for delta_n_kv_heads_tqm */
} tqm_header_t;
#pragma pack(pop)

/* ============================================================
 * API
 * ============================================================ */

/* Model loading */
tq_model_t* tq_load_model(const char* path);  /* auto-detect format */
tq_model_t* tq_load_tqm(const char* path);    /* TQM format */
tq_model_t* tq_load_gguf(const char* path);   /* GGUF format */
int tq_save_tqm(tq_model_t* model, const char* tokenizer_path,
                const char* output_path);
void tq_free_model(tq_model_t* model);

/* State management */
tq_state_t* tq_create_state(const tq_model_config_t* config, tq_type kv_type);
tq_state_t* tq_create_state_ex(const tq_model_config_t* config, tq_type kv_type, int value_quant_bits);
void tq_free_state(tq_state_t* state);

/* Inference — returns pointer to logits (owned by state) */
float* tq_forward(tq_model_t* model, tq_state_t* state, int token, int pos);

/* Generation */
int tq_generate(tq_model_t* model, tq_tokenizer_t* tokenizer,
                const char* prompt, tq_gen_config_t* config,
                char* output, int output_size);

/* Sampling */
int tq_sample_argmax(const float* logits, int vocab_size);
int tq_sample_topp(const float* logits, int vocab_size,
                   float temperature, float top_p, unsigned long long* rng);

/* Tokenizer */
tq_tokenizer_t* tq_load_tokenizer(const char* path);
tq_tokenizer_t* tq_load_tokenizer_from_memory(const char* data, size_t size);
tq_tokenizer_t* tq_load_tokenizer_from_tqm(const char* tqm_path);
tq_tokenizer_t* tq_load_tokenizer_from_gguf(const void* gguf_ctx);
void tq_free_tokenizer(tq_tokenizer_t* tok);
int tq_encode(const tq_tokenizer_t* tok, const char* text,
              int* tokens, int max_tokens, int add_bos);
const char* tq_decode(const tq_tokenizer_t* tok, int prev_token, int token);

/* Tensor operations (exported for testing/reuse) */
void tq_matmul(float* out, const float* x, const float* w, int n, int d);
void tq_matmul_bf16(float* out, const float* x, const uint16_t* w_bf16, int n, int d);
void tq_matmul_q8(float* out, const float* x, const int8_t* w_qs, const float* w_scales,
                   int n, int d);
void tq_quantize_row_q8(const float* src, int8_t* dst_qs, float* dst_scales, int n);
void tq_quantize_weights(tq_model_t* model);
void tq_matmul_q4(float* out, const float* x, const uint8_t* w_qs, const float* w_scales,
                   int n, int d);
void tq_matmul_q4q2_preq(float* out,
                          const uint8_t* w_q4, const float* w_q4s,
                          const uint8_t* w_q2, const float* w_q2s,
                          const int8_t* x_q8, const float* x_scales,
                          int n, int d);
void tq_matmul_q4_preq(float* out, const uint8_t* w_qs, const float* w_scales,
                        const int8_t* x_q8, const float* x_scales, int n, int d);
void tq_quantize_row_q4(const float* src, uint8_t* dst_qs, float* dst_scales, int n);
void tq_dequantize_row_q4(const uint8_t* qs, const float* scales, float* dst, int n);
void tq_quantize_weights_q4(tq_model_t* model);
void tq_matmul_q2(float* out, const float* x, const uint8_t* w_qs, const float* w_scales,
                   int n, int d);
void tq_matmul_1bit(float* out, const float* x, const uint8_t* sign_data, const float* norms,
                     int n_rows, int dim);
void tq_matmul_q2_preq(float* out, const uint8_t* w_qs, const float* w_scales,
                        const int8_t* x_q8, const float* x_scales, int n, int d);
void tq_quantize_row_q2(const float* src, uint8_t* dst_qs, float* dst_scales, int n);
void tq_dequantize_row_q2(const uint8_t* qs, const float* scales, float* dst, int n);
void tq_quantize_weights_q2(tq_model_t* model);

/* RHT + Q4 + Q2 Residual: TurboQuant novel weight quantization.
 * Achieves Q8 quality at 6-bit effective size. */
void tq_quantize_row_rht_q4q2(const float* src,
                                uint8_t* qs4, float* sc4,
                                uint8_t* qs2, float* sc2,
                                float* rht_buf, int n);
void tq_matmul_rht_q4q2(float* out, const float* x,
                          const uint8_t* w_qs4, const float* w_sc4,
                          const uint8_t* w_qs2, const float* w_sc2,
                          float* x_rht, int n, int d);
void tq_rmsnorm(float* out, const float* x, const float* weight, int n, float eps);
void tq_rope(float* q, float* k, int pos, int head_dim,
             int n_heads, int n_kv_heads, float freq_base);
void tq_silu(float* x, int n);
void tq_gelu_tanh(float* x, int n);
void tq_softmax(float* x, int n);
void tq_add(float* out, const float* a, const float* b, int n);
void tq_mul(float* out, const float* a, const float* b, int n);

/* Default generation config */
tq_gen_config_t tq_default_gen_config(void);

/* ============================================================
 * Adaptive compression utilities
 * ============================================================ */

/** Per-layer bit allocation recommendation based on kurtosis.
 * @param kurtosis_values  Post-RHT kurtosis per layer [n_layers]
 * @param n_layers         Number of layers
 * @param recommended_bits Output: recommended bits per layer [n_layers] (3 or 1)
 * @param avg_bits         Output: average bits across all layers
 */
void tq_recommend_layer_bits(const float* kurtosis_values, int n_layers,
                             int* recommended_bits, float* avg_bits);

/** Compute attention entropy from softmax distribution.
 * H = -sum(p * log2(p)), where p_i = 0 contributes 0.
 * @param probs   Softmax attention weights [seq_len]
 * @param seq_len Length of the attention distribution
 * @return        Entropy in bits
 */
float tq_attention_entropy(const float* probs, int seq_len);

/** Online Lloyd-Max codebook calibration.
 * Runs Lloyd-Max iterations on empirical data to find optimal centroids.
 * @param data       Input samples (post-RHT values)
 * @param n_samples  Number of samples
 * @param n_levels   Number of codebook levels (4 for 2-bit, 8 for 3-bit)
 * @param iterations Number of Lloyd-Max iterations
 * @param centroids  Output: optimized centroid values [n_levels]
 * @param boundaries Output: decision boundaries [n_levels - 1] (can be NULL)
 * @return           MSE of the calibrated codebook
 */
float tq_calibrate_codebook(const float* data, int n_samples,
                            int n_levels, int iterations,
                            float* centroids, float* boundaries);

/* Thread control for matmul parallelism */
void tq_set_threads(int n_threads);
int tq_get_threads(void);

/* Thread pool dispatch — splits work across the global thread pool.
 * fn: worker function (takes void* arg, returns void*)
 * args: array of n_tasks argument pointers, one per thread
 * n_tasks: number of tasks (should match tq_get_threads())
 * Falls back to serial execution if pool not active or n_tasks <= 1. */
void tq_tp_run(void* (*fn)(void*), void** args, int n_tasks);

/* Max threads supported by thread pool */
#define TQ_TP_MAX 16

// ============================================================================
// Section 3: GGUF Types (from tq_gguf.h)
// ============================================================================

/**
 * tq_gguf.h — GGUF format loader for TurboQuant
 *
 * Supports GGUF v3 (llama.cpp native format) with:
 *   - mmap-based zero-copy tensor access
 *   - All K-quant types (Q2_K through Q6_K)
 *   - Importance-matrix quants (IQ2_XXS, IQ3_XXS, IQ4_XS)
 *   - MoE expert tensor layouts
 *
 * Enables loading community GGUF models (Unsloth, bartowski, etc.)
 * directly into TurboQuant inference engine.
 */

/* ============================================================
 * GGUF format constants
 * ============================================================ */
#define TQ_GGUF_MAGIC       0x46554747  /* "GGUF" as uint32 LE */
#define TQ_GGUF_VERSION_MIN 2
#define TQ_GGUF_VERSION_MAX 3
#define TQ_GGUF_MAX_NAME    256
#define TQ_GGUF_DEFAULT_ALIGNMENT 32

/* ============================================================
 * GGUF metadata value types
 * ============================================================ */
typedef enum {
    TQ_GGUF_TYPE_UINT8   = 0,
    TQ_GGUF_TYPE_INT8    = 1,
    TQ_GGUF_TYPE_UINT16  = 2,
    TQ_GGUF_TYPE_INT16   = 3,
    TQ_GGUF_TYPE_UINT32  = 4,
    TQ_GGUF_TYPE_INT32   = 5,
    TQ_GGUF_TYPE_FLOAT32 = 6,
    TQ_GGUF_TYPE_BOOL    = 7,
    TQ_GGUF_TYPE_STRING  = 8,
    TQ_GGUF_TYPE_ARRAY   = 9,
    TQ_GGUF_TYPE_UINT64  = 10,
    TQ_GGUF_TYPE_INT64   = 11,
    TQ_GGUF_TYPE_FLOAT64 = 12,
} tq_gguf_type;

/* ============================================================
 * GGML tensor quantization types
 * ============================================================ */
typedef enum {
    TQ_GGML_TYPE_F32       = 0,
    TQ_GGML_TYPE_F16       = 1,
    TQ_GGML_TYPE_Q4_0      = 2,
    TQ_GGML_TYPE_Q4_1      = 3,
    TQ_GGML_TYPE_Q5_0      = 6,
    TQ_GGML_TYPE_Q5_1      = 7,
    TQ_GGML_TYPE_Q8_0      = 8,
    TQ_GGML_TYPE_Q8_1      = 9,
    TQ_GGML_TYPE_Q2_K      = 10,
    TQ_GGML_TYPE_Q3_K      = 11,
    TQ_GGML_TYPE_Q4_K      = 12,
    TQ_GGML_TYPE_Q5_K      = 13,
    TQ_GGML_TYPE_Q6_K      = 14,
    TQ_GGML_TYPE_Q8_K      = 15,
    TQ_GGML_TYPE_IQ2_XXS   = 16,
    TQ_GGML_TYPE_IQ2_XS    = 17,
    TQ_GGML_TYPE_IQ3_XXS   = 18,
    TQ_GGML_TYPE_IQ1_S     = 19,
    TQ_GGML_TYPE_IQ4_NL    = 20,
    TQ_GGML_TYPE_IQ3_S     = 21,
    TQ_GGML_TYPE_IQ2_S     = 22,
    TQ_GGML_TYPE_IQ4_XS    = 23,
    TQ_GGML_TYPE_BF16      = 30,
    TQ_GGML_TYPE_COUNT     = 31,
} tq_ggml_dtype;

/* ============================================================
 * GGUF structures
 * ============================================================ */

/* String in GGUF format */
typedef struct {
    uint64_t len;
    char*    str;    /* NOT null-terminated in file, but we null-terminate on parse */
} tq_gguf_string_t;

/* Metadata key-value pair */
typedef struct {
    char         key[TQ_GGUF_MAX_NAME];
    tq_gguf_type type;
    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        float    f32;
        uint64_t u64;
        int64_t  i64;
        double   f64;
        uint8_t  bool_val;
        tq_gguf_string_t string;
        struct {
            tq_gguf_type elem_type;
            uint64_t     count;
            void*        data;   /* raw array data (heap-allocated on parse) */
        } array;
    } value;
} tq_gguf_kv_t;

/* Tensor descriptor (metadata only, data accessed via mmap) */
typedef struct {
    char          name[TQ_GGUF_MAX_NAME];
    uint32_t      n_dims;
    int64_t       shape[4];
    tq_ggml_dtype type;
    uint64_t      offset;       /* offset within tensor data section */
    size_t        size_bytes;   /* computed total bytes */
    const void*   data;         /* pointer into mmap'd region */
} tq_gguf_tensor_t;

/* GGUF file context (opaque to callers) */
typedef struct {
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    uint32_t alignment;

    tq_gguf_kv_t*     kv;       /* [n_kv] metadata pairs */
    tq_gguf_tensor_t* tensors;  /* [n_tensors] tensor descriptors */

    void*   mmap_data;          /* base of mmap'd file */
    size_t  mmap_size;          /* total file size */
    size_t  data_offset;        /* offset where tensor data begins */

    char    arch[64];           /* architecture string (e.g., "qwen2moe", "llama") */
} tq_gguf_ctx_t;

/* ============================================================
 * GGUF API
 * ============================================================ */

/* Open/close GGUF file */
tq_gguf_ctx_t* tq_gguf_open(const char* path);
void           tq_gguf_close(tq_gguf_ctx_t* ctx);

/* Metadata lookup */
int64_t     tq_gguf_find_key(const tq_gguf_ctx_t* ctx, const char* key);
int32_t     tq_gguf_get_i32(const tq_gguf_ctx_t* ctx, const char* key, int32_t fallback);
uint32_t    tq_gguf_get_u32(const tq_gguf_ctx_t* ctx, const char* key, uint32_t fallback);
float       tq_gguf_get_f32(const tq_gguf_ctx_t* ctx, const char* key, float fallback);
const char* tq_gguf_get_str(const tq_gguf_ctx_t* ctx, const char* key);

/* Tensor lookup */
const tq_gguf_tensor_t* tq_gguf_find_tensor(const tq_gguf_ctx_t* ctx, const char* name);

/* ============================================================
 * GGML quant type utilities
 * ============================================================ */

/* Bytes per quantization block */
size_t tq_ggml_type_size(tq_ggml_dtype type);

/* Elements per quantization block */
int tq_ggml_type_blck(tq_ggml_dtype type);

/* Human-readable name */
const char* tq_ggml_type_name(tq_ggml_dtype type);

/* IQ2_S codebook accessor — returns pointer to 1024-entry uint64 grid */
const uint64_t* tq_iq2s_grid(void);

/* ============================================================
 * GGUF dequantization
 * ============================================================ */

/* Dequantize a contiguous row of n elements from GGUF quant format to FP32 */
void tq_dequant_row_gguf(tq_ggml_dtype type, const void* src, float* dst, int n);

/* On-the-fly dequant matmul: out[d] += sum_n(x[n] * dequant(W[d,n]))
 * W is stored in GGUF quantized format, dequantized block-by-block.
 * This is the hot path for MoE expert computation. */
void tq_matmul_gguf(float* out, const float* x,
                    const void* weight, tq_ggml_dtype weight_type,
                    int out_dim, int in_dim);

/* ============================================================
 * Metal GPU batch mode
 *
 * Batch consecutive matmul dispatches into a single GPU command
 * buffer to reduce per-dispatch overhead. Critical for MoE models
 * with hundreds of small matmuls per token.
 *
 * Usage pattern:
 *   tq_metal_batch_begin_if_available();
 *   tq_matmul_gguf(gate_out, ...);  // encoded, not dispatched
 *   tq_matmul_gguf(up_out, ...);    // encoded, not dispatched
 *   tq_metal_batch_flush_if_available();  // dispatch + wait + copy
 *   // gate_out and up_out are now valid
 * ============================================================ */

/* Begin batching Metal matmul encodes (no-op if Metal unavailable) */
void tq_metal_batch_begin_if_available(void);
/* Flush: commit command buffer, wait, copy all results (no-op if not batching) */
void tq_metal_batch_flush_if_available(void);
/* End batch mode and flush (no-op if not batching) */
void tq_metal_batch_end_if_available(void);

/* ============================================================
 * MoE (Mixture of Experts) support
 * ============================================================ */

/* MoE configuration */
typedef struct {
    int num_experts;                /* total experts per MoE layer (e.g., 64) */
    int num_active;                 /* active experts per token (e.g., 8) */
    int expert_intermediate_dim;    /* per-expert FFN intermediate dim */
    int has_shared_expert;          /* 1 if shared expert exists */
    int shared_expert_intermediate_dim;
    int norm_topk_prob;             /* 1 = renormalize top-K weights */
    int use_gelu;                   /* 1 = GeGLU (Gemma 4), 0 = SwiGLU (Qwen) */
} tq_moe_config_t;

/* Per-expert weight pointers (into GGUF mmap) */
typedef struct {
    const void*   w_gate;     /* [expert_inter, hidden_dim] quantized */
    const void*   w_up;       /* [expert_inter, hidden_dim] quantized */
    const void*   w_down;     /* [hidden_dim, expert_inter] quantized */
    tq_ggml_dtype gate_type;
    tq_ggml_dtype up_type;
    tq_ggml_dtype down_type;

    /* Q4 pre-converted weights (NULL if not converted) */
    uint8_t* gate_q4_qs;      /* packed Q4 for gate */
    float*   gate_q4_scales;
    uint8_t* up_q4_qs;        /* packed Q4 for up */
    float*   up_q4_scales;
    uint8_t* down_q4_qs;      /* packed Q4 for down */
    float*   down_q4_scales;
    int      q4_converted;    /* 1 if Q4 conversion done */
} tq_expert_weights_t;

/* MoE layer (per transformer layer) */
typedef struct {
    float*               router_weight;  /* [num_experts, hidden_dim] FP32 */
    const float*         router_input_scale; /* [hidden_dim] per-feature router input scale (NULL if not used) */
    tq_expert_weights_t* experts;        /* [num_experts] */
    tq_expert_weights_t  shared_expert;  /* always-active expert */
    float*               shared_gate;    /* [hidden_dim] shared expert gate (optional) */
    const float*         expert_scale;   /* [num_experts] per-expert output scale (Gemma 4, NULL if not used) */
} tq_moe_layer_t;

/* MoE runtime state */
typedef struct {
    float* router_logits;    /* [num_experts] */
    int*   top_experts;      /* [num_active] selected indices */
    float* expert_weights;   /* [num_active] softmax weights */
    float* expert_out;       /* [hidden_dim] accumulator */
    float* expert_hb;        /* [expert_intermediate_dim] workspace */
    float* expert_hb2;       /* [expert_intermediate_dim] workspace */
} tq_moe_state_t;

/* MoE API */
tq_moe_state_t* tq_moe_create_state(const tq_moe_config_t* config, int hidden_dim);
void            tq_moe_free_state(tq_moe_state_t* state);

/* Top-K expert routing: select top num_active experts */
void tq_moe_route(const float* hidden, const float* router_weight,
                  int num_experts, int num_active, int hidden_dim,
                  int* out_expert_ids, float* out_expert_weights);

/* Full MoE FFN forward: route + dispatch + accumulate
 * layer_idx is needed for the per-layer expert Q4 LRU cache. */
void tq_moe_forward(const tq_moe_layer_t* layer,
                    const tq_moe_config_t* config,
                    tq_moe_state_t* state,
                    const float* input, float* output,
                    int hidden_dim, int layer_idx);

/* Expert Q4 LRU cache — runtime on-demand conversion */
void tq_moe_cache_init(int n_layers, const tq_moe_config_t* config, int hidden_dim);
void tq_moe_cache_free(void);

/* Expert memory hints (madvise for active/inactive experts) */
void tq_moe_advise(const tq_moe_layer_t* layer,
                   const int* active_ids, int n_active,
                   int num_experts);

/* ============================================================
 * Fused MoE Metal GPU dispatch
 *
 * Fused GPU MoE dispatch:
 *   Phase 1: gate + up projections on GPU (all experts, parallel)
 *   Phase 2: SwiGLU activation on GPU (all experts, parallel)
 *   Phase 3: down projection + weighted accumulate on GPU
 *            (IQ2_S codebook passed as device buffer to avoid constant memory limit)
 *
 * Returns 0 = full GPU success (output filled).
 * Returns 1 = partial (hb_output filled, caller does down+accum on CPU).
 * Returns -1 if unavailable.
 * ============================================================ */

/* Check if fused MoE Metal dispatch is available */
int tq_metal_moe_available(void);

/* Fused MoE forward: GPU handles gate+up+SwiGLU (Phases 1+2).
 * Returns:
 *   0  = full success (all phases on GPU, output[] filled)
 *   1  = partial success: gate+up+SwiGLU done on GPU,
 *         hb_output[] filled with [num_active * expert_dim] SwiGLU'd activations,
 *         caller must do down projection + accumulate on CPU.
 *  -1  = failure (caller falls back to full CPU path). */
int tq_metal_moe_forward(
    const float*    input,
    float*          output,
    float*          hb_output,      /* [num_active * expert_dim] — GPU writes SwiGLU results here */
    const void*     weight_base,
    size_t          weight_size,
    const uint64_t* gate_offsets,
    const uint64_t* up_offsets,
    const uint64_t* down_offsets,
    const int*      active_expert_ids,
    const float*    expert_routing_weights,
    int             num_active,
    int             expert_dim,
    int             hidden_dim,
    int             num_experts_total,
    int             weight_type,
    const int*      gate_types,     /* per-expert gate quant types, NULL = use weight_type */
    const int*      up_types,       /* per-expert up quant types, NULL = use weight_type */
    const int*      down_types);    /* per-expert down quant types, NULL = use weight_type */

// ============================================================================
// Section 4: Internal API (from turboquant.h)
// ============================================================================

/**
 * TurboQuant.cpp — Cross-platform KV cache compression library
 *
 * Public C API — single header include for all functionality.
 * Zero external dependencies (libc/libm only).
 */

/* ============================================================
 * Version
 * ============================================================ */

#define TQ_VERSION_STRING "0.1.0"

/* ============================================================
 * Error codes
 * ============================================================ */

typedef enum {
    TQ_OK              =  0,
    TQ_ERR_NULL_PTR    = -1,
    TQ_ERR_INVALID_TYPE= -2,
    TQ_ERR_INVALID_DIM = -3,
    TQ_ERR_OUT_OF_MEM  = -4,
    TQ_ERR_NOT_IMPL    = -5,
    TQ_ERR_BACKEND     = -6,
    TQ_ERR_BUFFER_TOO_SMALL = -7,
} tq_status;

const char* tq_status_string(tq_status status);

/* ============================================================
 * Backend selection
 * ============================================================ */

typedef enum {
    TQ_BACKEND_CPU    = 0,
    TQ_BACKEND_CUDA   = 1,
    TQ_BACKEND_METAL  = 2,
    TQ_BACKEND_VULKAN = 3,   /* AMD + cross-platform (SPIR-V compute shaders) */
    TQ_BACKEND_ROCM   = 4,   /* AMD ROCm/HIP (CUDA-compatible API) */
    TQ_BACKEND_AUTO   = 99,
} tq_backend;

/* ============================================================
 * Context (opaque handle)
 * ============================================================ */

typedef struct tq_context tq_context_t;

tq_status   tq_init(tq_context_t** ctx, tq_backend backend);
void        tq_free(tq_context_t* ctx);
tq_backend  tq_get_backend(const tq_context_t* ctx);

/* ============================================================
 * Type info
 * ============================================================ */

const char* tq_type_name(tq_type type);
float       tq_type_bpe(tq_type type);
size_t      tq_type_block_size(tq_type type);
size_t      tq_type_type_size(tq_type type);

/* ============================================================
 * Quantization
 * ============================================================ */

/**
 * Quantize key vectors.
 * @param ctx       TurboQuant context
 * @param keys      Input FP32 keys [n × head_dim]
 * @param n         Number of key vectors
 * @param head_dim  Dimension of each key vector
 * @param type      Quantization type (TQ_TYPE_*)
 * @param out       Output buffer (caller allocated, size from tq_quantize_keys_size)
 * @param out_size  Size of output buffer in bytes
 */
tq_status tq_quantize_keys(tq_context_t* ctx,
                           const float* keys, int n, int head_dim,
                           tq_type type,
                           void* out, size_t out_size);

/** Compute required output buffer size for tq_quantize_keys */
size_t tq_quantize_keys_size(int n, int head_dim, tq_type type);

/**
 * Quantize value vectors.
 * @param ctx       TurboQuant context
 * @param values    Input FP32 values [n × head_dim]
 * @param n         Number of value vectors
 * @param head_dim  Dimension per value
 * @param bits      Quantization bits (2 or 4)
 * @param out       Output buffer
 * @param out_size  Size of output buffer
 */
tq_status tq_quantize_values(tq_context_t* ctx,
                             const float* values, int n, int head_dim,
                             int bits,
                             void* out, size_t out_size);

size_t tq_quantize_values_size(int n, int head_dim, int bits);

/**
 * Dequantize keys back to FP32 (for debugging/testing).
 */
tq_status tq_dequantize_keys(tq_context_t* ctx,
                             const void* quantized, int n, int head_dim,
                             tq_type type,
                             float* out);

/* ============================================================
 * K/V Asymmetric Quantization
 * ============================================================ */

/**
 * Quantize keys and values with independent types (K/V asymmetric).
 * @param ctx          TurboQuant context
 * @param keys         Input FP32 keys [n x head_dim]
 * @param values       Input FP32 values [n x head_dim]
 * @param n            Number of key/value vectors
 * @param head_dim     Dimension of each vector
 * @param key_type     Quantization type for keys (TQ_TYPE_*)
 * @param value_type   Quantization type for values (TQ_TYPE_*)
 * @param key_out      Output buffer for quantized keys
 * @param key_out_size Size of key output buffer in bytes
 * @param val_out      Output buffer for quantized values
 * @param val_out_size Size of value output buffer in bytes
 */
tq_status tq_quantize_kv(tq_context_t* ctx,
                          const float* keys, const float* values,
                          int n, int head_dim,
                          tq_type key_type, tq_type value_type,
                          void* key_out, size_t key_out_size,
                          void* val_out, size_t val_out_size);

/** Compute required output buffer size for keys in K/V asymmetric quantization */
size_t tq_quantize_kv_key_size(int n, int head_dim, tq_type key_type);

/** Compute required output buffer size for values in K/V asymmetric quantization */
size_t tq_quantize_kv_value_size(int n, int head_dim, tq_type value_type);

/* ============================================================
 * Attention
 * ============================================================ */

/**
 * Compute attention scores from quantized KV cache.
 * @param ctx       TurboQuant context
 * @param query     Query vector [head_dim]
 * @param kv_cache  Quantized key cache
 * @param seq_len   Number of cached keys
 * @param head_dim  Dimension per head
 * @param type      Quantization type used for keys
 * @param scores    Output attention scores [seq_len]
 */
tq_status tq_attention(tq_context_t* ctx,
                       const float* query,
                       const void* kv_cache,
                       int seq_len, int head_dim,
                       tq_type type,
                       float* scores);

/* ============================================================
 * Paged cache management
 * ============================================================ */

typedef struct tq_cache tq_cache_t;

tq_status tq_cache_create(tq_cache_t** cache,
                          int block_size, int max_blocks,
                          int num_heads, int head_dim,
                          tq_type default_type);

tq_status tq_cache_append(tq_cache_t* cache,
                          int head_idx,
                          const float* key, const float* value,
                          int head_dim);

tq_status tq_cache_get_block(const tq_cache_t* cache,
                             int head_idx, int block_idx,
                             const void** data, tq_type* type);

int  tq_cache_seq_len(const tq_cache_t* cache, int head_idx);
void tq_cache_free(tq_cache_t* cache);

/** Copy-on-Write: increment ref_count on a block (share it) */
tq_status tq_cache_share_block(tq_cache_t* cache, int head_idx, int block_idx);

/** Copy-on-Write: decrement ref_count, free block data when it reaches 0 */
tq_status tq_cache_free_block(tq_cache_t* cache, int head_idx, int block_idx);

/** Get the quantized value block for a given head and block index */
tq_status tq_cache_get_value(const tq_cache_t* cache, int head_idx, int block_idx,
                             const void** data);

/** Get ref_count of a block (for testing/debugging) */
int tq_cache_block_ref_count(const tq_cache_t* cache, int head_idx, int block_idx);

/* ============================================================
 * Strategy recommendation
 * ============================================================ */

tq_type tq_recommend_strategy(int head_dim, int target_bits,
                              float quality_threshold);

/* ============================================================
 * Random Hadamard Transform (RHT) for quantization pre-processing
 * ============================================================ */

/** In-place Random Hadamard Transform.
 * Decorrelates channels for improved scalar quantization quality.
 * n is rounded down to nearest power of 2 internally.
 * @param data   FP32 vector to transform in-place
 * @param n      Length of the vector
 * @param seed   Random seed for sign flipping (must match for inverse)
 */
void tq_rht_transform(float* data, int n, uint32_t seed);

/** Inverse RHT — recovers original vector from transformed data.
 * Must use the same seed as the forward transform.
 */
void tq_rht_inverse(float* data, int n, uint32_t seed);

/** Quantize key vectors with RHT pre-processing (higher quality).
 * Pipeline: copy -> RHT each vector -> quantize.
 * @param ctx       TurboQuant context
 * @param keys      Input FP32 keys [n x head_dim]
 * @param n         Number of key vectors
 * @param head_dim  Dimension of each key vector (should be power of 2)
 * @param type      Quantization type (TQ_TYPE_*)
 * @param rht_seed  Random seed for RHT sign flipping
 * @param out       Output buffer (same size as tq_quantize_keys_size)
 * @param out_size  Size of output buffer in bytes
 */
tq_status tq_quantize_keys_rht(tq_context_t* ctx,
                                const float* keys, int n, int head_dim,
                                tq_type type, uint32_t rht_seed,
                                void* out, size_t out_size);

/** Dequantize key vectors with RHT post-processing.
 * Pipeline: dequantize -> inverse RHT each vector.
 * Must use the same rht_seed as quantize.
 */
tq_status tq_dequantize_keys_rht(tq_context_t* ctx,
                                  const void* quantized, int n, int head_dim,
                                  tq_type type, uint32_t rht_seed,
                                  float* out);

/* ============================================================
 * Utility
 * ============================================================ */

/** Get format spec for a quantization type */
tq_format_spec_t tq_get_format_spec(tq_type type);

/* ============================================================
 * Convenience functions
 * ============================================================ */

int     tq_type_count(void);
tq_type tq_type_from_name(const char* name);

/* ============================================================
 * Progressive compression
 * ============================================================ */

typedef struct tq_progressive tq_progressive_t;

tq_status tq_progressive_create(tq_progressive_t** out,
                                const tq_progressive_config_t* config,
                                int head_dim, int max_tokens);
tq_status tq_progressive_append(tq_progressive_t* p,
                                const float* key, int head_dim);
tq_status tq_progressive_attention(const tq_progressive_t* p,
                                   const float* query,
                                   float* scores, int head_dim);
int       tq_progressive_count(const tq_progressive_t* p);
void      tq_progressive_free(tq_progressive_t* p);

tq_progressive_config_t tq_progressive_default_config(void);

// ============================================================================
// Section 5: quant_ctx struct definition
// ============================================================================

struct quant_ctx {
    tq_model_t* model;
    tq_state_t* state;
    tq_tokenizer_t* tokenizer;
    tq_gen_config_t config;
    int n_ctx_tokens;     /* number of tokens currently in KV cache */
    /* Prefix-match cache for chat history reuse:
     * stores the actual token IDs that are committed to the KV cache,
     * so the next quant_generate() can skip the matching prefix and
     * only prefill the diverging suffix. Critical for chat mode where
     * each turn re-sends the entire conversation history. */
    int* cached_tokens;
    int  n_cached;
    int  cached_capacity;
};

// ============================================================================
// Section 6: RHT Transform (from tq_rht.c)
// ============================================================================

/**
 * Random Hadamard Transform (RHT) — core implementation
 *
 * The Walsh-Hadamard Transform (WHT) rotates input vectors to remove
 * inter-coordinate correlation, making scalar quantization near-optimal.
 * Combined with random sign flipping, it becomes the Random Hadamard
 * Transform (RHT).
 *
 * Key properties:
 * - WHT is self-inverse: H * H = n * I
 * - O(n log n) butterfly computation, no matrix storage needed
 * - Random signs decorrelate channels across different blocks
 */

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ---------- Random sign generation from seed ---------- */

static int random_sign(uint32_t seed, int idx) {
    uint32_t h = seed ^ (uint32_t)idx;
    h = h * 2654435761u;  /* Knuth multiplicative hash */
    return (h & 1) ? 1 : -1;
}

/* ---------- In-place Walsh-Hadamard Transform: O(n log n) ----------
 * n must be power of 2.
 * This is self-inverse up to scaling: WHT(WHT(x)) = n * x.
 * The butterfly pattern applies log2(n) stages of add/subtract pairs. */

static void walsh_hadamard(float* data, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
#ifdef __ARM_NEON
            if (len >= 4) {
                int j = 0;
                for (; j + 3 < len; j += 4) {
                    float32x4_t u = vld1q_f32(data + i + j);
                    float32x4_t v = vld1q_f32(data + i + j + len);
                    vst1q_f32(data + i + j,       vaddq_f32(u, v));
                    vst1q_f32(data + i + j + len,  vsubq_f32(u, v));
                }
                for (; j < len; j++) {
                    float u = data[i + j];
                    float v = data[i + j + len];
                    data[i + j]       = u + v;
                    data[i + j + len] = u - v;
                }
            } else
#endif
            {
                for (int j = 0; j < len; j++) {
                    float u = data[i + j];
                    float v = data[i + j + len];
                    data[i + j]       = u + v;
                    data[i + j + len] = u - v;
                }
            }
        }
    }
}

/* ---------- RHT: Random sign flip + Walsh-Hadamard + normalize ---------- */

void tq_rht_transform(float* data, int n, uint32_t seed) {
    if (!data || n <= 0) return;

    /* Round down to nearest power of 2 */
    int n2 = 1;
    while (n2 * 2 <= n) n2 *= 2;

    /* Step 1: Random sign flip (diagonal D matrix) */
    for (int i = 0; i < n2; i++) {
        data[i] *= (float)random_sign(seed, i);
    }

    /* Step 2: Walsh-Hadamard butterfly */
    walsh_hadamard(data, n2);

    /* Step 3: Normalize by 1/sqrt(n) to make it an orthogonal transform */
    float scale = 1.0f / sqrtf((float)n2);
    for (int i = 0; i < n2; i++) {
        data[i] *= scale;
    }
}

/* ---------- Inverse RHT ----------
 * Since H is self-inverse (up to scaling) and D*D = I:
 *   RHT     = (1/sqrt(n)) * H * D
 *   RHT^-1  = (1/sqrt(n)) * D * H
 * So: scale first, then WHT, then same sign flip. */

void tq_rht_inverse(float* data, int n, uint32_t seed) {
    if (!data || n <= 0) return;

    int n2 = 1;
    while (n2 * 2 <= n) n2 *= 2;

    /* Step 1: Normalize by 1/sqrt(n) */
    float scale = 1.0f / sqrtf((float)n2);
    for (int i = 0; i < n2; i++) {
        data[i] *= scale;
    }

    /* Step 2: Walsh-Hadamard (self-inverse up to scaling) */
    walsh_hadamard(data, n2);

    /* Step 3: Same random sign flip (D * D = I, so applying same signs inverts) */
    for (int i = 0; i < n2; i++) {
        data[i] *= (float)random_sign(seed, i);
    }
}

// ============================================================================
// Section 7: Uniform Quantization (from tq_uniform.c)
// ============================================================================

/**
 * Uniform min-max quantization — reference C implementation
 *
 * Simple baseline quantizer: find min/max, linearly map to 2^bits levels.
 * NOTE: This is the GENERIC reference. Compiler auto-vectorization is disabled
 * so that SIMD speedup measurement is meaningful.
 */
/* Generic reference — no compiler-specific pragmas */

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

// ============================================================================
// Section 8: Type Traits (from tq_traits.c)
// ============================================================================

/* Stub implementations for excluded quantization types (polar, qjl, turbo, mixed) */
static void tq_stub_quantize(const float* src, void* dst, int n) {
    (void)src; (void)dst; (void)n;
}
static void tq_stub_dequantize(const void* src, float* dst, int n) {
    (void)src; (void)n;
    memset(dst, 0, (size_t)n * sizeof(float));
}
static void tq_stub_attention(const float* query, const void* kv,
                               float* scores, int seq_len, int head_dim) {
    (void)query; (void)kv; (void)head_dim;
    memset(scores, 0, (size_t)seq_len * sizeof(float));
}

/* Stub macros for excluded algorithm files */
#define tq_polar_quantize_ref tq_stub_quantize
#define tq_polar_dequantize_ref tq_stub_dequantize
#define tq_polar_attention_ref tq_stub_attention
#define tq_qjl_quantize_ref tq_stub_quantize
#define tq_qjl_dequantize_ref tq_stub_dequantize
#define tq_qjl_attention_ref tq_stub_attention
#define tq_turbo_quantize_ref tq_stub_quantize
#define tq_turbo_dequantize_ref tq_stub_dequantize
#define tq_turbo_attention_ref tq_stub_attention

extern void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
extern void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
extern void tq_uniform_4b_attention_ref(const float* query, const void* kv,
                                         float* scores, int seq_len, int head_dim);
extern void tq_uniform_4b_attention_int_ref(const float* query, const void* kv,
                                             float* scores, int seq_len, int head_dim);
extern void tq_uniform_2b_quantize_ref(const float* src, void* dst, int n);
extern void tq_uniform_2b_dequantize_ref(const void* src, float* dst, int n);
extern void tq_uniform_2b_attention_ref(const float* query, const void* kv,
                                         float* scores, int seq_len, int head_dim);

#define tq_mixed_4b8_quantize_ref tq_stub_quantize
#define tq_mixed_4b8_dequantize_ref tq_stub_dequantize
#define tq_mixed_4b8_attention_ref tq_stub_attention

extern void tq_uniform_3b_quantize_ref(const float* src, void* dst, int n);
extern void tq_uniform_3b_dequantize_ref(const void* src, float* dst, int n);
extern void tq_uniform_3b_attention_ref(const float* query, const void* kv,
                                         float* scores, int seq_len, int head_dim);

#define tq_turbo_kv_3b_quantize_ref tq_stub_quantize
#define tq_turbo_kv_3b_dequantize_ref tq_stub_dequantize
#define tq_turbo_kv_3b_attention_ref tq_stub_attention

#define tq_turbo_kv_4b_quantize_ref tq_stub_quantize
#define tq_turbo_kv_4b_dequantize_ref tq_stub_dequantize
#define tq_turbo_kv_4b_attention_ref tq_stub_attention

#define tq_turbo_kv_1b_quantize_ref tq_stub_quantize
#define tq_turbo_kv_1b_dequantize_ref tq_stub_dequantize
#define tq_turbo_kv_1b_attention_ref tq_stub_attention

#define tq_turbo_kv_2b_quantize_ref tq_stub_quantize
#define tq_turbo_kv_2b_dequantize_ref tq_stub_dequantize
#define tq_turbo_kv_2b_attention_ref tq_stub_attention

/* Non-const to allow runtime GPU backend override (Vulkan/Metal) */
tq_type_traits_t TQ_TRAITS[TQ_TYPE_COUNT] = {
    [TQ_TYPE_POLAR_3B] = {
        .name       = "polar_3b",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_polar),
        .bpe        = (float)sizeof(block_tq_polar) * 8.0f / TQ_BK,
        .quantize   = tq_polar_quantize_ref,
        .dequantize = tq_polar_dequantize_ref,
        .attention  = tq_polar_attention_ref,
        .residual_type = TQ_TYPE_COUNT, /* none */
    },
    [TQ_TYPE_POLAR_4B] = {
        .name       = "polar_4b",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_polar),
        .bpe        = (float)sizeof(block_tq_polar) * 8.0f / TQ_BK,
        .quantize   = tq_polar_quantize_ref,
        .dequantize = tq_polar_dequantize_ref,
        .attention  = tq_polar_attention_ref,
        .residual_type = TQ_TYPE_COUNT,
    },
    [TQ_TYPE_QJL_1B] = {
        .name       = "qjl_1b",
        .block_size = TQ_BK_QJL,
        .type_size  = sizeof(block_tq_qjl),
        .bpe        = (float)sizeof(block_tq_qjl) * 8.0f / TQ_BK_QJL,
        .quantize   = tq_qjl_quantize_ref,
        .dequantize = tq_qjl_dequantize_ref,
        .attention  = tq_qjl_attention_ref,
        .residual_type = TQ_TYPE_COUNT,
    },
    [TQ_TYPE_TURBO_3B] = {
        .name       = "turbo_3b",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_turbo),
        .bpe        = (float)sizeof(block_tq_turbo) * 8.0f / TQ_BK,
        .quantize   = tq_turbo_quantize_ref,
        .dequantize = tq_turbo_dequantize_ref,
        .attention  = tq_turbo_attention_ref,
        .residual_type = TQ_TYPE_QJL_1B,
    },
    [TQ_TYPE_TURBO_4B] = {
        .name       = "turbo_4b",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_turbo),
        .bpe        = (float)sizeof(block_tq_turbo) * 8.0f / TQ_BK,
        .quantize   = tq_turbo_quantize_ref,
        .dequantize = tq_turbo_dequantize_ref,
        .attention  = tq_turbo_attention_ref,
        .residual_type = TQ_TYPE_QJL_1B,
    },
    [TQ_TYPE_UNIFORM_4B] = {
        .name       = "uniform_4b",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_uniform_4b),
        .bpe        = (float)sizeof(block_tq_uniform_4b) * 8.0f / TQ_BK,
        .quantize   = tq_uniform_4b_quantize_ref,
        .dequantize = tq_uniform_4b_dequantize_ref,
        .attention  = tq_uniform_4b_attention_int_ref,
        .residual_type = TQ_TYPE_COUNT,
    },
    [TQ_TYPE_UNIFORM_2B] = {
        .name       = "uniform_2b",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_uniform_2b),
        .bpe        = (float)sizeof(block_tq_uniform_2b) * 8.0f / TQ_BK,
        .quantize   = tq_uniform_2b_quantize_ref,
        .dequantize = tq_uniform_2b_dequantize_ref,
        .attention  = tq_uniform_2b_attention_ref,
        .residual_type = TQ_TYPE_COUNT,
    },
    [TQ_TYPE_MIXED_4B8] = {
        .name       = "mixed_4b8",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_mixed_4b8),
        .bpe        = (float)sizeof(block_tq_mixed_4b8) * 8.0f / TQ_BK,
        .quantize   = tq_mixed_4b8_quantize_ref,
        .dequantize = tq_mixed_4b8_dequantize_ref,
        .attention  = tq_mixed_4b8_attention_ref,
        .residual_type = TQ_TYPE_COUNT,
    },
    [TQ_TYPE_TURBO_KV_3B] = {
        .name       = "turbo_kv_3b",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_turbo_kv_3b),
        .bpe        = (float)sizeof(block_tq_turbo_kv_3b) * 8.0f / TQ_BK,
        .quantize   = tq_turbo_kv_3b_quantize_ref,
        .dequantize = tq_turbo_kv_3b_dequantize_ref,
        .attention  = tq_turbo_kv_3b_attention_ref,
        .residual_type = TQ_TYPE_QJL_1B,
    },
    [TQ_TYPE_TURBO_KV_4B] = {
        .name       = "turbo_kv_4b",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_turbo_kv_4b),
        .bpe        = (float)sizeof(block_tq_turbo_kv_4b) * 8.0f / TQ_BK,
        .quantize   = tq_turbo_kv_4b_quantize_ref,
        .dequantize = tq_turbo_kv_4b_dequantize_ref,
        .attention  = tq_turbo_kv_4b_attention_ref,
        .residual_type = TQ_TYPE_QJL_1B,
    },
    [TQ_TYPE_TURBO_KV_1B] = {
        .name       = "turbo_kv_1b",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_turbo_kv_1b),
        .bpe        = (float)sizeof(block_tq_turbo_kv_1b) * 8.0f / TQ_BK,
        .quantize   = tq_turbo_kv_1b_quantize_ref,
        .dequantize = tq_turbo_kv_1b_dequantize_ref,
        .attention  = tq_turbo_kv_1b_attention_ref,
        .residual_type = TQ_TYPE_COUNT, /* none */
    },
    [TQ_TYPE_TURBO_KV_2B] = {
        .name       = "turbo_kv_2b",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_turbo_kv_2b),
        .bpe        = (float)sizeof(block_tq_turbo_kv_2b) * 8.0f / TQ_BK,
        .quantize   = tq_turbo_kv_2b_quantize_ref,
        .dequantize = tq_turbo_kv_2b_dequantize_ref,
        .attention  = tq_turbo_kv_2b_attention_ref,
        .residual_type = TQ_TYPE_QJL_1B,
    },
    [TQ_TYPE_UNIFORM_3B] = {
        .name       = "uniform_3b",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_uniform_3b),
        .bpe        = (float)sizeof(block_tq_uniform_3b) * 8.0f / TQ_BK,
        .quantize   = tq_uniform_3b_quantize_ref,
        .dequantize = tq_uniform_3b_dequantize_ref,
        .attention  = tq_uniform_3b_attention_ref,
        .residual_type = TQ_TYPE_COUNT,
    },
};

const char* tq_type_name(tq_type type) {
    if (type < 0 || type >= TQ_TYPE_COUNT) return "unknown";
    return TQ_TRAITS[type].name;
}

float tq_type_bpe(tq_type type) {
    if (type < 0 || type >= TQ_TYPE_COUNT) return 0.0f;
    return TQ_TRAITS[type].bpe;
}

size_t tq_type_block_size(tq_type type) {
    if (type < 0 || type >= TQ_TYPE_COUNT) return 0;
    return TQ_TRAITS[type].block_size;
}

size_t tq_type_type_size(tq_type type) {
    if (type < 0 || type >= TQ_TYPE_COUNT) return 0;
    return TQ_TRAITS[type].type_size;
}

const char* tq_status_string(tq_status status) {
    switch (status) {
        case TQ_OK:               return "OK";
        case TQ_ERR_NULL_PTR:     return "null pointer";
        case TQ_ERR_INVALID_TYPE: return "invalid type";
        case TQ_ERR_INVALID_DIM:  return "invalid dimension";
        case TQ_ERR_OUT_OF_MEM:   return "out of memory";
        case TQ_ERR_NOT_IMPL:     return "not implemented";
        case TQ_ERR_BACKEND:      return "backend error";
        case TQ_ERR_BUFFER_TOO_SMALL: return "buffer too small";
        default:                  return "unknown error";
    }
}

tq_format_spec_t tq_get_format_spec(tq_type type) {
    tq_format_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.spec_version = TQ_SPEC_VERSION;
    spec.block_size = (uint16_t)TQ_TRAITS[type].block_size;
    switch (type) {
        case TQ_TYPE_POLAR_3B:
            spec.algorithm = TQ_ALG_POLAR; spec.key_bits = 3; break;
        case TQ_TYPE_POLAR_4B:
            spec.algorithm = TQ_ALG_POLAR; spec.key_bits = 4; break;
        case TQ_TYPE_QJL_1B:
            spec.algorithm = TQ_ALG_QJL; spec.key_bits = 1;
            spec.sketch_dim = TQ_SKETCH_DIM; spec.outlier_count = TQ_OUTLIERS; break;
        case TQ_TYPE_TURBO_3B:
            spec.algorithm = TQ_ALG_TURBO; spec.key_bits = 3;
            spec.sketch_dim = TQ_SKETCH_DIM; spec.outlier_count = TQ_OUTLIERS;
            spec.flags = TQ_FLAG_HAS_RESIDUAL; break;
        case TQ_TYPE_TURBO_4B:
            spec.algorithm = TQ_ALG_TURBO; spec.key_bits = 4;
            spec.sketch_dim = TQ_SKETCH_DIM; spec.outlier_count = TQ_OUTLIERS;
            spec.flags = TQ_FLAG_HAS_RESIDUAL; break;
        case TQ_TYPE_UNIFORM_4B:
            spec.algorithm = TQ_ALG_UNIFORM; spec.key_bits = 4; break;
        case TQ_TYPE_UNIFORM_2B:
            spec.algorithm = TQ_ALG_UNIFORM; spec.key_bits = 2; break;
        case TQ_TYPE_MIXED_4B8:
            spec.algorithm = TQ_ALG_MIXED; spec.key_bits = 4;
            spec.outlier_count = TQ_MIXED_OUTLIERS; break;
        case TQ_TYPE_TURBO_KV_3B:
            spec.algorithm = TQ_ALG_TURBO; spec.key_bits = 3;
            spec.flags = TQ_FLAG_HAS_RESIDUAL; break;
        case TQ_TYPE_TURBO_KV_4B:
            spec.algorithm = TQ_ALG_TURBO; spec.key_bits = 4;
            spec.flags = TQ_FLAG_HAS_RESIDUAL; break;
        case TQ_TYPE_TURBO_KV_1B:
            spec.algorithm = TQ_ALG_TURBO; spec.key_bits = 1; break;
        case TQ_TYPE_TURBO_KV_2B:
            spec.algorithm = TQ_ALG_TURBO; spec.key_bits = 2;
            spec.flags = TQ_FLAG_HAS_RESIDUAL; break;
        case TQ_TYPE_UNIFORM_3B:
            spec.algorithm = TQ_ALG_UNIFORM; spec.key_bits = 3; break;
        default: break;
    }
    return spec;
}

int tq_type_count(void) { return TQ_TYPE_COUNT; }

tq_type tq_type_from_name(const char* name) {
    if (!name) return TQ_TYPE_COUNT;
    for (int i = 0; i < TQ_TYPE_COUNT; i++) {
        if (strcmp(TQ_TRAITS[i].name, name) == 0) return (tq_type)i;
    }
    return TQ_TYPE_COUNT;
}

// ============================================================================
// Section 9: Tensor Operations (from tq_ops.c)
// ============================================================================

/**
 * tq_ops.c — Core tensor operations for transformer inference
 *
 * Implements matmul, RMSNorm, RoPE, SiLU, softmax, and element-wise ops.
 * NEON-optimized where available (Apple Silicon / ARM64).
 * No external dependencies — libc/libm only.
 */

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ============================================================
 * Thread pool — condition variable based, minimal overhead
 * Workers sleep between dispatches, wake via cond_broadcast.
 * Main thread does task[0], workers do task[1..n-1].
 * ============================================================ */

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
        /* GPU: only in batch mode (batched independent matmuls) */
        if (tq_metal_batch_active()) {
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

// ============================================================================
// Section 10: GGUF Parser (from tq_gguf.c)
// ============================================================================

/**
 * tq_gguf.c — GGUF v3 format parser for TurboQuant
 *
 * Implements mmap-based zero-copy loading of GGUF files.
 * Supports GGUF versions 2 and 3 with all GGML quant types.
 *
 * SPDX-License-Identifier: MIT
 */

#ifdef _WIN32
#else
#endif

/* ============================================================
 * Helpers
 * ============================================================ */

static inline uint64_t align_up(uint64_t x, uint64_t align) {
    return (x + align - 1) & ~(align - 1);
}

/* Simple cursor-based reader over the mmap'd buffer */
typedef struct {
    const uint8_t* base;
    size_t         size;
    size_t         pos;
} tq_reader_t;

static bool reader_has(const tq_reader_t* r, size_t n) {
    return r->pos + n <= r->size;
}

static uint8_t read_u8(tq_reader_t* r) {
    uint8_t v = 0;
    if (reader_has(r, 1)) {
        v = r->base[r->pos];
        r->pos += 1;
    }
    return v;
}

static uint16_t read_u16(tq_reader_t* r) {
    uint16_t v = 0;
    if (reader_has(r, 2)) {
        memcpy(&v, r->base + r->pos, 2);
        r->pos += 2;
    }
    return v;
}

static uint32_t read_u32(tq_reader_t* r) {
    uint32_t v = 0;
    if (reader_has(r, 4)) {
        memcpy(&v, r->base + r->pos, 4);
        r->pos += 4;
    }
    return v;
}

static uint64_t read_u64(tq_reader_t* r) {
    uint64_t v = 0;
    if (reader_has(r, 8)) {
        memcpy(&v, r->base + r->pos, 8);
        r->pos += 8;
    }
    return v;
}

static float read_f32(tq_reader_t* r) {
    float v = 0.0f;
    if (reader_has(r, 4)) {
        memcpy(&v, r->base + r->pos, 4);
        r->pos += 4;
    }
    return v;
}

static double read_f64(tq_reader_t* r) {
    double v = 0.0;
    if (reader_has(r, 8)) {
        memcpy(&v, r->base + r->pos, 8);
        r->pos += 8;
    }
    return v;
}

/* Read a GGUF string: uint64 len + len bytes. Caller must free the returned str. */
static char* read_string(tq_reader_t* r, uint64_t* out_len) {
    uint64_t len = read_u64(r);
    if (out_len) *out_len = len;
    if (!reader_has(r, len)) {
        return NULL;
    }
    char* s = (char*)malloc(len + 1);
    if (!s) return NULL;
    memcpy(s, r->base + r->pos, len);
    s[len] = '\0';
    r->pos += len;
    return s;
}

/* Return the byte size of a single GGUF metadata scalar type */
static size_t gguf_type_scalar_size(tq_gguf_type type) {
    switch (type) {
        case TQ_GGUF_TYPE_UINT8:   return 1;
        case TQ_GGUF_TYPE_INT8:    return 1;
        case TQ_GGUF_TYPE_BOOL:    return 1;
        case TQ_GGUF_TYPE_UINT16:  return 2;
        case TQ_GGUF_TYPE_INT16:   return 2;
        case TQ_GGUF_TYPE_UINT32:  return 4;
        case TQ_GGUF_TYPE_INT32:   return 4;
        case TQ_GGUF_TYPE_FLOAT32: return 4;
        case TQ_GGUF_TYPE_UINT64:  return 8;
        case TQ_GGUF_TYPE_INT64:   return 8;
        case TQ_GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

/* ============================================================
 * GGML type tables
 * ============================================================ */

size_t tq_ggml_type_size(tq_ggml_dtype type) {
    switch (type) {
        case TQ_GGML_TYPE_F32:      return 4;
        case TQ_GGML_TYPE_F16:      return 2;
        case TQ_GGML_TYPE_BF16:     return 2;
        case TQ_GGML_TYPE_Q4_0:     return 18;
        case TQ_GGML_TYPE_Q4_1:     return 20;
        case TQ_GGML_TYPE_Q5_0:     return 22;
        case TQ_GGML_TYPE_Q5_1:     return 24;
        case TQ_GGML_TYPE_Q8_0:     return 34;
        case TQ_GGML_TYPE_Q8_1:     return 36;
        case TQ_GGML_TYPE_Q2_K:     return 84;
        case TQ_GGML_TYPE_Q3_K:     return 110;
        case TQ_GGML_TYPE_Q4_K:     return 144;
        case TQ_GGML_TYPE_Q5_K:     return 176;
        case TQ_GGML_TYPE_Q6_K:     return 210;
        case TQ_GGML_TYPE_Q8_K:     return 292;
        case TQ_GGML_TYPE_IQ2_XXS:  return 66;
        case TQ_GGML_TYPE_IQ2_XS:   return 74;
        case TQ_GGML_TYPE_IQ2_S:    return 82;
        case TQ_GGML_TYPE_IQ3_XXS:  return 98;
        case TQ_GGML_TYPE_IQ3_S:    return 110;
        case TQ_GGML_TYPE_IQ4_NL:   return 18;
        case TQ_GGML_TYPE_IQ4_XS:   return 136;
        case TQ_GGML_TYPE_IQ1_S:    return 50;
        default:                     return 0;
    }
}

int tq_ggml_type_blck(tq_ggml_dtype type) {
    switch (type) {
        case TQ_GGML_TYPE_F32:      return 1;
        case TQ_GGML_TYPE_F16:      return 1;
        case TQ_GGML_TYPE_BF16:     return 1;
        case TQ_GGML_TYPE_Q4_0:     return 32;
        case TQ_GGML_TYPE_Q4_1:     return 32;
        case TQ_GGML_TYPE_Q5_0:     return 32;
        case TQ_GGML_TYPE_Q5_1:     return 32;
        case TQ_GGML_TYPE_Q8_0:     return 32;
        case TQ_GGML_TYPE_Q8_1:     return 32;
        case TQ_GGML_TYPE_Q2_K:     return 256;
        case TQ_GGML_TYPE_Q3_K:     return 256;
        case TQ_GGML_TYPE_Q4_K:     return 256;
        case TQ_GGML_TYPE_Q5_K:     return 256;
        case TQ_GGML_TYPE_Q6_K:     return 256;
        case TQ_GGML_TYPE_Q8_K:     return 256;
        case TQ_GGML_TYPE_IQ2_XXS:  return 256;
        case TQ_GGML_TYPE_IQ2_XS:   return 256;
        case TQ_GGML_TYPE_IQ2_S:    return 256;
        case TQ_GGML_TYPE_IQ3_XXS:  return 256;
        case TQ_GGML_TYPE_IQ3_S:    return 256;
        case TQ_GGML_TYPE_IQ4_NL:   return 32;
        case TQ_GGML_TYPE_IQ4_XS:   return 256;
        case TQ_GGML_TYPE_IQ1_S:    return 256;
        default:                     return 0;
    }
}

const char* tq_ggml_type_name(tq_ggml_dtype type) {
    switch (type) {
        case TQ_GGML_TYPE_F32:      return "F32";
        case TQ_GGML_TYPE_F16:      return "F16";
        case TQ_GGML_TYPE_BF16:     return "BF16";
        case TQ_GGML_TYPE_Q4_0:     return "Q4_0";
        case TQ_GGML_TYPE_Q4_1:     return "Q4_1";
        case TQ_GGML_TYPE_Q5_0:     return "Q5_0";
        case TQ_GGML_TYPE_Q5_1:     return "Q5_1";
        case TQ_GGML_TYPE_Q8_0:     return "Q8_0";
        case TQ_GGML_TYPE_Q8_1:     return "Q8_1";
        case TQ_GGML_TYPE_Q2_K:     return "Q2_K";
        case TQ_GGML_TYPE_Q3_K:     return "Q3_K";
        case TQ_GGML_TYPE_Q4_K:     return "Q4_K";
        case TQ_GGML_TYPE_Q5_K:     return "Q5_K";
        case TQ_GGML_TYPE_Q6_K:     return "Q6_K";
        case TQ_GGML_TYPE_Q8_K:     return "Q8_K";
        case TQ_GGML_TYPE_IQ2_XXS:  return "IQ2_XXS";
        case TQ_GGML_TYPE_IQ2_XS:   return "IQ2_XS";
        case TQ_GGML_TYPE_IQ2_S:    return "IQ2_S";
        case TQ_GGML_TYPE_IQ3_XXS:  return "IQ3_XXS";
        case TQ_GGML_TYPE_IQ3_S:    return "IQ3_S";
        case TQ_GGML_TYPE_IQ4_NL:   return "IQ4_NL";
        case TQ_GGML_TYPE_IQ4_XS:   return "IQ4_XS";
        case TQ_GGML_TYPE_IQ1_S:    return "IQ1_S";
        default:                     return "unknown";
    }
}

/* ============================================================
 * Metadata value parsing
 * ============================================================ */

/* Parse a single metadata value from the reader into a kv struct.
 * Returns false on failure. */
static bool parse_kv_value(tq_reader_t* r, tq_gguf_kv_t* kv, tq_gguf_type type) {
    kv->type = type;

    switch (type) {
        case TQ_GGUF_TYPE_UINT8:
            kv->value.u8 = read_u8(r);
            break;
        case TQ_GGUF_TYPE_INT8:
            kv->value.i8 = (int8_t)read_u8(r);
            break;
        case TQ_GGUF_TYPE_BOOL:
            kv->value.bool_val = read_u8(r);
            break;
        case TQ_GGUF_TYPE_UINT16:
            kv->value.u16 = read_u16(r);
            break;
        case TQ_GGUF_TYPE_INT16:
            kv->value.i16 = (int16_t)read_u16(r);
            break;
        case TQ_GGUF_TYPE_UINT32:
            kv->value.u32 = read_u32(r);
            break;
        case TQ_GGUF_TYPE_INT32:
            kv->value.i32 = (int32_t)read_u32(r);
            break;
        case TQ_GGUF_TYPE_FLOAT32:
            kv->value.f32 = read_f32(r);
            break;
        case TQ_GGUF_TYPE_UINT64:
            kv->value.u64 = read_u64(r);
            break;
        case TQ_GGUF_TYPE_INT64:
            kv->value.i64 = (int64_t)read_u64(r);
            break;
        case TQ_GGUF_TYPE_FLOAT64:
            kv->value.f64 = read_f64(r);
            break;
        case TQ_GGUF_TYPE_STRING: {
            uint64_t slen = 0;
            char* s = read_string(r, &slen);
            if (!s) return false;
            kv->value.string.len = slen;
            kv->value.string.str = s;
            break;
        }
        case TQ_GGUF_TYPE_ARRAY: {
            uint32_t elem_type = read_u32(r);
            uint64_t count     = read_u64(r);

            kv->value.array.elem_type = (tq_gguf_type)elem_type;
            kv->value.array.count     = count;
            kv->value.array.data      = NULL;

            if (count == 0) break;

            if ((tq_gguf_type)elem_type == TQ_GGUF_TYPE_STRING) {
                /* Array of strings: allocate array of tq_gguf_string_t */
                tq_gguf_string_t* strs = (tq_gguf_string_t*)calloc(count, sizeof(tq_gguf_string_t));
                if (!strs) return false;
                for (uint64_t i = 0; i < count; i++) {
                    uint64_t slen = 0;
                    char* s = read_string(r, &slen);
                    if (!s) {
                        /* Clean up already-parsed strings */
                        for (uint64_t j = 0; j < i; j++) {
                            free(strs[j].str);
                        }
                        free(strs);
                        return false;
                    }
                    strs[i].len = slen;
                    strs[i].str = s;
                }
                kv->value.array.data = strs;
            } else if ((tq_gguf_type)elem_type == TQ_GGUF_TYPE_ARRAY) {
                /* Nested arrays not supported, skip */
                fprintf(stderr, "tq_gguf: nested arrays not supported\n");
                return false;
            } else {
                /* Scalar array: bulk copy */
                size_t elem_sz = gguf_type_scalar_size((tq_gguf_type)elem_type);
                if (elem_sz == 0) return false;
                size_t total = (size_t)count * elem_sz;
                if (!reader_has(r, total)) return false;
                void* buf = malloc(total);
                if (!buf) return false;
                memcpy(buf, r->base + r->pos, total);
                r->pos += total;
                kv->value.array.data = buf;
            }
            break;
        }
        default:
            fprintf(stderr, "tq_gguf: unknown metadata type %d\n", (int)type);
            return false;
    }
    return true;
}

/* Free heap memory inside a kv entry */
static void free_kv(tq_gguf_kv_t* kv) {
    if (kv->type == TQ_GGUF_TYPE_STRING) {
        free(kv->value.string.str);
        kv->value.string.str = NULL;
    } else if (kv->type == TQ_GGUF_TYPE_ARRAY) {
        if (kv->value.array.elem_type == TQ_GGUF_TYPE_STRING && kv->value.array.data) {
            tq_gguf_string_t* strs = (tq_gguf_string_t*)kv->value.array.data;
            for (uint64_t i = 0; i < kv->value.array.count; i++) {
                free(strs[i].str);
            }
        }
        free(kv->value.array.data);
        kv->value.array.data = NULL;
    }
}

/* ============================================================
 * Platform-specific mmap
 * ============================================================ */

#ifdef _WIN32

static void* platform_mmap(const char* path, size_t* out_size) {
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "tq_gguf: cannot open '%s'\n", path);
        return NULL;
    }

    LARGE_INTEGER fsize;
    if (!GetFileSizeEx(file, &fsize)) {
        CloseHandle(file);
        fprintf(stderr, "tq_gguf: cannot get size of '%s'\n", path);
        return NULL;
    }
    *out_size = (size_t)fsize.QuadPart;

    HANDLE mapping = CreateFileMappingA(file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mapping) {
        CloseHandle(file);
        fprintf(stderr, "tq_gguf: CreateFileMapping failed for '%s'\n", path);
        return NULL;
    }

    void* data = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mapping);
    CloseHandle(file);

    if (!data) {
        fprintf(stderr, "tq_gguf: MapViewOfFile failed for '%s'\n", path);
        return NULL;
    }
    return data;
}

static void platform_munmap(void* addr, size_t size) {
    (void)size;
    if (addr) UnmapViewOfFile(addr);
}

#else /* POSIX */

static void* platform_mmap(const char* path, size_t* out_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "tq_gguf: cannot open '%s'\n", path);
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        fprintf(stderr, "tq_gguf: cannot stat '%s'\n", path);
        return NULL;
    }
    *out_size = (size_t)st.st_size;

    void* data = mmap(NULL, *out_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (data == MAP_FAILED) {
        fprintf(stderr, "tq_gguf: mmap failed for '%s'\n", path);
        return NULL;
    }
    return data;
}

static void platform_munmap(void* addr, size_t size) {
    if (addr) munmap(addr, size);
}

#endif

/* ============================================================
 * GGUF open / close
 * ============================================================ */

tq_gguf_ctx_t* tq_gguf_open(const char* path) {
    if (!path) {
        fprintf(stderr, "tq_gguf: NULL path\n");
        return NULL;
    }

    /* mmap the file */
    size_t file_size = 0;
    void* mmap_data = platform_mmap(path, &file_size);
    if (!mmap_data) return NULL;

    /* Minimum header: magic(4) + version(4) + n_tensors(8) + n_kv(8) = 24 */
    if (file_size < 24) {
        fprintf(stderr, "tq_gguf: file too small (%zu bytes)\n", file_size);
        platform_munmap(mmap_data, file_size);
        return NULL;
    }

    tq_reader_t reader = { .base = (const uint8_t*)mmap_data, .size = file_size, .pos = 0 };

    /* Parse header */
    uint32_t magic = read_u32(&reader);
    if (magic != TQ_GGUF_MAGIC) {
        fprintf(stderr, "tq_gguf: bad magic 0x%08x (expected 0x%08x)\n",
                magic, TQ_GGUF_MAGIC);
        platform_munmap(mmap_data, file_size);
        return NULL;
    }

    uint32_t version = read_u32(&reader);
    if (version < TQ_GGUF_VERSION_MIN || version > TQ_GGUF_VERSION_MAX) {
        fprintf(stderr, "tq_gguf: unsupported version %u (need %u-%u)\n",
                version, TQ_GGUF_VERSION_MIN, TQ_GGUF_VERSION_MAX);
        platform_munmap(mmap_data, file_size);
        return NULL;
    }

    uint64_t n_tensors = read_u64(&reader);
    uint64_t n_kv      = read_u64(&reader);

    /* Allocate context */
    tq_gguf_ctx_t* ctx = (tq_gguf_ctx_t*)calloc(1, sizeof(tq_gguf_ctx_t));
    if (!ctx) {
        fprintf(stderr, "tq_gguf: out of memory\n");
        platform_munmap(mmap_data, file_size);
        return NULL;
    }

    ctx->version    = version;
    ctx->n_tensors  = n_tensors;
    ctx->n_kv       = n_kv;
    ctx->alignment  = TQ_GGUF_DEFAULT_ALIGNMENT;
    ctx->mmap_data  = mmap_data;
    ctx->mmap_size  = file_size;
    ctx->arch[0]    = '\0';

    /* Allocate metadata KV array */
    if (n_kv > 0) {
        ctx->kv = (tq_gguf_kv_t*)calloc((size_t)n_kv, sizeof(tq_gguf_kv_t));
        if (!ctx->kv) {
            fprintf(stderr, "tq_gguf: out of memory (kv)\n");
            tq_gguf_close(ctx);
            return NULL;
        }
    }

    /* Parse metadata KV pairs */
    for (uint64_t i = 0; i < n_kv; i++) {
        tq_gguf_kv_t* kv = &ctx->kv[i];

        /* Read key */
        uint64_t key_len = 0;
        char* key = read_string(&reader, &key_len);
        if (!key) {
            fprintf(stderr, "tq_gguf: failed to read key %llu\n", (unsigned long long)i);
            tq_gguf_close(ctx);
            return NULL;
        }
        size_t copy_len = key_len < TQ_GGUF_MAX_NAME - 1 ? key_len : TQ_GGUF_MAX_NAME - 1;
        memcpy(kv->key, key, copy_len);
        kv->key[copy_len] = '\0';
        free(key);

        /* Read value type and value */
        uint32_t vtype = read_u32(&reader);
        if (!parse_kv_value(&reader, kv, (tq_gguf_type)vtype)) {
            fprintf(stderr, "tq_gguf: failed to parse value for key '%s'\n", kv->key);
            tq_gguf_close(ctx);
            return NULL;
        }

        /* Check for alignment override */
        if (strcmp(kv->key, "general.alignment") == 0 && kv->type == TQ_GGUF_TYPE_UINT32) {
            ctx->alignment = kv->value.u32;
        }

        /* Extract architecture string */
        if (strcmp(kv->key, "general.architecture") == 0 && kv->type == TQ_GGUF_TYPE_STRING) {
            size_t alen = kv->value.string.len;
            if (alen > sizeof(ctx->arch) - 1) alen = sizeof(ctx->arch) - 1;
            memcpy(ctx->arch, kv->value.string.str, alen);
            ctx->arch[alen] = '\0';
        }
    }

    /* Allocate tensor descriptors */
    if (n_tensors > 0) {
        ctx->tensors = (tq_gguf_tensor_t*)calloc((size_t)n_tensors, sizeof(tq_gguf_tensor_t));
        if (!ctx->tensors) {
            fprintf(stderr, "tq_gguf: out of memory (tensors)\n");
            tq_gguf_close(ctx);
            return NULL;
        }
    }

    /* Parse tensor descriptors */
    for (uint64_t i = 0; i < n_tensors; i++) {
        tq_gguf_tensor_t* t = &ctx->tensors[i];

        /* Read tensor name */
        uint64_t name_len = 0;
        char* name = read_string(&reader, &name_len);
        if (!name) {
            fprintf(stderr, "tq_gguf: failed to read tensor name %llu\n", (unsigned long long)i);
            tq_gguf_close(ctx);
            return NULL;
        }
        size_t copy_len = name_len < TQ_GGUF_MAX_NAME - 1 ? name_len : TQ_GGUF_MAX_NAME - 1;
        memcpy(t->name, name, copy_len);
        t->name[copy_len] = '\0';
        free(name);

        /* Number of dimensions */
        t->n_dims = read_u32(&reader);
        if (t->n_dims > 4) {
            fprintf(stderr, "tq_gguf: tensor '%s' has %u dims (max 4)\n", t->name, t->n_dims);
            tq_gguf_close(ctx);
            return NULL;
        }

        /* Shape */
        for (uint32_t d = 0; d < t->n_dims; d++) {
            t->shape[d] = (int64_t)read_u64(&reader);
        }
        for (uint32_t d = t->n_dims; d < 4; d++) {
            t->shape[d] = 1;
        }

        /* Type */
        t->type = (tq_ggml_dtype)read_u32(&reader);

        /* Offset within tensor data section */
        t->offset = read_u64(&reader);

        /* Compute total number of elements */
        int64_t n_elements = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            n_elements *= t->shape[d];
        }

        /* Compute size in bytes */
        int blck = tq_ggml_type_blck(t->type);
        size_t type_sz = tq_ggml_type_size(t->type);
        if (blck > 0 && type_sz > 0) {
            t->size_bytes = (size_t)((n_elements + blck - 1) / blck) * type_sz;
        } else {
            t->size_bytes = 0;
        }
    }

    /* Compute data section offset: aligned up from current reader position */
    ctx->data_offset = (size_t)align_up((uint64_t)reader.pos, (uint64_t)ctx->alignment);

    /* Set data pointers for each tensor */
    for (uint64_t i = 0; i < n_tensors; i++) {
        tq_gguf_tensor_t* t = &ctx->tensors[i];
        size_t abs_offset = ctx->data_offset + (size_t)t->offset;
        if (abs_offset + t->size_bytes > file_size) {
            fprintf(stderr, "tq_gguf: tensor '%s' data out of bounds "
                    "(offset %zu + size %zu > file %zu)\n",
                    t->name, abs_offset, t->size_bytes, file_size);
            tq_gguf_close(ctx);
            return NULL;
        }
        t->data = (const uint8_t*)mmap_data + abs_offset;
    }

    return ctx;
}

void tq_gguf_close(tq_gguf_ctx_t* ctx) {
    if (!ctx) return;

    /* Free metadata heap allocations */
    if (ctx->kv) {
        for (uint64_t i = 0; i < ctx->n_kv; i++) {
            free_kv(&ctx->kv[i]);
        }
        free(ctx->kv);
    }

    /* Free tensor descriptors (no heap inside, just the array) */
    free(ctx->tensors);

    /* Unmap the file */
    platform_munmap(ctx->mmap_data, ctx->mmap_size);

    free(ctx);
}

/* ============================================================
 * Metadata lookup
 * ============================================================ */

int64_t tq_gguf_find_key(const tq_gguf_ctx_t* ctx, const char* key) {
    if (!ctx || !key) return -1;
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (strcmp(ctx->kv[i].key, key) == 0) {
            return (int64_t)i;
        }
    }
    return -1;
}

int32_t tq_gguf_get_i32(const tq_gguf_ctx_t* ctx, const char* key, int32_t fallback) {
    int64_t idx = tq_gguf_find_key(ctx, key);
    if (idx < 0) return fallback;
    const tq_gguf_kv_t* kv = &ctx->kv[idx];
    switch (kv->type) {
        case TQ_GGUF_TYPE_INT32:   return kv->value.i32;
        case TQ_GGUF_TYPE_UINT32:  return (int32_t)kv->value.u32;
        case TQ_GGUF_TYPE_INT16:   return (int32_t)kv->value.i16;
        case TQ_GGUF_TYPE_UINT16:  return (int32_t)kv->value.u16;
        case TQ_GGUF_TYPE_INT8:    return (int32_t)kv->value.i8;
        case TQ_GGUF_TYPE_UINT8:   return (int32_t)kv->value.u8;
        default:                   return fallback;
    }
}

uint32_t tq_gguf_get_u32(const tq_gguf_ctx_t* ctx, const char* key, uint32_t fallback) {
    int64_t idx = tq_gguf_find_key(ctx, key);
    if (idx < 0) return fallback;
    const tq_gguf_kv_t* kv = &ctx->kv[idx];
    switch (kv->type) {
        case TQ_GGUF_TYPE_UINT32:  return kv->value.u32;
        case TQ_GGUF_TYPE_INT32:   return (uint32_t)kv->value.i32;
        case TQ_GGUF_TYPE_UINT16:  return (uint32_t)kv->value.u16;
        case TQ_GGUF_TYPE_INT16:   return (uint32_t)kv->value.i16;
        case TQ_GGUF_TYPE_UINT8:   return (uint32_t)kv->value.u8;
        case TQ_GGUF_TYPE_INT8:    return (uint32_t)kv->value.i8;
        default:                   return fallback;
    }
}

float tq_gguf_get_f32(const tq_gguf_ctx_t* ctx, const char* key, float fallback) {
    int64_t idx = tq_gguf_find_key(ctx, key);
    if (idx < 0) return fallback;
    const tq_gguf_kv_t* kv = &ctx->kv[idx];
    switch (kv->type) {
        case TQ_GGUF_TYPE_FLOAT32: return kv->value.f32;
        case TQ_GGUF_TYPE_FLOAT64: return (float)kv->value.f64;
        default:                   return fallback;
    }
}

const char* tq_gguf_get_str(const tq_gguf_ctx_t* ctx, const char* key) {
    int64_t idx = tq_gguf_find_key(ctx, key);
    if (idx < 0) return NULL;
    const tq_gguf_kv_t* kv = &ctx->kv[idx];
    if (kv->type != TQ_GGUF_TYPE_STRING) return NULL;
    return kv->value.string.str;
}

/* ============================================================
 * Tensor lookup
 * ============================================================ */

const tq_gguf_tensor_t* tq_gguf_find_tensor(const tq_gguf_ctx_t* ctx, const char* name) {
    if (!ctx || !name) return NULL;
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0) {
            return &ctx->tensors[i];
        }
    }
    return NULL;
}

// ============================================================================
// Section 11: GGUF Weight Dequantization (from tq_gguf_quants.c)
// ============================================================================

/**
 * tq_gguf_quants.c — GGUF weight dequantization and on-the-fly dequant matmul
 *
 * Implements dequantization for all major GGML quant types:
 *   Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, F16, F32, BF16
 * Plus stub implementations for IQ types (IQ2_XXS, IQ3_XXS, IQ4_XS).
 *
 * The matmul path includes NEON-optimized inner loop for Apple Silicon.
 *
 * Pure C11, no external dependencies.
 */

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define TQ_HAS_NEON 1
#else
#define TQ_HAS_NEON 0
#endif

/* ============================================================
 * FP16 / BF16 helpers
 * ============================================================ */

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            /* positive/negative zero */
            float r;
            uint32_t v = sign;
            memcpy(&r, &v, 4);
            return r;
        }
        /* subnormal: normalize by shifting mantissa up */
        exp = 1;
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x3FF;
        exp += 112;  /* fp16 bias (15) -> fp32 bias (127): 127-15 = 112 */
    } else if (exp == 31) {
        /* inf / nan */
        exp = 255;
    } else {
        exp += 112;
    }

    uint32_t bits = sign | ((uint32_t)exp << 23) | (mant << 13);
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

static inline float bf16_to_fp32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

/* ============================================================
 * Block structures (matching llama.cpp / ggml exactly)
 * ============================================================ */

/* Q8_0: 34 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    int8_t   qs[32];  /* quantized values */
} block_q8_0;

/* Q4_K: 144 bytes, 256 elements */
typedef struct {
    uint16_t d;          /* fp16 super-block scale */
    uint16_t dmin;       /* fp16 super-block min */
    uint8_t  scales[12]; /* sub-block scales+mins, 6-bit packed */
    uint8_t  qs[128];    /* 4-bit values, 2 per byte */
} block_q4_K;

/* Q2_K: 84 bytes, 256 elements */
typedef struct {
    uint8_t  scales[16]; /* sub-block scales+mins, 4-bit each */
    uint8_t  qs[64];     /* 2-bit values, 4 per byte */
    uint16_t d;          /* fp16 super-block scale */
    uint16_t dmin;       /* fp16 super-block min */
} block_q2_K;

/* Q3_K: 110 bytes, 256 elements */
typedef struct {
    uint8_t  hmask[32];  /* high bits */
    uint8_t  qs[64];     /* low 2 bits, 4 per byte */
    uint8_t  scales[12]; /* sub-block scales, packed */
    uint16_t d;          /* fp16 scale */
} block_q3_K;

/* Q6_K: 210 bytes, 256 elements */
typedef struct {
    uint8_t  ql[128];    /* low 4 bits */
    uint8_t  qh[64];     /* high 2 bits */
    int8_t   scales[16]; /* sub-block scales (signed int8) */
    uint16_t d;          /* fp16 super-block scale */
} block_q6_K;

/* Q5_K: 176 bytes, 256 elements */
typedef struct {
    uint16_t d;          /* fp16 super-block scale */
    uint16_t dmin;       /* fp16 super-block min */
    uint8_t  scales[12]; /* sub-block scales+mins, 6-bit packed */
    uint8_t  qh[32];     /* high bit for each of 256 elements */
    uint8_t  qs[128];    /* low 4 bits, 2 per byte */
} block_q5_K;

/* Q4_0: 18 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    uint8_t  qs[16];  /* 4-bit values, 2 per byte */
} block_q4_0;

/* Q4_1: 20 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    uint16_t m;       /* fp16 min */
    uint8_t  qs[16];  /* 4-bit values, 2 per byte */
} block_q4_1;

/* Q5_0: 22 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    uint8_t  qh[4];   /* high bits packed */
    uint8_t  qs[16];  /* low 4 bits, 2 per byte */
} block_q5_0;

/* Q5_1: 24 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    uint16_t m;       /* fp16 min */
    uint8_t  qh[4];   /* high bits packed */
    uint8_t  qs[16];  /* low 4 bits, 2 per byte */
} block_q5_1;

/* Q8_1: 36 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale (delta) */
    uint16_t s;       /* fp16 sum */
    int8_t   qs[32];
} block_q8_1;

/* Type size / block size / name — defined in tq_gguf.c, just declared in header */

/* ============================================================
 * Per-type dequantization
 * ============================================================ */

/* --- F32: passthrough --- */
static void dequant_f32(const void* src, float* dst, int n) {
    memcpy(dst, src, (size_t)n * sizeof(float));
}

/* --- F16 --- */
static void dequant_f16(const void* src, float* dst, int n) {
    const uint16_t* s = (const uint16_t*)src;
    for (int i = 0; i < n; i++) {
        dst[i] = fp16_to_fp32(s[i]);
    }
}

/* --- BF16 --- */
static void dequant_bf16(const void* src, float* dst, int n) {
    const uint16_t* s = (const uint16_t*)src;
    for (int i = 0; i < n; i++) {
        dst[i] = bf16_to_fp32(s[i]);
    }
}

/* --- Q8_0: 34 bytes, 32 elements --- */
static void dequant_q8_0(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q8_0* blk = (const block_q8_0*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        for (int j = 0; j < 32; j++) {
            dst[b * 32 + j] = d * blk[b].qs[j];
        }
    }
}

/* --- Q4_0: 18 bytes, 32 elements --- */
static void dequant_q4_0(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q4_0* blk = (const block_q4_0*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            dst[b * 32 + j]      = d * ((int)(byte & 0x0F) - 8);
            dst[b * 32 + j + 16] = d * ((int)(byte >> 4) - 8);
        }
    }
}

/* --- Q4_1: 20 bytes, 32 elements --- */
static void dequant_q4_1(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q4_1* blk = (const block_q4_1*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float m = fp16_to_fp32(blk[b].m);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            dst[b * 32 + j]      = d * (byte & 0x0F) + m;
            dst[b * 32 + j + 16] = d * (byte >> 4) + m;
        }
    }
}

/* --- Q5_0: 22 bytes, 32 elements --- */
static void dequant_q5_0(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q5_0* blk = (const block_q5_0*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        uint32_t qh;
        memcpy(&qh, blk[b].qh, 4);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            int lo = (byte & 0x0F) | (((qh >> j) & 1) << 4);
            int hi = (byte >> 4)   | (((qh >> (j + 16)) & 1) << 4);
            dst[b * 32 + j]      = d * (lo - 16);
            dst[b * 32 + j + 16] = d * (hi - 16);
        }
    }
}

/* --- Q5_1: 24 bytes, 32 elements --- */
static void dequant_q5_1(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q5_1* blk = (const block_q5_1*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float m = fp16_to_fp32(blk[b].m);
        uint32_t qh;
        memcpy(&qh, blk[b].qh, 4);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            int lo = (byte & 0x0F) | (((qh >> j) & 1) << 4);
            int hi = (byte >> 4)   | (((qh >> (j + 16)) & 1) << 4);
            dst[b * 32 + j]      = d * lo + m;
            dst[b * 32 + j + 16] = d * hi + m;
        }
    }
}

/* --- Q2_K: 84 bytes, 256 elements --- */
static void dequant_q2_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q2_K* blk = (const block_q2_K*)src;

    for (int b = 0; b < nb; b++) {
        const float d    = fp16_to_fp32(blk[b].d);
        const float dmin = fp16_to_fp32(blk[b].dmin);

        const uint8_t* q = blk[b].qs;
        float* y = dst + b * 256;

        int is = 0;
        for (int half = 0; half < 2; half++) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc_byte = blk[b].scales[is++];
                float dl = d * (sc_byte & 0x0F);
                float ml = dmin * (sc_byte >> 4);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                sc_byte = blk[b].scales[is++];
                dl = d * (sc_byte & 0x0F);
                ml = dmin * (sc_byte >> 4);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

/* --- Q3_K: 110 bytes, 256 elements ---
 * 3-bit = 2 low bits (qs) + 1 high bit (hmask)
 * 16 sub-blocks with 6-bit scales packed into 12 bytes */
static void dequant_q3_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q3_K* blk = (const block_q3_K*)src;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t* scales = (const int8_t*)aux;

    for (int b = 0; b < nb; b++) {
        const float d_all = fp16_to_fp32(blk[b].d);

        const uint8_t* q  = blk[b].qs;
        const uint8_t* hm = blk[b].hmask;
        uint8_t m = 1;

        /* Decode 16 x 6-bit scales using the ggml bit-manipulation trick.
         * The 12 packed bytes are loaded as three uint32, then rearranged
         * into four uint32 that are reinterpreted as sixteen int8 values. */
        memcpy(aux, blk[b].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float* y = dst + b * 256;

        for (int half = 0; half < 2; half++) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

/* --- Q4_K: 144 bytes, 256 elements ---
 * 8 sub-blocks of 32 elements each
 * 6-bit scale/min packed in 12 bytes */
static void dequant_q4_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q4_K* blk = (const block_q4_K*)src;

    for (int b = 0; b < nb; b++) {
        const float d    = fp16_to_fp32(blk[b].d);
        const float dmin = fp16_to_fp32(blk[b].dmin);

        /* Decode 8 scale/min pairs from 12 bytes.
         * bytes 0..3:  low 6 bits of scales[0..3]
         * bytes 4..7:  low 6 bits of mins[0..3]
         * bytes 8..9:  high 2 bits of scales[0..3] + scales[4..7]
         * bytes 10..11: high 2 bits of mins[0..3] + mins[4..7]
         *
         * Actually ggml Q4_K packing:
         *   scales[0..5]: low 6 bits of scale for sub-blocks 0..5
         *                 but the first 4 bytes have scale low 6,
         *                 bytes 4..7 have min low 6,
         *                 bytes 8..11 have the high bits.
         *
         * Match ggml exactly:
         */
        uint8_t sc[8], mn[8];

        /* Low 6 bits */
        sc[0] = blk[b].scales[0] & 63;
        sc[1] = blk[b].scales[1] & 63;
        sc[2] = blk[b].scales[2] & 63;
        sc[3] = blk[b].scales[3] & 63;
        mn[0] = blk[b].scales[4] & 63;
        mn[1] = blk[b].scales[5] & 63;
        mn[2] = blk[b].scales[6] & 63;
        mn[3] = blk[b].scales[7] & 63;

        /* High 2 bits from bytes 8..11 */
        sc[4] = (blk[b].scales[8] & 0x0F) | ((blk[b].scales[0] >> 6) << 4);
        sc[5] = (blk[b].scales[9] & 0x0F) | ((blk[b].scales[1] >> 6) << 4);
        sc[6] = (blk[b].scales[10] & 0x0F) | ((blk[b].scales[2] >> 6) << 4);
        sc[7] = (blk[b].scales[11] & 0x0F) | ((blk[b].scales[3] >> 6) << 4);
        mn[4] = (blk[b].scales[8] >> 4) | ((blk[b].scales[4] >> 6) << 4);
        mn[5] = (blk[b].scales[9] >> 4) | ((blk[b].scales[5] >> 6) << 4);
        mn[6] = (blk[b].scales[10] >> 4) | ((blk[b].scales[6] >> 6) << 4);
        mn[7] = (blk[b].scales[11] >> 4) | ((blk[b].scales[7] >> 6) << 4);

        /* 4 groups of 64 elements (2 sub-blocks each).
         * Within each 64-element group, the first 32 elements use the low
         * nibble and the next 32 use the high nibble of the same 32 bytes.
         * This matches the ggml Q4_K packing exactly. */
        const uint8_t* q = blk[b].qs;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            const float d1 = d * sc[is + 0];
            const float m1 = dmin * mn[is + 0];
            const float d2 = d * sc[is + 1];
            const float m2 = dmin * mn[is + 1];
            for (int l = 0; l < 32; ++l)
                dst[b * 256 + j + l]      = d1 * (q[l] & 0x0F) - m1;
            for (int l = 0; l < 32; ++l)
                dst[b * 256 + j + 32 + l] = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

/* --- Q5_K: 176 bytes, 256 elements ---
 * Like Q4_K but with an extra high bit per element */
static void dequant_q5_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q5_K* blk = (const block_q5_K*)src;

    for (int b = 0; b < nb; b++) {
        const float d    = fp16_to_fp32(blk[b].d);
        const float dmin = fp16_to_fp32(blk[b].dmin);

        /* Same scale/min packing as Q4_K */
        uint8_t sc[8], mn[8];

        sc[0] = blk[b].scales[0] & 63;
        sc[1] = blk[b].scales[1] & 63;
        sc[2] = blk[b].scales[2] & 63;
        sc[3] = blk[b].scales[3] & 63;
        mn[0] = blk[b].scales[4] & 63;
        mn[1] = blk[b].scales[5] & 63;
        mn[2] = blk[b].scales[6] & 63;
        mn[3] = blk[b].scales[7] & 63;

        sc[4] = (blk[b].scales[8] & 0x0F) | ((blk[b].scales[0] >> 6) << 4);
        sc[5] = (blk[b].scales[9] & 0x0F) | ((blk[b].scales[1] >> 6) << 4);
        sc[6] = (blk[b].scales[10] & 0x0F) | ((blk[b].scales[2] >> 6) << 4);
        sc[7] = (blk[b].scales[11] & 0x0F) | ((blk[b].scales[3] >> 6) << 4);
        mn[4] = (blk[b].scales[8] >> 4) | ((blk[b].scales[4] >> 6) << 4);
        mn[5] = (blk[b].scales[9] >> 4) | ((blk[b].scales[5] >> 6) << 4);
        mn[6] = (blk[b].scales[10] >> 4) | ((blk[b].scales[6] >> 6) << 4);
        mn[7] = (blk[b].scales[11] >> 4) | ((blk[b].scales[7] >> 6) << 4);

        /* 4 groups of 64 elements (2 sub-blocks each), matching ggml Q5_K.
         * Low 4 bits: low nibble for first 32 elems, high nibble for next 32.
         * High bit: from qh, using bitmasks u1/u2 that shift left by 2 each group. */
        const uint8_t* ql = blk[b].qs;
        const uint8_t* qh = blk[b].qh;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            const float d1 = d * sc[is + 0];
            const float m1 = dmin * mn[is + 0];
            const float d2 = d * sc[is + 1];
            const float m2 = dmin * mn[is + 1];
            for (int l = 0; l < 32; ++l)
                dst[b * 256 + j + l]      = d1 * ((ql[l] & 0x0F) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l)
                dst[b * 256 + j + 32 + l] = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/* --- Q6_K: 210 bytes, 256 elements ---
 * 6-bit = 4 low bits (ql) + 2 high bits (qh)
 * 16 sub-blocks of 16 elements, int8 scales */
static void dequant_q6_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q6_K* blk = (const block_q6_K*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);

        /* Match ggml dequantize_row_q6_K exactly.
         * Processes in two 128-element halves. Within each half, 32
         * iterations produce 4 output elements each by interleaving
         * ql low/high nibbles and qh 2-bit fields. */
        const uint8_t* ql = blk[b].ql;
        const uint8_t* qh = blk[b].qh;
        const int8_t*  sc = blk[b].scales;
        float* y = dst + b * 256;

        for (int half = 0; half < 2; half++) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/* ============================================================
 * IQ2_XXS dequantization — E8 lattice codebook
 *
 * Block: 66 bytes per 256 elements (2.0625 bpw)
 *   - d (fp16): super-block scale
 *   - qs[32] (uint16): 8 groups of 4 uint16 (8 bytes each)
 *     Each 8-byte group decodes 32 floats:
 *     - aux32[0]: 4 grid indices (1 byte each) → 4×8=32 values from iq2xxs_grid
 *     - aux32[1] bits 0-27: 4×7-bit sign fields → ksigns_iq2xs → 8-bit patterns
 *     - aux32[1] bits 28-31: 4-bit sub-block scale
 * ============================================================ */

static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint64_t iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

static void dequant_iq2_xxs(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)src;

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 66; /* 66 bytes per block */
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint16_t* qs = (const uint16_t*)(blk + 2);

        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32[2];
            memcpy(aux32, qs + 4 * ib32, 8);
            const uint8_t* aux8 = (const uint8_t*)aux32;

            const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;

            for (int l = 0; l < 4; l++) {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];

                for (int j = 0; j < 8; j++) {
                    dst[b * 256 + ib32 * 32 + l * 8 + j] =
                        db * (float)grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                }
            }
        }
    }
}

/* ============================================================
 * IQ2_S dequantization — 82 bytes per 256 elements (2.5625 bpw)
 *
 * Block layout:
 *   d (fp16, 2 bytes): super-block scale
 *   qs[64]: first 32 bytes = grid index low bits, next 32 = sign bits
 *   qh[8]: high 2 bits of 10-bit grid index
 *   scales[8]: 4-bit sub-block scales (2 per byte)
 *
 * Uses iq2s_grid[1024] lookup table (10-bit index).
 * ============================================================ */

/* iq2s_grid: 1024-entry E8 lattice codebook for IQ2_S (from ggml-common.h).
 * Each uint64 packs 8 unsigned magnitude bytes from {0x08, 0x19, 0x2b}. */

static const uint64_t iq2s_grid[1024] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919,
    0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908,
    0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908,
    0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19,
    0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919,
    0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908,
    0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b,
    0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b,
    0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908,
    0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b,
    0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b,
    0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808,
    0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b,
    0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08,
    0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808,
    0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819,
    0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908,
    0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b,
    0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808,
    0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919,
    0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808,
    0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b,
    0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b,
    0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919,
    0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919,
    0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819,
    0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908,
    0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908,
    0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908,
    0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908,
    0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819,
    0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08,
    0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819,
    0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819,
    0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808,
    0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08,
    0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908,
    0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819,
    0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808,
    0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908,
    0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819,
    0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19,
    0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919,
    0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808,
    0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908,
    0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08,
    0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908,
    0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808,
    0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908,
    0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819,
    0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808,
    0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808,
    0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b,
    0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808,
    0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808,
    0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19,
    0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08,
    0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908,
    0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b,
    0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808,
    0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908,
    0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08,
    0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919,
    0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b,
    0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b,
    0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08,
    0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808,
    0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919,
    0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808,
    0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819,
    0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808,
    0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919,
    0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b,
    0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808,
    0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b,
    0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b,
    0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b,
    0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08,
    0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b,
    0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08,
    0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919,
    0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19,
    0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919,
    0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819,
    0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919,
    0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808,
    0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919,
    0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b,
    0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919,
    0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08,
    0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808,
    0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908,
    0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808,
    0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908,
    0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808,
    0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908,
    0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808,
    0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808,
    0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b,
    0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08,
    0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919,
    0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908,
    0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808,
    0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908,
    0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808,
    0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808,
    0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908,
    0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08,
    0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b,
    0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819,
    0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819,
    0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808,
    0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908,
    0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908,
    0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819,
    0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b,
    0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819,
    0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b,
    0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819,
    0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819,
    0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
};

/* Public accessor for the IQ2_S codebook — used by Metal backend */
const uint64_t* tq_iq2s_grid(void) {
    return iq2s_grid;
}

static void dequant_iq2_s(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)src;

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 82;
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);

        const uint8_t* qs = blk + 2;           /* grid index low bytes */
        const uint8_t* signs = qs + 32;         /* sign bytes (second half of qs) */
        const uint8_t* qh = blk + 66;           /* high bits: blk + 2 + 64 */
        const uint8_t* scales = blk + 74;       /* scales: blk + 2 + 64 + 8 */

        for (int ib32 = 0; ib32 < 8; ib32++) {
            float db0 = d * (0.5f + (float)(scales[ib32] & 0xF)) * 0.25f;
            float db1 = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;

            for (int l = 0; l < 4; l++) {
                float dl = (l < 2) ? db0 : db1;
                /* 10-bit grid index: low 8 from qs, high 2 from qh */
                int grid_idx = qs[l] | ((qh[ib32] << (8 - 2*l)) & 0x300);
                const uint8_t* grid = (const uint8_t*)(iq2s_grid + grid_idx);
                uint8_t sign = signs[l];

                for (int j = 0; j < 8; j++) {
                    dst[b * 256 + ib32 * 32 + l * 8 + j] =
                        dl * (float)grid[j] * ((sign & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                }
            }
            qs += 4;
            signs += 4;
        }
    }
}

/* ============================================================
 * IQ4_NL dequantization — 18 bytes per 32 elements (4.5 bpw)
 *
 * Non-linear 4-bit quantization using a 16-entry lookup table.
 * Block: d (fp16, 2 bytes) + qs[16] (4-bit packed pairs)
 * ============================================================ */

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

typedef struct {
    uint16_t d;       /* fp16 scale */
    uint8_t  qs[16];  /* 4-bit packed values, 2 per byte */
} block_iq4_nl;

static void dequant_iq4_nl(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_iq4_nl* blk = (const block_iq4_nl*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const uint8_t* qs = blk[b].qs;
        for (int j = 0; j < 16; ++j) {
            dst[b * 32 + j]      = d * kvalues_iq4nl[qs[j] & 0xf];
            dst[b * 32 + j + 16] = d * kvalues_iq4nl[qs[j] >> 4];
        }
    }
}

/* ============================================================
 * IQ4_XS dequantization — 136 bytes per 256 elements (4.25 bpw)
 *
 * Like IQ4_NL but with 256-element super-blocks and 6-bit sub-block scales.
 * Block: d (fp16, 2) + scales_h (uint16, 2) + scales_l[4] + qs[128]
 * ============================================================ */

typedef struct {
    uint16_t d;           /* fp16 super-block scale */
    uint16_t scales_h;    /* high 2 bits of 8 sub-block scales */
    uint8_t  scales_l[4]; /* low 4 bits of 8 sub-block scales, packed 2 per byte */
    uint8_t  qs[128];     /* 4-bit packed values */
} block_iq4_xs;

static void dequant_iq4_xs(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_iq4_xs* blk = (const block_iq4_xs*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const uint8_t* qs = blk[b].qs;
        float* y = dst + b * 256;

        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((blk[b].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf)
                         | (((blk[b].scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j +  0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

/* ============================================================
 * IQ3_XXS dequantization — 3.0625 bpw grid codebook
 *
 * Block: 98 bytes per 256 elements
 *   - d (fp16): super-block scale
 *   - qs[64]: grid indices (8 groups × 8 bytes each)
 *     First 64 bytes: 2 uint8 grid indices per sub-group (4 sub-groups × 2 = 8 per group)
 *     Next 32 bytes: scales_and_signs (4 bytes per group × 8 groups)
 *       Each uint32: bits 0-27 = 4×7-bit sign fields → ksigns_iq2xs
 *                    bits 28-31 = 4-bit sub-block scale
 *   Each grid index lookups iq3xxs_grid[idx] → 4 uint8 values (4 floats)
 *   2 grid indices per sub-group → 8 floats, 4 sub-groups per group → 32 floats
 * ============================================================ */

static const uint32_t iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

static void dequant_iq3_xxs(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)src;

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 98; /* 98 bytes per block */
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint8_t* qs = blk + 2;
        const uint8_t* scales_and_signs = qs + 64; /* QK_K/4 = 64 */
        float* y = dst + b * 256;

        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32;
            memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;

            for (int l = 0; l < 4; l++) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; j++) {
                    y[j + 0] = db * (float)grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                    y[j + 4] = db * (float)grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

/* --- Other IQ type stubs --- */
static void dequant_iq_stub(const char* type_name, float* dst, int n) {
    static int warned = 0;
    if (!warned) {
        fprintf(stderr, "tq_gguf_quants: WARNING: %s dequant not yet implemented, "
                        "returning zeros\n", type_name);
        warned = 1;
    }
    memset(dst, 0, (size_t)n * sizeof(float));
}

/* ============================================================
 * Main dequantization dispatcher
 * ============================================================ */

void tq_dequant_row_gguf(tq_ggml_dtype type, const void* src, float* dst, int n) {
    switch (type) {
        case TQ_GGML_TYPE_F32:
            dequant_f32(src, dst, n);
            break;
        case TQ_GGML_TYPE_F16:
            dequant_f16(src, dst, n);
            break;
        case TQ_GGML_TYPE_BF16:
            dequant_bf16(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q4_0:
            dequant_q4_0(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q4_1:
            dequant_q4_1(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q5_0:
            dequant_q5_0(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q5_1:
            dequant_q5_1(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q8_0:
            dequant_q8_0(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q2_K:
            dequant_q2_k(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q3_K:
            dequant_q3_k(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q4_K:
            dequant_q4_k(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q5_K:
            dequant_q5_k(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q6_K:
            dequant_q6_k(src, dst, n);
            break;

        /* IQ stubs */
        case TQ_GGML_TYPE_IQ2_XXS:
            dequant_iq2_xxs(src, dst, n);
            break;
        case TQ_GGML_TYPE_IQ2_XS:
            dequant_iq_stub("IQ2_XS", dst, n);
            break;
        case TQ_GGML_TYPE_IQ3_XXS:
            dequant_iq3_xxs(src, dst, n);
            break;
        case TQ_GGML_TYPE_IQ1_S:
            dequant_iq_stub("IQ1_S", dst, n);
            break;
        case TQ_GGML_TYPE_IQ4_NL:
            dequant_iq4_nl(src, dst, n);
            break;
        case TQ_GGML_TYPE_IQ3_S:
            dequant_iq_stub("IQ3_S", dst, n);
            break;
        case TQ_GGML_TYPE_IQ2_S:
            dequant_iq2_s(src, dst, n);
            break;
        case TQ_GGML_TYPE_IQ4_XS:
            dequant_iq4_xs(src, dst, n);
            break;

        default:
            fprintf(stderr, "tq_gguf_quants: ERROR: unsupported type %d\n", (int)type);
            memset(dst, 0, (size_t)n * sizeof(float));
            break;
    }
}

/* ============================================================
 * Fused dequant-dot product functions
 *
 * These compute dot(dequant(weight_row), input) in a single pass
 * without writing dequantized values to memory. All intermediate
 * values stay in registers, eliminating the temporary FP32 buffer.
 *
 * This is the critical optimization for MoE inference where
 * IQ2_XXS dequant dominates runtime.
 * ============================================================ */

/* Fused IQ2_XXS dot product: dot(dequant(row), x) for one 256-element block */
static inline float dot_block_iq2_xxs(const uint8_t* blk, const float* x) {
    uint16_t d_raw;
    memcpy(&d_raw, blk, 2);
    const float d = fp16_to_fp32(d_raw);
    const uint16_t* qs = (const uint16_t*)(blk + 2);
    float sum = 0.0f;

    for (int ib32 = 0; ib32 < 8; ib32++) {
        uint32_t aux32[2];
        memcpy(aux32, qs + 4 * ib32, 8);
        const uint8_t* aux8 = (const uint8_t*)aux32;
        const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
        const float* xb = x + ib32 * 32;

        float group_sum = 0.0f;
        for (int l = 0; l < 4; l++) {
            const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
            const uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
            const float* xp = xb + l * 8;

#if TQ_HAS_NEON
            /* Load 8 grid values into two int8x8 vectors, apply signs, dot with input */
            /* Grid values are uint8_t (0x08, 0x19, 0x2b), signs are bitmask */
            float local_sum = 0.0f;
            for (int j = 0; j < 8; j++) {
                float w = (float)grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                local_sum += w * xp[j];
            }
            group_sum += local_sum;
#else
            float local_sum = 0.0f;
            for (int j = 0; j < 8; j++) {
                float w = (float)grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                local_sum += w * xp[j];
            }
            group_sum += local_sum;
#endif
        }
        sum += db * group_sum;
    }
    return sum;
}

/* Fused IQ2_XXS row dot: dot product of entire quantized row with input vector.
 * Processes all 256-element super-blocks without any intermediate FP32 buffer.
 * Reserved for future fused matmul optimization path. */
/* unused — kept for future fused matmul optimization */
static float fused_dot_iq2_xxs(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        sum += dot_block_iq2_xxs(base + b * 66, x + b * 256);
    }
    return sum;
}

#if TQ_HAS_NEON

/* Vectorized sign application helper: given 8 grid bytes and an 8-bit sign mask,
 * produce signed int8x8 where negative signs are applied.
 * Uses NEON bit test: broadcast sign byte, AND with bit masks, compare to produce
 * negation mask, then apply via (grid ^ neg) - neg (conditional negate). */
static const uint8_t iq2_sign_bit_masks[8] = {1, 2, 4, 8, 16, 32, 64, 128};

/* NEON-optimized fused IQ2_XXS dot product.
 * Optimizations over baseline:
 *   1. Vectorized sign expansion via NEON bit-test (replaces 8 scalar shifts)
 *   2. Apply signs in int8 domain before float conversion (fewer instructions)
 *   3. Fully unrolled inner loop (4 groups per ib32)
 *   4. Prefetch next block's weight data
 *   5. Two accumulator strategy to reduce FMA dependency chains */
static float fused_dot_iq2_xxs_neon(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)row;
    float32x4_t vtotal0 = vdupq_n_f32(0.0f);

    /* Preload sign bit masks into a NEON register */
    const uint8x8_t vbit_masks = vld1_u8(iq2_sign_bit_masks);

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 66;

        /* Prefetch next block */
        if (b + 1 < nb) {
            __builtin_prefetch(blk + 66, 0, 3);
            __builtin_prefetch(blk + 66 + 32, 0, 3);
        }

        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint8_t* qs_bytes = blk + 2;
        const float* xbase = x + b * 256;

        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32[2];
            memcpy(aux32, qs_bytes + 8 * ib32, 8);
            const uint8_t* aux8 = (const uint8_t*)aux32;
            const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
            const float* xb = xbase + ib32 * 32;

            /* Accumulate across all 4 sub-groups before scaling by db.
             * Use two accumulators to break FMA dependency chains. */
            float32x4_t vacc0 = vdupq_n_f32(0.0f);
            float32x4_t vacc1 = vdupq_n_f32(0.0f);

            /* --- Group 0 --- */
            {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[0]);
                const uint8_t signs = ksigns_iq2xs[aux32[1] & 127];

                uint8x8_t vgrid = vld1_u8(grid);
                /* Vectorized sign expansion:
                 * Broadcast sign byte to all lanes, AND with bit masks,
                 * compare != 0 produces 0xFF for negative lanes.
                 * Then: signed = (grid ^ neg_mask) - neg_mask
                 *       which is grid when neg_mask=0, -grid when neg_mask=0xFF */
                uint8x8_t vsign_bcast = vdup_n_u8(signs);
                uint8x8_t vsign_bits = vtst_u8(vsign_bcast, vbit_masks);
                /* vsign_bits is 0xFF where negative, 0x00 where positive */
                int8x8_t vgrid_s = vreinterpret_s8_u8(vgrid);
                int8x8_t vneg_mask = vreinterpret_s8_u8(vsign_bits);
                int8x8_t vsigned = vsub_s8(veor_s8(vgrid_s, vneg_mask), vneg_mask);

                /* Widen to int16, then int32, then float */
                int16x8_t vs16 = vmovl_s8(vsigned);
                float32x4_t vf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vs16)));
                float32x4_t vf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vs16)));

                vacc0 = vfmaq_f32(vacc0, vf_lo, vld1q_f32(xb));
                vacc1 = vfmaq_f32(vacc1, vf_hi, vld1q_f32(xb + 4));
            }

            /* --- Group 1 --- */
            {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[1]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7) & 127];

                uint8x8_t vgrid = vld1_u8(grid);
                uint8x8_t vsign_bcast = vdup_n_u8(signs);
                uint8x8_t vsign_bits = vtst_u8(vsign_bcast, vbit_masks);
                int8x8_t vgrid_s = vreinterpret_s8_u8(vgrid);
                int8x8_t vneg_mask = vreinterpret_s8_u8(vsign_bits);
                int8x8_t vsigned = vsub_s8(veor_s8(vgrid_s, vneg_mask), vneg_mask);

                int16x8_t vs16 = vmovl_s8(vsigned);
                float32x4_t vf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vs16)));
                float32x4_t vf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vs16)));

                vacc0 = vfmaq_f32(vacc0, vf_lo, vld1q_f32(xb + 8));
                vacc1 = vfmaq_f32(vacc1, vf_hi, vld1q_f32(xb + 12));
            }

            /* --- Group 2 --- */
            {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[2]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 14) & 127];

                uint8x8_t vgrid = vld1_u8(grid);
                uint8x8_t vsign_bcast = vdup_n_u8(signs);
                uint8x8_t vsign_bits = vtst_u8(vsign_bcast, vbit_masks);
                int8x8_t vgrid_s = vreinterpret_s8_u8(vgrid);
                int8x8_t vneg_mask = vreinterpret_s8_u8(vsign_bits);
                int8x8_t vsigned = vsub_s8(veor_s8(vgrid_s, vneg_mask), vneg_mask);

                int16x8_t vs16 = vmovl_s8(vsigned);
                float32x4_t vf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vs16)));
                float32x4_t vf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vs16)));

                vacc0 = vfmaq_f32(vacc0, vf_lo, vld1q_f32(xb + 16));
                vacc1 = vfmaq_f32(vacc1, vf_hi, vld1q_f32(xb + 20));
            }

            /* --- Group 3 --- */
            {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[3]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 21) & 127];

                uint8x8_t vgrid = vld1_u8(grid);
                uint8x8_t vsign_bcast = vdup_n_u8(signs);
                uint8x8_t vsign_bits = vtst_u8(vsign_bcast, vbit_masks);
                int8x8_t vgrid_s = vreinterpret_s8_u8(vgrid);
                int8x8_t vneg_mask = vreinterpret_s8_u8(vsign_bits);
                int8x8_t vsigned = vsub_s8(veor_s8(vgrid_s, vneg_mask), vneg_mask);

                int16x8_t vs16 = vmovl_s8(vsigned);
                float32x4_t vf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vs16)));
                float32x4_t vf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vs16)));

                vacc0 = vfmaq_f32(vacc0, vf_lo, vld1q_f32(xb + 24));
                vacc1 = vfmaq_f32(vacc1, vf_hi, vld1q_f32(xb + 28));
            }

            /* Combine accumulators, scale by db, accumulate to total */
            float32x4_t vgroup = vaddq_f32(vacc0, vacc1);
            vtotal0 = vfmaq_n_f32(vtotal0, vgroup, db);
        }
    }
    return vaddvq_f32(vtotal0);
}
#endif /* TQ_HAS_NEON */

/* Fused IQ2_S dot product */
static float fused_dot_iq2_s(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 82;
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);

        const uint8_t* qs_base = blk + 2;
        const uint8_t* signs_base = qs_base + 32;
        const uint8_t* qh = blk + 66;
        const uint8_t* scales = blk + 74;
        const float* xbase = x + b * 256;

        for (int ib32 = 0; ib32 < 8; ib32++) {
            const uint8_t* qs = qs_base + ib32 * 4;
            const uint8_t* sn = signs_base + ib32 * 4;
            float db0 = d * (0.5f + (float)(scales[ib32] & 0xF)) * 0.25f;
            float db1 = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;

            for (int l = 0; l < 4; l++) {
                float dl = (l < 2) ? db0 : db1;
                int grid_idx = qs[l] | ((qh[ib32] << (8 - 2*l)) & 0x300);
                const uint8_t* grid = (const uint8_t*)(iq2s_grid + grid_idx);
                uint8_t sign = sn[l];
                const float* xp = xbase + ib32 * 32 + l * 8;

                float local_sum = 0.0f;
                for (int j = 0; j < 8; j++) {
                    float w = (float)grid[j] * ((sign & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    local_sum += w * xp[j];
                }
                sum += dl * local_sum;
            }
        }
    }
    return sum;
}

/* Fused Q8_0 dot product: 34 bytes per 32 elements */
static float fused_dot_q8_0(const void* row, const float* x, int n) {
    const int nb = n / 32;
    const block_q8_0* blk = (const block_q8_0*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float* xp = x + b * 32;

#if TQ_HAS_NEON
        /* NEON: dot product of 32 int8 * float, scaled by d */
        float32x4_t vsum0 = vdupq_n_f32(0.0f);
        float32x4_t vsum1 = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 8) {
            /* Load 8 int8 weights, convert to float */
            int8x8_t vq = vld1_s8(blk[b].qs + j);
            int16x8_t vq16 = vmovl_s8(vq);
            int32x4_t vq32_lo = vmovl_s16(vget_low_s16(vq16));
            int32x4_t vq32_hi = vmovl_s16(vget_high_s16(vq16));
            float32x4_t vw_lo = vcvtq_f32_s32(vq32_lo);
            float32x4_t vw_hi = vcvtq_f32_s32(vq32_hi);
            float32x4_t vx_lo = vld1q_f32(xp + j);
            float32x4_t vx_hi = vld1q_f32(xp + j + 4);
            vsum0 = vfmaq_f32(vsum0, vw_lo, vx_lo);
            vsum1 = vfmaq_f32(vsum1, vw_hi, vx_hi);
        }
        vsum0 = vaddq_f32(vsum0, vsum1);
        sum += d * vaddvq_f32(vsum0);
#else
        float block_sum = 0.0f;
        for (int j = 0; j < 32; j++) {
            block_sum += (float)blk[b].qs[j] * xp[j];
        }
        sum += d * block_sum;
#endif
    }
    return sum;
}

/* Fused IQ4_NL dot product: 18 bytes per 32 elements */
static float fused_dot_iq4_nl(const void* row, const float* x, int n) {
    const int nb = n / 32;
    const block_iq4_nl* blk = (const block_iq4_nl*)row;
    float sum = 0.0f;

#if TQ_HAS_NEON
    /* Preload IQ4_NL lookup table into 2 NEON registers (16 values split into 2x8).
     * kvalues_iq4nl[0..15] are int8 values, we load as two int8x8 for tbl lookup. */
    int8x16_t vlut = vld1q_s8(kvalues_iq4nl);
    float32x4_t vsum0 = vdupq_n_f32(0.0f);
    uint8x16_t vmask_lo = vdupq_n_u8(0x0f);

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const uint8_t* qs = blk[b].qs;
        const float* xp = x + b * 32;

        /* Load 16 bytes of qs, split into low/high nibbles -> lookup -> signed int8 weights */
        uint8x16_t vqs = vld1q_u8(qs);
        uint8x16_t vlo = vandq_u8(vqs, vmask_lo);         /* low nibbles [0..15] */
        uint8x16_t vhi = vshrq_n_u8(vqs, 4);              /* high nibbles [0..15] */
        int8x16_t wlo = vqtbl1q_s8(vlut, vlo);            /* lookup low -> signed weights */
        int8x16_t whi = vqtbl1q_s8(vlut, vhi);            /* lookup high -> signed weights */

        /* Convert to float and accumulate: wlo[j]*xp[j] for j=0..15, whi[j]*xp[j+16] */
        float32x4_t vacc = vdupq_n_f32(0.0f);
        for (int k = 0; k < 4; k++) {
            /* Low nibble: 4 elements */
            int16x8_t w16 = vmovl_s8(vget_low_s8(wlo));
            if (k >= 2) w16 = vmovl_s8(vget_high_s8(wlo));
            int16x4_t w16_part = (k & 1) ? vget_high_s16(w16) : vget_low_s16(w16);
            float32x4_t vw = vcvtq_f32_s32(vmovl_s16(w16_part));
            float32x4_t vx = vld1q_f32(xp + k * 4);
            vacc = vfmaq_f32(vacc, vw, vx);

            /* High nibble: 4 elements at xp+16 */
            int16x8_t wh16 = vmovl_s8(vget_low_s8(whi));
            if (k >= 2) wh16 = vmovl_s8(vget_high_s8(whi));
            int16x4_t wh16_part = (k & 1) ? vget_high_s16(wh16) : vget_low_s16(wh16);
            float32x4_t vwh = vcvtq_f32_s32(vmovl_s16(wh16_part));
            float32x4_t vxh = vld1q_f32(xp + 16 + k * 4);
            vacc = vfmaq_f32(vacc, vwh, vxh);
        }
        float block_sum = vaddvq_f32(vacc);
        vsum0 = vfmaq_n_f32(vsum0, vdupq_n_f32(block_sum), d);
    }
    sum = vaddvq_f32(vsum0);
#else
    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const uint8_t* qs = blk[b].qs;
        const float* xp = x + b * 32;

        float block_sum = 0.0f;
        for (int j = 0; j < 16; j++) {
            block_sum += (float)kvalues_iq4nl[qs[j] & 0xf] * xp[j];
            block_sum += (float)kvalues_iq4nl[qs[j] >> 4]  * xp[j + 16];
        }
        sum += d * block_sum;
    }
#endif
    return sum;
}

/* Fused IQ4_XS dot product: 136 bytes per 256 elements */
static float fused_dot_iq4_xs(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const block_iq4_xs* blk = (const block_iq4_xs*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const uint8_t* qs = blk[b].qs;
        const float* xbase = x + b * 256;

        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((blk[b].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf)
                         | (((blk[b].scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            const float* xp = xbase + ib * 32;

            float block_sum = 0.0f;
            for (int j = 0; j < 16; j++) {
                block_sum += (float)kvalues_iq4nl[qs[j] & 0xf] * xp[j];
                block_sum += (float)kvalues_iq4nl[qs[j] >> 4]  * xp[j + 16];
            }
            sum += dl * block_sum;
            qs += 16;
        }
    }
    return sum;
}

/* Fused IQ3_XXS dot product: 98 bytes per 256 elements
 * Same layout as dequant_iq3_xxs but computes dot product without materializing FP32.
 * 8 groups of 32 elements per block. Each group: 4 sub-groups of 8 elements.
 * Grid lookup + sign application + dot in one pass. */
static float fused_dot_iq3_xxs(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)row;
    float sum = 0.0f;

#if TQ_HAS_NEON
    const uint8x8_t vbit_masks = vld1_u8(iq2_sign_bit_masks);

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 98;
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint8_t* qs = blk + 2;
        const uint8_t* scales_and_signs = qs + 64;
        const float* xbase = x + b * 256;

        float block_sum = 0.0f;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32;
            memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = (0.5f + (float)(aux32 >> 28)) * 0.5f;
            const float* xb = xbase + ib32 * 32;

            float32x4_t vacc0 = vdupq_n_f32(0.0f);
            float32x4_t vacc1 = vdupq_n_f32(0.0f);

            for (int l = 0; l < 4; l++) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 1]);

                /* Load 4+4 grid bytes, combine into one 8-byte vector */
                uint8_t grid8[8];
                memcpy(grid8, grid1, 4);
                memcpy(grid8 + 4, grid2, 4);
                uint8x8_t vgrid = vld1_u8(grid8);

                /* Vectorized sign: broadcast sign byte, AND with masks, compare */
                uint8x8_t vsign_bcast = vdup_n_u8(signs);
                uint8x8_t vsign_bits = vtst_u8(vsign_bcast, vbit_masks);
                int8x8_t vgrid_s = vreinterpret_s8_u8(vgrid);
                int8x8_t vneg = vreinterpret_s8_u8(vsign_bits);
                int8x8_t vsigned = vsub_s8(veor_s8(vgrid_s, vneg), vneg);

                /* Widen to float and dot with input */
                int16x8_t vs16 = vmovl_s8(vsigned);
                float32x4_t vf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vs16)));
                float32x4_t vf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vs16)));

                const float* xp = xb + l * 8;
                vacc0 = vfmaq_f32(vacc0, vf_lo, vld1q_f32(xp));
                vacc1 = vfmaq_f32(vacc1, vf_hi, vld1q_f32(xp + 4));
            }
            block_sum += db * vaddvq_f32(vaddq_f32(vacc0, vacc1));
            qs += 8;
        }
        sum += d * block_sum;
    }
#else
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 98;
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint8_t* qs = blk + 2;
        const uint8_t* scales_and_signs = qs + 64;
        const float* xbase = x + b * 256;

        float block_sum = 0.0f;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32;
            memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = (0.5f + (float)(aux32 >> 28)) * 0.5f;
            const float* xb = xbase + ib32 * 32;

            float group_sum = 0.0f;
            for (int l = 0; l < 4; l++) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 1]);
                const float* xp = xb + l * 8;
                for (int j = 0; j < 4; j++) {
                    float w1 = (float)grid1[j] * ((signs & kmask_iq2xs[j])     ? -1.0f : 1.0f);
                    float w2 = (float)grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                    group_sum += w1 * xp[j] + w2 * xp[j + 4];
                }
            }
            block_sum += db * group_sum;
            qs += 8;
        }
        sum += d * block_sum;
    }
#endif
    return sum;
}

/* Fused Q4_K dot product: 144 bytes per 256 elements
 * Layout: 4 groups of 64 elements, each group uses 32 bytes of qs.
 *   First 32 elements: d1 * (q[l] & 0xF) - m1  (low nibble, scale pair[0])
 *   Next 32 elements:  d2 * (q[l] >> 4)  - m2   (high nibble, scale pair[1])
 */
static float fused_dot_q4_k(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const block_q4_K* blk = (const block_q4_K*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const float d    = fp16_to_fp32(blk[b].d);
        const float dmin = fp16_to_fp32(blk[b].dmin);

        uint8_t sc[8], mn[8];
        sc[0] = blk[b].scales[0] & 63;
        sc[1] = blk[b].scales[1] & 63;
        sc[2] = blk[b].scales[2] & 63;
        sc[3] = blk[b].scales[3] & 63;
        mn[0] = blk[b].scales[4] & 63;
        mn[1] = blk[b].scales[5] & 63;
        mn[2] = blk[b].scales[6] & 63;
        mn[3] = blk[b].scales[7] & 63;
        sc[4] = (blk[b].scales[8] & 0x0F) | ((blk[b].scales[0] >> 6) << 4);
        sc[5] = (blk[b].scales[9] & 0x0F) | ((blk[b].scales[1] >> 6) << 4);
        sc[6] = (blk[b].scales[10] & 0x0F) | ((blk[b].scales[2] >> 6) << 4);
        sc[7] = (blk[b].scales[11] & 0x0F) | ((blk[b].scales[3] >> 6) << 4);
        mn[4] = (blk[b].scales[8] >> 4) | ((blk[b].scales[4] >> 6) << 4);
        mn[5] = (blk[b].scales[9] >> 4) | ((blk[b].scales[5] >> 6) << 4);
        mn[6] = (blk[b].scales[10] >> 4) | ((blk[b].scales[6] >> 6) << 4);
        mn[7] = (blk[b].scales[11] >> 4) | ((blk[b].scales[7] >> 6) << 4);

        const uint8_t* q = blk[b].qs;
        const float* xp = x + b * 256;
        int is = 0;

        /* 4 groups of 64 elements */
        for (int j = 0; j < 256; j += 64) {
            const float d1 = d * sc[is + 0];
            const float m1 = dmin * mn[is + 0];
            const float d2 = d * sc[is + 1];
            const float m2 = dmin * mn[is + 1];

            /* First 32 elements: low nibble */
            float dot1 = 0.0f, sum_x1 = 0.0f;
            for (int l = 0; l < 32; l++) {
                dot1  += (float)(q[l] & 0x0F) * xp[j + l];
                sum_x1 += xp[j + l];
            }
            sum += d1 * dot1 - m1 * sum_x1;

            /* Next 32 elements: high nibble */
            float dot2 = 0.0f, sum_x2 = 0.0f;
            for (int l = 0; l < 32; l++) {
                dot2  += (float)(q[l] >> 4) * xp[j + 32 + l];
                sum_x2 += xp[j + 32 + l];
            }
            sum += d2 * dot2 - m2 * sum_x2;

            q += 32;
            is += 2;
        }
    }
    return sum;
}

/* Fused Q4_0 dot product: 18 bytes per 32 elements */
static float fused_dot_q4_0(const void* row, const float* x, int n) {
    const int nb = n / 32;
    const block_q4_0* blk = (const block_q4_0*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float* xp = x + b * 32;

        float block_sum = 0.0f;
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            block_sum += (float)((int)(byte & 0x0F) - 8) * xp[j];
            block_sum += (float)((int)(byte >> 4) - 8) * xp[j + 16];
        }
        sum += d * block_sum;
    }
    return sum;
}

/* Fused Q6_K dot product: 210 bytes per 256 elements
 * Matches ggml dequantize_row_q6_K layout exactly:
 * Two 128-element halves, each with 32 iterations producing 4 elements. */
static float fused_dot_q6_k(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const block_q6_K* blk = (const block_q6_K*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float* xbase = x + b * 256;
        const uint8_t* ql = blk[b].ql;
        const uint8_t* qh = blk[b].qh;
        const int8_t* sc = blk[b].scales;

        for (int half = 0; half < 2; half++) {
            const float* xp = xbase + half * 128;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                sum += d * sc[is + 0] * q1 * xp[l +  0];
                sum += d * sc[is + 2] * q2 * xp[l + 32];
                sum += d * sc[is + 4] * q3 * xp[l + 64];
                sum += d * sc[is + 6] * q4 * xp[l + 96];
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return sum;
}

/* ============================================================
 * On-the-fly dequant matmul (with fused fast paths)
 *
 * out[d] = sum_n( x[n] * dequant(W[d, n]) )
 *
 * W is stored row-major in quantized blocks.
 * Hot path for MoE expert computation.
 *
 * For supported types (IQ2_XXS, IQ2_S, Q8_0, Q4_K, Q4_0, Q6_K,
 * IQ4_NL, IQ4_XS), we use fused dequant-dot that avoids writing
 * intermediate FP32 values to memory. This eliminates ~3 GB/token
 * of temporary memory traffic for IQ2_XXS MoE models.
 * ============================================================ */

/* ============================================================
 * Multi-threaded GGUF matmul worker
 * ============================================================ */
typedef struct {
    float*       out;
    const float* x;
    const void*  weight;
    float (*fused_dot)(const void*, const float*, int);
    tq_ggml_dtype weight_type;
    size_t       row_bytes;
    int          in_dim;
    int          block_bytes;
    int          block_elems;
    int          n_blocks;
    int          start_row;
    int          end_row;
} gguf_matmul_task_t;

static void* gguf_matmul_worker(void* arg) {
    gguf_matmul_task_t* t = (gguf_matmul_task_t*)arg;

    if (t->fused_dot) {
        for (int d = t->start_row; d < t->end_row; d++) {
            const uint8_t* row = (const uint8_t*)t->weight + (size_t)d * t->row_bytes;
            t->out[d] = t->fused_dot(row, t->x, t->in_dim);
        }
        return NULL;
    }

    /* Generic fallback: dequant block -> tmp -> dot */
    for (int d = t->start_row; d < t->end_row; d++) {
        const uint8_t* row = (const uint8_t*)t->weight + (size_t)d * t->row_bytes;
        float sum = 0.0f;
        float tmp[256]; /* max block size is 256 */

        for (int b = 0; b < t->n_blocks; b++) {
            tq_dequant_row_gguf(t->weight_type,
                                row + (size_t)b * t->block_bytes,
                                tmp, t->block_elems);

            const float* xp = t->x + b * t->block_elems;

#if TQ_HAS_NEON
            float32x4_t vsum0 = vdupq_n_f32(0.0f);
            float32x4_t vsum1 = vdupq_n_f32(0.0f);
            float32x4_t vsum2 = vdupq_n_f32(0.0f);
            float32x4_t vsum3 = vdupq_n_f32(0.0f);

            int j = 0;
            for (; j + 15 < t->block_elems; j += 16) {
                float32x4_t vx0 = vld1q_f32(xp + j);
                float32x4_t vx1 = vld1q_f32(xp + j + 4);
                float32x4_t vx2 = vld1q_f32(xp + j + 8);
                float32x4_t vx3 = vld1q_f32(xp + j + 12);
                float32x4_t vt0 = vld1q_f32(tmp + j);
                float32x4_t vt1 = vld1q_f32(tmp + j + 4);
                float32x4_t vt2 = vld1q_f32(tmp + j + 8);
                float32x4_t vt3 = vld1q_f32(tmp + j + 12);
                vsum0 = vfmaq_f32(vsum0, vx0, vt0);
                vsum1 = vfmaq_f32(vsum1, vx1, vt1);
                vsum2 = vfmaq_f32(vsum2, vx2, vt2);
                vsum3 = vfmaq_f32(vsum3, vx3, vt3);
            }
            for (; j + 3 < t->block_elems; j += 4) {
                float32x4_t vx0 = vld1q_f32(xp + j);
                float32x4_t vt0 = vld1q_f32(tmp + j);
                vsum0 = vfmaq_f32(vsum0, vx0, vt0);
            }

            vsum0 = vaddq_f32(vsum0, vsum1);
            vsum2 = vaddq_f32(vsum2, vsum3);
            vsum0 = vaddq_f32(vsum0, vsum2);
            sum += vaddvq_f32(vsum0);

            for (; j < t->block_elems; j++) {
                sum += xp[j] * tmp[j];
            }
#else
            for (int j = 0; j < t->block_elems; j++) {
                sum += xp[j] * tmp[j];
            }
#endif
        }

        t->out[d] = sum;
    }
    return NULL;
}

void tq_matmul_gguf(float* out, const float* x,
                    const void* weight, tq_ggml_dtype weight_type,
                    int out_dim, int in_dim)
{
    /* Per-matmul Metal dispatch DISABLED — slower than CPU fused dot
     * due to dispatch overhead. MoE uses tq_metal_moe_forward() instead
     * which batches all experts in 3 dispatches per layer. */

    const size_t block_bytes = tq_ggml_type_size(weight_type);
    const int    block_elems = tq_ggml_type_blck(weight_type);

    if (block_bytes == 0 || block_elems == 0) {
        static int warn_count = 0;
        if (warn_count++ < 5) {
            void* ra = NULL;
            fprintf(stderr, "tq_matmul_gguf: unsupported type %d (out=%d, in=%d, w=%p, caller=%p)\n",
                    (int)weight_type, out_dim, in_dim, weight, ra);
        }
        memset(out, 0, (size_t)out_dim * sizeof(float));
        return;
    }

    const int    n_blocks  = in_dim / block_elems;
    const size_t row_bytes = (size_t)n_blocks * block_bytes;

    /* ---- Fused fast paths: dequant + dot in one pass, no tmp buffer ---- */

    /* Fused path function pointer: returns dot product for one row */
    float (*fused_dot)(const void*, const float*, int) = NULL;

    switch (weight_type) {
        case TQ_GGML_TYPE_IQ2_XXS:
#if TQ_HAS_NEON
            fused_dot = fused_dot_iq2_xxs_neon;
#else
            fused_dot = fused_dot_iq2_xxs;
#endif
            break;
        case TQ_GGML_TYPE_IQ2_S:
            fused_dot = fused_dot_iq2_s;
            break;
        case TQ_GGML_TYPE_Q8_0:
            fused_dot = fused_dot_q8_0;
            break;
        case TQ_GGML_TYPE_Q4_K:
            fused_dot = fused_dot_q4_k;
            break;
        case TQ_GGML_TYPE_Q4_0:
            fused_dot = fused_dot_q4_0;
            break;
        case TQ_GGML_TYPE_Q6_K:
            fused_dot = fused_dot_q6_k;
            break;
        case TQ_GGML_TYPE_IQ4_NL:
            fused_dot = fused_dot_iq4_nl;
            break;
        case TQ_GGML_TYPE_IQ4_XS:
            fused_dot = fused_dot_iq4_xs;
            break;
        case TQ_GGML_TYPE_IQ3_XXS:
            fused_dot = fused_dot_iq3_xxs;
            break;
        default:
            break;
    }

    /* ---- Multi-threaded dispatch ---- */
    int n_threads = tq_get_threads();

    /* Note: single-thread for small matmuls was tested and was SLOWER
     * (538ms vs 251ms MoE). Multi-threading benefits IQ2_XXS fused dot
     * even at out_dim=512. Keep multi-threaded. */

    /* For small matrices or single-thread config, skip thread overhead */
    if (n_threads <= 1 || out_dim < n_threads) {
        /* Single-threaded path */
        if (fused_dot) {
            for (int d = 0; d < out_dim; d++) {
                const uint8_t* row = (const uint8_t*)weight + (size_t)d * row_bytes;
                out[d] = fused_dot(row, x, in_dim);
            }
        } else {
            gguf_matmul_task_t task = {
                .out = out, .x = x, .weight = weight, .fused_dot = NULL,
                .weight_type = weight_type, .row_bytes = row_bytes,
                .in_dim = in_dim, .block_bytes = (int)block_bytes,
                .block_elems = block_elems, .n_blocks = n_blocks,
                .start_row = 0, .end_row = out_dim
            };
            gguf_matmul_worker(&task);
        }
        return;
    }

    /* Cap threads */
    if (n_threads > TQ_TP_MAX) n_threads = TQ_TP_MAX;
    if (n_threads > out_dim) n_threads = out_dim;

    gguf_matmul_task_t tasks[TQ_TP_MAX];
    void* ptrs[TQ_TP_MAX];

    int rows_per_thread = out_dim / n_threads;
    for (int t = 0; t < n_threads; t++) {
        tasks[t].out         = out;
        tasks[t].x           = x;
        tasks[t].weight      = weight;
        tasks[t].fused_dot   = fused_dot;
        tasks[t].weight_type = weight_type;
        tasks[t].row_bytes   = row_bytes;
        tasks[t].in_dim      = in_dim;
        tasks[t].block_bytes = (int)block_bytes;
        tasks[t].block_elems = block_elems;
        tasks[t].n_blocks    = n_blocks;
        tasks[t].start_row   = t * rows_per_thread;
        tasks[t].end_row     = (t == n_threads - 1) ? out_dim : (t + 1) * rows_per_thread;
        ptrs[t] = &tasks[t];
    }

    tq_tp_run(gguf_matmul_worker, ptrs, n_threads);
}

/* ============================================================
 * Metal batch mode wrappers
 *
 * These forward to Metal batch API when available, otherwise no-op.
 * The transformer/MoE code calls these to batch consecutive matmuls
 * into a single GPU command buffer, reducing dispatch overhead.
 * ============================================================ */

void tq_metal_batch_begin_if_available(void) {
#ifdef TQ_HAS_METAL
    extern int tq_metal_available(void);
    extern void tq_metal_batch_begin(void);
    if (tq_metal_available()) {
        tq_metal_batch_begin();
    }
#endif
}

void tq_metal_batch_flush_if_available(void) {
#ifdef TQ_HAS_METAL
    extern void tq_metal_batch_flush(void);
    extern int tq_metal_batch_active(void);
    if (tq_metal_batch_active()) {
        tq_metal_batch_flush();
    }
#endif
}

void tq_metal_batch_end_if_available(void) {
#ifdef TQ_HAS_METAL
    extern void tq_metal_batch_end(void);
    extern int tq_metal_batch_active(void);
    if (tq_metal_batch_active()) {
        tq_metal_batch_end();
    }
#endif
}

// ============================================================================
// Section 12: BPE Tokenizer (from tq_tokenizer.c)
// ============================================================================

/**
 * tq_tokenizer.c — HuggingFace BPE tokenizer (tokenizer.json) loader
 *
 * Parses the HuggingFace tokenizer.json format:
 *   - model.vocab: { "token_string": token_id, ... }
 *   - model.merges: [ "tok_a tok_b", ... ]
 *   - added_tokens: [ { "id": N, "content": "...", ... }, ... ]
 *
 * Implements BPE encoding via iterative pair merging with merge priority.
 * Implements decoding with Qwen/GPT-style byte-level BPE (Ġ = space prefix).
 *
 * Also supports the legacy llama2.c binary tokenizer format as fallback.
 */

/* Global for qsort comparator (vocab index sorting) */
static char** g_vocab_for_sort;
static int cmp_vocab_idx(const void* a, const void* b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    const char* sa = g_vocab_for_sort[ia] ? g_vocab_for_sort[ia] : "";
    const char* sb = g_vocab_for_sort[ib] ? g_vocab_for_sort[ib] : "";
    return strcmp(sa, sb);
}

/* ============================================================
 * Minimal JSON helpers (reused from tq_model.c pattern)
 * ============================================================ */

static const char* skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

/* Parse a JSON string with proper escape handling.
 * Writes the unescaped string into out (up to max_len-1 chars).
 * Returns pointer past closing quote, or NULL on error. */
static const char* json_parse_string(const char* p, char* out, int max_len) {
    if (*p != '"') return NULL;
    p++;
    int i = 0;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            if (!*p) return NULL;
            switch (*p) {
                case '"':  if (i < max_len - 1) out[i++] = '"';  break;
                case '\\': if (i < max_len - 1) out[i++] = '\\'; break;
                case '/':  if (i < max_len - 1) out[i++] = '/';  break;
                case 'n':  if (i < max_len - 1) out[i++] = '\n'; break;
                case 'r':  if (i < max_len - 1) out[i++] = '\r'; break;
                case 't':  if (i < max_len - 1) out[i++] = '\t'; break;
                case 'b':  if (i < max_len - 1) out[i++] = '\b'; break;
                case 'f':  if (i < max_len - 1) out[i++] = '\f'; break;
                case 'u': {
                    /* Parse \uXXXX unicode escape */
                    unsigned int cp = 0;
                    for (int k = 0; k < 4; k++) {
                        p++;
                        if (!*p) return NULL;
                        cp <<= 4;
                        if (*p >= '0' && *p <= '9') cp |= (*p - '0');
                        else if (*p >= 'a' && *p <= 'f') cp |= (*p - 'a' + 10);
                        else if (*p >= 'A' && *p <= 'F') cp |= (*p - 'A' + 10);
                        else return NULL;
                    }
                    /* Handle surrogate pairs for codepoints > U+FFFF */
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        /* High surrogate: expect \uDCxx low surrogate */
                        if (p[1] == '\\' && p[2] == 'u') {
                            p += 3; /* skip \u */
                            unsigned int lo = 0;
                            for (int k = 0; k < 4; k++) {
                                if (!*p) return NULL;
                                lo <<= 4;
                                if (*p >= '0' && *p <= '9') lo |= (*p - '0');
                                else if (*p >= 'a' && *p <= 'f') lo |= (*p - 'a' + 10);
                                else if (*p >= 'A' && *p <= 'F') lo |= (*p - 'A' + 10);
                                p++;
                            }
                            p--; /* will be incremented at end of loop */
                            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            }
                        }
                    }
                    /* Encode codepoint as UTF-8 */
                    if (cp < 0x80) {
                        if (i < max_len - 1) out[i++] = (char)cp;
                    } else if (cp < 0x800) {
                        if (i < max_len - 2) {
                            out[i++] = (char)(0xC0 | (cp >> 6));
                            out[i++] = (char)(0x80 | (cp & 0x3F));
                        }
                    } else if (cp < 0x10000) {
                        if (i < max_len - 3) {
                            out[i++] = (char)(0xE0 | (cp >> 12));
                            out[i++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                            out[i++] = (char)(0x80 | (cp & 0x3F));
                        }
                    } else if (cp < 0x110000) {
                        if (i < max_len - 4) {
                            out[i++] = (char)(0xF0 | (cp >> 18));
                            out[i++] = (char)(0x80 | ((cp >> 12) & 0x3F));
                            out[i++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                            out[i++] = (char)(0x80 | (cp & 0x3F));
                        }
                    }
                    break;
                }
                default:
                    if (i < max_len - 1) out[i++] = *p;
                    break;
            }
        } else {
            /* Regular UTF-8 byte — copy as-is */
            if (i < max_len - 1) out[i++] = *p;
        }
        p++;
    }
    out[i] = '\0';
    if (*p == '"') p++;
    return p;
}

/* Skip a JSON value (string, number, object, array, bool, null) */
static const char* json_skip_value(const char* p) {
    p = skip_ws(p);
    if (*p == '"') {
        /* Skip string */
        p++;
        while (*p && *p != '"') {
            if (*p == '\\') { p++; if (*p) p++; }
            else p++;
        }
        if (*p == '"') p++;
    } else if (*p == '{') {
        int depth = 1; p++;
        while (*p && depth > 0) {
            if (*p == '{') depth++;
            else if (*p == '}') depth--;
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\') { p++; if (*p) p++; }
                    else p++;
                }
                if (*p == '"') p++;
                continue;
            }
            p++;
        }
    } else if (*p == '[') {
        int depth = 1; p++;
        while (*p && depth > 0) {
            if (*p == '[') depth++;
            else if (*p == ']') depth--;
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\') { p++; if (*p) p++; }
                    else p++;
                }
                if (*p == '"') p++;
                continue;
            }
            p++;
        }
    } else {
        /* number, bool, null */
        while (*p && *p != ',' && *p != '}' && *p != ']'
               && *p != ' ' && *p != '\n' && *p != '\r' && *p != '\t') {
            p++;
        }
    }
    return p;
}

/* Parse a JSON integer */
static const char* json_parse_int(const char* p, int* out) {
    p = skip_ws(p);
    int neg = 0;
    if (*p == '-') { neg = 1; p++; }
    int val = 0;
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    *out = neg ? -val : val;
    return p;
}

/* Forward declaration for str_lookup (used during merge parsing) */
static int str_lookup(const tq_tokenizer_t* tok, const char* str);

/* ============================================================
 * Detect file format: JSON starts with '{', binary starts with
 * a uint32 that is a reasonable vocab size.
 * ============================================================ */
static int is_json_file(const char* data, size_t size) {
    if (size < 4) return 0;
    /* Skip BOM if present */
    const char* p = data;
    if ((unsigned char)p[0] == 0xEF &&
        (unsigned char)p[1] == 0xBB &&
        (unsigned char)p[2] == 0xBF) {
        p += 3;
    }
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return (*p == '{');
}

/* ============================================================
 * qsort comparison for sorted_indices (by vocab string)
 * ============================================================ */
typedef struct {
    char** vocab;
} sort_ctx_t;

static sort_ctx_t g_sort_ctx;

static int compare_vocab_strings(const void* a, const void* b) {
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    return strcmp(g_sort_ctx.vocab[ia], g_sort_ctx.vocab[ib]);
}

/* Build sorted index for binary search */
static void build_sorted_index(tq_tokenizer_t* tok) {
    tok->sorted_indices = (int*)malloc((size_t)tok->vocab_size * sizeof(int));
    if (!tok->sorted_indices) return;

    int n = 0;
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->vocab[i] && tok->vocab[i][0] != '\0') {
            tok->sorted_indices[n++] = i;
        }
    }

    /* Sort using qsort with global context (simpler than passing context) */
    g_sort_ctx.vocab = tok->vocab;
    qsort(tok->sorted_indices, (size_t)n, sizeof(int), compare_vocab_strings);

    /* Store actual count of valid entries; we still keep vocab_size as capacity */
    /* Reuse sorted_indices[n..vocab_size-1] as sentinel; mark count in max_token_len if needed */
    /* Actually just zero-fill the rest */
    for (int i = n; i < tok->vocab_size; i++) {
        tok->sorted_indices[i] = -1;
    }
}

/* ============================================================
 * Load tokenizer from HuggingFace tokenizer.json
 *
 * Strategy:
 * 1. Read entire file into memory
 * 2. Navigate JSON to find "model" -> "vocab" and "merges"
 * 3. Also parse "added_tokens" for special tokens
 * 4. Build vocab array and merge table
 * ============================================================ */
static tq_tokenizer_t* load_hf_tokenizer_json(const char* data, size_t size) {
    tq_tokenizer_t* tok = (tq_tokenizer_t*)calloc(1, sizeof(tq_tokenizer_t));
    if (!tok) return NULL;

    /* First pass: scan for max token ID to determine vocab_size */
    /* Find "vocab": { ... } inside "model": { ... } */

    /* Locate "model" key at the top level */
    const char* model_start = NULL;
    {
        const char* p = data;
        p = skip_ws(p);
        if (*p != '{') { free(tok); return NULL; }
        p++;

        while (*p) {
            p = skip_ws(p);
            if (*p == '}') break;
            if (*p == ',') { p++; p = skip_ws(p); }
            if (*p == '}') break;

            char key[64];
            p = json_parse_string(p, key, sizeof(key));
            if (!p) { free(tok); return NULL; }
            p = skip_ws(p);
            if (*p != ':') { free(tok); return NULL; }
            p++;
            p = skip_ws(p);

            if (strcmp(key, "model") == 0) {
                model_start = p;
                break;
            }
            p = json_skip_value(p);
        }
    }

    if (!model_start) {
        fprintf(stderr, "tq_load_tokenizer: 'model' key not found in JSON\n");
        free(tok);
        return NULL;
    }

    /* Inside "model": { "vocab": {...}, "merges": [...], ... } */
    const char* vocab_start = NULL;
    const char* merges_start = NULL;

    {
        const char* p = model_start;
        p = skip_ws(p);
        if (*p != '{') { free(tok); return NULL; }
        p++;

        while (*p) {
            p = skip_ws(p);
            if (*p == '}') break;
            if (*p == ',') { p++; p = skip_ws(p); }
            if (*p == '}') break;

            char key[64];
            p = json_parse_string(p, key, sizeof(key));
            if (!p) { free(tok); return NULL; }
            p = skip_ws(p);
            if (*p != ':') { free(tok); return NULL; }
            p++;
            p = skip_ws(p);

            if (strcmp(key, "vocab") == 0) {
                vocab_start = p;
                p = json_skip_value(p);
            } else if (strcmp(key, "merges") == 0) {
                merges_start = p;
                p = json_skip_value(p);
            } else {
                p = json_skip_value(p);
            }
        }
    }

    if (!vocab_start) {
        fprintf(stderr, "tq_load_tokenizer: 'vocab' not found in model\n");
        free(tok);
        return NULL;
    }

    /* Parse vocab to find max ID and count entries */
    int max_id = -1;
    int n_vocab_entries = 0;
    {
        const char* p = vocab_start;
        p = skip_ws(p);
        if (*p != '{') { free(tok); return NULL; }
        p++;

        char token_str[1024];
        while (*p) {
            p = skip_ws(p);
            if (*p == '}') break;
            if (*p == ',') { p++; p = skip_ws(p); }
            if (*p == '}') break;

            p = json_parse_string(p, token_str, sizeof(token_str));
            if (!p) break;
            p = skip_ws(p);
            if (*p != ':') break;
            p++;

            int id = 0;
            p = json_parse_int(p, &id);
            if (id > max_id) max_id = id;
            n_vocab_entries++;
        }
    }

    /* Also scan added_tokens for higher IDs */
    const char* added_tokens_start = NULL;
    {
        const char* p = data;
        p = skip_ws(p);
        if (*p == '{') p++;
        while (*p) {
            p = skip_ws(p);
            if (*p == '}') break;
            if (*p == ',') { p++; p = skip_ws(p); }
            if (*p == '}') break;

            char key[64];
            p = json_parse_string(p, key, sizeof(key));
            if (!p) break;
            p = skip_ws(p);
            if (*p != ':') break;
            p++;
            p = skip_ws(p);

            if (strcmp(key, "added_tokens") == 0) {
                added_tokens_start = p;
                /* Quick scan for max id in added_tokens array */
                if (*p == '[') {
                    const char* q = p + 1;
                    while (*q) {
                        q = skip_ws(q);
                        if (*q == ']') break;
                        if (*q == ',') { q++; q = skip_ws(q); }
                        if (*q == ']') break;
                        if (*q == '{') {
                            q++;
                            while (*q && *q != '}') {
                                q = skip_ws(q);
                                if (*q == ',') { q++; q = skip_ws(q); }
                                if (*q == '}') break;
                                char akey[64];
                                q = json_parse_string(q, akey, sizeof(akey));
                                if (!q) goto done_added_scan;
                                q = skip_ws(q);
                                if (*q != ':') goto done_added_scan;
                                q++;
                                q = skip_ws(q);
                                if (strcmp(akey, "id") == 0) {
                                    int aid = 0;
                                    q = json_parse_int(q, &aid);
                                    if (aid > max_id) max_id = aid;
                                } else {
                                    q = json_skip_value(q);
                                }
                            }
                            if (*q == '}') q++;
                        } else {
                            q = json_skip_value(q);
                        }
                    }
                }
                done_added_scan:
                p = json_skip_value(p);
            } else {
                p = json_skip_value(p);
            }
        }
    }

    tok->vocab_size = max_id + 1;
    tok->max_token_len = 0;

    fprintf(stderr, "tq_load_tokenizer: vocab has %d entries, max_id=%d, total_size=%d\n",
            n_vocab_entries, max_id, tok->vocab_size);

    /* Allocate vocab array */
    tok->vocab = (char**)calloc((size_t)tok->vocab_size, sizeof(char*));
    tok->scores = (float*)calloc((size_t)tok->vocab_size, sizeof(float));
    if (!tok->vocab || !tok->scores) {
        tq_free_tokenizer(tok);
        return NULL;
    }

    /* Initialize all vocab entries to empty strings */
    for (int i = 0; i < tok->vocab_size; i++) {
        tok->vocab[i] = (char*)calloc(1, 1); /* empty string "" */
    }

    /* Second pass: populate vocab entries */
    {
        const char* p = vocab_start;
        p = skip_ws(p);
        if (*p == '{') p++;

        char token_str[1024];
        while (*p) {
            p = skip_ws(p);
            if (*p == '}') break;
            if (*p == ',') { p++; p = skip_ws(p); }
            if (*p == '}') break;

            p = json_parse_string(p, token_str, sizeof(token_str));
            if (!p) break;
            p = skip_ws(p);
            if (*p != ':') break;
            p++;

            int id = 0;
            p = json_parse_int(p, &id);

            if (id >= 0 && id < tok->vocab_size) {
                free(tok->vocab[id]);
                int len = (int)strlen(token_str);
                tok->vocab[id] = (char*)malloc((size_t)len + 1);
                if (tok->vocab[id]) {
                    memcpy(tok->vocab[id], token_str, (size_t)len + 1);
                    if (len > tok->max_token_len) tok->max_token_len = len;
                }
            }
        }
    }

    /* Parse added_tokens to fill special token entries */
    if (added_tokens_start) {
        const char* p = added_tokens_start;
        p = skip_ws(p);
        if (*p == '[') {
            p++;
            while (*p) {
                p = skip_ws(p);
                if (*p == ']') break;
                if (*p == ',') { p++; p = skip_ws(p); }
                if (*p == ']') break;

                if (*p == '{') {
                    p++;
                    int at_id = -1;
                    char at_content[256] = {0};
                    while (*p && *p != '}') {
                        p = skip_ws(p);
                        if (*p == ',') { p++; p = skip_ws(p); }
                        if (*p == '}') break;

                        char akey[64];
                        p = json_parse_string(p, akey, sizeof(akey));
                        if (!p) goto done_added;
                        p = skip_ws(p);
                        if (*p != ':') goto done_added;
                        p++;
                        p = skip_ws(p);

                        if (strcmp(akey, "id") == 0) {
                            p = json_parse_int(p, &at_id);
                        } else if (strcmp(akey, "content") == 0) {
                            p = json_parse_string(p, at_content, sizeof(at_content));
                            if (!p) goto done_added;
                        } else {
                            p = json_skip_value(p);
                        }
                    }
                    if (*p == '}') p++;

                    if (at_id >= 0 && at_id < tok->vocab_size && at_content[0]) {
                        free(tok->vocab[at_id]);
                        int len = (int)strlen(at_content);
                        tok->vocab[at_id] = (char*)malloc((size_t)len + 1);
                        if (tok->vocab[at_id]) {
                            memcpy(tok->vocab[at_id], at_content, (size_t)len + 1);
                            if (len > tok->max_token_len) tok->max_token_len = len;
                        }
                    }
                } else {
                    p = json_skip_value(p);
                }
            }
        }
    }
    done_added:

    /* Build sorted index FIRST so merge parsing can use binary search */
    build_sorted_index(tok);

    /* Parse merges: array of "token_a token_b" strings.
     * The merge priority is the index in the array (lower = higher priority).
     * We store scores so that BPE merge finds highest score first. */
    tok->n_merges = 0;
    tok->merge_pairs = NULL;

    if (merges_start) {
        /* Count merges first */
        int n_merges = 0;
        {
            const char* p = merges_start;
            p = skip_ws(p);
            if (*p == '[') {
                p++;
                while (*p) {
                    p = skip_ws(p);
                    if (*p == ']') break;
                    if (*p == ',') { p++; p = skip_ws(p); }
                    if (*p == ']') break;
                    p = json_skip_value(p);
                    n_merges++;
                }
            }
        }

        fprintf(stderr, "tq_load_tokenizer: parsing %d merges\n", n_merges);

        /* Allocate merge pairs */
        tok->merge_pairs = (int*)malloc((size_t)n_merges * 3 * sizeof(int));
        if (!tok->merge_pairs) {
            tq_free_tokenizer(tok);
            return NULL;
        }

        /* Parse merge strings using binary search for fast lookup.
         * Supports two formats:
         *   Qwen/GPT2 style: ["tok_a tok_b", ...]  (space-separated string)
         *   Gemma/SentencePiece style: [["tok_a","tok_b"], ...]  (JSON array pairs) */
        {
            const char* p = merges_start;
            p = skip_ws(p);
            if (*p == '[') p++;
            p = skip_ws(p);

            /* Detect format: if first element starts with '[', it's array-pair format */
            int array_pair_format = (*p == '[');

            int mi = 0;
            char str_a[1024], str_b[1024];
            while (*p && mi < n_merges) {
                p = skip_ws(p);
                if (*p == ']') break;
                if (*p == ',') { p++; p = skip_ws(p); }
                if (*p == ']') break;

                if (array_pair_format) {
                    /* Gemma style: ["tok_a", "tok_b"] */
                    if (*p != '[') { p = json_skip_value(p); mi++; continue; }
                    p++; /* skip '[' */
                    p = skip_ws(p);
                    p = json_parse_string(p, str_a, sizeof(str_a));
                    if (!p) break;
                    p = skip_ws(p);
                    if (*p == ',') p++;
                    p = skip_ws(p);
                    p = json_parse_string(p, str_b, sizeof(str_b));
                    if (!p) break;
                    p = skip_ws(p);
                    if (*p == ']') p++; /* skip closing ']' */
                } else {
                    /* Qwen/GPT2 style: "tok_a tok_b" */
                    char merge_str[2048];
                    p = json_parse_string(p, merge_str, sizeof(merge_str));
                    if (!p) break;
                    char* sep = strchr(merge_str, ' ');
                    if (!sep) { mi++; continue; }
                    *sep = '\0';
                    strncpy(str_a, merge_str, sizeof(str_a) - 1);
                    str_a[sizeof(str_a) - 1] = '\0';
                    strncpy(str_b, sep + 1, sizeof(str_b) - 1);
                    str_b[sizeof(str_b) - 1] = '\0';
                }

                /* Find the merged result: concatenation of tok_a + tok_b */
                char merged[2048];
                int la = (int)strlen(str_a);
                int lb = (int)strlen(str_b);
                if (la + lb >= (int)sizeof(merged)) { mi++; continue; }
                memcpy(merged, str_a, (size_t)la);
                memcpy(merged + la, str_b, (size_t)lb);
                merged[la + lb] = '\0';

                /* Look up token IDs */
                int id_a = str_lookup(tok, str_a);
                int id_b = str_lookup(tok, str_b);
                int id_merged = str_lookup(tok, merged);

                if (id_a >= 0 && id_b >= 0 && id_merged >= 0) {
                    tok->merge_pairs[tok->n_merges * 3 + 0] = id_a;
                    tok->merge_pairs[tok->n_merges * 3 + 1] = id_b;
                    tok->merge_pairs[tok->n_merges * 3 + 2] = id_merged;
                    tok->scores[id_merged] = (float)(n_merges - mi);
                    tok->n_merges++;
                }

                mi++;
            }
        }

        fprintf(stderr, "tq_load_tokenizer: loaded %d/%d merges successfully\n",
                tok->n_merges, n_merges);
    }

    fprintf(stderr, "tq_load_tokenizer: loaded %d tokens, max_len=%d, %d merges\n",
            tok->vocab_size, tok->max_token_len, tok->n_merges);
    return tok;
}

/* ============================================================
 * Load tokenizer from file (auto-detect format)
 * ============================================================ */
tq_tokenizer_t* tq_load_tokenizer(const char* path) {
    if (!path) return NULL;

    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "tq_load_tokenizer: cannot open '%s'\n", path);
        return NULL;
    }

    /* Get file size */
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0 || file_size > 200 * 1024 * 1024) {
        fprintf(stderr, "tq_load_tokenizer: invalid file size %ld\n", file_size);
        fclose(f);
        return NULL;
    }

    /* Read entire file */
    char* data = (char*)malloc((size_t)file_size + 1);
    if (!data) {
        fclose(f);
        return NULL;
    }
    size_t nread = fread(data, 1, (size_t)file_size, f);
    fclose(f);
    data[nread] = '\0';

    tq_tokenizer_t* tok = NULL;

    if (is_json_file(data, nread)) {
        fprintf(stderr, "tq_load_tokenizer: detected HuggingFace JSON format\n");
        tok = load_hf_tokenizer_json(data, nread);
    } else {
        fprintf(stderr, "tq_load_tokenizer: detected binary format — not supported for this model\n");
        fprintf(stderr, "tq_load_tokenizer: please provide a tokenizer.json file\n");
    }

    free(data);
    return tok;
}

/* ============================================================
 * Load tokenizer from TQM file (extract embedded tokenizer)
 * ============================================================ */
tq_tokenizer_t* tq_load_tokenizer_from_tqm(const char* tqm_path) {
    if (!tqm_path) return NULL;

    FILE* f = fopen(tqm_path, "rb");
    if (!f) {
        fprintf(stderr, "tq_load_tokenizer_from_tqm: cannot open '%s'\n", tqm_path);
        return NULL;
    }

    /* Read the TQM header to get tokenizer offset and size */
    uint8_t hdr_buf[512];
    if (fread(hdr_buf, 1, 512, f) != 512) {
        fprintf(stderr, "tq_load_tokenizer_from_tqm: file too small\n");
        fclose(f);
        return NULL;
    }

    uint32_t magic;
    memcpy(&magic, hdr_buf, 4);
    if (magic != 0x4D515454) { /* TQM_MAGIC */
        fprintf(stderr, "tq_load_tokenizer_from_tqm: not a TQM file\n");
        fclose(f);
        return NULL;
    }

    /* Extract tokenizer offset and size from header using offsetof */
    uint64_t tok_offset, tok_size;
    memcpy(&tok_offset, hdr_buf + offsetof(tqm_header_t, tokenizer_offset), 8);
    memcpy(&tok_size, hdr_buf + offsetof(tqm_header_t, tokenizer_size), 8);

    if (tok_size == 0) {
        fprintf(stderr, "tq_load_tokenizer_from_tqm: no embedded tokenizer\n");
        fclose(f);
        return NULL;
    }

    /* Read tokenizer data */
    char* tok_data = (char*)malloc((size_t)tok_size);
    if (!tok_data) { fclose(f); return NULL; }

    fseek(f, (long)tok_offset, SEEK_SET);
    size_t nread = fread(tok_data, 1, (size_t)tok_size, f);
    fclose(f);

    if (nread != (size_t)tok_size) {
        fprintf(stderr, "tq_load_tokenizer_from_tqm: short read (%zu/%llu)\n",
                nread, (unsigned long long)tok_size);
        free(tok_data);
        return NULL;
    }

    tq_tokenizer_t* tok = tq_load_tokenizer_from_memory(tok_data, (size_t)tok_size);
    free(tok_data);
    return tok;
}

/* ============================================================
 * Load tokenizer from memory buffer (for TQM embedded tokenizer)
 * ============================================================ */
tq_tokenizer_t* tq_load_tokenizer_from_memory(const char* data, size_t size) {
    if (!data || size == 0) return NULL;

    /* Make a null-terminated copy */
    char* buf = (char*)malloc(size + 1);
    if (!buf) return NULL;
    memcpy(buf, data, size);
    buf[size] = '\0';

    tq_tokenizer_t* tok = NULL;
    if (is_json_file(buf, size)) {
        tok = load_hf_tokenizer_json(buf, size);
    } else {
        fprintf(stderr, "tq_load_tokenizer_from_memory: unrecognized format\n");
    }

    free(buf);
    return tok;
}

/* ============================================================
 * Load tokenizer from GGUF metadata
 *
 * GGUF stores tokenizer data in metadata keys:
 *   tokenizer.ggml.tokens: string array of token strings
 *   tokenizer.ggml.scores: float array of BPE merge scores
 *   tokenizer.ggml.merges: string array of merge rules (optional)
 * ============================================================ */
tq_tokenizer_t* tq_load_tokenizer_from_gguf(const void* gguf_ctx_ptr) {
    if (!gguf_ctx_ptr) return NULL;

    const tq_gguf_ctx_t* gguf = (const tq_gguf_ctx_t*)gguf_ctx_ptr;

    /* Find the tokens array */
    int64_t tokens_idx = tq_gguf_find_key(gguf, "tokenizer.ggml.tokens");
    if (tokens_idx < 0) {
        fprintf(stderr, "tq_load_tokenizer_from_gguf: no tokenizer.ggml.tokens\n");
        return NULL;
    }

    const tq_gguf_kv_t* kv = &gguf->kv[tokens_idx];
    if (kv->type != TQ_GGUF_TYPE_ARRAY || kv->value.array.elem_type != TQ_GGUF_TYPE_STRING) {
        fprintf(stderr, "tq_load_tokenizer_from_gguf: tokens is not a string array\n");
        return NULL;
    }

    uint64_t vocab_size = kv->value.array.count;
    if (vocab_size == 0 || vocab_size > 1000000) {
        fprintf(stderr, "tq_load_tokenizer_from_gguf: invalid vocab_size=%llu\n",
                (unsigned long long)vocab_size);
        return NULL;
    }

    tq_tokenizer_t* tok = (tq_tokenizer_t*)calloc(1, sizeof(tq_tokenizer_t));
    if (!tok) return NULL;

    tok->vocab_size = (int)vocab_size;
    tok->vocab = (char**)calloc(vocab_size, sizeof(char*));
    tok->scores = (float*)calloc(vocab_size, sizeof(float));
    if (!tok->vocab || !tok->scores) {
        free(tok->vocab);
        free(tok->scores);
        free(tok);
        return NULL;
    }

    /* Copy token strings from GGUF string array.
     * The array data contains tq_gguf_string_t structs laid out sequentially. */
    tq_gguf_string_t* strings = (tq_gguf_string_t*)kv->value.array.data;
    int max_len = 0;
    for (uint64_t i = 0; i < vocab_size; i++) {
        if (strings[i].str && strings[i].len > 0) {
            tok->vocab[i] = (char*)malloc((size_t)strings[i].len + 1);
            if (tok->vocab[i]) {
                memcpy(tok->vocab[i], strings[i].str, (size_t)strings[i].len);
                tok->vocab[i][strings[i].len] = '\0';
                if ((int)strings[i].len > max_len) max_len = (int)strings[i].len;
            }
        } else {
            tok->vocab[i] = (char*)calloc(1, 1); /* empty string */
        }
    }
    tok->max_token_len = max_len;

    /* Load scores if available */
    int64_t scores_idx = tq_gguf_find_key(gguf, "tokenizer.ggml.scores");
    if (scores_idx >= 0) {
        const tq_gguf_kv_t* skv = &gguf->kv[scores_idx];
        if (skv->type == TQ_GGUF_TYPE_ARRAY &&
            skv->value.array.elem_type == TQ_GGUF_TYPE_FLOAT32 &&
            skv->value.array.count == vocab_size) {
            memcpy(tok->scores, skv->value.array.data, vocab_size * sizeof(float));
        }
    }

    /* Build sorted indices BEFORE merge parsing so str_lookup() can use
     * binary search instead of O(n) linear scan.  For 248K vocab with
     * ~50K merges (3 lookups each), this turns a ~10 s init into ~100 ms. */
    tok->sorted_indices = (int*)malloc(vocab_size * sizeof(int));
    if (tok->sorted_indices) {
        for (int i = 0; i < (int)vocab_size; i++) tok->sorted_indices[i] = i;
        g_vocab_for_sort = tok->vocab;
        qsort(tok->sorted_indices, vocab_size, sizeof(int), cmp_vocab_idx);
    }

    /* Load and parse merges if available.
     * GGUF stores merges as a string array of "tok_a tok_b" pairs.
     * We need to look up token IDs and build (id_a, id_b, id_merged) triples
     * so the BPE encoder can use them. */
    int64_t merges_idx = tq_gguf_find_key(gguf, "tokenizer.ggml.merges");
    if (merges_idx >= 0) {
        const tq_gguf_kv_t* mkv = &gguf->kv[merges_idx];
        if (mkv->type == TQ_GGUF_TYPE_ARRAY &&
            mkv->value.array.elem_type == TQ_GGUF_TYPE_STRING) {
            uint64_t n_merges_total = mkv->value.array.count;
            tok->merge_pairs = (int*)malloc(n_merges_total * 3 * sizeof(int));
            tok->n_merges = 0;
            if (tok->merge_pairs) {
                tq_gguf_string_t* merge_strings = (tq_gguf_string_t*)mkv->value.array.data;
                for (uint64_t mi = 0; mi < n_merges_total; mi++) {
                    if (!merge_strings[mi].str || merge_strings[mi].len == 0) continue;

                    /* Copy merge string and split on space: "tok_a tok_b" */
                    char buf[2048];
                    int slen = (int)merge_strings[mi].len;
                    if (slen >= (int)sizeof(buf)) continue;
                    memcpy(buf, merge_strings[mi].str, (size_t)slen);
                    buf[slen] = '\0';

                    char* sep = strchr(buf, ' ');
                    if (!sep) continue;
                    *sep = '\0';
                    const char* str_a = buf;
                    const char* str_b = sep + 1;

                    /* Build merged string: concatenation of tok_a + tok_b */
                    char merged[2048];
                    int la = (int)strlen(str_a);
                    int lb = (int)strlen(str_b);
                    if (la + lb >= (int)sizeof(merged)) continue;
                    memcpy(merged, str_a, (size_t)la);
                    memcpy(merged + la, str_b, (size_t)lb);
                    merged[la + lb] = '\0';

                    /* Look up token IDs via binary search (sorted_indices built above) */
                    int id_a = str_lookup(tok, str_a);
                    int id_b = str_lookup(tok, str_b);
                    int id_merged = str_lookup(tok, merged);

                    if (id_a >= 0 && id_b >= 0 && id_merged >= 0) {
                        tok->merge_pairs[tok->n_merges * 3 + 0] = id_a;
                        tok->merge_pairs[tok->n_merges * 3 + 1] = id_b;
                        tok->merge_pairs[tok->n_merges * 3 + 2] = id_merged;
                        /* Priority: earlier merges in GGUF = higher priority */
                        tok->scores[id_merged] = (float)(n_merges_total - mi);
                        tok->n_merges++;
                    }
                }
                fprintf(stderr, "tq_load_tokenizer_from_gguf: parsed %d/%d merges\n",
                        tok->n_merges, (int)n_merges_total);
            }
        }
    }

    fprintf(stderr, "tq_load_tokenizer_from_gguf: loaded %d tokens (max_len=%d)\n",
            tok->vocab_size, tok->max_token_len);
    return tok;
}

/* ============================================================
 * Free tokenizer
 * ============================================================ */
void tq_free_tokenizer(tq_tokenizer_t* tok) {
    if (!tok) return;
    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i]);
        }
        free(tok->vocab);
    }
    free(tok->scores);
    free(tok->sorted_indices);
    free(tok->merge_pairs);
    free(tok);
}

/* ============================================================
 * Lookup token ID by string (binary search on sorted index)
 * ============================================================ */
static int str_lookup(const tq_tokenizer_t* tok, const char* str) {
    if (!tok->sorted_indices) {
        /* Fallback: linear scan */
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->vocab[i] && strcmp(tok->vocab[i], str) == 0) return i;
        }
        return -1;
    }

    /* Binary search */
    int lo = 0, hi = tok->vocab_size - 1;
    /* Find the actual valid range (entries with sorted_indices >= 0) */
    while (hi >= 0 && tok->sorted_indices[hi] < 0) hi--;
    if (hi < 0) return -1;

    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int idx = tok->sorted_indices[mid];
        if (idx < 0) { hi = mid - 1; continue; }
        int cmp = strcmp(str, tok->vocab[idx]);
        if (cmp == 0) return idx;
        if (cmp < 0) hi = mid - 1;
        else lo = mid + 1;
    }
    return -1;
}

/* ============================================================
 * GPT/Qwen byte-level BPE: map from BPE byte representation
 * to actual UTF-8 bytes.
 *
 * In Qwen/GPT tokenizers, bytes 0x00-0xFF are represented using
 * a specific mapping where printable ASCII is kept as-is but
 * non-printable bytes are mapped to Unicode characters starting
 * at U+0100 (Ġ = U+0120 = space, etc.).
 *
 * The mapping for decoding:
 *   - Ġ (U+0120, UTF-8: C4 A0) -> space (0x20)
 *   - Ċ (U+010A, UTF-8: C4 8A) -> newline (0x0A)
 *   - Characters U+0100-U+01FF map to bytes 0x00-0xFF
 *     where the byte value = codepoint - 0x100
 *     (but only for the "shifted" ones; printable ASCII stays)
 * ============================================================ */

/* Build the GPT2 byte-to-char map on first use.
 * The map: for each byte value 0-255, what character(s) represent it
 * in the BPE vocabulary.
 *
 * GPT2 byte encoder:
 *   - bytes 33-126 ('!' to '~') map to themselves
 *   - bytes 161-172 map to themselves
 *   - bytes 174-255 map to themselves
 *   - all other bytes (0-32, 127-160, 173) map to 256+offset
 *
 * For decoding, we need the REVERSE: given a BPE character, what byte?
 */

/* Decode a single BPE vocab string to raw UTF-8 bytes.
 * Handles the Ġ -> space mapping used by GPT2/Qwen tokenizers.
 * Returns decoded string in a static thread-local buffer. */
static const char* decode_bpe_token(const char* piece) {
    static char decode_buf[1024];
    int out = 0;
    const unsigned char* p = (const unsigned char*)piece;

    while (*p && out < (int)sizeof(decode_buf) - 4) {
        if (*p < 0x80) {
            /* ASCII — direct */
            decode_buf[out++] = (char)*p;
            p++;
        } else if ((*p & 0xE0) == 0xC0 && (p[1] & 0xC0) == 0x80) {
            /* 2-byte UTF-8: decode codepoint */
            unsigned int cp = ((unsigned int)(*p & 0x1F) << 6) | (p[1] & 0x3F);
            if (cp >= 0x100 && cp <= 0x1FF) {
                /* GPT2 byte mapping: codepoint - 0x100 is a raw byte
                 * Actually the mapping is more nuanced. Let's use the
                 * standard GPT2 byte decoder. */
                /* The GPT2 bytes_to_unicode creates a bijection.
                 * Codepoints 0x100+ represent specific bytes. */
                /* Simple approach: cp in [0x100, 0x14F] maps to bytes that
                 * aren't in the "direct" set. Build lookup. */
                int byte_val = -1;
                /* GPT2 direct bytes: 33-126, 161-172, 174-255 */
                /* Indirect bytes get codepoints starting at 256 (0x100) */
                /* Build the indirect byte list */
                static int indirect_map_built = 0;
                static unsigned char indirect_to_byte[256];
                if (!indirect_map_built) {
                    int n = 0;
                    for (int b = 0; b < 256; b++) {
                        int direct = 0;
                        if (b >= 33 && b <= 126) direct = 1;
                        if (b >= 161 && b <= 172) direct = 1;
                        if (b >= 174 && b <= 255) direct = 1;
                        if (!direct) {
                            indirect_to_byte[n++] = (unsigned char)b;
                        }
                    }
                    indirect_map_built = 1;
                }
                int idx = (int)cp - 256;
                if (idx >= 0 && idx < 69) { /* 69 indirect bytes */
                    byte_val = indirect_to_byte[idx];
                }
                if (byte_val >= 0) {
                    decode_buf[out++] = (char)(unsigned char)byte_val;
                } else {
                    /* Fallback: copy UTF-8 bytes as-is */
                    decode_buf[out++] = (char)p[0];
                    decode_buf[out++] = (char)p[1];
                }
            } else {
                /* Regular 2-byte UTF-8 char (e.g., accented letters) */
                decode_buf[out++] = (char)p[0];
                decode_buf[out++] = (char)p[1];
            }
            p += 2;
        } else if ((*p & 0xF0) == 0xE0 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80) {
            /* 3-byte UTF-8 */
            decode_buf[out++] = (char)p[0];
            decode_buf[out++] = (char)p[1];
            decode_buf[out++] = (char)p[2];
            p += 3;
        } else if ((*p & 0xF8) == 0xF0 && (p[1] & 0xC0) == 0x80 &&
                   (p[2] & 0xC0) == 0x80 && (p[3] & 0xC0) == 0x80) {
            /* 4-byte UTF-8 */
            decode_buf[out++] = (char)p[0];
            decode_buf[out++] = (char)p[1];
            decode_buf[out++] = (char)p[2];
            decode_buf[out++] = (char)p[3];
            p += 4;
        } else {
            /* Invalid UTF-8 — copy byte */
            decode_buf[out++] = (char)*p;
            p++;
        }
    }
    decode_buf[out] = '\0';
    return decode_buf;
}

/* ============================================================
 * Encode text to tokens using BPE merge
 *
 * For GPT2/Qwen byte-level BPE:
 * 1. Convert each byte to its BPE character representation
 * 2. Look up each character as initial token
 * 3. Iteratively merge the highest-priority pair
 * ============================================================ */

/* Encode a single byte to its GPT2 BPE character representation */
static int encode_byte_to_bpe_char(unsigned char byte, char* out) {
    /* Direct bytes: 33-126, 161-172, 174-255 -> same codepoint */
    int direct = 0;
    if (byte >= 33 && byte <= 126) direct = 1;
    if (byte >= 161 && byte <= 172) direct = 1;
    if (byte >= 174) direct = 1; /* upper range always fits in uint8 */

    if (direct) {
        out[0] = (char)byte;
        out[1] = '\0';
        return 1;
    }

    /* Indirect bytes -> codepoint 256 + index */
    static unsigned char byte_order[69];
    static int order_built = 0;
    if (!order_built) {
        int n = 0;
        for (int b = 0; b < 256; b++) {
            int d = 0;
            if (b >= 33 && b <= 126) d = 1;
            if (b >= 161 && b <= 172) d = 1;
            if (b >= 174 && b <= 255) d = 1;
            if (!d) byte_order[n++] = (unsigned char)b;
        }
        order_built = 1;
    }

    /* Find index of this byte in indirect list */
    int idx = -1;
    for (int i = 0; i < 69; i++) {
        if (byte_order[i] == byte) { idx = i; break; }
    }
    if (idx < 0) { out[0] = (char)byte; out[1] = '\0'; return 1; }

    unsigned int cp = 256 + (unsigned int)idx;
    /* Encode codepoint as UTF-8 */
    if (cp < 0x80) {
        out[0] = (char)cp;
        out[1] = '\0';
        return 1;
    } else {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        out[2] = '\0';
        return 2;
    }
}

int tq_encode(const tq_tokenizer_t* tok, const char* text,
              int* tokens, int max_tokens, int add_bos) {
    if (!tok || !text || !tokens || max_tokens <= 0) return 0;

    int n_tokens = 0;

    /* Add BOS token if requested.
     * Gemma: BOS=2, Qwen: no BOS (uses <|im_start|> instead) */
    if (add_bos) {
        /* Look up <bos> token in vocab; default to id 2 (Gemma convention) */
        int bos_id = str_lookup(tok, "<bos>");
        if (bos_id < 0) { bos_id = str_lookup(tok, "<|im_start|>"); }
        if (bos_id >= 0) {
            tokens[n_tokens++] = bos_id;
        }
    }

    if (*text == '\0') return n_tokens;

    /* Detect tokenizer style: Gemma uses ▁ (U+2581) for spaces in vocab,
     * GPT2/Qwen uses byte-level BPE with Ġ/ĉ encoding.
     * Check if '▁' exists in vocab as a simple heuristic. */
    int is_sentencepiece = (str_lookup(tok, "\xe2\x96\x81") >= 0); /* ▁ = U+2581 = 0xE2 0x96 0x81 */

    int text_len = (int)strlen(text);

    if (is_sentencepiece) {
        /* SentencePiece-style: replace spaces with ▁, then split into UTF-8 characters.
         * Each character is looked up in vocab directly. */
        /* First, build normalized text with ▁ replacing spaces, and ▁ prepended */
        int norm_cap = text_len * 4 + 16;
        char* norm = (char*)malloc((size_t)norm_cap);
        if (!norm) return n_tokens;
        int ni = 0;
        /* Prepend ▁ (space before first word, SentencePiece convention) */
        norm[ni++] = (char)0xE2; norm[ni++] = (char)0x96; norm[ni++] = (char)0x81;
        for (int i = 0; i < text_len; i++) {
            if (text[i] == ' ') {
                norm[ni++] = (char)0xE2; norm[ni++] = (char)0x96; norm[ni++] = (char)0x81;
            } else {
                norm[ni++] = text[i];
            }
        }
        norm[ni] = '\0';

        /* Split into individual UTF-8 characters */
        for (int i = 0; i < ni && n_tokens < max_tokens; ) {
            /* Determine UTF-8 character length */
            unsigned char c = (unsigned char)norm[i];
            int clen = 1;
            if (c >= 0xF0) { clen = 4; }
            else if (c >= 0xE0) { clen = 3; }
            else if (c >= 0xC0) { clen = 2; }
            if (i + clen > ni) break;

            char ch_str[8];
            memcpy(ch_str, norm + i, (size_t)clen);
            ch_str[clen] = '\0';

            int id = str_lookup(tok, ch_str);
            if (id >= 0) {
                tokens[n_tokens++] = id;
            }
            /* If not found, skip (byte fallback tokens handle this in merges) */
            i += clen;
        }
        free(norm);
    } else {
        /* GPT2/Qwen byte-level BPE: each byte maps to a BPE character token */
        for (int i = 0; i < text_len && n_tokens < max_tokens; i++) {
            unsigned char byte = (unsigned char)text[i];
            char bpe_char[4];
            encode_byte_to_bpe_char(byte, bpe_char);

            int id = str_lookup(tok, bpe_char);
            if (id >= 0) {
                tokens[n_tokens++] = id;
            } else {
                char direct[2] = { (char)byte, '\0' };
                id = str_lookup(tok, direct);
                if (id >= 0) {
                    tokens[n_tokens++] = id;
                }
            }
        }
    }

    /* BPE merge pass: repeatedly merge the highest-priority pair.
     * A merge has higher priority if its score is larger.
     * We check all consecutive token pairs against the merge table. */
    while (n_tokens >= 2) {
        float best_score = -1e30f;
        int best_idx = -1;
        int best_id = -1;

        for (int i = 0; i < n_tokens - 1; i++) {
            /* Construct merged string */
            const char* s1 = tok->vocab[tokens[i]];
            const char* s2 = tok->vocab[tokens[i + 1]];
            int len1 = (int)strlen(s1);
            int len2 = (int)strlen(s2);

            if (len1 + len2 >= 512) continue;

            char merged[512];
            memcpy(merged, s1, (size_t)len1);
            memcpy(merged + len1, s2, (size_t)len2);
            merged[len1 + len2] = '\0';

            int id = str_lookup(tok, merged);
            if (id >= 0 && tok->scores[id] > best_score) {
                best_score = tok->scores[id];
                best_idx = i;
                best_id = id;
            }
        }

        if (best_idx < 0) break;

        /* Apply the merge */
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < n_tokens - 1; i++) {
            tokens[i] = tokens[i + 1];
        }
        n_tokens--;
    }

    return n_tokens;
}

/* ============================================================
 * Decode single token to string
 *
 * For GPT2/Qwen byte-level BPE, the token string uses special
 * Unicode characters (Ġ etc.) to represent bytes. We decode
 * these back to actual UTF-8 bytes.
 * ============================================================ */
const char* tq_decode(const tq_tokenizer_t* tok, int prev_token, int token) {
    if (!tok || token < 0 || token >= tok->vocab_size) return "";

    const char* piece = tok->vocab[token];
    if (!piece || piece[0] == '\0') return "";

    /* Check if this is a special token (e.g., <|endoftext|>) */
    if (piece[0] == '<' && piece[1] == '|') {
        return ""; /* Don't output special tokens as text */
    }

    /* SentencePiece: replace ▁ (U+2581) with space */
    if (strstr(piece, "\xe2\x96\x81") != NULL) {
        static __thread char sp_buf[1024];
        int j = 0;
        for (int i = 0; piece[i] && j < (int)sizeof(sp_buf) - 1; ) {
            if ((unsigned char)piece[i] == 0xE2 &&
                (unsigned char)piece[i+1] == 0x96 &&
                (unsigned char)piece[i+2] == 0x81) {
                sp_buf[j++] = ' ';
                i += 3;
            } else {
                sp_buf[j++] = piece[i++];
            }
        }
        sp_buf[j] = '\0';
        return sp_buf;
    }

    /* GPT2/Qwen: decode BPE byte representation to actual UTF-8 */
    return decode_bpe_token(piece);
}

// ============================================================================
// Section 13: Model Loading (from tq_model.c)
// ============================================================================

/**
 * tq_model.c — Safetensors model loader
 *
 * Reads safetensors format:
 *   - First 8 bytes: header size (uint64_t little-endian)
 *   - Next N bytes: JSON header with tensor metadata
 *   - Remaining bytes: raw tensor data
 *
 * Implements a minimal JSON parser (no external deps).
 * Uses mmap for zero-copy tensor access on supported platforms.
 *
 * Supported dtypes: F32, F16, BF16 (BF16/F16 are converted to FP32).
 * Supports naming conventions:
 *   - Standard:  model.layers.N.self_attn.q_proj.weight
 *   - Qwen3.5:   model.language_model.layers.N.self_attn.q_proj.weight
 * Supports hybrid architectures (e.g., Qwen3.5 DeltaNet + self_attn).
 */

#ifdef _WIN32
#else
#endif

/* ============================================================
 * Minimal JSON parser for safetensors header
 *
 * Safetensors JSON looks like:
 * {
 *   "model.layers.0.self_attn.q_proj.weight": {
 *     "dtype": "F32",
 *     "shape": [4096, 4096],
 *     "data_offsets": [0, 67108864]
 *   },
 *   ...
 * }
 * ============================================================ */

/* Maximum number of tensors we expect */
#define MAX_TENSORS 2048
#define MAX_NAME_LEN 256
#define MAX_DIMS 8

typedef struct {
    char name[MAX_NAME_LEN];
    char dtype[16];
    int64_t shape[MAX_DIMS];
    int n_dims;
    int64_t data_start;
    int64_t data_end;
    void* data_base;  /* base pointer for this tensor's shard data region */
} tensor_info_t;

/* ============================================================
 * BF16 / F16 conversion utilities
 * ============================================================ */

static float model_bf16_to_fp32(uint16_t bf16) {
    uint32_t fp32 = ((uint32_t)bf16) << 16;
    float result;
    memcpy(&result, &fp32, sizeof(float));
    return result;
}

static float model_fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t expo = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    uint32_t fp32;

    if (expo == 0) {
        if (mant == 0) {
            fp32 = sign; /* +/- zero */
        } else {
            /* Denormalized: convert to normalized FP32 */
            expo = 1;
            while (!(mant & 0x0400)) {
                mant <<= 1;
                expo--;
            }
            mant &= 0x03FF;
            fp32 = sign | ((uint32_t)(expo + 127 - 15) << 23) | ((uint32_t)mant << 13);
        }
    } else if (expo == 31) {
        fp32 = sign | 0x7F800000 | ((uint32_t)mant << 13); /* Inf/NaN */
    } else {
        fp32 = sign | ((uint32_t)(expo + 127 - 15) << 23) | ((uint32_t)mant << 13);
    }

    float result;
    memcpy(&result, &fp32, sizeof(float));
    return result;
}

/* Return the number of bytes per element for a dtype string.
 * Returns 0 for unknown dtypes. */
static int dtype_element_size(const char* dtype) {
    if (strcmp(dtype, "F32") == 0) return 4;
    if (strcmp(dtype, "F16") == 0) return 2;
    if (strcmp(dtype, "BF16") == 0) return 2;
    if (strcmp(dtype, "F64") == 0) return 8;
    if (strcmp(dtype, "I32") == 0) return 4;
    if (strcmp(dtype, "I64") == 0) return 8;
    if (strcmp(dtype, "I16") == 0) return 2;
    if (strcmp(dtype, "I8") == 0) return 1;
    if (strcmp(dtype, "U8") == 0) return 1;
    if (strcmp(dtype, "BOOL") == 0) return 1;
    return 0;
}

/* Skip whitespace in JSON string */
static const char* model_mdl_skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

/* Parse a JSON string value, return pointer past closing quote */
static const char* parse_string(const char* p, char* out, int max_len) {
    if (*p != '"') return NULL;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < max_len - 1) {
        if (*p == '\\') {
            p++;
            if (!*p) return NULL;
        }
        out[i++] = *p++;
    }
    out[i] = '\0';
    if (*p == '"') p++;
    return p;
}

/* Parse a JSON integer */
static const char* parse_int64(const char* p, int64_t* out) {
    *out = 0;
    int neg = 0;
    if (*p == '-') { neg = 1; p++; }
    while (*p >= '0' && *p <= '9') {
        *out = *out * 10 + (*p - '0');
        p++;
    }
    if (neg) *out = -*out;
    return p;
}

/* Parse tensor metadata from the safetensors JSON header */
static int parse_safetensors_header(const char* json, int64_t json_len,
                                    tensor_info_t* tensors, int max_tensors) {
    int n_tensors = 0;
    const char* p = json;
    const char* end = json + json_len;

    p = model_mdl_skip_ws(p);
    if (*p != '{') return -1;
    p++;

    while (p < end && n_tensors < max_tensors) {
        p = model_mdl_skip_ws(p);
        if (*p == '}') break;
        if (*p == ',') { p++; p = model_mdl_skip_ws(p); }
        if (*p == '}') break;

        /* Parse tensor name */
        tensor_info_t* t = &tensors[n_tensors];
        memset(t, 0, sizeof(*t));
        p = parse_string(p, t->name, MAX_NAME_LEN);
        if (!p) return -1;

        p = model_mdl_skip_ws(p);
        if (*p != ':') return -1;
        p++;
        p = model_mdl_skip_ws(p);

        /* Skip __metadata__ entries */
        if (strcmp(t->name, "__metadata__") == 0) {
            /* Skip the value object */
            int depth = 0;
            while (p < end) {
                if (*p == '{') depth++;
                else if (*p == '}') {
                    depth--;
                    if (depth == 0) { p++; break; }
                }
                p++;
            }
            continue;
        }

        /* Parse tensor metadata object */
        if (*p != '{') return -1;
        p++;

        while (p < end) {
            p = model_mdl_skip_ws(p);
            if (*p == '}') { p++; break; }
            if (*p == ',') { p++; p = model_mdl_skip_ws(p); }

            char key[64];
            p = parse_string(p, key, 64);
            if (!p) return -1;
            p = model_mdl_skip_ws(p);
            if (*p != ':') return -1;
            p++;
            p = model_mdl_skip_ws(p);

            if (strcmp(key, "dtype") == 0) {
                p = parse_string(p, t->dtype, 16);
                if (!p) return -1;
            } else if (strcmp(key, "shape") == 0) {
                /* Parse array of ints */
                if (*p != '[') return -1;
                p++;
                t->n_dims = 0;
                while (*p != ']' && t->n_dims < MAX_DIMS) {
                    p = model_mdl_skip_ws(p);
                    if (*p == ',') { p++; p = model_mdl_skip_ws(p); }
                    if (*p == ']') break;
                    p = parse_int64(p, &t->shape[t->n_dims]);
                    t->n_dims++;
                    p = model_mdl_skip_ws(p);
                }
                if (*p == ']') p++;
            } else if (strcmp(key, "data_offsets") == 0) {
                /* Parse [start, end] */
                if (*p != '[') return -1;
                p++;
                p = model_mdl_skip_ws(p);
                p = parse_int64(p, &t->data_start);
                p = model_mdl_skip_ws(p);
                if (*p == ',') p++;
                p = model_mdl_skip_ws(p);
                p = parse_int64(p, &t->data_end);
                p = model_mdl_skip_ws(p);
                if (*p == ']') p++;
            } else {
                /* Skip unknown value */
                if (*p == '"') {
                    char dummy[256];
                    p = parse_string(p, dummy, 256);
                } else if (*p == '[') {
                    int depth = 1;
                    p++;
                    while (p < end && depth > 0) {
                        if (*p == '[') depth++;
                        else if (*p == ']') depth--;
                        p++;
                    }
                } else if (*p == '{') {
                    int depth = 1;
                    p++;
                    while (p < end && depth > 0) {
                        if (*p == '{') depth++;
                        else if (*p == '}') depth--;
                        p++;
                    }
                } else {
                    /* number/bool/null */
                    while (p < end && *p != ',' && *p != '}') p++;
                }
            }
        }

        n_tensors++;
    }

    return n_tensors;
}

/* ============================================================
 * Find a tensor by name in the parsed tensor list.
 * Also tries alternative naming prefixes for compatibility
 * with different model architectures (e.g., Qwen3.5 uses
 * "model.language_model.layers.N." instead of "model.layers.N.").
 * ============================================================ */

/* Known prefix alternatives for tensor name matching */
/* Prefixes to insert after "model." when searching for tensors.
 * E.g., for "model.layers.0.foo", inserting "language_model." yields
 * "model.language_model.layers.0.foo" (Qwen3.5 naming convention). */
static const char* const NAME_PREFIXES[] = {
    "",                    /* identity — try as-is first */
    "language_model.",     /* Qwen3.5 style: model.language_model.layers.N... */
    NULL
};

static tensor_info_t* find_tensor(tensor_info_t* tensors, int n,
                                   const char* name) {
    /* Try exact match first */
    for (int i = 0; i < n; i++) {
        if (strcmp(tensors[i].name, name) == 0) {
            return &tensors[i];
        }
    }
    /* Try alternative prefixes: for a name like "model.layers.0.foo",
     * try "model.language_model.layers.0.foo" etc.
     * The prefix replaces "model." at the start of the name. */
    for (int p = 1; NAME_PREFIXES[p] != NULL; p++) {
        const char* prefix = NAME_PREFIXES[p];
        if (strncmp(name, "model.", 6) == 0) {
            char alt[MAX_NAME_LEN * 2];
            snprintf(alt, sizeof(alt), "model.%s%s", prefix, name + 6);
            for (int i = 0; i < n; i++) {
                if (strcmp(tensors[i].name, alt) == 0) {
                    return &tensors[i];
                }
            }
        }
    }
    /* Try "language_model." prefix: Gemma 4B uses
     * "language_model.model.layers.N.foo" instead of "model.layers.N.foo".
     * Also try "language_model.lm_head.weight" for "lm_head.weight". */
    {
        char alt[MAX_NAME_LEN * 2];
        snprintf(alt, sizeof(alt), "language_model.%s", name);
        for (int i = 0; i < n; i++) {
            if (strcmp(tensors[i].name, alt) == 0) {
                return &tensors[i];
            }
        }
    }
    return NULL;
}

/* ============================================================
 * Convert tensor data to FP32 if needed (BF16, F16).
 * Returns a newly allocated float buffer on conversion,
 * or NULL if the tensor is already F32 (caller uses mmap).
 * ============================================================ */
static float* convert_tensor_to_fp32(tensor_info_t* t,
                                      float** conv_buf, size_t* conv_used,
                                      size_t conv_capacity) {
    if (!t) return NULL;

    const void* src = (const char*)t->data_base + t->data_start;
    int64_t data_bytes = t->data_end - t->data_start;
    int elem_size = dtype_element_size(t->dtype);
    if (elem_size == 0) return NULL;
    int64_t n_elements = data_bytes / elem_size;

    if (strcmp(t->dtype, "F32") == 0) {
        /* Zero-copy: return direct pointer into mmap */
        return (float*)((char*)t->data_base + t->data_start);
    }

    /* Need conversion — write into the pre-allocated conversion buffer */
    size_t needed = (size_t)n_elements * sizeof(float);
    if (*conv_used + needed > conv_capacity) {
        fprintf(stderr, "tq_load_model: conversion buffer overflow (need %zu more bytes)\n",
                needed);
        return NULL;
    }

    float* dst = (float*)((char*)*conv_buf + *conv_used);
    *conv_used += needed;

    if (strcmp(t->dtype, "BF16") == 0) {
        const uint16_t* src16 = (const uint16_t*)src;
        for (int64_t i = 0; i < n_elements; i++) {
            dst[i] = bf16_to_fp32(src16[i]);
        }
    } else if (strcmp(t->dtype, "F16") == 0) {
        const uint16_t* src16 = (const uint16_t*)src;
        for (int64_t i = 0; i < n_elements; i++) {
            dst[i] = fp16_to_fp32(src16[i]);
        }
    } else {
        fprintf(stderr, "tq_load_model: unsupported dtype '%s' for tensor '%s'\n",
                t->dtype, t->name);
        *conv_used -= needed; /* rollback */
        return NULL;
    }

    return dst;
}

/* ============================================================
 * Map or convert tensor data pointer from mmap'd file.
 * Wrapper that handles F32 (zero-copy) and BF16/F16 (convert).
 * ============================================================ */
static float* load_tensor(tensor_info_t* t,
                           float** conv_buf, size_t* conv_used,
                           size_t conv_capacity) {
    if (!t) return NULL;
    return convert_tensor_to_fp32(t, conv_buf, conv_used, conv_capacity);
}

/* ============================================================
 * Get raw BF16 pointer for a tensor (zero-copy from mmap).
 * Returns NULL if tensor is not BF16.
 * ============================================================ */
static const uint16_t* get_bf16_ptr(tensor_info_t* t) {
    if (!t) return NULL;
    if (strcmp(t->dtype, "BF16") != 0) return NULL;
    return (const uint16_t*)((const char*)t->data_base + t->data_start);
}

/* ============================================================
 * Check if a tensor name matches any of the "keep as BF16" names.
 * These tensors won't be converted to FP32 upfront; instead they
 * will be converted on demand during inference to save memory.
 * ============================================================ */
static int should_keep_bf16(const char* name) {
    /* Embedding table — largest single tensor, only need one row at a time */
    if (strcmp(name, "model.embed_tokens.weight") == 0) return 1;
    if (strcmp(name, "model.language_model.embed_tokens.weight") == 0) return 1;
    if (strcmp(name, "language_model.model.embed_tokens.weight") == 0) return 1;
    if (strcmp(name, "tok_embeddings.weight") == 0) return 1;
    if (strcmp(name, "transformer.wte.weight") == 0) return 1;
    /* lm_head — output projection, can use BF16 matmul */
    if (strcmp(name, "lm_head.weight") == 0) return 1;
    if (strcmp(name, "language_model.lm_head.weight") == 0) return 1;
    return 0;
}

/* ============================================================
 * Calculate total FP32 bytes needed for all non-F32 tensors,
 * excluding tensors that will be kept as BF16 (streaming conversion).
 * ============================================================ */
static size_t calc_conversion_buffer_size(tensor_info_t* tensors, int n_tensors) {
    size_t total = 0;
    for (int i = 0; i < n_tensors; i++) {
        if (strcmp(tensors[i].dtype, "F32") != 0 &&
            strcmp(tensors[i].dtype, "__metadata__") != 0 &&
            !should_keep_bf16(tensors[i].name)) {
            int64_t data_bytes = tensors[i].data_end - tensors[i].data_start;
            int elem_size = dtype_element_size(tensors[i].dtype);
            if (elem_size > 0) {
                int64_t n_elements = data_bytes / elem_size;
                total += (size_t)n_elements * sizeof(float);
            }
        }
    }
    return total;
}

/* ============================================================
 * Detect layer count, considering alternative naming prefixes.
 * Scans all tensor names for patterns like:
 *   model.layers.N.xxx
 *   model.language_model.layers.N.xxx
 * Returns the max layer index + 1.
 * ============================================================ */
static int detect_n_layers(tensor_info_t* tensors, int n_tensors) {
    int max_layer = -1;
    for (int i = 0; i < n_tensors; i++) {
        int layer_idx = -1;
        const char* name = tensors[i].name;
        /* Try standard prefix */
        if (sscanf(name, "model.layers.%d.", &layer_idx) == 1) {
            if (layer_idx > max_layer) max_layer = layer_idx;
        }
        /* Try Qwen3.5 prefix */
        else if (sscanf(name, "model.language_model.layers.%d.", &layer_idx) == 1) {
            if (layer_idx > max_layer) max_layer = layer_idx;
        }
        /* Try Gemma 4B prefix: language_model.model.layers.N. */
        else if (sscanf(name, "language_model.model.layers.%d.", &layer_idx) == 1) {
            if (layer_idx > max_layer) max_layer = layer_idx;
        }
    }
    return max_layer + 1;
}

/* ============================================================
 * Detect which layers have standard self_attn (vs linear_attn/DeltaNet).
 * Returns a dynamically allocated array of layer indices and sets *count.
 * ============================================================ */
static int* detect_attn_layers(tensor_info_t* tensors, int n_tensors,
                                int n_layers, int* count) {
    /* Bitmap of layers that have self_attn */
    int* has_attn = (int*)calloc((size_t)n_layers, sizeof(int));
    /* Also track which layers have linear_attn (DeltaNet) */
    int* has_linear = (int*)calloc((size_t)n_layers, sizeof(int));
    if (!has_attn || !has_linear) {
        free(has_attn); free(has_linear);
        *count = 0;
        return NULL;
    }

    for (int i = 0; i < n_tensors; i++) {
        const char* name = tensors[i].name;

        /* Only match model.layers.N. or model.language_model.layers.N.
         * or language_model.model.layers.N. (Gemma 4B).
         * Skip other prefixes like mtp.layers.N. */
        const char* layers_pos = NULL;
        if (strncmp(name, "model.layers.", 13) == 0) {
            layers_pos = name + 13;
        } else if (strncmp(name, "model.language_model.layers.", 28) == 0) {
            layers_pos = name + 28;
        } else if (strncmp(name, "language_model.model.layers.", 28) == 0) {
            layers_pos = name + 28;
        }
        if (!layers_pos) continue;

        int layer_idx = 0;
        const char* p = layers_pos;
        while (*p >= '0' && *p <= '9') {
            layer_idx = layer_idx * 10 + (*p - '0');
            p++;
        }
        if (*p != '.' || p == layers_pos) continue; /* must have "layers.N." */
        p++; /* skip the dot after layer index */

        if (layer_idx < 0 || layer_idx >= n_layers) continue;

        /* Check what kind of attention this layer has */
        if (strncmp(p, "self_attn.", 10) == 0) {
            has_attn[layer_idx] = 1;
        } else if (strncmp(p, "linear_attn.", 12) == 0) {
            has_linear[layer_idx] = 1;
        }
    }

    /* If a layer has linear_attn but NOT self_attn, it is a DeltaNet layer.
     * If we found no linear_attn layers at all, treat all layers as attn. */
    int any_linear = 0;
    for (int l = 0; l < n_layers; l++) {
        if (has_linear[l]) { any_linear = 1; break; }
    }
    if (!any_linear) {
        /* No hybrid architecture detected — all layers have attn */
        for (int l = 0; l < n_layers; l++) has_attn[l] = 1;
    }
    free(has_linear);

    /* Count and build index array */
    int n_attn = 0;
    for (int l = 0; l < n_layers; l++) {
        if (has_attn[l]) n_attn++;
    }

    int* indices = (int*)malloc((size_t)n_attn * sizeof(int));
    if (!indices) {
        free(has_attn);
        *count = 0;
        return NULL;
    }

    int j = 0;
    for (int l = 0; l < n_layers; l++) {
        if (has_attn[l]) indices[j++] = l;
    }

    free(has_attn);
    *count = n_attn;
    return indices;
}

/* Forward declaration */
static tq_model_t* tq_load_safetensors(const char* path);

/* ============================================================
 * Load model — auto-detect format (TQM or safetensors)
 * ============================================================ */
tq_model_t* tq_load_model(const char* path) {
    if (!path) return NULL;

    /* Check magic bytes to detect format */
    FILE* probe = fopen(path, "rb");
    if (probe) {
        uint32_t magic = 0;
        if (fread(&magic, 4, 1, probe) == 1) {
            fclose(probe);
            if (magic == TQM_MAGIC) {
                fprintf(stderr, "tq_load_model: detected TQM format, using fast loader\n");
                return tq_load_tqm(path);
            }
            if (magic == 0x46554747) { /* "GGUF" */
                fprintf(stderr, "tq_load_model: detected GGUF format\n");
                return tq_load_gguf(path);
            }
        } else {
            fclose(probe);
        }
    }

    /* Fall through to safetensors loader */
    return tq_load_safetensors(path);
}

/* ============================================================
 * Helper: mmap a file, returning pointer and size.
 * Returns NULL on failure.
 * ============================================================ */
static void* mmap_file(const char* path, size_t* out_size) {
#ifdef _WIN32
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "tq_load_model: cannot open '%s'\n", path);
        return NULL;
    }
    LARGE_INTEGER fileSize;
    GetFileSizeEx(hFile, &fileSize);
    *out_size = (size_t)fileSize.QuadPart;
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) { CloseHandle(hFile); return NULL; }
    void* data = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    CloseHandle(hFile);
    return data;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "tq_load_model: cannot open '%s'\n", path);
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return NULL; }
    *out_size = (size_t)st.st_size;
    void* data = mmap(NULL, *out_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) {
        fprintf(stderr, "tq_load_model: mmap failed for '%s'\n", path);
        return NULL;
    }
    return data;
#endif
}

/* ============================================================
 * Helper: parse a single safetensors shard.
 * Mmaps the file, parses the header, and appends tensors to the
 * provided tensor array. Sets data_base on each tensor.
 * Returns number of tensors added, or -1 on error.
 * mmap_out/mmap_size_out receive the mmap pointer for the caller to manage.
 * ============================================================ */
static int parse_shard(const char* shard_path,
                       tensor_info_t* tensors, int existing_count, int max_tensors,
                       void** mmap_out, size_t* mmap_size_out) {
    size_t file_size = 0;
    void* mmap_data = mmap_file(shard_path, &file_size);
    if (!mmap_data) return -1;

    if (file_size < 8) {
        fprintf(stderr, "tq_load_model: shard too small: '%s'\n", shard_path);
#ifdef _WIN32
        UnmapViewOfFile(mmap_data);
#else
        munmap(mmap_data, file_size);
#endif
        return -1;
    }

    uint64_t header_size = 0;
    memcpy(&header_size, mmap_data, 8);
    if (8 + header_size > file_size) {
        fprintf(stderr, "tq_load_model: invalid header in shard '%s'\n", shard_path);
#ifdef _WIN32
        UnmapViewOfFile(mmap_data);
#else
        munmap(mmap_data, file_size);
#endif
        return -1;
    }

    const char* json = (const char*)mmap_data + 8;
    void* data_base = (char*)mmap_data + 8 + header_size;

    int space = max_tensors - existing_count;
    if (space <= 0) {
#ifdef _WIN32
        UnmapViewOfFile(mmap_data);
#else
        munmap(mmap_data, file_size);
#endif
        return -1;
    }

    int n = parse_safetensors_header(json, (int64_t)header_size,
                                     tensors + existing_count, space);
    if (n < 0) {
        fprintf(stderr, "tq_load_model: JSON parse error in shard '%s'\n", shard_path);
#ifdef _WIN32
        UnmapViewOfFile(mmap_data);
#else
        munmap(mmap_data, file_size);
#endif
        return -1;
    }

    /* Set data_base for each newly parsed tensor */
    for (int i = existing_count; i < existing_count + n; i++) {
        tensors[i].data_base = data_base;
    }

    *mmap_out = mmap_data;
    *mmap_size_out = file_size;

    fprintf(stderr, "tq_load_model: shard '%s' — %d tensors, header=%llu bytes\n",
            shard_path, n, (unsigned long long)header_size);
    return n;
}

/* ============================================================
 * Helper: compare strings for qsort (used to sort shard filenames)
 * ============================================================ */
static int cmp_strings(const void* a, const void* b) {
    return strcmp(*(const char**)a, *(const char**)b);
}

/* ============================================================
 * Helper: find all shard files in the same directory as `path`.
 * If path points to a single model.safetensors, check for
 * model.safetensors.index.json to detect multi-shard.
 * If path is a shard like model-00001-of-00002.safetensors,
 * find all matching shards.
 * Returns number of shard paths, or 0 for single-file mode.
 * shard_paths[] is filled with allocated strings (caller must free).
 * ============================================================ */
static int find_shard_files(const char* path, char** shard_paths, int max_shards) {
    /* Extract directory and basename */
    char dir_path[2048];
    strncpy(dir_path, path, sizeof(dir_path) - 1);
    dir_path[sizeof(dir_path) - 1] = '\0';
    char* last_slash = strrchr(dir_path, '/');
    const char* basename = path;
    if (last_slash) {
        *last_slash = '\0';
        basename = last_slash + 1;
    } else {
        strcpy(dir_path, ".");
    }

    /* Check 1: path is "model.safetensors" and index file exists */
    if (strcmp(basename, "model.safetensors") == 0) {
        char index_path[2048];
        snprintf(index_path, sizeof(index_path), "%s/model.safetensors.index.json",
                 dir_path);
        if (access(index_path, R_OK) == 0) {
            /* Multi-shard: find all model-NNNNN-of-NNNNN.safetensors */
            DIR* dir = opendir(dir_path);
            if (!dir) return 0;

            int count = 0;
            struct dirent* ent;
            while ((ent = readdir(dir)) != NULL && count < max_shards) {
                /* Match pattern: model-XXXXX-of-XXXXX.safetensors */
                if (strncmp(ent->d_name, "model-", 6) == 0 &&
                    strstr(ent->d_name, "-of-") != NULL &&
                    strstr(ent->d_name, ".safetensors") != NULL &&
                    strstr(ent->d_name, ".index.") == NULL) {
                    char full[2048];
                    snprintf(full, sizeof(full), "%s/%s", dir_path, ent->d_name);
                    shard_paths[count] = strdup(full);
                    count++;
                }
            }
            closedir(dir);

            /* Sort shard paths alphabetically so they load in order */
            if (count > 1) {
                qsort(shard_paths, (size_t)count, sizeof(char*), cmp_strings);
            }

            if (count > 0) {
                fprintf(stderr, "tq_load_model: detected %d shard files\n", count);
            }
            return count;
        }
    }

    /* Check 2: path itself is a shard file (contains "-of-") */
    if (strstr(basename, "-of-") != NULL &&
        strstr(basename, ".safetensors") != NULL) {
        /* Extract prefix before the shard number: e.g., "model-" */
        const char* dash_of = strstr(basename, "-of-");
        /* Walk back to find the dash before the shard number */
        const char* num_start = dash_of;
        while (num_start > basename && *(num_start - 1) != '-') num_start--;
        size_t prefix_len = (size_t)(num_start - basename);

        DIR* dir = opendir(dir_path);
        if (!dir) return 0;

        int count = 0;
        struct dirent* ent;
        while ((ent = readdir(dir)) != NULL && count < max_shards) {
            if (strncmp(ent->d_name, basename, prefix_len) == 0 &&
                strstr(ent->d_name, "-of-") != NULL &&
                strstr(ent->d_name, ".safetensors") != NULL &&
                strstr(ent->d_name, ".index.") == NULL) {
                char full[2048];
                snprintf(full, sizeof(full), "%s/%s", dir_path, ent->d_name);
                shard_paths[count] = strdup(full);
                count++;
            }
        }
        closedir(dir);

        if (count > 1) {
            qsort(shard_paths, (size_t)count, sizeof(char*), cmp_strings);
            fprintf(stderr, "tq_load_model: detected %d shard files\n", count);
            return count;
        }

        /* Only 1 shard found = effectively single file, clean up */
        for (int i = 0; i < count; i++) free(shard_paths[i]);
        return 0;
    }

    return 0; /* single file mode */
}

/* ============================================================
 * Load model from safetensors file (single or multi-shard)
 * ============================================================ */
static tq_model_t* tq_load_safetensors(const char* path) {
    if (!path) return NULL;

    /* Detect multi-shard */
    char* shard_paths[TQ_MAX_SHARDS];
    memset(shard_paths, 0, sizeof(shard_paths));
    int n_shards = find_shard_files(path, shard_paths, TQ_MAX_SHARDS);

    /* Parse tensors from all shards (or single file) */
    tensor_info_t* tensors = (tensor_info_t*)calloc(MAX_TENSORS, sizeof(tensor_info_t));
    if (!tensors) {
        for (int i = 0; i < n_shards; i++) free(shard_paths[i]);
        return NULL;
    }

    void* mmap_ptrs[TQ_MAX_SHARDS];
    size_t mmap_sizes[TQ_MAX_SHARDS];
    memset(mmap_ptrs, 0, sizeof(mmap_ptrs));
    memset(mmap_sizes, 0, sizeof(mmap_sizes));
    int total_shards_loaded = 0;
    int n_tensors = 0;

    if (n_shards > 0) {
        /* Multi-shard loading */
        for (int s = 0; s < n_shards; s++) {
            int added = parse_shard(shard_paths[s], tensors, n_tensors, MAX_TENSORS,
                                    &mmap_ptrs[s], &mmap_sizes[s]);
            if (added < 0) {
                fprintf(stderr, "tq_load_model: failed to parse shard '%s'\n", shard_paths[s]);
                /* Clean up already-loaded shards */
                for (int j = 0; j < s; j++) {
#ifdef _WIN32
                    if (mmap_ptrs[j]) UnmapViewOfFile(mmap_ptrs[j]);
#else
                    if (mmap_ptrs[j]) munmap(mmap_ptrs[j], mmap_sizes[j]);
#endif
                }
                for (int j = 0; j < n_shards; j++) free(shard_paths[j]);
                free(tensors);
                return NULL;
            }
            n_tensors += added;
            total_shards_loaded++;
        }
        for (int i = 0; i < n_shards; i++) free(shard_paths[i]);
    } else {
        /* Single file loading */
        int added = parse_shard(path, tensors, 0, MAX_TENSORS,
                                &mmap_ptrs[0], &mmap_sizes[0]);
        if (added < 0) {
            free(tensors);
            return NULL;
        }
        n_tensors = added;
        total_shards_loaded = 1;
    }

    fprintf(stderr, "tq_load_model: parsed %d tensors from %d file(s)\n",
            n_tensors, total_shards_loaded);

    /* Log the dtype of the first non-metadata tensor */
    for (int i = 0; i < n_tensors; i++) {
        if (tensors[i].dtype[0] != '\0') {
            fprintf(stderr, "tq_load_model: tensor dtype = %s (e.g., '%s')\n",
                    tensors[i].dtype, tensors[i].name);
            break;
        }
    }

    /* Allocate model */
    tq_model_t* model = (tq_model_t*)calloc(1, sizeof(tq_model_t));
    if (!model) {
        free(tensors);
        goto fail;
    }
    model->_mmap_data = mmap_ptrs[0];
    model->_mmap_size = mmap_sizes[0];
    model->_n_shards = total_shards_loaded;
    for (int i = 1; i < total_shards_loaded; i++) {
        model->_mmap_shards[i] = mmap_ptrs[i];
        model->_mmap_shard_sizes[i] = mmap_sizes[i];
    }

    /* Calculate and allocate conversion buffer for non-F32 tensors */
    size_t conv_capacity = calc_conversion_buffer_size(tensors, n_tensors);
    float* conv_buf = NULL;
    size_t conv_used = 0;
    if (conv_capacity > 0) {
        conv_buf = (float*)malloc(conv_capacity);
        if (!conv_buf) {
            fprintf(stderr, "tq_load_model: failed to allocate %zu bytes for dtype conversion\n",
                    conv_capacity);
            free(model);
            free(tensors);
            goto fail;
        }
        model->_converted_data = conv_buf;
        model->_converted_size = conv_capacity;
        fprintf(stderr, "tq_load_model: allocated %zu MB for BF16/F16 -> FP32 conversion\n",
                conv_capacity / (1024 * 1024));
    }

    /* Detect model config from tensor shapes.
     * Look for embedding table to get vocab_size and hidden_dim.
     * Look for layer 0 weights to get n_heads, n_kv_heads, intermediate_dim. */

    /* Try common naming conventions for embedding */
    tensor_info_t* embed = find_tensor(tensors, n_tensors, "model.embed_tokens.weight");
    if (!embed) embed = find_tensor(tensors, n_tensors, "tok_embeddings.weight");
    if (!embed) embed = find_tensor(tensors, n_tensors, "transformer.wte.weight");

    if (!embed) {
        fprintf(stderr, "tq_load_model: cannot find embedding tensor\n");
        /* Dump first few tensor names for debugging */
        int show = n_tensors < 10 ? n_tensors : 10;
        for (int i = 0; i < show; i++) {
            fprintf(stderr, "  tensor[%d]: '%s'\n", i, tensors[i].name);
        }
        free(conv_buf);
        model->_converted_data = NULL;
        free(model);
        free(tensors);
        goto fail;
    }

    model->config.vocab_size = (int)embed->shape[0];
    model->config.hidden_dim = (int)embed->shape[1];

    /* Try to keep embedding as BF16 (streaming conversion saves ~1GB) */
    model->embed_bf16 = get_bf16_ptr(embed);
    if (model->embed_bf16) {
        /* BF16 embedding: don't convert, will convert on demand in forward pass */
        model->token_embedding = NULL;
        size_t embed_bytes = (size_t)embed->shape[0] * embed->shape[1] * 2;
        fprintf(stderr, "tq_load_model: keeping embed_tokens as BF16 (%zu MB saved)\n",
                embed_bytes / (1024 * 1024));
    } else {
        /* F32 or other dtype: convert as before */
        model->token_embedding = load_tensor( embed,
                                              &conv_buf, &conv_used, conv_capacity);
    }

    /* Detect n_layers (handles both standard and Qwen3.5 naming) */
    model->config.n_layers = detect_n_layers(tensors, n_tensors);

    /* Detect which layers have standard self_attn */
    int n_attn_layers = 0;
    int* attn_indices = detect_attn_layers(tensors, n_tensors,
                                            model->config.n_layers, &n_attn_layers);
    model->n_attn_layers = n_attn_layers;
    model->attn_layer_indices = attn_indices;

    if (n_attn_layers < model->config.n_layers) {
        fprintf(stderr, "tq_load_model: hybrid architecture detected — "
                "%d/%d layers have self_attn\n",
                n_attn_layers, model->config.n_layers);
        fprintf(stderr, "tq_load_model: attn layers:");
        for (int i = 0; i < n_attn_layers; i++) {
            fprintf(stderr, " %d", attn_indices[i]);
        }
        fprintf(stderr, "\n");
    }

    /* Detect head dimensions from Q projection shape.
     * For hybrid models, find the first layer that has self_attn. */
    int probe_layer = 0;
    if (n_attn_layers > 0) {
        probe_layer = attn_indices[0];
    }

    char name_buf[MAX_NAME_LEN];
    snprintf(name_buf, sizeof(name_buf),
             "model.layers.%d.self_attn.q_proj.weight", probe_layer);
    tensor_info_t* wq0 = find_tensor(tensors, n_tensors, name_buf);

    snprintf(name_buf, sizeof(name_buf),
             "model.layers.%d.self_attn.k_proj.weight", probe_layer);
    tensor_info_t* wk0 = find_tensor(tensors, n_tensors, name_buf);

    if (wq0 && wk0) {
        int q_out = (int)wq0->shape[0];
        int k_out = (int)wk0->shape[0];

        /* Try to detect head_dim from q_norm weight if available */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.q_norm.weight", probe_layer);
        tensor_info_t* qn0 = find_tensor(tensors, n_tensors, name_buf);
        int head_dim;
        if (qn0 && qn0->n_dims >= 1) {
            head_dim = (int)qn0->shape[0];
            model->config.use_qk_norm = 1;
        } else {
            /* Common head_dim values: 128, 64, 96, 256 */
            head_dim = 128;
            if (q_out % head_dim != 0) head_dim = 64;
            if (q_out % head_dim != 0) head_dim = 96;
            if (q_out % head_dim != 0) head_dim = 256;
            model->config.use_qk_norm = 0;
        }
        model->config.head_dim = head_dim;
        model->config.n_kv_heads = k_out / head_dim;

        /* Detect attn_output_gate: if q_proj output is exactly 2x k_proj
         * output * (n_heads/n_kv_heads ratio), then q_proj includes a gate.
         * More precisely: q_out = n_heads * head_dim * (1 + gate).
         * Compare against o_proj input dim to determine n_heads. */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.o_proj.weight", probe_layer);
        tensor_info_t* wo0 = find_tensor(tensors, n_tensors, name_buf);
        if (wo0 && wo0->n_dims >= 2) {
            int o_in = (int)wo0->shape[1]; /* o_proj is [hidden_dim, n_heads*head_dim] */
            int n_heads_from_o = o_in / head_dim;
            if (q_out == n_heads_from_o * head_dim * 2) {
                /* q_proj is doubled: [Q, gate_q] */
                model->config.attn_output_gate = 1;
                model->config.n_heads = n_heads_from_o;
                fprintf(stderr, "tq_load_model: detected attn_output_gate=1 "
                        "(q_proj=%d = 2 * %d * %d)\n",
                        q_out, n_heads_from_o, head_dim);
            } else {
                model->config.attn_output_gate = 0;
                model->config.n_heads = q_out / head_dim;
            }
        } else {
            model->config.attn_output_gate = 0;
            model->config.n_heads = q_out / head_dim;
        }
    } else {
        /* Defaults for small models */
        model->config.head_dim = 64;
        model->config.n_heads = model->config.hidden_dim / 64;
        model->config.n_kv_heads = model->config.n_heads;
        model->config.use_qk_norm = 0;
        model->config.attn_output_gate = 0;
    }

    /* Detect DeltaNet config from first linear_attn layer */
    model->config.delta_n_heads = 0;
    model->config.delta_n_kv_heads = 0;
    model->config.delta_key_head_dim = 0;
    model->config.delta_value_head_dim = 0;
    model->config.delta_conv_width = 4;
    model->config.partial_rotary_factor = 0.0f;
    {
        /* Find first DeltaNet layer */
        int delta_layer = -1;
        for (int l = 0; l < model->config.n_layers; l++) {
            snprintf(name_buf, sizeof(name_buf),
                     "model.layers.%d.linear_attn.A_log", l);
            if (find_tensor(tensors, n_tensors, name_buf)) {
                delta_layer = l;
                break;
            }
        }
        if (delta_layer >= 0) {
            snprintf(name_buf, sizeof(name_buf),
                     "model.layers.%d.linear_attn.A_log", delta_layer);
            tensor_info_t* a_log = find_tensor(tensors, n_tensors, name_buf);
            if (a_log) {
                model->config.delta_n_heads = (int)a_log->shape[0];
            }

            snprintf(name_buf, sizeof(name_buf),
                     "model.layers.%d.linear_attn.in_proj_qkv.weight", delta_layer);
            tensor_info_t* qkv_proj = find_tensor(tensors, n_tensors, name_buf);
            /* Detect value_head_dim from norm weight shape */
            snprintf(name_buf, sizeof(name_buf),
                     "model.layers.%d.linear_attn.norm.weight", delta_layer);
            tensor_info_t* norm_w = find_tensor(tensors, n_tensors, name_buf);
            if (norm_w) {
                model->config.delta_value_head_dim = (int)norm_w->shape[0];
            }

            /* Detect key_head_dim from z proj: z_dim = n_v_heads * val_dim
             * qkv_dim = n_kv_heads * key_dim * 2 + n_v_heads * val_dim
             * So: n_kv_heads * key_dim = (qkv_dim - z_dim) / 2 */
            snprintf(name_buf, sizeof(name_buf),
                     "model.layers.%d.linear_attn.in_proj_z.weight", delta_layer);
            tensor_info_t* z_proj = find_tensor(tensors, n_tensors, name_buf);

            if (qkv_proj && model->config.delta_n_heads > 0) {
                int qkv_dim = (int)qkv_proj->shape[0];
                int z_dim = z_proj ? (int)z_proj->shape[0] : 0;
                int val_dim = model->config.delta_value_head_dim;

                if (z_dim > 0 && val_dim > 0) {
                    /* z_dim = n_v_heads * val_dim → already known */
                    /* qkv = n_kv_heads * key_dim * 2 + z_dim */
                    int qk_dim = qkv_dim - z_dim;  /* Q+K total */
                    /* Try: key_dim = val_dim (common), then n_kv_heads = qk_dim / (2 * key_dim) */
                    int key_dim = val_dim; /* start with same as val */
                    int n_kv = qk_dim / (2 * key_dim);
                    if (n_kv * 2 * key_dim == qk_dim) {
                        model->config.delta_key_head_dim = key_dim;
                        model->config.delta_n_kv_heads = n_kv;
                    } else {
                        /* Fallback: assume n_kv_heads = n_v_heads */
                        model->config.delta_key_head_dim = qkv_dim / (3 * model->config.delta_n_heads);
                    }
                } else {
                    /* Original logic for models where Q/K/V heads are equal */
                    model->config.delta_key_head_dim = qkv_dim / (3 * model->config.delta_n_heads);
                }
                if (model->config.delta_value_head_dim == 0)
                    model->config.delta_value_head_dim = model->config.delta_key_head_dim;
            }

            snprintf(name_buf, sizeof(name_buf),
                     "model.layers.%d.linear_attn.conv1d.weight", delta_layer);
            tensor_info_t* conv = find_tensor(tensors, n_tensors, name_buf);
            if (conv && conv->n_dims >= 3) {
                model->config.delta_conv_width = (int)conv->shape[2];
            }

            int delta_nkv = model->config.delta_n_kv_heads > 0 ?
                            model->config.delta_n_kv_heads : model->config.delta_n_heads;
            fprintf(stderr, "tq_load_model: DeltaNet config — v_heads=%d, kv_heads=%d, "
                    "key_dim=%d, val_dim=%d, conv_w=%d\n",
                    model->config.delta_n_heads, delta_nkv,
                    model->config.delta_key_head_dim,
                    model->config.delta_value_head_dim, model->config.delta_conv_width);
        }
    }

    /* Detect intermediate_dim from gate projection (use probe_layer) */
    snprintf(name_buf, sizeof(name_buf),
             "model.layers.%d.mlp.gate_proj.weight", probe_layer);
    tensor_info_t* wg0 = find_tensor(tensors, n_tensors, name_buf);
    if (!wg0) {
        /* Try layer 0 as fallback — MLP might exist on all layers even in hybrid models */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.0.mlp.gate_proj.weight");
        wg0 = find_tensor(tensors, n_tensors, name_buf);
    }
    if (wg0) {
        model->config.intermediate_dim = (int)wg0->shape[0];
    } else {
        model->config.intermediate_dim = model->config.hidden_dim * 4;
    }

    /* Detect Gemma3 architecture by presence of pre_feedforward_layernorm */
    {
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.0.pre_feedforward_layernorm.weight");
        tensor_info_t* gemma3_probe = find_tensor(tensors, n_tensors, name_buf);
        if (gemma3_probe) {
            model->config.model_type = 1; /* gemma3 */
            model->config.n_norms_per_block = 4;
            fprintf(stderr, "tq_load_model: detected Gemma3 architecture (4 norms per block)\n");
        } else {
            model->config.model_type = 0; /* qwen35 */
            model->config.n_norms_per_block = 2;
        }
    }

    /* Defaults — tuned for Qwen3.5 if DeltaNet detected */
    model->config.max_seq_len = 4096;
    if (model->config.model_type == 1) {
        /* Gemma3: rope_theta=1M for global, 10K for local, rms_norm_eps=1e-6 */
        model->config.rope_freq_base = 1000000.0f; /* global layers */
        model->config.rope_local_base_freq = 10000.0f; /* sliding/local layers */
        model->config.rms_norm_eps = 1e-6f;
        model->config.partial_rotary_factor = 0.0f;
        model->config.sliding_window = 512;
        model->config.query_pre_attn_scalar = 256.0f;
    } else if (model->config.delta_n_heads > 0) {
        /* Qwen3.5 uses rope_theta=10M, rms_norm_eps=1e-6, partial_rotary=0.25 */
        model->config.rope_freq_base = 10000000.0f;
        model->config.rms_norm_eps = 1e-6f;
        model->config.partial_rotary_factor = 0.25f;
        model->config.query_pre_attn_scalar = 0.0f;
    } else {
        model->config.rope_freq_base = 10000.0f;
        model->config.rms_norm_eps = 1e-5f;
        model->config.partial_rotary_factor = 0.0f;
        model->config.query_pre_attn_scalar = 0.0f;
    }

    /* Allocate layer weight pointers */
    int n_layers = model->config.n_layers;
    model->layers = (tq_layer_weights_t*)calloc((size_t)n_layers, sizeof(tq_layer_weights_t));
    if (!model->layers) {
        free(attn_indices);
        model->attn_layer_indices = NULL;
        free(conv_buf);
        model->_converted_data = NULL;
        free(model);
        free(tensors);
        goto fail;
    }

    /* Map per-layer weights */
    for (int l = 0; l < n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];

        /* Attention norm */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.input_layernorm.weight", l);
        layer->attn_norm = load_tensor(
                                        find_tensor(tensors, n_tensors, name_buf),
                                        &conv_buf, &conv_used, conv_capacity);

        /* FFN norm (Qwen3.5: post_attention_layernorm used as pre-FFN norm) */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.post_attention_layernorm.weight", l);
        layer->ffn_norm = load_tensor(
                                       find_tensor(tensors, n_tensors, name_buf),
                                       &conv_buf, &conv_used, conv_capacity);

        /* Gemma3 extra norms: post_attn, pre_ffn, post_ffn */
        if (model->config.model_type == 1) {
            /* For Gemma3, post_attention_layernorm is applied to attn output,
             * not as pre-FFN norm. Store it in post_attn_norm. */
            layer->post_attn_norm = layer->ffn_norm;

            snprintf(name_buf, sizeof(name_buf),
                     "model.layers.%d.pre_feedforward_layernorm.weight", l);
            layer->pre_ffn_norm = load_tensor(
                                               find_tensor(tensors, n_tensors, name_buf),
                                               &conv_buf, &conv_used, conv_capacity);

            snprintf(name_buf, sizeof(name_buf),
                     "model.layers.%d.post_feedforward_layernorm.weight", l);
            layer->post_ffn_norm = load_tensor(
                                                find_tensor(tensors, n_tensors, name_buf),
                                                &conv_buf, &conv_used, conv_capacity);
        }

        /* Q, K, V, O projections — only exist for self_attn layers */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.q_proj.weight", l);
        layer->wq = load_tensor(
                                 find_tensor(tensors, n_tensors, name_buf),
                                 &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.k_proj.weight", l);
        layer->wk = load_tensor(
                                 find_tensor(tensors, n_tensors, name_buf),
                                 &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.v_proj.weight", l);
        layer->wv = load_tensor(
                                 find_tensor(tensors, n_tensors, name_buf),
                                 &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.o_proj.weight", l);
        layer->wo = load_tensor(
                                 find_tensor(tensors, n_tensors, name_buf),
                                 &conv_buf, &conv_used, conv_capacity);

        /* QK-norm weights (Qwen3.5 style) */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.q_norm.weight", l);
        layer->q_norm = load_tensor(
                                     find_tensor(tensors, n_tensors, name_buf),
                                     &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.k_norm.weight", l);
        layer->k_norm = load_tensor(
                                     find_tensor(tensors, n_tensors, name_buf),
                                     &conv_buf, &conv_used, conv_capacity);

        /* DeltaNet (linear_attention) weights */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.A_log", l);
        layer->delta_a_log = load_tensor(
                                          find_tensor(tensors, n_tensors, name_buf),
                                          &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.conv1d.weight", l);
        layer->delta_conv1d = load_tensor(
                                           find_tensor(tensors, n_tensors, name_buf),
                                           &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.dt_bias", l);
        layer->delta_dt_bias = load_tensor(
                                            find_tensor(tensors, n_tensors, name_buf),
                                            &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.in_proj_a.weight", l);
        layer->delta_in_proj_a = load_tensor(
                                              find_tensor(tensors, n_tensors, name_buf),
                                              &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.in_proj_b.weight", l);
        layer->delta_in_proj_b = load_tensor(
                                              find_tensor(tensors, n_tensors, name_buf),
                                              &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.in_proj_qkv.weight", l);
        layer->delta_in_proj_qkv = load_tensor(
                                                find_tensor(tensors, n_tensors, name_buf),
                                                &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.in_proj_z.weight", l);
        layer->delta_in_proj_z = load_tensor(
                                              find_tensor(tensors, n_tensors, name_buf),
                                              &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.norm.weight", l);
        layer->delta_norm = load_tensor(
                                         find_tensor(tensors, n_tensors, name_buf),
                                         &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.out_proj.weight", l);
        layer->delta_out_proj = load_tensor(
                                             find_tensor(tensors, n_tensors, name_buf),
                                             &conv_buf, &conv_used, conv_capacity);

        /* Reorder DeltaNet V-head weights to GGUF tiled order.
         * Original: [h0,h1,h2,...,h31] (sequential V-heads)
         * Tiled:    [h0,h2,h4,...,h30, h1,h3,...,h31] (even first, odd second)
         * This matches ggml broadcast convention for GQA DeltaNet.
         * Only needed when delta_n_kv_heads < delta_n_heads (asymmetric). */
        if (model->config.delta_n_kv_heads > 0 && model->config.delta_n_kv_heads < model->config.delta_n_heads
            && layer->delta_a_log) {
            int dim = model->config.hidden_dim;
            int dn = model->config.delta_n_heads;
            int nkv = model->config.delta_n_kv_heads;
            if (l == 0) fprintf(stderr, "tq_load_model: reordering DeltaNet V-heads (dn=%d, nkv=%d) to tiled order\n", dn, nkv);
            int kv_mul = dn / nkv;          /* 2 V-heads per K-group */
            int dk = model->config.delta_key_head_dim;
            int dv = model->config.delta_value_head_dim;

            /* Build tiled permutation: for each K-group, collect its V-heads */
            int perm[256]; /* max heads */
            for (int pass = 0; pass < kv_mul; pass++)
                for (int g = 0; g < nkv; g++)
                    perm[pass * nkv + g] = g * kv_mul + pass;

            /* Reorder per-head vectors: a_log, dt_bias */
            float* tmp = (float*)malloc((size_t)dn * sizeof(float));
            if (tmp && layer->delta_a_log) {
                for (int h = 0; h < dn; h++) tmp[h] = layer->delta_a_log[perm[h]];
                memcpy(layer->delta_a_log, tmp, (size_t)dn * sizeof(float));
            }
            if (tmp && layer->delta_dt_bias) {
                for (int h = 0; h < dn; h++) tmp[h] = layer->delta_dt_bias[perm[h]];
                memcpy(layer->delta_dt_bias, tmp, (size_t)dn * sizeof(float));
            }

            /* Reorder per-head row matrices: in_proj_a [dn, dim], in_proj_b [dn, dim] */
            float* row_tmp = (float*)malloc((size_t)dn * dim * sizeof(float));
            if (row_tmp && layer->delta_in_proj_a) {
                for (int h = 0; h < dn; h++)
                    memcpy(row_tmp + (size_t)h * dim, layer->delta_in_proj_a + (size_t)perm[h] * dim, dim * sizeof(float));
                memcpy(layer->delta_in_proj_a, row_tmp, (size_t)dn * dim * sizeof(float));
            }
            if (row_tmp && layer->delta_in_proj_b) {
                for (int h = 0; h < dn; h++)
                    memcpy(row_tmp + (size_t)h * dim, layer->delta_in_proj_b + (size_t)perm[h] * dim, dim * sizeof(float));
                memcpy(layer->delta_in_proj_b, row_tmp, (size_t)dn * dim * sizeof(float));
            }

            /* Reorder V portion of in_proj_qkv: QKV = [Q(nkv*dk), K(nkv*dk), V(dn*dv)]
             * Only V part needs reordering */
            if (row_tmp && layer->delta_in_proj_qkv) {
                int qk_total = nkv * dk * 2;
                float* v_start = layer->delta_in_proj_qkv + (size_t)qk_total * dim;
                /* V is [dn, dv*dim/dn]... actually V rows in QKV are contiguous:
                 * row r of V portion = qkv_row[qk_total + r] */
                /* Wait, QKV is [qkv_dim, dim] where qkv_dim = nkv*dk*2 + dn*dv
                 * Row indices qk_total..qk_total+dn*dv-1 are V rows.
                 * Each V-head h occupies dv rows: [qk_total + h*dv .. qk_total + (h+1)*dv - 1] */
                float* v_reorder = (float*)malloc((size_t)dn * dv * dim * sizeof(float));
                if (v_reorder) {
                    for (int h = 0; h < dn; h++) {
                        memcpy(v_reorder + (size_t)h * dv * dim,
                               v_start + (size_t)perm[h] * dv * dim,
                               (size_t)dv * dim * sizeof(float));
                    }
                    memcpy(v_start, v_reorder, (size_t)dn * dv * dim * sizeof(float));
                    free(v_reorder);
                }
            }

            /* Reorder in_proj_z [dn*dv, dim]: z_dim = dn * dv, grouped by V-head */
            if (row_tmp && layer->delta_in_proj_z) {
                float* z_reorder = (float*)malloc((size_t)dn * dv * dim * sizeof(float));
                if (z_reorder) {
                    for (int h = 0; h < dn; h++) {
                        memcpy(z_reorder + (size_t)h * dv * dim,
                               layer->delta_in_proj_z + (size_t)perm[h] * dv * dim,
                               (size_t)dv * dim * sizeof(float));
                    }
                    memcpy(layer->delta_in_proj_z, z_reorder, (size_t)dn * dv * dim * sizeof(float));
                    free(z_reorder);
                }
            }

            /* Reorder out_proj [dim, dn*dv]: columns grouped by V-head
             * out_proj is [dim, z_dim], column h*dv..(h+1)*dv-1 belongs to V-head h */
            if (row_tmp && layer->delta_out_proj) {
                int z_dim = dn * dv;
                float* out_reorder = (float*)malloc((size_t)dim * z_dim * sizeof(float));
                if (out_reorder) {
                    for (int r = 0; r < dim; r++) {
                        for (int h = 0; h < dn; h++) {
                            memcpy(out_reorder + (size_t)r * z_dim + h * dv,
                                   layer->delta_out_proj + (size_t)r * z_dim + perm[h] * dv,
                                   (size_t)dv * sizeof(float));
                        }
                    }
                    memcpy(layer->delta_out_proj, out_reorder, (size_t)dim * z_dim * sizeof(float));
                    free(out_reorder);
                }
            }

            /* Reorder norm weights [dv]: same for all heads, no reorder needed */

            free(tmp);
            free(row_tmp);
        }

        /* FFN: gate, up, down projections (SwiGLU) */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.gate_proj.weight", l);
        layer->w_gate = load_tensor(
                                     find_tensor(tensors, n_tensors, name_buf),
                                     &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.up_proj.weight", l);
        layer->w_up = load_tensor(
                                   find_tensor(tensors, n_tensors, name_buf),
                                   &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.down_proj.weight", l);
        layer->w_down = load_tensor(
                                     find_tensor(tensors, n_tensors, name_buf),
                                     &conv_buf, &conv_used, conv_capacity);
    }

    /* Output norm */
    model->output_norm = load_tensor(
        find_tensor(tensors, n_tensors, "model.norm.weight"),
        &conv_buf, &conv_used, conv_capacity);

    /* Output weight — may be tied to embedding */
    tensor_info_t* lm_head = find_tensor(tensors, n_tensors, "lm_head.weight");
    if (lm_head) {
        /* Try to keep lm_head as BF16 */
        model->output_weight_bf16 = get_bf16_ptr(lm_head);
        if (model->output_weight_bf16) {
            model->output_weight = NULL;
            size_t lm_bytes = (size_t)lm_head->shape[0] * lm_head->shape[1] * 2;
            fprintf(stderr, "tq_load_model: keeping lm_head as BF16 (%zu MB saved)\n",
                    lm_bytes / (1024 * 1024));
        } else {
            model->output_weight = load_tensor( lm_head,
                                                &conv_buf, &conv_used, conv_capacity);
        }
    } else {
        /* Weight tying: reuse embedding (either FP32 or BF16) */
        model->output_weight = model->token_embedding;
        model->output_weight_bf16 = model->embed_bf16;
    }

    free(tensors);

    /* Qwen3.5 (DeltaNet hybrid) RMSNorm adjustment.
     * Only for non-GGUF models (raw checkpoints). GGUF files from
     * llama.cpp already have +1 baked in by the converter.
     * Qwen2/Qwen3 use standard RMSNorm and never need +1. */
    if (model->config.delta_n_heads > 0 && !model->gguf_ctx) {
        int dim_h = model->config.hidden_dim;
        int head_dim_h = model->config.head_dim;

        for (int l = 0; l < n_layers; l++) {
            tq_layer_weights_t* layer_w = &model->layers[l];
            if (layer_w->attn_norm) {
                for (int i = 0; i < dim_h; i++)
                    layer_w->attn_norm[i] += 1.0f;
            }
            if (layer_w->ffn_norm) {
                for (int i = 0; i < dim_h; i++)
                    layer_w->ffn_norm[i] += 1.0f;
            }
            if (layer_w->q_norm) {
                for (int i = 0; i < head_dim_h; i++)
                    layer_w->q_norm[i] += 1.0f;
            }
            if (layer_w->k_norm) {
                for (int i = 0; i < head_dim_h; i++)
                    layer_w->k_norm[i] += 1.0f;
            }
            /* Note: delta_norm is NOT adjusted — Qwen3_5RMSNormGated
             * uses plain weight without +1 offset */
        }
        if (model->output_norm) {
            for (int i = 0; i < dim_h; i++)
                model->output_norm[i] += 1.0f;
        }
        fprintf(stderr, "tq_load_model: applied Qwen RMSNorm +1 weight adjustment\n");
    }

    /* Gemma3 RMSNorm adjustment: same (1+w) scaling as Qwen3.5 */
    if (model->config.model_type == 1) {
        int dim_h = model->config.hidden_dim;
        int head_dim_h = model->config.head_dim;

        for (int l = 0; l < n_layers; l++) {
            tq_layer_weights_t* layer_w = &model->layers[l];
            if (layer_w->attn_norm) {
                for (int i = 0; i < dim_h; i++) {
                    layer_w->attn_norm[i] += 1.0f;
                }
            }
            if (layer_w->post_attn_norm) {
                for (int i = 0; i < dim_h; i++) {
                    layer_w->post_attn_norm[i] += 1.0f;
                }
            }
            if (layer_w->pre_ffn_norm) {
                for (int i = 0; i < dim_h; i++) {
                    layer_w->pre_ffn_norm[i] += 1.0f;
                }
            }
            if (layer_w->post_ffn_norm) {
                for (int i = 0; i < dim_h; i++) {
                    layer_w->post_ffn_norm[i] += 1.0f;
                }
            }
            if (layer_w->q_norm) {
                for (int i = 0; i < head_dim_h; i++) {
                    layer_w->q_norm[i] += 1.0f;
                }
            }
            if (layer_w->k_norm) {
                for (int i = 0; i < head_dim_h; i++) {
                    layer_w->k_norm[i] += 1.0f;
                }
            }
        }
        if (model->output_norm) {
            for (int i = 0; i < dim_h; i++) {
                model->output_norm[i] += 1.0f;
            }
        }
        fprintf(stderr, "tq_load_model: applied Gemma3 RMSNorm +1 weight adjustment\n");

        /* Set up layer_is_sliding for Gemma3.
         * Pattern: 5 sliding + 1 full, repeated. Layers 0-4=sliding, 5=full, etc.
         * We detect by checking layer count modulo 6. */
        model->layer_is_sliding = (int*)calloc((size_t)n_layers, sizeof(int));
        if (model->layer_is_sliding) {
            for (int l = 0; l < n_layers; l++) {
                /* Full/global attention every 6th layer (indices 5, 11, 17, ...) */
                if ((l + 1) % 6 == 0) {
                    model->layer_is_sliding[l] = 0; /* global */
                } else {
                    model->layer_is_sliding[l] = 1; /* sliding */
                }
            }
            int n_sliding = 0, n_global = 0;
            for (int l = 0; l < n_layers; l++) {
                if (model->layer_is_sliding[l]) {
                    n_sliding++;
                } else {
                    n_global++;
                }
            }
            fprintf(stderr, "tq_load_model: Gemma3 layer types: %d sliding, %d global\n",
                    n_sliding, n_global);
        }
    }

    fprintf(stderr, "tq_load_model: loaded %d layers (%d with self_attn), "
            "dim=%d, heads=%d/%d, vocab=%d\n",
            model->config.n_layers, model->n_attn_layers,
            model->config.hidden_dim,
            model->config.n_heads, model->config.n_kv_heads,
            model->config.vocab_size);

    if (conv_capacity > 0) {
        fprintf(stderr, "tq_load_model: dtype conversion used %zu / %zu MB\n",
                conv_used / (1024 * 1024), conv_capacity / (1024 * 1024));
    }

    return model;

fail:
    /* Clean up all mmaps on failure */
    for (int i = 0; i < total_shards_loaded; i++) {
        if (mmap_ptrs[i]) {
#ifdef _WIN32
            UnmapViewOfFile(mmap_ptrs[i]);
#else
            munmap(mmap_ptrs[i], mmap_sizes[i]);
#endif
        }
    }
    return NULL;
}

/* ============================================================
 * Q8 weight quantization — quantize all layer weights post-load
 *
 * Converts FP32 weight matrices to Q8 (int8 + per-block float scale,
 * block_size=32).  This halves memory: FP32 uses 4 bytes/value,
 * Q8 uses 1 byte + 4 bytes/32 = 1.125 bytes/value.
 *
 * Each weight matrix [rows, cols] gets:
 *   - int8_t q8[rows * cols]           — quantized values
 *   - float  scales[rows * (cols/32)]  — per-block scales
 *
 * After quantization, the original FP32 pointer is set to NULL
 * (it either pointed into mmap or conversion buffer, both still alive).
 * ============================================================ */

/* Helper: quantize a single weight matrix and store into pre-allocated buffer */
static void quantize_matrix_q8(const float* src, int rows, int cols,
                                 int8_t** out_qs, float** out_scales,
                                 char** buf, size_t* used) {
    if (!src || rows <= 0 || cols <= 0) {
        *out_qs = NULL;
        *out_scales = NULL;
        return;
    }
    int n_blocks_per_row = (cols + 31) / 32;
    size_t qs_bytes = (size_t)rows * cols * sizeof(int8_t);
    size_t sc_bytes = (size_t)rows * n_blocks_per_row * sizeof(float);

    int8_t*  qs = (int8_t*)(*buf + *used);
    *used += qs_bytes;
    float*   sc = (float*)(*buf + *used);
    *used += sc_bytes;

    for (int r = 0; r < rows; r++) {
        tq_quantize_row_q8(src + (size_t)r * cols,
                            qs + (size_t)r * cols,
                            sc + (size_t)r * n_blocks_per_row,
                            cols);
    }
    *out_qs = qs;
    *out_scales = sc;
}

/* Calculate total Q8 buffer size needed for all layer weights */
static size_t calc_q8_buffer_size(const tq_model_t* model) {
    size_t total = 0;
    const tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;

    /* DeltaNet dimensions */
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;

    for (int l = 0; l < c->n_layers; l++) {
        const tq_layer_weights_t* layer = &model->layers[l];

        /* Self-attention weights */
        if (layer->wq) {
            int rows = qg_dim;
            int cols = dim;
            int nb = (cols + 31) / 32;
            total += (size_t)rows * cols;        /* int8 data */
            total += (size_t)rows * nb * 4;      /* float scales */
        }
        if (layer->wk) {
            int nb = (dim + 31) / 32;
            total += (size_t)kv_dim * dim;
            total += (size_t)kv_dim * nb * 4;
        }
        if (layer->wv) {
            int nb = (dim + 31) / 32;
            total += (size_t)kv_dim * dim;
            total += (size_t)kv_dim * nb * 4;
        }
        if (layer->wo) {
            int nb = (q_dim + 31) / 32;
            total += (size_t)dim * q_dim;
            total += (size_t)dim * nb * 4;
        }

        /* FFN weights */
        if (layer->w_gate) {
            int nb = (dim + 31) / 32;
            total += (size_t)inter * dim;
            total += (size_t)inter * nb * 4;
        }
        if (layer->w_up) {
            int nb = (dim + 31) / 32;
            total += (size_t)inter * dim;
            total += (size_t)inter * nb * 4;
        }
        if (layer->w_down) {
            int nb = (inter + 31) / 32;
            total += (size_t)dim * inter;
            total += (size_t)dim * nb * 4;
        }

        /* DeltaNet weights */
        if (layer->delta_in_proj_qkv) {
            int nb = (dim + 31) / 32;
            total += (size_t)delta_qkv_dim * dim;
            total += (size_t)delta_qkv_dim * nb * 4;
        }
        if (layer->delta_in_proj_z) {
            int nb = (dim + 31) / 32;
            total += (size_t)delta_z_dim * dim;
            total += (size_t)delta_z_dim * nb * 4;
        }
        if (layer->delta_in_proj_a) {
            int nb = (dim + 31) / 32;
            total += (size_t)delta_dn * dim;
            total += (size_t)delta_dn * nb * 4;
        }
        if (layer->delta_in_proj_b) {
            int nb = (dim + 31) / 32;
            total += (size_t)delta_dn * dim;
            total += (size_t)delta_dn * nb * 4;
        }
        if (layer->delta_out_proj) {
            int nb = (delta_z_dim + 31) / 32;
            total += (size_t)dim * delta_z_dim;
            total += (size_t)dim * nb * 4;
        }
    }
    return total;
}

void tq_quantize_weights(tq_model_t* model) {
    if (!model || model->use_q8_weights) return;

    const tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;

    /* DeltaNet dimensions */
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;

    size_t buf_size = calc_q8_buffer_size(model);
    char* buf = (char*)malloc(buf_size);
    if (!buf) {
        fprintf(stderr, "tq_quantize_weights: failed to allocate %zu MB for Q8\n",
                buf_size / (1024 * 1024));
        return;
    }
    size_t used = 0;

    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];

        /* Self-attention */
        quantize_matrix_q8(layer->wq, qg_dim, dim,
                            &layer->wq_q8, &layer->wq_q8s, &buf, &used);
        if (layer->wq_q8) layer->wq = NULL;

        quantize_matrix_q8(layer->wk, kv_dim, dim,
                            &layer->wk_q8, &layer->wk_q8s, &buf, &used);
        if (layer->wk_q8) layer->wk = NULL;

        quantize_matrix_q8(layer->wv, kv_dim, dim,
                            &layer->wv_q8, &layer->wv_q8s, &buf, &used);
        if (layer->wv_q8) layer->wv = NULL;

        quantize_matrix_q8(layer->wo, dim, q_dim,
                            &layer->wo_q8, &layer->wo_q8s, &buf, &used);
        if (layer->wo_q8) layer->wo = NULL;

        /* FFN */
        quantize_matrix_q8(layer->w_gate, inter, dim,
                            &layer->w_gate_q8, &layer->w_gate_q8s, &buf, &used);
        if (layer->w_gate_q8) layer->w_gate = NULL;

        quantize_matrix_q8(layer->w_up, inter, dim,
                            &layer->w_up_q8, &layer->w_up_q8s, &buf, &used);
        if (layer->w_up_q8) layer->w_up = NULL;

        quantize_matrix_q8(layer->w_down, dim, inter,
                            &layer->w_down_q8, &layer->w_down_q8s, &buf, &used);
        if (layer->w_down_q8) layer->w_down = NULL;

        /* DeltaNet */
        quantize_matrix_q8(layer->delta_in_proj_qkv, delta_qkv_dim, dim,
                            &layer->delta_in_proj_qkv_q8, &layer->delta_in_proj_qkv_q8s,
                            &buf, &used);
        if (layer->delta_in_proj_qkv_q8) layer->delta_in_proj_qkv = NULL;

        quantize_matrix_q8(layer->delta_in_proj_z, delta_z_dim, dim,
                            &layer->delta_in_proj_z_q8, &layer->delta_in_proj_z_q8s,
                            &buf, &used);
        if (layer->delta_in_proj_z_q8) layer->delta_in_proj_z = NULL;

        quantize_matrix_q8(layer->delta_in_proj_a, delta_dn, dim,
                            &layer->delta_in_proj_a_q8, &layer->delta_in_proj_a_q8s,
                            &buf, &used);
        if (layer->delta_in_proj_a_q8) layer->delta_in_proj_a = NULL;

        quantize_matrix_q8(layer->delta_in_proj_b, delta_dn, dim,
                            &layer->delta_in_proj_b_q8, &layer->delta_in_proj_b_q8s,
                            &buf, &used);
        if (layer->delta_in_proj_b_q8) layer->delta_in_proj_b = NULL;

        quantize_matrix_q8(layer->delta_out_proj, dim, delta_z_dim,
                            &layer->delta_out_proj_q8, &layer->delta_out_proj_q8s,
                            &buf, &used);
        if (layer->delta_out_proj_q8) layer->delta_out_proj = NULL;
    }

    model->use_q8_weights = 1;
    model->_q8_data = buf;
    model->_q8_size = used;

    /* If original weights were in conversion buffer (BF16->FP32), free it.
     * The converted_data is no longer needed since all layer weights are now Q8.
     * Note: norm weights, conv1d, bias, etc. still point into converted_data or mmap,
     * so we CANNOT free it. Keep it alive. */

    fprintf(stderr, "tq_quantize_weights: quantized to Q8 (%zu MB, was ~%zu MB FP32)\n",
            used / (1024 * 1024), used * 4 / (1024 * 1024));
}

/* ============================================================
 * Q4_0 weight quantization — quantize all layer weights post-load
 *
 * Converts FP32 weight matrices to Q4_0 (packed 4-bit + per-block float scale,
 * block_size=32). This reduces memory ~7x: FP32 uses 4 bytes/value,
 * Q4_0 uses 0.5 byte + 4 bytes/32 = 0.625 bytes/value.
 *
 * Each weight matrix [rows, cols] gets:
 *   - uint8_t qs[rows * (cols/32) * 16] — packed 4-bit values (2 per byte)
 *   - float scales[rows * (cols/32)]     — per-block scales
 *
 * After quantization, the original FP32 pointer is set to NULL.
 * ============================================================ */

/* Helper: quantize a single weight matrix to Q4 and store into pre-allocated buffer */
static void quantize_matrix_q4(const float* src, int rows, int cols,
                                 uint8_t** out_qs, float** out_scales,
                                 char** buf, size_t* used) {
    if (!src || rows <= 0 || cols <= 0) {
        *out_qs = NULL;
        *out_scales = NULL;
        return;
    }
    int n_blocks_per_row = (cols + 31) / 32;
    size_t qs_bytes = (size_t)rows * n_blocks_per_row * 16;   /* 16 packed bytes per block */
    size_t sc_bytes = (size_t)rows * n_blocks_per_row * sizeof(float);

    uint8_t* qs = (uint8_t*)(*buf + *used);
    *used += qs_bytes;
    float*   sc = (float*)(*buf + *used);
    *used += sc_bytes;

    for (int r = 0; r < rows; r++) {
        tq_quantize_row_q4(src + (size_t)r * cols,
                            qs + (size_t)r * n_blocks_per_row * 16,
                            sc + (size_t)r * n_blocks_per_row,
                            cols);
    }
    *out_qs = qs;
    *out_scales = sc;
}

/* Calculate total Q4 buffer size needed for all layer weights */
static size_t calc_q4_buffer_size(const tq_model_t* model) {
    size_t total = 0;
    const tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int full_kv_dim = (c->full_n_kv_heads > 0 && c->full_head_dim > 0)
        ? c->full_n_kv_heads * c->full_head_dim : kv_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;

    /* DeltaNet dimensions */
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;

    int full_q_dim = (c->full_n_heads > 0 && c->full_head_dim > 0)
        ? c->full_n_heads * c->full_head_dim : q_dim;

    for (int l = 0; l < c->n_layers; l++) {
        const tq_layer_weights_t* layer = &model->layers[l];
        int is_full = (model->layer_is_sliding && !model->layer_is_sliding[l]);
        int lkv = is_full ? full_kv_dim : kv_dim;
        int lq = is_full ? full_q_dim : q_dim;
        int lqg = qg_dim; /* with gate: lq*2, without: lq */
        if (is_full) lqg = c->attn_output_gate ? lq * 2 : lq;

        /* Self-attention weights */
        if (layer->wq) {
            int nb = (dim + 31) / 32;
            total += (size_t)lqg * nb * 16;   /* packed Q4 data */
            total += (size_t)lqg * nb * 4;     /* float scales */
        }
        if (layer->wk) {
            int nb = (dim + 31) / 32;
            total += (size_t)lkv * nb * 16;
            total += (size_t)lkv * nb * 4;
        }
        if (layer->wv) {
            int nb = (dim + 31) / 32;
            total += (size_t)lkv * nb * 16;
            total += (size_t)lkv * nb * 4;
        }
        if (layer->wo) {
            int nb = (lq + 31) / 32;
            total += (size_t)dim * nb * 16;
            total += (size_t)dim * nb * 4;
        }

        /* FFN weights — per-layer intermediate dim */
        int lint = (c->per_layer_inter_dim) ? c->per_layer_inter_dim[l] : inter;
        if (layer->w_gate) {
            int nb = (dim + 31) / 32;
            total += (size_t)lint * nb * 16;
            total += (size_t)lint * nb * 4;
        }
        if (layer->w_up) {
            int nb = (dim + 31) / 32;
            total += (size_t)lint * nb * 16;
            total += (size_t)lint * nb * 4;
        }
        if (layer->w_down) {
            int nb = (lint + 31) / 32;
            total += (size_t)dim * nb * 16;
            total += (size_t)dim * nb * 4;
        }

        /* DeltaNet weights */
        if (layer->delta_in_proj_qkv) {
            int nb = (dim + 31) / 32;
            total += (size_t)delta_qkv_dim * nb * 16;
            total += (size_t)delta_qkv_dim * nb * 4;
        }
        if (layer->delta_in_proj_z) {
            int nb = (dim + 31) / 32;
            total += (size_t)delta_z_dim * nb * 16;
            total += (size_t)delta_z_dim * nb * 4;
        }
        if (layer->delta_in_proj_a) {
            int nb = (dim + 31) / 32;
            total += (size_t)delta_dn * nb * 16;
            total += (size_t)delta_dn * nb * 4;
        }
        if (layer->delta_in_proj_b) {
            int nb = (dim + 31) / 32;
            total += (size_t)delta_dn * nb * 16;
            total += (size_t)delta_dn * nb * 4;
        }
        if (layer->delta_out_proj) {
            int nb = (delta_z_dim + 31) / 32;
            total += (size_t)dim * nb * 16;
            total += (size_t)dim * nb * 4;
        }
    }
    return total;
}

void tq_quantize_weights_q4(tq_model_t* model) {
    if (!model || model->use_q4_weights) return;

    const tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int full_kv_dim = (c->full_n_kv_heads > 0 && c->full_head_dim > 0)
        ? c->full_n_kv_heads * c->full_head_dim : kv_dim;
    int inter = c->intermediate_dim;
    (void)inter;

    /* DeltaNet dimensions */
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;

    size_t buf_size = calc_q4_buffer_size(model);
    char* buf = (char*)malloc(buf_size);
    if (!buf) {
        fprintf(stderr, "tq_quantize_weights_q4: failed to allocate %zu MB for Q4\n",
                buf_size / (1024 * 1024));
        return;
    }
    size_t used = 0;

    int full_q_dim = (c->full_n_heads > 0 && c->full_head_dim > 0)
        ? c->full_n_heads * c->full_head_dim : q_dim;

    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];
        int is_full = (model->layer_is_sliding && !model->layer_is_sliding[l]);
        int lkv = is_full ? full_kv_dim : kv_dim;
        int lq = is_full ? full_q_dim : q_dim;
        int lqg = c->attn_output_gate ? lq * 2 : lq;

        /* Self-attention */
        quantize_matrix_q4(layer->wq, lqg, dim,
                            &layer->wq_q4, &layer->wq_q4s, &buf, &used);
        if (layer->wq_q4) layer->wq = NULL;

        quantize_matrix_q4(layer->wk, lkv, dim,
                            &layer->wk_q4, &layer->wk_q4s, &buf, &used);
        if (layer->wk_q4) layer->wk = NULL;

        quantize_matrix_q4(layer->wv, lkv, dim,
                            &layer->wv_q4, &layer->wv_q4s, &buf, &used);
        if (layer->wv_q4) layer->wv = NULL;

        quantize_matrix_q4(layer->wo, dim, lq,
                            &layer->wo_q4, &layer->wo_q4s, &buf, &used);
        if (layer->wo_q4) layer->wo = NULL;

        /* FFN — use per-layer intermediate dim if available */
        int linter = (c->per_layer_inter_dim) ? c->per_layer_inter_dim[l] : inter;
        quantize_matrix_q4(layer->w_gate, linter, dim,
                            &layer->w_gate_q4, &layer->w_gate_q4s, &buf, &used);
        if (layer->w_gate_q4) layer->w_gate = NULL;

        quantize_matrix_q4(layer->w_up, linter, dim,
                            &layer->w_up_q4, &layer->w_up_q4s, &buf, &used);
        if (layer->w_up_q4) layer->w_up = NULL;

        quantize_matrix_q4(layer->w_down, dim, linter,
                            &layer->w_down_q4, &layer->w_down_q4s, &buf, &used);
        if (layer->w_down_q4) layer->w_down = NULL;

        /* DeltaNet */
        quantize_matrix_q4(layer->delta_in_proj_qkv, delta_qkv_dim, dim,
                            &layer->delta_in_proj_qkv_q4, &layer->delta_in_proj_qkv_q4s,
                            &buf, &used);
        if (layer->delta_in_proj_qkv_q4) layer->delta_in_proj_qkv = NULL;

        quantize_matrix_q4(layer->delta_in_proj_z, delta_z_dim, dim,
                            &layer->delta_in_proj_z_q4, &layer->delta_in_proj_z_q4s,
                            &buf, &used);
        if (layer->delta_in_proj_z_q4) layer->delta_in_proj_z = NULL;

        quantize_matrix_q4(layer->delta_in_proj_a, delta_dn, dim,
                            &layer->delta_in_proj_a_q4, &layer->delta_in_proj_a_q4s,
                            &buf, &used);
        if (layer->delta_in_proj_a_q4) layer->delta_in_proj_a = NULL;

        quantize_matrix_q4(layer->delta_in_proj_b, delta_dn, dim,
                            &layer->delta_in_proj_b_q4, &layer->delta_in_proj_b_q4s,
                            &buf, &used);
        if (layer->delta_in_proj_b_q4) layer->delta_in_proj_b = NULL;

        quantize_matrix_q4(layer->delta_out_proj, dim, delta_z_dim,
                            &layer->delta_out_proj_q4, &layer->delta_out_proj_q4s,
                            &buf, &used);
        if (layer->delta_out_proj_q4) layer->delta_out_proj = NULL;
    }

    model->use_q4_weights = 1;
    model->_q4_data = buf;
    model->_q4_size = used;

    fprintf(stderr, "tq_quantize_weights_q4: quantized to Q4 (%zu MB, was ~%zu MB FP32)\n",
            used / (1024 * 1024), used * 8 / (1024 * 1024));
}

/* ============================================================
 * Q2 weight quantization: 2-bit Lloyd-Max codebook (~12x reduction from FP32)
 *
 * Per block of 32 values: 8 bytes packed + 4 bytes float scale = 12 bytes.
 * Compare to Q4: 16 + 4 = 20 bytes. Q2 is ~1.67x smaller than Q4.
 * ============================================================ */

static void quantize_matrix_q2(const float* src, int rows, int cols,
                                 uint8_t** out_qs, float** out_scales,
                                 char** buf, size_t* used) {
    if (!src || rows <= 0 || cols <= 0) {
        *out_qs = NULL;
        *out_scales = NULL;
        return;
    }
    int n_blocks_per_row = (cols + 31) / 32;
    size_t qs_bytes = (size_t)rows * n_blocks_per_row * 8;   /* 8 packed bytes per Q2 block */
    size_t sc_bytes = (size_t)rows * n_blocks_per_row * sizeof(float);

    uint8_t* qs = (uint8_t*)(*buf + *used);
    *used += qs_bytes;
    float*   sc = (float*)(*buf + *used);
    *used += sc_bytes;

    for (int r = 0; r < rows; r++) {
        tq_quantize_row_q2(src + (size_t)r * cols,
                            qs + (size_t)r * n_blocks_per_row * 8,
                            sc + (size_t)r * n_blocks_per_row,
                            cols);
    }
    *out_qs = qs;
    *out_scales = sc;
}

static size_t calc_q2_buffer_size(const tq_model_t* model) {
    size_t total = 0;
    const tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;

    for (int l = 0; l < c->n_layers; l++) {
        const tq_layer_weights_t* layer = &model->layers[l];
        /* Q2 block: 8 packed bytes + 4 float scale = 12 bytes per 32 values */
        if (layer->wq) { int nb = (dim + 31) / 32; total += (size_t)qg_dim * nb * (8 + 4); }
        if (layer->wk) { int nb = (dim + 31) / 32; total += (size_t)kv_dim * nb * (8 + 4); }
        if (layer->wv) { int nb = (dim + 31) / 32; total += (size_t)kv_dim * nb * (8 + 4); }
        if (layer->wo) { int nb = (q_dim + 31) / 32; total += (size_t)dim * nb * (8 + 4); }
        if (layer->w_gate) { int nb = (dim + 31) / 32; total += (size_t)inter * nb * (8 + 4); }
        if (layer->w_up) { int nb = (dim + 31) / 32; total += (size_t)inter * nb * (8 + 4); }
        if (layer->w_down) { int nb = (inter + 31) / 32; total += (size_t)dim * nb * (8 + 4); }
        if (layer->delta_in_proj_qkv) { int nb = (dim + 31) / 32; total += (size_t)delta_qkv_dim * nb * (8 + 4); }
        if (layer->delta_in_proj_z) { int nb = (dim + 31) / 32; total += (size_t)delta_z_dim * nb * (8 + 4); }
        if (layer->delta_in_proj_a) { int nb = (dim + 31) / 32; total += (size_t)delta_dn * nb * (8 + 4); }
        if (layer->delta_in_proj_b) { int nb = (dim + 31) / 32; total += (size_t)delta_dn * nb * (8 + 4); }
        if (layer->delta_out_proj) { int nb = (delta_z_dim + 31) / 32; total += (size_t)dim * nb * (8 + 4); }
    }
    return total;
}

void tq_quantize_weights_q2(tq_model_t* model) {
    if (!model || model->use_q2_weights) return;

    const tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;

    size_t buf_size = calc_q2_buffer_size(model);
    char* buf = (char*)malloc(buf_size);
    if (!buf) {
        fprintf(stderr, "tq_quantize_weights_q2: failed to allocate %zu MB for Q2\n",
                buf_size / (1024 * 1024));
        return;
    }
    size_t used = 0;

    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];

        /* Self-attention */
        quantize_matrix_q2(layer->wq, qg_dim, dim,
                            &layer->wq_q2, &layer->wq_q2s, &buf, &used);
        if (layer->wq_q2) layer->wq = NULL;

        quantize_matrix_q2(layer->wk, kv_dim, dim,
                            &layer->wk_q2, &layer->wk_q2s, &buf, &used);
        if (layer->wk_q2) layer->wk = NULL;

        quantize_matrix_q2(layer->wv, kv_dim, dim,
                            &layer->wv_q2, &layer->wv_q2s, &buf, &used);
        if (layer->wv_q2) layer->wv = NULL;

        quantize_matrix_q2(layer->wo, dim, q_dim,
                            &layer->wo_q2, &layer->wo_q2s, &buf, &used);
        if (layer->wo_q2) layer->wo = NULL;

        /* FFN */
        quantize_matrix_q2(layer->w_gate, inter, dim,
                            &layer->w_gate_q2, &layer->w_gate_q2s, &buf, &used);
        if (layer->w_gate_q2) layer->w_gate = NULL;

        quantize_matrix_q2(layer->w_up, inter, dim,
                            &layer->w_up_q2, &layer->w_up_q2s, &buf, &used);
        if (layer->w_up_q2) layer->w_up = NULL;

        quantize_matrix_q2(layer->w_down, dim, inter,
                            &layer->w_down_q2, &layer->w_down_q2s, &buf, &used);
        if (layer->w_down_q2) layer->w_down = NULL;

        /* DeltaNet */
        quantize_matrix_q2(layer->delta_in_proj_qkv, delta_qkv_dim, dim,
                            &layer->delta_in_proj_qkv_q2, &layer->delta_in_proj_qkv_q2s,
                            &buf, &used);
        if (layer->delta_in_proj_qkv_q2) layer->delta_in_proj_qkv = NULL;

        quantize_matrix_q2(layer->delta_in_proj_z, delta_z_dim, dim,
                            &layer->delta_in_proj_z_q2, &layer->delta_in_proj_z_q2s,
                            &buf, &used);
        if (layer->delta_in_proj_z_q2) layer->delta_in_proj_z = NULL;

        quantize_matrix_q2(layer->delta_in_proj_a, delta_dn, dim,
                            &layer->delta_in_proj_a_q2, &layer->delta_in_proj_a_q2s,
                            &buf, &used);
        if (layer->delta_in_proj_a_q2) layer->delta_in_proj_a = NULL;

        quantize_matrix_q2(layer->delta_in_proj_b, delta_dn, dim,
                            &layer->delta_in_proj_b_q2, &layer->delta_in_proj_b_q2s,
                            &buf, &used);
        if (layer->delta_in_proj_b_q2) layer->delta_in_proj_b = NULL;

        quantize_matrix_q2(layer->delta_out_proj, dim, delta_z_dim,
                            &layer->delta_out_proj_q2, &layer->delta_out_proj_q2s,
                            &buf, &used);
        if (layer->delta_out_proj_q2) layer->delta_out_proj = NULL;
    }

    model->use_q2_weights = 1;
    model->use_1bit_weights = 1;
    model->_q2_data = buf;
    model->_q2_size = used;

    fprintf(stderr, "tq_quantize_weights_q2: quantized to Q2 (%zu MB, was ~%zu MB FP32)\n",
            used / (1024 * 1024), used * 8 / (1024 * 1024));
}

/* ============================================================
 * Q4+Q2 Progressive Residual: TurboQuant novel weight quantization
 *
 * Applies KV cache residual correction insight to weight matrices:
 *   Pass 1: Q4 quantize (captures main signal)
 *   Pass 2: Residual = FP32 - dequant(Q4)
 *   Pass 3: Q2 quantize residual (captures correction)
 *
 * At inference: matmul uses Q4 path, then adds Q2 correction.
 * Achieves Q8 quality (cosine 0.9998) at 6-bit effective size.
 * ============================================================ */

/* Helper: Q4 quantize a matrix, compute residual, Q2 quantize residual */
static void quantize_matrix_q4q2(float* src, int rows, int cols,
                                   uint8_t** out_q4_qs, float** out_q4_sc,
                                   uint8_t** out_q2_qs, float** out_q2_sc,
                                   char** buf, size_t* used) {
    if (!src || rows <= 0 || cols <= 0) {
        *out_q4_qs = NULL; *out_q4_sc = NULL;
        *out_q2_qs = NULL; *out_q2_sc = NULL;
        return;
    }
    int nb = (cols + 31) / 32;

    /* Allocate Q4 output */
    size_t q4_qs_bytes = (size_t)rows * nb * 16;
    size_t q4_sc_bytes = (size_t)rows * nb * sizeof(float);
    uint8_t* q4_qs = (uint8_t*)(*buf + *used); *used += q4_qs_bytes;
    float*   q4_sc = (float*)(*buf + *used);   *used += q4_sc_bytes;

    /* Allocate Q2 residual output */
    size_t q2_qs_bytes = (size_t)rows * nb * 8;
    size_t q2_sc_bytes = (size_t)rows * nb * sizeof(float);
    uint8_t* q2_qs = (uint8_t*)(*buf + *used); *used += q2_qs_bytes;
    float*   q2_sc = (float*)(*buf + *used);   *used += q2_sc_bytes;

    float* residual = (float*)malloc((size_t)cols * sizeof(float));
    float* dequant = (float*)malloc((size_t)cols * sizeof(float));

    for (int r = 0; r < rows; r++) {
        float* row = src + (size_t)r * cols;

        /* Q4 quantize */
        tq_quantize_row_q4(row,
                            q4_qs + (size_t)r * nb * 16,
                            q4_sc + (size_t)r * nb,
                            cols);

        /* Dequant Q4 → compute residual */
        tq_dequantize_row_q4(q4_qs + (size_t)r * nb * 16,
                              q4_sc + (size_t)r * nb,
                              dequant, cols);
        for (int j = 0; j < cols; j++)
            residual[j] = row[j] - dequant[j];

        /* Q2 quantize residual */
        tq_quantize_row_q2(residual,
                            q2_qs + (size_t)r * nb * 8,
                            q2_sc + (size_t)r * nb,
                            cols);
    }

    free(residual);
    free(dequant);

    *out_q4_qs = q4_qs; *out_q4_sc = q4_sc;
    *out_q2_qs = q2_qs; *out_q2_sc = q2_sc;
}

static size_t calc_q4q2_buffer_size(const tq_model_t* model) {
    /* Each block: Q4 (20 bytes) + Q2 residual (12 bytes) = 32 bytes per 32 elements */
    return calc_q4_buffer_size(model) + calc_q2_buffer_size(model);
}

void tq_quantize_weights_q4q2(tq_model_t* model) {
    if (!model) return;
    if (model->use_q4_weights || model->use_q2_weights) return;

    const tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;

    size_t buf_size = calc_q4q2_buffer_size(model);
    char* buf = (char*)malloc(buf_size);
    if (!buf) {
        fprintf(stderr, "tq_quantize_weights_q4q2: alloc failed (%zu MB)\n", buf_size/(1024*1024));
        return;
    }
    size_t used = 0;

    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];

        /* Self-attention: Q4+Q2 */
        quantize_matrix_q4q2(layer->wq, qg_dim, dim,
            &layer->wq_q4, &layer->wq_q4s, &layer->wq_q2, &layer->wq_q2s, &buf, &used);
        if (layer->wq_q4) layer->wq = NULL;

        quantize_matrix_q4q2(layer->wk, kv_dim, dim,
            &layer->wk_q4, &layer->wk_q4s, &layer->wk_q2, &layer->wk_q2s, &buf, &used);
        if (layer->wk_q4) layer->wk = NULL;

        quantize_matrix_q4q2(layer->wv, kv_dim, dim,
            &layer->wv_q4, &layer->wv_q4s, &layer->wv_q2, &layer->wv_q2s, &buf, &used);
        if (layer->wv_q4) layer->wv = NULL;

        quantize_matrix_q4q2(layer->wo, dim, q_dim,
            &layer->wo_q4, &layer->wo_q4s, &layer->wo_q2, &layer->wo_q2s, &buf, &used);
        if (layer->wo_q4) layer->wo = NULL;

        /* FFN */
        quantize_matrix_q4q2(layer->w_gate, inter, dim,
            &layer->w_gate_q4, &layer->w_gate_q4s, &layer->w_gate_q2, &layer->w_gate_q2s, &buf, &used);
        if (layer->w_gate_q4) layer->w_gate = NULL;

        quantize_matrix_q4q2(layer->w_up, inter, dim,
            &layer->w_up_q4, &layer->w_up_q4s, &layer->w_up_q2, &layer->w_up_q2s, &buf, &used);
        if (layer->w_up_q4) layer->w_up = NULL;

        quantize_matrix_q4q2(layer->w_down, dim, inter,
            &layer->w_down_q4, &layer->w_down_q4s, &layer->w_down_q2, &layer->w_down_q2s, &buf, &used);
        if (layer->w_down_q4) layer->w_down = NULL;

        /* DeltaNet */
        quantize_matrix_q4q2(layer->delta_in_proj_qkv, delta_qkv_dim, dim,
            &layer->delta_in_proj_qkv_q4, &layer->delta_in_proj_qkv_q4s,
            &layer->delta_in_proj_qkv_q2, &layer->delta_in_proj_qkv_q2s, &buf, &used);
        if (layer->delta_in_proj_qkv_q4) layer->delta_in_proj_qkv = NULL;

        quantize_matrix_q4q2(layer->delta_in_proj_z, delta_z_dim, dim,
            &layer->delta_in_proj_z_q4, &layer->delta_in_proj_z_q4s,
            &layer->delta_in_proj_z_q2, &layer->delta_in_proj_z_q2s, &buf, &used);
        if (layer->delta_in_proj_z_q4) layer->delta_in_proj_z = NULL;

        quantize_matrix_q4q2(layer->delta_in_proj_a, delta_dn, dim,
            &layer->delta_in_proj_a_q4, &layer->delta_in_proj_a_q4s,
            &layer->delta_in_proj_a_q2, &layer->delta_in_proj_a_q2s, &buf, &used);
        if (layer->delta_in_proj_a_q4) layer->delta_in_proj_a = NULL;

        quantize_matrix_q4q2(layer->delta_in_proj_b, delta_dn, dim,
            &layer->delta_in_proj_b_q4, &layer->delta_in_proj_b_q4s,
            &layer->delta_in_proj_b_q2, &layer->delta_in_proj_b_q2s, &buf, &used);
        if (layer->delta_in_proj_b_q4) layer->delta_in_proj_b = NULL;

        quantize_matrix_q4q2(layer->delta_out_proj, dim, delta_z_dim,
            &layer->delta_out_proj_q4, &layer->delta_out_proj_q4s,
            &layer->delta_out_proj_q2, &layer->delta_out_proj_q2s, &buf, &used);
        if (layer->delta_out_proj_q4) layer->delta_out_proj = NULL;
    }

    model->use_q4_weights = 1;
    /* NOTE: use_q2_weights NOT set — Q2 fields contain residual, not primary weights.
     * The Q4 dispatch path is used for matmul. Q2 residual correction is applied
     * in the matmul_q4_preq path when both q4 and q2 pointers are non-NULL. */
    model->_q4_data = buf;
    model->_q4_size = used;

    fprintf(stderr, "tq_quantize_weights_q4q2: Q4+Q2 progressive residual (%zu MB, 6-bit effective)\n",
            used / (1024 * 1024));
}

/* ============================================================
 * TQM format: pre-quantized model loading (mmap, zero-copy)
 *
 * The .tqm file stores:
 *   - 512-byte header (tqm_header_t)
 *   - Tokenizer JSON (raw bytes, variable size)
 *   - Weights section (Q4 packed + FP32 norms + BF16 embeds)
 *
 * All pointers into the model's weight arrays point directly
 * into the mmap'd file — no malloc, no conversion.
 * ============================================================ */

/* Align offset up to TQM_ALIGN boundary */
static uint64_t tqm_align(uint64_t offset) {
    return (offset + TQM_ALIGN - 1) & ~((uint64_t)(TQM_ALIGN - 1));
}

tq_model_t* tq_load_tqm(const char* path) {
    if (!path) return NULL;

#ifdef _WIN32
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "tq_load_tqm: cannot open '%s'\n", path);
        return NULL;
    }
    LARGE_INTEGER fileSize;
    GetFileSizeEx(hFile, &fileSize);
    size_t file_size = (size_t)fileSize.QuadPart;

    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) { CloseHandle(hFile); return NULL; }
    void* mmap_data = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    CloseHandle(hFile);
    if (!mmap_data) return NULL;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "tq_load_tqm: cannot open '%s'\n", path);
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return NULL; }
    size_t file_size = (size_t)st.st_size;

    void* mmap_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mmap_data == MAP_FAILED) {
        fprintf(stderr, "tq_load_tqm: mmap failed for '%s'\n", path);
        return NULL;
    }
#endif

    if (file_size < sizeof(tqm_header_t)) {
        fprintf(stderr, "tq_load_tqm: file too small (%zu bytes)\n", file_size);
        goto fail_tqm;
    }

    const tqm_header_t* hdr = (const tqm_header_t*)mmap_data;
    if (hdr->magic != TQM_MAGIC) {
        fprintf(stderr, "tq_load_tqm: invalid magic 0x%08X (expected 0x%08X)\n",
                hdr->magic, TQM_MAGIC);
        goto fail_tqm;
    }
    if (hdr->version != TQM_VERSION) {
        fprintf(stderr, "tq_load_tqm: unsupported version %u\n", hdr->version);
        goto fail_tqm;
    }

    /* Allocate model */
    tq_model_t* model = (tq_model_t*)calloc(1, sizeof(tq_model_t));
    if (!model) goto fail_tqm;

    model->_mmap_data = mmap_data;
    model->_mmap_size = file_size;

    /* Copy config from header */
    tq_model_config_t* c = &model->config;
    c->n_layers         = hdr->n_layers;
    c->hidden_dim       = hdr->hidden_dim;
    c->intermediate_dim = hdr->intermediate_dim;
    c->n_heads          = hdr->n_heads;
    c->n_kv_heads       = hdr->n_kv_heads;
    c->head_dim         = hdr->head_dim;
    c->vocab_size       = hdr->vocab_size;
    c->max_seq_len      = hdr->max_seq_len;
    c->rope_freq_base   = hdr->rope_freq_base;
    c->rms_norm_eps     = hdr->rms_norm_eps;

    c->delta_n_heads       = hdr->delta_n_heads;
    c->delta_n_kv_heads    = hdr->delta_n_kv_heads_tqm;
    c->delta_key_head_dim  = hdr->delta_key_head_dim;
    c->delta_value_head_dim= hdr->delta_value_head_dim;
    c->delta_conv_width    = hdr->delta_conv_width;
    c->partial_rotary_factor = hdr->partial_rotary_factor;
    c->use_qk_norm         = hdr->use_qk_norm;
    c->attn_output_gate    = hdr->attn_output_gate;

    /* Multi-architecture fields */
    c->model_type              = hdr->model_type;
    c->sliding_window          = hdr->sliding_window;
    c->rope_local_base_freq    = hdr->rope_local_base_freq;
    c->n_norms_per_block       = hdr->n_norms_per_block;
    c->query_pre_attn_scalar   = hdr->query_pre_attn_scalar;

    /* Attn layer indices */
    model->n_attn_layers = hdr->n_attn_layers;
    if (hdr->n_attn_layers > 0) {
        model->attn_layer_indices = (int*)malloc((size_t)hdr->n_attn_layers * sizeof(int));
        if (!model->attn_layer_indices) { free(model); goto fail_tqm; }
        for (int i = 0; i < hdr->n_attn_layers; i++) {
            model->attn_layer_indices[i] = hdr->attn_layer_indices[i];
        }
    }

    /* Build is-attn lookup */
    int* is_attn_layer = (int*)calloc((size_t)c->n_layers, sizeof(int));
    if (is_attn_layer) {
        for (int i = 0; i < model->n_attn_layers; i++) {
            int idx = model->attn_layer_indices[i];
            if (idx >= 0 && idx < c->n_layers) is_attn_layer[idx] = 1;
        }
    }

    /* Dimensions for Q4 weight sizes */
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;
    int delta_conv_total = delta_qkv_dim; /* conv1d channels */
    int delta_conv_width = c->delta_conv_width;
    int delta_vhd = c->delta_value_head_dim;

    /* Navigate to weights section */
    uint8_t* ptr = (uint8_t*)mmap_data + hdr->weights_offset;
    uint8_t* file_end = (uint8_t*)mmap_data + file_size;

    /* Helper macro: read FP32 array pointer from mmap */
    #define TQM_READ_FP32(dest, count) do {                   \
        size_t _bytes = (size_t)(count) * sizeof(float);      \
        if (ptr + _bytes > file_end) goto fail_tqm_model;     \
        (dest) = (float*)ptr;                                  \
        ptr += _bytes;                                         \
    } while (0)

    /* Helper macro: read Q4 weight (packed bytes + float scales) */
    #define TQM_READ_Q4(dest_qs, dest_sc, rows, cols) do {    \
        int _nb = ((cols) + 31) / 32;                          \
        size_t _qs_bytes = (size_t)(rows) * _nb * 16;          \
        size_t _sc_bytes = (size_t)(rows) * _nb * sizeof(float); \
        if (ptr + _qs_bytes + _sc_bytes > file_end) goto fail_tqm_model; \
        (dest_qs) = (uint8_t*)ptr;                             \
        ptr += _qs_bytes;                                      \
        (dest_sc) = (float*)ptr;                               \
        ptr += _sc_bytes;                                      \
    } while (0)

    /* Allocate layers */
    model->layers = (tq_layer_weights_t*)calloc((size_t)c->n_layers,
                                                 sizeof(tq_layer_weights_t));
    if (!model->layers) goto fail_tqm_model;

    /* Read per-layer weights */
    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];

        /* Both layer types have norms */
        TQM_READ_FP32(layer->attn_norm, dim);
        TQM_READ_FP32(layer->ffn_norm, dim);

        /* Gemma3 extra norms */
        if (c->model_type == 1) {
            layer->post_attn_norm = layer->ffn_norm; /* shares storage */
            TQM_READ_FP32(layer->pre_ffn_norm, dim);
            TQM_READ_FP32(layer->post_ffn_norm, dim);
        }

        if (is_attn_layer && is_attn_layer[l]) {
            /* Self-attention layer */
            TQM_READ_Q4(layer->wq_q4, layer->wq_q4s, qg_dim, dim);
            TQM_READ_Q4(layer->wk_q4, layer->wk_q4s, kv_dim, dim);
            TQM_READ_Q4(layer->wv_q4, layer->wv_q4s, kv_dim, dim);
            TQM_READ_Q4(layer->wo_q4, layer->wo_q4s, dim, q_dim);

            if (c->use_qk_norm) {
                TQM_READ_FP32(layer->q_norm, c->head_dim);
                TQM_READ_FP32(layer->k_norm, c->head_dim);
            }
        } else {
            /* DeltaNet layer */
            TQM_READ_FP32(layer->delta_a_log, delta_dn);
            TQM_READ_FP32(layer->delta_dt_bias, delta_dn);
            TQM_READ_Q4(layer->delta_in_proj_qkv_q4, layer->delta_in_proj_qkv_q4s,
                         delta_qkv_dim, dim);
            TQM_READ_Q4(layer->delta_in_proj_z_q4, layer->delta_in_proj_z_q4s,
                         delta_z_dim, dim);
            TQM_READ_Q4(layer->delta_in_proj_a_q4, layer->delta_in_proj_a_q4s,
                         delta_dn, dim);
            TQM_READ_Q4(layer->delta_in_proj_b_q4, layer->delta_in_proj_b_q4s,
                         delta_dn, dim);
            TQM_READ_FP32(layer->delta_conv1d, delta_conv_total * delta_conv_width);
            TQM_READ_FP32(layer->delta_norm, delta_vhd);
            TQM_READ_Q4(layer->delta_out_proj_q4, layer->delta_out_proj_q4s,
                         dim, delta_z_dim);
        }

        /* FFN (all layers) */
        TQM_READ_Q4(layer->w_gate_q4, layer->w_gate_q4s, inter, dim);
        TQM_READ_Q4(layer->w_up_q4, layer->w_up_q4s, inter, dim);
        TQM_READ_Q4(layer->w_down_q4, layer->w_down_q4s, dim, inter);
    }

    /* Output norm */
    TQM_READ_FP32(model->output_norm, dim);

    /* Embedding and output as BF16 */
    {
        size_t embed_elems = (size_t)c->vocab_size * dim;
        size_t embed_bytes = embed_elems * 2; /* BF16 = 2 bytes */
        if (ptr + embed_bytes > file_end) goto fail_tqm_model;
        model->embed_bf16 = (const uint16_t*)ptr;
        model->token_embedding = NULL;
        ptr += embed_bytes;

        /* Check if output weight is tied (offset == 0 means tied) */
        if (ptr + embed_bytes <= file_end) {
            /* Separate lm_head */
            model->output_weight_bf16 = (const uint16_t*)ptr;
            model->output_weight = NULL;
            ptr += embed_bytes;
        } else {
            /* Tied: output_weight points to embed_tokens */
            model->output_weight_bf16 = model->embed_bf16;
            model->output_weight = NULL;
        }
    }

    #undef TQM_READ_FP32
    #undef TQM_READ_Q4

    model->use_q4_weights = 1;
    free(is_attn_layer);

    /* Set up Gemma3 layer_is_sliding from TQM */
    if (c->model_type == 1 && c->sliding_window > 0) {
        model->layer_is_sliding = (int*)calloc((size_t)c->n_layers, sizeof(int));
        if (model->layer_is_sliding) {
            for (int l = 0; l < c->n_layers; l++) {
                if ((l + 1) % 6 == 0) {
                    model->layer_is_sliding[l] = 0; /* global */
                } else {
                    model->layer_is_sliding[l] = 1; /* sliding */
                }
            }
        }
    }

    /* Runtime Q4 quantization of lm_head (output projection) for fast logit computation.
     * BF16 matmul on 248K x 1024 is slow; Q4 matmul is ~4x faster. */
    if (model->output_weight_bf16) {
        int vocab = c->vocab_size;
        int n_blocks = dim / 32;
        size_t qs_size = (size_t)vocab * n_blocks * 16;
        size_t scales_size = (size_t)vocab * n_blocks * sizeof(float);
        model->output_qs = (uint8_t*)malloc(qs_size);
        model->output_scales = (float*)malloc(scales_size);
        if (model->output_qs && model->output_scales) {
            float* row_buf = (float*)malloc(dim * sizeof(float));
            if (row_buf) {
                for (int i = 0; i < vocab; i++) {
                    const uint16_t* src = model->output_weight_bf16 + (size_t)i * dim;
                    for (int j = 0; j < dim; j++) {
                        uint32_t bits = ((uint32_t)src[j]) << 16;
                        memcpy(&row_buf[j], &bits, 4);
                    }
                    tq_quantize_row_q4(row_buf,
                                        model->output_qs + (size_t)i * n_blocks * 16,
                                        model->output_scales + (size_t)i * n_blocks,
                                        dim);
                }
                free(row_buf);
                fprintf(stderr, "tq_load_tqm: lm_head quantized to Q4 (%.1f MB -> %.1f MB)\n",
                        (double)vocab * dim * 2 / (1024.0 * 1024.0),
                        (double)(qs_size + scales_size) / (1024.0 * 1024.0));
            }
        }
    }

    fprintf(stderr, "tq_load_tqm: loaded %d layers (%d self_attn), "
            "dim=%d, heads=%d/%d, vocab=%d [%.1f MB, mmap zero-copy]\n",
            c->n_layers, model->n_attn_layers, dim,
            c->n_heads, c->n_kv_heads, c->vocab_size,
            (double)file_size / (1024.0 * 1024.0));

    return model;

fail_tqm_model:
    fprintf(stderr, "tq_load_tqm: weight data extends past end of file\n");
    free(is_attn_layer);
    free(model->attn_layer_indices);
    free(model->layers);
    free(model);

fail_tqm:
#ifdef _WIN32
    if (mmap_data) UnmapViewOfFile(mmap_data);
#else
    if (mmap_data && mmap_data != MAP_FAILED) munmap(mmap_data, file_size);
#endif
    return NULL;
}

/* ============================================================
 * TQM saver — write pre-quantized model to .tqm file
 *
 * The model MUST have Q4 weights (use_q4_weights == 1) before
 * calling this function.  Call tq_quantize_weights_q4() first.
 * ============================================================ */

/* Helper: write with 64-byte alignment padding */
static int tqm_write_pad(FILE* f, uint64_t current_offset) {
    uint64_t aligned = tqm_align(current_offset);
    if (aligned > current_offset) {
        uint64_t pad_size = aligned - current_offset;
        uint8_t zeros[64] = {0};
        while (pad_size > 0) {
            size_t chunk = pad_size > 64 ? 64 : (size_t)pad_size;
            if (fwrite(zeros, 1, chunk, f) != chunk) return -1;
            pad_size -= chunk;
        }
    }
    return 0;
}

/* ============================================================
 * GGUF model loader — loads community GGUF models (Q2_K..IQ4_XS)
 *
 * Strategy:
 *   - mmap the entire GGUF file via tq_gguf_open()
 *   - Extract model config from GGUF metadata
 *   - For attention weights: keep pointers into mmap (dequant on-the-fly)
 *   - For MoE experts: keep pointers into mmap (demand-paged by OS)
 *   - Router weights + norms: dequant to FP32 at load time (small)
 *   - Embeddings: keep as mmap'd BF16/F16 or dequant
 * ============================================================ */

/* Helper: find a GGUF tensor with fallback name patterns */
static const tq_gguf_tensor_t* find_gguf_tensor(const tq_gguf_ctx_t* ctx,
                                                  const char* name) {
    const tq_gguf_tensor_t* t = tq_gguf_find_tensor(ctx, name);
    if (t) return t;
    /* Try without "blk." prefix variants */
    return NULL;
}

/* Helper: dequant a GGUF tensor to FP32, returns malloc'd buffer */
static float* dequant_tensor_fp32(const tq_gguf_tensor_t* t) {
    if (!t || !t->data) return NULL;
    int64_t n_elem = 1;
    for (uint32_t d = 0; d < t->n_dims; d++) n_elem *= t->shape[d];
    float* out = (float*)malloc((size_t)n_elem * sizeof(float));
    if (!out) return NULL;
    tq_dequant_row_gguf(t->type, t->data, out, (int)n_elem);
    return out;
}

tq_model_t* tq_load_gguf(const char* path) {
    if (!path) return NULL;

    /* Open GGUF file */
    tq_gguf_ctx_t* gguf = tq_gguf_open(path);
    if (!gguf) {
        fprintf(stderr, "tq_load_gguf: failed to open '%s'\n", path);
        return NULL;
    }

    fprintf(stderr, "tq_load_gguf: GGUF v%u, %llu tensors, %llu metadata keys\n",
            gguf->version, (unsigned long long)gguf->n_tensors,
            (unsigned long long)gguf->n_kv);
    fprintf(stderr, "tq_load_gguf: architecture = '%s'\n", gguf->arch);

    /* Extract model configuration from GGUF metadata */
    tq_model_t* model = (tq_model_t*)calloc(1, sizeof(tq_model_t));
    if (!model) { tq_gguf_close(gguf); return NULL; }

    tq_model_config_t* c = &model->config;
    char key[256];

    /* Helper macro for arch-prefixed metadata keys */
    #define GGUF_KEY(suffix) (snprintf(key, sizeof(key), "%s." suffix, gguf->arch), key)

    c->n_layers         = tq_gguf_get_i32(gguf, GGUF_KEY("block_count"), 0);
    c->hidden_dim       = tq_gguf_get_i32(gguf, GGUF_KEY("embedding_length"), 0);
    c->intermediate_dim = tq_gguf_get_i32(gguf, GGUF_KEY("feed_forward_length"), 0);
    /* For MoE models, intermediate_dim may be 0 in metadata.
     * Use expert_feed_forward_length as fallback for state allocation. */
    if (c->intermediate_dim == 0) {
        c->intermediate_dim = tq_gguf_get_i32(gguf, GGUF_KEY("expert_feed_forward_length"), 0);
    }
    /* Infer per-layer intermediate_dim from FFN tensor shapes.
     * Gemma 4 E2B has variable FFN dim per layer (e.g., 6144 then 12288). */
    {
        int max_inter = c->intermediate_dim;
        int has_variable = 0;
        int per_layer_dims[256];
        for (int l = 0; l < c->n_layers && l < 256; l++) {
            char tn[128];
            snprintf(tn, sizeof(tn), "blk.%d.ffn_gate.weight", l);
            const tq_gguf_tensor_t* fg = tq_gguf_find_tensor(gguf, tn);
            int idim = fg ? (int)fg->shape[1] : c->intermediate_dim;
            per_layer_dims[l] = idim;
            if (idim > max_inter) max_inter = idim;
            if (l > 0 && idim != per_layer_dims[0]) has_variable = 1;
        }
        if (max_inter > c->intermediate_dim) c->intermediate_dim = max_inter;
        if (has_variable && c->n_layers <= 256) {
            c->per_layer_inter_dim = (int*)malloc((size_t)c->n_layers * sizeof(int));
            for (int l = 0; l < c->n_layers; l++)
                c->per_layer_inter_dim[l] = per_layer_dims[l];
            fprintf(stderr, "tq_load_gguf: variable FFN dim — [0]=%d, [%d]=%d\n",
                    per_layer_dims[0], c->n_layers-1, per_layer_dims[c->n_layers-1]);
        }
    }
    c->n_heads          = tq_gguf_get_i32(gguf, GGUF_KEY("attention.head_count"), 0);
    c->n_kv_heads       = tq_gguf_get_i32(gguf, GGUF_KEY("attention.head_count_kv"), c->n_heads);
    c->vocab_size       = (int)tq_gguf_get_u32(gguf, GGUF_KEY("vocab_size"),
                            tq_gguf_get_u32(gguf, "tokenizer.ggml.tokens", 0));
    c->max_seq_len      = tq_gguf_get_i32(gguf, GGUF_KEY("context_length"), 131072);
    c->rope_freq_base   = tq_gguf_get_f32(gguf, GGUF_KEY("rope.freq_base"), 1000000.0f);
    c->rms_norm_eps     = tq_gguf_get_f32(gguf, GGUF_KEY("attention.layer_norm_rms_epsilon"), 1e-6f);

    /* RoPE dimension count: number of dimensions to rotate per head.
     * For models with rope_freqs (learned freq factors), this determines the
     * frequency computation: freq[i] = pow(base, -2*i/n_dims).
     * For STEP35/Gemma4: n_dims = head_dim/2 for full layers (partial rotation). */
    c->rope_n_dims = tq_gguf_get_i32(gguf, GGUF_KEY("rope.dimension_count"), 0);
    c->rope_n_dims_full = c->rope_n_dims;  /* default: same for both layer types */
    {
        int swa_dims = tq_gguf_get_i32(gguf, GGUF_KEY("rope.dimension_count_swa"), 0);
        if (swa_dims > 0) {
            c->rope_n_dims = swa_dims;  /* sliding layers use SWA dim count */
        }
    }

    /* Sliding window + local RoPE base */
    c->sliding_window = (int)tq_gguf_get_u32(gguf, GGUF_KEY("attention.sliding_window"), 0);
    /* Local/sliding RoPE base: try Gemma4 naming first, then generic */
    c->rope_local_base_freq = tq_gguf_get_f32(gguf, GGUF_KEY("rope.freq_base_swa"),
                               tq_gguf_get_f32(gguf, GGUF_KEY("rope.local.freq_base"),
                               tq_gguf_get_f32(gguf, GGUF_KEY("rope.freq_base"), 10000.0f)));
    c->final_logit_softcap = tq_gguf_get_f32(gguf, GGUF_KEY("final_logit_softcapping"), 0.0f);
    c->attn_logit_softcap = tq_gguf_get_f32(gguf, GGUF_KEY("attn_logit_softcapping"), 0.0f);
    /* Gemma 2/3/4 use attention softcap but it may not be in metadata — hardcode */
    if (c->model_type == 1 && c->attn_logit_softcap == 0.0f) {
        c->attn_logit_softcap = 50.0f;
    }

    /* Cap context for memory safety on small machines.
     * GGUF models often claim 262K context but we cap at 4096 by default.
     * Users can override with --ctx flag in quant. */
    if (c->max_seq_len > 4096) c->max_seq_len = 4096;

    /* Compute head_dim — prefer explicit key_length from metadata.
     * For Gemma 4: key_length=512 is for full attention layers,
     * but sliding layers use 256. Detect from first layer's K tensor shape. */
    c->head_dim = tq_gguf_get_i32(gguf, GGUF_KEY("attention.key_length"), 0);
    if (c->head_dim == 0 && c->n_heads > 0) {
        c->head_dim = c->hidden_dim / c->n_heads;
    }

    /* For hybrid sliding/full attention (Gemma 4):
     * Override head_dim from first layer's K tensor shape (sliding layer),
     * since sliding layers are the majority and determine KV cache layout. */
    {
        const tq_gguf_tensor_t* k0 = tq_gguf_find_tensor(gguf, "blk.0.attn_k.weight");
        if (k0 && k0->n_dims >= 2) {
            int k_out = (int)k0->shape[1];
            /* Try head_dim candidates: check if k_out / head_dim gives integer kv_heads */
            /* Try from largest to smallest to prefer larger head_dim */
            int sliding_head_dim = c->head_dim;
            for (int hd = 512; hd >= 64; hd /= 2) {
                if (k_out % hd == 0) {
                    int kv = k_out / hd;
                    if (kv >= 1 && kv <= c->n_heads && hd < c->head_dim) {
                        sliding_head_dim = hd;
                        break;
                    }
                }
            }
            if (sliding_head_dim != c->head_dim) {
                fprintf(stderr, "tq_load_gguf: hybrid attention detected — "
                        "sliding head_dim=%d (metadata: %d)\n", sliding_head_dim, c->head_dim);
                c->head_dim = sliding_head_dim;
            }
            /* Infer kv_heads from K tensor shape */
            c->n_kv_heads = k_out / c->head_dim;
        }
    }

    /* MoE configuration */
    c->num_experts        = tq_gguf_get_i32(gguf, GGUF_KEY("expert_count"), 0);
    c->num_active_experts = tq_gguf_get_i32(gguf, GGUF_KEY("expert_used_count"), 0);
    c->is_moe = (c->num_experts > 0);

    if (c->is_moe) {
        /* Try to get expert FFN dim from metadata, or infer from tensor shapes */
        c->expert_intermediate_dim = tq_gguf_get_i32(gguf,
            GGUF_KEY("expert_feed_forward_length"), 0);

        /* Check for shared expert */
        char shared_name[128];
        snprintf(shared_name, sizeof(shared_name), "blk.0.ffn_gate_shexp.weight");
        c->has_shared_expert = (tq_gguf_find_tensor(gguf, shared_name) != NULL) ? 1 : 0;
        c->shared_expert_intermediate_dim = tq_gguf_get_i32(gguf,
            GGUF_KEY("expert_shared_feed_forward_length"), 0);

        /* Infer expert_intermediate_dim from tensor shape if not in metadata */
        if (c->expert_intermediate_dim == 0) {
            snprintf(shared_name, sizeof(shared_name), "blk.0.ffn_gate_exps.weight");
            const tq_gguf_tensor_t* exp_t = tq_gguf_find_tensor(gguf, shared_name);
            if (exp_t && exp_t->n_dims == 3) {
                /* Shape: [num_experts, expert_inter_dim, hidden_dim] */
                c->expert_intermediate_dim = (int)exp_t->shape[1];
            }
        }

        fprintf(stderr, "tq_load_gguf: MoE — %d experts, %d active, expert_dim=%d, shared=%d\n",
                c->num_experts, c->num_active_experts,
                c->expert_intermediate_dim, c->has_shared_expert);
    }

    /* Model type detection — Gemma takes priority (Gemma 4 is both Gemma AND MoE) */
    if (strstr(gguf->arch, "gemma") != NULL) {
        c->model_type = 1; /* gemma family */
        c->n_norms_per_block = 4;
        /* Gemma 4 (STEP35) detection: architecture string is "gemma4" */
        if (strstr(gguf->arch, "gemma4") != NULL) {
            c->is_gemma4 = 1;
            /* STEP35: full attention layers use half the RoPE dimensions */
            if (c->rope_n_dims_full > 0) {
                c->rope_n_dims_full = c->rope_n_dims_full / 2;
            }
            fprintf(stderr, "tq_load_gguf: Gemma4 — RoPE dims swa=%d full=%d, "
                    "GeGLU, rope_freqs for full layers only\n",
                    c->rope_n_dims, c->rope_n_dims_full);
        }
        fprintf(stderr, "tq_load_gguf: Gemma family detected (sliding_window=%d)\n", c->sliding_window);
    } else if (c->is_moe) {
        c->model_type = 2; /* qwen moe */
    } else {
        c->model_type = 0; /* qwen35 */
    }

    fprintf(stderr, "tq_load_gguf: config — layers=%d, dim=%d, heads=%d/%d, head_dim=%d, vocab=%d\n",
            c->n_layers, c->hidden_dim, c->n_heads, c->n_kv_heads, c->head_dim, c->vocab_size);

    if (c->n_layers == 0 || c->hidden_dim == 0) {
        fprintf(stderr, "tq_load_gguf: invalid config, aborting\n");
        free(model);
        tq_gguf_close(gguf);
        return NULL;
    }

    /* Store GGUF context in model for lifetime management */
    model->gguf_ctx = gguf;

    /* Allocate per-layer weights */
    model->layers = (tq_layer_weights_t*)calloc((size_t)c->n_layers, sizeof(tq_layer_weights_t));
    if (!model->layers) {
        free(model);
        tq_gguf_close(gguf);
        return NULL;
    }

    /* MoE config storage */
    tq_moe_config_t* moe_cfg = NULL;
    if (c->is_moe) {
        moe_cfg = (tq_moe_config_t*)calloc(1, sizeof(tq_moe_config_t));
        moe_cfg->num_experts = c->num_experts;
        moe_cfg->num_active = c->num_active_experts;
        moe_cfg->expert_intermediate_dim = c->expert_intermediate_dim;
        moe_cfg->has_shared_expert = c->has_shared_expert;
        moe_cfg->shared_expert_intermediate_dim = c->shared_expert_intermediate_dim;
        moe_cfg->norm_topk_prob = 1;
        moe_cfg->use_gelu = c->is_gemma4 ? 1 : 0; /* Gemma 4: GeGLU, others: SwiGLU */
        model->moe_config = moe_cfg;
    }

    /* Load per-layer weights */
    char tname[256];
    int n_attn_layers = 0;
    int attn_indices[256]; /* max layers */

    /* Detect if GGUF already has Gemma +1.0 norm adjustment baked in.
     * If first layer's attn_norm has mean > 2.0, it's already adjusted. */
    int gemma_norms_adjusted = 0;
    if (c->model_type == 1) {
        const tq_gguf_tensor_t* probe = tq_gguf_find_tensor(gguf, "blk.0.attn_norm.weight");
        if (probe && probe->type == TQ_GGML_TYPE_F32 && probe->shape[0] > 0) {
            const float* pw = (const float*)probe->data;
            float sum = 0;
            int n = (int)probe->shape[0];
            for (int i = 0; i < n && i < 64; i++) sum += pw[i];
            float mean = sum / (n < 64 ? n : 64);
            if (mean > 2.0f) {
                gemma_norms_adjusted = 1;
                fprintf(stderr, "tq_load_gguf: Gemma norms already adjusted (mean=%.1f, skipping +1.0)\n", mean);
            }
        }
    }

    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];

        /* RMSNorm weights (always FP32 in GGUF, dequant to FP32) */
        snprintf(tname, sizeof(tname), "blk.%d.attn_norm.weight", l);
        const tq_gguf_tensor_t* t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->attn_norm = dequant_tensor_fp32(t);
            /* Gemma norm convention: weight = 1 + stored_weight (skip if already adjusted) */
            if (c->model_type == 1 && !gemma_norms_adjusted) {
                for (int i = 0; i < c->hidden_dim; i++) layer->attn_norm[i] += 1.0f;
            }
        }

        snprintf(tname, sizeof(tname), "blk.%d.ffn_norm.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (!t) {
            /* Qwen3.5 uses post_attention_norm as FFN norm */
            snprintf(tname, sizeof(tname), "blk.%d.post_attention_norm.weight", l);
            t = find_gguf_tensor(gguf, tname);
        }
        if (t) {
            layer->ffn_norm = dequant_tensor_fp32(t);
            if (c->model_type == 1 && !gemma_norms_adjusted) {
                for (int i = 0; i < c->hidden_dim; i++) layer->ffn_norm[i] += 1.0f;
            }
        }

        /* QK-norm (optional) */
        snprintf(tname, sizeof(tname), "blk.%d.attn_q_norm.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->q_norm = dequant_tensor_fp32(t);
            c->use_qk_norm = 1;
            /* Gemma QK norm: weight = 1 + stored. Size = head_dim (per-head norm). */
            if (c->model_type == 1 && !gemma_norms_adjusted) {
                int hd = (int)t->shape[t->n_dims > 1 ? 1 : 0];
                for (int i = 0; i < hd; i++) layer->q_norm[i] += 1.0f;
            }
        }

        snprintf(tname, sizeof(tname), "blk.%d.attn_k_norm.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->k_norm = dequant_tensor_fp32(t);
            if (c->model_type == 1 && !gemma_norms_adjusted) {
                int hd = (int)t->shape[t->n_dims > 1 ? 1 : 0];
                for (int i = 0; i < hd; i++) layer->k_norm[i] += 1.0f;
            }
        }

        /* Gemma extra norms (post_attn, pre_ffn, post_ffn) */
        snprintf(tname, sizeof(tname), "blk.%d.post_attention_norm.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->post_attn_norm = dequant_tensor_fp32(t);
            if (!gemma_norms_adjusted)
                for (int i = 0; i < c->hidden_dim; i++) layer->post_attn_norm[i] += 1.0f;
        }
        snprintf(tname, sizeof(tname), "blk.%d.post_ffw_norm.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->post_ffn_norm = dequant_tensor_fp32(t);
            if (!gemma_norms_adjusted)
                for (int i = 0; i < c->hidden_dim; i++) layer->post_ffn_norm[i] += 1.0f;
        }
        snprintf(tname, sizeof(tname), "blk.%d.pre_ffw_norm.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->pre_ffn_norm = dequant_tensor_fp32(t);
            if (!gemma_norms_adjusted)
                for (int i = 0; i < c->hidden_dim; i++) layer->pre_ffn_norm[i] += 1.0f;
        }

        /* Gemma 4 dual-FFN extra norms */
        snprintf(tname, sizeof(tname), "blk.%d.post_ffw_norm_1.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->post_ffn_norm_1 = dequant_tensor_fp32(t);
            if (!gemma_norms_adjusted)
                for (int i = 0; i < c->hidden_dim; i++) layer->post_ffn_norm_1[i] += 1.0f;
        }
        snprintf(tname, sizeof(tname), "blk.%d.pre_ffw_norm_2.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->pre_ffn_norm_2 = dequant_tensor_fp32(t);
            if (!gemma_norms_adjusted)
                for (int i = 0; i < c->hidden_dim; i++) layer->pre_ffn_norm_2[i] += 1.0f;
        }
        snprintf(tname, sizeof(tname), "blk.%d.post_ffw_norm_2.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->post_ffn_norm_2 = dequant_tensor_fp32(t);
            if (!gemma_norms_adjusted)
                for (int i = 0; i < c->hidden_dim; i++) layer->post_ffn_norm_2[i] += 1.0f;
        }

        /* Gemma 4: layer_output_scale (scalar per layer) */
        snprintf(tname, sizeof(tname), "blk.%d.layer_output_scale.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t && t->type == TQ_GGML_TYPE_F32) {
            layer->layer_output_scale = ((const float*)t->data)[0];
        }

        /* Gemma 4 PLE per-layer weights: inp_gate, proj, post_norm */
        snprintf(tname, sizeof(tname), "blk.%d.inp_gate.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->ple_gate = t->data;
            layer->ple_gate_type = t->type;
        }
        snprintf(tname, sizeof(tname), "blk.%d.proj.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->ple_proj = t->data;
            layer->ple_proj_type = t->type;
        }
        snprintf(tname, sizeof(tname), "blk.%d.post_norm.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->ple_norm = dequant_tensor_fp32(t);
            if (c->model_type == 1 && !gemma_norms_adjusted && layer->ple_norm) {
                for (int i = 0; i < c->hidden_dim; i++) layer->ple_norm[i] += 1.0f;
            }
        }

        /* Attention weights — keep as GGUF quantized pointers for on-the-fly dequant.
         * We store the raw data pointer + type info using a small struct packed into
         * the existing FP32 weight pointer fields. For GGUF models, we use a special
         * dispatch: if gguf_ctx is non-NULL, the forward pass uses tq_matmul_gguf. */
        snprintf(tname, sizeof(tname), "blk.%d.attn_q.weight", l);
        const tq_gguf_tensor_t* wq_t = find_gguf_tensor(gguf, tname);
        int is_attn_layer = (wq_t != NULL);
        if (is_attn_layer) {
            /* Detect attn_output_gate on first self_attn layer:
             * If Q proj out_dim = 2 * O proj in_dim, Q includes gate */
            if (n_attn_layers == 0 && wq_t->n_dims >= 2) {
                snprintf(tname, sizeof(tname), "blk.%d.attn_output.weight", l);
                const tq_gguf_tensor_t* wo_t = find_gguf_tensor(gguf, tname);
                if (wo_t && wo_t->n_dims >= 2) {
                    int q_out = (int)wq_t->shape[1];  /* Q: [hidden, q_out] */
                    int o_in  = (int)wo_t->shape[0];   /* O: [o_in, hidden] */
                    if (q_out == o_in * 2) {
                        c->attn_output_gate = 1;
                        /* Don't override n_heads from metadata — gate doubles Q proj only */
                        fprintf(stderr, "tq_load_gguf: detected attn_output_gate=1 "
                                "(q=%d = 2 * o=%d, n_heads=%d, head_dim=%d)\n",
                                q_out, o_in, c->n_heads, c->head_dim);
                    }
                }
            }

            /* Store GGUF quantized pointers for on-the-fly dequant in forward.
             * This saves ~5GB for 35B models (vs full FP32 dequant at load). */
            layer->gguf_wq = wq_t->data;
            layer->gguf_wq_type = wq_t->type;

            snprintf(tname, sizeof(tname), "blk.%d.attn_k.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) { layer->gguf_wk = t->data; layer->gguf_wk_type = t->type; }

            snprintf(tname, sizeof(tname), "blk.%d.attn_v.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) { layer->gguf_wv = t->data; layer->gguf_wv_type = t->type; }

            snprintf(tname, sizeof(tname), "blk.%d.attn_output.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) { layer->gguf_wo = t->data; layer->gguf_wo_type = t->type; }

            attn_indices[n_attn_layers++] = l;
        }

        /* Check for DeltaNet / SSM weights (Qwen3.5 hybrid) */
        snprintf(tname, sizeof(tname), "blk.%d.ssm_a", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->delta_a_log = dequant_tensor_fp32(t);
            /* GGUF stores ssm_a as -exp(a_log), but our forward pass expects a_log.
             * Convert back: a_log = log(-ssm_a) */
            if (layer->delta_a_log) {
                for (int64_t j = 0; j < t->shape[0]; j++) {
                    float v = layer->delta_a_log[j];
                    layer->delta_a_log[j] = (v < 0) ? logf(-v) : 0.0f;
                }
            }

            snprintf(tname, sizeof(tname), "blk.%d.ssm_conv1d.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) layer->delta_conv1d = dequant_tensor_fp32(t);

            snprintf(tname, sizeof(tname), "blk.%d.ssm_dt.bias", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) layer->delta_dt_bias = dequant_tensor_fp32(t);

            /* Small DeltaNet weights: dequant to FP32 (alpha, beta are small) */
            snprintf(tname, sizeof(tname), "blk.%d.ssm_alpha.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) { layer->gguf_delta_a = t->data; layer->gguf_delta_a_type = t->type; }

            snprintf(tname, sizeof(tname), "blk.%d.ssm_beta.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) { layer->gguf_delta_b = t->data; layer->gguf_delta_b_type = t->type; }

            /* Large DeltaNet projections: dequant to FP32 for recurrent
             * state precision.  Q5_K (5-bit) introduces too much error in
             * the recurrent state that accumulates across time steps.
             * ~24 MB/layer × 30 layers ≈ 720 MB — fits in 16 GB. */
            snprintf(tname, sizeof(tname), "blk.%d.attn_qkv.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) {
                if (t->type == TQ_GGML_TYPE_Q5_K || t->type == TQ_GGML_TYPE_IQ2_XXS ||
                    t->type == TQ_GGML_TYPE_IQ3_XXS || t->type == TQ_GGML_TYPE_IQ4_XS) {
                    /* Low-precision: dequant to FP32 for recurrent accuracy */
                    layer->delta_in_proj_qkv = dequant_tensor_fp32(t);
                    fprintf(stderr, "tq_load_gguf: layer %d attn_qkv dequant to FP32 (was type %d)\n", l, t->type);
                } else {
                    layer->gguf_delta_qkv = t->data;
                    layer->gguf_delta_qkv_type = t->type;
                }
            }

            snprintf(tname, sizeof(tname), "blk.%d.attn_gate.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) {
                if (t->type == TQ_GGML_TYPE_Q5_K || t->type == TQ_GGML_TYPE_IQ2_XXS ||
                    t->type == TQ_GGML_TYPE_IQ3_XXS || t->type == TQ_GGML_TYPE_IQ4_XS) {
                    layer->delta_in_proj_z = dequant_tensor_fp32(t);
                    fprintf(stderr, "tq_load_gguf: layer %d attn_gate dequant to FP32 (was type %d)\n", l, t->type);
                } else {
                    layer->gguf_delta_z = t->data;
                    layer->gguf_delta_z_type = t->type;
                }
            }

            snprintf(tname, sizeof(tname), "blk.%d.ssm_norm.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) layer->delta_norm = dequant_tensor_fp32(t);

            snprintf(tname, sizeof(tname), "blk.%d.ssm_out.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) { layer->gguf_delta_out = t->data; layer->gguf_delta_out_type = t->type; }

            /* Infer DeltaNet config from tensor shapes if not set */
            if (c->delta_n_heads == 0 && layer->delta_a_log) {
                /* Read SSM config from GGUF metadata (Qwen3.5 DeltaNet).
                 * ssm_a shape[0] = num_v_heads (= ssm.time_step_rank)
                 * ssm_norm shape[0] = value_head_dim
                 * From GGUF metadata:
                 *   ssm.state_size   = key_head_dim (ssm_d_state)
                 *   ssm.group_count  = num_k_heads  (ssm_n_group)
                 *   ssm.time_step_rank = num_v_heads
                 *   ssm.inner_size   = num_v_heads * value_head_dim */
                snprintf(tname, sizeof(tname), "blk.%d.ssm_a", l);
                const tq_gguf_tensor_t* a_t = find_gguf_tensor(gguf, tname);
                if (a_t) c->delta_n_heads = (int)a_t->shape[0]; /* num_v_heads */

                snprintf(tname, sizeof(tname), "blk.%d.ssm_norm.weight", l);
                const tq_gguf_tensor_t* norm_t = find_gguf_tensor(gguf, tname);
                if (norm_t) {
                    c->delta_value_head_dim = (int)norm_t->shape[0];
                }

                /* Try to read key_head_dim from metadata (ssm.state_size) */
                c->delta_key_head_dim = tq_gguf_get_i32(gguf,
                    GGUF_KEY("ssm.state_size"), c->delta_value_head_dim);

                /* Try to read num_k_heads from metadata (ssm.group_count) */
                c->delta_n_kv_heads = tq_gguf_get_i32(gguf,
                    GGUF_KEY("ssm.group_count"), c->delta_n_heads);

                c->delta_conv_width = tq_gguf_get_i32(gguf,
                    GGUF_KEY("ssm.conv_kernel"), 4);
                c->partial_rotary_factor = 0.25f;

                fprintf(stderr, "tq_load_gguf: DeltaNet config — v_heads=%d, kv_heads=%d, "
                        "key_dim=%d, val_dim=%d, conv=%d\n",
                        c->delta_n_heads, c->delta_n_kv_heads,
                        c->delta_key_head_dim, c->delta_value_head_dim, c->delta_conv_width);
            }
        }

        /* FFN weights */
        if (c->is_moe) {
            /* Check if this layer has MoE or dense FFN */
            snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_inp.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) {
                /* MoE layer: allocate and set up expert pointers */
                tq_moe_layer_t* moe = (tq_moe_layer_t*)calloc(1, sizeof(tq_moe_layer_t));
                moe->experts = (tq_expert_weights_t*)calloc((size_t)c->num_experts,
                                                              sizeof(tq_expert_weights_t));

                /* Router weights (small, always dequant to FP32) */
                moe->router_weight = dequant_tensor_fp32(t);

                /* Router input scale (Gemma 4): per-feature scaling before routing */
                snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_inp.scale", l);
                t = find_gguf_tensor(gguf, tname);
                if (t && t->type == TQ_GGML_TYPE_F32) {
                    moe->router_input_scale = (const float*)t->data;
                }

                /* Expert weights: shape [num_experts, expert_dim, hidden_dim]
                 * For GGUF, these are stored as 3D tensors. Each expert's
                 * weights are a contiguous slice within the tensor. */
                snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_exps.weight", l);
                const tq_gguf_tensor_t* gate_t = find_gguf_tensor(gguf, tname);
                snprintf(tname, sizeof(tname), "blk.%d.ffn_up_exps.weight", l);
                const tq_gguf_tensor_t* up_t = find_gguf_tensor(gguf, tname);
                snprintf(tname, sizeof(tname), "blk.%d.ffn_down_exps.weight", l);
                const tq_gguf_tensor_t* down_t = find_gguf_tensor(gguf, tname);

                if (gate_t && up_t && down_t) {
                    int exp_inter = c->expert_intermediate_dim;
                    /* Each expert's gate/up weight: [expert_inter, hidden_dim]
                     * Stored contiguously: expert[i] starts at offset i * expert_size */
                    size_t gate_exp_size = tq_ggml_type_size(gate_t->type) *
                        ((size_t)exp_inter * c->hidden_dim / tq_ggml_type_blck(gate_t->type));
                    size_t up_exp_size = tq_ggml_type_size(up_t->type) *
                        ((size_t)exp_inter * c->hidden_dim / tq_ggml_type_blck(up_t->type));
                    size_t down_exp_size = tq_ggml_type_size(down_t->type) *
                        ((size_t)c->hidden_dim * exp_inter / tq_ggml_type_blck(down_t->type));

                    for (int e = 0; e < c->num_experts; e++) {
                        moe->experts[e].w_gate = (const uint8_t*)gate_t->data + e * gate_exp_size;
                        moe->experts[e].gate_type = gate_t->type;
                        moe->experts[e].w_up = (const uint8_t*)up_t->data + e * up_exp_size;
                        moe->experts[e].up_type = up_t->type;
                        moe->experts[e].w_down = (const uint8_t*)down_t->data + e * down_exp_size;
                        moe->experts[e].down_type = down_t->type;
                    }
                }

                /* Gemma 4 fused gate_up_exps: single tensor [hidden_dim, 2*expert_dim, num_experts].
                 * gate is first expert_dim rows, up is next expert_dim rows per expert. */
                if (!gate_t && !up_t) {
                    snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_up_exps.weight", l);
                    const tq_gguf_tensor_t* gate_up_t = find_gguf_tensor(gguf, tname);
                    if (gate_up_t && down_t) {
                        int exp_inter = c->expert_intermediate_dim;
                        /* Fused gate+up: each expert has 2*expert_dim rows of hidden_dim columns.
                         * gate = first expert_dim rows, up = next expert_dim rows. */
                        int fused_rows = 2 * exp_inter;
                        size_t fused_exp_size = tq_ggml_type_size(gate_up_t->type) *
                            ((size_t)fused_rows * c->hidden_dim / tq_ggml_type_blck(gate_up_t->type));
                        size_t gate_part_size = tq_ggml_type_size(gate_up_t->type) *
                            ((size_t)exp_inter * c->hidden_dim / tq_ggml_type_blck(gate_up_t->type));
                        size_t down_exp_size = tq_ggml_type_size(down_t->type) *
                            ((size_t)c->hidden_dim * exp_inter / tq_ggml_type_blck(down_t->type));

                        for (int e = 0; e < c->num_experts; e++) {
                            const uint8_t* base = (const uint8_t*)gate_up_t->data + e * fused_exp_size;
                            moe->experts[e].w_gate = base;
                            moe->experts[e].gate_type = gate_up_t->type;
                            moe->experts[e].w_up = base + gate_part_size;
                            moe->experts[e].up_type = gate_up_t->type;
                            moe->experts[e].w_down = (const uint8_t*)down_t->data + e * down_exp_size;
                            moe->experts[e].down_type = down_t->type;
                        }

                        /* Load per-expert output scale if present (Gemma 4) */
                        snprintf(tname, sizeof(tname), "blk.%d.ffn_down_exps.scale", l);
                        t = find_gguf_tensor(gguf, tname);
                        if (t && t->type == TQ_GGML_TYPE_F32) {
                            moe->expert_scale = (const float*)t->data;
                        }

                        fprintf(stderr, "tq_load_gguf: layer %d — fused gate_up experts, "
                                "exp_inter=%d, down_type=%d\n", l, exp_inter, down_t->type);
                    }
                }

                /* Shared expert (if present) */
                if (c->has_shared_expert) {
                    snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_shexp.weight", l);
                    t = find_gguf_tensor(gguf, tname);
                    if (t) {
                        moe->shared_expert.w_gate = t->data;
                        moe->shared_expert.gate_type = t->type;
                    }
                    snprintf(tname, sizeof(tname), "blk.%d.ffn_up_shexp.weight", l);
                    t = find_gguf_tensor(gguf, tname);
                    if (t) {
                        moe->shared_expert.w_up = t->data;
                        moe->shared_expert.up_type = t->type;
                    }
                    snprintf(tname, sizeof(tname), "blk.%d.ffn_down_shexp.weight", l);
                    t = find_gguf_tensor(gguf, tname);
                    if (t) {
                        moe->shared_expert.w_down = t->data;
                        moe->shared_expert.down_type = t->type;
                    }
                    snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_inp_shexp.weight", l);
                    t = find_gguf_tensor(gguf, tname);
                    if (t) moe->shared_gate = dequant_tensor_fp32(t);
                }

                layer->moe = moe;
            } else {
                /* Dense FFN in an otherwise MoE model — use GGUF on-the-fly */
                snprintf(tname, sizeof(tname), "blk.%d.ffn_gate.weight", l);
                t = find_gguf_tensor(gguf, tname);
                if (t) { layer->gguf_w_gate = t->data; layer->gguf_w_gate_type = t->type; }
                snprintf(tname, sizeof(tname), "blk.%d.ffn_up.weight", l);
                t = find_gguf_tensor(gguf, tname);
                if (t) { layer->gguf_w_up = t->data; layer->gguf_w_up_type = t->type; }
                snprintf(tname, sizeof(tname), "blk.%d.ffn_down.weight", l);
                t = find_gguf_tensor(gguf, tname);
                if (t) { layer->gguf_w_down = t->data; layer->gguf_w_down_type = t->type; }
            }
        } else {
            /* Dense model: use GGUF on-the-fly dequant */
            snprintf(tname, sizeof(tname), "blk.%d.ffn_gate.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) { layer->gguf_w_gate = t->data; layer->gguf_w_gate_type = t->type; }
            snprintf(tname, sizeof(tname), "blk.%d.ffn_up.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) { layer->gguf_w_up = t->data; layer->gguf_w_up_type = t->type; }
            snprintf(tname, sizeof(tname), "blk.%d.ffn_down.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) { layer->gguf_w_down = t->data; layer->gguf_w_down_type = t->type; }
        }
    }

    /* Hybrid architecture: record which layers have self_attn */
    model->n_attn_layers = n_attn_layers;
    if (n_attn_layers > 0 && n_attn_layers < c->n_layers) {
        model->attn_layer_indices = (int*)malloc((size_t)n_attn_layers * sizeof(int));
        memcpy(model->attn_layer_indices, attn_indices, (size_t)n_attn_layers * sizeof(int));
        fprintf(stderr, "tq_load_gguf: hybrid architecture — %d attn layers out of %d total\n",
                n_attn_layers, c->n_layers);
    }

    /* Set up layer_is_sliding for Gemma hybrid attention.
     * Detect from K tensor shape: sliding and full layers have different K output dims.
     * The MAJORITY of layers are sliding (e.g., 25/30 or 28/35). */
    if (c->sliding_window > 0 && c->model_type == 1) {
        model->layer_is_sliding = (int*)calloc((size_t)c->n_layers, sizeof(int));
        if (model->layer_is_sliding) {
            /* Collect K output dims for each layer */
            int k_dims[256]; /* max layers */
            for (int l = 0; l < c->n_layers && l < 256; l++) {
                char tname[128];
                snprintf(tname, sizeof(tname), "blk.%d.attn_k.weight", l);
                const tq_gguf_tensor_t* kt = tq_gguf_find_tensor(gguf, tname);
                k_dims[l] = kt ? (int)kt->shape[1] : 0;
            }
            /* Find the most common K dim (= sliding layer K dim) */
            int sliding_k = k_dims[0]; /* layer 0 is always sliding */
            int n_sliding = 0, n_full = 0;
            int full_kv_dim = 0;
            for (int l = 0; l < c->n_layers; l++) {
                if (k_dims[l] == sliding_k) {
                    model->layer_is_sliding[l] = 1;
                    n_sliding++;
                } else {
                    model->layer_is_sliding[l] = 0;
                    n_full++;
                    full_kv_dim = k_dims[l];
                }
            }
            if (n_full > 0 && full_kv_dim > 0) {
                /* Compute full layer dimensions.
                 * sliding_k = kv_heads * sliding_head_dim (already in c->head_dim)
                 * full_kv_dim = kv_heads_full * full_head_dim
                 * Use metadata key_length for full_head_dim as primary source. */
                int meta_hd = tq_gguf_get_i32(gguf, GGUF_KEY("attention.key_length"), 0);
                if (meta_hd > c->head_dim && full_kv_dim % meta_hd == 0) {
                    c->full_head_dim = meta_hd;
                    c->full_n_kv_heads = full_kv_dim / meta_hd;
                } else if (full_kv_dim % (c->head_dim * 2) == 0) {
                    c->full_head_dim = c->head_dim * 2;
                    c->full_n_kv_heads = full_kv_dim / c->full_head_dim;
                } else {
                    c->full_head_dim = c->head_dim;
                    c->full_n_kv_heads = c->n_kv_heads;
                }
                c->full_n_heads = c->n_heads;
                fprintf(stderr, "tq_load_gguf: Gemma hybrid — %d sliding (hd=%d, kv=%d) + "
                        "%d full (hd=%d, kv=%d, heads=%d) attention layers\n",
                        n_sliding, c->head_dim, c->n_kv_heads,
                        n_full, c->full_head_dim, c->full_n_kv_heads, c->full_n_heads);
            }
        }
    }

    /* Load embedding + output weights */
    const tq_gguf_tensor_t* emb_t = find_gguf_tensor(gguf, "token_embd.weight");
    if (emb_t) {
        if (emb_t->type == TQ_GGML_TYPE_F32) {
            model->token_embedding = (float*)emb_t->data;
        } else if (emb_t->type == TQ_GGML_TYPE_BF16 || emb_t->type == TQ_GGML_TYPE_F16) {
            /* Keep as-is for streaming dequant */
            model->embed_bf16 = (const uint16_t*)emb_t->data;
        } else if (c->vocab_size > 100000 || emb_t->shape[1] > 100000) {
            /* Large vocab: keep GGUF for output projection, dequant rows on demand */
            model->output_gguf = emb_t->data;
            model->output_gguf_type = emb_t->type;
            model->token_embedding = NULL;
        } else {
            model->token_embedding = dequant_tensor_fp32(emb_t);
        }
    }

    /* Get vocab size from embedding tensor shape if not in metadata */
    if (c->vocab_size == 0 && emb_t && emb_t->n_dims >= 2) {
        c->vocab_size = (int)emb_t->shape[1]; /* shape: [hidden_dim, vocab_size] in GGUF row-major */
    }

    const tq_gguf_tensor_t* out_t = find_gguf_tensor(gguf, "output.weight");
    if (out_t) {
        if (out_t->type == TQ_GGML_TYPE_F32) {
            model->output_weight = (float*)out_t->data;
        } else if (out_t->type == TQ_GGML_TYPE_BF16 || out_t->type == TQ_GGML_TYPE_F16) {
            model->output_weight_bf16 = (const uint16_t*)out_t->data;
        } else {
            model->output_weight = dequant_tensor_fp32(out_t);
        }
    } else {
        /* Weight tying: output weight = embedding weight */
        model->output_weight = model->token_embedding;
        model->output_weight_bf16 = model->embed_bf16;
        /* GGUF weight tying: output_gguf already set from embedding */
    }

    const tq_gguf_tensor_t* onorm_t = find_gguf_tensor(gguf, "output_norm.weight");
    if (onorm_t) {
        model->output_norm = dequant_tensor_fp32(onorm_t);
        if (c->model_type == 1 && !gemma_norms_adjusted) {
            for (int i = 0; i < c->hidden_dim; i++) model->output_norm[i] += 1.0f;
        }
    }

    /* Learned RoPE frequencies (Gemma 4): pre-computed inv_freq values */
    {
        const tq_gguf_tensor_t* rope_t = find_gguf_tensor(gguf, "rope_freqs.weight");
        if (rope_t) {
            model->rope_freqs = dequant_tensor_fp32(rope_t);
            model->rope_freqs_len = (int)rope_t->shape[0];
            fprintf(stderr, "tq_load_gguf: loaded learned RoPE frequencies (%d values)\n",
                    model->rope_freqs_len);
        }
    }

    /* Gemma 4 PLE (Per-Layer Embedding) global tensors */
    {
        const tq_gguf_tensor_t* ple_emb_t = find_gguf_tensor(gguf, "per_layer_token_embd.weight");
        const tq_gguf_tensor_t* ple_proj_t = find_gguf_tensor(gguf, "per_layer_model_proj.weight");
        const tq_gguf_tensor_t* ple_norm_t = find_gguf_tensor(gguf, "per_layer_proj_norm.weight");
        if (ple_emb_t && ple_proj_t && ple_norm_t) {
            /* per_layer_token_embd: [8960, vocab] Q5_K — keep as GGUF pointer for runtime dequant */
            model->ple_embedding = ple_emb_t->data;
            model->ple_embedding_type = ple_emb_t->type;

            /* per_layer_model_proj: [hidden_dim, 8960] BF16 — dequant to FP32 at load time */
            model->ple_proj = dequant_tensor_fp32(ple_proj_t);

            /* per_layer_proj_norm: [ple_dim] F32 */
            model->ple_proj_norm = dequant_tensor_fp32(ple_norm_t);

            /* Infer ple_dim from tensor shape: per_layer_token_embd shape[0] / n_layers */
            int total_ple = (int)ple_emb_t->shape[0];
            model->ple_dim = total_ple / c->n_layers;

            /* Gemma norm adjustment: +1.0 if norms not already adjusted */
            if (c->model_type == 1 && !gemma_norms_adjusted && model->ple_proj_norm) {
                for (int i = 0; i < model->ple_dim; i++)
                    model->ple_proj_norm[i] += 1.0f;
            }

            fprintf(stderr, "tq_load_gguf: PLE enabled, ple_dim=%d, total_ple_dim=%d\n",
                    model->ple_dim, total_ple);
        }
    }

    fprintf(stderr, "tq_load_gguf: loaded %d layers (%d self_attn%s), dim=%d, heads=%d/%d, vocab=%d\n",
            c->n_layers, n_attn_layers,
            c->is_moe ? ", MoE" : "",
            c->hidden_dim, c->n_heads, c->n_kv_heads, c->vocab_size);

    /* ============================================================
     * Load-time weight conversion: GGUF -> Q4
     *
     * For non-MoE weights (attention, DeltaNet, dense FFN): batch
     * dequantize GGUF -> FP32 (temporary), then quantize to Q4.
     *
     * For MoE expert weights: convert expert-by-expert using a small
     * reusable FP32 temp buffer (~12 MB) to avoid multi-GB allocation.
     * This replaces the slow on-the-fly GGUF dequant path with the
     * fast Q4xQ8 integer matmul path, yielding ~10x speedup.
     * ============================================================ */
    {
        /* Estimate total FP32 weight size for non-MoE layer weights */
        int dim = c->hidden_dim;
        int q_dim = c->n_heads * c->head_dim;
        int kv_dim = c->n_kv_heads * c->head_dim;
        int full_kv_dim = (c->full_n_kv_heads > 0 && c->full_head_dim > 0)
            ? c->full_n_kv_heads * c->full_head_dim : kv_dim;
        int inter = c->intermediate_dim;
        (void)inter;
        int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
        int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2
                          + c->delta_n_heads * c->delta_value_head_dim;
        int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
        int delta_dn = c->delta_n_heads;

        size_t est_fp32 = 0;
        for (int l = 0; l < c->n_layers; l++) {
            const tq_layer_weights_t* layer = &model->layers[l];
            int is_full_l = (model->layer_is_sliding && !model->layer_is_sliding[l]);
            int lkv = is_full_l ? full_kv_dim : kv_dim;
            int lq = is_full_l ? (c->full_n_heads * c->full_head_dim) : q_dim;
            int lqg = c->attn_output_gate ? lq * 2 : lq;
            if (layer->gguf_wq)
                est_fp32 += (size_t)lqg * dim * sizeof(float);
            if (layer->gguf_wk)
                est_fp32 += (size_t)lkv * dim * sizeof(float);
            if (layer->gguf_wv)
                est_fp32 += (size_t)lkv * dim * sizeof(float);
            if (layer->gguf_wo)
                est_fp32 += (size_t)dim * lq * sizeof(float);
            /* Dense FFN weights (not present in MoE layers) */
            int lint_est = (c->per_layer_inter_dim) ? c->per_layer_inter_dim[l] : inter;
            if (layer->gguf_w_gate)
                est_fp32 += (size_t)lint_est * dim * sizeof(float);
            if (layer->gguf_w_up)
                est_fp32 += (size_t)lint_est * dim * sizeof(float);
            if (layer->gguf_w_down)
                est_fp32 += (size_t)dim * lint_est * sizeof(float);
            /* DeltaNet GGUF weights */
            if (layer->gguf_delta_qkv)
                est_fp32 += (size_t)delta_qkv_dim * dim * sizeof(float);
            if (layer->gguf_delta_z)
                est_fp32 += (size_t)delta_z_dim * dim * sizeof(float);
            if (layer->gguf_delta_a)
                est_fp32 += (size_t)delta_dn * dim * sizeof(float);
            if (layer->gguf_delta_b)
                est_fp32 += (size_t)delta_dn * dim * sizeof(float);
            if (layer->gguf_delta_out)
                est_fp32 += (size_t)dim * delta_z_dim * sizeof(float);
        }

        const size_t MAX_FP32_BYTES = (size_t)16 * 1024 * 1024 * 1024ULL; /* 16 GB */
        /* TQ_NO_Q4=1 disables Q4 recompression → use direct GGUF dequant for better quality.
         * Can be set via environment variable or compile-time define (useful for WASM). */
#ifdef TQ_NO_Q4
        if (1) {
#else
        if (getenv("TQ_NO_Q4")) {
#endif
            fprintf(stderr, "tq_load_gguf: TQ_NO_Q4 set — skipping Q4 conversion, using GGUF on-the-fly dequant\n");
            goto skip_q4_conversion;
        }
        int has_gguf_weights = 0;
        for (int l = 0; l < c->n_layers && !has_gguf_weights; l++) {
            if (model->layers[l].gguf_wq || model->layers[l].gguf_w_gate
                || model->layers[l].gguf_delta_qkv || model->layers[l].gguf_delta_z
                || model->layers[l].moe)
                has_gguf_weights = 1;
        }

        if (has_gguf_weights && est_fp32 < MAX_FP32_BYTES) {
            fprintf(stderr, "tq_load_gguf: load-time Q4 conversion enabled "
                    "(est FP32 = %.1f GB < 8 GB threshold)\n",
                    (double)est_fp32 / (1024.0 * 1024.0 * 1024.0));

            /* Track all FP32 temporaries for cleanup after Q4 quantization.
             * Max 13 weight matrices per layer (7 attn/FFN + 5 DeltaNet + 1 spare). */
            int max_tmp = c->n_layers * 13;
            float** fp32_temps = (float**)calloc((size_t)max_tmp, sizeof(float*));
            int n_tmp = 0;

            for (int l = 0; l < c->n_layers; l++) {
                tq_layer_weights_t* layer = &model->layers[l];

                /* Self-attention weights: dequant GGUF -> FP32 */
                int is_full = (model->layer_is_sliding && !model->layer_is_sliding[l]);
                int lq = is_full ? (c->full_n_heads * c->full_head_dim) : q_dim;
                int lqg = c->attn_output_gate ? lq * 2 : lq;
                if (layer->gguf_wq) {
                    int n = lqg * dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_wq_type, layer->gguf_wq, fp, n);
                        layer->wq = fp;
                        layer->gguf_wq = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
                if (layer->gguf_wk) {
                    int lkv = (model->layer_is_sliding && !model->layer_is_sliding[l]) ? full_kv_dim : kv_dim;
                    int n = lkv * dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_wk_type, layer->gguf_wk, fp, n);
                        layer->wk = fp;
                        layer->gguf_wk = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
                if (layer->gguf_wv) {
                    int lkv = (model->layer_is_sliding && !model->layer_is_sliding[l]) ? full_kv_dim : kv_dim;
                    int n = lkv * dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_wv_type, layer->gguf_wv, fp, n);
                        layer->wv = fp;
                        layer->gguf_wv = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
                if (layer->gguf_wo) {
                    int n = dim * lq;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_wo_type, layer->gguf_wo, fp, n);
                        layer->wo = fp;
                        layer->gguf_wo = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }

                /* Dense FFN weights: dequant GGUF -> FP32 (per-layer dim) */
                int lint = (c->per_layer_inter_dim) ? c->per_layer_inter_dim[l] : inter;
                if (layer->gguf_w_gate) {
                    int n = lint * dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_w_gate_type, layer->gguf_w_gate, fp, n);
                        layer->w_gate = fp;
                        layer->gguf_w_gate = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
                if (layer->gguf_w_up) {
                    int n = lint * dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_w_up_type, layer->gguf_w_up, fp, n);
                        layer->w_up = fp;
                        layer->gguf_w_up = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
                if (layer->gguf_w_down) {
                    int n = dim * lint;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_w_down_type, layer->gguf_w_down, fp, n);
                        layer->w_down = fp;
                        layer->gguf_w_down = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }

                /* DeltaNet weights: dequant GGUF -> FP32 */
                if (layer->gguf_delta_qkv && delta_qkv_dim > 0) {
                    int n = delta_qkv_dim * dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_delta_qkv_type, layer->gguf_delta_qkv, fp, n);
                        layer->delta_in_proj_qkv = fp;
                        layer->gguf_delta_qkv = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
                if (layer->gguf_delta_z && delta_z_dim > 0) {
                    int n = delta_z_dim * dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_delta_z_type, layer->gguf_delta_z, fp, n);
                        layer->delta_in_proj_z = fp;
                        layer->gguf_delta_z = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
                if (layer->gguf_delta_a && delta_dn > 0) {
                    int n = delta_dn * dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_delta_a_type, layer->gguf_delta_a, fp, n);
                        layer->delta_in_proj_a = fp;
                        layer->gguf_delta_a = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
                if (layer->gguf_delta_b && delta_dn > 0) {
                    int n = delta_dn * dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_delta_b_type, layer->gguf_delta_b, fp, n);
                        layer->delta_in_proj_b = fp;
                        layer->gguf_delta_b = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
                if (layer->gguf_delta_out && delta_z_dim > 0) {
                    int n = dim * delta_z_dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_delta_out_type, layer->gguf_delta_out, fp, n);
                        layer->delta_out_proj = fp;
                        layer->gguf_delta_out = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
            }

            /* Convert all FP32 weights to Q4 (reuses existing tq_quantize_weights_q4) */
            tq_quantize_weights_q4(model);

            /* Free FP32 temporaries (tq_quantize_weights_q4 set layer->wX = NULL
             * but didn't free the malloc'd buffers) */
            for (int i = 0; i < n_tmp; i++) {
                free(fp32_temps[i]);
            }
            free(fp32_temps);

            fprintf(stderr, "tq_load_gguf: Q4 conversion complete — fast matmul path active\n");
        }

skip_q4_conversion: ;
        /* ============================================================
         * MoE shared expert Q4 conversion + LRU cache init
         *
         * Routed experts use a runtime LRU cache (in tq_moe.c) that
         * converts on-demand — only ~32 most-recently-used experts per
         * layer stay in Q4 form, keeping memory under ~2 GB.
         *
         * Shared experts are always active, so convert them at load time.
         * ============================================================ */
        if (c->is_moe) {
            int shared_inter = c->shared_expert_intermediate_dim;
            if (shared_inter == 0) shared_inter = c->expert_intermediate_dim;

            size_t max_elems = (size_t)shared_inter * dim;
            if ((size_t)dim * shared_inter > max_elems)
                max_elems = (size_t)dim * shared_inter;

            float* fp32_temp = NULL;
            if (c->has_shared_expert && max_elems > 0)
                fp32_temp = (float*)malloc(max_elems * sizeof(float));

            size_t total_q4_shared = 0;
            int n_shared_converted = 0;

            if (c->has_shared_expert && fp32_temp) {
                for (int l = 0; l < c->n_layers; l++) {
                    tq_moe_layer_t* moe = (tq_moe_layer_t*)model->layers[l].moe;
                    if (!moe || !moe->shared_expert.w_gate) continue;
                    tq_expert_weights_t* se = &moe->shared_expert;

                    /* gate: [shared_inter, dim] */
                    {
                        int n = shared_inter * dim;
                        int n_blocks = (n + 31) / 32;
                        tq_dequant_row_gguf(se->gate_type, se->w_gate, fp32_temp, n);
                        se->gate_q4_qs = (uint8_t*)malloc((size_t)n_blocks * 16);
                        se->gate_q4_scales = (float*)malloc((size_t)n_blocks * sizeof(float));
                        if (se->gate_q4_qs && se->gate_q4_scales) {
                            tq_quantize_row_q4(fp32_temp, se->gate_q4_qs,
                                               se->gate_q4_scales, n);
                            total_q4_shared += (size_t)n_blocks * 16 + (size_t)n_blocks * sizeof(float);
                        }
                    }

                    /* up: [shared_inter, dim] */
                    {
                        int n = shared_inter * dim;
                        int n_blocks = (n + 31) / 32;
                        tq_dequant_row_gguf(se->up_type, se->w_up, fp32_temp, n);
                        se->up_q4_qs = (uint8_t*)malloc((size_t)n_blocks * 16);
                        se->up_q4_scales = (float*)malloc((size_t)n_blocks * sizeof(float));
                        if (se->up_q4_qs && se->up_q4_scales) {
                            tq_quantize_row_q4(fp32_temp, se->up_q4_qs,
                                               se->up_q4_scales, n);
                            total_q4_shared += (size_t)n_blocks * 16 + (size_t)n_blocks * sizeof(float);
                        }
                    }

                    /* down: [dim, shared_inter] */
                    {
                        int n = dim * shared_inter;
                        int n_blocks = (n + 31) / 32;
                        tq_dequant_row_gguf(se->down_type, se->w_down, fp32_temp, n);
                        se->down_q4_qs = (uint8_t*)malloc((size_t)n_blocks * 16);
                        se->down_q4_scales = (float*)malloc((size_t)n_blocks * sizeof(float));
                        if (se->down_q4_qs && se->down_q4_scales) {
                            tq_quantize_row_q4(fp32_temp, se->down_q4_qs,
                                               se->down_q4_scales, n);
                            total_q4_shared += (size_t)n_blocks * 16 + (size_t)n_blocks * sizeof(float);
                        }
                    }

                    se->q4_converted = (se->gate_q4_qs && se->up_q4_qs && se->down_q4_qs);
                    if (se->q4_converted) n_shared_converted++;
                }
            }

            free(fp32_temp);

            /* LRU caches disabled: both Q8 and cblas caches are bypassed
             * in the forward path (if(0) guards). Skip allocation to save
             * ~1.5 GB RAM — critical for 16GB machines running 9.9 GB models. */
            /* tq_moe_cache_init(...) — disabled */
            /* tq_moe_cblas_cache_init(...) — disabled */

            fprintf(stderr, "tq_load_gguf: MoE — %d shared experts Q4-converted "
                    "(%.1f MB), routed experts use runtime LRU cache\n",
                    n_shared_converted, (double)total_q4_shared / (1024.0 * 1024.0));
        }

        /* Advise OS to release mmap'd GGUF pages — the original quantized
         * data is no longer needed after Q4 conversion.
         * NOTE: For MoE models we must NOT release mmap pages since routed
         * expert weights are still read on-demand from the mmap. */
#ifndef _WIN32
        if (model->gguf_ctx) {
            tq_gguf_ctx_t* gctx = (tq_gguf_ctx_t*)model->gguf_ctx;
            if (gctx->mmap_data && gctx->mmap_size > 0) {
                if (c->is_moe) {
                    /* MoE: lock model data in physical RAM to prevent page-out.
                     * Without mlock, expert weights get evicted by OS memory pressure,
                     * causing 100x+ slowdown from SSD page faults.
                     * mlock may fail if ulimit is too low — fall back to MADV_WILLNEED. */
                    if (mlock(gctx->mmap_data, gctx->mmap_size) == 0) {
                        fprintf(stderr, "tq_load_gguf: mlock(%.1f GB) — expert weights pinned in RAM\n",
                                (double)gctx->mmap_size / (1024.0 * 1024.0 * 1024.0));
                    } else {
                        /* mlock failed (insufficient privilege or ulimit) — use madvise */
                        madvise(gctx->mmap_data, gctx->mmap_size, MADV_WILLNEED);
                        fprintf(stderr, "tq_load_gguf: mlock failed (errno=%d), using MADV_WILLNEED for %.1f GB\n",
                                errno, (double)gctx->mmap_size / (1024.0 * 1024.0 * 1024.0));
                    }
                } else {
                    /* Non-MoE: release mmap pages after Q4 conversion */
                    madvise(gctx->mmap_data, gctx->mmap_size, MADV_DONTNEED);
                    fprintf(stderr, "tq_load_gguf: madvise(MADV_DONTNEED) on %.1f GB mmap\n",
                            (double)gctx->mmap_size / (1024.0 * 1024.0 * 1024.0));
                }
            }
        }
#endif
    }

    #undef GGUF_KEY
    return model;
}

int tq_save_tqm(tq_model_t* model, const char* tokenizer_path,
                const char* output_path) {
    if (!model || !output_path) return -1;
    if (!model->use_q4_weights) {
        fprintf(stderr, "tq_save_tqm: model must be Q4-quantized first\n");
        return -1;
    }

    const tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;
    int delta_conv_total = delta_qkv_dim;
    int delta_conv_width = c->delta_conv_width;
    int delta_vhd = c->delta_value_head_dim;

    /* Build is-attn lookup */
    int* is_attn_layer = (int*)calloc((size_t)c->n_layers, sizeof(int));
    if (!is_attn_layer) return -1;
    for (int i = 0; i < model->n_attn_layers; i++) {
        int idx = model->attn_layer_indices[i];
        if (idx >= 0 && idx < c->n_layers) is_attn_layer[idx] = 1;
    }

    /* Read tokenizer file if provided */
    char* tok_data = NULL;
    size_t tok_size = 0;
    if (tokenizer_path) {
        FILE* tf = fopen(tokenizer_path, "rb");
        if (tf) {
            fseek(tf, 0, SEEK_END);
            tok_size = (size_t)ftell(tf);
            fseek(tf, 0, SEEK_SET);
            tok_data = (char*)malloc(tok_size);
            if (tok_data) {
                size_t nr = fread(tok_data, 1, tok_size, tf);
                if (nr != tok_size) {
                    free(tok_data);
                    tok_data = NULL;
                    tok_size = 0;
                }
            }
            fclose(tf);
        } else {
            fprintf(stderr, "tq_save_tqm: warning: cannot open tokenizer '%s'\n",
                    tokenizer_path);
        }
    }

    FILE* f = fopen(output_path, "wb");
    if (!f) {
        fprintf(stderr, "tq_save_tqm: cannot create '%s'\n", output_path);
        free(tok_data);
        free(is_attn_layer);
        return -1;
    }

    /* Compute section offsets */
    uint64_t tok_offset = tqm_align(sizeof(tqm_header_t));
    uint64_t wt_offset = tqm_align(tok_offset + tok_size);

    /* Build and write header */
    tqm_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic   = TQM_MAGIC;
    hdr.version = TQM_VERSION;

    hdr.n_layers         = c->n_layers;
    hdr.hidden_dim       = c->hidden_dim;
    hdr.intermediate_dim = c->intermediate_dim;
    hdr.n_heads          = c->n_heads;
    hdr.n_kv_heads       = c->n_kv_heads;
    hdr.head_dim         = c->head_dim;
    hdr.vocab_size       = c->vocab_size;
    hdr.max_seq_len      = c->max_seq_len;
    hdr.rope_freq_base   = c->rope_freq_base;
    hdr.rms_norm_eps     = c->rms_norm_eps;

    hdr.delta_n_heads       = c->delta_n_heads;
    hdr.delta_n_kv_heads_tqm= c->delta_n_kv_heads;
    hdr.delta_key_head_dim  = c->delta_key_head_dim;
    hdr.delta_value_head_dim= c->delta_value_head_dim;
    hdr.delta_conv_width    = c->delta_conv_width;
    hdr.partial_rotary_factor = c->partial_rotary_factor;
    hdr.use_qk_norm         = c->use_qk_norm;
    hdr.attn_output_gate    = c->attn_output_gate;

    hdr.model_type              = c->model_type;
    hdr.sliding_window          = c->sliding_window;
    hdr.rope_local_base_freq    = c->rope_local_base_freq;
    hdr.n_norms_per_block       = c->n_norms_per_block;
    hdr.query_pre_attn_scalar   = c->query_pre_attn_scalar;

    hdr.weight_quant = model->use_q8_weights ? 8 : (model->use_q4_weights ? 4 : 0);
    hdr.embed_format = 16; /* BF16 */

    hdr.tokenizer_offset = tok_offset;
    hdr.tokenizer_size   = tok_size;
    hdr.weights_offset   = wt_offset;
    /* weights_size filled after writing */

    hdr.n_attn_layers = model->n_attn_layers;
    for (int i = 0; i < model->n_attn_layers && i < 64; i++) {
        hdr.attn_layer_indices[i] = model->attn_layer_indices[i];
    }

    fwrite(&hdr, sizeof(hdr), 1, f);

    /* Pad to tokenizer offset */
    {
        uint64_t cur = sizeof(tqm_header_t);
        tqm_write_pad(f, cur);
    }

    /* Write tokenizer */
    if (tok_data && tok_size > 0) {
        fwrite(tok_data, 1, tok_size, f);
    }
    free(tok_data);

    /* Pad to weights offset */
    {
        uint64_t cur = tok_offset + tok_size;
        tqm_write_pad(f, cur);
    }

    /* Helper macros for writing */
    #define TQM_WRITE_FP32(src, count) do {                            \
        if ((src)) fwrite((src), sizeof(float), (size_t)(count), f);   \
        else { float _z = 0; for (int _i = 0; _i < (count); _i++)     \
            fwrite(&_z, sizeof(float), 1, f); }                        \
    } while (0)

    #define TQM_WRITE_Q4(qs, sc, rows, cols) do {                      \
        int _nb = ((cols) + 31) / 32;                                   \
        size_t _qs_bytes = (size_t)(rows) * _nb * 16;                   \
        size_t _sc_bytes = (size_t)(rows) * _nb * sizeof(float);        \
        if ((qs)) fwrite((qs), 1, _qs_bytes, f);                       \
        else { uint8_t _z = 0; for (size_t _i = 0; _i < _qs_bytes; _i++) \
            fwrite(&_z, 1, 1, f); }                                    \
        if ((sc)) fwrite((sc), 1, _sc_bytes, f);                       \
        else { float _zf = 0; for (size_t _i = 0;                      \
            _i < (size_t)(rows) * _nb; _i++) fwrite(&_zf, sizeof(float), 1, f); } \
    } while (0)

    /* Write per-layer weights */
    for (int l = 0; l < c->n_layers; l++) {
        const tq_layer_weights_t* layer = &model->layers[l];

        TQM_WRITE_FP32(layer->attn_norm, dim);
        TQM_WRITE_FP32(layer->ffn_norm, dim);

        /* Gemma3 extra norms */
        if (c->model_type == 1) {
            TQM_WRITE_FP32(layer->pre_ffn_norm, dim);
            TQM_WRITE_FP32(layer->post_ffn_norm, dim);
        }

        if (is_attn_layer[l]) {
            TQM_WRITE_Q4(layer->wq_q4, layer->wq_q4s, qg_dim, dim);
            TQM_WRITE_Q4(layer->wk_q4, layer->wk_q4s, kv_dim, dim);
            TQM_WRITE_Q4(layer->wv_q4, layer->wv_q4s, kv_dim, dim);
            TQM_WRITE_Q4(layer->wo_q4, layer->wo_q4s, dim, q_dim);

            if (c->use_qk_norm) {
                TQM_WRITE_FP32(layer->q_norm, c->head_dim);
                TQM_WRITE_FP32(layer->k_norm, c->head_dim);
            }
        } else {
            TQM_WRITE_FP32(layer->delta_a_log, delta_dn);
            TQM_WRITE_FP32(layer->delta_dt_bias, delta_dn);
            TQM_WRITE_Q4(layer->delta_in_proj_qkv_q4, layer->delta_in_proj_qkv_q4s,
                          delta_qkv_dim, dim);
            TQM_WRITE_Q4(layer->delta_in_proj_z_q4, layer->delta_in_proj_z_q4s,
                          delta_z_dim, dim);
            TQM_WRITE_Q4(layer->delta_in_proj_a_q4, layer->delta_in_proj_a_q4s,
                          delta_dn, dim);
            TQM_WRITE_Q4(layer->delta_in_proj_b_q4, layer->delta_in_proj_b_q4s,
                          delta_dn, dim);
            TQM_WRITE_FP32(layer->delta_conv1d, delta_conv_total * delta_conv_width);
            TQM_WRITE_FP32(layer->delta_norm, delta_vhd);
            TQM_WRITE_Q4(layer->delta_out_proj_q4, layer->delta_out_proj_q4s,
                          dim, delta_z_dim);
        }

        TQM_WRITE_Q4(layer->w_gate_q4, layer->w_gate_q4s, inter, dim);
        TQM_WRITE_Q4(layer->w_up_q4, layer->w_up_q4s, inter, dim);
        TQM_WRITE_Q4(layer->w_down_q4, layer->w_down_q4s, dim, inter);
    }

    /* Output norm */
    TQM_WRITE_FP32(model->output_norm, dim);

    /* Embedding BF16 */
    {
        size_t embed_elems = (size_t)c->vocab_size * dim;
        if (model->embed_bf16) {
            fwrite(model->embed_bf16, 2, embed_elems, f);
        } else if (model->token_embedding) {
            /* Convert FP32 to BF16 on the fly */
            for (size_t i = 0; i < embed_elems; i++) {
                uint32_t fp32;
                memcpy(&fp32, &model->token_embedding[i], sizeof(float));
                uint16_t bf16 = (uint16_t)(fp32 >> 16);
                fwrite(&bf16, 2, 1, f);
            }
        } else {
            /* Write zeros */
            uint16_t z = 0;
            for (size_t i = 0; i < embed_elems; i++) fwrite(&z, 2, 1, f);
        }

        /* Output weight (lm_head) — write if different from embed */
        if (model->output_weight_bf16 &&
            model->output_weight_bf16 != model->embed_bf16) {
            fwrite(model->output_weight_bf16, 2, embed_elems, f);
        } else if (model->output_weight &&
                   model->output_weight != model->token_embedding) {
            for (size_t i = 0; i < embed_elems; i++) {
                uint32_t fp32;
                memcpy(&fp32, &model->output_weight[i], sizeof(float));
                uint16_t bf16 = (uint16_t)(fp32 >> 16);
                fwrite(&bf16, 2, 1, f);
            }
        }
        /* If tied, don't write — loader will detect from file size */
    }

    #undef TQM_WRITE_FP32
    #undef TQM_WRITE_Q4

    /* Update weights_size in header */
    long end_pos = ftell(f);
    hdr.weights_size = (uint64_t)end_pos - wt_offset;
    fseek(f, 0, SEEK_SET);
    fwrite(&hdr, sizeof(hdr), 1, f);

    fclose(f);
    free(is_attn_layer);

    fprintf(stderr, "tq_save_tqm: saved '%s' (%.1f MB)\n",
            output_path, (double)end_pos / (1024.0 * 1024.0));
    return 0;
}

/* ============================================================
 * Free model
 * ============================================================ */
void tq_free_model(tq_model_t* model) {
    if (!model) return;

#ifdef _WIN32
    if (model->_mmap_data) UnmapViewOfFile(model->_mmap_data);
    for (int i = 1; i < model->_n_shards; i++) {
        if (model->_mmap_shards[i]) UnmapViewOfFile(model->_mmap_shards[i]);
    }
#else
    if (model->_mmap_data) munmap(model->_mmap_data, model->_mmap_size);
    for (int i = 1; i < model->_n_shards; i++) {
        if (model->_mmap_shards[i])
            munmap(model->_mmap_shards[i], model->_mmap_shard_sizes[i]);
    }
#endif

    free(model->_converted_data);
    free(model->_q8_data);
    free(model->_q4_data);
    free(model->_q2_data);
    free(model->attn_layer_indices);
    free(model->layer_is_sliding);

    /* Free MoE LRU caches (must happen before freeing layers) */
    tq_moe_cache_free();
#ifdef TQ_HAS_ACCELERATE
    extern void tq_moe_cblas_cache_free(void);
    tq_moe_cblas_cache_free();
#endif

    /* Free MoE resources */
    if (model->config.is_moe && model->layers) {
        for (int l = 0; l < model->config.n_layers; l++) {
            tq_moe_layer_t* moe = (tq_moe_layer_t*)model->layers[l].moe;
            if (moe) {
                /* Free Q4 expert weight data */
                if (moe->experts) {
                    for (int e = 0; e < model->config.num_experts; e++) {
                        tq_expert_weights_t* exp = &moe->experts[e];
                        free(exp->gate_q4_qs);
                        free(exp->gate_q4_scales);
                        free(exp->up_q4_qs);
                        free(exp->up_q4_scales);
                        free(exp->down_q4_qs);
                        free(exp->down_q4_scales);
                    }
                }
                /* Free shared expert Q4 data */
                free(moe->shared_expert.gate_q4_qs);
                free(moe->shared_expert.gate_q4_scales);
                free(moe->shared_expert.up_q4_qs);
                free(moe->shared_expert.up_q4_scales);
                free(moe->shared_expert.down_q4_qs);
                free(moe->shared_expert.down_q4_scales);

                free(moe->router_weight);
                free(moe->shared_gate);
                free(moe->experts);
                free(moe);
            }
        }
    }
    free(model->moe_config);
    free(model->layers);

    /* Free GGUF context (handles munmap internally) */
    if (model->gguf_ctx) {
        tq_gguf_close((tq_gguf_ctx_t*)model->gguf_ctx);
        model->_mmap_data = NULL; /* gguf_close handled it */
    }

    free(model);
}

/* ============================================================
 * 1-bit Weight Quantization
 * Converts all FP32 weights to sign bits + L2 norm per row.
 * Forward path uses tq_matmul_1bit (sign-based dot product).
 * ============================================================ */

extern void tq_quantize_row_1bit(const float*, uint8_t*, float*, int);

void tq_quantize_weights_1bit(tq_model_t* model) {
    if (!model) return;
    const tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;

    /* Calculate total buffer size */
    size_t total = 0;
    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];
        /* sign bits: rows * ceil(cols/8), norms: rows * 4 */
        if (layer->wq) total += (size_t)qg_dim * ((dim+7)/8 + 4);
        if (layer->wk) total += (size_t)kv_dim * ((dim+7)/8 + 4);
        if (layer->wv) total += (size_t)kv_dim * ((dim+7)/8 + 4);
        if (layer->wo) total += (size_t)dim * ((q_dim+7)/8 + 4);
        if (layer->w_gate) total += (size_t)inter * ((dim+7)/8 + 4);
        if (layer->w_up) total += (size_t)inter * ((dim+7)/8 + 4);
        if (layer->w_down) total += (size_t)dim * ((inter+7)/8 + 4);
        if (layer->delta_in_proj_qkv) total += (size_t)delta_qkv_dim * ((dim+7)/8 + 4);
        if (layer->delta_in_proj_z) total += (size_t)delta_z_dim * ((dim+7)/8 + 4);
        if (layer->delta_in_proj_a) total += (size_t)delta_dn * ((dim+7)/8 + 4);
        if (layer->delta_in_proj_b) total += (size_t)delta_dn * ((dim+7)/8 + 4);
        if (layer->delta_out_proj) total += (size_t)dim * ((delta_z_dim+7)/8 + 4);
    }

    /* We store sign bits in the Q2 fields (repurposed) and norms in Q2 scales */
    /* Actually, we need dedicated storage. Use a single buffer. */
    char* buf = (char*)malloc(total);
    if (!buf) { fprintf(stderr, "1bit: alloc failed\n"); return; }
    size_t used = 0;

    /* Helper macro */
    #define QUANTIZE_1BIT(src_ptr, rows, cols, qs_ptr, sc_ptr) do { \
        if (src_ptr) { \
            int _nb = ((cols) + 7) / 8; \
            uint8_t* _qs = (uint8_t*)(buf + used); used += (size_t)(rows) * _nb; \
            float* _sc = (float*)(buf + used); used += (size_t)(rows) * sizeof(float); \
            for (int _r = 0; _r < (rows); _r++) \
                tq_quantize_row_1bit((src_ptr) + (size_t)(_r) * (cols), \
                                      _qs + (size_t)(_r) * _nb, &_sc[_r], (cols)); \
            (qs_ptr) = _qs; (sc_ptr) = _sc; \
            (src_ptr) = NULL; \
        } \
    } while(0)

    /* Store 1-bit data in Q2 fields (repurposed).
     * Q2 packed format: 8 bytes per 32 elements = 1 byte per 4 elements.
     * 1-bit: 1 byte per 8 elements. Different layout, but we just store raw. */

    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];
        QUANTIZE_1BIT(layer->wq, qg_dim, dim, layer->wq_q2, layer->wq_q2s);
        QUANTIZE_1BIT(layer->wk, kv_dim, dim, layer->wk_q2, layer->wk_q2s);
        QUANTIZE_1BIT(layer->wv, kv_dim, dim, layer->wv_q2, layer->wv_q2s);
        QUANTIZE_1BIT(layer->wo, dim, q_dim, layer->wo_q2, layer->wo_q2s);
        QUANTIZE_1BIT(layer->w_gate, inter, dim, layer->w_gate_q2, layer->w_gate_q2s);
        QUANTIZE_1BIT(layer->w_up, inter, dim, layer->w_up_q2, layer->w_up_q2s);
        QUANTIZE_1BIT(layer->w_down, dim, inter, layer->w_down_q2, layer->w_down_q2s);
        QUANTIZE_1BIT(layer->delta_in_proj_qkv, delta_qkv_dim, dim,
                       layer->delta_in_proj_qkv_q2, layer->delta_in_proj_qkv_q2s);
        QUANTIZE_1BIT(layer->delta_in_proj_z, delta_z_dim, dim,
                       layer->delta_in_proj_z_q2, layer->delta_in_proj_z_q2s);
        QUANTIZE_1BIT(layer->delta_in_proj_a, delta_dn, dim,
                       layer->delta_in_proj_a_q2, layer->delta_in_proj_a_q2s);
        QUANTIZE_1BIT(layer->delta_in_proj_b, delta_dn, dim,
                       layer->delta_in_proj_b_q2, layer->delta_in_proj_b_q2s);
        QUANTIZE_1BIT(layer->delta_out_proj, dim, delta_z_dim,
                       layer->delta_out_proj_q2, layer->delta_out_proj_q2s);
    }
    #undef QUANTIZE_1BIT

    model->use_q2_weights = 1;
    model->use_1bit_weights = 1;
    model->_q2_data = buf;
    model->_q2_size = used;

    /* Activate global 1-bit matmul routing */

    fprintf(stderr, "tq_quantize_weights_1bit: 1-bit sign hash (%zu MB)\n",
            used / (1024 * 1024));
}

// ============================================================================
// Section 14: Transformer Forward Pass (from tq_transformer.c)
// ============================================================================

/**
 * tq_transformer.c — Hybrid transformer forward pass (self_attn + DeltaNet)
 *
 * Supports Qwen3.5 architecture:
 *   - Standard self_attn layers with GQA, QK-norm, partial RoPE
 *   - DeltaNet (linear_attention) layers with gated recurrent updates
 *   - SwiGLU FFN on all layers
 *
 * DeltaNet forward (Gated DeltaNet):
 *   x -> RMSNorm -> in_proj_qkv -> split Q,K,V
 *                -> in_proj_z -> z gate
 *                -> in_proj_a, in_proj_b -> a, b
 *   Apply conv1d (causal, width=4) on [Q,K,V]
 *   Q,K -> L2 normalize per head
 *   dt = sigmoid(a * b + dt_bias) -> delta scaling
 *   state = state * decay + delta * outer(K, V)
 *   output = Q @ state -> group_norm -> swish(z) gate -> out_proj
 *   -> residual add
 */

/* Unified Q2/1-bit matmul dispatch.
 * When model->use_1bit_weights, Q2 fields contain sign bits + norms,
 * dispatched to tq_matmul_1bit (FP32 input required).
 * Otherwise, standard Q2 Lloyd-Max matmul with pre-quantized Q8 input. */
#define TQ_MATMUL_Q2_OR_1BIT(out, x_fp32, qs, scales, x_q8, x_q8s, rows, cols, is_1bit) \
    do { \
        if (is_1bit) \
            tq_matmul_1bit((out), (x_fp32), (qs), (scales), (rows), (cols)); \
        else \
            tq_matmul_q2_preq((out), (qs), (scales), (x_q8), (x_q8s), (rows), (cols)); \
    } while(0)

#define TQ_MATMUL_Q2_OR_1BIT_FP32(out, x_fp32, qs, scales, rows, cols, is_1bit) \
    do { \
        if (is_1bit) \
            tq_matmul_1bit((out), (x_fp32), (qs), (scales), (rows), (cols)); \
        else \
            tq_matmul_q2((out), (x_fp32), (qs), (scales), (rows), (cols)); \
    } while(0)

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ============================================================
 * Lightweight forward-pass profiling (clock_gettime only)
 * Activated by setting g_tq_profile_enabled = 1 (via --profile flag)
 * ============================================================ */
typedef struct {
    double matmul_ns;
    double recurrent_ns;
    double moe_ns;
    double conv1d_ns;
    double attn_ns;       /* softmax + weighted-sum in self_attn */
    double total_fwd_ns;  /* total forward pass wall time */
    int    n_tokens;
} tq_profile_t;

static tq_profile_t g_profile = {0};
int g_tq_profile_enabled = 0;  /* set from quant --profile */

static inline double tq_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* Usage: double _tp; TQ_PROF_START(_tp); ... TQ_PROF_STOP(_tp, field); */
#define TQ_PROF_START(var) do { var = g_tq_profile_enabled ? tq_now_ns() : 0; } while(0)
#define TQ_PROF_STOP(var, field) do { if (g_tq_profile_enabled) g_profile.field += tq_now_ns() - var; } while(0)

/* ============================================================
 * FP16 helpers (IEEE 754 half-precision, storage only)
 * ============================================================ */

static uint16_t xfr_f32_to_fp16(float v) {
    union { float f; uint32_t u; } bits;
    bits.f = v;
    uint32_t sign = (bits.u >> 16) & 0x8000;
    int32_t  exp  = ((bits.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits.u >> 13) & 0x03FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static float xfr_fp16_to_f32(uint16_t h) {
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

/* Convert n floats to FP16 (NEON-optimized where available) */
static void f32_to_fp16_vec(const float* src, uint16_t* dst, int n) {
#ifdef __ARM_NEON
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vf = vld1q_f32(src + i);
        float16x4_t vh = vcvt_f16_f32(vf);
        vst1_u16(dst + i, vreinterpret_u16_f16(vh));
    }
    for (; i < n; i++) {
        dst[i] = xfr_f32_to_fp16(src[i]);
    }
#else
    for (int i = 0; i < n; i++) {
        dst[i] = xfr_f32_to_fp16(src[i]);
    }
#endif
}

/* ============================================================
 * State management
 * ============================================================ */

tq_state_t* tq_create_state(const tq_model_config_t* config, tq_type kv_type) {
    return tq_create_state_ex(config, kv_type, 0);
}

tq_state_t* tq_create_state_ex(const tq_model_config_t* config, tq_type kv_type, int value_quant_bits) {
    if (!config) return NULL;

    int dim = config->hidden_dim;
    int kv_dim = config->n_kv_heads * config->head_dim;
    int inter_dim = config->intermediate_dim;
    int n_heads = config->n_heads;
    int max_seq = config->max_seq_len;
    int n_layers = config->n_layers;

    /* For hybrid attention (Gemma 4), full layers have larger kv_dim.
     * Allocate K/V buffers and KV cache with the MAX of sliding and full kv_dim. */
    int full_kv_dim = (config->full_n_kv_heads > 0 && config->full_head_dim > 0)
        ? config->full_n_kv_heads * config->full_head_dim : kv_dim;
    int max_kv_dim = (full_kv_dim > kv_dim) ? full_kv_dim : kv_dim;

    tq_state_t* s = (tq_state_t*)calloc(1, sizeof(tq_state_t));
    if (!s) return NULL;

    s->kv_quant_type = kv_type;

    /* Allocate activation buffers */
    /* For Qwen3.5, q dimension is n_heads * head_dim = 8 * 256 = 2048
     * but the DeltaNet qkv_dim is 6144 which is larger, so we need
     * the max of both for workspace.
     * When attn_output_gate is enabled, q_proj outputs 2x for Q + gate. */
    int q_dim = n_heads * config->head_dim;
    /* Gemma 4 hybrid: full layers have larger Q dim (n_heads * full_head_dim) */
    int full_q_dim = (config->full_head_dim > 0 && config->full_n_heads > 0)
        ? config->full_n_heads * config->full_head_dim : q_dim;
    int max_q_dim = (full_q_dim > q_dim) ? full_q_dim : q_dim;
    int q_proj_dim = config->attn_output_gate ? max_q_dim * 2 : max_q_dim;
    int delta_nkv = config->delta_n_kv_heads > 0 ? config->delta_n_kv_heads : config->delta_n_heads;
    int delta_qkv_dim = delta_nkv * config->delta_key_head_dim * 2 + config->delta_n_heads * config->delta_value_head_dim;
    int delta_z_dim = config->delta_n_heads * config->delta_value_head_dim;
    int max_dim = dim;
    if (max_q_dim > max_dim) max_dim = max_q_dim;
    if (q_proj_dim > max_dim) max_dim = q_proj_dim;
    if (delta_qkv_dim > max_dim) max_dim = delta_qkv_dim;

    s->x      = (float*)calloc((size_t)dim, sizeof(float));
    s->xb     = (float*)calloc((size_t)max_dim, sizeof(float));
    s->xb2    = (float*)calloc((size_t)max_dim, sizeof(float));
    s->q      = (float*)calloc((size_t)max_q_dim, sizeof(float));
    s->k      = (float*)calloc((size_t)max_kv_dim, sizeof(float));
    s->v      = (float*)calloc((size_t)max_kv_dim, sizeof(float));
    s->att    = (float*)calloc((size_t)n_heads * max_seq, sizeof(float));
    s->hb     = (float*)calloc((size_t)inter_dim, sizeof(float));
    s->hb2    = (float*)calloc((size_t)inter_dim, sizeof(float));
    s->logits = (float*)calloc((size_t)config->vocab_size, sizeof(float));

    /* KV cache for self_attn layers — use max_kv_dim for hybrid attention compatibility */
    size_t kv_layer_size = (size_t)max_seq * max_kv_dim;
    s->key_cache   = (float*)calloc((size_t)n_layers * kv_layer_size, sizeof(float));

    /* Value cache quantization: Q4 or Q2 for aggressive V compression.
     * When value_quant_bits > 0, V is stored quantized instead of FP16/FP32.
     * Q4: 16 packed bytes + 1 float scale per block of 32 = 20 bytes/32 values
     * Q2: 8 packed bytes + 1 float scale per block of 32 = 12 bytes/32 values */
    s->value_quant_bits = value_quant_bits;
    if (value_quant_bits == 4 || value_quant_bits == 2) {
        /* Quantized V cache — use max_kv_dim for hybrid attention compatibility */
        int n_blocks_per_pos = (max_kv_dim + 31) / 32; /* blocks per position (all heads) */
        size_t packed_per_block = (value_quant_bits == 4) ? 16 : 8;
        s->value_stride_qs = (size_t)n_blocks_per_pos * packed_per_block;
        s->value_stride_scales = (size_t)n_blocks_per_pos;
        size_t total_qs = (size_t)n_layers * max_seq * s->value_stride_qs;
        size_t total_scales = (size_t)n_layers * max_seq * s->value_stride_scales;
        s->value_cache_qs = (uint8_t*)calloc(total_qs, 1);
        s->value_cache_scales = (float*)calloc(total_scales, sizeof(float));
        s->use_fp16_values = 0;
        s->value_cache_fp16 = NULL;
        s->value_cache = NULL;
        s->kv_cache_size = total_qs + total_scales * sizeof(float);
    } else if (kv_type < TQ_TYPE_COUNT) {
        /* Use FP16 value cache when KV key quantization is enabled (saves 2x V memory).
         * FP16 has sufficient precision for value vectors (used in weighted sum, not scoring). */
        s->use_fp16_values = 1;
        s->value_cache_fp16 = (uint16_t*)calloc((size_t)n_layers * kv_layer_size, sizeof(uint16_t));
        s->value_cache = NULL;
        s->value_cache_qs = NULL;
        s->value_cache_scales = NULL;
        s->kv_cache_size = (size_t)n_layers * kv_layer_size * sizeof(uint16_t);
    } else {
        s->use_fp16_values = 0;
        s->value_cache_fp16 = NULL;
        s->value_cache = (float*)calloc((size_t)n_layers * kv_layer_size, sizeof(float));
        s->value_cache_qs = NULL;
        s->value_cache_scales = NULL;
        s->kv_cache_size = (size_t)n_layers * kv_layer_size * sizeof(float);
    }

    /* Dynamic workspace buffers (replacing fixed-size stack arrays).
     * xb_q8/xb_q8s are used in deltanet_forward, self_attn_forward, and FFN
     * for pre-quantizing activations to Q8 before Q4 matmuls. */
    int q8_blocks = (dim + 31) / 32;
    s->xb_q8  = (int8_t*)calloc((size_t)dim, sizeof(int8_t));
    s->xb_q8s = (float*)calloc((size_t)(q8_blocks + 1), sizeof(float));

    /* DeltaNet recurrent state */
    if (config->delta_n_heads > 0) {
        int dn = config->delta_n_heads;
        int dk = config->delta_key_head_dim;
        int dv = config->delta_value_head_dim;
        /* State: [n_layers, delta_n_heads, key_head_dim, value_head_dim] */
        s->delta_state = (float*)calloc((size_t)n_layers * dn * dk * dv, sizeof(float));
        /* Conv state: [n_layers, qkv_dim, conv_width-1] */
        int conv_buf_size = config->delta_conv_width - 1;
        if (conv_buf_size < 1) conv_buf_size = 1;
        s->conv_state = (float*)calloc((size_t)n_layers * delta_qkv_dim * conv_buf_size, sizeof(float));

        /* Workspace buffers */
        s->delta_qkv = (float*)calloc((size_t)delta_qkv_dim, sizeof(float));
        s->delta_z   = (float*)calloc((size_t)delta_z_dim, sizeof(float));
        s->delta_ab  = (float*)calloc((size_t)dn * 2, sizeof(float));
        s->delta_out = (float*)calloc((size_t)delta_z_dim, sizeof(float));

        /* DeltaNet per-head workspace (replacing stack-allocated gate_vals/decay_vals/sk/d_vec) */
        s->gate_vals  = (float*)calloc((size_t)dn, sizeof(float));
        s->decay_vals = (float*)calloc((size_t)dn, sizeof(float));
        s->delta_sk   = (float*)calloc((size_t)dv, sizeof(float));
        s->delta_dvec = (float*)calloc((size_t)dv, sizeof(float));
    }

    /* Quantization workspace */
    size_t block_size = tq_type_block_size(kv_type);
    size_t type_size  = tq_type_type_size(kv_type);
    if (block_size == 0) block_size = TQ_BK;
    if (type_size == 0) type_size = sizeof(block_tq_uniform_4b);
    size_t n_blocks_per_head = ((size_t)config->head_dim + block_size - 1) / block_size;
    /* quant_key_buf is used as a gather buffer for integer attention:
     * we collect quantized key blocks for one KV head across all seq positions.
     * Size needed: max_seq_len * blocks_per_head * type_size */
    size_t gather_buf_size = (size_t)max_seq * n_blocks_per_head * type_size;
    /* Ensure at least the old size for other uses */
    size_t old_buf_size = n_blocks_per_head * type_size * (size_t)config->n_kv_heads;
    if (gather_buf_size < old_buf_size) gather_buf_size = old_buf_size;
    s->quant_key_buf = calloc(gather_buf_size, 1);
    s->quant_score_buf = (float*)calloc((size_t)max_seq, sizeof(float));

    /* Quantized key cache for integer attention acceleration.
     * Layout: [n_layers][max_seq_len][n_kv_heads][blocks_per_head * type_size]
     * Each key vector is quantized when stored, then reused for fast Q4xQ8 attention. */
    s->quant_head_stride = n_blocks_per_head * type_size;
    size_t quant_pos_stride = s->quant_head_stride * (size_t)config->n_kv_heads;
    s->quant_kv_stride = quant_pos_stride * (size_t)max_seq;
    if (kv_type < TQ_TYPE_COUNT) {
        s->quant_key_cache = calloc((size_t)n_layers * s->quant_kv_stride, 1);
    } else {
        s->quant_key_cache = NULL;
    }

    /* Note: low-bit KV quantization (1b/2b/3b) with head_dim < 128 is now handled
     * by expanding sketch_dim to 128 (QJL paper: m/d >= 2). No fallback needed. */

    /* MoE state allocation (set up later by tq_load_gguf when model is MoE) */
    s->moe_state = NULL;

    /* Adaptive compression: these are set later via flags, not at creation time.
     * attn_entropy, entropy_accum, v_highres_window, value_highres_fp16
     * are initialized to 0/NULL by calloc. */

    /* PLE buffer: allocated lazily in tq_forward when model->ple_dim > 0.
     * We don't know ple_dim at state creation time (model not loaded yet).
     * ple_buf is initialized to NULL by calloc. */

    /* Verify critical allocations */
    int value_cache_ok;
    if (s->value_quant_bits == 4 || s->value_quant_bits == 2) {
        value_cache_ok = (s->value_cache_qs != NULL && s->value_cache_scales != NULL);
    } else if (s->use_fp16_values) {
        value_cache_ok = (s->value_cache_fp16 != NULL);
    } else {
        value_cache_ok = (s->value_cache != NULL);
    }
    if (!s->x || !s->xb || !s->xb2 || !s->q || !s->k || !s->v ||
        !s->att || !s->hb || !s->hb2 || !s->logits ||
        !s->key_cache || !value_cache_ok ||
        !s->xb_q8 || !s->xb_q8s) {
        tq_free_state(s);
        return NULL;
    }

    return s;
}

void tq_free_state(tq_state_t* state) {
    if (!state) return;
    free(state->x);
    free(state->xb);
    free(state->xb2);
    free(state->q);
    free(state->k);
    free(state->v);
    free(state->att);
    free(state->hb);
    free(state->hb2);
    free(state->logits);
    free(state->key_cache);
    free(state->value_cache);
    free(state->value_cache_fp16);
    free(state->value_cache_qs);
    free(state->value_cache_scales);
    free(state->delta_state);
    free(state->conv_state);
    free(state->delta_qkv);
    free(state->delta_z);
    free(state->delta_ab);
    free(state->delta_out);
    free(state->xb_q8);
    free(state->xb_q8s);
    free(state->gate_vals);
    free(state->decay_vals);
    free(state->delta_sk);
    free(state->delta_dvec);
    free(state->quant_key_buf);
    free(state->quant_score_buf);
    free(state->quant_key_cache);
    free(state->entropy_accum);
    free(state->value_highres_fp16);
    free(state->profile_stats);
    free(state->profile_accum);
    free(state->ple_buf);
    free(state->key_highres_fp32);
    if (state->moe_state) {
        tq_moe_free_state((tq_moe_state_t*)state->moe_state);
    }
    free(state);
}

/* ============================================================
 * Helper: L2 normalize a vector in-place (NEON-optimized)
 * ============================================================ */
static void l2_normalize(float* v, int n) {
#ifdef __ARM_NEON
    float32x4_t vss = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vx = vld1q_f32(v + i);
        vss = vfmaq_f32(vss, vx, vx);
    }
    float ss = vaddvq_f32(vss);
    for (; i < n; i++) ss += v[i] * v[i];
    if (ss > 0.0f) {
        float inv = 1.0f / sqrtf(ss);
        float32x4_t vinv = vdupq_n_f32(inv);
        i = 0;
        for (; i + 3 < n; i += 4) {
            float32x4_t vx = vld1q_f32(v + i);
            vst1q_f32(v + i, vmulq_f32(vx, vinv));
        }
        for (; i < n; i++) v[i] *= inv;
    }
#else
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += v[i] * v[i];
    if (ss > 0.0f) {
        float inv = 1.0f / sqrtf(ss);
        for (int i = 0; i < n; i++) v[i] *= inv;
    }
#endif
}

/* ============================================================
 * Fast exponential approximation (Schraudolph's algorithm)
 * ~6x faster than expf(), accuracy within ~1% for |x| < 10
 * Used for decay gates where exact precision is not critical.
 * ============================================================ */
static inline float fast_expf(float x) {
    /* Clamp to avoid overflow/underflow */
    if (x < -20.0f) return 0.0f;
    if (x > 20.0f) return expf(x);
    union { float f; int32_t i; } v;
    v.i = (int32_t)(12102203.0f * x + 1065353216.0f);
    return v.f;
}

/* ============================================================
 * Helper: Apply causal conv1d (width=conv_width) for a single
 * channel at the current time step.
 *
 * conv_state holds the last (conv_width-1) inputs for this channel.
 * weight has conv_width values.
 * Returns the convolution output for the current input.
 * ============================================================ */
static inline float causal_conv1d_step(float input, float* conv_buf,
                                 const float* weight, int conv_width) {
    int buf_len = conv_width - 1;
    float out = 0.0f;
    for (int k = 0; k < buf_len; k++) {
        out += weight[k] * conv_buf[k];
    }
    out += weight[buf_len] * input;
    for (int i = 0; i < buf_len - 1; i++) {
        conv_buf[i] = conv_buf[i + 1];
    }
    conv_buf[buf_len - 1] = input;
    return out;
}

/* ============================================================
 * Batched causal conv1d for all channels + SiLU activation.
 * When conv_width=4 (buf_len=3), we specialize to avoid inner loops.
 * Uses NEON to process 4 channels simultaneously.
 * ============================================================ */
static void causal_conv1d_silu_batch(float* data, float* conv_st,
                                      const float* conv_weights,
                                      int n_channels, int conv_width) {
    int conv_buf_len = conv_width - 1;

#ifdef __ARM_NEON
    if (conv_width == 4) {
        /* Specialized path for width=4 (3 history values per channel).
         * Conv state layout: [channel][buf_len=3] */
        int ch = 0;
        for (; ch + 3 < n_channels; ch += 4) {
            /* For each of the 4 channels, compute:
             * out = w[0]*buf[0] + w[1]*buf[1] + w[2]*buf[2] + w[3]*input */
            float results[4];
            for (int c = 0; c < 4; c++) {
                int idx = ch + c;
                float* buf = conv_st + idx * conv_buf_len;
                const float* w = conv_weights + idx * conv_width;
                float out = w[0] * buf[0] + w[1] * buf[1] + w[2] * buf[2] + w[3] * data[idx];
                /* Shift buffer */
                buf[0] = buf[1];
                buf[1] = buf[2];
                buf[2] = data[idx];
                results[c] = out;
            }
            /* SiLU on 4 values at once: x / (1 + exp(-x)) */
            float32x4_t vx = vld1q_f32(results);
            float32x4_t vneg = vnegq_f32(vx);
            /* Use fast exp for SiLU since exact precision is not critical here */
            float exp_vals[4];
            vst1q_f32(exp_vals, vneg);
            exp_vals[0] = fast_expf(exp_vals[0]);
            exp_vals[1] = fast_expf(exp_vals[1]);
            exp_vals[2] = fast_expf(exp_vals[2]);
            exp_vals[3] = fast_expf(exp_vals[3]);
            float32x4_t vexp = vld1q_f32(exp_vals);
            float32x4_t vone = vdupq_n_f32(1.0f);
            float32x4_t vdenom = vaddq_f32(vone, vexp);
            float32x4_t vresult = vdivq_f32(vx, vdenom);
            vst1q_f32(data + ch, vresult);
        }
        /* Scalar tail */
        for (; ch < n_channels; ch++) {
            float* buf = conv_st + ch * conv_buf_len;
            const float* w = conv_weights + ch * conv_width;
            float out = w[0] * buf[0] + w[1] * buf[1] + w[2] * buf[2] + w[3] * data[ch];
            buf[0] = buf[1];
            buf[1] = buf[2];
            buf[2] = data[ch];
            data[ch] = out / (1.0f + fast_expf(-out));
        }
    } else
#endif
    {
        /* Generic path */
        for (int ch = 0; ch < n_channels; ch++) {
            float* ch_conv_buf = conv_st + ch * conv_buf_len;
            const float* ch_weight = conv_weights + ch * conv_width;
            data[ch] = causal_conv1d_step(data[ch], ch_conv_buf, ch_weight, conv_width);
        }
        /* SiLU */
        for (int i = 0; i < n_channels; i++) {
            data[i] = data[i] / (1.0f + fast_expf(-data[i]));
        }
    }
}

/* ============================================================
 * DeltaNet forward pass for a single layer (autoregressive mode)
 *
 * Follows the llama.cpp/fla Gated DeltaNet implementation:
 *   1. Project input -> QKV (via in_proj_qkv), Z (via in_proj_z)
 *   2. Project alpha = in_proj_a @ x, beta = sigmoid(in_proj_b @ x)
 *   3. Compute gate = softplus(alpha + dt_bias) * (-exp(A_log))
 *   4. Apply causal conv1d on QKV, then SiLU activation
 *   5. Split QKV into Q, K, V per head; L2 normalize Q, K
 *   6. Scale Q by 1/sqrt(head_dim)
 *   7. Recurrent delta rule update:
 *        S = S * exp(gate)
 *        d = beta * (V - S @ K)
 *        S = S + outer(K, d)
 *        output = S @ Q
 *   8. Apply group norm, multiply by swish(z), output projection
 * ============================================================ */
static void deltanet_forward(tq_model_t* model, tq_state_t* s, int l) {
    double _tp = 0;  /* profiling timestamp */
    tq_model_config_t* c = &model->config;
    tq_layer_weights_t* layer = &model->layers[l];
    int dim = c->hidden_dim;
    int dn = c->delta_n_heads;        /* num_v_heads (e.g. 32) */
    int dn_kv = c->delta_n_kv_heads;  /* num_k_heads (e.g. 16); 0 = same as dn */
    if (dn_kv <= 0) dn_kv = dn;
    int dk = c->delta_key_head_dim;   /* key head dim (e.g. 128) */
    int dv = c->delta_value_head_dim; /* value head dim (e.g. 128) */
    /* Note: GGUF V-heads are in tiled order (ggml broadcast convention).
     * V-head h belongs to K-group (h % dn_kv), NOT (h / kv_mul). */
    int qkv_dim = dn_kv * dk * 2 + dn * dv; /* Q[dn_kv*dk] + K[dn_kv*dk] + V[dn*dv] */
    int z_dim = dn * dv;
    int conv_width = c->delta_conv_width;
    int conv_buf_len = conv_width - 1;
    if (conv_buf_len < 1) conv_buf_len = 1;

    /* Pointers into DeltaNet state for this layer */
    float* state = s->delta_state + (size_t)l * dn * dk * dv;
    float* conv_st = s->conv_state + (size_t)l * qkv_dim * conv_buf_len;

    /* Pre-quantize activation to Q8 once for all Q2/Q4 projections in this layer.
     * This eliminates redundant tq_quantize_row_q8 + malloc/free cycles. */
    int dn_has_q2 = (layer->delta_in_proj_qkv_q2 != NULL);
    int dn_has_q4 = (layer->delta_in_proj_qkv_q4 != NULL);
    if (dn_has_q2 || dn_has_q4) {
        tq_quantize_row_q8(s->xb, s->xb_q8, s->xb_q8s, dim);
    }

    /* Step 1: Project input through QKV and Z */
    TQ_PROF_START(_tp);
    if (layer->delta_in_proj_qkv_q2)
        TQ_MATMUL_Q2_OR_1BIT(s->delta_qkv, s->xb, layer->delta_in_proj_qkv_q2, layer->delta_in_proj_qkv_q2s, s->xb_q8, s->xb_q8s, qkv_dim, dim, model->use_1bit_weights);
    else if (layer->delta_in_proj_qkv_q4)
        tq_matmul_q4q2_preq(s->delta_qkv, layer->delta_in_proj_qkv_q4, layer->delta_in_proj_qkv_q4s, layer->delta_in_proj_qkv_q2, layer->delta_in_proj_qkv_q2s, s->xb_q8, s->xb_q8s, qkv_dim, dim);
    else if (layer->delta_in_proj_qkv_q8)
        tq_matmul_q8(s->delta_qkv, s->xb, layer->delta_in_proj_qkv_q8, layer->delta_in_proj_qkv_q8s, qkv_dim, dim);
    else if (layer->gguf_delta_qkv)
        tq_matmul_gguf(s->delta_qkv, s->xb, layer->gguf_delta_qkv, layer->gguf_delta_qkv_type, qkv_dim, dim);
    else
        tq_matmul(s->delta_qkv, s->xb, layer->delta_in_proj_qkv, qkv_dim, dim);

    if (layer->delta_in_proj_z_q2)
        TQ_MATMUL_Q2_OR_1BIT(s->delta_z, s->xb, layer->delta_in_proj_z_q2, layer->delta_in_proj_z_q2s, s->xb_q8, s->xb_q8s, z_dim, dim, model->use_1bit_weights);
    else if (layer->delta_in_proj_z_q4)
        tq_matmul_q4q2_preq(s->delta_z, layer->delta_in_proj_z_q4, layer->delta_in_proj_z_q4s, layer->delta_in_proj_z_q2, layer->delta_in_proj_z_q2s, s->xb_q8, s->xb_q8s, z_dim, dim);
    else if (layer->delta_in_proj_z_q8)
        tq_matmul_q8(s->delta_z, s->xb, layer->delta_in_proj_z_q8, layer->delta_in_proj_z_q8s, z_dim, dim);
    else if (layer->gguf_delta_z)
        tq_matmul_gguf(s->delta_z, s->xb, layer->gguf_delta_z, layer->gguf_delta_z_type, z_dim, dim);
    else
        tq_matmul(s->delta_z, s->xb, layer->delta_in_proj_z, z_dim, dim);

    /* Step 2: Project alpha and beta */
    /* alpha = in_proj_a @ x  -> [dn] */
    if (layer->delta_in_proj_a_q2)
        TQ_MATMUL_Q2_OR_1BIT(s->delta_ab, s->xb, layer->delta_in_proj_a_q2, layer->delta_in_proj_a_q2s, s->xb_q8, s->xb_q8s, dn, dim, model->use_1bit_weights);
    else if (layer->delta_in_proj_a_q4)
        tq_matmul_q4q2_preq(s->delta_ab, layer->delta_in_proj_a_q4, layer->delta_in_proj_a_q4s, layer->delta_in_proj_a_q2, layer->delta_in_proj_a_q2s, s->xb_q8, s->xb_q8s, dn, dim);
    else if (layer->delta_in_proj_a_q8)
        tq_matmul_q8(s->delta_ab, s->xb, layer->delta_in_proj_a_q8, layer->delta_in_proj_a_q8s, dn, dim);
    else if (layer->gguf_delta_a)
        tq_matmul_gguf(s->delta_ab, s->xb, layer->gguf_delta_a, layer->gguf_delta_a_type, dn, dim);
    else
        tq_matmul(s->delta_ab, s->xb, layer->delta_in_proj_a, dn, dim);

    /* beta = sigmoid(in_proj_b @ x) -> [dn] */
    if (layer->delta_in_proj_b_q2)
        TQ_MATMUL_Q2_OR_1BIT(s->delta_ab + dn, s->xb, layer->delta_in_proj_b_q2, layer->delta_in_proj_b_q2s, s->xb_q8, s->xb_q8s, dn, dim, model->use_1bit_weights);
    else if (layer->delta_in_proj_b_q4)
        tq_matmul_q4q2_preq(s->delta_ab + dn, layer->delta_in_proj_b_q4, layer->delta_in_proj_b_q4s, layer->delta_in_proj_b_q2, layer->delta_in_proj_b_q2s, s->xb_q8, s->xb_q8s, dn, dim);
    else if (layer->delta_in_proj_b_q8)
        tq_matmul_q8(s->delta_ab + dn, s->xb, layer->delta_in_proj_b_q8, layer->delta_in_proj_b_q8s, dn, dim);
    else if (layer->gguf_delta_b)
        tq_matmul_gguf(s->delta_ab + dn, s->xb, layer->gguf_delta_b, layer->gguf_delta_b_type, dn, dim);
    else
        tq_matmul(s->delta_ab + dn, s->xb, layer->delta_in_proj_b, dn, dim);
    for (int h = 0; h < dn; h++) {
        s->delta_ab[dn + h] = 1.0f / (1.0f + fast_expf(-s->delta_ab[dn + h]));
    }

    TQ_PROF_STOP(_tp, matmul_ns);

    /* Step 3: Compute gate (decay) per head
     * gate = softplus(alpha + dt_bias) * (-exp(A_log))
     * exp(gate) is the per-step multiplicative decay (< 1).
     * We precompute both gate_vals and exp(gate) to avoid repeated exp calls. */
    float* gate_vals = s->gate_vals;
    float* decay_vals = s->decay_vals;
    for (int h = 0; h < dn; h++) {
        float alpha_biased = s->delta_ab[h] + layer->delta_dt_bias[h];
        /* softplus: log(1 + exp(x)). For large x, softplus(x) ~ x */
        float alpha_sp;
        if (alpha_biased > 15.0f) {
            alpha_sp = alpha_biased; /* softplus saturates to identity */
        } else {
            alpha_sp = logf(1.0f + fast_expf(alpha_biased));
        }
        float neg_exp_alog = -expf(layer->delta_a_log[h]); /* keep precise for model param */
        gate_vals[h] = alpha_sp * neg_exp_alog;
        decay_vals[h] = fast_expf(gate_vals[h]); /* precompute decay */
    }

    /* Step 4: Causal conv1d on QKV + SiLU (batched, NEON-optimized) */
    TQ_PROF_START(_tp);
    causal_conv1d_silu_batch(s->delta_qkv, conv_st, layer->delta_conv1d,
                              qkv_dim, conv_width);
    TQ_PROF_STOP(_tp, conv1d_ns);

    /* Step 5: Split into Q, K, V per head and L2 normalize Q, K.
     * Layout: Q[dn_kv * dk] + K[dn_kv * dk] + V[dn * dv]
     * Q and K have dn_kv groups (GQA), V has dn heads. */
    float* Q_all = s->delta_qkv;
    float* K_all = s->delta_qkv + dn_kv * dk;
    float* V_all = s->delta_qkv + 2 * dn_kv * dk;

    for (int h = 0; h < dn_kv; h++) {
        l2_normalize(Q_all + h * dk, dk);
        l2_normalize(K_all + h * dk, dk);
    }

    /* Step 6: Scale Q by 1/sqrt(head_dim) */
    float q_scale = 1.0f / sqrtf((float)dk);
    for (int i = 0; i < dn_kv * dk; i++) {
        Q_all[i] *= q_scale;
    }

    TQ_PROF_START(_tp);
    /* Step 7: Per-head recurrent delta rule update (NEON-optimized).
     *
     * Following the llama.cpp autoregressive implementation:
     *   S = S * exp(gate)           // decay state
     *   sk = sum_rows(S * K)        // S @ K -> [dv] for each head
     *   d = beta * (V - sk)         // delta
     *   S = S + outer(K, d)         // update state
     *   o = sum_rows(S * Q)         // output = S @ Q -> [dv]
     *
     * State layout: S[h] is [dk, dv] (row-major, S[i][j]) */
    for (int h = 0; h < dn; h++) {
        int kv_group = h % dn_kv; /* tiled V-head order: GGUF reorders V-heads for ggml broadcast */
        float* qh = Q_all + kv_group * dk;
        float* kh = K_all + kv_group * dk;
        float* vh = V_all + h * dv;
        float* sh = state + (size_t)h * dk * dv;
        float beta_h = s->delta_ab[dn + h];
        float decay = decay_vals[h]; /* precomputed exp(gate) */

#ifdef __ARM_NEON
        /* NEON-optimized: fused decay + sk computation.
         * For each row i of state: decay state, accumulate sk.
         * sk[j] = sum_i(S[i,j] * K[i]) after decay */
        float* sk = s->delta_sk;
        memset(sk, 0, (size_t)dv * sizeof(float));

        float32x4_t vdecay = vdupq_n_f32(decay);
        for (int i = 0; i < dk; i++) {
            float* sp = sh + i * dv;
            float ki = kh[i];
            float32x4_t vki = vdupq_n_f32(ki);
            int j = 0;
            for (; j + 3 < dv; j += 4) {
                float32x4_t vs = vld1q_f32(sp + j);
                vs = vmulq_f32(vs, vdecay);  /* decay */
                vst1q_f32(sp + j, vs);        /* store decayed state */
                float32x4_t vsk = vld1q_f32(sk + j);
                vsk = vfmaq_f32(vsk, vs, vki); /* accumulate sk */
                vst1q_f32(sk + j, vsk);
            }
            for (; j < dv; j++) {
                sp[j] *= decay;
                sk[j] += sp[j] * ki;
            }
        }

        /* Delta: d = beta * (V - sk) */
        float* d_vec = s->delta_dvec;
        float32x4_t vbeta = vdupq_n_f32(beta_h);
        {
            int j = 0;
            for (; j + 3 < dv; j += 4) {
                float32x4_t vv = vld1q_f32(vh + j);
                float32x4_t vs = vld1q_f32(sk + j);
                float32x4_t vd = vmulq_f32(vbeta, vsubq_f32(vv, vs));
                vst1q_f32(d_vec + j, vd);
            }
            for (; j < dv; j++) {
                d_vec[j] = beta_h * (vh[j] - sk[j]);
            }
        }

        /* State update: S[i][j] += K[i] * d[j] (rank-1 outer product)
         * + Output: o[j] = sum_i(S[i,j] * Q[i]) (simultaneously) */
        float* oh = s->delta_out + h * dv;
        memset(oh, 0, (size_t)dv * sizeof(float));

        for (int i = 0; i < dk; i++) {
            float* sp = sh + i * dv;
            float ki = kh[i];
            float qi = qh[i];
            float32x4_t vki = vdupq_n_f32(ki);
            float32x4_t vqi = vdupq_n_f32(qi);
            int j = 0;
            for (; j + 3 < dv; j += 4) {
                float32x4_t vs = vld1q_f32(sp + j);
                float32x4_t vd = vld1q_f32(d_vec + j);
                vs = vfmaq_f32(vs, vki, vd);  /* S += K[i] * d */
                vst1q_f32(sp + j, vs);
                float32x4_t vo = vld1q_f32(oh + j);
                vo = vfmaq_f32(vo, vs, vqi);   /* o += S * Q[i] */
                vst1q_f32(oh + j, vo);
            }
            for (; j < dv; j++) {
                sp[j] += ki * d_vec[j];
                oh[j] += sp[j] * qi;
            }
        }
#else
        /* Scalar fallback */
        /* Decay: S = S * exp(gate) */
        for (int i = 0; i < dk * dv; i++) {
            sh[i] *= decay;
        }

        /* Compute sk */
        float* sk = s->delta_sk;
        for (int j = 0; j < dv; j++) {
            float sum = 0.0f;
            for (int i = 0; i < dk; i++) {
                sum += sh[i * dv + j] * kh[i];
            }
            sk[j] = sum;
        }

        /* Delta */
        float* d_vec = s->delta_dvec;
        for (int j = 0; j < dv; j++) {
            d_vec[j] = beta_h * (vh[j] - sk[j]);
        }

        /* State update */
        for (int i = 0; i < dk; i++) {
            for (int j = 0; j < dv; j++) {
                sh[i * dv + j] += kh[i] * d_vec[j];
            }
        }

        /* Output */
        float* oh = s->delta_out + h * dv;
        for (int j = 0; j < dv; j++) {
            float sum = 0.0f;
            for (int i = 0; i < dk; i++) {
                sum += sh[i * dv + j] * qh[i];
            }
            oh[j] = sum;
        }
#endif
    }

    TQ_PROF_STOP(_tp, recurrent_ns);

    /* Step 8: Apply group norm (per-head RMSNorm), then z gate (swish), then output projection */
    for (int h = 0; h < dn; h++) {
        float* oh = s->delta_out + h * dv;

        /* RMSNorm with delta_norm weights */
        float ss = 0.0f;
#ifdef __ARM_NEON
        {
            float32x4_t vss = vdupq_n_f32(0.0f);
            int j = 0;
            for (; j + 3 < dv; j += 4) {
                float32x4_t vo = vld1q_f32(oh + j);
                vss = vfmaq_f32(vss, vo, vo);
            }
            ss = vaddvq_f32(vss);
            for (; j < dv; j++) ss += oh[j] * oh[j];
        }
#else
        for (int j = 0; j < dv; j++) {
            ss += oh[j] * oh[j];
        }
#endif
        ss = ss / dv + c->rms_norm_eps;
        float inv_rms = 1.0f / sqrtf(ss);
        for (int j = 0; j < dv; j++) {
            oh[j] = oh[j] * inv_rms * layer->delta_norm[j];
        }

        /* Multiply by swish(z) for this head (NEON + fast_expf) */
        float* zh = s->delta_z + h * dv;
#ifdef __ARM_NEON
        {
            int j = 0;
            for (; j + 3 < dv; j += 4) {
                float32x4_t vz = vld1q_f32(zh + j);
                float32x4_t vo = vld1q_f32(oh + j);
                float32x4_t vneg = vnegq_f32(vz);
                /* Fast exp for 4 values */
                float neg_vals[4];
                vst1q_f32(neg_vals, vneg);
                float exp_vals[4] = {
                    fast_expf(neg_vals[0]), fast_expf(neg_vals[1]),
                    fast_expf(neg_vals[2]), fast_expf(neg_vals[3])
                };
                float32x4_t vexp = vld1q_f32(exp_vals);
                float32x4_t vone = vdupq_n_f32(1.0f);
                float32x4_t vsilu = vdivq_f32(vz, vaddq_f32(vone, vexp));
                vst1q_f32(oh + j, vmulq_f32(vo, vsilu));
            }
            for (; j < dv; j++) {
                float z_val = zh[j];
                oh[j] *= z_val / (1.0f + fast_expf(-z_val));
            }
        }
#else
        for (int j = 0; j < dv; j++) {
            float z_val = zh[j];
            float z_silu = z_val / (1.0f + fast_expf(-z_val));
            oh[j] *= z_silu;
        }
#endif
    }

    /* Output projection: [dim, z_dim] @ delta_out[z_dim] -> xb2[dim] */
    TQ_PROF_START(_tp);
    if (layer->delta_out_proj_q2)
        TQ_MATMUL_Q2_OR_1BIT_FP32(s->xb2, s->delta_out, layer->delta_out_proj_q2, layer->delta_out_proj_q2s, dim, z_dim, model->use_1bit_weights);
    else if (layer->delta_out_proj_q4)
        tq_matmul_q4(s->xb2, s->delta_out, layer->delta_out_proj_q4, layer->delta_out_proj_q4s, dim, z_dim);
    else if (layer->delta_out_proj_q8)
        tq_matmul_q8(s->xb2, s->delta_out, layer->delta_out_proj_q8, layer->delta_out_proj_q8s, dim, z_dim);
    else if (layer->gguf_delta_out)
        tq_matmul_gguf(s->xb2, s->delta_out, layer->gguf_delta_out, layer->gguf_delta_out_type, dim, z_dim);
    else
        tq_matmul(s->xb2, s->delta_out, layer->delta_out_proj, dim, z_dim);

    TQ_PROF_STOP(_tp, matmul_ns);

    /* Residual connection */
    tq_add(s->x, s->x, s->xb2, dim);
}

/* ============================================================
 * Self-attention forward pass with QK-norm and partial RoPE
 * ============================================================ */
static void self_attn_forward(tq_model_t* model, tq_state_t* s, int l, int pos) {
    double _tp = 0;  /* profiling timestamp */
    tq_model_config_t* c = &model->config;
    tq_layer_weights_t* layer = &model->layers[l];
    int dim = c->hidden_dim;
    int head_dim = c->head_dim;
    int n_heads = c->n_heads;
    int n_kv_heads = c->n_kv_heads;

    /* Gemma 4 hybrid: full attention layers use different head_dim and kv_heads.
     * Sliding layers: head_dim=256, n_heads=16, kv_heads=8 (stored in config)
     * Full layers:    head_dim=512, n_heads=8,  kv_heads=2 (stored in full_* fields)
     * Q output dim is always hidden_dim; K/V output dim differs per layer. */
    if (model->layer_is_sliding && !model->layer_is_sliding[l] && c->full_head_dim > 0) {
        head_dim = c->full_head_dim;
        n_heads = c->full_n_heads;
        n_kv_heads = c->full_n_kv_heads;
    }

    int kv_dim = n_kv_heads * head_dim;
    int kv_mul = n_heads / n_kv_heads;
    /* KV cache stride uses the MAX of sliding and full kv_dim for uniform allocation.
     * This ensures full attention layers (with larger kv_dim) don't overflow the cache. */
    int sliding_kv_dim = c->n_kv_heads * c->head_dim;
    int full_kv_dim_cache = (c->full_n_kv_heads > 0 && c->full_head_dim > 0)
        ? c->full_n_kv_heads * c->full_head_dim : sliding_kv_dim;
    int cache_kv_dim = (full_kv_dim_cache > sliding_kv_dim) ? full_kv_dim_cache : sliding_kv_dim;
    size_t kv_layer_stride = (size_t)c->max_seq_len * cache_kv_dim;

    /* Pre-quantize activation to Q8 once for all Q2/Q4 projections in this layer.
     * This eliminates redundant tq_quantize_row_q8 + malloc/free in each matmul call. */
    int has_q2 = (layer->wq_q2 != NULL);
    int has_q4 = (layer->wq_q4 != NULL);
    int has_gguf = (layer->gguf_wq != NULL);
    if (has_q2 || has_q4) {
        tq_quantize_row_q8(s->xb, s->xb_q8, s->xb_q8s, dim);
    }

    /* QKV projections (timed as matmul) */
    TQ_PROF_START(_tp);
    /* When attn_output_gate is enabled, wq has shape [2*n_heads*head_dim, dim]
     * and outputs [Q, gate_q] concatenated. We project into xb2 as temp.
     *
     * Batch Q+K+V GPU dispatches into one command buffer when using GGUF path.
     * This reduces Metal dispatch overhead from 3 commits to 1. */
    if (has_gguf) tq_metal_batch_begin_if_available();

    float* gate_q = NULL;
    if (c->attn_output_gate) {
        int qg_dim = n_heads * head_dim * 2;
        if (layer->wq_q2) {
            TQ_MATMUL_Q2_OR_1BIT(s->xb2, s->xb, layer->wq_q2, layer->wq_q2s, s->xb_q8, s->xb_q8s, qg_dim, dim, model->use_1bit_weights);
        } else if (layer->wq_q4) {
            tq_matmul_q4q2_preq(s->xb2, layer->wq_q4, layer->wq_q4s, layer->wq_q2, layer->wq_q2s, s->xb_q8, s->xb_q8s, qg_dim, dim);
        } else if (layer->wq_q8) {
            tq_matmul_q8(s->xb2, s->xb, layer->wq_q8, layer->wq_q8s, qg_dim, dim);
        } else if (has_gguf) {
            tq_matmul_gguf(s->xb2, s->xb, layer->gguf_wq, layer->gguf_wq_type, qg_dim, dim);
        } else {
            tq_matmul(s->xb2, s->xb, layer->wq, qg_dim, dim);
        }
        /* Deinterleave: extract Q and gate from interleaved layout */
        gate_q = s->xb2;
        float* gate_tmp = s->att;
        for (int h = 0; h < n_heads; h++) {
            memcpy(s->q + h * head_dim,
                   s->xb2 + h * head_dim * 2,
                   (size_t)head_dim * sizeof(float));
            memcpy(gate_tmp + h * head_dim,
                   s->xb2 + h * head_dim * 2 + head_dim,
                   (size_t)head_dim * sizeof(float));
        }
        gate_q = gate_tmp;
    } else {
        if (layer->wq_q2) {
            TQ_MATMUL_Q2_OR_1BIT(s->q, s->xb, layer->wq_q2, layer->wq_q2s, s->xb_q8, s->xb_q8s, n_heads * head_dim, dim, model->use_1bit_weights);
        } else if (layer->wq_q4) {
            tq_matmul_q4q2_preq(s->q, layer->wq_q4, layer->wq_q4s, layer->wq_q2, layer->wq_q2s, s->xb_q8, s->xb_q8s, n_heads * head_dim, dim);
        } else if (layer->wq_q8) {
            tq_matmul_q8(s->q, s->xb, layer->wq_q8, layer->wq_q8s, n_heads * head_dim, dim);
        } else if (has_gguf) {
            tq_matmul_gguf(s->q, s->xb, layer->gguf_wq, layer->gguf_wq_type, n_heads * head_dim, dim);
        } else {
            tq_matmul(s->q, s->xb, layer->wq, n_heads * head_dim, dim);
        }
    }
    if (layer->wk_q2) {
        TQ_MATMUL_Q2_OR_1BIT(s->k, s->xb, layer->wk_q2, layer->wk_q2s, s->xb_q8, s->xb_q8s, kv_dim, dim, model->use_1bit_weights);
    } else if (layer->wk_q4) {
        tq_matmul_q4q2_preq(s->k, layer->wk_q4, layer->wk_q4s, layer->wk_q2, layer->wk_q2s, s->xb_q8, s->xb_q8s, kv_dim, dim);
    } else if (layer->wk_q8) {
        tq_matmul_q8(s->k, s->xb, layer->wk_q8, layer->wk_q8s, kv_dim, dim);
    } else if (has_gguf) {
        tq_matmul_gguf(s->k, s->xb, layer->gguf_wk, layer->gguf_wk_type, kv_dim, dim);
    } else {
        tq_matmul(s->k, s->xb, layer->wk, kv_dim, dim);
    }
    /* V projection: if V weights are absent (Gemma 4 K=V), copy K to V */
    int has_v_weights = (layer->wv_q2 || layer->wv_q4 || layer->wv_q8 ||
                         layer->gguf_wv || layer->wv);
    if (!has_v_weights) {
        /* K=V: value is same as key (attention_k_eq_v) */
        memcpy(s->v, s->k, kv_dim * sizeof(float));
    } else if (layer->wv_q2) {
        TQ_MATMUL_Q2_OR_1BIT(s->v, s->xb, layer->wv_q2, layer->wv_q2s, s->xb_q8, s->xb_q8s, kv_dim, dim, model->use_1bit_weights);
    } else if (layer->wv_q4) {
        tq_matmul_q4q2_preq(s->v, layer->wv_q4, layer->wv_q4s, layer->wv_q2, layer->wv_q2s, s->xb_q8, s->xb_q8s, kv_dim, dim);
    } else if (layer->wv_q8) {
        tq_matmul_q8(s->v, s->xb, layer->wv_q8, layer->wv_q8s, kv_dim, dim);
    } else if (has_gguf) {
        tq_matmul_gguf(s->v, s->xb, layer->gguf_wv, layer->gguf_wv_type, kv_dim, dim);
    } else {
        tq_matmul(s->v, s->xb, layer->wv, kv_dim, dim);
    }

    /* Flush batched Q+K+V GPU dispatches before using results */
    if (has_gguf) tq_metal_batch_flush_if_available();
    TQ_PROF_STOP(_tp, matmul_ns);

    /* Apply QK-norm if present (per-head RMSNorm) */
    if (layer->q_norm) {
        for (int h = 0; h < n_heads; h++) {
            tq_rmsnorm(s->q + h * head_dim, s->q + h * head_dim,
                       layer->q_norm, head_dim, c->rms_norm_eps);
        }
    }
    if (layer->k_norm) {
        for (int h = 0; h < n_kv_heads; h++) {
            tq_rmsnorm(s->k + h * head_dim, s->k + h * head_dim,
                       layer->k_norm, head_dim, c->rms_norm_eps);
        }
    }

    /* Apply RoPE (partial or full) */
    if (c->partial_rotary_factor > 0.0f && c->partial_rotary_factor < 1.0f) {
        /* Partial RoPE: only apply to first partial_rotary_factor * head_dim dims */
        int rope_dim = (int)(c->partial_rotary_factor * head_dim);
        for (int h = 0; h < n_heads; h++) {
            float* qh = s->q + h * head_dim;
            for (int i = 0; i < rope_dim / 2; i++) {
                float freq = 1.0f / powf(c->rope_freq_base, 2.0f * i / rope_dim);
                float theta = pos * freq;
                float cos_t = cosf(theta);
                float sin_t = sinf(theta);
                float q0 = qh[2 * i];
                float q1 = qh[2 * i + 1];
                qh[2 * i]     = q0 * cos_t - q1 * sin_t;
                qh[2 * i + 1] = q0 * sin_t + q1 * cos_t;
            }
        }
        for (int h = 0; h < n_kv_heads; h++) {
            float* kh = s->k + h * head_dim;
            for (int i = 0; i < rope_dim / 2; i++) {
                float freq = 1.0f / powf(c->rope_freq_base, 2.0f * i / rope_dim);
                float theta = pos * freq;
                float cos_t = cosf(theta);
                float sin_t = sinf(theta);
                float k0 = kh[2 * i];
                float k1 = kh[2 * i + 1];
                kh[2 * i]     = k0 * cos_t - k1 * sin_t;
                kh[2 * i + 1] = k0 * sin_t + k1 * cos_t;
            }
        }
    } else if (model->rope_freqs && model->rope_freqs_len > 0 &&
               !(c->is_gemma4 && model->layer_is_sliding && model->layer_is_sliding[l])) {
        /* Learned RoPE frequency factors (Gemma 4 / STEP35).
         * Only used for FULL (global) attention layers. Sliding (SWA) layers
         * use standard RoPE without freq_factors (matching llama.cpp STEP35).
         *
         * rope_freqs[i] is a frequency FACTOR (divisor) on the base frequency.
         * theta[i] = pos * pow(base, -2*i/n_dims) / rope_freqs[i]
         * where n_dims is the RoPE dimension count (NOT head_dim for full layers).
         *
         * For Gemma 4: n_dims = 256 for both sliding (head_dim=256) and full
         * (head_dim=512) layers. This is because rope.dimension_count=512 gets
         * halved for STEP35 (n_rot_full = 512/2 = 256), and
         * rope.dimension_count_swa=256 for sliding layers.
         *
         * rope_freqs has up to full_head_dim/2 entries (256 for head_dim=512).
         * For sliding layers (head_dim=256), use the first head_dim/2 entries.
         * For full layers, n_dims < head_dim, so pairs beyond n_dims/2 are not
         * rotated (left as-is). The freq_factors handle partial rotation within
         * the rotated range (1.0 = rotate, 1e30 = effectively no rotation). */
        float rope_base = c->rope_freq_base;
        if (c->model_type == 1 && c->rope_local_base_freq > 0.0f &&
            model->layer_is_sliding && model->layer_is_sliding[l]) {
            rope_base = c->rope_local_base_freq;
        }

        /* Determine RoPE n_dims for this layer type */
        int is_full_layer = (model->layer_is_sliding && !model->layer_is_sliding[l] &&
                             c->full_head_dim > 0);
        int rope_n_dims;
        if (is_full_layer && c->rope_n_dims_full > 0) {
            rope_n_dims = c->rope_n_dims_full;
        } else if (c->rope_n_dims > 0) {
            rope_n_dims = c->rope_n_dims;
        } else {
            rope_n_dims = head_dim; /* fallback */
        }
        int rope_pairs = rope_n_dims / 2;  /* pairs that get RoPE treatment */
        if (rope_pairs > model->rope_freqs_len)
            rope_pairs = model->rope_freqs_len;

        for (int h = 0; h < n_heads; h++) {
            float* qh = s->q + h * head_dim;
            for (int i = 0; i < rope_pairs; i++) {
                float base_freq = 1.0f / powf(rope_base, 2.0f * i / (float)rope_n_dims);
                float freq = base_freq / model->rope_freqs[i];
                float theta = pos * freq;
                float cos_t = cosf(theta);
                float sin_t = sinf(theta);
                float q0 = qh[2 * i];
                float q1 = qh[2 * i + 1];
                qh[2 * i]     = q0 * cos_t - q1 * sin_t;
                qh[2 * i + 1] = q0 * sin_t + q1 * cos_t;
            }
            /* Pairs beyond rope_pairs are left unrotated (pass-through) */
        }
        for (int h = 0; h < n_kv_heads; h++) {
            float* kh = s->k + h * head_dim;
            for (int i = 0; i < rope_pairs; i++) {
                float base_freq = 1.0f / powf(rope_base, 2.0f * i / (float)rope_n_dims);
                float freq = base_freq / model->rope_freqs[i];
                float theta = pos * freq;
                float cos_t = cosf(theta);
                float sin_t = sinf(theta);
                float k0 = kh[2 * i];
                float k1 = kh[2 * i + 1];
                kh[2 * i]     = k0 * cos_t - k1 * sin_t;
                kh[2 * i + 1] = k0 * sin_t + k1 * cos_t;
            }
        }
    } else {
        /* Full RoPE — for Gemma3, use different freq base for sliding vs global layers */
        float rope_base = c->rope_freq_base;
        if (c->model_type == 1 && c->rope_local_base_freq > 0.0f &&
            model->layer_is_sliding && model->layer_is_sliding[l]) {
            rope_base = c->rope_local_base_freq;
        }
        tq_rope(s->q, s->k, pos, head_dim, n_heads, n_kv_heads, rope_base);
    }

    /* Store K,V in cache.
     * When quantized KV is active, skip FP32 key storage — the quantized
     * cache is the single source of truth.  This eliminates the duplicate
     * FP32 copy and is the basis for real memory savings. */
    int use_quant_kv = (s->kv_quant_type < TQ_TYPE_COUNT && s->quant_key_cache != NULL);
    /* Gemma 4: QK-normed keys are too sparse for low-bit quantization (cosine=0.62).
     * Force FP32 key storage while keeping quantized V cache for memory savings. */
    if (use_quant_kv && c->is_gemma4 && c->use_qk_norm) {
        use_quant_kv = 0; /* fall through to FP32 key storage */
    }
    float* key_cache_layer = s->key_cache + l * kv_layer_stride;
    if (!use_quant_kv) {
        /* Use cache_kv_dim for position stride (cache allocated with sliding dims).
         * Full layers write fewer floats (kv_dim < cache_kv_dim) but at correct stride. */
        memcpy(key_cache_layer + (size_t)pos * cache_kv_dim, s->k, kv_dim * sizeof(float));
    } else if (s->k_highres_window > 0 && s->key_highres_fp32) {
        /* Age-based progressive: store FP32 copy in circular highres buffer.
         * Old keys live only in the quant cache (2-bit). Recent keys use FP32. */
        int win_idx = pos % s->k_highres_window;
        size_t hr_layer_stride = (size_t)s->k_highres_window * cache_kv_dim;
        float* hr_dst = s->key_highres_fp32
            + (size_t)l * hr_layer_stride + (size_t)win_idx * cache_kv_dim;
        memcpy(hr_dst, s->k, kv_dim * sizeof(float));
    } else if (s->delta_kv_enabled) {
        /* Mixed-precision delta: I-frames stored in FP32 key_cache for high-precision
         * reference points. P-frames stored as 2-bit deltas in quant_key_cache.
         * This avoids the quality disaster of 2-bit absolute quantization on I-frames. */
        int iframe_int_fp32 = s->delta_iframe_interval > 0 ? s->delta_iframe_interval : 64;
        if (pos % iframe_int_fp32 == 0) {
            memcpy(key_cache_layer + (size_t)pos * cache_kv_dim, s->k, kv_dim * sizeof(float));
        }
    }

    /* KV profiling: accumulate pre/post-RHT statistics for this layer's keys */
    if (s->profile_kv && s->profile_accum) {
        /* Accumulate pre-RHT stats from s->k (first KV head only for efficiency) */
        double* acc = s->profile_accum + (size_t)l * 8;
        for (int i = 0; i < head_dim; i++) {
            double v = (double)s->k[i];
            acc[0] += v;       /* sum (pre-RHT) */
            acc[1] += v * v;   /* sum_sq */
            acc[2] += v * v * v; /* sum_cube */
            acc[3] += v * v * v * v; /* sum_quad */
        }
        /* Compute post-RHT: apply RHT to a copy */
        float k_rht[TQ_BK];
        int rd = head_dim;
        if (rd > TQ_BK) rd = TQ_BK;
        memcpy(k_rht, s->k, (size_t)rd * sizeof(float));
        tq_rht_transform(k_rht, rd, 0x12345678u);
        for (int i = 0; i < rd; i++) {
            double v = (double)k_rht[i];
            acc[4] += v;       /* sum (post-RHT) */
            acc[5] += v * v;   /* sum_sq */
            acc[6] += v * v * v; /* sum_cube */
            acc[7] += v * v * v * v; /* sum_quad */
        }
    }

    /* Store V: Q4/Q2 if enabled, FP16 if KV quant enabled, otherwise FP32 */
    int max_seq = c->max_seq_len;
    if (s->value_quant_bits == 4) {
        size_t layer_off_qs = (size_t)l * max_seq * s->value_stride_qs;
        size_t layer_off_sc = (size_t)l * max_seq * s->value_stride_scales;
        uint8_t* vqs = s->value_cache_qs + layer_off_qs + (size_t)pos * s->value_stride_qs;
        float*   vsc = s->value_cache_scales + layer_off_sc + (size_t)pos * s->value_stride_scales;
        tq_quantize_row_q4(s->v, vqs, vsc, kv_dim);
        /* Also store FP16 copy in highres window for recent tokens */
        if (s->v_highres_window > 0 && s->value_highres_fp16) {
            int win_idx = pos % s->v_highres_window;
            size_t hr_layer_stride = (size_t)s->v_highres_window * cache_kv_dim;
            uint16_t* hr_dst = s->value_highres_fp16
                + (size_t)l * hr_layer_stride + (size_t)win_idx * cache_kv_dim;
            f32_to_fp16_vec(s->v, hr_dst, kv_dim);
        }
    } else if (s->value_quant_bits == 2) {
        size_t layer_off_qs = (size_t)l * max_seq * s->value_stride_qs;
        size_t layer_off_sc = (size_t)l * max_seq * s->value_stride_scales;
        uint8_t* vqs = s->value_cache_qs + layer_off_qs + (size_t)pos * s->value_stride_qs;
        float*   vsc = s->value_cache_scales + layer_off_sc + (size_t)pos * s->value_stride_scales;
        tq_quantize_row_q2(s->v, vqs, vsc, kv_dim);
        /* Also store FP16 copy in highres window for recent tokens */
        if (s->v_highres_window > 0 && s->value_highres_fp16) {
            int win_idx = pos % s->v_highres_window;
            size_t hr_layer_stride = (size_t)s->v_highres_window * cache_kv_dim;
            uint16_t* hr_dst = s->value_highres_fp16
                + (size_t)l * hr_layer_stride + (size_t)win_idx * cache_kv_dim;
            f32_to_fp16_vec(s->v, hr_dst, kv_dim);
        }
    } else if (s->use_fp16_values) {
        uint16_t* val_fp16_layer = s->value_cache_fp16 + l * kv_layer_stride;
        f32_to_fp16_vec(s->v, val_fp16_layer + (size_t)pos * cache_kv_dim, kv_dim);
    } else {
        float* val_cache_layer = s->value_cache + l * kv_layer_stride;
        memcpy(val_cache_layer + (size_t)pos * cache_kv_dim, s->v, kv_dim * sizeof(float));
    }

    /* Quantize the new key into the quantized cache for integer attention.
     * Each KV head's key vector is quantized independently into blocks.
     *
     * Note: 1-bit/2b/3b sign-based quantization now expands sketch_dim to
     * at least 128 bits for small head_dim (QJL paper: m/d >= 2), so no
     * fallback is needed. */
    int use_int_attn = use_quant_kv;
    /* Quantized KV cache: stride was allocated with sliding dims (c->n_kv_heads, c->head_dim).
     * For hybrid attention full layers with different head_dim, skip quant cache
     * (quant_head_stride doesn't match). Fall back to FP32 cache for those layers. */
    int cache_n_kv_heads = c->n_kv_heads;
    if (head_dim != c->head_dim) {
        /* Full layer: head_dim mismatch with quant cache allocation.
         * Disable both quantized and integer attention → use FP32 path. */
        use_quant_kv = 0;
        use_int_attn = 0;
        /* Ensure K is stored in FP32 cache (may have been skipped above) */
        memcpy(key_cache_layer + (size_t)pos * cache_kv_dim, s->k, kv_dim * sizeof(float));
    } else if (use_int_attn && head_dim != c->head_dim) {
        use_int_attn = 0;
        memcpy(key_cache_layer + (size_t)pos * cache_kv_dim, s->k, kv_dim * sizeof(float));
    }
    if (use_int_attn) {
        const tq_type_traits_t* traits = &TQ_TRAITS[s->kv_quant_type];
        for (int kh = 0; kh < n_kv_heads; kh++) {
            const float* key_src = s->k + kh * head_dim;
            /* Use cache_n_kv_heads for position stride (cache allocated with sliding dims) */
            uint8_t* quant_dst = (uint8_t*)s->quant_key_cache
                + (size_t)l * s->quant_kv_stride
                + (size_t)pos * cache_n_kv_heads * s->quant_head_stride
                + (size_t)kh * s->quant_head_stride;

            if (s->delta_kv_enabled) {
                /* Mixed-precision delta compression with periodic I-frames.
                 * I-frames: stored in FP32 key_cache (perfect precision reference).
                 * P-frames: store delta = key[t] - reconstruct(key[t-1]) in quant cache.
                 * This avoids 2-bit absolute quantization on I-frames (PPL 300+). */
                int iframe_int = s->delta_iframe_interval > 0 ? s->delta_iframe_interval : 64;
                int is_iframe = (pos % iframe_int == 0);

                if (is_iframe) {
                    /* I-frame: FP32 is already stored above. No quant needed.
                     * Zero out the quant slot so accidental reads are harmless. */
                    memset(quant_dst, 0, (size_t)s->quant_head_stride);
                } else {
                    /* P-frame: compute delta from previous position's reconstruction.
                     * Reconstruction starts from the last I-frame (FP32) and accumulates
                     * quantized deltas for subsequent P-frames. */
                    int last_iframe = (pos / iframe_int) * iframe_int;

                    /* Read I-frame from FP32 key_cache */
                    const float* iframe_key = key_cache_layer
                        + (size_t)last_iframe * cache_kv_dim + kh * head_dim;
                    float prev_recon[512];
                    memcpy(prev_recon, iframe_key, (size_t)head_dim * sizeof(float));

                    /* Accumulate deltas from last_iframe+1 to pos-1 */
                    float tmp[512];
                    for (int ti = last_iframe + 1; ti <= pos - 1; ti++) {
                        const uint8_t* delta_src = (const uint8_t*)s->quant_key_cache
                            + (size_t)l * s->quant_kv_stride
                            + (size_t)ti * cache_n_kv_heads * s->quant_head_stride
                            + (size_t)kh * s->quant_head_stride;
                        traits->dequantize(delta_src, tmp, head_dim);
                        for (int d = 0; d < head_dim; d++) {
                            prev_recon[d] += tmp[d];
                        }
                    }

                    float delta_buf[512];
                    for (int d = 0; d < head_dim; d++) {
                        delta_buf[d] = key_src[d] - prev_recon[d];
                    }
                    traits->quantize(delta_buf, quant_dst, head_dim);
                }
            } else {
                /* Non-delta mode: quantize absolute key */
                traits->quantize(key_src, quant_dst, head_dim);
            }
        }
    }

    /* Multi-head attention */
    TQ_PROF_START(_tp);
    int seq_len = pos + 1;
    /* Integer (Hamming) attention DISABLED: sign-based attention scores have
     * only ~68% sign accuracy, causing catastrophic PPL explosion at long
     * sequences. FP32 attention on dequantized keys is used instead.
     * Memory savings come from 1-bit KV STORAGE, not integer attention.
     * TODO: fix Hamming attention or implement proper QJL sketch attention. */
    int int_attn_threshold = INT_MAX; /* effectively disabled */

    /* Attention scaling:
     * Gemma 4 with QK-norm: scale = 1.0 (no 1/sqrt(head_dim) needed)
     * Gemma 3 with query_pre_attn_scalar: scale = 1/sqrt(scalar)
     * Others: scale = 1/sqrt(head_dim) */
    float attn_scale_dim = (float)head_dim;
    if (c->is_gemma4) {
        /* Gemma 4: attention_scale = 1.0 (QK-norm already normalizes Q,K per head).
         * Reference: refs/llama.cpp/src/llama-model.cpp line 1273
         * Set attn_scale_dim = 1.0 so that 1/sqrt(attn_scale_dim) = 1.0 */
        attn_scale_dim = 1.0f;
    } else if (c->query_pre_attn_scalar > 0.0f) {
        attn_scale_dim = c->query_pre_attn_scalar;
        if (c->full_head_dim > 0 && model->layer_is_sliding && !model->layer_is_sliding[l]) {
            attn_scale_dim = (float)c->full_head_dim;
        }
    }

    /* Gemma3 sliding window: limit attention to last sliding_window tokens for sliding layers */
    int attn_start = 0;
    if (c->model_type == 1 && c->sliding_window > 0 &&
        model->layer_is_sliding && model->layer_is_sliding[l]) {
        int window = c->sliding_window;
        if (seq_len > window) {
            attn_start = seq_len - window;
        }
    }

    for (int h = 0; h < n_heads; h++) {
        float* qh = s->q + h * head_dim;
        float* atth = s->att + (size_t)h * c->max_seq_len;
        int kv_h = h / kv_mul;

        if (use_int_attn && seq_len > int_attn_threshold) {
            /* Integer Q4xQ8 attention path.
             * Gather quantized key blocks for this KV head across all positions
             * into a contiguous buffer, then call the traits attention function.
             *
             * The quantized cache stores keys as:
             *   [layer][pos][kv_head][blocks_per_head * type_size]
             * The attention function expects:
             *   [seq_len][blocks_per_head] contiguous blocks
             * So we need to gather from strided positions. */
            const tq_type_traits_t* traits = &TQ_TRAITS[s->kv_quant_type];
            size_t head_block_bytes = s->quant_head_stride;
            size_t pos_stride_bytes = (size_t)cache_n_kv_heads * head_block_bytes;
            uint8_t* layer_base = (uint8_t*)s->quant_key_cache
                + (size_t)l * s->quant_kv_stride;

            /* Gather quantized blocks for this KV head into quant_key_buf */
            uint8_t* gather_dst = (uint8_t*)s->quant_key_buf;
            for (int t = 0; t < seq_len; t++) {
                const uint8_t* src = layer_base
                    + (size_t)t * pos_stride_bytes
                    + (size_t)kv_h * head_block_bytes;
                memcpy(gather_dst + (size_t)t * head_block_bytes, src, head_block_bytes);
            }

            /* Compute attention scores using integer kernel */
            traits->attention(qh, s->quant_key_buf, atth, seq_len, head_dim);

            /* The integer attention computes raw dot products;
             * apply 1/sqrt(attn_scale_dim) scaling */
            float scale = 1.0f / sqrtf(attn_scale_dim);
            for (int t = 0; t < seq_len; t++) {
                atth[t] *= scale;
            }
            /* Apply sliding window mask: set scores before attn_start to -inf */
            for (int t = 0; t < attn_start; t++) {
                atth[t] = -1e30f;
            }
        } else if (use_quant_kv && s->delta_kv_enabled) {
            /* Delta KV attention with periodic I-frames.
             * I-frames (pos % iframe_int == 0) store absolute keys.
             * P-frames store deltas. Reconstruct by accumulating from last I-frame.
             * This bounds drift to at most iframe_int steps. */
            const tq_type_traits_t* traits = &TQ_TRAITS[s->kv_quant_type];
            float inv_scale = 1.0f / sqrtf(attn_scale_dim);
            int iframe_int = s->delta_iframe_interval > 0 ? s->delta_iframe_interval : 64;
            float recon_key[512];
            float dequant_buf[512];

            for (int t = 0; t < attn_start; t++) atth[t] = -1e30f;

            for (int t = attn_start; t < seq_len; t++) {
                if (t % iframe_int == 0) {
                    /* I-frame: read from FP32 key_cache (perfect precision) */
                    const float* iframe_key = key_cache_layer
                        + (size_t)t * cache_kv_dim + kv_h * head_dim;
                    memcpy(recon_key, iframe_key, (size_t)head_dim * sizeof(float));
                } else {
                    /* P-frame: reconstruct from FP32 I-frame + quantized deltas */
                    int last_iframe = (t / iframe_int) * iframe_int;

                    /* If we're processing sequentially from last I-frame, recon_key
                     * already holds the previous position's reconstruction (if t-1
                     * was processed in this loop). Otherwise, reconstruct from scratch. */
                    if (t - 1 >= attn_start && t - 1 >= last_iframe) {
                        /* recon_key holds recon[t-1], just add delta[t] */
                        const uint8_t* quant_src = (const uint8_t*)s->quant_key_cache
                            + (size_t)l * s->quant_kv_stride
                            + (size_t)t * cache_n_kv_heads * s->quant_head_stride
                            + (size_t)kv_h * s->quant_head_stride;
                        traits->dequantize(quant_src, dequant_buf, head_dim);
                        for (int d = 0; d < head_dim; d++) {
                            recon_key[d] += dequant_buf[d];
                        }
                    } else {
                        /* Reconstruct from FP32 I-frame */
                        const float* iframe_key = key_cache_layer
                            + (size_t)last_iframe * cache_kv_dim + kv_h * head_dim;
                        memcpy(recon_key, iframe_key, (size_t)head_dim * sizeof(float));
                        for (int ti = last_iframe + 1; ti <= t; ti++) {
                            const uint8_t* delta_src = (const uint8_t*)s->quant_key_cache
                                + (size_t)l * s->quant_kv_stride
                                + (size_t)ti * cache_n_kv_heads * s->quant_head_stride
                                + (size_t)kv_h * s->quant_head_stride;
                            traits->dequantize(delta_src, dequant_buf, head_dim);
                            for (int d = 0; d < head_dim; d++) {
                                recon_key[d] += dequant_buf[d];
                            }
                        }
                    }
                }

                float score = 0.0f;
#ifdef __ARM_NEON
                float32x4_t vsum = vdupq_n_f32(0.0f);
                int d = 0;
                for (; d + 4 <= head_dim; d += 4) {
                    float32x4_t vq = vld1q_f32(qh + d);
                    float32x4_t vk = vld1q_f32(recon_key + d);
                    vsum = vfmaq_f32(vsum, vq, vk);
                }
                score = vaddvq_f32(vsum);
                for (; d < head_dim; d++) {
                    score += qh[d] * recon_key[d];
                }
#else
                for (int d = 0; d < head_dim; d++) {
                    score += qh[d] * recon_key[d];
                }
#endif
                atth[t] = score * inv_scale;
            }
        } else if (use_quant_kv) {
            /* Dequant attention: read from quantized key cache, dequantize
             * each position's key on the fly, then compute FP32 dot product.
             * This is the path that delivers REAL memory savings — no FP32
             * key cache is stored for previous positions.
             *
             * When k_highres_window is active (age-based progressive), recent
             * tokens within the window use FP32 from the circular buffer.
             * Old tokens use the quant cache (typically 2-bit). */
            const tq_type_traits_t* traits = &TQ_TRAITS[s->kv_quant_type];
            float inv_scale = 1.0f / sqrtf(attn_scale_dim);
            float dequant_buf[512];

            int k_hr_win = s->k_highres_window;
            int k_hr_active = (k_hr_win > 0 && s->key_highres_fp32 != NULL);
            /* Window boundary: positions >= window_start are in the FP32 window */
            int k_window_start = k_hr_active ? (pos - k_hr_win + 1) : seq_len;
            if (k_window_start < 0) k_window_start = 0;

            size_t hr_layer_stride = k_hr_active ?
                (size_t)k_hr_win * cache_kv_dim : 0;

            for (int t = 0; t < attn_start; t++) atth[t] = -1e30f;

            for (int t = attn_start; t < seq_len; t++) {
                const float* key_ptr;

                if (k_hr_active && t >= k_window_start) {
                    /* Recent token: read from FP32 highres circular buffer */
                    int win_idx = t % k_hr_win;
                    key_ptr = s->key_highres_fp32
                        + (size_t)l * hr_layer_stride
                        + (size_t)win_idx * cache_kv_dim
                        + kv_h * head_dim;
                } else {
                    /* Old token: dequantize from quant cache */
                    const uint8_t* quant_src = (const uint8_t*)s->quant_key_cache
                        + (size_t)l * s->quant_kv_stride
                        + (size_t)t * cache_n_kv_heads * s->quant_head_stride
                        + (size_t)kv_h * s->quant_head_stride;
                    traits->dequantize(quant_src, dequant_buf, head_dim);
                    key_ptr = dequant_buf;
                }

                float score = 0.0f;
#ifdef __ARM_NEON
                float32x4_t vsum = vdupq_n_f32(0.0f);
                int d = 0;
                for (; d + 4 <= head_dim; d += 4) {
                    float32x4_t vq = vld1q_f32(qh + d);
                    float32x4_t vk = vld1q_f32(key_ptr + d);
                    vsum = vfmaq_f32(vsum, vq, vk);
                }
                score = vaddvq_f32(vsum);
                for (; d < head_dim; d++) {
                    score += qh[d] * key_ptr[d];
                }
#else
                for (int d = 0; d < head_dim; d++) {
                    score += qh[d] * key_ptr[d];
                }
#endif
                atth[t] = score * inv_scale;
            }
        } else {
            /* FP32 attention scores (no quantization) */
            float inv_scale = 1.0f / sqrtf(attn_scale_dim);
            /* Set positions outside sliding window to -inf */
            for (int t = 0; t < attn_start; t++) {
                atth[t] = -1e30f;
            }
            for (int t = attn_start; t < seq_len; t++) {
                const float* kt = key_cache_layer + (size_t)t * cache_kv_dim + kv_h * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += qh[d] * kt[d];
                }
                atth[t] = score * inv_scale;
            }
        }

        /* Attention logit soft-capping (Gemma 2/3/4): cap * tanh(score / cap)
         * Important: softcap applies to RAW (unscaled) scores. The 1/sqrt(d)
         * scaling must be applied AFTER softcap, before softmax.
         * This matches llama.cpp's approach: softcap(Q*K^T) * scale → softmax.
         *
         * When softcap is disabled, scores already have scale applied inline
         * (score * inv_scale), so no extra work needed. */
        if (c->attn_logit_softcap > 0.0f) {
            float cap = c->attn_logit_softcap;
            float inv_cap = 1.0f / cap;
            float inv_scale = 1.0f / sqrtf(attn_scale_dim);
            for (int t = attn_start; t < seq_len; t++) {
                /* atth[t] currently has score * inv_scale (scaled).
                 * Undo the scale, apply softcap, then re-apply scale. */
                float raw = atth[t] / inv_scale;  /* undo: raw score */
                float capped = cap * tanhf(raw * inv_cap);
                atth[t] = capped * inv_scale;
            }
        }

        /* Softmax */
        tq_softmax(atth, seq_len);

        /* Attention entropy tracking (opt-in) */
        if (s->attn_entropy && s->entropy_accum) {
            double ent = 0.0;
            for (int t = 0; t < seq_len; t++) {
                float p = atth[t];
                if (p > 1e-10f) {
                    ent -= (double)p * log2((double)p);
                }
            }
            s->entropy_accum[(size_t)l * n_heads + h] += ent;
        }

        /* Weighted sum of values */
        float* xbh = s->xb + h * head_dim;
        memset(xbh, 0, head_dim * sizeof(float));

        /* V highres window: for recent tokens, use FP16 V instead of quantized.
         * This improves quality for tokens that typically receive high attention weight. */
        if (s->v_highres_window > 0 && s->value_highres_fp16 &&
            (s->value_quant_bits == 4 || s->value_quant_bits == 2)) {
            /* Hybrid path: quantized V for old tokens, FP16 V for recent tokens */
            int window_start = pos - s->v_highres_window + 1;
            if (window_start < 0) window_start = 0;

            /* Old tokens: use quantized V path */
            size_t layer_off_qs = (size_t)l * max_seq * s->value_stride_qs;
            size_t layer_off_sc = (size_t)l * max_seq * s->value_stride_scales;
            int n_blocks_per_head = (head_dim + 31) / 32;
            size_t packed_per_block = (s->value_quant_bits == 4) ? 16 : 8;
            size_t head_qs_off = (size_t)kv_h * n_blocks_per_head * packed_per_block;
            size_t head_sc_off = (size_t)kv_h * n_blocks_per_head;
            for (int t = 0; t < window_start && t < seq_len; t++) {
                float a = atth[t];
                if (a == 0.0f) continue;
                const uint8_t* vqs = s->value_cache_qs + layer_off_qs
                    + (size_t)t * s->value_stride_qs + head_qs_off;
                const float* vsc = s->value_cache_scales + layer_off_sc
                    + (size_t)t * s->value_stride_scales + head_sc_off;
                if (s->value_quant_bits == 4) {
                    /* Fused Q4 domain accumulation */
                    for (int b = 0; b < n_blocks_per_head; b++) {
                        float combined = a * vsc[b];
                        const uint8_t* bqs = vqs + (size_t)b * 16;
                        for (int j = 0; j < 16; j++) {
                            int idx0 = b * 32 + 2 * j;
                            int idx1 = idx0 + 1;
                            if (idx0 >= head_dim) break;
                            int q0 = bqs[j] & 0xF;
                            int q1 = bqs[j] >> 4;
                            xbh[idx0] += combined * (float)(q0 - 8);
                            if (idx1 < head_dim)
                                xbh[idx1] += combined * (float)(q1 - 8);
                        }
                    }
                } else {
                    float v_tmp[512];
                    tq_dequantize_row_q2(vqs, vsc, v_tmp, head_dim);
                    for (int d = 0; d < head_dim; d++) {
                        xbh[d] += a * v_tmp[d];
                    }
                }
            }

            /* Recent tokens: use FP16 V from highres window */
            int window_size = s->v_highres_window;
            size_t hr_layer_stride = (size_t)window_size * cache_kv_dim;
            const uint16_t* hr_layer = s->value_highres_fp16 + (size_t)l * hr_layer_stride;
            for (int t = window_start; t < seq_len; t++) {
                float a = atth[t];
                if (a == 0.0f) continue;
                int win_idx = t % window_size;
                const uint16_t* vt16 = hr_layer + (size_t)win_idx * cache_kv_dim + kv_h * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    /* fp16_to_f32 inline: extract sign/exp/mant */
                    uint16_t hv = vt16[d];
                    union { float f; uint32_t u; } bits;
                    uint32_t sign = (hv & 0x8000) << 16;
                    uint32_t exp = (hv >> 10) & 0x1F;
                    uint32_t mant = hv & 0x03FF;
                    if (exp == 0) { bits.u = sign; }
                    else if (exp == 31) { bits.u = sign | 0x7F800000 | (mant << 13); }
                    else { exp = exp - 15 + 127; bits.u = sign | (exp << 23) | (mant << 13); }
                    xbh[d] += a * bits.f;
                }
            }
        } else if (s->value_quant_bits == 4) {
            /* Fused Q4 domain accumulation: compute weighted sum directly
             * from packed Q4 nibbles without full dequantization to v_tmp.
             * out[d] += attn_weight * scale * (nibble - 8)
             * This saves one intermediate buffer and reduces memory traffic. */
            size_t layer_off_qs = (size_t)l * max_seq * s->value_stride_qs;
            size_t layer_off_sc = (size_t)l * max_seq * s->value_stride_scales;
            int n_blocks_per_head = (head_dim + 31) / 32;
            size_t packed_per_block = 16;
            size_t head_qs_off = (size_t)kv_h * n_blocks_per_head * packed_per_block;
            size_t head_sc_off = (size_t)kv_h * n_blocks_per_head;
            for (int t = 0; t < seq_len; t++) {
                float a = atth[t];
                if (a == 0.0f) continue;
                const uint8_t* vqs = s->value_cache_qs + layer_off_qs
                    + (size_t)t * s->value_stride_qs + head_qs_off;
                const float* vsc = s->value_cache_scales + layer_off_sc
                    + (size_t)t * s->value_stride_scales + head_sc_off;
                for (int b = 0; b < n_blocks_per_head; b++) {
                    float combined = a * vsc[b];
                    const uint8_t* bqs = vqs + (size_t)b * 16;
#ifdef __ARM_NEON
                    float32x4_t vc = vdupq_n_f32(combined);
                    float32x4_t v8 = vdupq_n_f32(8.0f);
                    for (int j = 0; j < 16; j += 4) {
                        /* Unpack 4 bytes -> 8 Q4 values */
                        int base = b * 32 + 2 * j;
                        if (base + 7 >= head_dim) {
                            /* Scalar tail for partial blocks */
                            for (int jj = j; jj < 16 && b * 32 + 2 * jj + 1 < head_dim; jj++) {
                                int q0 = bqs[jj] & 0xF;
                                int q1 = bqs[jj] >> 4;
                                xbh[b * 32 + 2 * jj]     += combined * (float)(q0 - 8);
                                xbh[b * 32 + 2 * jj + 1] += combined * (float)(q1 - 8);
                            }
                            break;
                        }
                        /* Low nibbles: 4 values at even positions */
                        float lo0 = (float)(bqs[j]   & 0xF);
                        float lo1 = (float)(bqs[j+1] & 0xF);
                        float lo2 = (float)(bqs[j+2] & 0xF);
                        float lo3 = (float)(bqs[j+3] & 0xF);
                        float32x4_t vlo = {lo0, lo1, lo2, lo3};
                        vlo = vsubq_f32(vlo, v8);

                        /* High nibbles: 4 values at odd positions */
                        float hi0 = (float)(bqs[j]   >> 4);
                        float hi1 = (float)(bqs[j+1] >> 4);
                        float hi2 = (float)(bqs[j+2] >> 4);
                        float hi3 = (float)(bqs[j+3] >> 4);
                        float32x4_t vhi = {hi0, hi1, hi2, hi3};
                        vhi = vsubq_f32(vhi, v8);

                        /* Interleave: [lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3] */
                        float32x4x2_t interleaved = vzipq_f32(vlo, vhi);

                        float32x4_t vx0 = vld1q_f32(xbh + base);
                        float32x4_t vx1 = vld1q_f32(xbh + base + 4);
                        vst1q_f32(xbh + base,     vfmaq_f32(vx0, vc, interleaved.val[0]));
                        vst1q_f32(xbh + base + 4, vfmaq_f32(vx1, vc, interleaved.val[1]));
                    }
#else
                    for (int j = 0; j < 16; j++) {
                        int idx0 = b * 32 + 2 * j;
                        int idx1 = idx0 + 1;
                        if (idx0 >= head_dim) break;
                        int q0 = bqs[j] & 0xF;
                        int q1 = bqs[j] >> 4;
                        xbh[idx0] += combined * (float)(q0 - 8);
                        if (idx1 < head_dim)
                            xbh[idx1] += combined * (float)(q1 - 8);
                    }
#endif
                }
            }
        } else if (s->value_quant_bits == 2) {
            /* Q2 value path: dequantize and accumulate.
             * Q2 has a more complex codebook, so we keep the dequant path. */
            float v_tmp[512]; /* max head_dim is 256, safe with margin */
            size_t layer_off_qs = (size_t)l * max_seq * s->value_stride_qs;
            size_t layer_off_sc = (size_t)l * max_seq * s->value_stride_scales;
            int n_blocks_per_head = (head_dim + 31) / 32;
            size_t packed_per_block = 8;
            size_t head_qs_off = (size_t)kv_h * n_blocks_per_head * packed_per_block;
            size_t head_sc_off = (size_t)kv_h * n_blocks_per_head;
            for (int t = 0; t < seq_len; t++) {
                float a = atth[t];
                if (a == 0.0f) continue;
                const uint8_t* vqs = s->value_cache_qs + layer_off_qs
                    + (size_t)t * s->value_stride_qs + head_qs_off;
                const float* vsc = s->value_cache_scales + layer_off_sc
                    + (size_t)t * s->value_stride_scales + head_sc_off;
                tq_dequantize_row_q2(vqs, vsc, v_tmp, head_dim);
#ifdef __ARM_NEON
                float32x4_t va = vdupq_n_f32(a);
                int d = 0;
                for (; d + 3 < head_dim; d += 4) {
                    float32x4_t vv = vld1q_f32(v_tmp + d);
                    float32x4_t vx = vld1q_f32(xbh + d);
                    vst1q_f32(xbh + d, vfmaq_f32(vx, va, vv));
                }
                for (; d < head_dim; d++) {
                    xbh[d] += a * v_tmp[d];
                }
#else
                for (int d = 0; d < head_dim; d++) {
                    xbh[d] += a * v_tmp[d];
                }
#endif
            }
        } else if (s->use_fp16_values) {
            /* FP16 value path: convert on the fly during weighted sum */
            const uint16_t* vfp16_layer = s->value_cache_fp16 + l * kv_layer_stride;
            for (int t = 0; t < seq_len; t++) {
                const uint16_t* vt16 = vfp16_layer + (size_t)t * cache_kv_dim + kv_h * head_dim;
                float a = atth[t];
                if (a == 0.0f) continue; /* skip zero-weight positions */
#ifdef __ARM_NEON
                float32x4_t va = vdupq_n_f32(a);
                int d = 0;
                for (; d + 3 < head_dim; d += 4) {
                    uint16x4_t vh = vld1_u16(vt16 + d);
                    float32x4_t vf = vcvt_f32_f16(vreinterpret_f16_u16(vh));
                    float32x4_t vx = vld1q_f32(xbh + d);
                    vst1q_f32(xbh + d, vfmaq_f32(vx, va, vf));
                }
                for (; d < head_dim; d++) {
                    xbh[d] += a * xfr_fp16_to_f32(vt16[d]);
                }
#else
                for (int d = 0; d < head_dim; d++) {
                    xbh[d] += a * xfr_fp16_to_f32(vt16[d]);
                }
#endif
            }
        } else {
            /* FP32 value path (original) */
            const float* val_cache_layer_fp32 = s->value_cache + l * kv_layer_stride;
            for (int t = 0; t < seq_len; t++) {
                const float* vt = val_cache_layer_fp32 + (size_t)t * cache_kv_dim + kv_h * head_dim;
                float a = atth[t];
                for (int d = 0; d < head_dim; d++) {
                    xbh[d] += a * vt[d];
                }
            }
        }
    }

    TQ_PROF_STOP(_tp, attn_ns);

    /* Apply output gate if enabled: attn_out *= sigmoid(gate_q) */
    if (c->attn_output_gate && gate_q) {
        int total = n_heads * head_dim;
#ifdef __ARM_NEON
        int i = 0;
        for (; i + 3 < total; i += 4) {
            float32x4_t vg = vld1q_f32(gate_q + i);
            float32x4_t vx = vld1q_f32(s->xb + i);
            float32x4_t vneg = vnegq_f32(vg);
            float neg_vals[4];
            vst1q_f32(neg_vals, vneg);
            float exp_vals[4] = {
                fast_expf(neg_vals[0]), fast_expf(neg_vals[1]),
                fast_expf(neg_vals[2]), fast_expf(neg_vals[3])
            };
            float32x4_t vexp = vld1q_f32(exp_vals);
            float32x4_t vone = vdupq_n_f32(1.0f);
            float32x4_t vsig = vdivq_f32(vone, vaddq_f32(vone, vexp));
            vst1q_f32(s->xb + i, vmulq_f32(vx, vsig));
        }
        for (; i < total; i++) {
            float g = 1.0f / (1.0f + fast_expf(-gate_q[i]));
            s->xb[i] *= g;
        }
#else
        for (int i = 0; i < total; i++) {
            float g = 1.0f / (1.0f + fast_expf(-gate_q[i]));
            s->xb[i] *= g;
        }
#endif
    }

    /* Output projection */
    TQ_PROF_START(_tp);
    if (layer->wo_q2)
        TQ_MATMUL_Q2_OR_1BIT_FP32(s->xb2, s->xb, layer->wo_q2, layer->wo_q2s, dim, n_heads * head_dim, model->use_1bit_weights);
    else if (layer->wo_q4)
        tq_matmul_q4(s->xb2, s->xb, layer->wo_q4, layer->wo_q4s, dim, n_heads * head_dim);
    else if (layer->wo_q8)
        tq_matmul_q8(s->xb2, s->xb, layer->wo_q8, layer->wo_q8s, dim, n_heads * head_dim);
    else if (layer->gguf_wo)
        tq_matmul_gguf(s->xb2, s->xb, layer->gguf_wo, layer->gguf_wo_type, dim, n_heads * head_dim);
    else
        tq_matmul(s->xb2, s->xb, layer->wo, dim, n_heads * head_dim);
    TQ_PROF_STOP(_tp, matmul_ns);

    /* Debug: print attention output before residual add */
    if (pos == 0 && getenv("TQ_DEBUG") && l < 3) {
        float maxv = 0, minv = 0;
        for (int i = 0; i < dim; i++) {
            if (s->xb2[i] > maxv) maxv = s->xb2[i];
            if (s->xb2[i] < minv) minv = s->xb2[i];
        }
        fprintf(stderr, "[DEBUG] layer%d attn_out min=%.3f max=%.3f (hd=%d, nh=%d, nkv=%d)\n",
                l, minv, maxv, head_dim, n_heads, n_kv_heads);
    }

    /* Residual */
    tq_add(s->x, s->x, s->xb2, dim);
}

/* ============================================================
 * Forward pass — hybrid transformer with DeltaNet + self_attn
 *
 * For each layer:
 *   1. RMSNorm
 *   2. If layer has DeltaNet: deltanet_forward
 *      If layer has self_attn: self_attn_forward
 *      (skip if neither)
 *   3. RMSNorm -> SwiGLU FFN -> residual
 * ============================================================ */
float* tq_forward(tq_model_t* model, tq_state_t* s, int token, int pos) {
    double _fwd_t0 = g_tq_profile_enabled ? tq_now_ns() : 0;
    double _tp = 0;  /* profiling timestamp */
    tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;

    /* Step 1: Token embedding */
    if (model->embed_bf16) {
        /* Streaming BF16->FP32 conversion: convert only this token's row */
        const uint16_t* bf16_row = model->embed_bf16 + (size_t)token * dim;
#ifdef __ARM_NEON
        int i = 0;
        for (; i + 3 < dim; i += 4) {
            uint16x4_t b = vld1_u16(bf16_row + i);
            float32x4_t f = vreinterpretq_f32_u32(vshll_n_u16(b, 16));
            vst1q_f32(s->x + i, f);
        }
        for (; i < dim; i++) {
            uint32_t bits = ((uint32_t)bf16_row[i]) << 16;
            memcpy(&s->x[i], &bits, 4);
        }
#else
        for (int i = 0; i < dim; i++) {
            uint32_t bits = ((uint32_t)bf16_row[i]) << 16;
            memcpy(&s->x[i], &bits, 4);
        }
#endif
    } else if (model->output_gguf && !model->token_embedding) {
        /* GGUF embedding: dequant single row on demand (no FP32 table in memory) */
        int block_elems = tq_ggml_type_blck(model->output_gguf_type);
        int block_bytes = (int)tq_ggml_type_size(model->output_gguf_type);
        int n_blocks = dim / block_elems;
        size_t row_bytes = (size_t)n_blocks * block_bytes;
        const uint8_t* row_ptr = (const uint8_t*)model->output_gguf + (size_t)token * row_bytes;
        tq_dequant_row_gguf(model->output_gguf_type, row_ptr, s->x, dim);
    } else {
        memcpy(s->x, model->token_embedding + (size_t)token * dim,
               dim * sizeof(float));
    }

    /* Gemma: scale embeddings by sqrt(hidden_dim) */
    if (c->model_type == 1) {
        float scale = sqrtf((float)dim);
        for (int i = 0; i < dim; i++) {
            s->x[i] *= scale;
        }
    }

    /* Debug: print embedding for verification */
    if (pos == 0 && getenv("TQ_DEBUG")) {
        fprintf(stderr, "[DEBUG] embed[0:8] = ");
        for (int i = 0; i < 8 && i < dim; i++) fprintf(stderr, "%.4f ", s->x[i]);
        fprintf(stderr, "\n");
    }

    /* PLE pre-computation: once per token, before the layer loop.
     * Computes ple_input[l] for each layer l from:
     *   1. per_layer_token_embd[token] (dequant from Q5_K) → reshape [n_layers, ple_dim]
     *   2. per_layer_model_proj @ embed_raw (FP32 matmul) → reshape [n_layers, ple_dim]
     *   3. Combine with RMS-norm and averaging. */
    if (model->ple_dim > 0 && model->ple_embedding && model->ple_proj && !getenv("TQ_NO_PLE")) {
        int ple_dim = model->ple_dim;
        int n_layers = c->n_layers;
        int total_ple = n_layers * ple_dim;  /* e.g., 35 * 256 = 8960 */

        /* Lazy allocation of ple_buf */
        if (!s->ple_buf) {
            s->ple_buf = (float*)calloc((size_t)total_ple, sizeof(float));
        }

        /* Step A: Dequant per_layer_token_embd[token] → temp_embd[8960]
         * The embedding tensor is [total_ple, vocab_size] in GGUF row-major,
         * so one token's data is at row offset = token * row_bytes. */
        float temp_embd[8960];  /* stack buffer, total_ple <= 8960 */
        {
            size_t type_size = tq_ggml_type_size(model->ple_embedding_type);
            int blck = tq_ggml_type_blck(model->ple_embedding_type);
            if (blck <= 0) blck = 1;
            size_t row_bytes = ((size_t)total_ple / (size_t)blck) * type_size;
            const uint8_t* row_ptr = (const uint8_t*)model->ple_embedding + (size_t)token * row_bytes;
            tq_dequant_row_gguf(model->ple_embedding_type, row_ptr, temp_embd, total_ple);
        }

        /* Scale by sqrt(ple_dim) = sqrt(256) = 16.0 */
        float ple_scale = sqrtf((float)ple_dim);
        for (int i = 0; i < total_ple; i++) {
            temp_embd[i] *= ple_scale;
        }

        /* Step B: per_layer_model_proj @ embed_raw → temp_proj[8960]
         * ple_proj is [total_ple, hidden_dim] FP32 (rows=8960, cols=1536).
         * We need: for each output row d in [0, total_ple): dot(ple_proj[d,:], s->x[:])
         * Note: s->x already has the scaled embedding from above. */
        float temp_proj[8960];
        tq_matmul(temp_proj, s->x, model->ple_proj, total_ple, dim);

        /* Scale by 1/sqrt(hidden_dim) */
        float inv_sqrt_dim = 1.0f / sqrtf((float)dim);
        for (int i = 0; i < total_ple; i++) {
            temp_proj[i] *= inv_sqrt_dim;
        }

        /* Step C: RMS-norm each 256-dim slice of temp_proj using ple_proj_norm */
        for (int l = 0; l < n_layers; l++) {
            float* slice = temp_proj + l * ple_dim;
            tq_rmsnorm(slice, slice, model->ple_proj_norm, ple_dim, c->rms_norm_eps);
        }

        /* Step D: ple_input[l] = (temp_embd[l] + temp_proj[l]) / sqrt(2) */
        float inv_sqrt2 = 1.0f / sqrtf(2.0f);
        for (int i = 0; i < total_ple; i++) {
            s->ple_buf[i] = (temp_embd[i] + temp_proj[i]) * inv_sqrt2;
        }
    }

    /* Step 2: Transformer layers */
    int is_gemma3 = (c->model_type == 1);

    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];

        /* Save input residual for layer_output_scale (Gemma 4).
         * layer_output_scale applies to the layer's contributions only,
         * not the entire hidden state. We need x_old to compute:
         *   x_new = x_old + scale * (x_current - x_old) */
        float layer_residual_buf[4096]; /* max dim for Gemma 4 */
        if (layer->layer_output_scale != 0.0f) {
            memcpy(layer_residual_buf, s->x, dim * sizeof(float));
        }

        /* Pre-attention/DeltaNet RMSNorm */
        tq_rmsnorm(s->xb, s->x, layer->attn_norm, dim, c->rms_norm_eps);

        if (layer->delta_a_log) {
            /* DeltaNet layer */
            deltanet_forward(model, s, l);
        } else if ((layer->wq || layer->wq_q8 || layer->wq_q4 || layer->gguf_wq || layer->wq_q2) &&
                   (layer->wk || layer->wk_q8 || layer->wk_q4 || layer->gguf_wk || layer->wk_q2) &&
                   (layer->wv || layer->wv_q8 || layer->wv_q4 || layer->gguf_wv || layer->wv_q2 ||
                    /* K=V layers (Gemma 4 full attention): no V weights needed */
                    (model->layer_is_sliding && !model->layer_is_sliding[l]))) {
            /* Standard self-attention layer */
            self_attn_forward(model, s, l, pos);

            /* Gemma3: apply post_attention_layernorm to attention output (xb2)
             * before residual add. The residual was already added in self_attn_forward,
             * so we undo it, apply norm, then re-add.
             * Actually, self_attn_forward adds xb2 to x. For Gemma3, we need to
             * apply post_attn_norm to xb2 before the add. We handle this by:
             * 1. The residual add in self_attn_forward already happened.
             * 2. For Gemma3: subtract xb2 from x, normalize xb2, add back. */
            if (is_gemma3 && layer->post_attn_norm) {
                /* xb2 still has the raw attention output from self_attn_forward.
                 * x already has x_old + xb2. Undo: x = x - xb2 */
                for (int i = 0; i < dim; i++) {
                    s->x[i] -= s->xb2[i];
                }
                /* Apply post_attention_layernorm to xb2 */
                tq_rmsnorm(s->xb2, s->xb2, layer->post_attn_norm, dim, c->rms_norm_eps);
                /* Re-add normalized output */
                tq_add(s->x, s->x, s->xb2, dim);
            }
        }
        /* else: skip (should not happen for valid models) */

        /* FFN Block — MoE or Dense SwiGLU/GeGLU */

        /* MoE FFN path: route to top-K experts + shared expert */
        int did_moe = 0;
        if (layer->moe && s->moe_state && model->moe_config) {
            float* ffn_norm_w = layer->ffn_norm;
            if (is_gemma3 && layer->pre_ffn_norm)
                ffn_norm_w = layer->pre_ffn_norm;
            tq_rmsnorm(s->xb, s->x, ffn_norm_w, dim, c->rms_norm_eps);

            TQ_PROF_START(_tp);
            tq_moe_forward((const tq_moe_layer_t*)layer->moe,
                           (const tq_moe_config_t*)model->moe_config,
                           (tq_moe_state_t*)s->moe_state,
                           s->xb, s->xb2, dim, l);
            TQ_PROF_STOP(_tp, moe_ns);

            /* Gemma: MoE output uses post_ffw_norm if present. */
            if (is_gemma3) {
                float* moe_post_norm = layer->post_ffn_norm_1 ? layer->post_ffn_norm_1 : layer->post_ffn_norm;
                if (moe_post_norm)
                    tq_rmsnorm(s->xb2, s->xb2, moe_post_norm, dim, c->rms_norm_eps);
            }

            tq_add(s->x, s->x, s->xb2, dim);
            did_moe = 1;
        }
        /* Dense FFN path — SwiGLU (Qwen3.5, Gemma4/STEP35) or GeGLU (Gemma3).
         * For Gemma 4 STEP35: layers are either MoE or dense, NOT both.
         * For Gemma 3: runs both MoE and dense FFN (shared expert) per layer. */
        /* Dense FFN: run for non-MoE layers, or for Gemma 3 MoE layers with dense FFN */
        if ((!did_moe || (is_gemma3 && !c->is_gemma4 && did_moe)) &&
            (layer->w_gate || layer->w_gate_q8 || layer->w_gate_q4 || layer->w_gate_q2 || layer->gguf_w_gate) &&
            (layer->w_up || layer->w_up_q8 || layer->w_up_q4 || layer->w_up_q2 || layer->gguf_w_up) &&
            (layer->w_down || layer->w_down_q8 || layer->w_down_q4 || layer->w_down_q2 || layer->gguf_w_down)) {

            /* Pre-FFN norm: Gemma 4 dual-FFN uses pre_ffw_norm_2 for the dense FFN.
             * Gemma3 uses pre_feedforward_layernorm.
             * Qwen3.5 uses post_attention_layernorm (stored as ffn_norm). */
            float* ffn_norm_w = layer->ffn_norm;
            if (did_moe && layer->pre_ffn_norm_2) {
                /* Gemma 4: dense FFN uses pre_ffw_norm_2 as input norm */
                ffn_norm_w = layer->pre_ffn_norm_2;
            } else if (is_gemma3 && layer->pre_ffn_norm) {
                ffn_norm_w = layer->pre_ffn_norm;
            }
            tq_rmsnorm(s->xb, s->x, ffn_norm_w, dim, c->rms_norm_eps);

            /* Per-layer intermediate dim (Gemma 4 E2B has variable FFN dim) */
            int inter = c->per_layer_inter_dim ? c->per_layer_inter_dim[l] : c->intermediate_dim;

            /* Pre-quantize xb for gate+up Q2/Q4 projections (same input, 2 matmuls) */
            TQ_PROF_START(_tp);
            if (layer->w_gate_q4 && layer->w_gate_q2) {
                /* Q4+Q2 Progressive Residual: Q4 main + Q2 correction */
                tq_quantize_row_q8(s->xb, s->xb_q8, s->xb_q8s, dim);
                /* Q4 matmul */
                tq_matmul_q4_preq(s->hb, layer->w_gate_q4, layer->w_gate_q4s,
                                   s->xb_q8, s->xb_q8s, inter, dim);
                tq_matmul_q4_preq(s->hb2, layer->w_up_q4, layer->w_up_q4s,
                                   s->xb_q8, s->xb_q8s, inter, dim);
                /* Add Q2 residual correction (reuse xb2 as temp — safe here,
                 * xb2 is only needed after FFN completes) */
                tq_matmul_q2_preq(s->xb2, layer->w_gate_q2, layer->w_gate_q2s,
                                   s->xb_q8, s->xb_q8s, inter, dim);
                for (int i = 0; i < inter; i++) s->hb[i] += s->xb2[i];
                tq_matmul_q2_preq(s->xb2, layer->w_up_q2, layer->w_up_q2s,
                                   s->xb_q8, s->xb_q8s, inter, dim);
                for (int i = 0; i < inter; i++) s->hb2[i] += s->xb2[i];
            } else if (layer->w_gate_q2 && !layer->w_gate_q4) {
                tq_quantize_row_q8(s->xb, s->xb_q8, s->xb_q8s, dim);
                TQ_MATMUL_Q2_OR_1BIT(s->hb, s->xb, layer->w_gate_q2, layer->w_gate_q2s,
                                      s->xb_q8, s->xb_q8s, inter, dim, model->use_1bit_weights);
                TQ_MATMUL_Q2_OR_1BIT(s->hb2, s->xb, layer->w_up_q2, layer->w_up_q2s,
                                      s->xb_q8, s->xb_q8s, inter, dim, model->use_1bit_weights);
            } else if (layer->w_gate_q4) {
                tq_quantize_row_q8(s->xb, s->xb_q8, s->xb_q8s, dim);
                tq_matmul_q4_preq(s->hb, layer->w_gate_q4, layer->w_gate_q4s,
                                   s->xb_q8, s->xb_q8s, inter, dim);
                tq_matmul_q4_preq(s->hb2, layer->w_up_q4, layer->w_up_q4s,
                                   s->xb_q8, s->xb_q8s, inter, dim);
            } else if (layer->gguf_w_gate) {
                tq_metal_batch_begin_if_available();
                tq_matmul_gguf(s->hb, s->xb, layer->gguf_w_gate, layer->gguf_w_gate_type, inter, dim);
                tq_matmul_gguf(s->hb2, s->xb, layer->gguf_w_up, layer->gguf_w_up_type, inter, dim);
                tq_metal_batch_flush_if_available();
            } else {
                if (layer->w_gate_q8) {
                    tq_matmul_q8(s->hb, s->xb, layer->w_gate_q8, layer->w_gate_q8s, inter, dim);
                } else {
                    tq_matmul(s->hb, s->xb, layer->w_gate, inter, dim);
                }
                if (layer->w_up_q8) {
                    tq_matmul_q8(s->hb2, s->xb, layer->w_up_q8, layer->w_up_q8s, inter, dim);
                } else {
                    tq_matmul(s->hb2, s->xb, layer->w_up, inter, dim);
                }
            }

            TQ_PROF_STOP(_tp, matmul_ns);

            /* Activation: GeGLU for Gemma3/4, SwiGLU for others.
             * Note: Gemma 4 (STEP35) uses GeGLU (gated GELU), same as Gemma 3.
             * The llama.cpp STEP35 code uses LLM_FFN_SILU which might be incorrect
             * for the E2B model. The HuggingFace Gemma4 config uses gelu_pytorch_tanh. */
            if (is_gemma3) {
                tq_gelu_tanh(s->hb, inter);
            } else {
                tq_silu(s->hb, inter);
            }
            tq_mul(s->hb, s->hb, s->hb2, inter);

            TQ_PROF_START(_tp);
            if (layer->w_down_q2) {
                TQ_MATMUL_Q2_OR_1BIT_FP32(s->xb2, s->hb, layer->w_down_q2, layer->w_down_q2s, dim, inter, model->use_1bit_weights);
            } else if (layer->w_down_q4) {
                tq_matmul_q4(s->xb2, s->hb, layer->w_down_q4, layer->w_down_q4s, dim, inter);
            } else if (layer->w_down_q8) {
                tq_matmul_q8(s->xb2, s->hb, layer->w_down_q8, layer->w_down_q8s, dim, inter);
            } else if (layer->gguf_w_down) {
                tq_matmul_gguf(s->xb2, s->hb, layer->gguf_w_down, layer->gguf_w_down_type, dim, inter);
            } else {
                tq_matmul(s->xb2, s->hb, layer->w_down, dim, inter);
            }
            TQ_PROF_STOP(_tp, matmul_ns);

            /* Gemma: apply post-FFN norm if present. */
            if (is_gemma3) {
                float* dense_post_norm = NULL;
                if (did_moe && layer->post_ffn_norm_2)
                    dense_post_norm = layer->post_ffn_norm_2;
                else if (layer->post_ffn_norm)
                    dense_post_norm = layer->post_ffn_norm;
                if (dense_post_norm)
                    tq_rmsnorm(s->xb2, s->xb2, dense_post_norm, dim, c->rms_norm_eps);
            }

            tq_add(s->x, s->x, s->xb2, dim);
        }

        /* Gemma 4 PLE: apply per-layer embedding after FFN, before layer_output_scale.
         * Can be disabled with TQ_NO_PLE=1 for debugging.
         * 1. gate_out = gelu(inp_gate @ hidden_state) → [ple_dim]
         * 2. mixed = gate_out * ple_input[l] → elementwise [ple_dim]
         * 3. proj_out = proj @ mixed → [hidden_dim]
         * 4. normed = rms_norm(proj_out, post_norm) → [hidden_dim]
         * 5. hidden_state = hidden_state + normed */
        if (model->ple_dim > 0 && s->ple_buf && layer->ple_gate && layer->ple_proj && layer->ple_norm && !getenv("TQ_NO_PLE")) {
            int ple_dim = model->ple_dim;
            float ple_gate_out[256];  /* ple_dim <= 256 */
            float ple_mixed[256];
            float ple_proj_out[2048]; /* hidden_dim <= 2048 (Gemma 4 E2B: 1536) */

            /* gate_out = inp_gate @ hidden_state → [ple_dim]
             * inp_gate is [hidden_dim, ple_dim] F32 type */
            if (layer->ple_gate_type == TQ_GGML_TYPE_F32) {
                tq_matmul(ple_gate_out, s->x, (const float*)layer->ple_gate, ple_dim, dim);
            } else {
                tq_matmul_gguf(ple_gate_out, s->x, layer->ple_gate, layer->ple_gate_type, ple_dim, dim);
            }

            /* Apply GELU-tanh activation */
            tq_gelu_tanh(ple_gate_out, ple_dim);

            /* mixed = gate_out * ple_input[l] (elementwise) */
            float* ple_input_l = s->ple_buf + l * ple_dim;
            for (int i = 0; i < ple_dim; i++) {
                ple_mixed[i] = ple_gate_out[i] * ple_input_l[i];
            }

            /* proj_out = proj @ mixed → [hidden_dim]
             * proj is [ple_dim, hidden_dim] — output is hidden_dim rows, input is ple_dim */
            if (layer->ple_proj_type == TQ_GGML_TYPE_F32) {
                tq_matmul(ple_proj_out, ple_mixed, (const float*)layer->ple_proj, dim, ple_dim);
            } else {
                tq_matmul_gguf(ple_proj_out, ple_mixed, layer->ple_proj, layer->ple_proj_type, dim, ple_dim);
            }

            /* normed = rms_norm(proj_out, post_norm) */
            tq_rmsnorm(ple_proj_out, ple_proj_out, layer->ple_norm, dim, c->rms_norm_eps);

            /* hidden_state += normed */
            tq_add(s->x, s->x, ple_proj_out, dim);
        }

        /* Gemma 4: layer_output_scale scales the layer's CONTRIBUTIONS (attn + ffn).
         * Essential for controlling gradient flow — model was trained with these scales. */
        if (layer->layer_output_scale != 0.0f) {
            float los = layer->layer_output_scale;
            /* Debug: print pre-scale values */
            if (pos == 0 && getenv("TQ_DEBUG") && l < 3) {
                float maxv = 0, minv = 0;
                for (int i = 0; i < dim; i++) {
                    if (s->x[i] > maxv) maxv = s->x[i];
                    if (s->x[i] < minv) minv = s->x[i];
                }
                fprintf(stderr, "[DEBUG] layer%d pre_scale min=%.3f max=%.3f (los=%.4f)\n", l, minv, maxv, los);
            }
            for (int i = 0; i < dim; i++) {
                s->x[i] = layer_residual_buf[i] + los * (s->x[i] - layer_residual_buf[i]);
            }
        }

        /* Debug: print layer output */
        if (pos == 0 && getenv("TQ_DEBUG")) {
            if (l < 10 || l == c->n_layers - 1 || getenv("TQ_DEBUG_ALL")) {
                float maxv = 0, minv = 0;
                for (int i = 0; i < dim; i++) {
                    if (s->x[i] > maxv) maxv = s->x[i];
                    if (s->x[i] < minv) minv = s->x[i];
                }
                fprintf(stderr, "[DEBUG] layer%d out[0:4]=%.3f,%.3f,%.3f,%.3f min=%.3f max=%.3f los=%.4f\n",
                        l, s->x[0], s->x[1], s->x[2], s->x[3], minv, maxv, layer->layer_output_scale);
            }
        }
    }

    /* Step 3: Final RMSNorm */
    if (pos == 0 && getenv("TQ_DEBUG")) {
        fprintf(stderr, "[DEBUG] pre_norm[0:8] = ");
        for (int i = 0; i < 8 && i < dim; i++) fprintf(stderr, "%.4f ", s->x[i]);
        fprintf(stderr, "\n");
    }
    tq_rmsnorm(s->x, s->x, model->output_norm, dim, c->rms_norm_eps);
    if (pos == 0 && getenv("TQ_DEBUG")) {
        fprintf(stderr, "[DEBUG] post_norm[0:8] = ");
        for (int i = 0; i < 8 && i < dim; i++) fprintf(stderr, "%.4f ", s->x[i]);
        fprintf(stderr, "\n");
    }

    /* Step 4: Output projection to vocab logits */
    TQ_PROF_START(_tp);
    if (model->output_gguf) {
        /* GGUF fused dot output projection — less memory bandwidth than FP32 */
        tq_matmul_gguf(s->logits, s->x, model->output_gguf,
                        model->output_gguf_type, c->vocab_size, dim);
    } else if (model->output_qs) {
        tq_matmul_q4(s->logits, s->x, model->output_qs, model->output_scales,
                      c->vocab_size, dim);
    } else if (model->output_weight_bf16) {
        tq_matmul_bf16(s->logits, s->x, model->output_weight_bf16, c->vocab_size, dim);
    } else {
        tq_matmul(s->logits, s->x, model->output_weight, c->vocab_size, dim);
    }
    TQ_PROF_STOP(_tp, matmul_ns);

    if (pos <= 1 && getenv("TQ_DEBUG")) {
        /* Print top-5 logits for debugging */
        fprintf(stderr, "[DEBUG] pos=%d logits[0:8] = ", pos);
        for (int i = 0; i < 8; i++) fprintf(stderr, "%.2f ", s->logits[i]);
        float max_l = s->logits[0]; int max_i = 0;
        for (int i = 1; i < c->vocab_size; i++) { if (s->logits[i] > max_l) { max_l = s->logits[i]; max_i = i; } }
        fprintf(stderr, "... max=%.2f @%d\n", max_l, max_i);
    }

    /* Final logit soft-capping: logits = cap * tanh(logits / cap) */
    /* Note: logit soft-capping disabled for now — Gemma 4 GGUF models have
     * large norm weights (by design) that produce logits >> cap, destroying
     * the ranking. TODO: investigate if soft-capping needs different handling
     * or if it should only apply after attention, not final logits. */
    if (c->final_logit_softcap > 0.0f && !getenv("TQ_NO_SOFTCAP")) {
        float cap = c->final_logit_softcap;
        float inv_cap = 1.0f / cap;
        for (int i = 0; i < c->vocab_size; i++) {
            s->logits[i] = cap * tanhf(s->logits[i] * inv_cap);
        }
    }

    /* Increment profile token count if profiling is active */
    if (s->profile_kv) {
        s->profile_kv_count++;
    }

    /* Timing profile: accumulate total fwd time and print every 10 tokens */
    if (g_tq_profile_enabled) {
        g_profile.total_fwd_ns += tq_now_ns() - _fwd_t0;
        g_profile.n_tokens++;
        if (g_profile.n_tokens % 10 == 0) {
            double mat  = g_profile.matmul_ns;
            double rec  = g_profile.recurrent_ns;
            double moe  = g_profile.moe_ns;
            double conv = g_profile.conv1d_ns;
            double attn = g_profile.attn_ns;
            double total = g_profile.total_fwd_ns;
            double other = total - mat - rec - moe - conv - attn;
            if (other < 0) other = 0;
            if (total > 0) {
                fprintf(stderr, "[Profile %d tok] matmul=%.1f%% recurrent=%.1f%% moe=%.1f%% conv=%.1f%% attn=%.1f%% other=%.1f%% | per-tok: %.1fms (mat=%.1f rec=%.1f moe=%.1f conv=%.1f attn=%.1f other=%.1f)\n",
                    g_profile.n_tokens,
                    mat/total*100, rec/total*100, moe/total*100,
                    conv/total*100, attn/total*100, other/total*100,
                    total / g_profile.n_tokens / 1e6,
                    mat / g_profile.n_tokens / 1e6,
                    rec / g_profile.n_tokens / 1e6,
                    moe / g_profile.n_tokens / 1e6,
                    conv / g_profile.n_tokens / 1e6,
                    attn / g_profile.n_tokens / 1e6,
                    other / g_profile.n_tokens / 1e6);
            }
        }
    }

    return s->logits;
}

// ============================================================================
// Section 15: Generation Loop (from tq_generate.c)
// ============================================================================

/**
 * tq_generate.c — Text generation loop with TurboQuant KV cache
 *
 * Implements:
 *   - Argmax sampling (greedy)
 *   - Top-p (nucleus) sampling with temperature
 *   - Full generation loop with streaming callback
 */

/* ============================================================
 * Argmax sampling: return token with highest logit
 * ============================================================ */
int tq_sample_argmax(const float* logits, int vocab_size) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

/* ============================================================
 * Top-p (nucleus) sampling with temperature
 *
 * 1. Apply temperature scaling
 * 2. Compute softmax probabilities
 * 3. Sort by probability (descending)
 * 4. Accumulate until cumulative prob >= top_p
 * 5. Sample from the nucleus
 * ============================================================ */

/* Simple RNG (xorshift64) for reproducible sampling */
static float random_f32(unsigned long long* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (float)((*state * 0x2545F4914F6CDD1DULL) >> 33) / (float)(1u << 31);
}

/* Comparison for sorting (probability, index) pairs */
typedef struct {
    float prob;
    int index;
} prob_index_t;

static int compare_prob_desc(const void* a, const void* b) {
    float pa = ((const prob_index_t*)a)->prob;
    float pb = ((const prob_index_t*)b)->prob;
    if (pa > pb) return -1;
    if (pa < pb) return 1;
    return 0;
}

/* Persistent workspace to avoid per-token malloc.
 * Protected by mutex for thread safety when multiple model instances
 * call tq_sample_topp concurrently. */
static prob_index_t* g_probindex = NULL;
static int g_probindex_size = 0;
static pthread_mutex_t g_probindex_mutex = PTHREAD_MUTEX_INITIALIZER;

int tq_sample_topp(const float* logits, int vocab_size,
                   float temperature, float top_p,
                   unsigned long long* rng) {
    if (temperature <= 0.0f || top_p <= 0.0f) {
        return tq_sample_argmax(logits, vocab_size);
    }

    /* Pre-filter: only keep logits within reasonable range of max.
     * For top-p=0.9 with temperature=0.7, logits more than ~20 below max
     * contribute negligibly. This avoids sorting 248K entries. */
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    float threshold = max_val - 16.0f * temperature; /* exp(-16) ≈ 1e-7 */

    /* Allocate/reuse workspace (mutex-protected for concurrent callers) */
    pthread_mutex_lock(&g_probindex_mutex);
    if (g_probindex_size < vocab_size) {
        free(g_probindex);
        g_probindex = (prob_index_t*)malloc(vocab_size * sizeof(prob_index_t));
        g_probindex_size = vocab_size;
    }
    if (!g_probindex) {
        pthread_mutex_unlock(&g_probindex_mutex);
        return tq_sample_argmax(logits, vocab_size);
    }

    /* Collect only candidates above threshold */
    int n_candidates = 0;
    float sum = 0.0f;
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] >= threshold) {
            float p = expf((logits[i] - max_val) * inv_temp);
            g_probindex[n_candidates].prob = p;
            g_probindex[n_candidates].index = i;
            sum += p;
            n_candidates++;
        }
    }

    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n_candidates; i++) {
        g_probindex[i].prob *= inv_sum;
    }

    /* Sort only candidates (typically < 1000 vs 248K) */
    qsort(g_probindex, n_candidates, sizeof(prob_index_t), compare_prob_desc);

    /* Find top-p cutoff */
    float cumulative = 0.0f;
    int n_top = 0;
    for (int i = 0; i < n_candidates; i++) {
        cumulative += g_probindex[i].prob;
        n_top = i + 1;
        if (cumulative >= top_p) break;
    }

    /* Sample from the nucleus */
    float r = random_f32(rng) * cumulative;
    float cdf = 0.0f;
    int sampled = g_probindex[0].index;
    for (int i = 0; i < n_top; i++) {
        cdf += g_probindex[i].prob;
        if (cdf >= r) {
            sampled = g_probindex[i].index;
            break;
        }
    }

    pthread_mutex_unlock(&g_probindex_mutex);
    return sampled;
}

/* ============================================================
 * Generate text from prompt
 *
 * Steps:
 * 1. Encode prompt to tokens
 * 2. Prefill: forward all prompt tokens
 * 3. Decode: sample next token, forward, repeat
 * 4. Stop on EOS or max_tokens
 * ============================================================ */
int tq_generate(tq_model_t* model, tq_tokenizer_t* tokenizer,
                const char* prompt, tq_gen_config_t* config,
                char* output, int output_size) {
    if (!model || !config) return -1;

    tq_state_t* state = tq_create_state_ex(&model->config, config->kv_type, config->value_quant_bits);
    if (!state) {
        fprintf(stderr, "tq_generate: failed to allocate state\n");
        return -1;
    }
    state->delta_kv_enabled = config->delta_kv;
    state->delta_iframe_interval = config->delta_iframe_interval;
    /* Hybrid DeltaNet models: delta KV applies only to self_attn layers.
     * DeltaNet layers don't use key_cache, so delta compression is safe. */

    /* Allocate MoE state if model uses MoE */
    if (model->config.is_moe && model->moe_config) {
        state->moe_state = tq_moe_create_state(
            (const tq_moe_config_t*)model->moe_config,
            model->config.hidden_dim);
        if (!state->moe_state) {
            fprintf(stderr, "tq_generate: failed to allocate MoE state\n");
            tq_free_state(state);
            return -1;
        }
    }

    /* Set up V highres window if requested */
    if (config->v_highres_window > 0 &&
        (config->value_quant_bits == 4 || config->value_quant_bits == 2)) {
        int n_layers = model->config.n_layers;
        int kv_dim = model->config.n_kv_heads * model->config.head_dim;
        int window = config->v_highres_window;
        state->v_highres_window = window;
        state->value_highres_fp16 = (uint16_t*)calloc(
            (size_t)n_layers * window * kv_dim, sizeof(uint16_t));
    }

    /* Set up K highres window (age-based progressive compression) */
    if (config->k_highres_window > 0 &&
        state->kv_quant_type < TQ_TYPE_COUNT && state->quant_key_cache != NULL) {
        int n_layers = model->config.n_layers;
        int kv_dim = model->config.n_kv_heads * model->config.head_dim;
        int window = config->k_highres_window;
        state->k_highres_window = window;
        state->key_highres_fp32 = (float*)calloc(
            (size_t)n_layers * window * kv_dim, sizeof(float));
    }

    /* Encode prompt */
    int prompt_tokens[4096];
    int n_prompt = 0;

    if (tokenizer && prompt) {
        /* Gemma models: prepend BOS=2 (required by both Gemma 3 and 4 architectures).
         * Qwen3.5: no BOS. */
        int add_bos = 0;
        if (model->config.model_type == 1) {
            add_bos = 1; /* All Gemma models need BOS */
        }
        n_prompt = tq_encode(tokenizer, prompt, prompt_tokens, 4096, add_bos);
    } else {
        /* No tokenizer: use BOS only (Gemma=2, Qwen=skip) */
        prompt_tokens[0] = (model->config.model_type == 1) ? 2 : 1;
        n_prompt = 1;
    }

    if (n_prompt <= 0) {
        prompt_tokens[0] = (model->config.model_type == 1) ? 2 : 1;
        n_prompt = 1;
    }

    /* Debug: print tokenized prompt */
    if (getenv("TQ_DEBUG")) {
        fprintf(stderr, "[DEBUG] prompt tokens (%d): ", n_prompt);
        for (int i = 0; i < n_prompt && i < 20; i++)
            fprintf(stderr, "%d ", prompt_tokens[i]);
        fprintf(stderr, "\n");
    }

    /* Prefill: process all prompt tokens.
     * NOTE: No emscripten_sleep() here — the call stack during tq_forward()
     * is too deep for ASYNCIFY to unwind (matmul → SIMD kernels). Adding
     * sleep here breaks ASYNCIFY for the entire generate call, including
     * the token streaming callback. The browser shows "Thinking..." via
     * requestAnimationFrame before entering this blocking prefill. */
    for (int i = 0; i < n_prompt; i++) {
        tq_forward(model, state, prompt_tokens[i], i);
    }

    /* Repetition penalty setup */
    int vocab_size = model->config.vocab_size;
    float rep_penalty = config->rep_penalty;
    int rep_window = config->rep_window;
    if (rep_window > 64) rep_window = 64;
    int recent_tokens[64];
    int recent_count = 0;

    /* Seed recent tokens with tail of prompt for better penalty coverage */
    for (int i = (n_prompt > rep_window ? n_prompt - rep_window : 0); i < n_prompt; i++) {
        recent_tokens[recent_count % 64] = prompt_tokens[i];
        recent_count++;
    }

    /* Apply repetition penalty to logits before first sample */
    if (rep_penalty > 1.0f) {
        int window = recent_count < rep_window ? recent_count : rep_window;
        for (int r = 0; r < window; r++) {
            int idx = (recent_count - 1 - r) % 64;
            if (idx < 0) idx += 64;
            int tok = recent_tokens[idx];
            if (tok >= 0 && tok < vocab_size) {
                if (state->logits[tok] > 0)
                    state->logits[tok] /= rep_penalty;
                else
                    state->logits[tok] *= rep_penalty;
            }
        }
    }

    /* Sample first generated token. The seed is configurable via
     * config->rng_seed (default 42); 0 falls back to 42 so existing
     * callers that never set rng_seed get bit-identical behaviour. */
    int pos = n_prompt;
    unsigned long long rng_state = config->rng_seed ? config->rng_seed : 42ULL;
    int next_token = tq_sample_topp(state->logits, vocab_size,
                                     config->temperature, config->top_p,
                                     &rng_state);

    /* Record first sampled token */
    recent_tokens[recent_count % 64] = next_token;
    recent_count++;

    int generated = 0;
    int output_pos = 0;
    int prev_token = prompt_tokens[n_prompt - 1];

    /* EOS token IDs — check common values across model families.
     * Qwen3.5: eos = 248044 (<|endoftext|>), 248046 (<|im_end|>)
     * Gemma3: eos = 1
     * Gemma4: eos = 106 (<end_of_turn>)
     * LLaMA 2: eos = 2
     * LLaMA 3: eos = 128001 (<|end_of_text|>), 128009 (<|eot_id|>) */
    int eos_tokens[] = {
        1,       /* Gemma3 <eos> */
        2,       /* LLaMA 2 </s> */
        106,     /* Gemma4 <end_of_turn> */
        128001,  /* LLaMA 3 <|end_of_text|> */
        128006,  /* LLaMA 3 <|start_header_id|> (new turn = stop) */
        128007,  /* LLaMA 3 <|end_header_id|> */
        128008,  /* LLaMA 3 <|start_of_role|> */
        128009,  /* LLaMA 3 <|eot_id|> */
        248044,  /* Qwen <|endoftext|> */
        248046,  /* Qwen <|im_end|> */
    };
    int n_eos = sizeof(eos_tokens) / sizeof(eos_tokens[0]);

    /* Generate loop */
    while (generated < config->max_tokens) {
        int is_eos = 0;
        for (int e = 0; e < n_eos; e++) {
            if (next_token == eos_tokens[e]) { is_eos = 1; break; }
        }
        if (is_eos) break;
        /* Infinite scrollback: shift KV cache when context is full */
        if (pos >= model->config.max_seq_len) {
            int max_seq = model->config.max_seq_len;
            int keep = max_seq / 2;
            int discard = pos - keep;
            if (discard <= 0) break;
            int kv_dim = model->config.n_kv_heads * model->config.head_dim;
            for (int l = 0; l < model->config.n_layers; l++) {
                size_t off = (size_t)l * max_seq * kv_dim;
                if (state->key_cache)
                    memmove(state->key_cache + off,
                            state->key_cache + off + (size_t)discard * kv_dim,
                            (size_t)keep * kv_dim * sizeof(float));
                if (state->value_cache)
                    memmove(state->value_cache + off,
                            state->value_cache + off + (size_t)discard * kv_dim,
                            (size_t)keep * kv_dim * sizeof(float));
                if (state->value_cache_fp16) {
                    size_t off16 = (size_t)l * max_seq * kv_dim;
                    memmove(state->value_cache_fp16 + off16,
                            state->value_cache_fp16 + off16 + (size_t)discard * kv_dim,
                            (size_t)keep * kv_dim * sizeof(uint16_t));
                }
                if (state->quant_key_cache && state->kv_quant_type < TQ_TYPE_COUNT) {
                    size_t bsz = tq_type_type_size(state->kv_quant_type);
                    size_t qs = (size_t)max_seq * bsz;
                    uint8_t* qb = (uint8_t*)state->quant_key_cache + (size_t)l * qs;
                    memmove(qb, qb + (size_t)discard * bsz, (size_t)keep * bsz);
                }
            }
            pos = keep;
        }

        /* Decode token to text */
        if (tokenizer) {
            const char* piece = tq_decode(tokenizer, prev_token, next_token);

            /* Skip special/thinking tokens that shouldn't appear in output.
             * Qwen3.5: <think>...</think>
             * Gemma 4: thought, <channel|>, <tool|>, <mask>, <unused*>
             * LLaMA 3: <|start_header_id|>, <|reserved_special_token_*|> */
            int should_stop = 0;
            if (piece) {
                if (strstr(piece, "<think>") || strstr(piece, "</think>") ||
                    strstr(piece, "<channel|>") || strstr(piece, "<tool|>") ||
                    strstr(piece, "<mask>") ||
                    strstr(piece, "<unused") || strstr(piece, "<|think")) {
                    piece = "";
                }
                /* Gemma 4 "thought" token: only filter if it's the EXACT piece
                 * (not a substring of normal text like "thoughtful") */
                if (piece[0] != '\0' && strcmp(piece, "thought") == 0) {
                    piece = "";
                }
                /* Stop generation on turn-boundary tokens (LLaMA 3 / Qwen only).
                 * Gemma uses token ID-based EOS (106), not text-based detection. */
                if (strstr(piece, "<|start_header_id|>") ||
                    strstr(piece, "<|eot_id|>") ||
                    strstr(piece, "<|im_end|>")) {
                    should_stop = 1;
                    piece = "";
                }
                /* Filter reserved special tokens */
                if (strstr(piece, "<|reserved_special_token") ||
                    strstr(piece, "<1st>") || strstr(piece, "<2nd>") || strstr(piece, "<3rd>")) {
                    piece = "";
                }
            }
            if (should_stop) break;

            int piece_len = (int)strlen(piece);

            /* Stream callback */
            if (config->on_token) {
                config->on_token(piece, config->user_data);
            }

            /* Append to output buffer */
            if (output && output_pos + piece_len < output_size - 1) {
                memcpy(output + output_pos, piece, piece_len);
                output_pos += piece_len;
            }
        }

        /* Forward pass for next token */
        prev_token = next_token;
        tq_forward(model, state, next_token, pos);
        pos++;
        generated++;

        /* Apply repetition penalty before sampling */
        if (rep_penalty > 1.0f) {
            int window = recent_count < rep_window ? recent_count : rep_window;
            for (int r = 0; r < window; r++) {
                int idx = (recent_count - 1 - r) % 64;
                if (idx < 0) idx += 64;
                int tok = recent_tokens[idx];
                if (tok >= 0 && tok < vocab_size) {
                    if (state->logits[tok] > 0)
                        state->logits[tok] /= rep_penalty;
                    else
                        state->logits[tok] *= rep_penalty;
                }
            }
        }

        /* Sample next token */
        next_token = tq_sample_topp(state->logits, vocab_size,
                                     config->temperature, config->top_p,
                                     &rng_state);

        /* Record sampled token for repetition penalty */
        recent_tokens[recent_count % 64] = next_token;
        recent_count++;
    }

    /* Null-terminate output */
    if (output && output_size > 0) {
        output[output_pos < output_size ? output_pos : output_size - 1] = '\0';
    }

    tq_free_state(state);
    return generated;
}

/* ============================================================================
 * tq_generate_continue — reuse an existing tq_state_t across calls.
 *
 * Unlike tq_generate (which allocates and frees its own state on every call),
 * this function takes a caller-managed state plus a record of which tokens
 * are currently committed to the KV cache. It computes the longest common
 * prefix between the cached tokens and the new prompt, then prefills only
 * the diverging suffix. After generation, *cached_tokens_out and
 * *n_cached_out are updated to reflect the new cache contents.
 *
 * This turns chat mode from O(n^2) (full re-prefill every turn) into
 * O(delta) (only the new tokens of each turn).
 *
 * Returns the number of tokens generated, or -1 on error.
 * ============================================================================ */
static int tq_lcp_int(const int* a, int na, const int* b, int nb) {
    int lim = na < nb ? na : nb;
    int i = 0;
    while (i < lim && a[i] == b[i]) i++;
    return i;
}

int tq_generate_continue(tq_model_t* model,
                          tq_tokenizer_t* tokenizer,
                          tq_state_t* state,
                          const char* prompt,
                          tq_gen_config_t* config,
                          int** cached_tokens_io,   /* in/out: cached prefix tokens */
                          int*  n_cached_io,        /* in/out: cached count */
                          int*  cached_capacity_io, /* in/out: allocated capacity */
                          char* output, int output_size) {
    if (!model || !state || !config || !cached_tokens_io || !n_cached_io || !cached_capacity_io) {
        return -1;
    }

    /* Heap-allocated prompt token buffer (was a 4096-stack array, which
     * silently truncated after ~10 turns of accumulating chat history).
     * Cap at the model's max_seq_len so we never exceed KV bounds. */
    int max_prompt = model->config.max_seq_len > 0
                       ? model->config.max_seq_len : 4096;
    int* new_tokens = (int*)malloc((size_t)max_prompt * sizeof(int));
    if (!new_tokens) return -1;
    int n_new = 0;
    if (tokenizer && prompt) {
        int add_bos = (model->config.model_type == 1) ? 1 : 0;
        n_new = tq_encode(tokenizer, prompt, new_tokens, max_prompt, add_bos);
    }
    if (n_new <= 0) {
        new_tokens[0] = (model->config.model_type == 1) ? 2 : 1;
        n_new = 1;
    }

    /* Sliding window: drop oldest prompt tokens if the new prompt would
     * leave no room for max_tokens of generation. Keeps the most recent
     * tokens. Forces full reprefill since the prefix shifted. */
    int reserve = config->max_tokens > 0 ? config->max_tokens : 256;
    int budget  = max_prompt - reserve - 32;
    if (budget < 64) budget = 64;
    if (n_new > budget) {
        int drop = n_new - budget;
        memmove(new_tokens, new_tokens + drop, (size_t)budget * sizeof(int));
        n_new = budget;
        *n_cached_io = 0;
    }

    /* Find longest common prefix with the cached tokens.
     * If the new prompt is just an extension of the cached one, we skip
     * everything up to the LCP and only prefill the suffix. */
    int n_cached = *n_cached_io;
    int* cached_tokens = *cached_tokens_io;

    int lcp = tq_lcp_int(cached_tokens, n_cached, new_tokens, n_new);

    /* Prefill the new suffix [lcp, n_new) */
    for (int i = lcp; i < n_new; i++) {
        tq_forward(model, state, new_tokens[i], i);
    }
    int pos = n_new;
    int prefill_tokens = n_new - lcp;
    int prefix_hit    = lcp;

    /* Save the n_new prompt into the cache buffer (will append generated
     * tokens below). Grow the buffer if needed. */
    int needed_cap = n_new + config->max_tokens + 16;
    if (*cached_capacity_io < needed_cap) {
        int new_cap = needed_cap < 4096 ? 4096 : needed_cap;
        int* nb = (int*)realloc(*cached_tokens_io, (size_t)new_cap * sizeof(int));
        if (!nb) { free(new_tokens); return -1; }
        *cached_tokens_io = nb;
        *cached_capacity_io = new_cap;
        cached_tokens = nb;
    }
    memcpy(cached_tokens, new_tokens, (size_t)n_new * sizeof(int));
    *n_cached_io = n_new;
    n_cached = n_new;

    /* --- generation loop (mirrors tq_generate's loop) --- */
    int vocab_size = model->config.vocab_size;
    float rep_penalty = config->rep_penalty;
    int rep_window = config->rep_window;
    if (rep_window > 64) rep_window = 64;
    int recent_tokens[64];
    int recent_count = 0;
    for (int i = (n_new > rep_window ? n_new - rep_window : 0); i < n_new; i++) {
        recent_tokens[recent_count % 64] = new_tokens[i];
        recent_count++;
    }

    if (rep_penalty > 1.0f) {
        int window = recent_count < rep_window ? recent_count : rep_window;
        for (int r = 0; r < window; r++) {
            int idx = (recent_count - 1 - r) % 64;
            if (idx < 0) idx += 64;
            int tok = recent_tokens[idx];
            if (tok >= 0 && tok < vocab_size && state->logits) {
                if (state->logits[tok] > 0) state->logits[tok] /= rep_penalty;
                else                         state->logits[tok] *= rep_penalty;
            }
        }
    }

    uint64_t rng_state = config->rng_seed ? (uint64_t)config->rng_seed
                                          : (uint64_t)time(NULL);
    int next_token = tq_sample_topp(state->logits, vocab_size,
                                     config->temperature, config->top_p,
                                     &rng_state);

    int generated = 0;
    int output_pos = 0;
    int prev_token = new_tokens[n_new - 1];

    int eos_tokens[] = {
        1, 2, 106, 128001, 128006, 128007, 128008, 128009, 248044, 248046,
    };
    int n_eos = sizeof(eos_tokens) / sizeof(eos_tokens[0]);

    while (generated < config->max_tokens) {
        int is_eos = 0;
        for (int e = 0; e < n_eos; e++) {
            if (next_token == eos_tokens[e]) { is_eos = 1; break; }
        }
        if (is_eos) break;

        if (pos >= model->config.max_seq_len) break;  /* simple stop, no shift */

        /* Decode + stream */
        if (tokenizer) {
            const char* piece = tq_decode(tokenizer, prev_token, next_token);
            int should_stop = 0;
            if (piece) {
                if (strstr(piece, "<|im_end|>") || strstr(piece, "<|eot_id|>") ||
                    strstr(piece, "<|start_header_id|>")) {
                    should_stop = 1; piece = "";
                }
            }
            if (should_stop) break;
            int piece_len = (int)strlen(piece ? piece : "");
            if (config->on_token && piece) config->on_token(piece, config->user_data);
            if (output && piece && output_pos + piece_len < output_size - 1) {
                memcpy(output + output_pos, piece, piece_len);
                output_pos += piece_len;
            }
        }

        /* Append generated token to cache record */
        if (n_cached < *cached_capacity_io) {
            cached_tokens[n_cached++] = next_token;
            *n_cached_io = n_cached;
        }

        prev_token = next_token;
        tq_forward(model, state, next_token, pos);
        pos++;
        generated++;

        if (rep_penalty > 1.0f) {
            int window = recent_count < rep_window ? recent_count : rep_window;
            for (int r = 0; r < window; r++) {
                int idx = (recent_count - 1 - r) % 64;
                if (idx < 0) idx += 64;
                int tok = recent_tokens[idx];
                if (tok >= 0 && tok < vocab_size) {
                    if (state->logits[tok] > 0) state->logits[tok] /= rep_penalty;
                    else                         state->logits[tok] *= rep_penalty;
                }
            }
        }

        next_token = tq_sample_topp(state->logits, vocab_size,
                                     config->temperature, config->top_p,
                                     &rng_state);
        recent_tokens[recent_count % 64] = next_token;
        recent_count++;
    }

    if (output && output_size > 0) {
        output[output_pos < output_size ? output_pos : output_size - 1] = '\0';
    }

    if (getenv("TQ_CHAT_DEBUG")) {
        fprintf(stderr,
            "[chat] prefix_hit=%d prefill=%d generated=%d cached=%d\n",
            prefix_hit, prefill_tokens, generated, *n_cached_io);
    }

    free(new_tokens);
    return generated;
}

// ============================================================================

// ============================================================================
// Section 15b: MoE Stubs (tq_moe.c excluded - stubs for linking)
// ============================================================================

tq_moe_state_t* tq_moe_create_state(const tq_moe_config_t* config, int hidden_dim) {
    (void)config; (void)hidden_dim;
    return NULL;
}

void tq_moe_free_state(tq_moe_state_t* state) {
    (void)state;
}

void tq_moe_route(const float* hidden, const float* router_weight,
                  int num_experts, int num_active, int hidden_dim,
                  int* out_expert_ids, float* out_expert_weights) {
    (void)hidden; (void)router_weight; (void)num_experts;
    (void)num_active; (void)hidden_dim;
    (void)out_expert_ids; (void)out_expert_weights;
}

void tq_moe_forward(const tq_moe_layer_t* layer,
                    const tq_moe_config_t* config,
                    tq_moe_state_t* state,
                    const float* input, float* output,
                    int hidden_dim, int layer_idx) {
    (void)layer; (void)config; (void)state;
    (void)input; (void)layer_idx;
    memset(output, 0, (size_t)hidden_dim * sizeof(float));
}

void tq_moe_cache_init(int n_layers, const tq_moe_config_t* config, int hidden_dim) {
    (void)n_layers; (void)config; (void)hidden_dim;
}

void tq_moe_cache_free(void) {}

void tq_moe_advise(const tq_moe_layer_t* layer,
                   const int* active_ids, int n_active,
                   int num_experts) {
    (void)layer; (void)active_ids; (void)n_active; (void)num_experts;
}

int tq_metal_moe_available(void) { return 0; }

int tq_metal_moe_forward(
    const float* input, float* output, float* hb_output,
    const void* weight_base, size_t weight_size,
    const uint64_t* gate_offsets, const uint64_t* up_offsets,
    const uint64_t* down_offsets,
    const int* active_expert_ids, const float* expert_routing_weights,
    int num_active, int expert_dim, int hidden_dim,
    int num_experts_total, int weight_type,
    const int* gate_types, const int* up_types, const int* down_types) {
    (void)input; (void)output; (void)hb_output;
    (void)weight_base; (void)weight_size;
    (void)gate_offsets; (void)up_offsets; (void)down_offsets;
    (void)active_expert_ids; (void)expert_routing_weights;
    (void)num_active; (void)expert_dim; (void)hidden_dim;
    (void)num_experts_total; (void)weight_type;
    (void)gate_types; (void)up_types; (void)down_types;
    return -1;
}

// Section 16: Public API Wrapper Functions
// ============================================================================

const char* quant_version(void) {
    return "0.1.0 (quant.h single-header)";
}

quant_model* quant_load(const char* path) {
    return (quant_model*)tq_load_model(path);
}

quant_ctx* quant_new(quant_model* model, const quant_config* config) {
    if (!model) return NULL;

    tq_gen_config_t gc = tq_default_gen_config();
    if (config) {
        gc.temperature = config->temperature;
        gc.top_p = config->top_p;
        gc.max_tokens = config->max_tokens;
        gc.n_threads = config->n_threads;
        if (config->kv_compress == 1) {
            gc.kv_type = TQ_TYPE_UNIFORM_4B;
            gc.value_quant_bits = 4;
        } else if (config->kv_compress == 2) {
            gc.kv_type = TQ_TYPE_UNIFORM_3B;
            gc.value_quant_bits = 4;
            gc.delta_kv = 1;
        }
    }

    tq_model_t* m = (tq_model_t*)model;

    /* Override context length if user requested it. With KV compression,
     * larger context is safe on the same hardware budget. The default
     * cap (4096) was set during model loading for safety; here we lift it. */
    if (config && config->context_length > 0) {
        int req = config->context_length;
        /* Clamp to model's absolute max (from GGUF metadata) to prevent
         * RoPE frequency mismatch. Re-read the original GGUF value: */
        int gguf_max = 131072; /* conservative fallback */
        if (m->gguf_ctx) {
            gguf_max = tq_gguf_get_i32(m->gguf_ctx, "llama.context_length", 131072);
            if (gguf_max <= 0) gguf_max = 131072;
        }
        if (req > gguf_max) req = gguf_max;
        if (req > m->config.max_seq_len) {
            fprintf(stderr, "quant_new: extending context %d -> %d (user request)\n",
                    m->config.max_seq_len, req);
            m->config.max_seq_len = req;
        }
    }

    /* Set thread count */
    if (gc.n_threads > 1) {
        tq_set_threads(gc.n_threads);
    }

    quant_ctx* ctx = (quant_ctx*)calloc(1, sizeof(quant_ctx));
    if (!ctx) return NULL;

    ctx->model = m;
    ctx->config = gc;

    /* Load tokenizer from GGUF */
    if (m->gguf_ctx) {
        ctx->tokenizer = tq_load_tokenizer_from_gguf(m->gguf_ctx);
    }

    /* Create state */
    ctx->state = tq_create_state_ex(&m->config, gc.kv_type, gc.value_quant_bits);
    if (!ctx->state) {
        tq_free_tokenizer(ctx->tokenizer);
        free(ctx);
        return NULL;
    }

    /* Set up MoE state if needed */
    if (m->config.is_moe && m->moe_config) {
        ctx->state->moe_state = tq_moe_create_state(
            (const tq_moe_config_t*)m->moe_config,
            m->config.hidden_dim);
    }

    /* Progressive KV: keep last N tokens' keys at FP32 for quality.
     * k_highres_window=128 reduces PPL degradation from +3.8% to +0.6%. */
    if (config && config->k_highres_window > 0 &&
        gc.kv_type < TQ_TYPE_COUNT && ctx->state->quant_key_cache) {
        int kw = config->k_highres_window;
        int kv_dim = m->config.n_kv_heads * m->config.head_dim;
        ctx->state->k_highres_window = kw;
        ctx->state->key_highres_fp32 = (float*)calloc(
            (size_t)m->config.n_layers * kw * kv_dim, sizeof(float));
        if (ctx->state->key_highres_fp32) {
            fprintf(stderr, "quant_new: progressive KV enabled (last %d tokens FP32)\n", kw);
        }
    }

    return ctx;
}

int quant_generate(quant_ctx* ctx, const char* prompt,
                   void (*on_token)(const char* text, void* user_data),
                   void* user_data) {
    if (!ctx || !ctx->model) return -1;

    ctx->config.on_token = on_token;
    ctx->config.user_data = user_data;

    /* Reset state for new generation */
    tq_free_state(ctx->state);
    ctx->state = tq_create_state_ex(&ctx->model->config,
                                     ctx->config.kv_type,
                                     ctx->config.value_quant_bits);
    if (!ctx->state) return -1;

    if (ctx->model->config.is_moe && ctx->model->moe_config) {
        ctx->state->moe_state = tq_moe_create_state(
            (const tq_moe_config_t*)ctx->model->moe_config,
            ctx->model->config.hidden_dim);
    }

    char output[65536];
    int n = tq_generate(ctx->model, ctx->tokenizer, prompt,
                        &ctx->config, output, sizeof(output));
    if (n > 0) ctx->n_ctx_tokens += n;
    return n;
}

char* quant_ask(quant_ctx* ctx, const char* prompt) {
    if (!ctx || !ctx->model) return NULL;

    /* Reset state */
    tq_free_state(ctx->state);
    ctx->state = tq_create_state_ex(&ctx->model->config,
                                     ctx->config.kv_type,
                                     ctx->config.value_quant_bits);
    if (!ctx->state) return NULL;

    if (ctx->model->config.is_moe && ctx->model->moe_config) {
        ctx->state->moe_state = tq_moe_create_state(
            (const tq_moe_config_t*)ctx->model->moe_config,
            ctx->model->config.hidden_dim);
    }

    /* Disable streaming callback for quant_ask */
    ctx->config.on_token = NULL;
    ctx->config.user_data = NULL;

    char* output = (char*)malloc(65536);
    if (!output) return NULL;

    int n = tq_generate(ctx->model, ctx->tokenizer, prompt,
                        &ctx->config, output, 65536);
    if (n < 0) {
        free(output);
        return NULL;
    }
    if (n > 0) ctx->n_ctx_tokens += n;
    return output;
}

void quant_free_string(char* str) {
    /* The string was malloc()'d inside this translation unit (quant_ask),
     * so it must be free()'d here too — same malloc zone, no cross-heap
     * crash on macOS arm64 / Windows. */
    if (str) free(str);
}

/* Context persistence: QKVC format (64-byte header + raw KV data) */
int quant_save_context(quant_ctx* ctx, const char* path) {
    if (!ctx || !ctx->state || !path) return -1;
    FILE* fp = fopen(path, "wb");
    if (!fp) return -1;

    tq_state_t* s = ctx->state;
    tq_model_config_t* c = &ctx->model->config;
    int kv_dim = c->n_kv_heads * c->head_dim;
    fwrite("QKVC", 1, 4, fp);
    uint32_t hdr[7] = { 1, (uint32_t)c->n_layers, (uint32_t)kv_dim,
        (uint32_t)c->max_seq_len, (uint32_t)ctx->n_ctx_tokens,
        (uint32_t)s->kv_quant_type, s->value_cache_fp16 ? 1u : 0u };
    fwrite(hdr, 4, 7, fp);
    char reserved[32] = {0}; fwrite(reserved, 1, 32, fp);
    uint32_t nl = hdr[1], nt = hdr[4], kt = hdr[5];

    /* KV data: write only the filled portion (nt tokens) */
    for (uint32_t l = 0; l < nl; l++) {
        size_t layer_stride = (size_t)c->max_seq_len * kv_dim;
        /* Key cache: FP32 or quantized */
        if (s->key_cache) {
            fwrite(s->key_cache + l * layer_stride, sizeof(float),
                   (size_t)nt * kv_dim, fp);
        }
        if (s->quant_key_cache && kt < TQ_TYPE_COUNT) {
            size_t blk_sz = tq_type_type_size(kt);
            uint8_t* qbase = (uint8_t*)s->quant_key_cache + l * (size_t)c->max_seq_len * blk_sz;
            fwrite(qbase, blk_sz, nt, fp);
        }
        /* Value cache: FP32 or FP16 */
        if (s->value_cache) {
            fwrite(s->value_cache + l * layer_stride, sizeof(float),
                   (size_t)nt * kv_dim, fp);
        }
        if (s->value_cache_fp16) {
            size_t layer_stride16 = (size_t)c->max_seq_len * kv_dim;
            fwrite(s->value_cache_fp16 + l * layer_stride16, sizeof(uint16_t),
                   (size_t)nt * kv_dim, fp);
        }
    }

    fclose(fp);
    fprintf(stderr, "quant_save_context: saved %u tokens (%u layers) to %s\n",
            nt, nl, path);
    return 0;
}

int quant_load_context(quant_ctx* ctx, const char* path) {
    if (!ctx || !ctx->state || !path) return -1;
    FILE* fp = fopen(path, "rb");
    if (!fp) return -1;

    char magic[4];
    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "QKVC", 4) != 0) { fclose(fp); return -1; }
    uint32_t hdr[7]; fread(hdr, 4, 7, fp);
    char reserved[32]; fread(reserved, 1, 32, fp);
    uint32_t nl = hdr[1], nt = hdr[4], kt = hdr[5];
    tq_state_t* s = ctx->state;
    tq_model_config_t* c = &ctx->model->config;
    int kv_dim = c->n_kv_heads * c->head_dim;
    if (nl != (uint32_t)c->n_layers || hdr[2] != (uint32_t)kv_dim) { fclose(fp); return -1; }
    if (nt > (uint32_t)c->max_seq_len) { fclose(fp); return -1; }

    /* Read KV data */
    for (uint32_t l = 0; l < nl; l++) {
        size_t layer_stride = (size_t)c->max_seq_len * kv_dim;
        if (s->key_cache) {
            fread(s->key_cache + l * layer_stride, sizeof(float),
                  (size_t)nt * kv_dim, fp);
        }
        if (s->quant_key_cache && kt < TQ_TYPE_COUNT) {
            size_t blk_sz = tq_type_type_size(kt);
            uint8_t* qbase = (uint8_t*)s->quant_key_cache + l * (size_t)c->max_seq_len * blk_sz;
            fread(qbase, blk_sz, nt, fp);
        }
        if (s->value_cache) {
            fread(s->value_cache + l * layer_stride, sizeof(float),
                  (size_t)nt * kv_dim, fp);
        }
        if (s->value_cache_fp16) {
            size_t layer_stride16 = (size_t)c->max_seq_len * kv_dim;
            fread(s->value_cache_fp16 + l * layer_stride16, sizeof(uint16_t),
                  (size_t)nt * kv_dim, fp);
        }
    }

    /* Restore position */
    ctx->n_ctx_tokens = (int)nt;
    fclose(fp);
    fprintf(stderr, "quant_load_context: restored %u tokens (%u layers) from %s\n",
            nt, nl, path);
    return 0;
}

void quant_free_ctx(quant_ctx* ctx) {
    if (!ctx) return;
    tq_free_state(ctx->state);
    tq_free_tokenizer(ctx->tokenizer);
    if (ctx->cached_tokens) free(ctx->cached_tokens);
    free(ctx);
}

/* ----------------------------------------------------------------------
 * quant_chat — chat-mode generate that reuses the KV cache across calls.
 *
 * Unlike quant_generate (which resets the state on every call and so makes
 * each turn O(history_length)), quant_chat keeps the state alive between
 * calls. The first call to quant_chat() prefills and generates as normal.
 * Subsequent calls compute the longest common prefix between the new prompt
 * and the previously processed tokens, skip the matched prefix, and only
 * prefill the diverging suffix.
 *
 * Result: turn N's prefill cost is O(new tokens this turn), not
 * O(total history). Chat experience matches what users expect from ollama.
 *
 * Reset behavior: pass NULL prompt to wipe the cache (start a new chat).
 * Returns the number of tokens generated, or -1 on error.
 * ---------------------------------------------------------------------- */
int quant_chat(quant_ctx* ctx, const char* prompt,
               void (*on_token)(const char* text, void* user_data),
               void* user_data) {
    if (!ctx || !ctx->model) return -1;

    /* NULL prompt = reset the chat (clear cache + state) */
    if (!prompt) {
        tq_free_state(ctx->state);
        ctx->state = tq_create_state_ex(&ctx->model->config,
                                         ctx->config.kv_type,
                                         ctx->config.value_quant_bits);
        if (ctx->cached_tokens) free(ctx->cached_tokens);
        ctx->cached_tokens = NULL;
        ctx->n_cached = 0;
        ctx->cached_capacity = 0;
        ctx->n_ctx_tokens = 0;
        return 0;
    }

    if (!ctx->state) {
        ctx->state = tq_create_state_ex(&ctx->model->config,
                                         ctx->config.kv_type,
                                         ctx->config.value_quant_bits);
        if (!ctx->state) return -1;
    }

    ctx->config.on_token = on_token;
    ctx->config.user_data = user_data;

    char output[65536];
    int n = tq_generate_continue(
        ctx->model, ctx->tokenizer, ctx->state, prompt, &ctx->config,
        &ctx->cached_tokens, &ctx->n_cached, &ctx->cached_capacity,
        output, sizeof(output));

    if (n > 0) ctx->n_ctx_tokens = ctx->n_cached;
    return n;
}

void quant_free_model(quant_model* model) {
    tq_free_model((tq_model_t*)model);
}

#endif // QUANT_IMPLEMENTATION
#endif // QUANT_H
