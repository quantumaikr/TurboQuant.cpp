/**
 * TurboQuant context and high-level API implementation
 *
 * Thread safety: a pthread mutex protects quantize and attention calls.
 */

#include "turboquant/turboquant.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#ifdef _WIN32
#include <windows.h>
#define pthread_mutex_t SRWLOCK
#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
#define pthread_mutex_init(m, a) (InitializeSRWLock(m), 0)
#define pthread_mutex_lock(m) AcquireSRWLockExclusive(m)
#define pthread_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define pthread_mutex_destroy(m) ((void)0)
#else
#include <pthread.h>
#endif

struct tq_context {
    tq_backend backend;
    pthread_mutex_t mutex;
};

tq_status tq_init(tq_context_t** ctx, tq_backend backend) {
    if (!ctx) return TQ_ERR_NULL_PTR;

    tq_context_t* c = (tq_context_t*)calloc(1, sizeof(tq_context_t));
    if (!c) return TQ_ERR_OUT_OF_MEM;

    c->backend = (backend == TQ_BACKEND_AUTO) ? TQ_BACKEND_CPU : backend;

    if (pthread_mutex_init(&c->mutex, NULL) != 0) {
        free(c);
        return TQ_ERR_BACKEND;
    }

    *ctx = c;
    return TQ_OK;
}

void tq_free(tq_context_t* ctx) {
    if (!ctx) return;
    pthread_mutex_destroy(&ctx->mutex);
    free(ctx);
}

tq_backend tq_get_backend(const tq_context_t* ctx) {
    if (!ctx) return TQ_BACKEND_CPU;
    return ctx->backend;
}

size_t tq_quantize_keys_size(int n, int head_dim, tq_type type) {
    if (type < 0 || type >= TQ_TYPE_COUNT) return 0;
    if (n <= 0 || head_dim <= 0) return 0;
    if (n > TQ_MAX_SEQ_LEN) return 0;

    size_t block_size = TQ_TRAITS[type].block_size;
    size_t type_size  = TQ_TRAITS[type].type_size;
    /* Each key vector of head_dim elements uses ceil(head_dim/block_size) blocks */
    size_t blocks_per_key = ((size_t)head_dim + block_size - 1) / block_size;

    /* Check multiplication overflow: n * blocks_per_key * type_size */
    size_t nb = (size_t)n * blocks_per_key;
    if (blocks_per_key != 0 && nb / blocks_per_key != (size_t)n) return 0;
    size_t result = nb * type_size;
    if (type_size != 0 && result / type_size != nb) return 0;

    return result;
}

tq_status tq_quantize_keys(tq_context_t* ctx,
                           const float* keys, int n, int head_dim,
                           tq_type type,
                           void* out, size_t out_size) {
    if (!ctx || !keys || !out) return TQ_ERR_NULL_PTR;
    if (type < 0 || type >= TQ_TYPE_COUNT) return TQ_ERR_INVALID_TYPE;
    if (n == 0) return TQ_OK;
    if (head_dim < 2) return TQ_ERR_INVALID_DIM;

    /* PolarQuant and TurboQuant require even head_dim (polar coordinate pairs) */
    if (type == TQ_TYPE_POLAR_3B || type == TQ_TYPE_POLAR_4B ||
        type == TQ_TYPE_TURBO_3B || type == TQ_TYPE_TURBO_4B) {
        if (head_dim % 2 != 0) return TQ_ERR_INVALID_DIM;
    }

    size_t needed = tq_quantize_keys_size(n, head_dim, type);
    if (needed == 0) return TQ_ERR_INVALID_DIM;
    if (out_size < needed) return TQ_ERR_BUFFER_TOO_SMALL;

    tq_quantize_fn qfn = TQ_TRAITS[type].quantize;
    if (!qfn) return TQ_ERR_NOT_IMPL;

    pthread_mutex_lock(&ctx->mutex);

    size_t block_size = TQ_TRAITS[type].block_size;
    size_t type_size  = TQ_TRAITS[type].type_size;
    int blocks_per_key = (head_dim + (int)block_size - 1) / (int)block_size;
    uint8_t* dst = (uint8_t*)out;

    for (int i = 0; i < n; i++) {
        for (int b = 0; b < blocks_per_key; b++) {
            int offset = b * (int)block_size;
            int chunk = head_dim - offset;
            if (chunk > (int)block_size) chunk = (int)block_size;
            qfn(keys + i * head_dim + offset, dst, chunk);
            dst += type_size;
        }
    }

    pthread_mutex_unlock(&ctx->mutex);

    return TQ_OK;
}

tq_status tq_dequantize_keys(tq_context_t* ctx,
                             const void* quantized, int n, int head_dim,
                             tq_type type,
                             float* out) {
    if (!ctx || !quantized || !out) return TQ_ERR_NULL_PTR;
    if (type < 0 || type >= TQ_TYPE_COUNT) return TQ_ERR_INVALID_TYPE;
    if (head_dim <= 0) return TQ_ERR_INVALID_DIM;

    tq_dequantize_fn dfn = TQ_TRAITS[type].dequantize;
    if (!dfn) return TQ_ERR_NOT_IMPL;

    size_t block_size = TQ_TRAITS[type].block_size;
    size_t type_size  = TQ_TRAITS[type].type_size;
    int blocks_per_key = (head_dim + (int)block_size - 1) / (int)block_size;
    const uint8_t* src = (const uint8_t*)quantized;

    for (int i = 0; i < n; i++) {
        for (int b = 0; b < blocks_per_key; b++) {
            int offset = b * (int)block_size;
            int chunk = head_dim - offset;
            if (chunk > (int)block_size) chunk = (int)block_size;
            dfn(src, out + i * head_dim + offset, chunk);
            src += type_size;
        }
    }

    return TQ_OK;
}

tq_status tq_quantize_values(tq_context_t* ctx,
                             const float* values, int n, int head_dim,
                             int bits,
                             void* out, size_t out_size) {
    if (!ctx || !values || !out) return TQ_ERR_NULL_PTR;
    if (bits != 2 && bits != 4) return TQ_ERR_INVALID_TYPE;

    tq_type type = (bits == 4) ? TQ_TYPE_UNIFORM_4B : TQ_TYPE_UNIFORM_2B;
    tq_quantize_fn qfn = TQ_TRAITS[type].quantize;
    if (!qfn) return TQ_ERR_NOT_IMPL;

    size_t block_size = TQ_TRAITS[type].block_size;
    size_t type_size  = TQ_TRAITS[type].type_size;
    int blocks_per_key = ((int)head_dim + (int)block_size - 1) / (int)block_size;
    uint8_t* dst = (uint8_t*)out;

    for (int i = 0; i < n; i++) {
        for (int b = 0; b < blocks_per_key; b++) {
            int offset = b * (int)block_size;
            int chunk = head_dim - offset;
            if (chunk > (int)block_size) chunk = (int)block_size;
            qfn(values + i * head_dim + offset, dst, chunk);
            dst += type_size;
        }
    }

    return TQ_OK;
}

tq_status tq_attention(tq_context_t* ctx,
                       const float* query,
                       const void* kv_cache,
                       int seq_len, int head_dim,
                       tq_type type,
                       float* scores) {
    if (!ctx || !query || !kv_cache || !scores) return TQ_ERR_NULL_PTR;
    if (type < 0 || type >= TQ_TYPE_COUNT) return TQ_ERR_INVALID_TYPE;
    if (seq_len == 0) return TQ_OK;
    if (head_dim < 2) return TQ_ERR_INVALID_DIM;

    /* PolarQuant and TurboQuant require even head_dim */
    if (type == TQ_TYPE_POLAR_3B || type == TQ_TYPE_POLAR_4B ||
        type == TQ_TYPE_TURBO_3B || type == TQ_TYPE_TURBO_4B) {
        if (head_dim % 2 != 0) return TQ_ERR_INVALID_DIM;
    }

    tq_attention_fn afn = TQ_TRAITS[type].attention;
    if (!afn) return TQ_ERR_NOT_IMPL;

    pthread_mutex_lock(&ctx->mutex);
    afn(query, kv_cache, scores, seq_len, head_dim);
    pthread_mutex_unlock(&ctx->mutex);

    return TQ_OK;
}

/* ============================================================
 * K/V Asymmetric Quantization
 * ============================================================ */

size_t tq_quantize_kv_key_size(int n, int head_dim, tq_type key_type) {
    return tq_quantize_keys_size(n, head_dim, key_type);
}

size_t tq_quantize_kv_value_size(int n, int head_dim, tq_type value_type) {
    return tq_quantize_keys_size(n, head_dim, value_type);
}

tq_status tq_quantize_kv(tq_context_t* ctx,
                          const float* keys, const float* values,
                          int n, int head_dim,
                          tq_type key_type, tq_type value_type,
                          void* key_out, size_t key_out_size,
                          void* val_out, size_t val_out_size) {
    if (!ctx || !keys || !values || !key_out || !val_out)
        return TQ_ERR_NULL_PTR;
    if (key_type < 0 || key_type >= TQ_TYPE_COUNT)
        return TQ_ERR_INVALID_TYPE;
    if (value_type < 0 || value_type >= TQ_TYPE_COUNT)
        return TQ_ERR_INVALID_TYPE;
    if (n == 0) return TQ_OK;
    if (head_dim < 2) return TQ_ERR_INVALID_DIM;

    /* Validate buffer sizes */
    size_t key_needed = tq_quantize_kv_key_size(n, head_dim, key_type);
    if (key_needed == 0) return TQ_ERR_INVALID_DIM;
    if (key_out_size < key_needed) return TQ_ERR_BUFFER_TOO_SMALL;

    size_t val_needed = tq_quantize_kv_value_size(n, head_dim, value_type);
    if (val_needed == 0) return TQ_ERR_INVALID_DIM;
    if (val_out_size < val_needed) return TQ_ERR_BUFFER_TOO_SMALL;

    /* Quantize keys with key_type */
    tq_status status = tq_quantize_keys(ctx, keys, n, head_dim,
                                         key_type, key_out, key_out_size);
    if (status != TQ_OK) return status;

    /* Quantize values with value_type (same quantize path as keys) */
    status = tq_quantize_keys(ctx, values, n, head_dim,
                               value_type, val_out, val_out_size);
    return status;
}

/* ============================================================
 * RHT Pipeline — quantize/dequantize with Random Hadamard Transform
 * ============================================================ */

tq_status tq_quantize_keys_rht(tq_context_t* ctx,
                                const float* keys, int n, int head_dim,
                                tq_type type, uint32_t rht_seed,
                                void* out, size_t out_size) {
    if (!ctx || !keys || !out) return TQ_ERR_NULL_PTR;
    if (n == 0) return TQ_OK;
    if (head_dim < 2) return TQ_ERR_INVALID_DIM;

    /* Allocate temp buffer for RHT-transformed keys */
    size_t total_floats = (size_t)n * (size_t)head_dim;
    float* temp = (float*)malloc(total_floats * sizeof(float));
    if (!temp) return TQ_ERR_OUT_OF_MEM;

    /* Copy keys to temp buffer */
    memcpy(temp, keys, total_floats * sizeof(float));

    /* Apply RHT to each key vector independently */
    for (int i = 0; i < n; i++) {
        tq_rht_transform(temp + (size_t)i * head_dim, head_dim, rht_seed);
    }

    /* Quantize the rotated vectors using the standard path */
    tq_status status = tq_quantize_keys(ctx, temp, n, head_dim, type, out, out_size);

    free(temp);
    return status;
}

tq_status tq_dequantize_keys_rht(tq_context_t* ctx,
                                  const void* quantized, int n, int head_dim,
                                  tq_type type, uint32_t rht_seed,
                                  float* out) {
    if (!ctx || !quantized || !out) return TQ_ERR_NULL_PTR;
    if (n == 0) return TQ_OK;
    if (head_dim <= 0) return TQ_ERR_INVALID_DIM;

    /* Dequantize using the standard path */
    tq_status status = tq_dequantize_keys(ctx, quantized, n, head_dim, type, out);
    if (status != TQ_OK) return status;

    /* Apply inverse RHT to each dequantized vector */
    for (int i = 0; i < n; i++) {
        tq_rht_inverse(out + (size_t)i * head_dim, head_dim, rht_seed);
    }

    return TQ_OK;
}

/**
 * Recommend quantization strategy based on real-model A/B test findings.
 *
 * Validated on Qwen3.5-0.8B (real KV cache, not synthetic):
 *   uniform_4b:  cosine 0.994 (A+) at 7.5x compression
 *   mixed_4b8:   cosine 0.994 (A+) at 6.4x — best for outlier-heavy data
 *   uniform_2b:  cosine 0.953 (A)  at 14.2x — surprisingly good
 *   turbo_3b:    cosine 0.934 (B+) at 4.6x
 *   qjl_1b:      cosine 0.744 (C)  — NOT recommended (uniform_2b is better)
 *   polar_4b:    cosine 0.893 (B)  — uniform_4b is better at same bits
 *
 * Community finding (r/LocalLLaMA, llama.cpp #20969):
 *   MSE-only (uniform) outperforms QJL in practice.
 */
tq_type tq_recommend_strategy(int head_dim, int target_bits,
                              float quality_threshold) {
    /*
     * Decision tree (from real Qwen3.5-0.8B A/B test):
     *
     * quality_threshold > 0.99 → uniform_4b (safest A+ choice)
     * quality_threshold > 0.95 + head_dim >= 256 → mixed_4b8 (outlier handling)
     * quality_threshold > 0.95 → uniform_4b
     * target_bits <= 2 → uniform_2b (A grade at 14x compression)
     * target_bits <= 4 → uniform_4b (A+ default)
     * target_bits <= 5 + head_dim >= 256 → mixed_4b8
     * otherwise → uniform_4b
     */

    /* High quality requirement: always uniform_4b */
    if (quality_threshold > 0.99f) {
        return TQ_TYPE_UNIFORM_4B;
    }

    /* Large head_dim with outliers: mixed precision shines */
    if (head_dim >= 256 && quality_threshold > 0.95f && target_bits >= 4) {
        return TQ_TYPE_MIXED_4B8;
    }

    /* Max compression: uniform_2b is surprisingly good (cosine 0.953 on real data) */
    if (target_bits <= 2) {
        return TQ_TYPE_UNIFORM_2B;
    }

    /* 3-4 bits: uniform_4b beats polar and turbo in practice */
    if (target_bits <= 4) {
        return TQ_TYPE_UNIFORM_4B;
    }

    /* 5+ bits with large dims: mixed precision for outlier resilience */
    if (head_dim >= 256) {
        return TQ_TYPE_MIXED_4B8;
    }

    /* Default: uniform_4b — the community-validated production choice */
    return TQ_TYPE_UNIFORM_4B;
}
