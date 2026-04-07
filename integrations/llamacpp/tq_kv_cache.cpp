/**
 * quant.cpp <-> llama.cpp Integration
 *
 * This file provides the glue between quant.cpp's quantization types
 * and llama.cpp's GGML type system.
 *
 * Integration steps for llama.cpp users:
 * 1. Add quant.cpp as a subdirectory in CMakeLists.txt
 * 2. #include "integrations/llamacpp/tq_kv_cache.cpp"
 * 3. Call tq_ggml_register_types() during initialization
 * 4. Use --kv-cache-type turbo3 CLI option
 */

#include "tq_ggml_type.h"

extern "C" {
#include "turboquant/turboquant.h"
}

#include <cstdio>
#include <cstring>
#include <cstdlib>

/* ============================================================
 * GGML type IDs for quant.cpp types
 *
 * Standard GGML uses IDs 0-40 (as of the current ggml.h).
 * We use IDs starting at 256 to avoid any conflicts with
 * upstream additions or vendor extensions.
 * ============================================================ */

#define GGML_TYPE_TQ_BASE       256

enum {
    GGML_TYPE_TQ_POLAR_3B   = GGML_TYPE_TQ_BASE + 0,
    GGML_TYPE_TQ_POLAR_4B   = GGML_TYPE_TQ_BASE + 1,
    GGML_TYPE_TQ_QJL_1B     = GGML_TYPE_TQ_BASE + 2,
    GGML_TYPE_TQ_TURBO_3B   = GGML_TYPE_TQ_BASE + 3,
    GGML_TYPE_TQ_TURBO_4B   = GGML_TYPE_TQ_BASE + 4,
    GGML_TYPE_TQ_UNIFORM_4B = GGML_TYPE_TQ_BASE + 5,
    GGML_TYPE_TQ_UNIFORM_2B = GGML_TYPE_TQ_BASE + 6,
    GGML_TYPE_TQ_MIXED_4B8     = GGML_TYPE_TQ_BASE + 7,
    GGML_TYPE_TQ_TURBO_KV_3B  = GGML_TYPE_TQ_BASE + 8,
    GGML_TYPE_TQ_TURBO_KV_4B  = GGML_TYPE_TQ_BASE + 9,
    GGML_TYPE_TQ_TURBO_KV_1B  = GGML_TYPE_TQ_BASE + 10,
    GGML_TYPE_TQ_TURBO_KV_2B  = GGML_TYPE_TQ_BASE + 11,
    GGML_TYPE_TQ_UNIFORM_3B   = GGML_TYPE_TQ_BASE + 12,
    GGML_TYPE_TQ_TURBO_KV_5B  = GGML_TYPE_TQ_BASE + 13,
    GGML_TYPE_TQ_TURBO_KV_4BO = GGML_TYPE_TQ_BASE + 14,
    GGML_TYPE_TQ_COUNT         = 15,
};

/* ============================================================
 * Mapping helpers
 * ============================================================ */

static int tq_to_ggml_type(tq_type type) {
    switch (type) {
        case TQ_TYPE_POLAR_3B:   return GGML_TYPE_TQ_POLAR_3B;
        case TQ_TYPE_POLAR_4B:   return GGML_TYPE_TQ_POLAR_4B;
        case TQ_TYPE_QJL_1B:     return GGML_TYPE_TQ_QJL_1B;
        case TQ_TYPE_TURBO_3B:   return GGML_TYPE_TQ_TURBO_3B;
        case TQ_TYPE_TURBO_4B:   return GGML_TYPE_TQ_TURBO_4B;
        case TQ_TYPE_UNIFORM_4B: return GGML_TYPE_TQ_UNIFORM_4B;
        case TQ_TYPE_UNIFORM_2B: return GGML_TYPE_TQ_UNIFORM_2B;
        case TQ_TYPE_MIXED_4B8:     return GGML_TYPE_TQ_MIXED_4B8;
        case TQ_TYPE_TURBO_KV_3B:  return GGML_TYPE_TQ_TURBO_KV_3B;
        case TQ_TYPE_TURBO_KV_4B:  return GGML_TYPE_TQ_TURBO_KV_4B;
        case TQ_TYPE_TURBO_KV_1B:  return GGML_TYPE_TQ_TURBO_KV_1B;
        case TQ_TYPE_TURBO_KV_2B:  return GGML_TYPE_TQ_TURBO_KV_2B;
        case TQ_TYPE_UNIFORM_3B:   return GGML_TYPE_TQ_UNIFORM_3B;
        case TQ_TYPE_TURBO_KV_5B:  return GGML_TYPE_TQ_TURBO_KV_5B;
        case TQ_TYPE_TURBO_KV_4BO: return GGML_TYPE_TQ_TURBO_KV_4BO;
        default: return -1;
    }
}

static tq_type ggml_to_tq_type(int ggml_id) {
    switch (ggml_id) {
        case GGML_TYPE_TQ_POLAR_3B:   return TQ_TYPE_POLAR_3B;
        case GGML_TYPE_TQ_POLAR_4B:   return TQ_TYPE_POLAR_4B;
        case GGML_TYPE_TQ_QJL_1B:     return TQ_TYPE_QJL_1B;
        case GGML_TYPE_TQ_TURBO_3B:   return TQ_TYPE_TURBO_3B;
        case GGML_TYPE_TQ_TURBO_4B:   return TQ_TYPE_TURBO_4B;
        case GGML_TYPE_TQ_UNIFORM_4B: return TQ_TYPE_UNIFORM_4B;
        case GGML_TYPE_TQ_UNIFORM_2B: return TQ_TYPE_UNIFORM_2B;
        case GGML_TYPE_TQ_MIXED_4B8:     return TQ_TYPE_MIXED_4B8;
        case GGML_TYPE_TQ_TURBO_KV_3B:  return TQ_TYPE_TURBO_KV_3B;
        case GGML_TYPE_TQ_TURBO_KV_4B:  return TQ_TYPE_TURBO_KV_4B;
        case GGML_TYPE_TQ_TURBO_KV_1B:  return TQ_TYPE_TURBO_KV_1B;
        case GGML_TYPE_TQ_TURBO_KV_2B:  return TQ_TYPE_TURBO_KV_2B;
        case GGML_TYPE_TQ_UNIFORM_3B:   return TQ_TYPE_UNIFORM_3B;
        case GGML_TYPE_TQ_TURBO_KV_5B:  return TQ_TYPE_TURBO_KV_5B;
        case GGML_TYPE_TQ_TURBO_KV_4BO: return TQ_TYPE_TURBO_KV_4BO;
        default: return TQ_TYPE_COUNT;
    }
}

/* ============================================================
 * GGML-compatible from_float / to_float wrappers
 *
 * GGML expects:
 *   from_float(src_fp32, dst_quant, n_elements)
 *   to_float  (src_quant, dst_fp32, n_elements)
 *
 * These process n elements in block-sized chunks using the
 * TQ_TRAITS dispatch table.
 * ============================================================ */

static void tq_ggml_from_float(const float* src, void* dst, int64_t n, tq_type type) {
    int block_size = (int)tq_type_block_size(type);
    int type_size  = (int)tq_type_type_size(type);
    int num_blocks = (int)(n / block_size);

    const tq_type_traits_t* traits = &TQ_TRAITS[type];
    if (!traits->quantize) return;

    char* out = (char*)dst;
    for (int b = 0; b < num_blocks; b++) {
        traits->quantize(src + b * block_size, out + b * type_size, block_size);
    }
}

static void tq_ggml_to_float(const void* src, float* dst, int64_t n, tq_type type) {
    int block_size = (int)tq_type_block_size(type);
    int type_size  = (int)tq_type_type_size(type);
    int num_blocks = (int)(n / block_size);

    const tq_type_traits_t* traits = &TQ_TRAITS[type];
    if (!traits->dequantize) return;

    const char* in = (const char*)src;
    for (int b = 0; b < num_blocks; b++) {
        traits->dequantize(in + b * type_size, dst + b * block_size, block_size);
    }
}

/* Type-specific wrappers for GGML function pointer compatibility */
#define TQ_GGML_WRAPPERS(NAME, TYPE)                                              \
    static void tq_ggml_from_float_##NAME(const float* src, void* dst, int64_t n) { \
        tq_ggml_from_float(src, dst, n, TYPE);                                      \
    }                                                                                \
    static void tq_ggml_to_float_##NAME(const void* src, float* dst, int64_t n) {   \
        tq_ggml_to_float(src, dst, n, TYPE);                                         \
    }

TQ_GGML_WRAPPERS(polar_3b,   TQ_TYPE_POLAR_3B)
TQ_GGML_WRAPPERS(polar_4b,   TQ_TYPE_POLAR_4B)
TQ_GGML_WRAPPERS(qjl_1b,     TQ_TYPE_QJL_1B)
TQ_GGML_WRAPPERS(turbo_3b,   TQ_TYPE_TURBO_3B)
TQ_GGML_WRAPPERS(turbo_4b,   TQ_TYPE_TURBO_4B)
TQ_GGML_WRAPPERS(uniform_4b, TQ_TYPE_UNIFORM_4B)
TQ_GGML_WRAPPERS(uniform_2b, TQ_TYPE_UNIFORM_2B)
TQ_GGML_WRAPPERS(mixed_4b8,     TQ_TYPE_MIXED_4B8)
TQ_GGML_WRAPPERS(turbo_kv_3b,  TQ_TYPE_TURBO_KV_3B)
TQ_GGML_WRAPPERS(turbo_kv_4b,  TQ_TYPE_TURBO_KV_4B)
TQ_GGML_WRAPPERS(turbo_kv_1b,  TQ_TYPE_TURBO_KV_1B)
TQ_GGML_WRAPPERS(turbo_kv_2b,  TQ_TYPE_TURBO_KV_2B)
TQ_GGML_WRAPPERS(uniform_3b,  TQ_TYPE_UNIFORM_3B)
TQ_GGML_WRAPPERS(turbo_kv_5b, TQ_TYPE_TURBO_KV_5B)
TQ_GGML_WRAPPERS(turbo_kv_4bo, TQ_TYPE_TURBO_KV_4BO)

/* ============================================================
 * vec_dot wrappers (quantized key . FP32 query -> scalar)
 *
 * GGML vec_dot signature:
 *   void vec_dot(int n, float* result, const void* x, const float* y)
 *
 * We dequantize x into a temporary buffer and compute the dot
 * product with y. Stack allocation handles head_dim <= 512;
 * larger dimensions fall back to heap allocation.
 * ============================================================ */

static void tq_ggml_vec_dot_generic(tq_type type, int n, float* result,
                                     const void* x, const float* y) {
    tq_dequantize_fn dfn = TQ_TRAITS[type].dequantize;
    if (!dfn) {
        *result = 0.0f;
        return;
    }

    float tmp[512];
    float* buf = (n <= 512) ? tmp : (float*)malloc((size_t)n * sizeof(float));
    if (!buf) { *result = 0.0f; return; }

    dfn(x, buf, n);

    float dot = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += buf[i] * y[i];
    }
    *result = dot;

    if (buf != tmp) free(buf);
}

#define TQ_GGML_VEC_DOT(NAME, TYPE)                                                \
    static void tq_ggml_vec_dot_##NAME(int n, float* s, const void* x, const float* y) { \
        tq_ggml_vec_dot_generic(TYPE, n, s, x, y);                                  \
    }

TQ_GGML_VEC_DOT(polar_3b,   TQ_TYPE_POLAR_3B)
TQ_GGML_VEC_DOT(polar_4b,   TQ_TYPE_POLAR_4B)
TQ_GGML_VEC_DOT(qjl_1b,     TQ_TYPE_QJL_1B)
TQ_GGML_VEC_DOT(turbo_3b,   TQ_TYPE_TURBO_3B)
TQ_GGML_VEC_DOT(turbo_4b,   TQ_TYPE_TURBO_4B)
TQ_GGML_VEC_DOT(uniform_4b, TQ_TYPE_UNIFORM_4B)
TQ_GGML_VEC_DOT(uniform_2b, TQ_TYPE_UNIFORM_2B)
TQ_GGML_VEC_DOT(mixed_4b8,     TQ_TYPE_MIXED_4B8)
TQ_GGML_VEC_DOT(turbo_kv_3b,  TQ_TYPE_TURBO_KV_3B)
TQ_GGML_VEC_DOT(turbo_kv_4b,  TQ_TYPE_TURBO_KV_4B)
TQ_GGML_VEC_DOT(turbo_kv_1b,  TQ_TYPE_TURBO_KV_1B)
TQ_GGML_VEC_DOT(turbo_kv_2b,  TQ_TYPE_TURBO_KV_2B)
TQ_GGML_VEC_DOT(uniform_3b,  TQ_TYPE_UNIFORM_3B)
TQ_GGML_VEC_DOT(turbo_kv_5b, TQ_TYPE_TURBO_KV_5B)
TQ_GGML_VEC_DOT(turbo_kv_4bo, TQ_TYPE_TURBO_KV_4BO)

/* ============================================================
 * GGML type trait table
 *
 * In a real llama.cpp integration this struct maps 1:1 to
 * ggml_type_traits_t / ggml_type_traits_cpu. We define a
 * compatible struct here so the integration can be tested
 * independently.
 * ============================================================ */

typedef void (*ggml_from_float_fn)(const float* src, void* dst, int64_t n);
typedef void (*ggml_to_float_fn)(const void* src, float* dst, int64_t n);
typedef void (*ggml_vec_dot_fn)(int n, float* s, const void* x, const float* y);

struct tq_ggml_type_trait {
    const char*        type_name;
    int                ggml_type_id;
    tq_type            tq_type_id;
    size_t             type_size;       /* bytes per block */
    size_t             block_size;      /* elements per block */
    float              bpe;             /* bits per element (with metadata) */
    ggml_from_float_fn from_float;
    ggml_to_float_fn   to_float;
    ggml_vec_dot_fn    vec_dot;
};

static const tq_ggml_type_trait TQ_GGML_TRAITS[GGML_TYPE_TQ_COUNT] = {
    {
        "tq_polar_3b", GGML_TYPE_TQ_POLAR_3B, TQ_TYPE_POLAR_3B,
        sizeof(block_tq_polar), TQ_BK, 4.5f,
        tq_ggml_from_float_polar_3b,
        tq_ggml_to_float_polar_3b,
        tq_ggml_vec_dot_polar_3b,
    },
    {
        "tq_polar_4b", GGML_TYPE_TQ_POLAR_4B, TQ_TYPE_POLAR_4B,
        sizeof(block_tq_polar), TQ_BK, 4.5f,
        tq_ggml_from_float_polar_4b,
        tq_ggml_to_float_polar_4b,
        tq_ggml_vec_dot_polar_4b,
    },
    {
        "tq_qjl_1b", GGML_TYPE_TQ_QJL_1B, TQ_TYPE_QJL_1B,
        sizeof(block_tq_qjl), TQ_BK_QJL, 1.25f,
        tq_ggml_from_float_qjl_1b,
        tq_ggml_to_float_qjl_1b,
        tq_ggml_vec_dot_qjl_1b,
    },
    {
        "tq_turbo_3b", GGML_TYPE_TQ_TURBO_3B, TQ_TYPE_TURBO_3B,
        sizeof(block_tq_turbo), TQ_BK, 5.75f,
        tq_ggml_from_float_turbo_3b,
        tq_ggml_to_float_turbo_3b,
        tq_ggml_vec_dot_turbo_3b,
    },
    {
        "tq_turbo_4b", GGML_TYPE_TQ_TURBO_4B, TQ_TYPE_TURBO_4B,
        sizeof(block_tq_turbo), TQ_BK, 5.75f,
        tq_ggml_from_float_turbo_4b,
        tq_ggml_to_float_turbo_4b,
        tq_ggml_vec_dot_turbo_4b,
    },
    {
        "tq_uniform_4b", GGML_TYPE_TQ_UNIFORM_4B, TQ_TYPE_UNIFORM_4B,
        sizeof(block_tq_uniform_4b), TQ_BK, 4.25f,
        tq_ggml_from_float_uniform_4b,
        tq_ggml_to_float_uniform_4b,
        tq_ggml_vec_dot_uniform_4b,
    },
    {
        "tq_uniform_2b", GGML_TYPE_TQ_UNIFORM_2B, TQ_TYPE_UNIFORM_2B,
        sizeof(block_tq_uniform_2b), TQ_BK, 3.0f,
        tq_ggml_from_float_uniform_2b,
        tq_ggml_to_float_uniform_2b,
        tq_ggml_vec_dot_uniform_2b,
    },
    {
        "tq_mixed_4b8", GGML_TYPE_TQ_MIXED_4B8, TQ_TYPE_MIXED_4B8,
        sizeof(block_tq_mixed_4b8), TQ_BK, 5.0f,
        tq_ggml_from_float_mixed_4b8,
        tq_ggml_to_float_mixed_4b8,
        tq_ggml_vec_dot_mixed_4b8,
    },
    {
        "tq_turbo_kv_3b", GGML_TYPE_TQ_TURBO_KV_3B, TQ_TYPE_TURBO_KV_3B,
        sizeof(block_tq_turbo_kv_3b), TQ_BK,
        (float)sizeof(block_tq_turbo_kv_3b) * 8.0f / TQ_BK,
        tq_ggml_from_float_turbo_kv_3b,
        tq_ggml_to_float_turbo_kv_3b,
        tq_ggml_vec_dot_turbo_kv_3b,
    },
    {
        "tq_turbo_kv_4b", GGML_TYPE_TQ_TURBO_KV_4B, TQ_TYPE_TURBO_KV_4B,
        sizeof(block_tq_turbo_kv_4b), TQ_BK,
        (float)sizeof(block_tq_turbo_kv_4b) * 8.0f / TQ_BK,
        tq_ggml_from_float_turbo_kv_4b,
        tq_ggml_to_float_turbo_kv_4b,
        tq_ggml_vec_dot_turbo_kv_4b,
    },
    {
        "tq_turbo_kv_1b", GGML_TYPE_TQ_TURBO_KV_1B, TQ_TYPE_TURBO_KV_1B,
        sizeof(block_tq_turbo_kv_1b), TQ_BK,
        (float)sizeof(block_tq_turbo_kv_1b) * 8.0f / TQ_BK,
        tq_ggml_from_float_turbo_kv_1b,
        tq_ggml_to_float_turbo_kv_1b,
        tq_ggml_vec_dot_turbo_kv_1b,
    },
    {
        "tq_turbo_kv_2b", GGML_TYPE_TQ_TURBO_KV_2B, TQ_TYPE_TURBO_KV_2B,
        sizeof(block_tq_turbo_kv_2b), TQ_BK,
        (float)sizeof(block_tq_turbo_kv_2b) * 8.0f / TQ_BK,
        tq_ggml_from_float_turbo_kv_2b,
        tq_ggml_to_float_turbo_kv_2b,
        tq_ggml_vec_dot_turbo_kv_2b,
    },
    {
        "tq_uniform_3b", GGML_TYPE_TQ_UNIFORM_3B, TQ_TYPE_UNIFORM_3B,
        sizeof(block_tq_uniform_3b), TQ_BK,
        (float)sizeof(block_tq_uniform_3b) * 8.0f / TQ_BK,
        tq_ggml_from_float_uniform_3b,
        tq_ggml_to_float_uniform_3b,
        tq_ggml_vec_dot_uniform_3b,
    },
    {
        "tq_turbo_kv_5b", GGML_TYPE_TQ_TURBO_KV_5B, TQ_TYPE_TURBO_KV_5B,
        sizeof(block_tq_turbo_kv_5b), TQ_BK,
        (float)sizeof(block_tq_turbo_kv_5b) * 8.0f / TQ_BK,
        tq_ggml_from_float_turbo_kv_5b,
        tq_ggml_to_float_turbo_kv_5b,
        tq_ggml_vec_dot_turbo_kv_5b,
    },
    {
        "tq_turbo_kv_4bo", GGML_TYPE_TQ_TURBO_KV_4BO, TQ_TYPE_TURBO_KV_4BO,
        sizeof(block_tq_turbo_kv_4bo), TQ_BK,
        (float)sizeof(block_tq_turbo_kv_4bo) * 8.0f / TQ_BK,
        tq_ggml_from_float_turbo_kv_4bo,
        tq_ggml_to_float_turbo_kv_4bo,
        tq_ggml_vec_dot_turbo_kv_4bo,
    },
};

#define TQ_GGML_NUM_TYPES (sizeof(TQ_GGML_TRAITS) / sizeof(TQ_GGML_TRAITS[0]))

/* ============================================================
 * Registration
 * ============================================================ */

static int g_tq_registered = 0;

tq_status tq_ggml_register_types(void) {
    if (g_tq_registered) return TQ_OK;

    /*
     * In a real llama.cpp integration, this function would iterate
     * TQ_GGML_TRAITS and call:
     *
     *   ggml_register_custom_type(trait->ggml_type_id,
     *       trait->type_name,
     *       trait->type_size,
     *       trait->block_size,
     *       trait->from_float,
     *       trait->to_float,
     *       trait->vec_dot);
     *
     * Since ggml_register_custom_type() is not available without
     * linking against llama.cpp, we validate the trait table here
     * and mark registration as complete.
     */

    for (size_t i = 0; i < TQ_GGML_NUM_TYPES; i++) {
        const tq_ggml_type_trait* t = &TQ_GGML_TRAITS[i];
        if (!t->type_name || !t->from_float || !t->to_float || !t->vec_dot) {
            fprintf(stderr, "tq_ggml_register_types: trait %zu incomplete\n", i);
            return TQ_ERR_NOT_IMPL;
        }
        if (t->type_size == 0 || t->block_size == 0) {
            fprintf(stderr, "tq_ggml_register_types: trait %zu has zero size\n", i);
            return TQ_ERR_INVALID_DIM;
        }
    }

    g_tq_registered = 1;
    fprintf(stderr, "[quant.cpp] Registered %d GGML types (IDs %d..%d)\n",
            (int)TQ_GGML_NUM_TYPES, GGML_TYPE_TQ_BASE,
            GGML_TYPE_TQ_BASE + (int)TQ_GGML_NUM_TYPES - 1);

    return TQ_OK;
}

/* ============================================================
 * CLI option parsing helper
 *
 * Parses --kv-cache-type argument and returns the corresponding
 * quant.cpp type. Returns TQ_TYPE_COUNT on unrecognized input.
 * ============================================================ */

tq_type tq_parse_kv_cache_type(const char* arg) {
    if (!arg) return TQ_TYPE_COUNT;

    /* Try the canonical name first (via tq_type_from_name) */
    tq_type result = tq_type_from_name(arg);
    if (result != TQ_TYPE_COUNT) return result;

    /* Then try common short aliases */
    struct { const char* name; tq_type type; } map[] = {
        { "turbo3",       TQ_TYPE_TURBO_3B   },
        { "turbo_3b",     TQ_TYPE_TURBO_3B   },
        { "tq-turbo-3b",  TQ_TYPE_TURBO_3B   },
        { "turbo4",       TQ_TYPE_TURBO_4B   },
        { "turbo_4b",     TQ_TYPE_TURBO_4B   },
        { "tq-turbo-4b",  TQ_TYPE_TURBO_4B   },
        { "polar3",       TQ_TYPE_POLAR_3B   },
        { "polar_3b",     TQ_TYPE_POLAR_3B   },
        { "tq-polar-3b",  TQ_TYPE_POLAR_3B   },
        { "polar4",       TQ_TYPE_POLAR_4B   },
        { "polar_4b",     TQ_TYPE_POLAR_4B   },
        { "tq-polar-4b",  TQ_TYPE_POLAR_4B   },
        { "qjl1",         TQ_TYPE_QJL_1B     },
        { "qjl_1b",       TQ_TYPE_QJL_1B     },
        { "tq-qjl-1b",    TQ_TYPE_QJL_1B     },
        { "uniform4",     TQ_TYPE_UNIFORM_4B },
        { "uniform_4b",   TQ_TYPE_UNIFORM_4B },
        { "tq-uniform-4b",TQ_TYPE_UNIFORM_4B },
        { "uniform2",        TQ_TYPE_UNIFORM_2B },
        { "uniform_2b",     TQ_TYPE_UNIFORM_2B },
        { "tq-uniform-2b",  TQ_TYPE_UNIFORM_2B },
        { "turbo_kv_3b",    TQ_TYPE_TURBO_KV_3B },
        { "tq-turbo-kv-3b", TQ_TYPE_TURBO_KV_3B },
        { "turbokv3",       TQ_TYPE_TURBO_KV_3B },
        { "turbo_kv_4b",    TQ_TYPE_TURBO_KV_4B },
        { "turbo_kv_5b",    TQ_TYPE_TURBO_KV_5B },
        { "turbo_kv_4bo",   TQ_TYPE_TURBO_KV_4BO },
        { "tq-turbo-kv-4b", TQ_TYPE_TURBO_KV_4B },
        { "turbokv4",       TQ_TYPE_TURBO_KV_4B },
        { "turbo_kv_1b",    TQ_TYPE_TURBO_KV_1B },
        { "tq-turbo-kv-1b", TQ_TYPE_TURBO_KV_1B },
        { "turbokv1",       TQ_TYPE_TURBO_KV_1B },
        { "turbo_kv_2b",    TQ_TYPE_TURBO_KV_2B },
        { "tq-turbo-kv-2b", TQ_TYPE_TURBO_KV_2B },
        { "turbokv2",       TQ_TYPE_TURBO_KV_2B },
    };

    for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); i++) {
        if (strcmp(arg, map[i].name) == 0) {
            return map[i].type;
        }
    }

    return TQ_TYPE_COUNT;
}

/* Print all available quant.cpp KV cache types */
void tq_print_kv_cache_types(void) {
    fprintf(stderr, "Available quant.cpp KV cache types:\n");
    for (int i = 0; i < TQ_TYPE_COUNT; i++) {
        float bpe = tq_type_bpe((tq_type)i);
        fprintf(stderr, "  %-14s  %.1f bpe  %.1fx compression\n",
                tq_type_name((tq_type)i), bpe,
                (bpe > 0.0f) ? 32.0f / bpe : 0.0f);
    }
}

/* ============================================================
 * KV cache integration helpers
 *
 * These wrap quant.cpp's context-based API for use within
 * llama.cpp's KV cache management code.
 * ============================================================ */

/**
 * Create a quant.cpp context suitable for llama.cpp integration.
 * Caller must free with tq_free().
 */
tq_context_t* tq_llamacpp_create_context(void) {
    tq_context_t* ctx = nullptr;
    tq_status status = tq_init(&ctx, TQ_BACKEND_AUTO);
    if (status != TQ_OK) {
        fprintf(stderr, "[quant.cpp] Failed to create context: %s\n",
                tq_status_string(status));
        return nullptr;
    }
    return ctx;
}

/**
 * Quantize a batch of key vectors into the KV cache.
 *
 * @param ctx        quant.cpp context
 * @param keys       Input FP32 keys [n_tokens x head_dim]
 * @param n_tokens   Number of tokens to quantize
 * @param head_dim   Dimension per attention head
 * @param type       quant.cpp quantization type
 * @param out        Output buffer (pre-allocated by llama.cpp cache manager)
 * @param out_bytes  Size of output buffer
 * @return           TQ_OK on success
 */
tq_status tq_llamacpp_quantize_keys(tq_context_t* ctx,
                                     const float* keys,
                                     int n_tokens, int head_dim,
                                     tq_type type,
                                     void* out, size_t out_bytes) {
    return tq_quantize_keys(ctx, keys, n_tokens, head_dim, type, out, out_bytes);
}

/**
 * Quantize a batch of value vectors into the KV cache.
 *
 * @param ctx        quant.cpp context
 * @param values     Input FP32 values [n_tokens x head_dim]
 * @param n_tokens   Number of tokens
 * @param head_dim   Dimension per head
 * @param bits       Quantization bits (2 or 4)
 * @param out        Output buffer
 * @param out_bytes  Buffer size
 * @return           TQ_OK on success
 */
tq_status tq_llamacpp_quantize_values(tq_context_t* ctx,
                                       const float* values,
                                       int n_tokens, int head_dim,
                                       int bits,
                                       void* out, size_t out_bytes) {
    return tq_quantize_values(ctx, values, n_tokens, head_dim, bits, out, out_bytes);
}

/**
 * Compute attention scores from quantized KV cache.
 *
 * @param ctx       quant.cpp context
 * @param query     Query vector [head_dim]
 * @param kv_cache  Quantized key cache
 * @param seq_len   Sequence length (cached tokens)
 * @param head_dim  Head dimension
 * @param type      Quantization type
 * @param scores    Output scores [seq_len]
 * @return          TQ_OK on success
 */
tq_status tq_llamacpp_attention(tq_context_t* ctx,
                                 const float* query,
                                 const void* kv_cache,
                                 int seq_len, int head_dim,
                                 tq_type type,
                                 float* scores) {
    return tq_attention(ctx, query, kv_cache, seq_len, head_dim, type, scores);
}

/**
 * Compute the quantized KV cache memory per token for a given type.
 * Useful for llama.cpp's memory estimation in --kv-cache-type mode.
 *
 * @param head_dim   Dimension per head
 * @param key_type   quant.cpp key quantization type
 * @param value_bits Value quantization bits (2 or 4), 0 for no value quant
 * @return           Bytes per token per head (key + value)
 */
size_t tq_llamacpp_bytes_per_token(int head_dim, tq_type key_type, int value_bits) {
    size_t key_bytes = tq_quantize_keys_size(1, head_dim, key_type);
    size_t val_bytes = 0;
    if (value_bits > 0) {
        val_bytes = tq_quantize_values_size(1, head_dim, value_bits);
    } else {
        /* Default: store values as FP16 (2 bytes per element) */
        val_bytes = (size_t)head_dim * 2;
    }
    return key_bytes + val_bytes;
}

/**
 * Print a summary of memory savings for a given configuration.
 * Useful for llama.cpp's startup info output.
 */
void tq_llamacpp_print_config(tq_type key_type, int value_bits,
                               int n_heads, int head_dim, int max_seq_len) {
    size_t fp16_per_token = (size_t)head_dim * 2 * 2;  /* key + value, FP16 */
    size_t tq_per_token = tq_llamacpp_bytes_per_token(head_dim, key_type, value_bits);
    double ratio = (tq_per_token > 0) ?
        (double)fp16_per_token / (double)tq_per_token : 0.0;

    size_t fp16_total = fp16_per_token * (size_t)n_heads * (size_t)max_seq_len;
    size_t tq_total   = tq_per_token   * (size_t)n_heads * (size_t)max_seq_len;

    fprintf(stderr, "[quant.cpp] KV cache config:\n");
    fprintf(stderr, "  Key type:        %s\n", tq_type_name(key_type));
    fprintf(stderr, "  Value bits:      %d\n", value_bits > 0 ? value_bits : 16);
    fprintf(stderr, "  Heads:           %d x %d\n", n_heads, head_dim);
    fprintf(stderr, "  Max seq length:  %d\n", max_seq_len);
    fprintf(stderr, "  FP16 KV memory:  %.1f MB\n", (double)fp16_total / (1024.0 * 1024.0));
    fprintf(stderr, "  TQ KV memory:    %.1f MB\n", (double)tq_total / (1024.0 * 1024.0));
    fprintf(stderr, "  Compression:     %.2fx\n", ratio);
}
