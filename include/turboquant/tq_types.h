#ifndef TQ_TYPES_H
#define TQ_TYPES_H

#include <stdint.h>
#include <stddef.h>

/* Cross-language static assert: works in both C11 and C++11/17 */
#ifdef __cplusplus
#define TQ_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define TQ_STATIC_ASSERT(cond, msg) TQ_STATIC_ASSERT(cond, msg)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Constants
 * ============================================================ */

#define TQ_BK          128   /* Default block size (elements per block) */
#define TQ_BK_QJL      256   /* QJL block size */
#define TQ_SKETCH_DIM  256   /* QJL sketch dimension */
#define TQ_OUTLIERS    4     /* QJL outlier count */
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
    TQ_TYPE_COUNT     = 7
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

typedef struct {
    uint16_t scale;
    uint16_t zero_point;
    uint8_t  qs[TQ_BK / 4];         /* 2-bit: 4 values/byte, LSB-first */
} block_tq_uniform_2b;

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

/* Global traits table — initialized by tq_init() */
extern const tq_type_traits_t TQ_TRAITS[TQ_TYPE_COUNT];

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

#ifdef __cplusplus
}
#endif

/* ============================================================
 * Block size verification (compile-time, C/C++ compatible)
 * Uses negative-size array trick for universal compatibility.
 * ============================================================ */
#define TQ_CHECK_SIZE(type, expected) \
    typedef char tq_check_##type[(sizeof(type) == (expected)) ? 1 : -1]

TQ_CHECK_SIZE(block_tq_polar,      8 + TQ_BK / 2);
TQ_CHECK_SIZE(block_tq_qjl,        4 + TQ_SKETCH_DIM / 8 + TQ_OUTLIERS);
TQ_CHECK_SIZE(block_tq_uniform_4b, 4 + TQ_BK / 2);
TQ_CHECK_SIZE(block_tq_uniform_2b, 4 + TQ_BK / 4);

#endif /* TQ_TYPES_H */
