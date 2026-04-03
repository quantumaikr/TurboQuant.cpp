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

/* Cross-platform math constants (some platforms lack M_PI) */
#ifndef TQ_PI
#define TQ_PI   3.14159265358979323846f
#endif
#ifndef TQ_PI_2
#define TQ_PI_2 1.5707963267948966f
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

#endif /* TQ_TYPES_H */
