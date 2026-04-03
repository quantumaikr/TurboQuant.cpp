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

#ifndef TQ_GGUF_H
#define TQ_GGUF_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif
#endif /* TQ_GGUF_H */
