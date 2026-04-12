#ifndef TQ_ENGINE_H
#define TQ_ENGINE_H

#include "tq_types.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

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

    /* Phi-3 LongRoPE parameters */
    int rope_orig_ctx_len;             /* original context length (e.g., 4096) */
    float rope_attn_factor;            /* attention magnitude scaling */
    const float* rope_factors_short;   /* [head_dim/2] for short context */
    const float* rope_factors_long;    /* [head_dim/2] for long context */

    /* Phi-3 fused-tensor flags — drive state buffer sizing */
    int has_fused_qkv;                 /* any layer has gguf_w_qkv */
    int has_fused_up_gate;             /* any layer has gguf_w_up_gate */
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
    /* Phi-3 fused projections — one matmul + memcpy split */
    const void* gguf_w_qkv;     int gguf_w_qkv_type;     /* [hidden, q+k+v] fused QKV */
    const void* gguf_w_up_gate; int gguf_w_up_gate_type;  /* [hidden, 2*inter] fused gate||up */

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

    /* GGUF output weight — keep quantized for fused dot output projection */
    const void* output_gguf;  /* mmap'd quantized weight, or NULL */
    int output_gguf_type;     /* tq_ggml_dtype */

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
    /* KV cache persistence (Document-Level RAG: read once, query forever) */
    const char* save_kv_path; /* save KV cache after generation (NULL = don't save) */
    const char* load_kv_path; /* load pre-computed KV cache before generation (NULL = normal) */
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

#ifdef __cplusplus
}
#endif
#endif /* TQ_ENGINE_H */
