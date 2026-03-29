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
    int delta_n_heads;       /* number of DeltaNet heads (e.g., 16) */
    int delta_key_head_dim;  /* key head dim (e.g., 128) */
    int delta_value_head_dim;/* value head dim (e.g., 128) */
    int delta_conv_width;    /* conv1d kernel width (e.g., 4) */
    float partial_rotary_factor; /* fraction of head_dim that uses RoPE (e.g., 0.25) */

    /* QK-norm for self_attn (Qwen3.5 style) */
    int use_qk_norm;         /* 1 if q_norm/k_norm weights present */
    int attn_output_gate;    /* 1 if q_proj includes output gate (doubled q_proj output) */
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

    /* SwiGLU FFN weights (present on ALL layers) */
    float* w_gate;        /* [intermediate_dim, hidden_dim] */
    float* w_up;          /* [intermediate_dim, hidden_dim] */
    float* w_down;        /* [hidden_dim, intermediate_dim] */

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
    float* token_embedding;   /* [vocab_size, hidden_dim] */

    /* Per-layer weights */
    tq_layer_weights_t* layers;

    /* Output */
    float* output_norm;       /* [hidden_dim] */
    float* output_weight;     /* [vocab_size, hidden_dim] (may be tied to embedding) */

    /* Hybrid architecture support (e.g., Qwen3.5 with DeltaNet layers) */
    int n_attn_layers;        /* number of layers with standard self_attn */
    int* attn_layer_indices;  /* which layer indices have self_attn [n_attn_layers] */

    /* Memory management */
    void* _mmap_data;
    size_t _mmap_size;
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
    float* value_cache;  /* [n_layers, max_seq_len, n_kv_heads * head_dim] */
    tq_type kv_quant_type; /* quantization type for KV attention */
    size_t kv_cache_size;

    /* DeltaNet recurrent state */
    float* delta_state;  /* [n_layers, delta_n_heads, key_head_dim, value_head_dim] */
    float* conv_state;   /* [n_layers, qkv_dim, conv_width-1] */

    /* DeltaNet workspace buffers */
    float* delta_qkv;    /* [qkv_dim] workspace for QKV projection */
    float* delta_z;      /* [z_dim] workspace for Z gate */
    float* delta_ab;     /* [delta_n_heads * 2] workspace for a,b projections */
    float* delta_out;    /* [z_dim] workspace for output */

    /* Quantization workspace */
    void* quant_key_buf;    /* workspace for quantized keys */
    float* quant_score_buf; /* workspace for quantized attention scores */
} tq_state_t;

/* ============================================================
 * Generation config
 * ============================================================ */
typedef struct {
    float temperature;
    float top_p;
    int max_tokens;
    tq_type kv_type;     /* KV cache quantization type */
    int n_threads;
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
 * API
 * ============================================================ */

/* Model loading */
tq_model_t* tq_load_model(const char* path);
void tq_free_model(tq_model_t* model);

/* State management */
tq_state_t* tq_create_state(const tq_model_config_t* config, tq_type kv_type);
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
void tq_free_tokenizer(tq_tokenizer_t* tok);
int tq_encode(const tq_tokenizer_t* tok, const char* text,
              int* tokens, int max_tokens, int add_bos);
const char* tq_decode(const tq_tokenizer_t* tok, int prev_token, int token);

/* Tensor operations (exported for testing/reuse) */
void tq_matmul(float* out, const float* x, const float* w, int n, int d);
void tq_rmsnorm(float* out, const float* x, const float* weight, int n, float eps);
void tq_rope(float* q, float* k, int pos, int head_dim,
             int n_heads, int n_kv_heads, float freq_base);
void tq_silu(float* x, int n);
void tq_softmax(float* x, int n);
void tq_add(float* out, const float* a, const float* b, int n);
void tq_mul(float* out, const float* a, const float* b, int n);

/* Default generation config */
tq_gen_config_t tq_default_gen_config(void);

/* Thread control for matmul parallelism */
void tq_set_threads(int n_threads);
int tq_get_threads(void);

#ifdef __cplusplus
}
#endif
#endif /* TQ_ENGINE_H */
