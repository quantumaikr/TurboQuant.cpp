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
    int n_heads;         /* query heads */
    int n_kv_heads;      /* KV heads (GQA) */
    int head_dim;
    int vocab_size;
    int max_seq_len;
    float rope_freq_base;
    float rms_norm_eps;
} tq_model_config_t;

/* ============================================================
 * Model weights (in memory)
 * ============================================================ */
typedef struct {
    float* attn_norm;     /* [hidden_dim] */
    float* ffn_norm;      /* [hidden_dim] */
    float* wq;            /* [n_heads * head_dim, hidden_dim] */
    float* wk;            /* [n_kv_heads * head_dim, hidden_dim] */
    float* wv;            /* [n_kv_heads * head_dim, hidden_dim] */
    float* wo;            /* [hidden_dim, n_heads * head_dim] */
    float* w_gate;        /* [intermediate_dim, hidden_dim] */
    float* w_up;          /* [intermediate_dim, hidden_dim] */
    float* w_down;        /* [hidden_dim, intermediate_dim] */
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

    /* Memory management */
    void* _mmap_data;
    size_t _mmap_size;
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

    /* KV cache — FP32 for values, quantized for keys via TurboQuant */
    float* key_cache;    /* [n_layers, max_seq_len, n_kv_heads * head_dim] */
    float* value_cache;  /* [n_layers, max_seq_len, n_kv_heads * head_dim] */
    tq_type kv_quant_type; /* quantization type for KV attention */
    size_t kv_cache_size;

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
    char** vocab;        /* token strings */
    float* scores;       /* BPE merge scores */
    int vocab_size;
    int max_token_len;
    /* Sorted vocab for encoding */
    int* sorted_indices;
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

#ifdef __cplusplus
}
#endif
#endif /* TQ_ENGINE_H */
