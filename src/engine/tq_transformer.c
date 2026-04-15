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

#include "turboquant/tq_engine.h"
#include "turboquant/turboquant.h"
#include "turboquant/tq_gguf.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#ifdef __APPLE__
#include <unistd.h> /* getpagesize, posix_memalign */
#endif
#ifdef _WIN32
#include <windows.h> /* QueryPerformanceCounter, LARGE_INTEGER */
#endif

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
#ifdef _WIN32
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart * 1e9;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
#endif
}

/* Usage: double _tp; TQ_PROF_START(_tp); ... TQ_PROF_STOP(_tp, field); */
#define TQ_PROF_START(var) do { var = g_tq_profile_enabled ? tq_now_ns() : 0; } while(0)
#define TQ_PROF_STOP(var, field) do { if (g_tq_profile_enabled) g_profile.field += tq_now_ns() - var; } while(0)

/* ============================================================
 * FP16 helpers (IEEE 754 half-precision, storage only)
 * ============================================================ */

static uint16_t f32_to_fp16(float v) {
    union { float f; uint32_t u; } bits;
    bits.f = v;
    uint32_t sign = (bits.u >> 16) & 0x8000;
    int32_t  exp  = ((bits.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits.u >> 13) & 0x03FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static float fp16_to_f32(uint16_t h) {
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
        dst[i] = f32_to_fp16(src[i]);
    }
#else
    for (int i = 0; i < n; i++) {
        dst[i] = f32_to_fp16(src[i]);
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
    /* Phi-3 fused QKV: xb2 is used as temp buffer for [Q|K|V] output */
    if (config->has_fused_qkv) {
        int fused_qkv_dim = q_dim + 2 * kv_dim;
        if (fused_qkv_dim > max_dim) max_dim = fused_qkv_dim;
    }
    /* Phi-3 fused gate||up: hb must hold 2*inter for the fused matmul */
    int hb_dim = inter_dim;
    if (config->has_fused_up_gate) hb_dim = 2 * inter_dim;

    s->x      = (float*)calloc((size_t)dim, sizeof(float));
    s->xb     = (float*)calloc((size_t)max_dim, sizeof(float));
    s->xb2    = (float*)calloc((size_t)max_dim, sizeof(float));
    s->q      = (float*)calloc((size_t)max_q_dim, sizeof(float));
    s->k      = (float*)calloc((size_t)max_kv_dim, sizeof(float));
    s->v      = (float*)calloc((size_t)max_kv_dim, sizeof(float));
    s->att    = (float*)calloc((size_t)n_heads * max_seq, sizeof(float));
    s->hb     = (float*)calloc((size_t)hb_dim, sizeof(float));
    s->hb2    = (float*)calloc((size_t)inter_dim, sizeof(float));
    s->logits = (float*)calloc((size_t)config->vocab_size, sizeof(float));

    /* KV cache for self_attn layers — use max_kv_dim for hybrid attention compatibility.
     * Page-aligned allocation for Metal GPU zero-copy (newBufferWithBytesNoCopy). */
    size_t kv_layer_size = (size_t)max_seq * max_kv_dim;
    size_t kv_total_bytes = (size_t)n_layers * kv_layer_size * sizeof(float);
#ifdef __APPLE__
    {
        void* kv_ptr = NULL;
        size_t page_sz = (size_t)getpagesize();
        size_t aligned_sz = (kv_total_bytes + page_sz - 1) & ~(page_sz - 1);
        if (posix_memalign(&kv_ptr, page_sz, aligned_sz) == 0) {
            memset(kv_ptr, 0, aligned_sz);
            s->key_cache = (float*)kv_ptr;
        } else {
            s->key_cache = (float*)calloc(1, kv_total_bytes);
        }
    }
#else
    s->key_cache   = (float*)calloc(1, kv_total_bytes);
#endif
    if (!s->key_cache) { free(s); return NULL; }

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
        /* Page-aligned for Metal GPU zero-copy */
#ifdef __APPLE__
        {
            void* vc_ptr = NULL;
            size_t page_sz = (size_t)getpagesize();
            size_t vc_bytes = (size_t)n_layers * kv_layer_size * sizeof(float);
            size_t aligned_sz = (vc_bytes + page_sz - 1) & ~(page_sz - 1);
            if (posix_memalign(&vc_ptr, page_sz, aligned_sz) == 0) {
                memset(vc_ptr, 0, aligned_sz);
                s->value_cache = (float*)vc_ptr;
            } else {
                s->value_cache = (float*)calloc(1, vc_bytes);
            }
        }
#else
        s->value_cache = (float*)calloc((size_t)n_layers * kv_layer_size, sizeof(float));
#endif
        if (!s->value_cache) { free(s->key_cache); free(s); return NULL; }
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

    /* Quantization workspace — use MAX head_dim for hybrid attention (Gemma 4).
     * Sliding layers have head_dim=256, full layers have head_dim=512.
     * Quantized cache must accommodate the larger dimension. */
    size_t block_size = tq_type_block_size(kv_type);
    size_t type_size  = tq_type_type_size(kv_type);
    if (block_size == 0) block_size = TQ_BK;
    if (type_size == 0) type_size = sizeof(block_tq_uniform_4b);
    int max_head_dim = config->head_dim;
    if (config->full_head_dim > max_head_dim) max_head_dim = config->full_head_dim;
    size_t n_blocks_per_head = ((size_t)max_head_dim + block_size - 1) / block_size;
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
    /* Use max kv_heads for position stride (hybrid: sliding=8, full=2 but larger heads) */
    int max_kv_heads = config->n_kv_heads;
    if (config->full_n_kv_heads > max_kv_heads) max_kv_heads = config->full_n_kv_heads;
    /* Position stride = max(sliding_kv_dim, full_kv_dim) in quantized blocks */
    size_t quant_pos_stride = s->quant_head_stride * (size_t)max_kv_heads;
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
    int has_fused_qkv_layer = (layer->gguf_w_qkv != NULL);
    if (has_q2 || has_q4) {
        tq_quantize_row_q8(s->xb, s->xb_q8, s->xb_q8s, dim);
    }

    /* Note: int8×int8 Q8 path was tested but Apple Silicon's FP FMA pipeline
     * is wider than integer multiply, so float fused dot is already optimal.
     * Pre-quantized int8 path kept in tq_gguf_quants.c for x86/AVX-512 VNNI. */

    /* QKV projections (timed as matmul) */
    TQ_PROF_START(_tp);
    /* When attn_output_gate is enabled, wq has shape [2*n_heads*head_dim, dim]
     * and outputs [Q, gate_q] concatenated. We project into xb2 as temp.
     *
     * Q+K+V GPU dispatches are batched into one command buffer by the
     * layer-level batch scope in tq_forward(). */

    float* gate_q = NULL;
    if (has_fused_qkv_layer) {
        /* Phi-3 fused QKV: one matmul produces [Q | K | V].
         * Force CPU path — Metal has a buffer sizing bug with the large
         * output dim (total_out = 3*dim for Phi-3.5, much larger than
         * typical single-projection matmuls). */
        extern void tq_matmul_gguf_cpu(float*, const float*, const void*,
                                        int, int, int);
        int q_out  = n_heads * head_dim;
        int kv_out = kv_dim;
        int total_out = q_out + 2 * kv_out;
        tq_matmul_gguf_cpu(s->xb2, s->xb,
                           layer->gguf_w_qkv, layer->gguf_w_qkv_type,
                           total_out, dim);
        memcpy(s->q, s->xb2,                       (size_t)q_out  * sizeof(float));
        memcpy(s->k, s->xb2 + q_out,               (size_t)kv_out * sizeof(float));
        memcpy(s->v, s->xb2 + q_out + kv_out,      (size_t)kv_out * sizeof(float));
    } else if (c->attn_output_gate) {
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
        /* Note: Metal GPU QKV batch was benchmarked but is SLOWER than CPU NEON
         * for batch-1 inference on Apple Silicon unified memory (5.4 vs 17 tok/s).
         * GPU wins only for batch inference (multiple tokens). Keeping CPU path. */
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
    /* Apply Q bias (Qwen2/2.5/3) */
    if (layer->q_bias) {
        int q_dim = n_heads * head_dim;
        for (int i = 0; i < q_dim; i++) s->q[i] += layer->q_bias[i];
    }

    /* Check if this is a KV-shared layer (reuse KV cache from source layer).
     * KV-shared layers DO have K/V weights in GGUF but we skip them, matching
     * llama.cpp's has_kv(il) which returns false for l >= n_layer - n_kv_shared_layers. */
    int has_v_weights = (layer->wv_q2 || layer->wv_q4 || layer->wv_q8 ||
                         layer->gguf_wv || layer->wv);
    int is_kv_shared = (c->n_kv_shared_layers > 0 && model->kv_source_layer &&
                         l >= c->n_layers - c->n_kv_shared_layers);

    if (!has_fused_qkv_layer && !is_kv_shared) {
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
    }

    /* Apply K/V biases (Qwen2/2.5/3) */
    if (!is_kv_shared) {
        if (layer->k_bias) {
            for (int i = 0; i < kv_dim; i++) s->k[i] += layer->k_bias[i];
        }
        if (layer->v_bias) {
            for (int i = 0; i < kv_dim; i++) s->v[i] += layer->v_bias[i];
        }
    }

    /* Flush batched Q+K+V GPU dispatches before CPU-side RoPE/attention */
    if (has_gguf) tq_metal_batch_flush_if_available();
    /* (int8 preq cleared — path disabled on Apple Silicon, see note above) */
    TQ_PROF_STOP(_tp, matmul_ns);

    /* Gemma 4: save pre-QK-norm keys for quantized cache (better distribution).
     * QK-norm compresses keys to unit sphere (rms≈0.06), destroying quantization quality.
     * Pre-norm keys have rms≈1.0 which quantizes well (cosine>0.99).
     * During attention, we apply QK-norm to dequantized keys on-the-fly. */
    float pre_norm_keys[4096]; /* max kv_dim for Gemma 4 */
    /* Gemma 4 QK-normed keys: 4-bit quantization gives cosine=0.62 (unusable).
     * Both pre-norm and post-norm keys quantize poorly due to extreme sparsity.
     * Solution: skip key quantization for QK-normed models → use FP32 keys + Q4 values.
     * This gives 2x V memory savings while preserving key precision. */
    int save_pre_norm_keys = 0; /* disabled — see above */

    /* Apply QK-norm if present (per-head RMSNorm).
     * For KV-shared layers: only Q-norm is applied (no K/V to normalize). */
    if (layer->q_norm) {
        for (int h = 0; h < n_heads; h++) {
            tq_rmsnorm(s->q + h * head_dim, s->q + h * head_dim,
                       layer->q_norm, head_dim, c->rms_norm_eps);
        }
    }
    if (layer->k_norm && !is_kv_shared) {
        for (int h = 0; h < n_kv_heads; h++) {
            tq_rmsnorm(s->k + h * head_dim, s->k + h * head_dim,
                       layer->k_norm, head_dim, c->rms_norm_eps);
        }
    }

    /* Gemma 4: V normalization (RMS norm without learned weights).
     * Reference: refs/llama.cpp/src/models/gemma4-iswa.cpp line 82 */
    if (c->is_gemma4 && has_v_weights && !is_kv_shared) {
        for (int h = 0; h < n_kv_heads; h++) {
            float* vh = s->v + h * head_dim;
            float ss = 0.0f;
            for (int i = 0; i < head_dim; i++) ss += vh[i] * vh[i];
            ss = 1.0f / sqrtf(ss / (float)head_dim + c->rms_norm_eps);
            for (int i = 0; i < head_dim; i++) vh[i] *= ss;
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

        /* Gemma 4 / STEP35 uses NeoX-style RoPE: pairs are (q[i], q[i+half])
         * where half = head_dim/2. Other models use interleaved: (q[2i], q[2i+1]).
         * The GGUF converter generates rope_freqs assuming NeoX layout. */
        int neox = c->is_gemma4;
        int half = head_dim / 2;

        for (int h = 0; h < n_heads; h++) {
            float* qh = s->q + h * head_dim;
            for (int i = 0; i < rope_pairs; i++) {
                float base_freq = 1.0f / powf(rope_base, 2.0f * i / (float)rope_n_dims);
                float freq = base_freq / model->rope_freqs[i];
                float theta = pos * freq;
                float cos_t = cosf(theta);
                float sin_t = sinf(theta);
                int a = neox ? i        : 2 * i;
                int b = neox ? i + half  : 2 * i + 1;
                float q0 = qh[a];
                float q1 = qh[b];
                qh[a] = q0 * cos_t - q1 * sin_t;
                qh[b] = q0 * sin_t + q1 * cos_t;
            }
            /* Pairs beyond rope_pairs are left unrotated (pass-through) */
        }
        if (!is_kv_shared) {
            for (int h = 0; h < n_kv_heads; h++) {
                float* kh = s->k + h * head_dim;
                for (int i = 0; i < rope_pairs; i++) {
                    float base_freq = 1.0f / powf(rope_base, 2.0f * i / (float)rope_n_dims);
                    float freq = base_freq / model->rope_freqs[i];
                    float theta = pos * freq;
                    float cos_t = cosf(theta);
                    float sin_t = sinf(theta);
                    int a = neox ? i        : 2 * i;
                    int b = neox ? i + half  : 2 * i + 1;
                    float k0 = kh[a];
                    float k1 = kh[b];
                    kh[a] = k0 * cos_t - k1 * sin_t;
                    kh[b] = k0 * sin_t + k1 * cos_t;
                }
            }
        }
    } else {
        /* Full RoPE — for Gemma3, use different freq base for sliding vs global layers */
        float rope_base = c->rope_freq_base;
        if (c->model_type == 1 && c->rope_local_base_freq > 0.0f &&
            model->layer_is_sliding && model->layer_is_sliding[l]) {
            rope_base = c->rope_local_base_freq;
        }
        if (c->rope_factors_short || c->rope_factors_long) {
            /* Phi-3 LongRoPE with NeoX-style rotation (non-interleaved pairs) */
            const float* factors =
                (pos >= c->rope_orig_ctx_len && c->rope_factors_long)
                    ? c->rope_factors_long
                    : (c->rope_factors_short ? c->rope_factors_short : c->rope_factors_long);
            int half = head_dim / 2;
            for (int h = 0; h < n_heads; h++) {
                float* qh = s->q + h * head_dim;
                for (int i = 0; i < half; i++) {
                    float base_freq = 1.0f / powf(rope_base, 2.0f * i / (float)head_dim);
                    float freq = base_freq / factors[i];
                    float theta = pos * freq;
                    float cos_t = cosf(theta);
                    float sin_t = sinf(theta);
                    float q0 = qh[i], q1 = qh[i + half];
                    qh[i]        = q0 * cos_t - q1 * sin_t;
                    qh[i + half] = q0 * sin_t + q1 * cos_t;
                }
            }
            for (int h = 0; h < n_kv_heads; h++) {
                float* kh = s->k + h * head_dim;
                for (int i = 0; i < half; i++) {
                    float base_freq = 1.0f / powf(rope_base, 2.0f * i / (float)head_dim);
                    float freq = base_freq / factors[i];
                    float theta = pos * freq;
                    float cos_t = cosf(theta);
                    float sin_t = sinf(theta);
                    float k0 = kh[i], k1 = kh[i + half];
                    kh[i]        = k0 * cos_t - k1 * sin_t;
                    kh[i + half] = k0 * sin_t + k1 * cos_t;
                }
            }
            if (pos >= c->rope_orig_ctx_len && c->rope_attn_factor > 0.0f) {
                float scale = c->rope_attn_factor;
                for (int i = 0; i < n_heads * head_dim; i++) s->q[i] *= scale;
            }
        } else if (c->is_gemma4) {
            /* Gemma 4 uses NeoX-style RoPE: pairs are (q[i], q[i+half]) */
            int half = head_dim / 2;
            for (int h = 0; h < n_heads; h++) {
                float* qh = s->q + h * head_dim;
                for (int i = 0; i < half; i++) {
                    float freq = 1.0f / powf(rope_base, 2.0f * i / (float)head_dim);
                    float theta = pos * freq;
                    float cos_t = cosf(theta);
                    float sin_t = sinf(theta);
                    float q0 = qh[i], q1 = qh[i + half];
                    qh[i]        = q0 * cos_t - q1 * sin_t;
                    qh[i + half] = q0 * sin_t + q1 * cos_t;
                }
            }
            if (!is_kv_shared) {
                for (int h = 0; h < n_kv_heads; h++) {
                    float* kh = s->k + h * head_dim;
                    for (int i = 0; i < half; i++) {
                        float freq = 1.0f / powf(rope_base, 2.0f * i / (float)head_dim);
                        float theta = pos * freq;
                        float cos_t = cosf(theta);
                        float sin_t = sinf(theta);
                        float k0 = kh[i], k1 = kh[i + half];
                        kh[i]        = k0 * cos_t - k1 * sin_t;
                        kh[i + half] = k0 * sin_t + k1 * cos_t;
                    }
                }
            }
        } else if (is_kv_shared) {
            /* KV-shared layer: only apply RoPE to Q, not K */
            int half_dim = head_dim / 2;
            for (int h = 0; h < n_heads; h++) {
                float* qh = s->q + h * head_dim;
                float r_base = rope_base;
                for (int i = 0; i < half_dim; i++) {
                    float freq = 1.0f / powf(r_base, 2.0f * i / (float)head_dim);
                    float theta = pos * freq;
                    float cos_t = cosf(theta);
                    float sin_t = sinf(theta);
                    /* NeoX style for Gemma 4, interleaved for others */
                    int a = c->is_gemma4 ? i : 2 * i;
                    int b = c->is_gemma4 ? i + half_dim : 2 * i + 1;
                    float q0 = qh[a], q1 = qh[b];
                    qh[a] = q0 * cos_t - q1 * sin_t;
                    qh[b] = q0 * sin_t + q1 * cos_t;
                }
            }
        } else {
            tq_rope(s->q, s->k, pos, head_dim, n_heads, n_kv_heads, rope_base);
        }
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
    /* For KV-shared layers, use the source layer's cache instead of this layer's.
     * The source layer already wrote K/V at each position, so we just read from it. */
    int kv_cache_layer_idx = l;
    if (is_kv_shared && model->kv_source_layer) {
        kv_cache_layer_idx = model->kv_source_layer[l];
    }
    float* key_cache_layer = s->key_cache + kv_cache_layer_idx * kv_layer_stride;

    /* Skip K/V cache write for KV-shared layers (source layer already wrote) */
    if (is_kv_shared) {
        /* No K/V to store — skip to attention */
    } else if (!use_quant_kv) {
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
        uint16_t* val_fp16_layer = s->value_cache_fp16 + kv_cache_layer_idx * kv_layer_stride;
        if (!is_kv_shared) {
            f32_to_fp16_vec(s->v, val_fp16_layer + (size_t)pos * cache_kv_dim, kv_dim);
        }
    } else {
        float* val_cache_layer = s->value_cache + kv_cache_layer_idx * kv_layer_stride;
        if (!is_kv_shared) {
            memcpy(val_cache_layer + (size_t)pos * cache_kv_dim, s->v, kv_dim * sizeof(float));
        }
    }

    /* Quantize the new key into the quantized cache for integer attention.
     * Each KV head's key vector is quantized independently into blocks.
     *
     * Note: 1-bit/2b/3b sign-based quantization now expands sketch_dim to
     * at least 128 bits for small head_dim (QJL paper: m/d >= 2), so no
     * fallback is needed. */
    int use_int_attn = use_quant_kv;
    /* Hybrid attention KV cache: now allocated with max(sliding, full) dimensions.
     * quant_head_stride uses max_head_dim, quant_pos_stride uses max_kv_heads.
     * Both sliding and full layers can use the quantized cache. */
    int cache_n_kv_heads = c->n_kv_heads;
    if (c->full_n_kv_heads > cache_n_kv_heads) cache_n_kv_heads = c->full_n_kv_heads;
    /* Debug: measure KV quantization roundtrip error (before RHT) */
    if (0 && pos == 0 && l == 0 && getenv("TQ_DEBUG") && use_int_attn) {
        float kmin = s->k[0], kmax = s->k[0], krms = 0;
        for (int i = 0; i < kv_dim; i++) {
            if (s->k[i] < kmin) kmin = s->k[i];
            if (s->k[i] > kmax) kmax = s->k[i];
            krms += s->k[i] * s->k[i];
        }
        krms = sqrtf(krms / kv_dim);
        /* Measure roundtrip: quantize → dequantize → MSE */
        const tq_type_traits_t* dbg_traits = &TQ_TRAITS[s->kv_quant_type];
        float mse = 0, cos_num = 0, cos_d1 = 0, cos_d2 = 0;
        uint8_t tmp_buf[1024];
        float recon[512]; /* max head_dim is 512 (Gemma 4 full layers) */
        for (int kh = 0; kh < 1; kh++) { /* first head only */
            const float* key_src = s->k + kh * head_dim;
            dbg_traits->quantize(key_src, tmp_buf, head_dim);
            dbg_traits->dequantize(tmp_buf, recon, head_dim);
            for (int i = 0; i < head_dim; i++) {
                float diff = key_src[i] - recon[i];
                mse += diff * diff;
                cos_num += key_src[i] * recon[i];
                cos_d1 += key_src[i] * key_src[i];
                cos_d2 += recon[i] * recon[i];
            }
        }
        mse /= head_dim;
        float cosine = cos_num / (sqrtf(cos_d1) * sqrtf(cos_d2) + 1e-10f);
        fprintf(stderr, "[DEBUG] layer0 key: min=%.4f max=%.4f rms=%.4f | quant MSE=%.6f cosine=%.6f (hd=%d)\n",
                kmin, kmax, krms, mse, cosine, head_dim);
    }
    /* Gemma 4 QK-normed keys: pre-scale before quantization.
     * QK-norm produces keys with rms≈0.12, range≈3.0 → sparse distribution.
     * Uniform quantization has terrible cosine (0.62) due to most values → 0.
     * Fix: scale keys by 1/rms to fill dynamic range, store scale factor.
     * Attention: q·k = q·(k/rms)*rms → scale cancels in dot product.
     * Since attention_scale=1.0 for Gemma 4, we apply 1/rms during quantization
     * and multiply by rms when computing attention scores. */
    /* Gemma 4 QK-normed keys: fixed prescale to fill quantization range.
     * QK-norm produces ||k|| = sqrt(head_dim), so per-element rms ≈ 1/sqrt(head_dim).
     * Scale by sqrt(head_dim) so values are O(1), then undo in attention scores.
     * This is equivalent to storing un-normalized keys (before QK-norm application),
     * and attention score computation implicitly includes the 1/sqrt(head_dim). */
    /* Gemma 4 QK-normed keys: apply RHT before quantization to Gaussianize
     * the leptokurtic distribution. Without RHT: cosine=0.62. With RHT: ~0.97+.
     * RHT is orthogonal, so <q, RHT⁻¹(k_hat)> = <RHT(q), k_hat>.
     * We rotate the query once during attention instead of inverse-rotating every key. */
    #define KV_RHT_SEED 0xDEAD4B00u  /* fixed seed for reproducible RHT */
    int kv_use_rht = 0; /* Disabled: pre-norm key storage eliminates need for RHT */
    if (kv_use_rht) {
        /* Apply RHT to key in-place, block_size=128 at a time */
        extern void tq_rht_transform(float* data, int n, uint32_t seed);
        for (int kh = 0; kh < n_kv_heads; kh++) {
            float* kh_ptr = s->k + kh * head_dim;
            for (int blk_start = 0; blk_start < head_dim; blk_start += TQ_BK) {
                int blk_len = head_dim - blk_start;
                if (blk_len > TQ_BK) blk_len = TQ_BK;
                tq_rht_transform(kh_ptr + blk_start, blk_len, KV_RHT_SEED + blk_start);
            }
        }
    }
    /* Debug: measure roundtrip of the keys ACTUALLY stored in quant cache */
    if (pos == 0 && l == 0 && getenv("TQ_DEBUG") && use_int_attn) {
        const tq_type_traits_t* dt = &TQ_TRAITS[s->kv_quant_type];
        const float* dbg_key = save_pre_norm_keys ? pre_norm_keys : s->k;
        float mse=0,cn=0,cd1=0,cd2=0; uint8_t tb[1024]; float rc[512];
        dt->quantize(dbg_key, tb, head_dim);
        dt->dequantize(tb, rc, head_dim);
        for(int i=0;i<head_dim;i++){float d=dbg_key[i]-rc[i];mse+=d*d;cn+=dbg_key[i]*rc[i];cd1+=dbg_key[i]*dbg_key[i];cd2+=rc[i]*rc[i];}
        /* Also check min/max of stored key */
        float dbg_mn=dbg_key[0],dbg_mx=dbg_key[0];
        int nz=0;
        for(int i=0;i<head_dim;i++){if(dbg_key[i]<dbg_mn)dbg_mn=dbg_key[i];if(dbg_key[i]>dbg_mx)dbg_mx=dbg_key[i];if(fabsf(dbg_key[i])>sqrtf(cd1/head_dim)*0.5f)nz++;}
        fprintf(stderr,"[DEBUG] key dist: min=%.2f max=%.2f nonzero(>0.5rms)=%d/%d\n",dbg_mn,dbg_mx,nz,head_dim);
        fprintf(stderr,"[DEBUG] quant key (%s): rms=%.4f | MSE=%.6f cosine=%.6f\n",
                save_pre_norm_keys ? "pre-norm" : "post-norm",
                sqrtf(cd1/head_dim), mse/head_dim, cn/(sqrtf(cd1)*sqrtf(cd2)+1e-10f));
    }
    float kv_prescale = 1.0f;
    if (use_int_attn && !is_kv_shared) {
        const tq_type_traits_t* traits = &TQ_TRAITS[s->kv_quant_type];
        for (int kh = 0; kh < n_kv_heads; kh++) {
            /* Use pre-QK-norm keys for quantization (better rms→cosine) */
            const float* key_src = save_pre_norm_keys ? pre_norm_keys + kh * head_dim
                                                       : s->k + kh * head_dim;
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
                /* Non-delta mode: quantize absolute key.
                 * For head_dim > TQ_BK (e.g. Qwen3.5 head_dim=256),
                 * process multiple TQ_BK-sized blocks per head. */
                for (int blk = 0; blk < head_dim; blk += TQ_BK) {
                    int blen = head_dim - blk;
                    if (blen > TQ_BK) blen = TQ_BK;
                    traits->quantize(key_src + blk,
                                     quant_dst + (blk / TQ_BK) * traits->type_size,
                                     blen);
                }
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

    /* If RHT was applied to keys, pre-rotate ALL queries once.
     * RHT is applied per 128-element block with fixed seed.
     * Since <RHT(q), k_rot> = <q, RHT⁻¹(k_rot)> = <q, k_orig>, we rotate q. */
    float q_rot_buf[8192]; /* max n_heads * head_dim for Gemma 4 */
    float* q_base = s->q;
    if (kv_use_rht && use_quant_kv && n_heads * head_dim <= 8192) {
        extern void tq_rht_transform(float* data, int n, uint32_t seed);
        memcpy(q_rot_buf, s->q, (size_t)n_heads * head_dim * sizeof(float));
        for (int h = 0; h < n_heads; h++) {
            float* qh_rot = q_rot_buf + h * head_dim;
            for (int blk_start = 0; blk_start < head_dim; blk_start += TQ_BK) {
                int blk_len = head_dim - blk_start;
                if (blk_len > TQ_BK) blk_len = TQ_BK;
                tq_rht_transform(qh_rot + blk_start, blk_len, KV_RHT_SEED + blk_start);
            }
        }
        q_base = q_rot_buf;
    }

    for (int h = 0; h < n_heads; h++) {
        /* Use rotated query for quantized KV attention, original for FP32 */
        float* qh = (kv_use_rht && use_quant_kv) ? q_base + h * head_dim : s->q + h * head_dim;
        float* atth = s->att + (size_t)h * c->max_seq_len;
        int kv_h = h / kv_mul;

        if (use_int_attn && seq_len > int_attn_threshold && head_dim <= TQ_BK) {
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
                        /* Multi-block dequant for head_dim > TQ_BK */
                        for (int blk = 0; blk < head_dim; blk += TQ_BK) {
                            int blen = head_dim - blk;
                            if (blen > TQ_BK) blen = TQ_BK;
                            traits->dequantize(quant_src + (blk / TQ_BK) * traits->type_size,
                                               dequant_buf + blk, blen);
                        }
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
            /* Dequant attention: gather quantized key blocks for this kv head
             * then call the type's optimized fused attention kernel which
             * pre-rotates the query ONCE and skips per-position inverse RHT.
             *
             * Round 5 optimization: previously this loop called
             *   traits->dequantize(quant_src, buf, head_dim)
             * per position, which paid O(d log d) inverse RHT per call. The
             * fused traits->attention path eliminates that completely.
             *
             * Fast path conditions: no QK-norm-on-stored-keys, no high-res
             * window, no Gemma 4 prescale. Falls back to per-position dequant
             * for the complex cases.
             */
            const tq_type_traits_t* traits = &TQ_TRAITS[s->kv_quant_type];
            float prescale_inv = (c->is_gemma4 && kv_prescale != 1.0f) ? (1.0f / kv_prescale) : 1.0f;
            float inv_scale = prescale_inv / sqrtf(attn_scale_dim);

            int k_hr_win = s->k_highres_window;
            int k_hr_active = (k_hr_win > 0 && s->key_highres_fp32 != NULL);
            int k_window_start = k_hr_active ? (pos - k_hr_win + 1) : seq_len;
            if (k_window_start < 0) k_window_start = 0;
            size_t hr_layer_stride = k_hr_active ?
                (size_t)k_hr_win * cache_kv_dim : 0;

            int needs_post_norm = (save_pre_norm_keys && layer->k_norm != NULL);

            for (int t = 0; t < attn_start; t++) atth[t] = -1e30f;

            /* Fast path: no post-norm, no high-res window, attention type
             * supports the fused kernel (which is true for all turbo_kv_*).
             *
             * Round 9: still gather + bulk attention. Skipping gather with
             * strided per-position attention turned out NOT to help —
             * Apple Silicon's prefetcher handles the strided pattern fine,
             * and the gather lets the CPU's L1 prefetcher walk through a
             * contiguous block, which is cache-efficient.
             */
            if (!needs_post_norm && !k_hr_active && traits->attention != NULL
                && attn_start == 0 && head_dim <= TQ_BK) {
                size_t head_block_bytes = s->quant_head_stride;
                size_t pos_stride_bytes = (size_t)cache_n_kv_heads * head_block_bytes;
                uint8_t* layer_base = (uint8_t*)s->quant_key_cache
                    + (size_t)l * s->quant_kv_stride;
                uint8_t* gather_dst = (uint8_t*)s->quant_key_buf;
                for (int t = 0; t < seq_len; t++) {
                    const uint8_t* src = layer_base
                        + (size_t)t * pos_stride_bytes
                        + (size_t)kv_h * head_block_bytes;
                    memcpy(gather_dst + (size_t)t * head_block_bytes, src, head_block_bytes);
                }

                /* Single bulk attention call (query pre-rotated inside) */
                traits->attention(qh, s->quant_key_buf, atth, seq_len, head_dim);

                /* Apply scale */
                for (int t = 0; t < seq_len; t++) {
                    atth[t] *= inv_scale;
                }
            } else {
                /* Slow path: per-position dequant for complex cases */
                float dequant_buf[512];
                for (int t = attn_start; t < seq_len; t++) {
                    const float* key_ptr;
                    if (k_hr_active && t >= k_window_start) {
                        int win_idx = t % k_hr_win;
                        key_ptr = s->key_highres_fp32
                            + (size_t)l * hr_layer_stride
                            + (size_t)win_idx * cache_kv_dim
                            + kv_h * head_dim;
                    } else {
                        const uint8_t* quant_src = (const uint8_t*)s->quant_key_cache
                            + (size_t)l * s->quant_kv_stride
                            + (size_t)t * cache_n_kv_heads * s->quant_head_stride
                            + (size_t)kv_h * s->quant_head_stride;
                        /* Multi-block dequant for head_dim > TQ_BK */
                        for (int blk = 0; blk < head_dim; blk += TQ_BK) {
                            int blen = head_dim - blk;
                            if (blen > TQ_BK) blen = TQ_BK;
                            traits->dequantize(quant_src + (blk / TQ_BK) * traits->type_size,
                                               dequant_buf + blk, blen);
                        }
                        if (needs_post_norm) {
                            tq_rmsnorm(dequant_buf, dequant_buf, layer->k_norm,
                                       head_dim, c->rms_norm_eps);
                        }
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
            }
        } else {
            /* FP32 attention scores (no quantization) — NEON-optimized */
            float inv_scale = 1.0f / sqrtf(attn_scale_dim);
            /* Set positions outside sliding window to -inf */
            for (int t = 0; t < attn_start; t++) {
                atth[t] = -1e30f;
            }
            for (int t = attn_start; t < seq_len; t++) {
                const float* kt = key_cache_layer + (size_t)t * cache_kv_dim + kv_h * head_dim;
                float score = 0.0f;
#ifdef __ARM_NEON
                float32x4_t vsum = vdupq_n_f32(0.0f);
                int d = 0;
                for (; d + 4 <= head_dim; d += 4) {
                    float32x4_t vq = vld1q_f32(qh + d);
                    float32x4_t vk = vld1q_f32(kt + d);
                    vsum = vfmaq_f32(vsum, vq, vk);
                }
                score = vaddvq_f32(vsum);
                for (; d < head_dim; d++) {
                    score += qh[d] * kt[d];
                }
#else
                for (int d = 0; d < head_dim; d++) {
                    score += qh[d] * kt[d];
                }
#endif
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
            float v_tmp[512]; /* max head_dim is 512 (Gemma 4 full layers) */
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
            const uint16_t* vfp16_layer = s->value_cache_fp16 + kv_cache_layer_idx * kv_layer_stride;
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
                    xbh[d] += a * fp16_to_f32(vt16[d]);
                }
#else
                for (int d = 0; d < head_dim; d++) {
                    xbh[d] += a * fp16_to_f32(vt16[d]);
                }
#endif
            }
        } else {
            /* FP32 value path (original) */
            const float* val_cache_layer_fp32 = s->value_cache + kv_cache_layer_idx * kv_layer_stride;
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
    /* Flush wo GPU dispatch before CPU reads xb2 for residual add */
    if (has_gguf) tq_metal_batch_flush_if_available();
    TQ_PROF_STOP(_tp, matmul_ns);

    if (l == 3 && pos <= 1 && getenv("TQ_DEBUG_PREFILL")) {
        double _sum=0, _sabs=0;
        for (int i = 0; i < dim; i++) { _sum += s->xb[i]; _sabs += (s->xb[i]<0?-s->xb[i]:s->xb[i]); }
        fprintf(stderr, "[fwd]   L3 pos=%d xb sum=%.6f sumabs=%.6f [0:4]=%.6f,%.6f,%.6f,%.6f [dim/2]=%.6f\n",
            pos, _sum, _sabs, s->xb[0], s->xb[1], s->xb[2], s->xb[3], s->xb[dim/2]);
        fprintf(stderr, "[fwd]   L3 pos=%d xb2 (after wo) [0:8] = ", pos);
        for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", s->xb2[i]);
        fprintf(stderr, "\n");
    }
    /* Debug: print attention output before residual add */
    if (pos == 0 && getenv("TQ_DEBUG") && (l < 3 || l == 5 || l == 11)) {
        float maxv = 0, minv = 0;
        for (int i = 0; i < dim; i++) {
            if (s->xb2[i] > maxv) maxv = s->xb2[i];
            if (s->xb2[i] < minv) minv = s->xb2[i];
        }
        int is_full_dbg = (model->layer_is_sliding && !model->layer_is_sliding[l]);
        fprintf(stderr, "[DEBUG] layer%d attn_out min=%.3f max=%.3f (hd=%d, nh=%d, nkv=%d, %s)\n",
                l, minv, maxv, head_dim, n_heads, n_kv_heads,
                is_full_dbg ? "FULL" : "sliding");
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

    /* Phi-3: force all GGUF matmuls to CPU for the entire forward pass.
     * Metal kernels produce garbage for Phi-3.5's architecture (fused QKV,
     * head_dim=96, LongRoPE). CPU NEON path is correct and fast enough
     * on Apple Silicon. Restored at function exit. */
    int _phi3_force_cpu = c->has_fused_qkv;
    if (_phi3_force_cpu) {
        extern int tq_matmul_force_cpu;
        tq_matmul_force_cpu = 1;
    }

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
    } else if (model->embed_gguf && !model->token_embedding) {
        /* GGUF embedding: dequant single row on demand (no FP32 table in memory) */
        int block_elems = tq_ggml_type_blck(model->embed_gguf_type);
        int block_bytes = (int)tq_ggml_type_size(model->embed_gguf_type);
        int n_blocks = dim / block_elems;
        size_t row_bytes = (size_t)n_blocks * block_bytes;
        const uint8_t* row_ptr = (const uint8_t*)model->embed_gguf + (size_t)token * row_bytes;
        tq_dequant_row_gguf(model->embed_gguf_type, row_ptr, s->x, dim);
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
        int total_ple = n_layers * ple_dim;  /* E2B: 35*256=8960, E4B: 42*256=10752 */

        /* Lazy allocation of ple_buf */
        if (!s->ple_buf) {
            s->ple_buf = (float*)calloc((size_t)total_ple, sizeof(float));
        }

        /* Step A: Dequant per_layer_token_embd[token] → temp_embd
         * The embedding tensor is [total_ple, vocab_size] in GGUF row-major,
         * so one token's data is at row offset = token * row_bytes. */
        float temp_embd[16384];  /* stack buffer, sized for total_ple up to 16384 (E4B=10752) */
        if (total_ple > 16384) return s->logits;  /* safety guard — should not happen */
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

        /* Step B: per_layer_model_proj @ embed_raw → temp_proj
         * ple_proj is [total_ple, hidden_dim] FP32. For E4B: [10752, 2560].
         * We need: for each output row d in [0, total_ple): dot(ple_proj[d,:], s->x[:])
         * Note: s->x already has the scaled embedding from above. */
        float temp_proj[16384];
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

        /* layer_output_scale: simple scalar multiply on entire hidden state (Gemma 4).
         * No need to save/restore residual. */

        /* ============================================================
         * GPU Compute Graph: Full-layer GPU forward
         *
         * When ALL Q4 weights are available for a standard self-attn layer
         * (no DeltaNet, no hybrid attention, no attn_output_gate, no QK-norm),
         * attempt to run the entire layer on GPU with a single command buffer.
         * This eliminates ~20 per-kernel dispatch overheads per layer.
         *
         * Falls through to CPU path on failure (returns -1).
         * ============================================================ */
        int gpu_layer_done = 0;
#ifdef __APPLE__
        /* GPU compute graph: full layer forward on Metal.
         * Currently 2-commit-per-layer (Phase A: QKV+RoPE, Phase B: attn+FFN).
         * Benchmark: 0.6 tok/s vs 17 tok/s CPU — disabled until 1-commit design.
         * Root cause: waitUntilCompleted overhead (~0.3ms × 2 × 28 layers = 17ms).
         * TODO: move KV cache to GPU to eliminate Phase A commit.
         * Infrastructure kept for batch inference (multiple tokens per forward). */
        /* GPU compute graph: 1-commit full-layer forward.
         * Benchmarked: Q4 Metal kernel is 4x slower than CPU NEON Q4×Q8 fused dot.
         * Root cause: Q4 nibble extraction in GPU shader is inefficient.
         * Fix needed: weight repacking to GPU-friendly layout at load time.
         * Infrastructure ready — enable when repacked weights are implemented. */
        /* GPU compute graph with repacked Q4 weights.
         * Benchmarked with tile-major repacking + 1-commit design:
         * SmolLM2: 27 tok/s GPU vs 96 tok/s CPU (3.5x slower)
         * Root cause: Q4 nibble extraction (integer bit ops) is slow on Apple GPU.
         * Apple Silicon GPU excels at float/half ops, not integer bit manipulation.
         * CPU NEON Q4×Q8 fused dot saturates memory bandwidth more efficiently.
         * Infrastructure preserved for FP16/BF16 weight format (no bit extraction). */
        /* GPU graph: fast Q4 kernel (uint16 mask + SIMD-group) benchmarked at
         * 27 tok/s (SmolLM2) vs CPU 96 tok/s. Dispatch overhead remains dominant.
         * Needs: entire forward without CPU↔GPU sync (graph compilation). */
        if (0 && layer->wq_q4 && layer->wk_q4 && layer->wv_q4 && layer->wo_q4 &&
            layer->w_gate_q4 && layer->w_up_q4 && layer->w_down_q4 &&
            !layer->delta_a_log &&  /* not DeltaNet */
            !layer->q_norm &&       /* no QK-norm (would need per-head norm on GPU) */
            !c->attn_output_gate && /* no attention output gate */
            !layer->moe &&          /* not MoE */
            !(model->layer_is_sliding && !model->layer_is_sliding[l] && c->full_head_dim > 0) && /* not hybrid attention */
            c->partial_rotary_factor <= 0.0f &&  /* full RoPE only (no partial) */
            !model->rope_freqs &&   /* no learned RoPE frequencies */
            !s->use_fp16_values &&  /* FP32 value cache only */
            s->value_quant_bits == 0)  /* no quantized value cache */
        {
            int kv_dim_l = c->n_kv_heads * c->head_dim;
            int inter = c->per_layer_inter_dim ? c->per_layer_inter_dim[l] : c->intermediate_dim;
            size_t kv_layer_stride = (size_t)c->max_seq_len * kv_dim_l;

            int rc = tq_metal_forward_layer(
                s->x,
                s->key_cache + (size_t)l * kv_layer_stride,
                s->value_cache + (size_t)l * kv_layer_stride,
                layer->attn_norm, layer->ffn_norm,
                layer->wq_q4, layer->wq_q4s,
                layer->wk_q4, layer->wk_q4s,
                layer->wv_q4, layer->wv_q4s,
                layer->wo_q4, layer->wo_q4s,
                layer->w_gate_q4, layer->w_gate_q4s,
                layer->w_up_q4, layer->w_up_q4s,
                layer->w_down_q4, layer->w_down_q4s,
                dim, c->n_heads, c->n_kv_heads, c->head_dim,
                inter, pos, pos + 1, c->rope_freq_base, c->rms_norm_eps,
                is_gemma3);

            if (rc == 0) {
                gpu_layer_done = 1;
                /* GPU path handled everything including residual adds.
                 * Skip directly to post-layer processing (PLE, layer_output_scale). */
            }
        }
#endif

        int layer_has_gguf = (layer->gguf_wq != NULL || layer->gguf_w_qkv != NULL);

        if (gpu_layer_done) goto layer_postprocess;

        /* Pre-attention/DeltaNet RMSNorm */
        tq_rmsnorm(s->xb, s->x, layer->attn_norm, dim, c->rms_norm_eps);

        /* Begin layer-level GPU batch scope: all GGUF matmuls in this layer
         * (QKV, wo, gate, up, down) encode into shared command buffers.
         * Intermediate flushes synchronize where CPU needs GPU results.
         * This keeps batch mode active throughout the layer so even single
         * matmuls (wo, down) benefit from batch-mode GPU dispatch. */
        /* Metal batch mode: GGUF on-the-fly path only (Gemma 4 MoE).
         * Q4 converted weights: CPU NEON Q4×Q8 is faster than Metal GPU
         * due to per-dispatch overhead exceeding compute time on small matrices.
         * Benchmarked: Metal Q4 batch → 38 tok/s vs CPU Q4 → 95 tok/s (SmolLM2).
         *
         * Skip Metal batch for Phi-3 fused layers: mixing CPU fallback
         * (fused QKV/FFN) with GPU matmuls (wo, down) in the same batch
         * causes buffer synchronization issues. All matmuls fall through
         * to CPU NEON path instead — still fast on Apple Silicon. */
        /* Phi-3 fused layers: force ALL GGUF matmuls to CPU for this layer.
         * Metal immediate-mode matmul also produces garbage for Phi-3.5's
         * dimensions (hidden=3072, inter=8192) — root cause TBD. CPU NEON
         * path works correctly and is fast enough on Apple Silicon. */
        int skip_metal_batch = (layer->gguf_w_qkv != NULL);
        if (layer_has_gguf && !skip_metal_batch) tq_metal_batch_begin_if_available();

        if (layer->delta_a_log) {
            /* DeltaNet layer */
            deltanet_forward(model, s, l);
        } else if (layer->gguf_w_qkv) {
            /* Phi-3 fused QKV — `gguf_wq/wk/wv` are NULL because Q, K
             * and V are concatenated into `gguf_w_qkv`. self_attn_forward
             * handles the fused dispatch internally. */
            self_attn_forward(model, s, l, pos);
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

        /* Gemma 4 dual-FFN: Dense (shared MLP) and MoE run IN PARALLEL from same input,
         * outputs summed, then final post_ffw_norm, then residual add.
         * Reference: refs/llama.cpp/src/models/gemma4-iswa.cpp lines 117-190 */
        int did_moe = 0;
        int has_dense_ffn = (layer->w_gate || layer->w_gate_q8 || layer->w_gate_q4 || layer->w_gate_q2 || layer->gguf_w_gate);
        int gemma4_dual_ffn = (c->is_gemma4 && layer->moe && s->moe_state && model->moe_config && has_dense_ffn);

        if (pos == 0 && l == 0 && getenv("TQ_DEBUG"))
            fprintf(stderr, "[DEBUG] layer0 gemma4_dual_ffn=%d (is_gemma4=%d moe=%p moe_state=%p dense=%d)\n",
                    gemma4_dual_ffn, c->is_gemma4, (void*)layer->moe, (void*)s->moe_state, has_dense_ffn);

        if (gemma4_dual_ffn) {
            /* ---- Gemma 4 dual-FFN path ----
             * Both Dense and MoE branch from s->x (attn_out), outputs summed into xb2 */
            float* dense_out = s->xb2; /* reuse xb2 for dense output */

            /* Dense FFN (shared MLP): attn_out → ffn_norm → GELU FFN → post_ffw_norm_1 */
            {
                float* dense_norm_w = layer->ffn_norm;
                tq_rmsnorm(s->xb, s->x, dense_norm_w, dim, c->rms_norm_eps);

                int inter = c->per_layer_inter_dim ? c->per_layer_inter_dim[l] : c->intermediate_dim;
                TQ_PROF_START(_tp);
                if (layer->w_gate_q4) {
                    tq_quantize_row_q8(s->xb, s->xb_q8, s->xb_q8s, dim);
                    tq_matmul_q4_preq(s->hb, layer->w_gate_q4, layer->w_gate_q4s,
                                       s->xb_q8, s->xb_q8s, inter, dim);
                    tq_matmul_q4_preq(s->hb2, layer->w_up_q4, layer->w_up_q4s,
                                       s->xb_q8, s->xb_q8s, inter, dim);
                } else if (layer->gguf_w_gate) {
                    tq_matmul_gguf(s->hb, s->xb, layer->gguf_w_gate, layer->gguf_w_gate_type, inter, dim);
                    tq_matmul_gguf(s->hb2, s->xb, layer->gguf_w_up, layer->gguf_w_up_type, inter, dim);
                    tq_metal_batch_flush_if_available();
                } else if (layer->w_gate_q8) {
                    tq_matmul_q8(s->hb, s->xb, layer->w_gate_q8, layer->w_gate_q8s, inter, dim);
                    tq_matmul_q8(s->hb2, s->xb, layer->w_up_q8, layer->w_up_q8s, inter, dim);
                } else {
                    tq_matmul(s->hb, s->xb, layer->w_gate, inter, dim);
                    tq_matmul(s->hb2, s->xb, layer->w_up, inter, dim);
                }
                TQ_PROF_STOP(_tp, matmul_ns);

                tq_gelu_tanh(s->hb, inter);
                tq_mul(s->hb, s->hb, s->hb2, inter);

                TQ_PROF_START(_tp);
                if (layer->w_down_q4)
                    tq_matmul_q4(dense_out, s->hb, layer->w_down_q4, layer->w_down_q4s, dim, inter);
                else if (layer->gguf_w_down)
                    tq_matmul_gguf(dense_out, s->hb, layer->gguf_w_down, layer->gguf_w_down_type, dim, inter);
                else if (layer->w_down_q8)
                    tq_matmul_q8(dense_out, s->hb, layer->w_down_q8, layer->w_down_q8s, dim, inter);
                else
                    tq_matmul(dense_out, s->hb, layer->w_down, dim, inter);
                tq_metal_batch_flush_if_available();
                TQ_PROF_STOP(_tp, matmul_ns);

                /* Apply post_ffw_norm_1 to dense output */
                float* dense_post = layer->post_ffn_norm_1 ? layer->post_ffn_norm_1 : NULL;
                if (dense_post)
                    tq_rmsnorm(dense_out, dense_out, dense_post, dim, c->rms_norm_eps);
            }

            /* MoE FFN: attn_out → pre_ffw_norm_2 → MoE → post_ffw_norm_2
             * Router uses separate norm: rms_norm(attn_out) / sqrt(dim) * scale
             * (refs/llama.cpp/src/models/gemma4-iswa.cpp line 141-145) */
            float moe_out_buf[4096]; /* stack buffer for MoE output */
            {
                /* Pre-compute routing from raw attn_out (s->x) */
                float route_buf[4096];
                tq_moe_layer_t* moe_layer = (tq_moe_layer_t*)layer->moe;
                tq_moe_state_t* moe_st = (tq_moe_state_t*)s->moe_state;
                tq_moe_config_t* moe_cfg = (tq_moe_config_t*)model->moe_config;

                /* Unweighted RMS norm of attn_out */
                float ss = 0.0f;
                for (int i = 0; i < dim; i++) ss += s->x[i] * s->x[i];
                ss = 1.0f / sqrtf(ss / (float)dim + c->rms_norm_eps);
                float inv_sqrt_dim = 1.0f / sqrtf((float)dim);
                for (int i = 0; i < dim; i++)
                    route_buf[i] = s->x[i] * ss * inv_sqrt_dim;

                /* Apply per-feature scale (ffn_gate_inp.scale) */
                if (moe_layer->router_input_scale) {
                    for (int i = 0; i < dim; i++)
                        route_buf[i] *= moe_layer->router_input_scale[i];
                }

                /* Route: select top-K experts */
                tq_moe_route(route_buf, moe_layer->router_weight,
                             moe_cfg->num_experts, moe_cfg->num_active, dim,
                             moe_st->top_experts, moe_st->expert_weights);
                moe_st->routing_precomputed = 1;

                /* Norm input for expert FFN */
                float* moe_norm_w = layer->pre_ffn_norm_2 ? layer->pre_ffn_norm_2 : layer->ffn_norm;
                tq_rmsnorm(s->xb, s->x, moe_norm_w, dim, c->rms_norm_eps);

                TQ_PROF_START(_tp);
                tq_moe_forward((const tq_moe_layer_t*)layer->moe,
                               (const tq_moe_config_t*)model->moe_config,
                               (tq_moe_state_t*)s->moe_state,
                               s->xb, moe_out_buf, dim, l);
                TQ_PROF_STOP(_tp, moe_ns);

                /* Apply post_ffw_norm_2 to MoE output */
                float* moe_post = layer->post_ffn_norm_2 ? layer->post_ffn_norm_2 : NULL;
                if (moe_post)
                    tq_rmsnorm(moe_out_buf, moe_out_buf, moe_post, dim, c->rms_norm_eps);
            }

            /* Sum dense + MoE outputs */
            for (int i = 0; i < dim; i++)
                dense_out[i] += moe_out_buf[i];

            /* Apply final post_ffw_norm to combined output */
            if (layer->post_ffn_norm)
                tq_rmsnorm(dense_out, dense_out, layer->post_ffn_norm, dim, c->rms_norm_eps);

            /* Residual: x = attn_out + combined_ffn */
            tq_add(s->x, s->x, dense_out, dim);
            did_moe = 1;
        }

        /* MoE-only FFN path (non-Gemma4: Qwen MoE, Gemma 3 MoE) */
        if (!gemma4_dual_ffn && layer->moe && s->moe_state && model->moe_config) {
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
        /* Dense FFN path — SwiGLU (Qwen3.5) or GeGLU (Gemma3).
         * Qwen: layers are either MoE or dense, NOT both.
         * Gemma 3 non-MoE layers: run dense FFN. */
        if (!did_moe &&
            (layer->w_gate || layer->w_gate_q8 || layer->w_gate_q4 || layer->w_gate_q2 || layer->gguf_w_gate || layer->gguf_w_up_gate) &&
            (layer->w_up || layer->w_up_q8 || layer->w_up_q4 || layer->w_up_q2 || layer->gguf_w_up || layer->gguf_w_up_gate) &&
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
            } else if (layer->gguf_w_up_gate) {
                /* Phi-3 fused gate||up — force CPU (same Metal issue as fused QKV) */
                extern void tq_matmul_gguf_cpu(float*, const float*, const void*,
                                                int, int, int);
                tq_matmul_gguf_cpu(s->hb, s->xb,
                                   layer->gguf_w_up_gate, layer->gguf_w_up_gate_type,
                                   2 * inter, dim);
                memcpy(s->hb2, s->hb + inter, (size_t)inter * sizeof(float));
            } else if (layer->gguf_w_gate) {
                /* Gate+up GPU dispatches batched by layer-level batch scope */
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
            /* Flush w_down GPU dispatch before CPU reads xb2 for post-FFN norm / residual */
            tq_metal_batch_flush_if_available();
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

    layer_postprocess:
        if (l <= 3 && pos <= 1 && getenv("TQ_DEBUG_PREFILL")) {
            double _s=0, _sa=0;
            for (int i = 0; i < dim; i++) { _s += s->x[i]; _sa += (s->x[i]<0?-s->x[i]:s->x[i]); }
            fprintf(stderr, "[fwd]   L%d pos=%d final x sum=%.9f sumabs=%.9f\n", l, pos, _s, _sa);
        }
        /* Post-layer processing: PLE, layer_output_scale.
         * GPU graph path jumps here after full-layer GPU forward. */

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

        /* End layer-level GPU batch scope */
        if (layer_has_gguf && !skip_metal_batch) tq_metal_batch_end_if_available();

        /* Gemma 4: layer_output_scale is a simple scalar multiply on the entire hidden state.
         * Reference: refs/llama.cpp/src/models/gemma4-iswa.cpp line 216-218
         *   cur = ggml_mul(ctx0, cur, model.layers[il].out_scale); */
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
                s->x[i] *= los;
            }
        }

        /* Debug: print layer output */
        if (pos == 0 && getenv("TQ_DEBUG")) {
            if (l < 10 || l == c->n_layers - 1 || l % 10 == 0) {
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
        /* GGUF fused dot output projection — 3.5x less memory bandwidth than FP32 */
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
        /* Find top-5 tokens */
        int top5[5] = {0,1,2,3,4}; float top5v[5];
        for (int i = 0; i < 5; i++) top5v[i] = s->logits[top5[i]];
        for (int i = 5; i < c->vocab_size; i++) {
            int minj = 0;
            for (int j = 1; j < 5; j++) if (top5v[j] < top5v[minj]) minj = j;
            if (s->logits[i] > top5v[minj]) { top5[minj] = i; top5v[minj] = s->logits[i]; }
        }
        /* Sort top5 by value descending */
        for (int i = 0; i < 4; i++) for (int j = i+1; j < 5; j++)
            if (top5v[j] > top5v[i]) { int ti=top5[i]; top5[i]=top5[j]; top5[j]=ti; float tv=top5v[i]; top5v[i]=top5v[j]; top5v[j]=tv; }
        fprintf(stderr, "... top5: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%d(%.1f) ", top5[i], top5v[i]);
        fprintf(stderr, "\n");
        /* Check output weight row norms for top token */
        if (pos == 0 && model->output_weight) {
            int ti = top5[0];
            float norm_ti = 0, norm_2 = 0;
            const float* row_ti = model->output_weight + (size_t)ti * dim;
            const float* row_2 = model->output_weight + (size_t)2 * dim;
            for (int i = 0; i < dim; i++) { norm_ti += row_ti[i]*row_ti[i]; norm_2 += row_2[i]*row_2[i]; }
            fprintf(stderr, "[DEBUG] output_weight norms: tok2=%.4f tok%d=%.4f\n",
                    sqrtf(norm_2), ti, sqrtf(norm_ti));
        }
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

    /* Restore Metal dispatch for non-Phi3 models */
    if (_phi3_force_cpu) {
        extern int tq_matmul_force_cpu;
        tq_matmul_force_cpu = 0;
    }
    return s->logits;
}

/* ============================================================
 * Batched prefill — process N consecutive tokens in one call.
 *
 * Strategy: walk the layers exactly like tq_forward does, but with the
 * matmul calls replaced by tq_batched_matmul_q4 over an [N, D] activation
 * matrix. Per-token operations (RoPE, attention against history) are
 * still sequential — this is fine because attention is small for short
 * context; the matmul gains dominate. KV cache writes are per-token but
 * to consecutive positions [pos_start..pos_start+N).
 *
 * Currently supported: standard Llama family with load-time Q4 conversion
 * (use_q4_weights=1, separate Q/K/V/O + gate/up/down). Returns -1 for
 * unsupported architectures so the caller falls back to a per-token loop.
 *
 * Output: KV cache populated for [pos_start..pos_start+N). Logits NOT
 * computed (would require an extra batched lm_head matmul; the caller
 * can still tq_forward(token[N-1], pos_start+N-1) for the final token).
 * ============================================================ */
int tq_forward_batch(tq_model_t* model, tq_state_t* s,
                     const int* tokens, int N, int pos_start) {
    if (N <= 0) return pos_start;
    /* SANITY CHECK MODE: just call tq_forward N times. If THIS gives
     * different results than the per-token tq_generate loop, the bug
     * is in the orchestration outside the matmul work. Set
     * TQ_BATCH_SANITY=1 to enable. */
    if (getenv("TQ_BATCH_SANITY")) {
        for (int n = 0; n < N; n++) {
            tq_forward(model, s, tokens[n], pos_start + n);
        }
        return pos_start + N;
    }
    tq_model_config_t* c = &model->config;

    /* Architectural gating: only standard Llama for now. */
    int dbg = (getenv("TQ_DEBUG_PREFILL") != NULL);
    if (!model->use_q4_weights)               { if (dbg) fprintf(stderr, "[batch] bail: !use_q4_weights\n"); return -1; }
    if (c->is_moe || c->is_gemma4)             { if (dbg) fprintf(stderr, "[batch] bail: moe/gemma4\n"); return -1; }
    if (c->has_fused_qkv || c->has_fused_up_gate) { if (dbg) fprintf(stderr, "[batch] bail: fused qkv/up\n"); return -1; }
    if (c->n_kv_shared_layers > 0)             { if (dbg) fprintf(stderr, "[batch] bail: kv_shared\n"); return -1; }
    if (s->delta_kv_enabled)                   { if (dbg) fprintf(stderr, "[batch] bail: delta_kv\n"); return -1; }
    /* k_highres_window supported — circular FP32 buffer for recent keys. */
    if (s->value_quant_bits != 0)              { if (dbg) fprintf(stderr, "[batch] bail: quant_V\n"); return -1; }
    /* DeltaNet: hybrid batched is WIP. Default bail to per-token; opt-in
     * via TQ_DELTANET_BATCH=1 for development. Known issue: FFN handling
     * for DeltaNet layers in Qwen3.5 still produces empty output. */
    if (!getenv("TQ_DELTANET_BATCH")) {
        for (int l = 0; l < c->n_layers; l++) {
            if (model->layers[l].delta_a_log) {
                if (dbg) fprintf(stderr, "[batch] bail: deltanet l=%d\n", l);
                return -1;
            }
        }
    }

    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;

    /* Allocate batch scratch (N×{dim, q_dim, kv_dim, inter} FP32).
     * For Phi-3.5 N=32: 7 buffers × 32 × 8192 × 4 = 7 MB. Trivial. */
    size_t bytes_x  = (size_t)N * dim   * sizeof(float);
    size_t bytes_q  = (size_t)N * q_dim * sizeof(float);
    size_t bytes_kv = (size_t)N * kv_dim * sizeof(float);
    size_t bytes_h  = (size_t)N * inter * sizeof(float);

    float* X     = (float*)malloc(bytes_x);
    float* Xres  = (float*)malloc(bytes_x);  /* residual stream */
    float* XBN   = (float*)malloc(bytes_x);  /* normed for matmul */
    float* QB    = (float*)malloc(bytes_q);
    float* KB    = (float*)malloc(bytes_kv);
    float* VB    = (float*)malloc(bytes_kv);
    float* OB    = (float*)malloc(bytes_x);
    float* GB    = (float*)malloc(bytes_h);
    float* UB    = (float*)malloc(bytes_h);
    if (!X || !Xres || !XBN || !QB || !KB || !VB || !OB || !GB || !UB) {
        free(X); free(Xres); free(XBN); free(QB); free(KB); free(VB);
        free(OB); free(GB); free(UB);
        return -1;
    }

    /* Step 1: token embeddings (per-token). Mirror tq_forward's lookup.
     * Three sources: BF16 mmap, GGUF on-demand dequant, or FP32 table. */
    for (int n = 0; n < N; n++) {
        int tok = tokens[n];
        float* dst = Xres + (size_t)n * dim;
        if (model->embed_bf16) {
            const uint16_t* src = model->embed_bf16 + (size_t)tok * dim;
            for (int i = 0; i < dim; i++) {
                uint32_t bits = ((uint32_t)src[i]) << 16;
                memcpy(&dst[i], &bits, 4);
            }
        } else if (model->embed_gguf && !model->token_embedding) {
            int block_elems = tq_ggml_type_blck(model->embed_gguf_type);
            int block_bytes = (int)tq_ggml_type_size(model->embed_gguf_type);
            int n_blocks = dim / block_elems;
            size_t row_bytes = (size_t)n_blocks * block_bytes;
            const uint8_t* row_ptr = (const uint8_t*)model->embed_gguf + (size_t)tok * row_bytes;
            tq_dequant_row_gguf(model->embed_gguf_type, row_ptr, dst, dim);
        } else if (model->token_embedding) {
            memcpy(dst, model->token_embedding + (size_t)tok * dim,
                   (size_t)dim * sizeof(float));
        } else {
            if (dbg) fprintf(stderr, "[batch] bail: no embed source (n=%d tok=%d)\n", n, tok);
            free(X); free(Xres); free(XBN); free(QB); free(KB); free(VB);
            free(OB); free(GB); free(UB);
            return -1;
        }
    }

    /* Per-layer KV cache stride (FP32 K and V). */
    size_t kv_layer_stride = (size_t)c->max_seq_len * (size_t)kv_dim;

    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];

        /* DeltaNet layer (Qwen3.5 hybrid): recurrent state can't be batched
         * across the sequence dim, so drive each token through the existing
         * tq_forward per-layer path that updates s->delta_state and
         * s->conv_state in sequence order. FFN for this layer is still done
         * per-token because deltanet_forward writes residual into s->x and
         * we continue from there. */
        if (layer->delta_a_log) {
            /* DeltaNet: SSM recurrent state can't be batched. Process each
             * token in order so state advances correctly; no final
             * tq_forward runs after this function (logits computed below). */
            extern void deltanet_forward(tq_model_t* model, tq_state_t* s, int l);
            for (int n = 0; n < N; n++) {
                memcpy(s->x, Xres + (size_t)n * dim, (size_t)dim * sizeof(float));
                tq_rmsnorm(s->xb, s->x, layer->attn_norm, dim, c->rms_norm_eps);
                deltanet_forward(model, s, l);
                /* deltanet_forward adds residual into s->x. Copy back. */
                memcpy(Xres + (size_t)n * dim, s->x, (size_t)dim * sizeof(float));

                /* FFN for this token — use the existing tq_forward's logic
                 * inline. Most Qwen3.5 layers have FFN norm → gate+up → silu
                 * → down → residual. */
                if (layer->w_gate_q4 && layer->w_up_q4 && layer->w_down_q4) {
                    tq_rmsnorm(s->xb, s->x, layer->ffn_norm, dim, c->rms_norm_eps);
                    /* Use tq_matmul_q4 via per-token path */
                    int inter_l = c->intermediate_dim;
                    float* tmp_g = (float*)malloc((size_t)inter_l * sizeof(float));
                    float* tmp_u = (float*)malloc((size_t)inter_l * sizeof(float));
                    float* tmp_d = (float*)malloc((size_t)dim * sizeof(float));
                    if (tmp_g && tmp_u && tmp_d) {
                        tq_quantize_row_q8(s->xb, s->xb_q8, s->xb_q8s, dim);
                        tq_matmul_q4_preq(tmp_g, layer->w_gate_q4, layer->w_gate_q4s, s->xb_q8, s->xb_q8s, inter_l, dim);
                        tq_matmul_q4_preq(tmp_u, layer->w_up_q4, layer->w_up_q4s, s->xb_q8, s->xb_q8s, inter_l, dim);
                        for (int i = 0; i < inter_l; i++) {
                            float g = tmp_g[i];
                            tmp_g[i] = (g / (1.0f + expf(-g))) * tmp_u[i];
                        }
                        tq_quantize_row_q8(tmp_g, s->xb_q8, s->xb_q8s, inter_l);
                        tq_matmul_q4_preq(tmp_d, layer->w_down_q4, layer->w_down_q4s, s->xb_q8, s->xb_q8s, dim, inter_l);
                        for (int i = 0; i < dim; i++) s->x[i] += tmp_d[i];
                        memcpy(Xres + (size_t)n * dim, s->x, (size_t)dim * sizeof(float));
                    }
                    free(tmp_g); free(tmp_u); free(tmp_d);
                }
            }
            continue;  /* skip the self-attention layer code below */
        }

        /* Required Q4 weights for this fast path (self_attn layers). */
        if (!layer->wq_q4 || !layer->wk_q4 || !layer->wv_q4 || !layer->wo_q4 ||
            !layer->w_gate_q4 || !layer->w_up_q4 || !layer->w_down_q4) {
            if (dbg) fprintf(stderr, "[batch] bail: layer %d missing q4 weights (wq=%p wk=%p wv=%p wo=%p g=%p u=%p d=%p)\n",
                l, (void*)layer->wq_q4, (void*)layer->wk_q4, (void*)layer->wv_q4,
                (void*)layer->wo_q4, (void*)layer->w_gate_q4, (void*)layer->w_up_q4, (void*)layer->w_down_q4);
            free(X); free(Xres); free(XBN); free(QB); free(KB); free(VB);
            free(OB); free(GB); free(UB);
            return -1;
        }

        /* 1. attn RMSNorm (per-row) */
        for (int n = 0; n < N; n++) {
            tq_rmsnorm(XBN + (size_t)n * dim, Xres + (size_t)n * dim,
                       layer->attn_norm, dim, c->rms_norm_eps);
        }

        /* 2. Q, K, V batched matmul (Q4 main weights) */
        tq_batched_matmul_q4(QB, layer->wq_q4, layer->wq_q4s, XBN, q_dim,  dim, N, NULL);
        tq_batched_matmul_q4(KB, layer->wk_q4, layer->wk_q4s, XBN, kv_dim, dim, N, NULL);
        tq_batched_matmul_q4(VB, layer->wv_q4, layer->wv_q4s, XBN, kv_dim, dim, N, NULL);

        if (l == 0 && dbg) {
            fprintf(stderr, "[batch] L0 VB (post-matmul) tok0 [0:8] = ");
            for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", VB[i]);
            fprintf(stderr, "\n");
        }
        /* 2-r. Add Q2 residual correction per-token (matches tq_matmul_q4q2_preq).
         * Load-time Q4 conversion stores BOTH Q4 main + Q2 residual. Skipping the
         * Q2 part causes large numerical drift. We do the Q2 part per-token using
         * the existing primitive — Q2 is small (2 bits) so the per-token cost is
         * a fraction of the Q4 batched savings. */
        if (layer->wq_q2 || layer->wk_q2 || layer->wv_q2) {
            int n_blocks_d = dim / 32;
            int8_t* xq = s->xb_q8;     /* reuse state's per-token Q8 buffer */
            float*  xs = s->xb_q8s;
            float* tmp_q = (float*)malloc((size_t)q_dim  * sizeof(float));
            float* tmp_k = (float*)malloc((size_t)kv_dim * sizeof(float));
            float* tmp_v = (float*)malloc((size_t)kv_dim * sizeof(float));
            for (int n = 0; n < N; n++) {
                /* Quantize this row's XBN to Q8 once. */
                tq_quantize_row_q8(XBN + (size_t)n * dim, xq, xs, dim);
                if (layer->wq_q2) {
                    tq_matmul_q2_preq(tmp_q, layer->wq_q2, layer->wq_q2s, xq, xs, q_dim, dim);
                    for (int i = 0; i < q_dim; i++) QB[(size_t)n * q_dim + i] += tmp_q[i];
                }
                if (layer->wk_q2) {
                    tq_matmul_q2_preq(tmp_k, layer->wk_q2, layer->wk_q2s, xq, xs, kv_dim, dim);
                    for (int i = 0; i < kv_dim; i++) KB[(size_t)n * kv_dim + i] += tmp_k[i];
                }
                if (layer->wv_q2) {
                    tq_matmul_q2_preq(tmp_v, layer->wv_q2, layer->wv_q2s, xq, xs, kv_dim, dim);
                    for (int i = 0; i < kv_dim; i++) VB[(size_t)n * kv_dim + i] += tmp_v[i];
                }
            }
            free(tmp_q); free(tmp_k); free(tmp_v);
            (void)n_blocks_d;
        }

        /* 2a. Apply Q/K/V biases (Qwen2/2.5/3 — NULL for Llama). */
        if (layer->q_bias) {
            for (int n = 0; n < N; n++)
                for (int i = 0; i < q_dim; i++) QB[(size_t)n * q_dim + i] += layer->q_bias[i];
        }
        if (layer->k_bias) {
            for (int n = 0; n < N; n++)
                for (int i = 0; i < kv_dim; i++) KB[(size_t)n * kv_dim + i] += layer->k_bias[i];
        }
        if (layer->v_bias) {
            for (int n = 0; n < N; n++)
                for (int i = 0; i < kv_dim; i++) VB[(size_t)n * kv_dim + i] += layer->v_bias[i];
        }
        /* 2b. QK-norm (Qwen3 — NULL for Llama). */
        if (layer->q_norm) {
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < c->n_heads; h++) {
                    float* qh = QB + (size_t)n * q_dim + h * c->head_dim;
                    tq_rmsnorm(qh, qh, layer->q_norm, c->head_dim, c->rms_norm_eps);
                }
            }
        }
        if (layer->k_norm) {
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < c->n_kv_heads; h++) {
                    float* kh = KB + (size_t)n * kv_dim + h * c->head_dim;
                    tq_rmsnorm(kh, kh, layer->k_norm, c->head_dim, c->rms_norm_eps);
                }
            }
        }

        /* 3. RoPE + KV cache write (per-token).
         * Mirror tq_forward's RoPE selection: if model->rope_freqs is set
         * (Llama 3.x learned RoPE scaling, 64 freq factors), apply per-pair
         * factor; otherwise plain interleaved RoPE. */
        for (int n = 0; n < N; n++) {
            float* qn = QB + (size_t)n * q_dim;
            float* kn = KB + (size_t)n * kv_dim;
            int pos = pos_start + n;
            if (model->rope_freqs && model->rope_freqs_len > 0) {
                /* Match tq_forward's rope_n_dims selection: c->rope_n_dims may
                 * differ from head_dim (e.g., Gemma partial RoPE). */
                int rope_n_dims = (c->rope_n_dims > 0) ? c->rope_n_dims : c->head_dim;
                int rope_pairs = rope_n_dims / 2;
                if (rope_pairs > model->rope_freqs_len) rope_pairs = model->rope_freqs_len;
                for (int h = 0; h < c->n_heads; h++) {
                    float* qh = qn + h * c->head_dim;
                    for (int i = 0; i < rope_pairs; i++) {
                        float base = 1.0f / powf(c->rope_freq_base, 2.0f * i / (float)rope_n_dims);
                        float freq = base / model->rope_freqs[i];
                        float theta = pos * freq;
                        float ct = cosf(theta), st = sinf(theta);
                        float q0 = qh[2*i], q1 = qh[2*i+1];
                        qh[2*i]   = q0 * ct - q1 * st;
                        qh[2*i+1] = q0 * st + q1 * ct;
                    }
                }
                for (int h = 0; h < c->n_kv_heads; h++) {
                    float* kh = kn + h * c->head_dim;
                    for (int i = 0; i < rope_pairs; i++) {
                        float base = 1.0f / powf(c->rope_freq_base, 2.0f * i / (float)rope_n_dims);
                        float freq = base / model->rope_freqs[i];
                        float theta = pos * freq;
                        float ct = cosf(theta), st = sinf(theta);
                        float k0 = kh[2*i], k1 = kh[2*i+1];
                        kh[2*i]   = k0 * ct - k1 * st;
                        kh[2*i+1] = k0 * st + k1 * ct;
                    }
                }
            } else {
                tq_rope(qn, kn, pos, c->head_dim, c->n_heads, c->n_kv_heads,
                        c->rope_freq_base);
            }
            if (n == 0 && l == 0 && dbg) {
                fprintf(stderr, "[batch] L0 QB (post-RoPE) tok0 [0:8] = ");
                for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", qn[i]);
                fprintf(stderr, "\n");
                fprintf(stderr, "[batch] L0 KB (post-RoPE) tok0 [0:8] = ");
                for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", kn[i]);
                fprintf(stderr, "\n");
            }
            int use_quant_kv_batch = (s->quant_key_cache && s->kv_quant_type < TQ_TYPE_COUNT);
            /* Always write FP32 K to s->key_cache — the batched attention
             * loop below reads from it. Baseline with use_quant_kv ONLY writes
             * to quant cache (saves memory) but we need the FP32 for our
             * batched attention (which doesn't call the traits dequantize).
             * The extra memory is negligible (same size as already-allocated
             * cache). */
            memcpy(s->key_cache + (size_t)l * kv_layer_stride + (size_t)pos * kv_dim,
                   kn, (size_t)kv_dim * sizeof(float));
            /* Also populate highres FP32 circular buffer when active. */
            if (use_quant_kv_batch && s->k_highres_window > 0 && s->key_highres_fp32) {
                int win_idx = pos % s->k_highres_window;
                size_t hr_layer_stride = (size_t)s->k_highres_window * kv_dim;
                float* hr_dst = s->key_highres_fp32
                    + (size_t)l * hr_layer_stride + (size_t)win_idx * kv_dim;
                memcpy(hr_dst, kn, (size_t)kv_dim * sizeof(float));
            }
            /* quant_key_cache write for baseline's attention to read later. */
            if (use_quant_kv_batch && !s->delta_kv_enabled) {
                const tq_type_traits_t* traits = &TQ_TRAITS[s->kv_quant_type];
                int cache_n_kv_heads = c->n_kv_heads;
                if (c->full_n_kv_heads > cache_n_kv_heads) cache_n_kv_heads = c->full_n_kv_heads;
                for (int kh = 0; kh < c->n_kv_heads; kh++) {
                    const float* key_src = kn + kh * c->head_dim;
                    uint8_t* quant_dst = (uint8_t*)s->quant_key_cache
                        + (size_t)l * s->quant_kv_stride
                        + (size_t)pos * cache_n_kv_heads * s->quant_head_stride
                        + (size_t)kh * s->quant_head_stride;
                    for (int blk = 0; blk < c->head_dim; blk += TQ_BK) {
                        int blen = c->head_dim - blk;
                        if (blen > TQ_BK) blen = TQ_BK;
                        traits->quantize(key_src + blk,
                                         quant_dst + (blk / TQ_BK) * traits->type_size,
                                         blen);
                    }
                }
            }
            if (s->value_cache) {
                memcpy(s->value_cache + (size_t)l * kv_layer_stride + (size_t)pos * kv_dim,
                       VB + (size_t)n * kv_dim, (size_t)kv_dim * sizeof(float));
            } else if (s->value_cache_fp16) {
                uint16_t* dst = s->value_cache_fp16
                              + (size_t)l * kv_layer_stride + (size_t)pos * kv_dim;
                f32_to_fp16_vec(VB + (size_t)n * kv_dim, dst, kv_dim);
            } else {
                if (dbg) fprintf(stderr, "[batch] bail: no FP32/FP16 V cache\n");
                free(X); free(Xres); free(XBN); free(QB); free(KB); free(VB);
                free(OB); free(GB); free(UB);
                return -1;
            }
        }

        /* 4. Attention (per-token, sequential — needs all preceding KV). */
        int n_kv_heads = c->n_kv_heads;
        int head_dim   = c->head_dim;
        int n_heads    = c->n_heads;
        int kv_mul     = n_heads / n_kv_heads;
        float*    K_layer = s->key_cache + (size_t)l * kv_layer_stride;
        float*    V_layer = s->value_cache ? (s->value_cache + (size_t)l * kv_layer_stride) : NULL;
        uint16_t* V_layer_fp16 = s->value_cache_fp16
                               ? (s->value_cache_fp16 + (size_t)l * kv_layer_stride) : NULL;

        for (int n = 0; n < N; n++) {
            int pos = pos_start + n;
            float* qn = QB + (size_t)n * q_dim;
            float* on = OB + (size_t)n * dim;
            for (int h = 0; h < n_heads; h++) {
                int kvh = h / kv_mul;
                float* qh = qn + h * head_dim;
                float* att = s->att + (size_t)h * c->max_seq_len;
                float scale = 1.0f / sqrtf((float)head_dim);
                for (int t = 0; t <= pos; t++) {
                    float* kh = K_layer + (size_t)t * kv_dim + kvh * head_dim;
                    float score = 0.0f;
#ifdef __ARM_NEON
                    float32x4_t vsum = vdupq_n_f32(0.0f);
                    int d = 0;
                    for (; d + 3 < head_dim; d += 4) {
                        float32x4_t vq = vld1q_f32(qh + d);
                        float32x4_t vk = vld1q_f32(kh + d);
                        vsum = vfmaq_f32(vsum, vq, vk);
                    }
                    score = vaddvq_f32(vsum);
                    for (; d < head_dim; d++) score += qh[d] * kh[d];
#else
                    for (int i = 0; i < head_dim; i++) score += qh[i] * kh[i];
#endif
                    att[t] = score * scale;
                }
                tq_softmax(att, pos + 1);
                float* oh = on + h * head_dim;
                memset(oh, 0, (size_t)head_dim * sizeof(float));
                if (V_layer) {
                    for (int t = 0; t <= pos; t++) {
                        float* vh = V_layer + (size_t)t * kv_dim + kvh * head_dim;
                        float w = att[t];
                        for (int i = 0; i < head_dim; i++) oh[i] += w * vh[i];
                    }
                } else {
                    /* FP16 V cache: use NEON vcvt_f32_f16 to exactly match the
                     * per-token attention path. */
                    for (int t = 0; t <= pos; t++) {
                        uint16_t* vh = V_layer_fp16 + (size_t)t * kv_dim + kvh * head_dim;
                        float w = att[t];
                        if (w == 0.0f) continue;
#ifdef __ARM_NEON
                        float32x4_t va = vdupq_n_f32(w);
                        int i = 0;
                        for (; i + 3 < head_dim; i += 4) {
                            uint16x4_t vh4 = vld1_u16(vh + i);
                            float32x4_t vf = vcvt_f32_f16(vreinterpret_f16_u16(vh4));
                            float32x4_t vx = vld1q_f32(oh + i);
                            vst1q_f32(oh + i, vfmaq_f32(vx, va, vf));
                        }
                        for (; i < head_dim; i++) {
                            uint16_t h16 = vh[i];
                            __fp16 hf = *(const __fp16*)&h16;
                            oh[i] += w * (float)hf;
                        }
#else
                        for (int i = 0; i < head_dim; i++) {
                            uint16_t h16 = vh[i];
                            __fp16 hf = *(const __fp16*)&h16;
                            oh[i] += w * (float)hf;
                        }
#endif
                    }
                }
            }
        }

        if (l == 3 && dbg) {
            /* Compute hash over OB to detect drift anywhere in the vector */
            for (int tn = 0; tn < N && tn < 2; tn++) {
                float* obr = OB + (size_t)tn * dim;
                double s=0, sabs=0;
                for (int i = 0; i < dim; i++) { s += obr[i]; sabs += (obr[i]<0?-obr[i]:obr[i]); }
                fprintf(stderr, "[batch] L3 OB tok%d sum=%.6f sumabs=%.6f [0:4]=%.6f,%.6f,%.6f,%.6f [dim/2]=%.6f\n",
                    tn, s, sabs, obr[0], obr[1], obr[2], obr[3], obr[dim/2]);
            }
        }
        /* 5. O matmul batched + Q2 residual */
        tq_batched_matmul_q4(X, layer->wo_q4, layer->wo_q4s, OB, dim, q_dim, N, NULL);
        if (layer->wo_q2) {
            int8_t* xq = s->xb_q8;
            float*  xs = s->xb_q8s;
            float* tmp = (float*)malloc((size_t)dim * sizeof(float));
            for (int n = 0; n < N; n++) {
                tq_quantize_row_q8(OB + (size_t)n * q_dim, xq, xs, q_dim);
                tq_matmul_q2_preq(tmp, layer->wo_q2, layer->wo_q2s, xq, xs, dim, q_dim);
                for (int i = 0; i < dim; i++) X[(size_t)n * dim + i] += tmp[i];
            }
            free(tmp);
        }

        if (l == 0 && dbg) {
            fprintf(stderr, "[batch] L0 X (after wo matmul) tok0 [0:8] = ");
            for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", X[i]);
            fprintf(stderr, "\n");
        }
        /* 6. Residual: Xres += X */
        for (size_t i = 0; i < (size_t)N * dim; i++) Xres[i] += X[i];

        if (l <= 3 && dbg) {
            for (int tn = 0; tn < N && tn < 2; tn++) {
                fprintf(stderr, "[batch] L%d after-attn-residual tok%d [0:8] = ", l, tn);
                for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", Xres[(size_t)tn * dim + i]);
                fprintf(stderr, "\n");
            }
        }

        /* 7. ffn_norm */
        for (int n = 0; n < N; n++) {
            tq_rmsnorm(XBN + (size_t)n * dim, Xres + (size_t)n * dim,
                       layer->ffn_norm, dim, c->rms_norm_eps);
        }

        /* 8. gate, up batched matmul + Q2 residuals */
        tq_batched_matmul_q4(GB, layer->w_gate_q4, layer->w_gate_q4s, XBN, inter, dim, N, NULL);
        tq_batched_matmul_q4(UB, layer->w_up_q4,   layer->w_up_q4s,   XBN, inter, dim, N, NULL);
        if (layer->w_gate_q2 || layer->w_up_q2) {
            int8_t* xq = s->xb_q8;
            float*  xs = s->xb_q8s;
            float* tmp = (float*)malloc((size_t)inter * sizeof(float));
            for (int n = 0; n < N; n++) {
                tq_quantize_row_q8(XBN + (size_t)n * dim, xq, xs, dim);
                if (layer->w_gate_q2) {
                    tq_matmul_q2_preq(tmp, layer->w_gate_q2, layer->w_gate_q2s, xq, xs, inter, dim);
                    for (int i = 0; i < inter; i++) GB[(size_t)n * inter + i] += tmp[i];
                }
                if (layer->w_up_q2) {
                    tq_matmul_q2_preq(tmp, layer->w_up_q2, layer->w_up_q2s, xq, xs, inter, dim);
                    for (int i = 0; i < inter; i++) UB[(size_t)n * inter + i] += tmp[i];
                }
            }
            free(tmp);
        }

        /* 9. SiLU(gate) * up (per-element) */
        for (size_t i = 0; i < (size_t)N * inter; i++) {
            float g = GB[i];
            float silu = g / (1.0f + expf(-g));
            GB[i] = silu * UB[i];
        }

        /* 10. down matmul batched (output back into X) + Q2 residual */
        tq_batched_matmul_q4(X, layer->w_down_q4, layer->w_down_q4s, GB, dim, inter, N, NULL);
        if (layer->w_down_q2) {
            int8_t* xq = s->xb_q8;
            float*  xs = s->xb_q8s;
            float* tmp = (float*)malloc((size_t)dim * sizeof(float));
            for (int n = 0; n < N; n++) {
                tq_quantize_row_q8(GB + (size_t)n * inter, xq, xs, inter);
                tq_matmul_q2_preq(tmp, layer->w_down_q2, layer->w_down_q2s, xq, xs, dim, inter);
                for (int i = 0; i < dim; i++) X[(size_t)n * dim + i] += tmp[i];
            }
            free(tmp);
        }

        /* 11. Residual: Xres += X */
        for (size_t i = 0; i < (size_t)N * dim; i++) Xres[i] += X[i];

        if ((l <= 3) && dbg) {
            for (int tn = 0; tn < N && tn < 2; tn++) {
                float* row = Xres + (size_t)tn * dim;
                double s=0, sabs=0;
                for (int i = 0; i < dim; i++) { s += row[i]; sabs += (row[i]<0?-row[i]:row[i]); }
                fprintf(stderr, "[batch] L%d final Xres tok%d sum=%.9f sumabs=%.9f\n",
                    l, tn, s, sabs);
            }
        }
    }

    /* Compute logits for the LAST token in the batch so the caller can
     * skip running tq_forward again. For non-DeltaNet models this is just
     * a convenience; for DeltaNet it's required to avoid double-advancing
     * the recurrent state. */
    {
        int last = N - 1;
        float* x_last = Xres + (size_t)last * dim;
        memcpy(s->x, x_last, (size_t)dim * sizeof(float));
        tq_rmsnorm(s->x, s->x, model->output_norm, dim, c->rms_norm_eps);
        if (model->output_gguf) {
            tq_matmul_gguf(s->logits, s->x, model->output_gguf,
                            model->output_gguf_type, c->vocab_size, dim);
        } else if (model->output_qs) {
            tq_matmul_q4(s->logits, s->x, model->output_qs, model->output_scales,
                          c->vocab_size, dim);
        } else if (model->output_weight_bf16) {
            tq_matmul_bf16(s->logits, s->x, model->output_weight_bf16, c->vocab_size, dim);
        } else if (model->output_weight) {
            tq_matmul(s->logits, s->x, model->output_weight, c->vocab_size, dim);
        }
    }

    free(X); free(XBN); free(QB); free(KB); free(VB); free(OB); free(GB); free(UB);
    free(Xres);
    return pos_start + N;
}
