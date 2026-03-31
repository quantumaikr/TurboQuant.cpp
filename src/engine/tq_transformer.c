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
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ============================================================
 * State management
 * ============================================================ */

tq_state_t* tq_create_state(const tq_model_config_t* config, tq_type kv_type) {
    if (!config) return NULL;

    int dim = config->hidden_dim;
    int kv_dim = config->n_kv_heads * config->head_dim;
    int inter_dim = config->intermediate_dim;
    int n_heads = config->n_heads;
    int max_seq = config->max_seq_len;
    int n_layers = config->n_layers;

    tq_state_t* s = (tq_state_t*)calloc(1, sizeof(tq_state_t));
    if (!s) return NULL;

    s->kv_quant_type = kv_type;

    /* Allocate activation buffers */
    /* For Qwen3.5, q dimension is n_heads * head_dim = 8 * 256 = 2048
     * but the DeltaNet qkv_dim is 6144 which is larger, so we need
     * the max of both for workspace.
     * When attn_output_gate is enabled, q_proj outputs 2x for Q + gate. */
    int q_dim = n_heads * config->head_dim;
    int q_proj_dim = config->attn_output_gate ? q_dim * 2 : q_dim;
    int delta_qkv_dim = 3 * config->delta_n_heads * config->delta_key_head_dim;
    int delta_z_dim = config->delta_n_heads * config->delta_value_head_dim;
    int max_dim = dim;
    if (q_dim > max_dim) max_dim = q_dim;
    if (q_proj_dim > max_dim) max_dim = q_proj_dim;
    if (delta_qkv_dim > max_dim) max_dim = delta_qkv_dim;

    s->x      = (float*)calloc((size_t)dim, sizeof(float));
    s->xb     = (float*)calloc((size_t)max_dim, sizeof(float));
    s->xb2    = (float*)calloc((size_t)max_dim, sizeof(float));
    s->q      = (float*)calloc((size_t)q_dim, sizeof(float));
    s->k      = (float*)calloc((size_t)kv_dim, sizeof(float));
    s->v      = (float*)calloc((size_t)kv_dim, sizeof(float));
    s->att    = (float*)calloc((size_t)n_heads * max_seq, sizeof(float));
    s->hb     = (float*)calloc((size_t)inter_dim, sizeof(float));
    s->hb2    = (float*)calloc((size_t)inter_dim, sizeof(float));
    s->logits = (float*)calloc((size_t)config->vocab_size, sizeof(float));

    /* KV cache for self_attn layers */
    size_t kv_layer_size = (size_t)max_seq * kv_dim;
    s->key_cache   = (float*)calloc((size_t)n_layers * kv_layer_size, sizeof(float));
    s->value_cache = (float*)calloc((size_t)n_layers * kv_layer_size, sizeof(float));
    s->kv_cache_size = (size_t)n_layers * kv_layer_size * sizeof(float);

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

    /* Verify critical allocations */
    if (!s->x || !s->xb || !s->xb2 || !s->q || !s->k || !s->v ||
        !s->att || !s->hb || !s->hb2 || !s->logits ||
        !s->key_cache || !s->value_cache ||
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
    tq_model_config_t* c = &model->config;
    tq_layer_weights_t* layer = &model->layers[l];
    int dim = c->hidden_dim;
    int dn = c->delta_n_heads;
    int dk = c->delta_key_head_dim;
    int dv = c->delta_value_head_dim;
    int qkv_dim = 3 * dn * dk;
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
    if (layer->delta_in_proj_qkv_q2)
        tq_matmul_q2_preq(s->delta_qkv, layer->delta_in_proj_qkv_q2, layer->delta_in_proj_qkv_q2s, s->xb_q8, s->xb_q8s, qkv_dim, dim);
    else if (layer->delta_in_proj_qkv_q4)
        tq_matmul_q4_preq(s->delta_qkv, layer->delta_in_proj_qkv_q4, layer->delta_in_proj_qkv_q4s, s->xb_q8, s->xb_q8s, qkv_dim, dim);
    else if (layer->delta_in_proj_qkv_q8)
        tq_matmul_q8(s->delta_qkv, s->xb, layer->delta_in_proj_qkv_q8, layer->delta_in_proj_qkv_q8s, qkv_dim, dim);
    else
        tq_matmul(s->delta_qkv, s->xb, layer->delta_in_proj_qkv, qkv_dim, dim);

    if (layer->delta_in_proj_z_q2)
        tq_matmul_q2_preq(s->delta_z, layer->delta_in_proj_z_q2, layer->delta_in_proj_z_q2s, s->xb_q8, s->xb_q8s, z_dim, dim);
    else if (layer->delta_in_proj_z_q4)
        tq_matmul_q4_preq(s->delta_z, layer->delta_in_proj_z_q4, layer->delta_in_proj_z_q4s, s->xb_q8, s->xb_q8s, z_dim, dim);
    else if (layer->delta_in_proj_z_q8)
        tq_matmul_q8(s->delta_z, s->xb, layer->delta_in_proj_z_q8, layer->delta_in_proj_z_q8s, z_dim, dim);
    else
        tq_matmul(s->delta_z, s->xb, layer->delta_in_proj_z, z_dim, dim);

    /* Step 2: Project alpha and beta */
    /* alpha = in_proj_a @ x  -> [dn] */
    if (layer->delta_in_proj_a_q2)
        tq_matmul_q2_preq(s->delta_ab, layer->delta_in_proj_a_q2, layer->delta_in_proj_a_q2s, s->xb_q8, s->xb_q8s, dn, dim);
    else if (layer->delta_in_proj_a_q4)
        tq_matmul_q4_preq(s->delta_ab, layer->delta_in_proj_a_q4, layer->delta_in_proj_a_q4s, s->xb_q8, s->xb_q8s, dn, dim);
    else if (layer->delta_in_proj_a_q8)
        tq_matmul_q8(s->delta_ab, s->xb, layer->delta_in_proj_a_q8, layer->delta_in_proj_a_q8s, dn, dim);
    else
        tq_matmul(s->delta_ab, s->xb, layer->delta_in_proj_a, dn, dim);

    /* beta = sigmoid(in_proj_b @ x) -> [dn] */
    if (layer->delta_in_proj_b_q2)
        tq_matmul_q2_preq(s->delta_ab + dn, layer->delta_in_proj_b_q2, layer->delta_in_proj_b_q2s, s->xb_q8, s->xb_q8s, dn, dim);
    else if (layer->delta_in_proj_b_q4)
        tq_matmul_q4_preq(s->delta_ab + dn, layer->delta_in_proj_b_q4, layer->delta_in_proj_b_q4s, s->xb_q8, s->xb_q8s, dn, dim);
    else if (layer->delta_in_proj_b_q8)
        tq_matmul_q8(s->delta_ab + dn, s->xb, layer->delta_in_proj_b_q8, layer->delta_in_proj_b_q8s, dn, dim);
    else
        tq_matmul(s->delta_ab + dn, s->xb, layer->delta_in_proj_b, dn, dim);
    for (int h = 0; h < dn; h++) {
        s->delta_ab[dn + h] = 1.0f / (1.0f + fast_expf(-s->delta_ab[dn + h]));
    }

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
    causal_conv1d_silu_batch(s->delta_qkv, conv_st, layer->delta_conv1d,
                              qkv_dim, conv_width);

    /* Step 5: Split into Q, K, V per head and L2 normalize Q, K */
    float* Q_all = s->delta_qkv;
    float* K_all = s->delta_qkv + dn * dk;
    float* V_all = s->delta_qkv + 2 * dn * dk;

    for (int h = 0; h < dn; h++) {
        l2_normalize(Q_all + h * dk, dk);
        l2_normalize(K_all + h * dk, dk);
    }

    /* Step 6: Scale Q by 1/sqrt(head_dim) */
    float q_scale = 1.0f / sqrtf((float)dk);
    for (int i = 0; i < dn * dk; i++) {
        Q_all[i] *= q_scale;
    }

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
        float* qh = Q_all + h * dk;
        float* kh = K_all + h * dk;
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
    if (layer->delta_out_proj_q2)
        tq_matmul_q2(s->xb2, s->delta_out, layer->delta_out_proj_q2, layer->delta_out_proj_q2s, dim, z_dim);
    else if (layer->delta_out_proj_q4)
        tq_matmul_q4(s->xb2, s->delta_out, layer->delta_out_proj_q4, layer->delta_out_proj_q4s, dim, z_dim);
    else if (layer->delta_out_proj_q8)
        tq_matmul_q8(s->xb2, s->delta_out, layer->delta_out_proj_q8, layer->delta_out_proj_q8s, dim, z_dim);
    else
        tq_matmul(s->xb2, s->delta_out, layer->delta_out_proj, dim, z_dim);

    /* Residual connection */
    tq_add(s->x, s->x, s->xb2, dim);
}

/* ============================================================
 * Self-attention forward pass with QK-norm and partial RoPE
 * ============================================================ */
static void self_attn_forward(tq_model_t* model, tq_state_t* s, int l, int pos) {
    tq_model_config_t* c = &model->config;
    tq_layer_weights_t* layer = &model->layers[l];
    int dim = c->hidden_dim;
    int head_dim = c->head_dim;
    int n_heads = c->n_heads;
    int n_kv_heads = c->n_kv_heads;
    int kv_dim = n_kv_heads * head_dim;
    int kv_mul = n_heads / n_kv_heads;
    size_t kv_layer_stride = (size_t)c->max_seq_len * kv_dim;

    /* Pre-quantize activation to Q8 once for all Q2/Q4 projections in this layer.
     * This eliminates redundant tq_quantize_row_q8 + malloc/free in each matmul call. */
    int has_q2 = (layer->wq_q2 != NULL);
    int has_q4 = (layer->wq_q4 != NULL);
    if (has_q2 || has_q4) {
        tq_quantize_row_q8(s->xb, s->xb_q8, s->xb_q8s, dim);
    }

    /* QKV projections.
     * When attn_output_gate is enabled, wq has shape [2*n_heads*head_dim, dim]
     * and outputs [Q, gate_q] concatenated. We project into xb2 as temp. */
    float* gate_q = NULL;
    if (c->attn_output_gate) {
        int qg_dim = n_heads * head_dim * 2;
        if (layer->wq_q2) {
            tq_matmul_q2_preq(s->xb2, layer->wq_q2, layer->wq_q2s, s->xb_q8, s->xb_q8s, qg_dim, dim);
        } else if (layer->wq_q4) {
            tq_matmul_q4_preq(s->xb2, layer->wq_q4, layer->wq_q4s, s->xb_q8, s->xb_q8s, qg_dim, dim);
        } else if (layer->wq_q8) {
            tq_matmul_q8(s->xb2, s->xb, layer->wq_q8, layer->wq_q8s, qg_dim, dim);
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
            tq_matmul_q2_preq(s->q, layer->wq_q2, layer->wq_q2s, s->xb_q8, s->xb_q8s, n_heads * head_dim, dim);
        } else if (layer->wq_q4) {
            tq_matmul_q4_preq(s->q, layer->wq_q4, layer->wq_q4s, s->xb_q8, s->xb_q8s, n_heads * head_dim, dim);
        } else if (layer->wq_q8) {
            tq_matmul_q8(s->q, s->xb, layer->wq_q8, layer->wq_q8s, n_heads * head_dim, dim);
        } else {
            tq_matmul(s->q, s->xb, layer->wq, n_heads * head_dim, dim);
        }
    }
    if (layer->wk_q2) {
        tq_matmul_q2_preq(s->k, layer->wk_q2, layer->wk_q2s, s->xb_q8, s->xb_q8s, kv_dim, dim);
    } else if (layer->wk_q4) {
        tq_matmul_q4_preq(s->k, layer->wk_q4, layer->wk_q4s, s->xb_q8, s->xb_q8s, kv_dim, dim);
    } else if (layer->wk_q8) {
        tq_matmul_q8(s->k, s->xb, layer->wk_q8, layer->wk_q8s, kv_dim, dim);
    } else {
        tq_matmul(s->k, s->xb, layer->wk, kv_dim, dim);
    }
    if (layer->wv_q2) {
        tq_matmul_q2_preq(s->v, layer->wv_q2, layer->wv_q2s, s->xb_q8, s->xb_q8s, kv_dim, dim);
    } else if (layer->wv_q4) {
        tq_matmul_q4_preq(s->v, layer->wv_q4, layer->wv_q4s, s->xb_q8, s->xb_q8s, kv_dim, dim);
    } else if (layer->wv_q8) {
        tq_matmul_q8(s->v, s->xb, layer->wv_q8, layer->wv_q8s, kv_dim, dim);
    } else {
        tq_matmul(s->v, s->xb, layer->wv, kv_dim, dim);
    }

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
        /* Apply RoPE only to the first rope_dim dimensions of each head */
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
    } else {
        /* Full RoPE — for Gemma3, use different freq base for sliding vs global layers */
        float rope_base = c->rope_freq_base;
        if (c->model_type == 1 && c->rope_local_base_freq > 0.0f &&
            model->layer_is_sliding && model->layer_is_sliding[l]) {
            rope_base = c->rope_local_base_freq;
        }
        tq_rope(s->q, s->k, pos, head_dim, n_heads, n_kv_heads, rope_base);
    }

    /* Store K,V in cache */
    float* key_cache_layer = s->key_cache + l * kv_layer_stride;
    float* val_cache_layer = s->value_cache + l * kv_layer_stride;
    memcpy(key_cache_layer + (size_t)pos * kv_dim, s->k, kv_dim * sizeof(float));
    memcpy(val_cache_layer + (size_t)pos * kv_dim, s->v, kv_dim * sizeof(float));

    /* Quantize the new key into the quantized cache for integer attention.
     * Each KV head's key vector is quantized independently into blocks. */
    int use_int_attn = (s->kv_quant_type < TQ_TYPE_COUNT && s->quant_key_cache != NULL);
    if (use_int_attn) {
        const tq_type_traits_t* traits = &TQ_TRAITS[s->kv_quant_type];
        for (int kh = 0; kh < n_kv_heads; kh++) {
            const float* key_src = s->k + kh * head_dim;
            /* Destination in quantized cache:
             * offset = layer * quant_kv_stride + pos * (n_kv_heads * quant_head_stride) + kh * quant_head_stride */
            uint8_t* quant_dst = (uint8_t*)s->quant_key_cache
                + (size_t)l * s->quant_kv_stride
                + (size_t)pos * n_kv_heads * s->quant_head_stride
                + (size_t)kh * s->quant_head_stride;
            traits->quantize(key_src, quant_dst, head_dim);
        }
    }

    /* Multi-head attention */
    int seq_len = pos + 1;
    /* Use integer attention when enough cached keys to amortize overhead */
    int int_attn_threshold = 128; /* only use integer attention for long contexts */

    /* Attention scaling: Gemma3 uses 1/sqrt(query_pre_attn_scalar), others use 1/sqrt(head_dim) */
    float attn_scale_dim = (float)head_dim;
    if (c->query_pre_attn_scalar > 0.0f) {
        attn_scale_dim = c->query_pre_attn_scalar;
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
            size_t pos_stride_bytes = (size_t)n_kv_heads * head_block_bytes;
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
        } else {
            /* FP32 attention scores (short sequences or no quantization) */
            float inv_scale = 1.0f / sqrtf(attn_scale_dim);
            /* Set positions outside sliding window to -inf */
            for (int t = 0; t < attn_start; t++) {
                atth[t] = -1e30f;
            }
            for (int t = attn_start; t < seq_len; t++) {
                const float* kt = key_cache_layer + (size_t)t * kv_dim + kv_h * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += qh[d] * kt[d];
                }
                atth[t] = score * inv_scale;
            }
        }

        /* Softmax */
        tq_softmax(atth, seq_len);

        /* Weighted sum of values */
        float* xbh = s->xb + h * head_dim;
        memset(xbh, 0, head_dim * sizeof(float));
        for (int t = 0; t < seq_len; t++) {
            const float* vt = val_cache_layer + (size_t)t * kv_dim + kv_h * head_dim;
            float a = atth[t];
            for (int d = 0; d < head_dim; d++) {
                xbh[d] += a * vt[d];
            }
        }
    }

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
    if (layer->wo_q2)
        tq_matmul_q2(s->xb2, s->xb, layer->wo_q2, layer->wo_q2s, dim, n_heads * head_dim);
    else if (layer->wo_q4)
        tq_matmul_q4(s->xb2, s->xb, layer->wo_q4, layer->wo_q4s, dim, n_heads * head_dim);
    else if (layer->wo_q8)
        tq_matmul_q8(s->xb2, s->xb, layer->wo_q8, layer->wo_q8s, dim, n_heads * head_dim);
    else
        tq_matmul(s->xb2, s->xb, layer->wo, dim, n_heads * head_dim);

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
    } else {
        memcpy(s->x, model->token_embedding + (size_t)token * dim,
               dim * sizeof(float));
    }

    /* Gemma3: scale embeddings by sqrt(hidden_dim) */
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

    /* Step 2: Transformer layers */
    int is_gemma3 = (c->model_type == 1);

    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];

        /* Pre-attention/DeltaNet RMSNorm */
        tq_rmsnorm(s->xb, s->x, layer->attn_norm, dim, c->rms_norm_eps);

        if (layer->delta_a_log) {
            /* DeltaNet layer */
            deltanet_forward(model, s, l);
        } else if ((layer->wq || layer->wq_q8 || layer->wq_q4) &&
                   (layer->wk || layer->wk_q8 || layer->wk_q4) &&
                   (layer->wv || layer->wv_q8 || layer->wv_q4)) {
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

        /* FFN Block — SwiGLU (Qwen3.5) or GeGLU (Gemma3).
         * Optimization: cache Q8 quantization of xb for gate+up projections,
         * and cache Q8 of hb for down projection. */
        if ((layer->w_gate || layer->w_gate_q8 || layer->w_gate_q4 || layer->w_gate_q2) &&
            (layer->w_up || layer->w_up_q8 || layer->w_up_q4 || layer->w_up_q2) &&
            (layer->w_down || layer->w_down_q8 || layer->w_down_q4 || layer->w_down_q2)) {

            /* Pre-FFN norm: Gemma3 uses pre_feedforward_layernorm,
             * Qwen3.5 uses post_attention_layernorm (stored as ffn_norm) */
            float* ffn_norm_w = layer->ffn_norm;
            if (is_gemma3 && layer->pre_ffn_norm) {
                ffn_norm_w = layer->pre_ffn_norm;
            }
            tq_rmsnorm(s->xb, s->x, ffn_norm_w, dim, c->rms_norm_eps);

            /* Pre-quantize xb for gate+up Q2/Q4 projections (same input, 2 matmuls) */
            if (layer->w_gate_q2) {
                tq_quantize_row_q8(s->xb, s->xb_q8, s->xb_q8s, dim);

                tq_matmul_q2_preq(s->hb, layer->w_gate_q2, layer->w_gate_q2s,
                                   s->xb_q8, s->xb_q8s, c->intermediate_dim, dim);
                tq_matmul_q2_preq(s->hb2, layer->w_up_q2, layer->w_up_q2s,
                                   s->xb_q8, s->xb_q8s, c->intermediate_dim, dim);
            } else if (layer->w_gate_q4) {
                tq_quantize_row_q8(s->xb, s->xb_q8, s->xb_q8s, dim);

                tq_matmul_q4_preq(s->hb, layer->w_gate_q4, layer->w_gate_q4s,
                                   s->xb_q8, s->xb_q8s, c->intermediate_dim, dim);
                tq_matmul_q4_preq(s->hb2, layer->w_up_q4, layer->w_up_q4s,
                                   s->xb_q8, s->xb_q8s, c->intermediate_dim, dim);
            } else {
                if (layer->w_gate_q8) {
                    tq_matmul_q8(s->hb, s->xb, layer->w_gate_q8, layer->w_gate_q8s, c->intermediate_dim, dim);
                } else {
                    tq_matmul(s->hb, s->xb, layer->w_gate, c->intermediate_dim, dim);
                }
                if (layer->w_up_q8) {
                    tq_matmul_q8(s->hb2, s->xb, layer->w_up_q8, layer->w_up_q8s, c->intermediate_dim, dim);
                } else {
                    tq_matmul(s->hb2, s->xb, layer->w_up, c->intermediate_dim, dim);
                }
            }

            /* Activation: GeGLU for Gemma3, SwiGLU for others */
            if (is_gemma3) {
                tq_gelu_tanh(s->hb, c->intermediate_dim);
            } else {
                tq_silu(s->hb, c->intermediate_dim);
            }
            tq_mul(s->hb, s->hb, s->hb2, c->intermediate_dim);

            if (layer->w_down_q2) {
                tq_matmul_q2(s->xb2, s->hb, layer->w_down_q2, layer->w_down_q2s, dim, c->intermediate_dim);
            } else if (layer->w_down_q4) {
                tq_matmul_q4(s->xb2, s->hb, layer->w_down_q4, layer->w_down_q4s, dim, c->intermediate_dim);
            } else if (layer->w_down_q8) {
                tq_matmul_q8(s->xb2, s->hb, layer->w_down_q8, layer->w_down_q8s, dim, c->intermediate_dim);
            } else {
                tq_matmul(s->xb2, s->hb, layer->w_down, dim, c->intermediate_dim);
            }

            /* Gemma3: apply post_feedforward_layernorm to FFN output before residual */
            if (is_gemma3 && layer->post_ffn_norm) {
                tq_rmsnorm(s->xb2, s->xb2, layer->post_ffn_norm, dim, c->rms_norm_eps);
            }

            tq_add(s->x, s->x, s->xb2, dim);
        }

        /* Debug: print layer output */
        if (pos == 0 && getenv("TQ_DEBUG") && (l == 0 || l == 5 || l == c->n_layers - 1)) {
            fprintf(stderr, "[DEBUG] layer%d out[0:8] = ", l);
            for (int i = 0; i < 8 && i < dim; i++) fprintf(stderr, "%.4f ", s->x[i]);
            fprintf(stderr, "\n");
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
    if (model->output_qs) {
        tq_matmul_q4(s->logits, s->x, model->output_qs, model->output_scales,
                      c->vocab_size, dim);
    } else if (model->output_weight_bf16) {
        tq_matmul_bf16(s->logits, s->x, model->output_weight_bf16, c->vocab_size, dim);
    } else {
        tq_matmul(s->logits, s->x, model->output_weight, c->vocab_size, dim);
    }

    return s->logits;
}
