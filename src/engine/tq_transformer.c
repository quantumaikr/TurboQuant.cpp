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
        !s->key_cache || !s->value_cache) {
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
    free(state->quant_key_buf);
    free(state->quant_score_buf);
    free(state->quant_key_cache);
    free(state);
}

/* ============================================================
 * Helper: L2 normalize a vector in-place
 * ============================================================ */
static void l2_normalize(float* v, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += v[i] * v[i];
    if (ss > 0.0f) {
        float inv = 1.0f / sqrtf(ss);
        for (int i = 0; i < n; i++) v[i] *= inv;
    }
}

/* ============================================================
 * Helper: Apply causal conv1d (width=conv_width) for a single
 * channel at the current time step.
 *
 * conv_state holds the last (conv_width-1) inputs for this channel.
 * weight has conv_width values.
 * Returns the convolution output for the current input.
 * ============================================================ */
static float causal_conv1d_step(float input, float* conv_buf,
                                 const float* weight, int conv_width) {
    /* conv_buf holds the previous (conv_width - 1) inputs:
     *   conv_buf[0] = x[t-K+1], ..., conv_buf[K-2] = x[t-1]
     * The current input x[t] is NOT in the buffer yet.
     *
     * Causal conv1d output at time t:
     *   out = sum_{k=0}^{K-2} weight[k] * conv_buf[k]  +  weight[K-1] * input
     *
     * After computing, shift buffer left and insert input for next step. */
    int buf_len = conv_width - 1;

    /* Compute output BEFORE updating buffer */
    float out = 0.0f;
    for (int k = 0; k < buf_len; k++) {
        out += weight[k] * conv_buf[k];
    }
    out += weight[buf_len] * input;

    /* Now update buffer: shift left and insert current input */
    for (int i = 0; i < buf_len - 1; i++) {
        conv_buf[i] = conv_buf[i + 1];
    }
    conv_buf[buf_len - 1] = input;

    return out;
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

    /* Step 1: Project input through QKV and Z */
    tq_matmul(s->delta_qkv, s->xb, layer->delta_in_proj_qkv, qkv_dim, dim);
    tq_matmul(s->delta_z, s->xb, layer->delta_in_proj_z, z_dim, dim);

    /* Step 2: Project alpha and beta */
    /* alpha = in_proj_a @ x  -> [dn] */
    tq_matmul(s->delta_ab, s->xb, layer->delta_in_proj_a, dn, dim);
    /* beta = sigmoid(in_proj_b @ x) -> [dn] */
    tq_matmul(s->delta_ab + dn, s->xb, layer->delta_in_proj_b, dn, dim);
    for (int h = 0; h < dn; h++) {
        s->delta_ab[dn + h] = 1.0f / (1.0f + expf(-s->delta_ab[dn + h]));
    }

    /* Step 3: Compute gate (decay) per head
     * In the GGUF conversion, A_log is stored as -exp(A_log).
     * Since we load from safetensors (raw A_log), we compute:
     *   gate = softplus(alpha + dt_bias) * (-exp(A_log))
     * This produces a negative gate value (decay in log space).
     * exp(gate) is the per-step multiplicative decay (< 1). */
    float gate_vals[128]; /* max 128 heads, stack-allocated for speed */
    for (int h = 0; h < dn; h++) {
        float alpha_biased = s->delta_ab[h] + layer->delta_dt_bias[h];
        float alpha_sp = logf(1.0f + expf(alpha_biased)); /* softplus */
        /* A_log in safetensors is raw; compute -exp(A_log) as the decay factor */
        float neg_exp_alog = -expf(layer->delta_a_log[h]);
        gate_vals[h] = alpha_sp * neg_exp_alog;
    }

    /* Step 4: Causal conv1d on QKV, then SiLU */
    for (int ch = 0; ch < qkv_dim; ch++) {
        float* ch_conv_buf = conv_st + ch * conv_buf_len;
        const float* ch_weight = layer->delta_conv1d + ch * conv_width;
        s->delta_qkv[ch] = causal_conv1d_step(s->delta_qkv[ch],
                                                ch_conv_buf, ch_weight,
                                                conv_width);
    }
    /* SiLU activation on conv output */
    for (int i = 0; i < qkv_dim; i++) {
        s->delta_qkv[i] = s->delta_qkv[i] / (1.0f + expf(-s->delta_qkv[i]));
    }

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

    /* Step 7: Per-head recurrent delta rule update.
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

        /* Decay: S = S * exp(gate) */
        float decay = expf(gate_vals[h]);
        for (int i = 0; i < dk * dv; i++) {
            sh[i] *= decay;
        }

        /* Compute sk = S @ K per value dimension:
         * sk[j] = sum_i(S[i][j] * K[i]) for j=0..dv-1
         * Actually, following the reference more carefully:
         * sk = sum_rows(S * K) where K is broadcast across value dim.
         * This means sk[j] = sum_i(S[i,j] * K[i]) */
        float sk[128]; /* max head_dim */
        for (int j = 0; j < dv; j++) {
            float sum = 0.0f;
            for (int i = 0; i < dk; i++) {
                sum += sh[i * dv + j] * kh[i];
            }
            sk[j] = sum;
        }

        /* Delta: d = beta * (V - sk) */
        float d[128];
        for (int j = 0; j < dv; j++) {
            d[j] = beta_h * (vh[j] - sk[j]);
        }

        /* State update: S = S + outer(K, d)
         * S[i][j] += K[i] * d[j] */
        for (int i = 0; i < dk; i++) {
            for (int j = 0; j < dv; j++) {
                sh[i * dv + j] += kh[i] * d[j];
            }
        }

        /* Output: o = S @ Q
         * o[j] = sum_i(S[i,j] * Q[i]) */
        float* oh = s->delta_out + h * dv;
        for (int j = 0; j < dv; j++) {
            float sum = 0.0f;
            for (int i = 0; i < dk; i++) {
                sum += sh[i * dv + j] * qh[i];
            }
            oh[j] = sum;
        }
    }

    /* Step 8: Apply group norm (per-head RMSNorm), then z gate (swish), then output projection */
    /* norm(output, z): normalized = RMSNorm(output), gated = silu(z), result = normalized * gated */
    for (int h = 0; h < dn; h++) {
        float* oh = s->delta_out + h * dv;

        /* RMSNorm with delta_norm weights */
        float ss = 0.0f;
        for (int j = 0; j < dv; j++) {
            ss += oh[j] * oh[j];
        }
        ss = ss / dv + c->rms_norm_eps;
        float inv_rms = 1.0f / sqrtf(ss);
        for (int j = 0; j < dv; j++) {
            oh[j] = oh[j] * inv_rms * layer->delta_norm[j];
        }

        /* Multiply by swish(z) for this head */
        float* zh = s->delta_z + h * dv;
        for (int j = 0; j < dv; j++) {
            float z_val = zh[j];
            float z_silu = z_val / (1.0f + expf(-z_val));
            oh[j] *= z_silu;
        }
    }

    /* Output projection: [dim, z_dim] @ delta_out[z_dim] -> xb2[dim] */
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

    /* QKV projections.
     * When attn_output_gate is enabled, wq has shape [2*n_heads*head_dim, dim]
     * and outputs [Q, gate_q] concatenated. We project into xb2 as temp. */
    float* gate_q = NULL;
    if (c->attn_output_gate) {
        /* Project full Q+gate interleaved:
         * wq output is [n_heads * head_dim * 2] arranged as:
         *   [Q_head0(head_dim), Gate_head0(head_dim), Q_head1, Gate_head1, ...]
         * We need to deinterleave into Q and gate. */
        int qg_dim = n_heads * head_dim * 2;
        tq_matmul(s->xb2, s->xb, layer->wq, qg_dim, dim);
        /* Deinterleave: extract Q and gate from interleaved layout */
        gate_q = s->xb2; /* reuse xb2 for gate after we extract Q */
        /* We need a temp buffer. Use att buffer as it's large enough. */
        float* gate_tmp = s->att; /* repurpose att temporarily */
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
        tq_matmul(s->q, s->xb, layer->wq, n_heads * head_dim, dim);
    }
    tq_matmul(s->k, s->xb, layer->wk, kv_dim, dim);
    tq_matmul(s->v, s->xb, layer->wv, kv_dim, dim);

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
        /* Full RoPE */
        tq_rope(s->q, s->k, pos, head_dim, n_heads, n_kv_heads,
                c->rope_freq_base);
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
             * apply 1/sqrt(head_dim) scaling */
            float scale = 1.0f / sqrtf((float)head_dim);
            for (int t = 0; t < seq_len; t++) {
                atth[t] *= scale;
            }
        } else {
            /* FP32 attention scores (short sequences or no quantization) */
            for (int t = 0; t < seq_len; t++) {
                const float* kt = key_cache_layer + (size_t)t * kv_dim + kv_h * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += qh[d] * kt[d];
                }
                atth[t] = score / sqrtf((float)head_dim);
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
        for (int i = 0; i < n_heads * head_dim; i++) {
            float g = 1.0f / (1.0f + expf(-gate_q[i]));
            s->xb[i] *= g;
        }
    }

    /* Output projection */
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

    /* Step 2: Transformer layers */
    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];

        /* Pre-attention/DeltaNet RMSNorm */
        tq_rmsnorm(s->xb, s->x, layer->attn_norm, dim, c->rms_norm_eps);

        if (layer->delta_a_log) {
            /* DeltaNet layer */
            deltanet_forward(model, s, l);
        } else if (layer->wq && layer->wk && layer->wv) {
            /* Standard self-attention layer */
            self_attn_forward(model, s, l, pos);
        }
        /* else: skip (should not happen for valid models) */

        /* FFN Block (SwiGLU) — present on ALL layers */
        if (layer->w_gate && layer->w_up && layer->w_down) {
            tq_rmsnorm(s->xb, s->x, layer->ffn_norm, dim, c->rms_norm_eps);
            tq_matmul(s->hb,  s->xb, layer->w_gate, c->intermediate_dim, dim);
            tq_matmul(s->hb2, s->xb, layer->w_up,   c->intermediate_dim, dim);
            tq_silu(s->hb, c->intermediate_dim);
            tq_mul(s->hb, s->hb, s->hb2, c->intermediate_dim);
            tq_matmul(s->xb2, s->hb, layer->w_down, dim, c->intermediate_dim);
            tq_add(s->x, s->x, s->xb2, dim);
        }
    }

    /* Step 3: Final RMSNorm */
    tq_rmsnorm(s->x, s->x, model->output_norm, dim, c->rms_norm_eps);

    /* Step 4: Output projection to vocab logits */
    if (model->output_weight_bf16) {
        tq_matmul_bf16(s->logits, s->x, model->output_weight_bf16, c->vocab_size, dim);
    } else {
        tq_matmul(s->logits, s->x, model->output_weight, c->vocab_size, dim);
    }

    return s->logits;
}
