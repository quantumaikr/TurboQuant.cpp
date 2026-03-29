/**
 * debug_compare.c -- Compare C engine forward pass with PyTorch reference
 *
 * Loads the model, runs forward pass on token 9419 ("Hello") using
 * the real tq_forward(), then compares per-layer outputs with saved
 * PyTorch reference from /tmp/tq_ref/.npy files.
 *
 * Since tq_forward() processes all layers internally and we only get
 * the final logits back, this tool hooks in by running layer-by-layer
 * manually, calling the exact same functions as tq_forward().
 *
 * To avoid code duplication, we expose deltanet_forward and
 * self_attn_forward by making them non-static (via a small shim).
 * Instead, we just run tq_forward and add per-layer debug output
 * to tq_transformer.c via environment variable TQ_DEBUG.
 *
 * Alternative approach used here: we replicate only the OUTER loop
 * of tq_forward, calling the real ops functions, and compare after
 * each layer.
 */

#include "turboquant/tq_engine.h"
#include "turboquant/turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================
 * NPY file reader
 * ============================================================ */
static float* load_npy(const char* path, int* out_n) {
    FILE* f = fopen(path, "rb");
    if (!f) { *out_n = 0; return NULL; }

    unsigned char header[10];
    if (fread(header, 1, 10, f) != 10) { fclose(f); *out_n = 0; return NULL; }
    if (header[0] != 0x93 || header[1] != 'N' || header[2] != 'U' ||
        header[3] != 'M' || header[4] != 'P' || header[5] != 'Y') {
        fclose(f); *out_n = 0; return NULL;
    }

    int major = header[6];
    unsigned int header_len = 0;
    if (major == 1) {
        header_len = header[8] | ((unsigned int)header[9] << 8);
    } else if (major == 2) {
        unsigned char extra[2];
        if (fread(extra, 1, 2, f) != 2) { fclose(f); *out_n = 0; return NULL; }
        header_len = header[8] | ((unsigned int)header[9] << 8) |
                     ((unsigned int)extra[0] << 16) | ((unsigned int)extra[1] << 24);
    }

    char* hdr_str = (char*)malloc(header_len + 1);
    if (fread(hdr_str, 1, header_len, f) != header_len) {
        free(hdr_str); fclose(f); *out_n = 0; return NULL;
    }
    hdr_str[header_len] = '\0';

    int total_elements = 1;
    char* shape_start = strstr(hdr_str, "'shape':");
    if (!shape_start) shape_start = strstr(hdr_str, "\"shape\":");
    if (shape_start) {
        char* paren = strchr(shape_start, '(');
        if (paren) {
            paren++;
            total_elements = 1;
            int found_dim = 0;
            while (*paren && *paren != ')') {
                while (*paren == ' ' || *paren == ',') paren++;
                if (*paren == ')') break;
                int dim = atoi(paren);
                if (dim > 0) { total_elements *= dim; found_dim = 1; }
                while (*paren && *paren != ',' && *paren != ')') paren++;
            }
            if (!found_dim) total_elements = 0;
        }
    }
    free(hdr_str);
    if (total_elements <= 0) { fclose(f); *out_n = 0; return NULL; }

    float* data = (float*)malloc((size_t)total_elements * sizeof(float));
    if (!data) { fclose(f); *out_n = 0; return NULL; }
    size_t rd = fread(data, sizeof(float), (size_t)total_elements, f);
    fclose(f);
    *out_n = (int)rd;
    return data;
}

/* ============================================================
 * Comparison metrics
 * ============================================================ */
static float cosine_similarity(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na < 1e-30 || nb < 1e-30) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

static float compute_mse(const float* a, const float* b, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        sum += d * d;
    }
    return (float)(sum / n);
}

static float max_abs_diff(const float* a, const float* b, int n, int* max_idx) {
    float max_d = 0; int idx = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_d) { max_d = d; idx = i; }
    }
    if (max_idx) *max_idx = idx;
    return max_d;
}

static int compare_with_ref(const char* name, const float* computed, int dim) {
    char path[512];
    snprintf(path, sizeof(path), "/tmp/tq_ref/%s.npy", name);
    int ref_n = 0;
    float* ref = load_npy(path, &ref_n);
    if (!ref) {
        printf("  %-30s  [no reference file]\n", name);
        return -1;
    }
    int cmp_n = ref_n < dim ? ref_n : dim;
    float cos = cosine_similarity(computed, ref, cmp_n);
    float mse = compute_mse(computed, ref, cmp_n);
    int max_idx = 0;
    float max_d = max_abs_diff(computed, ref, cmp_n, &max_idx);

    const char* status;
    if (cos > 0.9999f)     status = "MATCH";
    else if (cos > 0.999f) status = "CLOSE";
    else if (cos > 0.99f)  status = "DRIFT";
    else if (cos > 0.9f)   status = "DIVERGE";
    else                   status = "WRONG";

    printf("  %-30s cos=%.6f  mse=%.2e  max_diff=%.4f @[%d]  %s\n",
           name, cos, mse, max_d, max_idx, status);

    if (cos < 0.999f) {
        printf("    C:   [");
        for (int i = 0; i < 5 && i < cmp_n; i++)
            printf("%.6f%s", computed[i], i < 4 ? ", " : "");
        printf("]\n    Ref: [");
        for (int i = 0; i < 5 && i < cmp_n; i++)
            printf("%.6f%s", ref[i], i < 4 ? ", " : "");
        printf("]\n");
        if (max_idx > 2) {
            int s = max_idx - 2;
            printf("    C   @[%d..]: [", s);
            for (int i = s; i < s + 5 && i < cmp_n; i++)
                printf("%.6f%s", computed[i], i < s + 4 ? ", " : "");
            printf("]\n    Ref @[%d..]: [", s);
            for (int i = s; i < s + 5 && i < cmp_n; i++)
                printf("%.6f%s", ref[i], i < s + 4 ? ", " : "");
            printf("]\n");
        }
    }
    int is_match = (cos > 0.999f) ? 1 : 0;
    free(ref);
    return is_match;
}

/* ============================================================
 * Main: run tq_forward and compare results
 * ============================================================ */
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.safetensors>\n", argv[0]);
        return 1;
    }

    /* Check reference data */
    { int n; float* t = load_npy("/tmp/tq_ref/embed.npy", &n);
      if (!t) { fprintf(stderr, "No ref data. Run: python3 tools/debug_deltanet.py\n"); return 1; }
      free(t); }

    fprintf(stderr, "Loading model...\n");
    tq_model_t* model = tq_load_model(argv[1]);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    tq_model_config_t* c = &model->config;
    fprintf(stderr, "Model: %d layers, dim=%d, heads=%d/%d, vocab=%d\n",
            c->n_layers, c->hidden_dim, c->n_heads, c->n_kv_heads, c->vocab_size);

    tq_state_t* s = tq_create_state(c, TQ_TYPE_COUNT);
    if (!s) { fprintf(stderr, "Failed to create state\n"); tq_free_model(model); return 1; }

    int token = 9419;
    int dim = c->hidden_dim;
    int pos = 0;

    printf("=== Debug Forward Pass (using real tq_forward code path) ===\n");
    printf("Token: %d, dim=%d, n_layers=%d\n", token, dim, c->n_layers);
    printf("DeltaNet: n_heads=%d, key_dim=%d, val_dim=%d, conv_w=%d\n",
           c->delta_n_heads, c->delta_key_head_dim,
           c->delta_value_head_dim, c->delta_conv_width);
    printf("Attn: n_heads=%d, n_kv_heads=%d, head_dim=%d\n\n",
           c->n_heads, c->n_kv_heads, c->head_dim);

    /* Run the REAL tq_forward - single call */
    float* logits = tq_forward(model, s, token, pos);

    /* After tq_forward completes, s->x has final_norm output (overwritten by logits calc),
     * but s->logits has the logits. We can compare logits directly.
     * For per-layer comparison, we need a different approach.
     *
     * Let's re-run layer by layer using the real functions: */

    /* Reset state */
    tq_free_state(s);
    s = tq_create_state(c, TQ_TYPE_COUNT);

    /* Embedding */
    memcpy(s->x, model->token_embedding + (size_t)token * dim, dim * sizeof(float));
    printf("--- Embedding ---\n");
    compare_with_ref("embed", s->x, dim);
    printf("\n");

    int first_diverge = -1;
    int n_layers = c->n_layers;

    for (int l = 0; l < n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];
        int is_deltanet = (layer->delta_a_log != NULL);

        printf("--- Layer %d (%s) ---\n", l,
               is_deltanet ? "DeltaNet" : "self_attn");

        /* Pre-attention RMSNorm */
        tq_rmsnorm(s->xb, s->x, layer->attn_norm, dim, c->rms_norm_eps);
        if (l < 4) {
            char nm[64]; snprintf(nm, 64, "layer%02d_attn_norm", l);
            compare_with_ref(nm, s->xb, dim);
        }

        /* Save x before attention/deltanet for residual */
        /* The tq_forward code runs deltanet_forward or self_attn_forward
         * which internally handles residual. Since these are static,
         * we need to call tq_forward for a single layer. Instead,
         * let's replicate the outer loop manually using the public ops. */

        if (is_deltanet) {
            /* ---- DeltaNet forward (matching tq_transformer.c exactly) ---- */
            int dn = c->delta_n_heads;
            int dk = c->delta_key_head_dim;
            int dv = c->delta_value_head_dim;
            int qkv_dim = 3 * dn * dk;
            int z_dim = dn * dv;
            int conv_width = c->delta_conv_width;
            int conv_buf_len = conv_width - 1;
            if (conv_buf_len < 1) conv_buf_len = 1;

            float* state = s->delta_state + (size_t)l * dn * dk * dv;
            float* conv_st = s->conv_state + (size_t)l * qkv_dim * conv_buf_len;

            /* Project QKV, Z, a, b */
            tq_matmul(s->delta_qkv, s->xb, layer->delta_in_proj_qkv, qkv_dim, dim);
            tq_matmul(s->delta_z, s->xb, layer->delta_in_proj_z, z_dim, dim);
            tq_matmul(s->delta_ab, s->xb, layer->delta_in_proj_a, dn, dim);
            tq_matmul(s->delta_ab + dn, s->xb, layer->delta_in_proj_b, dn, dim);
            for (int h = 0; h < dn; h++)
                s->delta_ab[dn + h] = 1.0f / (1.0f + expf(-s->delta_ab[dn + h]));

            /* Gate computation */
            float gate_vals[128];
            for (int h = 0; h < dn; h++) {
                float alpha_biased = s->delta_ab[h] + layer->delta_dt_bias[h];
                float alpha_sp = logf(1.0f + expf(alpha_biased));
                float neg_exp_alog = -expf(layer->delta_a_log[h]);
                gate_vals[h] = alpha_sp * neg_exp_alog;
            }

            /* Conv1d + SiLU (FIXED: compute before updating buffer) */
            for (int ch = 0; ch < qkv_dim; ch++) {
                float* ch_conv_buf = conv_st + ch * conv_buf_len;
                const float* ch_weight = layer->delta_conv1d + ch * conv_width;
                float input = s->delta_qkv[ch];

                /* Compute output BEFORE updating buffer */
                float out = 0.0f;
                for (int k = 0; k < conv_buf_len; k++)
                    out += ch_weight[k] * ch_conv_buf[k];
                out += ch_weight[conv_buf_len] * input;

                /* Update buffer */
                for (int i = 0; i < conv_buf_len - 1; i++)
                    ch_conv_buf[i] = ch_conv_buf[i + 1];
                ch_conv_buf[conv_buf_len - 1] = input;

                s->delta_qkv[ch] = out;
            }
            /* SiLU */
            for (int i = 0; i < qkv_dim; i++)
                s->delta_qkv[i] = s->delta_qkv[i] / (1.0f + expf(-s->delta_qkv[i]));

            /* Compare after conv+silu */
            if (l == 0) {
                int ref_n = 0;
                float* ref = load_npy("/tmp/tq_ref/l0_after_conv_silu.npy", &ref_n);
                if (ref) {
                    float cos = cosine_similarity(s->delta_qkv, ref, qkv_dim < ref_n ? qkv_dim : ref_n);
                    printf("  %-30s cos=%.6f\n", "conv+silu (vs ref)", cos);
                    if (cos < 0.999f) {
                        printf("    C:   [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                               s->delta_qkv[0], s->delta_qkv[1], s->delta_qkv[2],
                               s->delta_qkv[3], s->delta_qkv[4]);
                        printf("    Ref: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                               ref[0], ref[1], ref[2], ref[3], ref[4]);
                    }
                    free(ref);
                }
            }

            /* Split Q, K, V and L2 normalize */
            float* Q_all = s->delta_qkv;
            float* K_all = s->delta_qkv + dn * dk;
            float* V_all = s->delta_qkv + 2 * dn * dk;

            for (int h = 0; h < dn; h++) {
                float ss;
                ss = 0.0f;
                for (int i = 0; i < dk; i++) ss += Q_all[h*dk+i] * Q_all[h*dk+i];
                if (ss > 0.0f) { float inv = 1.0f / sqrtf(ss); for (int i = 0; i < dk; i++) Q_all[h*dk+i] *= inv; }
                ss = 0.0f;
                for (int i = 0; i < dk; i++) ss += K_all[h*dk+i] * K_all[h*dk+i];
                if (ss > 0.0f) { float inv = 1.0f / sqrtf(ss); for (int i = 0; i < dk; i++) K_all[h*dk+i] *= inv; }
            }

            /* Scale Q */
            float q_scale = 1.0f / sqrtf((float)dk);
            for (int i = 0; i < dn * dk; i++) Q_all[i] *= q_scale;

            /* Recurrent delta rule */
            for (int h = 0; h < dn; h++) {
                float* qh = Q_all + h * dk;
                float* kh = K_all + h * dk;
                float* vh = V_all + h * dv;
                float* sh = state + (size_t)h * dk * dv;
                float beta_h = s->delta_ab[dn + h];

                float decay = expf(gate_vals[h]);
                for (int i = 0; i < dk * dv; i++) sh[i] *= decay;

                float sk[256];
                for (int j = 0; j < dv; j++) {
                    float sum = 0.0f;
                    for (int i = 0; i < dk; i++) sum += sh[i * dv + j] * kh[i];
                    sk[j] = sum;
                }
                float d[256];
                for (int j = 0; j < dv; j++) d[j] = beta_h * (vh[j] - sk[j]);

                for (int i = 0; i < dk; i++)
                    for (int j = 0; j < dv; j++)
                        sh[i * dv + j] += kh[i] * d[j];

                float* oh = s->delta_out + h * dv;
                for (int j = 0; j < dv; j++) {
                    float sum = 0.0f;
                    for (int i = 0; i < dk; i++) sum += sh[i * dv + j] * qh[i];
                    oh[j] = sum;
                }
            }

            /* Group norm + z gate */
            for (int h = 0; h < dn; h++) {
                float* oh = s->delta_out + h * dv;
                float ss = 0.0f;
                for (int j = 0; j < dv; j++) ss += oh[j] * oh[j];
                ss = ss / dv + c->rms_norm_eps;
                float inv_rms = 1.0f / sqrtf(ss);
                for (int j = 0; j < dv; j++)
                    oh[j] = oh[j] * inv_rms * layer->delta_norm[j];

                float* zh = s->delta_z + h * dv;
                for (int j = 0; j < dv; j++) {
                    float z_val = zh[j];
                    float z_silu = z_val / (1.0f + expf(-z_val));
                    oh[j] *= z_silu;
                }
            }

            /* Out proj */
            tq_matmul(s->xb2, s->delta_out, layer->delta_out_proj, dim, z_dim);

            if (l < 4) {
                char nm[64]; snprintf(nm, 64, "layer%02d_linear_attn", l);
                compare_with_ref(nm, s->xb2, dim);
            }

            /* Residual */
            tq_add(s->x, s->x, s->xb2, dim);

        } else if (layer->wq && layer->wk && layer->wv) {
            /* ---- Self-attention (matching tq_transformer.c exactly) ---- */
            int head_dim_h = c->head_dim;
            int n_heads_h = c->n_heads;
            int n_kv_heads_h = c->n_kv_heads;
            int kv_dim_h = n_kv_heads_h * head_dim_h;
            int kv_mul_h = n_heads_h / n_kv_heads_h;
            size_t kv_layer_stride = (size_t)c->max_seq_len * kv_dim_h;

            float* gate_q = NULL;
            if (c->attn_output_gate) {
                int qg_dim = n_heads_h * head_dim_h * 2;
                tq_matmul(s->xb2, s->xb, layer->wq, qg_dim, dim);
                float* gate_tmp = s->att;
                for (int h = 0; h < n_heads_h; h++) {
                    memcpy(s->q + h * head_dim_h,
                           s->xb2 + h * head_dim_h * 2,
                           (size_t)head_dim_h * sizeof(float));
                    memcpy(gate_tmp + h * head_dim_h,
                           s->xb2 + h * head_dim_h * 2 + head_dim_h,
                           (size_t)head_dim_h * sizeof(float));
                }
                gate_q = gate_tmp;
            } else {
                tq_matmul(s->q, s->xb, layer->wq, n_heads_h * head_dim_h, dim);
            }
            tq_matmul(s->k, s->xb, layer->wk, kv_dim_h, dim);
            tq_matmul(s->v, s->xb, layer->wv, kv_dim_h, dim);

            if (layer->q_norm) {
                for (int h = 0; h < n_heads_h; h++)
                    tq_rmsnorm(s->q + h * head_dim_h, s->q + h * head_dim_h,
                               layer->q_norm, head_dim_h, c->rms_norm_eps);
            }
            if (layer->k_norm) {
                for (int h = 0; h < n_kv_heads_h; h++)
                    tq_rmsnorm(s->k + h * head_dim_h, s->k + h * head_dim_h,
                               layer->k_norm, head_dim_h, c->rms_norm_eps);
            }

            if (c->partial_rotary_factor > 0.0f && c->partial_rotary_factor < 1.0f) {
                int rope_dim = (int)(c->partial_rotary_factor * head_dim_h);
                for (int h = 0; h < n_heads_h; h++) {
                    float* qh = s->q + h * head_dim_h;
                    for (int i = 0; i < rope_dim / 2; i++) {
                        float freq = 1.0f / powf(c->rope_freq_base, 2.0f * i / rope_dim);
                        float theta = pos * freq;
                        float cos_t = cosf(theta); float sin_t = sinf(theta);
                        float q0 = qh[2*i], q1 = qh[2*i+1];
                        qh[2*i] = q0*cos_t - q1*sin_t;
                        qh[2*i+1] = q0*sin_t + q1*cos_t;
                    }
                }
                for (int h = 0; h < n_kv_heads_h; h++) {
                    float* kh = s->k + h * head_dim_h;
                    for (int i = 0; i < rope_dim / 2; i++) {
                        float freq = 1.0f / powf(c->rope_freq_base, 2.0f * i / rope_dim);
                        float theta = pos * freq;
                        float cos_t = cosf(theta); float sin_t = sinf(theta);
                        float k0 = kh[2*i], k1 = kh[2*i+1];
                        kh[2*i] = k0*cos_t - k1*sin_t;
                        kh[2*i+1] = k0*sin_t + k1*cos_t;
                    }
                }
            } else {
                tq_rope(s->q, s->k, pos, head_dim_h, n_heads_h, n_kv_heads_h, c->rope_freq_base);
            }

            float* key_cache_l = s->key_cache + l * kv_layer_stride;
            float* val_cache_l = s->value_cache + l * kv_layer_stride;
            memcpy(key_cache_l + (size_t)pos * kv_dim_h, s->k, kv_dim_h * sizeof(float));
            memcpy(val_cache_l + (size_t)pos * kv_dim_h, s->v, kv_dim_h * sizeof(float));

            int seq_len = pos + 1;
            for (int h = 0; h < n_heads_h; h++) {
                float* qh = s->q + h * head_dim_h;
                float* atth = s->att + (size_t)h * c->max_seq_len;
                int kv_h = h / kv_mul_h;
                for (int t = 0; t < seq_len; t++) {
                    const float* kt = key_cache_l + (size_t)t * kv_dim_h + kv_h * head_dim_h;
                    float score = 0.0f;
                    for (int d = 0; d < head_dim_h; d++) score += qh[d] * kt[d];
                    atth[t] = score / sqrtf((float)head_dim_h);
                }
                tq_softmax(atth, seq_len);
                float* xbh = s->xb + h * head_dim_h;
                memset(xbh, 0, head_dim_h * sizeof(float));
                for (int t = 0; t < seq_len; t++) {
                    const float* vt = val_cache_l + (size_t)t * kv_dim_h + kv_h * head_dim_h;
                    float a = atth[t];
                    for (int d = 0; d < head_dim_h; d++) xbh[d] += a * vt[d];
                }
            }

            if (c->attn_output_gate && gate_q) {
                for (int i = 0; i < n_heads_h * head_dim_h; i++) {
                    float g = 1.0f / (1.0f + expf(-gate_q[i]));
                    s->xb[i] *= g;
                }
            }

            tq_matmul(s->xb2, s->xb, layer->wo, dim, n_heads_h * head_dim_h);

            if (l < 4) {
                char nm[64]; snprintf(nm, 64, "layer%02d_self_attn", l);
                compare_with_ref(nm, s->xb2, dim);
            }

            tq_add(s->x, s->x, s->xb2, dim);
        }

        /* FFN */
        if (layer->w_gate && layer->w_up && layer->w_down) {
            tq_rmsnorm(s->xb, s->x, layer->ffn_norm, dim, c->rms_norm_eps);
            if (l < 4) {
                char nm[64]; snprintf(nm, 64, "layer%02d_ffn_norm", l);
                compare_with_ref(nm, s->xb, dim);
            }
            tq_matmul(s->hb,  s->xb, layer->w_gate, c->intermediate_dim, dim);
            tq_matmul(s->hb2, s->xb, layer->w_up,   c->intermediate_dim, dim);
            tq_silu(s->hb, c->intermediate_dim);
            tq_mul(s->hb, s->hb, s->hb2, c->intermediate_dim);
            tq_matmul(s->xb2, s->hb, layer->w_down, dim, c->intermediate_dim);
            if (l < 4) {
                char nm[64]; snprintf(nm, 64, "layer%02d_mlp", l);
                compare_with_ref(nm, s->xb2, dim);
            }
            tq_add(s->x, s->x, s->xb2, dim);
        }

        /* Full layer output */
        char nm[64]; snprintf(nm, 64, "layer%02d", l);
        int match = compare_with_ref(nm, s->x, dim);
        if (match == 0 && first_diverge < 0) first_diverge = l;
        printf("\n");
    }

    /* Final norm */
    tq_rmsnorm(s->x, s->x, model->output_norm, dim, c->rms_norm_eps);
    printf("--- Final Norm ---\n");
    compare_with_ref("final_norm", s->x, dim);
    printf("\n");

    /* Logits */
    tq_matmul(s->logits, s->x, model->output_weight, c->vocab_size, dim);
    printf("--- Logits ---\n");
    compare_with_ref("logits", s->logits, c->vocab_size);

    int top_id = 0; float top_val = s->logits[0];
    for (int i = 1; i < c->vocab_size; i++)
        if (s->logits[i] > top_val) { top_val = s->logits[i]; top_id = i; }
    printf("  C  logits: top_id=%d val=%.4f [0:5]=[%.4f,%.4f,%.4f,%.4f,%.4f]\n",
           top_id, top_val, s->logits[0], s->logits[1], s->logits[2], s->logits[3], s->logits[4]);

    int rn = 0; float* rl = load_npy("/tmp/tq_ref/logits.npy", &rn);
    if (rl) {
        int ri = 0; float rv = rl[0];
        for (int i = 1; i < rn; i++) if (rl[i] > rv) { rv = rl[i]; ri = i; }
        printf("  Ref logits: top_id=%d val=%.4f [0:5]=[%.4f,%.4f,%.4f,%.4f,%.4f]\n",
               ri, rv, rl[0], rl[1], rl[2], rl[3], rl[4]);
        free(rl);
    }

    printf("\n");
    if (first_diverge >= 0) {
        printf("=== FIRST DIVERGENCE at layer %d ===\n", first_diverge);
    } else {
        printf("=== ALL LAYERS MATCH ===\n");
    }

    tq_free_state(s);
    tq_free_model(model);
    return 0;
}
