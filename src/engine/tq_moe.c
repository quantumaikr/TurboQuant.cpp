/**
 * tq_moe.c — Mixture of Experts routing and expert dispatch
 *
 * Implements top-K expert selection with softmax renormalization,
 * SwiGLU FFN dispatch per expert, shared expert support,
 * runtime LRU Q4 cache for routed experts, and memory advise hints.
 */

#include "turboquant/tq_gguf.h"
#include "turboquant/tq_engine.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define TQ_MOE_HAS_NEON 1
#else
#define TQ_MOE_HAS_NEON 0
#endif

/* ============================================================
 * Fast SiLU (Swish) approximation
 *
 * silu(x) = x / (1 + exp(-x))
 *
 * Uses Schraudolph's fast exp approximation (~1% accuracy for |x|<10).
 * Called 270 times/token with expert_dim=512, so ~138K calls saved
 * vs. standard expf per token.
 * ============================================================ */
static inline float fast_expf_moe(float x) {
    if (x < -20.0f) return 0.0f;
    if (x > 20.0f) return expf(x);
    union { float f; int32_t i; } v;
    v.i = (int32_t)(12102203.0f * x + 1065353216.0f);
    return v.f;
}

/* Vectorized SwiGLU: hb[i] = silu(hb[i]) * hb2[i] */
static void swiglu_fused(float* restrict hb, const float* restrict hb2, int n) {
#if TQ_MOE_HAS_NEON
    int i = 0;
    float32x4_t vone = vdupq_n_f32(1.0f);
    for (; i + 7 < n; i += 8) {
        /* Process 8 elements: 2x float32x4_t */
        float32x4_t vg0 = vld1q_f32(hb + i);
        float32x4_t vg1 = vld1q_f32(hb + i + 4);
        float32x4_t vu0 = vld1q_f32(hb2 + i);
        float32x4_t vu1 = vld1q_f32(hb2 + i + 4);

        /* Fast sigmoid via Schraudolph: exp(-x) approx */
        float neg0[4], neg1[4];
        vst1q_f32(neg0, vnegq_f32(vg0));
        vst1q_f32(neg1, vnegq_f32(vg1));
        float32x4_t vexp0 = {fast_expf_moe(neg0[0]), fast_expf_moe(neg0[1]),
                              fast_expf_moe(neg0[2]), fast_expf_moe(neg0[3])};
        float32x4_t vexp1 = {fast_expf_moe(neg1[0]), fast_expf_moe(neg1[1]),
                              fast_expf_moe(neg1[2]), fast_expf_moe(neg1[3])};
        /* sigmoid = 1 / (1 + exp(-x)) */
        float32x4_t vsig0 = vrecpeq_f32(vaddq_f32(vone, vexp0));
        vsig0 = vmulq_f32(vsig0, vrecpsq_f32(vaddq_f32(vone, vexp0), vsig0));
        float32x4_t vsig1 = vrecpeq_f32(vaddq_f32(vone, vexp1));
        vsig1 = vmulq_f32(vsig1, vrecpsq_f32(vaddq_f32(vone, vexp1), vsig1));
        /* silu = x * sigmoid(x) */
        float32x4_t vsilu0 = vmulq_f32(vg0, vsig0);
        float32x4_t vsilu1 = vmulq_f32(vg1, vsig1);
        /* silu * up */
        vst1q_f32(hb + i, vmulq_f32(vsilu0, vu0));
        vst1q_f32(hb + i + 4, vmulq_f32(vsilu1, vu1));
    }
    for (; i < n; i++) {
        float g = hb[i];
        hb[i] = (g / (1.0f + fast_expf_moe(-g))) * hb2[i];
    }
#else
    for (int i = 0; i < n; i++) {
        float g = hb[i];
        hb[i] = (g / (1.0f + fast_expf_moe(-g))) * hb2[i];
    }
#endif
}

#ifndef _WIN32
#include <sys/mman.h>
#endif

/* ============================================================
 * Runtime Expert Q4 LRU Cache
 *
 * MoE models with 256 experts x 40 layers would need ~19 GB
 * if all experts were pre-converted to Q4. Instead, we cache
 * only the EXPERT_CACHE_SIZE most-recently-used experts per
 * layer. With 32 slots per layer, this is ~1.9 GB total.
 *
 * Cache hits use fast Q4 matmul; misses dequant from GGUF
 * mmap on the fly, then cache the result for next time.
 * ============================================================ */

#define EXPERT_CACHE_SIZE 32  /* per layer */

typedef struct {
    int      expert_id;       /* -1 = empty slot */
    uint8_t* gate_q4_qs;
    float*   gate_q4_scales;
    uint8_t* up_q4_qs;
    float*   up_q4_scales;
    uint8_t* down_q4_qs;
    float*   down_q4_scales;
    int      last_used;       /* token counter for LRU eviction */
} expert_cache_entry_t;

typedef struct {
    expert_cache_entry_t entries[EXPERT_CACHE_SIZE];
    int count;                /* number of occupied slots */
} expert_layer_cache_t;

static expert_layer_cache_t* g_expert_cache = NULL; /* [n_layers] */
static int    g_cache_n_layers   = 0;
static int    g_cache_hidden_dim = 0;
static int    g_cache_exp_inter  = 0;
static int    g_token_counter    = 0;
static float* g_cache_fp32_temp  = NULL;  /* reusable dequant buffer */

void tq_moe_cache_init(int n_layers, const tq_moe_config_t* config, int hidden_dim)
{
    if (g_expert_cache) return; /* already initialized */
    if (!config) return;

    g_cache_n_layers   = n_layers;
    g_cache_hidden_dim = hidden_dim;
    g_cache_exp_inter  = config->expert_intermediate_dim;
    g_token_counter    = 0;

    g_expert_cache = (expert_layer_cache_t*)calloc(
        (size_t)n_layers, sizeof(expert_layer_cache_t));
    if (!g_expert_cache) {
        fprintf(stderr, "tq_moe_cache_init: allocation failed\n");
        return;
    }

    /* Mark all slots empty */
    for (int l = 0; l < n_layers; l++) {
        for (int s = 0; s < EXPERT_CACHE_SIZE; s++) {
            g_expert_cache[l].entries[s].expert_id = -1;
        }
    }

    /* Allocate reusable FP32 temp buffer (max of gate/up and down sizes) */
    size_t gate_up_elems = (size_t)g_cache_exp_inter * hidden_dim;
    size_t down_elems    = (size_t)hidden_dim * g_cache_exp_inter;
    size_t max_elems     = gate_up_elems > down_elems ? gate_up_elems : down_elems;
    g_cache_fp32_temp = (float*)malloc(max_elems * sizeof(float));

    float cache_mb = (float)(n_layers * EXPERT_CACHE_SIZE) *
                     (3.0f * (float)(gate_up_elems + 31) / 32.0f * 20.0f) /
                     (1024.0f * 1024.0f);
    fprintf(stderr, "tq_moe_cache_init: LRU cache for %d layers x %d slots "
            "(max %.0f MB)\n", n_layers, EXPERT_CACHE_SIZE, (double)cache_mb);
}

static void free_cache_entry(expert_cache_entry_t* e)
{
    free(e->gate_q4_qs);     e->gate_q4_qs = NULL;
    free(e->gate_q4_scales); e->gate_q4_scales = NULL;
    free(e->up_q4_qs);       e->up_q4_qs = NULL;
    free(e->up_q4_scales);   e->up_q4_scales = NULL;
    free(e->down_q4_qs);     e->down_q4_qs = NULL;
    free(e->down_q4_scales); e->down_q4_scales = NULL;
    e->expert_id = -1;
}

void tq_moe_cache_free(void)
{
    if (!g_expert_cache) return;
    for (int l = 0; l < g_cache_n_layers; l++) {
        for (int s = 0; s < EXPERT_CACHE_SIZE; s++) {
            free_cache_entry(&g_expert_cache[l].entries[s]);
        }
    }
    free(g_expert_cache);
    g_expert_cache = NULL;
    free(g_cache_fp32_temp);
    g_cache_fp32_temp = NULL;
    g_cache_n_layers = 0;
}

/* Find a cached entry for expert_id in layer, or evict LRU and create one.
 * Returns the entry with Q4 data populated. */
static expert_cache_entry_t* cache_get_or_create(
    int layer_idx, int expert_id, const tq_expert_weights_t* exp)
{
    expert_layer_cache_t* lc = &g_expert_cache[layer_idx];

    /* Search for existing entry */
    for (int s = 0; s < EXPERT_CACHE_SIZE; s++) {
        if (lc->entries[s].expert_id == expert_id) {
            lc->entries[s].last_used = g_token_counter;
            return &lc->entries[s];
        }
    }

    /* Cache miss: find an empty slot or evict LRU */
    int target = -1;
    if (lc->count < EXPERT_CACHE_SIZE) {
        /* Find first empty slot */
        for (int s = 0; s < EXPERT_CACHE_SIZE; s++) {
            if (lc->entries[s].expert_id < 0) {
                target = s;
                break;
            }
        }
        lc->count++;
    } else {
        /* Evict least-recently-used */
        int oldest_time = g_token_counter + 1;
        for (int s = 0; s < EXPERT_CACHE_SIZE; s++) {
            if (lc->entries[s].last_used < oldest_time) {
                oldest_time = lc->entries[s].last_used;
                target = s;
            }
        }
        free_cache_entry(&lc->entries[target]);
    }

    expert_cache_entry_t* ce = &lc->entries[target];
    ce->expert_id = expert_id;
    ce->last_used = g_token_counter;

    int dim = g_cache_hidden_dim;
    int inter = g_cache_exp_inter;

    /* Convert gate: [inter, dim] */
    {
        int n = inter * dim;
        int n_blocks = (n + 31) / 32;
        tq_dequant_row_gguf(exp->gate_type, exp->w_gate, g_cache_fp32_temp, n);
        ce->gate_q4_qs = (uint8_t*)malloc((size_t)n_blocks * 16);
        ce->gate_q4_scales = (float*)malloc((size_t)n_blocks * sizeof(float));
        if (ce->gate_q4_qs && ce->gate_q4_scales)
            tq_quantize_row_q4(g_cache_fp32_temp, ce->gate_q4_qs,
                               ce->gate_q4_scales, n);
    }

    /* Convert up: [inter, dim] */
    {
        int n = inter * dim;
        int n_blocks = (n + 31) / 32;
        tq_dequant_row_gguf(exp->up_type, exp->w_up, g_cache_fp32_temp, n);
        ce->up_q4_qs = (uint8_t*)malloc((size_t)n_blocks * 16);
        ce->up_q4_scales = (float*)malloc((size_t)n_blocks * sizeof(float));
        if (ce->up_q4_qs && ce->up_q4_scales)
            tq_quantize_row_q4(g_cache_fp32_temp, ce->up_q4_qs,
                               ce->up_q4_scales, n);
    }

    /* Convert down: [dim, inter] */
    {
        int n = dim * inter;
        int n_blocks = (n + 31) / 32;
        tq_dequant_row_gguf(exp->down_type, exp->w_down, g_cache_fp32_temp, n);
        ce->down_q4_qs = (uint8_t*)malloc((size_t)n_blocks * 16);
        ce->down_q4_scales = (float*)malloc((size_t)n_blocks * sizeof(float));
        if (ce->down_q4_qs && ce->down_q4_scales)
            tq_quantize_row_q4(g_cache_fp32_temp, ce->down_q4_qs,
                               ce->down_q4_scales, n);
    }

    return ce;
}

/* ============================================================
 * State management
 * ============================================================ */

tq_moe_state_t* tq_moe_create_state(const tq_moe_config_t* config, int hidden_dim)
{
    tq_moe_state_t* s = (tq_moe_state_t*)calloc(1, sizeof(tq_moe_state_t));
    if (!s) return NULL;

    s->router_logits  = (float*)malloc((size_t)config->num_experts * sizeof(float));
    s->top_experts    = (int*)calloc((size_t)config->num_active, sizeof(int));
    s->expert_weights = (float*)malloc((size_t)config->num_active * sizeof(float));
    s->expert_out     = (float*)malloc((size_t)hidden_dim * sizeof(float));

    /* Workspace buffers sized to the larger of expert / shared-expert intermediate dim */
    int inter = config->expert_intermediate_dim;
    if (config->has_shared_expert && config->shared_expert_intermediate_dim > inter)
        inter = config->shared_expert_intermediate_dim;

    s->expert_hb  = (float*)malloc((size_t)inter * sizeof(float));
    s->expert_hb2 = (float*)malloc((size_t)inter * sizeof(float));

    return s;
}

void tq_moe_free_state(tq_moe_state_t* state)
{
    if (!state) return;
    free(state->router_logits);
    free(state->top_experts);
    free(state->expert_weights);
    free(state->expert_out);
    free(state->expert_hb);
    free(state->expert_hb2);
    free(state);
}

/* ============================================================
 * Top-K expert routing
 * ============================================================ */

void tq_moe_route(const float* hidden, const float* router_weight,
                  int num_experts, int num_active, int hidden_dim,
                  int* out_expert_ids, float* out_expert_weights)
{
    /*
     * We need scratch space for router logits. num_experts can be up to 256,
     * so we heap-allocate to avoid large VLAs on the stack.
     */
    float* logits = (float*)malloc((size_t)num_experts * sizeof(float));
    if (!logits) return;

    /* Step 1: Compute router logits — logits[e] = dot(hidden, router_weight[e]) */
    for (int e = 0; e < num_experts; e++) {
        const float* row = router_weight + (size_t)e * hidden_dim;
        float sum = 0.0f;
        for (int j = 0; j < hidden_dim; j++)
            sum += hidden[j] * row[j];
        logits[e] = sum;
    }

    /* Step 2: Top-K selection via partial sort (K passes, K << num_experts)
     *
     * Use >= for tie-breaking so that when multiple experts have equal logits,
     * the first unused one always wins. Also guard against NaN logits (NaN
     * comparisons return false, so without >= the loop could leave best == -1).
     */
    uint8_t* used = (uint8_t*)calloc((size_t)num_experts, sizeof(uint8_t));
    if (!used) { free(logits); return; }

    int n_valid = 0;
    for (int k = 0; k < num_active; k++) {
        int best = -1;
        float best_val = -HUGE_VALF;
        for (int e = 0; e < num_experts; e++) {
            if (!used[e] && logits[e] >= best_val) {
                best_val = logits[e];
                best = e;
            }
        }
        out_expert_ids[k] = best;
        if (best >= 0) {
            used[best] = 1;
            n_valid++;
        } else {
            out_expert_weights[k] = 0.0f;
        }
    }

    /* Step 3: Softmax over selected experts (renormalize top-K) */
    if (n_valid == 0) {
        /* All experts invalid (NaN logits or num_experts=0) — uniform fallback */
        for (int k = 0; k < num_active; k++) {
            out_expert_weights[k] = 0.0f;
        }
        free(used);
        free(logits);
        return;
    }

    float max_val = -HUGE_VALF;
    for (int k = 0; k < num_active; k++) {
        if (out_expert_ids[k] < 0) continue;
        float v = logits[out_expert_ids[k]];
        if (v > max_val) max_val = v;
    }

    float sum_exp = 0.0f;
    for (int k = 0; k < num_active; k++) {
        if (out_expert_ids[k] < 0) { out_expert_weights[k] = 0.0f; continue; }
        float e = expf(logits[out_expert_ids[k]] - max_val);
        out_expert_weights[k] = e;
        sum_exp += e;
    }

    if (sum_exp > 0.0f) {
        float inv_sum = 1.0f / sum_exp;
        for (int k = 0; k < num_active; k++)
            out_expert_weights[k] *= inv_sum;
    }

    free(used);
    free(logits);
}

/* ============================================================
 * Full MoE FFN forward pass
 * ============================================================ */

void tq_moe_forward(const tq_moe_layer_t* layer,
                    const tq_moe_config_t* config,
                    tq_moe_state_t* state,
                    const float* input, float* output,
                    int hidden_dim, int layer_idx)
{
    int num_active = config->num_active;
    int expert_dim = config->expert_intermediate_dim;

    /* Step 1: Route — select top-K experts */
    tq_moe_route(input, layer->router_weight,
                 config->num_experts, num_active, hidden_dim,
                 state->top_experts, state->expert_weights);

    /* Step 2: Zero the output accumulator */
    memset(output, 0, (size_t)hidden_dim * sizeof(float));

    /* Advance the global token counter for LRU tracking */
    g_token_counter++;

    /* Step 3: For each selected expert, compute SwiGLU FFN and accumulate */
    for (int k = 0; k < num_active; k++) {
        int eid = state->top_experts[k];
        float w = state->expert_weights[k];
        if (eid < 0 || eid >= config->num_experts) continue; /* safety check */
        const tq_expert_weights_t* exp = &layer->experts[eid];

        /* LRU cache disabled — cache miss dequant+Q4 overhead dominates.
         * Direct fused GGUF dot product is faster than cache miss penalty. */
        if (0 && g_expert_cache && layer_idx >= 0 && layer_idx < g_cache_n_layers
            && exp->w_gate) {
            expert_cache_entry_t* ce = cache_get_or_create(layer_idx, eid, exp);
            if (ce->gate_q4_qs && ce->up_q4_qs && ce->down_q4_qs) {
                /* Fast Q4 matmul path from LRU cache */
                tq_matmul_q4(state->expert_hb, input,
                             ce->gate_q4_qs, ce->gate_q4_scales,
                             expert_dim, hidden_dim);
                tq_matmul_q4(state->expert_hb2, input,
                             ce->up_q4_qs, ce->up_q4_scales,
                             expert_dim, hidden_dim);

                /* SwiGLU activation: hb = silu(gate) * up */
                swiglu_fused(state->expert_hb, state->expert_hb2, expert_dim);

                tq_matmul_q4(state->expert_out, state->expert_hb,
                             ce->down_q4_qs, ce->down_q4_scales,
                             hidden_dim, expert_dim);

                /* Weighted accumulation: output += weight * down_proj */
                for (int i = 0; i < hidden_dim; i++)
                    output[i] += w * state->expert_out[i];
                continue;
            }
        }

        if (exp->q4_converted) {
            /* Fast Q4 matmul path — pre-converted expert weights (shared expert)
             * tq_matmul_q4(out, x, w_qs, w_scales, n=out_rows, d=in_cols) */
            tq_matmul_q4(state->expert_hb, input,
                         exp->gate_q4_qs, exp->gate_q4_scales,
                         expert_dim, hidden_dim);
            tq_matmul_q4(state->expert_hb2, input,
                         exp->up_q4_qs, exp->up_q4_scales,
                         expert_dim, hidden_dim);

            /* SwiGLU activation: hb = silu(gate) * up */
            swiglu_fused(state->expert_hb, state->expert_hb2, expert_dim);

            tq_matmul_q4(state->expert_out, state->expert_hb,
                         exp->down_q4_qs, exp->down_q4_scales,
                         hidden_dim, expert_dim);
        } else {
            /* Fallback: on-the-fly GGUF dequant path */
            tq_metal_batch_begin_if_available();

            /* gate = input @ w_gate^T   -> [expert_dim] */
            tq_matmul_gguf(state->expert_hb, input,
                           exp->w_gate, exp->gate_type,
                           expert_dim, hidden_dim);

            /* up = input @ w_up^T   -> [expert_dim] */
            tq_matmul_gguf(state->expert_hb2, input,
                           exp->w_up, exp->up_type,
                           expert_dim, hidden_dim);

            /* Flush: commit + wait + copy results before CPU-side SwiGLU */
            tq_metal_batch_flush_if_available();

            /* SwiGLU activation: hb = silu(gate) * up */
            swiglu_fused(state->expert_hb, state->expert_hb2, expert_dim);

            /* down = hb @ w_down^T   -> [hidden_dim] */
            tq_matmul_gguf(state->expert_out, state->expert_hb,
                           exp->w_down, exp->down_type,
                           hidden_dim, expert_dim);
        }

        /* Weighted accumulation: output += weight * down_proj */
        for (int i = 0; i < hidden_dim; i++)
            output[i] += w * state->expert_out[i];
    }

    /* Step 4: Shared expert (always-active, if present) */
    if (config->has_shared_expert) {
        int shared_dim = config->shared_expert_intermediate_dim;
        if (shared_dim == 0) shared_dim = expert_dim;

        /* Optional shared expert gating (sigmoid scalar gate) */
        float shared_gate_val = 1.0f;
        if (layer->shared_gate) {
            float dot = 0.0f;
            for (int j = 0; j < hidden_dim; j++)
                dot += input[j] * layer->shared_gate[j];
            shared_gate_val = 1.0f / (1.0f + fast_expf_moe(-dot)); /* sigmoid */
        }

        if (layer->shared_expert.q4_converted) {
            /* Fast Q4 path for shared expert
             * tq_matmul_q4(out, x, w_qs, w_scales, n=out_rows, d=in_cols) */
            tq_matmul_q4(state->expert_hb, input,
                         layer->shared_expert.gate_q4_qs, layer->shared_expert.gate_q4_scales,
                         shared_dim, hidden_dim);
            tq_matmul_q4(state->expert_hb2, input,
                         layer->shared_expert.up_q4_qs, layer->shared_expert.up_q4_scales,
                         shared_dim, hidden_dim);

            swiglu_fused(state->expert_hb, state->expert_hb2, shared_dim);

            tq_matmul_q4(state->expert_out, state->expert_hb,
                         layer->shared_expert.down_q4_qs, layer->shared_expert.down_q4_scales,
                         hidden_dim, shared_dim);
        } else {
            /* Fallback: on-the-fly GGUF dequant */
            tq_metal_batch_begin_if_available();

            tq_matmul_gguf(state->expert_hb, input,
                           layer->shared_expert.w_gate, layer->shared_expert.gate_type,
                           shared_dim, hidden_dim);

            tq_matmul_gguf(state->expert_hb2, input,
                           layer->shared_expert.w_up, layer->shared_expert.up_type,
                           shared_dim, hidden_dim);

            tq_metal_batch_flush_if_available();

            swiglu_fused(state->expert_hb, state->expert_hb2, shared_dim);

            tq_matmul_gguf(state->expert_out, state->expert_hb,
                           layer->shared_expert.w_down, layer->shared_expert.down_type,
                           hidden_dim, shared_dim);
        }

        for (int i = 0; i < hidden_dim; i++)
            output[i] += shared_gate_val * state->expert_out[i];
    }
}

/* ============================================================
 * Expert memory advise (madvise hints for paging)
 * ============================================================ */

void tq_moe_advise(const tq_moe_layer_t* layer,
                   const int* active_ids, int n_active,
                   int num_experts)
{
    /*
     * TODO: Implement madvise(MADV_WILLNEED) for active experts and
     *       madvise(MADV_DONTNEED) for inactive experts once tensor
     *       size information is available in tq_expert_weights_t.
     *
     * The idea is:
     *   - For each active expert, call madvise(MADV_WILLNEED) on
     *     w_gate, w_up, w_down data regions to prefetch pages.
     *   - For inactive experts, optionally call madvise(MADV_DONTNEED)
     *     to allow the OS to reclaim those pages.
     *
     * This requires knowing the byte size of each weight tensor,
     * which currently isn't stored in tq_expert_weights_t.
     */
    (void)layer;
    (void)active_ids;
    (void)n_active;
    (void)num_experts;
}
