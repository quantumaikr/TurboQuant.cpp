/**
 * tq_moe.c — Mixture of Experts routing and expert dispatch
 *
 * Implements top-K expert selection with softmax renormalization,
 * SwiGLU FFN dispatch per expert, shared expert support, and
 * memory advise hints for expert paging.
 */

#include "turboquant/tq_gguf.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

#ifndef _WIN32
#include <sys/mman.h>
#endif

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
                    int hidden_dim)
{
    int num_active = config->num_active;
    int expert_dim = config->expert_intermediate_dim;

    /* Step 1: Route — select top-K experts */
    tq_moe_route(input, layer->router_weight,
                 config->num_experts, num_active, hidden_dim,
                 state->top_experts, state->expert_weights);

    /* Step 2: Zero the output accumulator */
    memset(output, 0, (size_t)hidden_dim * sizeof(float));

    /* Step 3: For each selected expert, compute SwiGLU FFN and accumulate */
    for (int k = 0; k < num_active; k++) {
        int eid = state->top_experts[k];
        float w = state->expert_weights[k];
        if (eid < 0 || eid >= config->num_experts) continue; /* safety check */
        const tq_expert_weights_t* exp = &layer->experts[eid];

        /* gate = input @ w_gate^T   -> [expert_dim] */
        tq_matmul_gguf(state->expert_hb, input,
                       exp->w_gate, exp->gate_type,
                       expert_dim, hidden_dim);

        /* up = input @ w_up^T   -> [expert_dim] */
        tq_matmul_gguf(state->expert_hb2, input,
                       exp->w_up, exp->up_type,
                       expert_dim, hidden_dim);

        /* SwiGLU activation: hb = silu(gate) * up */
        for (int i = 0; i < expert_dim; i++) {
            float g = state->expert_hb[i];
            state->expert_hb[i] = (g / (1.0f + expf(-g))) * state->expert_hb2[i];
        }

        /* down = hb @ w_down^T   -> [hidden_dim] */
        tq_matmul_gguf(state->expert_out, state->expert_hb,
                       exp->w_down, exp->down_type,
                       hidden_dim, expert_dim);

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
            shared_gate_val = 1.0f / (1.0f + expf(-dot)); /* sigmoid */
        }

        /* SwiGLU for shared expert */
        tq_matmul_gguf(state->expert_hb, input,
                       layer->shared_expert.w_gate, layer->shared_expert.gate_type,
                       shared_dim, hidden_dim);

        tq_matmul_gguf(state->expert_hb2, input,
                       layer->shared_expert.w_up, layer->shared_expert.up_type,
                       shared_dim, hidden_dim);

        for (int i = 0; i < shared_dim; i++) {
            float g = state->expert_hb[i];
            state->expert_hb[i] = (g / (1.0f + expf(-g))) * state->expert_hb2[i];
        }

        tq_matmul_gguf(state->expert_out, state->expert_hb,
                       layer->shared_expert.w_down, layer->shared_expert.down_type,
                       hidden_dim, shared_dim);

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
