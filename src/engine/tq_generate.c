/**
 * tq_generate.c — Text generation loop with TurboQuant KV cache
 *
 * Implements:
 *   - Argmax sampling (greedy)
 *   - Top-p (nucleus) sampling with temperature
 *   - Full generation loop with streaming callback
 */

#include "turboquant/tq_engine.h"
#include "turboquant/tq_gguf.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <pthread.h>

/* ============================================================
 * Argmax sampling: return token with highest logit
 * ============================================================ */
int tq_sample_argmax(const float* logits, int vocab_size) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

/* ============================================================
 * Top-p (nucleus) sampling with temperature
 *
 * 1. Apply temperature scaling
 * 2. Compute softmax probabilities
 * 3. Sort by probability (descending)
 * 4. Accumulate until cumulative prob >= top_p
 * 5. Sample from the nucleus
 * ============================================================ */

/* Simple RNG (xorshift64) for reproducible sampling */
static float random_f32(unsigned long long* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (float)((*state * 0x2545F4914F6CDD1DULL) >> 33) / (float)(1u << 31);
}

/* Comparison for sorting (probability, index) pairs */
typedef struct {
    float prob;
    int index;
} prob_index_t;

static int compare_prob_desc(const void* a, const void* b) {
    float pa = ((const prob_index_t*)a)->prob;
    float pb = ((const prob_index_t*)b)->prob;
    if (pa > pb) return -1;
    if (pa < pb) return 1;
    return 0;
}

/* Persistent workspace to avoid per-token malloc.
 * Protected by mutex for thread safety when multiple model instances
 * call tq_sample_topp concurrently. */
static prob_index_t* g_probindex = NULL;
static int g_probindex_size = 0;
static pthread_mutex_t g_probindex_mutex = PTHREAD_MUTEX_INITIALIZER;

int tq_sample_topp(const float* logits, int vocab_size,
                   float temperature, float top_p,
                   unsigned long long* rng) {
    if (temperature <= 0.0f || top_p <= 0.0f) {
        return tq_sample_argmax(logits, vocab_size);
    }

    /* Pre-filter: only keep logits within reasonable range of max.
     * For top-p=0.9 with temperature=0.7, logits more than ~20 below max
     * contribute negligibly. This avoids sorting 248K entries. */
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    float threshold = max_val - 16.0f * temperature; /* exp(-16) ≈ 1e-7 */

    /* Allocate/reuse workspace (mutex-protected for concurrent callers) */
    pthread_mutex_lock(&g_probindex_mutex);
    if (g_probindex_size < vocab_size) {
        free(g_probindex);
        g_probindex = (prob_index_t*)malloc(vocab_size * sizeof(prob_index_t));
        g_probindex_size = vocab_size;
    }
    if (!g_probindex) {
        pthread_mutex_unlock(&g_probindex_mutex);
        return tq_sample_argmax(logits, vocab_size);
    }

    /* Collect only candidates above threshold */
    int n_candidates = 0;
    float sum = 0.0f;
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] >= threshold) {
            float p = expf((logits[i] - max_val) * inv_temp);
            g_probindex[n_candidates].prob = p;
            g_probindex[n_candidates].index = i;
            sum += p;
            n_candidates++;
        }
    }

    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n_candidates; i++) {
        g_probindex[i].prob *= inv_sum;
    }

    /* Sort only candidates (typically < 1000 vs 248K) */
    qsort(g_probindex, n_candidates, sizeof(prob_index_t), compare_prob_desc);

    /* Find top-p cutoff */
    float cumulative = 0.0f;
    int n_top = 0;
    for (int i = 0; i < n_candidates; i++) {
        cumulative += g_probindex[i].prob;
        n_top = i + 1;
        if (cumulative >= top_p) break;
    }

    /* Sample from the nucleus */
    float r = random_f32(rng) * cumulative;
    float cdf = 0.0f;
    int sampled = g_probindex[0].index;
    for (int i = 0; i < n_top; i++) {
        cdf += g_probindex[i].prob;
        if (cdf >= r) {
            sampled = g_probindex[i].index;
            break;
        }
    }

    pthread_mutex_unlock(&g_probindex_mutex);
    return sampled;
}

/* ============================================================
 * Generate text from prompt
 *
 * Steps:
 * 1. Encode prompt to tokens
 * 2. Prefill: forward all prompt tokens
 * 3. Decode: sample next token, forward, repeat
 * 4. Stop on EOS or max_tokens
 * ============================================================ */
int tq_generate(tq_model_t* model, tq_tokenizer_t* tokenizer,
                const char* prompt, tq_gen_config_t* config,
                char* output, int output_size) {
    if (!model || !config) return -1;

    tq_state_t* state = tq_create_state_ex(&model->config, config->kv_type, config->value_quant_bits);
    if (!state) {
        fprintf(stderr, "tq_generate: failed to allocate state\n");
        return -1;
    }
    state->delta_kv_enabled = config->delta_kv;
    state->delta_iframe_interval = config->delta_iframe_interval;
    /* Hybrid DeltaNet models: delta KV applies only to self_attn layers.
     * DeltaNet layers don't use key_cache, so delta compression is safe. */

    /* Allocate MoE state if model uses MoE */
    if (model->config.is_moe && model->moe_config) {
        state->moe_state = tq_moe_create_state(
            (const tq_moe_config_t*)model->moe_config,
            model->config.hidden_dim);
        if (!state->moe_state) {
            fprintf(stderr, "tq_generate: failed to allocate MoE state\n");
            tq_free_state(state);
            return -1;
        }
    }

    /* Set up V highres window if requested */
    if (config->v_highres_window > 0 &&
        (config->value_quant_bits == 4 || config->value_quant_bits == 2)) {
        int n_layers = model->config.n_layers;
        int kv_dim = model->config.n_kv_heads * model->config.head_dim;
        int window = config->v_highres_window;
        state->v_highres_window = window;
        state->value_highres_fp16 = (uint16_t*)calloc(
            (size_t)n_layers * window * kv_dim, sizeof(uint16_t));
    }

    /* Set up K highres window (age-based progressive compression) */
    if (config->k_highres_window > 0 &&
        state->kv_quant_type < TQ_TYPE_COUNT && state->quant_key_cache != NULL) {
        int n_layers = model->config.n_layers;
        int kv_dim = model->config.n_kv_heads * model->config.head_dim;
        int window = config->k_highres_window;
        state->k_highres_window = window;
        state->key_highres_fp32 = (float*)calloc(
            (size_t)n_layers * window * kv_dim, sizeof(float));
    }

    /* Encode prompt */
    int prompt_tokens[4096];
    int n_prompt = 0;

    if (tokenizer && prompt) {
        /* Gemma models: prepend BOS=2 (required by both Gemma 3 and 4 architectures).
         * Qwen3.5: no BOS. */
        int add_bos = 0;
        if (model->config.model_type == 1) {
            add_bos = 1; /* All Gemma models need BOS */
        }
        n_prompt = tq_encode(tokenizer, prompt, prompt_tokens, 4096, add_bos);
    } else {
        /* No tokenizer: use BOS only (Gemma=2, Qwen=skip) */
        prompt_tokens[0] = (model->config.model_type == 1) ? 2 : 1;
        n_prompt = 1;
    }

    if (n_prompt <= 0) {
        prompt_tokens[0] = (model->config.model_type == 1) ? 2 : 1;
        n_prompt = 1;
    }

    /* Debug: print tokenized prompt */
    if (getenv("TQ_DEBUG")) {
        fprintf(stderr, "[DEBUG] prompt tokens (%d): ", n_prompt);
        for (int i = 0; i < n_prompt && i < 20; i++)
            fprintf(stderr, "%d ", prompt_tokens[i]);
        fprintf(stderr, "\n");
    }

    /* Prefill: process all prompt tokens */
    for (int i = 0; i < n_prompt; i++) {
        tq_forward(model, state, prompt_tokens[i], i);
    }

    /* Repetition penalty setup */
    int vocab_size = model->config.vocab_size;
    float rep_penalty = config->rep_penalty;
    int rep_window = config->rep_window;
    if (rep_window > 64) rep_window = 64;
    int recent_tokens[64];
    int recent_count = 0;

    /* Seed recent tokens with tail of prompt for better penalty coverage */
    for (int i = (n_prompt > rep_window ? n_prompt - rep_window : 0); i < n_prompt; i++) {
        recent_tokens[recent_count % 64] = prompt_tokens[i];
        recent_count++;
    }

    /* Apply repetition penalty to logits before first sample */
    if (rep_penalty > 1.0f) {
        int window = recent_count < rep_window ? recent_count : rep_window;
        for (int r = 0; r < window; r++) {
            int idx = (recent_count - 1 - r) % 64;
            if (idx < 0) idx += 64;
            int tok = recent_tokens[idx];
            if (tok >= 0 && tok < vocab_size) {
                if (state->logits[tok] > 0)
                    state->logits[tok] /= rep_penalty;
                else
                    state->logits[tok] *= rep_penalty;
            }
        }
    }

    /* Sample first generated token */
    int pos = n_prompt;
    unsigned long long rng_state = 42;
    int next_token = tq_sample_topp(state->logits, vocab_size,
                                     config->temperature, config->top_p,
                                     &rng_state);

    /* Record first sampled token */
    recent_tokens[recent_count % 64] = next_token;
    recent_count++;

    int generated = 0;
    int output_pos = 0;
    int prev_token = prompt_tokens[n_prompt - 1];

    /* EOS token IDs — check common values.
     * Qwen3.5: eos = 248044 (<|endoftext|>), also 248046 (<|im_end|>)
     * Gemma3: eos = 1
     * Gemma4: eos = 106 (<end_of_turn>)
     * LLaMA: eos = 2 */
    int eos_token1 = 1;       /* Gemma3 <eos>, also common default */
    int eos_token2 = 248044;  /* Qwen <|endoftext|> */
    int eos_token3 = 248046;  /* Qwen <|im_end|> */
    int eos_token4 = 106;     /* Gemma4 <end_of_turn> */

    /* Generate loop */
    while (generated < config->max_tokens) {
        if (next_token == eos_token1 || next_token == eos_token2 ||
            next_token == eos_token3 || next_token == eos_token4) break;
        if (pos >= model->config.max_seq_len) break;

        /* Decode token to text */
        if (tokenizer) {
            const char* piece = tq_decode(tokenizer, prev_token, next_token);

            /* Skip thinking tokens (e.g. Qwen3.5 <think>...</think>) */
            if (piece && (strstr(piece, "<think>") || strstr(piece, "</think>"))) {
                piece = "";
            }

            int piece_len = (int)strlen(piece);

            /* Stream callback */
            if (config->on_token) {
                config->on_token(piece, config->user_data);
            }

            /* Append to output buffer */
            if (output && output_pos + piece_len < output_size - 1) {
                memcpy(output + output_pos, piece, piece_len);
                output_pos += piece_len;
            }
        }

        /* Forward pass for next token */
        prev_token = next_token;
        tq_forward(model, state, next_token, pos);
        pos++;
        generated++;

        /* Apply repetition penalty before sampling */
        if (rep_penalty > 1.0f) {
            int window = recent_count < rep_window ? recent_count : rep_window;
            for (int r = 0; r < window; r++) {
                int idx = (recent_count - 1 - r) % 64;
                if (idx < 0) idx += 64;
                int tok = recent_tokens[idx];
                if (tok >= 0 && tok < vocab_size) {
                    if (state->logits[tok] > 0)
                        state->logits[tok] /= rep_penalty;
                    else
                        state->logits[tok] *= rep_penalty;
                }
            }
        }

        /* Sample next token */
        next_token = tq_sample_topp(state->logits, vocab_size,
                                     config->temperature, config->top_p,
                                     &rng_state);

        /* Record sampled token for repetition penalty */
        recent_tokens[recent_count % 64] = next_token;
        recent_count++;
    }

    /* Null-terminate output */
    if (output && output_size > 0) {
        output[output_pos < output_size ? output_pos : output_size - 1] = '\0';
    }

    tq_free_state(state);
    return generated;
}
