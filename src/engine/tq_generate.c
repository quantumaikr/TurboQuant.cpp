/**
 * tq_generate.c — Text generation loop with TurboQuant KV cache
 *
 * Implements:
 *   - Argmax sampling (greedy)
 *   - Top-p (nucleus) sampling with temperature
 *   - Full generation loop with streaming callback
 */

#include "turboquant/tq_engine.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

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

int tq_sample_topp(const float* logits, int vocab_size,
                   float temperature, float top_p,
                   unsigned long long* rng) {
    if (temperature <= 0.0f || top_p <= 0.0f) {
        return tq_sample_argmax(logits, vocab_size);
    }

    /* Allocate workspace for probabilities */
    prob_index_t* probindex = (prob_index_t*)malloc(vocab_size * sizeof(prob_index_t));
    if (!probindex) return tq_sample_argmax(logits, vocab_size);

    /* Apply temperature and compute softmax */
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float p = expf((logits[i] - max_val) / temperature);
        probindex[i].prob = p;
        probindex[i].index = i;
        sum += p;
    }

    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < vocab_size; i++) {
        probindex[i].prob *= inv_sum;
    }

    /* Sort by probability descending */
    qsort(probindex, vocab_size, sizeof(prob_index_t), compare_prob_desc);

    /* Find top-p cutoff */
    float cumulative = 0.0f;
    int n_top = 0;
    for (int i = 0; i < vocab_size; i++) {
        cumulative += probindex[i].prob;
        n_top = i + 1;
        if (cumulative >= top_p) break;
    }

    /* Re-normalize the nucleus */
    float nucleus_sum = 0.0f;
    for (int i = 0; i < n_top; i++) {
        nucleus_sum += probindex[i].prob;
    }

    /* Sample from the nucleus */
    float r = random_f32(rng) * nucleus_sum;
    float cdf = 0.0f;
    int sampled = probindex[0].index;
    for (int i = 0; i < n_top; i++) {
        cdf += probindex[i].prob;
        if (cdf >= r) {
            sampled = probindex[i].index;
            break;
        }
    }

    free(probindex);
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

    tq_state_t* state = tq_create_state(&model->config, config->kv_type);
    if (!state) {
        fprintf(stderr, "tq_generate: failed to allocate state\n");
        return -1;
    }

    /* Encode prompt */
    int prompt_tokens[4096];
    int n_prompt = 0;

    if (tokenizer && prompt) {
        n_prompt = tq_encode(tokenizer, prompt, prompt_tokens, 4096, 1);
    } else {
        /* No tokenizer: use BOS only */
        prompt_tokens[0] = 1; /* BOS */
        n_prompt = 1;
    }

    if (n_prompt <= 0) {
        prompt_tokens[0] = 1;
        n_prompt = 1;
    }

    /* Prefill: process all prompt tokens */
    for (int i = 0; i < n_prompt; i++) {
        tq_forward(model, state, prompt_tokens[i], i);
    }

    /* Sample first generated token */
    int pos = n_prompt;
    unsigned long long rng_state = 42;
    int next_token = tq_sample_topp(state->logits, model->config.vocab_size,
                                     config->temperature, config->top_p,
                                     &rng_state);

    int generated = 0;
    int output_pos = 0;
    int prev_token = prompt_tokens[n_prompt - 1];

    /* EOS token IDs — common values */
    int eos_token = 2;

    /* Generate loop */
    while (generated < config->max_tokens) {
        if (next_token == eos_token) break;
        if (pos >= model->config.max_seq_len) break;

        /* Decode token to text */
        if (tokenizer) {
            const char* piece = tq_decode(tokenizer, prev_token, next_token);
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

        /* Sample next token */
        next_token = tq_sample_topp(state->logits, model->config.vocab_size,
                                     config->temperature, config->top_p,
                                     &rng_state);
    }

    /* Null-terminate output */
    if (output && output_size > 0) {
        output[output_pos < output_size ? output_pos : output_size - 1] = '\0';
    }

    tq_free_state(state);
    return generated;
}
