/**
 * tq_generate.c — Text generation loop with TurboQuant KV cache
 *
 * Implements:
 *   - Argmax sampling (greedy)
 *   - Top-p (nucleus) sampling with temperature
 *   - Full generation loop with streaming callback
 */

#include "turboquant/turboquant.h"
#include "turboquant/tq_engine.h"
#include "turboquant/tq_gguf.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#define pthread_mutex_t SRWLOCK
#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
#define pthread_mutex_lock(m) AcquireSRWLockExclusive(m)
#define pthread_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#else
#include <pthread.h>
#endif

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
        /* BOS token handling:
         * Gemma 3/4: BOS=2 (required)
         * LLaMA 3: BOS=128000 (<|begin_of_text|>) — but tokenizer usually adds it
         * Qwen3.5: no BOS needed */
        int add_bos = 0;
        if (model->config.model_type == 1) {
            add_bos = 1; /* Gemma: always prepend BOS=2 */
        }
        n_prompt = tq_encode(tokenizer, prompt, prompt_tokens, 4096, add_bos);
    } else {
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

    /* Load pre-computed KV cache if available (skip prefill) */
    int pos_after_prefill = n_prompt;
    if (config->load_kv_path) {
        FILE* kv_fp = fopen(config->load_kv_path, "rb");
        if (kv_fp) {
            int32_t saved_pos = 0;
            size_t kv_dim_save = 0;
            fread(&saved_pos, sizeof(int32_t), 1, kv_fp);
            fread(&kv_dim_save, sizeof(size_t), 1, kv_fp);
            size_t kv_dim = (size_t)model->config.n_kv_heads * model->config.head_dim;
            int max_seq = model->config.max_seq_len;
            size_t layer_stride = (size_t)max_seq * kv_dim;
            /* Read per-layer, respecting stride */
            for (int l = 0; l < model->config.n_layers; l++) {
                if (state->key_cache)
                    fread(state->key_cache + l * layer_stride, sizeof(float), (size_t)saved_pos * kv_dim, kv_fp);
                if (state->value_cache_fp16)
                    fread(state->value_cache_fp16 + l * layer_stride, sizeof(uint16_t), (size_t)saved_pos * kv_dim, kv_fp);
                else if (state->value_cache)
                    fread(state->value_cache + l * layer_stride, sizeof(float), (size_t)saved_pos * kv_dim, kv_fp);
            }
            fclose(kv_fp);
            pos_after_prefill = saved_pos;
            size_t total_bytes = (size_t)model->config.n_layers * saved_pos * kv_dim * (sizeof(float) + (state->value_cache_fp16 ? sizeof(uint16_t) : sizeof(float)));
            fprintf(stderr, "[load-kv] Loaded %d tokens from %s (%.1f MB)\n",
                    saved_pos, config->load_kv_path,
                    (double)total_bytes / (1024.0 * 1024.0));
        } else {
            fprintf(stderr, "[load-kv] Cannot open %s, running normal prefill\n", config->load_kv_path);
        }
    }

    /* Prefill: process prompt tokens.
     * If KV was loaded, the loaded context occupies positions [0..pos_after_prefill).
     * The new prompt is appended starting at pos_after_prefill. */
    int prefill_start = 0;
    if (config->load_kv_path && pos_after_prefill > 0) {
        prefill_start = pos_after_prefill;
    }
    for (int i = 0; i < n_prompt; i++) {
        tq_forward(model, state, prompt_tokens[i], prefill_start + i);
    }
    pos_after_prefill = prefill_start + n_prompt;

    /* Save KV cache after prefill if requested */
    if (config->save_kv_path && pos_after_prefill > 0) {
        FILE* kv_fp = fopen(config->save_kv_path, "wb");
        if (kv_fp) {
            int32_t save_pos = (int32_t)pos_after_prefill;
            size_t kv_dim = (size_t)model->config.n_kv_heads * model->config.head_dim;
            int max_seq = model->config.max_seq_len;
            size_t layer_stride = (size_t)max_seq * kv_dim;
            fwrite(&save_pos, sizeof(int32_t), 1, kv_fp);
            fwrite(&kv_dim, sizeof(size_t), 1, kv_fp);
            /* Write per-layer, only saved_pos positions */
            size_t total = 0;
            for (int l = 0; l < model->config.n_layers; l++) {
                if (state->key_cache) {
                    fwrite(state->key_cache + l * layer_stride, sizeof(float), (size_t)save_pos * kv_dim, kv_fp);
                    total += (size_t)save_pos * kv_dim * sizeof(float);
                }
                if (state->value_cache_fp16) {
                    fwrite(state->value_cache_fp16 + l * layer_stride, sizeof(uint16_t), (size_t)save_pos * kv_dim, kv_fp);
                    total += (size_t)save_pos * kv_dim * sizeof(uint16_t);
                } else if (state->value_cache) {
                    fwrite(state->value_cache + l * layer_stride, sizeof(float), (size_t)save_pos * kv_dim, kv_fp);
                    total += (size_t)save_pos * kv_dim * sizeof(float);
                }
            }
            fclose(kv_fp);
            fprintf(stderr, "[save-kv] Saved %d tokens to %s (%.1f MB)\n",
                    save_pos, config->save_kv_path, (double)total / (1024.0 * 1024.0));
        }
    }

    /* Repetition penalty setup */
    int vocab_size = model->config.vocab_size;
    float rep_penalty = config->rep_penalty;
    int rep_window = config->rep_window;
    if (rep_window > 128) rep_window = 128;
    int recent_tokens[128];
    int recent_count = 0;

    /* N-gram loop detection: track recent 4-grams to detect infinite loops.
     * Small models with T=0 greedy decoding enter repetition loops where
     * the same ~30-token pattern repeats endlessly. KV quantization error
     * compounds through these repetitions, eventually collapsing output
     * into garbage. Detecting loops early prevents wasted compute. */
    uint32_t ngram_hashes[64];
    int ngram_hash_count = 0;
    int loop_detected = 0;

    /* Seed recent tokens with tail of prompt for better penalty coverage */
    for (int i = (n_prompt > rep_window ? n_prompt - rep_window : 0); i < n_prompt; i++) {
        recent_tokens[recent_count % 128] = prompt_tokens[i];
        recent_count++;
    }

    /* Apply repetition penalty to logits before first sample */
    if (rep_penalty > 1.0f) {
        int window = recent_count < rep_window ? recent_count : rep_window;
        for (int r = 0; r < window; r++) {
            int idx = (recent_count - 1 - r) % 128;
            if (idx < 0) idx += 128;
            int tok = recent_tokens[idx];
            if (tok >= 0 && tok < vocab_size) {
                if (state->logits[tok] > 0)
                    state->logits[tok] /= rep_penalty;
                else
                    state->logits[tok] *= rep_penalty;
            }
        }
    }

    /* Sample first generated token. The seed is configurable via
     * config->rng_seed to support reproducible sampling sweeps; 0 falls
     * back to the historical default of 42 so existing callers that
     * never set rng_seed get bit-identical behaviour. */
    int pos = pos_after_prefill;
    unsigned long long rng_state = config->rng_seed ? config->rng_seed : 42ULL;
    int next_token = tq_sample_topp(state->logits, vocab_size,
                                     config->temperature, config->top_p,
                                     &rng_state);

    /* Record first sampled token */
    recent_tokens[recent_count % 128] = next_token;
    recent_count++;

    int generated = 0;
    int output_pos = 0;
    int prev_token = prompt_tokens[n_prompt - 1];

    /* EOS token IDs — check common values across model families.
     * Qwen3.5: eos = 248044 (<|endoftext|>), 248046 (<|im_end|>)
     * Gemma3: eos = 1
     * Gemma4: eos = 106 (<end_of_turn>)
     * LLaMA 2: eos = 2
     * LLaMA 3: eos = 128001 (<|end_of_text|>), 128009 (<|eot_id|>) */
    int eos_tokens[] = {
        1,       /* Gemma3 <eos> */
        2,       /* LLaMA 2 </s> */
        106,     /* Gemma4 <end_of_turn> */
        128001,  /* LLaMA 3 <|end_of_text|> */
        128006,  /* LLaMA 3 <|start_header_id|> (new turn = stop) */
        128007,  /* LLaMA 3 <|end_header_id|> */
        128008,  /* LLaMA 3 <|start_of_role|> */
        128009,  /* LLaMA 3 <|eot_id|> */
        248044,  /* Qwen <|endoftext|> */
        248046,  /* Qwen <|im_end|> */
    };
    int n_eos = sizeof(eos_tokens) / sizeof(eos_tokens[0]);

    /* Generate loop */
    while (generated < config->max_tokens) {
        int is_eos = 0;
        for (int e = 0; e < n_eos; e++) {
            if (next_token == eos_tokens[e]) { is_eos = 1; break; }
        }
        if (is_eos) break;
        /* Infinite scrollback: when context is full, shift the KV cache
         * instead of stopping. Keep the last half of the context (including
         * the FP32 hot window) and discard the oldest half. This mirrors
         * human memory: ancient context fades, recent stays sharp.
         *
         * After shift, pos is reset to keep_count and generation continues.
         * The KV cache data for discarded positions is simply overwritten
         * by future tokens — no explicit deletion needed for the quantized
         * cache (block-indexed by position modulo max_seq_len). */
        if (pos >= model->config.max_seq_len) {
            int max_seq = model->config.max_seq_len;
            int keep_count = max_seq / 2;  /* keep most recent half */
            int discard = pos - keep_count;
            if (discard <= 0) break;  /* safety: can't shift if nothing to discard */

            fprintf(stderr, "[infinite scrollback] context full at %d, "
                    "shifting: discard oldest %d, keep %d\n",
                    pos, discard, keep_count);

            /* Shift FP32 key/value caches (if present) */
            int kv_dim = model->config.n_kv_heads * model->config.head_dim;
            for (int l = 0; l < model->config.n_layers; l++) {
                size_t layer_off = (size_t)l * max_seq * kv_dim;
                if (state->key_cache) {
                    memmove(state->key_cache + layer_off,
                            state->key_cache + layer_off + (size_t)discard * kv_dim,
                            (size_t)keep_count * kv_dim * sizeof(float));
                }
                if (state->value_cache) {
                    memmove(state->value_cache + layer_off,
                            state->value_cache + layer_off + (size_t)discard * kv_dim,
                            (size_t)keep_count * kv_dim * sizeof(float));
                }
                if (state->value_cache_fp16) {
                    size_t layer_off16 = (size_t)l * max_seq * kv_dim;
                    memmove(state->value_cache_fp16 + layer_off16,
                            state->value_cache_fp16 + layer_off16 + (size_t)discard * kv_dim,
                            (size_t)keep_count * kv_dim * sizeof(uint16_t));
                }
                /* Quantized K cache: shift block-level data */
                if (state->quant_key_cache && state->kv_quant_type < TQ_TYPE_COUNT) {
                    size_t blk_sz = tq_type_type_size(state->kv_quant_type);
                    size_t q_stride = (size_t)max_seq * blk_sz;
                    uint8_t* qbase = (uint8_t*)state->quant_key_cache + (size_t)l * q_stride;
                    memmove(qbase,
                            qbase + (size_t)discard * blk_sz,
                            (size_t)keep_count * blk_sz);
                }
            }

            /* Reset position: keep absolute position for correct RoPE.
             * Keys in the KV cache have RoPE baked in at their original
             * positions. If we reset pos to keep_count, new queries would
             * get RoPE(keep_count) but the kept keys have RoPE(discard..pos),
             * giving wrong relative distances. Instead, DON'T change pos —
             * continue from the same absolute position. The attention will
             * only scan positions [discard..pos] which are now at cache
             * indices [0..keep_count]. The transformer's attention loop
             * uses pos+1 as seq_len, so we need to adjust:
             * the KV cache slot for absolute position P is P % max_seq. */
            /* For now: use the simpler approach matching llama.cpp's
             * context shift: keep pos as-is but wrap cache indices. */
            pos = keep_count;
            /* NOTE: this has a RoPE mismatch — same as llama.cpp's
             * basic context shift. Quality degrades ~2-5% per shift.
             * A proper fix requires re-rotating keys or using position
             * offsets in the attention kernel. Tracked for v0.11. */
        }

        /* Decode token to text */
        if (tokenizer) {
            const char* piece = tq_decode(tokenizer, prev_token, next_token);

            /* Skip special/thinking tokens that shouldn't appear in output.
             * Qwen3.5: <think>...</think>
             * Gemma 4: thought, <channel|>, <tool|>, <mask>, <unused*>
             * LLaMA 3: <|start_header_id|>, <|reserved_special_token_*|> */
            int should_stop = 0;
            if (piece) {
                if (strstr(piece, "<think>") || strstr(piece, "</think>") ||
                    strstr(piece, "<channel|>") || strstr(piece, "<tool|>") ||
                    strstr(piece, "<mask>") ||
                    strstr(piece, "<unused") || strstr(piece, "<|think")) {
                    piece = "";
                }
                /* Gemma 4 "thought" token: only filter if it's the EXACT piece
                 * (not a substring of normal text like "thoughtful") */
                if (piece[0] != '\0' && strcmp(piece, "thought") == 0) {
                    piece = "";
                }
                /* Stop generation on turn-boundary tokens (LLaMA 3 / Qwen only).
                 * Gemma uses token ID-based EOS (106), not text-based detection. */
                if (strstr(piece, "<|start_header_id|>") ||
                    strstr(piece, "<|eot_id|>") ||
                    strstr(piece, "<|im_end|>")) {
                    should_stop = 1;
                    piece = "";
                }
                /* Filter reserved special tokens */
                if (strstr(piece, "<|reserved_special_token") ||
                    strstr(piece, "<1st>") || strstr(piece, "<2nd>") || strstr(piece, "<3rd>")) {
                    piece = "";
                }
            }
            if (should_stop) break;

            /* Also check accumulated output for turn markers that span multiple tokens */
            if (output && output_pos > 5) {
                const char* tail = output + (output_pos > 20 ? output_pos - 20 : 0);
                if (strstr(tail, "<|start_header") || strstr(tail, "<|eot_id") ||
                    strstr(tail, "<end_of_turn") || strstr(tail, "<|im_end")) {
                    /* Trim the marker from output */
                    char* marker = strstr(output + (output_pos > 30 ? output_pos - 30 : 0), "<|");
                    if (!marker) marker = strstr(output + (output_pos > 30 ? output_pos - 30 : 0), "<end");
                    if (marker) { *marker = '\0'; output_pos = (int)(marker - output); }
                    break;
                }
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
        recent_tokens[recent_count % 128] = next_token;
        recent_count++;

        /* N-gram loop detection: hash recent 4-gram and check for repeats */
        if (recent_count >= 4) {
            uint32_t h = 0;
            for (int r = 0; r < 4; r++) {
                int gi = (recent_count - 4 + r) % 128;
                h = h * 31 + (uint32_t)recent_tokens[gi];
            }
            int matches = 0;
            int ring_len = ngram_hash_count < 64 ? ngram_hash_count : 64;
            for (int r = 0; r < ring_len; r++) {
                if (ngram_hashes[r] == h) matches++;
            }
            ngram_hashes[ngram_hash_count % 64] = h;
            ngram_hash_count++;
            if (matches >= 3) {
                loop_detected = 1;
                break;
            }
        }
    }

    if (loop_detected) {
        fprintf(stderr, "[generate] repetition loop detected after %d tokens, stopping\n", generated);
    }

    /* Null-terminate output */
    if (output && output_size > 0) {
        output[output_pos < output_size ? output_pos : output_size - 1] = '\0';
    }

    tq_free_state(state);
    return generated;
}

/* ============================================================================
 * tq_generate_continue — chat-mode generation with KV cache reuse.
 *
 * Caller-managed state: state and cached_tokens persist across calls.
 * Each call computes the longest common prefix between cached_tokens and
 * the new prompt, prefills only the diverging suffix, and updates the
 * cache record. Turns chat from O(history^2) into O(new_tokens_per_turn).
 * ============================================================================ */
static int tq_lcp_int(const int* a, int na, const int* b, int nb) {
    int lim = na < nb ? na : nb;
    int i = 0;
    while (i < lim && a[i] == b[i]) i++;
    return i;
}

int tq_generate_continue(tq_model_t* model,
                          tq_tokenizer_t* tokenizer,
                          tq_state_t* state,
                          const char* prompt,
                          tq_gen_config_t* config,
                          int** cached_tokens_io,
                          int*  n_cached_io,
                          int*  cached_capacity_io,
                          char* output, int output_size) {
    if (!model || !state || !config || !cached_tokens_io || !n_cached_io || !cached_capacity_io) {
        return -1;
    }

    /* Encode new prompt — use a heap buffer that grows on demand instead
     * of a fixed stack array. The previous int new_tokens[4096] silently
     * truncated long contexts (10+ turns of accumulated chat history).
     * Cap at the model's max_seq_len so we never exceed KV cache bounds. */
    int max_prompt = model->config.max_seq_len > 0
                       ? model->config.max_seq_len : 4096;
    int* new_tokens = (int*)malloc((size_t)max_prompt * sizeof(int));
    if (!new_tokens) return -1;
    int n_new = 0;
    if (tokenizer && prompt) {
        int add_bos = (model->config.model_type == 1) ? 1 : 0;
        n_new = tq_encode(tokenizer, prompt, new_tokens, max_prompt, add_bos);
    }
    if (n_new <= 0) {
        new_tokens[0] = (model->config.model_type == 1) ? 2 : 1;
        n_new = 1;
    }

    /* Sliding window: if the new prompt + reserved generation room would
     * exceed max_seq_len, drop the oldest tokens from the front of the
     * prompt. We keep the most recent (max_seq_len - max_tokens - 32) tokens.
     * Note: this discards conversation history; ideally callers send
     * pre-trimmed prompts, but this prevents catastrophic failure. */
    int reserve = config->max_tokens > 0 ? config->max_tokens : 256;
    int budget  = max_prompt - reserve - 32;
    if (budget < 64) budget = 64;
    if (n_new > budget) {
        int drop = n_new - budget;
        memmove(new_tokens, new_tokens + drop, (size_t)budget * sizeof(int));
        n_new = budget;
        /* Force full reprefill since the prefix shifted */
        *n_cached_io = 0;
    }

    int n_cached = *n_cached_io;
    int* cached_tokens = *cached_tokens_io;
    int lcp = tq_lcp_int(cached_tokens, n_cached, new_tokens, n_new);

    /* Prefill only the new suffix [lcp, n_new) */
    for (int i = lcp; i < n_new; i++) {
        tq_forward(model, state, new_tokens[i], i);
    }
    int pos = n_new;

    /* Track prefill metrics for observability */
    int prefill_tokens = n_new - lcp;
    int prefix_hit    = lcp;

    /* Grow cache buffer if needed */
    int needed_cap = n_new + config->max_tokens + 16;
    if (*cached_capacity_io < needed_cap) {
        int new_cap = needed_cap < 4096 ? 4096 : needed_cap;
        int* nb = (int*)realloc(*cached_tokens_io, (size_t)new_cap * sizeof(int));
        if (!nb) { free(new_tokens); return -1; }
        *cached_tokens_io = nb;
        *cached_capacity_io = new_cap;
        cached_tokens = nb;
    }
    memcpy(cached_tokens, new_tokens, (size_t)n_new * sizeof(int));
    *n_cached_io = n_new;
    n_cached = n_new;

    int vocab_size = model->config.vocab_size;
    float rep_penalty = config->rep_penalty;
    int rep_window = config->rep_window;
    if (rep_window > 64) rep_window = 64;
    int recent_tokens[64];
    int recent_count = 0;
    for (int i = (n_new > rep_window ? n_new - rep_window : 0); i < n_new; i++) {
        recent_tokens[recent_count % 64] = new_tokens[i];
        recent_count++;
    }

    if (rep_penalty > 1.0f) {
        int window = recent_count < rep_window ? recent_count : rep_window;
        for (int r = 0; r < window; r++) {
            int idx = (recent_count - 1 - r) % 64;
            if (idx < 0) idx += 64;
            int tok = recent_tokens[idx];
            if (tok >= 0 && tok < vocab_size && state->logits) {
                if (state->logits[tok] > 0) state->logits[tok] /= rep_penalty;
                else                         state->logits[tok] *= rep_penalty;
            }
        }
    }

    unsigned long long rng_state = config->rng_seed ? (unsigned long long)config->rng_seed
                                                    : (unsigned long long)time(NULL);
    int next_token = tq_sample_topp(state->logits, vocab_size,
                                     config->temperature, config->top_p,
                                     &rng_state);

    int generated = 0;
    int output_pos = 0;
    int prev_token = new_tokens[n_new - 1];

    int eos_tokens[] = {
        1, 2, 106, 128001, 128006, 128007, 128008, 128009, 248044, 248046,
    };
    int n_eos = sizeof(eos_tokens) / sizeof(eos_tokens[0]);

    while (generated < config->max_tokens) {
        int is_eos = 0;
        for (int e = 0; e < n_eos; e++) {
            if (next_token == eos_tokens[e]) { is_eos = 1; break; }
        }
        if (is_eos) break;
        if (pos >= model->config.max_seq_len) break;

        if (tokenizer) {
            const char* piece = tq_decode(tokenizer, prev_token, next_token);
            int should_stop = 0;
            if (piece) {
                if (strstr(piece, "<|im_end|>") || strstr(piece, "<|eot_id|>") ||
                    strstr(piece, "<|start_header_id|>")) {
                    should_stop = 1; piece = "";
                }
            }
            if (should_stop) break;
            int piece_len = (int)strlen(piece ? piece : "");
            if (config->on_token && piece) config->on_token(piece, config->user_data);
            if (output && piece && output_pos + piece_len < output_size - 1) {
                memcpy(output + output_pos, piece, piece_len);
                output_pos += piece_len;
            }
        }

        if (n_cached < *cached_capacity_io) {
            cached_tokens[n_cached++] = next_token;
            *n_cached_io = n_cached;
        }

        prev_token = next_token;
        tq_forward(model, state, next_token, pos);
        pos++;
        generated++;

        if (rep_penalty > 1.0f) {
            int window = recent_count < rep_window ? recent_count : rep_window;
            for (int r = 0; r < window; r++) {
                int idx = (recent_count - 1 - r) % 64;
                if (idx < 0) idx += 64;
                int tok = recent_tokens[idx];
                if (tok >= 0 && tok < vocab_size) {
                    if (state->logits[tok] > 0) state->logits[tok] /= rep_penalty;
                    else                         state->logits[tok] *= rep_penalty;
                }
            }
        }

        next_token = tq_sample_topp(state->logits, vocab_size,
                                     config->temperature, config->top_p,
                                     &rng_state);
        recent_tokens[recent_count % 64] = next_token;
        recent_count++;
    }

    if (output && output_size > 0) {
        output[output_pos < output_size ? output_pos : output_size - 1] = '\0';
    }

    /* Log cache metrics: prefix_hit / prefill_tokens / generated.
     * Useful for tuning chat clients that want to maximize KV reuse. */
    if (getenv("TQ_CHAT_DEBUG")) {
        fprintf(stderr,
            "[chat] prefix_hit=%d prefill=%d generated=%d cached=%d\n",
            prefix_hit, prefill_tokens, generated, *n_cached_io);
    }

    free(new_tokens);
    return generated;
}

/* ============================================================================
 * tq_generate_chat_text — text-prefix matching for chat reuse
 *
 * Solves the BPE re-tokenization issue: when the model generates response
 * tokens via sample_topp, those token IDs may not match what tq_encode()
 * produces from the same response text in the next turn's prompt. The
 * token-level LCP in tq_generate_continue truncates at that boundary.
 *
 * This function tracks the *text* of the last prompt (which includes the
 * model's response from previous turns, accumulated by the caller). On the
 * next call, if the new prompt starts with cached_text byte-for-byte, the
 * entire cached state is valid — we tokenize only the new SUFFIX text and
 * prefill those tokens at positions [n_cached..]. No LCP, no truncation.
 *
 * After generation, *cached_text_io is updated to:
 *   prompt + (generated tokens decoded back to text)
 * so the next call can fast-path again.
 *
 * Caller owns *cached_text_io (must free with free()).
 * Pass cached_text_io == NULL to disable text-prefix tracking and behave
 * exactly like tq_generate_continue.
 * ============================================================================ */

typedef struct {
    char*  buf;
    size_t len;
    size_t cap;
    void (*user_cb)(const char*, void*);
    void*  user_data;
} chat_accum_t;

static void chat_accum_callback(const char* tok, void* u) {
    chat_accum_t* ctx = (chat_accum_t*)u;
    if (!tok) return;
    size_t tlen = strlen(tok);
    if (ctx->len + tlen + 1 > ctx->cap) {
        size_t new_cap = (ctx->cap + tlen + 64) * 2;
        char* nb = (char*)realloc(ctx->buf, new_cap);
        if (!nb) return;
        ctx->buf = nb;
        ctx->cap = new_cap;
    }
    memcpy(ctx->buf + ctx->len, tok, tlen);
    ctx->len += tlen;
    ctx->buf[ctx->len] = '\0';
    if (ctx->user_cb) ctx->user_cb(tok, ctx->user_data);
}

int tq_generate_chat_text(tq_model_t* model,
                           tq_tokenizer_t* tokenizer,
                           tq_state_t* state,
                           const char* prompt,
                           tq_gen_config_t* config,
                           char** cached_text_io,
                           int** cached_tokens_io,
                           int*  n_cached_io,
                           int*  cached_capacity_io,
                           char* output, int output_size) {
    if (!model || !state || !config || !cached_tokens_io || !n_cached_io || !cached_capacity_io || !prompt) {
        return -1;
    }

    /* --- 1. Check for text-level prefix match --- */
    int matched_text_len = 0;
    int prefix_pos = 0;  /* tokens already in KV cache that we trust */

    if (cached_text_io && *cached_text_io && *n_cached_io > 0) {
        size_t cached_len = strlen(*cached_text_io);
        if (cached_len > 0 && strncmp(*cached_text_io, prompt, cached_len) == 0) {
            matched_text_len = (int)cached_len;
            prefix_pos = *n_cached_io;
        } else if (getenv("TQ_CHAT_DEBUG")) {
            /* Find where they diverge to help diagnose */
            size_t diverge = 0;
            size_t plen = strlen(prompt);
            size_t lim = cached_len < plen ? cached_len : plen;
            while (diverge < lim && (*cached_text_io)[diverge] == prompt[diverge]) diverge++;
            fprintf(stderr,
                "[chat-text] no match: cached_len=%zu prompt_len=%zu diverge_at=%zu\n"
                "  cached[%zu..]: %.40s\n"
                "  prompt[%zu..]: %.40s\n",
                cached_len, plen, diverge,
                diverge, *cached_text_io + diverge,
                diverge, prompt + diverge);
        }
    }

    /* Wrap user callback to capture generated text into a buffer for the
     * next call's cached_text update. */
    chat_accum_t accum = { .buf = NULL, .len = 0, .cap = 0,
                            .user_cb = config->on_token,
                            .user_data = config->user_data };
    void (*orig_cb)(const char*, void*) = config->on_token;
    void*  orig_ud = config->user_data;
    config->on_token = chat_accum_callback;
    config->user_data = &accum;

    int generated = 0;

    if (matched_text_len > 0) {
        /* --- Fast path: text prefix matches --- */
        const char* suffix = prompt + matched_text_len;
        int max_prompt = model->config.max_seq_len > 0
                           ? model->config.max_seq_len : 4096;
        int* suffix_toks = (int*)malloc((size_t)max_prompt * sizeof(int));
        if (!suffix_toks) {
            config->on_token = orig_cb; config->user_data = orig_ud;
            return -1;
        }
        int n_suffix = 0;
        if (*suffix != '\0') {
            n_suffix = tq_encode(tokenizer, suffix, suffix_toks, max_prompt, 0);
            if (n_suffix < 0) n_suffix = 0;
        }

        /* Sliding window if needed (drop from start of cached) */
        int reserve = config->max_tokens > 0 ? config->max_tokens : 256;
        if (prefix_pos + n_suffix + reserve + 32 > max_prompt) {
            /* Force a full reprefill — simpler than partial cache shift */
            free(suffix_toks);
            config->on_token = orig_cb; config->user_data = orig_ud;
            *n_cached_io = 0;
            if (cached_text_io && *cached_text_io) {
                free(*cached_text_io); *cached_text_io = NULL;
            }
            int n2 = tq_generate_continue(model, tokenizer, state, prompt, config,
                                           cached_tokens_io, n_cached_io, cached_capacity_io,
                                           output, output_size);
            /* fall-through path captures cached_text below */
            generated = n2;
            goto update_cache;
        }

        /* Grow cache buffer */
        int needed = prefix_pos + n_suffix + reserve + 16;
        if (*cached_capacity_io < needed) {
            int new_cap = needed < 4096 ? 4096 : needed;
            int* nb = (int*)realloc(*cached_tokens_io, (size_t)new_cap * sizeof(int));
            if (!nb) { free(suffix_toks); config->on_token = orig_cb; config->user_data = orig_ud; return -1; }
            *cached_tokens_io = nb;
            *cached_capacity_io = new_cap;
        }

        /* Append suffix tokens to cache + prefill at correct positions */
        int* cached = *cached_tokens_io;
        for (int i = 0; i < n_suffix; i++) {
            cached[prefix_pos + i] = suffix_toks[i];
            tq_forward(model, state, suffix_toks[i], prefix_pos + i);
        }
        *n_cached_io = prefix_pos + n_suffix;
        free(suffix_toks);

        if (getenv("TQ_CHAT_DEBUG")) {
            fprintf(stderr, "[chat-text] FAST text_match=%d new_suffix_tokens=%d\n",
                    matched_text_len, n_suffix);
        }

        /* --- Run generation loop directly --- */
        int vocab_size = model->config.vocab_size;
        int n_cached = *n_cached_io;
        int pos = n_cached;
        int prev_token = n_cached > 0 ? cached[n_cached - 1] : 1;

        unsigned long long rng_state = config->rng_seed
            ? (unsigned long long)config->rng_seed : (unsigned long long)time(NULL);
        int next_token = tq_sample_topp(state->logits, vocab_size,
                                         config->temperature, config->top_p,
                                         &rng_state);

        int output_pos = 0;
        int eos_tokens[] = { 1, 2, 106, 128001, 128006, 128007, 128008, 128009, 248044, 248046 };
        int n_eos = sizeof(eos_tokens) / sizeof(eos_tokens[0]);

        while (generated < config->max_tokens) {
            int is_eos = 0;
            for (int e = 0; e < n_eos; e++) {
                if (next_token == eos_tokens[e]) { is_eos = 1; break; }
            }
            if (is_eos) break;
            if (pos >= model->config.max_seq_len) break;

            const char* piece = tokenizer ? tq_decode(tokenizer, prev_token, next_token) : "";
            int should_stop = 0;
            if (piece) {
                if (strstr(piece, "<|im_end|>") || strstr(piece, "<|eot_id|>") ||
                    strstr(piece, "<|start_header_id|>")) {
                    should_stop = 1; piece = "";
                }
            }
            if (should_stop) break;

            int piece_len = (int)strlen(piece ? piece : "");
            if (config->on_token && piece) config->on_token(piece, config->user_data);
            if (output && piece && output_pos + piece_len < output_size - 1) {
                memcpy(output + output_pos, piece, piece_len);
                output_pos += piece_len;
            }

            if (n_cached < *cached_capacity_io) {
                cached[n_cached++] = next_token;
                *n_cached_io = n_cached;
            }

            prev_token = next_token;
            tq_forward(model, state, next_token, pos);
            pos++;
            generated++;

            next_token = tq_sample_topp(state->logits, vocab_size,
                                         config->temperature, config->top_p,
                                         &rng_state);
        }

        if (output && output_size > 0) {
            output[output_pos < output_size ? output_pos : output_size - 1] = '\0';
        }
    } else {
        /* --- Slow path: no text-prefix match, use token LCP fallback --- */
        if (getenv("TQ_CHAT_DEBUG")) {
            fprintf(stderr, "[chat-text] SLOW no text-prefix match, full tokenize\n");
        }
        generated = tq_generate_continue(
            model, tokenizer, state, prompt, config,
            cached_tokens_io, n_cached_io, cached_capacity_io,
            output, output_size);
    }

update_cache:
    /* Restore the original callback before returning to caller */
    config->on_token = orig_cb;
    config->user_data = orig_ud;

    /* Update cached_text = prompt + generated text. The next call can
     * fast-path against this if its prompt starts with this string. */
    if (cached_text_io) {
        size_t plen = strlen(prompt);
        size_t glen = accum.len;
        size_t new_len = plen + glen;
        char* nt = (char*)malloc(new_len + 1);
        if (nt) {
            memcpy(nt, prompt, plen);
            if (glen > 0 && accum.buf) memcpy(nt + plen, accum.buf, glen);
            nt[new_len] = '\0';
            if (*cached_text_io) free(*cached_text_io);
            *cached_text_io = nt;
        }
    }
    if (accum.buf) free(accum.buf);

    return generated;
}
