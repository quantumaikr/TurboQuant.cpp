/**
 * tq_run — TurboQuant inference CLI
 *
 * Usage:
 *   tq_run <model.safetensors> [options]
 *
 * Options:
 *   -t <tokenizer>   Path to tokenizer binary file
 *   -p <prompt>      Input prompt (default: "Hello")
 *   -n <max_tokens>  Maximum tokens to generate (default: 256)
 *   -T <temperature> Sampling temperature (default: 0.7)
 *   -P <top_p>       Top-p nucleus sampling (default: 0.9)
 *   -k <kv_type>     KV cache type: fp32, uniform_4b, uniform_2b,
 *                     polar_3b, polar_4b, turbo_3b, turbo_4b,
 *                     turbo_kv_3b, turbo_kv_4b (default: uniform_4b)
 *   -j <threads>     Number of threads for matmul (default: 4)
 *   -s <seed>        Random seed (default: 42)
 *   --info           Print model info and exit
 *   -M, --memory     Print KV cache memory stats after generation
 */

#include "turboquant/tq_engine.h"
#include "turboquant/turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Streaming token callback */
static void print_token(const char* text, void* user_data) {
    (void)user_data;
    fputs(text, stdout);
    fflush(stdout);
}

/* Parse KV type from string */
static tq_type parse_kv_type(const char* s) {
    if (!s) return TQ_TYPE_UNIFORM_4B;
    if (strcmp(s, "fp32") == 0)       return TQ_TYPE_COUNT; /* sentinel for FP32 */
    if (strcmp(s, "uniform_4b") == 0) return TQ_TYPE_UNIFORM_4B;
    if (strcmp(s, "uniform_2b") == 0) return TQ_TYPE_UNIFORM_2B;
    if (strcmp(s, "polar_3b") == 0)   return TQ_TYPE_POLAR_3B;
    if (strcmp(s, "polar_4b") == 0)   return TQ_TYPE_POLAR_4B;
    if (strcmp(s, "turbo_3b") == 0)   return TQ_TYPE_TURBO_3B;
    if (strcmp(s, "turbo_4b") == 0)   return TQ_TYPE_TURBO_4B;
    if (strcmp(s, "turbo_kv_3b") == 0) return TQ_TYPE_TURBO_KV_3B;
    if (strcmp(s, "turbo_kv_4b") == 0) return TQ_TYPE_TURBO_KV_4B;
    if (strcmp(s, "qjl_1b") == 0)     return TQ_TYPE_QJL_1B;
    if (strcmp(s, "mixed_4b8") == 0)  return TQ_TYPE_MIXED_4B8;
    fprintf(stderr, "Unknown KV type: %s (using uniform_4b)\n", s);
    return TQ_TYPE_UNIFORM_4B;
}

static void print_usage(const char* prog) {
    fprintf(stderr, "TurboQuant Inference Engine\n");
    fprintf(stderr, "Usage: %s <model.safetensors> [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <tokenizer>   Tokenizer binary file\n");
    fprintf(stderr, "  -p <prompt>      Input prompt (default: \"Hello\")\n");
    fprintf(stderr, "  -n <max_tokens>  Max tokens to generate (default: 256)\n");
    fprintf(stderr, "  -T <temperature> Sampling temperature (default: 0.7)\n");
    fprintf(stderr, "  -P <top_p>       Top-p sampling (default: 0.9)\n");
    fprintf(stderr, "  -k <kv_type>     KV cache quantization type\n");
    fprintf(stderr, "  -j <threads>     Number of threads for matmul (default: 4)\n");
    fprintf(stderr, "  -s <seed>        Random seed (default: 42)\n");
    fprintf(stderr, "  -q <type>        Quantize weights: q4 (4-bit, ~6x reduction, default),\n");
    fprintf(stderr, "                   q8 (int8, ~3.5x reduction), or none (FP32)\n");
    fprintf(stderr, "  --info           Print model info and exit\n");
    fprintf(stderr, "  -M, --memory     Print KV cache memory stats after generation\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    /* Parse arguments */
    const char* model_path = NULL;
    const char* tokenizer_path = NULL;
    const char* prompt = "Hello";
    int max_tokens = 256;
    float temperature = 0.7f;
    float top_p = 0.9f;
    tq_type kv_type = TQ_TYPE_UNIFORM_4B;
    int n_threads = 4;
    int quant_mode = 0;   /* 0 = none (default), 4 = Q4, 8 = Q8 */
    int info_only = 0;
    int show_memory = 0;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            model_path = argv[i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-T") == 0 && i + 1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-P") == 0 && i + 1 < argc) {
            top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            kv_type = parse_kv_type(argv[++i]);
        } else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-q") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                const char* qarg = argv[++i];
                if (strcmp(qarg, "q4") == 0 || strcmp(qarg, "4") == 0) {
                    quant_mode = 4;
                } else if (strcmp(qarg, "q8") == 0 || strcmp(qarg, "8") == 0) {
                    quant_mode = 8;
                } else if (strcmp(qarg, "none") == 0 || strcmp(qarg, "fp32") == 0) {
                    quant_mode = 0;
                } else {
                    fprintf(stderr, "Unknown quant type: %s (using q4)\n", qarg);
                    quant_mode = 4;
                }
            } else {
                quant_mode = 4;  /* -q alone defaults to Q4 */
            }
        } else if (strcmp(argv[i], "--info") == 0) {
            info_only = 1;
        } else if (strcmp(argv[i], "-M") == 0 || strcmp(argv[i], "--memory") == 0) {
            show_memory = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: model path required\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Load model */
    fprintf(stderr, "Loading model from %s...\n", model_path);
    tq_model_t* model = tq_load_model(model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    /* Print model info */
    tq_model_config_t* c = &model->config;
    fprintf(stderr, "Model: %d layers, dim=%d, heads=%d/%d, head_dim=%d, vocab=%d, inter=%d\n",
            c->n_layers, c->hidden_dim, c->n_heads, c->n_kv_heads,
            c->head_dim, c->vocab_size, c->intermediate_dim);
    fprintf(stderr, "KV cache type: %s\n",
            kv_type < TQ_TYPE_COUNT ? tq_type_name(kv_type) : "fp32");

    if (quant_mode == 4) {
        fprintf(stderr, "Quantizing weights to Q4 (4-bit)...\n");
        tq_quantize_weights_q4(model);
    } else if (quant_mode == 8) {
        fprintf(stderr, "Quantizing weights to Q8 (int8)...\n");
        tq_quantize_weights(model);
    }

    if (info_only) {
        tq_free_model(model);
        return 0;
    }

    /* Load tokenizer */
    tq_tokenizer_t* tokenizer = NULL;
    if (tokenizer_path) {
        tokenizer = tq_load_tokenizer(tokenizer_path);
        if (!tokenizer) {
            fprintf(stderr, "Warning: failed to load tokenizer, using raw IDs\n");
        }
    } else {
        /* Try to load embedded tokenizer from TQM file */
        tokenizer = tq_load_tokenizer_from_tqm(model_path);
        if (tokenizer) {
            fprintf(stderr, "Loaded embedded tokenizer from TQM file\n");
        }
    }

    /* Set thread count for matmul parallelism */
    tq_set_threads(n_threads);
    fprintf(stderr, "Threads: %d\n", tq_get_threads());

    /* Configure generation */
    tq_gen_config_t config = tq_default_gen_config();
    config.temperature = temperature;
    config.top_p = top_p;
    config.max_tokens = max_tokens;
    config.kv_type = kv_type;
    config.on_token = print_token;
    config.user_data = NULL;

    /* Generate */
    fprintf(stderr, "Prompt: %s\n", prompt);
    fprintf(stderr, "---\n");

    char output[65536];

    /* Measure generation time for tok/s reporting */
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    int n_generated = tq_generate(model, tokenizer, prompt, &config,
                                   output, sizeof(output));

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed = (double)(ts_end.tv_sec - ts_start.tv_sec)
                   + (double)(ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    fprintf(stderr, "\n---\n");
    if (n_generated > 0 && elapsed > 0.0) {
        double tok_per_sec = (double)n_generated / elapsed;
        const char* wq_name = model->use_q4_weights ? "Q4" : (model->use_q8_weights ? "Q8" : "FP32");
        fprintf(stderr, "%d tokens in %.1fs (%.1f tok/s, %d threads, weights=%s, kv=%s)\n",
                n_generated, elapsed, tok_per_sec, tq_get_threads(), wq_name,
                kv_type < TQ_TYPE_COUNT ? tq_type_name(kv_type) : "fp32");
    } else {
        fprintf(stderr, "Generated %d tokens\n", n_generated);
    }

    /* Print KV cache memory stats if requested */
    if (show_memory && n_generated > 0) {
        int total_tokens = n_generated;

        /* FP16 KV baseline (llama.cpp default):
         * 2 (K+V) * n_layers * n_kv_heads * head_dim * 2 bytes per token */
        size_t fp16_per_token = (size_t)2 * c->n_layers * c->n_kv_heads * c->head_dim * 2;

        /* Compressed KV: both keys and values quantized to same type.
         * blocks_per_head * type_size bytes per head per layer, times 2 for K+V */
        size_t block_size = tq_type_block_size(kv_type);
        size_t type_size_bytes = tq_type_type_size(kv_type);
        if (block_size == 0) block_size = TQ_BK;
        if (type_size_bytes == 0) type_size_bytes = sizeof(block_tq_uniform_4b);
        size_t blocks_per_head = ((size_t)c->head_dim + block_size - 1) / block_size;

        /* Q4 K+V per token: 2 * n_layers * n_kv_heads * blocks_per_head * type_size */
        size_t compressed_per_token = (size_t)2 * c->n_layers * c->n_kv_heads
                                    * blocks_per_head * type_size_bytes;

        /* If kv_type is fp32 (sentinel), both key and value are FP32 */
        if (kv_type >= TQ_TYPE_COUNT) {
            compressed_per_token = (size_t)2 * c->n_layers * c->n_kv_heads
                                 * c->head_dim * sizeof(float);
        }

        /* Total bytes for all generated tokens */
        size_t total_compressed = compressed_per_token * (size_t)total_tokens;
        size_t total_fp16 = fp16_per_token * (size_t)total_tokens;

        float ratio = (total_compressed > 0) ? (float)total_fp16 / (float)total_compressed : 0.0f;

        fprintf(stderr, "\n=== KV Cache Memory Stats ===\n");
        fprintf(stderr, "Tokens in cache:      %d\n", total_tokens);
        fprintf(stderr, "Model config:         %d layers, %d kv_heads, head_dim=%d\n",
                c->n_layers, c->n_kv_heads, c->head_dim);
        fprintf(stderr, "KV type:              %s\n",
                kv_type < TQ_TYPE_COUNT ? tq_type_name(kv_type) : "fp32");
        fprintf(stderr, "Per-token KV (Q4):    %.2f KB\n",
                (double)compressed_per_token / 1024.0);
        fprintf(stderr, "Per-token KV (FP16):  %.2f KB\n",
                (double)fp16_per_token / 1024.0);
        fprintf(stderr, "Total KV (Q4):        %.2f MB\n",
                (double)total_compressed / (1024.0 * 1024.0));
        fprintf(stderr, "Total KV (FP16):      %.2f MB\n",
                (double)total_fp16 / (1024.0 * 1024.0));
        fprintf(stderr, "Compression ratio:    %.2fx\n", ratio);
        fprintf(stderr, "Memory saved:         %.2f MB\n",
                (double)(total_fp16 - total_compressed) / (1024.0 * 1024.0));
        fprintf(stderr, "=============================\n");

        /* Machine-parseable line for scripts */
        fprintf(stderr, "MEMORY_CSV:%d,%zu,%zu,%.4f\n",
                total_tokens, total_compressed, total_fp16, ratio);
    }

    /* Cleanup */
    if (tokenizer) tq_free_tokenizer(tokenizer);
    tq_free_model(model);

    return 0;
}
