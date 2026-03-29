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
 *                     polar_3b, polar_4b, turbo_3b, turbo_4b (default: uniform_4b)
 *   -s <seed>        Random seed (default: 42)
 *   --info           Print model info and exit
 */

#include "turboquant/tq_engine.h"
#include "turboquant/turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    fprintf(stderr, "  -s <seed>        Random seed (default: 42)\n");
    fprintf(stderr, "  --info           Print model info and exit\n");
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
    int info_only = 0;

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
        } else if (strcmp(argv[i], "--info") == 0) {
            info_only = 1;
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
    fprintf(stderr, "Model: %d layers, dim=%d, heads=%d/%d, vocab=%d, inter=%d\n",
            c->n_layers, c->hidden_dim, c->n_heads, c->n_kv_heads,
            c->vocab_size, c->intermediate_dim);
    fprintf(stderr, "KV cache type: %s\n",
            kv_type < TQ_TYPE_COUNT ? tq_type_name(kv_type) : "fp32");

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
    }

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
    int n_generated = tq_generate(model, tokenizer, prompt, &config,
                                   output, sizeof(output));

    fprintf(stderr, "\n---\n");
    fprintf(stderr, "Generated %d tokens\n", n_generated);

    /* Cleanup */
    if (tokenizer) tq_free_tokenizer(tokenizer);
    tq_free_model(model);

    return 0;
}
