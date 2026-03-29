/**
 * tq_convert — Convert safetensors model to TQM (TurboQuant Model) format
 *
 * Usage:
 *   tq_convert <model.safetensors> [tokenizer.json] -o <output.tqm>
 *
 * The .tqm format stores pre-quantized Q4 weights that can be mmap'd
 * directly, eliminating the BF16->FP32->Q4 conversion at load time.
 * Typical loading speedup: 6s -> 0.5s for an 0.8B model.
 */

#include "turboquant/tq_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void print_usage(const char* prog) {
    fprintf(stderr, "TQM Converter — Pre-quantize models for instant loading\n\n");
    fprintf(stderr, "Usage: %s <model.safetensors> [tokenizer.json] -o <output.tqm>\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -o <path>     Output .tqm file path (required)\n");
    fprintf(stderr, "  -j <threads>  Number of threads for quantization (default: 4)\n");
    fprintf(stderr, "  -h, --help    Show this help\n");
    fprintf(stderr, "\nThe converter:\n");
    fprintf(stderr, "  1. Loads the safetensors model (BF16/FP32)\n");
    fprintf(stderr, "  2. Quantizes all weights to Q4 (4-bit, ~6x reduction)\n");
    fprintf(stderr, "  3. Writes a .tqm file with the tokenizer embedded\n");
    fprintf(stderr, "  4. The .tqm file can be mmap'd directly — no conversion needed\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* model_path = NULL;
    const char* tokenizer_path = NULL;
    const char* output_path = NULL;
    int n_threads = 4;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (argv[i][0] != '-') {
            if (!model_path) {
                model_path = argv[i];
            } else if (!tokenizer_path) {
                tokenizer_path = argv[i];
            }
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: model path required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (!output_path) {
        fprintf(stderr, "Error: output path required (-o)\n");
        print_usage(argv[0]);
        return 1;
    }

    tq_set_threads(n_threads);

    /* Step 1: Load model */
    fprintf(stderr, "[1/3] Loading model from %s...\n", model_path);
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    tq_model_t* model = tq_load_model(model_path);
    if (!model) {
        fprintf(stderr, "Error: failed to load model\n");
        return 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double load_time = (double)(ts1.tv_sec - ts0.tv_sec)
                     + (double)(ts1.tv_nsec - ts0.tv_nsec) / 1e9;

    tq_model_config_t* c = &model->config;
    fprintf(stderr, "  Model: %d layers, dim=%d, heads=%d/%d, vocab=%d\n",
            c->n_layers, c->hidden_dim, c->n_heads, c->n_kv_heads, c->vocab_size);
    fprintf(stderr, "  Load time: %.2f s\n", load_time);

    /* Step 2: Quantize to Q4 */
    fprintf(stderr, "[2/3] Quantizing weights to Q4...\n");
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    tq_quantize_weights_q4(model);

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double quant_time = (double)(ts1.tv_sec - ts0.tv_sec)
                      + (double)(ts1.tv_nsec - ts0.tv_nsec) / 1e9;
    fprintf(stderr, "  Quantization time: %.2f s\n", quant_time);

    /* Step 3: Write TQM */
    fprintf(stderr, "[3/3] Writing TQM to %s...\n", output_path);
    if (tokenizer_path) {
        fprintf(stderr, "  Embedding tokenizer from %s\n", tokenizer_path);
    }
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    int ret = tq_save_tqm(model, tokenizer_path, output_path);

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double write_time = (double)(ts1.tv_sec - ts0.tv_sec)
                      + (double)(ts1.tv_nsec - ts0.tv_nsec) / 1e9;

    if (ret != 0) {
        fprintf(stderr, "Error: failed to write TQM file\n");
        tq_free_model(model);
        return 1;
    }

    fprintf(stderr, "  Write time: %.2f s\n", write_time);
    fprintf(stderr, "\nDone! Total: %.2f s (load=%.2f, quant=%.2f, write=%.2f)\n",
            load_time + quant_time + write_time,
            load_time, quant_time, write_time);
    fprintf(stderr, "\nTo use: tq_run %s -t tokenizer.json -p \"Hello\"\n", output_path);
    fprintf(stderr, "  (tokenizer is embedded — -t flag is optional with TQM)\n");

    tq_free_model(model);
    return 0;
}
