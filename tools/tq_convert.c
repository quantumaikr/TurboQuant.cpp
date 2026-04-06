/**
 * tq_convert — Convert safetensors model to TQM format
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
#ifndef _WIN32
#include <unistd.h>
#endif
#include <dirent.h>

static void print_usage(const char* prog) {
    fprintf(stderr, "TQM Converter — Pre-quantize models for instant loading\n\n");
    fprintf(stderr, "Usage: %s [model.safetensors] [tokenizer.json] [-o output.tqm]\n\n", prog);
    fprintf(stderr, "  All arguments are optional — auto-detects Qwen3.5-0.8B or Gemma-3-270m from HuggingFace cache.\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -o <path>     Output file (default: model.tqm)\n");
    fprintf(stderr, "  -j <threads>  Threads for quantization (default: 4)\n");
    fprintf(stderr, "  -h, --help    Show this help\n");
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  %s                              # auto-detect + convert\n", prog);
    fprintf(stderr, "  %s -o qwen.tqm                  # auto-detect, custom output\n", prog);
    fprintf(stderr, "  %s model.safetensors tok.json -o out.tqm  # explicit paths\n", prog);
}

int main(int argc, char** argv) {
    /* argc < 2 is OK — auto-detect will find model */

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

    /* Auto-detect model from HuggingFace cache if not specified */
    if (!model_path) {
        const char* home = getenv("HOME");
        if (home) {
            static char auto_model[4096];
            static char auto_tok[4096];
            /* Try common Qwen3.5-0.8B cache locations */
            const char* base = "/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots";
            snprintf(auto_model, sizeof(auto_model), "%s%s", home, base);
            /* Find snapshot directory */
            DIR* dir = opendir(auto_model);
            if (dir) {
                struct dirent* ent;
                while ((ent = readdir(dir)) != NULL) {
                    if (ent->d_name[0] == '.') continue;
                    char try_path[2048];
                    /* Try single-file safetensors */
                    snprintf(try_path, sizeof(try_path), "%s/%s/model.safetensors",
                             auto_model, ent->d_name);
                    if (access(try_path, R_OK) == 0) {
                        snprintf(auto_model, sizeof(auto_model), "%s", try_path);
                        model_path = auto_model;
                    }
                    /* Try multi-shard */
                    if (!model_path) {
                        snprintf(try_path, sizeof(try_path),
                                 "%s/%s/model.safetensors-00001-of-00001.safetensors",
                                 auto_model, ent->d_name);
                        /* auto_model was overwritten, reconstruct */
                        snprintf(auto_model, sizeof(auto_model), "%s%s", home, base);
                        snprintf(try_path, sizeof(try_path),
                                 "%s/%s/model.safetensors-00001-of-00001.safetensors",
                                 auto_model, ent->d_name);
                        if (access(try_path, R_OK) == 0) {
                            snprintf(auto_model, sizeof(auto_model), "%s", try_path);
                            model_path = auto_model;
                        }
                    }
                    /* Auto-detect tokenizer too */
                    if (model_path && !tokenizer_path) {
                        char* last_slash = strrchr(auto_model, '/');
                        if (last_slash) {
                            size_t dir_len = last_slash - auto_model;
                            snprintf(auto_tok, sizeof(auto_tok), "%.*s/tokenizer.json",
                                     (int)dir_len, auto_model);
                            if (access(auto_tok, R_OK) == 0) {
                                tokenizer_path = auto_tok;
                            }
                        }
                    }
                    if (model_path) break;
                }
                closedir(dir);
            }
        }
    }

    /* Try Gemma3 model paths if Qwen not found */
    if (!model_path) {
        const char* home = getenv("HOME");
        if (home) {
            static char auto_model_g[4096];
            static char auto_tok_g[4096];
            /* Try Gemma3 270M cache locations */
            const char* gemma_bases[] = {
                "/.cache/huggingface/hub/models--unsloth--gemma-3-270m-it/snapshots",
                "/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots",
                NULL
            };
            for (int gi = 0; gemma_bases[gi] && !model_path; gi++) {
                snprintf(auto_model_g, sizeof(auto_model_g), "%s%s", home, gemma_bases[gi]);
                DIR* dir = opendir(auto_model_g);
                if (dir) {
                    struct dirent* ent;
                    while ((ent = readdir(dir)) != NULL) {
                        if (ent->d_name[0] == '.') { continue; }
                        char try_path[2048];
                        snprintf(try_path, sizeof(try_path), "%s/%s/model.safetensors",
                                 auto_model_g, ent->d_name);
                        if (access(try_path, R_OK) == 0) {
                            snprintf(auto_model_g, sizeof(auto_model_g), "%s", try_path);
                            model_path = auto_model_g;
                        }
                        if (model_path && !tokenizer_path) {
                            char* last_slash = strrchr(auto_model_g, '/');
                            if (last_slash) {
                                size_t dir_len = (size_t)(last_slash - auto_model_g);
                                snprintf(auto_tok_g, sizeof(auto_tok_g), "%.*s/tokenizer.json",
                                         (int)dir_len, auto_model_g);
                                if (access(auto_tok_g, R_OK) == 0) {
                                    tokenizer_path = auto_tok_g;
                                }
                            }
                        }
                        if (model_path) { break; }
                    }
                    closedir(dir);
                }
            }
        }
    }

    /* Try Gemma3 4B model paths (multi-shard) if still not found */
    if (!model_path) {
        const char* home = getenv("HOME");
        if (home) {
            static char auto_model_g4[2048];
            static char auto_tok_g4[2048];
            const char* gemma4b_bases[] = {
                "/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots",
                "/.cache/huggingface/hub/models--unsloth--gemma-3-4b-it/snapshots",
                NULL
            };
            for (int gi = 0; gemma4b_bases[gi] && !model_path; gi++) {
                char snap_dir[2048];
                snprintf(snap_dir, sizeof(snap_dir), "%s%s", home, gemma4b_bases[gi]);
                DIR* dir = opendir(snap_dir);
                if (dir) {
                    struct dirent* ent;
                    while ((ent = readdir(dir)) != NULL) {
                        if (ent->d_name[0] == '.') continue;
                        char try_path[2048];
                        /* Multi-shard: look for model.safetensors.index.json */
                        snprintf(try_path, sizeof(try_path), "%s/%s/model.safetensors.index.json",
                                 snap_dir, ent->d_name);
                        if (access(try_path, R_OK) == 0) {
                            /* Point to model.safetensors (the loader detects index.json) */
                            snprintf(auto_model_g4, sizeof(auto_model_g4),
                                     "%s/%s/model.safetensors", snap_dir, ent->d_name);
                            model_path = auto_model_g4;
                        }
                        /* Fallback: single file */
                        if (!model_path) {
                            snprintf(try_path, sizeof(try_path), "%s/%s/model.safetensors",
                                     snap_dir, ent->d_name);
                            if (access(try_path, R_OK) == 0) {
                                snprintf(auto_model_g4, sizeof(auto_model_g4), "%s", try_path);
                                model_path = auto_model_g4;
                            }
                        }
                        if (model_path && !tokenizer_path) {
                            char* last_slash = strrchr(auto_model_g4, '/');
                            if (last_slash) {
                                size_t dir_len = (size_t)(last_slash - auto_model_g4);
                                snprintf(auto_tok_g4, sizeof(auto_tok_g4), "%.*s/tokenizer.json",
                                         (int)dir_len, auto_model_g4);
                                if (access(auto_tok_g4, R_OK) == 0) {
                                    tokenizer_path = auto_tok_g4;
                                }
                            }
                        }
                        if (model_path) break;
                    }
                    closedir(dir);
                }
            }
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: model not found.\n");
        fprintf(stderr, "  Auto-detect searched:\n");
        fprintf(stderr, "    ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/\n");
        fprintf(stderr, "    ~/.cache/huggingface/hub/models--unsloth--gemma-3-270m-it/\n");
        fprintf(stderr, "    ~/.cache/huggingface/hub/models--google--gemma-3-270m-it/\n");
        fprintf(stderr, "    ~/.cache/huggingface/hub/models--google--gemma-3-4b-it/\n");
        fprintf(stderr, "  Specify manually: %s <model.safetensors> [tokenizer.json] -o output.tqm\n", argv[0]);
        return 1;
    }
    if (!output_path) {
        output_path = "model.tqm"; /* default output name */
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

    /* Step 2: Quantize weights */
    int use_q8 = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-q8") == 0 || strcmp(argv[i], "--q8") == 0) use_q8 = 1;
    }

    if (use_q8) {
        fprintf(stderr, "[2/3] Quantizing weights to Q8 (higher quality)...\n");
    } else {
        fprintf(stderr, "[2/3] Quantizing weights to Q4...\n");
    }
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    if (use_q8) {
        tq_quantize_weights(model); /* Q8 */
    } else {
        tq_quantize_weights_q4(model); /* Q4 */
    }

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
    fprintf(stderr, "\nTo use: quant %s -t tokenizer.json -p \"Hello\"\n", output_path);
    fprintf(stderr, "  (tokenizer is embedded — -t flag is optional with TQM)\n");

    tq_free_model(model);
    return 0;
}
