/**
 * quant — Minimal C inference engine. Zero dependencies.
 *
 * Usage:
 *   quant <model.gguf> [options]
 *
 * Options:
 *   -t <tokenizer>   Path to tokenizer binary file
 *   -p <prompt>      Input prompt (default: "Hello")
 *   -n <max_tokens>  Maximum tokens to generate (default: 256)
 *   -T <temperature> Sampling temperature (default: 0.7)
 *   -P <top_p>       Top-p nucleus sampling (default: 0.9)
 *   -k <kv_type>     KV cache type: fp32, uniform_4b, uniform_2b,
 *                     polar_3b, polar_4b, turbo_3b, turbo_4b,
 *                     turbo_kv_1b, turbo_kv_3b, turbo_kv_4b (default: turbo_kv_4b)
 *   -v <vq>          Value cache quantization: q4 (4-bit), q2 (2-bit),
 *                     or fp16 (default: fp16 when -k is set, fp32 otherwise)
 *   -j <threads>     Number of threads for matmul (default: 4)
 *   -s <seed>        Random seed (default: 42)
 *   --info           Print model info and exit
 *   -M, --memory     Print KV cache memory stats after generation
 *   --profile-kv     Profile KV activation distributions (pre/post RHT)
 *   --recommend      Run --profile-kv and output per-layer bit recommendations
 *   --attn-entropy   Compute per-layer, per-head attention entropy during generation
 *   -V <N>           V highres window: recent N tokens get FP16 V (0=disabled)
 *   --calibrate      Run online Lloyd-Max codebook calibration analysis
 *   --ppl <file>     Compute perplexity on a text file (teacher-forced)
 *   --bench-memory   Benchmark memory bandwidth (tok/s at varying context lengths)
 *   --bench-prefill  Benchmark prefill speed with/without KV quantization
 */

#include "turboquant/tq_engine.h"
#include "turboquant/turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>  /* sysconf for default thread count */
#if defined(__APPLE__)
#include <sys/sysctl.h>  /* sysctlbyname for hw.perflevel0.physicalcpu */
#endif

/* MSVC: clock_gettime compatibility */
#ifdef _WIN32
#include <windows.h>
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
#if defined(_WIN32) && !defined(_TIMESPEC_DEFINED) && !defined(__struct_timespec_defined) && (!defined(_MSC_VER) || _MSC_VER < 1900)
#define _TIMESPEC_DEFINED 1
#define __struct_timespec_defined 1
struct timespec { long tv_sec; long tv_nsec; };
#endif
static int clock_gettime(int id, struct timespec* ts) {
    (void)id;
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    ts->tv_sec  = (long)(cnt.QuadPart / freq.QuadPart);
    ts->tv_nsec = (long)((cnt.QuadPart % freq.QuadPart) * 1000000000LL / freq.QuadPart);
    return 0;
}
#endif

/* Forward-pass profiling flag (defined in tq_transformer.c) */
extern int g_tq_profile_enabled;

/* Streaming token callback — filters chat-template control tokens that
 * would otherwise leak into CLI output when --chat is active. */
static void print_token(const char* text, void* user_data) {
    (void)user_data;
    if (!text || !text[0]) return;

    /* Skip thinking / template tokens (same list as server). Gemma 4 raw
     * output contains <|think|>, Qwen3 uses <think>/</think>, chat-templated
     * models may emit <|end|>/<|im_end|>/<|eot_id|> etc. */
    if (strstr(text, "<|think|>") || strstr(text, "<think>") ||
        strstr(text, "</think>") || strstr(text, "<|channel>") ||
        strstr(text, "<|turn>") || strstr(text, "<turn|>") ||
        strstr(text, "<|end|>") || strstr(text, "<|assistant|>") ||
        strstr(text, "<|user|>") || strstr(text, "<|system|>") ||
        strstr(text, "<|im_end|>") || strstr(text, "<|im_start|>") ||
        strstr(text, "<start_of_turn>") || strstr(text, "<end_of_turn>") ||
        strstr(text, "<|begin_of_text|>") || strstr(text, "<|end_of_text|>") ||
        strstr(text, "<|start_header_id|>") || strstr(text, "<|end_header_id|>") ||
        strstr(text, "<|eot_id|>"))
        return;

    fputs(text, stdout);
    fflush(stdout);
}

/* Parse KV type from string */
static tq_type parse_kv_type(const char* s) {
    if (!s) return TQ_TYPE_TURBO_KV_4B;
    if (strcmp(s, "fp32") == 0)       return TQ_TYPE_COUNT; /* sentinel for FP32 */
    if (strcmp(s, "uniform_4b") == 0) return TQ_TYPE_UNIFORM_4B;
    if (strcmp(s, "uniform_2b") == 0) return TQ_TYPE_UNIFORM_2B;
    if (strcmp(s, "polar_3b") == 0)   return TQ_TYPE_POLAR_3B;
    if (strcmp(s, "polar_4b") == 0)   return TQ_TYPE_POLAR_4B;
    if (strcmp(s, "turbo_3b") == 0)   return TQ_TYPE_TURBO_3B;
    if (strcmp(s, "turbo_4b") == 0)   return TQ_TYPE_TURBO_4B;
    if (strcmp(s, "turbo_kv_3b") == 0) return TQ_TYPE_TURBO_KV_3B;
    if (strcmp(s, "turbo_kv_4b") == 0) return TQ_TYPE_TURBO_KV_4B;
    if (strcmp(s, "turbo_kv_5b") == 0) return TQ_TYPE_TURBO_KV_5B;
    if (strcmp(s, "turbo_kv_4bo") == 0) return TQ_TYPE_TURBO_KV_4BO;
    if (strcmp(s, "turbo_kv_3bo") == 0) return TQ_TYPE_TURBO_KV_3BO;
    if (strcmp(s, "turbo_kv_5b_fast") == 0) return TQ_TYPE_TURBO_KV_5B_FAST;
    if (strcmp(s, "turbo_kv_1b") == 0) return TQ_TYPE_TURBO_KV_1B;
    if (strcmp(s, "qjl_1b") == 0)     return TQ_TYPE_QJL_1B;
    if (strcmp(s, "mixed_4b8") == 0)  return TQ_TYPE_MIXED_4B8;
    if (strcmp(s, "uniform_3b") == 0) return TQ_TYPE_UNIFORM_3B;
    fprintf(stderr, "Unknown KV type: %s (using turbo_kv_4b)\n", s);
    return TQ_TYPE_TURBO_KV_4B;
}

#define QUANT_VERSION "0.2.0"

static void print_version(void) {
    printf("quant.cpp v%s\n", QUANT_VERSION);
    printf("Embeddable LLM inference in pure C\n");
    printf("https://github.com/quantumaikr/quant.cpp\n");
}

static void print_usage(const char* prog) {
    fprintf(stderr, "quant — Minimal C inference engine. Zero dependencies.\n");
    fprintf(stderr, "Usage: %s <model.gguf> [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <tokenizer>   Tokenizer binary file\n");
    fprintf(stderr, "  -p <prompt>      Input prompt (default: \"Hello\")\n");
    fprintf(stderr, "  -n <max_tokens>  Max tokens to generate (default: 256)\n");
    fprintf(stderr, "  -T <temperature> Sampling temperature (default: 0.7)\n");
    fprintf(stderr, "  -P <top_p>       Top-p sampling (default: 0.9)\n");
    fprintf(stderr, "  -k <kv_type>     KV cache quantization type\n");
    fprintf(stderr, "  -v <vq>          Value cache quant: q4 (4-bit), q2 (2-bit), fp16 (default)\n");
    fprintf(stderr, "  -j <threads>     Number of threads for matmul (default: 4)\n");
    fprintf(stderr, "  -s <seed>        Random seed (default: 42)\n");
    fprintf(stderr, "  -q <type>        Quantize weights: q2 (2-bit Lloyd-Max, ~12x reduction),\n");
    fprintf(stderr, "                   q4 (4-bit, ~6x reduction, default),\n");
    fprintf(stderr, "                   q8 (int8, ~3.5x reduction), or none (FP32)\n");
    fprintf(stderr, "  -c, --chat       Auto-wrap prompt with model chat template\n");
    fprintf(stderr, "  --info           Print model info and exit\n");
    fprintf(stderr, "  -M, --memory     Print KV cache memory stats after generation\n");
    fprintf(stderr, "  --profile        Profile forward pass timing (matmul/recurrent/moe/conv/attn)\n");
    fprintf(stderr, "  --profile-kv     Profile KV activation distributions (pre/post RHT)\n");
    fprintf(stderr, "  --recommend      Per-layer bit allocation recommendation (kurtosis-based)\n");
    fprintf(stderr, "  --attn-entropy   Compute per-layer, per-head attention entropy\n");
    fprintf(stderr, "  -V <N>           V highres window: recent N tokens get FP16 V (0=disabled)\n");
    fprintf(stderr, "  --calibrate      Online Lloyd-Max codebook calibration analysis\n");
    fprintf(stderr, "  --ppl <file>     Compute perplexity on text file (teacher-forced)\n");
    fprintf(stderr, "  --bench-memory   Benchmark memory bandwidth at varying context lengths\n");
    fprintf(stderr, "  --bench-prefill  Benchmark prefill speed with/without KV quantization\n");
    fprintf(stderr, "  --ctx <N>        Override max context length (default: 4096)\n");
    fprintf(stderr, "  --delta, -D      Enable delta KV compression (store key deltas)\n");
    fprintf(stderr, "  --k-window <N>   Age-based K: recent N tokens FP32, rest quantized\n");
    fprintf(stderr, "  --save-kv <file> Save KV cache after generation (read once, query forever)\n");
    fprintf(stderr, "  --load-kv <file> Load pre-computed KV cache (skip prefill)\n");
    fprintf(stderr, "  --version        Print version and exit\n");
    fprintf(stderr, "  --json           JSON output for --ppl (machine-parseable)\n");
    fprintf(stderr, "  --save-logits <f> Save per-token softmax (fp16) to file during --ppl\n");
    fprintf(stderr, "  --kl-baseline <f> Read baseline softmax from file and report KL divergence\n");
}

/* ---------- fp16 helpers (local) for KL save/load ---------- */
static uint16_t qtool_fp32_to_fp16(float v) {
    union { float f; uint32_t u; } b; b.f = v;
    uint32_t sign = (b.u >> 16) & 0x8000;
    int32_t  exp  = ((b.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (b.u >> 13) & 0x03FF;
    if (exp <= 0)  return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}
static float qtool_fp16_to_fp32(uint16_t h) {
    union { float f; uint32_t u; } b;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0)  { b.u = sign; return b.f; }
    if (exp == 31) { b.u = sign | 0x7F800000 | (mant << 13); return b.f; }
    exp = exp - 15 + 127;
    b.u = sign | (exp << 23) | (mant << 13);
    return b.f;
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
    tq_type kv_type = TQ_TYPE_TURBO_KV_4B;
    /* Default: P-core count on macOS, total core count elsewhere.
     * On Apple Silicon, mixing P+E cores at the same priority makes
     * the slow E threads become stragglers — total throughput drops.
     * Tests on M1 Pro: 8P-only (8 threads) ≈ 8P+2E (10 threads). */
    int n_threads;
#if defined(__APPLE__)
    {
        size_t sz = sizeof(int);
        if (sysctlbyname("hw.perflevel0.physicalcpu", &n_threads, &sz, NULL, 0) != 0)
            n_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    }
#else
    n_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    if (n_threads < 1) n_threads = 4;
    if (n_threads > 16) n_threads = 16;  /* matches TQ_TP_MAX */
    int quant_mode = 0;   /* 0 = none (default), 2 = Q2, 4 = Q4, 8 = Q8 */
    int value_quant_bits = 0; /* 0 = FP16/FP32 (default), 4 = Q4, 2 = Q2 */
    int info_only = 0;
    int show_memory = 0;
    int profile_kv = 0;
    int recommend_mode = 0;
    int attn_entropy_mode = 0;
    int v_highres_window = 0;
    int calibrate_mode = 0;
    const char* ppl_file = NULL;
    int bench_memory = 0;
    int bench_prefill = 0;
    int override_ctx = 0;  /* 0 = use model default (capped at 4096) */
    int delta_kv = 0;      /* 1 = delta KV compression (store key deltas) */
    int delta_iframe_int = 0; /* I-frame interval for delta KV (0 = auto = 64) */
    int k_highres_window = -1; /* -1=auto (128 when KV compressed), 0=off, N=explicit */
    int json_output = 0;     /* 1 = JSON output for --ppl */
    int chat_mode = 0;       /* 1 = auto-wrap prompt with chat template */
    const char* save_logits_file = NULL;
    const char* kl_baseline_file = NULL;
    const char* save_kv_file = NULL;   /* --save-kv: save KV cache after generation */
    const char* load_kv_file = NULL;   /* --load-kv: load pre-computed KV cache */
    unsigned long long rng_seed = 42ULL; /* -s <seed>: sampling RNG seed */

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
        } else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) {
            const char* varg = argv[++i];
            if (strcmp(varg, "q4") == 0 || strcmp(varg, "4") == 0) {
                value_quant_bits = 4;
            } else if (strcmp(varg, "q2") == 0 || strcmp(varg, "2") == 0) {
                value_quant_bits = 2;
            } else if (strcmp(varg, "fp16") == 0 || strcmp(varg, "none") == 0) {
                value_quant_bits = 0;
            } else {
                fprintf(stderr, "Unknown value quant type: %s (using fp16)\n", varg);
                value_quant_bits = 0;
            }
        } else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            rng_seed = strtoull(argv[++i], NULL, 10);
            if (rng_seed == 0ULL) rng_seed = 42ULL; /* 0 reserved for "use default" */
        } else if (strcmp(argv[i], "-q") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                const char* qarg = argv[++i];
                if (strcmp(qarg, "q1") == 0 || strcmp(qarg, "1") == 0 || strcmp(qarg, "1bit") == 0) {
                    quant_mode = 1;
                } else if (strcmp(qarg, "q2") == 0 || strcmp(qarg, "2") == 0) {
                    quant_mode = 2;
                } else if (strcmp(qarg, "q4") == 0 || strcmp(qarg, "4") == 0) {
                    quant_mode = 4;
                } else if (strcmp(qarg, "q6") == 0 || strcmp(qarg, "6") == 0 || strcmp(qarg, "q4q2") == 0) {
                    quant_mode = 6;
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
        } else if (strcmp(argv[i], "--profile") == 0) {
            g_tq_profile_enabled = 1;
        } else if (strcmp(argv[i], "--profile-kv") == 0) {
            profile_kv = 1;
        } else if (strcmp(argv[i], "--recommend") == 0) {
            recommend_mode = 1;
            profile_kv = 1;  /* --recommend implies --profile-kv */
        } else if (strcmp(argv[i], "--attn-entropy") == 0) {
            attn_entropy_mode = 1;
        } else if (strcmp(argv[i], "-V") == 0 && i + 1 < argc) {
            v_highres_window = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--calibrate") == 0) {
            calibrate_mode = 1;
            profile_kv = 1;  /* --calibrate implies --profile-kv */
        } else if (strcmp(argv[i], "--ppl") == 0 && i + 1 < argc) {
            ppl_file = argv[++i];
        } else if (strcmp(argv[i], "--bench-memory") == 0) {
            bench_memory = 1;
        } else if (strcmp(argv[i], "--bench-prefill") == 0) {
            bench_prefill = 1;
        } else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            override_ctx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--delta") == 0 || strcmp(argv[i], "-D") == 0) {
            delta_kv = 1;
        } else if (strcmp(argv[i], "--iframe") == 0 && i + 1 < argc) {
            delta_iframe_int = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--k-window") == 0 && i + 1 < argc) {
            k_highres_window = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--version") == 0) {
            print_version();
            return 0;
        } else if (strcmp(argv[i], "--save-kv") == 0 && i + 1 < argc) {
            save_kv_file = argv[++i];
        } else if (strcmp(argv[i], "--load-kv") == 0 && i + 1 < argc) {
            load_kv_file = argv[++i];
        } else if (strcmp(argv[i], "--save-logits") == 0 && i + 1 < argc) {
            save_logits_file = argv[++i];
        } else if (strcmp(argv[i], "--kl-baseline") == 0 && i + 1 < argc) {
            kl_baseline_file = argv[++i];
        } else if (strcmp(argv[i], "--json") == 0) {
            json_output = 1;
        } else if (strcmp(argv[i], "--chat") == 0 || strcmp(argv[i], "-c") == 0) {
            chat_mode = 1;
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

    /* Override context length if requested */
    tq_model_config_t* c = &model->config;
    if (override_ctx > 0) {
        fprintf(stderr, "Context length override: %d → %d\n", c->max_seq_len, override_ctx);
        c->max_seq_len = override_ctx;
    }

    /* Print model info */
    fprintf(stderr, "Model: %d layers, dim=%d, heads=%d/%d, head_dim=%d, vocab=%d, inter=%d\n",
            c->n_layers, c->hidden_dim, c->n_heads, c->n_kv_heads,
            c->head_dim, c->vocab_size, c->intermediate_dim);
    fprintf(stderr, "KV cache type: %s, V quant: %s\n",
            kv_type < TQ_TYPE_COUNT ? tq_type_name(kv_type) : "fp32",
            value_quant_bits == 4 ? "Q4" : (value_quant_bits == 2 ? "Q2" : "FP16"));

    if (quant_mode == 1) {
        fprintf(stderr, "Quantizing weights to 1-bit (quant.cpp sign hash)...\n");
        extern void tq_quantize_weights_1bit(tq_model_t*);
        tq_quantize_weights_1bit(model);
    }
    if (quant_mode == 6) {
        fprintf(stderr, "Quantizing weights to Q4+Q2 (quant.cpp Progressive Residual, 6-bit)...\n");
        extern void tq_quantize_weights_q4q2(tq_model_t*);
        tq_quantize_weights_q4q2(model);
    } else
    if (quant_mode == 2) {
        fprintf(stderr, "Quantizing weights to Q2 (2-bit Lloyd-Max codebook)...\n");
        tq_quantize_weights_q2(model);
    } else if (quant_mode == 4) {
        fprintf(stderr, "Quantizing weights to Q4 (4-bit)...\n");
        tq_quantize_weights_q4(model);
    } else if (quant_mode == 8) {
        fprintf(stderr, "Quantizing weights to Q8 (int8)...\n");
        tq_quantize_weights(model);
    }

    /* GPU backend detection and initialization */
#ifdef TQ_BUILD_VULKAN
    {
        extern int tq_init_vulkan_backend(void);
        if (tq_init_vulkan_backend() == 0) {
            fprintf(stderr, "Vulkan backend: ready (KV cache quantization on GPU)\n");
        }
    }
#endif
#ifdef TQ_BUILD_CUDA
    {
        extern int  tq_init_cuda_backend(void);
        extern void tq_cuda_override_traits(void);
        if (tq_init_cuda_backend() == 0) {
            tq_cuda_override_traits();
            fprintf(stderr, "CUDA backend: ready (KV cache quantization on GPU)\n");
        } else {
            fprintf(stderr, "CUDA backend: init failed, falling back to CPU\n");
        }
    }
#endif

    if (info_only) {
        tq_free_model(model);
        return 0;
    }

    /* ================================================================
     * Mode: --ppl  (Perplexity evaluation)
     * Teacher-forced: for each token position, compute cross-entropy
     * loss against the ground truth next token.
     * ================================================================ */
    if (ppl_file) {
        /* Load tokenizer first */
        tq_tokenizer_t* tok = NULL;
        if (tokenizer_path) {
            tok = tq_load_tokenizer(tokenizer_path);
        } else {
            tok = tq_load_tokenizer_from_tqm(model_path);
        }
        if (!tok && model->gguf_ctx) {
            tok = tq_load_tokenizer_from_gguf(model->gguf_ctx);
        }
        if (!tok) {
            fprintf(stderr, "Error: --ppl requires a tokenizer\n");
            tq_free_model(model);
            return 1;
        }

        /* Read text file */
        FILE* fp = fopen(ppl_file, "r");
        if (!fp) {
            fprintf(stderr, "Error: cannot open %s\n", ppl_file);
            tq_free_tokenizer(tok);
            tq_free_model(model);
            return 1;
        }
        fseek(fp, 0, SEEK_END);
        long fsize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        char* text = (char*)malloc((size_t)fsize + 1);
        if (!text) {
            fclose(fp);
            tq_free_tokenizer(tok);
            tq_free_model(model);
            return 1;
        }
        size_t nread = fread(text, 1, (size_t)fsize, fp);
        text[nread] = '\0';
        fclose(fp);

        /* Tokenize. BPE merge now uses O(n log n) heap-based algorithm,
         * so we can allocate a buffer large enough for the full text. */
        int max_tok = (int)(nread + 256);
        int* tokens = (int*)malloc((size_t)max_tok * sizeof(int));
        if (!tokens) {
            free(text);
            tq_free_tokenizer(tok);
            tq_free_model(model);
            return 1;
        }
        int n_tokens = tq_encode(tok, text, tokens, max_tok, 1);
        free(text);
        /* Truncate to model's context window for eval */
        if (n_tokens > c->max_seq_len) n_tokens = c->max_seq_len;
        fprintf(stderr, "PPL evaluation: %d tokens from %s\n", n_tokens, ppl_file);

        if (n_tokens < 2) {
            fprintf(stderr, "Error: need at least 2 tokens for perplexity\n");
            free(tokens);
            tq_free_tokenizer(tok);
            tq_free_model(model);
            return 1;
        }

        /* Apply weight quantization */
        if (quant_mode == 2) tq_quantize_weights_q2(model);
        else if (quant_mode == 4) tq_quantize_weights_q4(model);
        else if (quant_mode == 8) tq_quantize_weights(model);

        tq_set_threads(n_threads);

        /* Create state */
        tq_state_t* state = tq_create_state_ex(&model->config, kv_type, value_quant_bits);
        if (!state) {
            fprintf(stderr, "Error: failed to allocate state\n");
            free(tokens);
            tq_free_tokenizer(tok);
            tq_free_model(model);
            return 1;
        }
        state->delta_kv_enabled = delta_kv;
        state->delta_iframe_interval = delta_iframe_int;
        /* Hybrid DeltaNet models: delta KV applies only to self_attn layers.
         * DeltaNet layers don't use key_cache, so delta compression is safe. */
        if (state->delta_kv_enabled) {
            int ifi = delta_iframe_int > 0 ? delta_iframe_int : 64;
            fprintf(stderr, "Delta KV compression: ENABLED (mixed-precision, I-frame=%d)\n", ifi);
        }

        /* Progressive KV: auto-enable k128 when KV is compressed.
         * Verified on 3 models: strictly better quality at 1.75 MB cost.
         * User can override with --k-window 0 to disable. */
        if (k_highres_window == -1 && state->kv_quant_type < TQ_TYPE_COUNT && state->quant_key_cache) {
            k_highres_window = 128;
        } else if (k_highres_window == -1) {
            k_highres_window = 0;
        }
        if (k_highres_window > 0 && state->kv_quant_type < TQ_TYPE_COUNT && state->quant_key_cache) {
            int kv_dim_e = model->config.n_kv_heads * model->config.head_dim;
            int cache_kv_dim_e = model->config.n_kv_heads * model->config.head_dim;
            state->k_highres_window = k_highres_window;
            state->key_highres_fp32 = (float*)calloc(
                (size_t)model->config.n_layers * k_highres_window * cache_kv_dim_e, sizeof(float));
            fprintf(stderr, "K highres window: %d tokens at FP32 (age-based progressive)\n", k_highres_window);
        }

        /* KL/save-logits setup. Format: int32 n_tokens, int32 vocab, then
         * n_tokens × vocab × fp16 softmax probabilities. */
        FILE* save_fp = NULL;
        FILE* kl_fp   = NULL;
        double total_kl = 0.0;
        long n_kl = 0;
        uint16_t* kl_buf = NULL;
        if (save_logits_file) {
            save_fp = fopen(save_logits_file, "wb");
            if (!save_fp) { fprintf(stderr, "Error: cannot open --save-logits %s\n", save_logits_file); return 1; }
            int32_t hdr[2] = { n_tokens - 1, c->vocab_size };
            fwrite(hdr, sizeof(int32_t), 2, save_fp);
        }
        if (kl_baseline_file) {
            kl_fp = fopen(kl_baseline_file, "rb");
            if (!kl_fp) { fprintf(stderr, "Error: cannot open --kl-baseline %s\n", kl_baseline_file); return 1; }
            int32_t hdr[2] = {0,0};
            if (fread(hdr, sizeof(int32_t), 2, kl_fp) != 2 || hdr[1] != c->vocab_size) {
                fprintf(stderr, "Error: KL baseline header mismatch (expected vocab=%d)\n", c->vocab_size);
                return 1;
            }
            fprintf(stderr, "KL baseline: %d tokens × vocab %d\n", hdr[0], hdr[1]);
        }
        if (save_fp || kl_fp) {
            kl_buf = (uint16_t*)malloc((size_t)c->vocab_size * sizeof(uint16_t));
            if (!kl_buf) { fprintf(stderr, "Error: oom\n"); return 1; }
        }

        /* Teacher-forced forward: accumulate negative log-likelihood */
        double total_nll = 0.0;
        int n_eval = 0;

        struct timespec ppl_start, ppl_end;
        clock_gettime(CLOCK_MONOTONIC, &ppl_start);

        for (int i = 0; i < n_tokens - 1; i++) {
            float* logits = tq_forward(model, state, tokens[i], i);

            /* Compute log_softmax(logits)[tokens[i+1]] */
            int target = tokens[i + 1];
            if (target < 0 || target >= c->vocab_size) continue;

            /* Find max for numerical stability */
            float max_logit = logits[0];
            for (int j = 1; j < c->vocab_size; j++) {
                if (logits[j] > max_logit) max_logit = logits[j];
            }

            /* log(sum(exp(logits - max))) */
            double log_sum = 0.0;
            for (int j = 0; j < c->vocab_size; j++) {
                log_sum += exp((double)(logits[j] - max_logit));
            }
            log_sum = log(log_sum);

            /* log_softmax[target] = (logits[target] - max) - log_sum */
            double log_prob = (double)(logits[target] - max_logit) - log_sum;
            total_nll -= log_prob;
            n_eval++;

            /* Optional: compute full softmax for save / KL divergence. */
            if (save_fp || kl_fp) {
                /* p[j] = exp(logits[j] - max_logit) / exp(log_sum)
                 *      = exp((logits[j] - max_logit) - log_sum)            */
                double cur_kl = 0.0;
                for (int j = 0; j < c->vocab_size; j++) {
                    float p = (float)exp((double)(logits[j] - max_logit) - log_sum);
                    if (save_fp) kl_buf[j] = qtool_fp32_to_fp16(p);
                    if (kl_fp) {
                        /* Fold into KL accumulation: read baseline below. */
                        kl_buf[j] = qtool_fp32_to_fp16(p); /* reuse buf for current */
                    }
                    (void)cur_kl;
                }
                if (save_fp) {
                    fwrite(kl_buf, sizeof(uint16_t), (size_t)c->vocab_size, save_fp);
                }
                if (kl_fp) {
                    /* Read baseline row, compute KL(baseline || current). */
                    static uint16_t* base_buf = NULL;
                    static int base_buf_v = 0;
                    if (base_buf_v != c->vocab_size) {
                        free(base_buf);
                        base_buf = (uint16_t*)malloc((size_t)c->vocab_size * sizeof(uint16_t));
                        base_buf_v = c->vocab_size;
                    }
                    if (fread(base_buf, sizeof(uint16_t), (size_t)c->vocab_size, kl_fp)
                            == (size_t)c->vocab_size) {
                        double kl = 0.0;
                        for (int j = 0; j < c->vocab_size; j++) {
                            float pb = qtool_fp16_to_fp32(base_buf[j]);
                            float pc = qtool_fp16_to_fp32(kl_buf[j]);
                            if (pb > 1e-12f) {
                                if (pc < 1e-20f) pc = 1e-20f;
                                kl += (double)pb * (log((double)pb) - log((double)pc));
                            }
                        }
                        total_kl += kl;
                        n_kl++;
                    }
                }
            }

            if ((i + 1) % 50 == 0) {
                double ppl_so_far = exp(total_nll / (double)n_eval);
                fprintf(stderr, "  [%d/%d] PPL so far: %.4f\n", i + 1, n_tokens - 1, ppl_so_far);
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &ppl_end);
        double ppl_elapsed = (double)(ppl_end.tv_sec - ppl_start.tv_sec)
                           + (double)(ppl_end.tv_nsec - ppl_start.tv_nsec) / 1e9;

        double perplexity = exp(total_nll / (double)n_eval);
        double avg_nll = total_nll / (double)n_eval;

        fprintf(stderr, "\n=== Perplexity Results ===\n");
        fprintf(stderr, "File:         %s\n", ppl_file);
        fprintf(stderr, "Tokens:       %d (evaluated %d)\n", n_tokens, n_eval);
        fprintf(stderr, "KV type:      %s\n", kv_type < TQ_TYPE_COUNT ? tq_type_name(kv_type) : "fp32");
        fprintf(stderr, "Delta KV:     %s%s\n", delta_kv ? "ON (mixed-precision)" : "OFF",
                delta_kv && delta_iframe_int > 0 ? "" : "");
        if (delta_kv) {
            int ifi = delta_iframe_int > 0 ? delta_iframe_int : 64;
            fprintf(stderr, "I-frame int:  %d (FP32 I-frames, %d-bit P-frames)\n",
                    ifi, kv_type == TQ_TYPE_UNIFORM_2B ? 2 : 4);
        }
        fprintf(stderr, "V quant:      %s\n", value_quant_bits == 4 ? "Q4" : (value_quant_bits == 2 ? "Q2" : "FP16"));
        fprintf(stderr, "Avg NLL:      %.6f\n", avg_nll);
        fprintf(stderr, "Perplexity:   %.4f\n", perplexity);
        fprintf(stderr, "Time:         %.1fs (%.1f tok/s)\n", ppl_elapsed,
                (double)n_eval / ppl_elapsed);
        fprintf(stderr, "==========================\n");

        if (kl_fp && n_kl > 0) {
            double mean_kl = total_kl / (double)n_kl;
            fprintf(stderr, "KL divergence (baseline || quantized): mean = %.6f over %ld tokens\n",
                    mean_kl, n_kl);
        }
        if (save_fp) { fclose(save_fp); fprintf(stderr, "Saved logits to %s\n", save_logits_file); }
        if (kl_fp)   { fclose(kl_fp); }
        free(kl_buf);

        /* Machine-parseable */
        fprintf(stderr, "PPL_CSV:%d,%.6f,%.4f\n", n_eval, avg_nll, perplexity);

#ifdef TQ_HAS_METAL
        {
            extern void tq_metal_diag_get(unsigned long*, unsigned long*);
            unsigned long n_flushes = 0, n_ops = 0;
            tq_metal_diag_get(&n_flushes, &n_ops);
            if (n_flushes > 0 && n_eval > 0) {
                fprintf(stderr, "Metal diag: %lu flushes, %lu ops total, "
                                "%.1f flushes/token, %.1f ops/flush\n",
                        n_flushes, n_ops,
                        (double)n_flushes / (double)n_eval,
                        (double)n_ops / (double)n_flushes);
            }
        }
#endif

        /* JSON output (--json flag) */
        if (json_output) {
            const char* kv_name = kv_type < TQ_TYPE_COUNT ? tq_type_name(kv_type) : "fp32";
            const char* v_name = value_quant_bits == 4 ? "q4" : (value_quant_bits == 2 ? "q2" : "fp16");
            printf("{\n");
            printf("  \"model\": \"%s\",\n", model_path);
            printf("  \"benchmark\": \"%s\",\n", ppl_file);
            printf("  \"tokens\": %d,\n", n_tokens);
            printf("  \"tokens_evaluated\": %d,\n", n_eval);
            printf("  \"kv_type\": \"%s\",\n", kv_name);
            printf("  \"v_quant\": \"%s\",\n", v_name);
            printf("  \"delta_kv\": %s,\n", delta_kv ? "true" : "false");
            printf("  \"perplexity\": %.4f,\n", perplexity);
            printf("  \"avg_nll\": %.6f,\n", avg_nll);
            printf("  \"elapsed_s\": %.2f,\n", ppl_elapsed);
            printf("  \"tok_per_s\": %.1f\n", (double)n_eval / ppl_elapsed);
            printf("}\n");
        }

        tq_free_state(state);
        free(tokens);
        tq_free_tokenizer(tok);
        tq_free_model(model);
        return 0;
    }

    /* ================================================================
     * Mode: --bench-memory  (Memory bandwidth benchmark)
     * Runs inference at varying context lengths and measures tok/s.
     * ================================================================ */
    if (bench_memory) {
        /* Load tokenizer */
        tq_tokenizer_t* tok = NULL;
        if (tokenizer_path) {
            tok = tq_load_tokenizer(tokenizer_path);
        } else {
            tok = tq_load_tokenizer_from_tqm(model_path);
        }

        /* Apply weight quantization */
        if (quant_mode == 2) tq_quantize_weights_q2(model);
        else if (quant_mode == 4) tq_quantize_weights_q4(model);
        else if (quant_mode == 8) tq_quantize_weights(model);

        tq_set_threads(n_threads);

        /* Fixed prompt token for prefill */
        int bos_token = (c->model_type == 1) ? 2 : 1;

        /* Context lengths to test */
        int ctx_lengths[] = {10, 50, 100, 200, 500};
        int n_ctx = 5;

        fprintf(stderr, "\n=== Memory Bandwidth Benchmark ===\n");
        fprintf(stderr, "KV type: %s, V quant: %s\n",
                kv_type < TQ_TYPE_COUNT ? tq_type_name(kv_type) : "fp32",
                value_quant_bits == 4 ? "Q4" : (value_quant_bits == 2 ? "Q2" : "FP16"));
        fprintf(stderr, "%-12s %-12s %-12s\n", "Context", "Tok/s", "Time(s)");
        fprintf(stderr, "-------- -------- --------\n");

        for (int ci = 0; ci < n_ctx; ci++) {
            int ctx = ctx_lengths[ci];
            if (ctx >= c->max_seq_len) continue;

            tq_state_t* st = tq_create_state_ex(&model->config, kv_type, value_quant_bits);
            if (!st) continue;

            /* Prefill context with BOS tokens */
            for (int i = 0; i < ctx; i++) {
                tq_forward(model, st, bos_token, i);
            }

            /* Measure decode speed: generate 20 tokens */
            int gen_count = 20;
            if (ctx + gen_count >= c->max_seq_len) {
                gen_count = c->max_seq_len - ctx - 1;
            }
            if (gen_count < 1) { tq_free_state(st); continue; }

            struct timespec bm_start, bm_end;
            clock_gettime(CLOCK_MONOTONIC, &bm_start);

            for (int g = 0; g < gen_count; g++) {
                tq_forward(model, st, bos_token, ctx + g);
            }

            clock_gettime(CLOCK_MONOTONIC, &bm_end);
            double bm_elapsed = (double)(bm_end.tv_sec - bm_start.tv_sec)
                              + (double)(bm_end.tv_nsec - bm_start.tv_nsec) / 1e9;
            double tok_s = (double)gen_count / bm_elapsed;

            fprintf(stderr, "%-12d %-12.1f %-12.3f\n", ctx, tok_s, bm_elapsed);
            fprintf(stderr, "BENCH_CSV:%d,%.2f,%.4f\n", ctx, tok_s, bm_elapsed);

            tq_free_state(st);
        }
        fprintf(stderr, "==================================\n");

        if (tok) tq_free_tokenizer(tok);
        tq_free_model(model);
        return 0;
    }

    /* ================================================================
     * Mode: --bench-prefill  (Prefill speed benchmark)
     * Measures prefill tok/s for different KV quantization types
     * to demonstrate that KV quantization overhead is minimal.
     * ================================================================ */
    if (bench_prefill) {
        tq_set_threads(n_threads);

        int prefill_len = 200;
        if (prefill_len > c->max_seq_len - 1) prefill_len = c->max_seq_len - 1;
        int bos_token = (c->model_type == 1) ? 2 : 1;

        /* Configurations to benchmark */
        typedef struct { const char* name; tq_type kv; int vq; } prefill_config_t;
        prefill_config_t configs[] = {
            {"fp32 (no quant)", TQ_TYPE_COUNT, 0},
            {"uniform_4b + FP16 V", TQ_TYPE_UNIFORM_4B, 0},
            {"uniform_4b + Q4 V", TQ_TYPE_UNIFORM_4B, 4},
            {"turbo_kv_1b + FP16 V", TQ_TYPE_TURBO_KV_1B, 0},
            {"turbo_kv_3b + Q4 V", TQ_TYPE_TURBO_KV_3B, 4},
            {"turbo_kv_3b + Q2 V", TQ_TYPE_TURBO_KV_3B, 2},
        };
        int n_configs = (int)(sizeof(configs) / sizeof(configs[0]));

        fprintf(stderr, "\n=== Prefill Speed Benchmark ===\n");
        fprintf(stderr, "Prefill length: %d tokens, Threads: %d\n", prefill_len, n_threads);
        fprintf(stderr, "%-28s %-12s %-12s\n", "KV Config", "Tok/s", "Time(s)");
        fprintf(stderr, "---------------------------- ------------ ------------\n");

        for (int ci = 0; ci < n_configs; ci++) {
            tq_state_t* st = tq_create_state_ex(&model->config, configs[ci].kv, configs[ci].vq);
            if (!st) {
                fprintf(stderr, "%-28s (failed to allocate state)\n", configs[ci].name);
                continue;
            }

            struct timespec pf_start, pf_end;
            clock_gettime(CLOCK_MONOTONIC, &pf_start);

            for (int i = 0; i < prefill_len; i++) {
                tq_forward(model, st, bos_token, i);
            }

            clock_gettime(CLOCK_MONOTONIC, &pf_end);
            double pf_elapsed = (double)(pf_end.tv_sec - pf_start.tv_sec)
                              + (double)(pf_end.tv_nsec - pf_start.tv_nsec) / 1e9;
            double tok_s = (double)prefill_len / pf_elapsed;

            fprintf(stderr, "%-28s %-12.1f %-12.3f\n", configs[ci].name, tok_s, pf_elapsed);
            fprintf(stderr, "PREFILL_CSV:%s,%d,%.2f,%.4f\n", configs[ci].name, prefill_len, tok_s, pf_elapsed);

            tq_free_state(st);
        }

        fprintf(stderr, "===============================\n");
        fprintf(stderr, "Note: prefill speed is dominated by weight matmuls, not KV quantization.\n");
        fprintf(stderr, "KV quantization overhead should be <5%% of total prefill time.\n");

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
        /* Try GGUF tokenizer if model was loaded from GGUF */
        if (!tokenizer && model->gguf_ctx) {
            tokenizer = tq_load_tokenizer_from_gguf(model->gguf_ctx);
            if (tokenizer) {
                fprintf(stderr, "Loaded tokenizer from GGUF metadata\n");
            }
        }
    }

    /* Set thread count for matmul parallelism */
    tq_set_threads(n_threads);
    fprintf(stderr, "Threads: %d\n", tq_get_threads());

    /* ================================================================
     * Mode: --profile-kv  (KV activation distribution profiling)
     * Runs forward on prompt tokens, collects pre/post-RHT stats per layer.
     * ================================================================ */
    if (profile_kv) {
        tq_state_t* state = tq_create_state_ex(&model->config, kv_type, value_quant_bits);
        if (!state) {
            fprintf(stderr, "Error: failed to allocate state\n");
            if (tokenizer) tq_free_tokenizer(tokenizer);
            tq_free_model(model);
            return 1;
        }

        /* Enable profiling */
        state->profile_kv = 1;
        state->profile_kv_count = 0;
        state->profile_accum = (double*)calloc((size_t)c->n_layers * 8, sizeof(double));
        state->profile_stats = (float*)calloc((size_t)c->n_layers * 8, sizeof(float));
        if (!state->profile_accum || !state->profile_stats) {
            fprintf(stderr, "Error: failed to allocate profile buffers\n");
            tq_free_state(state);
            if (tokenizer) tq_free_tokenizer(tokenizer);
            tq_free_model(model);
            return 1;
        }

        /* Encode prompt */
        int ptokens[4096];
        int n_prompt = 0;
        if (tokenizer) {
            n_prompt = tq_encode(tokenizer, prompt, ptokens, 4096, 1);
        }
        if (n_prompt <= 0) {
            ptokens[0] = (c->model_type == 1) ? 2 : 1;
            n_prompt = 1;
        }

        /* Run forward on all prompt + generated tokens */
        int total_run = n_prompt + max_tokens;
        if (total_run > c->max_seq_len) total_run = c->max_seq_len;

        fprintf(stderr, "Profiling KV for %d tokens...\n", total_run);
        for (int i = 0; i < total_run; i++) {
            int tok = (i < n_prompt) ? ptokens[i] : 1; /* use token 1 for generated positions */
            float* logits = tq_forward(model, state, tok, i);
            if (i >= n_prompt && logits) {
                /* Use argmax for the next token */
                int next = tq_sample_argmax(logits, c->vocab_size);
                (void)next; /* just forward, not generating text */
            }
        }

        /* Compute and print statistics */
        int n_tok = state->profile_kv_count;
        int head_dim = c->head_dim;
        double n_samples = (double)n_tok * head_dim; /* samples per layer */

        fprintf(stderr, "\n=== KV Activation Distribution Profile ===\n");
        fprintf(stderr, "Tokens profiled: %d, head_dim: %d, samples/layer: %.0f\n",
                n_tok, head_dim, n_samples);
        fprintf(stderr, "%-8s %-10s %-10s %-10s %-10s | %-10s %-10s %-10s %-10s %-10s\n",
                "Layer", "PreMean", "PreStd", "PreSkew", "PreKurt",
                "PostMean", "PostStd", "PostSkew", "PostKurt", "KL-div");
        fprintf(stderr, "-------- ---------- ---------- ---------- ---------- | ---------- ---------- ---------- ---------- ----------\n");

        for (int l = 0; l < c->n_layers; l++) {
            double* acc = state->profile_accum + (size_t)l * 8;
            if (n_samples < 1.0) continue;

            /* Pre-RHT stats */
            double pre_mean = acc[0] / n_samples;
            double pre_var  = acc[1] / n_samples - pre_mean * pre_mean;
            double pre_std  = (pre_var > 0) ? sqrt(pre_var) : 1e-10;
            double pre_skew = (pre_std > 1e-10) ?
                (acc[2] / n_samples - 3.0 * pre_mean * pre_var - pre_mean * pre_mean * pre_mean)
                / (pre_std * pre_std * pre_std) : 0.0;
            double pre_kurt = (pre_std > 1e-10) ?
                (acc[3] / n_samples - 4.0 * pre_mean * acc[2] / n_samples
                 + 6.0 * pre_mean * pre_mean * acc[1] / n_samples
                 - 3.0 * pre_mean * pre_mean * pre_mean * pre_mean)
                / (pre_var * pre_var) : 0.0;

            /* Post-RHT stats */
            double post_mean = acc[4] / n_samples;
            double post_var  = acc[5] / n_samples - post_mean * post_mean;
            double post_std  = (post_var > 0) ? sqrt(post_var) : 1e-10;
            double post_skew = (post_std > 1e-10) ?
                (acc[6] / n_samples - 3.0 * post_mean * post_var - post_mean * post_mean * post_mean)
                / (post_std * post_std * post_std) : 0.0;
            double post_kurt = (post_std > 1e-10) ?
                (acc[7] / n_samples - 4.0 * post_mean * acc[6] / n_samples
                 + 6.0 * post_mean * post_mean * acc[5] / n_samples
                 - 3.0 * post_mean * post_mean * post_mean * post_mean)
                / (post_var * post_var) : 0.0;

            /* Approximate KL-divergence vs N(0, post_std):
             * KL(p || N(0,1)) = 0.5 * (var + mean^2 - 1 - log(var))
             * where p is assumed Gaussian with measured mean/var. */
            double post_var_for_kl = post_var > 1e-20 ? post_var : 1e-20;
            /* Normalize: we want KL against N(0, sigma) where sigma is the observed std.
             * Actually KL against standard normal: KL = 0.5 * (sigma^2 + mu^2 - 1 - ln(sigma^2))
             * But we need to standardize first. The RHT output should be ~N(0, 1/sqrt(dim)). */
            double scaled_var = post_var * (double)head_dim; /* should be ~1.0 after RHT */
            double scaled_mean = post_mean * sqrt((double)head_dim);
            double kl_div = 0.5 * (scaled_var + scaled_mean * scaled_mean - 1.0
                                   - log(scaled_var > 1e-20 ? scaled_var : 1e-20));
            if (kl_div < 0.0) kl_div = 0.0; /* numerical guard */

            fprintf(stderr, "%-8d %-10.4f %-10.4f %-10.4f %-10.2f | %-10.4f %-10.4f %-10.4f %-10.2f %-10.4f\n",
                    l, pre_mean, pre_std, pre_skew, pre_kurt,
                    post_mean, post_std, post_skew, post_kurt, kl_div);
        }
        fprintf(stderr, "==========================================\n");
        fprintf(stderr, "Target: post-RHT kurtosis ~ 3.0 (normal), KL-div < 0.05\n");

        /* --recommend: per-layer bit allocation based on kurtosis */
        if (recommend_mode) {
            /* Collect post-RHT kurtosis values */
            float* kurtosis_vals = (float*)calloc((size_t)c->n_layers, sizeof(float));
            int* rec_bits = (int*)calloc((size_t)c->n_layers, sizeof(int));
            if (kurtosis_vals && rec_bits) {
                for (int l = 0; l < c->n_layers; l++) {
                    double* acc = state->profile_accum + (size_t)l * 8;
                    double pm = acc[4] / n_samples;
                    double pv = acc[5] / n_samples - pm * pm;
                    double ps = (pv > 1e-20) ? sqrt(pv) : 1e-10;
                    double pk = (ps > 1e-10) ?
                        (acc[7] / n_samples - 4.0 * pm * acc[6] / n_samples
                         + 6.0 * pm * pm * acc[5] / n_samples
                         - 3.0 * pm * pm * pm * pm)
                        / (pv * pv) : 0.0;
                    kurtosis_vals[l] = (float)pk;
                }
                float avg_bits = 0.0f;
                tq_recommend_layer_bits(kurtosis_vals, c->n_layers, rec_bits, &avg_bits);

                fprintf(stderr, "\n=== Per-Layer Bit Allocation Recommendations ===\n");
                int n_3bit = 0, n_1bit = 0;
                for (int l = 0; l < c->n_layers; l++) {
                    const char* type_name = (rec_bits[l] == 3) ? "turbo_kv_3b" : "turbo_kv_1b";
                    fprintf(stderr, "Layer %2d: kurtosis=%.2f -> recommend %s (%d-bit)\n",
                            l, kurtosis_vals[l], type_name, rec_bits[l]);
                    if (rec_bits[l] == 3) n_3bit++;
                    else n_1bit++;
                }
                fprintf(stderr, "Summary: %d layers @ 3-bit, %d layers @ 1-bit\n", n_3bit, n_1bit);
                fprintf(stderr, "Average: %.1f bits (vs 3.0 uniform) -> %.0f%% memory savings\n",
                        avg_bits, (1.0f - avg_bits / 3.0f) * 100.0f);
                fprintf(stderr, "=================================================\n");
            }
            free(kurtosis_vals);
            free(rec_bits);
        }

        /* --calibrate: Lloyd-Max codebook optimization on post-RHT data */
        if (calibrate_mode) {
            fprintf(stderr, "\n=== Online Codebook Calibration ===\n");
            fprintf(stderr, "Collecting post-RHT activations for calibration...\n");

            /* Re-run a smaller forward pass to collect actual values */
            tq_state_t* cal_state = tq_create_state_ex(&model->config, kv_type, value_quant_bits);
            if (cal_state) {
                int cal_tokens = 100;
                if (cal_tokens > c->max_seq_len) cal_tokens = c->max_seq_len;

                /* Allocate sample buffer: cal_tokens * head_dim values per layer */
                int sample_size = cal_tokens * head_dim;
                float* samples = (float*)calloc((size_t)sample_size, sizeof(float));
                if (samples) {
                    /* Run forward, collect post-RHT key values from layer 0 */
                    for (int i = 0; i < cal_tokens; i++) {
                        int tok = (i < n_prompt) ? ptokens[i] : 1;
                        tq_forward(model, cal_state, tok, i);
                        /* Copy key from layer 0, apply RHT */
                        float k_rht[TQ_BK];
                        int rd = head_dim;
                        if (rd > TQ_BK) rd = TQ_BK;
                        memcpy(k_rht, cal_state->k, (size_t)rd * sizeof(float));
                        tq_rht_transform(k_rht, rd, 0x12345678u);
                        memcpy(samples + (size_t)i * head_dim, k_rht, (size_t)rd * sizeof(float));
                    }

                    /* Normalize samples to unit variance for codebook comparison */
                    double sum = 0.0, sum_sq = 0.0;
                    for (int i = 0; i < sample_size; i++) {
                        sum += (double)samples[i];
                        sum_sq += (double)samples[i] * (double)samples[i];
                    }
                    double smean = sum / (double)sample_size;
                    double svar = sum_sq / (double)sample_size - smean * smean;
                    double sstd = (svar > 0) ? sqrt(svar) : 1.0;
                    for (int i = 0; i < sample_size; i++) {
                        samples[i] = (float)(((double)samples[i] - smean) / sstd);
                    }

                    /* Calibrate 2-bit codebook (4 levels) */
                    float centroids_4[4], boundaries_3[3];
                    float mse_cal = tq_calibrate_codebook(samples, sample_size, 4, 20,
                                                          centroids_4, boundaries_3);

                    /* Compare with default N(0,1) Lloyd-Max centroids */
                    float default_centroids[4] = {-1.510f, -0.453f, 0.453f, 1.510f};
                    double mse_default = 0.0;
                    for (int i = 0; i < sample_size; i++) {
                        float val = samples[i];
                        float best_dist = fabsf(val - default_centroids[0]);
                        for (int ci = 1; ci < 4; ci++) {
                            float dist = fabsf(val - default_centroids[ci]);
                            if (dist < best_dist) best_dist = dist;
                        }
                        mse_default += (double)(best_dist * best_dist);
                    }
                    mse_default /= (double)sample_size;

                    fprintf(stderr, "\n2-bit codebook (4 levels):\n");
                    fprintf(stderr, "  Default N(0,1):  centroids = [%.3f, %.3f, %.3f, %.3f]  MSE = %.6f\n",
                            default_centroids[0], default_centroids[1],
                            default_centroids[2], default_centroids[3], mse_default);
                    fprintf(stderr, "  Calibrated:      centroids = [%.3f, %.3f, %.3f, %.3f]  MSE = %.6f\n",
                            centroids_4[0], centroids_4[1], centroids_4[2], centroids_4[3],
                            (double)mse_cal);
                    fprintf(stderr, "  Boundaries:      [%.3f, %.3f, %.3f]\n",
                            boundaries_3[0], boundaries_3[1], boundaries_3[2]);
                    double improvement = (mse_default > 0) ?
                        (1.0 - (double)mse_cal / mse_default) * 100.0 : 0.0;
                    fprintf(stderr, "  MSE improvement: %.1f%%\n", improvement);

                    /* Calibrate 3-bit codebook (8 levels) */
                    float centroids_8[8], boundaries_7[7];
                    float mse_cal_8 = tq_calibrate_codebook(samples, sample_size, 8, 20,
                                                            centroids_8, boundaries_7);
                    fprintf(stderr, "\n3-bit codebook (8 levels):\n");
                    fprintf(stderr, "  Calibrated:      centroids = [");
                    for (int ci = 0; ci < 8; ci++) {
                        fprintf(stderr, "%.3f%s", centroids_8[ci], ci < 7 ? ", " : "");
                    }
                    fprintf(stderr, "]  MSE = %.6f\n", (double)mse_cal_8);

                    free(samples);
                }
                tq_free_state(cal_state);
            }
            fprintf(stderr, "===================================\n");
        }

        free(state->profile_accum);
        state->profile_accum = NULL;
        free(state->profile_stats);
        state->profile_stats = NULL;
        tq_free_state(state);
        if (tokenizer) tq_free_tokenizer(tokenizer);
        tq_free_model(model);
        return 0;
    }

    /* ================================================================
     * Mode: --attn-entropy  (Attention entropy analysis)
     * Runs inference and tracks per-layer, per-head attention entropy.
     * ================================================================ */
    if (attn_entropy_mode) {
        tq_state_t* state = tq_create_state_ex(&model->config, kv_type, value_quant_bits);
        if (!state) {
            fprintf(stderr, "Error: failed to allocate state\n");
            if (tokenizer) tq_free_tokenizer(tokenizer);
            tq_free_model(model);
            return 1;
        }

        /* Enable entropy tracking */
        state->attn_entropy = 1;
        state->entropy_count = 0;
        state->entropy_accum = (double*)calloc(
            (size_t)c->n_layers * c->n_heads, sizeof(double));
        if (!state->entropy_accum) {
            fprintf(stderr, "Error: failed to allocate entropy buffers\n");
            tq_free_state(state);
            if (tokenizer) tq_free_tokenizer(tokenizer);
            tq_free_model(model);
            return 1;
        }

        /* Set up V highres window if requested */
        if (v_highres_window > 0 && (value_quant_bits == 4 || value_quant_bits == 2)) {
            int kv_dim_e = c->n_kv_heads * c->head_dim;
            state->v_highres_window = v_highres_window;
            state->value_highres_fp16 = (uint16_t*)calloc(
                (size_t)c->n_layers * v_highres_window * kv_dim_e, sizeof(uint16_t));
        }

        /* Encode prompt */
        int ptokens_e[4096];
        int n_prompt_e = 0;
        if (tokenizer) {
            n_prompt_e = tq_encode(tokenizer, prompt, ptokens_e, 4096, 1);
        }
        if (n_prompt_e <= 0) {
            ptokens_e[0] = (c->model_type == 1) ? 2 : 1;
            n_prompt_e = 1;
        }

        int total_run = n_prompt_e + max_tokens;
        if (total_run > c->max_seq_len) total_run = c->max_seq_len;

        fprintf(stderr, "Running attention entropy analysis for %d tokens...\n", total_run);
        for (int i = 0; i < total_run; i++) {
            int tok = (i < n_prompt_e) ? ptokens_e[i] : 1;
            float* logits_e = tq_forward(model, state, tok, i);
            if (i >= 1) {
                state->entropy_count++;
            }
            if (i >= n_prompt_e && logits_e) {
                int next = tq_sample_argmax(logits_e, c->vocab_size);
                (void)next;
            }
        }

        /* Print entropy report */
        int en_count = state->entropy_count;
        fprintf(stderr, "\n=== Attention Entropy Analysis ===\n");
        fprintf(stderr, "Tokens analyzed: %d\n", en_count);
        fprintf(stderr, "Entropy unit: bits (log2). Low = sharp, High = diffuse.\n\n");

        /* Per-layer summary */
        fprintf(stderr, "%-8s %-12s %-12s %-12s\n",
                "Layer", "AvgEntropy", "MinHead", "MaxHead");
        fprintf(stderr, "-------- ------------ ------------ ------------\n");

        for (int l = 0; l < c->n_layers; l++) {
            double layer_sum = 0.0;
            double min_h = 1e30, max_h = -1e30;
            for (int h = 0; h < c->n_heads; h++) {
                double avg_e = (en_count > 0) ?
                    state->entropy_accum[(size_t)l * c->n_heads + h] / (double)en_count : 0.0;
                layer_sum += avg_e;
                if (avg_e < min_h) min_h = avg_e;
                if (avg_e > max_h) max_h = avg_e;
            }
            double layer_avg = layer_sum / (double)c->n_heads;
            fprintf(stderr, "%-8d %-12.4f %-12.4f %-12.4f\n",
                    l, layer_avg, min_h, max_h);
        }

        /* Sliding window utilization analysis */
        if (c->sliding_window > 0) {
            /* For sliding window models, report what fraction of total tokens
             * fall within the window. This shows whether out-of-window tokens
             * (candidates for lower precision) represent a significant fraction. */
            int window = c->sliding_window;
            int final_seq = total_run;
            int in_window = (final_seq <= window) ? final_seq : window;
            double utilization = (final_seq > 0) ? (double)in_window / (double)final_seq * 100.0 : 100.0;
            fprintf(stderr, "\n--- Sliding Window Utilization ---\n");
            fprintf(stderr, "Window size:        %d tokens\n", window);
            fprintf(stderr, "Final context:      %d tokens\n", final_seq);
            fprintf(stderr, "In-window tokens:   %d (%.1f%% of context)\n", in_window, utilization);
            fprintf(stderr, "Out-of-window:      %d tokens (candidates for lower precision)\n",
                    final_seq - in_window);
            fprintf(stderr, "Insight: %.1f%% of attention weight is concentrated on recent %d tokens.\n",
                    utilization, in_window);
            fprintf(stderr, "  Tokens outside the window are never attended to in sliding layers,\n");
            fprintf(stderr, "  so they could use lower precision (e.g., 1-bit QJL) without quality impact.\n");
        } else {
            /* For non-sliding-window models, compute average entropy-based
             * estimate of how concentrated attention is on recent tokens.
             * High entropy = attention spread widely, low = focused on few tokens. */
            double total_avg_entropy = 0.0;
            for (int l = 0; l < c->n_layers; l++) {
                for (int h = 0; h < c->n_heads; h++) {
                    double avg_e = (en_count > 0) ?
                        state->entropy_accum[(size_t)l * c->n_heads + h] / (double)en_count : 0.0;
                    total_avg_entropy += avg_e;
                }
            }
            total_avg_entropy /= (double)(c->n_layers * c->n_heads);
            /* Effective window: 2^entropy gives rough # of tokens attended to */
            double effective_window = pow(2.0, total_avg_entropy);
            fprintf(stderr, "\n--- Attention Concentration ---\n");
            fprintf(stderr, "Average entropy:     %.2f bits\n", total_avg_entropy);
            fprintf(stderr, "Effective window:    ~%.0f tokens (2^entropy)\n", effective_window);
            fprintf(stderr, "Insight: attention is effectively concentrated on ~%.0f recent tokens.\n",
                    effective_window);
            fprintf(stderr, "  Older tokens beyond this window could use lower precision.\n");
        }

        /* Interpretation */
        fprintf(stderr, "\nInterpretation:\n");
        fprintf(stderr, "  Low entropy layers: quantization error matters less (sharp attention)\n");
        fprintf(stderr, "  High entropy layers: quantization error matters more (diffuse attention)\n");
        fprintf(stderr, "  Consider using higher precision for high-entropy layers.\n");
        fprintf(stderr, "==================================\n");

        tq_free_state(state);
        if (tokenizer) tq_free_tokenizer(tokenizer);
        tq_free_model(model);
        return 0;
    }

    /* Auto-wrap prompt with chat template when --chat is used.
     * Template detection order:
     *   1. Gemma 4 → <|turn>...<turn|> + thinking mode
     *   2. Gemma 2/3 → <start_of_turn>...<end_of_turn>
     *   3. Phi-3/Phi-4 (by filename) → <|user|>...<|end|>
     *   4. Llama 3.x (by filename) → <|start_header_id|>...<|eot_id|>
     *   5. Default → ChatML <|im_start|>...<|im_end|> (Qwen/Qwen3/Qwen3.5) */
    char chat_prompt[8192];
    if (chat_mode) {
        tq_model_config_t* mc = &model->config;
        const char* mp = model_path ? model_path : "";
        /* Basename for filename detection */
        const char* bn = strrchr(mp, '/');
        bn = bn ? bn + 1 : mp;

        int is_phi = (strstr(bn, "phi-3") || strstr(bn, "phi3") ||
                      strstr(bn, "Phi-3") || strstr(bn, "Phi3") ||
                      strstr(bn, "phi-4") || strstr(bn, "phi4") ||
                      strstr(bn, "Phi-4") || strstr(bn, "Phi4"));
        int is_llama3 = (strstr(bn, "Llama-3") || strstr(bn, "llama-3") ||
                         strstr(bn, "Llama3") || strstr(bn, "llama3") ||
                         strstr(bn, "Meta-Llama-3"));

        if (mc->model_type == 1 && mc->is_gemma4) {
            /* Skip <|think|> in CLI — the server suppresses it via logit mask,
             * but the CLI has no such suppression. Without it, the CLI uses
             * plain Gemma 4 format without thinking mode. */
            snprintf(chat_prompt, sizeof(chat_prompt),
                "<|turn>user\n%s<turn|>\n<|turn>model\n", prompt);
        } else if (mc->model_type == 1) {
            snprintf(chat_prompt, sizeof(chat_prompt),
                "<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", prompt);
        } else if (is_phi) {
            /* Phi-3/4: <|user|>...<|end|>\n<|assistant|>\n */
            snprintf(chat_prompt, sizeof(chat_prompt),
                "<|user|>\n%s<|end|>\n<|assistant|>\n", prompt);
        } else if (is_llama3) {
            snprintf(chat_prompt, sizeof(chat_prompt),
                "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n", prompt);
        } else {
            /* Default ChatML (Qwen/Qwen3/Qwen3.5) */
            snprintf(chat_prompt, sizeof(chat_prompt),
                "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt);
        }
        prompt = chat_prompt;
    }

    /* Configure generation */
    tq_gen_config_t config = tq_default_gen_config();
    config.temperature = temperature;
    config.top_p = top_p;
    config.max_tokens = max_tokens;
    config.kv_type = kv_type;
    config.value_quant_bits = value_quant_bits;
    config.v_highres_window = v_highres_window;
    config.delta_kv = delta_kv;
    config.delta_iframe_interval = delta_iframe_int;
    /* Auto progressive for generation path too */
    if (k_highres_window == -1) {
        k_highres_window = (kv_type < TQ_TYPE_COUNT) ? 128 : 0;
    }
    config.k_highres_window = k_highres_window;
    config.save_kv_path = save_kv_file;
    config.load_kv_path = load_kv_file;
    config.rng_seed = rng_seed;
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
        const char* wq_name = model->use_q2_weights ? "Q2" : (model->use_q4_weights ? "Q4" : (model->use_q8_weights ? "Q8" : "FP32"));
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

        /* Compressed KV: keys quantized, values remain FP32.
         * K: blocks_per_head * type_size bytes per head per layer
         * V: n_kv_heads * head_dim * 4 bytes (FP32) per layer */
        size_t block_size = tq_type_block_size(kv_type);
        size_t type_size_bytes = tq_type_type_size(kv_type);
        if (block_size == 0) { block_size = TQ_BK; }
        if (type_size_bytes == 0) { type_size_bytes = sizeof(block_tq_uniform_4b); }
        size_t blocks_per_head = ((size_t)c->head_dim + block_size - 1) / block_size;

        /* K (compressed) + V (Q4/Q2/FP16/FP32) per token */
        size_t k_per_token = (size_t)c->n_layers * c->n_kv_heads
                            * blocks_per_head * type_size_bytes;
        size_t v_per_token;
        const char* v_format_name;
        if (value_quant_bits == 4) {
            /* Q4 V: 16 packed bytes + 4 byte scale per block of 32 */
            int v_blocks = (c->head_dim + 31) / 32;
            v_per_token = (size_t)c->n_layers * c->n_kv_heads * v_blocks * (16 + sizeof(float));
            v_format_name = "Q4";
        } else if (value_quant_bits == 2) {
            /* Q2 V: 8 packed bytes + 4 byte scale per block of 32 */
            int v_blocks = (c->head_dim + 31) / 32;
            v_per_token = (size_t)c->n_layers * c->n_kv_heads * v_blocks * (8 + sizeof(float));
            v_format_name = "Q2";
        } else if (kv_type < TQ_TYPE_COUNT) {
            v_per_token = (size_t)c->n_layers * c->n_kv_heads * c->head_dim * sizeof(uint16_t);
            v_format_name = "FP16";
        } else {
            v_per_token = (size_t)c->n_layers * c->n_kv_heads * c->head_dim * sizeof(float);
            v_format_name = "FP32";
        }
        size_t compressed_per_token = k_per_token + v_per_token;

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
        fprintf(stderr, "Per-token K (%s): %.2f KB\n",
                kv_type < TQ_TYPE_COUNT ? tq_type_name(kv_type) : "fp32",
                (double)k_per_token / 1024.0);
        fprintf(stderr, "Per-token V (%s):   %.2f KB\n",
                v_format_name,
                (double)v_per_token / 1024.0);
        fprintf(stderr, "Per-token K+V total:  %.2f KB\n",
                (double)compressed_per_token / 1024.0);
        fprintf(stderr, "Per-token K+V (FP16): %.2f KB\n",
                (double)fp16_per_token / 1024.0);
        fprintf(stderr, "Total K+V:            %.2f MB\n",
                (double)total_compressed / (1024.0 * 1024.0));
        fprintf(stderr, "Total K+V (FP16):     %.2f MB\n",
                (double)total_fp16 / (1024.0 * 1024.0));
        fprintf(stderr, "Compression ratio:    %.2fx (K+V combined)\n", ratio);
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
