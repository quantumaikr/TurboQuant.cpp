/* phi3_kvcomp_test — validate Phi-3 forward path against quant.cpp's KV
 * cache compression layers.
 *
 * The Phi-3 architecture support PR (#65) was validated end-to-end with
 * KV compression DISABLED (kv_compress=0). The fused QKV / fused gate+up
 * forward branches do not touch the KV cache code path directly, but
 * the way s->k is written into the cache (and read back during attention)
 * goes through the same KV-quant code paths as Llama / SmolLM2. This
 * test exercises that interaction.
 *
 * Modes covered:
 *   off                          baseline (matches PR #65 validation)
 *   1 / 4-bit                    UNIFORM_4B K + 4-bit V
 *   1 / 4-bit + progressive 128  + last 128 tokens of K kept FP32
 *   1 / 4-bit + aggressive  512  + last 512 tokens of K kept FP32
 *   2 / delta+3-bit              UNIFORM_3B K + 4-bit V + delta encoding
 *
 * For each mode we generate 80 greedy tokens from a fixed prompt and
 * print the output. A working mode produces coherent English; a broken
 * mode produces fragmented garbage. Compare modes side-by-side.
 */
#define QUANT_IMPLEMENTATION
#include "../quant.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

static void print_token(const char* text, void* ud) {
    (void)ud;
    fputs(text, stdout);
    fflush(stdout);
}

static int run_one(quant_model* model, const char* label,
                    int kv_compress, int k_highres_window,
                    const char* prompt) {
    /* max_tokens=256 deliberately exceeds the 128-token progressive
     * window so we actually exercise the boundary where recent keys
     * shift from FP32 (highres buffer) into the quantized cache. */
    quant_config cfg = {
        .temperature = 0.0f,           /* greedy */
        .top_p = 1.0f,
        .max_tokens = 256,
        .n_threads = 4,
        .kv_compress = kv_compress,
        .context_length = 0,
        .k_highres_window = k_highres_window,
    };
    quant_ctx* ctx = quant_new(model, &cfg);
    if (!ctx) {
        fprintf(stderr, "quant_new failed for mode %s\n", label);
        return -1;
    }

    fprintf(stderr, "\n=== %s ===\n", label);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int n = quant_generate(ctx, prompt, print_token, NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    fprintf(stderr, "\n[%s] %d tokens in %.2fs (%.1f tok/s)\n",
            label, n, secs, secs > 0 ? n / secs : 0.0);

    quant_free_ctx(ctx);
    return n;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [prompt]\n", argv[0]);
        return 1;
    }
    const char* user_msg = (argc >= 3) ? argv[2]
        : "Explain in two paragraphs why the sky appears blue during the day.";

    /* Phi-3 chat template */
    char prompt[1024];
    snprintf(prompt, sizeof(prompt),
             "<|user|>\n%s<|end|>\n<|assistant|>\n", user_msg);

    fprintf(stderr, "Loading %s ...\n", argv[1]);
    quant_model* model = quant_load(argv[1]);
    if (!model) {
        fprintf(stderr, "quant_load failed\n");
        return 2;
    }

    int rc = 0;
    rc |= run_one(model, "off (baseline, FP32 KV)",      0, 0,   prompt) < 0;
    rc |= run_one(model, "kv_compress=1 (4-bit, no progressive)", 1, 0,   prompt) < 0;
    rc |= run_one(model, "kv_compress=1 + progressive(128)",      1, 128, prompt) < 0;
    rc |= run_one(model, "kv_compress=1 + aggressive(512)",       1, 512, prompt) < 0;
    rc |= run_one(model, "kv_compress=2 (delta+3-bit)",           2, 0,   prompt) < 0;

    quant_free_model(model);
    fputc('\n', stderr);
    return rc ? 4 : 0;
}
