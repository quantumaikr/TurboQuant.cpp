/* phi3_infer_test — minimal end-to-end inference test for Phi-3.
 *
 * Loads the model, prefills a known prompt, generates ~80 tokens with
 * greedy sampling, and prints them. We're not validating quality
 * against a reference here — just checking that the output is coherent
 * English text instead of garbage tokens. */
#define QUANT_IMPLEMENTATION
#include "../quant.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static void print_token(const char* text, void* ud) {
    (void)ud;
    fputs(text, stdout);
    fflush(stdout);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [prompt]\n", argv[0]);
        return 1;
    }

    /* Phi-3 chat template:
     *   <|user|>\n{msg}<|end|>\n<|assistant|>\n
     * (verified against the GGUF chat_template metadata) */
    const char* user_msg = (argc >= 3) ? argv[2] : "What is the capital of France?";
    char prompt[1024];
    snprintf(prompt, sizeof(prompt),
             "<|user|>\n%s<|end|>\n<|assistant|>\n", user_msg);

    fprintf(stderr, "Loading %s ...\n", argv[1]);
    quant_model* model = quant_load(argv[1]);
    if (!model) {
        fprintf(stderr, "quant_load failed\n");
        return 2;
    }

    quant_config cfg = {
        .temperature = 0.0f,   /* greedy */
        .top_p = 1.0f,
        .max_tokens = 80,
        .n_threads = 4,
        .kv_compress = 0,
    };
    quant_ctx* ctx = quant_new(model, &cfg);
    if (!ctx) {
        fprintf(stderr, "quant_new failed\n");
        quant_free_model(model);
        return 3;
    }

    fprintf(stderr, "\n--- prompt ---\n%s\n--- response ---\n", prompt);
    int n = quant_generate(ctx, prompt, print_token, NULL);
    fprintf(stderr, "\n--- end ---\ngenerated %d tokens\n", n);

    quant_free_ctx(ctx);
    quant_free_model(model);
    return n > 0 ? 0 : 4;
}
