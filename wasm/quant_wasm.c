/**
 * quant_wasm.c — WASM entry point for quant.cpp browser demo
 *
 * Compiled with Emscripten: emcc quant_wasm.c -o quant.js
 * Uses the single-header quant.h for zero-dependency builds.
 *
 * Build with -sASYNCIFY to enable wasm_generate_async(), which
 * yields to the browser event loop between tokens for real-time
 * streaming output.
 */

#define QUANT_IMPLEMENTATION
#include "../quant.h"

#include <emscripten.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Global state */
static quant_model* g_model = NULL;
static quant_ctx*   g_ctx = NULL;
static char         g_output[65536];
static int          g_output_pos = 0;
static int          g_generating = 0;

/* JS callback: called for each generated token */
EM_JS(void, js_on_token, (const char* text), {
    if (Module.onToken) Module.onToken(UTF8ToString(text));
});

EM_JS(void, js_on_done, (int n_tokens, double elapsed_ms), {
    if (Module.onDone) Module.onDone(n_tokens, elapsed_ms);
});

EM_JS(void, js_on_status, (const char* msg), {
    if (Module.onStatus) Module.onStatus(UTF8ToString(msg));
});

/* Token callback for streaming — calls JS then yields to browser.
 * Yields every 4 tokens to reduce ASYNCIFY stack unwind/rewind overhead. */
static int g_stream_count = 0;
static void on_token_streaming(const char* text, void* ud) {
    (void)ud;
    js_on_token(text);
    int len = (int)strlen(text);
    if (g_output_pos + len < (int)sizeof(g_output) - 1) {
        memcpy(g_output + g_output_pos, text, len);
        g_output_pos += len;
        g_output[g_output_pos] = '\0';
    }
    if (++g_stream_count % 4 == 0) {
        emscripten_sleep(0);
    }
}

/* Non-yielding callback (fallback for non-ASYNCIFY builds) */
static void on_token_sync(const char* text, void* ud) {
    (void)ud;
    js_on_token(text);
    int len = (int)strlen(text);
    if (g_output_pos + len < (int)sizeof(g_output) - 1) {
        memcpy(g_output + g_output_pos, text, len);
        g_output_pos += len;
        g_output[g_output_pos] = '\0';
    }
}

/* Load model from WASM filesystem (pre-loaded via fetch) */
EMSCRIPTEN_KEEPALIVE
int wasm_load_model(const char* path) {
    js_on_status("Loading model...");

    if (g_model) {
        quant_free_model(g_model);
        g_model = NULL;
    }
    if (g_ctx) {
        quant_free_ctx(g_ctx);
        g_ctx = NULL;
    }

    g_model = quant_load(path);
    if (!g_model) {
        js_on_status("Error: failed to load model");
        return -1;
    }

    quant_config cfg = {
        .temperature = 0.7f,
        .top_p = 0.9f,
        .max_tokens = 512,
        .n_threads = 1,  /* WASM: single thread for compatibility */
        .kv_compress = 1, /* 4-bit KV compression */
    };
    g_ctx = quant_new(g_model, &cfg);
    if (!g_ctx) {
        js_on_status("Error: failed to create context");
        return -1;
    }

    js_on_status("Model loaded! Ready to chat.");
    return 0;
}

/* Async generate — yields to browser between tokens (requires -sASYNCIFY) */
EMSCRIPTEN_KEEPALIVE
int wasm_generate_async(const char* prompt, float temperature, int max_tokens) {
    if (!g_model || !g_ctx) {
        js_on_status("Error: no model loaded");
        return -1;
    }
    if (g_generating) {
        js_on_status("Error: generation in progress");
        return -1;
    }

    g_generating = 1;
    g_output_pos = 0;
    g_output[0] = '\0';
    g_stream_count = 0;

    quant_config cfg = {
        .temperature = temperature,
        .top_p = 0.9f,
        .max_tokens = max_tokens > 0 ? max_tokens : 256,
        .n_threads = 1,
        .kv_compress = 1,
    };

    if (g_ctx) quant_free_ctx(g_ctx);
    g_ctx = quant_new(g_model, &cfg);

    double t0 = emscripten_get_now();

    /* Streaming generation — yields every 4 tokens to browser. */
    int n_tokens = quant_generate(g_ctx, prompt, on_token_streaming, NULL);

    double elapsed = emscripten_get_now() - t0;

    if (n_tokens > 0) {
        js_on_done(n_tokens, elapsed);
    } else {
        js_on_done(0, elapsed);
        if (g_output_pos == 0)
            js_on_status("No output \xe2\x80\x94 try a different prompt");
    }

    g_generating = 0;
    return 0;
}

/* Sync generate — does NOT yield to browser (fallback) */
EMSCRIPTEN_KEEPALIVE
int wasm_generate(const char* prompt, float temperature, int max_tokens) {
    if (!g_model || !g_ctx) {
        js_on_status("Error: no model loaded");
        return -1;
    }
    if (g_generating) {
        js_on_status("Error: generation in progress");
        return -1;
    }

    g_generating = 1;
    g_output_pos = 0;
    g_output[0] = '\0';

    quant_config cfg = {
        .temperature = temperature,
        .top_p = 0.9f,
        .max_tokens = max_tokens > 0 ? max_tokens : 256,
        .n_threads = 1,
        .kv_compress = 1,
    };

    if (g_ctx) quant_free_ctx(g_ctx);
    g_ctx = quant_new(g_model, &cfg);

    double t0 = emscripten_get_now();
    int n_tokens = quant_generate(g_ctx, prompt, on_token_sync, NULL);
    double elapsed = emscripten_get_now() - t0;

    if (n_tokens > 0) {
        js_on_done(n_tokens, elapsed);
    } else {
        js_on_done(0, elapsed);
        if (g_output_pos == 0)
            js_on_status("No output \xe2\x80\x94 try a different prompt");
    }

    g_generating = 0;
    return 0;
}

/* Get model info */
EMSCRIPTEN_KEEPALIVE
const char* wasm_model_info(void) {
    static char info[256];
    if (!g_model) {
        snprintf(info, sizeof(info), "No model loaded");
    } else {
        snprintf(info, sizeof(info), "Model loaded and ready");
    }
    return info;
}

/* Check if model is loaded */
EMSCRIPTEN_KEEPALIVE
int wasm_is_ready(void) {
    return (g_model != NULL && g_ctx != NULL) ? 1 : 0;
}

int main() {
    js_on_status("quant.cpp WASM runtime ready. Choose a model to start.");
    return 0;
}
