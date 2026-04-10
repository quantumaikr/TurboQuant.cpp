/**
 * quant_wasm.c — WASM entry point for quant.cpp browser demo
 *
 * Runs inside a Web Worker (inference-worker.js). No ASYNCIFY needed —
 * the worker blocks on quant_generate() while postMessage streams tokens
 * to the main thread. This eliminates ~30-50% ASYNCIFY overhead.
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
static int          g_wasm_threads = 1;

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

/* Query thread count from JS (works in both window and worker scope) */
EM_JS(int, js_get_hw_concurrency, (void), {
    return Math.min(typeof navigator !== 'undefined' ?
        (navigator.hardwareConcurrency || 1) : 1, 8);
});

/* Token callback — calls JS which posts to main thread via Worker */
static void on_token(const char* text, void* ud) {
    (void)ud;
    js_on_token(text);
    int len = (int)strlen(text);
    if (g_output_pos + len < (int)sizeof(g_output) - 1) {
        memcpy(g_output + g_output_pos, text, len);
        g_output_pos += len;
        g_output[g_output_pos] = '\0';
    }
}

/* Load model from WASM filesystem */
EMSCRIPTEN_KEEPALIVE
int wasm_load_model(const char* path) {
    js_on_status("Loading model...");

    if (g_model) { quant_free_model(g_model); g_model = NULL; }
    if (g_ctx)   { quant_free_ctx(g_ctx);     g_ctx = NULL; }

    g_model = quant_load(path);
    if (!g_model) {
        js_on_status("Error: failed to load model");
        return -1;
    }

    g_wasm_threads = js_get_hw_concurrency();

    quant_config cfg = {
        .temperature = 0.7f,
        .top_p = 0.9f,
        .max_tokens = 512,
        .n_threads = g_wasm_threads,
        .kv_compress = 1,
    };
    g_ctx = quant_new(g_model, &cfg);
    if (!g_ctx) {
        js_on_status("Error: failed to create context");
        return -1;
    }

    char status_msg[128];
    snprintf(status_msg, sizeof(status_msg),
             "Model loaded! Ready to chat. (%d threads)", g_wasm_threads);
    js_on_status(status_msg);
    return 0;
}

/* Generate — blocks the worker thread (main thread stays responsive) */
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
        .n_threads = g_wasm_threads,
        .kv_compress = 1,
    };

    if (g_ctx) quant_free_ctx(g_ctx);
    g_ctx = quant_new(g_model, &cfg);

    double t0 = emscripten_get_now();
    int n_tokens = quant_generate(g_ctx, prompt, on_token, NULL);
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

EMSCRIPTEN_KEEPALIVE
const char* wasm_model_info(void) {
    static char info[256];
    if (!g_model) snprintf(info, sizeof(info), "No model loaded");
    else snprintf(info, sizeof(info), "Model loaded (%d threads)", g_wasm_threads);
    return info;
}

EMSCRIPTEN_KEEPALIVE
int wasm_is_ready(void) {
    return (g_model != NULL && g_ctx != NULL) ? 1 : 0;
}

int main() {
    js_on_status("quant.cpp WASM runtime ready.");
    return 0;
}
