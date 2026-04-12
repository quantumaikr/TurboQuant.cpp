/**
 * quant_wasm.c — WASM entry point for quant.cpp browser demo
 *
 * Architecture: main thread runs inference with ASYNCIFY for UI yield.
 * pthreads run internally inside quant.h for parallel matmul.
 */

#define QUANT_IMPLEMENTATION
#include "../quant.h"

#include <emscripten.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static quant_model* g_model = NULL;
static quant_ctx*   g_ctx = NULL;
static char         g_output[65536];
static int          g_output_pos = 0;
static int          g_generating = 0;
static int          g_wasm_threads = 1;
static int          g_stream_count = 0;

EM_JS(void, js_on_token, (const char* text), {
    if (Module.onToken) Module.onToken(UTF8ToString(text));
});
EM_JS(void, js_on_done, (int n_tokens, double elapsed_ms), {
    if (Module.onDone) Module.onDone(n_tokens, elapsed_ms);
});
EM_JS(void, js_on_status, (const char* msg), {
    if (Module.onStatus) Module.onStatus(UTF8ToString(msg));
});
EM_JS(int, js_get_hw_concurrency, (void), {
    return Math.min(navigator.hardwareConcurrency || 1, 8);
});

/* Token callback — yield every 4 tokens for UI responsiveness */
static void on_token_streaming(const char* text, void* ud) {
    (void)ud;
    js_on_token(text);
    int len = (int)strlen(text);
    if (g_output_pos + len < (int)sizeof(g_output) - 1) {
        memcpy(g_output + g_output_pos, text, len);
        g_output_pos += len;
        g_output[g_output_pos] = '\0';
    }
    if (++g_stream_count % 4 == 0) emscripten_sleep(0);
}

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

EMSCRIPTEN_KEEPALIVE
int wasm_load_model(const char* path) {
    js_on_status("Loading model...");
    if (g_model) { quant_free_model(g_model); g_model = NULL; }
    if (g_ctx)   { quant_free_ctx(g_ctx);     g_ctx = NULL; }

    g_model = quant_load(path);
    if (!g_model) { js_on_status("Error: failed to load model"); return -1; }

    g_wasm_threads = js_get_hw_concurrency();
    quant_config cfg = {
        .temperature = 0.7f, .top_p = 0.9f, .max_tokens = 512,
        .n_threads = g_wasm_threads, .kv_compress = 1,
    };
    g_ctx = quant_new(g_model, &cfg);
    if (!g_ctx) { js_on_status("Error: failed to create context"); return -1; }

    char msg[128];
    snprintf(msg, sizeof(msg), "Model loaded! Ready to chat. (%d threads)", g_wasm_threads);
    js_on_status(msg);
    return 0;
}

EMSCRIPTEN_KEEPALIVE
int wasm_generate_async(const char* prompt, float temperature, int max_tokens) {
    if (!g_model || !g_ctx || g_generating) return -1;
    g_generating = 1; g_output_pos = 0; g_output[0] = '\0'; g_stream_count = 0;

    /* Update generation params on the existing context (don't recreate —
     * that would wipe the chat KV cache built up by previous turns).
     * kv_compress is set at quant_new() time and is immutable on the ctx. */
    g_ctx->config.temperature = temperature;
    g_ctx->config.top_p = 0.9f;
    g_ctx->config.max_tokens = max_tokens > 0 ? max_tokens : 256;

    double t0 = emscripten_get_now();
    /* quant_chat reuses the KV cache across calls — turn N's prefill is
     * O(new tokens since last call), not O(full history). The browser
     * sees a near-instant response on every turn after the first. */
    int n = quant_chat(g_ctx, prompt, on_token_streaming, NULL);
    double elapsed = emscripten_get_now() - t0;
    if (n == -2) {
        /* Context overflow — auto-reset and inform the JS caller so it
         * can show a "context full, starting new chat" message and
         * optionally retry with a shorter history. */
        js_on_status("Context full \xe2\x80\x94 chat reset. Send a shorter message.");
        quant_chat(g_ctx, NULL, NULL, NULL);
        g_output_pos = 0; g_output[0] = '\0'; g_stream_count = 0;
        js_on_done(0, elapsed);
        g_generating = 0;
        return -2;
    }
    js_on_done(n > 0 ? n : 0, elapsed);
    g_generating = 0;
    return 0;
}

EMSCRIPTEN_KEEPALIVE
int wasm_generate(const char* prompt, float temperature, int max_tokens) {
    if (!g_model || !g_ctx || g_generating) return -1;
    g_generating = 1; g_output_pos = 0; g_output[0] = '\0'; g_stream_count = 0;

    g_ctx->config.temperature = temperature;
    g_ctx->config.top_p = 0.9f;
    g_ctx->config.max_tokens = max_tokens > 0 ? max_tokens : 256;

    double t0 = emscripten_get_now();
    int n = quant_chat(g_ctx, prompt, on_token_sync, NULL);
    double elapsed = emscripten_get_now() - t0;
    if (n == -2) {
        js_on_status("Context full \xe2\x80\x94 chat reset.");
        quant_chat(g_ctx, NULL, NULL, NULL);
        g_output_pos = 0; g_output[0] = '\0'; g_stream_count = 0;
        js_on_done(0, elapsed);
        g_generating = 0;
        return -2;
    }
    js_on_done(n > 0 ? n : 0, elapsed);
    g_generating = 0;
    return 0;
}

/* Reset the chat session — wipes KV cache. Call when user starts new chat. */
EMSCRIPTEN_KEEPALIVE
void wasm_reset_chat(void) {
    if (g_ctx) quant_chat(g_ctx, NULL, NULL, NULL);
    /* Also reset the streaming output buffer state — otherwise the next
     * generation would append to stale text from the previous chat. */
    g_output_pos = 0;
    g_output[0] = '\0';
    g_stream_count = 0;
}

EMSCRIPTEN_KEEPALIVE const char* wasm_model_info(void) {
    static char info[256];
    snprintf(info, sizeof(info), g_model ? "Model loaded (%d threads)" : "No model loaded", g_wasm_threads);
    return info;
}
EMSCRIPTEN_KEEPALIVE int wasm_is_ready(void) { return (g_model && g_ctx) ? 1 : 0; }

int main() { js_on_status("quant.cpp WASM runtime ready. Choose a model to start."); return 0; }
