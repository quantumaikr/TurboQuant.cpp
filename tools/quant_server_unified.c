/**
 * quant_server_unified.c — OpenAI-compatible server built on quant.h
 *
 * Uses quant.h's public API (quant_load, quant_chat, quant_generate)
 * instead of libturboquant internals. This guarantees the server's
 * inference path is identical to the single-header library.
 *
 * Build:
 *   cc -O2 -o quant-server-unified tools/quant_server_unified.c -lm -lpthread
 *
 * Usage:
 *   ./quant-server-unified model.gguf [-p PORT] [-j THREADS]
 */
#define QUANT_IMPLEMENTATION
#include "../quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>

/* ============================================================
 * Configuration
 * ============================================================ */
#define MAX_BODY      (256 * 1024)  /* 256 KB max request body */
#define MAX_MESSAGES  64
#define MAX_OUTPUT    (128 * 1024)  /* 128 KB max response */
#define MAX_HEADER    8192

typedef struct {
    quant_model* model;
    quant_ctx*   ctx;
    const char*  model_path;
    const char*  model_id;
    int          port;
    int          n_threads;
    int          has_fused_qkv;   /* Phi-3 detection */
    pthread_mutex_t mutex;
} server_t;

/* ============================================================
 * Chat template
 * ============================================================ */
static char* build_prompt(const char** roles, const char** contents,
                           int n_msgs, int is_phi3) {
    size_t total = 256;
    for (int i = 0; i < n_msgs; i++)
        total += 64 + (contents[i] ? strlen(contents[i]) : 0);

    char* p = (char*)malloc(total);
    if (!p) return NULL;
    char* w = p;
    size_t rem = total;

    for (int i = 0; i < n_msgs; i++) {
        const char* c = contents[i] ? contents[i] : "";
        int n;
        if (is_phi3) {
            if (strcmp(roles[i], "system") == 0)
                n = snprintf(w, rem, "<|system|>\n%s<|end|>\n", c);
            else if (strcmp(roles[i], "user") == 0)
                n = snprintf(w, rem, "<|user|>\n%s<|end|>\n", c);
            else
                n = snprintf(w, rem, "<|assistant|>\n%s<|end|>\n", c);
        } else {
            n = snprintf(w, rem, "<|im_start|>%s\n%s<|im_end|>\n", roles[i], c);
        }
        if (n > 0 && (size_t)n < rem) { w += n; rem -= (size_t)n; }
    }
    if (is_phi3)
        snprintf(w, rem, "<|assistant|>\n");
    else
        snprintf(w, rem, "<|im_start|>assistant\n");

    return p;
}

/* ============================================================
 * Minimal JSON parser (good enough for OpenAI format)
 * ============================================================ */
static const char* json_find_key(const char* json, const char* key) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':') p++;
    return p;
}

static int json_get_int(const char* json, const char* key, int def) {
    const char* p = json_find_key(json, key);
    if (!p) return def;
    return atoi(p);
}

static double json_get_float(const char* json, const char* key, double def) {
    const char* p = json_find_key(json, key);
    if (!p) return def;
    return atof(p);
}

static int json_get_bool(const char* json, const char* key, int def) {
    const char* p = json_find_key(json, key);
    if (!p) return def;
    if (strncmp(p, "true", 4) == 0) return 1;
    if (strncmp(p, "false", 5) == 0) return 0;
    return def;
}

static int json_get_string(const char* json, const char* key, char* out, int max) {
    const char* p = json_find_key(json, key);
    if (!p || *p != '"') return -1;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < max - 1) {
        if (*p == '\\' && *(p+1)) { p++; }
        out[i++] = *p++;
    }
    out[i] = '\0';
    return i;
}

/* Parse messages array. Returns count. */
static int parse_messages(const char* body, const char** roles,
                           const char** contents, char** bufs, int max) {
    const char* p = json_find_key(body, "messages");
    if (!p || *p != '[') return 0;
    p++;

    int n = 0;
    while (*p && *p != ']' && n < max) {
        const char* obj = strchr(p, '{');
        if (!obj) break;
        const char* end = strchr(obj, '}');
        if (!end) break;

        /* Extract role and content */
        int len = (int)(end - obj + 1);
        bufs[n] = (char*)malloc(len + 1);
        memcpy(bufs[n], obj, len);
        bufs[n][len] = '\0';

        static char role_buf[MAX_MESSAGES][32];
        static char content_buf[MAX_MESSAGES][MAX_BODY];

        json_get_string(bufs[n], "role", role_buf[n], 32);
        json_get_string(bufs[n], "content", content_buf[n], MAX_BODY);

        roles[n] = role_buf[n];
        contents[n] = content_buf[n];
        n++;
        p = end + 1;
    }
    return n;
}

/* ============================================================
 * JSON escape
 * ============================================================ */
static void json_escape(const char* src, char* dst, size_t cap) {
    size_t j = 0;
    for (size_t i = 0; src[i] && j < cap - 2; i++) {
        unsigned char c = (unsigned char)src[i];
        if (c == '"')       { dst[j++] = '\\'; dst[j++] = '"'; }
        else if (c == '\\') { dst[j++] = '\\'; dst[j++] = '\\'; }
        else if (c == '\n') { dst[j++] = '\\'; dst[j++] = 'n'; }
        else if (c == '\r') { dst[j++] = '\\'; dst[j++] = 'r'; }
        else if (c == '\t') { dst[j++] = '\\'; dst[j++] = 't'; }
        else if (c < 0x20)  { j += snprintf(dst + j, cap - j, "\\u%04x", c); }
        else                { dst[j++] = c; }
    }
    dst[j] = '\0';
}

/* ============================================================
 * HTTP helpers
 * ============================================================ */
static void send_response(int fd, int code, const char* status,
                            const char* content_type, const char* body, int body_len) {
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Connection: close\r\n\r\n",
        code, status, content_type, body_len);
    write(fd, header, hlen);
    if (body_len > 0) write(fd, body, body_len);
}

static void send_json(int fd, int code, const char* status, const char* json) {
    send_response(fd, code, status, "application/json", json, (int)strlen(json));
}

static void send_sse_chunk(int fd, const char* data) {
    char buf[4096];
    int n = snprintf(buf, sizeof(buf), "data: %s\n\n", data);
    write(fd, buf, n);
}

/* ============================================================
 * Streaming token collector
 * ============================================================ */
typedef struct {
    int fd;
    char completion_id[32];
    const char* model_id;
    int token_count;
    int is_phi3;
} stream_ctx_t;

static void stream_on_token(const char* text, void* user_data) {
    stream_ctx_t* sc = (stream_ctx_t*)user_data;
    if (!text || !text[0]) return;

    /* Skip template tokens */
    if (strstr(text, "<|end|>") || strstr(text, "<|assistant|>") ||
        strstr(text, "<|user|>") || strstr(text, "<|system|>") ||
        strstr(text, "<|im_end|>") || strstr(text, "<|im_start|>") ||
        strstr(text, "<|endoftext|>")) return;

    /* JSON-escape the token */
    char escaped[1024];
    json_escape(text, escaped, sizeof(escaped));

    char chunk[2048];
    snprintf(chunk, sizeof(chunk),
        "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%ld,\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}",
        sc->completion_id, (long)time(NULL), sc->model_id, escaped);

    send_sse_chunk(sc->fd, chunk);
    sc->token_count++;
}

/* Non-streaming token collector */
typedef struct {
    char* buf;
    size_t len;
    size_t cap;
    int count;
    int is_phi3;
} collect_ctx_t;

static void collect_on_token(const char* text, void* user_data) {
    collect_ctx_t* cc = (collect_ctx_t*)user_data;
    if (!text || !text[0]) return;

    /* Skip template tokens */
    if (strstr(text, "<|end|>") || strstr(text, "<|assistant|>") ||
        strstr(text, "<|user|>") || strstr(text, "<|system|>") ||
        strstr(text, "<|im_end|>") || strstr(text, "<|im_start|>") ||
        strstr(text, "<|endoftext|>")) return;

    size_t tlen = strlen(text);
    if (cc->len + tlen >= cc->cap) {
        cc->cap = (cc->cap + tlen) * 2;
        cc->buf = (char*)realloc(cc->buf, cc->cap);
    }
    memcpy(cc->buf + cc->len, text, tlen);
    cc->len += tlen;
    cc->buf[cc->len] = '\0';
    cc->count++;
}

/* ============================================================
 * Request handler
 * ============================================================ */
static void handle_request(server_t* srv, int fd) {
    char header[MAX_HEADER];
    int hlen = 0;
    while (hlen < MAX_HEADER - 1) {
        int n = read(fd, header + hlen, 1);
        if (n <= 0) break;
        hlen++;
        if (hlen >= 4 && memcmp(header + hlen - 4, "\r\n\r\n", 4) == 0) break;
    }
    header[hlen] = '\0';

    /* Parse method and path */
    char method[8] = {0}, path[256] = {0};
    sscanf(header, "%7s %255s", method, path);

    /* OPTIONS (CORS preflight) */
    if (strcmp(method, "OPTIONS") == 0) {
        send_response(fd, 204, "No Content", "text/plain", "", 0);
        return;
    }

    /* GET /health */
    if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0) {
        char body[128];
        snprintf(body, sizeof(body), "{\"status\":\"ok\",\"version\":\"%s\"}", quant_version());
        send_json(fd, 200, "OK", body);
        return;
    }

    /* GET /v1/models */
    if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/models") == 0) {
        char body[512];
        snprintf(body, sizeof(body),
            "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\","
            "\"created\":%ld,\"owned_by\":\"quant.cpp\"}]}",
            srv->model_id, (long)time(NULL));
        send_json(fd, 200, "OK", body);
        return;
    }

    /* POST /v1/chat/completions */
    if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/chat/completions") == 0) {
        /* Read body */
        int content_length = 0;
        const char* cl = strstr(header, "Content-Length:");
        if (!cl) cl = strstr(header, "content-length:");
        if (cl) content_length = atoi(cl + 15);

        if (content_length <= 0 || content_length > MAX_BODY) {
            send_json(fd, 400, "Bad Request", "{\"error\":{\"message\":\"Invalid body\"}}");
            return;
        }

        char* body = (char*)malloc(content_length + 1);
        int bread = 0;
        while (bread < content_length) {
            int n = read(fd, body + bread, content_length - bread);
            if (n <= 0) break;
            bread += n;
        }
        body[bread] = '\0';

        /* Parse request */
        int max_tokens = json_get_int(body, "max_tokens", 256);
        float temperature = (float)json_get_float(body, "temperature", 0.7);
        int stream = json_get_bool(body, "stream", 0);

        const char* roles[MAX_MESSAGES];
        const char* contents[MAX_MESSAGES];
        char* bufs[MAX_MESSAGES];
        memset(bufs, 0, sizeof(bufs));
        int n_msgs = parse_messages(body, roles, contents, bufs, MAX_MESSAGES);

        if (n_msgs == 0) {
            free(body);
            send_json(fd, 400, "Bad Request", "{\"error\":{\"message\":\"No messages\"}}");
            return;
        }

        /* Build prompt */
        char* prompt = build_prompt(roles, contents, n_msgs, srv->has_fused_qkv);

        /* Generate completion ID */
        char comp_id[32];
        snprintf(comp_id, sizeof(comp_id), "chatcmpl-%lx", (long)time(NULL));

        fprintf(stderr, "[%s] POST /v1/chat/completions msgs=%d max_tokens=%d stream=%d\n",
                comp_id, n_msgs, max_tokens, stream);

        pthread_mutex_lock(&srv->mutex);

        /* Reuse context across requests — only update per-request config.
         * The old code called quant_free_ctx + quant_new per request,
         * which re-parsed the tokenizer (32K tokens from GGUF!) and
         * double-allocated state buffers. quant_generate() internally
         * resets the KV state anyway, so we only need to update
         * temperature and max_tokens on the existing context. */
        srv->ctx->config.temperature = temperature;
        srv->ctx->config.max_tokens = max_tokens;

        if (stream) {
            /* SSE streaming */
            char sse_header[512];
            int shlen = snprintf(sse_header, sizeof(sse_header),
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Cache-Control: no-cache\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
                "Connection: close\r\n\r\n");
            write(fd, sse_header, shlen);

            /* Initial chunk with role */
            char init_chunk[512];
            snprintf(init_chunk, sizeof(init_chunk),
                "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
                "\"created\":%ld,\"model\":\"%s\","
                "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}",
                comp_id, (long)time(NULL), srv->model_id);
            send_sse_chunk(fd, init_chunk);

            /* Stream tokens */
            stream_ctx_t sc = {
                .fd = fd,
                .model_id = srv->model_id,
                .token_count = 0,
                .is_phi3 = srv->has_fused_qkv,
            };
            strncpy(sc.completion_id, comp_id, sizeof(sc.completion_id) - 1);

            quant_generate(srv->ctx, prompt, stream_on_token, &sc);

            /* Final chunk */
            char final_chunk[512];
            snprintf(final_chunk, sizeof(final_chunk),
                "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
                "\"created\":%ld,\"model\":\"%s\","
                "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}",
                comp_id, (long)time(NULL), srv->model_id);
            send_sse_chunk(fd, final_chunk);
            send_sse_chunk(fd, "[DONE]");

            fprintf(stderr, "[%s] Streamed %d tokens\n", comp_id, sc.token_count);
        } else {
            /* Non-streaming */
            collect_ctx_t cc = { .buf = (char*)calloc(1, 4096), .len = 0, .cap = 4096, .count = 0 };

            quant_generate(srv->ctx, prompt, collect_on_token, &cc);

            char* escaped = (char*)malloc(cc.len * 2 + 64);
            json_escape(cc.buf, escaped, cc.len * 2 + 64);

            int prompt_tokens = (int)(strlen(prompt) / 4);
            char* resp = (char*)malloc(strlen(escaped) + 1024);
            snprintf(resp, strlen(escaped) + 1024,
                "{\"id\":\"%s\",\"object\":\"chat.completion\","
                "\"created\":%ld,\"model\":\"%s\","
                "\"choices\":[{\"index\":0,"
                "\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
                "\"finish_reason\":\"stop\"}],"
                "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
                comp_id, (long)time(NULL), srv->model_id,
                escaped, prompt_tokens, cc.count, prompt_tokens + cc.count);

            send_json(fd, 200, "OK", resp);
            fprintf(stderr, "[%s] Generated %d tokens\n", comp_id, cc.count);

            free(resp);
            free(escaped);
            free(cc.buf);
        }

        pthread_mutex_unlock(&srv->mutex);

        free(prompt);
        for (int i = 0; i < n_msgs; i++) free(bufs[i]);
        free(body);
        return;
    }

    /* 404 */
    send_json(fd, 404, "Not Found", "{\"error\":{\"message\":\"Not found\"}}");
}

/* ============================================================
 * Main
 * ============================================================ */
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "quant-server-unified — OpenAI-compatible server (quant.h unified build)\n\n"
            "Usage: %s <model.gguf> [options]\n\n"
            "Options:\n"
            "  -p <port>     Listen port (default: 8080)\n"
            "  -j <threads>  Threads per inference (default: 4)\n"
            "  --help        Show this help\n\n"
            "Example:\n"
            "  %s model.gguf -p 8080 -j 8\n"
            "  curl http://localhost:8080/v1/chat/completions \\\n"
            "    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'\n",
            argv[0], argv[0]);
        return 1;
    }

    signal(SIGPIPE, SIG_IGN);

    const char* model_path = argv[1];
    int port = 8080;
    int n_threads = 4;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) port = atoi(argv[++i]);
        else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
    }

    fprintf(stderr, "Loading %s ...\n", model_path);
    quant_model* model = quant_load(model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    quant_config cfg = {
        .temperature = 0.7f,
        .top_p = 0.9f,
        .max_tokens = 256,
        .n_threads = n_threads,
    };
    quant_ctx* ctx = quant_new(model, &cfg);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        quant_free_model(model);
        return 1;
    }

    /* Detect Phi-3 architecture by checking if the model loaded fused QKV.
     * We do a quick test: try a dummy generate to see if model works. */
    /* Simple heuristic: check model_path for "phi" */
    int has_fused_qkv = 0;
    const char* bn = strrchr(model_path, '/');
    bn = bn ? bn + 1 : model_path;
    if (strstr(bn, "hi-3") || strstr(bn, "hi3") || strstr(bn, "Hi-3") || strstr(bn, "Hi3") ||
        strstr(bn, "phi-3") || strstr(bn, "phi3") || strstr(bn, "Phi-3") || strstr(bn, "Phi3")) {
        has_fused_qkv = 1;
        fprintf(stderr, "Detected Phi-3 model — using Phi-3 chat template\n");
    }

    /* Extract model ID from filename */
    char model_id[256];
    snprintf(model_id, sizeof(model_id), "quant.cpp/%s", bn);

    server_t srv = {
        .model = model,
        .ctx = ctx,
        .model_path = model_path,
        .model_id = model_id,
        .port = port,
        .n_threads = n_threads,
        .has_fused_qkv = has_fused_qkv,
    };
    pthread_mutex_init(&srv.mutex, NULL);

    /* Create socket */
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = INADDR_ANY,
        .sin_port = htons(port),
    };

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "Error: port %d is already in use\n", port);
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, 16) < 0) { perror("listen"); return 1; }

    fprintf(stderr, "\nquant-server-unified listening on http://0.0.0.0:%d\n", port);
    fprintf(stderr, "  Model: %s\n", model_id);
    fprintf(stderr, "  Threads: %d\n", n_threads);
    fprintf(stderr, "  Template: %s\n", has_fused_qkv ? "phi3" : "chatml");
    fprintf(stderr, "  POST /v1/chat/completions\n");
    fprintf(stderr, "  GET  /v1/models\n");
    fprintf(stderr, "  GET  /health\n\n");

    while (1) {
        struct sockaddr_in client;
        socklen_t client_len = sizeof(client);
        int client_fd = accept(server_fd, (struct sockaddr*)&client, &client_len);
        if (client_fd < 0) continue;
        handle_request(&srv, client_fd);
        close(client_fd);
    }

    quant_free_ctx(ctx);
    quant_free_model(model);
    close(server_fd);
    return 0;
}
