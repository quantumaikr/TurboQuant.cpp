/**
 * tq_server.c -- Minimal OpenAI-compatible HTTP server for quant.cpp
 *
 * Pure C, zero external dependencies. Uses POSIX sockets + pthreads.
 * Implements just enough HTTP/1.1 to serve JSON API requests and SSE streams.
 *
 * Endpoints:
 *   POST /v1/chat/completions  — OpenAI chat completions (streaming + non-streaming)
 *   GET  /v1/models            — List loaded model
 *   GET  /health               — Health check
 *
 * Build: linked against libturboquant, libc, libm, libpthread.
 */

#include "tq_server.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>

/* Forward decl: defined in src/engine/tq_generate.c.
 * Not yet exposed in turboquant.h since it's a chat-mode helper. */
extern int tq_generate_continue(tq_model_t* model,
                                 tq_tokenizer_t* tokenizer,
                                 tq_state_t* state,
                                 const char* prompt,
                                 tq_gen_config_t* config,
                                 int** cached_tokens_io,
                                 int*  n_cached_io,
                                 int*  cached_capacity_io,
                                 char* output, int output_size);
#if defined(_MSC_VER)
#include <intrin.h>
typedef volatile long atomic_int;
#define atomic_store(p, v) _InterlockedExchange((p), (v))
#define atomic_load(p) _InterlockedCompareExchange((p), 0, 0)
#else
#include <stdatomic.h>
#endif
#include <signal.h>
#include <errno.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#define pthread_mutex_t SRWLOCK
#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
#define pthread_mutex_lock(m) AcquireSRWLockExclusive(m)
#define pthread_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#else
#include <pthread.h>
#endif
#include <unistd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>

/* ============================================================
 * Constants
 * ============================================================ */

#define HTTP_BUF_SIZE      (64 * 1024)   /* 64 KB per-request buffer  */
#define MAX_BODY_SIZE      (256 * 1024)  /* 256 KB max request body   */
#define MAX_HEADERS        64
#define MAX_TOKENS_DEFAULT 256
#define RESPONSE_BUF_SIZE  (512 * 1024)  /* 512 KB response buffer    */
#define SSE_CHUNK_SIZE     4096
#define SOCKET_TIMEOUT_SEC 30            /* read timeout per socket    */
#define MAX_ACTIVE_CONNS   32            /* hard limit on threads      */

/* ============================================================
 * Server state
 * ============================================================ */

struct tq_server {
    tq_server_config_t config;
    int                listen_fd;
    atomic_int         running;
    atomic_int         active_connections;  /* track concurrent threads */
    pthread_mutex_t    inference_mutex;     /* serialize inference (single model state) */

    /* Persistent inference state — shared across requests for chat-mode
     * KV cache reuse. The inference_mutex above serializes access. */
    tq_state_t*        kv_state;
    int*               cached_tokens;
    int                n_cached;
    int                cached_capacity;
};

/* Global server pointer for signal handler */
static tq_server_t* g_server = NULL;

/* ============================================================
 * Logging
 * ============================================================ */

static void server_log(const char* level, const char* fmt, ...) {
    va_list ap;
    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);
    fprintf(stderr, "[%04d-%02d-%02d %02d:%02d:%02d] [%s] ",
            tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
            tm.tm_hour, tm.tm_min, tm.tm_sec, level);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
}

#define LOG_INFO(...)  server_log("INFO",  __VA_ARGS__)
#define LOG_ERROR(...) server_log("ERROR", __VA_ARGS__)

/* ============================================================
 * Minimal JSON parser — just enough for chat completions
 *
 * We parse:
 *   "model": "string"
 *   "temperature": number
 *   "top_p": number
 *   "max_tokens": number
 *   "stream": bool
 *   "n_threads": number
 *   "kv_type": "string"
 *   "value_quant_bits": number
 *   "delta_kv": bool
 *   "messages": [ {"role": "...", "content": "..."}, ... ]
 *
 * Strategy: scan for known keys, extract values. Not a general parser.
 * ============================================================ */

/* Skip whitespace */
static const char* json_skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

/* Extract a JSON string value (unescaped). Returns pointer past closing quote.
 * Writes result into buf (up to buf_size-1 chars). */
static const char* json_extract_string(const char* p, char* buf, int buf_size) {
    p = json_skip_ws(p);
    if (*p != '"') return NULL;
    p++;
    int i = 0;
    while (*p && *p != '"') {
        if (*p == '\\' && *(p + 1)) {
            p++;
            char c = *p;
            switch (c) {
                case '"':  c = '"';  break;
                case '\\': c = '\\'; break;
                case '/':  c = '/';  break;
                case 'n':  c = '\n'; break;
                case 'r':  c = '\r'; break;
                case 't':  c = '\t'; break;
                case 'b':  c = '\b'; break;
                case 'f':  c = '\f'; break;
                default:   break; /* keep as-is for \uXXXX etc. */
            }
            if (i < buf_size - 1) buf[i++] = c;
        } else {
            if (i < buf_size - 1) buf[i++] = *p;
        }
        p++;
    }
    buf[i] = '\0';
    if (*p == '"') p++;
    return p;
}

/* Find a key in JSON and return pointer to value (past the colon).
 * Simple scan — works for flat or lightly nested objects. */
static const char* json_find_key(const char* json, const char* key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    p = json_skip_ws(p);
    if (*p != ':') return NULL;
    p++;
    return json_skip_ws(p);
}

/* Extract a number (int or float) from current position */
static double json_extract_number(const char* p) {
    return strtod(p, NULL);
}

/* Extract a boolean from current position */
static bool json_extract_bool(const char* p) {
    p = json_skip_ws(p);
    return (strncmp(p, "true", 4) == 0);
}

/* ============================================================
 * Chat message parsing
 *
 * Build a single prompt string from the messages array.
 * Uses ChatML format: <|im_start|>role\ncontent<|im_end|>
 * ============================================================ */

typedef struct {
    char role[32];
    char* content;   /* heap-allocated */
} chat_message_t;

#define MAX_MESSAGES 128

typedef struct {
    /* Parsed request fields */
    char           model[128];
    float          temperature;
    float          top_p;
    int            max_tokens;
    bool           stream;
    int            n_threads;
    char           kv_type_str[32];
    int            value_quant_bits;
    bool           delta_kv;

    /* Messages */
    chat_message_t messages[MAX_MESSAGES];
    int            n_messages;

    /* Built prompt */
    char*          prompt;    /* heap-allocated */
} chat_request_t;

static void free_chat_request(chat_request_t* req) {
    for (int i = 0; i < req->n_messages; i++) {
        free(req->messages[i].content);
    }
    free(req->prompt);
    memset(req, 0, sizeof(*req));
}

/* Parse messages array from JSON body.
 * Input: pointer to '[' of the messages array. */
static int parse_messages(const char* p, chat_request_t* req) {
    if (!p || *p != '[') return -1;
    p++;

    req->n_messages = 0;

    while (*p && *p != ']' && req->n_messages < MAX_MESSAGES) {
        p = json_skip_ws(p);
        if (*p == ',') { p++; p = json_skip_ws(p); }
        if (*p != '{') break;
        p++; /* skip '{' */

        chat_message_t* msg = &req->messages[req->n_messages];
        memset(msg, 0, sizeof(*msg));

        /* Scan for role and content within this object */
        int depth = 1;
        const char* obj_start = p;

        /* Find the end of this object first */
        const char* scan = p;
        while (*scan && depth > 0) {
            if (*scan == '{') depth++;
            else if (*scan == '}') depth--;
            else if (*scan == '"') {
                scan++;
                while (*scan && *scan != '"') {
                    if (*scan == '\\') scan++;
                    scan++;
                }
            }
            if (depth > 0) scan++;
        }
        /* scan now points to closing '}' */

        /* Extract role */
        const char* role_val = json_find_key(obj_start, "role");
        if (role_val) {
            json_extract_string(role_val, msg->role, sizeof(msg->role));
        }

        /* Extract content */
        const char* content_val = json_find_key(obj_start, "content");
        if (content_val) {
            /* Allocate generous buffer for content */
            int max_content = (int)(scan - content_val) + 1;
            if (max_content < 256) max_content = 256;
            msg->content = (char*)malloc(max_content);
            if (msg->content) {
                json_extract_string(content_val, msg->content, max_content);
            }
        } else {
            msg->content = strdup("");
        }

        req->n_messages++;
        p = scan;
        if (*p == '}') p++;
    }

    return 0;
}

/* Build a ChatML-formatted prompt from messages */
static char* build_prompt(const chat_request_t* req) {
    /* Calculate total size needed */
    size_t total = 1; /* null terminator */
    for (int i = 0; i < req->n_messages; i++) {
        /* <|im_start|>role\ncontent<|im_end|>\n */
        total += 14 + strlen(req->messages[i].role) + 1 +
                 (req->messages[i].content ? strlen(req->messages[i].content) : 0) +
                 12 + 1;
    }
    /* Add assistant prompt at end */
    total += 14 + 9 + 1; /* <|im_start|>assistant\n */

    char* prompt = (char*)malloc(total);
    if (!prompt) return NULL;

    char* w = prompt;
    size_t remaining = total;
    for (int i = 0; i < req->n_messages; i++) {
        int n = snprintf(w, remaining, "<|im_start|>%s\n%s<|im_end|>\n",
                         req->messages[i].role,
                         req->messages[i].content ? req->messages[i].content : "");
        if (n > 0 && (size_t)n < remaining) { w += n; remaining -= (size_t)n; }
    }
    snprintf(w, remaining, "<|im_start|>assistant\n");

    return prompt;
}

/* Parse a chat completion request from JSON body */
static int parse_chat_request(const char* body, chat_request_t* req) {
    memset(req, 0, sizeof(*req));

    /* Defaults */
    req->temperature = 0.7f;
    req->top_p = 0.9f;
    req->max_tokens = MAX_TOKENS_DEFAULT;
    req->stream = false;
    req->n_threads = 0;  /* 0 = use server default */
    req->value_quant_bits = -1; /* -1 = use server default */
    req->delta_kv = false;
    strcpy(req->model, "default");

    /* Extract top-level fields */
    const char* v;

    v = json_find_key(body, "model");
    if (v) json_extract_string(v, req->model, sizeof(req->model));

    v = json_find_key(body, "temperature");
    if (v) req->temperature = (float)json_extract_number(v);

    v = json_find_key(body, "top_p");
    if (v) req->top_p = (float)json_extract_number(v);

    v = json_find_key(body, "max_tokens");
    if (v) req->max_tokens = (int)json_extract_number(v);

    v = json_find_key(body, "stream");
    if (v) req->stream = json_extract_bool(v);

    v = json_find_key(body, "n_threads");
    if (v) req->n_threads = (int)json_extract_number(v);

    v = json_find_key(body, "kv_type");
    if (v) json_extract_string(v, req->kv_type_str, sizeof(req->kv_type_str));

    v = json_find_key(body, "value_quant_bits");
    if (v) req->value_quant_bits = (int)json_extract_number(v);

    v = json_find_key(body, "delta_kv");
    if (v) req->delta_kv = json_extract_bool(v);

    /* Parse messages */
    v = json_find_key(body, "messages");
    if (!v) {
        LOG_ERROR("Missing 'messages' field in request");
        return -1;
    }
    if (parse_messages(v, req) < 0) {
        LOG_ERROR("Failed to parse messages array");
        return -1;
    }

    if (req->n_messages == 0) {
        LOG_ERROR("Empty messages array");
        return -1;
    }

    /* Build prompt */
    req->prompt = build_prompt(req);
    if (!req->prompt) {
        LOG_ERROR("Failed to build prompt");
        return -1;
    }

    return 0;
}

/* ============================================================
 * KV type string -> enum
 * ============================================================ */

static tq_type parse_kv_type_str(const char* s, tq_type fallback) {
    if (!s || !s[0]) return fallback;
    if (strcmp(s, "fp32") == 0)       return TQ_TYPE_COUNT; /* sentinel */
    if (strcmp(s, "uniform_4b") == 0) return TQ_TYPE_UNIFORM_4B;
    if (strcmp(s, "uniform_2b") == 0) return TQ_TYPE_UNIFORM_2B;
    if (strcmp(s, "polar_3b") == 0)   return TQ_TYPE_POLAR_3B;
    if (strcmp(s, "polar_4b") == 0)   return TQ_TYPE_POLAR_4B;
    if (strcmp(s, "turbo_3b") == 0)   return TQ_TYPE_TURBO_3B;
    if (strcmp(s, "turbo_4b") == 0)   return TQ_TYPE_TURBO_4B;
    if (strcmp(s, "turbo_kv_3b") == 0) return TQ_TYPE_TURBO_KV_3B;
    if (strcmp(s, "turbo_kv_4b") == 0) return TQ_TYPE_TURBO_KV_4B;
    if (strcmp(s, "turbo_kv_1b") == 0) return TQ_TYPE_TURBO_KV_1B;
    return fallback;
}

/* ============================================================
 * HTTP response helpers
 * ============================================================ */

static int send_all(int fd, const char* buf, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = write(fd, buf + sent, len - sent);
        if (n <= 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        sent += (size_t)n;
    }
    return 0;
}

static int send_response(int fd, int status, const char* status_text,
                         const char* content_type, const char* body, size_t body_len) {
    char header[1024];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text, content_type, body_len);

    if (send_all(fd, header, (size_t)hlen) < 0) return -1;
    if (body_len > 0 && send_all(fd, body, body_len) < 0) return -1;
    return 0;
}

static int send_json(int fd, int status, const char* status_text, const char* json) {
    return send_response(fd, status, status_text,
                         "application/json", json, strlen(json));
}

static int send_sse_headers(int fd) {
    const char* header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";
    return send_all(fd, header, strlen(header));
}

static int send_sse_event(int fd, const char* data) {
    char buf[SSE_CHUNK_SIZE + 64];
    int len = snprintf(buf, sizeof(buf), "data: %s\n\n", data);
    return send_all(fd, buf, (size_t)len);
}

/* ============================================================
 * JSON escaping utility
 * ============================================================ */

static size_t json_escape(const char* src, char* dst, size_t dst_size) {
    size_t w = 0;
    for (const char* p = src; *p && w < dst_size - 6; p++) {
        switch (*p) {
            case '"':  dst[w++] = '\\'; dst[w++] = '"';  break;
            case '\\': dst[w++] = '\\'; dst[w++] = '\\'; break;
            case '\n': dst[w++] = '\\'; dst[w++] = 'n';  break;
            case '\r': dst[w++] = '\\'; dst[w++] = 'r';  break;
            case '\t': dst[w++] = '\\'; dst[w++] = 't';  break;
            case '\b': dst[w++] = '\\'; dst[w++] = 'b';  break;
            case '\f': dst[w++] = '\\'; dst[w++] = 'f';  break;
            default:
                if ((unsigned char)*p < 0x20) {
                    w += (size_t)snprintf(dst + w, dst_size - w, "\\u%04x", (unsigned char)*p);
                } else {
                    dst[w++] = *p;
                }
        }
    }
    dst[w] = '\0';
    return w;
}

/* ============================================================
 * Generate a pseudo-unique ID
 * ============================================================ */

static void generate_id(char* buf, size_t size) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    snprintf(buf, size, "chatcmpl-%lx%06lx",
             (unsigned long)ts.tv_sec, (unsigned long)(ts.tv_nsec / 1000));
}

/* ============================================================
 * Streaming token callback context
 * ============================================================ */

typedef struct {
    int         fd;
    char        completion_id[64];
    const char* model_id;
    int         token_count;
    bool        error;
} sse_ctx_t;

static void sse_token_callback(const char* text, void* user_data) {
    sse_ctx_t* ctx = (sse_ctx_t*)user_data;
    if (ctx->error) return;

    char escaped[4096];
    json_escape(text, escaped, sizeof(escaped));

    char chunk[SSE_CHUNK_SIZE];
    snprintf(chunk, sizeof(chunk),
        "{"
            "\"id\":\"%s\","
            "\"object\":\"chat.completion.chunk\","
            "\"created\":%ld,"
            "\"model\":\"%s\","
            "\"choices\":[{"
                "\"index\":0,"
                "\"delta\":{\"content\":\"%s\"},"
                "\"finish_reason\":null"
            "}]"
        "}",
        ctx->completion_id,
        (long)time(NULL),
        ctx->model_id,
        escaped);

    if (send_sse_event(ctx->fd, chunk) < 0) {
        ctx->error = true;
    }
    ctx->token_count++;
}

/* ============================================================
 * Non-streaming collection callback
 * ============================================================ */

typedef struct {
    char*  buf;
    size_t len;
    size_t cap;
    int    token_count;
} collect_ctx_t;

static void collect_token_callback(const char* text, void* user_data) {
    collect_ctx_t* ctx = (collect_ctx_t*)user_data;
    size_t tlen = strlen(text);
    if (ctx->len + tlen >= ctx->cap) {
        size_t new_cap = (ctx->cap == 0) ? 4096 : ctx->cap * 2;
        while (new_cap <= ctx->len + tlen) new_cap *= 2;
        char* new_buf = (char*)realloc(ctx->buf, new_cap);
        if (!new_buf) return;
        ctx->buf = new_buf;
        ctx->cap = new_cap;
    }
    memcpy(ctx->buf + ctx->len, text, tlen);
    ctx->len += tlen;
    ctx->buf[ctx->len] = '\0';
    ctx->token_count++;
}

/* ============================================================
 * Request handler: POST /v1/chat/completions
 * ============================================================ */

static void handle_chat_completions(tq_server_t* server, int fd, const char* body) {
    chat_request_t req;
    if (parse_chat_request(body, &req) < 0) {
        send_json(fd, 400, "Bad Request",
            "{\"error\":{\"message\":\"Invalid request body\","
            "\"type\":\"invalid_request_error\",\"code\":\"bad_request\"}}");
        return;
    }

    LOG_INFO("Chat request: model=%s, stream=%s, max_tokens=%d, messages=%d",
             req.model, req.stream ? "true" : "false", req.max_tokens, req.n_messages);

    /* Resolve inference parameters */
    tq_type kv_type = server->config.kv_type;
    if (req.kv_type_str[0]) {
        kv_type = parse_kv_type_str(req.kv_type_str, kv_type);
    }

    int vq_bits = server->config.value_quant_bits;
    if (req.value_quant_bits >= 0) {
        vq_bits = req.value_quant_bits;
    }

    int n_threads = server->config.n_threads;
    if (req.n_threads > 0) {
        n_threads = req.n_threads;
    }

    /* Build generation config */
    tq_gen_config_t gen_cfg = tq_default_gen_config();
    gen_cfg.temperature = req.temperature;
    gen_cfg.top_p = req.top_p;
    gen_cfg.max_tokens = req.max_tokens;
    gen_cfg.kv_type = kv_type;
    gen_cfg.value_quant_bits = vq_bits;
    gen_cfg.n_threads = n_threads;
    gen_cfg.delta_kv = (req.delta_kv || server->config.delta_kv) ? 1 : 0;

    const char* model_id = server->config.model_id ? server->config.model_id : "quant.cpp";
    char completion_id[64];
    generate_id(completion_id, sizeof(completion_id));

    /* Serialize inference (one request at a time) */
    pthread_mutex_lock(&server->inference_mutex);

    if (req.stream) {
        /* --- Streaming (SSE) --- */
        if (send_sse_headers(fd) < 0) {
            pthread_mutex_unlock(&server->inference_mutex);
            free_chat_request(&req);
            return;
        }

        /* Send initial role delta */
        char role_chunk[SSE_CHUNK_SIZE];
        snprintf(role_chunk, sizeof(role_chunk),
            "{"
                "\"id\":\"%s\","
                "\"object\":\"chat.completion.chunk\","
                "\"created\":%ld,"
                "\"model\":\"%s\","
                "\"choices\":[{"
                    "\"index\":0,"
                    "\"delta\":{\"role\":\"assistant\",\"content\":\"\"},"
                    "\"finish_reason\":null"
                "}]"
            "}",
            completion_id, (long)time(NULL), model_id);
        send_sse_event(fd, role_chunk);

        /* Generate with streaming callback */
        sse_ctx_t sse_ctx = {
            .fd = fd,
            .model_id = model_id,
            .token_count = 0,
            .error = false,
        };
        strncpy(sse_ctx.completion_id, completion_id, sizeof(sse_ctx.completion_id) - 1);

        gen_cfg.on_token = sse_token_callback;
        gen_cfg.user_data = &sse_ctx;

        char output[1]; /* writes via callback, output buffer unused */
        /* Use tq_generate_continue with persistent KV state for chat reuse:
         * matches the longest common prefix of req.prompt against
         * server->cached_tokens, prefills only the suffix. Turns chat
         * latency from O(history^2) into O(new_tokens). */
        if (!server->kv_state) {
            server->kv_state = tq_create_state_ex(
                &server->config.model->config, gen_cfg.kv_type, gen_cfg.value_quant_bits);
        }
        tq_generate_continue(server->config.model, server->config.tokenizer,
                              server->kv_state, req.prompt, &gen_cfg,
                              &server->cached_tokens, &server->n_cached,
                              &server->cached_capacity,
                              output, sizeof(output));

        /* Send final chunk with finish_reason */
        char final_chunk[SSE_CHUNK_SIZE];
        snprintf(final_chunk, sizeof(final_chunk),
            "{"
                "\"id\":\"%s\","
                "\"object\":\"chat.completion.chunk\","
                "\"created\":%ld,"
                "\"model\":\"%s\","
                "\"choices\":[{"
                    "\"index\":0,"
                    "\"delta\":{},"
                    "\"finish_reason\":\"stop\""
                "}]"
            "}",
            completion_id, (long)time(NULL), model_id);
        send_sse_event(fd, final_chunk);
        send_sse_event(fd, "[DONE]");

        LOG_INFO("Streaming complete: %d tokens", sse_ctx.token_count);

    } else {
        /* --- Non-streaming --- */
        collect_ctx_t collect = { .buf = NULL, .len = 0, .cap = 0, .token_count = 0 };

        gen_cfg.on_token = collect_token_callback;
        gen_cfg.user_data = &collect;

        char output[1];
        if (!server->kv_state) {
            server->kv_state = tq_create_state_ex(
                &server->config.model->config, gen_cfg.kv_type, gen_cfg.value_quant_bits);
        }
        tq_generate_continue(server->config.model, server->config.tokenizer,
                              server->kv_state, req.prompt, &gen_cfg,
                              &server->cached_tokens, &server->n_cached,
                              &server->cached_capacity,
                              output, sizeof(output));

        const char* content = collect.buf ? collect.buf : "";

        /* Build JSON response */
        size_t escaped_cap = (collect.len + 1) * 2 + 64;
        char* escaped = (char*)malloc(escaped_cap);
        if (!escaped) {
            free(collect.buf);
            pthread_mutex_unlock(&server->inference_mutex);
            free_chat_request(&req);
            send_json(fd, 500, "Internal Server Error",
                "{\"error\":{\"message\":\"Out of memory\","
                "\"type\":\"server_error\",\"code\":\"oom\"}}");
            return;
        }
        json_escape(content, escaped, escaped_cap);

        size_t resp_cap = escaped_cap + 1024;
        char* resp = (char*)malloc(resp_cap);
        if (!resp) {
            free(escaped);
            free(collect.buf);
            pthread_mutex_unlock(&server->inference_mutex);
            free_chat_request(&req);
            send_json(fd, 500, "Internal Server Error",
                "{\"error\":{\"message\":\"Out of memory\","
                "\"type\":\"server_error\",\"code\":\"oom\"}}");
            return;
        }

        /* Rough token estimate: prompt tokens ~ strlen/4, completion tokens counted */
        int prompt_tokens = (int)(strlen(req.prompt) / 4);
        int completion_tokens = collect.token_count;

        snprintf(resp, resp_cap,
            "{"
                "\"id\":\"%s\","
                "\"object\":\"chat.completion\","
                "\"created\":%ld,"
                "\"model\":\"%s\","
                "\"choices\":[{"
                    "\"index\":0,"
                    "\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
                    "\"finish_reason\":\"stop\""
                "}],"
                "\"usage\":{"
                    "\"prompt_tokens\":%d,"
                    "\"completion_tokens\":%d,"
                    "\"total_tokens\":%d"
                "}"
            "}",
            completion_id,
            (long)time(NULL),
            model_id,
            escaped,
            prompt_tokens,
            completion_tokens,
            prompt_tokens + completion_tokens);

        send_json(fd, 200, "OK", resp);

        LOG_INFO("Completion: %d tokens generated", completion_tokens);

        free(resp);
        free(escaped);
        free(collect.buf);
    }

    pthread_mutex_unlock(&server->inference_mutex);
    free_chat_request(&req);
}

/* ============================================================
 * Request handler: GET /v1/models
 * ============================================================ */

static void handle_models(tq_server_t* server, int fd) {
    const char* model_id = server->config.model_id ? server->config.model_id : "quant.cpp";

    char resp[2048];
    snprintf(resp, sizeof(resp),
        "{"
            "\"object\":\"list\","
            "\"data\":[{"
                "\"id\":\"%s\","
                "\"object\":\"model\","
                "\"created\":%ld,"
                "\"owned_by\":\"quant.cpp\""
            "}]"
        "}",
        model_id, (long)time(NULL));

    send_json(fd, 200, "OK", resp);
}

/* ============================================================
 * Request handler: GET /health
 * ============================================================ */

static void handle_health(tq_server_t* server, int fd) {
    (void)server;
    send_json(fd, 200, "OK",
        "{\"status\":\"ok\",\"version\":\"" TQ_VERSION_STRING "\"}");
}

/* ============================================================
 * HTTP/1.1 request parser — minimal, sufficient for API use
 * ============================================================ */

typedef struct {
    char method[16];
    char path[256];
    char content_type[128];
    int  content_length;
    char* body;           /* points into the read buffer */
    int  body_len;
} http_request_t;

/* Read a full HTTP request from a socket.
 * Returns 0 on success, -1 on error/disconnect. */
static int read_http_request(int fd, char* buf, int buf_size, http_request_t* req) {
    memset(req, 0, sizeof(*req));

    /* Read until we find \r\n\r\n (end of headers) */
    int total = 0;
    char* header_end = NULL;

    while (total < buf_size - 1) {
        ssize_t n = read(fd, buf + total, (size_t)(buf_size - 1 - total));
        if (n <= 0) {
            if (n == 0) return -1; /* connection closed */
            if (errno == EINTR) continue;
            return -1;
        }
        total += (int)n;
        buf[total] = '\0';

        header_end = strstr(buf, "\r\n\r\n");
        if (header_end) break;
    }

    if (!header_end) return -1; /* headers too large */

    /* Parse request line: METHOD PATH HTTP/1.x */
    char* line_end = strstr(buf, "\r\n");
    if (!line_end) return -1;

    /* Extract method */
    char* sp1 = memchr(buf, ' ', (size_t)(line_end - buf));
    if (!sp1) return -1;
    int mlen = (int)(sp1 - buf);
    if (mlen >= (int)sizeof(req->method)) mlen = (int)sizeof(req->method) - 1;
    memcpy(req->method, buf, (size_t)mlen);
    req->method[mlen] = '\0';

    /* Extract path */
    sp1++;
    char* sp2 = memchr(sp1, ' ', (size_t)(line_end - sp1));
    if (!sp2) sp2 = line_end;
    int plen = (int)(sp2 - sp1);
    if (plen >= (int)sizeof(req->path)) plen = (int)sizeof(req->path) - 1;
    memcpy(req->path, sp1, (size_t)plen);
    req->path[plen] = '\0';

    /* Strip query string */
    char* qmark = strchr(req->path, '?');
    if (qmark) *qmark = '\0';

    /* Parse headers */
    const char* hp = line_end + 2; /* past first \r\n */
    while (hp < header_end) {
        const char* next = strstr(hp, "\r\n");
        if (!next) break;

        /* Content-Length — use strtol to avoid UB on overflow */
        if (strncasecmp(hp, "Content-Length:", 15) == 0) {
            long cl = strtol(hp + 15, NULL, 10);
            req->content_length = (cl > 0 && cl <= MAX_BODY_SIZE) ? (int)cl : 0;
        }
        /* Content-Type */
        if (strncasecmp(hp, "Content-Type:", 13) == 0) {
            const char* val = hp + 13;
            while (*val == ' ') val++;
            int clen = (int)(next - val);
            if (clen >= (int)sizeof(req->content_type))
                clen = (int)sizeof(req->content_type) - 1;
            memcpy(req->content_type, val, (size_t)clen);
            req->content_type[clen] = '\0';
        }

        hp = next + 2;
    }

    /* Read body if Content-Length > 0 */
    char* body_start = header_end + 4;
    int headers_size = (int)(body_start - buf);
    int body_received = total - headers_size;

    if (req->content_length > 0) {
        if (req->content_length > MAX_BODY_SIZE) return -1; /* body too large */

        /* May need to read more body data */
        while (body_received < req->content_length &&
               total < buf_size - 1) {
            ssize_t n = read(fd, buf + total,
                            (size_t)(buf_size - 1 - total));
            if (n <= 0) {
                if (errno == EINTR) continue;
                break;
            }
            total += (int)n;
            body_received += (int)n;
            buf[total] = '\0';
        }
    }

    req->body = body_start;
    req->body_len = body_received;
    return 0;
}

/* ============================================================
 * Connection handler (runs in thread pool or inline)
 * ============================================================ */

typedef struct {
    tq_server_t* server;
    int          fd;
} conn_ctx_t;

static void* handle_connection(void* arg) {
    conn_ctx_t* ctx = (conn_ctx_t*)arg;
    int fd = ctx->fd;
    tq_server_t* server = ctx->server;
    free(ctx);

    /* Set socket read/write timeout to prevent slow-loris attacks */
    struct timeval sock_tv = { .tv_sec = SOCKET_TIMEOUT_SEC, .tv_usec = 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &sock_tv, sizeof(sock_tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &sock_tv, sizeof(sock_tv));

    char* buf = (char*)malloc(HTTP_BUF_SIZE + MAX_BODY_SIZE);
    if (!buf) {
        close(fd);
        return NULL;
    }

    http_request_t req;
    if (read_http_request(fd, buf, HTTP_BUF_SIZE + MAX_BODY_SIZE, &req) < 0) {
        free(buf);
        close(fd);
        return NULL;
    }

    LOG_INFO("%s %s (body: %d bytes)", req.method, req.path, req.body_len);

    /* Route request */
    if (strcmp(req.method, "OPTIONS") == 0) {
        /* CORS preflight */
        send_response(fd, 204, "No Content", "text/plain", "", 0);
    }
    else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/health") == 0) {
        handle_health(server, fd);
    }
    else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/v1/models") == 0) {
        handle_models(server, fd);
    }
    else if (strcmp(req.method, "POST") == 0 &&
             strcmp(req.path, "/v1/chat/completions") == 0) {
        if (req.body_len <= 0) {
            send_json(fd, 400, "Bad Request",
                "{\"error\":{\"message\":\"Empty request body\","
                "\"type\":\"invalid_request_error\",\"code\":\"bad_request\"}}");
        } else {
            handle_chat_completions(server, fd, req.body);
        }
    }
    else {
        send_json(fd, 404, "Not Found",
            "{\"error\":{\"message\":\"Unknown endpoint\","
            "\"type\":\"invalid_request_error\",\"code\":\"not_found\"}}");
    }

    close(fd);
    free(buf);
    atomic_fetch_sub(&server->active_connections, 1);
    return NULL;
}

/* ============================================================
 * Signal handler
 * ============================================================ */

static void signal_handler(int sig) {
    (void)sig;
    if (g_server) {
        atomic_store(&g_server->running, 0);
    }
}

/* ============================================================
 * Public API
 * ============================================================ */

tq_server_config_t tq_server_default_config(void) {
    tq_server_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.port = 8080;
    cfg.max_connections = 8;
    cfg.host = "0.0.0.0";
    cfg.model = NULL;
    cfg.tokenizer = NULL;
    cfg.model_id = "quant.cpp";
    cfg.kv_type = TQ_TYPE_UNIFORM_4B;
    cfg.value_quant_bits = 0;
    cfg.n_threads = 4;
    cfg.delta_kv = 0;
    return cfg;
}

int tq_server_start(tq_server_t** out, const tq_server_config_t* config) {
    if (!config || !config->model || !config->tokenizer) {
        LOG_ERROR("Server requires model and tokenizer to be loaded");
        return -1;
    }

    tq_server_t* server = (tq_server_t*)calloc(1, sizeof(tq_server_t));
    if (!server) return -1;

    server->config = *config;
    atomic_store(&server->running, 1);
    atomic_store(&server->active_connections, 0);
    pthread_mutex_init(&server->inference_mutex, NULL);

    /* Install signal handlers */
    g_server = server;
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGPIPE, SIG_IGN); /* ignore broken pipe from disconnected clients */

    /* Create listening socket */
    server->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server->listen_fd < 0) {
        LOG_ERROR("Failed to create socket: %s", strerror(errno));
        free(server);
        return -1;
    }

    int opt = 1;
    setsockopt(server->listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)config->port);

    if (config->host && strcmp(config->host, "0.0.0.0") != 0) {
        inet_pton(AF_INET, config->host, &addr.sin_addr);
    } else {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    }

    if (bind(server->listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        LOG_ERROR("Failed to bind to %s:%d: %s",
                  config->host, config->port, strerror(errno));
        close(server->listen_fd);
        free(server);
        return -1;
    }

    if (listen(server->listen_fd, config->max_connections) < 0) {
        LOG_ERROR("Failed to listen: %s", strerror(errno));
        close(server->listen_fd);
        free(server);
        return -1;
    }

    LOG_INFO("quant.cpp server listening on %s:%d", config->host, config->port);
    LOG_INFO("Model: %s", config->model_id);
    LOG_INFO("KV type: %s, V quant: %d-bit, Threads: %d",
             tq_type_name(config->kv_type),
             config->value_quant_bits,
             config->n_threads);
    LOG_INFO("Endpoints:");
    LOG_INFO("  POST /v1/chat/completions");
    LOG_INFO("  GET  /v1/models");
    LOG_INFO("  GET  /health");

    *out = server;

    /* Accept loop */
    while (atomic_load(&server->running)) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        /* Use a timeout on accept so we can check the running flag */
        struct timeval tv = { .tv_sec = 1, .tv_usec = 0 };
        setsockopt(server->listen_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        int client_fd = accept(server->listen_fd,
                               (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                continue; /* timeout or signal — check running flag */
            }
            if (atomic_load(&server->running)) {
                LOG_ERROR("accept() failed: %s", strerror(errno));
            }
            continue;
        }

        /* Enforce connection limit to prevent resource exhaustion */
        if (atomic_load(&server->active_connections) >= MAX_ACTIVE_CONNS) {
            LOG_ERROR("Connection limit reached (%d), rejecting", MAX_ACTIVE_CONNS);
            close(client_fd);
            continue;
        }
        atomic_fetch_add(&server->active_connections, 1);

        /* Spawn a thread for this connection */
        conn_ctx_t* conn = (conn_ctx_t*)malloc(sizeof(conn_ctx_t));
        if (!conn) {
            close(client_fd);
            atomic_fetch_sub(&server->active_connections, 1);
            continue;
        }
        conn->server = server;
        conn->fd = client_fd;

        pthread_t tid;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        if (pthread_create(&tid, &attr, handle_connection, conn) != 0) {
            LOG_ERROR("Failed to create thread: %s", strerror(errno));
            close(client_fd);
            free(conn);
            atomic_fetch_sub(&server->active_connections, 1);
        }
        pthread_attr_destroy(&attr);
    }

    LOG_INFO("Server shutting down...");
    close(server->listen_fd);
    return 0;
}

void tq_server_stop(tq_server_t* server) {
    if (server) {
        atomic_store(&server->running, 0);
    }
}

void tq_server_free(tq_server_t* server) {
    if (!server) return;
    pthread_mutex_destroy(&server->inference_mutex);
    if (server->kv_state) tq_free_state(server->kv_state);
    if (server->cached_tokens) free(server->cached_tokens);
    if (g_server == server) g_server = NULL;
    free(server);
}

/* ============================================================
 * Standalone main — builds as `quant-server` executable
 * ============================================================ */

#ifdef TQ_SERVER_MAIN

static void print_usage(const char* prog) {
    fprintf(stderr,
        "quant-server -- OpenAI-compatible LLM inference server\n"
        "\n"
        "Usage: %s <model> [options]\n"
        "\n"
        "Options:\n"
        "  -t <tokenizer>      Tokenizer file (auto-detected from GGUF if not set)\n"
        "  -p <port>           Listen port (default: 8080)\n"
        "  -H <host>           Bind address (default: 0.0.0.0)\n"
        "  --model-id <name>   Model name for /v1/models (default: filename)\n"
        "  -j <threads>        Threads per inference (default: 4)\n"
        "  -k <kv_type>        KV cache type: fp32, uniform_4b, polar_3b, turbo_3b, etc.\n"
        "  -v <vq>             Value cache quant: q4, q2, fp16 (default)\n"
        "  --delta              Enable delta KV compression\n"
        "  --help              Show this help\n"
        "\n"
        "Example:\n"
        "  %s model.gguf -p 8080 -j 8 -k turbo_kv_3b\n"
        "  curl http://localhost:8080/v1/chat/completions \\\n"
        "    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'\n",
        prog, prog);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* model_path = NULL;
    const char* tokenizer_path = NULL;
    const char* model_id = NULL;
    const char* host = "0.0.0.0";
    int port = 8080;
    int n_threads = 4;
    tq_type kv_type = TQ_TYPE_UNIFORM_4B;
    int value_quant_bits = 0;
    int delta_kv = 0;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-' && !model_path) {
            model_path = argv[i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-H") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "--model-id") == 0 && i + 1 < argc) {
            model_id = argv[++i];
        } else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            kv_type = parse_kv_type_str(argv[++i], TQ_TYPE_UNIFORM_4B);
        } else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "q4") == 0) value_quant_bits = 4;
            else if (strcmp(argv[i], "q2") == 0) value_quant_bits = 2;
            else value_quant_bits = 0;
        } else if (strcmp(argv[i], "--delta") == 0 || strcmp(argv[i], "-D") == 0) {
            delta_kv = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
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
    LOG_INFO("Loading model: %s", model_path);
    tq_model_t* model = tq_load_model(model_path);
    if (!model) {
        LOG_ERROR("Failed to load model: %s", model_path);
        return 1;
    }
    LOG_INFO("Model loaded: %d layers, %d heads, dim=%d, vocab=%d",
             model->config.n_layers, model->config.n_heads,
             model->config.hidden_dim, model->config.vocab_size);

    /* Load tokenizer */
    tq_tokenizer_t* tokenizer = NULL;
    if (tokenizer_path) {
        tokenizer = tq_load_tokenizer(tokenizer_path);
    } else if (model->gguf_ctx) {
        tokenizer = tq_load_tokenizer_from_gguf(model->gguf_ctx);
    }
    if (!tokenizer) {
        LOG_ERROR("Failed to load tokenizer. Use -t <path> or provide a GGUF model.");
        tq_free_model(model);
        return 1;
    }

    /* Derive model_id from filename if not specified */
    char auto_model_id[256];
    if (!model_id) {
        const char* slash = strrchr(model_path, '/');
        const char* name = slash ? slash + 1 : model_path;
        snprintf(auto_model_id, sizeof(auto_model_id), "quant.cpp/%s", name);
        model_id = auto_model_id;
    }

    /* Configure and start server */
    tq_server_config_t cfg = tq_server_default_config();
    cfg.port = port;
    cfg.host = host;
    cfg.model = model;
    cfg.tokenizer = tokenizer;
    cfg.model_id = model_id;
    cfg.kv_type = kv_type;
    cfg.value_quant_bits = value_quant_bits;
    cfg.n_threads = n_threads;
    cfg.delta_kv = delta_kv;

    tq_server_t* server = NULL;
    int rc = tq_server_start(&server, &cfg);

    /* Cleanup */
    tq_server_free(server);
    tq_free_tokenizer(tokenizer);
    tq_free_model(model);

    return rc;
}

#endif /* TQ_SERVER_MAIN */
