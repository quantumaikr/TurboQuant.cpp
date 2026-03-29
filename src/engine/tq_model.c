/**
 * tq_model.c — Safetensors model loader
 *
 * Reads safetensors format:
 *   - First 8 bytes: header size (uint64_t little-endian)
 *   - Next N bytes: JSON header with tensor metadata
 *   - Remaining bytes: raw tensor data
 *
 * Implements a minimal JSON parser (no external deps).
 * Uses mmap for zero-copy tensor access on supported platforms.
 */

#include "turboquant/tq_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

/* ============================================================
 * Minimal JSON parser for safetensors header
 *
 * Safetensors JSON looks like:
 * {
 *   "model.layers.0.self_attn.q_proj.weight": {
 *     "dtype": "F32",
 *     "shape": [4096, 4096],
 *     "data_offsets": [0, 67108864]
 *   },
 *   ...
 * }
 * ============================================================ */

/* Maximum number of tensors we expect */
#define MAX_TENSORS 2048
#define MAX_NAME_LEN 256
#define MAX_DIMS 4

typedef struct {
    char name[MAX_NAME_LEN];
    char dtype[16];
    int64_t shape[MAX_DIMS];
    int n_dims;
    int64_t data_start;
    int64_t data_end;
} tensor_info_t;

/* Skip whitespace in JSON string */
static const char* skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

/* Parse a JSON string value, return pointer past closing quote */
static const char* parse_string(const char* p, char* out, int max_len) {
    if (*p != '"') return NULL;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < max_len - 1) {
        if (*p == '\\') {
            p++;
            if (!*p) return NULL;
        }
        out[i++] = *p++;
    }
    out[i] = '\0';
    if (*p == '"') p++;
    return p;
}

/* Parse a JSON integer */
static const char* parse_int64(const char* p, int64_t* out) {
    *out = 0;
    int neg = 0;
    if (*p == '-') { neg = 1; p++; }
    while (*p >= '0' && *p <= '9') {
        *out = *out * 10 + (*p - '0');
        p++;
    }
    if (neg) *out = -*out;
    return p;
}

/* Parse tensor metadata from the safetensors JSON header */
static int parse_safetensors_header(const char* json, int64_t json_len,
                                    tensor_info_t* tensors, int max_tensors) {
    int n_tensors = 0;
    const char* p = json;
    const char* end = json + json_len;

    p = skip_ws(p);
    if (*p != '{') return -1;
    p++;

    while (p < end && n_tensors < max_tensors) {
        p = skip_ws(p);
        if (*p == '}') break;
        if (*p == ',') { p++; p = skip_ws(p); }
        if (*p == '}') break;

        /* Parse tensor name */
        tensor_info_t* t = &tensors[n_tensors];
        memset(t, 0, sizeof(*t));
        p = parse_string(p, t->name, MAX_NAME_LEN);
        if (!p) return -1;

        p = skip_ws(p);
        if (*p != ':') return -1;
        p++;
        p = skip_ws(p);

        /* Skip __metadata__ entries */
        if (strcmp(t->name, "__metadata__") == 0) {
            /* Skip the value object */
            int depth = 0;
            while (p < end) {
                if (*p == '{') depth++;
                else if (*p == '}') {
                    depth--;
                    if (depth == 0) { p++; break; }
                }
                p++;
            }
            continue;
        }

        /* Parse tensor metadata object */
        if (*p != '{') return -1;
        p++;

        while (p < end) {
            p = skip_ws(p);
            if (*p == '}') { p++; break; }
            if (*p == ',') { p++; p = skip_ws(p); }

            char key[64];
            p = parse_string(p, key, 64);
            if (!p) return -1;
            p = skip_ws(p);
            if (*p != ':') return -1;
            p++;
            p = skip_ws(p);

            if (strcmp(key, "dtype") == 0) {
                p = parse_string(p, t->dtype, 16);
                if (!p) return -1;
            } else if (strcmp(key, "shape") == 0) {
                /* Parse array of ints */
                if (*p != '[') return -1;
                p++;
                t->n_dims = 0;
                while (*p != ']' && t->n_dims < MAX_DIMS) {
                    p = skip_ws(p);
                    if (*p == ',') { p++; p = skip_ws(p); }
                    if (*p == ']') break;
                    p = parse_int64(p, &t->shape[t->n_dims]);
                    t->n_dims++;
                    p = skip_ws(p);
                }
                if (*p == ']') p++;
            } else if (strcmp(key, "data_offsets") == 0) {
                /* Parse [start, end] */
                if (*p != '[') return -1;
                p++;
                p = skip_ws(p);
                p = parse_int64(p, &t->data_start);
                p = skip_ws(p);
                if (*p == ',') p++;
                p = skip_ws(p);
                p = parse_int64(p, &t->data_end);
                p = skip_ws(p);
                if (*p == ']') p++;
            } else {
                /* Skip unknown value */
                if (*p == '"') {
                    char dummy[256];
                    p = parse_string(p, dummy, 256);
                } else if (*p == '[') {
                    int depth = 1;
                    p++;
                    while (p < end && depth > 0) {
                        if (*p == '[') depth++;
                        else if (*p == ']') depth--;
                        p++;
                    }
                } else if (*p == '{') {
                    int depth = 1;
                    p++;
                    while (p < end && depth > 0) {
                        if (*p == '{') depth++;
                        else if (*p == '}') depth--;
                        p++;
                    }
                } else {
                    /* number/bool/null */
                    while (p < end && *p != ',' && *p != '}') p++;
                }
            }
        }

        n_tensors++;
    }

    return n_tensors;
}

/* ============================================================
 * Find a tensor by name in the parsed tensor list
 * ============================================================ */
static tensor_info_t* find_tensor(tensor_info_t* tensors, int n,
                                   const char* name) {
    for (int i = 0; i < n; i++) {
        if (strcmp(tensors[i].name, name) == 0) {
            return &tensors[i];
        }
    }
    return NULL;
}

/* ============================================================
 * Map tensor data pointer from mmap'd file
 * ============================================================ */
static float* map_tensor(void* data_base, tensor_info_t* t) {
    if (!t) return NULL;
    return (float*)((char*)data_base + t->data_start);
}

/* ============================================================
 * Load model from safetensors file
 * ============================================================ */
tq_model_t* tq_load_model(const char* path) {
    if (!path) return NULL;

#ifdef _WIN32
    /* Windows file mapping */
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "tq_load_model: cannot open '%s'\n", path);
        return NULL;
    }
    LARGE_INTEGER fileSize;
    GetFileSizeEx(hFile, &fileSize);
    size_t file_size = (size_t)fileSize.QuadPart;

    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) {
        CloseHandle(hFile);
        return NULL;
    }
    void* mmap_data = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    CloseHandle(hFile);
    if (!mmap_data) return NULL;
#else
    /* POSIX mmap */
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "tq_load_model: cannot open '%s'\n", path);
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return NULL;
    }
    size_t file_size = (size_t)st.st_size;

    void* mmap_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mmap_data == MAP_FAILED) {
        fprintf(stderr, "tq_load_model: mmap failed for '%s'\n", path);
        return NULL;
    }
#endif

    /* Parse safetensors header */
    if (file_size < 8) {
        fprintf(stderr, "tq_load_model: file too small\n");
        goto fail;
    }

    uint64_t header_size = 0;
    memcpy(&header_size, mmap_data, 8); /* little-endian */

    if (8 + header_size > file_size) {
        fprintf(stderr, "tq_load_model: invalid header size\n");
        goto fail;
    }

    const char* json = (const char*)mmap_data + 8;
    void* data_base = (char*)mmap_data + 8 + header_size;

    /* Parse tensors */
    tensor_info_t* tensors = (tensor_info_t*)calloc(MAX_TENSORS, sizeof(tensor_info_t));
    if (!tensors) goto fail;

    int n_tensors = parse_safetensors_header(json, (int64_t)header_size,
                                              tensors, MAX_TENSORS);
    if (n_tensors < 0) {
        fprintf(stderr, "tq_load_model: JSON parse error\n");
        free(tensors);
        goto fail;
    }

    /* Allocate model */
    tq_model_t* model = (tq_model_t*)calloc(1, sizeof(tq_model_t));
    if (!model) {
        free(tensors);
        goto fail;
    }
    model->_mmap_data = mmap_data;
    model->_mmap_size = file_size;

    /* Detect model config from tensor shapes.
     * Look for embedding table to get vocab_size and hidden_dim.
     * Look for layer 0 weights to get n_heads, n_kv_heads, intermediate_dim. */

    /* Try common naming conventions */
    tensor_info_t* embed = find_tensor(tensors, n_tensors, "model.embed_tokens.weight");
    if (!embed) embed = find_tensor(tensors, n_tensors, "tok_embeddings.weight");
    if (!embed) embed = find_tensor(tensors, n_tensors, "transformer.wte.weight");

    if (!embed) {
        fprintf(stderr, "tq_load_model: cannot find embedding tensor\n");
        free(model);
        free(tensors);
        goto fail;
    }

    model->config.vocab_size = (int)embed->shape[0];
    model->config.hidden_dim = (int)embed->shape[1];
    model->token_embedding = map_tensor(data_base, embed);

    /* Detect n_layers by counting layer indices */
    int max_layer = -1;
    for (int i = 0; i < n_tensors; i++) {
        int layer_idx = -1;
        if (sscanf(tensors[i].name, "model.layers.%d.", &layer_idx) == 1) {
            if (layer_idx > max_layer) max_layer = layer_idx;
        }
    }
    model->config.n_layers = max_layer + 1;

    /* Detect head dimensions from Q projection shape */
    char name_buf[MAX_NAME_LEN];
    snprintf(name_buf, sizeof(name_buf),
             "model.layers.0.self_attn.q_proj.weight");
    tensor_info_t* wq0 = find_tensor(tensors, n_tensors, name_buf);

    snprintf(name_buf, sizeof(name_buf),
             "model.layers.0.self_attn.k_proj.weight");
    tensor_info_t* wk0 = find_tensor(tensors, n_tensors, name_buf);

    if (wq0 && wk0) {
        int q_out = (int)wq0->shape[0];
        int k_out = (int)wk0->shape[0];
        /* Common head_dim values: 64, 128 */
        /* Try head_dim = 128, then 64 */
        int head_dim = 128;
        if (q_out % head_dim != 0) head_dim = 64;
        if (q_out % head_dim != 0) head_dim = 96;
        model->config.head_dim = head_dim;
        model->config.n_heads = q_out / head_dim;
        model->config.n_kv_heads = k_out / head_dim;
    } else {
        /* Defaults for small models */
        model->config.head_dim = 64;
        model->config.n_heads = model->config.hidden_dim / 64;
        model->config.n_kv_heads = model->config.n_heads;
    }

    /* Detect intermediate_dim from gate projection */
    snprintf(name_buf, sizeof(name_buf),
             "model.layers.0.mlp.gate_proj.weight");
    tensor_info_t* wg0 = find_tensor(tensors, n_tensors, name_buf);
    if (wg0) {
        model->config.intermediate_dim = (int)wg0->shape[0];
    } else {
        model->config.intermediate_dim = model->config.hidden_dim * 4;
    }

    /* Defaults */
    model->config.max_seq_len = 4096;
    model->config.rope_freq_base = 10000.0f;
    model->config.rms_norm_eps = 1e-5f;

    /* Allocate layer weight pointers */
    int n_layers = model->config.n_layers;
    model->layers = (tq_layer_weights_t*)calloc(n_layers, sizeof(tq_layer_weights_t));
    if (!model->layers) {
        free(model);
        free(tensors);
        goto fail;
    }

    /* Map per-layer weights */
    for (int l = 0; l < n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];

        /* Attention norm */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.input_layernorm.weight", l);
        layer->attn_norm = map_tensor(data_base,
                                       find_tensor(tensors, n_tensors, name_buf));

        /* FFN norm */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.post_attention_layernorm.weight", l);
        layer->ffn_norm = map_tensor(data_base,
                                      find_tensor(tensors, n_tensors, name_buf));

        /* Q, K, V, O projections */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.q_proj.weight", l);
        layer->wq = map_tensor(data_base,
                                find_tensor(tensors, n_tensors, name_buf));

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.k_proj.weight", l);
        layer->wk = map_tensor(data_base,
                                find_tensor(tensors, n_tensors, name_buf));

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.v_proj.weight", l);
        layer->wv = map_tensor(data_base,
                                find_tensor(tensors, n_tensors, name_buf));

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.o_proj.weight", l);
        layer->wo = map_tensor(data_base,
                                find_tensor(tensors, n_tensors, name_buf));

        /* FFN: gate, up, down projections (SwiGLU) */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.gate_proj.weight", l);
        layer->w_gate = map_tensor(data_base,
                                    find_tensor(tensors, n_tensors, name_buf));

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.up_proj.weight", l);
        layer->w_up = map_tensor(data_base,
                                  find_tensor(tensors, n_tensors, name_buf));

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.down_proj.weight", l);
        layer->w_down = map_tensor(data_base,
                                    find_tensor(tensors, n_tensors, name_buf));
    }

    /* Output norm */
    model->output_norm = map_tensor(data_base,
        find_tensor(tensors, n_tensors, "model.norm.weight"));

    /* Output weight — may be tied to embedding */
    tensor_info_t* lm_head = find_tensor(tensors, n_tensors, "lm_head.weight");
    if (lm_head) {
        model->output_weight = map_tensor(data_base, lm_head);
    } else {
        /* Weight tying: reuse embedding */
        model->output_weight = model->token_embedding;
    }

    free(tensors);

    fprintf(stderr, "tq_load_model: loaded %d layers, dim=%d, heads=%d/%d, vocab=%d\n",
            model->config.n_layers, model->config.hidden_dim,
            model->config.n_heads, model->config.n_kv_heads,
            model->config.vocab_size);

    return model;

fail:
#ifdef _WIN32
    if (mmap_data) UnmapViewOfFile(mmap_data);
#else
    if (mmap_data && mmap_data != MAP_FAILED) munmap(mmap_data, file_size);
#endif
    return NULL;
}

/* ============================================================
 * Free model
 * ============================================================ */
void tq_free_model(tq_model_t* model) {
    if (!model) return;

#ifdef _WIN32
    if (model->_mmap_data) UnmapViewOfFile(model->_mmap_data);
#else
    if (model->_mmap_data) munmap(model->_mmap_data, model->_mmap_size);
#endif

    free(model->layers);
    free(model);
}
