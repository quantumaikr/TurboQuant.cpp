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
 *
 * Supported dtypes: F32, F16, BF16 (BF16/F16 are converted to FP32).
 * Supports naming conventions:
 *   - Standard:  model.layers.N.self_attn.q_proj.weight
 *   - Qwen3.5:   model.language_model.layers.N.self_attn.q_proj.weight
 * Supports hybrid architectures (e.g., Qwen3.5 DeltaNet + self_attn).
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
#define MAX_DIMS 8

typedef struct {
    char name[MAX_NAME_LEN];
    char dtype[16];
    int64_t shape[MAX_DIMS];
    int n_dims;
    int64_t data_start;
    int64_t data_end;
} tensor_info_t;

/* ============================================================
 * BF16 / F16 conversion utilities
 * ============================================================ */

static float bf16_to_fp32(uint16_t bf16) {
    uint32_t fp32 = ((uint32_t)bf16) << 16;
    float result;
    memcpy(&result, &fp32, sizeof(float));
    return result;
}

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t expo = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    uint32_t fp32;

    if (expo == 0) {
        if (mant == 0) {
            fp32 = sign; /* +/- zero */
        } else {
            /* Denormalized: convert to normalized FP32 */
            expo = 1;
            while (!(mant & 0x0400)) {
                mant <<= 1;
                expo--;
            }
            mant &= 0x03FF;
            fp32 = sign | ((uint32_t)(expo + 127 - 15) << 23) | ((uint32_t)mant << 13);
        }
    } else if (expo == 31) {
        fp32 = sign | 0x7F800000 | ((uint32_t)mant << 13); /* Inf/NaN */
    } else {
        fp32 = sign | ((uint32_t)(expo + 127 - 15) << 23) | ((uint32_t)mant << 13);
    }

    float result;
    memcpy(&result, &fp32, sizeof(float));
    return result;
}

/* Return the number of bytes per element for a dtype string.
 * Returns 0 for unknown dtypes. */
static int dtype_element_size(const char* dtype) {
    if (strcmp(dtype, "F32") == 0) return 4;
    if (strcmp(dtype, "F16") == 0) return 2;
    if (strcmp(dtype, "BF16") == 0) return 2;
    if (strcmp(dtype, "F64") == 0) return 8;
    if (strcmp(dtype, "I32") == 0) return 4;
    if (strcmp(dtype, "I64") == 0) return 8;
    if (strcmp(dtype, "I16") == 0) return 2;
    if (strcmp(dtype, "I8") == 0) return 1;
    if (strcmp(dtype, "U8") == 0) return 1;
    if (strcmp(dtype, "BOOL") == 0) return 1;
    return 0;
}

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
 * Find a tensor by name in the parsed tensor list.
 * Also tries alternative naming prefixes for compatibility
 * with different model architectures (e.g., Qwen3.5 uses
 * "model.language_model.layers.N." instead of "model.layers.N.").
 * ============================================================ */

/* Known prefix alternatives for tensor name matching */
/* Prefixes to insert after "model." when searching for tensors.
 * E.g., for "model.layers.0.foo", inserting "language_model." yields
 * "model.language_model.layers.0.foo" (Qwen3.5 naming convention). */
static const char* const NAME_PREFIXES[] = {
    "",                    /* identity — try as-is first */
    "language_model.",     /* Qwen3.5 style: model.language_model.layers.N... */
    NULL
};

static tensor_info_t* find_tensor(tensor_info_t* tensors, int n,
                                   const char* name) {
    /* Try exact match first */
    for (int i = 0; i < n; i++) {
        if (strcmp(tensors[i].name, name) == 0) {
            return &tensors[i];
        }
    }
    /* Try alternative prefixes: for a name like "model.layers.0.foo",
     * try "model.language_model.layers.0.foo" etc.
     * The prefix replaces "model." at the start of the name. */
    for (int p = 1; NAME_PREFIXES[p] != NULL; p++) {
        const char* prefix = NAME_PREFIXES[p];
        if (strncmp(name, "model.", 6) == 0) {
            char alt[MAX_NAME_LEN];
            /* Replace "model." with "model." + prefix
             * e.g., "model.layers.0.foo" -> "model.language_model.layers.0.foo" */
            snprintf(alt, sizeof(alt), "model.%s%s", prefix, name + 6);
            for (int i = 0; i < n; i++) {
                if (strcmp(tensors[i].name, alt) == 0) {
                    return &tensors[i];
                }
            }
        }
    }
    return NULL;
}

/* ============================================================
 * Convert tensor data to FP32 if needed (BF16, F16).
 * Returns a newly allocated float buffer on conversion,
 * or NULL if the tensor is already F32 (caller uses mmap).
 * ============================================================ */
static float* convert_tensor_to_fp32(void* data_base, tensor_info_t* t,
                                      float** conv_buf, size_t* conv_used,
                                      size_t conv_capacity) {
    if (!t) return NULL;

    const void* src = (const char*)data_base + t->data_start;
    int64_t data_bytes = t->data_end - t->data_start;
    int elem_size = dtype_element_size(t->dtype);
    if (elem_size == 0) return NULL;
    int64_t n_elements = data_bytes / elem_size;

    if (strcmp(t->dtype, "F32") == 0) {
        /* Zero-copy: return direct pointer into mmap */
        return (float*)((char*)data_base + t->data_start);
    }

    /* Need conversion — write into the pre-allocated conversion buffer */
    size_t needed = (size_t)n_elements * sizeof(float);
    if (*conv_used + needed > conv_capacity) {
        fprintf(stderr, "tq_load_model: conversion buffer overflow (need %zu more bytes)\n",
                needed);
        return NULL;
    }

    float* dst = (float*)((char*)*conv_buf + *conv_used);
    *conv_used += needed;

    if (strcmp(t->dtype, "BF16") == 0) {
        const uint16_t* src16 = (const uint16_t*)src;
        for (int64_t i = 0; i < n_elements; i++) {
            dst[i] = bf16_to_fp32(src16[i]);
        }
    } else if (strcmp(t->dtype, "F16") == 0) {
        const uint16_t* src16 = (const uint16_t*)src;
        for (int64_t i = 0; i < n_elements; i++) {
            dst[i] = fp16_to_fp32(src16[i]);
        }
    } else {
        fprintf(stderr, "tq_load_model: unsupported dtype '%s' for tensor '%s'\n",
                t->dtype, t->name);
        *conv_used -= needed; /* rollback */
        return NULL;
    }

    return dst;
}

/* ============================================================
 * Map or convert tensor data pointer from mmap'd file.
 * Wrapper that handles F32 (zero-copy) and BF16/F16 (convert).
 * ============================================================ */
static float* load_tensor(void* data_base, tensor_info_t* t,
                           float** conv_buf, size_t* conv_used,
                           size_t conv_capacity) {
    if (!t) return NULL;
    return convert_tensor_to_fp32(data_base, t, conv_buf, conv_used, conv_capacity);
}

/* ============================================================
 * Calculate total FP32 bytes needed for all non-F32 tensors
 * ============================================================ */
static size_t calc_conversion_buffer_size(tensor_info_t* tensors, int n_tensors) {
    size_t total = 0;
    for (int i = 0; i < n_tensors; i++) {
        if (strcmp(tensors[i].dtype, "F32") != 0 &&
            strcmp(tensors[i].dtype, "__metadata__") != 0) {
            int64_t data_bytes = tensors[i].data_end - tensors[i].data_start;
            int elem_size = dtype_element_size(tensors[i].dtype);
            if (elem_size > 0) {
                int64_t n_elements = data_bytes / elem_size;
                total += (size_t)n_elements * sizeof(float);
            }
        }
    }
    return total;
}

/* ============================================================
 * Detect layer count, considering alternative naming prefixes.
 * Scans all tensor names for patterns like:
 *   model.layers.N.xxx
 *   model.language_model.layers.N.xxx
 * Returns the max layer index + 1.
 * ============================================================ */
static int detect_n_layers(tensor_info_t* tensors, int n_tensors) {
    int max_layer = -1;
    for (int i = 0; i < n_tensors; i++) {
        int layer_idx = -1;
        const char* name = tensors[i].name;
        /* Try standard prefix */
        if (sscanf(name, "model.layers.%d.", &layer_idx) == 1) {
            if (layer_idx > max_layer) max_layer = layer_idx;
        }
        /* Try Qwen3.5 prefix */
        else if (sscanf(name, "model.language_model.layers.%d.", &layer_idx) == 1) {
            if (layer_idx > max_layer) max_layer = layer_idx;
        }
    }
    return max_layer + 1;
}

/* ============================================================
 * Detect which layers have standard self_attn (vs linear_attn/DeltaNet).
 * Returns a dynamically allocated array of layer indices and sets *count.
 * ============================================================ */
static int* detect_attn_layers(tensor_info_t* tensors, int n_tensors,
                                int n_layers, int* count) {
    /* Bitmap of layers that have self_attn */
    int* has_attn = (int*)calloc((size_t)n_layers, sizeof(int));
    /* Also track which layers have linear_attn (DeltaNet) */
    int* has_linear = (int*)calloc((size_t)n_layers, sizeof(int));
    if (!has_attn || !has_linear) {
        free(has_attn); free(has_linear);
        *count = 0;
        return NULL;
    }

    for (int i = 0; i < n_tensors; i++) {
        const char* name = tensors[i].name;

        /* Only match model.layers.N. or model.language_model.layers.N.
         * Skip other prefixes like mtp.layers.N. */
        const char* layers_pos = NULL;
        if (strncmp(name, "model.layers.", 13) == 0) {
            layers_pos = name + 13;
        } else if (strncmp(name, "model.language_model.layers.", 28) == 0) {
            layers_pos = name + 28;
        }
        if (!layers_pos) continue;

        int layer_idx = 0;
        const char* p = layers_pos;
        while (*p >= '0' && *p <= '9') {
            layer_idx = layer_idx * 10 + (*p - '0');
            p++;
        }
        if (*p != '.' || p == layers_pos) continue; /* must have "layers.N." */
        p++; /* skip the dot after layer index */

        if (layer_idx < 0 || layer_idx >= n_layers) continue;

        /* Check what kind of attention this layer has */
        if (strncmp(p, "self_attn.", 10) == 0) {
            has_attn[layer_idx] = 1;
        } else if (strncmp(p, "linear_attn.", 12) == 0) {
            has_linear[layer_idx] = 1;
        }
    }

    /* If a layer has linear_attn but NOT self_attn, it is a DeltaNet layer.
     * If we found no linear_attn layers at all, treat all layers as attn. */
    int any_linear = 0;
    for (int l = 0; l < n_layers; l++) {
        if (has_linear[l]) { any_linear = 1; break; }
    }
    if (!any_linear) {
        /* No hybrid architecture detected — all layers have attn */
        for (int l = 0; l < n_layers; l++) has_attn[l] = 1;
    }
    free(has_linear);

    /* Count and build index array */
    int n_attn = 0;
    for (int l = 0; l < n_layers; l++) {
        if (has_attn[l]) n_attn++;
    }

    int* indices = (int*)malloc((size_t)n_attn * sizeof(int));
    if (!indices) {
        free(has_attn);
        *count = 0;
        return NULL;
    }

    int j = 0;
    for (int l = 0; l < n_layers; l++) {
        if (has_attn[l]) indices[j++] = l;
    }

    free(has_attn);
    *count = n_attn;
    return indices;
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
        fprintf(stderr, "tq_load_model: invalid header size (%llu, file=%zu)\n",
                (unsigned long long)header_size, file_size);
        goto fail;
    }

    fprintf(stderr, "tq_load_model: header size = %llu bytes\n",
            (unsigned long long)header_size);

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

    fprintf(stderr, "tq_load_model: parsed %d tensors\n", n_tensors);

    /* Log the dtype of the first non-metadata tensor */
    for (int i = 0; i < n_tensors; i++) {
        if (tensors[i].dtype[0] != '\0') {
            fprintf(stderr, "tq_load_model: tensor dtype = %s (e.g., '%s')\n",
                    tensors[i].dtype, tensors[i].name);
            break;
        }
    }

    /* Allocate model */
    tq_model_t* model = (tq_model_t*)calloc(1, sizeof(tq_model_t));
    if (!model) {
        free(tensors);
        goto fail;
    }
    model->_mmap_data = mmap_data;
    model->_mmap_size = file_size;

    /* Calculate and allocate conversion buffer for non-F32 tensors */
    size_t conv_capacity = calc_conversion_buffer_size(tensors, n_tensors);
    float* conv_buf = NULL;
    size_t conv_used = 0;
    if (conv_capacity > 0) {
        conv_buf = (float*)malloc(conv_capacity);
        if (!conv_buf) {
            fprintf(stderr, "tq_load_model: failed to allocate %zu bytes for dtype conversion\n",
                    conv_capacity);
            free(model);
            free(tensors);
            goto fail;
        }
        model->_converted_data = conv_buf;
        model->_converted_size = conv_capacity;
        fprintf(stderr, "tq_load_model: allocated %zu MB for BF16/F16 -> FP32 conversion\n",
                conv_capacity / (1024 * 1024));
    }

    /* Detect model config from tensor shapes.
     * Look for embedding table to get vocab_size and hidden_dim.
     * Look for layer 0 weights to get n_heads, n_kv_heads, intermediate_dim. */

    /* Try common naming conventions for embedding */
    tensor_info_t* embed = find_tensor(tensors, n_tensors, "model.embed_tokens.weight");
    if (!embed) embed = find_tensor(tensors, n_tensors, "tok_embeddings.weight");
    if (!embed) embed = find_tensor(tensors, n_tensors, "transformer.wte.weight");

    if (!embed) {
        fprintf(stderr, "tq_load_model: cannot find embedding tensor\n");
        /* Dump first few tensor names for debugging */
        int show = n_tensors < 10 ? n_tensors : 10;
        for (int i = 0; i < show; i++) {
            fprintf(stderr, "  tensor[%d]: '%s'\n", i, tensors[i].name);
        }
        free(conv_buf);
        model->_converted_data = NULL;
        free(model);
        free(tensors);
        goto fail;
    }

    model->config.vocab_size = (int)embed->shape[0];
    model->config.hidden_dim = (int)embed->shape[1];
    model->token_embedding = load_tensor(data_base, embed,
                                          &conv_buf, &conv_used, conv_capacity);

    /* Detect n_layers (handles both standard and Qwen3.5 naming) */
    model->config.n_layers = detect_n_layers(tensors, n_tensors);

    /* Detect which layers have standard self_attn */
    int n_attn_layers = 0;
    int* attn_indices = detect_attn_layers(tensors, n_tensors,
                                            model->config.n_layers, &n_attn_layers);
    model->n_attn_layers = n_attn_layers;
    model->attn_layer_indices = attn_indices;

    if (n_attn_layers < model->config.n_layers) {
        fprintf(stderr, "tq_load_model: hybrid architecture detected — "
                "%d/%d layers have self_attn\n",
                n_attn_layers, model->config.n_layers);
        fprintf(stderr, "tq_load_model: attn layers:");
        for (int i = 0; i < n_attn_layers; i++) {
            fprintf(stderr, " %d", attn_indices[i]);
        }
        fprintf(stderr, "\n");
    }

    /* Detect head dimensions from Q projection shape.
     * For hybrid models, find the first layer that has self_attn. */
    int probe_layer = 0;
    if (n_attn_layers > 0) {
        probe_layer = attn_indices[0];
    }

    char name_buf[MAX_NAME_LEN];
    snprintf(name_buf, sizeof(name_buf),
             "model.layers.%d.self_attn.q_proj.weight", probe_layer);
    tensor_info_t* wq0 = find_tensor(tensors, n_tensors, name_buf);

    snprintf(name_buf, sizeof(name_buf),
             "model.layers.%d.self_attn.k_proj.weight", probe_layer);
    tensor_info_t* wk0 = find_tensor(tensors, n_tensors, name_buf);

    if (wq0 && wk0) {
        int q_out = (int)wq0->shape[0];
        int k_out = (int)wk0->shape[0];

        /* Try to detect head_dim from q_norm weight if available */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.q_norm.weight", probe_layer);
        tensor_info_t* qn0 = find_tensor(tensors, n_tensors, name_buf);
        int head_dim;
        if (qn0 && qn0->n_dims >= 1) {
            head_dim = (int)qn0->shape[0];
            model->config.use_qk_norm = 1;
        } else {
            /* Common head_dim values: 128, 64, 96, 256 */
            head_dim = 128;
            if (q_out % head_dim != 0) head_dim = 64;
            if (q_out % head_dim != 0) head_dim = 96;
            if (q_out % head_dim != 0) head_dim = 256;
            model->config.use_qk_norm = 0;
        }
        model->config.head_dim = head_dim;
        model->config.n_kv_heads = k_out / head_dim;

        /* Detect attn_output_gate: if q_proj output is exactly 2x k_proj
         * output * (n_heads/n_kv_heads ratio), then q_proj includes a gate.
         * More precisely: q_out = n_heads * head_dim * (1 + gate).
         * Compare against o_proj input dim to determine n_heads. */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.o_proj.weight", probe_layer);
        tensor_info_t* wo0 = find_tensor(tensors, n_tensors, name_buf);
        if (wo0 && wo0->n_dims >= 2) {
            int o_in = (int)wo0->shape[1]; /* o_proj is [hidden_dim, n_heads*head_dim] */
            int n_heads_from_o = o_in / head_dim;
            if (q_out == n_heads_from_o * head_dim * 2) {
                /* q_proj is doubled: [Q, gate_q] */
                model->config.attn_output_gate = 1;
                model->config.n_heads = n_heads_from_o;
                fprintf(stderr, "tq_load_model: detected attn_output_gate=1 "
                        "(q_proj=%d = 2 * %d * %d)\n",
                        q_out, n_heads_from_o, head_dim);
            } else {
                model->config.attn_output_gate = 0;
                model->config.n_heads = q_out / head_dim;
            }
        } else {
            model->config.attn_output_gate = 0;
            model->config.n_heads = q_out / head_dim;
        }
    } else {
        /* Defaults for small models */
        model->config.head_dim = 64;
        model->config.n_heads = model->config.hidden_dim / 64;
        model->config.n_kv_heads = model->config.n_heads;
        model->config.use_qk_norm = 0;
        model->config.attn_output_gate = 0;
    }

    /* Detect DeltaNet config from first linear_attn layer */
    model->config.delta_n_heads = 0;
    model->config.delta_key_head_dim = 0;
    model->config.delta_value_head_dim = 0;
    model->config.delta_conv_width = 4;
    model->config.partial_rotary_factor = 0.0f;
    {
        /* Find first DeltaNet layer */
        int delta_layer = -1;
        for (int l = 0; l < model->config.n_layers; l++) {
            snprintf(name_buf, sizeof(name_buf),
                     "model.layers.%d.linear_attn.A_log", l);
            if (find_tensor(tensors, n_tensors, name_buf)) {
                delta_layer = l;
                break;
            }
        }
        if (delta_layer >= 0) {
            snprintf(name_buf, sizeof(name_buf),
                     "model.layers.%d.linear_attn.A_log", delta_layer);
            tensor_info_t* a_log = find_tensor(tensors, n_tensors, name_buf);
            if (a_log) {
                model->config.delta_n_heads = (int)a_log->shape[0];
            }

            snprintf(name_buf, sizeof(name_buf),
                     "model.layers.%d.linear_attn.in_proj_qkv.weight", delta_layer);
            tensor_info_t* qkv_proj = find_tensor(tensors, n_tensors, name_buf);
            if (qkv_proj && model->config.delta_n_heads > 0) {
                int qkv_dim = (int)qkv_proj->shape[0];
                /* qkv_dim = 3 * n_heads * head_dim */
                model->config.delta_key_head_dim = qkv_dim / (3 * model->config.delta_n_heads);
                model->config.delta_value_head_dim = model->config.delta_key_head_dim;
            }

            snprintf(name_buf, sizeof(name_buf),
                     "model.layers.%d.linear_attn.conv1d.weight", delta_layer);
            tensor_info_t* conv = find_tensor(tensors, n_tensors, name_buf);
            if (conv && conv->n_dims >= 3) {
                model->config.delta_conv_width = (int)conv->shape[2];
            }

            fprintf(stderr, "tq_load_model: DeltaNet config — %d heads, key_dim=%d, val_dim=%d, conv_w=%d\n",
                    model->config.delta_n_heads, model->config.delta_key_head_dim,
                    model->config.delta_value_head_dim, model->config.delta_conv_width);
        }
    }

    /* Detect intermediate_dim from gate projection (use probe_layer) */
    snprintf(name_buf, sizeof(name_buf),
             "model.layers.%d.mlp.gate_proj.weight", probe_layer);
    tensor_info_t* wg0 = find_tensor(tensors, n_tensors, name_buf);
    if (!wg0) {
        /* Try layer 0 as fallback — MLP might exist on all layers even in hybrid models */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.0.mlp.gate_proj.weight");
        wg0 = find_tensor(tensors, n_tensors, name_buf);
    }
    if (wg0) {
        model->config.intermediate_dim = (int)wg0->shape[0];
    } else {
        model->config.intermediate_dim = model->config.hidden_dim * 4;
    }

    /* Defaults — tuned for Qwen3.5 if DeltaNet detected */
    model->config.max_seq_len = 4096;
    if (model->config.delta_n_heads > 0) {
        /* Qwen3.5 uses rope_theta=10M, rms_norm_eps=1e-6, partial_rotary=0.25 */
        model->config.rope_freq_base = 10000000.0f;
        model->config.rms_norm_eps = 1e-6f;
        model->config.partial_rotary_factor = 0.25f;
    } else {
        model->config.rope_freq_base = 10000.0f;
        model->config.rms_norm_eps = 1e-5f;
        model->config.partial_rotary_factor = 0.0f;
    }

    /* Allocate layer weight pointers */
    int n_layers = model->config.n_layers;
    model->layers = (tq_layer_weights_t*)calloc((size_t)n_layers, sizeof(tq_layer_weights_t));
    if (!model->layers) {
        free(attn_indices);
        model->attn_layer_indices = NULL;
        free(conv_buf);
        model->_converted_data = NULL;
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
        layer->attn_norm = load_tensor(data_base,
                                        find_tensor(tensors, n_tensors, name_buf),
                                        &conv_buf, &conv_used, conv_capacity);

        /* FFN norm */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.post_attention_layernorm.weight", l);
        layer->ffn_norm = load_tensor(data_base,
                                       find_tensor(tensors, n_tensors, name_buf),
                                       &conv_buf, &conv_used, conv_capacity);

        /* Q, K, V, O projections — only exist for self_attn layers */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.q_proj.weight", l);
        layer->wq = load_tensor(data_base,
                                 find_tensor(tensors, n_tensors, name_buf),
                                 &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.k_proj.weight", l);
        layer->wk = load_tensor(data_base,
                                 find_tensor(tensors, n_tensors, name_buf),
                                 &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.v_proj.weight", l);
        layer->wv = load_tensor(data_base,
                                 find_tensor(tensors, n_tensors, name_buf),
                                 &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.o_proj.weight", l);
        layer->wo = load_tensor(data_base,
                                 find_tensor(tensors, n_tensors, name_buf),
                                 &conv_buf, &conv_used, conv_capacity);

        /* QK-norm weights (Qwen3.5 style) */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.q_norm.weight", l);
        layer->q_norm = load_tensor(data_base,
                                     find_tensor(tensors, n_tensors, name_buf),
                                     &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.k_norm.weight", l);
        layer->k_norm = load_tensor(data_base,
                                     find_tensor(tensors, n_tensors, name_buf),
                                     &conv_buf, &conv_used, conv_capacity);

        /* DeltaNet (linear_attention) weights */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.A_log", l);
        layer->delta_a_log = load_tensor(data_base,
                                          find_tensor(tensors, n_tensors, name_buf),
                                          &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.conv1d.weight", l);
        layer->delta_conv1d = load_tensor(data_base,
                                           find_tensor(tensors, n_tensors, name_buf),
                                           &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.dt_bias", l);
        layer->delta_dt_bias = load_tensor(data_base,
                                            find_tensor(tensors, n_tensors, name_buf),
                                            &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.in_proj_a.weight", l);
        layer->delta_in_proj_a = load_tensor(data_base,
                                              find_tensor(tensors, n_tensors, name_buf),
                                              &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.in_proj_b.weight", l);
        layer->delta_in_proj_b = load_tensor(data_base,
                                              find_tensor(tensors, n_tensors, name_buf),
                                              &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.in_proj_qkv.weight", l);
        layer->delta_in_proj_qkv = load_tensor(data_base,
                                                find_tensor(tensors, n_tensors, name_buf),
                                                &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.in_proj_z.weight", l);
        layer->delta_in_proj_z = load_tensor(data_base,
                                              find_tensor(tensors, n_tensors, name_buf),
                                              &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.norm.weight", l);
        layer->delta_norm = load_tensor(data_base,
                                         find_tensor(tensors, n_tensors, name_buf),
                                         &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.out_proj.weight", l);
        layer->delta_out_proj = load_tensor(data_base,
                                             find_tensor(tensors, n_tensors, name_buf),
                                             &conv_buf, &conv_used, conv_capacity);

        /* FFN: gate, up, down projections (SwiGLU) */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.gate_proj.weight", l);
        layer->w_gate = load_tensor(data_base,
                                     find_tensor(tensors, n_tensors, name_buf),
                                     &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.up_proj.weight", l);
        layer->w_up = load_tensor(data_base,
                                   find_tensor(tensors, n_tensors, name_buf),
                                   &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.down_proj.weight", l);
        layer->w_down = load_tensor(data_base,
                                     find_tensor(tensors, n_tensors, name_buf),
                                     &conv_buf, &conv_used, conv_capacity);
    }

    /* Output norm */
    model->output_norm = load_tensor(data_base,
        find_tensor(tensors, n_tensors, "model.norm.weight"),
        &conv_buf, &conv_used, conv_capacity);

    /* Output weight — may be tied to embedding */
    tensor_info_t* lm_head = find_tensor(tensors, n_tensors, "lm_head.weight");
    if (lm_head) {
        model->output_weight = load_tensor(data_base, lm_head,
                                            &conv_buf, &conv_used, conv_capacity);
    } else {
        /* Weight tying: reuse embedding */
        model->output_weight = model->token_embedding;
    }

    free(tensors);

    /* Qwen3.5 RMSNorm adjustment: Qwen3_5RMSNorm computes
     * output = norm(x) * (1.0 + weight), NOT norm(x) * weight.
     * We bake the "+1" into the weight so tq_rmsnorm can stay as
     * out = x * rsqrt * weight.
     *
     * This applies to: input_layernorm, post_attention_layernorm,
     * model.norm, q_norm, k_norm.
     * It does NOT apply to: linear_attn.norm (Qwen3_5RMSNormGated
     * uses plain weight without +1).
     *
     * We detect Qwen3.5 by the presence of DeltaNet layers. */
    if (model->config.delta_n_heads > 0) {
        int dim_h = model->config.hidden_dim;
        int head_dim_h = model->config.head_dim;

        for (int l = 0; l < n_layers; l++) {
            tq_layer_weights_t* layer_w = &model->layers[l];
            if (layer_w->attn_norm) {
                for (int i = 0; i < dim_h; i++)
                    layer_w->attn_norm[i] += 1.0f;
            }
            if (layer_w->ffn_norm) {
                for (int i = 0; i < dim_h; i++)
                    layer_w->ffn_norm[i] += 1.0f;
            }
            if (layer_w->q_norm) {
                for (int i = 0; i < head_dim_h; i++)
                    layer_w->q_norm[i] += 1.0f;
            }
            if (layer_w->k_norm) {
                for (int i = 0; i < head_dim_h; i++)
                    layer_w->k_norm[i] += 1.0f;
            }
            /* Note: delta_norm is NOT adjusted — Qwen3_5RMSNormGated
             * uses plain weight without +1 offset */
        }
        if (model->output_norm) {
            for (int i = 0; i < dim_h; i++)
                model->output_norm[i] += 1.0f;
        }
        fprintf(stderr, "tq_load_model: applied Qwen3.5 RMSNorm +1 weight adjustment\n");
    }

    fprintf(stderr, "tq_load_model: loaded %d layers (%d with self_attn), "
            "dim=%d, heads=%d/%d, vocab=%d\n",
            model->config.n_layers, model->n_attn_layers,
            model->config.hidden_dim,
            model->config.n_heads, model->config.n_kv_heads,
            model->config.vocab_size);

    if (conv_capacity > 0) {
        fprintf(stderr, "tq_load_model: dtype conversion used %zu / %zu MB\n",
                conv_used / (1024 * 1024), conv_capacity / (1024 * 1024));
    }

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

    free(model->_converted_data);
    free(model->attn_layer_indices);
    free(model->layers);
    free(model);
}
