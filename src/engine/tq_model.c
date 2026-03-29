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
 * Get raw BF16 pointer for a tensor (zero-copy from mmap).
 * Returns NULL if tensor is not BF16.
 * ============================================================ */
static const uint16_t* get_bf16_ptr(void* data_base, tensor_info_t* t) {
    if (!t) return NULL;
    if (strcmp(t->dtype, "BF16") != 0) return NULL;
    return (const uint16_t*)((const char*)data_base + t->data_start);
}

/* ============================================================
 * Check if a tensor name matches any of the "keep as BF16" names.
 * These tensors won't be converted to FP32 upfront; instead they
 * will be converted on demand during inference to save memory.
 * ============================================================ */
static int should_keep_bf16(const char* name) {
    /* Embedding table — largest single tensor, only need one row at a time */
    if (strcmp(name, "model.embed_tokens.weight") == 0) return 1;
    if (strcmp(name, "model.language_model.embed_tokens.weight") == 0) return 1;
    if (strcmp(name, "tok_embeddings.weight") == 0) return 1;
    if (strcmp(name, "transformer.wte.weight") == 0) return 1;
    /* lm_head — output projection, can use BF16 matmul */
    if (strcmp(name, "lm_head.weight") == 0) return 1;
    return 0;
}

/* ============================================================
 * Calculate total FP32 bytes needed for all non-F32 tensors,
 * excluding tensors that will be kept as BF16 (streaming conversion).
 * ============================================================ */
static size_t calc_conversion_buffer_size(tensor_info_t* tensors, int n_tensors) {
    size_t total = 0;
    for (int i = 0; i < n_tensors; i++) {
        if (strcmp(tensors[i].dtype, "F32") != 0 &&
            strcmp(tensors[i].dtype, "__metadata__") != 0 &&
            !should_keep_bf16(tensors[i].name)) {
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

/* Forward declaration */
static tq_model_t* tq_load_safetensors(const char* path);

/* ============================================================
 * Load model — auto-detect format (TQM or safetensors)
 * ============================================================ */
tq_model_t* tq_load_model(const char* path) {
    if (!path) return NULL;

    /* Check magic bytes to detect TQM format */
    FILE* probe = fopen(path, "rb");
    if (probe) {
        uint32_t magic = 0;
        if (fread(&magic, 4, 1, probe) == 1 && magic == TQM_MAGIC) {
            fclose(probe);
            fprintf(stderr, "tq_load_model: detected TQM format, using fast loader\n");
            return tq_load_tqm(path);
        }
        fclose(probe);
    }

    /* Fall through to safetensors loader */
    return tq_load_safetensors(path);
}

/* ============================================================
 * Load model from safetensors file
 * ============================================================ */
static tq_model_t* tq_load_safetensors(const char* path) {
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

    /* Try to keep embedding as BF16 (streaming conversion saves ~1GB) */
    model->embed_bf16 = get_bf16_ptr(data_base, embed);
    if (model->embed_bf16) {
        /* BF16 embedding: don't convert, will convert on demand in forward pass */
        model->token_embedding = NULL;
        size_t embed_bytes = (size_t)embed->shape[0] * embed->shape[1] * 2;
        fprintf(stderr, "tq_load_model: keeping embed_tokens as BF16 (%zu MB saved)\n",
                embed_bytes / (1024 * 1024));
    } else {
        /* F32 or other dtype: convert as before */
        model->token_embedding = load_tensor(data_base, embed,
                                              &conv_buf, &conv_used, conv_capacity);
    }

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
        /* Try to keep lm_head as BF16 */
        model->output_weight_bf16 = get_bf16_ptr(data_base, lm_head);
        if (model->output_weight_bf16) {
            model->output_weight = NULL;
            size_t lm_bytes = (size_t)lm_head->shape[0] * lm_head->shape[1] * 2;
            fprintf(stderr, "tq_load_model: keeping lm_head as BF16 (%zu MB saved)\n",
                    lm_bytes / (1024 * 1024));
        } else {
            model->output_weight = load_tensor(data_base, lm_head,
                                                &conv_buf, &conv_used, conv_capacity);
        }
    } else {
        /* Weight tying: reuse embedding (either FP32 or BF16) */
        model->output_weight = model->token_embedding;
        model->output_weight_bf16 = model->embed_bf16;
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
 * Q8 weight quantization — quantize all layer weights post-load
 *
 * Converts FP32 weight matrices to Q8 (int8 + per-block float scale,
 * block_size=32).  This halves memory: FP32 uses 4 bytes/value,
 * Q8 uses 1 byte + 4 bytes/32 = 1.125 bytes/value.
 *
 * Each weight matrix [rows, cols] gets:
 *   - int8_t q8[rows * cols]           — quantized values
 *   - float  scales[rows * (cols/32)]  — per-block scales
 *
 * After quantization, the original FP32 pointer is set to NULL
 * (it either pointed into mmap or conversion buffer, both still alive).
 * ============================================================ */

/* Helper: quantize a single weight matrix and store into pre-allocated buffer */
static void quantize_matrix_q8(const float* src, int rows, int cols,
                                 int8_t** out_qs, float** out_scales,
                                 char** buf, size_t* used) {
    if (!src || rows <= 0 || cols <= 0) {
        *out_qs = NULL;
        *out_scales = NULL;
        return;
    }
    int n_blocks_per_row = (cols + 31) / 32;
    size_t qs_bytes = (size_t)rows * cols * sizeof(int8_t);
    size_t sc_bytes = (size_t)rows * n_blocks_per_row * sizeof(float);

    int8_t*  qs = (int8_t*)(*buf + *used);
    *used += qs_bytes;
    float*   sc = (float*)(*buf + *used);
    *used += sc_bytes;

    for (int r = 0; r < rows; r++) {
        tq_quantize_row_q8(src + (size_t)r * cols,
                            qs + (size_t)r * cols,
                            sc + (size_t)r * n_blocks_per_row,
                            cols);
    }
    *out_qs = qs;
    *out_scales = sc;
}

/* Calculate total Q8 buffer size needed for all layer weights */
static size_t calc_q8_buffer_size(const tq_model_t* model) {
    size_t total = 0;
    const tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;

    /* DeltaNet dimensions */
    int delta_qkv_dim = 3 * c->delta_n_heads * c->delta_key_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;

    for (int l = 0; l < c->n_layers; l++) {
        const tq_layer_weights_t* layer = &model->layers[l];

        /* Self-attention weights */
        if (layer->wq) {
            int rows = qg_dim;
            int cols = dim;
            int nb = (cols + 31) / 32;
            total += (size_t)rows * cols;        /* int8 data */
            total += (size_t)rows * nb * 4;      /* float scales */
        }
        if (layer->wk) {
            int nb = (dim + 31) / 32;
            total += (size_t)kv_dim * dim;
            total += (size_t)kv_dim * nb * 4;
        }
        if (layer->wv) {
            int nb = (dim + 31) / 32;
            total += (size_t)kv_dim * dim;
            total += (size_t)kv_dim * nb * 4;
        }
        if (layer->wo) {
            int nb = (q_dim + 31) / 32;
            total += (size_t)dim * q_dim;
            total += (size_t)dim * nb * 4;
        }

        /* FFN weights */
        if (layer->w_gate) {
            int nb = (dim + 31) / 32;
            total += (size_t)inter * dim;
            total += (size_t)inter * nb * 4;
        }
        if (layer->w_up) {
            int nb = (dim + 31) / 32;
            total += (size_t)inter * dim;
            total += (size_t)inter * nb * 4;
        }
        if (layer->w_down) {
            int nb = (inter + 31) / 32;
            total += (size_t)dim * inter;
            total += (size_t)dim * nb * 4;
        }

        /* DeltaNet weights */
        if (layer->delta_in_proj_qkv) {
            int nb = (dim + 31) / 32;
            total += (size_t)delta_qkv_dim * dim;
            total += (size_t)delta_qkv_dim * nb * 4;
        }
        if (layer->delta_in_proj_z) {
            int nb = (dim + 31) / 32;
            total += (size_t)delta_z_dim * dim;
            total += (size_t)delta_z_dim * nb * 4;
        }
        if (layer->delta_in_proj_a) {
            int nb = (dim + 31) / 32;
            total += (size_t)delta_dn * dim;
            total += (size_t)delta_dn * nb * 4;
        }
        if (layer->delta_in_proj_b) {
            int nb = (dim + 31) / 32;
            total += (size_t)delta_dn * dim;
            total += (size_t)delta_dn * nb * 4;
        }
        if (layer->delta_out_proj) {
            int nb = (delta_z_dim + 31) / 32;
            total += (size_t)dim * delta_z_dim;
            total += (size_t)dim * nb * 4;
        }
    }
    return total;
}

void tq_quantize_weights(tq_model_t* model) {
    if (!model || model->use_q8_weights) return;

    const tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;

    /* DeltaNet dimensions */
    int delta_qkv_dim = 3 * c->delta_n_heads * c->delta_key_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;

    size_t buf_size = calc_q8_buffer_size(model);
    char* buf = (char*)malloc(buf_size);
    if (!buf) {
        fprintf(stderr, "tq_quantize_weights: failed to allocate %zu MB for Q8\n",
                buf_size / (1024 * 1024));
        return;
    }
    size_t used = 0;

    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];

        /* Self-attention */
        quantize_matrix_q8(layer->wq, qg_dim, dim,
                            &layer->wq_q8, &layer->wq_q8s, &buf, &used);
        if (layer->wq_q8) layer->wq = NULL;

        quantize_matrix_q8(layer->wk, kv_dim, dim,
                            &layer->wk_q8, &layer->wk_q8s, &buf, &used);
        if (layer->wk_q8) layer->wk = NULL;

        quantize_matrix_q8(layer->wv, kv_dim, dim,
                            &layer->wv_q8, &layer->wv_q8s, &buf, &used);
        if (layer->wv_q8) layer->wv = NULL;

        quantize_matrix_q8(layer->wo, dim, q_dim,
                            &layer->wo_q8, &layer->wo_q8s, &buf, &used);
        if (layer->wo_q8) layer->wo = NULL;

        /* FFN */
        quantize_matrix_q8(layer->w_gate, inter, dim,
                            &layer->w_gate_q8, &layer->w_gate_q8s, &buf, &used);
        if (layer->w_gate_q8) layer->w_gate = NULL;

        quantize_matrix_q8(layer->w_up, inter, dim,
                            &layer->w_up_q8, &layer->w_up_q8s, &buf, &used);
        if (layer->w_up_q8) layer->w_up = NULL;

        quantize_matrix_q8(layer->w_down, dim, inter,
                            &layer->w_down_q8, &layer->w_down_q8s, &buf, &used);
        if (layer->w_down_q8) layer->w_down = NULL;

        /* DeltaNet */
        quantize_matrix_q8(layer->delta_in_proj_qkv, delta_qkv_dim, dim,
                            &layer->delta_in_proj_qkv_q8, &layer->delta_in_proj_qkv_q8s,
                            &buf, &used);
        if (layer->delta_in_proj_qkv_q8) layer->delta_in_proj_qkv = NULL;

        quantize_matrix_q8(layer->delta_in_proj_z, delta_z_dim, dim,
                            &layer->delta_in_proj_z_q8, &layer->delta_in_proj_z_q8s,
                            &buf, &used);
        if (layer->delta_in_proj_z_q8) layer->delta_in_proj_z = NULL;

        quantize_matrix_q8(layer->delta_in_proj_a, delta_dn, dim,
                            &layer->delta_in_proj_a_q8, &layer->delta_in_proj_a_q8s,
                            &buf, &used);
        if (layer->delta_in_proj_a_q8) layer->delta_in_proj_a = NULL;

        quantize_matrix_q8(layer->delta_in_proj_b, delta_dn, dim,
                            &layer->delta_in_proj_b_q8, &layer->delta_in_proj_b_q8s,
                            &buf, &used);
        if (layer->delta_in_proj_b_q8) layer->delta_in_proj_b = NULL;

        quantize_matrix_q8(layer->delta_out_proj, dim, delta_z_dim,
                            &layer->delta_out_proj_q8, &layer->delta_out_proj_q8s,
                            &buf, &used);
        if (layer->delta_out_proj_q8) layer->delta_out_proj = NULL;
    }

    model->use_q8_weights = 1;
    model->_q8_data = buf;
    model->_q8_size = used;

    /* If original weights were in conversion buffer (BF16->FP32), free it.
     * The converted_data is no longer needed since all layer weights are now Q8.
     * Note: norm weights, conv1d, bias, etc. still point into converted_data or mmap,
     * so we CANNOT free it. Keep it alive. */

    fprintf(stderr, "tq_quantize_weights: quantized to Q8 (%zu MB, was ~%zu MB FP32)\n",
            used / (1024 * 1024), used * 4 / (1024 * 1024));
}

/* ============================================================
 * Q4_0 weight quantization — quantize all layer weights post-load
 *
 * Converts FP32 weight matrices to Q4_0 (packed 4-bit + per-block float scale,
 * block_size=32). This reduces memory ~7x: FP32 uses 4 bytes/value,
 * Q4_0 uses 0.5 byte + 4 bytes/32 = 0.625 bytes/value.
 *
 * Each weight matrix [rows, cols] gets:
 *   - uint8_t qs[rows * (cols/32) * 16] — packed 4-bit values (2 per byte)
 *   - float scales[rows * (cols/32)]     — per-block scales
 *
 * After quantization, the original FP32 pointer is set to NULL.
 * ============================================================ */

/* Helper: quantize a single weight matrix to Q4 and store into pre-allocated buffer */
static void quantize_matrix_q4(const float* src, int rows, int cols,
                                 uint8_t** out_qs, float** out_scales,
                                 char** buf, size_t* used) {
    if (!src || rows <= 0 || cols <= 0) {
        *out_qs = NULL;
        *out_scales = NULL;
        return;
    }
    int n_blocks_per_row = (cols + 31) / 32;
    size_t qs_bytes = (size_t)rows * n_blocks_per_row * 16;   /* 16 packed bytes per block */
    size_t sc_bytes = (size_t)rows * n_blocks_per_row * sizeof(float);

    uint8_t* qs = (uint8_t*)(*buf + *used);
    *used += qs_bytes;
    float*   sc = (float*)(*buf + *used);
    *used += sc_bytes;

    for (int r = 0; r < rows; r++) {
        tq_quantize_row_q4(src + (size_t)r * cols,
                            qs + (size_t)r * n_blocks_per_row * 16,
                            sc + (size_t)r * n_blocks_per_row,
                            cols);
    }
    *out_qs = qs;
    *out_scales = sc;
}

/* Calculate total Q4 buffer size needed for all layer weights */
static size_t calc_q4_buffer_size(const tq_model_t* model) {
    size_t total = 0;
    const tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;

    /* DeltaNet dimensions */
    int delta_qkv_dim = 3 * c->delta_n_heads * c->delta_key_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;

    for (int l = 0; l < c->n_layers; l++) {
        const tq_layer_weights_t* layer = &model->layers[l];

        /* Self-attention weights */
        if (layer->wq) {
            int nb = (dim + 31) / 32;
            total += (size_t)qg_dim * nb * 16;   /* packed Q4 data */
            total += (size_t)qg_dim * nb * 4;     /* float scales */
        }
        if (layer->wk) {
            int nb = (dim + 31) / 32;
            total += (size_t)kv_dim * nb * 16;
            total += (size_t)kv_dim * nb * 4;
        }
        if (layer->wv) {
            int nb = (dim + 31) / 32;
            total += (size_t)kv_dim * nb * 16;
            total += (size_t)kv_dim * nb * 4;
        }
        if (layer->wo) {
            int nb = (q_dim + 31) / 32;
            total += (size_t)dim * nb * 16;
            total += (size_t)dim * nb * 4;
        }

        /* FFN weights */
        if (layer->w_gate) {
            int nb = (dim + 31) / 32;
            total += (size_t)inter * nb * 16;
            total += (size_t)inter * nb * 4;
        }
        if (layer->w_up) {
            int nb = (dim + 31) / 32;
            total += (size_t)inter * nb * 16;
            total += (size_t)inter * nb * 4;
        }
        if (layer->w_down) {
            int nb = (inter + 31) / 32;
            total += (size_t)dim * nb * 16;
            total += (size_t)dim * nb * 4;
        }

        /* DeltaNet weights */
        if (layer->delta_in_proj_qkv) {
            int nb = (dim + 31) / 32;
            total += (size_t)delta_qkv_dim * nb * 16;
            total += (size_t)delta_qkv_dim * nb * 4;
        }
        if (layer->delta_in_proj_z) {
            int nb = (dim + 31) / 32;
            total += (size_t)delta_z_dim * nb * 16;
            total += (size_t)delta_z_dim * nb * 4;
        }
        if (layer->delta_in_proj_a) {
            int nb = (dim + 31) / 32;
            total += (size_t)delta_dn * nb * 16;
            total += (size_t)delta_dn * nb * 4;
        }
        if (layer->delta_in_proj_b) {
            int nb = (dim + 31) / 32;
            total += (size_t)delta_dn * nb * 16;
            total += (size_t)delta_dn * nb * 4;
        }
        if (layer->delta_out_proj) {
            int nb = (delta_z_dim + 31) / 32;
            total += (size_t)dim * nb * 16;
            total += (size_t)dim * nb * 4;
        }
    }
    return total;
}

void tq_quantize_weights_q4(tq_model_t* model) {
    if (!model || model->use_q4_weights) return;

    const tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;

    /* DeltaNet dimensions */
    int delta_qkv_dim = 3 * c->delta_n_heads * c->delta_key_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;

    size_t buf_size = calc_q4_buffer_size(model);
    char* buf = (char*)malloc(buf_size);
    if (!buf) {
        fprintf(stderr, "tq_quantize_weights_q4: failed to allocate %zu MB for Q4\n",
                buf_size / (1024 * 1024));
        return;
    }
    size_t used = 0;

    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];

        /* Self-attention */
        quantize_matrix_q4(layer->wq, qg_dim, dim,
                            &layer->wq_q4, &layer->wq_q4s, &buf, &used);
        if (layer->wq_q4) layer->wq = NULL;

        quantize_matrix_q4(layer->wk, kv_dim, dim,
                            &layer->wk_q4, &layer->wk_q4s, &buf, &used);
        if (layer->wk_q4) layer->wk = NULL;

        quantize_matrix_q4(layer->wv, kv_dim, dim,
                            &layer->wv_q4, &layer->wv_q4s, &buf, &used);
        if (layer->wv_q4) layer->wv = NULL;

        quantize_matrix_q4(layer->wo, dim, q_dim,
                            &layer->wo_q4, &layer->wo_q4s, &buf, &used);
        if (layer->wo_q4) layer->wo = NULL;

        /* FFN */
        quantize_matrix_q4(layer->w_gate, inter, dim,
                            &layer->w_gate_q4, &layer->w_gate_q4s, &buf, &used);
        if (layer->w_gate_q4) layer->w_gate = NULL;

        quantize_matrix_q4(layer->w_up, inter, dim,
                            &layer->w_up_q4, &layer->w_up_q4s, &buf, &used);
        if (layer->w_up_q4) layer->w_up = NULL;

        quantize_matrix_q4(layer->w_down, dim, inter,
                            &layer->w_down_q4, &layer->w_down_q4s, &buf, &used);
        if (layer->w_down_q4) layer->w_down = NULL;

        /* DeltaNet */
        quantize_matrix_q4(layer->delta_in_proj_qkv, delta_qkv_dim, dim,
                            &layer->delta_in_proj_qkv_q4, &layer->delta_in_proj_qkv_q4s,
                            &buf, &used);
        if (layer->delta_in_proj_qkv_q4) layer->delta_in_proj_qkv = NULL;

        quantize_matrix_q4(layer->delta_in_proj_z, delta_z_dim, dim,
                            &layer->delta_in_proj_z_q4, &layer->delta_in_proj_z_q4s,
                            &buf, &used);
        if (layer->delta_in_proj_z_q4) layer->delta_in_proj_z = NULL;

        quantize_matrix_q4(layer->delta_in_proj_a, delta_dn, dim,
                            &layer->delta_in_proj_a_q4, &layer->delta_in_proj_a_q4s,
                            &buf, &used);
        if (layer->delta_in_proj_a_q4) layer->delta_in_proj_a = NULL;

        quantize_matrix_q4(layer->delta_in_proj_b, delta_dn, dim,
                            &layer->delta_in_proj_b_q4, &layer->delta_in_proj_b_q4s,
                            &buf, &used);
        if (layer->delta_in_proj_b_q4) layer->delta_in_proj_b = NULL;

        quantize_matrix_q4(layer->delta_out_proj, dim, delta_z_dim,
                            &layer->delta_out_proj_q4, &layer->delta_out_proj_q4s,
                            &buf, &used);
        if (layer->delta_out_proj_q4) layer->delta_out_proj = NULL;
    }

    model->use_q4_weights = 1;
    model->_q4_data = buf;
    model->_q4_size = used;

    fprintf(stderr, "tq_quantize_weights_q4: quantized to Q4 (%zu MB, was ~%zu MB FP32)\n",
            used / (1024 * 1024), used * 8 / (1024 * 1024));
}

/* ============================================================
 * TQM format: pre-quantized model loading (mmap, zero-copy)
 *
 * The .tqm file stores:
 *   - 512-byte header (tqm_header_t)
 *   - Tokenizer JSON (raw bytes, variable size)
 *   - Weights section (Q4 packed + FP32 norms + BF16 embeds)
 *
 * All pointers into the model's weight arrays point directly
 * into the mmap'd file — no malloc, no conversion.
 * ============================================================ */

/* Align offset up to TQM_ALIGN boundary */
static uint64_t tqm_align(uint64_t offset) {
    return (offset + TQM_ALIGN - 1) & ~((uint64_t)(TQM_ALIGN - 1));
}

tq_model_t* tq_load_tqm(const char* path) {
    if (!path) return NULL;

#ifdef _WIN32
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "tq_load_tqm: cannot open '%s'\n", path);
        return NULL;
    }
    LARGE_INTEGER fileSize;
    GetFileSizeEx(hFile, &fileSize);
    size_t file_size = (size_t)fileSize.QuadPart;

    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) { CloseHandle(hFile); return NULL; }
    void* mmap_data = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    CloseHandle(hFile);
    if (!mmap_data) return NULL;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "tq_load_tqm: cannot open '%s'\n", path);
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return NULL; }
    size_t file_size = (size_t)st.st_size;

    void* mmap_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mmap_data == MAP_FAILED) {
        fprintf(stderr, "tq_load_tqm: mmap failed for '%s'\n", path);
        return NULL;
    }
#endif

    if (file_size < sizeof(tqm_header_t)) {
        fprintf(stderr, "tq_load_tqm: file too small (%zu bytes)\n", file_size);
        goto fail_tqm;
    }

    const tqm_header_t* hdr = (const tqm_header_t*)mmap_data;
    if (hdr->magic != TQM_MAGIC) {
        fprintf(stderr, "tq_load_tqm: invalid magic 0x%08X (expected 0x%08X)\n",
                hdr->magic, TQM_MAGIC);
        goto fail_tqm;
    }
    if (hdr->version != TQM_VERSION) {
        fprintf(stderr, "tq_load_tqm: unsupported version %u\n", hdr->version);
        goto fail_tqm;
    }

    /* Allocate model */
    tq_model_t* model = (tq_model_t*)calloc(1, sizeof(tq_model_t));
    if (!model) goto fail_tqm;

    model->_mmap_data = mmap_data;
    model->_mmap_size = file_size;

    /* Copy config from header */
    tq_model_config_t* c = &model->config;
    c->n_layers         = hdr->n_layers;
    c->hidden_dim       = hdr->hidden_dim;
    c->intermediate_dim = hdr->intermediate_dim;
    c->n_heads          = hdr->n_heads;
    c->n_kv_heads       = hdr->n_kv_heads;
    c->head_dim         = hdr->head_dim;
    c->vocab_size       = hdr->vocab_size;
    c->max_seq_len      = hdr->max_seq_len;
    c->rope_freq_base   = hdr->rope_freq_base;
    c->rms_norm_eps     = hdr->rms_norm_eps;

    c->delta_n_heads       = hdr->delta_n_heads;
    c->delta_key_head_dim  = hdr->delta_key_head_dim;
    c->delta_value_head_dim= hdr->delta_value_head_dim;
    c->delta_conv_width    = hdr->delta_conv_width;
    c->partial_rotary_factor = hdr->partial_rotary_factor;
    c->use_qk_norm         = hdr->use_qk_norm;
    c->attn_output_gate    = hdr->attn_output_gate;

    /* Attn layer indices */
    model->n_attn_layers = hdr->n_attn_layers;
    if (hdr->n_attn_layers > 0) {
        model->attn_layer_indices = (int*)malloc((size_t)hdr->n_attn_layers * sizeof(int));
        if (!model->attn_layer_indices) { free(model); goto fail_tqm; }
        for (int i = 0; i < hdr->n_attn_layers; i++) {
            model->attn_layer_indices[i] = hdr->attn_layer_indices[i];
        }
    }

    /* Build is-attn lookup */
    int* is_attn_layer = (int*)calloc((size_t)c->n_layers, sizeof(int));
    if (is_attn_layer) {
        for (int i = 0; i < model->n_attn_layers; i++) {
            int idx = model->attn_layer_indices[i];
            if (idx >= 0 && idx < c->n_layers) is_attn_layer[idx] = 1;
        }
    }

    /* Dimensions for Q4 weight sizes */
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;
    int delta_qkv_dim = 3 * c->delta_n_heads * c->delta_key_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;
    int delta_conv_total = delta_qkv_dim; /* conv1d channels */
    int delta_conv_width = c->delta_conv_width;
    int delta_vhd = c->delta_value_head_dim;

    /* Navigate to weights section */
    uint8_t* ptr = (uint8_t*)mmap_data + hdr->weights_offset;
    uint8_t* file_end = (uint8_t*)mmap_data + file_size;

    /* Helper macro: read FP32 array pointer from mmap */
    #define TQM_READ_FP32(dest, count) do {                   \
        size_t _bytes = (size_t)(count) * sizeof(float);      \
        if (ptr + _bytes > file_end) goto fail_tqm_model;     \
        (dest) = (float*)ptr;                                  \
        ptr += _bytes;                                         \
    } while (0)

    /* Helper macro: read Q4 weight (packed bytes + float scales) */
    #define TQM_READ_Q4(dest_qs, dest_sc, rows, cols) do {    \
        int _nb = ((cols) + 31) / 32;                          \
        size_t _qs_bytes = (size_t)(rows) * _nb * 16;          \
        size_t _sc_bytes = (size_t)(rows) * _nb * sizeof(float); \
        if (ptr + _qs_bytes + _sc_bytes > file_end) goto fail_tqm_model; \
        (dest_qs) = (uint8_t*)ptr;                             \
        ptr += _qs_bytes;                                      \
        (dest_sc) = (float*)ptr;                               \
        ptr += _sc_bytes;                                      \
    } while (0)

    /* Allocate layers */
    model->layers = (tq_layer_weights_t*)calloc((size_t)c->n_layers,
                                                 sizeof(tq_layer_weights_t));
    if (!model->layers) goto fail_tqm_model;

    /* Read per-layer weights */
    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];

        /* Both layer types have norms */
        TQM_READ_FP32(layer->attn_norm, dim);
        TQM_READ_FP32(layer->ffn_norm, dim);

        if (is_attn_layer && is_attn_layer[l]) {
            /* Self-attention layer */
            TQM_READ_Q4(layer->wq_q4, layer->wq_q4s, qg_dim, dim);
            TQM_READ_Q4(layer->wk_q4, layer->wk_q4s, kv_dim, dim);
            TQM_READ_Q4(layer->wv_q4, layer->wv_q4s, kv_dim, dim);
            TQM_READ_Q4(layer->wo_q4, layer->wo_q4s, dim, q_dim);

            if (c->use_qk_norm) {
                TQM_READ_FP32(layer->q_norm, c->head_dim);
                TQM_READ_FP32(layer->k_norm, c->head_dim);
            }
        } else {
            /* DeltaNet layer */
            TQM_READ_FP32(layer->delta_a_log, delta_dn);
            TQM_READ_FP32(layer->delta_dt_bias, delta_dn);
            TQM_READ_Q4(layer->delta_in_proj_qkv_q4, layer->delta_in_proj_qkv_q4s,
                         delta_qkv_dim, dim);
            TQM_READ_Q4(layer->delta_in_proj_z_q4, layer->delta_in_proj_z_q4s,
                         delta_z_dim, dim);
            TQM_READ_Q4(layer->delta_in_proj_a_q4, layer->delta_in_proj_a_q4s,
                         delta_dn, dim);
            TQM_READ_Q4(layer->delta_in_proj_b_q4, layer->delta_in_proj_b_q4s,
                         delta_dn, dim);
            TQM_READ_FP32(layer->delta_conv1d, delta_conv_total * delta_conv_width);
            TQM_READ_FP32(layer->delta_norm, delta_vhd);
            TQM_READ_Q4(layer->delta_out_proj_q4, layer->delta_out_proj_q4s,
                         dim, delta_z_dim);
        }

        /* FFN (all layers) */
        TQM_READ_Q4(layer->w_gate_q4, layer->w_gate_q4s, inter, dim);
        TQM_READ_Q4(layer->w_up_q4, layer->w_up_q4s, inter, dim);
        TQM_READ_Q4(layer->w_down_q4, layer->w_down_q4s, dim, inter);
    }

    /* Output norm */
    TQM_READ_FP32(model->output_norm, dim);

    /* Embedding and output as BF16 */
    {
        size_t embed_elems = (size_t)c->vocab_size * dim;
        size_t embed_bytes = embed_elems * 2; /* BF16 = 2 bytes */
        if (ptr + embed_bytes > file_end) goto fail_tqm_model;
        model->embed_bf16 = (const uint16_t*)ptr;
        model->token_embedding = NULL;
        ptr += embed_bytes;

        /* Check if output weight is tied (offset == 0 means tied) */
        if (ptr + embed_bytes <= file_end) {
            /* Separate lm_head */
            model->output_weight_bf16 = (const uint16_t*)ptr;
            model->output_weight = NULL;
            ptr += embed_bytes;
        } else {
            /* Tied: output_weight points to embed_tokens */
            model->output_weight_bf16 = model->embed_bf16;
            model->output_weight = NULL;
        }
    }

    #undef TQM_READ_FP32
    #undef TQM_READ_Q4

    model->use_q4_weights = 1;
    free(is_attn_layer);

    fprintf(stderr, "tq_load_tqm: loaded %d layers (%d self_attn), "
            "dim=%d, heads=%d/%d, vocab=%d [%.1f MB, mmap zero-copy]\n",
            c->n_layers, model->n_attn_layers, dim,
            c->n_heads, c->n_kv_heads, c->vocab_size,
            (double)file_size / (1024.0 * 1024.0));

    return model;

fail_tqm_model:
    fprintf(stderr, "tq_load_tqm: weight data extends past end of file\n");
    free(is_attn_layer);
    free(model->attn_layer_indices);
    free(model->layers);
    free(model);

fail_tqm:
#ifdef _WIN32
    if (mmap_data) UnmapViewOfFile(mmap_data);
#else
    if (mmap_data && mmap_data != MAP_FAILED) munmap(mmap_data, file_size);
#endif
    return NULL;
}

/* ============================================================
 * TQM saver — write pre-quantized model to .tqm file
 *
 * The model MUST have Q4 weights (use_q4_weights == 1) before
 * calling this function.  Call tq_quantize_weights_q4() first.
 * ============================================================ */

/* Helper: write with 64-byte alignment padding */
static int tqm_write_pad(FILE* f, uint64_t current_offset) {
    uint64_t aligned = tqm_align(current_offset);
    if (aligned > current_offset) {
        uint64_t pad_size = aligned - current_offset;
        uint8_t zeros[64] = {0};
        while (pad_size > 0) {
            size_t chunk = pad_size > 64 ? 64 : (size_t)pad_size;
            if (fwrite(zeros, 1, chunk, f) != chunk) return -1;
            pad_size -= chunk;
        }
    }
    return 0;
}

int tq_save_tqm(tq_model_t* model, const char* tokenizer_path,
                const char* output_path) {
    if (!model || !output_path) return -1;
    if (!model->use_q4_weights) {
        fprintf(stderr, "tq_save_tqm: model must be Q4-quantized first\n");
        return -1;
    }

    const tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;
    int delta_qkv_dim = 3 * c->delta_n_heads * c->delta_key_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;
    int delta_conv_total = delta_qkv_dim;
    int delta_conv_width = c->delta_conv_width;
    int delta_vhd = c->delta_value_head_dim;

    /* Build is-attn lookup */
    int* is_attn_layer = (int*)calloc((size_t)c->n_layers, sizeof(int));
    if (!is_attn_layer) return -1;
    for (int i = 0; i < model->n_attn_layers; i++) {
        int idx = model->attn_layer_indices[i];
        if (idx >= 0 && idx < c->n_layers) is_attn_layer[idx] = 1;
    }

    /* Read tokenizer file if provided */
    char* tok_data = NULL;
    size_t tok_size = 0;
    if (tokenizer_path) {
        FILE* tf = fopen(tokenizer_path, "rb");
        if (tf) {
            fseek(tf, 0, SEEK_END);
            tok_size = (size_t)ftell(tf);
            fseek(tf, 0, SEEK_SET);
            tok_data = (char*)malloc(tok_size);
            if (tok_data) {
                size_t nr = fread(tok_data, 1, tok_size, tf);
                if (nr != tok_size) {
                    free(tok_data);
                    tok_data = NULL;
                    tok_size = 0;
                }
            }
            fclose(tf);
        } else {
            fprintf(stderr, "tq_save_tqm: warning: cannot open tokenizer '%s'\n",
                    tokenizer_path);
        }
    }

    FILE* f = fopen(output_path, "wb");
    if (!f) {
        fprintf(stderr, "tq_save_tqm: cannot create '%s'\n", output_path);
        free(tok_data);
        free(is_attn_layer);
        return -1;
    }

    /* Compute section offsets */
    uint64_t tok_offset = tqm_align(sizeof(tqm_header_t));
    uint64_t wt_offset = tqm_align(tok_offset + tok_size);

    /* Build and write header */
    tqm_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic   = TQM_MAGIC;
    hdr.version = TQM_VERSION;

    hdr.n_layers         = c->n_layers;
    hdr.hidden_dim       = c->hidden_dim;
    hdr.intermediate_dim = c->intermediate_dim;
    hdr.n_heads          = c->n_heads;
    hdr.n_kv_heads       = c->n_kv_heads;
    hdr.head_dim         = c->head_dim;
    hdr.vocab_size       = c->vocab_size;
    hdr.max_seq_len      = c->max_seq_len;
    hdr.rope_freq_base   = c->rope_freq_base;
    hdr.rms_norm_eps     = c->rms_norm_eps;

    hdr.delta_n_heads       = c->delta_n_heads;
    hdr.delta_key_head_dim  = c->delta_key_head_dim;
    hdr.delta_value_head_dim= c->delta_value_head_dim;
    hdr.delta_conv_width    = c->delta_conv_width;
    hdr.partial_rotary_factor = c->partial_rotary_factor;
    hdr.use_qk_norm         = c->use_qk_norm;
    hdr.attn_output_gate    = c->attn_output_gate;

    hdr.weight_quant = 4; /* Q4 */
    hdr.embed_format = 16; /* BF16 */

    hdr.tokenizer_offset = tok_offset;
    hdr.tokenizer_size   = tok_size;
    hdr.weights_offset   = wt_offset;
    /* weights_size filled after writing */

    hdr.n_attn_layers = model->n_attn_layers;
    for (int i = 0; i < model->n_attn_layers && i < 64; i++) {
        hdr.attn_layer_indices[i] = model->attn_layer_indices[i];
    }

    fwrite(&hdr, sizeof(hdr), 1, f);

    /* Pad to tokenizer offset */
    {
        uint64_t cur = sizeof(tqm_header_t);
        tqm_write_pad(f, cur);
    }

    /* Write tokenizer */
    if (tok_data && tok_size > 0) {
        fwrite(tok_data, 1, tok_size, f);
    }
    free(tok_data);

    /* Pad to weights offset */
    {
        uint64_t cur = tok_offset + tok_size;
        tqm_write_pad(f, cur);
    }

    /* Helper macros for writing */
    #define TQM_WRITE_FP32(src, count) do {                            \
        if ((src)) fwrite((src), sizeof(float), (size_t)(count), f);   \
        else { float _z = 0; for (int _i = 0; _i < (count); _i++)     \
            fwrite(&_z, sizeof(float), 1, f); }                        \
    } while (0)

    #define TQM_WRITE_Q4(qs, sc, rows, cols) do {                      \
        int _nb = ((cols) + 31) / 32;                                   \
        size_t _qs_bytes = (size_t)(rows) * _nb * 16;                   \
        size_t _sc_bytes = (size_t)(rows) * _nb * sizeof(float);        \
        if ((qs)) fwrite((qs), 1, _qs_bytes, f);                       \
        else { uint8_t _z = 0; for (size_t _i = 0; _i < _qs_bytes; _i++) \
            fwrite(&_z, 1, 1, f); }                                    \
        if ((sc)) fwrite((sc), 1, _sc_bytes, f);                       \
        else { float _zf = 0; for (size_t _i = 0;                      \
            _i < (size_t)(rows) * _nb; _i++) fwrite(&_zf, sizeof(float), 1, f); } \
    } while (0)

    /* Write per-layer weights */
    for (int l = 0; l < c->n_layers; l++) {
        const tq_layer_weights_t* layer = &model->layers[l];

        TQM_WRITE_FP32(layer->attn_norm, dim);
        TQM_WRITE_FP32(layer->ffn_norm, dim);

        if (is_attn_layer[l]) {
            TQM_WRITE_Q4(layer->wq_q4, layer->wq_q4s, qg_dim, dim);
            TQM_WRITE_Q4(layer->wk_q4, layer->wk_q4s, kv_dim, dim);
            TQM_WRITE_Q4(layer->wv_q4, layer->wv_q4s, kv_dim, dim);
            TQM_WRITE_Q4(layer->wo_q4, layer->wo_q4s, dim, q_dim);

            if (c->use_qk_norm) {
                TQM_WRITE_FP32(layer->q_norm, c->head_dim);
                TQM_WRITE_FP32(layer->k_norm, c->head_dim);
            }
        } else {
            TQM_WRITE_FP32(layer->delta_a_log, delta_dn);
            TQM_WRITE_FP32(layer->delta_dt_bias, delta_dn);
            TQM_WRITE_Q4(layer->delta_in_proj_qkv_q4, layer->delta_in_proj_qkv_q4s,
                          delta_qkv_dim, dim);
            TQM_WRITE_Q4(layer->delta_in_proj_z_q4, layer->delta_in_proj_z_q4s,
                          delta_z_dim, dim);
            TQM_WRITE_Q4(layer->delta_in_proj_a_q4, layer->delta_in_proj_a_q4s,
                          delta_dn, dim);
            TQM_WRITE_Q4(layer->delta_in_proj_b_q4, layer->delta_in_proj_b_q4s,
                          delta_dn, dim);
            TQM_WRITE_FP32(layer->delta_conv1d, delta_conv_total * delta_conv_width);
            TQM_WRITE_FP32(layer->delta_norm, delta_vhd);
            TQM_WRITE_Q4(layer->delta_out_proj_q4, layer->delta_out_proj_q4s,
                          dim, delta_z_dim);
        }

        TQM_WRITE_Q4(layer->w_gate_q4, layer->w_gate_q4s, inter, dim);
        TQM_WRITE_Q4(layer->w_up_q4, layer->w_up_q4s, inter, dim);
        TQM_WRITE_Q4(layer->w_down_q4, layer->w_down_q4s, dim, inter);
    }

    /* Output norm */
    TQM_WRITE_FP32(model->output_norm, dim);

    /* Embedding BF16 */
    {
        size_t embed_elems = (size_t)c->vocab_size * dim;
        if (model->embed_bf16) {
            fwrite(model->embed_bf16, 2, embed_elems, f);
        } else if (model->token_embedding) {
            /* Convert FP32 to BF16 on the fly */
            for (size_t i = 0; i < embed_elems; i++) {
                uint32_t fp32;
                memcpy(&fp32, &model->token_embedding[i], sizeof(float));
                uint16_t bf16 = (uint16_t)(fp32 >> 16);
                fwrite(&bf16, 2, 1, f);
            }
        } else {
            /* Write zeros */
            uint16_t z = 0;
            for (size_t i = 0; i < embed_elems; i++) fwrite(&z, 2, 1, f);
        }

        /* Output weight (lm_head) — write if different from embed */
        if (model->output_weight_bf16 &&
            model->output_weight_bf16 != model->embed_bf16) {
            fwrite(model->output_weight_bf16, 2, embed_elems, f);
        } else if (model->output_weight &&
                   model->output_weight != model->token_embedding) {
            for (size_t i = 0; i < embed_elems; i++) {
                uint32_t fp32;
                memcpy(&fp32, &model->output_weight[i], sizeof(float));
                uint16_t bf16 = (uint16_t)(fp32 >> 16);
                fwrite(&bf16, 2, 1, f);
            }
        }
        /* If tied, don't write — loader will detect from file size */
    }

    #undef TQM_WRITE_FP32
    #undef TQM_WRITE_Q4

    /* Update weights_size in header */
    long end_pos = ftell(f);
    hdr.weights_size = (uint64_t)end_pos - wt_offset;
    fseek(f, 0, SEEK_SET);
    fwrite(&hdr, sizeof(hdr), 1, f);

    fclose(f);
    free(is_attn_layer);

    fprintf(stderr, "tq_save_tqm: saved '%s' (%.1f MB)\n",
            output_path, (double)end_pos / (1024.0 * 1024.0));
    return 0;
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
    free(model->_q8_data);
    free(model->_q4_data);
    free(model->attn_layer_indices);
    free(model->layers);
    free(model);
}
