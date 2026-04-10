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
#include "turboquant/tq_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#ifndef R_OK
#define R_OK 4
#endif
#define access _access
/* Minimal dirent.h for MSVC (FindFirstFile-based) */
#ifndef _DIRENT_H_WIN32
#define _DIRENT_H_WIN32
struct dirent { char d_name[260]; };
typedef struct { HANDLE h; WIN32_FIND_DATAA fd; int first; } DIR;
static DIR* opendir(const char* p) {
    char buf[270]; snprintf(buf, sizeof(buf), "%s\\*", p);
    DIR* d = (DIR*)malloc(sizeof(DIR));
    d->h = FindFirstFileA(buf, &d->fd); d->first = 1;
    if (d->h == INVALID_HANDLE_VALUE) { free(d); return NULL; }
    return d;
}
static struct dirent* readdir(DIR* d) {
    static struct dirent e;
    if (d->first) { d->first = 0; } else if (!FindNextFileA(d->h, &d->fd)) return NULL;
    strncpy(e.d_name, d->fd.cFileName, 259); e.d_name[259] = 0;
    return &e;
}
static void closedir(DIR* d) { FindClose(d->h); free(d); }
#endif
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
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
    void* data_base;  /* base pointer for this tensor's shard data region */
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
            char alt[MAX_NAME_LEN * 2];
            snprintf(alt, sizeof(alt), "model.%s%s", prefix, name + 6);
            for (int i = 0; i < n; i++) {
                if (strcmp(tensors[i].name, alt) == 0) {
                    return &tensors[i];
                }
            }
        }
    }
    /* Try "language_model." prefix: Gemma 4B uses
     * "language_model.model.layers.N.foo" instead of "model.layers.N.foo".
     * Also try "language_model.lm_head.weight" for "lm_head.weight". */
    {
        char alt[MAX_NAME_LEN * 2];
        snprintf(alt, sizeof(alt), "language_model.%s", name);
        for (int i = 0; i < n; i++) {
            if (strcmp(tensors[i].name, alt) == 0) {
                return &tensors[i];
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
static float* convert_tensor_to_fp32(tensor_info_t* t,
                                      float** conv_buf, size_t* conv_used,
                                      size_t conv_capacity) {
    if (!t) return NULL;

    const void* src = (const char*)t->data_base + t->data_start;
    int64_t data_bytes = t->data_end - t->data_start;
    int elem_size = dtype_element_size(t->dtype);
    if (elem_size == 0) return NULL;
    int64_t n_elements = data_bytes / elem_size;

    if (strcmp(t->dtype, "F32") == 0) {
        /* Zero-copy: return direct pointer into mmap */
        return (float*)((char*)t->data_base + t->data_start);
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
static float* load_tensor(tensor_info_t* t,
                           float** conv_buf, size_t* conv_used,
                           size_t conv_capacity) {
    if (!t) return NULL;
    return convert_tensor_to_fp32(t, conv_buf, conv_used, conv_capacity);
}

/* ============================================================
 * Get raw BF16 pointer for a tensor (zero-copy from mmap).
 * Returns NULL if tensor is not BF16.
 * ============================================================ */
static const uint16_t* get_bf16_ptr(tensor_info_t* t) {
    if (!t) return NULL;
    if (strcmp(t->dtype, "BF16") != 0) return NULL;
    return (const uint16_t*)((const char*)t->data_base + t->data_start);
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
    if (strcmp(name, "language_model.model.embed_tokens.weight") == 0) return 1;
    if (strcmp(name, "tok_embeddings.weight") == 0) return 1;
    if (strcmp(name, "transformer.wte.weight") == 0) return 1;
    /* lm_head — output projection, can use BF16 matmul */
    if (strcmp(name, "lm_head.weight") == 0) return 1;
    if (strcmp(name, "language_model.lm_head.weight") == 0) return 1;
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
        /* Try Gemma 4B prefix: language_model.model.layers.N. */
        else if (sscanf(name, "language_model.model.layers.%d.", &layer_idx) == 1) {
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
         * or language_model.model.layers.N. (Gemma 4B).
         * Skip other prefixes like mtp.layers.N. */
        const char* layers_pos = NULL;
        if (strncmp(name, "model.layers.", 13) == 0) {
            layers_pos = name + 13;
        } else if (strncmp(name, "model.language_model.layers.", 28) == 0) {
            layers_pos = name + 28;
        } else if (strncmp(name, "language_model.model.layers.", 28) == 0) {
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

    /* Check magic bytes to detect format */
    FILE* probe = fopen(path, "rb");
    if (probe) {
        uint32_t magic = 0;
        if (fread(&magic, 4, 1, probe) == 1) {
            fclose(probe);
            if (magic == TQM_MAGIC) {
                fprintf(stderr, "tq_load_model: detected TQM format, using fast loader\n");
                return tq_load_tqm(path);
            }
            if (magic == 0x46554747) { /* "GGUF" */
                fprintf(stderr, "tq_load_model: detected GGUF format\n");
                return tq_load_gguf(path);
            }
        } else {
            fclose(probe);
        }
    }

    /* Fall through to safetensors loader */
    return tq_load_safetensors(path);
}

/* ============================================================
 * Helper: mmap a file, returning pointer and size.
 * Returns NULL on failure.
 * ============================================================ */
static void* mmap_file(const char* path, size_t* out_size) {
#ifdef _WIN32
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "tq_load_model: cannot open '%s'\n", path);
        return NULL;
    }
    LARGE_INTEGER fileSize;
    GetFileSizeEx(hFile, &fileSize);
    *out_size = (size_t)fileSize.QuadPart;
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) { CloseHandle(hFile); return NULL; }
    void* data = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    CloseHandle(hFile);
    return data;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "tq_load_model: cannot open '%s'\n", path);
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return NULL; }
    *out_size = (size_t)st.st_size;
    void* data = mmap(NULL, *out_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) {
        fprintf(stderr, "tq_load_model: mmap failed for '%s'\n", path);
        return NULL;
    }
    return data;
#endif
}

/* ============================================================
 * Helper: parse a single safetensors shard.
 * Mmaps the file, parses the header, and appends tensors to the
 * provided tensor array. Sets data_base on each tensor.
 * Returns number of tensors added, or -1 on error.
 * mmap_out/mmap_size_out receive the mmap pointer for the caller to manage.
 * ============================================================ */
static int parse_shard(const char* shard_path,
                       tensor_info_t* tensors, int existing_count, int max_tensors,
                       void** mmap_out, size_t* mmap_size_out) {
    size_t file_size = 0;
    void* mmap_data = mmap_file(shard_path, &file_size);
    if (!mmap_data) return -1;

    if (file_size < 8) {
        fprintf(stderr, "tq_load_model: shard too small: '%s'\n", shard_path);
#ifdef _WIN32
        UnmapViewOfFile(mmap_data);
#else
        munmap(mmap_data, file_size);
#endif
        return -1;
    }

    uint64_t header_size = 0;
    memcpy(&header_size, mmap_data, 8);
    if (8 + header_size > file_size) {
        fprintf(stderr, "tq_load_model: invalid header in shard '%s'\n", shard_path);
#ifdef _WIN32
        UnmapViewOfFile(mmap_data);
#else
        munmap(mmap_data, file_size);
#endif
        return -1;
    }

    const char* json = (const char*)mmap_data + 8;
    void* data_base = (char*)mmap_data + 8 + header_size;

    int space = max_tensors - existing_count;
    if (space <= 0) {
#ifdef _WIN32
        UnmapViewOfFile(mmap_data);
#else
        munmap(mmap_data, file_size);
#endif
        return -1;
    }

    int n = parse_safetensors_header(json, (int64_t)header_size,
                                     tensors + existing_count, space);
    if (n < 0) {
        fprintf(stderr, "tq_load_model: JSON parse error in shard '%s'\n", shard_path);
#ifdef _WIN32
        UnmapViewOfFile(mmap_data);
#else
        munmap(mmap_data, file_size);
#endif
        return -1;
    }

    /* Set data_base for each newly parsed tensor */
    for (int i = existing_count; i < existing_count + n; i++) {
        tensors[i].data_base = data_base;
    }

    *mmap_out = mmap_data;
    *mmap_size_out = file_size;

    fprintf(stderr, "tq_load_model: shard '%s' — %d tensors, header=%llu bytes\n",
            shard_path, n, (unsigned long long)header_size);
    return n;
}

/* ============================================================
 * Helper: compare strings for qsort (used to sort shard filenames)
 * ============================================================ */
static int cmp_strings(const void* a, const void* b) {
    return strcmp(*(const char**)a, *(const char**)b);
}

/* ============================================================
 * Helper: find all shard files in the same directory as `path`.
 * If path points to a single model.safetensors, check for
 * model.safetensors.index.json to detect multi-shard.
 * If path is a shard like model-00001-of-00002.safetensors,
 * find all matching shards.
 * Returns number of shard paths, or 0 for single-file mode.
 * shard_paths[] is filled with allocated strings (caller must free).
 * ============================================================ */
static int find_shard_files(const char* path, char** shard_paths, int max_shards) {
    /* Extract directory and basename */
    char dir_path[2048];
    strncpy(dir_path, path, sizeof(dir_path) - 1);
    dir_path[sizeof(dir_path) - 1] = '\0';
    char* last_slash = strrchr(dir_path, '/');
    const char* basename = path;
    if (last_slash) {
        *last_slash = '\0';
        basename = last_slash + 1;
    } else {
        strcpy(dir_path, ".");
    }

    /* Check 1: path is "model.safetensors" and index file exists */
    if (strcmp(basename, "model.safetensors") == 0) {
        char index_path[2048];
        snprintf(index_path, sizeof(index_path), "%s/model.safetensors.index.json",
                 dir_path);
        if (access(index_path, R_OK) == 0) {
            /* Multi-shard: find all model-NNNNN-of-NNNNN.safetensors */
            DIR* dir = opendir(dir_path);
            if (!dir) return 0;

            int count = 0;
            struct dirent* ent;
            while ((ent = readdir(dir)) != NULL && count < max_shards) {
                /* Match pattern: model-XXXXX-of-XXXXX.safetensors */
                if (strncmp(ent->d_name, "model-", 6) == 0 &&
                    strstr(ent->d_name, "-of-") != NULL &&
                    strstr(ent->d_name, ".safetensors") != NULL &&
                    strstr(ent->d_name, ".index.") == NULL) {
                    char full[2048];
                    snprintf(full, sizeof(full), "%s/%s", dir_path, ent->d_name);
                    shard_paths[count] = strdup(full);
                    count++;
                }
            }
            closedir(dir);

            /* Sort shard paths alphabetically so they load in order */
            if (count > 1) {
                qsort(shard_paths, (size_t)count, sizeof(char*), cmp_strings);
            }

            if (count > 0) {
                fprintf(stderr, "tq_load_model: detected %d shard files\n", count);
            }
            return count;
        }
    }

    /* Check 2: path itself is a shard file (contains "-of-") */
    if (strstr(basename, "-of-") != NULL &&
        strstr(basename, ".safetensors") != NULL) {
        /* Extract prefix before the shard number: e.g., "model-" */
        const char* dash_of = strstr(basename, "-of-");
        /* Walk back to find the dash before the shard number */
        const char* num_start = dash_of;
        while (num_start > basename && *(num_start - 1) != '-') num_start--;
        size_t prefix_len = (size_t)(num_start - basename);

        DIR* dir = opendir(dir_path);
        if (!dir) return 0;

        int count = 0;
        struct dirent* ent;
        while ((ent = readdir(dir)) != NULL && count < max_shards) {
            if (strncmp(ent->d_name, basename, prefix_len) == 0 &&
                strstr(ent->d_name, "-of-") != NULL &&
                strstr(ent->d_name, ".safetensors") != NULL &&
                strstr(ent->d_name, ".index.") == NULL) {
                char full[2048];
                snprintf(full, sizeof(full), "%s/%s", dir_path, ent->d_name);
                shard_paths[count] = strdup(full);
                count++;
            }
        }
        closedir(dir);

        if (count > 1) {
            qsort(shard_paths, (size_t)count, sizeof(char*), cmp_strings);
            fprintf(stderr, "tq_load_model: detected %d shard files\n", count);
            return count;
        }

        /* Only 1 shard found = effectively single file, clean up */
        for (int i = 0; i < count; i++) free(shard_paths[i]);
        return 0;
    }

    return 0; /* single file mode */
}

/* ============================================================
 * Load model from safetensors file (single or multi-shard)
 * ============================================================ */
static tq_model_t* tq_load_safetensors(const char* path) {
    if (!path) return NULL;

    /* Detect multi-shard */
    char* shard_paths[TQ_MAX_SHARDS];
    memset(shard_paths, 0, sizeof(shard_paths));
    int n_shards = find_shard_files(path, shard_paths, TQ_MAX_SHARDS);

    /* Parse tensors from all shards (or single file) */
    tensor_info_t* tensors = (tensor_info_t*)calloc(MAX_TENSORS, sizeof(tensor_info_t));
    if (!tensors) {
        for (int i = 0; i < n_shards; i++) free(shard_paths[i]);
        return NULL;
    }

    void* mmap_ptrs[TQ_MAX_SHARDS];
    size_t mmap_sizes[TQ_MAX_SHARDS];
    memset(mmap_ptrs, 0, sizeof(mmap_ptrs));
    memset(mmap_sizes, 0, sizeof(mmap_sizes));
    int total_shards_loaded = 0;
    int n_tensors = 0;

    if (n_shards > 0) {
        /* Multi-shard loading */
        for (int s = 0; s < n_shards; s++) {
            int added = parse_shard(shard_paths[s], tensors, n_tensors, MAX_TENSORS,
                                    &mmap_ptrs[s], &mmap_sizes[s]);
            if (added < 0) {
                fprintf(stderr, "tq_load_model: failed to parse shard '%s'\n", shard_paths[s]);
                /* Clean up already-loaded shards */
                for (int j = 0; j < s; j++) {
#ifdef _WIN32
                    if (mmap_ptrs[j]) UnmapViewOfFile(mmap_ptrs[j]);
#else
                    if (mmap_ptrs[j]) munmap(mmap_ptrs[j], mmap_sizes[j]);
#endif
                }
                for (int j = 0; j < n_shards; j++) free(shard_paths[j]);
                free(tensors);
                return NULL;
            }
            n_tensors += added;
            total_shards_loaded++;
        }
        for (int i = 0; i < n_shards; i++) free(shard_paths[i]);
    } else {
        /* Single file loading */
        int added = parse_shard(path, tensors, 0, MAX_TENSORS,
                                &mmap_ptrs[0], &mmap_sizes[0]);
        if (added < 0) {
            free(tensors);
            return NULL;
        }
        n_tensors = added;
        total_shards_loaded = 1;
    }

    fprintf(stderr, "tq_load_model: parsed %d tensors from %d file(s)\n",
            n_tensors, total_shards_loaded);

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
    model->_mmap_data = mmap_ptrs[0];
    model->_mmap_size = mmap_sizes[0];
    model->_n_shards = total_shards_loaded;
    for (int i = 1; i < total_shards_loaded; i++) {
        model->_mmap_shards[i] = mmap_ptrs[i];
        model->_mmap_shard_sizes[i] = mmap_sizes[i];
    }

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
    model->embed_bf16 = get_bf16_ptr(embed);
    if (model->embed_bf16) {
        /* BF16 embedding: don't convert, will convert on demand in forward pass */
        model->token_embedding = NULL;
        size_t embed_bytes = (size_t)embed->shape[0] * embed->shape[1] * 2;
        fprintf(stderr, "tq_load_model: keeping embed_tokens as BF16 (%zu MB saved)\n",
                embed_bytes / (1024 * 1024));
    } else {
        /* F32 or other dtype: convert as before */
        model->token_embedding = load_tensor( embed,
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
    model->config.delta_n_kv_heads = 0;
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
            /* Detect value_head_dim from norm weight shape */
            snprintf(name_buf, sizeof(name_buf),
                     "model.layers.%d.linear_attn.norm.weight", delta_layer);
            tensor_info_t* norm_w = find_tensor(tensors, n_tensors, name_buf);
            if (norm_w) {
                model->config.delta_value_head_dim = (int)norm_w->shape[0];
            }

            /* Detect key_head_dim from z proj: z_dim = n_v_heads * val_dim
             * qkv_dim = n_kv_heads * key_dim * 2 + n_v_heads * val_dim
             * So: n_kv_heads * key_dim = (qkv_dim - z_dim) / 2 */
            snprintf(name_buf, sizeof(name_buf),
                     "model.layers.%d.linear_attn.in_proj_z.weight", delta_layer);
            tensor_info_t* z_proj = find_tensor(tensors, n_tensors, name_buf);

            if (qkv_proj && model->config.delta_n_heads > 0) {
                int qkv_dim = (int)qkv_proj->shape[0];
                int z_dim = z_proj ? (int)z_proj->shape[0] : 0;
                int val_dim = model->config.delta_value_head_dim;

                if (z_dim > 0 && val_dim > 0) {
                    /* z_dim = n_v_heads * val_dim → already known */
                    /* qkv = n_kv_heads * key_dim * 2 + z_dim */
                    int qk_dim = qkv_dim - z_dim;  /* Q+K total */
                    /* Try: key_dim = val_dim (common), then n_kv_heads = qk_dim / (2 * key_dim) */
                    int key_dim = val_dim; /* start with same as val */
                    int n_kv = qk_dim / (2 * key_dim);
                    if (n_kv * 2 * key_dim == qk_dim) {
                        model->config.delta_key_head_dim = key_dim;
                        model->config.delta_n_kv_heads = n_kv;
                    } else {
                        /* Fallback: assume n_kv_heads = n_v_heads */
                        model->config.delta_key_head_dim = qkv_dim / (3 * model->config.delta_n_heads);
                    }
                } else {
                    /* Original logic for models where Q/K/V heads are equal */
                    model->config.delta_key_head_dim = qkv_dim / (3 * model->config.delta_n_heads);
                }
                if (model->config.delta_value_head_dim == 0)
                    model->config.delta_value_head_dim = model->config.delta_key_head_dim;
            }

            snprintf(name_buf, sizeof(name_buf),
                     "model.layers.%d.linear_attn.conv1d.weight", delta_layer);
            tensor_info_t* conv = find_tensor(tensors, n_tensors, name_buf);
            if (conv && conv->n_dims >= 3) {
                model->config.delta_conv_width = (int)conv->shape[2];
            }

            int delta_nkv = model->config.delta_n_kv_heads > 0 ?
                            model->config.delta_n_kv_heads : model->config.delta_n_heads;
            fprintf(stderr, "tq_load_model: DeltaNet config — v_heads=%d, kv_heads=%d, "
                    "key_dim=%d, val_dim=%d, conv_w=%d\n",
                    model->config.delta_n_heads, delta_nkv,
                    model->config.delta_key_head_dim,
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

    /* Detect Gemma3 architecture by presence of pre_feedforward_layernorm */
    {
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.0.pre_feedforward_layernorm.weight");
        tensor_info_t* gemma3_probe = find_tensor(tensors, n_tensors, name_buf);
        if (gemma3_probe) {
            model->config.model_type = 1; /* gemma3 */
            model->config.n_norms_per_block = 4;
            fprintf(stderr, "tq_load_model: detected Gemma3 architecture (4 norms per block)\n");
        } else {
            model->config.model_type = 0; /* qwen35 */
            model->config.n_norms_per_block = 2;
        }
    }

    /* Defaults — tuned for Qwen3.5 if DeltaNet detected */
    model->config.max_seq_len = 4096;
    if (model->config.model_type == 1) {
        /* Gemma3: rope_theta=1M for global, 10K for local, rms_norm_eps=1e-6 */
        model->config.rope_freq_base = 1000000.0f; /* global layers */
        model->config.rope_local_base_freq = 10000.0f; /* sliding/local layers */
        model->config.rms_norm_eps = 1e-6f;
        model->config.partial_rotary_factor = 0.0f;
        model->config.sliding_window = 512;
        model->config.query_pre_attn_scalar = 256.0f;
    } else if (model->config.delta_n_heads > 0) {
        /* Qwen3.5 uses rope_theta=10M, rms_norm_eps=1e-6, partial_rotary=0.25 */
        model->config.rope_freq_base = 10000000.0f;
        model->config.rms_norm_eps = 1e-6f;
        model->config.partial_rotary_factor = 0.25f;
        model->config.query_pre_attn_scalar = 0.0f;
    } else {
        model->config.rope_freq_base = 10000.0f;
        model->config.rms_norm_eps = 1e-5f;
        model->config.partial_rotary_factor = 0.0f;
        model->config.query_pre_attn_scalar = 0.0f;
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
        layer->attn_norm = load_tensor(
                                        find_tensor(tensors, n_tensors, name_buf),
                                        &conv_buf, &conv_used, conv_capacity);

        /* FFN norm (Qwen3.5: post_attention_layernorm used as pre-FFN norm) */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.post_attention_layernorm.weight", l);
        layer->ffn_norm = load_tensor(
                                       find_tensor(tensors, n_tensors, name_buf),
                                       &conv_buf, &conv_used, conv_capacity);

        /* Gemma3 extra norms: post_attn, pre_ffn, post_ffn */
        if (model->config.model_type == 1) {
            /* For Gemma3, post_attention_layernorm is applied to attn output,
             * not as pre-FFN norm. Store it in post_attn_norm. */
            layer->post_attn_norm = layer->ffn_norm;

            snprintf(name_buf, sizeof(name_buf),
                     "model.layers.%d.pre_feedforward_layernorm.weight", l);
            layer->pre_ffn_norm = load_tensor(
                                               find_tensor(tensors, n_tensors, name_buf),
                                               &conv_buf, &conv_used, conv_capacity);

            snprintf(name_buf, sizeof(name_buf),
                     "model.layers.%d.post_feedforward_layernorm.weight", l);
            layer->post_ffn_norm = load_tensor(
                                                find_tensor(tensors, n_tensors, name_buf),
                                                &conv_buf, &conv_used, conv_capacity);
        }

        /* Q, K, V, O projections — only exist for self_attn layers */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.q_proj.weight", l);
        layer->wq = load_tensor(
                                 find_tensor(tensors, n_tensors, name_buf),
                                 &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.k_proj.weight", l);
        layer->wk = load_tensor(
                                 find_tensor(tensors, n_tensors, name_buf),
                                 &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.v_proj.weight", l);
        layer->wv = load_tensor(
                                 find_tensor(tensors, n_tensors, name_buf),
                                 &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.o_proj.weight", l);
        layer->wo = load_tensor(
                                 find_tensor(tensors, n_tensors, name_buf),
                                 &conv_buf, &conv_used, conv_capacity);

        /* QK-norm weights (Qwen3.5 style) */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.q_norm.weight", l);
        layer->q_norm = load_tensor(
                                     find_tensor(tensors, n_tensors, name_buf),
                                     &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.k_norm.weight", l);
        layer->k_norm = load_tensor(
                                     find_tensor(tensors, n_tensors, name_buf),
                                     &conv_buf, &conv_used, conv_capacity);

        /* DeltaNet (linear_attention) weights */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.A_log", l);
        layer->delta_a_log = load_tensor(
                                          find_tensor(tensors, n_tensors, name_buf),
                                          &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.conv1d.weight", l);
        layer->delta_conv1d = load_tensor(
                                           find_tensor(tensors, n_tensors, name_buf),
                                           &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.dt_bias", l);
        layer->delta_dt_bias = load_tensor(
                                            find_tensor(tensors, n_tensors, name_buf),
                                            &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.in_proj_a.weight", l);
        layer->delta_in_proj_a = load_tensor(
                                              find_tensor(tensors, n_tensors, name_buf),
                                              &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.in_proj_b.weight", l);
        layer->delta_in_proj_b = load_tensor(
                                              find_tensor(tensors, n_tensors, name_buf),
                                              &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.in_proj_qkv.weight", l);
        layer->delta_in_proj_qkv = load_tensor(
                                                find_tensor(tensors, n_tensors, name_buf),
                                                &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.in_proj_z.weight", l);
        layer->delta_in_proj_z = load_tensor(
                                              find_tensor(tensors, n_tensors, name_buf),
                                              &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.norm.weight", l);
        layer->delta_norm = load_tensor(
                                         find_tensor(tensors, n_tensors, name_buf),
                                         &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.out_proj.weight", l);
        layer->delta_out_proj = load_tensor(
                                             find_tensor(tensors, n_tensors, name_buf),
                                             &conv_buf, &conv_used, conv_capacity);

        /* Reorder DeltaNet V-head weights to GGUF tiled order.
         * Original: [h0,h1,h2,...,h31] (sequential V-heads)
         * Tiled:    [h0,h2,h4,...,h30, h1,h3,...,h31] (even first, odd second)
         * This matches ggml broadcast convention for GQA DeltaNet.
         * Only needed when delta_n_kv_heads < delta_n_heads (asymmetric). */
        if (model->config.delta_n_kv_heads > 0 && model->config.delta_n_kv_heads < model->config.delta_n_heads
            && layer->delta_a_log) {
            int dim = model->config.hidden_dim;
            int dn = model->config.delta_n_heads;
            int nkv = model->config.delta_n_kv_heads;
            if (l == 0) fprintf(stderr, "tq_load_model: reordering DeltaNet V-heads (dn=%d, nkv=%d) to tiled order\n", dn, nkv);
            int kv_mul = dn / nkv;          /* 2 V-heads per K-group */
            int dk = model->config.delta_key_head_dim;
            int dv = model->config.delta_value_head_dim;

            /* Build tiled permutation: for each K-group, collect its V-heads */
            int perm[256]; /* max heads */
            for (int pass = 0; pass < kv_mul; pass++)
                for (int g = 0; g < nkv; g++)
                    perm[pass * nkv + g] = g * kv_mul + pass;

            /* Reorder per-head vectors: a_log, dt_bias */
            float* tmp = (float*)malloc((size_t)dn * sizeof(float));
            if (tmp && layer->delta_a_log) {
                for (int h = 0; h < dn; h++) tmp[h] = layer->delta_a_log[perm[h]];
                memcpy(layer->delta_a_log, tmp, (size_t)dn * sizeof(float));
            }
            if (tmp && layer->delta_dt_bias) {
                for (int h = 0; h < dn; h++) tmp[h] = layer->delta_dt_bias[perm[h]];
                memcpy(layer->delta_dt_bias, tmp, (size_t)dn * sizeof(float));
            }

            /* Reorder per-head row matrices: in_proj_a [dn, dim], in_proj_b [dn, dim] */
            float* row_tmp = (float*)malloc((size_t)dn * dim * sizeof(float));
            if (row_tmp && layer->delta_in_proj_a) {
                for (int h = 0; h < dn; h++)
                    memcpy(row_tmp + (size_t)h * dim, layer->delta_in_proj_a + (size_t)perm[h] * dim, dim * sizeof(float));
                memcpy(layer->delta_in_proj_a, row_tmp, (size_t)dn * dim * sizeof(float));
            }
            if (row_tmp && layer->delta_in_proj_b) {
                for (int h = 0; h < dn; h++)
                    memcpy(row_tmp + (size_t)h * dim, layer->delta_in_proj_b + (size_t)perm[h] * dim, dim * sizeof(float));
                memcpy(layer->delta_in_proj_b, row_tmp, (size_t)dn * dim * sizeof(float));
            }

            /* Reorder V portion of in_proj_qkv: QKV = [Q(nkv*dk), K(nkv*dk), V(dn*dv)]
             * Only V part needs reordering */
            if (row_tmp && layer->delta_in_proj_qkv) {
                int qk_total = nkv * dk * 2;
                float* v_start = layer->delta_in_proj_qkv + (size_t)qk_total * dim;
                /* V is [dn, dv*dim/dn]... actually V rows in QKV are contiguous:
                 * row r of V portion = qkv_row[qk_total + r] */
                /* Wait, QKV is [qkv_dim, dim] where qkv_dim = nkv*dk*2 + dn*dv
                 * Row indices qk_total..qk_total+dn*dv-1 are V rows.
                 * Each V-head h occupies dv rows: [qk_total + h*dv .. qk_total + (h+1)*dv - 1] */
                float* v_reorder = (float*)malloc((size_t)dn * dv * dim * sizeof(float));
                if (v_reorder) {
                    for (int h = 0; h < dn; h++) {
                        memcpy(v_reorder + (size_t)h * dv * dim,
                               v_start + (size_t)perm[h] * dv * dim,
                               (size_t)dv * dim * sizeof(float));
                    }
                    memcpy(v_start, v_reorder, (size_t)dn * dv * dim * sizeof(float));
                    free(v_reorder);
                }
            }

            /* Reorder in_proj_z [dn*dv, dim]: z_dim = dn * dv, grouped by V-head */
            if (row_tmp && layer->delta_in_proj_z) {
                float* z_reorder = (float*)malloc((size_t)dn * dv * dim * sizeof(float));
                if (z_reorder) {
                    for (int h = 0; h < dn; h++) {
                        memcpy(z_reorder + (size_t)h * dv * dim,
                               layer->delta_in_proj_z + (size_t)perm[h] * dv * dim,
                               (size_t)dv * dim * sizeof(float));
                    }
                    memcpy(layer->delta_in_proj_z, z_reorder, (size_t)dn * dv * dim * sizeof(float));
                    free(z_reorder);
                }
            }

            /* Reorder out_proj [dim, dn*dv]: columns grouped by V-head
             * out_proj is [dim, z_dim], column h*dv..(h+1)*dv-1 belongs to V-head h */
            if (row_tmp && layer->delta_out_proj) {
                int z_dim = dn * dv;
                float* out_reorder = (float*)malloc((size_t)dim * z_dim * sizeof(float));
                if (out_reorder) {
                    for (int r = 0; r < dim; r++) {
                        for (int h = 0; h < dn; h++) {
                            memcpy(out_reorder + (size_t)r * z_dim + h * dv,
                                   layer->delta_out_proj + (size_t)r * z_dim + perm[h] * dv,
                                   (size_t)dv * sizeof(float));
                        }
                    }
                    memcpy(layer->delta_out_proj, out_reorder, (size_t)dim * z_dim * sizeof(float));
                    free(out_reorder);
                }
            }

            /* Reorder norm weights [dv]: same for all heads, no reorder needed */

            free(tmp);
            free(row_tmp);
        }

        /* FFN: gate, up, down projections (SwiGLU) */
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.gate_proj.weight", l);
        layer->w_gate = load_tensor(
                                     find_tensor(tensors, n_tensors, name_buf),
                                     &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.up_proj.weight", l);
        layer->w_up = load_tensor(
                                   find_tensor(tensors, n_tensors, name_buf),
                                   &conv_buf, &conv_used, conv_capacity);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.down_proj.weight", l);
        layer->w_down = load_tensor(
                                     find_tensor(tensors, n_tensors, name_buf),
                                     &conv_buf, &conv_used, conv_capacity);
    }

    /* Output norm */
    model->output_norm = load_tensor(
        find_tensor(tensors, n_tensors, "model.norm.weight"),
        &conv_buf, &conv_used, conv_capacity);

    /* Output weight — may be tied to embedding */
    tensor_info_t* lm_head = find_tensor(tensors, n_tensors, "lm_head.weight");
    if (lm_head) {
        /* Try to keep lm_head as BF16 */
        model->output_weight_bf16 = get_bf16_ptr(lm_head);
        if (model->output_weight_bf16) {
            model->output_weight = NULL;
            size_t lm_bytes = (size_t)lm_head->shape[0] * lm_head->shape[1] * 2;
            fprintf(stderr, "tq_load_model: keeping lm_head as BF16 (%zu MB saved)\n",
                    lm_bytes / (1024 * 1024));
        } else {
            model->output_weight = load_tensor( lm_head,
                                                &conv_buf, &conv_used, conv_capacity);
        }
    } else {
        /* Weight tying: reuse embedding (either FP32 or BF16) */
        model->output_weight = model->token_embedding;
        model->output_weight_bf16 = model->embed_bf16;
    }

    free(tensors);

    /* Qwen RMSNorm adjustment: Qwen's RMSNorm computes
     * output = norm(x) * (1.0 + weight), NOT norm(x) * weight.
     * We bake the "+1" into the weight so tq_rmsnorm can stay as
     * out = x * rsqrt * weight.
     *
     * This applies to: input_layernorm, post_attention_layernorm,
     * model.norm, q_norm, k_norm.
     * It does NOT apply to: linear_attn.norm (Qwen3_5RMSNormGated
     * uses plain weight without +1).
     *
     * Applies to all Qwen-family models (qwen2, qwen3, qwen3_5, etc.)
     * Detected by arch string or DeltaNet presence. */
    int is_qwen_family = (model->config.delta_n_heads > 0);
    if (model->gguf_ctx) {
        const tq_gguf_ctx_t* gctx = (const tq_gguf_ctx_t*)model->gguf_ctx;
        if (strstr(gctx->arch, "qwen") != NULL) is_qwen_family = 1;
    }
    if (is_qwen_family) {
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
        fprintf(stderr, "tq_load_model: applied Qwen RMSNorm +1 weight adjustment\n");
    }

    /* Gemma3 RMSNorm adjustment: same (1+w) scaling as Qwen3.5 */
    if (model->config.model_type == 1) {
        int dim_h = model->config.hidden_dim;
        int head_dim_h = model->config.head_dim;

        for (int l = 0; l < n_layers; l++) {
            tq_layer_weights_t* layer_w = &model->layers[l];
            if (layer_w->attn_norm) {
                for (int i = 0; i < dim_h; i++) {
                    layer_w->attn_norm[i] += 1.0f;
                }
            }
            if (layer_w->post_attn_norm) {
                for (int i = 0; i < dim_h; i++) {
                    layer_w->post_attn_norm[i] += 1.0f;
                }
            }
            if (layer_w->pre_ffn_norm) {
                for (int i = 0; i < dim_h; i++) {
                    layer_w->pre_ffn_norm[i] += 1.0f;
                }
            }
            if (layer_w->post_ffn_norm) {
                for (int i = 0; i < dim_h; i++) {
                    layer_w->post_ffn_norm[i] += 1.0f;
                }
            }
            if (layer_w->q_norm) {
                for (int i = 0; i < head_dim_h; i++) {
                    layer_w->q_norm[i] += 1.0f;
                }
            }
            if (layer_w->k_norm) {
                for (int i = 0; i < head_dim_h; i++) {
                    layer_w->k_norm[i] += 1.0f;
                }
            }
        }
        if (model->output_norm) {
            for (int i = 0; i < dim_h; i++) {
                model->output_norm[i] += 1.0f;
            }
        }
        fprintf(stderr, "tq_load_model: applied Gemma3 RMSNorm +1 weight adjustment\n");

        /* Set up layer_is_sliding for Gemma3.
         * Pattern: 5 sliding + 1 full, repeated. Layers 0-4=sliding, 5=full, etc.
         * We detect by checking layer count modulo 6. */
        model->layer_is_sliding = (int*)calloc((size_t)n_layers, sizeof(int));
        if (model->layer_is_sliding) {
            for (int l = 0; l < n_layers; l++) {
                /* Full/global attention every 6th layer (indices 5, 11, 17, ...) */
                if ((l + 1) % 6 == 0) {
                    model->layer_is_sliding[l] = 0; /* global */
                } else {
                    model->layer_is_sliding[l] = 1; /* sliding */
                }
            }
            int n_sliding = 0, n_global = 0;
            for (int l = 0; l < n_layers; l++) {
                if (model->layer_is_sliding[l]) {
                    n_sliding++;
                } else {
                    n_global++;
                }
            }
            fprintf(stderr, "tq_load_model: Gemma3 layer types: %d sliding, %d global\n",
                    n_sliding, n_global);
        }
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
    /* Clean up all mmaps on failure */
    for (int i = 0; i < total_shards_loaded; i++) {
        if (mmap_ptrs[i]) {
#ifdef _WIN32
            UnmapViewOfFile(mmap_ptrs[i]);
#else
            munmap(mmap_ptrs[i], mmap_sizes[i]);
#endif
        }
    }
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
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
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
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
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
    int full_kv_dim = (c->full_n_kv_heads > 0 && c->full_head_dim > 0)
        ? c->full_n_kv_heads * c->full_head_dim : kv_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;

    /* DeltaNet dimensions */
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;

    int full_q_dim = (c->full_n_heads > 0 && c->full_head_dim > 0)
        ? c->full_n_heads * c->full_head_dim : q_dim;

    for (int l = 0; l < c->n_layers; l++) {
        const tq_layer_weights_t* layer = &model->layers[l];
        int is_full = (model->layer_is_sliding && !model->layer_is_sliding[l]);
        int lkv = is_full ? full_kv_dim : kv_dim;
        int lq = is_full ? full_q_dim : q_dim;
        int lqg = qg_dim; /* with gate: lq*2, without: lq */
        if (is_full) lqg = c->attn_output_gate ? lq * 2 : lq;

        /* Self-attention weights */
        if (layer->wq) {
            int nb = (dim + 31) / 32;
            total += (size_t)lqg * nb * 16;   /* packed Q4 data */
            total += (size_t)lqg * nb * 4;     /* float scales */
        }
        if (layer->wk) {
            int nb = (dim + 31) / 32;
            total += (size_t)lkv * nb * 16;
            total += (size_t)lkv * nb * 4;
        }
        if (layer->wv) {
            int nb = (dim + 31) / 32;
            total += (size_t)lkv * nb * 16;
            total += (size_t)lkv * nb * 4;
        }
        if (layer->wo) {
            int nb = (lq + 31) / 32;
            total += (size_t)dim * nb * 16;
            total += (size_t)dim * nb * 4;
        }

        /* FFN weights — per-layer intermediate dim */
        int lint = (c->per_layer_inter_dim) ? c->per_layer_inter_dim[l] : inter;
        if (layer->w_gate) {
            int nb = (dim + 31) / 32;
            total += (size_t)lint * nb * 16;
            total += (size_t)lint * nb * 4;
        }
        if (layer->w_up) {
            int nb = (dim + 31) / 32;
            total += (size_t)lint * nb * 16;
            total += (size_t)lint * nb * 4;
        }
        if (layer->w_down) {
            int nb = (lint + 31) / 32;
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
    int full_kv_dim = (c->full_n_kv_heads > 0 && c->full_head_dim > 0)
        ? c->full_n_kv_heads * c->full_head_dim : kv_dim;
    int inter = c->intermediate_dim;
    (void)inter;

    /* DeltaNet dimensions */
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
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

    int full_q_dim = (c->full_n_heads > 0 && c->full_head_dim > 0)
        ? c->full_n_heads * c->full_head_dim : q_dim;

    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];
        int is_full = (model->layer_is_sliding && !model->layer_is_sliding[l]);
        int lkv = is_full ? full_kv_dim : kv_dim;
        int lq = is_full ? full_q_dim : q_dim;
        int lqg = c->attn_output_gate ? lq * 2 : lq;

        /* Self-attention */
        quantize_matrix_q4(layer->wq, lqg, dim,
                            &layer->wq_q4, &layer->wq_q4s, &buf, &used);
        if (layer->wq_q4) layer->wq = NULL;

        quantize_matrix_q4(layer->wk, lkv, dim,
                            &layer->wk_q4, &layer->wk_q4s, &buf, &used);
        if (layer->wk_q4) layer->wk = NULL;

        quantize_matrix_q4(layer->wv, lkv, dim,
                            &layer->wv_q4, &layer->wv_q4s, &buf, &used);
        if (layer->wv_q4) layer->wv = NULL;

        quantize_matrix_q4(layer->wo, dim, lq,
                            &layer->wo_q4, &layer->wo_q4s, &buf, &used);
        if (layer->wo_q4) layer->wo = NULL;

        /* FFN — use per-layer intermediate dim if available */
        int linter = (c->per_layer_inter_dim) ? c->per_layer_inter_dim[l] : inter;
        quantize_matrix_q4(layer->w_gate, linter, dim,
                            &layer->w_gate_q4, &layer->w_gate_q4s, &buf, &used);
        if (layer->w_gate_q4) layer->w_gate = NULL;

        quantize_matrix_q4(layer->w_up, linter, dim,
                            &layer->w_up_q4, &layer->w_up_q4s, &buf, &used);
        if (layer->w_up_q4) layer->w_up = NULL;

        quantize_matrix_q4(layer->w_down, dim, linter,
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
 * Q2 weight quantization: 2-bit Lloyd-Max codebook (~12x reduction from FP32)
 *
 * Per block of 32 values: 8 bytes packed + 4 bytes float scale = 12 bytes.
 * Compare to Q4: 16 + 4 = 20 bytes. Q2 is ~1.67x smaller than Q4.
 * ============================================================ */

static void quantize_matrix_q2(const float* src, int rows, int cols,
                                 uint8_t** out_qs, float** out_scales,
                                 char** buf, size_t* used) {
    if (!src || rows <= 0 || cols <= 0) {
        *out_qs = NULL;
        *out_scales = NULL;
        return;
    }
    int n_blocks_per_row = (cols + 31) / 32;
    size_t qs_bytes = (size_t)rows * n_blocks_per_row * 8;   /* 8 packed bytes per Q2 block */
    size_t sc_bytes = (size_t)rows * n_blocks_per_row * sizeof(float);

    uint8_t* qs = (uint8_t*)(*buf + *used);
    *used += qs_bytes;
    float*   sc = (float*)(*buf + *used);
    *used += sc_bytes;

    for (int r = 0; r < rows; r++) {
        tq_quantize_row_q2(src + (size_t)r * cols,
                            qs + (size_t)r * n_blocks_per_row * 8,
                            sc + (size_t)r * n_blocks_per_row,
                            cols);
    }
    *out_qs = qs;
    *out_scales = sc;
}

static size_t calc_q2_buffer_size(const tq_model_t* model) {
    size_t total = 0;
    const tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;

    for (int l = 0; l < c->n_layers; l++) {
        const tq_layer_weights_t* layer = &model->layers[l];
        /* Q2 block: 8 packed bytes + 4 float scale = 12 bytes per 32 values */
        if (layer->wq) { int nb = (dim + 31) / 32; total += (size_t)qg_dim * nb * (8 + 4); }
        if (layer->wk) { int nb = (dim + 31) / 32; total += (size_t)kv_dim * nb * (8 + 4); }
        if (layer->wv) { int nb = (dim + 31) / 32; total += (size_t)kv_dim * nb * (8 + 4); }
        if (layer->wo) { int nb = (q_dim + 31) / 32; total += (size_t)dim * nb * (8 + 4); }
        if (layer->w_gate) { int nb = (dim + 31) / 32; total += (size_t)inter * nb * (8 + 4); }
        if (layer->w_up) { int nb = (dim + 31) / 32; total += (size_t)inter * nb * (8 + 4); }
        if (layer->w_down) { int nb = (inter + 31) / 32; total += (size_t)dim * nb * (8 + 4); }
        if (layer->delta_in_proj_qkv) { int nb = (dim + 31) / 32; total += (size_t)delta_qkv_dim * nb * (8 + 4); }
        if (layer->delta_in_proj_z) { int nb = (dim + 31) / 32; total += (size_t)delta_z_dim * nb * (8 + 4); }
        if (layer->delta_in_proj_a) { int nb = (dim + 31) / 32; total += (size_t)delta_dn * nb * (8 + 4); }
        if (layer->delta_in_proj_b) { int nb = (dim + 31) / 32; total += (size_t)delta_dn * nb * (8 + 4); }
        if (layer->delta_out_proj) { int nb = (delta_z_dim + 31) / 32; total += (size_t)dim * nb * (8 + 4); }
    }
    return total;
}

void tq_quantize_weights_q2(tq_model_t* model) {
    if (!model || model->use_q2_weights) return;

    const tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;

    size_t buf_size = calc_q2_buffer_size(model);
    char* buf = (char*)malloc(buf_size);
    if (!buf) {
        fprintf(stderr, "tq_quantize_weights_q2: failed to allocate %zu MB for Q2\n",
                buf_size / (1024 * 1024));
        return;
    }
    size_t used = 0;

    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];

        /* Self-attention */
        quantize_matrix_q2(layer->wq, qg_dim, dim,
                            &layer->wq_q2, &layer->wq_q2s, &buf, &used);
        if (layer->wq_q2) layer->wq = NULL;

        quantize_matrix_q2(layer->wk, kv_dim, dim,
                            &layer->wk_q2, &layer->wk_q2s, &buf, &used);
        if (layer->wk_q2) layer->wk = NULL;

        quantize_matrix_q2(layer->wv, kv_dim, dim,
                            &layer->wv_q2, &layer->wv_q2s, &buf, &used);
        if (layer->wv_q2) layer->wv = NULL;

        quantize_matrix_q2(layer->wo, dim, q_dim,
                            &layer->wo_q2, &layer->wo_q2s, &buf, &used);
        if (layer->wo_q2) layer->wo = NULL;

        /* FFN */
        quantize_matrix_q2(layer->w_gate, inter, dim,
                            &layer->w_gate_q2, &layer->w_gate_q2s, &buf, &used);
        if (layer->w_gate_q2) layer->w_gate = NULL;

        quantize_matrix_q2(layer->w_up, inter, dim,
                            &layer->w_up_q2, &layer->w_up_q2s, &buf, &used);
        if (layer->w_up_q2) layer->w_up = NULL;

        quantize_matrix_q2(layer->w_down, dim, inter,
                            &layer->w_down_q2, &layer->w_down_q2s, &buf, &used);
        if (layer->w_down_q2) layer->w_down = NULL;

        /* DeltaNet */
        quantize_matrix_q2(layer->delta_in_proj_qkv, delta_qkv_dim, dim,
                            &layer->delta_in_proj_qkv_q2, &layer->delta_in_proj_qkv_q2s,
                            &buf, &used);
        if (layer->delta_in_proj_qkv_q2) layer->delta_in_proj_qkv = NULL;

        quantize_matrix_q2(layer->delta_in_proj_z, delta_z_dim, dim,
                            &layer->delta_in_proj_z_q2, &layer->delta_in_proj_z_q2s,
                            &buf, &used);
        if (layer->delta_in_proj_z_q2) layer->delta_in_proj_z = NULL;

        quantize_matrix_q2(layer->delta_in_proj_a, delta_dn, dim,
                            &layer->delta_in_proj_a_q2, &layer->delta_in_proj_a_q2s,
                            &buf, &used);
        if (layer->delta_in_proj_a_q2) layer->delta_in_proj_a = NULL;

        quantize_matrix_q2(layer->delta_in_proj_b, delta_dn, dim,
                            &layer->delta_in_proj_b_q2, &layer->delta_in_proj_b_q2s,
                            &buf, &used);
        if (layer->delta_in_proj_b_q2) layer->delta_in_proj_b = NULL;

        quantize_matrix_q2(layer->delta_out_proj, dim, delta_z_dim,
                            &layer->delta_out_proj_q2, &layer->delta_out_proj_q2s,
                            &buf, &used);
        if (layer->delta_out_proj_q2) layer->delta_out_proj = NULL;
    }

    model->use_q2_weights = 1;
    model->use_1bit_weights = 1;
    model->_q2_data = buf;
    model->_q2_size = used;

    fprintf(stderr, "tq_quantize_weights_q2: quantized to Q2 (%zu MB, was ~%zu MB FP32)\n",
            used / (1024 * 1024), used * 8 / (1024 * 1024));
}

/* ============================================================
 * Q4+Q2 Progressive Residual: TurboQuant novel weight quantization
 *
 * Applies KV cache residual correction insight to weight matrices:
 *   Pass 1: Q4 quantize (captures main signal)
 *   Pass 2: Residual = FP32 - dequant(Q4)
 *   Pass 3: Q2 quantize residual (captures correction)
 *
 * At inference: matmul uses Q4 path, then adds Q2 correction.
 * Achieves Q8 quality (cosine 0.9998) at 6-bit effective size.
 * ============================================================ */

/* Helper: Q4 quantize a matrix, compute residual, Q2 quantize residual */
static void quantize_matrix_q4q2(float* src, int rows, int cols,
                                   uint8_t** out_q4_qs, float** out_q4_sc,
                                   uint8_t** out_q2_qs, float** out_q2_sc,
                                   char** buf, size_t* used) {
    if (!src || rows <= 0 || cols <= 0) {
        *out_q4_qs = NULL; *out_q4_sc = NULL;
        *out_q2_qs = NULL; *out_q2_sc = NULL;
        return;
    }
    int nb = (cols + 31) / 32;

    /* Allocate Q4 output */
    size_t q4_qs_bytes = (size_t)rows * nb * 16;
    size_t q4_sc_bytes = (size_t)rows * nb * sizeof(float);
    uint8_t* q4_qs = (uint8_t*)(*buf + *used); *used += q4_qs_bytes;
    float*   q4_sc = (float*)(*buf + *used);   *used += q4_sc_bytes;

    /* Allocate Q2 residual output */
    size_t q2_qs_bytes = (size_t)rows * nb * 8;
    size_t q2_sc_bytes = (size_t)rows * nb * sizeof(float);
    uint8_t* q2_qs = (uint8_t*)(*buf + *used); *used += q2_qs_bytes;
    float*   q2_sc = (float*)(*buf + *used);   *used += q2_sc_bytes;

    float* residual = (float*)malloc((size_t)cols * sizeof(float));
    float* dequant = (float*)malloc((size_t)cols * sizeof(float));

    for (int r = 0; r < rows; r++) {
        float* row = src + (size_t)r * cols;

        /* Q4 quantize */
        tq_quantize_row_q4(row,
                            q4_qs + (size_t)r * nb * 16,
                            q4_sc + (size_t)r * nb,
                            cols);

        /* Dequant Q4 → compute residual */
        tq_dequantize_row_q4(q4_qs + (size_t)r * nb * 16,
                              q4_sc + (size_t)r * nb,
                              dequant, cols);
        for (int j = 0; j < cols; j++)
            residual[j] = row[j] - dequant[j];

        /* Q2 quantize residual */
        tq_quantize_row_q2(residual,
                            q2_qs + (size_t)r * nb * 8,
                            q2_sc + (size_t)r * nb,
                            cols);
    }

    free(residual);
    free(dequant);

    *out_q4_qs = q4_qs; *out_q4_sc = q4_sc;
    *out_q2_qs = q2_qs; *out_q2_sc = q2_sc;
}

static size_t calc_q4q2_buffer_size(const tq_model_t* model) {
    /* Each block: Q4 (20 bytes) + Q2 residual (12 bytes) = 32 bytes per 32 elements */
    return calc_q4_buffer_size(model) + calc_q2_buffer_size(model);
}

void tq_quantize_weights_q4q2(tq_model_t* model) {
    if (!model) return;
    if (model->use_q4_weights || model->use_q2_weights) return;

    const tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;

    size_t buf_size = calc_q4q2_buffer_size(model);
    char* buf = (char*)malloc(buf_size);
    if (!buf) {
        fprintf(stderr, "tq_quantize_weights_q4q2: alloc failed (%zu MB)\n", buf_size/(1024*1024));
        return;
    }
    size_t used = 0;

    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];

        /* Self-attention: Q4+Q2 */
        quantize_matrix_q4q2(layer->wq, qg_dim, dim,
            &layer->wq_q4, &layer->wq_q4s, &layer->wq_q2, &layer->wq_q2s, &buf, &used);
        if (layer->wq_q4) layer->wq = NULL;

        quantize_matrix_q4q2(layer->wk, kv_dim, dim,
            &layer->wk_q4, &layer->wk_q4s, &layer->wk_q2, &layer->wk_q2s, &buf, &used);
        if (layer->wk_q4) layer->wk = NULL;

        quantize_matrix_q4q2(layer->wv, kv_dim, dim,
            &layer->wv_q4, &layer->wv_q4s, &layer->wv_q2, &layer->wv_q2s, &buf, &used);
        if (layer->wv_q4) layer->wv = NULL;

        quantize_matrix_q4q2(layer->wo, dim, q_dim,
            &layer->wo_q4, &layer->wo_q4s, &layer->wo_q2, &layer->wo_q2s, &buf, &used);
        if (layer->wo_q4) layer->wo = NULL;

        /* FFN */
        quantize_matrix_q4q2(layer->w_gate, inter, dim,
            &layer->w_gate_q4, &layer->w_gate_q4s, &layer->w_gate_q2, &layer->w_gate_q2s, &buf, &used);
        if (layer->w_gate_q4) layer->w_gate = NULL;

        quantize_matrix_q4q2(layer->w_up, inter, dim,
            &layer->w_up_q4, &layer->w_up_q4s, &layer->w_up_q2, &layer->w_up_q2s, &buf, &used);
        if (layer->w_up_q4) layer->w_up = NULL;

        quantize_matrix_q4q2(layer->w_down, dim, inter,
            &layer->w_down_q4, &layer->w_down_q4s, &layer->w_down_q2, &layer->w_down_q2s, &buf, &used);
        if (layer->w_down_q4) layer->w_down = NULL;

        /* DeltaNet */
        quantize_matrix_q4q2(layer->delta_in_proj_qkv, delta_qkv_dim, dim,
            &layer->delta_in_proj_qkv_q4, &layer->delta_in_proj_qkv_q4s,
            &layer->delta_in_proj_qkv_q2, &layer->delta_in_proj_qkv_q2s, &buf, &used);
        if (layer->delta_in_proj_qkv_q4) layer->delta_in_proj_qkv = NULL;

        quantize_matrix_q4q2(layer->delta_in_proj_z, delta_z_dim, dim,
            &layer->delta_in_proj_z_q4, &layer->delta_in_proj_z_q4s,
            &layer->delta_in_proj_z_q2, &layer->delta_in_proj_z_q2s, &buf, &used);
        if (layer->delta_in_proj_z_q4) layer->delta_in_proj_z = NULL;

        quantize_matrix_q4q2(layer->delta_in_proj_a, delta_dn, dim,
            &layer->delta_in_proj_a_q4, &layer->delta_in_proj_a_q4s,
            &layer->delta_in_proj_a_q2, &layer->delta_in_proj_a_q2s, &buf, &used);
        if (layer->delta_in_proj_a_q4) layer->delta_in_proj_a = NULL;

        quantize_matrix_q4q2(layer->delta_in_proj_b, delta_dn, dim,
            &layer->delta_in_proj_b_q4, &layer->delta_in_proj_b_q4s,
            &layer->delta_in_proj_b_q2, &layer->delta_in_proj_b_q2s, &buf, &used);
        if (layer->delta_in_proj_b_q4) layer->delta_in_proj_b = NULL;

        quantize_matrix_q4q2(layer->delta_out_proj, dim, delta_z_dim,
            &layer->delta_out_proj_q4, &layer->delta_out_proj_q4s,
            &layer->delta_out_proj_q2, &layer->delta_out_proj_q2s, &buf, &used);
        if (layer->delta_out_proj_q4) layer->delta_out_proj = NULL;
    }

    model->use_q4_weights = 1;
    /* NOTE: use_q2_weights NOT set — Q2 fields contain residual, not primary weights.
     * The Q4 dispatch path is used for matmul. Q2 residual correction is applied
     * in the matmul_q4_preq path when both q4 and q2 pointers are non-NULL. */
    model->_q4_data = buf;
    model->_q4_size = used;

    fprintf(stderr, "tq_quantize_weights_q4q2: Q4+Q2 progressive residual (%zu MB, 6-bit effective)\n",
            used / (1024 * 1024));
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
    c->delta_n_kv_heads    = hdr->delta_n_kv_heads_tqm;
    c->delta_key_head_dim  = hdr->delta_key_head_dim;
    c->delta_value_head_dim= hdr->delta_value_head_dim;
    c->delta_conv_width    = hdr->delta_conv_width;
    c->partial_rotary_factor = hdr->partial_rotary_factor;
    c->use_qk_norm         = hdr->use_qk_norm;
    c->attn_output_gate    = hdr->attn_output_gate;

    /* Multi-architecture fields */
    c->model_type              = hdr->model_type;
    c->sliding_window          = hdr->sliding_window;
    c->rope_local_base_freq    = hdr->rope_local_base_freq;
    c->n_norms_per_block       = hdr->n_norms_per_block;
    c->query_pre_attn_scalar   = hdr->query_pre_attn_scalar;

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
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
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

        /* Gemma3 extra norms */
        if (c->model_type == 1) {
            layer->post_attn_norm = layer->ffn_norm; /* shares storage */
            TQM_READ_FP32(layer->pre_ffn_norm, dim);
            TQM_READ_FP32(layer->post_ffn_norm, dim);
        }

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

    /* Set up Gemma3 layer_is_sliding from TQM */
    if (c->model_type == 1 && c->sliding_window > 0) {
        model->layer_is_sliding = (int*)calloc((size_t)c->n_layers, sizeof(int));
        if (model->layer_is_sliding) {
            for (int l = 0; l < c->n_layers; l++) {
                if ((l + 1) % 6 == 0) {
                    model->layer_is_sliding[l] = 0; /* global */
                } else {
                    model->layer_is_sliding[l] = 1; /* sliding */
                }
            }
        }
    }

    /* Runtime Q4 quantization of lm_head (output projection) for fast logit computation.
     * BF16 matmul on 248K x 1024 is slow; Q4 matmul is ~4x faster. */
    if (model->output_weight_bf16) {
        int vocab = c->vocab_size;
        int n_blocks = dim / 32;
        size_t qs_size = (size_t)vocab * n_blocks * 16;
        size_t scales_size = (size_t)vocab * n_blocks * sizeof(float);
        model->output_qs = (uint8_t*)malloc(qs_size);
        model->output_scales = (float*)malloc(scales_size);
        if (model->output_qs && model->output_scales) {
            float* row_buf = (float*)malloc(dim * sizeof(float));
            if (row_buf) {
                for (int i = 0; i < vocab; i++) {
                    const uint16_t* src = model->output_weight_bf16 + (size_t)i * dim;
                    for (int j = 0; j < dim; j++) {
                        uint32_t bits = ((uint32_t)src[j]) << 16;
                        memcpy(&row_buf[j], &bits, 4);
                    }
                    tq_quantize_row_q4(row_buf,
                                        model->output_qs + (size_t)i * n_blocks * 16,
                                        model->output_scales + (size_t)i * n_blocks,
                                        dim);
                }
                free(row_buf);
                fprintf(stderr, "tq_load_tqm: lm_head quantized to Q4 (%.1f MB -> %.1f MB)\n",
                        (double)vocab * dim * 2 / (1024.0 * 1024.0),
                        (double)(qs_size + scales_size) / (1024.0 * 1024.0));
            }
        }
    }

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

/* ============================================================
 * GGUF model loader — loads community GGUF models (Q2_K..IQ4_XS)
 *
 * Strategy:
 *   - mmap the entire GGUF file via tq_gguf_open()
 *   - Extract model config from GGUF metadata
 *   - For attention weights: keep pointers into mmap (dequant on-the-fly)
 *   - For MoE experts: keep pointers into mmap (demand-paged by OS)
 *   - Router weights + norms: dequant to FP32 at load time (small)
 *   - Embeddings: keep as mmap'd BF16/F16 or dequant
 * ============================================================ */

/* Helper: find a GGUF tensor with fallback name patterns */
static const tq_gguf_tensor_t* find_gguf_tensor(const tq_gguf_ctx_t* ctx,
                                                  const char* name) {
    const tq_gguf_tensor_t* t = tq_gguf_find_tensor(ctx, name);
    if (t) return t;
    /* Try without "blk." prefix variants */
    return NULL;
}

/* Helper: dequant a GGUF tensor to FP32, returns malloc'd buffer */
static float* dequant_tensor_fp32(const tq_gguf_tensor_t* t) {
    if (!t || !t->data) return NULL;
    int64_t n_elem = 1;
    for (uint32_t d = 0; d < t->n_dims; d++) n_elem *= t->shape[d];
    float* out = (float*)malloc((size_t)n_elem * sizeof(float));
    if (!out) return NULL;
    tq_dequant_row_gguf(t->type, t->data, out, (int)n_elem);
    return out;
}

tq_model_t* tq_load_gguf(const char* path) {
    if (!path) return NULL;

    /* Open GGUF file */
    tq_gguf_ctx_t* gguf = tq_gguf_open(path);
    if (!gguf) {
        fprintf(stderr, "tq_load_gguf: failed to open '%s'\n", path);
        return NULL;
    }

    fprintf(stderr, "tq_load_gguf: GGUF v%u, %llu tensors, %llu metadata keys\n",
            gguf->version, (unsigned long long)gguf->n_tensors,
            (unsigned long long)gguf->n_kv);
    fprintf(stderr, "tq_load_gguf: architecture = '%s'\n", gguf->arch);

    /* Extract model configuration from GGUF metadata */
    tq_model_t* model = (tq_model_t*)calloc(1, sizeof(tq_model_t));
    if (!model) { tq_gguf_close(gguf); return NULL; }

    tq_model_config_t* c = &model->config;
    char key[256];

    /* Helper macro for arch-prefixed metadata keys */
    #define GGUF_KEY(suffix) (snprintf(key, sizeof(key), "%s." suffix, gguf->arch), key)

    c->n_layers         = tq_gguf_get_i32(gguf, GGUF_KEY("block_count"), 0);
    c->hidden_dim       = tq_gguf_get_i32(gguf, GGUF_KEY("embedding_length"), 0);
    c->intermediate_dim = tq_gguf_get_i32(gguf, GGUF_KEY("feed_forward_length"), 0);
    /* For MoE models, intermediate_dim may be 0 in metadata.
     * Use expert_feed_forward_length as fallback for state allocation. */
    if (c->intermediate_dim == 0) {
        c->intermediate_dim = tq_gguf_get_i32(gguf, GGUF_KEY("expert_feed_forward_length"), 0);
    }
    /* Infer per-layer intermediate_dim from FFN tensor shapes.
     * Gemma 4 E2B has variable FFN dim per layer (e.g., 6144 then 12288). */
    {
        int max_inter = c->intermediate_dim;
        int has_variable = 0;
        int per_layer_dims[256];
        for (int l = 0; l < c->n_layers && l < 256; l++) {
            char tn[128];
            snprintf(tn, sizeof(tn), "blk.%d.ffn_gate.weight", l);
            const tq_gguf_tensor_t* fg = tq_gguf_find_tensor(gguf, tn);
            int idim = fg ? (int)fg->shape[1] : c->intermediate_dim;
            per_layer_dims[l] = idim;
            if (idim > max_inter) max_inter = idim;
            if (l > 0 && idim != per_layer_dims[0]) has_variable = 1;
        }
        if (max_inter > c->intermediate_dim) c->intermediate_dim = max_inter;
        if (has_variable && c->n_layers <= 256) {
            c->per_layer_inter_dim = (int*)malloc((size_t)c->n_layers * sizeof(int));
            for (int l = 0; l < c->n_layers; l++)
                c->per_layer_inter_dim[l] = per_layer_dims[l];
            fprintf(stderr, "tq_load_gguf: variable FFN dim — [0]=%d, [%d]=%d\n",
                    per_layer_dims[0], c->n_layers-1, per_layer_dims[c->n_layers-1]);
        }
    }
    c->n_heads          = tq_gguf_get_i32(gguf, GGUF_KEY("attention.head_count"), 0);
    c->n_kv_heads       = tq_gguf_get_i32(gguf, GGUF_KEY("attention.head_count_kv"), c->n_heads);
    c->vocab_size       = (int)tq_gguf_get_u32(gguf, GGUF_KEY("vocab_size"),
                            tq_gguf_get_u32(gguf, "tokenizer.ggml.tokens", 0));
    c->max_seq_len      = tq_gguf_get_i32(gguf, GGUF_KEY("context_length"), 131072);
    c->rope_freq_base   = tq_gguf_get_f32(gguf, GGUF_KEY("rope.freq_base"), 1000000.0f);
    c->rms_norm_eps     = tq_gguf_get_f32(gguf, GGUF_KEY("attention.layer_norm_rms_epsilon"), 1e-6f);

    /* RoPE dimension count: number of dimensions to rotate per head.
     * For models with rope_freqs (learned freq factors), this determines the
     * frequency computation: freq[i] = pow(base, -2*i/n_dims).
     * For STEP35/Gemma4: n_dims = head_dim/2 for full layers (partial rotation). */
    c->rope_n_dims = tq_gguf_get_i32(gguf, GGUF_KEY("rope.dimension_count"), 0);
    c->rope_n_dims_full = c->rope_n_dims;  /* default: same for both layer types */
    {
        int swa_dims = tq_gguf_get_i32(gguf, GGUF_KEY("rope.dimension_count_swa"), 0);
        if (swa_dims > 0) {
            c->rope_n_dims = swa_dims;  /* sliding layers use SWA dim count */
        }
    }

    /* Sliding window + local RoPE base */
    c->sliding_window = (int)tq_gguf_get_u32(gguf, GGUF_KEY("attention.sliding_window"), 0);
    /* Local/sliding RoPE base: try Gemma4 naming first, then generic */
    c->rope_local_base_freq = tq_gguf_get_f32(gguf, GGUF_KEY("rope.freq_base_swa"),
                               tq_gguf_get_f32(gguf, GGUF_KEY("rope.local.freq_base"),
                               tq_gguf_get_f32(gguf, GGUF_KEY("rope.freq_base"), 10000.0f)));
    c->final_logit_softcap = tq_gguf_get_f32(gguf, GGUF_KEY("final_logit_softcapping"), 0.0f);
    c->attn_logit_softcap = tq_gguf_get_f32(gguf, GGUF_KEY("attn_logit_softcapping"), 0.0f);
    /* Gemma 2/3 use attention softcap (50.0) but Gemma 4 does NOT.
     * Ref: llama.cpp — attn_soft_cap is only set for Gemma 2/3, not Gemma 4. */
    if (c->model_type == 1 && !c->is_gemma4 && c->attn_logit_softcap == 0.0f) {
        c->attn_logit_softcap = 50.0f;
    }

    /* Cap context for memory safety on small machines.
     * GGUF models often claim 262K context but we cap at 4096 by default.
     * Users can override with --ctx flag in quant. */
    if (c->max_seq_len > 4096) c->max_seq_len = 4096;

    /* Compute head_dim — prefer explicit key_length from metadata.
     * For Gemma 4: key_length=512 is for full attention layers,
     * but sliding layers use 256. Detect from first layer's K tensor shape. */
    c->head_dim = tq_gguf_get_i32(gguf, GGUF_KEY("attention.key_length"), 0);
    if (c->head_dim == 0 && c->n_heads > 0) {
        c->head_dim = c->hidden_dim / c->n_heads;
    }

    /* For hybrid sliding/full attention (Gemma 3/4 only):
     * Override head_dim from first layer's K tensor shape (sliding layer),
     * since sliding layers are the majority and determine KV cache layout.
     * NOTE: only for Gemma family — Llama/Qwen use uniform head_dim.
     * Use arch string directly since model_type hasn't been set yet. */
    int is_gemma_arch = (strstr(gguf->arch, "gemma") != NULL);
    if (is_gemma_arch && c->sliding_window > 0) {
        const tq_gguf_tensor_t* k0 = tq_gguf_find_tensor(gguf, "blk.0.attn_k.weight");
        if (k0 && k0->n_dims >= 2) {
            int k_out = (int)k0->shape[1];
            int sliding_head_dim = c->head_dim;
            for (int hd = 512; hd >= 64; hd /= 2) {
                if (k_out % hd == 0) {
                    int kv = k_out / hd;
                    if (kv >= 1 && kv <= c->n_heads && hd < c->head_dim) {
                        sliding_head_dim = hd;
                        break;
                    }
                }
            }
            if (sliding_head_dim != c->head_dim) {
                fprintf(stderr, "tq_load_gguf: hybrid attention detected — "
                        "sliding head_dim=%d (metadata: %d)\n", sliding_head_dim, c->head_dim);
                c->head_dim = sliding_head_dim;
            }
            c->n_kv_heads = k_out / c->head_dim;
        }
    } else {
        /* Non-Gemma: infer kv_heads from K tensor shape with metadata head_dim */
        const tq_gguf_tensor_t* k0 = tq_gguf_find_tensor(gguf, "blk.0.attn_k.weight");
        if (k0 && k0->n_dims >= 2) {
            int k_out = (int)k0->shape[1];
            if (c->head_dim > 0 && k_out % c->head_dim == 0) {
                c->n_kv_heads = k_out / c->head_dim;
            }
        }
    }

    /* MoE configuration */
    c->num_experts        = tq_gguf_get_i32(gguf, GGUF_KEY("expert_count"), 0);
    c->num_active_experts = tq_gguf_get_i32(gguf, GGUF_KEY("expert_used_count"), 0);
    c->is_moe = (c->num_experts > 0);

    if (c->is_moe) {
        /* Try to get expert FFN dim from metadata, or infer from tensor shapes */
        c->expert_intermediate_dim = tq_gguf_get_i32(gguf,
            GGUF_KEY("expert_feed_forward_length"), 0);

        /* Check for shared expert */
        char shared_name[128];
        snprintf(shared_name, sizeof(shared_name), "blk.0.ffn_gate_shexp.weight");
        c->has_shared_expert = (tq_gguf_find_tensor(gguf, shared_name) != NULL) ? 1 : 0;
        c->shared_expert_intermediate_dim = tq_gguf_get_i32(gguf,
            GGUF_KEY("expert_shared_feed_forward_length"), 0);

        /* Infer expert_intermediate_dim from tensor shape if not in metadata */
        if (c->expert_intermediate_dim == 0) {
            snprintf(shared_name, sizeof(shared_name), "blk.0.ffn_gate_exps.weight");
            const tq_gguf_tensor_t* exp_t = tq_gguf_find_tensor(gguf, shared_name);
            if (exp_t && exp_t->n_dims == 3) {
                /* Shape: [num_experts, expert_inter_dim, hidden_dim] */
                c->expert_intermediate_dim = (int)exp_t->shape[1];
            }
        }

        fprintf(stderr, "tq_load_gguf: MoE — %d experts, %d active, expert_dim=%d, shared=%d\n",
                c->num_experts, c->num_active_experts,
                c->expert_intermediate_dim, c->has_shared_expert);
    }

    /* Model type detection — Gemma takes priority (Gemma 4 is both Gemma AND MoE) */
    if (strstr(gguf->arch, "gemma") != NULL) {
        c->model_type = 1; /* gemma family */
        c->n_norms_per_block = 4;
        /* Gemma 4 detection: architecture string is "gemma4" */
        if (strstr(gguf->arch, "gemma4") != NULL) {
            c->is_gemma4 = 1;
            /* Gemma 4 (LLM_ARCH_GEMMA4): n_rot_full = rope.dimension_count (no halving).
             * rope_n_dims_full is already set from rope.dimension_count.
             * Refs: llama.cpp llama-model.cpp line 472-474 — reads ROPE_DIMENSION_COUNT directly. */
            fprintf(stderr, "tq_load_gguf: Gemma4 — RoPE dims swa=%d full=%d, "
                    "GeGLU, rope_freqs for full layers only\n",
                    c->rope_n_dims, c->rope_n_dims_full);
        }
        fprintf(stderr, "tq_load_gguf: Gemma family detected (sliding_window=%d)\n", c->sliding_window);
    } else if (c->is_moe) {
        c->model_type = 2; /* qwen moe */
    } else {
        c->model_type = 0; /* qwen35 */
    }

    fprintf(stderr, "tq_load_gguf: config — layers=%d, dim=%d, heads=%d/%d, head_dim=%d, vocab=%d\n",
            c->n_layers, c->hidden_dim, c->n_heads, c->n_kv_heads, c->head_dim, c->vocab_size);

    if (c->n_layers == 0 || c->hidden_dim == 0) {
        fprintf(stderr, "tq_load_gguf: invalid config, aborting\n");
        free(model);
        tq_gguf_close(gguf);
        return NULL;
    }

    /* Store GGUF context in model for lifetime management */
    model->gguf_ctx = gguf;

    /* Allocate per-layer weights */
    model->layers = (tq_layer_weights_t*)calloc((size_t)c->n_layers, sizeof(tq_layer_weights_t));
    if (!model->layers) {
        free(model);
        tq_gguf_close(gguf);
        return NULL;
    }

    /* MoE config storage */
    tq_moe_config_t* moe_cfg = NULL;
    if (c->is_moe) {
        moe_cfg = (tq_moe_config_t*)calloc(1, sizeof(tq_moe_config_t));
        moe_cfg->num_experts = c->num_experts;
        moe_cfg->num_active = c->num_active_experts;
        moe_cfg->expert_intermediate_dim = c->expert_intermediate_dim;
        moe_cfg->has_shared_expert = c->has_shared_expert;
        moe_cfg->shared_expert_intermediate_dim = c->shared_expert_intermediate_dim;
        moe_cfg->norm_topk_prob = 1;
        moe_cfg->use_gelu = c->is_gemma4 ? 1 : 0; /* Gemma 4: GeGLU, others: SwiGLU */
        model->moe_config = moe_cfg;
    }

    /* Load per-layer weights */
    char tname[256];
    int n_attn_layers = 0;
    int attn_indices[256]; /* max layers */

    /* Detect if GGUF already has Gemma +1.0 norm adjustment baked in.
     * If first layer's attn_norm has mean > 2.0, it's already adjusted. */
    int gemma_norms_adjusted = 0;
    if (c->model_type == 1) {
        const tq_gguf_tensor_t* probe = tq_gguf_find_tensor(gguf, "blk.0.attn_norm.weight");
        if (probe && probe->type == TQ_GGML_TYPE_F32 && probe->shape[0] > 0) {
            const float* pw = (const float*)probe->data;
            float sum = 0;
            int n = (int)probe->shape[0];
            for (int i = 0; i < n && i < 64; i++) sum += pw[i];
            float mean = sum / (n < 64 ? n : 64);
            if (mean > 2.0f) {
                gemma_norms_adjusted = 1;
                fprintf(stderr, "tq_load_gguf: Gemma norms already adjusted (mean=%.1f, skipping +1.0)\n", mean);
            }
        }
    }

    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];

        /* RMSNorm weights (always FP32 in GGUF, dequant to FP32) */
        snprintf(tname, sizeof(tname), "blk.%d.attn_norm.weight", l);
        const tq_gguf_tensor_t* t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->attn_norm = dequant_tensor_fp32(t);
            /* Gemma norm convention: weight = 1 + stored_weight (skip if already adjusted) */
            if (c->model_type == 1 && !gemma_norms_adjusted) {
                for (int i = 0; i < c->hidden_dim; i++) layer->attn_norm[i] += 1.0f;
            }
        }

        snprintf(tname, sizeof(tname), "blk.%d.ffn_norm.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (!t) {
            /* Qwen3.5 uses post_attention_norm as FFN norm */
            snprintf(tname, sizeof(tname), "blk.%d.post_attention_norm.weight", l);
            t = find_gguf_tensor(gguf, tname);
        }
        if (t) {
            layer->ffn_norm = dequant_tensor_fp32(t);
            if (c->model_type == 1 && !gemma_norms_adjusted) {
                for (int i = 0; i < c->hidden_dim; i++) layer->ffn_norm[i] += 1.0f;
            }
        }

        /* QK-norm (optional) */
        snprintf(tname, sizeof(tname), "blk.%d.attn_q_norm.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->q_norm = dequant_tensor_fp32(t);
            c->use_qk_norm = 1;
            /* Gemma QK norm: weight = 1 + stored. Size = head_dim (per-head norm). */
            if (c->model_type == 1 && !gemma_norms_adjusted) {
                int hd = (int)t->shape[t->n_dims > 1 ? 1 : 0];
                for (int i = 0; i < hd; i++) layer->q_norm[i] += 1.0f;
            }
        }

        snprintf(tname, sizeof(tname), "blk.%d.attn_k_norm.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->k_norm = dequant_tensor_fp32(t);
            if (c->model_type == 1 && !gemma_norms_adjusted) {
                int hd = (int)t->shape[t->n_dims > 1 ? 1 : 0];
                for (int i = 0; i < hd; i++) layer->k_norm[i] += 1.0f;
            }
        }

        /* Gemma extra norms (post_attn, pre_ffn, post_ffn) */
        snprintf(tname, sizeof(tname), "blk.%d.post_attention_norm.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->post_attn_norm = dequant_tensor_fp32(t);
            if (!gemma_norms_adjusted)
                for (int i = 0; i < c->hidden_dim; i++) layer->post_attn_norm[i] += 1.0f;
        }
        snprintf(tname, sizeof(tname), "blk.%d.post_ffw_norm.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->post_ffn_norm = dequant_tensor_fp32(t);
            if (!gemma_norms_adjusted)
                for (int i = 0; i < c->hidden_dim; i++) layer->post_ffn_norm[i] += 1.0f;
        }
        snprintf(tname, sizeof(tname), "blk.%d.pre_ffw_norm.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->pre_ffn_norm = dequant_tensor_fp32(t);
            if (!gemma_norms_adjusted)
                for (int i = 0; i < c->hidden_dim; i++) layer->pre_ffn_norm[i] += 1.0f;
        }

        /* Gemma 4 dual-FFN extra norms */
        snprintf(tname, sizeof(tname), "blk.%d.post_ffw_norm_1.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->post_ffn_norm_1 = dequant_tensor_fp32(t);
            if (!gemma_norms_adjusted)
                for (int i = 0; i < c->hidden_dim; i++) layer->post_ffn_norm_1[i] += 1.0f;
        }
        snprintf(tname, sizeof(tname), "blk.%d.pre_ffw_norm_2.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->pre_ffn_norm_2 = dequant_tensor_fp32(t);
            if (!gemma_norms_adjusted)
                for (int i = 0; i < c->hidden_dim; i++) layer->pre_ffn_norm_2[i] += 1.0f;
        }
        snprintf(tname, sizeof(tname), "blk.%d.post_ffw_norm_2.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->post_ffn_norm_2 = dequant_tensor_fp32(t);
            if (!gemma_norms_adjusted)
                for (int i = 0; i < c->hidden_dim; i++) layer->post_ffn_norm_2[i] += 1.0f;
        }

        /* Gemma 4: layer_output_scale (scalar per layer) */
        snprintf(tname, sizeof(tname), "blk.%d.layer_output_scale.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t && t->type == TQ_GGML_TYPE_F32) {
            layer->layer_output_scale = ((const float*)t->data)[0];
        }

        /* Gemma 4 PLE per-layer weights: inp_gate, proj, post_norm */
        snprintf(tname, sizeof(tname), "blk.%d.inp_gate.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->ple_gate = t->data;
            layer->ple_gate_type = t->type;
        }
        snprintf(tname, sizeof(tname), "blk.%d.proj.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->ple_proj = t->data;
            layer->ple_proj_type = t->type;
        }
        snprintf(tname, sizeof(tname), "blk.%d.post_norm.weight", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->ple_norm = dequant_tensor_fp32(t);
            if (c->model_type == 1 && !gemma_norms_adjusted && layer->ple_norm) {
                for (int i = 0; i < c->hidden_dim; i++) layer->ple_norm[i] += 1.0f;
            }
        }

        /* Attention weights — keep as GGUF quantized pointers for on-the-fly dequant.
         * We store the raw data pointer + type info using a small struct packed into
         * the existing FP32 weight pointer fields. For GGUF models, we use a special
         * dispatch: if gguf_ctx is non-NULL, the forward pass uses tq_matmul_gguf. */
        snprintf(tname, sizeof(tname), "blk.%d.attn_q.weight", l);
        const tq_gguf_tensor_t* wq_t = find_gguf_tensor(gguf, tname);
        int is_attn_layer = (wq_t != NULL);
        if (is_attn_layer) {
            /* Detect attn_output_gate on first self_attn layer:
             * If Q proj out_dim = 2 * O proj in_dim, Q includes gate */
            if (n_attn_layers == 0 && wq_t->n_dims >= 2) {
                snprintf(tname, sizeof(tname), "blk.%d.attn_output.weight", l);
                const tq_gguf_tensor_t* wo_t = find_gguf_tensor(gguf, tname);
                if (wo_t && wo_t->n_dims >= 2) {
                    int q_out = (int)wq_t->shape[1];  /* Q: [hidden, q_out] */
                    int o_in  = (int)wo_t->shape[0];   /* O: [o_in, hidden] */
                    if (q_out == o_in * 2) {
                        c->attn_output_gate = 1;
                        /* Don't override n_heads from metadata — gate doubles Q proj only */
                        fprintf(stderr, "tq_load_gguf: detected attn_output_gate=1 "
                                "(q=%d = 2 * o=%d, n_heads=%d, head_dim=%d)\n",
                                q_out, o_in, c->n_heads, c->head_dim);
                    }
                }
            }

            /* Store GGUF quantized pointers for on-the-fly dequant in forward.
             * This saves ~5GB for 35B models (vs full FP32 dequant at load). */
            layer->gguf_wq = wq_t->data;
            layer->gguf_wq_type = wq_t->type;

            snprintf(tname, sizeof(tname), "blk.%d.attn_k.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) { layer->gguf_wk = t->data; layer->gguf_wk_type = t->type; }

            snprintf(tname, sizeof(tname), "blk.%d.attn_v.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) { layer->gguf_wv = t->data; layer->gguf_wv_type = t->type; }

            snprintf(tname, sizeof(tname), "blk.%d.attn_output.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) { layer->gguf_wo = t->data; layer->gguf_wo_type = t->type; }

            attn_indices[n_attn_layers++] = l;
        }

        /* Check for DeltaNet / SSM weights (Qwen3.5 hybrid) */
        snprintf(tname, sizeof(tname), "blk.%d.ssm_a", l);
        t = find_gguf_tensor(gguf, tname);
        if (t) {
            layer->delta_a_log = dequant_tensor_fp32(t);
            /* GGUF stores ssm_a as -exp(a_log), but our forward pass expects a_log.
             * Convert back: a_log = log(-ssm_a) */
            if (layer->delta_a_log) {
                for (int64_t j = 0; j < t->shape[0]; j++) {
                    float v = layer->delta_a_log[j];
                    layer->delta_a_log[j] = (v < 0) ? logf(-v) : 0.0f;
                }
            }

            snprintf(tname, sizeof(tname), "blk.%d.ssm_conv1d.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) layer->delta_conv1d = dequant_tensor_fp32(t);

            snprintf(tname, sizeof(tname), "blk.%d.ssm_dt.bias", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) layer->delta_dt_bias = dequant_tensor_fp32(t);

            /* Small DeltaNet weights: dequant to FP32 (alpha, beta are small) */
            snprintf(tname, sizeof(tname), "blk.%d.ssm_alpha.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) { layer->gguf_delta_a = t->data; layer->gguf_delta_a_type = t->type; }

            snprintf(tname, sizeof(tname), "blk.%d.ssm_beta.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) { layer->gguf_delta_b = t->data; layer->gguf_delta_b_type = t->type; }

            /* Large DeltaNet projections: dequant to FP32 for recurrent
             * state precision.  Q5_K (5-bit) introduces too much error in
             * the recurrent state that accumulates across time steps.
             * ~24 MB/layer × 30 layers ≈ 720 MB — fits in 16 GB. */
            snprintf(tname, sizeof(tname), "blk.%d.attn_qkv.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) {
                if (t->type == TQ_GGML_TYPE_Q5_K || t->type == TQ_GGML_TYPE_IQ2_XXS ||
                    t->type == TQ_GGML_TYPE_IQ3_XXS || t->type == TQ_GGML_TYPE_IQ4_XS) {
                    /* Low-precision: dequant to FP32 for recurrent accuracy */
                    layer->delta_in_proj_qkv = dequant_tensor_fp32(t);
                    fprintf(stderr, "tq_load_gguf: layer %d attn_qkv dequant to FP32 (was type %d)\n", l, t->type);
                } else {
                    layer->gguf_delta_qkv = t->data;
                    layer->gguf_delta_qkv_type = t->type;
                }
            }

            snprintf(tname, sizeof(tname), "blk.%d.attn_gate.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) {
                if (t->type == TQ_GGML_TYPE_Q5_K || t->type == TQ_GGML_TYPE_IQ2_XXS ||
                    t->type == TQ_GGML_TYPE_IQ3_XXS || t->type == TQ_GGML_TYPE_IQ4_XS) {
                    layer->delta_in_proj_z = dequant_tensor_fp32(t);
                    fprintf(stderr, "tq_load_gguf: layer %d attn_gate dequant to FP32 (was type %d)\n", l, t->type);
                } else {
                    layer->gguf_delta_z = t->data;
                    layer->gguf_delta_z_type = t->type;
                }
            }

            snprintf(tname, sizeof(tname), "blk.%d.ssm_norm.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) layer->delta_norm = dequant_tensor_fp32(t);

            snprintf(tname, sizeof(tname), "blk.%d.ssm_out.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) { layer->gguf_delta_out = t->data; layer->gguf_delta_out_type = t->type; }

            /* Infer DeltaNet config from tensor shapes if not set */
            if (c->delta_n_heads == 0 && layer->delta_a_log) {
                /* Read SSM config from GGUF metadata (Qwen3.5 DeltaNet).
                 * ssm_a shape[0] = num_v_heads (= ssm.time_step_rank)
                 * ssm_norm shape[0] = value_head_dim
                 * From GGUF metadata:
                 *   ssm.state_size   = key_head_dim (ssm_d_state)
                 *   ssm.group_count  = num_k_heads  (ssm_n_group)
                 *   ssm.time_step_rank = num_v_heads
                 *   ssm.inner_size   = num_v_heads * value_head_dim */
                snprintf(tname, sizeof(tname), "blk.%d.ssm_a", l);
                const tq_gguf_tensor_t* a_t = find_gguf_tensor(gguf, tname);
                if (a_t) c->delta_n_heads = (int)a_t->shape[0]; /* num_v_heads */

                snprintf(tname, sizeof(tname), "blk.%d.ssm_norm.weight", l);
                const tq_gguf_tensor_t* norm_t = find_gguf_tensor(gguf, tname);
                if (norm_t) {
                    c->delta_value_head_dim = (int)norm_t->shape[0];
                }

                /* Try to read key_head_dim from metadata (ssm.state_size) */
                c->delta_key_head_dim = tq_gguf_get_i32(gguf,
                    GGUF_KEY("ssm.state_size"), c->delta_value_head_dim);

                /* Try to read num_k_heads from metadata (ssm.group_count) */
                c->delta_n_kv_heads = tq_gguf_get_i32(gguf,
                    GGUF_KEY("ssm.group_count"), c->delta_n_heads);

                c->delta_conv_width = tq_gguf_get_i32(gguf,
                    GGUF_KEY("ssm.conv_kernel"), 4);
                c->partial_rotary_factor = 0.25f;

                fprintf(stderr, "tq_load_gguf: DeltaNet config — v_heads=%d, kv_heads=%d, "
                        "key_dim=%d, val_dim=%d, conv=%d\n",
                        c->delta_n_heads, c->delta_n_kv_heads,
                        c->delta_key_head_dim, c->delta_value_head_dim, c->delta_conv_width);
            }
        }

        /* FFN weights */
        if (c->is_moe) {
            /* Check if this layer has MoE or dense FFN */
            snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_inp.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) {
                /* MoE layer: allocate and set up expert pointers */
                tq_moe_layer_t* moe = (tq_moe_layer_t*)calloc(1, sizeof(tq_moe_layer_t));
                moe->experts = (tq_expert_weights_t*)calloc((size_t)c->num_experts,
                                                              sizeof(tq_expert_weights_t));

                /* Router weights (small, always dequant to FP32) */
                moe->router_weight = dequant_tensor_fp32(t);

                /* Router input scale (Gemma 4): per-feature scaling before routing */
                snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_inp.scale", l);
                t = find_gguf_tensor(gguf, tname);
                if (t && t->type == TQ_GGML_TYPE_F32) {
                    moe->router_input_scale = (const float*)t->data;
                }

                /* Expert weights: shape [num_experts, expert_dim, hidden_dim]
                 * For GGUF, these are stored as 3D tensors. Each expert's
                 * weights are a contiguous slice within the tensor. */
                snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_exps.weight", l);
                const tq_gguf_tensor_t* gate_t = find_gguf_tensor(gguf, tname);
                snprintf(tname, sizeof(tname), "blk.%d.ffn_up_exps.weight", l);
                const tq_gguf_tensor_t* up_t = find_gguf_tensor(gguf, tname);
                snprintf(tname, sizeof(tname), "blk.%d.ffn_down_exps.weight", l);
                const tq_gguf_tensor_t* down_t = find_gguf_tensor(gguf, tname);

                if (gate_t && up_t && down_t) {
                    int exp_inter = c->expert_intermediate_dim;
                    /* Each expert's gate/up weight: [expert_inter, hidden_dim]
                     * Stored contiguously: expert[i] starts at offset i * expert_size */
                    size_t gate_exp_size = tq_ggml_type_size(gate_t->type) *
                        ((size_t)exp_inter * c->hidden_dim / tq_ggml_type_blck(gate_t->type));
                    size_t up_exp_size = tq_ggml_type_size(up_t->type) *
                        ((size_t)exp_inter * c->hidden_dim / tq_ggml_type_blck(up_t->type));
                    size_t down_exp_size = tq_ggml_type_size(down_t->type) *
                        ((size_t)c->hidden_dim * exp_inter / tq_ggml_type_blck(down_t->type));

                    for (int e = 0; e < c->num_experts; e++) {
                        moe->experts[e].w_gate = (const uint8_t*)gate_t->data + e * gate_exp_size;
                        moe->experts[e].gate_type = gate_t->type;
                        moe->experts[e].w_up = (const uint8_t*)up_t->data + e * up_exp_size;
                        moe->experts[e].up_type = up_t->type;
                        moe->experts[e].w_down = (const uint8_t*)down_t->data + e * down_exp_size;
                        moe->experts[e].down_type = down_t->type;
                    }
                }

                /* Gemma 4 fused gate_up_exps: single tensor [hidden_dim, 2*expert_dim, num_experts].
                 * gate is first expert_dim rows, up is next expert_dim rows per expert. */
                if (!gate_t && !up_t) {
                    snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_up_exps.weight", l);
                    const tq_gguf_tensor_t* gate_up_t = find_gguf_tensor(gguf, tname);
                    if (gate_up_t && down_t) {
                        int exp_inter = c->expert_intermediate_dim;
                        /* Fused gate+up: each expert has 2*expert_dim rows of hidden_dim columns.
                         * gate = first expert_dim rows, up = next expert_dim rows. */
                        int fused_rows = 2 * exp_inter;
                        size_t fused_exp_size = tq_ggml_type_size(gate_up_t->type) *
                            ((size_t)fused_rows * c->hidden_dim / tq_ggml_type_blck(gate_up_t->type));
                        size_t gate_part_size = tq_ggml_type_size(gate_up_t->type) *
                            ((size_t)exp_inter * c->hidden_dim / tq_ggml_type_blck(gate_up_t->type));
                        size_t down_exp_size = tq_ggml_type_size(down_t->type) *
                            ((size_t)c->hidden_dim * exp_inter / tq_ggml_type_blck(down_t->type));

                        for (int e = 0; e < c->num_experts; e++) {
                            const uint8_t* base = (const uint8_t*)gate_up_t->data + e * fused_exp_size;
                            moe->experts[e].w_gate = base;
                            moe->experts[e].gate_type = gate_up_t->type;
                            moe->experts[e].w_up = base + gate_part_size;
                            moe->experts[e].up_type = gate_up_t->type;
                            moe->experts[e].w_down = (const uint8_t*)down_t->data + e * down_exp_size;
                            moe->experts[e].down_type = down_t->type;
                        }

                        /* Load per-expert output scale if present (Gemma 4) */
                        snprintf(tname, sizeof(tname), "blk.%d.ffn_down_exps.scale", l);
                        t = find_gguf_tensor(gguf, tname);
                        if (t && t->type == TQ_GGML_TYPE_F32) {
                            moe->expert_scale = (const float*)t->data;
                        }

                        fprintf(stderr, "tq_load_gguf: layer %d — fused gate_up experts, "
                                "exp_inter=%d, down_type=%d\n", l, exp_inter, down_t->type);
                    }
                }

                /* Shared expert (if present) */
                if (c->has_shared_expert) {
                    snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_shexp.weight", l);
                    t = find_gguf_tensor(gguf, tname);
                    if (t) {
                        moe->shared_expert.w_gate = t->data;
                        moe->shared_expert.gate_type = t->type;
                    }
                    snprintf(tname, sizeof(tname), "blk.%d.ffn_up_shexp.weight", l);
                    t = find_gguf_tensor(gguf, tname);
                    if (t) {
                        moe->shared_expert.w_up = t->data;
                        moe->shared_expert.up_type = t->type;
                    }
                    snprintf(tname, sizeof(tname), "blk.%d.ffn_down_shexp.weight", l);
                    t = find_gguf_tensor(gguf, tname);
                    if (t) {
                        moe->shared_expert.w_down = t->data;
                        moe->shared_expert.down_type = t->type;
                    }
                    snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_inp_shexp.weight", l);
                    t = find_gguf_tensor(gguf, tname);
                    if (t) moe->shared_gate = dequant_tensor_fp32(t);
                }

                layer->moe = moe;

                /* Gemma 4 dual-FFN: MoE layers ALSO have dense FFN (ffn_gate/up/down).
                 * Load dense FFN weights alongside MoE experts. */
                if (c->is_gemma4) {
                    snprintf(tname, sizeof(tname), "blk.%d.ffn_gate.weight", l);
                    t = find_gguf_tensor(gguf, tname);
                    if (t) { layer->gguf_w_gate = t->data; layer->gguf_w_gate_type = t->type; }
                    snprintf(tname, sizeof(tname), "blk.%d.ffn_up.weight", l);
                    t = find_gguf_tensor(gguf, tname);
                    if (t) { layer->gguf_w_up = t->data; layer->gguf_w_up_type = t->type; }
                    snprintf(tname, sizeof(tname), "blk.%d.ffn_down.weight", l);
                    t = find_gguf_tensor(gguf, tname);
                    if (t) { layer->gguf_w_down = t->data; layer->gguf_w_down_type = t->type; }
                }
            } else {
                /* Dense FFN in an otherwise MoE model (non-Gemma4) */
                snprintf(tname, sizeof(tname), "blk.%d.ffn_gate.weight", l);
                t = find_gguf_tensor(gguf, tname);
                if (t) { layer->gguf_w_gate = t->data; layer->gguf_w_gate_type = t->type; }
                snprintf(tname, sizeof(tname), "blk.%d.ffn_up.weight", l);
                t = find_gguf_tensor(gguf, tname);
                if (t) { layer->gguf_w_up = t->data; layer->gguf_w_up_type = t->type; }
                snprintf(tname, sizeof(tname), "blk.%d.ffn_down.weight", l);
                t = find_gguf_tensor(gguf, tname);
                if (t) { layer->gguf_w_down = t->data; layer->gguf_w_down_type = t->type; }
            }
        } else {
            /* Dense model: use GGUF on-the-fly dequant */
            snprintf(tname, sizeof(tname), "blk.%d.ffn_gate.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) { layer->gguf_w_gate = t->data; layer->gguf_w_gate_type = t->type; }
            snprintf(tname, sizeof(tname), "blk.%d.ffn_up.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) { layer->gguf_w_up = t->data; layer->gguf_w_up_type = t->type; }
            snprintf(tname, sizeof(tname), "blk.%d.ffn_down.weight", l);
            t = find_gguf_tensor(gguf, tname);
            if (t) { layer->gguf_w_down = t->data; layer->gguf_w_down_type = t->type; }
        }
    }

    /* Hybrid architecture: record which layers have self_attn */
    model->n_attn_layers = n_attn_layers;
    if (n_attn_layers > 0 && n_attn_layers < c->n_layers) {
        model->attn_layer_indices = (int*)malloc((size_t)n_attn_layers * sizeof(int));
        memcpy(model->attn_layer_indices, attn_indices, (size_t)n_attn_layers * sizeof(int));
        fprintf(stderr, "tq_load_gguf: hybrid architecture — %d attn layers out of %d total\n",
                n_attn_layers, c->n_layers);
    }

    /* Set up layer_is_sliding for Gemma hybrid attention.
     * Detect from K tensor shape: sliding and full layers have different K output dims.
     * The MAJORITY of layers are sliding (e.g., 25/30 or 28/35). */
    if (c->sliding_window > 0 && c->model_type == 1) {
        model->layer_is_sliding = (int*)calloc((size_t)c->n_layers, sizeof(int));
        if (model->layer_is_sliding) {
            /* Collect K output dims for each layer */
            int k_dims[256]; /* max layers */
            for (int l = 0; l < c->n_layers && l < 256; l++) {
                char tname[128];
                snprintf(tname, sizeof(tname), "blk.%d.attn_k.weight", l);
                const tq_gguf_tensor_t* kt = tq_gguf_find_tensor(gguf, tname);
                k_dims[l] = kt ? (int)kt->shape[1] : 0;
            }
            /* Find the most common K dim (= sliding layer K dim) */
            int sliding_k = k_dims[0]; /* layer 0 is always sliding */
            int n_sliding = 0, n_full = 0;
            int full_kv_dim = 0;
            for (int l = 0; l < c->n_layers; l++) {
                if (k_dims[l] == sliding_k) {
                    model->layer_is_sliding[l] = 1;
                    n_sliding++;
                } else {
                    model->layer_is_sliding[l] = 0;
                    n_full++;
                    full_kv_dim = k_dims[l];
                }
            }
            if (n_full > 0 && full_kv_dim > 0) {
                /* Compute full layer dimensions.
                 * sliding_k = kv_heads * sliding_head_dim (already in c->head_dim)
                 * full_kv_dim = kv_heads_full * full_head_dim
                 * Use metadata key_length for full_head_dim as primary source. */
                int meta_hd = tq_gguf_get_i32(gguf, GGUF_KEY("attention.key_length"), 0);
                if (meta_hd > c->head_dim && full_kv_dim % meta_hd == 0) {
                    c->full_head_dim = meta_hd;
                    c->full_n_kv_heads = full_kv_dim / meta_hd;
                } else if (full_kv_dim % (c->head_dim * 2) == 0) {
                    c->full_head_dim = c->head_dim * 2;
                    c->full_n_kv_heads = full_kv_dim / c->full_head_dim;
                } else {
                    c->full_head_dim = c->head_dim;
                    c->full_n_kv_heads = c->n_kv_heads;
                }
                c->full_n_heads = c->n_heads;
                fprintf(stderr, "tq_load_gguf: Gemma hybrid — %d sliding (hd=%d, kv=%d) + "
                        "%d full (hd=%d, kv=%d, heads=%d) attention layers\n",
                        n_sliding, c->head_dim, c->n_kv_heads,
                        n_full, c->full_head_dim, c->full_n_kv_heads, c->full_n_heads);
            }
        }
    }

    /* Load embedding + output weights.
     * For large vocab (>100K) with quantized embedding, keep GGUF pointer for
     * fused dot output projection (saves 2.8GB FP32 allocation + faster matmul).
     * Token lookup still needs per-row dequant, done in forward pass. */
    const tq_gguf_tensor_t* emb_t = find_gguf_tensor(gguf, "token_embd.weight");
    if (emb_t) {
        if (emb_t->type == TQ_GGML_TYPE_F32) {
            model->token_embedding = (float*)emb_t->data;
        } else if (emb_t->type == TQ_GGML_TYPE_BF16 || emb_t->type == TQ_GGML_TYPE_F16) {
            model->embed_bf16 = (const uint16_t*)emb_t->data;
        } else if (c->vocab_size > 100000 || emb_t->shape[1] > 100000) {
            /* Large vocab: keep GGUF for output projection, dequant rows on demand */
            model->output_gguf = emb_t->data;
            model->output_gguf_type = emb_t->type;
            /* Still need FP32 embedding for token lookup — allocate just one row buffer.
             * The full FP32 table is NOT allocated. Token lookup uses on-the-fly dequant. */
            model->token_embedding = NULL;
            model->embed_bf16 = NULL;
            fprintf(stderr, "tq_load_gguf: large vocab (%d) — using GGUF embedding "
                    "(saves %.1f GB, fused dot output projection)\n",
                    (int)(emb_t->shape[1]),
                    (double)(emb_t->shape[0] * emb_t->shape[1] * 4) / (1024.0*1024.0*1024.0));
        } else {
            model->token_embedding = dequant_tensor_fp32(emb_t);
        }
    }

    /* Get vocab size from embedding tensor shape if not in metadata */
    if (c->vocab_size == 0 && emb_t && emb_t->n_dims >= 2) {
        c->vocab_size = (int)emb_t->shape[1]; /* shape: [hidden_dim, vocab_size] in GGUF row-major */
    }

    const tq_gguf_tensor_t* out_t = find_gguf_tensor(gguf, "output.weight");
    if (out_t) {
        if (out_t->type == TQ_GGML_TYPE_F32) {
            model->output_weight = (float*)out_t->data;
        } else if (out_t->type == TQ_GGML_TYPE_BF16 || out_t->type == TQ_GGML_TYPE_F16) {
            model->output_weight_bf16 = (const uint16_t*)out_t->data;
        } else {
            model->output_weight = dequant_tensor_fp32(out_t);
        }
    } else {
        /* Weight tying: output weight = embedding weight */
        model->output_weight = model->token_embedding;
        model->output_weight_bf16 = model->embed_bf16;
        /* GGUF weight tying: output_gguf already set from embedding */
    }

    const tq_gguf_tensor_t* onorm_t = find_gguf_tensor(gguf, "output_norm.weight");
    if (onorm_t) {
        model->output_norm = dequant_tensor_fp32(onorm_t);
        if (c->model_type == 1 && !gemma_norms_adjusted) {
            for (int i = 0; i < c->hidden_dim; i++) model->output_norm[i] += 1.0f;
        }
    }

    /* Learned RoPE frequencies (Gemma 4): pre-computed inv_freq values */
    {
        const tq_gguf_tensor_t* rope_t = find_gguf_tensor(gguf, "rope_freqs.weight");
        if (rope_t) {
            model->rope_freqs = dequant_tensor_fp32(rope_t);
            model->rope_freqs_len = (int)rope_t->shape[0];
            fprintf(stderr, "tq_load_gguf: loaded learned RoPE frequencies (%d values)\n",
                    model->rope_freqs_len);
        }
    }

    /* Gemma 4 PLE (Per-Layer Embedding) global tensors */
    {
        const tq_gguf_tensor_t* ple_emb_t = find_gguf_tensor(gguf, "per_layer_token_embd.weight");
        const tq_gguf_tensor_t* ple_proj_t = find_gguf_tensor(gguf, "per_layer_model_proj.weight");
        const tq_gguf_tensor_t* ple_norm_t = find_gguf_tensor(gguf, "per_layer_proj_norm.weight");
        if (ple_emb_t && ple_proj_t && ple_norm_t) {
            /* per_layer_token_embd: [8960, vocab] Q5_K — keep as GGUF pointer for runtime dequant */
            model->ple_embedding = ple_emb_t->data;
            model->ple_embedding_type = ple_emb_t->type;

            /* per_layer_model_proj: [hidden_dim, 8960] BF16 — dequant to FP32 at load time */
            model->ple_proj = dequant_tensor_fp32(ple_proj_t);

            /* per_layer_proj_norm: [ple_dim] F32 */
            model->ple_proj_norm = dequant_tensor_fp32(ple_norm_t);

            /* Infer ple_dim from tensor shape: per_layer_token_embd shape[0] / n_layers */
            int total_ple = (int)ple_emb_t->shape[0];
            model->ple_dim = total_ple / c->n_layers;

            /* Gemma norm adjustment: +1.0 if norms not already adjusted */
            if (c->model_type == 1 && !gemma_norms_adjusted && model->ple_proj_norm) {
                for (int i = 0; i < model->ple_dim; i++)
                    model->ple_proj_norm[i] += 1.0f;
            }

            fprintf(stderr, "tq_load_gguf: PLE enabled, ple_dim=%d, total_ple_dim=%d\n",
                    model->ple_dim, total_ple);
        }
    }

    fprintf(stderr, "tq_load_gguf: loaded %d layers (%d self_attn%s), dim=%d, heads=%d/%d, vocab=%d\n",
            c->n_layers, n_attn_layers,
            c->is_moe ? ", MoE" : "",
            c->hidden_dim, c->n_heads, c->n_kv_heads, c->vocab_size);

    /* ============================================================
     * Load-time weight conversion: GGUF -> Q4
     *
     * For non-MoE weights (attention, DeltaNet, dense FFN): batch
     * dequantize GGUF -> FP32 (temporary), then quantize to Q4.
     *
     * For MoE expert weights: convert expert-by-expert using a small
     * reusable FP32 temp buffer (~12 MB) to avoid multi-GB allocation.
     * This replaces the slow on-the-fly GGUF dequant path with the
     * fast Q4xQ8 integer matmul path, yielding ~10x speedup.
     * ============================================================ */
    {
        /* Estimate total FP32 weight size for non-MoE layer weights */
        int dim = c->hidden_dim;
        int q_dim = c->n_heads * c->head_dim;
        int kv_dim = c->n_kv_heads * c->head_dim;
        int full_kv_dim = (c->full_n_kv_heads > 0 && c->full_head_dim > 0)
            ? c->full_n_kv_heads * c->full_head_dim : kv_dim;
        int inter = c->intermediate_dim;
        (void)inter;
        int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
        int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2
                          + c->delta_n_heads * c->delta_value_head_dim;
        int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
        int delta_dn = c->delta_n_heads;

        size_t est_fp32 = 0;
        for (int l = 0; l < c->n_layers; l++) {
            const tq_layer_weights_t* layer = &model->layers[l];
            int is_full_l = (model->layer_is_sliding && !model->layer_is_sliding[l]);
            int lkv = is_full_l ? full_kv_dim : kv_dim;
            int lq = is_full_l ? (c->full_n_heads * c->full_head_dim) : q_dim;
            int lqg = c->attn_output_gate ? lq * 2 : lq;
            if (layer->gguf_wq)
                est_fp32 += (size_t)lqg * dim * sizeof(float);
            if (layer->gguf_wk)
                est_fp32 += (size_t)lkv * dim * sizeof(float);
            if (layer->gguf_wv)
                est_fp32 += (size_t)lkv * dim * sizeof(float);
            if (layer->gguf_wo)
                est_fp32 += (size_t)dim * lq * sizeof(float);
            /* Dense FFN weights (not present in MoE layers) */
            int lint_est = (c->per_layer_inter_dim) ? c->per_layer_inter_dim[l] : inter;
            if (layer->gguf_w_gate)
                est_fp32 += (size_t)lint_est * dim * sizeof(float);
            if (layer->gguf_w_up)
                est_fp32 += (size_t)lint_est * dim * sizeof(float);
            if (layer->gguf_w_down)
                est_fp32 += (size_t)dim * lint_est * sizeof(float);
            /* DeltaNet GGUF weights */
            if (layer->gguf_delta_qkv)
                est_fp32 += (size_t)delta_qkv_dim * dim * sizeof(float);
            if (layer->gguf_delta_z)
                est_fp32 += (size_t)delta_z_dim * dim * sizeof(float);
            if (layer->gguf_delta_a)
                est_fp32 += (size_t)delta_dn * dim * sizeof(float);
            if (layer->gguf_delta_b)
                est_fp32 += (size_t)delta_dn * dim * sizeof(float);
            if (layer->gguf_delta_out)
                est_fp32 += (size_t)dim * delta_z_dim * sizeof(float);
        }

        const size_t MAX_FP32_BYTES = (size_t)16 * 1024 * 1024 * 1024ULL; /* 16 GB */
        /* TQ_NO_Q4=1 disables Q4 recompression → use direct GGUF dequant for better quality */
        if (getenv("TQ_NO_Q4")) {
            fprintf(stderr, "tq_load_gguf: TQ_NO_Q4 set — skipping Q4 conversion, using GGUF on-the-fly dequant\n");
            goto skip_q4_conversion;
        }
        /* Gemma 4 MoE: skip Q4 conversion — Q8_0 fused dot is faster than Q4 matmul
         * because Q8_0→Q4 conversion adds overhead without speedup benefit.
         * Profiling: Q8_0 direct = 325ms/tok vs Q4 converted = 431ms/tok (-25%). */
        if (c->is_gemma4 && c->is_moe) {
            fprintf(stderr, "tq_load_gguf: Gemma4 MoE — skipping Q4 conversion (Q8_0 fused dot is faster)\n");
            goto skip_q4_conversion;
        }
        int has_gguf_weights = 0;
        for (int l = 0; l < c->n_layers && !has_gguf_weights; l++) {
            if (model->layers[l].gguf_wq || model->layers[l].gguf_w_gate
                || model->layers[l].gguf_delta_qkv || model->layers[l].gguf_delta_z
                || model->layers[l].moe)
                has_gguf_weights = 1;
        }

        if (has_gguf_weights && est_fp32 < MAX_FP32_BYTES) {
            fprintf(stderr, "tq_load_gguf: load-time Q4 conversion enabled "
                    "(est FP32 = %.1f GB < 8 GB threshold)\n",
                    (double)est_fp32 / (1024.0 * 1024.0 * 1024.0));

            /* Track all FP32 temporaries for cleanup after Q4 quantization.
             * Max 13 weight matrices per layer (7 attn/FFN + 5 DeltaNet + 1 spare). */
            int max_tmp = c->n_layers * 13;
            float** fp32_temps = (float**)calloc((size_t)max_tmp, sizeof(float*));
            int n_tmp = 0;

            for (int l = 0; l < c->n_layers; l++) {
                tq_layer_weights_t* layer = &model->layers[l];

                /* Self-attention weights: dequant GGUF -> FP32 */
                int is_full = (model->layer_is_sliding && !model->layer_is_sliding[l]);
                int lq = is_full ? (c->full_n_heads * c->full_head_dim) : q_dim;
                int lqg = c->attn_output_gate ? lq * 2 : lq;
                if (layer->gguf_wq) {
                    int n = lqg * dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_wq_type, layer->gguf_wq, fp, n);
                        layer->wq = fp;
                        layer->gguf_wq = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
                if (layer->gguf_wk) {
                    int lkv = (model->layer_is_sliding && !model->layer_is_sliding[l]) ? full_kv_dim : kv_dim;
                    int n = lkv * dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_wk_type, layer->gguf_wk, fp, n);
                        layer->wk = fp;
                        layer->gguf_wk = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
                if (layer->gguf_wv) {
                    int lkv = (model->layer_is_sliding && !model->layer_is_sliding[l]) ? full_kv_dim : kv_dim;
                    int n = lkv * dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_wv_type, layer->gguf_wv, fp, n);
                        layer->wv = fp;
                        layer->gguf_wv = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
                if (layer->gguf_wo) {
                    int n = dim * lq;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_wo_type, layer->gguf_wo, fp, n);
                        layer->wo = fp;
                        layer->gguf_wo = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }

                /* Dense FFN weights: dequant GGUF -> FP32 (per-layer dim) */
                int lint = (c->per_layer_inter_dim) ? c->per_layer_inter_dim[l] : inter;
                if (layer->gguf_w_gate) {
                    int n = lint * dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_w_gate_type, layer->gguf_w_gate, fp, n);
                        layer->w_gate = fp;
                        layer->gguf_w_gate = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
                if (layer->gguf_w_up) {
                    int n = lint * dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_w_up_type, layer->gguf_w_up, fp, n);
                        layer->w_up = fp;
                        layer->gguf_w_up = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
                if (layer->gguf_w_down) {
                    int n = dim * lint;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_w_down_type, layer->gguf_w_down, fp, n);
                        layer->w_down = fp;
                        layer->gguf_w_down = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }

                /* DeltaNet weights: dequant GGUF -> FP32 */
                if (layer->gguf_delta_qkv && delta_qkv_dim > 0) {
                    int n = delta_qkv_dim * dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_delta_qkv_type, layer->gguf_delta_qkv, fp, n);
                        layer->delta_in_proj_qkv = fp;
                        layer->gguf_delta_qkv = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
                if (layer->gguf_delta_z && delta_z_dim > 0) {
                    int n = delta_z_dim * dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_delta_z_type, layer->gguf_delta_z, fp, n);
                        layer->delta_in_proj_z = fp;
                        layer->gguf_delta_z = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
                if (layer->gguf_delta_a && delta_dn > 0) {
                    int n = delta_dn * dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_delta_a_type, layer->gguf_delta_a, fp, n);
                        layer->delta_in_proj_a = fp;
                        layer->gguf_delta_a = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
                if (layer->gguf_delta_b && delta_dn > 0) {
                    int n = delta_dn * dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_delta_b_type, layer->gguf_delta_b, fp, n);
                        layer->delta_in_proj_b = fp;
                        layer->gguf_delta_b = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
                if (layer->gguf_delta_out && delta_z_dim > 0) {
                    int n = dim * delta_z_dim;
                    float* fp = (float*)malloc((size_t)n * sizeof(float));
                    if (fp) {
                        tq_dequant_row_gguf(layer->gguf_delta_out_type, layer->gguf_delta_out, fp, n);
                        layer->delta_out_proj = fp;
                        layer->gguf_delta_out = NULL;
                        fp32_temps[n_tmp++] = fp;
                    }
                }
            }

            /* Convert all FP32 weights to Q4 (reuses existing tq_quantize_weights_q4) */
            tq_quantize_weights_q4(model);

            /* Free FP32 temporaries (tq_quantize_weights_q4 set layer->wX = NULL
             * but didn't free the malloc'd buffers) */
            for (int i = 0; i < n_tmp; i++) {
                free(fp32_temps[i]);
            }
            free(fp32_temps);

            fprintf(stderr, "tq_load_gguf: Q4 conversion complete — fast matmul path active\n");
        }

skip_q4_conversion: ;
        /* ============================================================
         * MoE shared expert Q4 conversion + LRU cache init
         *
         * Routed experts use a runtime LRU cache (in tq_moe.c) that
         * converts on-demand — only ~32 most-recently-used experts per
         * layer stay in Q4 form, keeping memory under ~2 GB.
         *
         * Shared experts are always active, so convert them at load time.
         * ============================================================ */
        if (c->is_moe) {
            int shared_inter = c->shared_expert_intermediate_dim;
            if (shared_inter == 0) shared_inter = c->expert_intermediate_dim;

            size_t max_elems = (size_t)shared_inter * dim;
            if ((size_t)dim * shared_inter > max_elems)
                max_elems = (size_t)dim * shared_inter;

            float* fp32_temp = NULL;
            if (c->has_shared_expert && max_elems > 0)
                fp32_temp = (float*)malloc(max_elems * sizeof(float));

            size_t total_q4_shared = 0;
            int n_shared_converted = 0;

            if (c->has_shared_expert && fp32_temp) {
                for (int l = 0; l < c->n_layers; l++) {
                    tq_moe_layer_t* moe = (tq_moe_layer_t*)model->layers[l].moe;
                    if (!moe || !moe->shared_expert.w_gate) continue;
                    tq_expert_weights_t* se = &moe->shared_expert;

                    /* gate: [shared_inter, dim] */
                    {
                        int n = shared_inter * dim;
                        int n_blocks = (n + 31) / 32;
                        tq_dequant_row_gguf(se->gate_type, se->w_gate, fp32_temp, n);
                        se->gate_q4_qs = (uint8_t*)malloc((size_t)n_blocks * 16);
                        se->gate_q4_scales = (float*)malloc((size_t)n_blocks * sizeof(float));
                        if (se->gate_q4_qs && se->gate_q4_scales) {
                            tq_quantize_row_q4(fp32_temp, se->gate_q4_qs,
                                               se->gate_q4_scales, n);
                            total_q4_shared += (size_t)n_blocks * 16 + (size_t)n_blocks * sizeof(float);
                        }
                    }

                    /* up: [shared_inter, dim] */
                    {
                        int n = shared_inter * dim;
                        int n_blocks = (n + 31) / 32;
                        tq_dequant_row_gguf(se->up_type, se->w_up, fp32_temp, n);
                        se->up_q4_qs = (uint8_t*)malloc((size_t)n_blocks * 16);
                        se->up_q4_scales = (float*)malloc((size_t)n_blocks * sizeof(float));
                        if (se->up_q4_qs && se->up_q4_scales) {
                            tq_quantize_row_q4(fp32_temp, se->up_q4_qs,
                                               se->up_q4_scales, n);
                            total_q4_shared += (size_t)n_blocks * 16 + (size_t)n_blocks * sizeof(float);
                        }
                    }

                    /* down: [dim, shared_inter] */
                    {
                        int n = dim * shared_inter;
                        int n_blocks = (n + 31) / 32;
                        tq_dequant_row_gguf(se->down_type, se->w_down, fp32_temp, n);
                        se->down_q4_qs = (uint8_t*)malloc((size_t)n_blocks * 16);
                        se->down_q4_scales = (float*)malloc((size_t)n_blocks * sizeof(float));
                        if (se->down_q4_qs && se->down_q4_scales) {
                            tq_quantize_row_q4(fp32_temp, se->down_q4_qs,
                                               se->down_q4_scales, n);
                            total_q4_shared += (size_t)n_blocks * 16 + (size_t)n_blocks * sizeof(float);
                        }
                    }

                    se->q4_converted = (se->gate_q4_qs && se->up_q4_qs && se->down_q4_qs);
                    if (se->q4_converted) n_shared_converted++;
                }
            }

            free(fp32_temp);

            /* LRU caches disabled: both Q8 and cblas caches are bypassed
             * in the forward path (if(0) guards). Skip allocation to save
             * ~1.5 GB RAM — critical for 16GB machines running 9.9 GB models. */
            /* tq_moe_cache_init(...) — disabled */
            /* tq_moe_cblas_cache_init(...) — disabled */

            fprintf(stderr, "tq_load_gguf: MoE — %d shared experts Q4-converted "
                    "(%.1f MB), routed experts use runtime LRU cache\n",
                    n_shared_converted, (double)total_q4_shared / (1024.0 * 1024.0));
        }

        /* Advise OS to release mmap'd GGUF pages — the original quantized
         * data is no longer needed after Q4 conversion.
         * NOTE: For MoE models we must NOT release mmap pages since routed
         * expert weights are still read on-demand from the mmap. */
#ifndef _WIN32
        if (model->gguf_ctx) {
            tq_gguf_ctx_t* gctx = (tq_gguf_ctx_t*)model->gguf_ctx;
            if (gctx->mmap_data && gctx->mmap_size > 0) {
                if (c->is_moe) {
                    /* MoE: lock model data in physical RAM to prevent page-out.
                     * Without mlock, expert weights get evicted by OS memory pressure,
                     * causing 100x+ slowdown from SSD page faults.
                     * mlock may fail if ulimit is too low — fall back to MADV_WILLNEED. */
                    if (mlock(gctx->mmap_data, gctx->mmap_size) == 0) {
                        fprintf(stderr, "tq_load_gguf: mlock(%.1f GB) — expert weights pinned in RAM\n",
                                (double)gctx->mmap_size / (1024.0 * 1024.0 * 1024.0));
                    } else {
                        /* mlock failed (insufficient privilege or ulimit) — use madvise */
                        madvise(gctx->mmap_data, gctx->mmap_size, MADV_WILLNEED);
                        fprintf(stderr, "tq_load_gguf: mlock failed (errno=%d), using MADV_WILLNEED for %.1f GB\n",
                                errno, (double)gctx->mmap_size / (1024.0 * 1024.0 * 1024.0));
                    }
                } else {
                    /* Non-MoE: release mmap pages after Q4 conversion */
                    madvise(gctx->mmap_data, gctx->mmap_size, MADV_DONTNEED);
                    fprintf(stderr, "tq_load_gguf: madvise(MADV_DONTNEED) on %.1f GB mmap\n",
                            (double)gctx->mmap_size / (1024.0 * 1024.0 * 1024.0));
                }
            }
        }
#endif
    }

    #undef GGUF_KEY

    /* NOTE: No runtime RMSNorm +1 adjustment for GGUF models.
     * - Qwen2/Qwen3: standard RMSNorm (weight * norm(x)), no +1 needed.
     * - Qwen3.5/Gemma: use (1+weight) convention, but llama.cpp's GGUF
     *   converter already bakes +1 into the weights during conversion.
     *   Adding +1 at runtime would double-apply and cause activation explosion.
     * The Gemma heuristic above (mean > 2.0 check) handles the Gemma case. */

    /* Initialize persistent Metal GPU buffers for layer-level compute */
#ifdef TQ_HAS_METAL
    {
        extern int tq_metal_gpu_init_buffers(int, int, int, int);
        extern int tq_metal_gpu_init_attn(int, int, int);
        int max_q_dim = c->n_heads * c->head_dim;
        int max_kv_dim = c->n_kv_heads * c->head_dim;
        if (c->full_n_heads > 0 && c->full_head_dim > 0) {
            int full_q = c->full_n_heads * c->full_head_dim;
            int full_kv = c->full_n_kv_heads * c->full_head_dim;
            if (full_q > max_q_dim) max_q_dim = full_q;
            if (full_kv > max_kv_dim) max_kv_dim = full_kv;
        }
        tq_metal_gpu_init_buffers(c->hidden_dim, c->intermediate_dim, max_q_dim, max_kv_dim);

        /* Initialize attention + KV cache GPU buffers for compute graph forward */
        tq_metal_gpu_init_attn(c->n_heads, c->max_seq_len, max_kv_dim);
    }
#endif

    return model;
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
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
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
    hdr.delta_n_kv_heads_tqm= c->delta_n_kv_heads;
    hdr.delta_key_head_dim  = c->delta_key_head_dim;
    hdr.delta_value_head_dim= c->delta_value_head_dim;
    hdr.delta_conv_width    = c->delta_conv_width;
    hdr.partial_rotary_factor = c->partial_rotary_factor;
    hdr.use_qk_norm         = c->use_qk_norm;
    hdr.attn_output_gate    = c->attn_output_gate;

    hdr.model_type              = c->model_type;
    hdr.sliding_window          = c->sliding_window;
    hdr.rope_local_base_freq    = c->rope_local_base_freq;
    hdr.n_norms_per_block       = c->n_norms_per_block;
    hdr.query_pre_attn_scalar   = c->query_pre_attn_scalar;

    hdr.weight_quant = model->use_q8_weights ? 8 : (model->use_q4_weights ? 4 : 0);
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

        /* Gemma3 extra norms */
        if (c->model_type == 1) {
            TQM_WRITE_FP32(layer->pre_ffn_norm, dim);
            TQM_WRITE_FP32(layer->post_ffn_norm, dim);
        }

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
    for (int i = 1; i < model->_n_shards; i++) {
        if (model->_mmap_shards[i]) UnmapViewOfFile(model->_mmap_shards[i]);
    }
#else
    if (model->_mmap_data) munmap(model->_mmap_data, model->_mmap_size);
    for (int i = 1; i < model->_n_shards; i++) {
        if (model->_mmap_shards[i])
            munmap(model->_mmap_shards[i], model->_mmap_shard_sizes[i]);
    }
#endif

    free(model->_converted_data);
    free(model->_q8_data);
    free(model->_q4_data);
    free(model->_q2_data);
    free(model->attn_layer_indices);
    free(model->layer_is_sliding);

    /* Free MoE LRU caches (must happen before freeing layers) */
    tq_moe_cache_free();
#ifdef TQ_HAS_ACCELERATE
    extern void tq_moe_cblas_cache_free(void);
    tq_moe_cblas_cache_free();
#endif

    /* Free MoE resources */
    if (model->config.is_moe && model->layers) {
        for (int l = 0; l < model->config.n_layers; l++) {
            tq_moe_layer_t* moe = (tq_moe_layer_t*)model->layers[l].moe;
            if (moe) {
                /* Free Q4 expert weight data */
                if (moe->experts) {
                    for (int e = 0; e < model->config.num_experts; e++) {
                        tq_expert_weights_t* exp = &moe->experts[e];
                        free(exp->gate_q4_qs);
                        free(exp->gate_q4_scales);
                        free(exp->up_q4_qs);
                        free(exp->up_q4_scales);
                        free(exp->down_q4_qs);
                        free(exp->down_q4_scales);
                    }
                }
                /* Free shared expert Q4 data */
                free(moe->shared_expert.gate_q4_qs);
                free(moe->shared_expert.gate_q4_scales);
                free(moe->shared_expert.up_q4_qs);
                free(moe->shared_expert.up_q4_scales);
                free(moe->shared_expert.down_q4_qs);
                free(moe->shared_expert.down_q4_scales);

                free(moe->router_weight);
                free(moe->shared_gate);
                free(moe->experts);
                free(moe);
            }
        }
    }
    free(model->moe_config);
    free(model->layers);

    /* Free GGUF context (handles munmap internally) */
    if (model->gguf_ctx) {
        tq_gguf_close((tq_gguf_ctx_t*)model->gguf_ctx);
        model->_mmap_data = NULL; /* gguf_close handled it */
    }

    free(model);
}

/* ============================================================
 * 1-bit Weight Quantization
 * Converts all FP32 weights to sign bits + L2 norm per row.
 * Forward path uses tq_matmul_1bit (sign-based dot product).
 * ============================================================ */

extern void tq_quantize_row_1bit(const float*, uint8_t*, float*, int);

void tq_quantize_weights_1bit(tq_model_t* model) {
    if (!model) return;
    const tq_model_config_t* c = &model->config;
    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    int qg_dim = c->attn_output_gate ? q_dim * 2 : q_dim;
    int delta_nkv = c->delta_n_kv_heads > 0 ? c->delta_n_kv_heads : c->delta_n_heads;
    int delta_qkv_dim = delta_nkv * c->delta_key_head_dim * 2 + c->delta_n_heads * c->delta_value_head_dim;
    int delta_z_dim = c->delta_n_heads * c->delta_value_head_dim;
    int delta_dn = c->delta_n_heads;

    /* Calculate total buffer size */
    size_t total = 0;
    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];
        /* sign bits: rows * ceil(cols/8), norms: rows * 4 */
        if (layer->wq) total += (size_t)qg_dim * ((dim+7)/8 + 4);
        if (layer->wk) total += (size_t)kv_dim * ((dim+7)/8 + 4);
        if (layer->wv) total += (size_t)kv_dim * ((dim+7)/8 + 4);
        if (layer->wo) total += (size_t)dim * ((q_dim+7)/8 + 4);
        if (layer->w_gate) total += (size_t)inter * ((dim+7)/8 + 4);
        if (layer->w_up) total += (size_t)inter * ((dim+7)/8 + 4);
        if (layer->w_down) total += (size_t)dim * ((inter+7)/8 + 4);
        if (layer->delta_in_proj_qkv) total += (size_t)delta_qkv_dim * ((dim+7)/8 + 4);
        if (layer->delta_in_proj_z) total += (size_t)delta_z_dim * ((dim+7)/8 + 4);
        if (layer->delta_in_proj_a) total += (size_t)delta_dn * ((dim+7)/8 + 4);
        if (layer->delta_in_proj_b) total += (size_t)delta_dn * ((dim+7)/8 + 4);
        if (layer->delta_out_proj) total += (size_t)dim * ((delta_z_dim+7)/8 + 4);
    }

    /* We store sign bits in the Q2 fields (repurposed) and norms in Q2 scales */
    /* Actually, we need dedicated storage. Use a single buffer. */
    char* buf = (char*)malloc(total);
    if (!buf) { fprintf(stderr, "1bit: alloc failed\n"); return; }
    size_t used = 0;

    /* Helper macro */
    #define QUANTIZE_1BIT(src_ptr, rows, cols, qs_ptr, sc_ptr) do { \
        if (src_ptr) { \
            int _nb = ((cols) + 7) / 8; \
            uint8_t* _qs = (uint8_t*)(buf + used); used += (size_t)(rows) * _nb; \
            float* _sc = (float*)(buf + used); used += (size_t)(rows) * sizeof(float); \
            for (int _r = 0; _r < (rows); _r++) \
                tq_quantize_row_1bit((src_ptr) + (size_t)(_r) * (cols), \
                                      _qs + (size_t)(_r) * _nb, &_sc[_r], (cols)); \
            (qs_ptr) = _qs; (sc_ptr) = _sc; \
            (src_ptr) = NULL; \
        } \
    } while(0)

    /* Store 1-bit data in Q2 fields (repurposed).
     * Q2 packed format: 8 bytes per 32 elements = 1 byte per 4 elements.
     * 1-bit: 1 byte per 8 elements. Different layout, but we just store raw. */

    for (int l = 0; l < c->n_layers; l++) {
        tq_layer_weights_t* layer = &model->layers[l];
        QUANTIZE_1BIT(layer->wq, qg_dim, dim, layer->wq_q2, layer->wq_q2s);
        QUANTIZE_1BIT(layer->wk, kv_dim, dim, layer->wk_q2, layer->wk_q2s);
        QUANTIZE_1BIT(layer->wv, kv_dim, dim, layer->wv_q2, layer->wv_q2s);
        QUANTIZE_1BIT(layer->wo, dim, q_dim, layer->wo_q2, layer->wo_q2s);
        QUANTIZE_1BIT(layer->w_gate, inter, dim, layer->w_gate_q2, layer->w_gate_q2s);
        QUANTIZE_1BIT(layer->w_up, inter, dim, layer->w_up_q2, layer->w_up_q2s);
        QUANTIZE_1BIT(layer->w_down, dim, inter, layer->w_down_q2, layer->w_down_q2s);
        QUANTIZE_1BIT(layer->delta_in_proj_qkv, delta_qkv_dim, dim,
                       layer->delta_in_proj_qkv_q2, layer->delta_in_proj_qkv_q2s);
        QUANTIZE_1BIT(layer->delta_in_proj_z, delta_z_dim, dim,
                       layer->delta_in_proj_z_q2, layer->delta_in_proj_z_q2s);
        QUANTIZE_1BIT(layer->delta_in_proj_a, delta_dn, dim,
                       layer->delta_in_proj_a_q2, layer->delta_in_proj_a_q2s);
        QUANTIZE_1BIT(layer->delta_in_proj_b, delta_dn, dim,
                       layer->delta_in_proj_b_q2, layer->delta_in_proj_b_q2s);
        QUANTIZE_1BIT(layer->delta_out_proj, dim, delta_z_dim,
                       layer->delta_out_proj_q2, layer->delta_out_proj_q2s);
    }
    #undef QUANTIZE_1BIT

    model->use_q2_weights = 1;
    model->use_1bit_weights = 1;
    model->_q2_data = buf;
    model->_q2_size = used;

    /* Activate global 1-bit matmul routing */

    fprintf(stderr, "tq_quantize_weights_1bit: 1-bit sign hash (%zu MB)\n",
            used / (1024 * 1024));
}
