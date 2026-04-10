/**
 * tq_tokenizer.c — HuggingFace BPE tokenizer (tokenizer.json) loader
 *
 * Parses the HuggingFace tokenizer.json format:
 *   - model.vocab: { "token_string": token_id, ... }
 *   - model.merges: [ "tok_a tok_b", ... ]
 *   - added_tokens: [ { "id": N, "content": "...", ... }, ... ]
 *
 * Implements BPE encoding via iterative pair merging with merge priority.
 * Implements decoding with Qwen/GPT-style byte-level BPE (Ġ = space prefix).
 *
 * Also supports the legacy llama2.c binary tokenizer format as fallback.
 */

#include "turboquant/tq_engine.h"
#include "turboquant/tq_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#if defined(_MSC_VER) && !defined(__thread)
#define __thread __declspec(thread)
#endif

/* Global for qsort comparator (vocab index sorting) */
static char** g_vocab_for_sort;
static int cmp_vocab_idx(const void* a, const void* b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    const char* sa = g_vocab_for_sort[ia] ? g_vocab_for_sort[ia] : "";
    const char* sb = g_vocab_for_sort[ib] ? g_vocab_for_sort[ib] : "";
    return strcmp(sa, sb);
}

/* ============================================================
 * Minimal JSON helpers (reused from tq_model.c pattern)
 * ============================================================ */

static const char* skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

/* Parse a JSON string with proper escape handling.
 * Writes the unescaped string into out (up to max_len-1 chars).
 * Returns pointer past closing quote, or NULL on error. */
static const char* json_parse_string(const char* p, char* out, int max_len) {
    if (*p != '"') return NULL;
    p++;
    int i = 0;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            if (!*p) return NULL;
            switch (*p) {
                case '"':  if (i < max_len - 1) out[i++] = '"';  break;
                case '\\': if (i < max_len - 1) out[i++] = '\\'; break;
                case '/':  if (i < max_len - 1) out[i++] = '/';  break;
                case 'n':  if (i < max_len - 1) out[i++] = '\n'; break;
                case 'r':  if (i < max_len - 1) out[i++] = '\r'; break;
                case 't':  if (i < max_len - 1) out[i++] = '\t'; break;
                case 'b':  if (i < max_len - 1) out[i++] = '\b'; break;
                case 'f':  if (i < max_len - 1) out[i++] = '\f'; break;
                case 'u': {
                    /* Parse \uXXXX unicode escape */
                    unsigned int cp = 0;
                    for (int k = 0; k < 4; k++) {
                        p++;
                        if (!*p) return NULL;
                        cp <<= 4;
                        if (*p >= '0' && *p <= '9') cp |= (*p - '0');
                        else if (*p >= 'a' && *p <= 'f') cp |= (*p - 'a' + 10);
                        else if (*p >= 'A' && *p <= 'F') cp |= (*p - 'A' + 10);
                        else return NULL;
                    }
                    /* Handle surrogate pairs for codepoints > U+FFFF */
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        /* High surrogate: expect \uDCxx low surrogate */
                        if (p[1] == '\\' && p[2] == 'u') {
                            p += 3; /* skip \u */
                            unsigned int lo = 0;
                            for (int k = 0; k < 4; k++) {
                                if (!*p) return NULL;
                                lo <<= 4;
                                if (*p >= '0' && *p <= '9') lo |= (*p - '0');
                                else if (*p >= 'a' && *p <= 'f') lo |= (*p - 'a' + 10);
                                else if (*p >= 'A' && *p <= 'F') lo |= (*p - 'A' + 10);
                                p++;
                            }
                            p--; /* will be incremented at end of loop */
                            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            }
                        }
                    }
                    /* Encode codepoint as UTF-8 */
                    if (cp < 0x80) {
                        if (i < max_len - 1) out[i++] = (char)cp;
                    } else if (cp < 0x800) {
                        if (i < max_len - 2) {
                            out[i++] = (char)(0xC0 | (cp >> 6));
                            out[i++] = (char)(0x80 | (cp & 0x3F));
                        }
                    } else if (cp < 0x10000) {
                        if (i < max_len - 3) {
                            out[i++] = (char)(0xE0 | (cp >> 12));
                            out[i++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                            out[i++] = (char)(0x80 | (cp & 0x3F));
                        }
                    } else if (cp < 0x110000) {
                        if (i < max_len - 4) {
                            out[i++] = (char)(0xF0 | (cp >> 18));
                            out[i++] = (char)(0x80 | ((cp >> 12) & 0x3F));
                            out[i++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                            out[i++] = (char)(0x80 | (cp & 0x3F));
                        }
                    }
                    break;
                }
                default:
                    if (i < max_len - 1) out[i++] = *p;
                    break;
            }
        } else {
            /* Regular UTF-8 byte — copy as-is */
            if (i < max_len - 1) out[i++] = *p;
        }
        p++;
    }
    out[i] = '\0';
    if (*p == '"') p++;
    return p;
}

/* Skip a JSON value (string, number, object, array, bool, null) */
static const char* json_skip_value(const char* p) {
    p = skip_ws(p);
    if (*p == '"') {
        /* Skip string */
        p++;
        while (*p && *p != '"') {
            if (*p == '\\') { p++; if (*p) p++; }
            else p++;
        }
        if (*p == '"') p++;
    } else if (*p == '{') {
        int depth = 1; p++;
        while (*p && depth > 0) {
            if (*p == '{') depth++;
            else if (*p == '}') depth--;
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\') { p++; if (*p) p++; }
                    else p++;
                }
                if (*p == '"') p++;
                continue;
            }
            p++;
        }
    } else if (*p == '[') {
        int depth = 1; p++;
        while (*p && depth > 0) {
            if (*p == '[') depth++;
            else if (*p == ']') depth--;
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\') { p++; if (*p) p++; }
                    else p++;
                }
                if (*p == '"') p++;
                continue;
            }
            p++;
        }
    } else {
        /* number, bool, null */
        while (*p && *p != ',' && *p != '}' && *p != ']'
               && *p != ' ' && *p != '\n' && *p != '\r' && *p != '\t') {
            p++;
        }
    }
    return p;
}

/* Parse a JSON integer */
static const char* json_parse_int(const char* p, int* out) {
    p = skip_ws(p);
    int neg = 0;
    if (*p == '-') { neg = 1; p++; }
    int val = 0;
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    *out = neg ? -val : val;
    return p;
}

/* Forward declaration for str_lookup (used during merge parsing) */
static int str_lookup(const tq_tokenizer_t* tok, const char* str);

/* ============================================================
 * Detect file format: JSON starts with '{', binary starts with
 * a uint32 that is a reasonable vocab size.
 * ============================================================ */
static int is_json_file(const char* data, size_t size) {
    if (size < 4) return 0;
    /* Skip BOM if present */
    const char* p = data;
    if ((unsigned char)p[0] == 0xEF &&
        (unsigned char)p[1] == 0xBB &&
        (unsigned char)p[2] == 0xBF) {
        p += 3;
    }
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return (*p == '{');
}

/* ============================================================
 * qsort comparison for sorted_indices (by vocab string)
 * ============================================================ */
typedef struct {
    char** vocab;
} sort_ctx_t;

static sort_ctx_t g_sort_ctx;

static int compare_vocab_strings(const void* a, const void* b) {
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    return strcmp(g_sort_ctx.vocab[ia], g_sort_ctx.vocab[ib]);
}

/* Build sorted index for binary search */
static void build_sorted_index(tq_tokenizer_t* tok) {
    tok->sorted_indices = (int*)malloc((size_t)tok->vocab_size * sizeof(int));
    if (!tok->sorted_indices) return;

    int n = 0;
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->vocab[i] && tok->vocab[i][0] != '\0') {
            tok->sorted_indices[n++] = i;
        }
    }

    /* Sort using qsort with global context (simpler than passing context) */
    g_sort_ctx.vocab = tok->vocab;
    qsort(tok->sorted_indices, (size_t)n, sizeof(int), compare_vocab_strings);

    /* Store actual count of valid entries; we still keep vocab_size as capacity */
    /* Reuse sorted_indices[n..vocab_size-1] as sentinel; mark count in max_token_len if needed */
    /* Actually just zero-fill the rest */
    for (int i = n; i < tok->vocab_size; i++) {
        tok->sorted_indices[i] = -1;
    }
}

/* ============================================================
 * Load tokenizer from HuggingFace tokenizer.json
 *
 * Strategy:
 * 1. Read entire file into memory
 * 2. Navigate JSON to find "model" -> "vocab" and "merges"
 * 3. Also parse "added_tokens" for special tokens
 * 4. Build vocab array and merge table
 * ============================================================ */
static tq_tokenizer_t* load_hf_tokenizer_json(const char* data, size_t size) {
    tq_tokenizer_t* tok = (tq_tokenizer_t*)calloc(1, sizeof(tq_tokenizer_t));
    if (!tok) return NULL;

    /* First pass: scan for max token ID to determine vocab_size */
    /* Find "vocab": { ... } inside "model": { ... } */

    /* Locate "model" key at the top level */
    const char* model_start = NULL;
    {
        const char* p = data;
        p = skip_ws(p);
        if (*p != '{') { free(tok); return NULL; }
        p++;

        while (*p) {
            p = skip_ws(p);
            if (*p == '}') break;
            if (*p == ',') { p++; p = skip_ws(p); }
            if (*p == '}') break;

            char key[64];
            p = json_parse_string(p, key, sizeof(key));
            if (!p) { free(tok); return NULL; }
            p = skip_ws(p);
            if (*p != ':') { free(tok); return NULL; }
            p++;
            p = skip_ws(p);

            if (strcmp(key, "model") == 0) {
                model_start = p;
                break;
            }
            p = json_skip_value(p);
        }
    }

    if (!model_start) {
        fprintf(stderr, "tq_load_tokenizer: 'model' key not found in JSON\n");
        free(tok);
        return NULL;
    }

    /* Inside "model": { "vocab": {...}, "merges": [...], ... } */
    const char* vocab_start = NULL;
    const char* merges_start = NULL;

    {
        const char* p = model_start;
        p = skip_ws(p);
        if (*p != '{') { free(tok); return NULL; }
        p++;

        while (*p) {
            p = skip_ws(p);
            if (*p == '}') break;
            if (*p == ',') { p++; p = skip_ws(p); }
            if (*p == '}') break;

            char key[64];
            p = json_parse_string(p, key, sizeof(key));
            if (!p) { free(tok); return NULL; }
            p = skip_ws(p);
            if (*p != ':') { free(tok); return NULL; }
            p++;
            p = skip_ws(p);

            if (strcmp(key, "vocab") == 0) {
                vocab_start = p;
                p = json_skip_value(p);
            } else if (strcmp(key, "merges") == 0) {
                merges_start = p;
                p = json_skip_value(p);
            } else {
                p = json_skip_value(p);
            }
        }
    }

    if (!vocab_start) {
        fprintf(stderr, "tq_load_tokenizer: 'vocab' not found in model\n");
        free(tok);
        return NULL;
    }

    /* Parse vocab to find max ID and count entries */
    int max_id = -1;
    int n_vocab_entries = 0;
    {
        const char* p = vocab_start;
        p = skip_ws(p);
        if (*p != '{') { free(tok); return NULL; }
        p++;

        char token_str[1024];
        while (*p) {
            p = skip_ws(p);
            if (*p == '}') break;
            if (*p == ',') { p++; p = skip_ws(p); }
            if (*p == '}') break;

            p = json_parse_string(p, token_str, sizeof(token_str));
            if (!p) break;
            p = skip_ws(p);
            if (*p != ':') break;
            p++;

            int id = 0;
            p = json_parse_int(p, &id);
            if (id > max_id) max_id = id;
            n_vocab_entries++;
        }
    }

    /* Also scan added_tokens for higher IDs */
    const char* added_tokens_start = NULL;
    {
        const char* p = data;
        p = skip_ws(p);
        if (*p == '{') p++;
        while (*p) {
            p = skip_ws(p);
            if (*p == '}') break;
            if (*p == ',') { p++; p = skip_ws(p); }
            if (*p == '}') break;

            char key[64];
            p = json_parse_string(p, key, sizeof(key));
            if (!p) break;
            p = skip_ws(p);
            if (*p != ':') break;
            p++;
            p = skip_ws(p);

            if (strcmp(key, "added_tokens") == 0) {
                added_tokens_start = p;
                /* Quick scan for max id in added_tokens array */
                if (*p == '[') {
                    const char* q = p + 1;
                    while (*q) {
                        q = skip_ws(q);
                        if (*q == ']') break;
                        if (*q == ',') { q++; q = skip_ws(q); }
                        if (*q == ']') break;
                        if (*q == '{') {
                            q++;
                            while (*q && *q != '}') {
                                q = skip_ws(q);
                                if (*q == ',') { q++; q = skip_ws(q); }
                                if (*q == '}') break;
                                char akey[64];
                                q = json_parse_string(q, akey, sizeof(akey));
                                if (!q) goto done_added_scan;
                                q = skip_ws(q);
                                if (*q != ':') goto done_added_scan;
                                q++;
                                q = skip_ws(q);
                                if (strcmp(akey, "id") == 0) {
                                    int aid = 0;
                                    q = json_parse_int(q, &aid);
                                    if (aid > max_id) max_id = aid;
                                } else {
                                    q = json_skip_value(q);
                                }
                            }
                            if (*q == '}') q++;
                        } else {
                            q = json_skip_value(q);
                        }
                    }
                }
                done_added_scan:
                p = json_skip_value(p);
            } else {
                p = json_skip_value(p);
            }
        }
    }

    tok->vocab_size = max_id + 1;
    tok->max_token_len = 0;

    fprintf(stderr, "tq_load_tokenizer: vocab has %d entries, max_id=%d, total_size=%d\n",
            n_vocab_entries, max_id, tok->vocab_size);

    /* Allocate vocab array */
    tok->vocab = (char**)calloc((size_t)tok->vocab_size, sizeof(char*));
    tok->scores = (float*)calloc((size_t)tok->vocab_size, sizeof(float));
    if (!tok->vocab || !tok->scores) {
        tq_free_tokenizer(tok);
        return NULL;
    }

    /* Initialize all vocab entries to empty strings */
    for (int i = 0; i < tok->vocab_size; i++) {
        tok->vocab[i] = (char*)calloc(1, 1); /* empty string "" */
    }

    /* Second pass: populate vocab entries */
    {
        const char* p = vocab_start;
        p = skip_ws(p);
        if (*p == '{') p++;

        char token_str[1024];
        while (*p) {
            p = skip_ws(p);
            if (*p == '}') break;
            if (*p == ',') { p++; p = skip_ws(p); }
            if (*p == '}') break;

            p = json_parse_string(p, token_str, sizeof(token_str));
            if (!p) break;
            p = skip_ws(p);
            if (*p != ':') break;
            p++;

            int id = 0;
            p = json_parse_int(p, &id);

            if (id >= 0 && id < tok->vocab_size) {
                free(tok->vocab[id]);
                int len = (int)strlen(token_str);
                tok->vocab[id] = (char*)malloc((size_t)len + 1);
                if (tok->vocab[id]) {
                    memcpy(tok->vocab[id], token_str, (size_t)len + 1);
                    if (len > tok->max_token_len) tok->max_token_len = len;
                }
            }
        }
    }

    /* Parse added_tokens to fill special token entries */
    if (added_tokens_start) {
        const char* p = added_tokens_start;
        p = skip_ws(p);
        if (*p == '[') {
            p++;
            while (*p) {
                p = skip_ws(p);
                if (*p == ']') break;
                if (*p == ',') { p++; p = skip_ws(p); }
                if (*p == ']') break;

                if (*p == '{') {
                    p++;
                    int at_id = -1;
                    char at_content[256] = {0};
                    while (*p && *p != '}') {
                        p = skip_ws(p);
                        if (*p == ',') { p++; p = skip_ws(p); }
                        if (*p == '}') break;

                        char akey[64];
                        p = json_parse_string(p, akey, sizeof(akey));
                        if (!p) goto done_added;
                        p = skip_ws(p);
                        if (*p != ':') goto done_added;
                        p++;
                        p = skip_ws(p);

                        if (strcmp(akey, "id") == 0) {
                            p = json_parse_int(p, &at_id);
                        } else if (strcmp(akey, "content") == 0) {
                            p = json_parse_string(p, at_content, sizeof(at_content));
                            if (!p) goto done_added;
                        } else {
                            p = json_skip_value(p);
                        }
                    }
                    if (*p == '}') p++;

                    if (at_id >= 0 && at_id < tok->vocab_size && at_content[0]) {
                        free(tok->vocab[at_id]);
                        int len = (int)strlen(at_content);
                        tok->vocab[at_id] = (char*)malloc((size_t)len + 1);
                        if (tok->vocab[at_id]) {
                            memcpy(tok->vocab[at_id], at_content, (size_t)len + 1);
                            if (len > tok->max_token_len) tok->max_token_len = len;
                        }
                    }
                } else {
                    p = json_skip_value(p);
                }
            }
        }
    }
    done_added:

    /* Build sorted index FIRST so merge parsing can use binary search */
    build_sorted_index(tok);

    /* Parse merges: array of "token_a token_b" strings.
     * The merge priority is the index in the array (lower = higher priority).
     * We store scores so that BPE merge finds highest score first. */
    tok->n_merges = 0;
    tok->merge_pairs = NULL;

    if (merges_start) {
        /* Count merges first */
        int n_merges = 0;
        {
            const char* p = merges_start;
            p = skip_ws(p);
            if (*p == '[') {
                p++;
                while (*p) {
                    p = skip_ws(p);
                    if (*p == ']') break;
                    if (*p == ',') { p++; p = skip_ws(p); }
                    if (*p == ']') break;
                    p = json_skip_value(p);
                    n_merges++;
                }
            }
        }

        fprintf(stderr, "tq_load_tokenizer: parsing %d merges\n", n_merges);

        /* Allocate merge pairs */
        tok->merge_pairs = (int*)malloc((size_t)n_merges * 3 * sizeof(int));
        if (!tok->merge_pairs) {
            tq_free_tokenizer(tok);
            return NULL;
        }

        /* Parse merge strings using binary search for fast lookup.
         * Supports two formats:
         *   Qwen/GPT2 style: ["tok_a tok_b", ...]  (space-separated string)
         *   Gemma/SentencePiece style: [["tok_a","tok_b"], ...]  (JSON array pairs) */
        {
            const char* p = merges_start;
            p = skip_ws(p);
            if (*p == '[') p++;
            p = skip_ws(p);

            /* Detect format: if first element starts with '[', it's array-pair format */
            int array_pair_format = (*p == '[');

            int mi = 0;
            char str_a[1024], str_b[1024];
            while (*p && mi < n_merges) {
                p = skip_ws(p);
                if (*p == ']') break;
                if (*p == ',') { p++; p = skip_ws(p); }
                if (*p == ']') break;

                if (array_pair_format) {
                    /* Gemma style: ["tok_a", "tok_b"] */
                    if (*p != '[') { p = json_skip_value(p); mi++; continue; }
                    p++; /* skip '[' */
                    p = skip_ws(p);
                    p = json_parse_string(p, str_a, sizeof(str_a));
                    if (!p) break;
                    p = skip_ws(p);
                    if (*p == ',') p++;
                    p = skip_ws(p);
                    p = json_parse_string(p, str_b, sizeof(str_b));
                    if (!p) break;
                    p = skip_ws(p);
                    if (*p == ']') p++; /* skip closing ']' */
                } else {
                    /* Qwen/GPT2 style: "tok_a tok_b" */
                    char merge_str[2048];
                    p = json_parse_string(p, merge_str, sizeof(merge_str));
                    if (!p) break;
                    char* sep = strchr(merge_str, ' ');
                    if (!sep) { mi++; continue; }
                    *sep = '\0';
                    strncpy(str_a, merge_str, sizeof(str_a) - 1);
                    str_a[sizeof(str_a) - 1] = '\0';
                    strncpy(str_b, sep + 1, sizeof(str_b) - 1);
                    str_b[sizeof(str_b) - 1] = '\0';
                }

                /* Find the merged result: concatenation of tok_a + tok_b */
                char merged[2048];
                int la = (int)strlen(str_a);
                int lb = (int)strlen(str_b);
                if (la + lb >= (int)sizeof(merged)) { mi++; continue; }
                memcpy(merged, str_a, (size_t)la);
                memcpy(merged + la, str_b, (size_t)lb);
                merged[la + lb] = '\0';

                /* Look up token IDs */
                int id_a = str_lookup(tok, str_a);
                int id_b = str_lookup(tok, str_b);
                int id_merged = str_lookup(tok, merged);

                if (id_a >= 0 && id_b >= 0 && id_merged >= 0) {
                    tok->merge_pairs[tok->n_merges * 3 + 0] = id_a;
                    tok->merge_pairs[tok->n_merges * 3 + 1] = id_b;
                    tok->merge_pairs[tok->n_merges * 3 + 2] = id_merged;
                    tok->scores[id_merged] = (float)(n_merges - mi);
                    tok->n_merges++;
                }

                mi++;
            }
        }

        fprintf(stderr, "tq_load_tokenizer: loaded %d/%d merges successfully\n",
                tok->n_merges, n_merges);
    }

    fprintf(stderr, "tq_load_tokenizer: loaded %d tokens, max_len=%d, %d merges\n",
            tok->vocab_size, tok->max_token_len, tok->n_merges);
    return tok;
}

/* ============================================================
 * Load tokenizer from file (auto-detect format)
 * ============================================================ */
tq_tokenizer_t* tq_load_tokenizer(const char* path) {
    if (!path) return NULL;

    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "tq_load_tokenizer: cannot open '%s'\n", path);
        return NULL;
    }

    /* Get file size */
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0 || file_size > 200 * 1024 * 1024) {
        fprintf(stderr, "tq_load_tokenizer: invalid file size %ld\n", file_size);
        fclose(f);
        return NULL;
    }

    /* Read entire file */
    char* data = (char*)malloc((size_t)file_size + 1);
    if (!data) {
        fclose(f);
        return NULL;
    }
    size_t nread = fread(data, 1, (size_t)file_size, f);
    fclose(f);
    data[nread] = '\0';

    tq_tokenizer_t* tok = NULL;

    if (is_json_file(data, nread)) {
        fprintf(stderr, "tq_load_tokenizer: detected HuggingFace JSON format\n");
        tok = load_hf_tokenizer_json(data, nread);
    } else {
        fprintf(stderr, "tq_load_tokenizer: detected binary format — not supported for this model\n");
        fprintf(stderr, "tq_load_tokenizer: please provide a tokenizer.json file\n");
    }

    free(data);
    return tok;
}

/* ============================================================
 * Load tokenizer from TQM file (extract embedded tokenizer)
 * ============================================================ */
tq_tokenizer_t* tq_load_tokenizer_from_tqm(const char* tqm_path) {
    if (!tqm_path) return NULL;

    FILE* f = fopen(tqm_path, "rb");
    if (!f) {
        fprintf(stderr, "tq_load_tokenizer_from_tqm: cannot open '%s'\n", tqm_path);
        return NULL;
    }

    /* Read the TQM header to get tokenizer offset and size */
    uint8_t hdr_buf[512];
    if (fread(hdr_buf, 1, 512, f) != 512) {
        fprintf(stderr, "tq_load_tokenizer_from_tqm: file too small\n");
        fclose(f);
        return NULL;
    }

    uint32_t magic;
    memcpy(&magic, hdr_buf, 4);
    if (magic != 0x4D515454) { /* TQM_MAGIC */
        fprintf(stderr, "tq_load_tokenizer_from_tqm: not a TQM file\n");
        fclose(f);
        return NULL;
    }

    /* Extract tokenizer offset and size from header using offsetof */
    uint64_t tok_offset, tok_size;
    memcpy(&tok_offset, hdr_buf + offsetof(tqm_header_t, tokenizer_offset), 8);
    memcpy(&tok_size, hdr_buf + offsetof(tqm_header_t, tokenizer_size), 8);

    if (tok_size == 0) {
        fprintf(stderr, "tq_load_tokenizer_from_tqm: no embedded tokenizer\n");
        fclose(f);
        return NULL;
    }

    /* Read tokenizer data */
    char* tok_data = (char*)malloc((size_t)tok_size);
    if (!tok_data) { fclose(f); return NULL; }

    fseek(f, (long)tok_offset, SEEK_SET);
    size_t nread = fread(tok_data, 1, (size_t)tok_size, f);
    fclose(f);

    if (nread != (size_t)tok_size) {
        fprintf(stderr, "tq_load_tokenizer_from_tqm: short read (%zu/%llu)\n",
                nread, (unsigned long long)tok_size);
        free(tok_data);
        return NULL;
    }

    tq_tokenizer_t* tok = tq_load_tokenizer_from_memory(tok_data, (size_t)tok_size);
    free(tok_data);
    return tok;
}

/* ============================================================
 * Load tokenizer from memory buffer (for TQM embedded tokenizer)
 * ============================================================ */
tq_tokenizer_t* tq_load_tokenizer_from_memory(const char* data, size_t size) {
    if (!data || size == 0) return NULL;

    /* Make a null-terminated copy */
    char* buf = (char*)malloc(size + 1);
    if (!buf) return NULL;
    memcpy(buf, data, size);
    buf[size] = '\0';

    tq_tokenizer_t* tok = NULL;
    if (is_json_file(buf, size)) {
        tok = load_hf_tokenizer_json(buf, size);
    } else {
        fprintf(stderr, "tq_load_tokenizer_from_memory: unrecognized format\n");
    }

    free(buf);
    return tok;
}

/* ============================================================
 * Load tokenizer from GGUF metadata
 *
 * GGUF stores tokenizer data in metadata keys:
 *   tokenizer.ggml.tokens: string array of token strings
 *   tokenizer.ggml.scores: float array of BPE merge scores
 *   tokenizer.ggml.merges: string array of merge rules (optional)
 * ============================================================ */
tq_tokenizer_t* tq_load_tokenizer_from_gguf(const void* gguf_ctx_ptr) {
    if (!gguf_ctx_ptr) return NULL;

    const tq_gguf_ctx_t* gguf = (const tq_gguf_ctx_t*)gguf_ctx_ptr;

    /* Find the tokens array */
    int64_t tokens_idx = tq_gguf_find_key(gguf, "tokenizer.ggml.tokens");
    if (tokens_idx < 0) {
        fprintf(stderr, "tq_load_tokenizer_from_gguf: no tokenizer.ggml.tokens\n");
        return NULL;
    }

    const tq_gguf_kv_t* kv = &gguf->kv[tokens_idx];
    if (kv->type != TQ_GGUF_TYPE_ARRAY || kv->value.array.elem_type != TQ_GGUF_TYPE_STRING) {
        fprintf(stderr, "tq_load_tokenizer_from_gguf: tokens is not a string array\n");
        return NULL;
    }

    uint64_t vocab_size = kv->value.array.count;
    if (vocab_size == 0 || vocab_size > 1000000) {
        fprintf(stderr, "tq_load_tokenizer_from_gguf: invalid vocab_size=%llu\n",
                (unsigned long long)vocab_size);
        return NULL;
    }

    tq_tokenizer_t* tok = (tq_tokenizer_t*)calloc(1, sizeof(tq_tokenizer_t));
    if (!tok) return NULL;

    tok->vocab_size = (int)vocab_size;
    tok->vocab = (char**)calloc(vocab_size, sizeof(char*));
    tok->scores = (float*)calloc(vocab_size, sizeof(float));
    if (!tok->vocab || !tok->scores) {
        free(tok->vocab);
        free(tok->scores);
        free(tok);
        return NULL;
    }

    /* Copy token strings from GGUF string array.
     * The array data contains tq_gguf_string_t structs laid out sequentially. */
    tq_gguf_string_t* strings = (tq_gguf_string_t*)kv->value.array.data;
    int max_len = 0;
    for (uint64_t i = 0; i < vocab_size; i++) {
        if (strings[i].str && strings[i].len > 0) {
            tok->vocab[i] = (char*)malloc((size_t)strings[i].len + 1);
            if (tok->vocab[i]) {
                memcpy(tok->vocab[i], strings[i].str, (size_t)strings[i].len);
                tok->vocab[i][strings[i].len] = '\0';
                if ((int)strings[i].len > max_len) max_len = (int)strings[i].len;
            }
        } else {
            tok->vocab[i] = (char*)calloc(1, 1); /* empty string */
        }
    }
    tok->max_token_len = max_len;

    /* Load scores if available */
    int64_t scores_idx = tq_gguf_find_key(gguf, "tokenizer.ggml.scores");
    if (scores_idx >= 0) {
        const tq_gguf_kv_t* skv = &gguf->kv[scores_idx];
        if (skv->type == TQ_GGUF_TYPE_ARRAY &&
            skv->value.array.elem_type == TQ_GGUF_TYPE_FLOAT32 &&
            skv->value.array.count == vocab_size) {
            memcpy(tok->scores, skv->value.array.data, vocab_size * sizeof(float));
        }
    }

    /* Build sorted indices BEFORE merge parsing so str_lookup() can use
     * binary search instead of O(n) linear scan.  For 248K vocab with
     * ~50K merges (3 lookups each), this turns a ~10 s init into ~100 ms. */
    tok->sorted_indices = (int*)malloc(vocab_size * sizeof(int));
    if (tok->sorted_indices) {
        for (int i = 0; i < (int)vocab_size; i++) tok->sorted_indices[i] = i;
        g_vocab_for_sort = tok->vocab;
        qsort(tok->sorted_indices, vocab_size, sizeof(int), cmp_vocab_idx);
    }

    /* Load and parse merges if available.
     * GGUF stores merges as a string array of "tok_a tok_b" pairs.
     * We need to look up token IDs and build (id_a, id_b, id_merged) triples
     * so the BPE encoder can use them. */
    int64_t merges_idx = tq_gguf_find_key(gguf, "tokenizer.ggml.merges");
    if (merges_idx >= 0) {
        const tq_gguf_kv_t* mkv = &gguf->kv[merges_idx];
        if (mkv->type == TQ_GGUF_TYPE_ARRAY &&
            mkv->value.array.elem_type == TQ_GGUF_TYPE_STRING) {
            uint64_t n_merges_total = mkv->value.array.count;
            tok->merge_pairs = (int*)malloc(n_merges_total * 3 * sizeof(int));
            tok->n_merges = 0;
            if (tok->merge_pairs) {
                tq_gguf_string_t* merge_strings = (tq_gguf_string_t*)mkv->value.array.data;
                for (uint64_t mi = 0; mi < n_merges_total; mi++) {
                    if (!merge_strings[mi].str || merge_strings[mi].len == 0) continue;

                    /* Copy merge string and split on space: "tok_a tok_b" */
                    char buf[2048];
                    int slen = (int)merge_strings[mi].len;
                    if (slen >= (int)sizeof(buf)) continue;
                    memcpy(buf, merge_strings[mi].str, (size_t)slen);
                    buf[slen] = '\0';

                    char* sep = strchr(buf, ' ');
                    if (!sep) continue;
                    *sep = '\0';
                    const char* str_a = buf;
                    const char* str_b = sep + 1;

                    /* Build merged string: concatenation of tok_a + tok_b */
                    char merged[2048];
                    int la = (int)strlen(str_a);
                    int lb = (int)strlen(str_b);
                    if (la + lb >= (int)sizeof(merged)) continue;
                    memcpy(merged, str_a, (size_t)la);
                    memcpy(merged + la, str_b, (size_t)lb);
                    merged[la + lb] = '\0';

                    /* Look up token IDs via binary search (sorted_indices built above) */
                    int id_a = str_lookup(tok, str_a);
                    int id_b = str_lookup(tok, str_b);
                    int id_merged = str_lookup(tok, merged);

                    if (id_a >= 0 && id_b >= 0 && id_merged >= 0) {
                        tok->merge_pairs[tok->n_merges * 3 + 0] = id_a;
                        tok->merge_pairs[tok->n_merges * 3 + 1] = id_b;
                        tok->merge_pairs[tok->n_merges * 3 + 2] = id_merged;
                        /* Priority: earlier merges in GGUF = higher priority */
                        tok->scores[id_merged] = (float)(n_merges_total - mi);
                        tok->n_merges++;
                    }
                }
                fprintf(stderr, "tq_load_tokenizer_from_gguf: parsed %d/%d merges\n",
                        tok->n_merges, (int)n_merges_total);
            }
        }
    }

    fprintf(stderr, "tq_load_tokenizer_from_gguf: loaded %d tokens (max_len=%d)\n",
            tok->vocab_size, tok->max_token_len);
    return tok;
}

/* ============================================================
 * Free tokenizer
 * ============================================================ */
void tq_free_tokenizer(tq_tokenizer_t* tok) {
    if (!tok) return;
    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i]);
        }
        free(tok->vocab);
    }
    free(tok->scores);
    free(tok->sorted_indices);
    free(tok->merge_pairs);
    free(tok);
}

/* ============================================================
 * Lookup token ID by string (binary search on sorted index)
 * ============================================================ */
static int str_lookup(const tq_tokenizer_t* tok, const char* str) {
    if (!tok->sorted_indices) {
        /* Fallback: linear scan */
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->vocab[i] && strcmp(tok->vocab[i], str) == 0) return i;
        }
        return -1;
    }

    /* Binary search */
    int lo = 0, hi = tok->vocab_size - 1;
    /* Find the actual valid range (entries with sorted_indices >= 0) */
    while (hi >= 0 && tok->sorted_indices[hi] < 0) hi--;
    if (hi < 0) return -1;

    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int idx = tok->sorted_indices[mid];
        if (idx < 0) { hi = mid - 1; continue; }
        int cmp = strcmp(str, tok->vocab[idx]);
        if (cmp == 0) return idx;
        if (cmp < 0) hi = mid - 1;
        else lo = mid + 1;
    }
    return -1;
}

/* ============================================================
 * GPT/Qwen byte-level BPE: map from BPE byte representation
 * to actual UTF-8 bytes.
 *
 * In Qwen/GPT tokenizers, bytes 0x00-0xFF are represented using
 * a specific mapping where printable ASCII is kept as-is but
 * non-printable bytes are mapped to Unicode characters starting
 * at U+0100 (Ġ = U+0120 = space, etc.).
 *
 * The mapping for decoding:
 *   - Ġ (U+0120, UTF-8: C4 A0) -> space (0x20)
 *   - Ċ (U+010A, UTF-8: C4 8A) -> newline (0x0A)
 *   - Characters U+0100-U+01FF map to bytes 0x00-0xFF
 *     where the byte value = codepoint - 0x100
 *     (but only for the "shifted" ones; printable ASCII stays)
 * ============================================================ */

/* Build the GPT2 byte-to-char map on first use.
 * The map: for each byte value 0-255, what character(s) represent it
 * in the BPE vocabulary.
 *
 * GPT2 byte encoder:
 *   - bytes 33-126 ('!' to '~') map to themselves
 *   - bytes 161-172 map to themselves
 *   - bytes 174-255 map to themselves
 *   - all other bytes (0-32, 127-160, 173) map to 256+offset
 *
 * For decoding, we need the REVERSE: given a BPE character, what byte?
 */

/* Decode a single BPE vocab string to raw UTF-8 bytes.
 * Handles the Ġ -> space mapping used by GPT2/Qwen tokenizers.
 * Returns decoded string in a static thread-local buffer. */
static const char* decode_bpe_token(const char* piece) {
    static char decode_buf[1024];
    int out = 0;
    const unsigned char* p = (const unsigned char*)piece;

    while (*p && out < (int)sizeof(decode_buf) - 4) {
        if (*p < 0x80) {
            /* ASCII — direct */
            decode_buf[out++] = (char)*p;
            p++;
        } else if ((*p & 0xE0) == 0xC0 && (p[1] & 0xC0) == 0x80) {
            /* 2-byte UTF-8: decode codepoint */
            unsigned int cp = ((unsigned int)(*p & 0x1F) << 6) | (p[1] & 0x3F);
            if (cp >= 0x100 && cp <= 0x1FF) {
                /* GPT2 byte mapping: codepoint - 0x100 is a raw byte
                 * Actually the mapping is more nuanced. Let's use the
                 * standard GPT2 byte decoder. */
                /* The GPT2 bytes_to_unicode creates a bijection.
                 * Codepoints 0x100+ represent specific bytes. */
                /* Simple approach: cp in [0x100, 0x14F] maps to bytes that
                 * aren't in the "direct" set. Build lookup. */
                int byte_val = -1;
                /* GPT2 direct bytes: 33-126, 161-172, 174-255 */
                /* Indirect bytes get codepoints starting at 256 (0x100) */
                /* Build the indirect byte list */
                static int indirect_map_built = 0;
                static unsigned char indirect_to_byte[256];
                if (!indirect_map_built) {
                    int n = 0;
                    for (int b = 0; b < 256; b++) {
                        int direct = 0;
                        if (b >= 33 && b <= 126) direct = 1;
                        if (b >= 161 && b <= 172) direct = 1;
                        if (b >= 174 && b <= 255) direct = 1;
                        if (!direct) {
                            indirect_to_byte[n++] = (unsigned char)b;
                        }
                    }
                    indirect_map_built = 1;
                }
                int idx = (int)cp - 256;
                if (idx >= 0 && idx < 69) { /* 69 indirect bytes */
                    byte_val = indirect_to_byte[idx];
                }
                if (byte_val >= 0) {
                    decode_buf[out++] = (char)(unsigned char)byte_val;
                } else {
                    /* Fallback: copy UTF-8 bytes as-is */
                    decode_buf[out++] = (char)p[0];
                    decode_buf[out++] = (char)p[1];
                }
            } else {
                /* Regular 2-byte UTF-8 char (e.g., accented letters) */
                decode_buf[out++] = (char)p[0];
                decode_buf[out++] = (char)p[1];
            }
            p += 2;
        } else if ((*p & 0xF0) == 0xE0 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80) {
            /* 3-byte UTF-8 */
            decode_buf[out++] = (char)p[0];
            decode_buf[out++] = (char)p[1];
            decode_buf[out++] = (char)p[2];
            p += 3;
        } else if ((*p & 0xF8) == 0xF0 && (p[1] & 0xC0) == 0x80 &&
                   (p[2] & 0xC0) == 0x80 && (p[3] & 0xC0) == 0x80) {
            /* 4-byte UTF-8 */
            decode_buf[out++] = (char)p[0];
            decode_buf[out++] = (char)p[1];
            decode_buf[out++] = (char)p[2];
            decode_buf[out++] = (char)p[3];
            p += 4;
        } else {
            /* Invalid UTF-8 — copy byte */
            decode_buf[out++] = (char)*p;
            p++;
        }
    }
    decode_buf[out] = '\0';
    return decode_buf;
}

/* ============================================================
 * Encode text to tokens using BPE merge
 *
 * For GPT2/Qwen byte-level BPE:
 * 1. Convert each byte to its BPE character representation
 * 2. Look up each character as initial token
 * 3. Iteratively merge the highest-priority pair
 * ============================================================ */

/* Encode a single byte to its GPT2 BPE character representation */
static int encode_byte_to_bpe_char(unsigned char byte, char* out) {
    /* Direct bytes: 33-126, 161-172, 174-255 -> same codepoint */
    int direct = 0;
    if (byte >= 33 && byte <= 126) direct = 1;
    if (byte >= 161 && byte <= 172) direct = 1;
    if (byte >= 174) direct = 1; /* upper range always fits in uint8 */

    if (direct) {
        out[0] = (char)byte;
        out[1] = '\0';
        return 1;
    }

    /* Indirect bytes -> codepoint 256 + index */
    static unsigned char byte_order[69];
    static int order_built = 0;
    if (!order_built) {
        int n = 0;
        for (int b = 0; b < 256; b++) {
            int d = 0;
            if (b >= 33 && b <= 126) d = 1;
            if (b >= 161 && b <= 172) d = 1;
            if (b >= 174 && b <= 255) d = 1;
            if (!d) byte_order[n++] = (unsigned char)b;
        }
        order_built = 1;
    }

    /* Find index of this byte in indirect list */
    int idx = -1;
    for (int i = 0; i < 69; i++) {
        if (byte_order[i] == byte) { idx = i; break; }
    }
    if (idx < 0) { out[0] = (char)byte; out[1] = '\0'; return 1; }

    unsigned int cp = 256 + (unsigned int)idx;
    /* Encode codepoint as UTF-8 */
    if (cp < 0x80) {
        out[0] = (char)cp;
        out[1] = '\0';
        return 1;
    } else {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        out[2] = '\0';
        return 2;
    }
}

int tq_encode(const tq_tokenizer_t* tok, const char* text,
              int* tokens, int max_tokens, int add_bos) {
    if (!tok || !text || !tokens || max_tokens <= 0) return 0;

    int n_tokens = 0;

    /* Add BOS token if requested.
     * Gemma: BOS=2, Qwen: no BOS (uses <|im_start|> instead) */
    if (add_bos) {
        /* Look up <bos> token in vocab; default to id 2 (Gemma convention) */
        int bos_id = str_lookup(tok, "<bos>");
        if (bos_id < 0) { bos_id = str_lookup(tok, "<|im_start|>"); }
        if (bos_id >= 0) {
            tokens[n_tokens++] = bos_id;
        }
    }

    if (*text == '\0') return n_tokens;

    /* Detect tokenizer style: Gemma uses ▁ (U+2581) for spaces in vocab,
     * GPT2/Qwen/Llama3 uses byte-level BPE with Ġ/ĉ encoding.
     * Heuristic: ▁ in vocab AND vocab_size < 100K → SentencePiece.
     * Llama 3.x (128K vocab) has ▁ from the base model but uses tiktoken
     * (GPT-style BPE). Using the sentencepiece path for these models drops
     * most characters and produces far too few tokens. */
    int has_spm_marker = (str_lookup(tok, "\xe2\x96\x81") >= 0);
    int is_sentencepiece = has_spm_marker && tok->vocab_size < 100000;
    static int dbg_once = 0;
    if (!dbg_once) {
        fprintf(stderr, "[tokenizer] vocab=%d, spm_marker=%d, is_sentencepiece=%d\n",
                tok->vocab_size, has_spm_marker, is_sentencepiece);
        dbg_once = 1;
    }

    int text_len = (int)strlen(text);

    if (is_sentencepiece) {
        /* SentencePiece-style: replace spaces with ▁, then split into UTF-8 characters.
         * Each character is looked up in vocab directly. */
        /* First, build normalized text with ▁ replacing spaces, and ▁ prepended */
        int norm_cap = text_len * 4 + 16;
        char* norm = (char*)malloc((size_t)norm_cap);
        if (!norm) return n_tokens;
        int ni = 0;
        /* Prepend ▁ (space before first word, SentencePiece convention) */
        norm[ni++] = (char)0xE2; norm[ni++] = (char)0x96; norm[ni++] = (char)0x81;
        for (int i = 0; i < text_len; i++) {
            if (text[i] == ' ') {
                norm[ni++] = (char)0xE2; norm[ni++] = (char)0x96; norm[ni++] = (char)0x81;
            } else {
                norm[ni++] = text[i];
            }
        }
        norm[ni] = '\0';

        /* Split into individual UTF-8 characters */
        for (int i = 0; i < ni && n_tokens < max_tokens; ) {
            /* Determine UTF-8 character length */
            unsigned char c = (unsigned char)norm[i];
            int clen = 1;
            if (c >= 0xF0) { clen = 4; }
            else if (c >= 0xE0) { clen = 3; }
            else if (c >= 0xC0) { clen = 2; }
            if (i + clen > ni) break;

            char ch_str[8];
            memcpy(ch_str, norm + i, (size_t)clen);
            ch_str[clen] = '\0';

            int id = str_lookup(tok, ch_str);
            if (id >= 0) {
                tokens[n_tokens++] = id;
            }
            /* If not found, skip (byte fallback tokens handle this in merges) */
            i += clen;
        }
        free(norm);
    } else {
        /* GPT2/Qwen byte-level BPE: each byte maps to a BPE character token */
        for (int i = 0; i < text_len && n_tokens < max_tokens; i++) {
            unsigned char byte = (unsigned char)text[i];
            char bpe_char[4];
            encode_byte_to_bpe_char(byte, bpe_char);

            int id = str_lookup(tok, bpe_char);
            if (id >= 0) {
                tokens[n_tokens++] = id;
            } else {
                char direct[2] = { (char)byte, '\0' };
                id = str_lookup(tok, direct);
                if (id >= 0) {
                    tokens[n_tokens++] = id;
                }
            }
        }
    }

    /* BPE merge pass using a max-heap for O(n log n) instead of O(n²).
     *
     * The naive algorithm scans all pairs on each merge step → O(n²).
     * For 17K initial tokens (GPT2 byte-level), that's ~289M ops = minutes.
     *
     * Heap approach:
     * 1. Build a heap of all mergeable consecutive pairs (score, position)
     * 2. Pop max-score pair, apply merge, invalidate stale entries
     * 3. Insert new pairs formed at the merge point
     * 4. O(n log n) total: n initial inserts + n pops + O(1) updates each
     *
     * We use a simple binary max-heap with lazy deletion (stale entries
     * are skipped when popped, identified by a generation counter). */
    {
        /* Linked list for O(1) neighbor access after merges */
        int* prev = (int*)malloc((size_t)n_tokens * sizeof(int));
        int* next = (int*)malloc((size_t)n_tokens * sizeof(int));
        if (!prev || !next) { free(prev); free(next); return n_tokens; }
        for (int i = 0; i < n_tokens; i++) { prev[i] = i - 1; next[i] = i + 1; }

        /* Heap entry: (score, left_pos, merge_id, generation) */
        typedef struct { float score; int pos; int merge_id; int gen; } heap_entry_t;
        int heap_cap = n_tokens + 16;
        heap_entry_t* heap = (heap_entry_t*)malloc((size_t)heap_cap * sizeof(heap_entry_t));
        int* gen = (int*)calloc((size_t)n_tokens, sizeof(int)); /* per-position generation */
        if (!heap || !gen) { free(prev); free(next); free(heap); free(gen); return n_tokens; }
        int heap_size = 0;

        /* Heap helpers (max-heap by score) */
        #define HEAP_PARENT(i) (((i)-1)/2)
        #define HEAP_LEFT(i)   (2*(i)+1)
        #define HEAP_RIGHT(i)  (2*(i)+2)
        #define HEAP_SWAP(a,b) { heap_entry_t _t = heap[a]; heap[a] = heap[b]; heap[b] = _t; }

        void* _dummy_ptr = NULL; (void)_dummy_ptr; /* suppress unused warning */

        /* Sift up */
        int sift_up_idx = 0;
        #define SIFT_UP(idx) do { \
            sift_up_idx = (idx); \
            while (sift_up_idx > 0 && heap[sift_up_idx].score > heap[HEAP_PARENT(sift_up_idx)].score) { \
                HEAP_SWAP(sift_up_idx, HEAP_PARENT(sift_up_idx)); \
                sift_up_idx = HEAP_PARENT(sift_up_idx); \
            } \
        } while(0)

        /* Sift down */
        #define SIFT_DOWN(idx) do { \
            int _si = (idx); \
            for (;;) { \
                int _best = _si; \
                int _l = HEAP_LEFT(_si), _r = HEAP_RIGHT(_si); \
                if (_l < heap_size && heap[_l].score > heap[_best].score) _best = _l; \
                if (_r < heap_size && heap[_r].score > heap[_best].score) _best = _r; \
                if (_best == _si) break; \
                HEAP_SWAP(_si, _best); _si = _best; \
            } \
        } while(0)

        /* Try to create a merge entry for position i and its next neighbor */
        #define TRY_INSERT_PAIR(i) do { \
            int _ni = next[i]; \
            if (_ni < n_tokens && tokens[_ni] >= 0) { \
                const char* _s1 = tok->vocab[tokens[i]]; \
                const char* _s2 = tok->vocab[tokens[_ni]]; \
                int _l1 = (int)strlen(_s1), _l2 = (int)strlen(_s2); \
                if (_l1 + _l2 < 512) { \
                    char _m[512]; memcpy(_m, _s1, _l1); memcpy(_m+_l1, _s2, _l2); _m[_l1+_l2]=0; \
                    int _mid = str_lookup(tok, _m); \
                    if (_mid >= 0) { \
                        if (heap_size >= heap_cap) { heap_cap *= 2; heap = realloc(heap, (size_t)heap_cap * sizeof(heap_entry_t)); } \
                        heap[heap_size] = (heap_entry_t){tok->scores[_mid], (i), _mid, gen[i]}; \
                        SIFT_UP(heap_size); heap_size++; \
                    } \
                } \
            } \
        } while(0)

        /* Build initial heap */
        for (int i = 0; i < n_tokens - 1; i++) {
            int ni = next[i];
            if (ni < n_tokens) {
                const char* s1 = tok->vocab[tokens[i]];
                const char* s2 = tok->vocab[tokens[ni]];
                int l1 = (int)strlen(s1), l2 = (int)strlen(s2);
                if (l1 + l2 < 512) {
                    char merged[512];
                    memcpy(merged, s1, (size_t)l1);
                    memcpy(merged + l1, s2, (size_t)l2);
                    merged[l1 + l2] = '\0';
                    int mid = str_lookup(tok, merged);
                    if (mid >= 0) {
                        if (heap_size >= heap_cap) { heap_cap *= 2; heap = realloc(heap, (size_t)heap_cap * sizeof(heap_entry_t)); }
                        heap[heap_size] = (heap_entry_t){tok->scores[mid], i, mid, 0};
                        SIFT_UP(heap_size);
                        heap_size++;
                    }
                }
            }
        }

        /* Merge loop */
        int active_count = n_tokens;
        while (heap_size > 0 && active_count >= 2) {
            /* Pop max */
            heap_entry_t top = heap[0];
            heap[0] = heap[--heap_size];
            if (heap_size > 0) { SIFT_DOWN(0); }

            /* Check if stale (position was already merged) */
            if (top.gen != gen[top.pos]) continue;
            int ri = next[top.pos];
            if (ri >= n_tokens || tokens[ri] < 0) continue;

            /* Apply merge: left absorbs right */
            tokens[top.pos] = top.merge_id;
            tokens[ri] = -1; /* mark dead */
            gen[top.pos]++;  /* invalidate old entries for this position */

            /* Update linked list: skip the dead right node */
            int rr = next[ri];
            next[top.pos] = rr;
            if (rr < n_tokens) prev[rr] = top.pos;
            active_count--;

            /* Insert new pairs: (prev_of_left, left) and (left, next_of_right) */
            if (prev[top.pos] >= 0 && tokens[prev[top.pos]] >= 0) {
                gen[prev[top.pos]]++;
                TRY_INSERT_PAIR(prev[top.pos]);
            }
            if (next[top.pos] < n_tokens && tokens[next[top.pos]] >= 0) {
                TRY_INSERT_PAIR(top.pos);
            }
        }

        /* Compact: remove dead tokens */
        int out = 0;
        for (int i = 0; i < n_tokens; i++) {
            if (tokens[i] >= 0) tokens[out++] = tokens[i];
        }
        n_tokens = out;

        free(prev); free(next); free(heap); free(gen);
        #undef HEAP_PARENT
        #undef HEAP_LEFT
        #undef HEAP_RIGHT
        #undef HEAP_SWAP
        #undef SIFT_UP
        #undef SIFT_DOWN
        #undef TRY_INSERT_PAIR
    }

    return n_tokens;
}

/* ============================================================
 * Decode single token to string
 *
 * For GPT2/Qwen byte-level BPE, the token string uses special
 * Unicode characters (Ġ etc.) to represent bytes. We decode
 * these back to actual UTF-8 bytes.
 * ============================================================ */
const char* tq_decode(const tq_tokenizer_t* tok, int prev_token, int token) {
    if (!tok || token < 0 || token >= tok->vocab_size) return "";

    const char* piece = tok->vocab[token];
    if (!piece || piece[0] == '\0') return "";

    /* Check if this is a special token (e.g., <|endoftext|>) */
    if (piece[0] == '<' && piece[1] == '|') {
        return ""; /* Don't output special tokens as text */
    }

    /* SentencePiece: replace ▁ (U+2581) with space */
    if (strstr(piece, "\xe2\x96\x81") != NULL) {
        static __thread char sp_buf[1024];
        int j = 0;
        for (int i = 0; piece[i] && j < (int)sizeof(sp_buf) - 1; ) {
            if ((unsigned char)piece[i] == 0xE2 &&
                (unsigned char)piece[i+1] == 0x96 &&
                (unsigned char)piece[i+2] == 0x81) {
                sp_buf[j++] = ' ';
                i += 3;
            } else {
                sp_buf[j++] = piece[i++];
            }
        }
        sp_buf[j] = '\0';
        return sp_buf;
    }

    /* GPT2/Qwen: decode BPE byte representation to actual UTF-8 */
    return decode_bpe_token(piece);
}
