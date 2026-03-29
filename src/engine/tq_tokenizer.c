/**
 * tq_tokenizer.c — Minimal BPE tokenizer for LLM inference
 *
 * Supports loading vocabulary from a simple binary format:
 *   - uint32: vocab_size
 *   - uint32: max_token_length
 *   - For each token:
 *     - float32: BPE merge score
 *     - uint32: token string length
 *     - bytes: token string (NOT null-terminated in file)
 *
 * This is the same format used by llama2.c / llama.cpp tokenizer files.
 *
 * Encoding uses the greedy BPE merge algorithm:
 *   1. Start with individual UTF-8 characters
 *   2. Repeatedly merge the pair with highest score
 *   3. Continue until no more merges possible
 */

#include "turboquant/tq_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Load tokenizer from binary vocab file
 * ============================================================ */
tq_tokenizer_t* tq_load_tokenizer(const char* path) {
    if (!path) return NULL;

    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "tq_load_tokenizer: cannot open '%s'\n", path);
        return NULL;
    }

    tq_tokenizer_t* tok = (tq_tokenizer_t*)calloc(1, sizeof(tq_tokenizer_t));
    if (!tok) { fclose(f); return NULL; }

    /* Read header */
    uint32_t vocab_size, max_token_len;
    if (fread(&vocab_size, sizeof(uint32_t), 1, f) != 1 ||
        fread(&max_token_len, sizeof(uint32_t), 1, f) != 1) {
        fprintf(stderr, "tq_load_tokenizer: header read error\n");
        free(tok);
        fclose(f);
        return NULL;
    }

    tok->vocab_size = (int)vocab_size;
    tok->max_token_len = (int)max_token_len;

    /* Allocate vocab and scores */
    tok->vocab = (char**)calloc(vocab_size, sizeof(char*));
    tok->scores = (float*)calloc(vocab_size, sizeof(float));
    if (!tok->vocab || !tok->scores) {
        tq_free_tokenizer(tok);
        fclose(f);
        return NULL;
    }

    /* Read each token */
    for (uint32_t i = 0; i < vocab_size; i++) {
        float score;
        uint32_t len;
        if (fread(&score, sizeof(float), 1, f) != 1 ||
            fread(&len, sizeof(uint32_t), 1, f) != 1) {
            fprintf(stderr, "tq_load_tokenizer: truncated at token %u\n", i);
            tq_free_tokenizer(tok);
            fclose(f);
            return NULL;
        }
        tok->scores[i] = score;

        tok->vocab[i] = (char*)malloc(len + 1);
        if (!tok->vocab[i]) {
            tq_free_tokenizer(tok);
            fclose(f);
            return NULL;
        }
        if (len > 0 && fread(tok->vocab[i], 1, len, f) != len) {
            fprintf(stderr, "tq_load_tokenizer: truncated token string at %u\n", i);
            tq_free_tokenizer(tok);
            fclose(f);
            return NULL;
        }
        tok->vocab[i][len] = '\0';
    }

    fclose(f);

    /* Build sorted index for efficient encoding lookup */
    tok->sorted_indices = (int*)malloc(vocab_size * sizeof(int));
    if (tok->sorted_indices) {
        for (int i = 0; i < tok->vocab_size; i++) {
            tok->sorted_indices[i] = i;
        }
        /* Simple insertion sort by token string — sufficient for typical vocab sizes
         * during initial load; not on the hot path */
        for (int i = 1; i < tok->vocab_size; i++) {
            int key = tok->sorted_indices[i];
            int j = i - 1;
            while (j >= 0 &&
                   strcmp(tok->vocab[tok->sorted_indices[j]], tok->vocab[key]) > 0) {
                tok->sorted_indices[j + 1] = tok->sorted_indices[j];
                j--;
            }
            tok->sorted_indices[j + 1] = key;
        }
    }

    fprintf(stderr, "tq_load_tokenizer: loaded %d tokens, max_len=%d\n",
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
    free(tok);
}

/* ============================================================
 * Lookup token ID by string (binary search on sorted index)
 * ============================================================ */
static int str_lookup(const tq_tokenizer_t* tok, const char* str) {
    if (!tok->sorted_indices) {
        /* Fallback: linear scan */
        for (int i = 0; i < tok->vocab_size; i++) {
            if (strcmp(tok->vocab[i], str) == 0) return i;
        }
        return -1;
    }

    /* Binary search */
    int lo = 0, hi = tok->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int cmp = strcmp(str, tok->vocab[tok->sorted_indices[mid]]);
        if (cmp == 0) return tok->sorted_indices[mid];
        if (cmp < 0) hi = mid - 1;
        else lo = mid + 1;
    }
    return -1;
}

/* ============================================================
 * Encode text to tokens using BPE merge
 *
 * Algorithm:
 * 1. Convert each UTF-8 character/byte to its token ID
 * 2. Repeatedly find the consecutive pair with highest merge score
 * 3. Replace that pair with the merged token
 * 4. Repeat until no more merges are possible
 * ============================================================ */
int tq_encode(const tq_tokenizer_t* tok, const char* text,
              int* tokens, int max_tokens, int add_bos) {
    if (!tok || !text || !tokens || max_tokens <= 0) return 0;

    int n_tokens = 0;

    /* Optionally add BOS token (token ID 1 by convention) */
    if (add_bos && n_tokens < max_tokens) {
        tokens[n_tokens++] = 1;
    }

    /* Handle empty input */
    if (*text == '\0') return n_tokens;

    /* First pass: encode each UTF-8 character as individual tokens */
    /* Prefix space handling: first character may get a space prefix */
    int text_len = (int)strlen(text);
    char str_buf[8]; /* enough for any UTF-8 character + prefix */

    for (int i = 0; i < text_len && n_tokens < max_tokens; ) {
        /* Determine UTF-8 character length */
        unsigned char c = (unsigned char)text[i];
        int char_len = 1;
        if ((c & 0x80) == 0)        char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        /* Copy character to buffer */
        if (char_len > text_len - i) char_len = text_len - i;
        memcpy(str_buf, text + i, char_len);
        str_buf[char_len] = '\0';

        int id = str_lookup(tok, str_buf);
        if (id >= 0) {
            tokens[n_tokens++] = id;
        } else {
            /* Fallback: encode as raw bytes (token IDs 3..258 typically) */
            for (int b = 0; b < char_len && n_tokens < max_tokens; b++) {
                /* Try byte token */
                unsigned char byte = (unsigned char)text[i + b];
                char byte_str[8];
                snprintf(byte_str, sizeof(byte_str), "<0x%02X>", byte);
                int byte_id = str_lookup(tok, byte_str);
                if (byte_id >= 0) {
                    tokens[n_tokens++] = byte_id;
                } else {
                    /* Last resort: UNK token (0) */
                    tokens[n_tokens++] = 0;
                }
            }
        }
        i += char_len;
    }

    /* BPE merge pass: repeatedly merge the best pair */
    while (n_tokens >= 2) {
        float best_score = -1e30f;
        int best_idx = -1;
        int best_id = -1;

        /* Find the merge with highest score */
        for (int i = 0; i < n_tokens - 1; i++) {
            /* Construct merged string */
            const char* s1 = tok->vocab[tokens[i]];
            const char* s2 = tok->vocab[tokens[i + 1]];
            int len1 = (int)strlen(s1);
            int len2 = (int)strlen(s2);

            if (len1 + len2 >= tok->max_token_len) continue;

            char merged[512];
            memcpy(merged, s1, len1);
            memcpy(merged + len1, s2, len2);
            merged[len1 + len2] = '\0';

            int id = str_lookup(tok, merged);
            if (id >= 0 && tok->scores[id] > best_score) {
                best_score = tok->scores[id];
                best_idx = i;
                best_id = id;
            }
        }

        if (best_idx < 0) break; /* No more merges possible */

        /* Apply the merge: replace tokens[best_idx] with merged token,
         * shift remaining tokens left */
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < n_tokens - 1; i++) {
            tokens[i] = tokens[i + 1];
        }
        n_tokens--;
    }

    return n_tokens;
}

/* ============================================================
 * Decode single token to string
 *
 * Handles special byte tokens like <0xAB>
 * ============================================================ */
const char* tq_decode(const tq_tokenizer_t* tok, int prev_token, int token) {
    if (!tok || token < 0 || token >= tok->vocab_size) return "";

    const char* piece = tok->vocab[token];

    /* Handle byte fallback tokens: <0xAB> -> raw byte */
    /* These are returned as static buffer (not thread-safe, but simple) */
    static char byte_buf[2];
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x') {
        unsigned int byte_val = 0;
        if (sscanf(piece, "<0x%02X>", &byte_val) == 1) {
            byte_buf[0] = (char)byte_val;
            byte_buf[1] = '\0';
            return byte_buf;
        }
    }

    /* Strip leading space if this is the first real token after BOS */
    if (prev_token == 1 && piece[0] == ' ') {
        return piece + 1;
    }

    return piece;
}
