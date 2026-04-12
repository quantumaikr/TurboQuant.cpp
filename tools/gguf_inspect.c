/* gguf_inspect — dump tensor names, shapes, types, and metadata from a GGUF.
 *
 * Used during architecture-support work to verify what tensor names and
 * shapes a given model file actually ships with, before writing loader
 * code that depends on those assumptions.
 *
 *   cc -O0 -o gguf_inspect tools/gguf_inspect.c -lm -lpthread
 *   ./gguf_inspect ~/.cache/quantcpp/Phi-3.5-mini-instruct-Q4_K_M.gguf
 */
#define QUANT_IMPLEMENTATION
#include "../quant.h"

#include <stdio.h>
#include <string.h>

static const char* type_name(tq_ggml_dtype t) {
    switch (t) {
        case TQ_GGML_TYPE_F32:    return "F32";
        case TQ_GGML_TYPE_F16:    return "F16";
        case TQ_GGML_TYPE_Q4_0:   return "Q4_0";
        case TQ_GGML_TYPE_Q4_1:   return "Q4_1";
        case TQ_GGML_TYPE_Q5_0:   return "Q5_0";
        case TQ_GGML_TYPE_Q5_1:   return "Q5_1";
        case TQ_GGML_TYPE_Q8_0:   return "Q8_0";
        case TQ_GGML_TYPE_Q8_1:   return "Q8_1";
        case TQ_GGML_TYPE_Q2_K:   return "Q2_K";
        case TQ_GGML_TYPE_Q3_K:   return "Q3_K";
        case TQ_GGML_TYPE_Q4_K:   return "Q4_K";
        case TQ_GGML_TYPE_Q5_K:   return "Q5_K";
        case TQ_GGML_TYPE_Q6_K:   return "Q6_K";
        case TQ_GGML_TYPE_Q8_K:   return "Q8_K";
        case TQ_GGML_TYPE_BF16:   return "BF16";
        default: {
            static char buf[16];
            snprintf(buf, sizeof(buf), "TYPE_%d", (int)t);
            return buf;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [--brief|--meta|--tensors|--layer N]\n", argv[0]);
        return 1;
    }
    int brief    = 0;
    int show_meta    = 1;
    int show_tensors = 1;
    int focus_layer  = -1;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--brief") == 0) brief = 1;
        else if (strcmp(argv[i], "--meta") == 0) { show_tensors = 0; }
        else if (strcmp(argv[i], "--tensors") == 0) { show_meta = 0; }
        else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            focus_layer = atoi(argv[++i]);
        }
    }

    tq_gguf_ctx_t* ctx = tq_gguf_open(argv[1]);
    if (!ctx) {
        fprintf(stderr, "failed to open %s\n", argv[1]);
        return 2;
    }

    printf("=== %s ===\n", argv[1]);
    printf("version  : %u\n", ctx->version);
    printf("arch     : %s\n", ctx->arch);
    printf("n_tensors: %llu\n", (unsigned long long)ctx->n_tensors);
    printf("n_kv     : %llu\n", (unsigned long long)ctx->n_kv);
    printf("file_size: %.2f MB\n", (double)ctx->mmap_size / (1024.0 * 1024.0));

    if (show_meta && !brief) {
        printf("\n--- metadata (selected keys) ---\n");
        const char* keys[] = {
            "general.architecture",
            "general.name",
            "general.basename",
            "general.size_label",
            "general.quantization_version",
            "general.file_type",
            "phi3.context_length",
            "phi3.embedding_length",
            "phi3.feed_forward_length",
            "phi3.block_count",
            "phi3.attention.head_count",
            "phi3.attention.head_count_kv",
            "phi3.attention.layer_norm_rms_epsilon",
            "phi3.rope.freq_base",
            "phi3.rope.scaling.factor",
            "phi3.rope.scaling.original_context_length",
            "phi3.rope.scaling.attn_factor",
            "phi3.rope.scaling.short_factor",
            "phi3.rope.scaling.long_factor",
            "phi3.rope.scaling.type",
            "phi3.rope.dimension_count",
            "phi3.attention.sliding_window",
            "tokenizer.ggml.model",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.padding_token_id",
            "tokenizer.ggml.unknown_token_id",
            "tokenizer.chat_template",
            NULL,
        };
        for (int i = 0; keys[i]; i++) {
            int64_t idx = tq_gguf_find_key(ctx, keys[i]);
            if (idx < 0) continue;
            tq_gguf_kv_t* kv = &ctx->kv[idx];
            printf("  %-50s = ", keys[i]);
            switch (kv->type) {
                case TQ_GGUF_TYPE_UINT32:
                    printf("%u (u32)\n", kv->value.u32);
                    break;
                case TQ_GGUF_TYPE_INT32:
                    printf("%d (i32)\n", kv->value.i32);
                    break;
                case TQ_GGUF_TYPE_UINT64:
                    printf("%llu (u64)\n", (unsigned long long)kv->value.u64);
                    break;
                case TQ_GGUF_TYPE_FLOAT32:
                    printf("%.6g (f32)\n", kv->value.f32);
                    break;
                case TQ_GGUF_TYPE_STRING: {
                    const char* s = tq_gguf_get_str(ctx, keys[i]);
                    if (s) {
                        size_t l = strlen(s);
                        if (l > 80) printf("\"%.80s...\" (string, %zu bytes)\n", s, l);
                        else        printf("\"%s\" (string)\n", s);
                    } else printf("(string, value unreadable)\n");
                    break;
                }
                case TQ_GGUF_TYPE_BOOL:
                    printf("%s (bool)\n", kv->value.bool_val ? "true" : "false");
                    break;
                case TQ_GGUF_TYPE_ARRAY:
                    printf("(array, elem_type=%d, count=%llu)\n",
                            (int)kv->value.array.elem_type,
                            (unsigned long long)kv->value.array.count);
                    break;
                default:
                    printf("(type=%d)\n", (int)kv->type);
                    break;
            }
        }
    }

    if (show_tensors) {
        printf("\n--- tensors ---\n");
        printf("%-50s %-8s %s\n", "name", "type", "shape");
        for (uint64_t i = 0; i < ctx->n_tensors; i++) {
            const tq_gguf_tensor_t* t = &ctx->tensors[i];
            if (focus_layer >= 0) {
                /* Only show blk.N. tensors for the requested layer */
                char prefix[32];
                snprintf(prefix, sizeof(prefix), "blk.%d.", focus_layer);
                if (strncmp(t->name, prefix, strlen(prefix)) != 0) continue;
            }
            char shape_buf[64];
            int n = 0;
            n += snprintf(shape_buf + n, sizeof(shape_buf) - n, "[");
            for (uint32_t d = 0; d < t->n_dims && n < (int)sizeof(shape_buf) - 1; d++) {
                n += snprintf(shape_buf + n, sizeof(shape_buf) - n,
                              "%lld%s", (long long)t->shape[d],
                              d + 1 < t->n_dims ? "," : "");
            }
            n += snprintf(shape_buf + n, sizeof(shape_buf) - n, "]");
            printf("%-50s %-8s %s\n", t->name, type_name(t->type), shape_buf);
        }
    }

    tq_gguf_close(ctx);
    return 0;
}
