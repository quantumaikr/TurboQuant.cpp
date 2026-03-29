/**
 * TurboQuant — Qwen3.5-0.8B KV Cache Validation
 *
 * Validates all quantization types (including v0.6: RHT, mixed precision)
 * on Qwen3.5-0.8B architecture: 2 KV heads, 256 head_dim, hybrid attention.
 */

extern "C" {
#include "turboquant/turboquant.h"

void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
void tq_uniform_2b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_2b_dequantize_ref(const void* src, float* dst, int n);
void tq_polar_quantize_ref(const float* src, void* dst, int n);
void tq_polar_dequantize_ref(const void* src, float* dst, int n);
void tq_qjl_quantize_ref(const float* src, void* dst, int n);
void tq_qjl_dequantize_ref(const void* src, float* dst, int n);
void tq_turbo_quantize_ref(const float* src, void* dst, int n);
void tq_turbo_dequantize_ref(const void* src, float* dst, int n);
void tq_mixed_4b8_quantize_ref(const float* src, void* dst, int n);
void tq_mixed_4b8_dequantize_ref(const void* src, float* dst, int n);
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#define MAGIC 0x544B5651

struct LayerData {
    int layer_idx;
    int num_heads;
    int seq_len;
    int head_dim;
    std::vector<float> keys;   /* [num_heads * seq_len * head_dim] */
};

static bool load_layer(const char* path, LayerData& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    uint32_t hdr[5];
    if (fread(hdr, sizeof(uint32_t), 5, f) != 5) { fclose(f); return false; }
    if (hdr[0] != MAGIC) { fclose(f); return false; }

    out.layer_idx = (int)hdr[1];
    out.num_heads = (int)hdr[2];
    out.seq_len   = (int)hdr[3];
    out.head_dim  = (int)hdr[4];

    size_t total = (size_t)out.num_heads * out.seq_len * out.head_dim;
    out.keys.resize(total);
    size_t read = fread(out.keys.data(), sizeof(float), total, f);
    fclose(f);
    return read == total;
}

struct TypeInfo {
    const char* name;
    tq_type type;
    tq_quantize_fn quantize;
    tq_dequantize_fn dequantize;
    size_t block_size_bytes;
    int block_elems;
};

int main() {
    printf("\n");
    printf("================================================================\n");
    printf("  TurboQuant — Qwen3.5-0.8B KV Cache Validation\n");
    printf("  Architecture: Hybrid (DeltaNet + Gated Attention)\n");
    printf("  KV Heads: 2, Head Dim: 256, Attention Layers: 6/24\n");
    printf("================================================================\n\n");

    /* Load layers */
    const char* base = "spec/test_vectors/qwen35_kv";
    int layer_indices[] = {3, 7, 11, 15};
    std::vector<LayerData> layers;

    for (int li : layer_indices) {
        char path[256];
        snprintf(path, sizeof(path), "%s/layer%d_keys.bin", base, li);
        LayerData ld;
        if (load_layer(path, ld)) {
            printf("Loaded layer %d: %d heads x %d seq x %d dim\n",
                   ld.layer_idx, ld.num_heads, ld.seq_len, ld.head_dim);
            layers.push_back(ld);
        }
    }

    if (layers.empty()) {
        printf("ERROR: No data found. Run: python3 tests/reference/dump_qwen35_kv.py\n");
        return 1;
    }

    int head_dim = layers[0].head_dim;

    /* Type table */
    TypeInfo types[] = {
        {"uniform_4b",  TQ_TYPE_UNIFORM_4B, tq_uniform_4b_quantize_ref, tq_uniform_4b_dequantize_ref,
         sizeof(block_tq_uniform_4b), TQ_BK},
        {"uniform_2b",  TQ_TYPE_UNIFORM_2B, tq_uniform_2b_quantize_ref, tq_uniform_2b_dequantize_ref,
         sizeof(block_tq_uniform_2b), TQ_BK},
        {"polar_4b",    TQ_TYPE_POLAR_4B,   tq_polar_quantize_ref, tq_polar_dequantize_ref,
         sizeof(block_tq_polar), TQ_BK},
        {"qjl_1b",      TQ_TYPE_QJL_1B,     tq_qjl_quantize_ref, tq_qjl_dequantize_ref,
         sizeof(block_tq_qjl), TQ_BK_QJL},
        {"turbo_3b",    TQ_TYPE_TURBO_3B,   tq_turbo_quantize_ref, tq_turbo_dequantize_ref,
         sizeof(block_tq_turbo), TQ_BK},
        {"mixed_4b8",   TQ_TYPE_MIXED_4B8,  tq_mixed_4b8_quantize_ref, tq_mixed_4b8_dequantize_ref,
         sizeof(block_tq_mixed_4b8), TQ_BK},
    };
    int n_types = sizeof(types) / sizeof(types[0]);

    /* RHT seed */
    uint32_t rht_seed = 42;

    /* ============================================================
     * Per-type, per-layer quality measurement
     * ============================================================ */
    printf("\n--- Per-Type Quality (averaged over all layers/heads) ---\n\n");
    printf("%-14s  %8s  %12s  %12s  %s\n",
           "Type", "BPE", "MSE", "Attn Cosine", "Grade");
    printf("%-14s  %8s  %12s  %12s  %s\n",
           "--------------", "--------", "------------", "------------", "-----");

    for (int ti = 0; ti < n_types; ti++) {
        const TypeInfo& t = types[ti];
        double total_mse = 0;
        double sum_dot = 0, sum_a2 = 0, sum_b2 = 0;
        int count = 0;

        for (const auto& layer : layers) {
            int hd = layer.head_dim;
            int blocks_per_key = (hd + t.block_elems - 1) / t.block_elems;
            size_t quant_size = blocks_per_key * t.block_size_bytes;

            /* Random query */
            std::vector<float> query(hd);
            uint32_t qseed = (uint32_t)(layer.layer_idx * 1000 + ti);
            for (int d = 0; d < hd; d++) {
                qseed = qseed * 1664525u + 1013904223u;
                query[d] = ((float)(qseed >> 8) / (float)(1 << 24)) * 2.0f - 1.0f;
            }

            for (int h = 0; h < layer.num_heads; h++) {
                for (int s = 0; s < layer.seq_len; s++) {
                    const float* key = layer.keys.data() +
                        (size_t)h * layer.seq_len * hd + (size_t)s * hd;

                    /* Quantize + dequantize */
                    std::vector<uint8_t> qbuf(quant_size, 0);
                    std::vector<float> deq(hd, 0.0f);

                    /* Process in blocks */
                    int offset = 0;
                    int qoffset = 0;
                    while (offset < hd) {
                        int chunk = (hd - offset > t.block_elems) ? t.block_elems : (hd - offset);
                        t.quantize(key + offset, qbuf.data() + qoffset, chunk);
                        t.dequantize(qbuf.data() + qoffset, deq.data() + offset, chunk);
                        offset += chunk;
                        qoffset += (int)t.block_size_bytes;
                    }

                    /* MSE */
                    double mse = 0;
                    for (int d = 0; d < hd; d++) {
                        double diff = (double)key[d] - (double)deq[d];
                        mse += diff * diff;
                    }
                    mse /= hd;
                    total_mse += mse;

                    /* Attention scores */
                    float fp32_dot = 0, quant_dot = 0;
                    for (int d = 0; d < hd; d++) {
                        fp32_dot += query[d] * key[d];
                        quant_dot += query[d] * deq[d];
                    }

                    sum_dot += (double)fp32_dot * (double)quant_dot;
                    sum_a2 += (double)fp32_dot * (double)fp32_dot;
                    sum_b2 += (double)quant_dot * (double)quant_dot;
                    count++;
                }
            }
        }

        double avg_mse = total_mse / count;
        double cosine = (sum_a2 > 0 && sum_b2 > 0) ?
            sum_dot / (sqrt(sum_a2) * sqrt(sum_b2)) : 0;

        float bpe = tq_type_bpe(t.type);
        const char* grade;
        if (cosine > 0.99) grade = "A+";
        else if (cosine > 0.95) grade = "A";
        else if (cosine > 0.90) grade = "B+";
        else if (cosine > 0.80) grade = "B";
        else grade = "C";

        printf("%-14s  %6.1f    %10.6f    %10.6f    %s\n",
               t.name, bpe, avg_mse, cosine, grade);

        /* Machine-readable */
        printf("qwen35_mse_%s=%.6f\n", t.name, avg_mse);
        printf("qwen35_cosine_%s=%.6f\n", t.name, cosine);
    }

    /* ============================================================
     * RHT comparison (uniform_4b with and without RHT)
     * ============================================================ */
    printf("\n--- RHT A/B Comparison (uniform_4b, head_dim=%d) ---\n\n", head_dim);

    double rht_mse_sum = 0, raw_mse_sum = 0;
    int rht_count = 0;

    for (const auto& layer : layers) {
        int hd = layer.head_dim;
        int blocks = (hd + TQ_BK - 1) / TQ_BK;
        size_t qsize = blocks * sizeof(block_tq_uniform_4b);

        for (int h = 0; h < layer.num_heads; h++) {
            for (int s = 0; s < layer.seq_len; s++) {
                const float* key = layer.keys.data() +
                    (size_t)h * layer.seq_len * hd + (size_t)s * hd;

                /* Raw quantize */
                std::vector<uint8_t> raw_buf(qsize, 0);
                std::vector<float> raw_deq(hd, 0.0f);
                int off = 0, qoff = 0;
                while (off < hd) {
                    int chunk = (hd - off > TQ_BK) ? TQ_BK : (hd - off);
                    tq_uniform_4b_quantize_ref(key + off, raw_buf.data() + qoff, chunk);
                    tq_uniform_4b_dequantize_ref(raw_buf.data() + qoff, raw_deq.data() + off, chunk);
                    off += chunk; qoff += sizeof(block_tq_uniform_4b);
                }

                /* RHT quantize */
                std::vector<float> rotated(hd);
                memcpy(rotated.data(), key, hd * sizeof(float));
                tq_rht_transform(rotated.data(), hd, rht_seed);

                std::vector<uint8_t> rht_buf(qsize, 0);
                std::vector<float> rht_deq(hd, 0.0f);
                off = 0; qoff = 0;
                while (off < hd) {
                    int chunk = (hd - off > TQ_BK) ? TQ_BK : (hd - off);
                    tq_uniform_4b_quantize_ref(rotated.data() + off, rht_buf.data() + qoff, chunk);
                    tq_uniform_4b_dequantize_ref(rht_buf.data() + qoff, rht_deq.data() + off, chunk);
                    off += chunk; qoff += sizeof(block_tq_uniform_4b);
                }
                tq_rht_inverse(rht_deq.data(), hd, rht_seed);

                /* MSE */
                double raw_mse = 0, rht_mse = 0;
                for (int d = 0; d < hd; d++) {
                    double d1 = (double)key[d] - (double)raw_deq[d];
                    double d2 = (double)key[d] - (double)rht_deq[d];
                    raw_mse += d1 * d1;
                    rht_mse += d2 * d2;
                }
                raw_mse_sum += raw_mse / hd;
                rht_mse_sum += rht_mse / hd;
                rht_count++;
            }
        }
    }

    double raw_avg = raw_mse_sum / rht_count;
    double rht_avg = rht_mse_sum / rht_count;
    printf("  Raw uniform_4b MSE:  %.6f\n", raw_avg);
    printf("  RHT+uniform_4b MSE:  %.6f\n", rht_avg);
    printf("  Improvement:         %.1fx\n", raw_avg / (rht_avg > 0 ? rht_avg : 1e-10));
    printf("qwen35_rht_raw_mse=%.6f\n", raw_avg);
    printf("qwen35_rht_improved_mse=%.6f\n", rht_avg);
    printf("qwen35_rht_improvement=%.1fx\n", raw_avg / (rht_avg > 0 ? rht_avg : 1e-10));

    /* ============================================================
     * K/V Asymmetric comparison
     * ============================================================ */
    printf("\n--- K/V Asymmetric: Key 4-bit + Value 2-bit ---\n\n");
    float k_bpe = tq_type_bpe(TQ_TYPE_UNIFORM_4B);
    float v_bpe = tq_type_bpe(TQ_TYPE_UNIFORM_2B);
    printf("  Key bits:    %.1f (uniform_4b)\n", k_bpe);
    printf("  Value bits:  %.1f (uniform_2b)\n", v_bpe);
    printf("  Average:     %.2f bits/element\n", (k_bpe + v_bpe) / 2.0f);
    printf("  FP16 equiv:  %.1fx compression\n", 32.0f / ((k_bpe + v_bpe) / 2.0f));

    /* ============================================================
     * Memory impact for Qwen3.5-0.8B
     * ============================================================ */
    printf("\n--- Memory Impact: Qwen3.5-0.8B ---\n\n");
    int kv_heads = 2, att_layers = 6;
    int ctx_lengths[] = {4096, 16384, 65536, 131072};
    printf("  %-8s  %-10s  %-10s  %-10s  %-6s\n",
           "Context", "FP16", "Uniform4b", "K4V2", "Saved");

    for (int ctx : ctx_lengths) {
        double fp16 = (double)att_layers * kv_heads * head_dim * ctx * 2 * 2 / (1024.0*1024.0*1024.0);
        double u4b = (double)att_layers * kv_heads * head_dim * ctx * 2 *
                     (k_bpe / 8.0) / (1024.0*1024.0*1024.0);
        double k4v2 = (double)att_layers * kv_heads * head_dim * ctx *
                      ((k_bpe + v_bpe) / 2.0 / 8.0) / (1024.0*1024.0*1024.0);

        char ctx_str[16];
        if (ctx >= 1024) snprintf(ctx_str, sizeof(ctx_str), "%dK", ctx/1024);
        else snprintf(ctx_str, sizeof(ctx_str), "%d", ctx);

        printf("  %-8s  %7.2f GB  %7.2f GB  %7.2f GB  %4.0f%%\n",
               ctx_str, fp16, u4b, k4v2, (1.0 - k4v2/fp16)*100);
    }

    printf("\n================================================================\n");
    printf("  Validation Complete\n");
    printf("================================================================\n\n");

    return 0;
}
