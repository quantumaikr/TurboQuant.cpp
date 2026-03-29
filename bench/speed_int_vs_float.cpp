/**
 * Speed benchmark: Integer-domain Q4xQ8 attention vs FP32 dequant+dot
 *
 * Compares three paths:
 * 1. FP32 dot product (baseline, no quantization overhead)
 * 2. Dequantize + FP32 dot (old quantized path)
 * 3. Integer Q4xQ8 dot (new integer-domain path)
 *
 * On ARM NEON (Apple M-series), the integer path should be 3-5x faster
 * than the dequant path due to avoiding FP32 conversion in the inner loop.
 */

#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>

extern "C" {
#include "turboquant/turboquant.h"

void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
void tq_quantize_query_q8(const float* query, int8_t* q8_out,
                           float* scale_out, float* sum_out, int n);
void tq_uniform_4b_attention_int_ref(const float* query, const void* kv,
                                      float* scores, int seq_len, int head_dim);

#ifdef __ARM_NEON
void tq_uniform_4b_attention_neon(const float* query, const void* kv,
                                   float* scores, int seq_len, int head_dim);
void tq_uniform_4b_attention_int_neon(const float* query, const void* kv,
                                       float* scores, int seq_len, int head_dim);
#endif
}

using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/* FP32 baseline: no quantization, just dot product */
static void fp32_attention(const float* query, const float* keys,
                           float* scores, int seq_len, int head_dim) {
    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int d = 0; d < head_dim; d++)
            dot += query[d] * keys[s * head_dim + d];
        scores[s] = dot;
    }
}

/* Dequant + dot: the old path */
static void dequant_attention(const float* query, const block_tq_uniform_4b* blocks,
                               float* scores, int seq_len, int head_dim) {
    for (int s = 0; s < seq_len; s++) {
        float deq[256];
        tq_uniform_4b_dequantize_ref(&blocks[s], deq, head_dim);
        float dot = 0;
        for (int d = 0; d < head_dim; d++)
            dot += query[d] * deq[d];
        scores[s] = dot;
    }
}

static void benchmark_config(int head_dim, int seq_len, int warmup_iters, int bench_iters) {
    printf("\n--- head_dim=%d, seq_len=%d ---\n", head_dim, seq_len);

    /* Setup data */
    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = cosf(i * 0.05f);

    std::vector<float> fp32_keys(seq_len * head_dim);
    std::vector<block_tq_uniform_4b> q4_blocks(seq_len);

    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < head_dim; d++) {
            fp32_keys[s * head_dim + d] = sinf(s * 0.1f + d * 0.05f);
        }
        tq_uniform_4b_quantize_ref(&fp32_keys[s * head_dim], &q4_blocks[s], head_dim);
    }

    std::vector<float> scores(seq_len);

    /* --- Benchmark FP32 baseline --- */
    for (int i = 0; i < warmup_iters; i++)
        fp32_attention(query.data(), fp32_keys.data(), scores.data(), seq_len, head_dim);

    auto t0 = Clock::now();
    for (int i = 0; i < bench_iters; i++)
        fp32_attention(query.data(), fp32_keys.data(), scores.data(), seq_len, head_dim);
    auto t1 = Clock::now();
    double fp32_ms = elapsed_ms(t0, t1) / bench_iters;

    /* --- Benchmark dequant+dot --- */
    for (int i = 0; i < warmup_iters; i++)
        dequant_attention(query.data(), q4_blocks.data(), scores.data(), seq_len, head_dim);

    t0 = Clock::now();
    for (int i = 0; i < bench_iters; i++)
        dequant_attention(query.data(), q4_blocks.data(), scores.data(), seq_len, head_dim);
    t1 = Clock::now();
    double dequant_ms = elapsed_ms(t0, t1) / bench_iters;

    /* --- Benchmark integer Q4xQ8 (ref) --- */
    for (int i = 0; i < warmup_iters; i++)
        tq_uniform_4b_attention_int_ref(query.data(), q4_blocks.data(),
                                         scores.data(), seq_len, head_dim);

    t0 = Clock::now();
    for (int i = 0; i < bench_iters; i++)
        tq_uniform_4b_attention_int_ref(query.data(), q4_blocks.data(),
                                         scores.data(), seq_len, head_dim);
    t1 = Clock::now();
    double int_ref_ms = elapsed_ms(t0, t1) / bench_iters;

    printf("  FP32 baseline:      %8.3f ms\n", fp32_ms);
    printf("  Dequant+dot (old):  %8.3f ms\n", dequant_ms);
    printf("  Int Q4xQ8 (ref):    %8.3f ms  (%.2fx vs dequant)\n",
           int_ref_ms, dequant_ms / int_ref_ms);

#ifdef __ARM_NEON
    /* --- Benchmark NEON dequant+dot --- */
    for (int i = 0; i < warmup_iters; i++)
        tq_uniform_4b_attention_neon(query.data(), q4_blocks.data(),
                                      scores.data(), seq_len, head_dim);

    t0 = Clock::now();
    for (int i = 0; i < bench_iters; i++)
        tq_uniform_4b_attention_neon(query.data(), q4_blocks.data(),
                                      scores.data(), seq_len, head_dim);
    t1 = Clock::now();
    double neon_dequant_ms = elapsed_ms(t0, t1) / bench_iters;

    /* --- Benchmark NEON integer Q4xQ8 --- */
    for (int i = 0; i < warmup_iters; i++)
        tq_uniform_4b_attention_int_neon(query.data(), q4_blocks.data(),
                                          scores.data(), seq_len, head_dim);

    t0 = Clock::now();
    for (int i = 0; i < bench_iters; i++)
        tq_uniform_4b_attention_int_neon(query.data(), q4_blocks.data(),
                                          scores.data(), seq_len, head_dim);
    t1 = Clock::now();
    double neon_int_ms = elapsed_ms(t0, t1) / bench_iters;

    printf("  NEON dequant+dot:   %8.3f ms  (%.2fx vs ref dequant)\n",
           neon_dequant_ms, dequant_ms / neon_dequant_ms);
    printf("  NEON Int Q4xQ8:     %8.3f ms  (%.2fx vs NEON dequant, %.2fx vs FP32)\n",
           neon_int_ms, neon_dequant_ms / neon_int_ms, fp32_ms / neon_int_ms);
#endif

    printf("\n");
}

int main() {
    printf("=== Integer-Domain Attention Speed Benchmark ===\n");
    printf("Comparing: FP32 | Dequant+Dot | Integer Q4xQ8\n");

    /* Typical LLM configurations */
    benchmark_config(128, 64,    10, 1000);   /* Small: GQA head_dim=128, short ctx */
    benchmark_config(128, 512,   10, 200);    /* Medium: head_dim=128, medium ctx */
    benchmark_config(128, 2048,  5,  50);     /* Large: head_dim=128, long ctx */
    benchmark_config(128, 8192,  3,  20);     /* XL: head_dim=128, very long ctx */

    printf("=== Benchmark Complete ===\n");
    return 0;
}
