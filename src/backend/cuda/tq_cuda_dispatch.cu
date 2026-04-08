/**
 * TurboQuant -- CUDA dispatch and initialization
 *
 * Initializes the CUDA backend, detects devices, and registers
 * CUDA kernel wrappers into the type traits function pointer table.
 * Manages CUDA streams and events for async execution.
 */
#ifdef TQ_BUILD_CUDA

#include "tq_cuda_common.cuh"
#include <cstdio>
#include <cstring>

/* ============================================================
 * Forward declarations of CUDA kernel wrappers
 * ============================================================ */

extern "C" void tq_polar_quantize_cuda(
    const float* d_keys, void* d_out,
    int n, int head_dim, cudaStream_t stream);

extern "C" void tq_polar_attention_cuda(
    const float* d_query, const void* d_keys,
    float* d_scores, int seq_len, int head_dim, cudaStream_t stream);

extern "C" void tq_qjl_quantize_cuda(
    const float* d_keys, void* d_out,
    int num_keys, int emb_dim, cudaStream_t stream);

extern "C" void tq_qjl_attention_cuda(
    const float* d_query, const void* d_keys,
    float* d_scores, int seq_len, int head_dim, cudaStream_t stream);

extern "C" void tq_turbo_quantize_cuda(
    const float* d_keys, void* d_out,
    int n, int head_dim, cudaStream_t stream);

extern "C" void tq_turbo_attention_cuda(
    const float* d_query, const void* d_keys,
    float* d_scores, int seq_len, int head_dim, cudaStream_t stream);

extern "C" void tq_value_quantize_4b_cuda(
    const float* d_values, void* d_out, int n, cudaStream_t stream);

extern "C" void tq_value_quantize_2b_cuda(
    const float* d_values, void* d_out, int n, cudaStream_t stream);

extern "C" void tq_fused_polar_cache_write(
    const float* d_keys, void* d_cache, const int* d_slot_mapping,
    int num_tokens, int num_heads, int head_dim, cudaStream_t stream);

/* ============================================================
 * CUDA backend state
 * ============================================================ */

typedef struct {
    int              initialized;
    int              device_id;
    int              compute_major;
    int              compute_minor;
    size_t           total_mem;
    cudaStream_t     default_stream;
    cudaStream_t     quant_stream;    /* stream for quantization ops */
    cudaStream_t     attn_stream;     /* stream for attention ops */
    cudaEvent_t      quant_done;      /* event to sync quant -> attn */
    char             device_name[256];
} tq_cuda_state_t;

static tq_cuda_state_t g_cuda_state = {0};

/* ============================================================
 * Wrapper functions matching tq_quantize_fn / tq_attention_fn
 * signatures from tq_types.h
 *
 * These wrappers handle device memory allocation and data
 * transfer when called with host pointers. For device-to-device
 * operation, use the _cuda functions directly.
 * ============================================================ */

static void tq_polar_quantize_cuda_wrapper(const float* src, void* dst, int n) {
    float* d_src = NULL;
    void*  d_dst = NULL;
    int num_blocks = (n + TQ_BK_CUDA - 1) / TQ_BK_CUDA;
    size_t out_size = num_blocks * sizeof(tq_polar_block_d);

    cudaMalloc(&d_src, n * sizeof(float));
    cudaMalloc(&d_dst, out_size);
    cudaMemcpy(d_src, src, n * sizeof(float), cudaMemcpyHostToDevice);

    tq_polar_quantize_cuda(d_src, d_dst, n, n, g_cuda_state.default_stream);
    cudaStreamSynchronize(g_cuda_state.default_stream);

    cudaMemcpy(dst, d_dst, out_size, cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);
}

static void tq_polar_attention_cuda_wrapper(
    const float* query, const void* kv_cache,
    float* scores, int seq_len, int head_dim)
{
    float* d_query  = NULL;
    void*  d_cache  = NULL;
    float* d_scores = NULL;
    size_t cache_size = seq_len * sizeof(tq_polar_block_d);

    cudaMalloc(&d_query,  head_dim * sizeof(float));
    cudaMalloc(&d_cache,  cache_size);
    cudaMalloc(&d_scores, seq_len * sizeof(float));

    cudaMemcpy(d_query, query, head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cache, kv_cache, cache_size, cudaMemcpyHostToDevice);

    tq_polar_attention_cuda(d_query, d_cache, d_scores,
                            seq_len, head_dim, g_cuda_state.default_stream);
    cudaStreamSynchronize(g_cuda_state.default_stream);

    cudaMemcpy(scores, d_scores, seq_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_query);
    cudaFree(d_cache);
    cudaFree(d_scores);
}

static void tq_qjl_quantize_cuda_wrapper(const float* src, void* dst, int n) {
    float* d_src = NULL;
    void*  d_dst = NULL;
    size_t out_size = sizeof(tq_qjl_block_d);

    cudaMalloc(&d_src, n * sizeof(float));
    cudaMalloc(&d_dst, out_size);
    cudaMemcpy(d_src, src, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemsetAsync(d_dst, 0, out_size, g_cuda_state.default_stream);

    tq_qjl_quantize_cuda(d_src, d_dst, 1, n, g_cuda_state.default_stream);
    cudaStreamSynchronize(g_cuda_state.default_stream);

    cudaMemcpy(dst, d_dst, out_size, cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);
}

static void tq_qjl_attention_cuda_wrapper(
    const float* query, const void* kv_cache,
    float* scores, int seq_len, int head_dim)
{
    float* d_query  = NULL;
    void*  d_cache  = NULL;
    float* d_scores = NULL;
    size_t cache_size = seq_len * sizeof(tq_qjl_block_d);

    cudaMalloc(&d_query,  head_dim * sizeof(float));
    cudaMalloc(&d_cache,  cache_size);
    cudaMalloc(&d_scores, seq_len * sizeof(float));

    cudaMemcpy(d_query, query, head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cache, kv_cache, cache_size, cudaMemcpyHostToDevice);

    tq_qjl_attention_cuda(d_query, d_cache, d_scores,
                           seq_len, head_dim, g_cuda_state.default_stream);
    cudaStreamSynchronize(g_cuda_state.default_stream);

    cudaMemcpy(scores, d_scores, seq_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_query);
    cudaFree(d_cache);
    cudaFree(d_scores);
}

static void tq_turbo_quantize_cuda_wrapper(const float* src, void* dst, int n) {
    float* d_src = NULL;
    void*  d_dst = NULL;
    int num_blocks = (n + TQ_BK_CUDA - 1) / TQ_BK_CUDA;
    size_t out_size = num_blocks * sizeof(tq_turbo_block_d);

    cudaMalloc(&d_src, n * sizeof(float));
    cudaMalloc(&d_dst, out_size);
    cudaMemcpy(d_src, src, n * sizeof(float), cudaMemcpyHostToDevice);

    tq_turbo_quantize_cuda(d_src, d_dst, n, n, g_cuda_state.default_stream);
    cudaStreamSynchronize(g_cuda_state.default_stream);

    cudaMemcpy(dst, d_dst, out_size, cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);
}

static void tq_turbo_attention_cuda_wrapper(
    const float* query, const void* kv_cache,
    float* scores, int seq_len, int head_dim)
{
    float* d_query  = NULL;
    void*  d_cache  = NULL;
    float* d_scores = NULL;
    size_t cache_size = seq_len * sizeof(tq_turbo_block_d);

    cudaMalloc(&d_query,  head_dim * sizeof(float));
    cudaMalloc(&d_cache,  cache_size);
    cudaMalloc(&d_scores, seq_len * sizeof(float));

    cudaMemcpy(d_query, query, head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cache, kv_cache, cache_size, cudaMemcpyHostToDevice);

    tq_turbo_attention_cuda(d_query, d_cache, d_scores,
                            seq_len, head_dim, g_cuda_state.default_stream);
    cudaStreamSynchronize(g_cuda_state.default_stream);

    cudaMemcpy(scores, d_scores, seq_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_query);
    cudaFree(d_cache);
    cudaFree(d_scores);
}

static void tq_uniform_4b_quantize_cuda_wrapper(const float* src, void* dst, int n) {
    float* d_src = NULL;
    void*  d_dst = NULL;
    int num_blocks = (n + TQ_BK_CUDA - 1) / TQ_BK_CUDA;
    size_t out_size = num_blocks * sizeof(tq_uniform_4b_block_d);

    cudaMalloc(&d_src, n * sizeof(float));
    cudaMalloc(&d_dst, out_size);
    cudaMemcpy(d_src, src, n * sizeof(float), cudaMemcpyHostToDevice);

    tq_value_quantize_4b_cuda(d_src, d_dst, n, g_cuda_state.default_stream);
    cudaStreamSynchronize(g_cuda_state.default_stream);

    cudaMemcpy(dst, d_dst, out_size, cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);
}

static void tq_uniform_2b_quantize_cuda_wrapper(const float* src, void* dst, int n) {
    float* d_src = NULL;
    void*  d_dst = NULL;
    int num_blocks = (n + TQ_BK_CUDA - 1) / TQ_BK_CUDA;
    size_t out_size = num_blocks * sizeof(tq_uniform_2b_block_d);

    cudaMalloc(&d_src, n * sizeof(float));
    cudaMalloc(&d_dst, out_size);
    cudaMemcpy(d_src, src, n * sizeof(float), cudaMemcpyHostToDevice);

    tq_value_quantize_2b_cuda(d_src, d_dst, n, g_cuda_state.default_stream);
    cudaStreamSynchronize(g_cuda_state.default_stream);

    cudaMemcpy(dst, d_dst, out_size, cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);
}

/* ============================================================
 * Backend initialization
 * ============================================================ */

extern "C" int tq_init_cuda_backend(void) {
    if (g_cuda_state.initialized) return 0;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "TQ CUDA: No CUDA devices found (%s)\n",
                cudaGetErrorString(err));
        return -1;
    }

    /* Select device 0 by default */
    g_cuda_state.device_id = 0;
    TQ_CUDA_CHECK_STATUS(cudaSetDevice(0));

    /* Query device properties */
    cudaDeviceProp prop;
    TQ_CUDA_CHECK_STATUS(cudaGetDeviceProperties(&prop, 0));

    g_cuda_state.compute_major = prop.major;
    g_cuda_state.compute_minor = prop.minor;
    g_cuda_state.total_mem     = prop.totalGlobalMem;
    strncpy(g_cuda_state.device_name, prop.name, sizeof(g_cuda_state.device_name) - 1);

    printf("TQ CUDA: Initialized on %s (SM %d.%d, %.1f GB)\n",
           g_cuda_state.device_name,
           g_cuda_state.compute_major,
           g_cuda_state.compute_minor,
           (double)g_cuda_state.total_mem / (1024.0 * 1024.0 * 1024.0));

    /* Create streams and events */
    TQ_CUDA_CHECK_STATUS(cudaStreamCreate(&g_cuda_state.default_stream));
    TQ_CUDA_CHECK_STATUS(cudaStreamCreate(&g_cuda_state.quant_stream));
    TQ_CUDA_CHECK_STATUS(cudaStreamCreate(&g_cuda_state.attn_stream));
    TQ_CUDA_CHECK_STATUS(cudaEventCreate(&g_cuda_state.quant_done));

    g_cuda_state.initialized = 1;
    return 0;
}

extern "C" void tq_shutdown_cuda_backend(void) {
    if (!g_cuda_state.initialized) return;

    cudaEventDestroy(g_cuda_state.quant_done);
    cudaStreamDestroy(g_cuda_state.attn_stream);
    cudaStreamDestroy(g_cuda_state.quant_stream);
    cudaStreamDestroy(g_cuda_state.default_stream);

    memset(&g_cuda_state, 0, sizeof(g_cuda_state));
}

extern "C" int tq_cuda_is_available(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0) ? 1 : 0;
}

extern "C" const char* tq_cuda_device_name(void) {
    if (!g_cuda_state.initialized) return "N/A";
    return g_cuda_state.device_name;
}

extern "C" cudaStream_t tq_cuda_get_stream(void) {
    return g_cuda_state.default_stream;
}

/* ============================================================
 * Dispatch table registration
 *
 * Call this after tq_init_cuda_backend() to override the default
 * CPU function pointers in TQ_TRAITS with CUDA-accelerated versions.
 *
 * NOTE: The actual traits table is defined in tq_traits.c as const.
 * In practice, the CUDA backend would use a mutable dispatch table
 * or function pointer overrides. This provides the mechanism.
 * ============================================================ */

typedef struct {
    void (*quantize)(const float*, void*, int);
    void (*attention)(const float*, const void*, float*, int, int);
} tq_cuda_dispatch_entry_t;

static tq_cuda_dispatch_entry_t g_cuda_dispatch[7] = {
    /* TQ_TYPE_POLAR_3B */
    { tq_polar_quantize_cuda_wrapper,  tq_polar_attention_cuda_wrapper },
    /* TQ_TYPE_POLAR_4B */
    { tq_polar_quantize_cuda_wrapper,  tq_polar_attention_cuda_wrapper },
    /* TQ_TYPE_QJL_1B */
    { tq_qjl_quantize_cuda_wrapper,    tq_qjl_attention_cuda_wrapper },
    /* TQ_TYPE_TURBO_3B */
    { tq_turbo_quantize_cuda_wrapper,  tq_turbo_attention_cuda_wrapper },
    /* TQ_TYPE_TURBO_4B */
    { tq_turbo_quantize_cuda_wrapper,  tq_turbo_attention_cuda_wrapper },
    /* TQ_TYPE_UNIFORM_4B */
    { tq_uniform_4b_quantize_cuda_wrapper, NULL },
    /* TQ_TYPE_UNIFORM_2B */
    { tq_uniform_2b_quantize_cuda_wrapper, NULL },
};

extern "C" void* tq_cuda_get_quantize_fn(int type_id) {
    if (type_id < 0 || type_id >= 7) return NULL;
    return (void*)g_cuda_dispatch[type_id].quantize;
}

extern "C" void* tq_cuda_get_attention_fn(int type_id) {
    if (type_id < 0 || type_id >= 7) return NULL;
    return (void*)g_cuda_dispatch[type_id].attention;
}

/* Override TQ_TRAITS with CUDA-accelerated KV quantize/attention functions.
 * Mirror of tq_vulkan_override_traits() — only KV cache ops are dispatched
 * to GPU; weight matmul still runs on CPU. */
#include "turboquant/tq_types.h"
extern "C" void tq_cuda_override_traits(void) {
    for (int i = 0; i < 7 && i < TQ_TYPE_COUNT; i++) {
        void* cu_quant = tq_cuda_get_quantize_fn(i);
        void* cu_attn  = tq_cuda_get_attention_fn(i);
        if (cu_quant) {
            void (*fn)(const float*, void*, int);
            memcpy(&fn, &cu_quant, sizeof(fn));
            TQ_TRAITS[i].quantize = fn;
            fprintf(stderr, "  CUDA: GPU-accelerated quantize for %s\n", TQ_TRAITS[i].name);
        }
        if (cu_attn) {
            void (*fn)(const float*, const void*, float*, int, int);
            memcpy(&fn, &cu_attn, sizeof(fn));
            TQ_TRAITS[i].attention = fn;
            fprintf(stderr, "  CUDA: GPU-accelerated attention for %s\n", TQ_TRAITS[i].name);
        }
    }
    fprintf(stderr, "quant.cpp CUDA: GPU acceleration covers KV cache "
                    "quantize/attention only; weight matmul runs on CPU.\n");
}

#endif /* TQ_BUILD_CUDA */
