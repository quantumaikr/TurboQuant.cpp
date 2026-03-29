/**
 * TurboQuant -- Value quantization CUDA kernels
 *
 * Group-wise min-max quantization for value vectors (2-bit and 4-bit).
 * Includes a fused dequantize + matrix multiply kernel for efficient
 * attention-weighted value aggregation.
 */
#ifdef TQ_BUILD_CUDA

#include "tq_cuda_common.cuh"
#include <math.h>

/* ============================================================
 * Value quantize kernel (4-bit uniform)
 *
 * Grid:  (num_blocks, 1, 1)
 * Block: (TQ_BK_CUDA, 1, 1)  -- one thread per element
 *
 * Each CUDA block processes one quantization group (TQ_BK elements).
 * Uses warp reduction for min/max, then each thread quantizes its value.
 * ============================================================ */
__global__ void tq_value_quantize_4b_kernel(
    const float* __restrict__ values,
    tq_uniform_4b_block_d* __restrict__ out,
    int n)
{
    const int block_idx = blockIdx.x;
    const int tid       = threadIdx.x;
    const int base      = block_idx * TQ_BK_CUDA;

    /* Load value into register */
    float val = 0.0f;
    int global_idx = base + tid;
    if (tid < TQ_BK_CUDA && global_idx < n) {
        val = values[global_idx];
    }

    /* Find block min/max using reduction */
    float local_val = (tid < TQ_BK_CUDA && global_idx < n) ? val :  1e30f;
    float gmin = blockReduceMin(local_val);
    local_val = (tid < TQ_BK_CUDA && global_idx < n) ? val : -1e30f;
    float gmax = blockReduceMax(local_val);

    __shared__ float s_minmax[2];
    if (tid == 0) {
        s_minmax[0] = gmin;
        s_minmax[1] = gmax;
    }
    __syncthreads();
    gmin = s_minmax[0];
    gmax = s_minmax[1];

    float range = fmaxf(gmax - gmin, 1e-8f);
    float scale = range / 16.0f; /* 4-bit: 16 bins of width range/16 */

    /* Thread 0 writes header */
    if (tid == 0) {
        out[block_idx].scale      = tq_float_to_half(scale);
        out[block_idx].zero_point = tq_float_to_half(gmin);
    }

    /* Each pair of threads packs two 4-bit values into one byte */
    if (tid < TQ_BK_CUDA && global_idx < n) {
        int q = __float2int_rd((val - gmin) / scale);
        q = max(0, min(15, q));

        /* Shared staging for byte packing */
        __shared__ uint8_t s_quant[TQ_BK_CUDA];
        s_quant[tid] = (uint8_t)q;
        __syncthreads();

        /* Even threads pack two values into one byte */
        if (tid % 2 == 0) {
            int idx = tid / 2;
            uint8_t lo = s_quant[tid];
            uint8_t hi = (tid + 1 < TQ_BK_CUDA) ? s_quant[tid + 1] : 0;
            out[block_idx].qs[idx] = (hi << 4) | lo; /* LSB-first */
        }
    }
}

/* ============================================================
 * Value quantize kernel (2-bit uniform)
 * ============================================================ */
__global__ void tq_value_quantize_2b_kernel(
    const float* __restrict__ values,
    tq_uniform_2b_block_d* __restrict__ out,
    int n)
{
    const int block_idx = blockIdx.x;
    const int tid       = threadIdx.x;
    const int base      = block_idx * TQ_BK_CUDA;

    float val = 0.0f;
    int global_idx = base + tid;
    if (tid < TQ_BK_CUDA && global_idx < n) {
        val = values[global_idx];
    }

    float local_val = (tid < TQ_BK_CUDA && global_idx < n) ? val :  1e30f;
    float gmin = blockReduceMin(local_val);
    local_val = (tid < TQ_BK_CUDA && global_idx < n) ? val : -1e30f;
    float gmax = blockReduceMax(local_val);

    __shared__ float s_minmax[2];
    if (tid == 0) {
        s_minmax[0] = gmin;
        s_minmax[1] = gmax;
    }
    __syncthreads();
    gmin = s_minmax[0];
    gmax = s_minmax[1];

    float range = fmaxf(gmax - gmin, 1e-8f);
    float scale = range / 4.0f; /* 2-bit: 4 bins of width range/4 */

    if (tid == 0) {
        out[block_idx].scale      = tq_float_to_half(scale);
        out[block_idx].zero_point = tq_float_to_half(gmin);
    }

    if (tid < TQ_BK_CUDA && global_idx < n) {
        int q = __float2int_rd((val - gmin) / scale);
        q = max(0, min(3, q));

        __shared__ uint8_t s_quant[TQ_BK_CUDA];
        s_quant[tid] = (uint8_t)q;
        __syncthreads();

        /* Four values per byte, LSB-first */
        if (tid % 4 == 0) {
            int idx = tid / 4;
            uint8_t byte = s_quant[tid];
            if (tid + 1 < TQ_BK_CUDA) byte |= (s_quant[tid + 1] << 2);
            if (tid + 2 < TQ_BK_CUDA) byte |= (s_quant[tid + 2] << 4);
            if (tid + 3 < TQ_BK_CUDA) byte |= (s_quant[tid + 3] << 6);
            out[block_idx].qs[idx] = byte;
        }
    }
}

/* ============================================================
 * Fused value dequantize + matmul kernel
 *
 * Computes: output[d] = sum_s (attention_weights[s] * dequant(value[s, d]))
 *
 * Grid:  (head_dim_blocks, 1, 1)
 * Block: (THREADS, 1, 1)
 *
 * Each block computes a chunk of the output dimension.
 * Values are dequantized on-the-fly to avoid materializing full FP32.
 * ============================================================ */
__global__ void tq_value_dequant_matmul_4b_kernel(
    const float* __restrict__ attn_weights,          /* [seq_len] */
    const tq_uniform_4b_block_d* __restrict__ values, /* [seq_len, num_dim_blocks] */
    float* __restrict__ output,                       /* [head_dim] */
    int seq_len,
    int head_dim)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;

    /* Determine which block and position within block this dimension falls in */
    int dim_block = d / TQ_BK_CUDA;
    int dim_offset = d % TQ_BK_CUDA;

    float accum = 0.0f;

    int num_dim_blocks = (head_dim + TQ_BK_CUDA - 1) / TQ_BK_CUDA;

    for (int s = 0; s < seq_len; s++) {
        float w = attn_weights[s];
        if (fabsf(w) < 1e-10f) continue; /* skip near-zero weights */

        /* Index into value cache: [s, dim_block] */
        const tq_uniform_4b_block_d* blk = &values[s * num_dim_blocks + dim_block];

        float scale = tq_half_to_float(blk->scale);
        float zero  = tq_half_to_float(blk->zero_point);

        /* Unpack 4-bit value (2 per byte, LSB-first) */
        int byte_idx = dim_offset / 2;
        uint8_t byte = blk->qs[byte_idx];
        int q;
        if (dim_offset % 2 == 0) {
            q = byte & 0x0F;
        } else {
            q = (byte >> 4) & 0x0F;
        }

        float dequant_val = zero + ((float)q + 0.5f) * scale;
        accum += w * dequant_val;
    }

    output[d] = accum;
}

/* ============================================================
 * Host-callable wrappers
 * ============================================================ */

extern "C" void tq_value_quantize_4b_cuda(
    const float* d_values,
    void* d_out,
    int n,
    cudaStream_t stream)
{
    int num_blocks = (n + TQ_BK_CUDA - 1) / TQ_BK_CUDA;
    tq_value_quantize_4b_kernel<<<num_blocks, TQ_BK_CUDA, 0, stream>>>(
        d_values,
        reinterpret_cast<tq_uniform_4b_block_d*>(d_out),
        n);
}

extern "C" void tq_value_quantize_2b_cuda(
    const float* d_values,
    void* d_out,
    int n,
    cudaStream_t stream)
{
    int num_blocks = (n + TQ_BK_CUDA - 1) / TQ_BK_CUDA;
    tq_value_quantize_2b_kernel<<<num_blocks, TQ_BK_CUDA, 0, stream>>>(
        d_values,
        reinterpret_cast<tq_uniform_2b_block_d*>(d_out),
        n);
}

extern "C" void tq_value_dequant_matmul_4b_cuda(
    const float* d_attn_weights,
    const void* d_values,
    float* d_output,
    int seq_len,
    int head_dim,
    cudaStream_t stream)
{
    int threads = 256;
    int grid = tq_cuda_num_blocks(head_dim, threads);

    tq_value_dequant_matmul_4b_kernel<<<grid, threads, 0, stream>>>(
        d_attn_weights,
        reinterpret_cast<const tq_uniform_4b_block_d*>(d_values),
        d_output,
        seq_len,
        head_dim);
}

#endif /* TQ_BUILD_CUDA */
