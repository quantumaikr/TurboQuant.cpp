/**
 * TurboQuant -- Fused quantize + cache write CUDA kernels
 *
 * Inspired by vLLM's reshape_and_cache pattern. These kernels combine
 * quantization and paged cache insertion into a single kernel launch
 * to eliminate redundant memory traffic.
 *
 * The slot_mapping array maps logical token positions to physical
 * cache block addresses (block_idx * block_size + offset).
 */
#ifdef TQ_BUILD_CUDA

#include "tq_cuda_common.cuh"
#include <math.h>

/* ============================================================
 * Fused PolarQuant quantize + cache write kernel
 *
 * Template parameters:
 *   scalar_t   -- input data type (float or __half)
 *   BLOCK_SIZE -- tokens per cache block
 *
 * Each CUDA block processes one token's key vector.
 * The slot_mapping translates the token index to a physical
 * location: slot = block_idx * BLOCK_SIZE + offset_in_block.
 *
 * Grid:  (num_tokens, 1, 1)
 * Block: (head_dim/2, 1, 1)
 * ============================================================ */
template<typename scalar_t, int BLOCK_SIZE>
__global__ void tq_fused_polar_cache_kernel(
    const scalar_t* __restrict__ keys,       /* [num_tokens, num_heads, head_dim] */
    tq_polar_block_d* __restrict__ cache,    /* paged cache: [num_blocks, num_heads, block_tokens, ...] */
    const int* __restrict__ slot_mapping,    /* [num_tokens] -> physical slot index */
    int num_tokens,
    int num_heads,
    int head_dim)
{
    const int token_idx = blockIdx.x;
    const int tid       = threadIdx.x;
    const int pairs     = head_dim / 2;

    if (token_idx >= num_tokens) return;

    /* Compute physical cache location from slot mapping */
    int slot = slot_mapping[token_idx];
    if (slot < 0) return; /* padding token, skip */
    int cache_block_idx = slot / BLOCK_SIZE;
    int offset_in_block = slot % BLOCK_SIZE;

    /* Process each head */
    for (int h = 0; h < num_heads; h++) {
        /* Input key offset: [token_idx, h, :] */
        const int key_offset = (token_idx * num_heads + h) * head_dim;

        /* Load key into shared memory */
        __shared__ float s_keys[TQ_BK_CUDA];
        __shared__ float s_theta[TQ_BK_CUDA / 2];
        __shared__ float s_radius[TQ_BK_CUDA / 2];
        __shared__ float s_params[4];

        if (tid < pairs) {
            float x, y;
            /* Convert to float if input is half */
            if constexpr (sizeof(scalar_t) == 2) {
                x = __half2float(reinterpret_cast<const __half*>(keys)[key_offset + 2 * tid]);
                y = __half2float(reinterpret_cast<const __half*>(keys)[key_offset + 2 * tid + 1]);
            } else {
                x = static_cast<float>(keys[key_offset + 2 * tid]);
                y = static_cast<float>(keys[key_offset + 2 * tid + 1]);
            }
            s_keys[2 * tid]     = x;
            s_keys[2 * tid + 1] = y;

            float r = sqrtf(x * x + y * y);
            float t = atan2f(y, x);
            if (t < 0.0f) t += 2.0f * 3.14159265358979323846f;
            s_theta[tid]  = t;
            s_radius[tid] = r;
        }
        __syncthreads();

        /* Min/max reduction */
        float t_val = (tid < pairs) ? s_theta[tid]  :  1e30f;
        float r_val = (tid < pairs) ? s_radius[tid] :  1e30f;
        float tmin = blockReduceMin(t_val);
        float rmin = blockReduceMin(r_val);

        t_val = (tid < pairs) ? s_theta[tid]  : -1e30f;
        r_val = (tid < pairs) ? s_radius[tid] : -1e30f;
        float tmax = blockReduceMax(t_val);
        float rmax = blockReduceMax(r_val);

        if (tid == 0) {
            s_params[0] = tmin; s_params[1] = tmax;
            s_params[2] = rmin; s_params[3] = rmax;
        }
        __syncthreads();

        tmin = s_params[0]; tmax = s_params[1];
        rmin = s_params[2]; rmax = s_params[3];

        float trange = fmaxf(tmax - tmin, 1e-8f);
        float rrange = fmaxf(rmax - rmin, 1e-8f);
        float tscale = trange / 4.0f;
        float rscale = rrange / 4.0f;

        /* Cache output location:
           Linear index = cache_block_idx * (num_heads * BLOCK_SIZE) + h * BLOCK_SIZE + offset */
        int cache_out_idx = cache_block_idx * (num_heads * BLOCK_SIZE)
                          + h * BLOCK_SIZE + offset_in_block;

        /* Write block header (thread 0 only) */
        if (tid == 0) {
            cache[cache_out_idx].tscale = tq_float_to_half(tscale);
            cache[cache_out_idx].tmn    = tq_float_to_half(tmin);
            cache[cache_out_idx].rscale = tq_float_to_half(rscale);
            cache[cache_out_idx].rmn    = tq_float_to_half(rmin);
        }

        /* Pack indices through shared staging */
        __shared__ uint8_t s_packed[TQ_BK_CUDA / 2];
        if (tid < pairs) {
            int tq = __float2int_rd((s_theta[tid] - tmin) / tscale);
            int rq = __float2int_rd((s_radius[tid] - rmin) / rscale);
            tq = max(0, min(3, tq));
            rq = max(0, min(3, rq));
            uint8_t packed = (uint8_t)((rq << 2) | tq);
            if (tid % 2 == 0) s_packed[tid / 2] = packed;
        }
        __syncthreads();

        if (tid < pairs && (tid % 2 == 1)) {
            int tq = __float2int_rd((s_theta[tid] - tmin) / tscale);
            int rq = __float2int_rd((s_radius[tid] - rmin) / rscale);
            tq = max(0, min(3, tq));
            rq = max(0, min(3, rq));
            uint8_t packed = (uint8_t)((rq << 2) | tq);
            s_packed[tid / 2] |= (packed << 4);
        }
        __syncthreads();

        /* Write packed indices to global cache */
        int idx_bytes = pairs / 2;
        if (tid < idx_bytes) {
            cache[cache_out_idx].indices[tid] = s_packed[tid];
        }
        __syncthreads();
    }
}

/* ============================================================
 * Fused uniform quantize + cache write kernel
 *
 * Handles both 4-bit and 2-bit uniform quantization with
 * direct cache insertion.
 * ============================================================ */
template<typename scalar_t, int BLOCK_SIZE, int QUANT_BITS>
__global__ void tq_fused_uniform_cache_kernel(
    const scalar_t* __restrict__ values,     /* [num_tokens, num_heads, head_dim] */
    void* __restrict__ cache,                /* paged value cache */
    const int* __restrict__ slot_mapping,
    int num_tokens,
    int num_heads,
    int head_dim)
{
    const int token_idx = blockIdx.x;
    const int tid       = threadIdx.x;

    if (token_idx >= num_tokens) return;

    int slot = slot_mapping[token_idx];
    if (slot < 0) return;
    int cache_block_idx = slot / BLOCK_SIZE;
    int offset_in_block = slot % BLOCK_SIZE;

    for (int h = 0; h < num_heads; h++) {
        const int val_offset = (token_idx * num_heads + h) * head_dim;

        /* Load values and find min/max */
        __shared__ float s_vals[TQ_BK_CUDA];
        __shared__ float s_minmax[2];

        float local_min =  1e30f;
        float local_max = -1e30f;

        for (int d = tid; d < head_dim; d += blockDim.x) {
            float v;
            if constexpr (sizeof(scalar_t) == 2) {
                v = __half2float(reinterpret_cast<const __half*>(values)[val_offset + d]);
            } else {
                v = static_cast<float>(values[val_offset + d]);
            }
            s_vals[d] = v;
            local_min = fminf(local_min, v);
            local_max = fmaxf(local_max, v);
        }

        float gmin = blockReduceMin(local_min);
        float gmax = blockReduceMax(local_max);

        if (tid == 0) {
            s_minmax[0] = gmin;
            s_minmax[1] = gmax;
        }
        __syncthreads();
        gmin = s_minmax[0];
        gmax = s_minmax[1];

        constexpr int max_val = (1 << QUANT_BITS) - 1;
        float range = fmaxf(gmax - gmin, 1e-8f);
        float scale = range / (float)max_val;

        /* Cache output index */
        int cache_out_idx = cache_block_idx * (num_heads * BLOCK_SIZE)
                          + h * BLOCK_SIZE + offset_in_block;

        if constexpr (QUANT_BITS == 4) {
            tq_uniform_4b_block_d* out = reinterpret_cast<tq_uniform_4b_block_d*>(cache);
            if (tid == 0) {
                out[cache_out_idx].scale      = tq_float_to_half(scale);
                out[cache_out_idx].zero_point = tq_float_to_half(gmin);
            }

            /* Each thread packs two values into one byte */
            int pack_elems = head_dim / 2;
            for (int i = tid; i < pack_elems; i += blockDim.x) {
                int d0 = 2 * i;
                int d1 = 2 * i + 1;
                int q0 = __float2int_rd((s_vals[d0] - gmin) / scale);
                int q1 = (d1 < head_dim) ? __float2int_rd((s_vals[d1] - gmin) / scale) : 0;
                q0 = max(0, min(max_val, q0));
                q1 = max(0, min(max_val, q1));
                out[cache_out_idx].qs[i] = (uint8_t)((q1 << 4) | q0);
            }
        } else if constexpr (QUANT_BITS == 2) {
            tq_uniform_2b_block_d* out = reinterpret_cast<tq_uniform_2b_block_d*>(cache);
            if (tid == 0) {
                out[cache_out_idx].scale      = tq_float_to_half(scale);
                out[cache_out_idx].zero_point = tq_float_to_half(gmin);
            }

            /* Each thread packs four values into one byte */
            int pack_elems = head_dim / 4;
            for (int i = tid; i < pack_elems; i += blockDim.x) {
                uint8_t byte = 0;
                for (int j = 0; j < 4; j++) {
                    int d = 4 * i + j;
                    if (d < head_dim) {
                        int q = __float2int_rd((s_vals[d] - gmin) / scale);
                        q = max(0, min(max_val, q));
                        byte |= (uint8_t)(q << (2 * j));
                    }
                }
                out[cache_out_idx].qs[i] = byte;
            }
        }
        __syncthreads();
    }
}

/* ============================================================
 * Fused paged attention kernel
 *
 * Reads quantized keys from paged cache using block_table,
 * dequantizes on-the-fly, and computes attention scores.
 * Supports mixed-precision blocks (for progressive compression).
 * ============================================================ */
template<int BLOCK_SIZE>
__global__ void tq_fused_paged_attention_kernel(
    const float* __restrict__ query,         /* [num_heads, head_dim] */
    const tq_polar_block_d* __restrict__ key_cache, /* paged */
    float* __restrict__ scores,               /* [num_heads, max_seq_len] */
    const int* __restrict__ block_table,     /* [max_blocks] */
    const uint8_t* __restrict__ quant_types, /* per-block quant type */
    int num_heads,
    int head_dim,
    int seq_len,
    int max_blocks)
{
    const int head_idx  = blockIdx.y;
    const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx >= seq_len) return;

    /* Determine physical block and offset */
    int logical_block = token_idx / BLOCK_SIZE;
    int offset        = token_idx % BLOCK_SIZE;

    if (logical_block >= max_blocks) return;
    int physical_block = block_table[logical_block];

    /* Cache index */
    int cache_idx = physical_block * (num_heads * BLOCK_SIZE)
                  + head_idx * BLOCK_SIZE + offset;

    /* Read quantized key block and dequantize */
    const tq_polar_block_d* block = &key_cache[cache_idx];
    float tscale = tq_half_to_float(block->tscale);
    float tmin   = tq_half_to_float(block->tmn);
    float rscale = tq_half_to_float(block->rscale);
    float rmin   = tq_half_to_float(block->rmn);

    int pairs = head_dim / 2;
    float dot = 0.0f;

    for (int p = 0; p < pairs; p++) {
        uint8_t byte = block->indices[p / 2];
        uint8_t packed = (p % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        int tq = packed & 0x03;
        int rq = (packed >> 2) & 0x03;

        float theta  = tmin + ((float)tq + 0.5f) * tscale;
        float radius = rmin + ((float)rq + 0.5f) * rscale;

        float dx = radius * cosf(theta);
        float dy = radius * sinf(theta);

        dot += query[head_idx * head_dim + 2 * p]     * dx
             + query[head_idx * head_dim + 2 * p + 1] * dy;
    }

    scores[head_idx * seq_len + token_idx] = dot;
}

/* ============================================================
 * Host-callable wrappers
 * ============================================================ */

extern "C" void tq_fused_polar_cache_write(
    const float* d_keys,
    void* d_cache,
    const int* d_slot_mapping,
    int num_tokens,
    int num_heads,
    int head_dim,
    cudaStream_t stream)
{
    int threads = head_dim / 2;
    if (threads > TQ_MAX_THREADS) threads = TQ_MAX_THREADS;

    tq_fused_polar_cache_kernel<float, 16><<<num_tokens, threads, 0, stream>>>(
        d_keys,
        reinterpret_cast<tq_polar_block_d*>(d_cache),
        d_slot_mapping,
        num_tokens,
        num_heads,
        head_dim);
}

extern "C" void tq_fused_uniform4b_cache_write(
    const float* d_values,
    void* d_cache,
    const int* d_slot_mapping,
    int num_tokens,
    int num_heads,
    int head_dim,
    cudaStream_t stream)
{
    int threads = min(head_dim, TQ_MAX_THREADS);

    tq_fused_uniform_cache_kernel<float, 16, 4><<<num_tokens, threads, 0, stream>>>(
        d_values,
        d_cache,
        d_slot_mapping,
        num_tokens,
        num_heads,
        head_dim);
}

extern "C" void tq_fused_paged_attention(
    const float* d_query,
    const void* d_key_cache,
    float* d_scores,
    const int* d_block_table,
    const uint8_t* d_quant_types,
    int num_heads,
    int head_dim,
    int seq_len,
    int max_blocks,
    cudaStream_t stream)
{
    int threads_per_block = 256;
    dim3 grid(tq_cuda_num_blocks(seq_len, threads_per_block), num_heads, 1);

    tq_fused_paged_attention_kernel<16><<<grid, threads_per_block, 0, stream>>>(
        d_query,
        reinterpret_cast<const tq_polar_block_d*>(d_key_cache),
        d_scores,
        d_block_table,
        d_quant_types,
        num_heads,
        head_dim,
        seq_len,
        max_blocks);
}

#endif /* TQ_BUILD_CUDA */
