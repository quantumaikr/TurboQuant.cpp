/**
 * TurboQuant -- PolarQuant CUDA kernels
 *
 * Quantizes key vectors using polar coordinates (theta, rho) on GPU.
 * Each CUDA block processes one quantization group of TQ_BK elements.
 * Uses shared memory for key loading and warp-level reductions
 * for min/max scale computation.
 */
#ifdef TQ_BUILD_CUDA

#include "tq_cuda_common.cuh"
#include <math.h>

/* ============================================================
 * PolarQuant quantize kernel
 *
 * Grid:  (num_blocks, 1, 1)
 * Block: (TQ_BK/2, 1, 1)  -- one thread per pair
 *
 * Each thread block processes one quantization group.
 * Shared memory holds the input keys for cooperative loading.
 * ============================================================ */
__global__ void tq_polar_quantize_kernel(
    const float* __restrict__ keys,
    tq_polar_block_d* __restrict__ out,
    int n,
    int head_dim)
{
    const int block_idx  = blockIdx.x;
    const int tid        = threadIdx.x;
    const int block_elems = TQ_BK_CUDA;
    const int pairs      = block_elems / 2;

    /* Shared memory for loading the key block */
    __shared__ float s_keys[TQ_BK_CUDA];
    /* Shared memory for polar coordinates */
    __shared__ float s_theta[TQ_BK_CUDA / 2];
    __shared__ float s_radius[TQ_BK_CUDA / 2];
    /* Shared memory for broadcasting scale/offset after reduction */
    __shared__ float s_params[4]; /* tmin, tmax, rmin, rmax */

    /* Global offset for this block */
    const int base = block_idx * block_elems;

    /* Cooperative load of key block into shared memory */
    if (tid < pairs) {
        int g0 = base + 2 * tid;
        int g1 = base + 2 * tid + 1;
        s_keys[2 * tid]     = (g0 < n) ? keys[g0] : 0.0f;
        s_keys[2 * tid + 1] = (g1 < n) ? keys[g1] : 0.0f;
    }
    __syncthreads();

    /* Each thread computes polar coords for its pair */
    float theta = 0.0f, radius = 0.0f;
    if (tid < pairs) {
        float x = s_keys[2 * tid];
        float y = s_keys[2 * tid + 1];
        radius = sqrtf(x * x + y * y);
        theta  = atan2f(y, x); /* [-pi, pi] */
        if (theta < 0.0f) theta += 2.0f * 3.14159265358979323846f;

        s_theta[tid]  = theta;
        s_radius[tid] = radius;
    } else {
        /* Padding threads contribute neutral values */
        theta  = 1e30f;   /* for min: will be overridden; for max: neutral */
        radius = 1e30f;
    }
    __syncthreads();

    /* Warp-level min/max reduction for theta */
    float t_local = (tid < pairs) ? theta : 1e30f;
    float r_local = (tid < pairs) ? radius : 1e30f;

    float tmin_v = blockReduceMin(t_local);
    float rmin_v = blockReduceMin(r_local);

    t_local = (tid < pairs) ? theta : -1e30f;
    r_local = (tid < pairs) ? radius : -1e30f;

    float tmax_v = blockReduceMax(t_local);
    float rmax_v = blockReduceMax(r_local);

    /* Thread 0 broadcasts results */
    if (tid == 0) {
        s_params[0] = tmin_v;
        s_params[1] = tmax_v;
        s_params[2] = rmin_v;
        s_params[3] = rmax_v;
    }
    __syncthreads();

    float tmin = s_params[0];
    float tmax = s_params[1];
    float rmin = s_params[2];
    float rmax = s_params[3];

    float trange = tmax - tmin;
    float rrange = rmax - rmin;
    if (trange < 1e-8f) trange = 1e-8f;
    if (rrange < 1e-8f) rrange = 1e-8f;

    float tscale = trange / 4.0f; /* 4 bins of width range/4 */
    float rscale = rrange / 4.0f;

    /* Thread 0 writes the block header */
    if (tid == 0) {
        out[block_idx].tscale = tq_float_to_half(tscale);
        out[block_idx].tmn    = tq_float_to_half(tmin);
        out[block_idx].rscale = tq_float_to_half(rscale);
        out[block_idx].rmn    = tq_float_to_half(rmin);
    }

    /* Each thread quantizes and packs its pair */
    if (tid < pairs) {
        float t = s_theta[tid];
        float r = s_radius[tid];

        int tq = __float2int_rd((t - tmin) / tscale);
        int rq = __float2int_rd((r - rmin) / rscale);
        tq = max(0, min(3, tq));
        rq = max(0, min(3, rq));

        uint8_t packed = (uint8_t)((rq << 2) | tq);

        /* Two pairs per byte, LSB-first packing */
        if (tid % 2 == 0) {
            out[block_idx].indices[tid / 2] = packed;
        } else {
            /* Atomic OR to avoid race with the even-index thread */
            atomicOr(reinterpret_cast<unsigned int*>(
                         &out[block_idx].indices[(tid / 2) & ~3u] /* align to 4B */),
                     (unsigned int)(packed << 4) << (8 * ((tid / 2) & 3)));
        }
    }

    /* Alternative safe packing: use shared memory staging */
    __syncthreads();

    /* Re-pack using shared staging to avoid atomics on bytes */
    __shared__ uint8_t s_packed[TQ_BK_CUDA / 2];
    if (tid < pairs) {
        float t = s_theta[tid];
        float r = s_radius[tid];

        int tq = __float2int_rd((t - tmin) / tscale);
        int rq = __float2int_rd((r - rmin) / rscale);
        tq = max(0, min(3, tq));
        rq = max(0, min(3, rq));

        uint8_t packed = (uint8_t)((rq << 2) | tq);

        /* Stage in shared: even pairs go to low nibble, odd to high */
        if (tid % 2 == 0) {
            s_packed[tid / 2] = packed;
        }
    }
    __syncthreads();

    if (tid < pairs && (tid % 2 == 1)) {
        float t = s_theta[tid];
        float r = s_radius[tid];
        int tq = __float2int_rd((t - tmin) / tscale);
        int rq = __float2int_rd((r - rmin) / rscale);
        tq = max(0, min(3, tq));
        rq = max(0, min(3, rq));
        uint8_t packed = (uint8_t)((rq << 2) | tq);
        s_packed[tid / 2] |= (packed << 4);
    }
    __syncthreads();

    /* Cooperative store of packed indices to global memory */
    int indices_bytes = pairs / 2;
    if (tid < indices_bytes) {
        out[block_idx].indices[tid] = s_packed[tid];
    }
}

/* ============================================================
 * PolarQuant attention kernel
 *
 * Grid:  (seq_len, 1, 1)
 * Block: (pairs=head_dim/2, 1, 1)
 *
 * Each block computes one attention score: dot(query, dequant(key[s]))
 * Uses shared memory for cos/sin lookup table and warp reduction.
 * ============================================================ */
__global__ void tq_polar_attention_kernel(
    const float* __restrict__ query,
    const tq_polar_block_d* __restrict__ keys,
    float* __restrict__ scores,
    int seq_len,
    int head_dim)
{
    const int s   = blockIdx.x;    /* sequence position */
    const int tid = threadIdx.x;
    const int pairs = head_dim / 2;

    if (s >= seq_len) return;

    /* Load block metadata */
    const tq_polar_block_d* block = &keys[s];
    float tscale = tq_half_to_float(block->tscale);
    float tmin   = tq_half_to_float(block->tmn);
    float rscale = tq_half_to_float(block->rscale);
    float rmin   = tq_half_to_float(block->rmn);

    /* Precompute cos/sin lookup for 4 theta levels in shared memory */
    __shared__ float s_cos_lut[4];
    __shared__ float s_sin_lut[4];
    if (tid < 4) {
        float theta = tmin + ((float)tid + 0.5f) * tscale;
        s_cos_lut[tid] = cosf(theta);
        s_sin_lut[tid] = sinf(theta);
    }
    __syncthreads();

    /* Each thread processes one pair, computing partial dot product */
    float partial_dot = 0.0f;

    if (tid < pairs) {
        /* Unpack index */
        uint8_t byte = block->indices[tid / 2];
        uint8_t packed = (tid % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        int tq = packed & 0x03;
        int rq = (packed >> 2) & 0x03;

        /* Dequantize to Cartesian */
        float radius = rmin + ((float)rq + 0.5f) * rscale;
        float dx = radius * s_cos_lut[tq];
        float dy = radius * s_sin_lut[tq];

        /* Dot product with query */
        float qx = query[2 * tid];
        float qy = query[2 * tid + 1];
        partial_dot = qx * dx + qy * dy;
    }

    /* Block-level reduction to get final dot product */
    float total = blockReduceSum(partial_dot);

    if (tid == 0) {
        scores[s] = total;
    }
}

/* ============================================================
 * Host-callable wrappers
 * ============================================================ */

extern "C" void tq_polar_quantize_cuda(
    const float* d_keys,
    void* d_out,
    int n,
    int head_dim,
    cudaStream_t stream)
{
    int num_blocks_q = (n + TQ_BK_CUDA - 1) / TQ_BK_CUDA;
    int threads = TQ_BK_CUDA / 2; /* one thread per pair */

    tq_polar_quantize_kernel<<<num_blocks_q, threads, 0, stream>>>(
        d_keys,
        reinterpret_cast<tq_polar_block_d*>(d_out),
        n,
        head_dim);
}

extern "C" void tq_polar_attention_cuda(
    const float* d_query,
    const void* d_keys,
    float* d_scores,
    int seq_len,
    int head_dim,
    cudaStream_t stream)
{
    int threads = head_dim / 2;
    if (threads > TQ_MAX_THREADS) threads = TQ_MAX_THREADS;

    tq_polar_attention_kernel<<<seq_len, threads, 0, stream>>>(
        d_query,
        reinterpret_cast<const tq_polar_block_d*>(d_keys),
        d_scores,
        seq_len,
        head_dim);
}

#endif /* TQ_BUILD_CUDA */
