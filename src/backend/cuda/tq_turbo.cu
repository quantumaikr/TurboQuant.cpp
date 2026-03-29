/**
 * TurboQuant -- Composite CUDA kernel (PolarQuant + QJL residual)
 *
 * Performs two-stage quantization in a single kernel launch:
 *  Stage 1: PolarQuant quantization of key vectors
 *  Stage 2: QJL sign hash of the residual (original - polar_reconstruction)
 *
 * Attention kernel combines polar and residual scores.
 */
#ifdef TQ_BUILD_CUDA

#include "tq_cuda_common.cuh"
#include <math.h>

/* ============================================================
 * TurboQuant quantize kernel (fused polar + QJL residual)
 *
 * Grid:  (num_blocks, 1, 1)
 * Block: (TQ_BK_CUDA/2, 1, 1)  -- one thread per pair
 *
 * This kernel:
 *  1. Loads the key block into shared memory
 *  2. Computes polar coordinates (theta, rho) per pair
 *  3. Reduces to find min/max for scale computation
 *  4. Quantizes and packs polar indices
 *  5. Dequantizes (reconstructs) the polar approximation
 *  6. Computes residual = original - reconstruction
 *  7. Computes QJL sign hash of the residual
 * ============================================================ */
__global__ void tq_turbo_quantize_kernel(
    const float* __restrict__ keys,
    tq_turbo_block_d* __restrict__ out,
    int n,
    int head_dim)
{
    const int block_idx = blockIdx.x;
    const int tid       = threadIdx.x;
    const int pairs     = TQ_BK_CUDA / 2;

    __shared__ float s_keys[TQ_BK_CUDA];
    __shared__ float s_theta[TQ_BK_CUDA / 2];
    __shared__ float s_radius[TQ_BK_CUDA / 2];
    __shared__ float s_params[4]; /* tmin, tmax, rmin, rmax */
    __shared__ float s_residual[TQ_BK_CUDA];

    const int base = block_idx * TQ_BK_CUDA;

    /* Load key block */
    if (tid < pairs) {
        int g0 = base + 2 * tid;
        int g1 = base + 2 * tid + 1;
        s_keys[2 * tid]     = (g0 < n) ? keys[g0] : 0.0f;
        s_keys[2 * tid + 1] = (g1 < n) ? keys[g1] : 0.0f;
    }
    __syncthreads();

    /* Stage 1: Polar coordinate conversion */
    float theta = 0.0f, radius = 0.0f;
    if (tid < pairs) {
        float x = s_keys[2 * tid];
        float y = s_keys[2 * tid + 1];
        radius = sqrtf(x * x + y * y);
        theta  = atan2f(y, x);
        if (theta < 0.0f) theta += 2.0f * 3.14159265358979323846f;
        s_theta[tid]  = theta;
        s_radius[tid] = radius;
    }
    __syncthreads();

    /* Min/max reduction */
    float t_val = (tid < pairs) ? theta  :  1e30f;
    float r_val = (tid < pairs) ? radius :  1e30f;
    float tmin_v = blockReduceMin(t_val);
    float rmin_v = blockReduceMin(r_val);

    t_val = (tid < pairs) ? theta  : -1e30f;
    r_val = (tid < pairs) ? radius : -1e30f;
    float tmax_v = blockReduceMax(t_val);
    float rmax_v = blockReduceMax(r_val);

    if (tid == 0) {
        s_params[0] = tmin_v; s_params[1] = tmax_v;
        s_params[2] = rmin_v; s_params[3] = rmax_v;
    }
    __syncthreads();

    float tmin = s_params[0], tmax = s_params[1];
    float rmin = s_params[2], rmax = s_params[3];
    float trange = fmaxf(tmax - tmin, 1e-8f);
    float rrange = fmaxf(rmax - rmin, 1e-8f);
    float tscale = trange / 4.0f;
    float rscale = rrange / 4.0f;

    /* Write polar block header */
    if (tid == 0) {
        out[block_idx].polar.tscale = tq_float_to_half(tscale);
        out[block_idx].polar.tmn    = tq_float_to_half(tmin);
        out[block_idx].polar.rscale = tq_float_to_half(rscale);
        out[block_idx].polar.rmn    = tq_float_to_half(rmin);
    }

    /* Quantize, pack, and compute residual */
    __shared__ uint8_t s_packed[TQ_BK_CUDA / 2];
    if (tid < pairs) {
        float t = s_theta[tid];
        float r = s_radius[tid];

        int tq = __float2int_rd((t - tmin) / tscale);
        int rq = __float2int_rd((r - rmin) / rscale);
        tq = max(0, min(3, tq));
        rq = max(0, min(3, rq));

        uint8_t packed = (uint8_t)((rq << 2) | tq);
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

    /* Store packed indices */
    int idx_bytes = pairs / 2;
    if (tid < idx_bytes) {
        out[block_idx].polar.indices[tid] = s_packed[tid];
    }

    /* Compute reconstruction and residual */
    if (tid < pairs) {
        uint8_t byte = s_packed[tid / 2];
        uint8_t pk = (tid % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        int tqi = pk & 0x03;
        int rqi = (pk >> 2) & 0x03;

        float recon_theta  = tmin + ((float)tqi + 0.5f) * tscale;
        float recon_radius = rmin + ((float)rqi + 0.5f) * rscale;

        float recon_x = recon_radius * cosf(recon_theta);
        float recon_y = recon_radius * sinf(recon_theta);

        s_residual[2 * tid]     = s_keys[2 * tid]     - recon_x;
        s_residual[2 * tid + 1] = s_keys[2 * tid + 1] - recon_y;
    }
    __syncthreads();

    /* Stage 2: QJL sign hash of the residual */
    int dim = (TQ_BK_CUDA < head_dim) ? TQ_BK_CUDA : head_dim;

    /* Compute residual L2 norm (use thread 0) */
    if (tid == 0) {
        float norm_sq = 0.0f;
        for (int d = 0; d < dim; d++) {
            norm_sq += s_residual[d] * s_residual[d];
        }
        out[block_idx].residual.norm = tq_float_to_half(sqrtf(norm_sq));

        /* Outlier detection */
        float abs_vals[TQ_BK_CUDA];
        for (int d = 0; d < dim; d++) abs_vals[d] = fabsf(s_residual[d]);

        float outlier_norm_sq = 0.0f;
        for (int o = 0; o < TQ_OUTLIERS_CUDA; o++) {
            int best = 0; float best_val = -1.0f;
            for (int d = 0; d < dim; d++) {
                if (abs_vals[d] > best_val) { best_val = abs_vals[d]; best = d; }
            }
            out[block_idx].residual.outlier_idx[o] = (uint8_t)(best < 256 ? best : 255);
            float v = s_residual[best];
            outlier_norm_sq += v * v;
            abs_vals[best] = -1.0f;
        }
        out[block_idx].residual.outlier_norm = tq_float_to_half(sqrtf(outlier_norm_sq));
    }

    /* QJL hash: each thread handles a subset of sketch dimensions */
    int sketches_per_thread = (TQ_SKETCH_DIM_CUDA + pairs - 1) / pairs;
    for (int si = 0; si < sketches_per_thread; si++) {
        int sketch_idx = tid * sketches_per_thread + si;
        if (sketch_idx >= TQ_SKETCH_DIM_CUDA) break;

        float proj = 0.0f;
        for (int d = 0; d < dim; d++) {
            proj += s_residual[d] * tq_random_entry_d(d, sketch_idx);
        }

        int byte_idx = sketch_idx / 8;
        int bit_pos  = sketch_idx % 8;
        if (proj >= 0.0f) {
            atomicOr(reinterpret_cast<unsigned int*>(
                         &out[block_idx].residual.hash[byte_idx & ~3u]),
                     (1u << bit_pos) << (8 * (byte_idx & 3)));
        }
    }
}

/* ============================================================
 * TurboQuant attention kernel
 *
 * Grid:  (seq_len, 1, 1)
 * Block: (head_dim/2, 1, 1)
 *
 * Computes attention by dequantizing both polar and QJL residual,
 * summing them, and computing dot product with query.
 * ============================================================ */
__global__ void tq_turbo_attention_kernel(
    const float* __restrict__ query,
    const tq_turbo_block_d* __restrict__ keys,
    float* __restrict__ scores,
    int seq_len,
    int head_dim)
{
    const int s   = blockIdx.x;
    const int tid = threadIdx.x;
    const int pairs = head_dim / 2;

    if (s >= seq_len) return;

    const tq_turbo_block_d* block = &keys[s];

    /* Load polar metadata */
    float tscale = tq_half_to_float(block->polar.tscale);
    float tmin   = tq_half_to_float(block->polar.tmn);
    float rscale = tq_half_to_float(block->polar.rscale);
    float rmin   = tq_half_to_float(block->polar.rmn);

    /* Precompute cos/sin lookup */
    __shared__ float s_cos_lut[4];
    __shared__ float s_sin_lut[4];
    if (tid < 4) {
        float theta = tmin + ((float)tid + 0.5f) * tscale;
        s_cos_lut[tid] = cosf(theta);
        s_sin_lut[tid] = sinf(theta);
    }

    /* Shared memory for QJL residual reconstruction */
    __shared__ float s_residual_recon[TQ_BK_CUDA];
    __syncthreads();

    /* Polar dequantize + dot product */
    float partial_dot = 0.0f;
    if (tid < pairs) {
        uint8_t byte = block->polar.indices[tid / 2];
        uint8_t packed = (tid % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        int tq = packed & 0x03;
        int rq = (packed >> 2) & 0x03;

        float radius = rmin + ((float)rq + 0.5f) * rscale;
        float dx = radius * s_cos_lut[tq];
        float dy = radius * s_sin_lut[tq];

        float qx = query[2 * tid];
        float qy = query[2 * tid + 1];
        partial_dot = qx * dx + qy * dy;
    }

    /* QJL residual reconstruction: each thread reconstructs its pair */
    int dim = (pairs * 2 < head_dim) ? pairs * 2 : head_dim;
    if (tid < pairs) {
        /* Reconstruct residual[2*tid] and residual[2*tid+1] from QJL hash */
        for (int c = 0; c < 2; c++) {
            int d = 2 * tid + c;
            float val = 0.0f;
            for (int sk = 0; sk < TQ_SKETCH_DIM_CUDA; sk++) {
                int bit = (block->residual.hash[sk / 8] >> (sk % 8)) & 1;
                float sign = bit ? 1.0f : -1.0f;
                val += sign * tq_random_entry_d(d, sk);
            }
            s_residual_recon[d] = val;
        }
    }
    __syncthreads();

    /* Normalize residual reconstruction to match stored norm */
    float recon_norm_sq = 0.0f;
    float local_sq = 0.0f;
    if (tid < pairs) {
        float v0 = s_residual_recon[2 * tid];
        float v1 = s_residual_recon[2 * tid + 1];
        local_sq = v0 * v0 + v1 * v1;
    }
    recon_norm_sq = blockReduceSum(local_sq);

    __shared__ float s_scale_factor;
    if (tid == 0) {
        float recon_norm = sqrtf(recon_norm_sq);
        float target_norm = tq_half_to_float(block->residual.norm);
        s_scale_factor = (recon_norm > 1e-8f) ? (target_norm / recon_norm) : 0.0f;
    }
    __syncthreads();

    /* Add scaled residual dot product contribution */
    if (tid < pairs) {
        float r0 = s_residual_recon[2 * tid]     * s_scale_factor;
        float r1 = s_residual_recon[2 * tid + 1] * s_scale_factor;
        partial_dot += query[2 * tid] * r0 + query[2 * tid + 1] * r1;
    }

    float total = blockReduceSum(partial_dot);
    if (tid == 0) {
        scores[s] = total;
    }
}

/* ============================================================
 * Host-callable wrappers
 * ============================================================ */

extern "C" void tq_turbo_quantize_cuda(
    const float* d_keys,
    void* d_out,
    int n,
    int head_dim,
    cudaStream_t stream)
{
    int num_blocks = (n + TQ_BK_CUDA - 1) / TQ_BK_CUDA;
    int threads = TQ_BK_CUDA / 2;

    /* Zero output first (QJL hash uses atomicOr) */
    cudaMemsetAsync(d_out, 0, num_blocks * sizeof(tq_turbo_block_d), stream);

    tq_turbo_quantize_kernel<<<num_blocks, threads, 0, stream>>>(
        d_keys,
        reinterpret_cast<tq_turbo_block_d*>(d_out),
        n,
        head_dim);
}

extern "C" void tq_turbo_attention_cuda(
    const float* d_query,
    const void* d_keys,
    float* d_scores,
    int seq_len,
    int head_dim,
    cudaStream_t stream)
{
    int threads = head_dim / 2;
    if (threads > TQ_MAX_THREADS) threads = TQ_MAX_THREADS;

    tq_turbo_attention_kernel<<<seq_len, threads, 0, stream>>>(
        d_query,
        reinterpret_cast<const tq_turbo_block_d*>(d_keys),
        d_scores,
        seq_len,
        head_dim);
}

#endif /* TQ_BUILD_CUDA */
