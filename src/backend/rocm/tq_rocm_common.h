/**
 * TurboQuant -- ROCm/HIP common utilities
 *
 * Provides warp-level reductions, fp16 conversion helpers, error checking
 * macros, and block/grid size helpers used across all TQ ROCm kernels.
 *
 * Mechanically converted from CUDA (tq_cuda_common.cuh).
 */
#ifndef TQ_ROCM_COMMON_H
#define TQ_ROCM_COMMON_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdint>

/* ============================================================
 * Error checking
 * ============================================================ */

#define TQ_HIP_CHECK(call)                                                    \
    do {                                                                       \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            fprintf(stderr, "TQ HIP error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, hipGetErrorString(err));                \
            return;                                                            \
        }                                                                      \
    } while (0)

#define TQ_HIP_CHECK_STATUS(call)                                             \
    do {                                                                       \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            fprintf(stderr, "TQ HIP error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, hipGetErrorString(err));                \
            return -1;                                                         \
        }                                                                      \
    } while (0)

/* ============================================================
 * Constants
 * ============================================================ */

#define TQ_WARP_SIZE       64  /* AMD wavefront size (64 on GCN/CDNA/RDNA1, 32 on RDNA2+ wave32 mode) */
#define TQ_MAX_THREADS     256
#define TQ_BK_ROCM        128
#define TQ_BK_QJL_ROCM    256
#define TQ_SKETCH_DIM_ROCM 256
#define TQ_OUTLIERS_ROCM   4

/* ============================================================
 * FP16 <-> FP32 device helpers
 * ============================================================ */

__device__ __forceinline__ float tq_half_to_float(uint16_t h) {
    __half hv;
    /* Reinterpret the raw bits as __half */
    *reinterpret_cast<uint16_t*>(&hv) = h;
    return __half2float(hv);
}

__device__ __forceinline__ uint16_t tq_float_to_half(float f) {
    __half hv = __float2half(f);
    return *reinterpret_cast<uint16_t*>(&hv);
}

/* ============================================================
 * Warp-level reductions
 * HIP note: __shfl_down_sync mask argument is not used in HIP;
 * use __shfl_down directly. AMD wavefronts are lock-step.
 * ============================================================ */

__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
    for (int offset = TQ_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__device__ __forceinline__ float warpReduceMax(float val) {
#pragma unroll
    for (int offset = TQ_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down(val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

__device__ __forceinline__ float warpReduceMin(float val) {
#pragma unroll
    for (int offset = TQ_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down(val, offset);
        val = fminf(val, other);
    }
    return val;
}

__device__ __forceinline__ int warpReduceSumInt(int val) {
#pragma unroll
    for (int offset = TQ_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

/* ============================================================
 * Block-level reductions using shared memory
 * ============================================================ */

__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[TQ_WARP_SIZE];
    int lane = threadIdx.x % TQ_WARP_SIZE;
    int wid  = threadIdx.x / TQ_WARP_SIZE;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + TQ_WARP_SIZE - 1) / TQ_WARP_SIZE;
    val = (threadIdx.x < (unsigned)num_warps) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

__device__ __forceinline__ float blockReduceMax(float val) {
    __shared__ float shared[TQ_WARP_SIZE];
    int lane = threadIdx.x % TQ_WARP_SIZE;
    int wid  = threadIdx.x / TQ_WARP_SIZE;

    val = warpReduceMax(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + TQ_WARP_SIZE - 1) / TQ_WARP_SIZE;
    val = (threadIdx.x < (unsigned)num_warps) ? shared[lane] : -1e30f;

    if (wid == 0) {
        val = warpReduceMax(val);
    }
    return val;
}

__device__ __forceinline__ float blockReduceMin(float val) {
    __shared__ float shared[TQ_WARP_SIZE];
    int lane = threadIdx.x % TQ_WARP_SIZE;
    int wid  = threadIdx.x / TQ_WARP_SIZE;

    val = warpReduceMin(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + TQ_WARP_SIZE - 1) / TQ_WARP_SIZE;
    val = (threadIdx.x < (unsigned)num_warps) ? shared[lane] : 1e30f;

    if (wid == 0) {
        val = warpReduceMin(val);
    }
    return val;
}

/* ============================================================
 * Grid/block size helpers
 * ============================================================ */

inline int tq_rocm_num_blocks(int n, int threads_per_block) {
    return (n + threads_per_block - 1) / threads_per_block;
}

inline dim3 tq_rocm_grid_1d(int n, int threads = TQ_MAX_THREADS) {
    return dim3(tq_rocm_num_blocks(n, threads));
}

/* ============================================================
 * Deterministic pseudo-random projection entry (device version)
 * Matches the CPU reference: Rademacher (+1/-1) random variable
 * ============================================================ */

__device__ __forceinline__ float tq_random_entry_d(int dim_idx, int sketch_idx) {
    uint32_t h = (uint32_t)(dim_idx * 2654435761u + sketch_idx * 340573321u);
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return (h & 1) ? 1.0f : -1.0f;
}

/* ============================================================
 * Block structure definitions for ROCm (matching tq_types.h)
 * Redefined here to avoid C11 _Static_assert in HIP context
 * ============================================================ */

struct tq_polar_block_d {
    uint16_t rscale;
    uint16_t rmn;
    uint16_t tscale;
    uint16_t tmn;
    uint8_t  indices[TQ_BK_ROCM / 2]; /* 64 bytes for BK=128 */
};

struct tq_qjl_block_d {
    uint16_t norm;
    uint16_t outlier_norm;
    uint8_t  hash[TQ_SKETCH_DIM_ROCM / 8]; /* 32 bytes @256 */
    uint8_t  outlier_idx[TQ_OUTLIERS_ROCM]; /* 4 bytes */
};

struct tq_turbo_block_d {
    tq_polar_block_d polar;
    tq_qjl_block_d   residual;
};

struct tq_uniform_4b_block_d {
    uint16_t scale;
    uint16_t zero_point;
    uint8_t  qs[TQ_BK_ROCM / 2];
};

struct tq_uniform_2b_block_d {
    uint16_t sub_scale[4];   /* per-sub-block scale (fp16) */
    uint16_t sub_min[4];     /* per-sub-block minimum (fp16) */
    uint8_t  qs[TQ_BK_ROCM / 4];
};

#endif /* TQ_ROCM_COMMON_H */
