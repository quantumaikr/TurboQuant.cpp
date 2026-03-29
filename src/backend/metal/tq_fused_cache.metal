/**
 * TurboQuant -- Fused quantize + cache write Metal shaders
 *
 * Combines quantization and paged cache insertion into a single
 * compute shader dispatch, eliminating redundant memory traffic.
 * Follows the vLLM reshape_and_cache pattern for slot-mapped writes.
 */
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

/* ============================================================
 * Constants and structures
 * ============================================================ */

constant int TQ_BK    = 128;
constant int TQ_PAIRS = 64;

struct block_tq_polar_fc {
    ushort rscale;
    ushort rmn;
    ushort tscale;
    ushort tmn;
    uchar  indices[64];
};

struct block_tq_uniform_4b_fc {
    ushort scale;
    ushort zero_point;
    uchar  qs[64]; /* TQ_BK / 2 */
};

struct block_tq_uniform_2b_fc {
    ushort scale;
    ushort zero_point;
    uchar  qs[32]; /* TQ_BK / 4 */
};

/* ============================================================
 * Helpers
 * ============================================================ */

inline float fp16_to_float_fc(ushort h) {
    return as_type<half>(h);
}

inline ushort float_to_fp16_fc(float f) {
    return as_type<ushort>(half(f));
}

inline float simd_reduce_min_fc(float val) {
    val = min(val, simd_shuffle_down(val, 16));
    val = min(val, simd_shuffle_down(val, 8));
    val = min(val, simd_shuffle_down(val, 4));
    val = min(val, simd_shuffle_down(val, 2));
    val = min(val, simd_shuffle_down(val, 1));
    return val;
}

inline float simd_reduce_max_fc(float val) {
    val = max(val, simd_shuffle_down(val, 16));
    val = max(val, simd_shuffle_down(val, 8));
    val = max(val, simd_shuffle_down(val, 4));
    val = max(val, simd_shuffle_down(val, 2));
    val = max(val, simd_shuffle_down(val, 1));
    return val;
}

/* ============================================================
 * Fused PolarQuant quantize + cache write kernel
 *
 * Each threadgroup processes one token's key vector for one head.
 * slot_mapping[token_idx] maps to physical cache location.
 *
 * Threadgroup: (TQ_PAIRS, 1, 1)
 * Grid:       (num_tokens * num_heads, 1, 1)
 * ============================================================ */
kernel void tq_fused_polar_cache(
    device const float*            keys         [[buffer(0)]],
    device block_tq_polar_fc*      cache        [[buffer(1)]],
    device const int*              slot_mapping  [[buffer(2)]],
    constant uint&                 num_tokens   [[buffer(3)]],
    constant uint&                 num_heads    [[buffer(4)]],
    constant uint&                 head_dim     [[buffer(5)]],
    constant uint&                 block_size   [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
    uint token_idx = tgid / num_heads;
    uint head_idx  = tgid % num_heads;

    if (token_idx >= num_tokens) return;

    int slot = slot_mapping[token_idx];
    if (slot < 0) return; /* padding */

    uint cache_block_idx = uint(slot) / block_size;
    uint offset_in_block = uint(slot) % block_size;

    uint key_offset = (token_idx * num_heads + head_idx) * head_dim;
    uint pairs = head_dim / 2;

    /* Load key pair and convert to polar */
    threadgroup float tg_theta[TQ_PAIRS];
    threadgroup float tg_radius[TQ_PAIRS];
    threadgroup float tg_scratch[8];
    threadgroup float tg_params[4];

    float theta = 0.0f, radius = 0.0f;
    if (tid < pairs) {
        float x = keys[key_offset + 2 * tid];
        float y = keys[key_offset + 2 * tid + 1];
        radius = sqrt(x * x + y * y);
        theta  = atan2(y, x);
        if (theta < 0.0f) theta += 2.0f * 3.14159265358979323846f;
        tg_theta[tid]  = theta;
        tg_radius[tid] = radius;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Reductions */
    uint num_sg = (pairs + 31) / 32;

    float tmin_v = (tid < pairs) ? theta  :  INFINITY;
    float rmin_v = (tid < pairs) ? radius :  INFINITY;
    tmin_v = simd_reduce_min_fc(tmin_v);
    if (lane == 0) tg_scratch[sgid] = tmin_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < num_sg) tmin_v = tg_scratch[tid]; else tmin_v = INFINITY;
    if (sgid == 0) tmin_v = simd_reduce_min_fc(tmin_v);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    rmin_v = simd_reduce_min_fc(rmin_v);
    if (lane == 0) tg_scratch[sgid] = rmin_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < num_sg) rmin_v = tg_scratch[tid]; else rmin_v = INFINITY;
    if (sgid == 0) rmin_v = simd_reduce_min_fc(rmin_v);

    float tmax_v = (tid < pairs) ? theta  : -INFINITY;
    float rmax_v = (tid < pairs) ? radius : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tmax_v = simd_reduce_max_fc(tmax_v);
    if (lane == 0) tg_scratch[sgid] = tmax_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < num_sg) tmax_v = tg_scratch[tid]; else tmax_v = -INFINITY;
    if (sgid == 0) tmax_v = simd_reduce_max_fc(tmax_v);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    rmax_v = simd_reduce_max_fc(rmax_v);
    if (lane == 0) tg_scratch[sgid] = rmax_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < num_sg) rmax_v = tg_scratch[tid]; else rmax_v = -INFINITY;
    if (sgid == 0) rmax_v = simd_reduce_max_fc(rmax_v);

    if (tid == 0) {
        tg_params[0] = tmin_v; tg_params[1] = tmax_v;
        tg_params[2] = rmin_v; tg_params[3] = rmax_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tmin = tg_params[0], tmax = tg_params[1];
    float rmin = tg_params[2], rmax = tg_params[3];
    float tscale = max(tmax - tmin, 1e-8f) / 4.0f;
    float rscale = max(rmax - rmin, 1e-8f) / 4.0f;

    /* Compute cache output index */
    uint cache_idx = cache_block_idx * (num_heads * block_size)
                   + head_idx * block_size + offset_in_block;

    if (tid == 0) {
        cache[cache_idx].tscale = float_to_fp16_fc(tscale);
        cache[cache_idx].tmn    = float_to_fp16_fc(tmin);
        cache[cache_idx].rscale = float_to_fp16_fc(rscale);
        cache[cache_idx].rmn    = float_to_fp16_fc(rmin);
    }

    /* Pack indices */
    threadgroup uchar tg_packed[TQ_PAIRS];
    if (tid < pairs) {
        int tq = clamp(int(floor((tg_theta[tid] - tmin) / tscale)), 0, 3);
        int rq = clamp(int(floor((tg_radius[tid] - rmin) / rscale)), 0, 3);
        uchar packed = uchar((rq << 2) | tq);
        if (tid % 2 == 0) tg_packed[tid / 2] = packed;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < pairs && (tid % 2 == 1)) {
        int tq = clamp(int(floor((tg_theta[tid] - tmin) / tscale)), 0, 3);
        int rq = clamp(int(floor((tg_radius[tid] - rmin) / rscale)), 0, 3);
        uchar packed = uchar((rq << 2) | tq);
        tg_packed[tid / 2] |= (packed << 4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint idx_bytes = pairs / 2;
    if (tid < idx_bytes) {
        cache[cache_idx].indices[tid] = tg_packed[tid];
    }
}

/* ============================================================
 * Fused uniform 4-bit value quantize + cache write
 *
 * Threadgroup: (TQ_BK, 1, 1)
 * Grid:       (num_tokens * num_heads, 1, 1)
 * ============================================================ */
kernel void tq_fused_uniform4b_cache(
    device const float*              values       [[buffer(0)]],
    device block_tq_uniform_4b_fc*   cache        [[buffer(1)]],
    device const int*                slot_mapping  [[buffer(2)]],
    constant uint&                   num_tokens   [[buffer(3)]],
    constant uint&                   num_heads    [[buffer(4)]],
    constant uint&                   head_dim     [[buffer(5)]],
    constant uint&                   block_size   [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
    uint token_idx = tgid / num_heads;
    uint head_idx  = tgid % num_heads;

    if (token_idx >= num_tokens) return;

    int slot = slot_mapping[token_idx];
    if (slot < 0) return;

    uint cache_block_idx = uint(slot) / block_size;
    uint offset_in_block = uint(slot) % block_size;
    uint val_offset = (token_idx * num_heads + head_idx) * head_dim;

    threadgroup float tg_vals[TQ_BK];
    threadgroup float tg_scratch[8];

    /* Load and find min/max */
    float val = 0.0f;
    float local_min =  INFINITY;
    float local_max = -INFINITY;

    if (tid < head_dim) {
        val = values[val_offset + tid];
        tg_vals[tid] = val;
        local_min = val;
        local_max = val;
    }

    uint num_sg = (head_dim + 31) / 32;
    local_min = simd_reduce_min_fc(local_min);
    if (lane == 0) tg_scratch[sgid] = local_min;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < num_sg) local_min = tg_scratch[tid]; else local_min = INFINITY;
    if (sgid == 0) local_min = simd_reduce_min_fc(local_min);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    local_max = simd_reduce_max_fc(local_max);
    if (lane == 0) tg_scratch[sgid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < num_sg) local_max = tg_scratch[tid]; else local_max = -INFINITY;
    if (sgid == 0) local_max = simd_reduce_max_fc(local_max);

    threadgroup float tg_minmax[2];
    if (tid == 0) {
        tg_minmax[0] = local_min;
        tg_minmax[1] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gmin = tg_minmax[0];
    float gmax = tg_minmax[1];
    float range = max(gmax - gmin, 1e-8f);
    float scale = range / 16.0f;

    uint cache_idx = cache_block_idx * (num_heads * block_size)
                   + head_idx * block_size + offset_in_block;

    if (tid == 0) {
        cache[cache_idx].scale      = float_to_fp16_fc(scale);
        cache[cache_idx].zero_point = float_to_fp16_fc(gmin);
    }

    /* Pack two 4-bit values per byte */
    threadgroup uchar tg_quant[TQ_BK];
    if (tid < head_dim) {
        int q = clamp(int(floor((tg_vals[tid] - gmin) / scale)), 0, 15);
        tg_quant[tid] = uchar(q);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint pack_elems = head_dim / 2;
    if (tid < pack_elems) {
        uchar lo = tg_quant[2 * tid];
        uchar hi = (2 * tid + 1 < head_dim) ? tg_quant[2 * tid + 1] : 0;
        cache[cache_idx].qs[tid] = (hi << 4) | lo;
    }
}
