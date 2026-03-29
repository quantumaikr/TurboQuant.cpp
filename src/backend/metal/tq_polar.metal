/**
 * TurboQuant -- PolarQuant Metal compute shaders
 *
 * Quantizes key vectors using polar coordinates (theta, rho) on Apple GPU.
 * Uses threadgroup memory for cooperative loading and SIMD-group
 * reductions for min/max scale computation.
 */
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

/* ============================================================
 * Constants matching tq_types.h
 * ============================================================ */

constant int TQ_BK           = 128;
constant int TQ_PAIRS        = 64;   /* TQ_BK / 2 */
constant int TQ_INDICES_SIZE = 32;   /* TQ_BK / 4  (pairs/2 bytes) */

/* ============================================================
 * Block structures (matching C layout)
 * ============================================================ */

struct block_tq_polar {
    ushort rscale;
    ushort rmn;
    ushort tscale;
    ushort tmn;
    uchar  indices[64]; /* TQ_BK / 2 */
};

/* ============================================================
 * FP16 conversion helpers
 * ============================================================ */

inline float fp16_to_float(ushort h) {
    return as_type<half>(h);
}

inline ushort float_to_fp16(float f) {
    return as_type<ushort>(half(f));
}

/* ============================================================
 * SIMD-group min/max reduction
 *
 * Metal SIMD-groups (equivalent to warps) are 32 threads wide
 * on Apple GPUs. We use simd_shuffle_down for efficient reduction.
 * ============================================================ */

inline float simd_reduce_min(float val) {
    val = min(val, simd_shuffle_down(val, 16));
    val = min(val, simd_shuffle_down(val, 8));
    val = min(val, simd_shuffle_down(val, 4));
    val = min(val, simd_shuffle_down(val, 2));
    val = min(val, simd_shuffle_down(val, 1));
    return val;
}

inline float simd_reduce_max(float val) {
    val = max(val, simd_shuffle_down(val, 16));
    val = max(val, simd_shuffle_down(val, 8));
    val = max(val, simd_shuffle_down(val, 4));
    val = max(val, simd_shuffle_down(val, 2));
    val = max(val, simd_shuffle_down(val, 1));
    return val;
}

inline float simd_reduce_sum(float val) {
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return val;
}

/* ============================================================
 * Threadgroup-level reduction using SIMD-group results
 * ============================================================ */

inline float threadgroup_reduce_min(
    float val,
    uint tid,
    uint simd_lane,
    uint simd_group_id,
    uint num_simd_groups,
    threadgroup float* scratch)
{
    val = simd_reduce_min(val);
    if (simd_lane == 0) scratch[simd_group_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        val = scratch[tid];
    } else {
        val = INFINITY;
    }
    if (simd_group_id == 0) {
        val = simd_reduce_min(val);
    }
    return val;
}

inline float threadgroup_reduce_max(
    float val,
    uint tid,
    uint simd_lane,
    uint simd_group_id,
    uint num_simd_groups,
    threadgroup float* scratch)
{
    val = simd_reduce_max(val);
    if (simd_lane == 0) scratch[simd_group_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        val = scratch[tid];
    } else {
        val = -INFINITY;
    }
    if (simd_group_id == 0) {
        val = simd_reduce_max(val);
    }
    return val;
}

inline float threadgroup_reduce_sum(
    float val,
    uint tid,
    uint simd_lane,
    uint simd_group_id,
    uint num_simd_groups,
    threadgroup float* scratch)
{
    val = simd_reduce_sum(val);
    if (simd_lane == 0) scratch[simd_group_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_simd_groups) {
        val = scratch[tid];
    } else {
        val = 0.0f;
    }
    if (simd_group_id == 0) {
        val = simd_reduce_sum(val);
    }
    return val;
}

/* ============================================================
 * PolarQuant quantize kernel
 *
 * Threadgroup: (TQ_PAIRS, 1, 1) -- one thread per pair
 * Grid:       (num_blocks, 1, 1) -- one threadgroup per quant block
 *
 * Each threadgroup:
 *  1. Loads TQ_BK elements into threadgroup memory
 *  2. Converts each pair to polar coordinates
 *  3. SIMD-group reduction for min/max of theta and radius
 *  4. Quantizes to 2-bit theta + 2-bit rho
 *  5. Packs into indices array
 * ============================================================ */
kernel void tq_polar_quantize(
    device const float*        input       [[buffer(0)]],
    device block_tq_polar*     output      [[buffer(1)]],
    constant uint&             total_elems [[buffer(2)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint simd_lane     [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    /* Threadgroup shared memory */
    threadgroup float tg_keys[TQ_BK];
    threadgroup float tg_theta[TQ_PAIRS];
    threadgroup float tg_radius[TQ_PAIRS];
    threadgroup float tg_scratch[8]; /* for reductions across SIMD groups */
    threadgroup float tg_params[4];  /* tmin, tmax, rmin, rmax */

    uint base = tgid * TQ_BK;
    uint num_simd_groups = (TQ_PAIRS + 31) / 32;

    /* Load key pair into threadgroup memory */
    if (tid < uint(TQ_PAIRS)) {
        uint g0 = base + 2 * tid;
        uint g1 = base + 2 * tid + 1;
        tg_keys[2 * tid]     = (g0 < total_elems) ? input[g0] : 0.0f;
        tg_keys[2 * tid + 1] = (g1 < total_elems) ? input[g1] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Compute polar coordinates */
    float theta = 0.0f, radius = 0.0f;
    if (tid < uint(TQ_PAIRS)) {
        float x = tg_keys[2 * tid];
        float y = tg_keys[2 * tid + 1];
        radius = sqrt(x * x + y * y);
        theta  = atan2(y, x);
        if (theta < 0.0f) theta += 2.0f * 3.14159265358979323846f;
        tg_theta[tid]  = theta;
        tg_radius[tid] = radius;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Reduce for min/max */
    float t_for_min = (tid < uint(TQ_PAIRS)) ? theta  :  INFINITY;
    float r_for_min = (tid < uint(TQ_PAIRS)) ? radius :  INFINITY;
    float t_for_max = (tid < uint(TQ_PAIRS)) ? theta  : -INFINITY;
    float r_for_max = (tid < uint(TQ_PAIRS)) ? radius : -INFINITY;

    float tmin = threadgroup_reduce_min(t_for_min, tid, simd_lane, simd_group_id, num_simd_groups, tg_scratch);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tmax = threadgroup_reduce_max(t_for_max, tid, simd_lane, simd_group_id, num_simd_groups, tg_scratch);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rmin = threadgroup_reduce_min(r_for_min, tid, simd_lane, simd_group_id, num_simd_groups, tg_scratch);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rmax = threadgroup_reduce_max(r_for_max, tid, simd_lane, simd_group_id, num_simd_groups, tg_scratch);

    /* Broadcast parameters */
    if (tid == 0) {
        tg_params[0] = tmin; tg_params[1] = tmax;
        tg_params[2] = rmin; tg_params[3] = rmax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tmin = tg_params[0]; tmax = tg_params[1];
    rmin = tg_params[2]; rmax = tg_params[3];

    float trange = max(tmax - tmin, 1e-8f);
    float rrange = max(rmax - rmin, 1e-8f);
    float tscale = trange / 4.0f;
    float rscale = rrange / 4.0f;

    /* Write block header */
    if (tid == 0) {
        output[tgid].tscale = float_to_fp16(tscale);
        output[tgid].tmn    = float_to_fp16(tmin);
        output[tgid].rscale = float_to_fp16(rscale);
        output[tgid].rmn    = float_to_fp16(rmin);
    }

    /* Quantize and pack */
    threadgroup uchar tg_packed[TQ_PAIRS];
    if (tid < uint(TQ_PAIRS)) {
        float t = tg_theta[tid];
        float r = tg_radius[tid];

        int tq = int(floor((t - tmin) / tscale));
        int rq = int(floor((r - rmin) / rscale));
        tq = clamp(tq, 0, 3);
        rq = clamp(rq, 0, 3);

        uchar packed = uchar((rq << 2) | tq);

        /* Even pair -> low nibble, odd pair -> high nibble */
        if (tid % 2 == 0) {
            tg_packed[tid / 2] = packed;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(TQ_PAIRS) && (tid % 2 == 1)) {
        float t = tg_theta[tid];
        float r = tg_radius[tid];
        int tq = clamp(int(floor((t - tmin) / tscale)), 0, 3);
        int rq = clamp(int(floor((r - rmin) / rscale)), 0, 3);
        uchar packed = uchar((rq << 2) | tq);
        tg_packed[tid / 2] |= (packed << 4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Store packed indices */
    uint idx_bytes = TQ_PAIRS / 2; /* 32 bytes */
    if (tid < idx_bytes) {
        output[tgid].indices[tid] = tg_packed[tid];
    }
}

/* ============================================================
 * PolarQuant attention kernel
 *
 * Threadgroup: (TQ_PAIRS, 1, 1) -- one thread per pair
 * Grid:       (seq_len, 1, 1)   -- one threadgroup per token
 *
 * Each threadgroup computes: dot(query, dequant(key[s]))
 * Uses cos/sin lookup table in threadgroup memory.
 * ============================================================ */
kernel void tq_polar_attention(
    device const float*           query    [[buffer(0)]],
    device const block_tq_polar*  keys     [[buffer(1)]],
    device float*                 scores   [[buffer(2)]],
    constant uint&                seq_len  [[buffer(3)]],
    constant uint&                head_dim [[buffer(4)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint simd_lane     [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint s = tgid;
    if (s >= seq_len) return;

    uint pairs = head_dim / 2;

    /* Load block metadata */
    float tscale = fp16_to_float(keys[s].tscale);
    float tmin   = fp16_to_float(keys[s].tmn);
    float rscale = fp16_to_float(keys[s].rscale);
    float rmin   = fp16_to_float(keys[s].rmn);

    /* Precompute cos/sin lookup (4 levels) */
    threadgroup float tg_cos[4];
    threadgroup float tg_sin[4];
    threadgroup float tg_scratch[8];

    if (tid < 4) {
        float t = tmin + (float(tid) + 0.5f) * tscale;
        tg_cos[tid] = cos(t);
        tg_sin[tid] = sin(t);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Each thread dequantizes its pair and computes partial dot */
    float partial_dot = 0.0f;
    if (tid < pairs) {
        uchar byte = keys[s].indices[tid / 2];
        uchar packed = (tid % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        int tq = packed & 0x03;
        int rq = (packed >> 2) & 0x03;

        float rad = rmin + (float(rq) + 0.5f) * rscale;
        float dx = rad * tg_cos[tq];
        float dy = rad * tg_sin[tq];

        float qx = query[2 * tid];
        float qy = query[2 * tid + 1];
        partial_dot = qx * dx + qy * dy;
    }

    uint num_simd_groups = (pairs + 31) / 32;
    float total = threadgroup_reduce_sum(partial_dot, tid, simd_lane,
                                          simd_group_id, num_simd_groups,
                                          tg_scratch);

    if (tid == 0) {
        scores[s] = total;
    }
}
