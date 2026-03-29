/**
 * TurboQuant -- Composite Metal compute shaders (PolarQuant + QJL residual)
 *
 * Performs two-stage quantization: PolarQuant on the key vector,
 * then QJL sign hash on the residual (original - polar_reconstruction).
 * Attention kernel combines both stages.
 */
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

/* ============================================================
 * Constants and structures
 * ============================================================ */

constant int TQ_BK         = 128;
constant int TQ_PAIRS      = 64;
constant int TQ_SKETCH_DIM = 256;
constant int TQ_HASH_BYTES = 32;
constant int TQ_OUTLIERS   = 4;

struct block_tq_polar_m {
    ushort rscale;
    ushort rmn;
    ushort tscale;
    ushort tmn;
    uchar  indices[64];
};

struct block_tq_qjl_m {
    ushort norm;
    ushort outlier_norm;
    uchar  hash[32];
    uchar  outlier_idx[4];
};

struct block_tq_turbo_m {
    block_tq_polar_m polar;
    block_tq_qjl_m   residual;
};

/* ============================================================
 * Helpers
 * ============================================================ */

inline float fp16_to_float_m(ushort h) {
    return as_type<half>(h);
}

inline ushort float_to_fp16_m(float f) {
    return as_type<ushort>(half(f));
}

inline float random_entry_m(int dim_idx, int sketch_idx) {
    uint h = uint(dim_idx * 2654435761u + sketch_idx * 340573321u);
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return (h & 1u) ? 1.0f : -1.0f;
}

inline float simd_reduce_min_m(float val) {
    val = min(val, simd_shuffle_down(val, 16));
    val = min(val, simd_shuffle_down(val, 8));
    val = min(val, simd_shuffle_down(val, 4));
    val = min(val, simd_shuffle_down(val, 2));
    val = min(val, simd_shuffle_down(val, 1));
    return val;
}

inline float simd_reduce_max_m(float val) {
    val = max(val, simd_shuffle_down(val, 16));
    val = max(val, simd_shuffle_down(val, 8));
    val = max(val, simd_shuffle_down(val, 4));
    val = max(val, simd_shuffle_down(val, 2));
    val = max(val, simd_shuffle_down(val, 1));
    return val;
}

inline float simd_reduce_sum_m(float val) {
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return val;
}

/* ============================================================
 * TurboQuant quantize kernel
 *
 * Threadgroup: (TQ_PAIRS, 1, 1) -- one thread per pair
 * Grid:       (num_blocks, 1, 1)
 *
 * Performs polar quantization then QJL on residual in a single dispatch.
 * ============================================================ */
kernel void tq_turbo_quantize(
    device const float*         input       [[buffer(0)]],
    device block_tq_turbo_m*    output      [[buffer(1)]],
    constant uint&              total_elems [[buffer(2)]],
    constant uint&              head_dim_c  [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tg_keys[TQ_BK];
    threadgroup float tg_theta[TQ_PAIRS];
    threadgroup float tg_radius[TQ_PAIRS];
    threadgroup float tg_scratch[8];
    threadgroup float tg_params[4];
    threadgroup float tg_residual[TQ_BK];
    threadgroup uchar tg_packed[TQ_PAIRS];

    uint base = tgid * TQ_BK;
    uint num_sg = (TQ_PAIRS + 31) / 32;

    /* Load keys */
    if (tid < uint(TQ_PAIRS)) {
        uint g0 = base + 2 * tid;
        uint g1 = base + 2 * tid + 1;
        tg_keys[2 * tid]     = (g0 < total_elems) ? input[g0] : 0.0f;
        tg_keys[2 * tid + 1] = (g1 < total_elems) ? input[g1] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Stage 1: Polar quantization */
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

    /* Min/max reduction */
    float tmin_v = (tid < uint(TQ_PAIRS)) ? theta  :  INFINITY;
    float rmin_v = (tid < uint(TQ_PAIRS)) ? radius :  INFINITY;
    tmin_v = simd_reduce_min_m(tmin_v);
    rmin_v = simd_reduce_min_m(rmin_v);
    if (lane == 0) tg_scratch[sgid] = tmin_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < num_sg) tmin_v = tg_scratch[tid]; else tmin_v = INFINITY;
    if (sgid == 0) tmin_v = simd_reduce_min_m(tmin_v);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rmin_reduced = rmin_v;
    if (lane == 0) tg_scratch[sgid] = rmin_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < num_sg) rmin_reduced = tg_scratch[tid]; else rmin_reduced = INFINITY;
    if (sgid == 0) rmin_reduced = simd_reduce_min_m(rmin_reduced);

    float tmax_v = (tid < uint(TQ_PAIRS)) ? theta  : -INFINITY;
    float rmax_v = (tid < uint(TQ_PAIRS)) ? radius : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tmax_v = simd_reduce_max_m(tmax_v);
    if (lane == 0) tg_scratch[sgid] = tmax_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < num_sg) tmax_v = tg_scratch[tid]; else tmax_v = -INFINITY;
    if (sgid == 0) tmax_v = simd_reduce_max_m(tmax_v);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    rmax_v = simd_reduce_max_m(rmax_v);
    if (lane == 0) tg_scratch[sgid] = rmax_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < num_sg) rmax_v = tg_scratch[tid]; else rmax_v = -INFINITY;
    if (sgid == 0) rmax_v = simd_reduce_max_m(rmax_v);

    if (tid == 0) {
        tg_params[0] = tmin_v; tg_params[1] = tmax_v;
        tg_params[2] = rmin_reduced; tg_params[3] = rmax_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tmin = tg_params[0], tmax = tg_params[1];
    float rmin = tg_params[2], rmax = tg_params[3];
    float tscale = max(tmax - tmin, 1e-8f) / 4.0f;
    float rscale = max(rmax - rmin, 1e-8f) / 4.0f;

    if (tid == 0) {
        output[tgid].polar.tscale = float_to_fp16_m(tscale);
        output[tgid].polar.tmn    = float_to_fp16_m(tmin);
        output[tgid].polar.rscale = float_to_fp16_m(rscale);
        output[tgid].polar.rmn    = float_to_fp16_m(rmin);
    }

    /* Quantize, pack, and compute residual */
    if (tid < uint(TQ_PAIRS)) {
        int tq = clamp(int(floor((tg_theta[tid] - tmin) / tscale)), 0, 3);
        int rq = clamp(int(floor((tg_radius[tid] - rmin) / rscale)), 0, 3);
        uchar packed = uchar((rq << 2) | tq);
        if (tid % 2 == 0) tg_packed[tid / 2] = packed;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < uint(TQ_PAIRS) && (tid % 2 == 1)) {
        int tq = clamp(int(floor((tg_theta[tid] - tmin) / tscale)), 0, 3);
        int rq = clamp(int(floor((tg_radius[tid] - rmin) / rscale)), 0, 3);
        uchar packed = uchar((rq << 2) | tq);
        tg_packed[tid / 2] |= (packed << 4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint idx_bytes = TQ_PAIRS / 2;
    if (tid < idx_bytes) {
        output[tgid].polar.indices[tid] = tg_packed[tid];
    }

    /* Compute residual */
    if (tid < uint(TQ_PAIRS)) {
        uchar byte = tg_packed[tid / 2];
        uchar pk = (tid % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        int tqi = pk & 0x03;
        int rqi = (pk >> 2) & 0x03;

        float rt = tmin + (float(tqi) + 0.5f) * tscale;
        float rr = rmin + (float(rqi) + 0.5f) * rscale;
        float rx = rr * cos(rt);
        float ry = rr * sin(rt);

        tg_residual[2 * tid]     = tg_keys[2 * tid]     - rx;
        tg_residual[2 * tid + 1] = tg_keys[2 * tid + 1] - ry;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Stage 2: QJL sign hash of residual */
    uint dim = min(uint(TQ_BK), head_dim_c);

    /* Compute residual norm and outliers (thread 0) */
    if (tid == 0) {
        float norm_sq = 0.0f;
        for (uint d = 0; d < dim; d++) {
            norm_sq += tg_residual[d] * tg_residual[d];
        }
        output[tgid].residual.norm = float_to_fp16_m(sqrt(norm_sq));

        float abs_v[TQ_BK];
        for (uint d = 0; d < dim; d++) abs_v[d] = abs(tg_residual[d]);

        float olnorm_sq = 0.0f;
        for (int o = 0; o < TQ_OUTLIERS; o++) {
            int best = 0; float bv = -1.0f;
            for (uint d = 0; d < dim; d++) {
                if (abs_v[d] > bv) { bv = abs_v[d]; best = int(d); }
            }
            output[tgid].residual.outlier_idx[o] = uchar(best < 256 ? best : 255);
            float v = tg_residual[best];
            olnorm_sq += v * v;
            abs_v[best] = -1.0f;
        }
        output[tgid].residual.outlier_norm = float_to_fp16_m(sqrt(olnorm_sq));

        /* Zero hash before OR-ing */
        for (int b = 0; b < TQ_HASH_BYTES; b++) {
            output[tgid].residual.hash[b] = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Each thread computes a subset of sketch projections */
    int per_thread = (TQ_SKETCH_DIM + int(TQ_PAIRS) - 1) / int(TQ_PAIRS);
    for (int si = 0; si < per_thread; si++) {
        int sketch_idx = int(tid) * per_thread + si;
        if (sketch_idx >= TQ_SKETCH_DIM) break;

        float proj = 0.0f;
        for (uint d = 0; d < dim; d++) {
            proj += tg_residual[d] * random_entry_m(int(d), sketch_idx);
        }

        if (proj >= 0.0f) {
            int byte_idx = sketch_idx / 8;
            int bit_pos  = sketch_idx % 8;
            /* Atomic OR at byte level via device atomic */
            atomic_fetch_or_explicit(
                (device atomic_uint*)&output[tgid].residual.hash[byte_idx & ~3],
                uint((1u << bit_pos) << (8 * (byte_idx & 3))),
                memory_order_relaxed);
        }
    }
}

/* ============================================================
 * TurboQuant attention kernel
 *
 * Threadgroup: (TQ_PAIRS, 1, 1)
 * Grid:       (seq_len, 1, 1)
 *
 * Dequantizes polar + QJL residual, computes dot product with query.
 * ============================================================ */
kernel void tq_turbo_attention(
    device const float*            query    [[buffer(0)]],
    device const block_tq_turbo_m* keys     [[buffer(1)]],
    device float*                  scores   [[buffer(2)]],
    constant uint&                 seq_len  [[buffer(3)]],
    constant uint&                 head_dim [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
    uint s = tgid;
    if (s >= seq_len) return;

    uint pairs = head_dim / 2;
    uint num_sg = (pairs + 31) / 32;

    device const block_tq_turbo_m* blk = &keys[s];

    float tscale = fp16_to_float_m(blk->polar.tscale);
    float tmin   = fp16_to_float_m(blk->polar.tmn);
    float rscale = fp16_to_float_m(blk->polar.rscale);
    float rmin   = fp16_to_float_m(blk->polar.rmn);

    threadgroup float tg_cos[4];
    threadgroup float tg_sin[4];
    threadgroup float tg_scratch[8];
    threadgroup float tg_residual[TQ_BK];

    if (tid < 4) {
        float t = tmin + (float(tid) + 0.5f) * tscale;
        tg_cos[tid] = cos(t);
        tg_sin[tid] = sin(t);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Polar dequantize + dot */
    float partial_dot = 0.0f;
    if (tid < pairs) {
        uchar byte = blk->polar.indices[tid / 2];
        uchar packed = (tid % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        int tq = packed & 0x03;
        int rq = (packed >> 2) & 0x03;

        float rad = rmin + (float(rq) + 0.5f) * rscale;
        float dx = rad * tg_cos[tq];
        float dy = rad * tg_sin[tq];

        partial_dot = query[2 * tid] * dx + query[2 * tid + 1] * dy;
    }

    /* QJL residual reconstruction */
    if (tid < pairs) {
        for (int c = 0; c < 2; c++) {
            int d = int(2 * tid + uint(c));
            float val = 0.0f;
            for (int sk = 0; sk < TQ_SKETCH_DIM; sk++) {
                int bit = (blk->residual.hash[sk / 8] >> (sk % 8)) & 1;
                float sign = bit ? 1.0f : -1.0f;
                val += sign * random_entry_m(d, sk);
            }
            tg_residual[d] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Normalize residual */
    float local_sq = 0.0f;
    if (tid < pairs) {
        float v0 = tg_residual[2 * tid];
        float v1 = tg_residual[2 * tid + 1];
        local_sq = v0 * v0 + v1 * v1;
    }

    float norm_sq = simd_reduce_sum_m(local_sq);
    if (lane == 0) tg_scratch[sgid] = norm_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < num_sg) norm_sq = tg_scratch[tid]; else norm_sq = 0.0f;
    if (sgid == 0) norm_sq = simd_reduce_sum_m(norm_sq);

    threadgroup float tg_scale;
    if (tid == 0) {
        float rn = sqrt(norm_sq);
        float tn = fp16_to_float_m(blk->residual.norm);
        tg_scale = (rn > 1e-8f) ? (tn / rn) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < pairs) {
        float r0 = tg_residual[2 * tid]     * tg_scale;
        float r1 = tg_residual[2 * tid + 1] * tg_scale;
        partial_dot += query[2 * tid] * r0 + query[2 * tid + 1] * r1;
    }

    /* Final reduction */
    float total = simd_reduce_sum_m(partial_dot);
    if (lane == 0) tg_scratch[sgid] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < num_sg) total = tg_scratch[tid]; else total = 0.0f;
    if (sgid == 0) total = simd_reduce_sum_m(total);

    if (tid == 0) {
        scores[s] = total;
    }
}
