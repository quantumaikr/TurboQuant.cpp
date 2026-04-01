/**
 * TurboQuant — GGUF weight matmul Metal compute shaders
 *
 * Fused dequant-matmul kernels for GGUF quantized weights.
 * Each kernel computes: out[row] = dot(dequant(weight_row), input)
 * where weight is in a GGUF quant format and input is FP32.
 *
 * Primary target: IQ2_XXS for 35B MoE inference on Apple Silicon.
 * Also includes Q8_0 and Q4_K for broader GGUF coverage.
 *
 * Threading model:
 *   - One threadgroup per output row
 *   - Threads within a threadgroup cooperatively reduce across blocks
 *   - Input vector cached in threadgroup memory for MoE (many small matmuls)
 */
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

/* ============================================================
 * IQ2_XXS codebook tables (constant memory)
 *
 * iq2xxs_grid: 256 entries, each uint64 encodes 8 lattice values
 * ksigns_iq2xs: 128 entries, each byte is a sign bitmask for 8 values
 * Total: 2048 + 128 = 2176 bytes — fits comfortably in constant memory
 * ============================================================ */

constant uint64_t iq2xxs_grid[256] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,
    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL,
    0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL,
    0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL,
    0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL,
    0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL,
    0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL,
    0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL,
    0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL,
    0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL,
    0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL,
    0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL,
    0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL,
    0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL,
    0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL,
    0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL,
    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL,
    0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL,
    0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL,
    0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL,
    0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL,
    0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL,
    0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL,
    0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL,
    0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL,
    0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL,
    0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL,
    0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL,
    0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL,
    0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL,
    0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL,
    0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL,
    0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL,
    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL,
    0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL,
    0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL,
    0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL,
    0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL,
    0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL,
    0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL,
    0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL,
    0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL,
    0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL,
    0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL,
    0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL,
    0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL,
    0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL,
    0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL,
    0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL,
    0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL,
    0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL,
    0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL,
    0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL,
    0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL,
    0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL,
    0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL,
    0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL,
    0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL,
    0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL,
    0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL,
    0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL,
    0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL,
    0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL,
};

constant uchar ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

/* ============================================================
 * IQ2_XXS fused dequant-matmul kernel
 *
 * Block format: 66 bytes per 256 elements (2.0625 bpw)
 *   - d (half, 2 bytes): super-block scale
 *   - qs[32] (uint16, 64 bytes): 8 groups x 4 uint16
 *     Each group decodes 32 floats via E8 lattice codebook
 *
 * Threading: one threadgroup per output row.
 * Threads within a threadgroup split blocks and use SIMD reduction.
 *
 * For MoE workloads (many 512x2048 matmuls), input is cached
 * in threadgroup shared memory to avoid redundant global reads.
 * ============================================================ */

/* Threadgroup size for matmul kernels */
constant uint TG_SIZE = 256;

/* Maximum input dimension cacheable in shared memory (32KB / 4 = 8192 floats) */
constant uint MAX_SHARED_DIM = 8192;

kernel void matmul_iq2_xxs(
    device const uchar*  weight    [[buffer(0)]],  /* [out_dim * row_bytes] */
    device const float*  input     [[buffer(1)]],  /* [in_dim] */
    device float*        output    [[buffer(2)]],  /* [out_dim] */
    constant uint&       in_dim    [[buffer(3)]],
    constant uint&       out_dim   [[buffer(4)]],
    threadgroup float*   shared_input [[threadgroup(0)]],
    uint  gid     [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]])
{
    uint row = gid;
    if (row >= out_dim) return;

    /* Cache input vector in shared memory (cooperative load) */
    for (uint i = tid; i < in_dim; i += tg_size) {
        shared_input[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_blocks = in_dim / 256;
    uint row_bytes = n_blocks * 66;
    device const uchar* row_ptr = weight + row * row_bytes;

    /* Each thread processes a subset of blocks */
    float partial_sum = 0.0f;

    for (uint b = tid; b < n_blocks; b += tg_size) {
        device const uchar* blk = row_ptr + b * 66;

        /* Read super-block scale (fp16) */
        half d_half = *reinterpret_cast<device const half*>(blk);
        float d = float(d_half);

        device const ushort* qs = reinterpret_cast<device const ushort*>(blk + 2);

        uint base_idx = b * 256;

        for (uint ib32 = 0; ib32 < 8; ib32++) {
            /* Reconstruct aux32 from 4 uint16 values */
            uint aux32_0 = uint(qs[4 * ib32])     | (uint(qs[4 * ib32 + 1]) << 16);
            uint aux32_1 = uint(qs[4 * ib32 + 2]) | (uint(qs[4 * ib32 + 3]) << 16);

            /* Sub-block scale: top 4 bits of aux32_1 */
            float db = d * (0.5f + float(aux32_1 >> 28)) * 0.25f;

            float group_sum = 0.0f;

            for (uint l = 0; l < 4; l++) {
                /* Grid index: one byte from aux32_0 */
                uchar grid_idx = uchar((aux32_0 >> (8 * l)) & 0xFF);
                uint64_t grid_val = iq2xxs_grid[grid_idx];

                /* Sign pattern: 7-bit index into ksigns table */
                uchar signs = ksigns_iq2xs[(aux32_1 >> (7 * l)) & 127];

                uint elem_idx = base_idx + ib32 * 32 + l * 8;

                /* Unrolled dot product of 8 dequantized weights with input */
                for (uint j = 0; j < 8; j++) {
                    float w = float((grid_val >> (8 * j)) & 0xFF);
                    if (signs & (1u << j)) w = -w;
                    group_sum += w * shared_input[elem_idx + j];
                }
            }

            partial_sum += db * group_sum;
        }
    }

    /* SIMD-group (warp) reduction */
    partial_sum += simd_shuffle_down(partial_sum, 16);
    partial_sum += simd_shuffle_down(partial_sum, 8);
    partial_sum += simd_shuffle_down(partial_sum, 4);
    partial_sum += simd_shuffle_down(partial_sum, 2);
    partial_sum += simd_shuffle_down(partial_sum, 1);

    /* Cross-SIMD-group reduction via shared memory */
    /* TG_SIZE / 32 = 8 SIMD groups max */
    threadgroup float simd_sums[8];

    uint simd_lane = tid % 32;
    uint simd_group = tid / 32;

    if (simd_lane == 0) {
        simd_sums[simd_group] = partial_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* First thread does final reduction across SIMD groups */
    if (tid == 0) {
        uint n_simd = (tg_size + 31) / 32;
        float total = 0.0f;
        for (uint s = 0; s < n_simd; s++) {
            total += simd_sums[s];
        }
        output[row] = total;
    }
}

/* ============================================================
 * Q8_0 fused dequant-matmul kernel
 *
 * Block format: 34 bytes per 32 elements
 *   - d (half, 2 bytes): scale
 *   - qs[32] (int8, 32 bytes): quantized values
 * ============================================================ */

kernel void matmul_q8_0(
    device const uchar*  weight    [[buffer(0)]],
    device const float*  input     [[buffer(1)]],
    device float*        output    [[buffer(2)]],
    constant uint&       in_dim    [[buffer(3)]],
    constant uint&       out_dim   [[buffer(4)]],
    threadgroup float*   shared_input [[threadgroup(0)]],
    uint  gid     [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]])
{
    uint row = gid;
    if (row >= out_dim) return;

    /* Cache input in shared memory */
    for (uint i = tid; i < in_dim; i += tg_size) {
        shared_input[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_blocks = in_dim / 32;
    uint row_bytes = n_blocks * 34;
    device const uchar* row_ptr = weight + row * row_bytes;

    float partial_sum = 0.0f;

    for (uint b = tid; b < n_blocks; b += tg_size) {
        device const uchar* blk = row_ptr + b * 34;

        float d = float(*reinterpret_cast<device const half*>(blk));
        device const char* qs = reinterpret_cast<device const char*>(blk + 2);

        uint base_idx = b * 32;
        float block_sum = 0.0f;

        /* Process 32 elements — unroll by 8 for better ILP */
        for (uint j = 0; j < 32; j += 8) {
            block_sum += float(qs[j + 0]) * shared_input[base_idx + j + 0];
            block_sum += float(qs[j + 1]) * shared_input[base_idx + j + 1];
            block_sum += float(qs[j + 2]) * shared_input[base_idx + j + 2];
            block_sum += float(qs[j + 3]) * shared_input[base_idx + j + 3];
            block_sum += float(qs[j + 4]) * shared_input[base_idx + j + 4];
            block_sum += float(qs[j + 5]) * shared_input[base_idx + j + 5];
            block_sum += float(qs[j + 6]) * shared_input[base_idx + j + 6];
            block_sum += float(qs[j + 7]) * shared_input[base_idx + j + 7];
        }

        partial_sum += d * block_sum;
    }

    /* SIMD reduction */
    partial_sum += simd_shuffle_down(partial_sum, 16);
    partial_sum += simd_shuffle_down(partial_sum, 8);
    partial_sum += simd_shuffle_down(partial_sum, 4);
    partial_sum += simd_shuffle_down(partial_sum, 2);
    partial_sum += simd_shuffle_down(partial_sum, 1);

    threadgroup float simd_sums[8];
    uint simd_lane = tid % 32;
    uint simd_group = tid / 32;

    if (simd_lane == 0) {
        simd_sums[simd_group] = partial_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint n_simd = (tg_size + 31) / 32;
        float total = 0.0f;
        for (uint s = 0; s < n_simd; s++) {
            total += simd_sums[s];
        }
        output[row] = total;
    }
}

/* ============================================================
 * Q4_K fused dequant-matmul kernel
 *
 * Block format: 144 bytes per 256 elements
 *   - d (half, 2 bytes): super-block scale
 *   - dmin (half, 2 bytes): super-block min
 *   - scales[12] (uchar, 12 bytes): 8 sub-block scales + mins packed
 *   - qs[128] (uchar, 128 bytes): 256 x 4-bit quantized values
 * ============================================================ */

kernel void matmul_q4_k(
    device const uchar*  weight    [[buffer(0)]],
    device const float*  input     [[buffer(1)]],
    device float*        output    [[buffer(2)]],
    constant uint&       in_dim    [[buffer(3)]],
    constant uint&       out_dim   [[buffer(4)]],
    threadgroup float*   shared_input [[threadgroup(0)]],
    uint  gid     [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]])
{
    uint row = gid;
    if (row >= out_dim) return;

    for (uint i = tid; i < in_dim; i += tg_size) {
        shared_input[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_blocks = in_dim / 256;
    uint row_bytes = n_blocks * 144;
    device const uchar* row_ptr = weight + row * row_bytes;

    float partial_sum = 0.0f;

    for (uint b = tid; b < n_blocks; b += tg_size) {
        device const uchar* blk = row_ptr + b * 144;

        float d    = float(*reinterpret_cast<device const half*>(blk));
        float dmin = float(*reinterpret_cast<device const half*>(blk + 2));
        device const uchar* sc = blk + 4;    /* 12 bytes of packed scales */
        device const uchar* qs = blk + 16;   /* 128 bytes of 4-bit data */

        uint base_idx = b * 256;

        /* Process 8 sub-blocks of 32 elements each */
        for (uint sb = 0; sb < 8; sb++) {
            /* Unpack 6-bit scale and 6-bit min from packed format */
            float scale, min_val;
            if (sb < 4) {
                scale   = float(sc[sb] & 0x3F) * d;
                min_val = float(sc[sb + 4] & 0x3F) * dmin;
            } else {
                scale   = float(((sc[sb + 4] & 0xF0) >> 4) | ((sc[sb - 4] >> 6) << 4)) * d;
                min_val = float(((sc[sb + 4] & 0x0F))       | ((sc[sb]     >> 6) << 4)) * dmin;
            }

            float block_sum = 0.0f;
            uint q_offset = sb * 16; /* 32 elements = 16 bytes in 4-bit */
            uint elem_offset = base_idx + sb * 32;

            for (uint j = 0; j < 16; j++) {
                uchar qbyte = qs[q_offset + j];
                float v0 = float(qbyte & 0x0F);
                float v1 = float(qbyte >> 4);
                block_sum += (scale * v0 - min_val) * shared_input[elem_offset + j * 2];
                block_sum += (scale * v1 - min_val) * shared_input[elem_offset + j * 2 + 1];
            }

            partial_sum += block_sum;
        }
    }

    /* SIMD reduction */
    partial_sum += simd_shuffle_down(partial_sum, 16);
    partial_sum += simd_shuffle_down(partial_sum, 8);
    partial_sum += simd_shuffle_down(partial_sum, 4);
    partial_sum += simd_shuffle_down(partial_sum, 2);
    partial_sum += simd_shuffle_down(partial_sum, 1);

    threadgroup float simd_sums[8];
    uint simd_lane = tid % 32;
    uint simd_group = tid / 32;

    if (simd_lane == 0) {
        simd_sums[simd_group] = partial_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint n_simd = (tg_size + 31) / 32;
        float total = 0.0f;
        for (uint s = 0; s < n_simd; s++) {
            total += simd_sums[s];
        }
        output[row] = total;
    }
}
