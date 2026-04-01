/**
 * TurboQuant — Value cache quantization Metal compute shaders
 */
#include <metal_stdlib>
using namespace metal;

constant int TQ_BK = 128;

kernel void tq_value_quantize_4b(
    device const float* input       [[buffer(0)]],
    device uchar*       output      [[buffer(1)]],
    device half2*       scale_zp    [[buffer(2)]],
    constant uint&      n           [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n / 2) return;

    uint i0 = tid * 2;
    uint i1 = tid * 2 + 1;

    float v0 = (i0 < n) ? input[i0] : 0.0f;
    float v1 = (i1 < n) ? input[i1] : 0.0f;

    /* Per-block scale loaded from buffer */
    float scale = float(scale_zp[0].x);
    float zp    = float(scale_zp[0].y);

    if (scale < 1e-8f) scale = 1e-8f;

    int q0 = clamp(int(floor((v0 - zp) / scale)), 0, 15);
    int q1 = clamp(int(floor((v1 - zp) / scale)), 0, 15);

    output[tid] = uchar((q1 << 4) | (q0 & 0x0F));
}

kernel void tq_value_dequantize_4b(
    device const uchar* input       [[buffer(0)]],
    device float*       output      [[buffer(1)]],
    device const half2* scale_zp    [[buffer(2)]],
    constant uint&      n           [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n / 2) return;

    uchar packed = input[tid];
    int q0 = packed & 0x0F;
    int q1 = (packed >> 4) & 0x0F;

    float scale = float(scale_zp[0].x);
    float zp    = float(scale_zp[0].y);

    uint i0 = tid * 2;
    uint i1 = tid * 2 + 1;
    if (i0 < n) output[i0] = (float(q0) + 0.5f) * scale + zp;
    if (i1 < n) output[i1] = (float(q1) + 0.5f) * scale + zp;
}

/* ============================================================
 * 2-bit value quantize
 *
 * Each thread packs 4 values into one byte (2 bits each, LSB-first).
 * ============================================================ */
kernel void tq_value_quantize_2b(
    device const float* input       [[buffer(0)]],
    device uchar*       output      [[buffer(1)]],
    device half2*       scale_zp    [[buffer(2)]],
    constant uint&      n           [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n / 4) return;

    uint base = tid * 4;

    float scale = float(scale_zp[0].x);
    float zp    = float(scale_zp[0].y);
    if (scale < 1e-8f) scale = 1e-8f;

    uchar byte_val = 0;
    for (uint k = 0; k < 4; k++) {
        uint idx = base + k;
        float v = (idx < n) ? input[idx] : 0.0f;
        int q = clamp(int(floor((v - zp) / scale)), 0, 3);
        byte_val |= uchar(q << (k * 2));
    }
    output[tid] = byte_val;
}

/* ============================================================
 * Fused dequantize + matmul for 4-bit values
 *
 * output[d] = sum_s( attn_weights[s] * dequant(values[s, d]) )
 *
 * Each thread computes one output dimension.
 * ============================================================ */
kernel void tq_value_dequant_matmul_4b(
    device const float* attn_weights  [[buffer(0)]],   /* [seq_len] */
    device const uchar* values        [[buffer(1)]],   /* packed 4-bit blocks */
    device const half2* scale_zp_buf  [[buffer(2)]],   /* [seq_len] scale/zp per block */
    device float*       output        [[buffer(3)]],   /* [head_dim] */
    constant uint&      seq_len       [[buffer(4)]],
    constant uint&      head_dim      [[buffer(5)]],
    uint d [[thread_position_in_grid]])
{
    if (d >= head_dim) return;

    uint dim_block  = d / TQ_BK;
    uint dim_offset = d % uint(TQ_BK);
    uint num_dim_blocks = (head_dim + uint(TQ_BK) - 1) / uint(TQ_BK);

    float accum = 0.0f;

    for (uint s = 0; s < seq_len; s++) {
        float w = attn_weights[s];
        if (abs(w) < 1e-10f) continue;

        uint blk_base = (s * num_dim_blocks + dim_block);
        float scale = float(scale_zp_buf[blk_base].x);
        float zp    = float(scale_zp_buf[blk_base].y);

        /* Unpack 4-bit value (2 per byte, LSB-first) */
        uint byte_idx = blk_base * (uint(TQ_BK) / 2) + dim_offset / 2;
        uchar byte_val = values[byte_idx];
        int q;
        if (dim_offset % 2 == 0) {
            q = byte_val & 0x0F;
        } else {
            q = (byte_val >> 4) & 0x0F;
        }

        float dequant_val = zp + (float(q) + 0.5f) * scale;
        accum += w * dequant_val;
    }

    output[d] = accum;
}

/* ============================================================
 * Fused dequantize + matmul for 2-bit values
 *
 * Same structure as 4b version but unpacks 2-bit values
 * (4 per byte, LSB-first).
 * ============================================================ */
kernel void tq_value_dequant_matmul_2b(
    device const float* attn_weights  [[buffer(0)]],
    device const uchar* values        [[buffer(1)]],
    device const half2* scale_zp_buf  [[buffer(2)]],
    device float*       output        [[buffer(3)]],
    constant uint&      seq_len       [[buffer(4)]],
    constant uint&      head_dim      [[buffer(5)]],
    uint d [[thread_position_in_grid]])
{
    if (d >= head_dim) return;

    uint dim_block  = d / TQ_BK;
    uint dim_offset = d % uint(TQ_BK);
    uint num_dim_blocks = (head_dim + uint(TQ_BK) - 1) / uint(TQ_BK);

    float accum = 0.0f;

    for (uint s = 0; s < seq_len; s++) {
        float w = attn_weights[s];
        if (abs(w) < 1e-10f) continue;

        uint blk_base = (s * num_dim_blocks + dim_block);
        float scale = float(scale_zp_buf[blk_base].x);
        float zp    = float(scale_zp_buf[blk_base].y);

        /* Unpack 2-bit value (4 per byte, LSB-first) */
        uint byte_idx = blk_base * (uint(TQ_BK) / 4) + dim_offset / 4;
        uchar byte_val = values[byte_idx];
        int shift = int(dim_offset % 4) * 2;
        int q = (byte_val >> shift) & 0x03;

        float dequant_val = zp + (float(q) + 0.5f) * scale;
        accum += w * dequant_val;
    }

    output[d] = accum;
}
