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
