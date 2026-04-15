/**
 * tq_gguf_quants.c — GGUF weight dequantization and on-the-fly dequant matmul
 *
 * Implements dequantization for all major GGML quant types:
 *   Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, F16, F32, BF16
 * Plus stub implementations for IQ types (IQ2_XXS, IQ3_XXS, IQ4_XS).
 *
 * The matmul path includes NEON-optimized inner loop for Apple Silicon.
 *
 * Pure C11, no external dependencies.
 */

#include "turboquant/tq_gguf.h"
#include "turboquant/tq_engine.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define TQ_HAS_NEON 1
#else
#define TQ_HAS_NEON 0
#endif

/* MSVC: no __builtin_* intrinsics */
#ifdef _MSC_VER
#define __builtin_prefetch(addr, ...) ((void)0)
#define __builtin_return_address(n) ((void*)0)
#endif

/* ============================================================
 * FP16 / BF16 helpers
 * ============================================================ */

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            /* positive/negative zero */
            float r;
            uint32_t v = sign;
            memcpy(&r, &v, 4);
            return r;
        }
        /* subnormal: normalize by shifting mantissa up */
        exp = 1;
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x3FF;
        exp += 112;  /* fp16 bias (15) -> fp32 bias (127): 127-15 = 112 */
    } else if (exp == 31) {
        /* inf / nan */
        exp = 255;
    } else {
        exp += 112;
    }

    uint32_t bits = sign | ((uint32_t)exp << 23) | (mant << 13);
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

static inline float bf16_to_fp32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

/* ============================================================
 * Block structures (matching llama.cpp / ggml exactly)
 * ============================================================ */

/* Q8_0: 34 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    int8_t   qs[32];  /* quantized values */
} block_q8_0;

/* Q4_K: 144 bytes, 256 elements */
typedef struct {
    uint16_t d;          /* fp16 super-block scale */
    uint16_t dmin;       /* fp16 super-block min */
    uint8_t  scales[12]; /* sub-block scales+mins, 6-bit packed */
    uint8_t  qs[128];    /* 4-bit values, 2 per byte */
} block_q4_K;

/* Q2_K: 84 bytes, 256 elements */
typedef struct {
    uint8_t  scales[16]; /* sub-block scales+mins, 4-bit each */
    uint8_t  qs[64];     /* 2-bit values, 4 per byte */
    uint16_t d;          /* fp16 super-block scale */
    uint16_t dmin;       /* fp16 super-block min */
} block_q2_K;

/* Q3_K: 110 bytes, 256 elements */
typedef struct {
    uint8_t  hmask[32];  /* high bits */
    uint8_t  qs[64];     /* low 2 bits, 4 per byte */
    uint8_t  scales[12]; /* sub-block scales, packed */
    uint16_t d;          /* fp16 scale */
} block_q3_K;

/* Q6_K: 210 bytes, 256 elements */
typedef struct {
    uint8_t  ql[128];    /* low 4 bits */
    uint8_t  qh[64];     /* high 2 bits */
    int8_t   scales[16]; /* sub-block scales (signed int8) */
    uint16_t d;          /* fp16 super-block scale */
} block_q6_K;

/* Q5_K: 176 bytes, 256 elements */
typedef struct {
    uint16_t d;          /* fp16 super-block scale */
    uint16_t dmin;       /* fp16 super-block min */
    uint8_t  scales[12]; /* sub-block scales+mins, 6-bit packed */
    uint8_t  qh[32];     /* high bit for each of 256 elements */
    uint8_t  qs[128];    /* low 4 bits, 2 per byte */
} block_q5_K;

/* Q4_0: 18 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    uint8_t  qs[16];  /* 4-bit values, 2 per byte */
} block_q4_0;

/* Q4_1: 20 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    uint16_t m;       /* fp16 min */
    uint8_t  qs[16];  /* 4-bit values, 2 per byte */
} block_q4_1;

/* Q5_0: 22 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    uint8_t  qh[4];   /* high bits packed */
    uint8_t  qs[16];  /* low 4 bits, 2 per byte */
} block_q5_0;

/* Q5_1: 24 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    uint16_t m;       /* fp16 min */
    uint8_t  qh[4];   /* high bits packed */
    uint8_t  qs[16];  /* low 4 bits, 2 per byte */
} block_q5_1;

/* Q8_1: 36 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale (delta) */
    uint16_t s;       /* fp16 sum */
    int8_t   qs[32];
} block_q8_1;

/* Type size / block size / name — defined in tq_gguf.c, just declared in header */

/* ============================================================
 * Per-type dequantization
 * ============================================================ */

/* --- F32: passthrough --- */
static void dequant_f32(const void* src, float* dst, int n) {
    memcpy(dst, src, (size_t)n * sizeof(float));
}

/* --- F16 --- */
static void dequant_f16(const void* src, float* dst, int n) {
    const uint16_t* s = (const uint16_t*)src;
    for (int i = 0; i < n; i++) {
        dst[i] = fp16_to_fp32(s[i]);
    }
}

/* --- BF16 --- */
static void dequant_bf16(const void* src, float* dst, int n) {
    const uint16_t* s = (const uint16_t*)src;
    for (int i = 0; i < n; i++) {
        dst[i] = bf16_to_fp32(s[i]);
    }
}

/* --- Q8_0: 34 bytes, 32 elements --- */
static void dequant_q8_0(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q8_0* blk = (const block_q8_0*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        for (int j = 0; j < 32; j++) {
            dst[b * 32 + j] = d * blk[b].qs[j];
        }
    }
}

/* --- Q8_1: 36 bytes, 32 elements --- */
static void dequant_q8_1(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q8_1* blk = (const block_q8_1*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        for (int j = 0; j < 32; j++) {
            dst[b * 32 + j] = d * blk[b].qs[j];
        }
    }
}

/* --- Q4_0: 18 bytes, 32 elements --- */
static void dequant_q4_0(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q4_0* blk = (const block_q4_0*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            dst[b * 32 + j]      = d * ((int)(byte & 0x0F) - 8);
            dst[b * 32 + j + 16] = d * ((int)(byte >> 4) - 8);
        }
    }
}

/* --- Q4_1: 20 bytes, 32 elements --- */
static void dequant_q4_1(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q4_1* blk = (const block_q4_1*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float m = fp16_to_fp32(blk[b].m);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            dst[b * 32 + j]      = d * (byte & 0x0F) + m;
            dst[b * 32 + j + 16] = d * (byte >> 4) + m;
        }
    }
}

/* --- Q5_0: 22 bytes, 32 elements --- */
static void dequant_q5_0(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q5_0* blk = (const block_q5_0*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        uint32_t qh;
        memcpy(&qh, blk[b].qh, 4);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            int lo = (byte & 0x0F) | (((qh >> j) & 1) << 4);
            int hi = (byte >> 4)   | (((qh >> (j + 16)) & 1) << 4);
            dst[b * 32 + j]      = d * (lo - 16);
            dst[b * 32 + j + 16] = d * (hi - 16);
        }
    }
}

/* --- Q5_1: 24 bytes, 32 elements --- */
static void dequant_q5_1(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q5_1* blk = (const block_q5_1*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float m = fp16_to_fp32(blk[b].m);
        uint32_t qh;
        memcpy(&qh, blk[b].qh, 4);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            int lo = (byte & 0x0F) | (((qh >> j) & 1) << 4);
            int hi = (byte >> 4)   | (((qh >> (j + 16)) & 1) << 4);
            dst[b * 32 + j]      = d * lo + m;
            dst[b * 32 + j + 16] = d * hi + m;
        }
    }
}

/* --- Q2_K: 84 bytes, 256 elements --- */
static void dequant_q2_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q2_K* blk = (const block_q2_K*)src;

    for (int b = 0; b < nb; b++) {
        const float d    = fp16_to_fp32(blk[b].d);
        const float dmin = fp16_to_fp32(blk[b].dmin);

        const uint8_t* q = blk[b].qs;
        float* y = dst + b * 256;

        int is = 0;
        for (int half = 0; half < 2; half++) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc_byte = blk[b].scales[is++];
                float dl = d * (sc_byte & 0x0F);
                float ml = dmin * (sc_byte >> 4);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                sc_byte = blk[b].scales[is++];
                dl = d * (sc_byte & 0x0F);
                ml = dmin * (sc_byte >> 4);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

/* --- Q3_K: 110 bytes, 256 elements ---
 * 3-bit = 2 low bits (qs) + 1 high bit (hmask)
 * 16 sub-blocks with 6-bit scales packed into 12 bytes */
static void dequant_q3_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q3_K* blk = (const block_q3_K*)src;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t* scales = (const int8_t*)aux;

    for (int b = 0; b < nb; b++) {
        const float d_all = fp16_to_fp32(blk[b].d);

        const uint8_t* q  = blk[b].qs;
        const uint8_t* hm = blk[b].hmask;
        uint8_t m = 1;

        /* Decode 16 x 6-bit scales using the ggml bit-manipulation trick.
         * The 12 packed bytes are loaded as three uint32, then rearranged
         * into four uint32 that are reinterpreted as sixteen int8 values. */
        memcpy(aux, blk[b].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float* y = dst + b * 256;

        for (int half = 0; half < 2; half++) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

/* --- Q4_K: 144 bytes, 256 elements ---
 * 8 sub-blocks of 32 elements each
 * 6-bit scale/min packed in 12 bytes */
static void dequant_q4_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q4_K* blk = (const block_q4_K*)src;

    for (int b = 0; b < nb; b++) {
        const float d    = fp16_to_fp32(blk[b].d);
        const float dmin = fp16_to_fp32(blk[b].dmin);

        /* Decode 8 scale/min pairs from 12 bytes.
         * bytes 0..3:  low 6 bits of scales[0..3]
         * bytes 4..7:  low 6 bits of mins[0..3]
         * bytes 8..9:  high 2 bits of scales[0..3] + scales[4..7]
         * bytes 10..11: high 2 bits of mins[0..3] + mins[4..7]
         *
         * Actually ggml Q4_K packing:
         *   scales[0..5]: low 6 bits of scale for sub-blocks 0..5
         *                 but the first 4 bytes have scale low 6,
         *                 bytes 4..7 have min low 6,
         *                 bytes 8..11 have the high bits.
         *
         * Match ggml exactly:
         */
        uint8_t sc[8], mn[8];

        /* Low 6 bits */
        sc[0] = blk[b].scales[0] & 63;
        sc[1] = blk[b].scales[1] & 63;
        sc[2] = blk[b].scales[2] & 63;
        sc[3] = blk[b].scales[3] & 63;
        mn[0] = blk[b].scales[4] & 63;
        mn[1] = blk[b].scales[5] & 63;
        mn[2] = blk[b].scales[6] & 63;
        mn[3] = blk[b].scales[7] & 63;

        /* High 2 bits from bytes 8..11 */
        sc[4] = (blk[b].scales[8] & 0x0F) | ((blk[b].scales[0] >> 6) << 4);
        sc[5] = (blk[b].scales[9] & 0x0F) | ((blk[b].scales[1] >> 6) << 4);
        sc[6] = (blk[b].scales[10] & 0x0F) | ((blk[b].scales[2] >> 6) << 4);
        sc[7] = (blk[b].scales[11] & 0x0F) | ((blk[b].scales[3] >> 6) << 4);
        mn[4] = (blk[b].scales[8] >> 4) | ((blk[b].scales[4] >> 6) << 4);
        mn[5] = (blk[b].scales[9] >> 4) | ((blk[b].scales[5] >> 6) << 4);
        mn[6] = (blk[b].scales[10] >> 4) | ((blk[b].scales[6] >> 6) << 4);
        mn[7] = (blk[b].scales[11] >> 4) | ((blk[b].scales[7] >> 6) << 4);

        /* 4 groups of 64 elements (2 sub-blocks each).
         * Within each 64-element group, the first 32 elements use the low
         * nibble and the next 32 use the high nibble of the same 32 bytes.
         * This matches the ggml Q4_K packing exactly. */
        const uint8_t* q = blk[b].qs;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            const float d1 = d * sc[is + 0];
            const float m1 = dmin * mn[is + 0];
            const float d2 = d * sc[is + 1];
            const float m2 = dmin * mn[is + 1];
            for (int l = 0; l < 32; ++l)
                dst[b * 256 + j + l]      = d1 * (q[l] & 0x0F) - m1;
            for (int l = 0; l < 32; ++l)
                dst[b * 256 + j + 32 + l] = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

/* --- Q5_K: 176 bytes, 256 elements ---
 * Like Q4_K but with an extra high bit per element */
static void dequant_q5_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q5_K* blk = (const block_q5_K*)src;

    for (int b = 0; b < nb; b++) {
        const float d    = fp16_to_fp32(blk[b].d);
        const float dmin = fp16_to_fp32(blk[b].dmin);

        /* Same scale/min packing as Q4_K */
        uint8_t sc[8], mn[8];

        sc[0] = blk[b].scales[0] & 63;
        sc[1] = blk[b].scales[1] & 63;
        sc[2] = blk[b].scales[2] & 63;
        sc[3] = blk[b].scales[3] & 63;
        mn[0] = blk[b].scales[4] & 63;
        mn[1] = blk[b].scales[5] & 63;
        mn[2] = blk[b].scales[6] & 63;
        mn[3] = blk[b].scales[7] & 63;

        sc[4] = (blk[b].scales[8] & 0x0F) | ((blk[b].scales[0] >> 6) << 4);
        sc[5] = (blk[b].scales[9] & 0x0F) | ((blk[b].scales[1] >> 6) << 4);
        sc[6] = (blk[b].scales[10] & 0x0F) | ((blk[b].scales[2] >> 6) << 4);
        sc[7] = (blk[b].scales[11] & 0x0F) | ((blk[b].scales[3] >> 6) << 4);
        mn[4] = (blk[b].scales[8] >> 4) | ((blk[b].scales[4] >> 6) << 4);
        mn[5] = (blk[b].scales[9] >> 4) | ((blk[b].scales[5] >> 6) << 4);
        mn[6] = (blk[b].scales[10] >> 4) | ((blk[b].scales[6] >> 6) << 4);
        mn[7] = (blk[b].scales[11] >> 4) | ((blk[b].scales[7] >> 6) << 4);

        /* 4 groups of 64 elements (2 sub-blocks each), matching ggml Q5_K.
         * Low 4 bits: low nibble for first 32 elems, high nibble for next 32.
         * High bit: from qh, using bitmasks u1/u2 that shift left by 2 each group. */
        const uint8_t* ql = blk[b].qs;
        const uint8_t* qh = blk[b].qh;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            const float d1 = d * sc[is + 0];
            const float m1 = dmin * mn[is + 0];
            const float d2 = d * sc[is + 1];
            const float m2 = dmin * mn[is + 1];
            for (int l = 0; l < 32; ++l)
                dst[b * 256 + j + l]      = d1 * ((ql[l] & 0x0F) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l)
                dst[b * 256 + j + 32 + l] = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/* --- Q6_K: 210 bytes, 256 elements ---
 * 6-bit = 4 low bits (ql) + 2 high bits (qh)
 * 16 sub-blocks of 16 elements, int8 scales */
static void dequant_q6_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q6_K* blk = (const block_q6_K*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);

        /* Match ggml dequantize_row_q6_K exactly.
         * Processes in two 128-element halves. Within each half, 32
         * iterations produce 4 output elements each by interleaving
         * ql low/high nibbles and qh 2-bit fields. */
        const uint8_t* ql = blk[b].ql;
        const uint8_t* qh = blk[b].qh;
        const int8_t*  sc = blk[b].scales;
        float* y = dst + b * 256;

        for (int half = 0; half < 2; half++) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/* ============================================================
 * IQ2_XXS dequantization — E8 lattice codebook
 *
 * Block: 66 bytes per 256 elements (2.0625 bpw)
 *   - d (fp16): super-block scale
 *   - qs[32] (uint16): 8 groups of 4 uint16 (8 bytes each)
 *     Each 8-byte group decodes 32 floats:
 *     - aux32[0]: 4 grid indices (1 byte each) → 4×8=32 values from iq2xxs_grid
 *     - aux32[1] bits 0-27: 4×7-bit sign fields → ksigns_iq2xs → 8-bit patterns
 *     - aux32[1] bits 28-31: 4-bit sub-block scale
 * ============================================================ */

static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint64_t iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

static void dequant_iq2_xxs(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)src;

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 66; /* 66 bytes per block */
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint16_t* qs = (const uint16_t*)(blk + 2);

        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32[2];
            memcpy(aux32, qs + 4 * ib32, 8);
            const uint8_t* aux8 = (const uint8_t*)aux32;

            const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;

            for (int l = 0; l < 4; l++) {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];

                for (int j = 0; j < 8; j++) {
                    dst[b * 256 + ib32 * 32 + l * 8 + j] =
                        db * (float)grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                }
            }
        }
    }
}

/* ============================================================
 * IQ2_S dequantization — 82 bytes per 256 elements (2.5625 bpw)
 *
 * Block layout:
 *   d (fp16, 2 bytes): super-block scale
 *   qs[64]: first 32 bytes = grid index low bits, next 32 = sign bits
 *   qh[8]: high 2 bits of 10-bit grid index
 *   scales[8]: 4-bit sub-block scales (2 per byte)
 *
 * Uses iq2s_grid[1024] lookup table (10-bit index).
 * ============================================================ */

/* iq2s_grid: 1024-entry E8 lattice codebook for IQ2_S (from ggml-common.h).
 * Each uint64 packs 8 unsigned magnitude bytes from {0x08, 0x19, 0x2b}. */

static const uint64_t iq2s_grid[1024] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919,
    0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908,
    0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908,
    0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19,
    0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919,
    0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908,
    0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b,
    0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b,
    0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908,
    0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b,
    0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b,
    0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808,
    0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b,
    0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08,
    0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808,
    0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819,
    0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908,
    0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b,
    0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808,
    0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919,
    0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808,
    0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b,
    0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b,
    0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919,
    0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919,
    0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819,
    0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908,
    0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908,
    0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908,
    0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908,
    0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819,
    0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08,
    0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819,
    0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819,
    0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808,
    0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08,
    0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908,
    0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819,
    0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808,
    0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908,
    0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819,
    0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19,
    0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919,
    0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808,
    0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908,
    0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08,
    0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908,
    0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808,
    0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908,
    0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819,
    0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808,
    0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808,
    0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b,
    0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808,
    0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808,
    0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19,
    0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08,
    0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908,
    0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b,
    0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808,
    0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908,
    0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08,
    0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919,
    0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b,
    0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b,
    0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08,
    0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808,
    0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919,
    0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808,
    0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819,
    0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808,
    0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919,
    0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b,
    0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808,
    0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b,
    0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b,
    0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b,
    0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08,
    0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b,
    0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08,
    0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919,
    0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19,
    0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919,
    0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819,
    0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919,
    0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808,
    0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919,
    0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b,
    0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919,
    0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08,
    0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808,
    0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908,
    0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808,
    0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908,
    0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808,
    0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908,
    0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808,
    0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808,
    0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b,
    0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08,
    0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919,
    0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908,
    0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808,
    0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908,
    0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808,
    0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808,
    0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908,
    0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08,
    0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b,
    0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819,
    0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819,
    0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808,
    0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908,
    0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908,
    0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819,
    0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b,
    0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819,
    0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b,
    0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819,
    0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819,
    0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
};

/* Public accessor for the IQ2_S codebook — used by Metal backend */
const uint64_t* tq_iq2s_grid(void) {
    return iq2s_grid;
}

static void dequant_iq2_s(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)src;

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 82;
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);

        const uint8_t* qs = blk + 2;           /* grid index low bytes */
        const uint8_t* signs = qs + 32;         /* sign bytes (second half of qs) */
        const uint8_t* qh = blk + 66;           /* high bits: blk + 2 + 64 */
        const uint8_t* scales = blk + 74;       /* scales: blk + 2 + 64 + 8 */

        for (int ib32 = 0; ib32 < 8; ib32++) {
            float db0 = d * (0.5f + (float)(scales[ib32] & 0xF)) * 0.25f;
            float db1 = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;

            for (int l = 0; l < 4; l++) {
                float dl = (l < 2) ? db0 : db1;
                /* 10-bit grid index: low 8 from qs, high 2 from qh */
                int grid_idx = qs[l] | ((qh[ib32] << (8 - 2*l)) & 0x300);
                const uint8_t* grid = (const uint8_t*)(iq2s_grid + grid_idx);
                uint8_t sign = signs[l];

                for (int j = 0; j < 8; j++) {
                    dst[b * 256 + ib32 * 32 + l * 8 + j] =
                        dl * (float)grid[j] * ((sign & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                }
            }
            qs += 4;
            signs += 4;
        }
    }
}

/* ============================================================
 * IQ4_NL dequantization — 18 bytes per 32 elements (4.5 bpw)
 *
 * Non-linear 4-bit quantization using a 16-entry lookup table.
 * Block: d (fp16, 2 bytes) + qs[16] (4-bit packed pairs)
 * ============================================================ */

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

typedef struct {
    uint16_t d;       /* fp16 scale */
    uint8_t  qs[16];  /* 4-bit packed values, 2 per byte */
} block_iq4_nl;

static void dequant_iq4_nl(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_iq4_nl* blk = (const block_iq4_nl*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const uint8_t* qs = blk[b].qs;
        for (int j = 0; j < 16; ++j) {
            dst[b * 32 + j]      = d * kvalues_iq4nl[qs[j] & 0xf];
            dst[b * 32 + j + 16] = d * kvalues_iq4nl[qs[j] >> 4];
        }
    }
}

/* ============================================================
 * IQ4_XS dequantization — 136 bytes per 256 elements (4.25 bpw)
 *
 * Like IQ4_NL but with 256-element super-blocks and 6-bit sub-block scales.
 * Block: d (fp16, 2) + scales_h (uint16, 2) + scales_l[4] + qs[128]
 * ============================================================ */

typedef struct {
    uint16_t d;           /* fp16 super-block scale */
    uint16_t scales_h;    /* high 2 bits of 8 sub-block scales */
    uint8_t  scales_l[4]; /* low 4 bits of 8 sub-block scales, packed 2 per byte */
    uint8_t  qs[128];     /* 4-bit packed values */
} block_iq4_xs;

static void dequant_iq4_xs(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_iq4_xs* blk = (const block_iq4_xs*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const uint8_t* qs = blk[b].qs;
        float* y = dst + b * 256;

        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((blk[b].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf)
                         | (((blk[b].scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j +  0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

/* ============================================================
 * IQ3_XXS dequantization — 3.0625 bpw grid codebook
 *
 * Block: 98 bytes per 256 elements
 *   - d (fp16): super-block scale
 *   - qs[64]: grid indices (8 groups × 8 bytes each)
 *     First 64 bytes: 2 uint8 grid indices per sub-group (4 sub-groups × 2 = 8 per group)
 *     Next 32 bytes: scales_and_signs (4 bytes per group × 8 groups)
 *       Each uint32: bits 0-27 = 4×7-bit sign fields → ksigns_iq2xs
 *                    bits 28-31 = 4-bit sub-block scale
 *   Each grid index lookups iq3xxs_grid[idx] → 4 uint8 values (4 floats)
 *   2 grid indices per sub-group → 8 floats, 4 sub-groups per group → 32 floats
 * ============================================================ */

static const uint32_t iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

static void dequant_iq3_xxs(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)src;

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 98; /* 98 bytes per block */
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint8_t* qs = blk + 2;
        const uint8_t* scales_and_signs = qs + 64; /* QK_K/4 = 64 */
        float* y = dst + b * 256;

        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32;
            memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;

            for (int l = 0; l < 4; l++) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; j++) {
                    y[j + 0] = db * (float)grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                    y[j + 4] = db * (float)grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

/* --- Other IQ type stubs --- */
static void dequant_iq_stub(const char* type_name, float* dst, int n) {
    static int warned_iq2_xs = 0, warned_iq1_s = 0, warned_iq3_s = 0, warned_other = 0;
    int* flag = &warned_other;
    if (strcmp(type_name, "IQ2_XS") == 0) flag = &warned_iq2_xs;
    else if (strcmp(type_name, "IQ1_S") == 0) flag = &warned_iq1_s;
    else if (strcmp(type_name, "IQ3_S") == 0) flag = &warned_iq3_s;
    if (!*flag) {
        fprintf(stderr, "tq_gguf_quants: WARNING: %s dequant not yet implemented, "
                        "returning zeros\n", type_name);
        *flag = 1;
    }
    memset(dst, 0, (size_t)n * sizeof(float));
}

/* ============================================================
 * Main dequantization dispatcher
 * ============================================================ */

void tq_dequant_row_gguf(tq_ggml_dtype type, const void* src, float* dst, int n) {
    switch (type) {
        case TQ_GGML_TYPE_F32:
            dequant_f32(src, dst, n);
            break;
        case TQ_GGML_TYPE_F16:
            dequant_f16(src, dst, n);
            break;
        case TQ_GGML_TYPE_BF16:
            dequant_bf16(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q4_0:
            dequant_q4_0(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q4_1:
            dequant_q4_1(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q5_0:
            dequant_q5_0(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q5_1:
            dequant_q5_1(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q8_0:
            dequant_q8_0(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q8_1:
            dequant_q8_1(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q2_K:
            dequant_q2_k(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q3_K:
            dequant_q3_k(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q4_K:
            dequant_q4_k(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q5_K:
            dequant_q5_k(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q6_K:
            dequant_q6_k(src, dst, n);
            break;

        /* IQ stubs */
        case TQ_GGML_TYPE_IQ2_XXS:
            dequant_iq2_xxs(src, dst, n);
            break;
        case TQ_GGML_TYPE_IQ2_XS:
            dequant_iq_stub("IQ2_XS", dst, n);
            break;
        case TQ_GGML_TYPE_IQ3_XXS:
            dequant_iq3_xxs(src, dst, n);
            break;
        case TQ_GGML_TYPE_IQ1_S:
            dequant_iq_stub("IQ1_S", dst, n);
            break;
        case TQ_GGML_TYPE_IQ4_NL:
            dequant_iq4_nl(src, dst, n);
            break;
        case TQ_GGML_TYPE_IQ3_S:
            dequant_iq_stub("IQ3_S", dst, n);
            break;
        case TQ_GGML_TYPE_IQ2_S:
            dequant_iq2_s(src, dst, n);
            break;
        case TQ_GGML_TYPE_IQ4_XS:
            dequant_iq4_xs(src, dst, n);
            break;

        default:
            fprintf(stderr, "tq_gguf_quants: ERROR: unsupported type %d\n", (int)type);
            memset(dst, 0, (size_t)n * sizeof(float));
            break;
    }
}

/* ============================================================
 * Fused dequant-dot product functions
 *
 * These compute dot(dequant(weight_row), input) in a single pass
 * without writing dequantized values to memory. All intermediate
 * values stay in registers, eliminating the temporary FP32 buffer.
 *
 * This is the critical optimization for MoE inference where
 * IQ2_XXS dequant dominates runtime.
 * ============================================================ */

/* Fused IQ2_XXS dot product: dot(dequant(row), x) for one 256-element block */
static inline float dot_block_iq2_xxs(const uint8_t* blk, const float* x) {
    uint16_t d_raw;
    memcpy(&d_raw, blk, 2);
    const float d = fp16_to_fp32(d_raw);
    const uint16_t* qs = (const uint16_t*)(blk + 2);
    float sum = 0.0f;

    for (int ib32 = 0; ib32 < 8; ib32++) {
        uint32_t aux32[2];
        memcpy(aux32, qs + 4 * ib32, 8);
        const uint8_t* aux8 = (const uint8_t*)aux32;
        const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
        const float* xb = x + ib32 * 32;

        float group_sum = 0.0f;
        for (int l = 0; l < 4; l++) {
            const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
            const uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
            const float* xp = xb + l * 8;

#if TQ_HAS_NEON
            /* Load 8 grid values into two int8x8 vectors, apply signs, dot with input */
            /* Grid values are uint8_t (0x08, 0x19, 0x2b), signs are bitmask */
            float local_sum = 0.0f;
            for (int j = 0; j < 8; j++) {
                float w = (float)grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                local_sum += w * xp[j];
            }
            group_sum += local_sum;
#else
            float local_sum = 0.0f;
            for (int j = 0; j < 8; j++) {
                float w = (float)grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                local_sum += w * xp[j];
            }
            group_sum += local_sum;
#endif
        }
        sum += db * group_sum;
    }
    return sum;
}

/* Fused IQ2_XXS row dot: dot product of entire quantized row with input vector.
 * Processes all 256-element super-blocks without any intermediate FP32 buffer.
 * Reserved for future fused matmul optimization path. */
#ifdef _MSC_VER
__pragma(warning(suppress: 4505))
#else
__attribute__((unused))
#endif
static float fused_dot_iq2_xxs(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        sum += dot_block_iq2_xxs(base + b * 66, x + b * 256);
    }
    return sum;
}

#if TQ_HAS_NEON

/* Vectorized sign application helper: given 8 grid bytes and an 8-bit sign mask,
 * produce signed int8x8 where negative signs are applied.
 * Uses NEON bit test: broadcast sign byte, AND with bit masks, compare to produce
 * negation mask, then apply via (grid ^ neg) - neg (conditional negate). */
static const uint8_t iq2_sign_bit_masks[8] = {1, 2, 4, 8, 16, 32, 64, 128};

/* NEON-optimized fused IQ2_XXS dot product.
 * Optimizations over baseline:
 *   1. Vectorized sign expansion via NEON bit-test (replaces 8 scalar shifts)
 *   2. Apply signs in int8 domain before float conversion (fewer instructions)
 *   3. Fully unrolled inner loop (4 groups per ib32)
 *   4. Prefetch next block's weight data
 *   5. Two accumulator strategy to reduce FMA dependency chains */
static float fused_dot_iq2_xxs_neon(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)row;
    float32x4_t vtotal0 = vdupq_n_f32(0.0f);

    /* Preload sign bit masks into a NEON register */
    const uint8x8_t vbit_masks = vld1_u8(iq2_sign_bit_masks);

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 66;

        /* Prefetch next block */
        if (b + 1 < nb) {
            __builtin_prefetch(blk + 66, 0, 3);
            __builtin_prefetch(blk + 66 + 32, 0, 3);
        }

        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint8_t* qs_bytes = blk + 2;
        const float* xbase = x + b * 256;

        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32[2];
            memcpy(aux32, qs_bytes + 8 * ib32, 8);
            const uint8_t* aux8 = (const uint8_t*)aux32;
            const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
            const float* xb = xbase + ib32 * 32;

            /* Accumulate across all 4 sub-groups before scaling by db.
             * Use two accumulators to break FMA dependency chains. */
            float32x4_t vacc0 = vdupq_n_f32(0.0f);
            float32x4_t vacc1 = vdupq_n_f32(0.0f);

            /* --- Group 0 --- */
            {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[0]);
                const uint8_t signs = ksigns_iq2xs[aux32[1] & 127];

                uint8x8_t vgrid = vld1_u8(grid);
                /* Vectorized sign expansion:
                 * Broadcast sign byte to all lanes, AND with bit masks,
                 * compare != 0 produces 0xFF for negative lanes.
                 * Then: signed = (grid ^ neg_mask) - neg_mask
                 *       which is grid when neg_mask=0, -grid when neg_mask=0xFF */
                uint8x8_t vsign_bcast = vdup_n_u8(signs);
                uint8x8_t vsign_bits = vtst_u8(vsign_bcast, vbit_masks);
                /* vsign_bits is 0xFF where negative, 0x00 where positive */
                int8x8_t vgrid_s = vreinterpret_s8_u8(vgrid);
                int8x8_t vneg_mask = vreinterpret_s8_u8(vsign_bits);
                int8x8_t vsigned = vsub_s8(veor_s8(vgrid_s, vneg_mask), vneg_mask);

                /* Widen to int16, then int32, then float */
                int16x8_t vs16 = vmovl_s8(vsigned);
                float32x4_t vf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vs16)));
                float32x4_t vf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vs16)));

                vacc0 = vfmaq_f32(vacc0, vf_lo, vld1q_f32(xb));
                vacc1 = vfmaq_f32(vacc1, vf_hi, vld1q_f32(xb + 4));
            }

            /* --- Group 1 --- */
            {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[1]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7) & 127];

                uint8x8_t vgrid = vld1_u8(grid);
                uint8x8_t vsign_bcast = vdup_n_u8(signs);
                uint8x8_t vsign_bits = vtst_u8(vsign_bcast, vbit_masks);
                int8x8_t vgrid_s = vreinterpret_s8_u8(vgrid);
                int8x8_t vneg_mask = vreinterpret_s8_u8(vsign_bits);
                int8x8_t vsigned = vsub_s8(veor_s8(vgrid_s, vneg_mask), vneg_mask);

                int16x8_t vs16 = vmovl_s8(vsigned);
                float32x4_t vf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vs16)));
                float32x4_t vf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vs16)));

                vacc0 = vfmaq_f32(vacc0, vf_lo, vld1q_f32(xb + 8));
                vacc1 = vfmaq_f32(vacc1, vf_hi, vld1q_f32(xb + 12));
            }

            /* --- Group 2 --- */
            {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[2]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 14) & 127];

                uint8x8_t vgrid = vld1_u8(grid);
                uint8x8_t vsign_bcast = vdup_n_u8(signs);
                uint8x8_t vsign_bits = vtst_u8(vsign_bcast, vbit_masks);
                int8x8_t vgrid_s = vreinterpret_s8_u8(vgrid);
                int8x8_t vneg_mask = vreinterpret_s8_u8(vsign_bits);
                int8x8_t vsigned = vsub_s8(veor_s8(vgrid_s, vneg_mask), vneg_mask);

                int16x8_t vs16 = vmovl_s8(vsigned);
                float32x4_t vf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vs16)));
                float32x4_t vf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vs16)));

                vacc0 = vfmaq_f32(vacc0, vf_lo, vld1q_f32(xb + 16));
                vacc1 = vfmaq_f32(vacc1, vf_hi, vld1q_f32(xb + 20));
            }

            /* --- Group 3 --- */
            {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[3]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 21) & 127];

                uint8x8_t vgrid = vld1_u8(grid);
                uint8x8_t vsign_bcast = vdup_n_u8(signs);
                uint8x8_t vsign_bits = vtst_u8(vsign_bcast, vbit_masks);
                int8x8_t vgrid_s = vreinterpret_s8_u8(vgrid);
                int8x8_t vneg_mask = vreinterpret_s8_u8(vsign_bits);
                int8x8_t vsigned = vsub_s8(veor_s8(vgrid_s, vneg_mask), vneg_mask);

                int16x8_t vs16 = vmovl_s8(vsigned);
                float32x4_t vf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vs16)));
                float32x4_t vf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vs16)));

                vacc0 = vfmaq_f32(vacc0, vf_lo, vld1q_f32(xb + 24));
                vacc1 = vfmaq_f32(vacc1, vf_hi, vld1q_f32(xb + 28));
            }

            /* Combine accumulators, scale by db, accumulate to total */
            float32x4_t vgroup = vaddq_f32(vacc0, vacc1);
            vtotal0 = vfmaq_n_f32(vtotal0, vgroup, db);
        }
    }
    return vaddvq_f32(vtotal0);
}
#endif /* TQ_HAS_NEON */

/* Fused IQ2_S dot product */
static float fused_dot_iq2_s(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 82;
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);

        const uint8_t* qs_base = blk + 2;
        const uint8_t* signs_base = qs_base + 32;
        const uint8_t* qh = blk + 66;
        const uint8_t* scales = blk + 74;
        const float* xbase = x + b * 256;

        for (int ib32 = 0; ib32 < 8; ib32++) {
            const uint8_t* qs = qs_base + ib32 * 4;
            const uint8_t* sn = signs_base + ib32 * 4;
            float db0 = d * (0.5f + (float)(scales[ib32] & 0xF)) * 0.25f;
            float db1 = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;

            for (int l = 0; l < 4; l++) {
                float dl = (l < 2) ? db0 : db1;
                int grid_idx = qs[l] | ((qh[ib32] << (8 - 2*l)) & 0x300);
                const uint8_t* grid = (const uint8_t*)(iq2s_grid + grid_idx);
                uint8_t sign = sn[l];
                const float* xp = xbase + ib32 * 32 + l * 8;

                float local_sum = 0.0f;
                for (int j = 0; j < 8; j++) {
                    float w = (float)grid[j] * ((sign & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    local_sum += w * xp[j];
                }
                sum += dl * local_sum;
            }
        }
    }
    return sum;
}

/* Fused Q8_0 dot product: 34 bytes per 32 elements */
static float fused_dot_q8_0(const void* row, const float* x, int n) {
    const int nb = n / 32;
    const block_q8_0* blk = (const block_q8_0*)row;
    float sum = 0.0f;

#if TQ_HAS_NEON
    /* Two-accumulator NEON with prefetch for Q8_0.
     * Process 2 blocks per iteration to break FMA dependency chains. */
    float32x4_t vtotal0 = vdupq_n_f32(0.0f);
    float32x4_t vtotal1 = vdupq_n_f32(0.0f);

    int b = 0;
    for (; b + 1 < nb; b += 2) {
        /* Prefetch next pair */
        if (b + 3 < nb) __builtin_prefetch(&blk[b + 2], 0, 3);

        /* Block b */
        const float d0 = fp16_to_fp32(blk[b].d);
        const float* xp0 = x + b * 32;
        float32x4_t vs0 = vdupq_n_f32(0.0f);
        float32x4_t vs1 = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 8) {
            int8x8_t vq = vld1_s8(blk[b].qs + j);
            int16x8_t vq16 = vmovl_s8(vq);
            vs0 = vfmaq_f32(vs0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vq16))),  vld1q_f32(xp0 + j));
            vs1 = vfmaq_f32(vs1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vq16))), vld1q_f32(xp0 + j + 4));
        }
        vtotal0 = vfmaq_n_f32(vtotal0, vaddq_f32(vs0, vs1), d0);

        /* Block b+1 */
        const float d1 = fp16_to_fp32(blk[b + 1].d);
        const float* xp1 = x + (b + 1) * 32;
        float32x4_t vs2 = vdupq_n_f32(0.0f);
        float32x4_t vs3 = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 8) {
            int8x8_t vq = vld1_s8(blk[b + 1].qs + j);
            int16x8_t vq16 = vmovl_s8(vq);
            vs2 = vfmaq_f32(vs2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vq16))),  vld1q_f32(xp1 + j));
            vs3 = vfmaq_f32(vs3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vq16))), vld1q_f32(xp1 + j + 4));
        }
        vtotal1 = vfmaq_n_f32(vtotal1, vaddq_f32(vs2, vs3), d1);
    }
    /* Handle odd block */
    for (; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float* xp = x + b * 32;
        float32x4_t vs0 = vdupq_n_f32(0.0f), vs1 = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 8) {
            int8x8_t vq = vld1_s8(blk[b].qs + j);
            int16x8_t vq16 = vmovl_s8(vq);
            vs0 = vfmaq_f32(vs0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vq16))),  vld1q_f32(xp + j));
            vs1 = vfmaq_f32(vs1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vq16))), vld1q_f32(xp + j + 4));
        }
        vtotal0 = vfmaq_n_f32(vtotal0, vaddq_f32(vs0, vs1), d);
    }
    sum = vaddvq_f32(vaddq_f32(vtotal0, vtotal1));
#else
    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float* xp = x + b * 32;
        float block_sum = 0.0f;
        for (int j = 0; j < 32; j++) block_sum += (float)blk[b].qs[j] * xp[j];
        sum += d * block_sum;
    }
#endif
    return sum;
}

/* Fused Q8_1 dot product: 36 bytes per 32 elements.
 * Same math as Q8_0 (val = d * qs[i]) but different block layout.
 * The `s` (sum) field is only used for Q4×Q8_1 mixed-type dot products. */
static float fused_dot_q8_1(const void* row, const float* x, int n) {
    const int nb = n / 32;
    const block_q8_1* blk = (const block_q8_1*)row;
    float sum = 0.0f;

#if TQ_HAS_NEON
    float32x4_t vtotal0 = vdupq_n_f32(0.0f);
    float32x4_t vtotal1 = vdupq_n_f32(0.0f);

    int b = 0;
    for (; b + 1 < nb; b += 2) {
        if (b + 3 < nb) __builtin_prefetch(&blk[b + 2], 0, 3);

        const float d0 = fp16_to_fp32(blk[b].d);
        const float* xp0 = x + b * 32;
        float32x4_t vs0 = vdupq_n_f32(0.0f), vs1 = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 8) {
            int8x8_t vq = vld1_s8(blk[b].qs + j);
            int16x8_t vq16 = vmovl_s8(vq);
            vs0 = vfmaq_f32(vs0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vq16))),  vld1q_f32(xp0 + j));
            vs1 = vfmaq_f32(vs1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vq16))), vld1q_f32(xp0 + j + 4));
        }
        vtotal0 = vfmaq_n_f32(vtotal0, vaddq_f32(vs0, vs1), d0);

        const float d1 = fp16_to_fp32(blk[b + 1].d);
        const float* xp1 = x + (b + 1) * 32;
        float32x4_t vs2 = vdupq_n_f32(0.0f), vs3 = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 8) {
            int8x8_t vq = vld1_s8(blk[b + 1].qs + j);
            int16x8_t vq16 = vmovl_s8(vq);
            vs2 = vfmaq_f32(vs2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vq16))),  vld1q_f32(xp1 + j));
            vs3 = vfmaq_f32(vs3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vq16))), vld1q_f32(xp1 + j + 4));
        }
        vtotal1 = vfmaq_n_f32(vtotal1, vaddq_f32(vs2, vs3), d1);
    }
    for (; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float* xp = x + b * 32;
        float32x4_t vs0 = vdupq_n_f32(0.0f), vs1 = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 8) {
            int8x8_t vq = vld1_s8(blk[b].qs + j);
            int16x8_t vq16 = vmovl_s8(vq);
            vs0 = vfmaq_f32(vs0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vq16))),  vld1q_f32(xp + j));
            vs1 = vfmaq_f32(vs1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vq16))), vld1q_f32(xp + j + 4));
        }
        vtotal0 = vfmaq_n_f32(vtotal0, vaddq_f32(vs0, vs1), d);
    }
    sum = vaddvq_f32(vaddq_f32(vtotal0, vtotal1));
#else
    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float* xp = x + b * 32;
        float block_sum = 0.0f;
        for (int j = 0; j < 32; j++) block_sum += (float)blk[b].qs[j] * xp[j];
        sum += d * block_sum;
    }
#endif
    return sum;
}

/* Fused IQ4_NL dot product: 18 bytes per 32 elements */
static float fused_dot_iq4_nl(const void* row, const float* x, int n) {
    const int nb = n / 32;
    const block_iq4_nl* blk = (const block_iq4_nl*)row;
    float sum = 0.0f;

#if TQ_HAS_NEON
    /* Preload IQ4_NL lookup table into 2 NEON registers (16 values split into 2×8).
     * kvalues_iq4nl[0..15] are int8 values, we load as two int8x8 for tbl lookup. */
    int8x16_t vlut = vld1q_s8(kvalues_iq4nl);
    float32x4_t vsum0 = vdupq_n_f32(0.0f);
    uint8x16_t vmask_lo = vdupq_n_u8(0x0f);

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const uint8_t* qs = blk[b].qs;
        const float* xp = x + b * 32;

        /* Load 16 bytes of qs, split into low/high nibbles → lookup → signed int8 weights */
        uint8x16_t vqs = vld1q_u8(qs);
        uint8x16_t vlo = vandq_u8(vqs, vmask_lo);         /* low nibbles [0..15] */
        uint8x16_t vhi = vshrq_n_u8(vqs, 4);              /* high nibbles [0..15] */
        int8x16_t wlo = vqtbl1q_s8(vlut, vlo);            /* lookup low → signed weights */
        int8x16_t whi = vqtbl1q_s8(vlut, vhi);            /* lookup high → signed weights */

        /* Convert to float and accumulate: wlo[j]*xp[j] for j=0..15, whi[j]*xp[j+16] */
        float32x4_t vacc = vdupq_n_f32(0.0f);
        for (int k = 0; k < 4; k++) {
            /* Low nibble: 4 elements */
            int16x8_t w16 = vmovl_s8(vget_low_s8(wlo));
            if (k >= 2) w16 = vmovl_s8(vget_high_s8(wlo));
            int16x4_t w16_part = (k & 1) ? vget_high_s16(w16) : vget_low_s16(w16);
            float32x4_t vw = vcvtq_f32_s32(vmovl_s16(w16_part));
            float32x4_t vx = vld1q_f32(xp + k * 4);
            vacc = vfmaq_f32(vacc, vw, vx);

            /* High nibble: 4 elements at xp+16 */
            int16x8_t wh16 = vmovl_s8(vget_low_s8(whi));
            if (k >= 2) wh16 = vmovl_s8(vget_high_s8(whi));
            int16x4_t wh16_part = (k & 1) ? vget_high_s16(wh16) : vget_low_s16(wh16);
            float32x4_t vwh = vcvtq_f32_s32(vmovl_s16(wh16_part));
            float32x4_t vxh = vld1q_f32(xp + 16 + k * 4);
            vacc = vfmaq_f32(vacc, vwh, vxh);
        }
        float block_sum = vaddvq_f32(vacc);
        vsum0 = vfmaq_n_f32(vsum0, vdupq_n_f32(block_sum), d);
    }
    sum = vaddvq_f32(vsum0);
#else
    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const uint8_t* qs = blk[b].qs;
        const float* xp = x + b * 32;

        float block_sum = 0.0f;
        for (int j = 0; j < 16; j++) {
            block_sum += (float)kvalues_iq4nl[qs[j] & 0xf] * xp[j];
            block_sum += (float)kvalues_iq4nl[qs[j] >> 4]  * xp[j + 16];
        }
        sum += d * block_sum;
    }
#endif
    return sum;
}

/* Fused IQ4_XS dot product: 136 bytes per 256 elements */
static float fused_dot_iq4_xs(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const block_iq4_xs* blk = (const block_iq4_xs*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const uint8_t* qs = blk[b].qs;
        const float* xbase = x + b * 256;

        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((blk[b].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf)
                         | (((blk[b].scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            const float* xp = xbase + ib * 32;

            float block_sum = 0.0f;
            for (int j = 0; j < 16; j++) {
                block_sum += (float)kvalues_iq4nl[qs[j] & 0xf] * xp[j];
                block_sum += (float)kvalues_iq4nl[qs[j] >> 4]  * xp[j + 16];
            }
            sum += dl * block_sum;
            qs += 16;
        }
    }
    return sum;
}

/* Fused IQ3_XXS dot product: 98 bytes per 256 elements
 * Same layout as dequant_iq3_xxs but computes dot product without materializing FP32.
 * 8 groups of 32 elements per block. Each group: 4 sub-groups of 8 elements.
 * Grid lookup + sign application + dot in one pass. */
static float fused_dot_iq3_xxs(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)row;
    float sum = 0.0f;

#if TQ_HAS_NEON
    const uint8x8_t vbit_masks = vld1_u8(iq2_sign_bit_masks);

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 98;
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint8_t* qs = blk + 2;
        const uint8_t* scales_and_signs = qs + 64;
        const float* xbase = x + b * 256;

        float block_sum = 0.0f;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32;
            memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = (0.5f + (float)(aux32 >> 28)) * 0.5f;
            const float* xb = xbase + ib32 * 32;

            float32x4_t vacc0 = vdupq_n_f32(0.0f);
            float32x4_t vacc1 = vdupq_n_f32(0.0f);

            for (int l = 0; l < 4; l++) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 1]);

                /* Load 4+4 grid bytes, combine into one 8-byte vector */
                uint8_t grid8[8];
                memcpy(grid8, grid1, 4);
                memcpy(grid8 + 4, grid2, 4);
                uint8x8_t vgrid = vld1_u8(grid8);

                /* Vectorized sign: broadcast sign byte, AND with masks, compare */
                uint8x8_t vsign_bcast = vdup_n_u8(signs);
                uint8x8_t vsign_bits = vtst_u8(vsign_bcast, vbit_masks);
                int8x8_t vgrid_s = vreinterpret_s8_u8(vgrid);
                int8x8_t vneg = vreinterpret_s8_u8(vsign_bits);
                int8x8_t vsigned = vsub_s8(veor_s8(vgrid_s, vneg), vneg);

                /* Widen to float and dot with input */
                int16x8_t vs16 = vmovl_s8(vsigned);
                float32x4_t vf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vs16)));
                float32x4_t vf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vs16)));

                const float* xp = xb + l * 8;
                vacc0 = vfmaq_f32(vacc0, vf_lo, vld1q_f32(xp));
                vacc1 = vfmaq_f32(vacc1, vf_hi, vld1q_f32(xp + 4));
            }
            block_sum += db * vaddvq_f32(vaddq_f32(vacc0, vacc1));
            qs += 8;
        }
        sum += d * block_sum;
    }
#else
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 98;
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint8_t* qs = blk + 2;
        const uint8_t* scales_and_signs = qs + 64;
        const float* xbase = x + b * 256;

        float block_sum = 0.0f;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32;
            memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = (0.5f + (float)(aux32 >> 28)) * 0.5f;
            const float* xb = xbase + ib32 * 32;

            float group_sum = 0.0f;
            for (int l = 0; l < 4; l++) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 1]);
                const float* xp = xb + l * 8;
                for (int j = 0; j < 4; j++) {
                    float w1 = (float)grid1[j] * ((signs & kmask_iq2xs[j])     ? -1.0f : 1.0f);
                    float w2 = (float)grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                    group_sum += w1 * xp[j] + w2 * xp[j + 4];
                }
            }
            block_sum += db * group_sum;
            qs += 8;
        }
        sum += d * block_sum;
    }
#endif
    return sum;
}

/* Fused Q4_K dot product: 144 bytes per 256 elements
 * Layout: 4 groups of 64 elements, each group uses 32 bytes of qs.
 *   First 32 elements: d1 * (q[l] & 0xF) - m1  (low nibble, scale pair[0])
 *   Next 32 elements:  d2 * (q[l] >> 4)  - m2   (high nibble, scale pair[1])
 */
/* Fused Q2_K dot product: 84 bytes per 256 elements.
 * 2-bit values packed 4-per-byte in qs[64]. scales[16] holds:
 *   low 4 bits = sub-block scale (× d)
 *   high 4 bits = sub-block min (× dmin)
 * 16 sub-blocks of 16 elements each. Formula per sub-block:
 *   sum += dl * dot(q2_values, x) - ml * sum(x)
 * where q2_values are unsigned 2-bit values in [0, 3]. */
static float fused_dot_q2_k(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const block_q2_K* blk = (const block_q2_K*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const float d    = fp16_to_fp32(blk[b].d);
        const float dmin = fp16_to_fp32(blk[b].dmin);
        const uint8_t* q = blk[b].qs;
        const float* xp = x + b * 256;

        int is = 0;
        /* 2 halves, 4 shifts per half, 2 sub-blocks per shift */
        for (int half = 0; half < 2; half++) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                /* Sub-block 0: q[0..15] >> shift & 3 */
                uint8_t sc0 = blk[b].scales[is++];
                float dl0 = d * (sc0 & 0x0F);
                float ml0 = dmin * (sc0 >> 4);

                /* Sub-block 1: q[16..31] >> shift & 3 */
                uint8_t sc1 = blk[b].scales[is++];
                float dl1 = d * (sc1 & 0x0F);
                float ml1 = dmin * (sc1 >> 4);

#if TQ_HAS_NEON
                /* Load 16 packed bytes, extract 2-bit values, dot with x.
                 * Mask is uint8x16_t of 0x03; shift applied via vshrq_n_u8. */
                uint8x16_t qv0 = vld1q_u8(q);        /* sub-block 0 bytes */
                uint8x16_t qv1 = vld1q_u8(q + 16);   /* sub-block 1 bytes */
                uint8x16_t m03 = vdupq_n_u8(0x03);
                uint8x16_t v0, v1;
                switch (shift) {
                    case 0: v0 = vandq_u8(qv0, m03); v1 = vandq_u8(qv1, m03); break;
                    case 2: v0 = vandq_u8(vshrq_n_u8(qv0, 2), m03); v1 = vandq_u8(vshrq_n_u8(qv1, 2), m03); break;
                    case 4: v0 = vandq_u8(vshrq_n_u8(qv0, 4), m03); v1 = vandq_u8(vshrq_n_u8(qv1, 4), m03); break;
                    default: v0 = vshrq_n_u8(qv0, 6); v1 = vshrq_n_u8(qv1, 6); break;
                }
                /* Expand u8 → float32, accumulate dot and sum_x */
                #define TQ_Q2K_ACC(nv, off, dl_v, ml_v)                                        \
                    do {                                                                        \
                        uint16x8_t w16_l = vmovl_u8(vget_low_u8(nv));                           \
                        uint16x8_t w16_h = vmovl_u8(vget_high_u8(nv));                          \
                        float32x4_t wf0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_l)));        \
                        float32x4_t wf1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_l)));       \
                        float32x4_t wf2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_h)));        \
                        float32x4_t wf3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_h)));       \
                        float32x4_t x0 = vld1q_f32(xp + (off));                                 \
                        float32x4_t x1 = vld1q_f32(xp + (off) + 4);                             \
                        float32x4_t x2 = vld1q_f32(xp + (off) + 8);                             \
                        float32x4_t x3 = vld1q_f32(xp + (off) + 12);                            \
                        float32x4_t vd = vdupq_n_f32(0.0f);                                     \
                        vd = vfmaq_f32(vd, wf0, x0);                                            \
                        vd = vfmaq_f32(vd, wf1, x1);                                            \
                        vd = vfmaq_f32(vd, wf2, x2);                                            \
                        vd = vfmaq_f32(vd, wf3, x3);                                            \
                        float dot_s = vaddvq_f32(vd);                                           \
                        float sum_s = vaddvq_f32(vaddq_f32(vaddq_f32(x0, x1), vaddq_f32(x2, x3))); \
                        sum += (dl_v) * dot_s - (ml_v) * sum_s;                                 \
                    } while (0)

                int yi = half * 128 + j * 32;
                TQ_Q2K_ACC(v0, yi, dl0, ml0);
                TQ_Q2K_ACC(v1, yi + 16, dl1, ml1);
                #undef TQ_Q2K_ACC
#else
                int yi = half * 128 + j * 32;
                float dot0 = 0, sumx0 = 0;
                for (int l = 0; l < 16; l++) {
                    float xv = xp[yi + l];
                    dot0 += (float)((q[l] >> shift) & 3) * xv;
                    sumx0 += xv;
                }
                sum += dl0 * dot0 - ml0 * sumx0;

                float dot1 = 0, sumx1 = 0;
                for (int l = 0; l < 16; l++) {
                    float xv = xp[yi + 16 + l];
                    dot1 += (float)((q[l + 16] >> shift) & 3) * xv;
                    sumx1 += xv;
                }
                sum += dl1 * dot1 - ml1 * sumx1;
#endif
                shift += 2;
            }
            q += 32;
        }
    }
    return sum;
}

static float fused_dot_q4_k(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const block_q4_K* blk = (const block_q4_K*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const float d    = fp16_to_fp32(blk[b].d);
        const float dmin = fp16_to_fp32(blk[b].dmin);

        uint8_t sc[8], mn[8];
        sc[0] = blk[b].scales[0] & 63;
        sc[1] = blk[b].scales[1] & 63;
        sc[2] = blk[b].scales[2] & 63;
        sc[3] = blk[b].scales[3] & 63;
        mn[0] = blk[b].scales[4] & 63;
        mn[1] = blk[b].scales[5] & 63;
        mn[2] = blk[b].scales[6] & 63;
        mn[3] = blk[b].scales[7] & 63;
        sc[4] = (blk[b].scales[8] & 0x0F) | ((blk[b].scales[0] >> 6) << 4);
        sc[5] = (blk[b].scales[9] & 0x0F) | ((blk[b].scales[1] >> 6) << 4);
        sc[6] = (blk[b].scales[10] & 0x0F) | ((blk[b].scales[2] >> 6) << 4);
        sc[7] = (blk[b].scales[11] & 0x0F) | ((blk[b].scales[3] >> 6) << 4);
        mn[4] = (blk[b].scales[8] >> 4) | ((blk[b].scales[4] >> 6) << 4);
        mn[5] = (blk[b].scales[9] >> 4) | ((blk[b].scales[5] >> 6) << 4);
        mn[6] = (blk[b].scales[10] >> 4) | ((blk[b].scales[6] >> 6) << 4);
        mn[7] = (blk[b].scales[11] >> 4) | ((blk[b].scales[7] >> 6) << 4);

        const uint8_t* q = blk[b].qs;
        const float* xp = x + b * 256;
        int is = 0;

#if TQ_HAS_NEON
        /* 4 groups of 64 elements, NEON-accelerated */
        const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
        for (int j = 0; j < 256; j += 64) {
            const float d1 = d * sc[is + 0];
            const float m1 = dmin * mn[is + 0];
            const float d2 = d * sc[is + 1];
            const float m2 = dmin * mn[is + 1];

            /* Load 32 bytes = 32 pairs of nibbles covering 64 elements */
            uint8x16_t qa = vld1q_u8(q);
            uint8x16_t qb = vld1q_u8(q + 16);
            /* Extract low nibbles (→ elements 0..31) and high nibbles (→ 32..63) */
            uint8x16_t lo_a = vandq_u8(qa, mask_lo);
            uint8x16_t lo_b = vandq_u8(qb, mask_lo);
            uint8x16_t hi_a = vshrq_n_u8(qa, 4);
            uint8x16_t hi_b = vshrq_n_u8(qb, 4);

            /* Convert u8 nibbles [0..15] to float32 and dot with xp[...] */
            float32x4_t vdot1 = vdupq_n_f32(0.0f);
            float32x4_t vsum1 = vdupq_n_f32(0.0f);
            float32x4_t vdot2 = vdupq_n_f32(0.0f);
            float32x4_t vsum2 = vdupq_n_f32(0.0f);
            /* Helper: process 16 elements of x[off:off+16] with nibble vector `nv` */
            #define TQ_Q4K_ACC(nv, off, vdot, vsum)                                              \
                do {                                                                             \
                    uint16x8_t w16_l = vmovl_u8(vget_low_u8(nv));                                \
                    uint16x8_t w16_h = vmovl_u8(vget_high_u8(nv));                               \
                    float32x4_t wf0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_l)));             \
                    float32x4_t wf1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_l)));            \
                    float32x4_t wf2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_h)));             \
                    float32x4_t wf3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_h)));            \
                    float32x4_t x0 = vld1q_f32(xp + (off));                                      \
                    float32x4_t x1 = vld1q_f32(xp + (off) + 4);                                  \
                    float32x4_t x2 = vld1q_f32(xp + (off) + 8);                                  \
                    float32x4_t x3 = vld1q_f32(xp + (off) + 12);                                 \
                    (vdot) = vfmaq_f32((vdot), wf0, x0);                                         \
                    (vdot) = vfmaq_f32((vdot), wf1, x1);                                         \
                    (vdot) = vfmaq_f32((vdot), wf2, x2);                                         \
                    (vdot) = vfmaq_f32((vdot), wf3, x3);                                         \
                    (vsum) = vaddq_f32((vsum), vaddq_f32(vaddq_f32(x0, x1), vaddq_f32(x2, x3))); \
                } while (0)
            TQ_Q4K_ACC(lo_a, j +  0, vdot1, vsum1);
            TQ_Q4K_ACC(lo_b, j + 16, vdot1, vsum1);
            TQ_Q4K_ACC(hi_a, j + 32, vdot2, vsum2);
            TQ_Q4K_ACC(hi_b, j + 48, vdot2, vsum2);
            #undef TQ_Q4K_ACC

            float dot1_s = vaddvq_f32(vdot1);
            float sum1_s = vaddvq_f32(vsum1);
            float dot2_s = vaddvq_f32(vdot2);
            float sum2_s = vaddvq_f32(vsum2);
            sum += d1 * dot1_s - m1 * sum1_s;
            sum += d2 * dot2_s - m2 * sum2_s;

            q += 32;
            is += 2;
        }
#else
        /* Scalar fallback (non-ARM) */
        for (int j = 0; j < 256; j += 64) {
            const float d1 = d * sc[is + 0];
            const float m1 = dmin * mn[is + 0];
            const float d2 = d * sc[is + 1];
            const float m2 = dmin * mn[is + 1];

            float dot1 = 0.0f, sum_x1 = 0.0f;
            for (int l = 0; l < 32; l++) {
                dot1  += (float)(q[l] & 0x0F) * xp[j + l];
                sum_x1 += xp[j + l];
            }
            sum += d1 * dot1 - m1 * sum_x1;

            float dot2 = 0.0f, sum_x2 = 0.0f;
            for (int l = 0; l < 32; l++) {
                dot2  += (float)(q[l] >> 4) * xp[j + 32 + l];
                sum_x2 += xp[j + 32 + l];
            }
            sum += d2 * dot2 - m2 * sum_x2;

            q += 32;
            is += 2;
        }
#endif
    }
    return sum;
}

/* Fused Q3_K dot product: 110 bytes per 256 elements
 * 3-bit = 2 low bits (qs) + 1 high bit (hmask)
 * 16 sub-blocks with 6-bit scales packed into 12 bytes */
static float fused_dot_q3_k(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const block_q3_K* blk = (const block_q3_K*)row;
    float sum = 0.0f;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t* scales = (const int8_t*)aux;

    for (int b = 0; b < nb; b++) {
        const float d_all = fp16_to_fp32(blk[b].d);

        const uint8_t* q  = blk[b].qs;
        const uint8_t* hm = blk[b].hmask;
        uint8_t m = 1;

        /* Decode 16 x 6-bit scales (same as dequant_q3_k) */
        memcpy(aux, blk[b].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        const float* xp = x + b * 256;
        int yi = 0;

        for (int half = 0; half < 2; half++) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                float dot = 0.0f;
                for (int l = 0; l < 16; ++l) {
                    dot += xp[yi + l] * (float)((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                }
                sum += dl * dot;
                yi += 16;

                dl = d_all * (scales[is++] - 32);
                dot = 0.0f;
                for (int l = 0; l < 16; ++l) {
                    dot += xp[yi + l] * (float)((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                }
                sum += dl * dot;
                yi += 16;

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
    return sum;
}

/* Fused Q4_0 dot product: 18 bytes per 32 elements */
static float fused_dot_q4_0(const void* row, const float* x, int n) {
    const int nb = n / 32;
    const block_q4_0* blk = (const block_q4_0*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float* xp = x + b * 32;

        float block_sum = 0.0f;
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            block_sum += (float)((int)(byte & 0x0F) - 8) * xp[j];
            block_sum += (float)((int)(byte >> 4) - 8) * xp[j + 16];
        }
        sum += d * block_sum;
    }
    return sum;
}

/* Fused Q6_K dot product: 210 bytes per 256 elements
 * Matches ggml dequantize_row_q6_K layout exactly:
 * Two 128-element halves, each with 32 iterations producing 4 elements. */
static float fused_dot_q6_k(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const block_q6_K* blk = (const block_q6_K*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float* xbase = x + b * 256;
        const uint8_t* ql = blk[b].ql;
        const uint8_t* qh = blk[b].qh;
        const int8_t* sc = blk[b].scales;

        for (int half = 0; half < 2; half++) {
            const float* xp = xbase + half * 128;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                sum += d * sc[is + 0] * q1 * xp[l +  0];
                sum += d * sc[is + 2] * q2 * xp[l + 32];
                sum += d * sc[is + 4] * q3 * xp[l + 64];
                sum += d * sc[is + 6] * q4 * xp[l + 96];
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return sum;
}

/* ============================================================
 * On-the-fly dequant matmul (with fused fast paths)
 *
 * out[d] = sum_n( x[n] * dequant(W[d, n]) )
 *
 * W is stored row-major in quantized blocks.
 * Hot path for MoE expert computation.
 *
 * For supported types (IQ2_XXS, IQ2_S, Q8_0, Q4_K, Q4_0, Q6_K,
 * IQ4_NL, IQ4_XS), we use fused dequant-dot that avoids writing
 * intermediate FP32 values to memory. This eliminates ~3 GB/token
 * of temporary memory traffic for IQ2_XXS MoE models.
 * ============================================================ */

/* ============================================================
 * Multi-threaded GGUF matmul worker
 * ============================================================ */
typedef struct {
    float*       out;
    const float* x;
    const void*  weight;
    float (*fused_dot)(const void*, const float*, int);
    tq_ggml_dtype weight_type;
    size_t       row_bytes;
    int          in_dim;
    int          block_bytes;
    int          block_elems;
    int          n_blocks;
    int          start_row;
    int          end_row;
} gguf_matmul_task_t;

static void* gguf_matmul_worker(void* arg) {
    gguf_matmul_task_t* t = (gguf_matmul_task_t*)arg;

    if (t->fused_dot) {
        for (int d = t->start_row; d < t->end_row; d++) {
            const uint8_t* row = (const uint8_t*)t->weight + (size_t)d * t->row_bytes;
            t->out[d] = t->fused_dot(row, t->x, t->in_dim);
        }
        return NULL;
    }

    /* Generic fallback: dequant block -> tmp -> dot */
    for (int d = t->start_row; d < t->end_row; d++) {
        const uint8_t* row = (const uint8_t*)t->weight + (size_t)d * t->row_bytes;
        float sum = 0.0f;
        float tmp[256]; /* max block size is 256 */

        for (int b = 0; b < t->n_blocks; b++) {
            tq_dequant_row_gguf(t->weight_type,
                                row + (size_t)b * t->block_bytes,
                                tmp, t->block_elems);

            const float* xp = t->x + b * t->block_elems;

#if TQ_HAS_NEON
            float32x4_t vsum0 = vdupq_n_f32(0.0f);
            float32x4_t vsum1 = vdupq_n_f32(0.0f);
            float32x4_t vsum2 = vdupq_n_f32(0.0f);
            float32x4_t vsum3 = vdupq_n_f32(0.0f);

            int j = 0;
            for (; j + 15 < t->block_elems; j += 16) {
                float32x4_t vx0 = vld1q_f32(xp + j);
                float32x4_t vx1 = vld1q_f32(xp + j + 4);
                float32x4_t vx2 = vld1q_f32(xp + j + 8);
                float32x4_t vx3 = vld1q_f32(xp + j + 12);
                float32x4_t vt0 = vld1q_f32(tmp + j);
                float32x4_t vt1 = vld1q_f32(tmp + j + 4);
                float32x4_t vt2 = vld1q_f32(tmp + j + 8);
                float32x4_t vt3 = vld1q_f32(tmp + j + 12);
                vsum0 = vfmaq_f32(vsum0, vx0, vt0);
                vsum1 = vfmaq_f32(vsum1, vx1, vt1);
                vsum2 = vfmaq_f32(vsum2, vx2, vt2);
                vsum3 = vfmaq_f32(vsum3, vx3, vt3);
            }
            for (; j + 3 < t->block_elems; j += 4) {
                float32x4_t vx0 = vld1q_f32(xp + j);
                float32x4_t vt0 = vld1q_f32(tmp + j);
                vsum0 = vfmaq_f32(vsum0, vx0, vt0);
            }

            vsum0 = vaddq_f32(vsum0, vsum1);
            vsum2 = vaddq_f32(vsum2, vsum3);
            vsum0 = vaddq_f32(vsum0, vsum2);
            sum += vaddvq_f32(vsum0);

            for (; j < t->block_elems; j++) {
                sum += xp[j] * tmp[j];
            }
#else
            for (int j = 0; j < t->block_elems; j++) {
                sum += xp[j] * tmp[j];
            }
#endif
        }

        t->out[d] = sum;
    }
    return NULL;
}

/* Pre-quantize input vector to Q8 format for int8×int8 matmul.
 * Called once in transformer, result reused for Q/K/V/O projections.
 * Stores int8 values in qs[n], per-block scales in ds[n/32]. */
void tq_preq_input_q8(const float* x, int8_t* qs, float* ds, int n) {
    int nb = n / 32;
    for (int b = 0; b < nb; b++) {
        const float* xp = x + b * 32;
        float amax = 0.0f;
#if TQ_HAS_NEON
        float32x4_t vmax = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 4) {
            float32x4_t vx = vld1q_f32(xp + j);
            vmax = vmaxq_f32(vmax, vabsq_f32(vx));
        }
        amax = vmaxvq_f32(vmax);
#else
        for (int j = 0; j < 32; j++) {
            float a = xp[j] < 0 ? -xp[j] : xp[j];
            if (a > amax) amax = a;
        }
#endif
        float d = amax / 127.0f;
        ds[b] = d;
        if (d > 0.0f) {
            float id = 127.0f / amax;
#if TQ_HAS_NEON
            float32x4_t vid = vdupq_n_f32(id);
            for (int j = 0; j < 32; j += 8) {
                float32x4_t v0 = vmulq_f32(vld1q_f32(xp + j), vid);
                float32x4_t v1 = vmulq_f32(vld1q_f32(xp + j + 4), vid);
                int32x4_t i0 = vcvtnq_s32_f32(v0);
                int32x4_t i1 = vcvtnq_s32_f32(v1);
                int16x4_t s0 = vqmovn_s32(i0);
                int16x4_t s1 = vqmovn_s32(i1);
                int8x8_t b8 = vqmovn_s16(vcombine_s16(s0, s1));
                vst1_s8(qs + b * 32 + j, b8);
            }
#else
            for (int j = 0; j < 32; j++) {
                int v = (int)roundf(xp[j] * id);
                qs[b * 32 + j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
            }
#endif
        } else {
            memset(qs + b * 32, 0, 32);
        }
    }
}

/* Thread-local pre-quantized input pointer (set by tq_matmul_gguf when available) */
#ifdef _MSC_VER
static __declspec(thread) const int8_t* g_preq_qs = NULL;
static __declspec(thread) const float*  g_preq_ds = NULL;
#else
static __thread const int8_t* g_preq_qs = NULL;
static __thread const float*  g_preq_ds = NULL;
#endif

void tq_set_preq(const int8_t* qs, const float* ds) {
    g_preq_qs = qs;
    g_preq_ds = ds;
}
void tq_clear_preq(void) {
    g_preq_qs = NULL;
    g_preq_ds = NULL;
}

/* Q8×Q8 integer dot worker — processes a range of output rows using int8 multiply-accumulate */
#if TQ_HAS_NEON
typedef struct {
    float* out; const void* weight; const int8_t* x_qs; const float* x_ds;
    size_t row_bytes; int n_blocks; int start_row; int end_row;
} q8_int_task_t;

void* q8_int_dot_worker(void* arg) {
    q8_int_task_t* t = (q8_int_task_t*)arg;
    for (int d = t->start_row; d < t->end_row; d++) {
        const block_q8_0* wblk = (const block_q8_0*)((const uint8_t*)t->weight + (size_t)d * t->row_bytes);
        if (d + 1 < t->end_row) {
            const uint8_t* next = (const uint8_t*)t->weight + (size_t)(d + 1) * t->row_bytes;
            /* Q8_0 row is n_blocks * 34 bytes — for hidden=3072, ~3.4 KB.
             * Prefetch first 4 cache lines; HW prefetcher takes the rest. */
            __builtin_prefetch(next +   0, 0, 0);
            __builtin_prefetch(next +  64, 0, 0);
            __builtin_prefetch(next + 128, 0, 0);
            __builtin_prefetch(next + 192, 0, 0);
        }
        float row_sum = 0.0f;
        for (int b = 0; b < t->n_blocks; b++) {
            const float wd = fp16_to_fp32(wblk[b].d);
            const int8_t* wqs = wblk[b].qs;
            const int8_t* xqs = t->x_qs + b * 32;
#ifdef __ARM_FEATURE_DOTPROD
            /* ARMv8.2 dotprod: 16 int8 MACs per instruction (M1+ have this). */
            int32x4_t vd = vdupq_n_s32(0);
            vd = vdotq_s32(vd, vld1q_s8(wqs +  0), vld1q_s8(xqs +  0));
            vd = vdotq_s32(vd, vld1q_s8(wqs + 16), vld1q_s8(xqs + 16));
            row_sum += wd * t->x_ds[b] * (float)vaddvq_s32(vd);
#else
            int32x4_t vd0 = vdupq_n_s32(0), vd1 = vdupq_n_s32(0);
            for (int j = 0; j < 32; j += 16) {
                int8x16_t vw = vld1q_s8(wqs + j);
                int8x16_t vx = vld1q_s8(xqs + j);
                vd0 = vpadalq_s16(vd0, vmull_s8(vget_low_s8(vw), vget_low_s8(vx)));
                vd1 = vpadalq_s16(vd1, vmull_s8(vget_high_s8(vw), vget_high_s8(vx)));
            }
            row_sum += wd * t->x_ds[b] * (float)vaddvq_s32(vaddq_s32(vd0, vd1));
#endif
        }
        t->out[d] = row_sum;
    }
    return NULL;
}

/* Q4_K int8 dot worker — same idea as q8_int but with on-the-fly nibble unpack.
 * Pre-quantized x: int8 array (x_qs), per-32-element scales (x_ds), and
 * pre-summed int sums per 32-element block (x_isums) so the dmin*mn correction
 * doesn't recompute sum(x_int8) per output row. */
typedef struct {
    float* out; const void* weight; const int8_t* x_qs; const float* x_ds;
    const int32_t* x_isums; size_t row_bytes; int nb_super; int start_row; int end_row;
} q4k_int_task_t;

/* Q5_K int8 dot worker — same pattern as Q4_K, plus 5th bit from qh.
 * The qh array has 1 bit per element across 8 sub-blocks; bit position
 * shifts by 2 per j-iteration (u1 = 1<<(2*iter), u2 = 2<<(2*iter)). */
typedef q4k_int_task_t q5k_int_task_t;

void* q5k_int_dot_worker(void* arg) {
    q5k_int_task_t* t = (q5k_int_task_t*)arg;
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    for (int d = t->start_row; d < t->end_row; d++) {
        const block_q5_K* wblk = (const block_q5_K*)((const uint8_t*)t->weight + (size_t)d * t->row_bytes);
        if (d + 1 < t->end_row) {
            const uint8_t* next = (const uint8_t*)t->weight + (size_t)(d + 1) * t->row_bytes;
            __builtin_prefetch(next +   0, 0, 0);
            __builtin_prefetch(next +  64, 0, 0);
            __builtin_prefetch(next + 128, 0, 0);
            __builtin_prefetch(next + 192, 0, 0);
        }
        float row_sum = 0.0f;
        for (int sb = 0; sb < t->nb_super; sb++) {
            const block_q5_K* blk = wblk + sb;
            const float dW    = fp16_to_fp32(blk->d);
            const float dminW = fp16_to_fp32(blk->dmin);

            uint8_t sc[8], mn[8];
            sc[0] = blk->scales[0] & 63;
            sc[1] = blk->scales[1] & 63;
            sc[2] = blk->scales[2] & 63;
            sc[3] = blk->scales[3] & 63;
            mn[0] = blk->scales[4] & 63;
            mn[1] = blk->scales[5] & 63;
            mn[2] = blk->scales[6] & 63;
            mn[3] = blk->scales[7] & 63;
            sc[4] = (blk->scales[8]  & 0x0F) | ((blk->scales[0] >> 6) << 4);
            sc[5] = (blk->scales[9]  & 0x0F) | ((blk->scales[1] >> 6) << 4);
            sc[6] = (blk->scales[10] & 0x0F) | ((blk->scales[2] >> 6) << 4);
            sc[7] = (blk->scales[11] & 0x0F) | ((blk->scales[3] >> 6) << 4);
            mn[4] = (blk->scales[8]  >> 4) | ((blk->scales[4] >> 6) << 4);
            mn[5] = (blk->scales[9]  >> 4) | ((blk->scales[5] >> 6) << 4);
            mn[6] = (blk->scales[10] >> 4) | ((blk->scales[6] >> 6) << 4);
            mn[7] = (blk->scales[11] >> 4) | ((blk->scales[7] >> 6) << 4);

            const uint8_t* ql = blk->qs;   /* 128 bytes of low nibbles */
            const uint8_t* qh = blk->qh;   /* 32 bytes of high bits */
            int sub_base = sb * 8;
            int is = 0;
            uint8_t u1 = 1, u2 = 2;

            for (int j = 0; j < 256; j += 64) {
                int sub_idx_a = sub_base + is;
                int sub_idx_b = sub_base + is + 1;

                /* Low 4 bits of weights */
                uint8x16_t qa = vld1q_u8(ql);
                uint8x16_t qb = vld1q_u8(ql + 16);
                uint8x16_t lo_a = vandq_u8(qa, mask_lo);
                uint8x16_t lo_b = vandq_u8(qb, mask_lo);
                uint8x16_t hi_a = vshrq_n_u8(qa, 4);
                uint8x16_t hi_b = vshrq_n_u8(qb, 4);

                /* 5th bit from qh: extract bit u1 (sub-block A) and u2 (B).
                 * qh[0..15] covers elements 0..15, qh[16..31] covers 16..31.
                 * For sub-block A (low nibbles), bit position = log2(u1).
                 * Convert "bit set" → byte value 16 by shifting to bit 4. */
                uint8x16_t qh_a = vld1q_u8(qh);
                uint8x16_t qh_b = vld1q_u8(qh + 16);
                uint8x16_t u1v = vdupq_n_u8(u1);
                uint8x16_t u2v = vdupq_n_u8(u2);
                /* Test bit, then convert to 0 or 16:
                 *   masked_a = qh & u1  →  0 or u1 (in {1,4,16,64})
                 *   want bit 4 set when u1 bit set → multiply masked_a by (16/u1)
                 * For u1 in {1,4,16,64}: 16/u1 in {16,4,1,1/4}.
                 * Easier: vceqq + select between 0 and 16. */
                uint8x16_t bit_a_lo = vceqq_u8(vandq_u8(qh_a, u1v), u1v);  /* 0xFF or 0x00 */
                uint8x16_t bit_a_hi = vceqq_u8(vandq_u8(qh_b, u1v), u1v);
                uint8x16_t bit_b_lo = vceqq_u8(vandq_u8(qh_a, u2v), u2v);
                uint8x16_t bit_b_hi = vceqq_u8(vandq_u8(qh_b, u2v), u2v);
                uint8x16_t v16 = vdupq_n_u8(16);
                /* And with 16 to get 0 or 16 */
                lo_a = vorrq_u8(lo_a, vandq_u8(bit_a_lo, v16));
                lo_b = vorrq_u8(lo_b, vandq_u8(bit_a_hi, v16));
                hi_a = vorrq_u8(hi_a, vandq_u8(bit_b_lo, v16));
                hi_b = vorrq_u8(hi_b, vandq_u8(bit_b_hi, v16));

                int8x16_t wa_lo = vreinterpretq_s8_u8(lo_a);
                int8x16_t wa_hi = vreinterpretq_s8_u8(lo_b);
                int8x16_t wb_lo = vreinterpretq_s8_u8(hi_a);
                int8x16_t wb_hi = vreinterpretq_s8_u8(hi_b);

                const int8_t* xa = t->x_qs + (size_t)sub_idx_a * 32;
                int8x16_t xa_lo = vld1q_s8(xa);
                int8x16_t xa_hi = vld1q_s8(xa + 16);
                const int8_t* xb = t->x_qs + (size_t)sub_idx_b * 32;
                int8x16_t xb_lo = vld1q_s8(xb);
                int8x16_t xb_hi = vld1q_s8(xb + 16);

#ifdef __ARM_FEATURE_DOTPROD
                int32x4_t accA = vdotq_s32(vdupq_n_s32(0), wa_lo, xa_lo);
                accA = vdotq_s32(accA, wa_hi, xa_hi);
                int32_t isumA = vaddvq_s32(accA);
                int32x4_t accB = vdotq_s32(vdupq_n_s32(0), wb_lo, xb_lo);
                accB = vdotq_s32(accB, wb_hi, xb_hi);
                int32_t isumB = vaddvq_s32(accB);
#else
                int32x4_t accA = vpadalq_s16(vdupq_n_s32(0),
                                              vmull_s8(vget_low_s8(wa_lo), vget_low_s8(xa_lo)));
                accA = vpadalq_s16(accA, vmull_s8(vget_high_s8(wa_lo), vget_high_s8(xa_lo)));
                accA = vpadalq_s16(accA, vmull_s8(vget_low_s8(wa_hi), vget_low_s8(xa_hi)));
                accA = vpadalq_s16(accA, vmull_s8(vget_high_s8(wa_hi), vget_high_s8(xa_hi)));
                int32_t isumA = vaddvq_s32(accA);
                int32x4_t accB = vpadalq_s16(vdupq_n_s32(0),
                                              vmull_s8(vget_low_s8(wb_lo), vget_low_s8(xb_lo)));
                accB = vpadalq_s16(accB, vmull_s8(vget_high_s8(wb_lo), vget_high_s8(xb_lo)));
                accB = vpadalq_s16(accB, vmull_s8(vget_low_s8(wb_hi), vget_low_s8(xb_hi)));
                accB = vpadalq_s16(accB, vmull_s8(vget_high_s8(wb_hi), vget_high_s8(xb_hi)));
                int32_t isumB = vaddvq_s32(accB);
#endif

                float xdA = t->x_ds[sub_idx_a];
                float xdB = t->x_ds[sub_idx_b];
                int32_t xisA = t->x_isums[sub_idx_a];
                int32_t xisB = t->x_isums[sub_idx_b];

                row_sum += (dW * sc[is + 0] * xdA) * (float)isumA
                         - (dminW * mn[is + 0] * xdA) * (float)xisA;
                row_sum += (dW * sc[is + 1] * xdB) * (float)isumB
                         - (dminW * mn[is + 1] * xdB) * (float)xisB;

                ql += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        t->out[d] = row_sum;
    }
    return NULL;
}

void* q4k_int_dot_worker(void* arg) {
    q4k_int_task_t* t = (q4k_int_task_t*)arg;
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    for (int d = t->start_row; d < t->end_row; d++) {
        const block_q4_K* wblk = (const block_q4_K*)((const uint8_t*)t->weight + (size_t)d * t->row_bytes);
        /* Prefetch the next row's weights while we're processing this one.
         * Each row is nb_super * 144 bytes; for typical Phi-3.5 (in_dim=3072,
         * 12 super-blocks) that's 1728 bytes. Prefetch 4 cache lines ahead. */
        if (d + 1 < t->end_row) {
            const uint8_t* next = (const uint8_t*)t->weight + (size_t)(d + 1) * t->row_bytes;
            __builtin_prefetch(next +   0, 0, 0);
            __builtin_prefetch(next +  64, 0, 0);
            __builtin_prefetch(next + 128, 0, 0);
            __builtin_prefetch(next + 192, 0, 0);
        }
        float row_sum = 0.0f;
        for (int sb = 0; sb < t->nb_super; sb++) {
            const block_q4_K* blk = wblk + sb;
            const float dW    = fp16_to_fp32(blk->d);
            const float dminW = fp16_to_fp32(blk->dmin);

            /* 6-bit packed sub-block scales (sc) and mins (mn).
             * Layout matches fused_dot_q4_k. */
            uint8_t sc[8], mn[8];
            sc[0] = blk->scales[0] & 63;
            sc[1] = blk->scales[1] & 63;
            sc[2] = blk->scales[2] & 63;
            sc[3] = blk->scales[3] & 63;
            mn[0] = blk->scales[4] & 63;
            mn[1] = blk->scales[5] & 63;
            mn[2] = blk->scales[6] & 63;
            mn[3] = blk->scales[7] & 63;
            sc[4] = (blk->scales[8]  & 0x0F) | ((blk->scales[0] >> 6) << 4);
            sc[5] = (blk->scales[9]  & 0x0F) | ((blk->scales[1] >> 6) << 4);
            sc[6] = (blk->scales[10] & 0x0F) | ((blk->scales[2] >> 6) << 4);
            sc[7] = (blk->scales[11] & 0x0F) | ((blk->scales[3] >> 6) << 4);
            mn[4] = (blk->scales[8]  >> 4) | ((blk->scales[4] >> 6) << 4);
            mn[5] = (blk->scales[9]  >> 4) | ((blk->scales[5] >> 6) << 4);
            mn[6] = (blk->scales[10] >> 4) | ((blk->scales[6] >> 6) << 4);
            mn[7] = (blk->scales[11] >> 4) | ((blk->scales[7] >> 6) << 4);

            const uint8_t* q = blk->qs;
            int sub_base = sb * 8;
            int is = 0;

            /* 4 j-iterations × 64 elements = 256-element super-block.
             * Each iteration handles two 32-element sub-blocks (lo+hi nibbles). */
            for (int j = 0; j < 256; j += 64) {
                int sub_idx_a = sub_base + is;     /* offset j..j+31  */
                int sub_idx_b = sub_base + is + 1; /* offset j+32..j+63 */

                /* Load 32 bytes of packed nibbles */
                uint8x16_t qa = vld1q_u8(q);
                uint8x16_t qb = vld1q_u8(q + 16);
                /* lo_a: weights j..j+15, lo_b: weights j+16..j+31 */
                int8x16_t wa_lo = vreinterpretq_s8_u8(vandq_u8(qa, mask_lo));
                int8x16_t wa_hi = vreinterpretq_s8_u8(vandq_u8(qb, mask_lo));
                /* hi_a: weights j+32..j+47, hi_b: weights j+48..j+63 */
                int8x16_t wb_lo = vreinterpretq_s8_u8(vshrq_n_u8(qa, 4));
                int8x16_t wb_hi = vreinterpretq_s8_u8(vshrq_n_u8(qb, 4));

                /* x for sub-block A: 32 int8 values starting at sub_idx_a*32 */
                const int8_t* xa = t->x_qs + (size_t)sub_idx_a * 32;
                int8x16_t xa_lo = vld1q_s8(xa);
                int8x16_t xa_hi = vld1q_s8(xa + 16);
                /* x for sub-block B */
                const int8_t* xb = t->x_qs + (size_t)sub_idx_b * 32;
                int8x16_t xb_lo = vld1q_s8(xb);
                int8x16_t xb_hi = vld1q_s8(xb + 16);

#ifdef __ARM_FEATURE_DOTPROD
                /* ARMv8.2 dotprod: 16 int8 MACs per call. 2 calls = full sub-block. */
                int32x4_t accA = vdotq_s32(vdupq_n_s32(0), wa_lo, xa_lo);
                accA = vdotq_s32(accA, wa_hi, xa_hi);
                int32_t isumA = vaddvq_s32(accA);
                int32x4_t accB = vdotq_s32(vdupq_n_s32(0), wb_lo, xb_lo);
                accB = vdotq_s32(accB, wb_hi, xb_hi);
                int32_t isumB = vaddvq_s32(accB);
#else
                /* int8 dot for sub-block A: 4 widening multiplies, padalq accumulates */
                int32x4_t accA = vpadalq_s16(vdupq_n_s32(0),
                                              vmull_s8(vget_low_s8(wa_lo),  vget_low_s8(xa_lo)));
                accA = vpadalq_s16(accA, vmull_s8(vget_high_s8(wa_lo), vget_high_s8(xa_lo)));
                accA = vpadalq_s16(accA, vmull_s8(vget_low_s8(wa_hi),  vget_low_s8(xa_hi)));
                accA = vpadalq_s16(accA, vmull_s8(vget_high_s8(wa_hi), vget_high_s8(xa_hi)));
                int32_t isumA = vaddvq_s32(accA);

                int32x4_t accB = vpadalq_s16(vdupq_n_s32(0),
                                              vmull_s8(vget_low_s8(wb_lo),  vget_low_s8(xb_lo)));
                accB = vpadalq_s16(accB, vmull_s8(vget_high_s8(wb_lo), vget_high_s8(xb_lo)));
                accB = vpadalq_s16(accB, vmull_s8(vget_low_s8(wb_hi),  vget_low_s8(xb_hi)));
                accB = vpadalq_s16(accB, vmull_s8(vget_high_s8(wb_hi), vget_high_s8(xb_hi)));
                int32_t isumB = vaddvq_s32(accB);
#endif

                /* Combine: weight_i = q_i * (d*sc) - (dmin*mn)
                 *   dot = sum(w_i * x_i)
                 *       = (d*sc) * sum(q_i * x_int8_i) * x_d
                 *         - (dmin*mn) * sum(x_int8_i) * x_d
                 *   First term uses isum (just computed). Second term uses
                 *   precomputed x_isums to avoid re-summing per row. */
                float xdA = t->x_ds[sub_idx_a];
                float xdB = t->x_ds[sub_idx_b];
                int32_t xisA = t->x_isums[sub_idx_a];
                int32_t xisB = t->x_isums[sub_idx_b];

                row_sum += (dW * sc[is + 0] * xdA) * (float)isumA
                         - (dminW * mn[is + 0] * xdA) * (float)xisA;
                row_sum += (dW * sc[is + 1] * xdB) * (float)isumB
                         - (dminW * mn[is + 1] * xdB) * (float)xisB;

                q += 32;
                is += 2;
            }
        }
        t->out[d] = row_sum;
    }
    return NULL;
}
#endif

/* Force-CPU variant: skips Metal dispatch entirely. Used for fused QKV
 * matmuls (Phi-3) where the Metal buffer management has a bug with the
 * large output dimension (out_dim = 3 * hidden_dim). The CPU NEON path
 * handles it correctly. */
void tq_matmul_gguf_cpu(float* out, const float* x,
                         const void* weight, tq_ggml_dtype weight_type,
                         int out_dim, int in_dim);

/* Thread-local flag to force CPU path in tq_matmul_gguf. Set to 1
 * before calling tq_matmul_gguf to skip Metal dispatch. Used for
 * Phi-3 fused QKV/FFN matmuls where Metal has a buffer sizing bug
 * with the unusually large output dimensions. */
/* Global flag (NOT thread-local) — worker threads in the matmul thread pool
 * must see the same value set by the main thread. Prior _Thread_local version
 * silently allowed Metal dispatch from worker threads despite the main thread
 * setting the flag to 1 (observed as Phi-3.5 Q4_K_M garbage output under Metal). */
int tq_matmul_force_cpu = 0;

void tq_matmul_gguf(float* out, const float* x,
                    const void* weight, tq_ggml_dtype weight_type,
                    int out_dim, int in_dim)
{
    /* Metal GPU dispatch for supported GGUF types.
     *
     * Two modes:
     *   1. Batch mode: when tq_metal_batch_begin_if_available() was called
     *      by the transformer, all matmuls encode into one command buffer.
     *      Dispatch overhead is amortized across the batch.
     *   2. Immediate mode: for large matmuls (out_dim >= 512), the GPU
     *      throughput beats CPU even with per-call dispatch overhead.
     *      Small matmuls stay on CPU where fused dot is faster.
     *
     * Returns 0 on success, -1 if type unsupported or Metal unavailable.
     * On -1, falls through to CPU path below. */
#ifdef TQ_HAS_METAL
    {
        extern int tq_metal_available(void);
        extern int tq_metal_matmul_gguf(float*, const float*, const void*,
                                        tq_ggml_dtype, int, int);
        extern int tq_metal_batch_active(void);

        if (!tq_matmul_force_cpu && tq_metal_available()) {
            /* In batch mode, always dispatch to GPU (overhead is amortized).
             * In immediate mode, only for types that have Metal pipelines
             * AND large matrices where GPU wins. */
            int has_metal_pipeline = (weight_type == TQ_GGML_TYPE_IQ2_XXS ||
                                      weight_type == TQ_GGML_TYPE_IQ2_S ||
                                      weight_type == TQ_GGML_TYPE_Q8_0 ||
                                      weight_type == TQ_GGML_TYPE_Q4_K);
            int use_gpu = has_metal_pipeline && (tq_metal_batch_active() || (out_dim >= 512));
            if (use_gpu) {
                int rc = tq_metal_matmul_gguf(out, x, weight, weight_type,
                                              out_dim, in_dim);
                if (rc == 0) return; /* GPU handled it */
                /* rc == -1: unsupported type, fall through to CPU */
            }
        }
    }
#endif

    const size_t block_bytes = tq_ggml_type_size(weight_type);
    const int    block_elems = tq_ggml_type_blck(weight_type);

    if (block_bytes == 0 || block_elems == 0) {
        static int warn_count = 0;
        if (warn_count++ < 5) {
            void* ra = __builtin_return_address(0);
            fprintf(stderr, "tq_matmul_gguf: unsupported type %d (out=%d, in=%d, w=%p, caller=%p)\n",
                    (int)weight_type, out_dim, in_dim, weight, ra);
        }
        memset(out, 0, (size_t)out_dim * sizeof(float));
        return;
    }

    const int    n_blocks  = in_dim / block_elems;
    const size_t row_bytes = (size_t)n_blocks * block_bytes;

    /* Q8×Q8 integer dot: input pre-quantized in transformer (once per layer).
     * Uses int8×int8 vmull_s8+vpadalq_s16 — ~2x faster than float fused dot. */
#if TQ_HAS_NEON
    if (weight_type == TQ_GGML_TYPE_Q8_0 && g_preq_qs != NULL && in_dim <= 4096) {
        const int8_t* xqs = g_preq_qs;
        const float*  xds = g_preq_ds;

        /* Always use thread pool — pass preq data via task struct (TLS doesn't propagate) */
        int n_threads = tq_get_threads();
        if (n_threads > TQ_TP_MAX) n_threads = TQ_TP_MAX;
        if (n_threads > out_dim) n_threads = out_dim;
        if (n_threads < 1) n_threads = 1;

        q8_int_task_t tasks[TQ_TP_MAX];
        void* ptrs[TQ_TP_MAX];
        int rows_per = out_dim / n_threads;
        for (int t = 0; t < n_threads; t++) {
            tasks[t] = (q8_int_task_t){
                .out = out, .weight = weight, .x_qs = xqs, .x_ds = xds,
                .row_bytes = row_bytes, .n_blocks = n_blocks,
                .start_row = t * rows_per,
                .end_row = (t == n_threads - 1) ? out_dim : (t + 1) * rows_per
            };
            ptrs[t] = &tasks[t];
        }
        if (n_threads == 1) {
            q8_int_dot_worker(ptrs[0]);
        } else {
            extern void tq_tp_run(void* (*fn)(void*), void** args, int n);
            tq_tp_run(q8_int_dot_worker, ptrs, n_threads);
        }
        return;
    }
#endif

    /* ---- Q8_0×Q8 integer dot fast path (auto-quantize activation) ----
     * When g_preq_qs is not set (most callers), quantize the input
     * activation inline and use the NEON int8×int8 path. Cost of
     * one-time activation quantization is O(in_dim); matmul is
     * O(in_dim * out_dim), so quantization overhead is negligible
     * for typical out_dim >= 256.
     *
     * Previously disabled with false claim "per-call overhead > benefit"
     * — actually this is 3-4x faster than float fused_dot for Q8_0. */
#if TQ_HAS_NEON
    if (weight_type == TQ_GGML_TYPE_Q8_0 && in_dim >= 32 && in_dim <= 16384) {
        /* Step 1: Quantize input x[in_dim] to Q8 blocks on stack.
         * Buffers sized for max FFN dim (Llama-70B uses 28672, but most
         * models <= 16384). Stack usage: 16KB + 2KB = 18KB. */
        int8_t  x_qs[16384];
        float   x_ds[512];
        if (in_dim <= 16384) {
            for (int b = 0; b < n_blocks; b++) {
                const float* xp = x + b * 32;
                /* Find absmax for this block */
                float amax = 0.0f;
                for (int j = 0; j < 32; j++) {
                    float a = xp[j] < 0 ? -xp[j] : xp[j];
                    if (a > amax) amax = a;
                }
                float d = amax / 127.0f;
                x_ds[b] = d;
                if (d > 0.0f) {
                    float id = 1.0f / d;
                    for (int j = 0; j < 32; j++) {
                        int v = (int)roundf(xp[j] * id);
                        x_qs[b * 32 + j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
                    }
                } else {
                    memset(x_qs + b * 32, 0, 32);
                }
            }

            /* Step 2: Integer dot for each output row */
            int n_threads = tq_get_threads();
            /* For large matmuls, use thread pool; for small ones, single thread */
            if (n_threads <= 1 || out_dim <= n_threads) {
                for (int d = 0; d < out_dim; d++) {
                    const block_q8_0* wblk = (const block_q8_0*)((const uint8_t*)weight + (size_t)d * row_bytes);
                    float row_sum = 0.0f;
                    for (int b = 0; b < n_blocks; b++) {
                        const float wd = fp16_to_fp32(wblk[b].d);
                        const int8_t* wqs = wblk[b].qs;
                        const int8_t* xqs = x_qs + b * 32;
                        /* NEON int8×int8 dot using vdotq_s32 (M1+) or widening multiply */
                        int32x4_t vdot0 = vdupq_n_s32(0);
                        int32x4_t vdot1 = vdupq_n_s32(0);
                        for (int j = 0; j < 32; j += 16) {
                            int8x16_t vw = vld1q_s8(wqs + j);
                            int8x16_t vx = vld1q_s8(xqs + j);
                            /* Widening multiply-accumulate: int8×int8 → int16 → int32 */
                            int16x8_t vprod_lo = vmull_s8(vget_low_s8(vw), vget_low_s8(vx));
                            int16x8_t vprod_hi = vmull_s8(vget_high_s8(vw), vget_high_s8(vx));
                            vdot0 = vpadalq_s16(vdot0, vprod_lo);
                            vdot1 = vpadalq_s16(vdot1, vprod_hi);
                        }
                        int32_t isum = vaddvq_s32(vaddq_s32(vdot0, vdot1));
                        row_sum += wd * x_ds[b] * (float)isum;
                    }
                    out[d] = row_sum;
                }
            } else {
                /* Multi-threaded: each thread processes a range of output rows */
                if (n_threads > TQ_TP_MAX) n_threads = TQ_TP_MAX;
                if (n_threads > out_dim) n_threads = out_dim;
                q8_int_task_t q8_tasks[TQ_TP_MAX];
                void* q8_ptrs[TQ_TP_MAX];

                int rows_per = out_dim / n_threads;
                for (int t = 0; t < n_threads; t++) {
                    q8_tasks[t].out = out;
                    q8_tasks[t].weight = weight;
                    q8_tasks[t].x_qs = x_qs;
                    q8_tasks[t].x_ds = x_ds;
                    q8_tasks[t].row_bytes = row_bytes;
                    q8_tasks[t].n_blocks = n_blocks;
                    q8_tasks[t].start_row = t * rows_per;
                    q8_tasks[t].end_row = (t == n_threads - 1) ? out_dim : (t + 1) * rows_per;
                    q8_ptrs[t] = &q8_tasks[t];
                }
                extern void tq_tp_run(void* (*fn)(void*), void** args, int n);
                extern void* q8_int_dot_worker(void* arg);
                tq_tp_run(q8_int_dot_worker, q8_ptrs, n_threads);
            }
            return;
        }
    }

    /* ---- Q4_K × int8 dot fast path (auto-quantize activation) ----
     * Same pattern as Q8_0: quantize x once to int8 (32-element blocks),
     * precompute per-block int sums for the dmin*mn correction, then
     * run vmull_s8 + vpadalq_s16 dots over 4-bit nibbles unpacked to int8.
     * Replaces the float fused_dot_q4_k path on Phi-3.5/Llama Q4_K_M models. */
    if ((weight_type == TQ_GGML_TYPE_Q4_K || weight_type == TQ_GGML_TYPE_Q5_K)
        && in_dim >= 256 && in_dim <= 16384 && (in_dim % 256) == 0)
    {
        /* Stack buffers: x as int8 (16KB), per-block scales (512 floats =
         * 2KB), per-block int sums (512 ints = 2KB). Total ~20KB. */
        int8_t  x_qs[16384];
        float   x_ds[512];
        int32_t x_isums[512];

        /* Step 1: Per-32-element-block quantization of x to int8. */
        const int n_blocks_x = in_dim / 32;
        for (int b = 0; b < n_blocks_x; b++) {
            const float* xp = x + b * 32;
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float a = xp[j] < 0 ? -xp[j] : xp[j];
                if (a > amax) amax = a;
            }
            float d = amax / 127.0f;
            x_ds[b] = d;
            int32_t isum = 0;
            if (d > 0.0f) {
                float id = 1.0f / d;
                for (int j = 0; j < 32; j++) {
                    int v = (int)roundf(xp[j] * id);
                    int8_t q = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
                    x_qs[b * 32 + j] = q;
                    isum += q;
                }
            } else {
                memset(x_qs + b * 32, 0, 32);
            }
            x_isums[b] = isum;
        }

        const int nb_super = in_dim / 256; /* number of 256-elem super-blocks */
        int n_threads = tq_get_threads();
        if (n_threads > TQ_TP_MAX) n_threads = TQ_TP_MAX;
        if (n_threads > out_dim) n_threads = out_dim;
        if (n_threads < 1) n_threads = 1;

        q4k_int_task_t tasks[TQ_TP_MAX];
        void* ptrs[TQ_TP_MAX];
        int rows_per = out_dim / n_threads;
        for (int t = 0; t < n_threads; t++) {
            tasks[t].out       = out;
            tasks[t].weight    = weight;
            tasks[t].x_qs      = x_qs;
            tasks[t].x_ds      = x_ds;
            tasks[t].x_isums   = x_isums;
            tasks[t].row_bytes = row_bytes;
            tasks[t].nb_super  = nb_super;
            tasks[t].start_row = t * rows_per;
            tasks[t].end_row   = (t == n_threads - 1) ? out_dim : (t + 1) * rows_per;
            ptrs[t] = &tasks[t];
        }
        void* (*worker)(void*) = (weight_type == TQ_GGML_TYPE_Q5_K)
                                  ? q5k_int_dot_worker : q4k_int_dot_worker;
        if (n_threads == 1) {
            worker(ptrs[0]);
        } else {
            extern void tq_tp_run(void* (*fn)(void*), void** args, int n);
            tq_tp_run(worker, ptrs, n_threads);
        }
        return;
    }
#endif

    /* ---- Fused fast paths: dequant + dot in one pass, no tmp buffer ---- */

    /* Fused path function pointer: returns dot product for one row */
    float (*fused_dot)(const void*, const float*, int) = NULL;

    switch (weight_type) {
        case TQ_GGML_TYPE_IQ2_XXS:
#if TQ_HAS_NEON
            fused_dot = fused_dot_iq2_xxs_neon;
#else
            fused_dot = fused_dot_iq2_xxs;
#endif
            break;
        case TQ_GGML_TYPE_IQ2_S:
            fused_dot = fused_dot_iq2_s;
            break;
        case TQ_GGML_TYPE_Q8_0:
            fused_dot = fused_dot_q8_0;
            break;
        case TQ_GGML_TYPE_Q2_K:
            fused_dot = fused_dot_q2_k;
            break;
        case TQ_GGML_TYPE_Q8_1:
            fused_dot = fused_dot_q8_1;
            break;
        case TQ_GGML_TYPE_Q3_K:
            fused_dot = fused_dot_q3_k;
            break;
        case TQ_GGML_TYPE_Q4_K:
            fused_dot = fused_dot_q4_k;
            break;
        case TQ_GGML_TYPE_Q4_0:
            fused_dot = fused_dot_q4_0;
            break;
        case TQ_GGML_TYPE_Q6_K:
            fused_dot = fused_dot_q6_k;
            break;
        case TQ_GGML_TYPE_IQ3_XXS:
            fused_dot = fused_dot_iq3_xxs;
            break;
        case TQ_GGML_TYPE_IQ4_NL:
            fused_dot = fused_dot_iq4_nl;
            break;
        case TQ_GGML_TYPE_IQ4_XS:
            fused_dot = fused_dot_iq4_xs;
            break;
        default:
            break;
    }

    /* ---- Multi-threaded dispatch ---- */
    int n_threads = tq_get_threads();

    /* Note: single-thread for small matmuls was tested and was SLOWER
     * (538ms vs 251ms MoE). Multi-threading benefits IQ2_XXS fused dot
     * even at out_dim=512. Keep multi-threaded. */

    /* For small matrices or single-thread config, skip thread overhead */
    if (n_threads <= 1 || out_dim < n_threads) {
        /* Single-threaded path */
        if (fused_dot) {
            for (int d = 0; d < out_dim; d++) {
                const uint8_t* row = (const uint8_t*)weight + (size_t)d * row_bytes;
                out[d] = fused_dot(row, x, in_dim);
            }
        } else {
            gguf_matmul_task_t task = {
                .out = out, .x = x, .weight = weight, .fused_dot = NULL,
                .weight_type = weight_type, .row_bytes = row_bytes,
                .in_dim = in_dim, .block_bytes = (int)block_bytes,
                .block_elems = block_elems, .n_blocks = n_blocks,
                .start_row = 0, .end_row = out_dim
            };
            gguf_matmul_worker(&task);
        }
        return;
    }

    /* Cap threads */
    if (n_threads > TQ_TP_MAX) n_threads = TQ_TP_MAX;
    if (n_threads > out_dim) n_threads = out_dim;

    gguf_matmul_task_t tasks[TQ_TP_MAX];
    void* ptrs[TQ_TP_MAX];

    int rows_per_thread = out_dim / n_threads;
    for (int t = 0; t < n_threads; t++) {
        tasks[t].out         = out;
        tasks[t].x           = x;
        tasks[t].weight      = weight;
        tasks[t].fused_dot   = fused_dot;
        tasks[t].weight_type = weight_type;
        tasks[t].row_bytes   = row_bytes;
        tasks[t].in_dim      = in_dim;
        tasks[t].block_bytes = (int)block_bytes;
        tasks[t].block_elems = block_elems;
        tasks[t].n_blocks    = n_blocks;
        tasks[t].start_row   = t * rows_per_thread;
        tasks[t].end_row     = (t == n_threads - 1) ? out_dim : (t + 1) * rows_per_thread;
        ptrs[t] = &tasks[t];
    }

    tq_tp_run(gguf_matmul_worker, ptrs, n_threads);
}

void tq_matmul_gguf_cpu(float* out, const float* x,
                         const void* weight, tq_ggml_dtype weight_type,
                         int out_dim, int in_dim)
{
    /* Save-and-restore, not hard-reset. A caller (e.g., tq_forward for
     * Phi-3) may have already set the flag to 1 as an invariant for the
     * entire forward pass; hard-resetting to 0 here would let subsequent
     * matmuls in the same forward dispatch to Metal and produce garbage. */
    int prev = tq_matmul_force_cpu;
    tq_matmul_force_cpu = 1;
    tq_matmul_gguf(out, x, weight, weight_type, out_dim, in_dim);
    tq_matmul_force_cpu = prev;
}

/* ============================================================
 * Metal batch mode wrappers
 *
 * These forward to Metal batch API when available, otherwise no-op.
 * The transformer/MoE code calls these to batch consecutive matmuls
 * into a single GPU command buffer, reducing dispatch overhead.
 * ============================================================ */

void tq_metal_batch_begin_if_available(void) {
#ifdef TQ_HAS_METAL
    extern int tq_metal_available(void);
    extern void tq_metal_batch_begin(void);
    if (tq_metal_available()) {
        tq_metal_batch_begin();
    }
#endif
}

void tq_metal_batch_flush_if_available(void) {
#ifdef TQ_HAS_METAL
    extern void tq_metal_batch_flush(void);
    extern int tq_metal_batch_active(void);
    if (tq_metal_batch_active()) {
        tq_metal_batch_flush();
    }
#endif
}

void tq_metal_batch_end_if_available(void) {
#ifdef TQ_HAS_METAL
    extern void tq_metal_batch_end(void);
    extern int tq_metal_batch_active(void);
    if (tq_metal_batch_active()) {
        tq_metal_batch_end();
    }
#endif
}
