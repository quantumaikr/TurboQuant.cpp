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
#include <math.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define TQ_HAS_NEON 1
#else
#define TQ_HAS_NEON 0
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

/* --- Other IQ type stubs --- */
static void dequant_iq_stub(const char* type_name, float* dst, int n) {
    static int warned = 0;
    if (!warned) {
        fprintf(stderr, "tq_gguf_quants: WARNING: %s dequant not yet implemented, "
                        "returning zeros\n", type_name);
        warned = 1;
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
            dequant_iq_stub("IQ3_XXS", dst, n);
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
__attribute__((unused))
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
/* NEON-optimized fused IQ2_XXS dot product.
 * Processes 8 grid values at a time using vectorized sign application. */
static float fused_dot_iq2_xxs_neon(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)row;
    float32x4_t vtotal = vdupq_n_f32(0.0f);

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 66;
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint16_t* qs = (const uint16_t*)(blk + 2);
        const float* xbase = x + b * 256;

        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32[2];
            memcpy(aux32, qs + 4 * ib32, 8);
            const uint8_t* aux8 = (const uint8_t*)aux32;
            const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
            const float* xb = xbase + ib32 * 32;

            float32x4_t vgroup = vdupq_n_f32(0.0f);

            for (int l = 0; l < 4; l++) {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
                const float* xp = xb + l * 8;

                /* Load 8 grid bytes, expand to float */
                uint8x8_t vgrid = vld1_u8(grid);
                int16x8_t vgrid16 = vreinterpretq_s16_u16(vmovl_u8(vgrid));
                int32x4_t vg_lo = vmovl_s16(vget_low_s16(vgrid16));
                int32x4_t vg_hi = vmovl_s16(vget_high_s16(vgrid16));
                float32x4_t vf_lo = vcvtq_f32_s32(vg_lo);
                float32x4_t vf_hi = vcvtq_f32_s32(vg_hi);

                /* Apply signs via float bit XOR: set sign bit where signs bit is 1.
                 * Expand each sign bit to a 32-bit mask with only the float sign bit set. */
                uint32x4_t sign_lo = {
                    (uint32_t)((signs >> 0) & 1) << 31,
                    (uint32_t)((signs >> 1) & 1) << 31,
                    (uint32_t)((signs >> 2) & 1) << 31,
                    (uint32_t)((signs >> 3) & 1) << 31
                };
                uint32x4_t sign_hi = {
                    (uint32_t)((signs >> 4) & 1) << 31,
                    (uint32_t)((signs >> 5) & 1) << 31,
                    (uint32_t)((signs >> 6) & 1) << 31,
                    (uint32_t)((signs >> 7) & 1) << 31
                };
                vf_lo = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(vf_lo), sign_lo));
                vf_hi = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(vf_hi), sign_hi));

                /* Dot with input */
                float32x4_t vx_lo = vld1q_f32(xp);
                float32x4_t vx_hi = vld1q_f32(xp + 4);

                vgroup = vfmaq_f32(vgroup, vf_lo, vx_lo);
                vgroup = vfmaq_f32(vgroup, vf_hi, vx_hi);
            }
            /* Scale by db and accumulate */
            vtotal = vfmaq_n_f32(vtotal, vgroup, db);
        }
    }
    return vaddvq_f32(vtotal);
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

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float* xp = x + b * 32;

#if TQ_HAS_NEON
        /* NEON: dot product of 32 int8 * float, scaled by d */
        float32x4_t vsum0 = vdupq_n_f32(0.0f);
        float32x4_t vsum1 = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 8) {
            /* Load 8 int8 weights, convert to float */
            int8x8_t vq = vld1_s8(blk[b].qs + j);
            int16x8_t vq16 = vmovl_s8(vq);
            int32x4_t vq32_lo = vmovl_s16(vget_low_s16(vq16));
            int32x4_t vq32_hi = vmovl_s16(vget_high_s16(vq16));
            float32x4_t vw_lo = vcvtq_f32_s32(vq32_lo);
            float32x4_t vw_hi = vcvtq_f32_s32(vq32_hi);
            float32x4_t vx_lo = vld1q_f32(xp + j);
            float32x4_t vx_hi = vld1q_f32(xp + j + 4);
            vsum0 = vfmaq_f32(vsum0, vw_lo, vx_lo);
            vsum1 = vfmaq_f32(vsum1, vw_hi, vx_hi);
        }
        vsum0 = vaddq_f32(vsum0, vsum1);
        sum += d * vaddvq_f32(vsum0);
#else
        float block_sum = 0.0f;
        for (int j = 0; j < 32; j++) {
            block_sum += (float)blk[b].qs[j] * xp[j];
        }
        sum += d * block_sum;
#endif
    }
    return sum;
}

/* Fused IQ4_NL dot product: 18 bytes per 32 elements */
static float fused_dot_iq4_nl(const void* row, const float* x, int n) {
    const int nb = n / 32;
    const block_iq4_nl* blk = (const block_iq4_nl*)row;
    float sum = 0.0f;

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

/* Fused Q4_K dot product: 144 bytes per 256 elements
 * Layout: 4 groups of 64 elements, each group uses 32 bytes of qs.
 *   First 32 elements: d1 * (q[l] & 0xF) - m1  (low nibble, scale pair[0])
 *   Next 32 elements:  d2 * (q[l] >> 4)  - m2   (high nibble, scale pair[1])
 */
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

        /* 4 groups of 64 elements */
        for (int j = 0; j < 256; j += 64) {
            const float d1 = d * sc[is + 0];
            const float m1 = dmin * mn[is + 0];
            const float d2 = d * sc[is + 1];
            const float m2 = dmin * mn[is + 1];

            /* First 32 elements: low nibble */
            float dot1 = 0.0f, sum_x1 = 0.0f;
            for (int l = 0; l < 32; l++) {
                dot1  += (float)(q[l] & 0x0F) * xp[j + l];
                sum_x1 += xp[j + l];
            }
            sum += d1 * dot1 - m1 * sum_x1;

            /* Next 32 elements: high nibble */
            float dot2 = 0.0f, sum_x2 = 0.0f;
            for (int l = 0; l < 32; l++) {
                dot2  += (float)(q[l] >> 4) * xp[j + 32 + l];
                sum_x2 += xp[j + 32 + l];
            }
            sum += d2 * dot2 - m2 * sum_x2;

            q += 32;
            is += 2;
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

void tq_matmul_gguf(float* out, const float* x,
                    const void* weight, tq_ggml_dtype weight_type,
                    int out_dim, int in_dim)
{
#ifdef TQ_HAS_METAL
    /* Try GPU-accelerated path first (IQ2_XXS, Q8_0, Q4_K) */
    extern int tq_metal_available(void);
    extern int tq_metal_matmul_gguf(float* out, const float* x, const void* weight,
                                     tq_ggml_dtype weight_type, int out_dim, int in_dim);
    if (tq_metal_available() &&
        tq_metal_matmul_gguf(out, x, weight, weight_type, out_dim, in_dim) == 0) {
        return; /* GPU handled it */
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
        case TQ_GGML_TYPE_Q4_K:
            fused_dot = fused_dot_q4_k;
            break;
        case TQ_GGML_TYPE_Q4_0:
            fused_dot = fused_dot_q4_0;
            break;
        case TQ_GGML_TYPE_Q6_K:
            fused_dot = fused_dot_q6_k;
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
