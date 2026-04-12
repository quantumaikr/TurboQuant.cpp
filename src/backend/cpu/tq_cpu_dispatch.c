/**
 * CPU runtime dispatch — detects SIMD capabilities and selects kernels
 *
 * Since TQ_TRAITS is const, we provide a separate mutable dispatch table
 * (tq_dispatch) that is initialized with the best available implementations.
 * Code that needs dispatched functions should use tq_dispatch instead of
 * TQ_TRAITS directly for quantize/dequantize/attention calls.
 */

#include "turboquant/turboquant.h"
#include <string.h>

/* ================================================================
 * Mutable dispatch table — initialized to reference implementations
 * ================================================================ */

typedef struct {
    tq_quantize_fn   quantize;
    tq_dequantize_fn dequantize;
    tq_attention_fn  attention;
} tq_dispatch_entry_t;

static tq_dispatch_entry_t tq_dispatch_table[TQ_TYPE_COUNT];
static int tq_dispatch_initialized = 0;

/* ================================================================
 * Forward declarations for SIMD implementations
 * ================================================================ */

/* Reference fallbacks (always available) */
extern void tq_polar_quantize_ref(const float* src, void* dst, int n);
extern void tq_polar_dequantize_ref(const void* src, float* dst, int n);
extern void tq_polar_attention_ref(const float* q, const void* kv,
                                   float* s, int seq, int hd);

extern void tq_qjl_quantize_ref(const float* src, void* dst, int n);
extern void tq_qjl_dequantize_ref(const void* src, float* dst, int n);
extern void tq_qjl_attention_ref(const float* q, const void* kv,
                                 float* s, int seq, int hd);

extern void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
extern void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
extern void tq_uniform_2b_quantize_ref(const float* src, void* dst, int n);
extern void tq_uniform_2b_dequantize_ref(const void* src, float* dst, int n);

#if defined(__ARM_NEON)
/* NEON optimized implementations */
extern void tq_uniform_4b_quantize_neon(const float* src, void* dst, int n);
extern void tq_uniform_4b_dequantize_neon(const void* src, float* dst, int n);
extern void tq_polar_quantize_neon(const float* src, void* dst, int n);
extern void tq_polar_dequantize_neon(const void* src, float* dst, int n);
extern void tq_qjl_quantize_neon(const float* src, void* dst, int n);
extern void tq_qjl_attention_neon(const float* q, const void* kv,
                                  float* s, int seq, int hd);
#endif

#if defined(__AVX2__)
/* AVX2 optimized implementations */
extern void tq_uniform_4b_quantize_avx2(const float* src, void* dst, int n);
extern void tq_uniform_4b_dequantize_avx2(const void* src, float* dst, int n);
extern void tq_polar_quantize_avx2(const float* src, void* dst, int n);
extern void tq_polar_dequantize_avx2(const void* src, float* dst, int n);
extern void tq_qjl_quantize_avx2(const float* src, void* dst, int n);
extern void tq_qjl_attention_avx2(const float* q, const void* kv,
                                   float* s, int seq, int hd);
#endif

#if defined(__ARM_FEATURE_SVE)
/* SVE optimized implementations (stubs — delegate to reference for now) */
extern void tq_uniform_4b_quantize_sve(const float* src, void* dst, int n);
extern void tq_uniform_4b_dequantize_sve(const void* src, float* dst, int n);
extern void tq_polar_quantize_sve(const float* src, void* dst, int n);
extern void tq_polar_dequantize_sve(const void* src, float* dst, int n);
extern void tq_qjl_quantize_sve(const float* src, void* dst, int n);
extern void tq_qjl_attention_sve(const float* q, const void* kv,
                                  float* s, int seq, int hd);
#endif

/* ================================================================
 * CPU feature detection
 * ================================================================ */

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#ifdef _MSC_VER
#include <intrin.h>
static int cpu_has_avx2(void) {
    int cpuinfo[4];
    __cpuidex(cpuinfo, 7, 0);
    return (cpuinfo[1] & (1 << 5)) != 0; /* EBX bit 5 = AVX2 */
}
#else
#include <cpuid.h>
static int cpu_has_avx2(void) {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
    return (ebx & (1 << 5)) != 0; /* EBX bit 5 = AVX2 */
}
#endif

#endif /* x86 */

/* ================================================================
 * Dispatch initialization
 * ================================================================ */

void tq_cpu_dispatch_init(void) {
    if (tq_dispatch_initialized) return;

    /* Start with reference implementations from TQ_TRAITS */
    for (int i = 0; i < TQ_TYPE_COUNT; i++) {
        tq_dispatch_table[i].quantize   = TQ_TRAITS[i].quantize;
        tq_dispatch_table[i].dequantize = TQ_TRAITS[i].dequantize;
        tq_dispatch_table[i].attention  = TQ_TRAITS[i].attention;
    }

    /* --- ARM NEON dispatch (compile-time detection) --- */
#if defined(__ARM_NEON)
    /* NEON is always available when compiled with __ARM_NEON */
    tq_dispatch_table[TQ_TYPE_UNIFORM_4B].quantize   = tq_uniform_4b_quantize_neon;
    tq_dispatch_table[TQ_TYPE_UNIFORM_4B].dequantize = tq_uniform_4b_dequantize_neon;

    tq_dispatch_table[TQ_TYPE_POLAR_3B].quantize   = tq_polar_quantize_neon;
    tq_dispatch_table[TQ_TYPE_POLAR_3B].dequantize = tq_polar_dequantize_neon;
    tq_dispatch_table[TQ_TYPE_POLAR_4B].quantize   = tq_polar_quantize_neon;
    tq_dispatch_table[TQ_TYPE_POLAR_4B].dequantize = tq_polar_dequantize_neon;

    tq_dispatch_table[TQ_TYPE_QJL_1B].quantize  = tq_qjl_quantize_neon;
    tq_dispatch_table[TQ_TYPE_QJL_1B].attention = tq_qjl_attention_neon;
#endif

    /* --- ARM SVE dispatch (compile-time detection) --- */
#if defined(__ARM_FEATURE_SVE)
    /* SVE takes priority over NEON when available (wider vectors).
     * Currently stubs that delegate to reference — swap with real
     * SVE implementations as they are developed. */
    tq_dispatch_table[TQ_TYPE_UNIFORM_4B].quantize   = tq_uniform_4b_quantize_sve;
    tq_dispatch_table[TQ_TYPE_UNIFORM_4B].dequantize = tq_uniform_4b_dequantize_sve;

    tq_dispatch_table[TQ_TYPE_POLAR_3B].quantize   = tq_polar_quantize_sve;
    tq_dispatch_table[TQ_TYPE_POLAR_3B].dequantize = tq_polar_dequantize_sve;
    tq_dispatch_table[TQ_TYPE_POLAR_4B].quantize   = tq_polar_quantize_sve;
    tq_dispatch_table[TQ_TYPE_POLAR_4B].dequantize = tq_polar_dequantize_sve;

    tq_dispatch_table[TQ_TYPE_QJL_1B].quantize  = tq_qjl_quantize_sve;
    tq_dispatch_table[TQ_TYPE_QJL_1B].attention = tq_qjl_attention_sve;
#endif

    /* --- x86 AVX2 dispatch (runtime detection) --- */
#if defined(__AVX2__)
    /* If compiled with -mavx2, AVX2 is always available */
    tq_dispatch_table[TQ_TYPE_UNIFORM_4B].quantize   = tq_uniform_4b_quantize_avx2;
    tq_dispatch_table[TQ_TYPE_UNIFORM_4B].dequantize = tq_uniform_4b_dequantize_avx2;

    tq_dispatch_table[TQ_TYPE_POLAR_3B].quantize   = tq_polar_quantize_avx2;
    tq_dispatch_table[TQ_TYPE_POLAR_3B].dequantize = tq_polar_dequantize_avx2;
    tq_dispatch_table[TQ_TYPE_POLAR_4B].quantize   = tq_polar_quantize_avx2;
    tq_dispatch_table[TQ_TYPE_POLAR_4B].dequantize = tq_polar_dequantize_avx2;

    tq_dispatch_table[TQ_TYPE_QJL_1B].quantize  = tq_qjl_quantize_avx2;
    tq_dispatch_table[TQ_TYPE_QJL_1B].attention = tq_qjl_attention_avx2;
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    /* Runtime detection for x86 when not compiled with -mavx2 */
    /* Only swap if AVX2 is available at runtime.
       Since we can't call AVX2 code without -mavx2, this path
       stays on reference. A real deployment would use multi-versioning
       or separate compilation units with -mavx2 flag. */
    (void)cpu_has_avx2; /* suppress unused warning */
#endif

    tq_dispatch_initialized = 1;
}

/* ================================================================
 * Dispatch accessors — get the best available implementation
 * ================================================================ */

tq_quantize_fn tq_get_quantize_fn(tq_type type) {
    if (!tq_dispatch_initialized) tq_cpu_dispatch_init();
    if (type < 0 || type >= TQ_TYPE_COUNT) return NULL;
    return tq_dispatch_table[type].quantize;
}

tq_dequantize_fn tq_get_dequantize_fn(tq_type type) {
    if (!tq_dispatch_initialized) tq_cpu_dispatch_init();
    if (type < 0 || type >= TQ_TYPE_COUNT) return NULL;
    return tq_dispatch_table[type].dequantize;
}

tq_attention_fn tq_get_attention_fn(tq_type type) {
    if (!tq_dispatch_initialized) tq_cpu_dispatch_init();
    if (type < 0 || type >= TQ_TYPE_COUNT) return NULL;
    return tq_dispatch_table[type].attention;
}

/* ================================================================
 * Query which backend is active for a given type
 * ================================================================ */

const char* tq_get_dispatch_backend(tq_type type) {
    if (!tq_dispatch_initialized) tq_cpu_dispatch_init();
    if (type < 0 || type >= TQ_TYPE_COUNT) return "unknown";

#if defined(__ARM_FEATURE_SVE)
    /* Check if using SVE versions */
    if (type == TQ_TYPE_UNIFORM_4B &&
        tq_dispatch_table[type].quantize == tq_uniform_4b_quantize_sve)
        return "sve";
    if ((type == TQ_TYPE_POLAR_3B || type == TQ_TYPE_POLAR_4B) &&
        tq_dispatch_table[type].quantize == tq_polar_quantize_sve)
        return "sve";
    if (type == TQ_TYPE_QJL_1B &&
        tq_dispatch_table[type].quantize == tq_qjl_quantize_sve)
        return "sve";
#endif

#if defined(__ARM_NEON)
    /* Check if using NEON versions */
    if (type == TQ_TYPE_UNIFORM_4B &&
        tq_dispatch_table[type].quantize == tq_uniform_4b_quantize_neon)
        return "neon";
    if ((type == TQ_TYPE_POLAR_3B || type == TQ_TYPE_POLAR_4B) &&
        tq_dispatch_table[type].quantize == tq_polar_quantize_neon)
        return "neon";
    if (type == TQ_TYPE_QJL_1B &&
        tq_dispatch_table[type].quantize == tq_qjl_quantize_neon)
        return "neon";
#endif

#if defined(__AVX2__)
    if (type == TQ_TYPE_UNIFORM_4B &&
        tq_dispatch_table[type].quantize == tq_uniform_4b_quantize_avx2)
        return "avx2";
    if ((type == TQ_TYPE_POLAR_3B || type == TQ_TYPE_POLAR_4B) &&
        tq_dispatch_table[type].quantize == tq_polar_quantize_avx2)
        return "avx2";
#endif

    return "generic";
}
