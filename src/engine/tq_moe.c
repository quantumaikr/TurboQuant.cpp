/**
 * tq_moe.c — Mixture of Experts routing and expert dispatch
 *
 * Implements top-K expert selection with softmax renormalization,
 * SwiGLU FFN dispatch per expert, shared expert support,
 * runtime LRU Q8_0 cache for routed experts, and memory advise hints.
 */

#include "turboquant/tq_gguf.h"
#include "turboquant/tq_engine.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define TQ_MOE_HAS_NEON 1
#else
#define TQ_MOE_HAS_NEON 0
#endif

#ifdef TQ_HAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

/* ============================================================
 * Fast SiLU (Swish) approximation
 *
 * silu(x) = x / (1 + exp(-x))
 *
 * Uses Schraudolph's fast exp approximation (~1% accuracy for |x|<10).
 * Called 270 times/token with expert_dim=512, so ~138K calls saved
 * vs. standard expf per token.
 * ============================================================ */
static inline float fast_expf_moe(float x) {
    if (x < -20.0f) return 0.0f;
    if (x > 20.0f) return expf(x);
    union { float f; int32_t i; } v;
    v.i = (int32_t)(12102203.0f * x + 1065353216.0f);
    return v.f;
}

/* Vectorized SwiGLU: hb[i] = silu(hb[i]) * hb2[i] */
static void swiglu_fused(float* restrict hb, const float* restrict hb2, int n) {
#if TQ_MOE_HAS_NEON
    int i = 0;
    float32x4_t vone = vdupq_n_f32(1.0f);
    for (; i + 7 < n; i += 8) {
        /* Process 8 elements: 2x float32x4_t */
        float32x4_t vg0 = vld1q_f32(hb + i);
        float32x4_t vg1 = vld1q_f32(hb + i + 4);
        float32x4_t vu0 = vld1q_f32(hb2 + i);
        float32x4_t vu1 = vld1q_f32(hb2 + i + 4);

        /* Fast sigmoid via Schraudolph: exp(-x) approx */
        float neg0[4], neg1[4];
        vst1q_f32(neg0, vnegq_f32(vg0));
        vst1q_f32(neg1, vnegq_f32(vg1));
        float32x4_t vexp0 = {fast_expf_moe(neg0[0]), fast_expf_moe(neg0[1]),
                              fast_expf_moe(neg0[2]), fast_expf_moe(neg0[3])};
        float32x4_t vexp1 = {fast_expf_moe(neg1[0]), fast_expf_moe(neg1[1]),
                              fast_expf_moe(neg1[2]), fast_expf_moe(neg1[3])};
        /* sigmoid = 1 / (1 + exp(-x)) */
        float32x4_t vsig0 = vrecpeq_f32(vaddq_f32(vone, vexp0));
        vsig0 = vmulq_f32(vsig0, vrecpsq_f32(vaddq_f32(vone, vexp0), vsig0));
        float32x4_t vsig1 = vrecpeq_f32(vaddq_f32(vone, vexp1));
        vsig1 = vmulq_f32(vsig1, vrecpsq_f32(vaddq_f32(vone, vexp1), vsig1));
        /* silu = x * sigmoid(x) */
        float32x4_t vsilu0 = vmulq_f32(vg0, vsig0);
        float32x4_t vsilu1 = vmulq_f32(vg1, vsig1);
        /* silu * up */
        vst1q_f32(hb + i, vmulq_f32(vsilu0, vu0));
        vst1q_f32(hb + i + 4, vmulq_f32(vsilu1, vu1));
    }
    for (; i < n; i++) {
        float g = hb[i];
        hb[i] = (g / (1.0f + fast_expf_moe(-g))) * hb2[i];
    }
#else
    for (int i = 0; i < n; i++) {
        float g = hb[i];
        hb[i] = (g / (1.0f + fast_expf_moe(-g))) * hb2[i];
    }
#endif
}

#ifndef _WIN32
#include <sys/mman.h>
#endif

/* ============================================================
 * Runtime Expert Q8_0 LRU Cache
 *
 * MoE models with 256 experts x 40 layers would need ~19 GB
 * if all experts were pre-converted. Instead, we cache only the
 * EXPERT_CACHE_SIZE most-recently-used experts per layer in Q8_0
 * block format (34 bytes per 32 elements = ~1.06 bytes/elem).
 *
 * Q8_0 fused dot is ~3-5x faster than IQ2_XXS fused dot because
 * it avoids E8 lattice codebook lookups — just int8*float FMA.
 * On cache miss, we dequant IQ2_XXS → FP32 → Q8_0 blocks once.
 * On cache hit, tq_matmul_gguf dispatches to fused_dot_q8_0.
 *
 * Memory: 34 bytes/32 elems ≈ 1.0625 B/elem. For expert with
 * 3M params (gate+up+down), that's ~3.2 MB per cached expert.
 * 32 slots/layer × 3.2 MB ≈ 102 MB/layer. For 30 layers: ~3 GB.
 * ============================================================ */

#define EXPERT_CACHE_SIZE 4   /* per layer — 4 × 3.2MB × 40 = 512 MB max */

/* FP32 → FP16 conversion for Q8_0 block scale fields */
static inline uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t  exp  = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;

    if (exp <= 0) {
        /* Underflow to zero */
        return (uint16_t)sign;
    } else if (exp >= 31) {
        /* Overflow to infinity */
        return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

/* Quantize FP32 array to Q8_0 block format in-place.
 * Q8_0 block: 2-byte fp16 scale + 32 int8 values = 34 bytes per 32 elements.
 * dst must have room for (n/32) * 34 bytes. n must be a multiple of 32. */
static void quantize_fp32_to_q8_0(const float* src, void* dst, int n) {
    const int nb = n / 32;
    uint8_t* out = (uint8_t*)dst;

    for (int b = 0; b < nb; b++) {
        const float* block = src + b * 32;

        /* Find max absolute value */
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            float a = block[j] < 0 ? -block[j] : block[j];
            if (a > amax) amax = a;
        }

        /* Scale: map [-amax, amax] to [-127, 127] */
        float d = amax / 127.0f;
        float id = (d > 0.0f) ? 127.0f / amax : 0.0f;

        /* Write fp16 scale */
        uint16_t d_fp16 = fp32_to_fp16(d);
        memcpy(out + b * 34, &d_fp16, 2);

        /* Write quantized int8 values */
        int8_t* qs = (int8_t*)(out + b * 34 + 2);
        for (int j = 0; j < 32; j++) {
            float v = block[j] * id;
            int32_t vi = (int32_t)(v + (v >= 0 ? 0.5f : -0.5f));
            if (vi > 127) vi = 127;
            if (vi < -127) vi = -127;
            qs[j] = (int8_t)vi;
        }
    }
}

typedef struct {
    int      expert_id;       /* -1 = empty slot */
    void*    gate_q8;         /* Q8_0 block data for gate [inter*dim elems] */
    void*    up_q8;           /* Q8_0 block data for up [inter*dim elems] */
    void*    down_q8;         /* Q8_0 block data for down [dim*inter elems] */
    int      last_used;       /* token counter for LRU eviction */
} expert_cache_entry_t;

typedef struct {
    expert_cache_entry_t entries[EXPERT_CACHE_SIZE];
    int count;                /* number of occupied slots */
} expert_layer_cache_t;

static expert_layer_cache_t* g_expert_cache = NULL; /* [n_layers] */
static int    g_cache_n_layers   = 0;
static int    g_cache_hidden_dim = 0;
static int    g_cache_exp_inter  = 0;
static int    g_token_counter    = 0;
static float* g_cache_fp32_temp  = NULL;  /* reusable dequant buffer */

void tq_moe_cache_init(int n_layers, const tq_moe_config_t* config, int hidden_dim)
{
    if (g_expert_cache) return; /* already initialized */
    if (!config) return;

    g_cache_n_layers   = n_layers;
    g_cache_hidden_dim = hidden_dim;
    g_cache_exp_inter  = config->expert_intermediate_dim;
    g_token_counter    = 0;

    g_expert_cache = (expert_layer_cache_t*)calloc(
        (size_t)n_layers, sizeof(expert_layer_cache_t));
    if (!g_expert_cache) {
        fprintf(stderr, "tq_moe_cache_init: allocation failed\n");
        return;
    }

    /* Mark all slots empty */
    for (int l = 0; l < n_layers; l++) {
        for (int s = 0; s < EXPERT_CACHE_SIZE; s++) {
            g_expert_cache[l].entries[s].expert_id = -1;
        }
    }

    /* Allocate reusable FP32 temp buffer (max of gate/up and down sizes) */
    size_t gate_up_elems = (size_t)g_cache_exp_inter * hidden_dim;
    size_t down_elems    = (size_t)hidden_dim * g_cache_exp_inter;
    size_t max_elems     = gate_up_elems > down_elems ? gate_up_elems : down_elems;
    g_cache_fp32_temp = (float*)malloc(max_elems * sizeof(float));

    /* Q8_0: 34 bytes per 32 elements = 1.0625 bytes/elem
     * Per expert: 3 matrices × (inter*dim) elems × 1.0625 ≈ 3.2 MB (for 1024×512) */
    float cache_mb = (float)(n_layers * EXPERT_CACHE_SIZE) *
                     (3.0f * (float)gate_up_elems * 34.0f / 32.0f) /
                     (1024.0f * 1024.0f);
    fprintf(stderr, "tq_moe_cache_init: Q8 LRU cache for %d layers x %d slots "
            "(max %.0f MB)\n", n_layers, EXPERT_CACHE_SIZE, (double)cache_mb);
}

static void free_cache_entry(expert_cache_entry_t* e)
{
    free(e->gate_q8);  e->gate_q8 = NULL;
    free(e->up_q8);    e->up_q8 = NULL;
    free(e->down_q8);  e->down_q8 = NULL;
    e->expert_id = -1;
}

void tq_moe_cache_free(void)
{
    if (!g_expert_cache) return;
    for (int l = 0; l < g_cache_n_layers; l++) {
        for (int s = 0; s < EXPERT_CACHE_SIZE; s++) {
            free_cache_entry(&g_expert_cache[l].entries[s]);
        }
    }
    free(g_expert_cache);
    g_expert_cache = NULL;
    free(g_cache_fp32_temp);
    g_cache_fp32_temp = NULL;
    g_cache_n_layers = 0;
}

/* Q8_0 block byte size for n elements (n must be multiple of 32) */
static inline size_t q8_0_bytes(int n) {
    return (size_t)(n / 32) * 34;
}

/* Find a cached entry for expert_id in layer, or evict LRU and create one.
 * Returns the entry with Q8_0 data populated, or NULL on allocation failure. */
static expert_cache_entry_t* cache_get_or_create(
    int layer_idx, int expert_id, const tq_expert_weights_t* exp)
{
    expert_layer_cache_t* lc = &g_expert_cache[layer_idx];

    /* Search for existing entry (cache hit) */
    for (int s = 0; s < EXPERT_CACHE_SIZE; s++) {
        if (lc->entries[s].expert_id == expert_id) {
            lc->entries[s].last_used = g_token_counter;
            return &lc->entries[s];
        }
    }

    /* Cache miss: find an empty slot or evict LRU */
    int target = -1;
    if (lc->count < EXPERT_CACHE_SIZE) {
        /* Find first empty slot */
        for (int s = 0; s < EXPERT_CACHE_SIZE; s++) {
            if (lc->entries[s].expert_id < 0) {
                target = s;
                break;
            }
        }
        lc->count++;
    } else {
        /* Evict least-recently-used */
        int oldest_time = g_token_counter + 1;
        for (int s = 0; s < EXPERT_CACHE_SIZE; s++) {
            if (lc->entries[s].last_used < oldest_time) {
                oldest_time = lc->entries[s].last_used;
                target = s;
            }
        }
        free_cache_entry(&lc->entries[target]);
    }

    expert_cache_entry_t* ce = &lc->entries[target];
    ce->expert_id = expert_id;
    ce->last_used = g_token_counter;

    int dim = g_cache_hidden_dim;
    int inter = g_cache_exp_inter;

    /* Convert gate: [inter, dim] — dequant IQ2_XXS → FP32 → Q8_0 blocks */
    {
        int n = inter * dim;
        ce->gate_q8 = malloc(q8_0_bytes(n));
        if (ce->gate_q8) {
            tq_dequant_row_gguf(exp->gate_type, exp->w_gate, g_cache_fp32_temp, n);
            quantize_fp32_to_q8_0(g_cache_fp32_temp, ce->gate_q8, n);
        }
    }

    /* Convert up: [inter, dim] */
    {
        int n = inter * dim;
        ce->up_q8 = malloc(q8_0_bytes(n));
        if (ce->up_q8) {
            tq_dequant_row_gguf(exp->up_type, exp->w_up, g_cache_fp32_temp, n);
            quantize_fp32_to_q8_0(g_cache_fp32_temp, ce->up_q8, n);
        }
    }

    /* Convert down: [dim, inter] */
    {
        int n = dim * inter;
        ce->down_q8 = malloc(q8_0_bytes(n));
        if (ce->down_q8) {
            tq_dequant_row_gguf(exp->down_type, exp->w_down, g_cache_fp32_temp, n);
            quantize_fp32_to_q8_0(g_cache_fp32_temp, ce->down_q8, n);
        }
    }

    return ce;
}

/* ============================================================
 * Accelerate/cblas FP32 Expert LRU Cache
 *
 * On Apple Silicon, cblas_sgemv leverages the AMX coprocessor which is
 * ~36x faster than manual FP32 dot and ~10x faster than NEON for sgemv.
 * The strategy: dequant IQ2_XXS -> FP32 once (on cache miss), then
 * use cblas_sgemv for all subsequent matmuls (cache hit = near-free).
 *
 * Memory per cached expert: 3 matrices x (inter*dim) x 4 bytes
 *   For 512x2048: 3 x 1M x 4 = 12 MB per expert
 *   32 cache slots: 384 MB total (fits comfortably in 16GB+)
 *
 * Cache hit: cblas only = ~0.019 ms per matmul (36x faster)
 * Cache miss: dequant + store + cblas ~ 1-2 ms (amortized quickly)
 * ============================================================ */

#ifdef TQ_HAS_ACCELERATE

#define CBLAS_CACHE_SIZE 2   /* per layer — 2 × 12 MB × 40 layers = 0.96 GB max */

typedef struct {
    int      expert_id;       /* -1 = empty slot */
    float*   gate_fp32;       /* [expert_dim x hidden_dim] row-major */
    float*   up_fp32;         /* [expert_dim x hidden_dim] row-major */
    float*   down_fp32;       /* [hidden_dim x expert_dim] row-major */
    int      last_used;       /* token counter for LRU eviction */
} cblas_cache_entry_t;

typedef struct {
    cblas_cache_entry_t entries[CBLAS_CACHE_SIZE];
    int count;
} cblas_layer_cache_t;

static cblas_layer_cache_t* g_cblas_cache     = NULL; /* [n_layers] */
static int                  g_cblas_n_layers   = 0;
static int                  g_cblas_hidden_dim = 0;
static int                  g_cblas_exp_inter  = 0;
static int                  g_cblas_token      = 0;
static float*               g_cblas_fp32_temp  = NULL; /* reusable dequant buffer */

void tq_moe_cblas_cache_init(int n_layers, const tq_moe_config_t* config, int hidden_dim)
{
    if (g_cblas_cache) return; /* already initialized */
    if (!config) return;

    g_cblas_n_layers   = n_layers;
    g_cblas_hidden_dim = hidden_dim;
    g_cblas_exp_inter  = config->expert_intermediate_dim;
    g_cblas_token      = 0;

    g_cblas_cache = (cblas_layer_cache_t*)calloc(
        (size_t)n_layers, sizeof(cblas_layer_cache_t));
    if (!g_cblas_cache) {
        fprintf(stderr, "tq_moe_cblas_cache_init: allocation failed\n");
        return;
    }

    for (int l = 0; l < n_layers; l++) {
        for (int s = 0; s < CBLAS_CACHE_SIZE; s++) {
            g_cblas_cache[l].entries[s].expert_id = -1;
        }
    }

    /* Reusable FP32 temp buffer for dequantization */
    size_t max_elems = (size_t)g_cblas_exp_inter * hidden_dim;
    size_t down_elems = (size_t)hidden_dim * g_cblas_exp_inter;
    if (down_elems > max_elems) max_elems = down_elems;
    g_cblas_fp32_temp = (float*)malloc(max_elems * sizeof(float));

    float cache_mb = (float)(n_layers * CBLAS_CACHE_SIZE) *
                     (3.0f * (float)((size_t)g_cblas_exp_inter * hidden_dim) * 4.0f) /
                     (1024.0f * 1024.0f);
    fprintf(stderr, "tq_moe_cblas_cache_init: FP32/AMX LRU cache for %d layers x %d slots "
            "(max %.0f MB)\n", n_layers, CBLAS_CACHE_SIZE, (double)cache_mb);
}

static void cblas_free_entry(cblas_cache_entry_t* e)
{
    free(e->gate_fp32);  e->gate_fp32 = NULL;
    free(e->up_fp32);    e->up_fp32 = NULL;
    free(e->down_fp32);  e->down_fp32 = NULL;
    e->expert_id = -1;
}

void tq_moe_cblas_cache_free(void)
{
    if (!g_cblas_cache) return;
    for (int l = 0; l < g_cblas_n_layers; l++) {
        for (int s = 0; s < CBLAS_CACHE_SIZE; s++) {
            cblas_free_entry(&g_cblas_cache[l].entries[s]);
        }
    }
    free(g_cblas_cache);
    g_cblas_cache = NULL;
    free(g_cblas_fp32_temp);
    g_cblas_fp32_temp = NULL;
    g_cblas_n_layers = 0;
}

/* Find cached FP32 entry or evict LRU and dequant from IQ2_XXS */
static cblas_cache_entry_t* cblas_cache_get_or_create(
    int layer_idx, int expert_id, const tq_expert_weights_t* exp)
{
    cblas_layer_cache_t* lc = &g_cblas_cache[layer_idx];

    /* Cache hit */
    for (int s = 0; s < CBLAS_CACHE_SIZE; s++) {
        if (lc->entries[s].expert_id == expert_id) {
            lc->entries[s].last_used = g_cblas_token;
            return &lc->entries[s];
        }
    }

    /* Cache miss: find empty slot or evict LRU */
    int target = -1;
    if (lc->count < CBLAS_CACHE_SIZE) {
        for (int s = 0; s < CBLAS_CACHE_SIZE; s++) {
            if (lc->entries[s].expert_id < 0) {
                target = s;
                break;
            }
        }
        lc->count++;
    } else {
        int oldest_time = g_cblas_token + 1;
        for (int s = 0; s < CBLAS_CACHE_SIZE; s++) {
            if (lc->entries[s].last_used < oldest_time) {
                oldest_time = lc->entries[s].last_used;
                target = s;
            }
        }
        cblas_free_entry(&lc->entries[target]);
    }

    cblas_cache_entry_t* ce = &lc->entries[target];
    ce->expert_id = expert_id;
    ce->last_used = g_cblas_token;

    int dim   = g_cblas_hidden_dim;
    int inter = g_cblas_exp_inter;

    /* Dequant gate: [inter, dim] -> FP32 */
    {
        int n = inter * dim;
        ce->gate_fp32 = (float*)malloc((size_t)n * sizeof(float));
        if (ce->gate_fp32) {
            tq_dequant_row_gguf(exp->gate_type, exp->w_gate, ce->gate_fp32, n);
        }
    }

    /* Dequant up: [inter, dim] -> FP32 */
    {
        int n = inter * dim;
        ce->up_fp32 = (float*)malloc((size_t)n * sizeof(float));
        if (ce->up_fp32) {
            tq_dequant_row_gguf(exp->up_type, exp->w_up, ce->up_fp32, n);
        }
    }

    /* Dequant down: [dim, inter] -> FP32 */
    {
        int n = dim * inter;
        ce->down_fp32 = (float*)malloc((size_t)n * sizeof(float));
        if (ce->down_fp32) {
            tq_dequant_row_gguf(exp->down_type, exp->w_down, ce->down_fp32, n);
        }
    }

    return ce;
}

#endif /* TQ_HAS_ACCELERATE */

/* ============================================================
 * State management
 * ============================================================ */

tq_moe_state_t* tq_moe_create_state(const tq_moe_config_t* config, int hidden_dim)
{
    tq_moe_state_t* s = (tq_moe_state_t*)calloc(1, sizeof(tq_moe_state_t));
    if (!s) return NULL;

    s->router_logits  = (float*)malloc((size_t)config->num_experts * sizeof(float));
    s->top_experts    = (int*)calloc((size_t)config->num_active, sizeof(int));
    s->expert_weights = (float*)malloc((size_t)config->num_active * sizeof(float));
    s->expert_out     = (float*)malloc((size_t)hidden_dim * sizeof(float));

    /* Workspace buffers sized to the larger of expert / shared-expert intermediate dim */
    int inter = config->expert_intermediate_dim;
    if (config->has_shared_expert && config->shared_expert_intermediate_dim > inter)
        inter = config->shared_expert_intermediate_dim;

    s->expert_hb  = (float*)malloc((size_t)inter * sizeof(float));
    s->expert_hb2 = (float*)malloc((size_t)inter * sizeof(float));

    return s;
}

void tq_moe_free_state(tq_moe_state_t* state)
{
    if (!state) return;
    free(state->router_logits);
    free(state->top_experts);
    free(state->expert_weights);
    free(state->expert_out);
    free(state->expert_hb);
    free(state->expert_hb2);
    free(state);
}

/* ============================================================
 * Top-K expert routing
 * ============================================================ */

void tq_moe_route(const float* hidden, const float* router_weight,
                  int num_experts, int num_active, int hidden_dim,
                  int* out_expert_ids, float* out_expert_weights)
{
    /*
     * We need scratch space for router logits. num_experts can be up to 256,
     * so we heap-allocate to avoid large VLAs on the stack.
     */
    float* logits = (float*)malloc((size_t)num_experts * sizeof(float));
    if (!logits) return;

    /* Step 1: Compute router logits — logits[e] = dot(hidden, router_weight[e]) */
    for (int e = 0; e < num_experts; e++) {
        const float* row = router_weight + (size_t)e * hidden_dim;
        float sum = 0.0f;
        for (int j = 0; j < hidden_dim; j++)
            sum += hidden[j] * row[j];
        logits[e] = sum;
    }

    /* Step 2: Top-K selection via partial sort (K passes, K << num_experts)
     *
     * Use >= for tie-breaking so that when multiple experts have equal logits,
     * the first unused one always wins. Also guard against NaN logits (NaN
     * comparisons return false, so without >= the loop could leave best == -1).
     */
    uint8_t* used = (uint8_t*)calloc((size_t)num_experts, sizeof(uint8_t));
    if (!used) { free(logits); return; }

    int n_valid = 0;
    for (int k = 0; k < num_active; k++) {
        int best = -1;
        float best_val = -HUGE_VALF;
        for (int e = 0; e < num_experts; e++) {
            if (!used[e] && logits[e] >= best_val) {
                best_val = logits[e];
                best = e;
            }
        }
        out_expert_ids[k] = best;
        if (best >= 0) {
            used[best] = 1;
            n_valid++;
        } else {
            out_expert_weights[k] = 0.0f;
        }
    }

    /* Step 3: Softmax over selected experts (renormalize top-K) */
    if (n_valid == 0) {
        /* All experts invalid (NaN logits or num_experts=0) — uniform fallback */
        for (int k = 0; k < num_active; k++) {
            out_expert_weights[k] = 0.0f;
        }
        free(used);
        free(logits);
        return;
    }

    float max_val = -HUGE_VALF;
    for (int k = 0; k < num_active; k++) {
        if (out_expert_ids[k] < 0) continue;
        float v = logits[out_expert_ids[k]];
        if (v > max_val) max_val = v;
    }

    float sum_exp = 0.0f;
    for (int k = 0; k < num_active; k++) {
        if (out_expert_ids[k] < 0) { out_expert_weights[k] = 0.0f; continue; }
        float e = expf(logits[out_expert_ids[k]] - max_val);
        out_expert_weights[k] = e;
        sum_exp += e;
    }

    if (sum_exp > 0.0f) {
        float inv_sum = 1.0f / sum_exp;
        for (int k = 0; k < num_active; k++)
            out_expert_weights[k] *= inv_sum;
    }

    free(used);
    free(logits);
}

/* ============================================================
 * Full MoE FFN forward pass
 * ============================================================ */

void tq_moe_forward(const tq_moe_layer_t* layer,
                    const tq_moe_config_t* config,
                    tq_moe_state_t* state,
                    const float* input, float* output,
                    int hidden_dim, int layer_idx)
{
    int num_active = config->num_active;
    int expert_dim = config->expert_intermediate_dim;

    /* Step 1: Route — select top-K experts */
    tq_moe_route(input, layer->router_weight,
                 config->num_experts, num_active, hidden_dim,
                 state->top_experts, state->expert_weights);

    /* Step 2: Zero the output accumulator */
    memset(output, 0, (size_t)hidden_dim * sizeof(float));

    /* Advance the global token counter for LRU tracking */
    g_token_counter++;
#ifdef TQ_HAS_ACCELERATE
    g_cblas_token++;
#endif

    /* Step 2.5: Try fused MoE Metal dispatch (3 dispatches for all experts).
     * This replaces the entire per-expert loop below when available.
     * Requires: all expert weights are same IQ2_XXS type and contiguous in mmap. */
#ifdef TQ_HAS_METAL
    extern int tq_metal_moe_available(void);
    extern int tq_metal_moe_forward(
        const float* input, float* output, float* hb_output,
        const void* weight_base, size_t weight_size,
        const uint64_t* gate_offsets, const uint64_t* up_offsets, const uint64_t* down_offsets,
        const int* active_expert_ids, const float* expert_routing_weights,
        int num_active, int expert_dim, int hidden_dim, int num_experts_total, int weight_type,
        const int* gate_types, const int* up_types, const int* down_types);

    /* Metal MoE: single command buffer with memoryBarrier between phases.
     * Eliminates per-phase waitUntilCompleted overhead. */
    if (tq_metal_moe_available() && num_active > 0) {
        /* Check that all active experts use IQ2_XXS and have valid weights */
        int can_fuse = 1;
        const void* base_ptr = NULL;
        size_t base_size = 0;

        /* Find the lowest and highest addressed weight to determine the
         * weight region span for the zero-copy Metal buffer. */
        uintptr_t min_addr = ~(uintptr_t)0;
        uintptr_t max_addr = 0;

        for (int k = 0; k < num_active; k++) {
            int eid = state->top_experts[k];
            if (eid < 0 || eid >= config->num_experts) { can_fuse = 0; break; }
            const tq_expert_weights_t* exp = &layer->experts[eid];
            if (exp->q4_converted) {
                can_fuse = 0; break;
            }
            /* Accept IQ2_XXS (16) and IQ2_S (22) — UD models use mixed types */
            int gt = exp->gate_type, ut = exp->up_type, dt = exp->down_type;
            int is_iq2 = (gt == TQ_GGML_TYPE_IQ2_XXS || gt == TQ_GGML_TYPE_IQ2_S) &&
                         (ut == TQ_GGML_TYPE_IQ2_XXS || ut == TQ_GGML_TYPE_IQ2_S) &&
                         (dt == TQ_GGML_TYPE_IQ2_XXS || dt == TQ_GGML_TYPE_IQ2_S);
            if (!is_iq2) {
                can_fuse = 0; break;
            }
            if (!exp->w_gate || !exp->w_up || !exp->w_down) { can_fuse = 0; break; }

            /* Track min/max addresses to compute base and size */
            uintptr_t addrs[3] = {
                (uintptr_t)exp->w_gate, (uintptr_t)exp->w_up, (uintptr_t)exp->w_down
            };
            /* gate/up: [expert_dim, hidden_dim], down: [hidden_dim, expert_dim]
             * Byte sizes depend on quant type: IQ2_XXS=66, IQ2_S=82 per 256 elements */
            int gate_blk = (gt == TQ_GGML_TYPE_IQ2_S) ? 82 : 66;
            int up_blk   = (ut == TQ_GGML_TYPE_IQ2_S) ? 82 : 66;
            int down_blk = (dt == TQ_GGML_TYPE_IQ2_S) ? 82 : 66;
            size_t gate_bytes = (size_t)(expert_dim * (hidden_dim / 256)) * gate_blk;
            size_t up_bytes   = (size_t)(expert_dim * (hidden_dim / 256)) * up_blk;
            size_t down_bytes = (size_t)(hidden_dim * (expert_dim / 256)) * down_blk;
            size_t sizes[3] = { gate_bytes, up_bytes, down_bytes };

            for (int i = 0; i < 3; i++) {
                if (addrs[i] < min_addr) min_addr = addrs[i];
                uintptr_t end = addrs[i] + sizes[i];
                if (end > max_addr) max_addr = end;
            }
        }

        if (can_fuse && min_addr < max_addr) {
            base_ptr = (const void*)min_addr;
            base_size = (size_t)(max_addr - min_addr);

            /* Compute byte offsets from base for each active expert */
            uint64_t gate_offs[8], up_offs[8], down_offs[8];
            int expert_ids[8];
            float routing_w[8];
            int per_gate_types[8], per_up_types[8], per_down_types[8];

            for (int k = 0; k < num_active; k++) {
                int eid = state->top_experts[k];
                const tq_expert_weights_t* exp = &layer->experts[eid];
                expert_ids[k] = eid;
                routing_w[k] = state->expert_weights[k];
                gate_offs[k] = (uint64_t)((uintptr_t)exp->w_gate - min_addr);
                up_offs[k]   = (uint64_t)((uintptr_t)exp->w_up   - min_addr);
                down_offs[k] = (uint64_t)((uintptr_t)exp->w_down - min_addr);
                per_gate_types[k] = (int)exp->gate_type;
                per_up_types[k]   = (int)exp->up_type;
                per_down_types[k] = (int)exp->down_type;
            }

            /* Allocate buffer for GPU SwiGLU results [num_active * expert_dim] */
            float* hb_gpu = (float*)malloc((size_t)num_active * (size_t)expert_dim * sizeof(float));
            if (!hb_gpu) goto moe_cpu_fallback;

            int rc = tq_metal_moe_forward(
                input, output, hb_gpu,
                base_ptr, base_size,
                gate_offs, up_offs, down_offs,
                expert_ids, routing_w,
                num_active, expert_dim, hidden_dim,
                config->num_experts,
                0, /* weight_type=0 means use per-expert types */
                per_gate_types, per_up_types, per_down_types);

            if (rc == 1) {
                /* Hybrid: GPU did gate+up+SwiGLU, we do down+accum on CPU */
                for (int k = 0; k < num_active; k++) {
                    int eid = state->top_experts[k];
                    float w = state->expert_weights[k];
                    const tq_expert_weights_t* exp = &layer->experts[eid];
                    float* hb_k = hb_gpu + k * expert_dim;

                    /* down = hb_k @ w_down^T -> [hidden_dim] */
                    tq_matmul_gguf(state->expert_out, hb_k,
                                   exp->w_down, exp->down_type,
                                   hidden_dim, expert_dim);

                    /* Weighted accumulation: output += weight * down_proj */
                    for (int i = 0; i < hidden_dim; i++)
                        output[i] += w * state->expert_out[i];
                }
                free(hb_gpu);
                goto moe_shared_expert;
            }
            free(hb_gpu);
            if (rc == 0) {
                /* Full GPU success (unlikely in hybrid mode, but handle it) */
                goto moe_shared_expert;
            }
            /* else: rc == -1, fall through to per-expert CPU path */
        }
    }
#endif /* TQ_HAS_METAL */

#ifdef TQ_HAS_METAL
moe_cpu_fallback: ;
#endif
    /* Step 3: For each selected expert, compute SwiGLU FFN and accumulate */
    for (int k = 0; k < num_active; k++) {
        int eid = state->top_experts[k];
        float w = state->expert_weights[k];
        if (eid < 0 || eid >= config->num_experts) continue; /* safety check */
        const tq_expert_weights_t* exp = &layer->experts[eid];

#ifdef TQ_HAS_ACCELERATE
        /* PRIMARY path on Apple: cblas_sgemv via AMX coprocessor.
         * Dequant IQ2_XXS -> FP32 once (LRU cached), then cblas_sgemv.
         * AMX is ~36x faster than manual FP32 dot for expert-sized matrices. */
        /* cblas/AMX DISABLED: IQ2→FP32 dequant cost dominates.
         * With 256 experts/layer, cache miss rate is too high.
         * Fused IQ2 dot (no dequant) is faster overall.
         * cblas would win if we could pre-dequant ALL experts at load time,
         * but 30 layers × 256 experts × 12 MB = 90 GB — impossible. */
        if (0 && g_cblas_cache && layer_idx >= 0 && layer_idx < g_cblas_n_layers
            && exp->w_gate && !exp->q4_converted) {
            cblas_cache_entry_t* ce = cblas_cache_get_or_create(layer_idx, eid, exp);
            if (ce && ce->gate_fp32 && ce->up_fp32 && ce->down_fp32) {
                /* gate: input[hidden_dim] @ gate[expert_dim, hidden_dim]^T -> hb[expert_dim]
                 * cblas_sgemv(RowMajor, NoTrans, M, N, alpha, A, lda, x, incx, beta, y, incy)
                 * A is [M x N] = [expert_dim x hidden_dim], x is [hidden_dim], y is [expert_dim] */
                cblas_sgemv(CblasRowMajor, CblasNoTrans,
                            expert_dim, hidden_dim, 1.0f,
                            ce->gate_fp32, hidden_dim,
                            input, 1, 0.0f, state->expert_hb, 1);

                /* up: same layout */
                cblas_sgemv(CblasRowMajor, CblasNoTrans,
                            expert_dim, hidden_dim, 1.0f,
                            ce->up_fp32, hidden_dim,
                            input, 1, 0.0f, state->expert_hb2, 1);

                /* SwiGLU activation: hb = silu(gate) * up */
                swiglu_fused(state->expert_hb, state->expert_hb2, expert_dim);

                /* down: hb[expert_dim] @ down[hidden_dim, expert_dim]^T -> out[hidden_dim]
                 * A is [hidden_dim x expert_dim], x is [expert_dim], y is [hidden_dim] */
                cblas_sgemv(CblasRowMajor, CblasNoTrans,
                            hidden_dim, expert_dim, 1.0f,
                            ce->down_fp32, expert_dim,
                            state->expert_hb, 1, 0.0f, state->expert_out, 1);

                /* Weighted accumulation: output += weight * down_proj */
                for (int i = 0; i < hidden_dim; i++)
                    output[i] += w * state->expert_out[i];
                continue;
            }
        }
#endif /* TQ_HAS_ACCELERATE */

        /* Q8 LRU cache DISABLED: cache miss conversion cost (IQ2->FP32->Q8)
         * exceeds fused IQ2 dot cost. Direct fused_dot_iq2_xxs_neon is faster
         * than any cache scheme when expert reuse rate is low. */
        if (0 && g_expert_cache && layer_idx >= 0 && layer_idx < g_cache_n_layers
            && exp->w_gate && !exp->q4_converted) {
            expert_cache_entry_t* ce = cache_get_or_create(layer_idx, eid, exp);
            if (ce && ce->gate_q8 && ce->up_q8 && ce->down_q8) {
                /* Fast Q8_0 matmul path — dispatches to fused_dot_q8_0 (NEON) */
                tq_matmul_gguf(state->expert_hb, input,
                               ce->gate_q8, TQ_GGML_TYPE_Q8_0,
                               expert_dim, hidden_dim);
                tq_matmul_gguf(state->expert_hb2, input,
                               ce->up_q8, TQ_GGML_TYPE_Q8_0,
                               expert_dim, hidden_dim);

                /* SwiGLU activation: hb = silu(gate) * up */
                swiglu_fused(state->expert_hb, state->expert_hb2, expert_dim);

                tq_matmul_gguf(state->expert_out, state->expert_hb,
                               ce->down_q8, TQ_GGML_TYPE_Q8_0,
                               hidden_dim, expert_dim);

                /* Weighted accumulation: output += weight * down_proj */
                for (int i = 0; i < hidden_dim; i++)
                    output[i] += w * state->expert_out[i];
                continue;
            }
        }

        if (exp->q4_converted) {
            /* Fast Q4 matmul path — pre-converted expert weights (shared expert)
             * tq_matmul_q4(out, x, w_qs, w_scales, n=out_rows, d=in_cols) */
            tq_matmul_q4(state->expert_hb, input,
                         exp->gate_q4_qs, exp->gate_q4_scales,
                         expert_dim, hidden_dim);
            tq_matmul_q4(state->expert_hb2, input,
                         exp->up_q4_qs, exp->up_q4_scales,
                         expert_dim, hidden_dim);

            /* SwiGLU activation: hb = silu(gate) * up */
            swiglu_fused(state->expert_hb, state->expert_hb2, expert_dim);

            tq_matmul_q4(state->expert_out, state->expert_hb,
                         exp->down_q4_qs, exp->down_q4_scales,
                         hidden_dim, expert_dim);
        } else {
            /* On-the-fly GGUF dequant path (IQ2_XXS/IQ2_S).
             * Metal batch DISABLED for MoE experts: GPU IQ2 dequant
             * overhead exceeds CPU NEON fused dot speed for small 512×2048 matmuls. */

            /* gate = input @ w_gate^T   -> [expert_dim] */
            tq_matmul_gguf(state->expert_hb, input,
                           exp->w_gate, exp->gate_type,
                           expert_dim, hidden_dim);

            /* up = input @ w_up^T   -> [expert_dim] */
            tq_matmul_gguf(state->expert_hb2, input,
                           exp->w_up, exp->up_type,
                           expert_dim, hidden_dim);

            /* SwiGLU activation: hb = silu(gate) * up */
            swiglu_fused(state->expert_hb, state->expert_hb2, expert_dim);

            /* down = hb @ w_down^T   -> [hidden_dim] */
            tq_matmul_gguf(state->expert_out, state->expert_hb,
                           exp->w_down, exp->down_type,
                           hidden_dim, expert_dim);
        }

        /* Weighted accumulation: output += weight * scale * down_proj
         * Gemma 4 has per-expert output scaling (ffn_down_exps.scale). */
        float exp_scale = (layer->expert_scale) ? layer->expert_scale[eid] : 1.0f;
        float ws = w * exp_scale;
        for (int i = 0; i < hidden_dim; i++)
            output[i] += ws * state->expert_out[i];
    }

#ifdef TQ_HAS_METAL
moe_shared_expert:
#endif
    /* Step 4: Shared expert (always-active, if present) */
    if (config->has_shared_expert) {
        int shared_dim = config->shared_expert_intermediate_dim;
        if (shared_dim == 0) shared_dim = expert_dim;

        /* Optional shared expert gating (sigmoid scalar gate) */
        float shared_gate_val = 1.0f;
        if (layer->shared_gate) {
            float dot = 0.0f;
            for (int j = 0; j < hidden_dim; j++)
                dot += input[j] * layer->shared_gate[j];
            shared_gate_val = 1.0f / (1.0f + fast_expf_moe(-dot)); /* sigmoid */
        }

        if (layer->shared_expert.q4_converted) {
            /* Fast Q4 path for shared expert
             * tq_matmul_q4(out, x, w_qs, w_scales, n=out_rows, d=in_cols) */
            tq_matmul_q4(state->expert_hb, input,
                         layer->shared_expert.gate_q4_qs, layer->shared_expert.gate_q4_scales,
                         shared_dim, hidden_dim);
            tq_matmul_q4(state->expert_hb2, input,
                         layer->shared_expert.up_q4_qs, layer->shared_expert.up_q4_scales,
                         shared_dim, hidden_dim);

            swiglu_fused(state->expert_hb, state->expert_hb2, shared_dim);

            tq_matmul_q4(state->expert_out, state->expert_hb,
                         layer->shared_expert.down_q4_qs, layer->shared_expert.down_q4_scales,
                         hidden_dim, shared_dim);
        } else {
            /* Fallback: on-the-fly GGUF dequant */
            tq_metal_batch_begin_if_available();

            tq_matmul_gguf(state->expert_hb, input,
                           layer->shared_expert.w_gate, layer->shared_expert.gate_type,
                           shared_dim, hidden_dim);

            tq_matmul_gguf(state->expert_hb2, input,
                           layer->shared_expert.w_up, layer->shared_expert.up_type,
                           shared_dim, hidden_dim);

            tq_metal_batch_flush_if_available();

            swiglu_fused(state->expert_hb, state->expert_hb2, shared_dim);

            tq_matmul_gguf(state->expert_out, state->expert_hb,
                           layer->shared_expert.w_down, layer->shared_expert.down_type,
                           hidden_dim, shared_dim);
        }

        for (int i = 0; i < hidden_dim; i++)
            output[i] += shared_gate_val * state->expert_out[i];
    }
}

/* ============================================================
 * Expert memory advise (madvise hints for paging)
 * ============================================================ */

void tq_moe_advise(const tq_moe_layer_t* layer,
                   const int* active_ids, int n_active,
                   int num_experts)
{
    /*
     * TODO: Implement madvise(MADV_WILLNEED) for active experts and
     *       madvise(MADV_DONTNEED) for inactive experts once tensor
     *       size information is available in tq_expert_weights_t.
     *
     * The idea is:
     *   - For each active expert, call madvise(MADV_WILLNEED) on
     *     w_gate, w_up, w_down data regions to prefetch pages.
     *   - For inactive experts, optionally call madvise(MADV_DONTNEED)
     *     to allow the OS to reclaim those pages.
     *
     * This requires knowing the byte size of each weight tensor,
     * which currently isn't stored in tq_expert_weights_t.
     */
    (void)layer;
    (void)active_ids;
    (void)n_active;
    (void)num_experts;
}
