/**
 * TurboQuant — Metal backend dispatch (Objective-C host code)
 *
 * Loads the .metallib shader library, creates compute pipelines,
 * and provides the dispatch interface for Metal GPU kernels.
 *
 * Includes matmul dispatch for GGUF quantized weight formats
 * (IQ2_XXS, Q8_0, Q4_K) with buffer caching for MoE workloads.
 *
 * Supports two dispatch modes:
 *   1. Immediate mode (default): each matmul dispatches and waits.
 *   2. Batched mode: multiple matmuls encoded into one command buffer,
 *      committed and waited once at flush. Reduces dispatch overhead
 *      by ~730x for MoE models with many small matmuls per token.
 *
 * MLX Metal patterns applied to MoE dispatch:
 *   P1: Cached intermediate buffers — gate/up/input/output/params buffers
 *       are allocated once and grown as needed, eliminating per-call
 *       Metal buffer creation overhead (~0.1ms × 30 layers = 3ms saved).
 *   P2: Conditional barriers — barriers only between phases within a
 *       layer (P1→P2, P2→P3), never between layers since they use
 *       independent buffers. Cross-encoder ordering is implicit in Metal.
 *   P3: Reduced threadgroup size — MoE matmul kernels use 64 threads
 *       (vs 256 for dense matmul) matching MLX's QMV pattern for small
 *       matrices, reducing synchronization overhead.
 *
 * Uses zero-copy weight buffers (newBufferWithBytesNoCopy) on Apple
 * Silicon unified memory to eliminate weight upload overhead.
 */
#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "turboquant/tq_gguf.h"
#include <string.h>

/* Pipeline cache */
static id<MTLDevice>       tq_mtl_device    = nil;
static id<MTLCommandQueue> tq_mtl_queue     = nil;
static id<MTLLibrary>      tq_mtl_library   = nil;

/* Cached pipelines — KV cache quantization */
static id<MTLComputePipelineState> tq_pipe_polar_quantize  = nil;
static id<MTLComputePipelineState> tq_pipe_polar_attention  = nil;
static id<MTLComputePipelineState> tq_pipe_qjl_quantize    = nil;
static id<MTLComputePipelineState> tq_pipe_qjl_attention   = nil;
static id<MTLComputePipelineState> tq_pipe_value_quantize  = nil;

/* Cached pipelines — matmul kernels */
static id<MTLComputePipelineState> tq_pipe_matmul_iq2_xxs  = nil;
static id<MTLComputePipelineState> tq_pipe_matmul_iq2_s    = nil;
static id<MTLComputePipelineState> tq_pipe_matmul_q8_0     = nil;
static id<MTLComputePipelineState> tq_pipe_matmul_q4_k     = nil;
static id<MTLComputePipelineState> tq_pipe_matmul_tq_q4   = nil;
static id<MTLComputePipelineState> tq_pipe_matmul_tq_q4_repacked = nil;

/* Cached pipelines — element-wise kernels */
static id<MTLComputePipelineState> tq_pipe_rmsnorm         = nil;
static id<MTLComputePipelineState> tq_pipe_silu            = nil;
static id<MTLComputePipelineState> tq_pipe_mul_elementwise = nil;
static id<MTLComputePipelineState> tq_pipe_add_vectors     = nil;
static id<MTLComputePipelineState> tq_pipe_add_inplace     = nil;

/* Cached pipelines — compute graph kernels (full-layer forward) */
static id<MTLComputePipelineState> tq_pipe_rope            = nil;
static id<MTLComputePipelineState> tq_pipe_gelu_tanh       = nil;
static id<MTLComputePipelineState> tq_pipe_softmax         = nil;
static id<MTLComputePipelineState> tq_pipe_attn_qk         = nil;
static id<MTLComputePipelineState> tq_pipe_attn_v          = nil;
static id<MTLComputePipelineState> tq_pipe_kv_cache_write  = nil;

/* Cached pipelines — fused MoE kernels */
static id<MTLComputePipelineState> tq_pipe_moe_gate_up     = nil;
static id<MTLComputePipelineState> tq_pipe_moe_swiglu      = nil;
static id<MTLComputePipelineState> tq_pipe_moe_down_accum  = nil;

/* IQ2_S codebook buffer — shared between matmul and MoE kernels */
static id<MTLBuffer> tq_iq2s_grid_buf = nil;

/* ============================================================
 * Zero-copy weight buffer cache
 *
 * On Apple Silicon, CPU and GPU share unified memory. We use
 * newBufferWithBytesNoCopy: to wrap mmap'd weight data directly
 * as Metal buffers without any memcpy. The cache maps weight
 * pointers to their MTLBuffer wrappers.
 * ============================================================ */

#define TQ_WEIGHT_CACHE_SIZE 128

typedef struct {
    const void*   ptr;
    size_t        size;
    id<MTLBuffer> buf;
} tq_weight_cache_entry_t;

static tq_weight_cache_entry_t tq_weight_cache[TQ_WEIGHT_CACHE_SIZE];
static int tq_weight_cache_count = 0;

/**
 * Get or create a zero-copy Metal buffer wrapping a weight pointer.
 * Falls back to newBufferWithBytes if zero-copy fails (e.g., unaligned).
 */
static id<MTLBuffer> tq_get_weight_buffer(const void* weight, size_t weight_size) {
    /* Search existing cache */
    for (int i = 0; i < tq_weight_cache_count; i++) {
        if (tq_weight_cache[i].ptr == weight &&
            tq_weight_cache[i].size == weight_size) {
            return tq_weight_cache[i].buf;
        }
    }

    /* Create new zero-copy buffer.
     * Page-align the pointer down and adjust offset/length.
     * newBufferWithBytesNoCopy requires page-aligned address and length. */
    id<MTLBuffer> buf = nil;
    size_t page_size = 16384; /* ARM64 page size on macOS */
    uintptr_t addr = (uintptr_t)weight;
    uintptr_t aligned_addr = addr & ~(page_size - 1);
    size_t offset = addr - aligned_addr;
    size_t aligned_size = ((offset + weight_size + page_size - 1) / page_size) * page_size;

    buf = [tq_mtl_device newBufferWithBytesNoCopy:(void*)aligned_addr
                                           length:aligned_size
                                          options:MTLResourceStorageModeShared
                                      deallocator:nil];
    if (buf && offset > 0) {
        /* We need the buffer to start at the actual weight pointer.
         * Metal doesn't support sub-buffer offsets in newBufferWithBytesNoCopy,
         * but since we pass offset via setBuffer:offset: at encode time,
         * we store the offset and use it later. For simplicity, just use
         * the copy path if there's an offset issue. */
        /* Actually, we can store the base buffer and use offset at bind time.
         * But the current API uses setBuffer:offset:0. For now, if the pointer
         * is page-aligned, zero-copy works directly. Otherwise, fall back. */
        if (offset != 0) {
            /* Pointer not page-aligned — fall back to copy */
            buf = [tq_mtl_device newBufferWithBytes:weight
                                             length:weight_size
                                            options:MTLResourceStorageModeShared];
        }
    }

    if (!buf) {
        /* Zero-copy failed — fall back to memcpy */
        buf = [tq_mtl_device newBufferWithBytes:weight
                                         length:weight_size
                                        options:MTLResourceStorageModeShared];
    }

    if (!buf) return nil;

    /* Add to cache (evict oldest if full) */
    if (tq_weight_cache_count < TQ_WEIGHT_CACHE_SIZE) {
        tq_weight_cache[tq_weight_cache_count].ptr = weight;
        tq_weight_cache[tq_weight_cache_count].size = weight_size;
        tq_weight_cache[tq_weight_cache_count].buf = buf;
        tq_weight_cache_count++;
    } else {
        /* Evict slot 0, shift down */
        tq_weight_cache[0].buf = nil;
        for (int i = 0; i < TQ_WEIGHT_CACHE_SIZE - 1; i++) {
            tq_weight_cache[i] = tq_weight_cache[i + 1];
        }
        tq_weight_cache[TQ_WEIGHT_CACHE_SIZE - 1].ptr = weight;
        tq_weight_cache[TQ_WEIGHT_CACHE_SIZE - 1].size = weight_size;
        tq_weight_cache[TQ_WEIGHT_CACHE_SIZE - 1].buf = buf;
    }

    return buf;
}

/* ============================================================
 * Batch mode state
 *
 * When batch mode is active, matmul calls encode into a shared
 * command buffer without committing. The caller must flush to
 * commit the command buffer and copy results.
 *
 * Each batched matmul gets its own output buffer (since they
 * write to different CPU destinations). We track pending copies
 * to perform after GPU completion.
 * ============================================================ */

#define TQ_BATCH_MAX_OPS 64

typedef struct {
    float*        cpu_dst;     /* CPU destination for memcpy */
    id<MTLBuffer> gpu_buf;     /* GPU output buffer */
    size_t        size;        /* bytes to copy */
} tq_batch_pending_copy_t;

typedef struct {
    int                      active;       /* 1 if batch mode is on */
    id<MTLCommandBuffer>     cmd_buf;      /* shared command buffer */
    id<MTLComputeCommandEncoder> encoder;  /* shared encoder */
    tq_batch_pending_copy_t  copies[TQ_BATCH_MAX_OPS];
    int                      n_copies;
} tq_batch_state_t;

static tq_batch_state_t tq_batch = {
    .active = 0, .cmd_buf = nil, .encoder = nil, .n_copies = 0
};

/* ============================================================
 * Cached buffer pools for batch mode
 *
 * Eliminates per-dispatch Metal buffer allocation overhead.
 * In batch mode, each dispatch previously allocated:
 *   - 1 output buffer (~4KB-16KB)
 *   - 2 dimension uniform buffers (4 bytes each)
 * With 30+ dispatches per layer * 35 layers, this caused
 * massive allocation churn (~2000+ allocations per token).
 *
 * Solution: pre-allocated pools that grow as needed.
 * Output buffers are pooled by slot index (up to TQ_BATCH_MAX_OPS).
 * Dimension buffers are cached by value (small lookup table).
 * ============================================================ */

/* Output buffer pool: one buffer per batch slot, grow-only */
static id<MTLBuffer> tq_batch_output_pool[TQ_BATCH_MAX_OPS];
static size_t        tq_batch_output_pool_size[TQ_BATCH_MAX_OPS];

/* Dimension uniform buffer cache: maps dimension value → MTLBuffer.
 * Typical models use 3-5 distinct dimension values (hidden_dim,
 * intermediate_dim, kv_dim, head_dim, etc.), so a small cache suffices. */
#define TQ_DIM_CACHE_SIZE 16

typedef struct {
    uint32_t      dim_value;
    id<MTLBuffer> buf;
} tq_dim_cache_entry_t;

static tq_dim_cache_entry_t tq_dim_cache[TQ_DIM_CACHE_SIZE];
static int tq_dim_cache_count = 0;

/**
 * Get or create a cached dimension uniform buffer for a given value.
 * Thread-safe for single-threaded Metal dispatch (which this is).
 */
static id<MTLBuffer> tq_get_dim_buffer(uint32_t dim_value) {
    /* Search cache */
    for (int i = 0; i < tq_dim_cache_count; i++) {
        if (tq_dim_cache[i].dim_value == dim_value) {
            return tq_dim_cache[i].buf;
        }
    }

    /* Create new buffer */
    id<MTLBuffer> buf = [tq_mtl_device newBufferWithLength:sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
    if (!buf) return nil;
    *(uint32_t*)[buf contents] = dim_value;

    /* Add to cache (evict oldest if full) */
    if (tq_dim_cache_count < TQ_DIM_CACHE_SIZE) {
        tq_dim_cache[tq_dim_cache_count].dim_value = dim_value;
        tq_dim_cache[tq_dim_cache_count].buf = buf;
        tq_dim_cache_count++;
    } else {
        /* Evict slot 0 */
        tq_dim_cache[0].buf = nil;
        for (int i = 0; i < TQ_DIM_CACHE_SIZE - 1; i++) {
            tq_dim_cache[i] = tq_dim_cache[i + 1];
        }
        tq_dim_cache[TQ_DIM_CACHE_SIZE - 1].dim_value = dim_value;
        tq_dim_cache[TQ_DIM_CACHE_SIZE - 1].buf = buf;
    }

    return buf;
}

/**
 * Get or grow a cached output buffer for a given batch slot index.
 * Buffers grow monotonically — never shrink.
 */
static id<MTLBuffer> tq_get_batch_output_buffer(int slot, size_t required_size) {
    if (slot < 0 || slot >= TQ_BATCH_MAX_OPS) return nil;

    if (tq_batch_output_pool_size[slot] < required_size || !tq_batch_output_pool[slot]) {
        tq_batch_output_pool[slot] = [tq_mtl_device
            newBufferWithLength:required_size
                        options:MTLResourceStorageModeShared];
        if (!tq_batch_output_pool[slot]) return nil;
        tq_batch_output_pool_size[slot] = required_size;
    }

    return tq_batch_output_pool[slot];
}

/* Reusable input/dimension buffers (shared across batch and immediate modes) */
static id<MTLBuffer> tq_shared_input_buf  = nil;
static uint32_t      tq_shared_input_dim  = 0;
static id<MTLBuffer> tq_shared_indim_buf  = nil;
static id<MTLBuffer> tq_shared_outdim_buf = nil;

/* Threadgroup size for matmul kernels — must match shader constant */
static const uint32_t TQ_MATMUL_TG_SIZE = 256;

/* MLX Pattern 3: Reduced threadgroup size for MoE kernels.
 * MLX's QMV uses 32-64 threads for small matrices. MoE expert matmuls
 * are smaller than dense layers, so fewer threads = less sync overhead.
 * reduce_sum() adapts dynamically via (tg_size + 31) / 32. */
static const uint32_t TQ_MOE_TG_SIZE = 64;

/* ============================================================
 * MLX Pattern 1: Cached intermediate buffers for MoE dispatch
 *
 * Instead of allocating new Metal buffers on every tq_metal_moe_forward()
 * call (30 layers × ~5 buffers = 150 allocations per token), we cache
 * them as statics and only reallocate when a larger size is needed.
 * This eliminates ~3ms of Metal buffer creation overhead per token.
 * ============================================================ */
static id<MTLBuffer> tq_moe_gate_buf   = nil;
static size_t        tq_moe_gate_size   = 0;
static id<MTLBuffer> tq_moe_up_buf     = nil;
static size_t        tq_moe_up_size     = 0;
static id<MTLBuffer> tq_moe_input_buf  = nil;
static size_t        tq_moe_input_size  = 0;
static id<MTLBuffer> tq_moe_output_buf = nil;
static size_t        tq_moe_output_size = 0;
static id<MTLBuffer> tq_moe_params_buf = nil;

/**
 * Initialize Metal backend.
 * Returns 0 on success, -1 on failure.
 */
int tq_init_metal_backend(void) {
    @autoreleasepool {
        /* Get default Metal device */
        tq_mtl_device = MTLCreateSystemDefaultDevice();
        if (!tq_mtl_device) {
            NSLog(@"TurboQuant: No Metal device found");
            return -1;
        }

        /* Create command queue */
        tq_mtl_queue = [tq_mtl_device newCommandQueue];
        if (!tq_mtl_queue) {
            NSLog(@"TurboQuant: Failed to create command queue");
            return -1;
        }

        /* Load shader library: try metallib first, then runtime compile from source */
        NSError *error = nil;

        /* Try pre-compiled metallib */
        NSString *libPath = [[NSBundle mainBundle] pathForResource:@"turboquant"
                                                           ofType:@"metallib"];
        if (libPath) {
            NSURL *libURL = [NSURL fileURLWithPath:libPath];
            tq_mtl_library = [tq_mtl_device newLibraryWithURL:libURL error:&error];
        }
        if (!tq_mtl_library) {
            tq_mtl_library = [tq_mtl_device newDefaultLibrary];
        }

        /* Fallback: runtime compile from .metal source files */
        if (!tq_mtl_library) {
            /* Find the matmul shader source file relative to executable */
            NSString *exePath = [[NSProcessInfo processInfo] arguments][0];
            NSString *exeDir = [exePath stringByDeletingLastPathComponent];

            /* Search paths for the metal source */
            NSArray *searchPaths = @[
                [exeDir stringByAppendingPathComponent:@"../src/backend/metal/tq_matmul.metal"],
                @"src/backend/metal/tq_matmul.metal",
                @"../src/backend/metal/tq_matmul.metal",
            ];

            NSString *sourceCode = nil;
            for (NSString *path in searchPaths) {
                sourceCode = [NSString stringWithContentsOfFile:path
                                                      encoding:NSUTF8StringEncoding
                                                         error:nil];
                if (sourceCode) {
                    NSLog(@"TurboQuant: Compiling Metal shaders from %@", path);
                    break;
                }
            }

            if (sourceCode) {
                MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
                if (@available(macOS 15.0, *)) {
                    opts.mathMode = MTLMathModeFast;
                } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                    opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
                }
                tq_mtl_library = [tq_mtl_device newLibraryWithSource:sourceCode
                                                             options:opts
                                                               error:&error];
                if (!tq_mtl_library) {
                    NSLog(@"TurboQuant: Metal shader compile failed: %@", error);
                    return -1;
                }
                NSLog(@"TurboQuant: Metal shaders compiled successfully");
            } else {
                NSLog(@"TurboQuant: No Metal library or source found");
                return -1;
            }
        }

        /* Helper block: create pipeline from kernel name */
        id<MTLComputePipelineState> (^makePipe)(NSString *) = ^(NSString *name) {
            id<MTLFunction> func = [tq_mtl_library newFunctionWithName:name];
            if (!func) return (id<MTLComputePipelineState>)nil;
            NSError *pipeErr = nil;
            id<MTLComputePipelineState> pipe =
                [tq_mtl_device newComputePipelineStateWithFunction:func error:&pipeErr];
            if (pipeErr) {
                NSLog(@"TurboQuant: Pipeline error for %@: %@", name, pipeErr);
            }
            return pipe;
        };

        /* Create compute pipelines — KV cache */
        tq_pipe_polar_quantize = makePipe(@"tq_polar_quantize");
        tq_pipe_polar_attention = makePipe(@"tq_polar_attention");
        tq_pipe_qjl_quantize = makePipe(@"tq_qjl_quantize");
        tq_pipe_qjl_attention = makePipe(@"tq_qjl_attention");
        tq_pipe_value_quantize = makePipe(@"tq_value_quantize_4b");

        /* Create compute pipelines — matmul */
        tq_pipe_matmul_iq2_xxs = makePipe(@"matmul_iq2_xxs");
        tq_pipe_matmul_iq2_s   = makePipe(@"matmul_iq2_s");
        tq_pipe_matmul_q8_0 = makePipe(@"matmul_q8_0");
        tq_pipe_matmul_q4_k = makePipe(@"matmul_q4_k");
        tq_pipe_matmul_tq_q4 = makePipe(@"matmul_tq_q4");
        tq_pipe_matmul_tq_q4_repacked = makePipe(@"matmul_tq_q4_fast");

        /* Create compute pipelines — element-wise ops */
        tq_pipe_rmsnorm         = makePipe(@"rmsnorm");
        tq_pipe_silu            = makePipe(@"silu");
        tq_pipe_mul_elementwise = makePipe(@"mul_elementwise");
        tq_pipe_add_vectors     = makePipe(@"add_vectors");
        tq_pipe_add_inplace     = makePipe(@"add_inplace");

        /* Create compute pipelines — compute graph kernels */
        tq_pipe_rope            = makePipe(@"rope");
        tq_pipe_gelu_tanh       = makePipe(@"gelu_tanh");
        tq_pipe_softmax         = makePipe(@"softmax_inplace");
        tq_pipe_attn_qk         = makePipe(@"attention_qk");
        tq_pipe_kv_cache_write  = makePipe(@"kv_cache_write");
        tq_pipe_attn_v          = makePipe(@"attention_v");

        /* Create IQ2_S codebook buffer (shared by matmul and MoE kernels) */
        {
            const uint64_t* grid = tq_iq2s_grid();
            tq_iq2s_grid_buf = [tq_mtl_device newBufferWithBytes:grid
                                                          length:1024 * sizeof(uint64_t)
                                                         options:MTLResourceStorageModeShared];
        }

        /* Create compute pipelines — fused MoE.
         * If the MoE kernel functions aren't in the main library
         * (i.e., compiled from tq_matmul.metal only), load and compile
         * the MoE shader source separately, then extract pipelines. */
        tq_pipe_moe_gate_up    = makePipe(@"moe_gate_up_fused");
        tq_pipe_moe_swiglu     = makePipe(@"moe_swiglu");
        tq_pipe_moe_down_accum = makePipe(@"moe_down_accum");

        if (!tq_pipe_moe_gate_up) {
            /* MoE kernels not in main library — try separate compile */
            NSString *exePath2 = [[NSProcessInfo processInfo] arguments][0];
            NSString *exeDir2 = [exePath2 stringByDeletingLastPathComponent];
            NSArray *moePaths = @[
                [exeDir2 stringByAppendingPathComponent:@"../src/backend/metal/tq_moe_kernel.metal"],
                @"src/backend/metal/tq_moe_kernel.metal",
                @"../src/backend/metal/tq_moe_kernel.metal",
            ];
            for (NSString *moePath in moePaths) {
                NSString *moeSrc = [NSString stringWithContentsOfFile:moePath
                                                            encoding:NSUTF8StringEncoding
                                                               error:nil];
                if (moeSrc) {
                    MTLCompileOptions *moeOpts = [[MTLCompileOptions alloc] init];
                    if (@available(macOS 15.0, *)) {
                        moeOpts.mathMode = MTLMathModeFast;
                    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                        moeOpts.fastMathEnabled = YES;
#pragma clang diagnostic pop
                    }
                    NSError *moeErr = nil;
                    id<MTLLibrary> moeLib = [tq_mtl_device newLibraryWithSource:moeSrc
                                                                        options:moeOpts
                                                                          error:&moeErr];
                    if (moeLib) {
                        id<MTLComputePipelineState> (^moePipe)(NSString *) = ^(NSString *name) {
                            id<MTLFunction> func = [moeLib newFunctionWithName:name];
                            if (!func) return (id<MTLComputePipelineState>)nil;
                            NSError *pErr = nil;
                            return [tq_mtl_device newComputePipelineStateWithFunction:func error:&pErr];
                        };
                        tq_pipe_moe_gate_up    = moePipe(@"moe_gate_up_fused");
                        tq_pipe_moe_swiglu     = moePipe(@"moe_swiglu");
                        tq_pipe_moe_down_accum = moePipe(@"moe_down_accum");
                        if (tq_pipe_moe_gate_up) {
                            NSLog(@"TurboQuant: MoE fused kernels compiled from %@", moePath);
                        }
                    } else {
                        NSLog(@"TurboQuant: MoE shader compile failed: %@", moeErr);
                    }
                    break;
                }
            }
        }

        if (tq_pipe_moe_gate_up) {
            NSLog(@"TurboQuant: Fused MoE kernels ready (3 dispatches per layer)");
        } else {
            NSLog(@"TurboQuant: Fused MoE kernels NOT available — using batched matmul fallback");
        }

        /* Create shared dimension uniform buffers */
        tq_shared_indim_buf = [tq_mtl_device
            newBufferWithLength:sizeof(uint32_t)
                        options:MTLResourceStorageModeShared];
        tq_shared_outdim_buf = [tq_mtl_device
            newBufferWithLength:sizeof(uint32_t)
                        options:MTLResourceStorageModeShared];

        NSLog(@"TurboQuant: Metal backend initialized on %@", tq_mtl_device.name);
        return 0;
    }
}

/**
 * Free Metal resources.
 */
void tq_free_metal_backend(void) {
    /* Flush any pending batch */
    if (tq_batch.active) {
        tq_batch.active = 0;
        if (tq_batch.encoder) {
            [tq_batch.encoder endEncoding];
            tq_batch.encoder = nil;
        }
        tq_batch.cmd_buf = nil;
        tq_batch.n_copies = 0;
    }

    /* KV cache pipelines */
    tq_pipe_polar_quantize = nil;
    tq_pipe_polar_attention = nil;
    tq_pipe_qjl_quantize = nil;
    tq_pipe_qjl_attention = nil;
    tq_pipe_value_quantize = nil;

    /* Matmul pipelines */
    tq_pipe_matmul_iq2_xxs = nil;
    tq_pipe_matmul_iq2_s   = nil;
    tq_pipe_matmul_q8_0 = nil;
    tq_pipe_matmul_q4_k = nil;
    tq_pipe_matmul_tq_q4 = nil;

    /* Element-wise pipelines */
    tq_pipe_rmsnorm = nil;
    tq_pipe_silu = nil;
    tq_pipe_mul_elementwise = nil;
    tq_pipe_add_vectors = nil;
    tq_pipe_add_inplace = nil;

    /* Compute graph pipelines */
    tq_pipe_rope = nil;
    tq_pipe_gelu_tanh = nil;
    tq_pipe_softmax = nil;
    tq_pipe_attn_qk = nil;
    tq_pipe_attn_v = nil;

    /* MoE pipelines */
    tq_pipe_moe_gate_up = nil;
    tq_pipe_moe_swiglu = nil;
    tq_pipe_moe_down_accum = nil;

    /* IQ2_S codebook buffer */
    tq_iq2s_grid_buf = nil;

    /* Weight cache */
    for (int i = 0; i < tq_weight_cache_count; i++) {
        tq_weight_cache[i].buf = nil;
        tq_weight_cache[i].ptr = NULL;
        tq_weight_cache[i].size = 0;
    }
    tq_weight_cache_count = 0;

    /* Shared buffers */
    tq_shared_input_buf = nil;
    tq_shared_input_dim = 0;
    tq_shared_indim_buf = nil;
    tq_shared_outdim_buf = nil;

    /* Batch output buffer pool */
    for (int i = 0; i < TQ_BATCH_MAX_OPS; i++) {
        tq_batch_output_pool[i] = nil;
        tq_batch_output_pool_size[i] = 0;
    }

    /* Dimension buffer cache */
    for (int i = 0; i < tq_dim_cache_count; i++) {
        tq_dim_cache[i].buf = nil;
        tq_dim_cache[i].dim_value = 0;
    }
    tq_dim_cache_count = 0;

    tq_mtl_library = nil;
    tq_mtl_queue = nil;
    tq_mtl_device = nil;
}

/**
 * Get Metal device name.
 */
const char* tq_metal_device_name(void) {
    if (!tq_mtl_device) return "not initialized";
    return [[tq_mtl_device name] UTF8String];
}

/**
 * Check if Metal backend is available and initialized.
 */
int tq_metal_available(void) {
    /* TEMP DEBUG: force Metal OFF to isolate Phi-3.5 garbage output.
     * If this produces correct output for Phi-3.5, the issue is in Metal
     * initialization or GPU buffer allocation, not in matmul code. */
    /* TQ_NO_METAL=1: disable Metal GPU entirely (Phi-3.5 workaround).
     * TQ_NO_METAL_COMPUTE=1: init Metal but skip GPU compute (for debugging). */
    static int force_off = -1;
    if (force_off < 0) {
        force_off = (getenv("TQ_NO_METAL") != NULL) ? 1 : 0;
    }
    if (force_off) return 0;

    /* Lazy initialization: first call triggers Metal setup */
    static int init_done = 0;
    if (!init_done) {
        init_done = 1;
        tq_init_metal_backend();
    }
    return (tq_mtl_device != nil && tq_mtl_queue != nil && tq_mtl_library != nil) ? 1 : 0;
}

/* ============================================================
 * Batch mode API
 *
 * tq_metal_batch_begin()  — Start batching matmul encodes.
 * tq_metal_batch_flush()  — Commit command buffer, wait, copy results.
 *
 * Between begin and flush, tq_metal_matmul_gguf() encodes compute
 * commands without committing. Each matmul gets its own output
 * buffer so multiple matmuls can write in parallel on GPU.
 *
 * If batch is full (TQ_BATCH_MAX_OPS), an automatic flush occurs.
 * ============================================================ */

/**
 * Begin a new batch. If a batch is already active, it is flushed first.
 */
void tq_metal_batch_begin(void) {
    if (tq_batch.active && tq_batch.n_copies > 0) {
        /* Auto-flush previous batch */
        extern void tq_metal_batch_flush(void);
        tq_metal_batch_flush();
    }

    tq_batch.active = 1;
    tq_batch.cmd_buf = nil;
    tq_batch.encoder = nil;
    tq_batch.n_copies = 0;
}

/**
 * Flush the current batch: end encoding, commit, wait, copy results.
 * Safe to call when no batch is active (no-op).
 */
/* Diagnostic counters (Issue #16: dispatch overhead investigation).
 * Reset by callers via tq_metal_diag_reset(); read via tq_metal_diag_get().
 * No-op when batch path isn't taken — overhead is one atomic add per flush. */
static unsigned long g_metal_flush_count    = 0;
static unsigned long g_metal_flush_op_count = 0; /* total dispatched ops across flushes */

void tq_metal_diag_reset(void) {
    g_metal_flush_count = 0;
    g_metal_flush_op_count = 0;
}

void tq_metal_diag_get(unsigned long* flushes, unsigned long* ops) {
    if (flushes) *flushes = g_metal_flush_count;
    if (ops)     *ops     = g_metal_flush_op_count;
}

void tq_metal_batch_flush(void) {
    if (!tq_batch.active) return;

    @autoreleasepool {
        if (tq_batch.encoder) {
            [tq_batch.encoder endEncoding];
            tq_batch.encoder = nil;
        }

        if (tq_batch.cmd_buf && tq_batch.n_copies > 0) {
            g_metal_flush_count    += 1;
            g_metal_flush_op_count += (unsigned long)tq_batch.n_copies;
            [tq_batch.cmd_buf commit];
            [tq_batch.cmd_buf waitUntilCompleted];

            /* Check for GPU errors */
            if (tq_batch.cmd_buf.status == MTLCommandBufferStatusError) {
                NSLog(@"TurboQuant: Metal batch error: %@", tq_batch.cmd_buf.error);
            }

            /* Copy all results from GPU output buffers to CPU destinations */
            for (int i = 0; i < tq_batch.n_copies; i++) {
                tq_batch_pending_copy_t* pc = &tq_batch.copies[i];
                memcpy(pc->cpu_dst, [pc->gpu_buf contents], pc->size);
                pc->gpu_buf = nil; /* Release output buffer */
            }
        }

        tq_batch.cmd_buf = nil;
        tq_batch.n_copies = 0;
        /* Keep active flag — caller can keep encoding until they call
         * tq_metal_batch_end() or begin a new batch. */
    }
}

/**
 * End batch mode. Flushes any pending operations.
 */
void tq_metal_batch_end(void) {
    if (tq_batch.active) {
        tq_metal_batch_flush();
        tq_batch.active = 0;
    }
}

/**
 * Check if batch mode is currently active.
 */
int tq_metal_batch_active(void) {
    return tq_batch.active;
}

/* ============================================================
 * Metal matmul dispatch
 *
 * Dispatches fused dequant-matmul on GPU for supported GGUF types.
 * Returns 0 on success, -1 if the type is not supported on Metal.
 *
 * In immediate mode: encodes, commits, waits, copies result.
 * In batch mode: encodes only, defers commit/wait/copy to flush.
 *
 * Weight buffers use zero-copy (newBufferWithBytesNoCopy) where
 * possible, leveraging Apple Silicon unified memory.
 * ============================================================ */

int tq_metal_matmul_gguf(float* out, const float* x, const void* weight,
                         tq_ggml_dtype weight_type, int out_dim, int in_dim)
{
    @autoreleasepool {
        /* Select pipeline based on weight type */
        id<MTLComputePipelineState> pipeline = nil;

        switch (weight_type) {
            case TQ_GGML_TYPE_IQ2_XXS:
                pipeline = tq_pipe_matmul_iq2_xxs;
                break;
            case TQ_GGML_TYPE_IQ2_S:
                pipeline = tq_pipe_matmul_iq2_s;
                break;
            case TQ_GGML_TYPE_Q8_0:
                pipeline = tq_pipe_matmul_q8_0;
                break;
            case TQ_GGML_TYPE_Q4_K:
                pipeline = tq_pipe_matmul_q4_k;
                break;
            default:
                return -1; /* Unsupported type — fall back to CPU */
        }

        if (!pipeline) {
            return -1; /* Pipeline not loaded */
        }

        /* Compute weight buffer size */
        size_t block_bytes = tq_ggml_type_size(weight_type);
        int    block_elems = tq_ggml_type_blck(weight_type);
        if (block_bytes == 0 || block_elems == 0) return -1;

        int    n_blocks    = in_dim / block_elems;
        size_t row_bytes   = (size_t)n_blocks * block_bytes;
        size_t weight_size = (size_t)out_dim * row_bytes;

        /* Align buffer sizes to 16 bytes (Metal requirement) */
        size_t input_size  = ((size_t)in_dim * sizeof(float) + 15) & ~15UL;
        size_t output_size = ((size_t)out_dim * sizeof(float) + 15) & ~15UL;

        /* --- Weight buffer: zero-copy cache lookup --- */
        id<MTLBuffer> weight_buf = tq_get_weight_buffer(weight, weight_size);
        if (!weight_buf) return -1;

        /* --- Input buffer: shared, reused across calls --- */
        if (tq_shared_input_dim != (uint32_t)in_dim || !tq_shared_input_buf) {
            tq_shared_input_buf = [tq_mtl_device
                newBufferWithLength:input_size
                            options:MTLResourceStorageModeShared];
            if (!tq_shared_input_buf) return -1;
            tq_shared_input_dim = (uint32_t)in_dim;
        }
        memcpy([tq_shared_input_buf contents], x, (size_t)in_dim * sizeof(float));

        /* --- Output buffer --- */
        id<MTLBuffer> output_buf = nil;
        if (tq_batch.active) {
            /* Batch mode: each matmul needs its own output buffer.
             * Use cached pool to avoid per-dispatch allocation. */
            if (tq_batch.n_copies >= TQ_BATCH_MAX_OPS) {
                tq_metal_batch_flush();
                /* Restart encoder for next operations */
            }
            output_buf = tq_get_batch_output_buffer(tq_batch.n_copies, output_size);
            if (!output_buf) return -1;
        } else {
            /* Immediate mode: reuse a single output buffer */
            static id<MTLBuffer> imm_output_buf = nil;
            static uint32_t imm_output_dim = 0;
            if (imm_output_dim != (uint32_t)out_dim || !imm_output_buf) {
                imm_output_buf = [tq_mtl_device
                    newBufferWithLength:output_size
                                options:MTLResourceStorageModeShared];
                if (!imm_output_buf) return -1;
                imm_output_dim = (uint32_t)out_dim;
            }
            output_buf = imm_output_buf;
        }

        /* --- Dimension uniform buffers --- */
        /* In batch mode, dimensions can change between matmuls, so we need
         * per-dispatch dimension buffers. Use cached lookup to avoid allocation. */
        id<MTLBuffer> indim_buf = nil;
        id<MTLBuffer> outdim_buf = nil;
        if (tq_batch.active) {
            indim_buf  = tq_get_dim_buffer((uint32_t)in_dim);
            outdim_buf = tq_get_dim_buffer((uint32_t)out_dim);
            if (!indim_buf || !outdim_buf) return -1;
        } else {
            indim_buf = tq_shared_indim_buf;
            outdim_buf = tq_shared_outdim_buf;
            *(uint32_t*)[indim_buf contents]  = (uint32_t)in_dim;
            *(uint32_t*)[outdim_buf contents] = (uint32_t)out_dim;
        }

        /* --- Encode compute command --- */
        id<MTLComputeCommandEncoder> enc = nil;

        if (tq_batch.active) {
            /* Batch mode: lazily create command buffer and encoder */
            if (!tq_batch.cmd_buf) {
                tq_batch.cmd_buf = [tq_mtl_queue commandBuffer];
                if (!tq_batch.cmd_buf) return -1;
            }
            if (!tq_batch.encoder) {
                tq_batch.encoder = [tq_batch.cmd_buf computeCommandEncoder];
                if (!tq_batch.encoder) return -1;
            }
            enc = tq_batch.encoder;
        } else {
            /* Immediate mode: create fresh command buffer */
            id<MTLCommandBuffer> cmdBuf = [tq_mtl_queue commandBuffer];
            if (!cmdBuf) return -1;
            enc = [cmdBuf computeCommandEncoder];
            if (!enc) return -1;

            /* Encode, commit, wait, copy in immediate mode */
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:weight_buf  offset:0 atIndex:0];
            [enc setBuffer:tq_shared_input_buf offset:0 atIndex:1];
            [enc setBuffer:output_buf  offset:0 atIndex:2];
            [enc setBuffer:indim_buf   offset:0 atIndex:3];
            [enc setBuffer:outdim_buf  offset:0 atIndex:4];

            /* IQ2_S needs codebook buffer at index 5 */
            if (weight_type == TQ_GGML_TYPE_IQ2_S && tq_iq2s_grid_buf) {
                [enc setBuffer:tq_iq2s_grid_buf offset:0 atIndex:5];
            }

            NSUInteger shared_mem = (NSUInteger)in_dim * sizeof(float);
            NSUInteger max_shared = [tq_mtl_device maxThreadgroupMemoryLength];
            if (shared_mem > max_shared) shared_mem = max_shared;
            [enc setThreadgroupMemoryLength:shared_mem atIndex:0];

            MTLSize gridSize      = MTLSizeMake((NSUInteger)out_dim, 1, 1);
            MTLSize threadgroupSz = MTLSizeMake(TQ_MATMUL_TG_SIZE, 1, 1);
            [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSz];
            [enc endEncoding];

            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            if (cmdBuf.status == MTLCommandBufferStatusError) {
                NSLog(@"TurboQuant: Metal matmul error: %@", cmdBuf.error);
                return -1;
            }

            memcpy(out, [output_buf contents], (size_t)out_dim * sizeof(float));
            return 0;
        }

        /* --- Batch mode: encode without committing --- */
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:weight_buf           offset:0 atIndex:0];
        [enc setBuffer:tq_shared_input_buf  offset:0 atIndex:1];
        [enc setBuffer:output_buf           offset:0 atIndex:2];
        [enc setBuffer:indim_buf            offset:0 atIndex:3];
        [enc setBuffer:outdim_buf           offset:0 atIndex:4];

        /* IQ2_S needs codebook buffer at index 5 */
        if (weight_type == TQ_GGML_TYPE_IQ2_S && tq_iq2s_grid_buf) {
            [enc setBuffer:tq_iq2s_grid_buf offset:0 atIndex:5];
        }

        NSUInteger shared_mem = (NSUInteger)in_dim * sizeof(float);
        NSUInteger max_shared = [tq_mtl_device maxThreadgroupMemoryLength];
        if (shared_mem > max_shared) shared_mem = max_shared;
        [enc setThreadgroupMemoryLength:shared_mem atIndex:0];

        MTLSize gridSize      = MTLSizeMake((NSUInteger)out_dim, 1, 1);
        MTLSize threadgroupSz = MTLSizeMake(TQ_MATMUL_TG_SIZE, 1, 1);
        [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSz];

        /* Record pending copy */
        tq_batch_pending_copy_t* pc = &tq_batch.copies[tq_batch.n_copies++];
        pc->cpu_dst = out;
        pc->gpu_buf = output_buf;
        pc->size    = (size_t)out_dim * sizeof(float);

        return 0;
    }
}

/**
 * TQ internal Q4 matmul dispatch.
 *
 * Format: block_size=32, 16 packed bytes (4-bit nibbles) + 1 float scale.
 * Dequant: (nibble - 8) * scale.  Layout matches matmul_tq_q4 shader.
 *
 * Supports both batch mode (encode only, defer commit) and immediate mode
 * (encode, commit, wait, copy).
 *
 * @param out      Output vector [n] (CPU pointer)
 * @param x        Input vector [d] (CPU pointer)
 * @param w_qs     Packed Q4 weights [n * (d/32) * 16] bytes
 * @param w_scales Weight scales [n * (d/32)] floats
 * @param n        Output dimension (number of rows)
 * @param d        Input dimension (must be multiple of 32)
 * @return 0 on success, -1 on failure (caller falls back to CPU)
 */
int tq_metal_matmul_q4(float* out, const float* x, const uint8_t* w_qs,
                        const float* w_scales, int n, int d) {
    if (!tq_metal_available()) return -1;
    if (!tq_pipe_matmul_tq_q4) return -1;

    @autoreleasepool {
        int n_blocks = d / 32;
        size_t qs_size  = (size_t)n * n_blocks * 16;
        size_t sc_size  = (size_t)n * n_blocks * sizeof(float);
        size_t input_size  = ((size_t)d * sizeof(float) + 15) & ~15UL;
        size_t output_size = ((size_t)n * sizeof(float) + 15) & ~15UL;

        /* Weight buffers: zero-copy via cache */
        id<MTLBuffer> qs_buf = tq_get_weight_buffer(w_qs, qs_size);
        if (!qs_buf) return -1;
        id<MTLBuffer> sc_buf = tq_get_weight_buffer(w_scales, sc_size);
        if (!sc_buf) return -1;

        /* Input buffer: reuse shared input buf */
        if (tq_shared_input_dim != (uint32_t)d || !tq_shared_input_buf) {
            tq_shared_input_buf = [tq_mtl_device
                newBufferWithLength:input_size
                            options:MTLResourceStorageModeShared];
            if (!tq_shared_input_buf) return -1;
            tq_shared_input_dim = (uint32_t)d;
        }
        memcpy([tq_shared_input_buf contents], x, (size_t)d * sizeof(float));

        /* Output buffer */
        id<MTLBuffer> output_buf = nil;
        if (tq_batch.active) {
            if (tq_batch.n_copies >= TQ_BATCH_MAX_OPS) {
                tq_metal_batch_flush();
            }
            output_buf = tq_get_batch_output_buffer(tq_batch.n_copies, output_size);
            if (!output_buf) return -1;
        } else {
            static id<MTLBuffer> imm_q4_output_buf = nil;
            static uint32_t imm_q4_output_dim = 0;
            if (imm_q4_output_dim != (uint32_t)n || !imm_q4_output_buf) {
                imm_q4_output_buf = [tq_mtl_device
                    newBufferWithLength:output_size
                                options:MTLResourceStorageModeShared];
                if (!imm_q4_output_buf) return -1;
                imm_q4_output_dim = (uint32_t)n;
            }
            output_buf = imm_q4_output_buf;
        }

        /* Dimension uniform buffers */
        id<MTLBuffer> indim_buf = nil;
        id<MTLBuffer> outdim_buf = nil;
        if (tq_batch.active) {
            /* Dim cache buffers have values pre-written at creation time */
            indim_buf  = tq_get_dim_buffer((uint32_t)d);
            outdim_buf = tq_get_dim_buffer((uint32_t)n);
            if (!indim_buf || !outdim_buf) return -1;
        } else {
            indim_buf = tq_shared_indim_buf;
            outdim_buf = tq_shared_outdim_buf;
            *(uint32_t*)[indim_buf contents]  = (uint32_t)d;
            *(uint32_t*)[outdim_buf contents] = (uint32_t)n;
        }

        /* Encode compute command.
         * Shader signature: input(0), output(1), weight_qs(2),
         *                   weight_sc(3), in_dim(4), out_dim(5) */
        if (tq_batch.active) {
            /* Batch mode: lazily create command buffer and encoder */
            if (!tq_batch.cmd_buf) {
                tq_batch.cmd_buf = [tq_mtl_queue commandBuffer];
                if (!tq_batch.cmd_buf) return -1;
            }
            if (!tq_batch.encoder) {
                tq_batch.encoder = [tq_batch.cmd_buf computeCommandEncoder];
                if (!tq_batch.encoder) return -1;
            }
            id<MTLComputeCommandEncoder> enc = tq_batch.encoder;

            [enc setComputePipelineState:tq_pipe_matmul_tq_q4];
            [enc setBuffer:tq_shared_input_buf offset:0 atIndex:0];
            [enc setBuffer:output_buf          offset:0 atIndex:1];
            [enc setBuffer:qs_buf              offset:0 atIndex:2];
            [enc setBuffer:sc_buf              offset:0 atIndex:3];
            [enc setBuffer:indim_buf           offset:0 atIndex:4];
            [enc setBuffer:outdim_buf          offset:0 atIndex:5];

            MTLSize gridSize      = MTLSizeMake((NSUInteger)n, 1, 1);
            MTLSize threadgroupSz = MTLSizeMake(TQ_MATMUL_TG_SIZE, 1, 1);
            [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSz];

            /* Record pending copy */
            tq_batch_pending_copy_t* pc = &tq_batch.copies[tq_batch.n_copies++];
            pc->cpu_dst = out;
            pc->gpu_buf = output_buf;
            pc->size    = (size_t)n * sizeof(float);

            return 0;
        } else {
            /* Immediate mode */
            id<MTLCommandBuffer> cmdBuf = [tq_mtl_queue commandBuffer];
            if (!cmdBuf) return -1;
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            if (!enc) return -1;

            [enc setComputePipelineState:tq_pipe_matmul_tq_q4];
            [enc setBuffer:tq_shared_input_buf offset:0 atIndex:0];
            [enc setBuffer:output_buf          offset:0 atIndex:1];
            [enc setBuffer:qs_buf              offset:0 atIndex:2];
            [enc setBuffer:sc_buf              offset:0 atIndex:3];
            [enc setBuffer:indim_buf           offset:0 atIndex:4];
            [enc setBuffer:outdim_buf          offset:0 atIndex:5];

            MTLSize gridSize      = MTLSizeMake((NSUInteger)n, 1, 1);
            MTLSize threadgroupSz = MTLSizeMake(TQ_MATMUL_TG_SIZE, 1, 1);
            [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSz];
            [enc endEncoding];

            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            if (cmdBuf.status == MTLCommandBufferStatusError) {
                NSLog(@"TurboQuant: Metal TQ Q4 matmul error: %@", cmdBuf.error);
                return -1;
            }

            memcpy(out, [output_buf contents], (size_t)n * sizeof(float));
            return 0;
        }
    }
}

/* ============================================================
 * Fused MoE forward pass — single-dispatch per phase
 *
 * Processes ALL active experts for one MoE layer using 3 GPU
 * dispatches instead of 810 per-matmul dispatches:
 *   Phase 1: gate + up projections for all experts
 *   Phase 2: SwiGLU activation
 *   Phase 3: down projection + weighted accumulation
 *
 * Returns 0 on success, -1 if fused MoE is not available.
 * On failure, caller should fall back to per-expert dispatch.
 * ============================================================ */

/* MoeGpuParams — must match the Metal shader struct exactly */
typedef struct {
    int      active_experts[8];
    float    expert_weights[8];
    int      num_active;
    int      expert_dim;
    int      hidden_dim;
    int      num_experts_total;
    uint64_t gate_offsets[8];
    uint64_t up_offsets[8];
    uint64_t down_offsets[8];
    uint32_t gate_type;
    int      blocks_per_row_gate;
    int      row_bytes_gate;
    int      blocks_per_row_down;
    int      row_bytes_down;
    /* Per-expert quant types: 16 = IQ2_XXS, 22 = IQ2_S */
    int      gate_types[8];
    int      up_types[8];
    int      down_types[8];
    int      gate_row_bytes;
    int      down_row_bytes;
} MoeGpuParams;

/**
 * Check if fused MoE Metal dispatch is available.
 */
int tq_metal_moe_available(void) {
    if (!tq_mtl_device || !tq_mtl_queue) return 0;
    return (tq_pipe_moe_gate_up != nil &&
            tq_pipe_moe_swiglu != nil &&
            tq_pipe_moe_down_accum != nil) ? 1 : 0;
}

/**
 * Hybrid GPU/CPU MoE forward: GPU does gate+up+SwiGLU (Phases 1+2),
 * CPU handles down projection (Phase 3 IQ2_S shader hangs on Metal).
 *
 * @param input        Input hidden state [hidden_dim] (CPU pointer, will be uploaded)
 * @param output       Output hidden state [hidden_dim] (unused in hybrid mode)
 * @param hb_output    SwiGLU'd activations [num_active * expert_dim] (filled by GPU)
 * @param weight_base  Base pointer for all expert weights (mmap'd GGUF data)
 * @param weight_size  Total size of weight region (for zero-copy buffer)
 * @param gate_offsets Byte offsets from weight_base to each active expert's gate weights [num_active]
 * @param up_offsets   Byte offsets from weight_base to each active expert's up weights [num_active]
 * @param down_offsets Byte offsets from weight_base to each active expert's down weights [num_active]
 * @param active_expert_ids  Expert indices [num_active]
 * @param expert_routing_weights  Routing weights [num_active]
 * @param num_active   Number of active experts
 * @param expert_dim   Expert intermediate dimension
 * @param hidden_dim   Model hidden dimension
 * @param num_experts_total  Total number of experts
 * @param weight_type  GGUF weight type (16=IQ2_XXS, 22=IQ2_S, or 0 for per-expert types)
 * @param gate_types_in  Per-expert gate quant types [num_active] (NULL = use weight_type for all)
 * @param up_types_in    Per-expert up quant types [num_active] (NULL = use weight_type for all)
 * @param down_types_in  Per-expert down quant types [num_active] (NULL = use weight_type for all)
 * @return 1 on partial success (hb_output filled, caller does down+accum on CPU),
 *        -1 on failure
 */
int tq_metal_moe_forward(
    const float*    input,
    float*          output,
    float*          hb_output,
    const void*     weight_base,
    size_t          weight_size,
    const uint64_t* gate_offsets,
    const uint64_t* up_offsets,
    const uint64_t* down_offsets,
    const int*      active_expert_ids,
    const float*    expert_routing_weights,
    int             num_active,
    int             expert_dim,
    int             hidden_dim,
    int             num_experts_total,
    int             weight_type,
    const int*      gate_types_in,
    const int*      up_types_in,
    const int*      down_types_in)
{
    if (!tq_metal_moe_available()) return -1;
    if (num_active <= 0 || num_active > 8) return -1;

    /* Accept IQ2_XXS (16) and IQ2_S (22) — UD models use mixed types */
    if (weight_type != 0 && weight_type != 16 && weight_type != 22) return -1;

    @autoreleasepool {
        /* --- Build params struct --- */
        MoeGpuParams params;
        memset(&params, 0, sizeof(params));
        params.num_active = num_active;
        params.expert_dim = expert_dim;
        params.hidden_dim = hidden_dim;
        params.num_experts_total = num_experts_total;
        params.gate_type = (uint32_t)weight_type;

        /* Block geometry: blocks_per_row is type-independent (always 256 elements/block).
         * row_bytes_gate/down use IQ2_XXS (66) as default; shader overrides per-expert
         * for IQ2_S (82) via the gate_types/down_types arrays. */
        params.blocks_per_row_gate = hidden_dim / 256;
        params.blocks_per_row_down = expert_dim / 256;

        /* Determine default row bytes from first expert's type */
        int first_gate_type = gate_types_in ? gate_types_in[0] : weight_type;
        int first_down_type = down_types_in ? down_types_in[0] : weight_type;
        int gate_blk_bytes = (first_gate_type == 22) ? 82 : 66; /* IQ2_S=82, IQ2_XXS=66 */
        int down_blk_bytes = (first_down_type == 22) ? 82 : 66;
        params.row_bytes_gate = params.blocks_per_row_gate * gate_blk_bytes;
        params.row_bytes_down = params.blocks_per_row_down * down_blk_bytes;
        params.gate_row_bytes = params.row_bytes_gate;
        params.down_row_bytes = params.row_bytes_down;

        for (int k = 0; k < num_active; k++) {
            params.active_experts[k] = active_expert_ids[k];
            params.expert_weights[k] = expert_routing_weights[k];
            params.gate_offsets[k] = gate_offsets[k];
            params.up_offsets[k] = up_offsets[k];
            params.down_offsets[k] = down_offsets[k];

            /* Per-expert quant types */
            params.gate_types[k] = gate_types_in ? gate_types_in[k] : weight_type;
            params.up_types[k]   = up_types_in   ? up_types_in[k]   : weight_type;
            params.down_types[k] = down_types_in ? down_types_in[k] : weight_type;
        }

        /* --- Get or create zero-copy weight buffer --- */
        id<MTLBuffer> weight_buf = tq_get_weight_buffer(weight_base, weight_size);
        if (!weight_buf) return -1;

        /* --- MLX Pattern 1: Cached buffer allocation ---
         * Reuse static Metal buffers across calls, only reallocating
         * when a larger size is needed. Eliminates ~0.1ms per buffer
         * creation × 5 buffers × 30 layers = ~15ms saved per token. */

        size_t input_bytes = (size_t)hidden_dim * sizeof(float);
        size_t inter_bytes = (size_t)num_active * (size_t)expert_dim * sizeof(float);
        size_t output_bytes = (size_t)hidden_dim * sizeof(float);

        /* Grow-only input buffer (memcpy new data each call) */
        if (!tq_moe_input_buf || tq_moe_input_size < input_bytes) {
            tq_moe_input_buf = [tq_mtl_device newBufferWithLength:input_bytes
                                                           options:MTLResourceStorageModeShared];
            tq_moe_input_size = input_bytes;
        }
        if (!tq_moe_input_buf) return -1;
        memcpy([tq_moe_input_buf contents], input, input_bytes);
        id<MTLBuffer> input_buf = tq_moe_input_buf;

        /* Grow-only intermediate buffers */
        if (!tq_moe_gate_buf || tq_moe_gate_size < inter_bytes) {
            tq_moe_gate_buf = [tq_mtl_device newBufferWithLength:inter_bytes
                                                          options:MTLResourceStorageModeShared];
            tq_moe_gate_size = inter_bytes;
        }
        if (!tq_moe_up_buf || tq_moe_up_size < inter_bytes) {
            tq_moe_up_buf = [tq_mtl_device newBufferWithLength:inter_bytes
                                                        options:MTLResourceStorageModeShared];
            tq_moe_up_size = inter_bytes;
        }
        id<MTLBuffer> gate_buf = tq_moe_gate_buf;
        id<MTLBuffer> up_buf   = tq_moe_up_buf;
        if (!gate_buf || !up_buf) return -1;

        /* Grow-only params buffer (memcpy new params each call) */
        if (!tq_moe_params_buf) {
            tq_moe_params_buf = [tq_mtl_device newBufferWithLength:sizeof(MoeGpuParams)
                                                            options:MTLResourceStorageModeShared];
        }
        if (!tq_moe_params_buf) return -1;
        memcpy([tq_moe_params_buf contents], &params, sizeof(MoeGpuParams));
        id<MTLBuffer> params_buf = tq_moe_params_buf;

        /* Grow-only output buffer */
        if (!tq_moe_output_buf || tq_moe_output_size < output_bytes) {
            tq_moe_output_buf = [tq_mtl_device newBufferWithLength:output_bytes
                                                            options:MTLResourceStorageModeShared];
            tq_moe_output_size = output_bytes;
        }
        id<MTLBuffer> output_buf = tq_moe_output_buf;
        if (!output_buf) {
            /* Fallback to hybrid if buffer creation fails */
            memcpy(hb_output, [gate_buf contents], inter_bytes);
            return 1;
        }

        /* --- Single command buffer for all 3 phases (MLX pattern) ---
         * Metal guarantees sequential execution of compute encoders within
         * one command buffer. memoryBarrierWithScope ensures buffer writes
         * from one encoder are visible to the next.
         *
         * MLX Pattern 2 (Conditional Barriers):
         * Barriers are only placed within a layer's 3 phases (P1→P2, P2→P3)
         * where data dependencies exist. No barriers needed between layers
         * because each layer call uses independent params and the cached
         * intermediate buffers are fully overwritten by each layer's Phase 1.
         * This is the MLX maybeInsertBarrier pattern — only barrier when
         * buffers actually alias between producer and consumer. */
        id<MTLCommandBuffer> cmdBuf = [tq_mtl_queue commandBuffer];
        if (!cmdBuf) return -1;

        /* Shared memory for Phase 1 (gate_up): hidden_dim floats for input + 8 for SIMD sums */
        NSUInteger shared_phase1 = ((NSUInteger)hidden_dim + 8) * sizeof(float);
        NSUInteger max_shared = [tq_mtl_device maxThreadgroupMemoryLength];
        if (shared_phase1 > max_shared) shared_phase1 = max_shared;

        /* ======== Phase 1: gate + up (fused) ======== */
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            if (!enc) return -1;

            [enc setComputePipelineState:tq_pipe_moe_gate_up];
            [enc setBuffer:weight_buf  offset:0 atIndex:0];
            [enc setBuffer:input_buf   offset:0 atIndex:1];
            [enc setBuffer:gate_buf    offset:0 atIndex:2];
            [enc setBuffer:up_buf      offset:0 atIndex:3];
            [enc setBuffer:params_buf  offset:0 atIndex:4];
            if (tq_iq2s_grid_buf) {
                [enc setBuffer:tq_iq2s_grid_buf offset:0 atIndex:5];
            }
            [enc setThreadgroupMemoryLength:shared_phase1 atIndex:0];

            /* MLX Pattern 3: 64 threads per TG (was 256).
             * reduce_sum() adapts via (tg_size+31)/32, so n_simd=2 with 64 threads.
             * Less sync overhead for MoE's small per-expert matmuls. */
            NSUInteger n_tgs = (NSUInteger)num_active * (NSUInteger)expert_dim;
            MTLSize gridSize = MTLSizeMake(n_tgs, 1, 1);
            MTLSize tgSize   = MTLSizeMake(TQ_MOE_TG_SIZE, 1, 1);
            [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];

            /* MLX Pattern 2: Conditional barrier — needed here because
             * Phase 2 reads gate_buf/up_buf written by this encoder. */
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc endEncoding];
        }

        /* ======== Phase 2: SwiGLU (reads gate_buf/up_buf from Phase 1) ======== */
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            if (!enc) return -1;

            [enc setComputePipelineState:tq_pipe_moe_swiglu];
            [enc setBuffer:gate_buf    offset:0 atIndex:0];
            [enc setBuffer:up_buf      offset:0 atIndex:1];
            [enc setBuffer:params_buf  offset:0 atIndex:2];

            /* SwiGLU is element-wise — uses thread_position_in_grid, no
             * threadgroup cooperation needed. 64 threads is sufficient. */
            NSUInteger n_threads = (NSUInteger)num_active * (NSUInteger)expert_dim;
            NSUInteger tg = TQ_MOE_TG_SIZE;
            NSUInteger n_tgs = (n_threads + tg - 1) / tg;
            MTLSize gridSize = MTLSizeMake(n_tgs, 1, 1);
            MTLSize tgSize   = MTLSizeMake(tg, 1, 1);
            [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];

            /* MLX Pattern 2: Conditional barrier — needed here because
             * Phase 3 reads gate_buf (SwiGLU output) written by this encoder. */
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc endEncoding];
        }

        /* ======== Phase 3: down projection + weighted accumulate (GPU) ========
         * IQ2_S codebook passed as device buffer (buffer 4). */
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            if (!enc) {
                memcpy(hb_output, [gate_buf contents], inter_bytes);
                return 1;
            }

            [enc setComputePipelineState:tq_pipe_moe_down_accum];
            [enc setBuffer:weight_buf   offset:0 atIndex:0];
            [enc setBuffer:gate_buf     offset:0 atIndex:1]; /* SwiGLU output = hb_all */
            [enc setBuffer:output_buf   offset:0 atIndex:2];
            [enc setBuffer:params_buf   offset:0 atIndex:3];
            if (tq_iq2s_grid_buf) {
                [enc setBuffer:tq_iq2s_grid_buf offset:0 atIndex:4];
            }

            /* Shared memory for Phase 3: expert_dim floats for hb + 8 for SIMD sums */
            NSUInteger shared_phase3 = ((NSUInteger)expert_dim + 8) * sizeof(float);
            if (shared_phase3 > max_shared) shared_phase3 = max_shared;
            [enc setThreadgroupMemoryLength:shared_phase3 atIndex:0];

            /* MLX Pattern 3: 64 threads for MoE down projection.
             * One threadgroup per output row (hidden_dim total). */
            MTLSize gridSize3 = MTLSizeMake((NSUInteger)hidden_dim, 1, 1);
            MTLSize tgSize3   = MTLSizeMake(TQ_MOE_TG_SIZE, 1, 1);
            [enc dispatchThreadgroups:gridSize3 threadsPerThreadgroup:tgSize3];
            [enc endEncoding];
        }

        /* ONE commit + wait for all 3 phases */
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status == MTLCommandBufferStatusError) {
            NSLog(@"TurboQuant MoE: GPU dispatch FAILED: %@", cmdBuf.error);
            /* Fallback to hybrid on failure */
            memcpy(hb_output, [gate_buf contents], inter_bytes);
            return 1;
        }

        /* Copy result to output */
        memcpy(output, [output_buf contents], output_bytes);

        /* Also copy hb for potential caller use */
        memcpy(hb_output, [gate_buf contents], inter_bytes);

        return 0; /* Full GPU success */
    }
}

/* ============================================================
 * Element-wise dispatch functions
 *
 * These run RMSNorm, SiLU, mul, and add on the GPU to avoid
 * GPU->CPU->GPU round-trips between matmul dispatches.
 *
 * NOT connected to tq_ops.c yet -- standalone dispatch ready
 * for integration when the full forward pass orchestrator is built.
 * ============================================================ */

/**
 * RMSNorm on Metal GPU.
 * out[i] = (x[i] / rms(x)) * weight[i], rms = sqrt(mean(x^2) + eps)
 * Returns 0 on success, -1 on failure.
 */
int tq_metal_rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
    @autoreleasepool {
        if (!tq_mtl_device || !tq_pipe_rmsnorm || n <= 0) return -1;

        size_t buf_bytes = (size_t)n * sizeof(float);
        id<MTLBuffer> x_buf = [tq_mtl_device newBufferWithBytes:x
                                                          length:buf_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> w_buf = [tq_mtl_device newBufferWithBytes:w
                                                          length:buf_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> o_buf = [tq_mtl_device newBufferWithLength:buf_bytes
                                                         options:MTLResourceStorageModeShared];
        uint32_t n_val = (uint32_t)n;
        id<MTLBuffer> n_buf = [tq_mtl_device newBufferWithBytes:&n_val
                                                          length:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> eps_buf = [tq_mtl_device newBufferWithBytes:&eps
                                                            length:sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        if (!x_buf || !w_buf || !o_buf || !n_buf || !eps_buf) return -1;

        id<MTLCommandBuffer> cmdBuf = [tq_mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:tq_pipe_rmsnorm];
        [enc setBuffer:x_buf   offset:0 atIndex:0];
        [enc setBuffer:w_buf   offset:0 atIndex:1];
        [enc setBuffer:o_buf   offset:0 atIndex:2];
        [enc setBuffer:n_buf   offset:0 atIndex:3];
        [enc setBuffer:eps_buf offset:0 atIndex:4];

        /* One threadgroup of 256 threads for the reduction */
        NSUInteger tg_size = 256;
        if (tg_size > tq_pipe_rmsnorm.maxTotalThreadsPerThreadgroup) {
            tg_size = tq_pipe_rmsnorm.maxTotalThreadsPerThreadgroup;
        }
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];

        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status == MTLCommandBufferStatusError) return -1;

        memcpy(out, [o_buf contents], buf_bytes);
        return 0;
    }
}

/**
 * SiLU activation on Metal GPU.
 * out[i] = x[i] / (1 + exp(-x[i]))
 * Returns 0 on success, -1 on failure.
 */
int tq_metal_silu(float* out, const float* x, int n) {
    @autoreleasepool {
        if (!tq_mtl_device || !tq_pipe_silu || n <= 0) return -1;

        size_t buf_bytes = (size_t)n * sizeof(float);
        id<MTLBuffer> x_buf = [tq_mtl_device newBufferWithBytes:x
                                                          length:buf_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> o_buf = [tq_mtl_device newBufferWithLength:buf_bytes
                                                         options:MTLResourceStorageModeShared];
        uint32_t n_val = (uint32_t)n;
        id<MTLBuffer> n_buf = [tq_mtl_device newBufferWithBytes:&n_val
                                                          length:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
        if (!x_buf || !o_buf || !n_buf) return -1;

        id<MTLCommandBuffer> cmdBuf = [tq_mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:tq_pipe_silu];
        [enc setBuffer:x_buf offset:0 atIndex:0];
        [enc setBuffer:o_buf offset:0 atIndex:1];
        [enc setBuffer:n_buf offset:0 atIndex:2];

        /* One thread per element, threadgroups of 256 */
        NSUInteger tg = 256;
        if (tg > tq_pipe_silu.maxTotalThreadsPerThreadgroup) {
            tg = tq_pipe_silu.maxTotalThreadsPerThreadgroup;
        }
        NSUInteger num_tg = ((NSUInteger)n + tg - 1) / tg;
        [enc dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status == MTLCommandBufferStatusError) return -1;

        memcpy(out, [o_buf contents], buf_bytes);
        return 0;
    }
}

/**
 * Element-wise multiply on Metal GPU.
 * out[i] = a[i] * b[i]
 * Returns 0 on success, -1 on failure.
 */
int tq_metal_mul(float* out, const float* a, const float* b, int n) {
    @autoreleasepool {
        if (!tq_mtl_device || !tq_pipe_mul_elementwise || n <= 0) return -1;

        size_t buf_bytes = (size_t)n * sizeof(float);
        id<MTLBuffer> a_buf = [tq_mtl_device newBufferWithBytes:a
                                                          length:buf_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_buf = [tq_mtl_device newBufferWithBytes:b
                                                          length:buf_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> o_buf = [tq_mtl_device newBufferWithLength:buf_bytes
                                                         options:MTLResourceStorageModeShared];
        uint32_t n_val = (uint32_t)n;
        id<MTLBuffer> n_buf = [tq_mtl_device newBufferWithBytes:&n_val
                                                          length:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
        if (!a_buf || !b_buf || !o_buf || !n_buf) return -1;

        id<MTLCommandBuffer> cmdBuf = [tq_mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:tq_pipe_mul_elementwise];
        [enc setBuffer:a_buf offset:0 atIndex:0];
        [enc setBuffer:b_buf offset:0 atIndex:1];
        [enc setBuffer:o_buf offset:0 atIndex:2];
        [enc setBuffer:n_buf offset:0 atIndex:3];

        NSUInteger tg = 256;
        if (tg > tq_pipe_mul_elementwise.maxTotalThreadsPerThreadgroup) {
            tg = tq_pipe_mul_elementwise.maxTotalThreadsPerThreadgroup;
        }
        NSUInteger num_tg = ((NSUInteger)n + tg - 1) / tg;
        [enc dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status == MTLCommandBufferStatusError) return -1;

        memcpy(out, [o_buf contents], buf_bytes);
        return 0;
    }
}

/**
 * Vector add on Metal GPU.
 * out[i] = a[i] + b[i]
 * Returns 0 on success, -1 on failure.
 */
int tq_metal_add(float* out, const float* a, const float* b, int n) {
    @autoreleasepool {
        if (!tq_mtl_device || !tq_pipe_add_vectors || n <= 0) return -1;

        size_t buf_bytes = (size_t)n * sizeof(float);
        id<MTLBuffer> a_buf = [tq_mtl_device newBufferWithBytes:a
                                                          length:buf_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_buf = [tq_mtl_device newBufferWithBytes:b
                                                          length:buf_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> o_buf = [tq_mtl_device newBufferWithLength:buf_bytes
                                                         options:MTLResourceStorageModeShared];
        uint32_t n_val = (uint32_t)n;
        id<MTLBuffer> n_buf = [tq_mtl_device newBufferWithBytes:&n_val
                                                          length:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
        if (!a_buf || !b_buf || !o_buf || !n_buf) return -1;

        id<MTLCommandBuffer> cmdBuf = [tq_mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:tq_pipe_add_vectors];
        [enc setBuffer:a_buf offset:0 atIndex:0];
        [enc setBuffer:b_buf offset:0 atIndex:1];
        [enc setBuffer:o_buf offset:0 atIndex:2];
        [enc setBuffer:n_buf offset:0 atIndex:3];

        NSUInteger tg = 256;
        if (tg > tq_pipe_add_vectors.maxTotalThreadsPerThreadgroup) {
            tg = tq_pipe_add_vectors.maxTotalThreadsPerThreadgroup;
        }
        NSUInteger num_tg = ((NSUInteger)n + tg - 1) / tg;
        [enc dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status == MTLCommandBufferStatusError) return -1;

        memcpy(out, [o_buf contents], buf_bytes);
        return 0;
    }
}

/* ============================================================
 * GPU Compute Graph Runtime — Full Layer Forward
 *
 * Encodes ALL operations for one transformer layer into a SINGLE
 * Metal command buffer with ZERO CPU<->GPU sync within a layer:
 *
 *   rmsnorm(x→xb) → matmul(xb→q,k,v) → rope(q,k) →
 *   attention(q,k_cache,v_cache→xb) → matmul(xb→xb2) →
 *   add(x,xb2→x) → rmsnorm(x→xb) → matmul(xb→hb,hb2) →
 *   silu/gelu(hb) → mul(hb,hb2) → matmul(hb→xb2) → add(x,xb2→x)
 *
 * Persistent GPU buffers allocated at init, reused every layer.
 * Weight buffers use zero-copy from mmap (unified memory).
 * Memory barriers between dependent operations ensure correctness.
 * ============================================================ */

/* Persistent activation buffers (allocated once, reused) */
static id<MTLBuffer> g_gpu_x   = nil;   /* [dim] residual state (persists across layers) */
static id<MTLBuffer> g_gpu_xb  = nil;   /* [max_dim] normed input */
static id<MTLBuffer> g_gpu_q   = nil;   /* [q_dim] query */
static id<MTLBuffer> g_gpu_k   = nil;   /* [kv_dim] key */
static id<MTLBuffer> g_gpu_v   = nil;   /* [kv_dim] value */
static id<MTLBuffer> g_gpu_xb2 = nil;   /* [max_dim] output */
static id<MTLBuffer> g_gpu_hb  = nil;   /* [inter_dim] FFN hidden */
static id<MTLBuffer> g_gpu_hb2 = nil;   /* [inter_dim] FFN hidden2 */
static id<MTLBuffer> g_gpu_att = nil;   /* [n_heads * max_seq] attention scores */
static id<MTLBuffer> g_gpu_key_cache = nil;  /* [max_seq * kv_dim] per-layer K cache */
static id<MTLBuffer> g_gpu_val_cache = nil;  /* [max_seq * kv_dim] per-layer V cache */
static uint32_t g_gpu_max_dim = 0;
static uint32_t g_gpu_max_inter = 0;
static uint32_t g_gpu_max_seq = 0;
static int g_gpu_graph_ready = 0;

/**
 * Initialize persistent GPU activation buffers for the compute graph.
 * Must be called once at model load time.
 * Returns 0 on success, -1 on failure.
 */
int tq_metal_gpu_init_buffers(int max_dim, int max_inter, int max_q_dim, int max_kv_dim) {
    @autoreleasepool {
        if (!tq_metal_available()) return -1;

        size_t dim_bytes   = (size_t)max_dim * sizeof(float);
        size_t inter_bytes = (size_t)max_inter * sizeof(float);
        size_t q_bytes     = (size_t)max_q_dim * sizeof(float);
        size_t kv_bytes    = (size_t)max_kv_dim * sizeof(float);

        g_gpu_x   = [tq_mtl_device newBufferWithLength:dim_bytes options:MTLResourceStorageModeShared];
        g_gpu_xb  = [tq_mtl_device newBufferWithLength:dim_bytes options:MTLResourceStorageModeShared];
        g_gpu_q   = [tq_mtl_device newBufferWithLength:q_bytes options:MTLResourceStorageModeShared];
        g_gpu_k   = [tq_mtl_device newBufferWithLength:kv_bytes options:MTLResourceStorageModeShared];
        g_gpu_v   = [tq_mtl_device newBufferWithLength:kv_bytes options:MTLResourceStorageModeShared];
        g_gpu_xb2 = [tq_mtl_device newBufferWithLength:dim_bytes options:MTLResourceStorageModeShared];
        g_gpu_hb  = [tq_mtl_device newBufferWithLength:inter_bytes options:MTLResourceStorageModeShared];
        g_gpu_hb2 = [tq_mtl_device newBufferWithLength:inter_bytes options:MTLResourceStorageModeShared];

        g_gpu_max_dim   = (uint32_t)max_dim;
        g_gpu_max_inter = (uint32_t)max_inter;

        if (!g_gpu_x || !g_gpu_xb || !g_gpu_q || !g_gpu_k || !g_gpu_v ||
            !g_gpu_xb2 || !g_gpu_hb || !g_gpu_hb2) return -1;

        NSLog(@"TurboQuant: GPU graph buffers initialized (dim=%d, inter=%d, q=%d, kv=%d)",
              max_dim, max_inter, max_q_dim, max_kv_dim);
        return 0;
    }
}

/**
 * Initialize attention score buffer and KV cache GPU buffers.
 * Called separately because n_heads and max_seq may not be known at init_buffers time.
 */
int tq_metal_gpu_init_attn(int n_heads, int max_seq, int kv_dim) {
    @autoreleasepool {
        if (!tq_metal_available()) return -1;

        size_t att_bytes  = (size_t)n_heads * max_seq * sizeof(float);
        size_t cache_bytes = (size_t)max_seq * kv_dim * sizeof(float);

        g_gpu_att = [tq_mtl_device newBufferWithLength:att_bytes options:MTLResourceStorageModeShared];
        g_gpu_key_cache = [tq_mtl_device newBufferWithLength:cache_bytes options:MTLResourceStorageModeShared];
        g_gpu_val_cache = [tq_mtl_device newBufferWithLength:cache_bytes options:MTLResourceStorageModeShared];
        g_gpu_max_seq = (uint32_t)max_seq;

        if (!g_gpu_att || !g_gpu_key_cache || !g_gpu_val_cache) return -1;

        g_gpu_graph_ready = 1;
        NSLog(@"TurboQuant: GPU compute graph ready (n_heads=%d, max_seq=%d, kv_dim=%d)",
              n_heads, max_seq, kv_dim);
        return 0;
    }
}

/**
 * Check if the full GPU compute graph forward is available.
 */
int tq_metal_graph_available(void) {
    return (g_gpu_graph_ready &&
            tq_pipe_matmul_tq_q4 &&
            tq_pipe_rmsnorm &&
            tq_pipe_rope &&
            tq_pipe_softmax &&
            tq_pipe_attn_qk &&
            tq_pipe_attn_v &&
            tq_pipe_silu &&
            tq_pipe_mul_elementwise &&
            tq_pipe_add_inplace) ? 1 : 0;
}

/* ---- Helper: encode a Q4 matmul into an existing encoder ---- */
/* Forward declaration */
void tq_metal_repack_q4(const uint8_t* src_qs, const float* src_scales,
                         id<MTLBuffer>* out_qs_buf, id<MTLBuffer>* out_sc_buf,
                         int out_dim, int in_dim);

/* Repacked weight cache: maps (w_qs pointer) → (repacked MTLBuffer pair) */
#define TQ_REPACK_CACHE_SIZE 128
static struct { const void* key; id<MTLBuffer> qs; id<MTLBuffer> sc; int out_dim; int in_dim; }
    g_repack_cache[TQ_REPACK_CACHE_SIZE];
static int g_repack_count __attribute__((unused)) = 0;

static void encode_q4_matmul(id<MTLComputeCommandEncoder> enc,
                              id<MTLBuffer> input_buf,
                              id<MTLBuffer> output_buf,
                              const uint8_t* w_qs, const float* w_scales,
                              int out_dim, int in_dim)
{
    if (!tq_pipe_matmul_tq_q4) return;

    int n_blocks = in_dim / 32;

    /* Fast Q4 kernel: llama.cpp-inspired uint16 mask trick + SIMD-group.
     * No repacking needed — reads original row-major Q4 layout.
     * 2 SIMD-groups per threadgroup, each processes 1 output row. */
    if (tq_pipe_matmul_tq_q4_repacked) {  /* reusing pipeline slot for fast kernel */
        size_t qs_size = (size_t)out_dim * n_blocks * 16;
        size_t sc_size = (size_t)out_dim * n_blocks * sizeof(float);
        id<MTLBuffer> w_qs_buf = tq_get_weight_buffer(w_qs, qs_size);
        id<MTLBuffer> w_sc_buf = tq_get_weight_buffer(w_scales, sc_size);
        if (w_qs_buf && w_sc_buf) {
            id<MTLBuffer> indim_buf  = tq_get_dim_buffer((uint32_t)in_dim);
            id<MTLBuffer> outdim_buf = tq_get_dim_buffer((uint32_t)out_dim);

            [enc setComputePipelineState:tq_pipe_matmul_tq_q4_repacked];
            [enc setBuffer:input_buf  offset:0 atIndex:0];
            [enc setBuffer:output_buf offset:0 atIndex:1];
            [enc setBuffer:w_qs_buf   offset:0 atIndex:2];
            [enc setBuffer:w_sc_buf   offset:0 atIndex:3];
            [enc setBuffer:indim_buf  offset:0 atIndex:4];
            [enc setBuffer:outdim_buf offset:0 atIndex:5];

            /* n_tiles threadgroups, 2 SIMD-groups (64 threads) per group */
            int n_rows_per_tg = 2;  /* NSG in kernel */
            int n_tg = (out_dim + n_rows_per_tg - 1) / n_rows_per_tg;
            MTLSize grid  = MTLSizeMake((NSUInteger)n_tg, 1, 1);
            MTLSize group = MTLSizeMake(64, 1, 1);  /* 2 × 32 threads */
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            return;
        }
    }

    /* Fallback: original non-repacked kernel */
    size_t qs_size = (size_t)out_dim * n_blocks * 16;
    size_t sc_size = (size_t)out_dim * n_blocks * sizeof(float);

    id<MTLBuffer> w_qs_buf = tq_get_weight_buffer(w_qs, qs_size);
    id<MTLBuffer> w_sc_buf = tq_get_weight_buffer(w_scales, sc_size);
    if (!w_qs_buf || !w_sc_buf) return;

    id<MTLBuffer> indim_buf  = tq_get_dim_buffer((uint32_t)in_dim);
    id<MTLBuffer> outdim_buf = tq_get_dim_buffer((uint32_t)out_dim);

    [enc setComputePipelineState:tq_pipe_matmul_tq_q4];
    [enc setBuffer:input_buf  offset:0 atIndex:0];
    [enc setBuffer:output_buf offset:0 atIndex:1];
    [enc setBuffer:w_qs_buf   offset:0 atIndex:2];
    [enc setBuffer:w_sc_buf   offset:0 atIndex:3];
    [enc setBuffer:indim_buf  offset:0 atIndex:4];
    [enc setBuffer:outdim_buf offset:0 atIndex:5];

    MTLSize grid  = MTLSizeMake((NSUInteger)out_dim, 1, 1);
    MTLSize group = MTLSizeMake(TQ_MATMUL_TG_SIZE, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

/* ---- Helper: encode rmsnorm ---- */
static void encode_rmsnorm(id<MTLComputeCommandEncoder> enc,
                            id<MTLBuffer> x_buf,
                            id<MTLBuffer> w_buf,
                            id<MTLBuffer> out_buf,
                            int n, float eps)
{
    id<MTLBuffer> n_buf   = tq_get_dim_buffer((uint32_t)n);
    /* eps buffer — use a small cached buffer */
    static id<MTLBuffer> eps_buf = nil;
    static float cached_eps = -1.0f;
    if (eps != cached_eps || !eps_buf) {
        eps_buf = [tq_mtl_device newBufferWithLength:sizeof(float)
                                             options:MTLResourceStorageModeShared];
        *(float*)[eps_buf contents] = eps;
        cached_eps = eps;
    }

    [enc setComputePipelineState:tq_pipe_rmsnorm];
    [enc setBuffer:x_buf   offset:0 atIndex:0];
    [enc setBuffer:w_buf   offset:0 atIndex:1];
    [enc setBuffer:out_buf offset:0 atIndex:2];
    [enc setBuffer:n_buf   offset:0 atIndex:3];
    [enc setBuffer:eps_buf offset:0 atIndex:4];

    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

/* ---- Helper: encode RoPE ---- */
static void encode_rope(id<MTLComputeCommandEncoder> enc,
                         id<MTLBuffer> q_buf, id<MTLBuffer> k_buf,
                         int pos, int head_dim, int n_heads, int n_kv_heads,
                         float rope_base)
{
    /* Uniform buffers for RoPE params */
    id<MTLBuffer> pos_buf    = tq_get_dim_buffer((uint32_t)pos);
    id<MTLBuffer> hd_buf     = tq_get_dim_buffer((uint32_t)head_dim);
    id<MTLBuffer> nh_buf     = tq_get_dim_buffer((uint32_t)n_heads);
    id<MTLBuffer> nkv_buf    = tq_get_dim_buffer((uint32_t)n_kv_heads);

    /* rope_base as float buffer */
    static id<MTLBuffer> rope_base_buf = nil;
    static float cached_rope_base = -1.0f;
    if (rope_base != cached_rope_base || !rope_base_buf) {
        rope_base_buf = [tq_mtl_device newBufferWithLength:sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        *(float*)[rope_base_buf contents] = rope_base;
        cached_rope_base = rope_base;
    }

    [enc setComputePipelineState:tq_pipe_rope];
    [enc setBuffer:q_buf         offset:0 atIndex:0];
    [enc setBuffer:k_buf         offset:0 atIndex:1];
    [enc setBuffer:pos_buf       offset:0 atIndex:2];
    [enc setBuffer:hd_buf        offset:0 atIndex:3];
    [enc setBuffer:nh_buf        offset:0 atIndex:4];
    [enc setBuffer:nkv_buf       offset:0 atIndex:5];
    [enc setBuffer:rope_base_buf offset:0 atIndex:6];

    uint total_pairs = (n_heads + n_kv_heads) * (head_dim / 2);
    NSUInteger tg = 256;
    NSUInteger n_tgs = ((NSUInteger)total_pairs + tg - 1) / tg;
    [enc dispatchThreadgroups:MTLSizeMake(n_tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

/* ---- Helper: encode attention scoring (Q*K) ---- */
static void encode_attn_qk(id<MTLComputeCommandEncoder> enc,
                             id<MTLBuffer> q_buf, id<MTLBuffer> k_cache_buf,
                             id<MTLBuffer> scores_buf,
                             int head_dim, int seq_len, int n_heads,
                             int n_kv_heads, int kv_dim)
{
    id<MTLBuffer> hd_buf  = tq_get_dim_buffer((uint32_t)head_dim);
    id<MTLBuffer> sl_buf  = tq_get_dim_buffer((uint32_t)seq_len);
    id<MTLBuffer> nh_buf  = tq_get_dim_buffer((uint32_t)n_heads);
    id<MTLBuffer> nkv_buf = tq_get_dim_buffer((uint32_t)n_kv_heads);
    id<MTLBuffer> kvd_buf = tq_get_dim_buffer((uint32_t)kv_dim);

    [enc setComputePipelineState:tq_pipe_attn_qk];
    [enc setBuffer:q_buf         offset:0 atIndex:0];
    [enc setBuffer:k_cache_buf   offset:0 atIndex:1];
    [enc setBuffer:scores_buf    offset:0 atIndex:2];
    [enc setBuffer:hd_buf        offset:0 atIndex:3];
    [enc setBuffer:sl_buf        offset:0 atIndex:4];
    [enc setBuffer:nh_buf        offset:0 atIndex:5];
    [enc setBuffer:nkv_buf       offset:0 atIndex:6];
    [enc setBuffer:kvd_buf       offset:0 atIndex:7];

    /* One threadgroup per (head, position) pair */
    NSUInteger total_tgs = (NSUInteger)n_heads * (NSUInteger)seq_len;
    [enc dispatchThreadgroups:MTLSizeMake(total_tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

/* ---- Helper: encode softmax ---- */
static void encode_softmax(id<MTLComputeCommandEncoder> enc,
                            id<MTLBuffer> scores_buf,
                            int n_heads, int seq_len)
{
    id<MTLBuffer> len_buf = tq_get_dim_buffer((uint32_t)seq_len);

    [enc setComputePipelineState:tq_pipe_softmax];
    [enc setBuffer:scores_buf offset:0 atIndex:0];
    [enc setBuffer:len_buf    offset:0 atIndex:1];

    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

/* ---- Helper: encode attention value weighted sum ---- */
static void encode_attn_v(id<MTLComputeCommandEncoder> enc,
                           id<MTLBuffer> attn_buf, id<MTLBuffer> v_cache_buf,
                           id<MTLBuffer> out_buf,
                           int head_dim, int seq_len, int n_heads,
                           int n_kv_heads, int kv_dim)
{
    id<MTLBuffer> hd_buf  = tq_get_dim_buffer((uint32_t)head_dim);
    id<MTLBuffer> sl_buf  = tq_get_dim_buffer((uint32_t)seq_len);
    id<MTLBuffer> nh_buf  = tq_get_dim_buffer((uint32_t)n_heads);
    id<MTLBuffer> nkv_buf = tq_get_dim_buffer((uint32_t)n_kv_heads);
    id<MTLBuffer> kvd_buf = tq_get_dim_buffer((uint32_t)kv_dim);

    [enc setComputePipelineState:tq_pipe_attn_v];
    [enc setBuffer:attn_buf      offset:0 atIndex:0];
    [enc setBuffer:v_cache_buf   offset:0 atIndex:1];
    [enc setBuffer:out_buf       offset:0 atIndex:2];
    [enc setBuffer:hd_buf        offset:0 atIndex:3];
    [enc setBuffer:sl_buf        offset:0 atIndex:4];
    [enc setBuffer:nh_buf        offset:0 atIndex:5];
    [enc setBuffer:nkv_buf       offset:0 atIndex:6];
    [enc setBuffer:kvd_buf       offset:0 atIndex:7];

    /* One threadgroup per (head, head_dim_element) pair */
    NSUInteger total_tgs = (NSUInteger)n_heads * (NSUInteger)head_dim;
    [enc dispatchThreadgroups:MTLSizeMake(total_tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

/* ---- Helper: encode element-wise silu (in-place) ---- */
static void encode_silu(id<MTLComputeCommandEncoder> enc,
                         id<MTLBuffer> x_buf, id<MTLBuffer> out_buf, int n)
{
    id<MTLBuffer> n_buf = tq_get_dim_buffer((uint32_t)n);

    [enc setComputePipelineState:tq_pipe_silu];
    [enc setBuffer:x_buf   offset:0 atIndex:0];
    [enc setBuffer:out_buf offset:0 atIndex:1];
    [enc setBuffer:n_buf   offset:0 atIndex:2];

    NSUInteger tg = 256;
    NSUInteger n_tgs = ((NSUInteger)n + tg - 1) / tg;
    [enc dispatchThreadgroups:MTLSizeMake(n_tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

/* ---- Helper: encode GELU-tanh (in-place) ---- */
static void encode_gelu_tanh(id<MTLComputeCommandEncoder> enc,
                              id<MTLBuffer> x_buf, int n)
{
    id<MTLBuffer> n_buf = tq_get_dim_buffer((uint32_t)n);

    [enc setComputePipelineState:tq_pipe_gelu_tanh];
    [enc setBuffer:x_buf offset:0 atIndex:0];
    [enc setBuffer:n_buf offset:0 atIndex:1];

    NSUInteger tg = 256;
    NSUInteger n_tgs = ((NSUInteger)n + tg - 1) / tg;
    [enc dispatchThreadgroups:MTLSizeMake(n_tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

/* ---- Helper: encode element-wise multiply ---- */
static void encode_mul(id<MTLComputeCommandEncoder> enc,
                        id<MTLBuffer> a_buf, id<MTLBuffer> b_buf,
                        id<MTLBuffer> out_buf, int n)
{
    id<MTLBuffer> n_buf = tq_get_dim_buffer((uint32_t)n);

    [enc setComputePipelineState:tq_pipe_mul_elementwise];
    [enc setBuffer:a_buf   offset:0 atIndex:0];
    [enc setBuffer:b_buf   offset:0 atIndex:1];
    [enc setBuffer:out_buf offset:0 atIndex:2];
    [enc setBuffer:n_buf   offset:0 atIndex:3];

    NSUInteger tg = 256;
    NSUInteger n_tgs = ((NSUInteger)n + tg - 1) / tg;
    [enc dispatchThreadgroups:MTLSizeMake(n_tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

/* ---- Helper: encode in-place add (a += b) ---- */
static void encode_add_inplace(id<MTLComputeCommandEncoder> enc,
                                id<MTLBuffer> a_buf, id<MTLBuffer> b_buf, int n)
{
    id<MTLBuffer> n_buf = tq_get_dim_buffer((uint32_t)n);

    [enc setComputePipelineState:tq_pipe_add_inplace];
    [enc setBuffer:a_buf offset:0 atIndex:0];
    [enc setBuffer:b_buf offset:0 atIndex:1];
    [enc setBuffer:n_buf offset:0 atIndex:2];

    NSUInteger tg = 256;
    NSUInteger n_tgs = ((NSUInteger)n + tg - 1) / tg;
    [enc dispatchThreadgroups:MTLSizeMake(n_tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

/* ============================================================
 * Q4 Weight Repacking: Row-major → Column-major blocks
 *
 * GPU threads read consecutive output rows, so transposing the
 * block layout ensures coalesced memory access.
 *
 * Original layout (row-major blocks):
 *   [row0_blk0][row0_blk1]...[row1_blk0][row1_blk1]...
 * Repacked layout (column-major blocks):
 *   [row0_blk0][row1_blk0]...[row0_blk1][row1_blk1]...
 * ============================================================ */

void tq_metal_repack_q4(const uint8_t* src_qs, const float* src_scales,
                         id<MTLBuffer>* out_qs_buf, id<MTLBuffer>* out_sc_buf,
                         int out_dim, int in_dim)
{
    if (!tq_metal_available()) return;

    int n_blocks_per_row = in_dim / 32;
    size_t qs_size = (size_t)out_dim * n_blocks_per_row * 16;
    size_t sc_size = (size_t)out_dim * n_blocks_per_row * sizeof(float);

    /* Allocate GPU buffers */
    *out_qs_buf = [tq_mtl_device newBufferWithLength:qs_size
                                             options:MTLResourceStorageModeShared];
    *out_sc_buf = [tq_mtl_device newBufferWithLength:sc_size
                                             options:MTLResourceStorageModeShared];
    if (!*out_qs_buf || !*out_sc_buf) return;

    uint8_t* dst_qs = (uint8_t*)[*out_qs_buf contents];
    float*   dst_sc = (float*)[*out_sc_buf contents];

    /* Repack to tile-major layout (TILE=32 rows per tile).
     * For each tile t and block b:
     *   dst[t * n_blocks * TILE + b * TILE + row_in_tile] = src[row, b]
     * This ensures SIMD-group threads (32 wide) read consecutive memory. */
    const int TILE = 32;
    int n_tiles = (out_dim + TILE - 1) / TILE;
    for (int t = 0; t < n_tiles; t++) {
        for (int b = 0; b < n_blocks_per_row; b++) {
            for (int tr = 0; tr < TILE; tr++) {
                int row = t * TILE + tr;
                if (row >= out_dim) {
                    /* Pad with zeros for incomplete last tile */
                    size_t dst_qs_off = ((size_t)t * n_blocks_per_row * TILE + (size_t)b * TILE + tr) * 16;
                    size_t dst_sc_off = (size_t)t * n_blocks_per_row * TILE + (size_t)b * TILE + tr;
                    memset(dst_qs + dst_qs_off, 0, 16);
                    dst_sc[dst_sc_off] = 0.0f;
                    continue;
                }
                size_t src_qs_off = ((size_t)row * n_blocks_per_row + b) * 16;
                size_t src_sc_off = (size_t)row * n_blocks_per_row + b;
                size_t dst_qs_off = ((size_t)t * n_blocks_per_row * TILE + (size_t)b * TILE + tr) * 16;
                size_t dst_sc_off = (size_t)t * n_blocks_per_row * TILE + (size_t)b * TILE + tr;
                memcpy(dst_qs + dst_qs_off, src_qs + src_qs_off, 16);
                dst_sc[dst_sc_off] = src_scales[src_sc_off];
            }
        }
    }
}

/* ============================================================
 * Full-layer GPU forward: ONE command buffer, ONE commit
 *
 * Pipeline:
 *   1. rmsnorm(x → xb, w_attn_norm)
 *   2. matmul(xb → q), matmul(xb → k), matmul(xb → v)
 *   3. rope(q, k)
 *   4. store k,v to cache; attention_qk; softmax; attention_v → xb
 *   5. matmul(xb → xb2, wo)
 *   6. add_inplace(x += xb2)
 *   7. rmsnorm(x → xb, w_ffn_norm)
 *   8. matmul(xb → hb, wg), matmul(xb → hb2, wu)
 *   9. silu(hb) or gelu_tanh(hb), mul(hb, hb2 → hb)
 *  10. matmul(hb → xb2, wd)
 *  11. add_inplace(x += xb2)
 *
 * Returns 0 on success, -1 if unavailable (caller uses CPU fallback).
 * ============================================================ */
int tq_metal_forward_layer(
    /* CPU activation state (input/output) */
    float* x,           /* [dim] hidden state — read on entry, written on exit */
    /* CPU KV cache pointers for this layer */
    float* key_cache,   /* [max_seq * kv_dim] — K cache for this layer */
    float* value_cache, /* [max_seq * kv_dim] — V cache for this layer */
    /* Weight pointers (Q4: packed nibbles + scales) */
    const float* w_attn_norm, const float* w_ffn_norm,
    const uint8_t* wq_qs, const float* wq_sc,
    const uint8_t* wk_qs, const float* wk_sc,
    const uint8_t* wv_qs, const float* wv_sc,
    const uint8_t* wo_qs, const float* wo_sc,
    const uint8_t* wg_qs, const float* wg_sc,
    const uint8_t* wu_qs, const float* wu_sc,
    const uint8_t* wd_qs, const float* wd_sc,
    /* Model parameters */
    int dim, int n_heads, int n_kv_heads, int head_dim,
    int inter_dim, int pos, int seq_len, float rope_base, float rms_eps,
    int use_gelu)
{
    @autoreleasepool {
        if (!tq_metal_graph_available()) return -1;
        if (!wq_qs || !wk_qs || !wv_qs || !wo_qs) return -1;
        if (!wg_qs || !wu_qs || !wd_qs) return -1;

        int kv_dim = n_kv_heads * head_dim;
        int q_dim  = n_heads * head_dim;

        /* Upload x to GPU (unified memory — just memcpy to shared buffer) */
        memcpy([g_gpu_x contents], x, (size_t)dim * sizeof(float));

        /* Zero-copy KV cache: wrap CPU cache pointers as Metal buffers.
         * Apple Silicon unified memory means no data copy needed.
         * The GPU reads/writes the same physical memory as CPU. */
        size_t cache_total = (size_t)seq_len * kv_dim * sizeof(float);
        if (cache_total == 0) cache_total = (size_t)kv_dim * sizeof(float);
        id<MTLBuffer> kc_buf = [tq_mtl_device newBufferWithBytesNoCopy:key_cache
                                length:cache_total
                                options:MTLResourceStorageModeShared
                                deallocator:nil];
        id<MTLBuffer> vc_buf = [tq_mtl_device newBufferWithBytesNoCopy:value_cache
                                length:cache_total
                                options:MTLResourceStorageModeShared
                                deallocator:nil];
        if (!kc_buf || !vc_buf) return -1;

        /* Weight norm buffers (zero-copy) */
        id<MTLBuffer> attn_norm_buf = tq_get_weight_buffer(w_attn_norm, (size_t)dim * sizeof(float));
        id<MTLBuffer> ffn_norm_buf  = tq_get_weight_buffer(w_ffn_norm, (size_t)dim * sizeof(float));
        if (!attn_norm_buf || !ffn_norm_buf) return -1;

        /* ===== ONE command buffer, ONE encoder, ONE commit =====
         * All operations encoded sequentially with memory barriers.
         * GPU executes the entire layer pipeline without CPU sync. */
        id<MTLCommandBuffer> cmdBuf = [tq_mtl_queue commandBuffer];
        if (!cmdBuf) return -1;
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        if (!enc) return -1;

        /* ---- Step 1: Pre-attention RMSNorm(x → xb) ---- */
        encode_rmsnorm(enc, g_gpu_x, attn_norm_buf, g_gpu_xb, dim, rms_eps);

        /* ---- Step 2: QKV projections ---- */
        encode_q4_matmul(enc, g_gpu_xb, g_gpu_q, wq_qs, wq_sc, q_dim, dim);
        encode_q4_matmul(enc, g_gpu_xb, g_gpu_k, wk_qs, wk_sc, kv_dim, dim);
        encode_q4_matmul(enc, g_gpu_xb, g_gpu_v, wv_qs, wv_sc, kv_dim, dim);

        /* ---- Step 3: RoPE on Q and K ---- */
        encode_rope(enc, g_gpu_q, g_gpu_k, pos, head_dim, n_heads, n_kv_heads, rope_base);

        /* ---- Step 4: Write K,V to cache ON GPU (no CPU sync!) ---- */
        {
            id<MTLBuffer> pos_buf = tq_get_dim_buffer((uint32_t)pos);
            id<MTLBuffer> kvd_buf = tq_get_dim_buffer((uint32_t)kv_dim);

            /* Write K to cache */
            [enc setComputePipelineState:tq_pipe_kv_cache_write];
            [enc setBuffer:kc_buf   offset:0 atIndex:0];
            [enc setBuffer:g_gpu_k  offset:0 atIndex:1];
            [enc setBuffer:pos_buf  offset:0 atIndex:2];
            [enc setBuffer:kvd_buf  offset:0 atIndex:3];
            {
                NSUInteger tg_w = (NSUInteger)(kv_dim < 256 ? kv_dim : 256);
                [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(tg_w, 1, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* Write V to cache */
            [enc setBuffer:vc_buf   offset:0 atIndex:0];
            [enc setBuffer:g_gpu_v  offset:0 atIndex:1];
            {
                NSUInteger tg_w = (NSUInteger)(kv_dim < 256 ? kv_dim : 256);
                [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(tg_w, 1, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        /* ---- Step 5: Attention (reads from GPU KV cache directly) ---- */
        int attn_seq_len = pos + 1;
        /* Attention uses same encoder — single command buffer! */
        encode_attn_qk(enc, g_gpu_q, kc_buf, g_gpu_att,
                        head_dim, attn_seq_len, n_heads, n_kv_heads, kv_dim);

        /* Softmax over attention scores per head */
        encode_softmax(enc, g_gpu_att, n_heads, attn_seq_len);

        /* Weighted sum of values → xb (reuse xb for attention output) */
        encode_attn_v(enc, g_gpu_att, vc_buf, g_gpu_xb,
                      head_dim, attn_seq_len, n_heads, n_kv_heads, kv_dim);

        /* ---- Step 5: Output projection (xb → xb2) ---- */
        encode_q4_matmul(enc, g_gpu_xb, g_gpu_xb2, wo_qs, wo_sc, dim, q_dim);

        /* ---- Step 6: Residual add (x += xb2) ---- */
        encode_add_inplace(enc, g_gpu_x, g_gpu_xb2, dim);

        /* ---- Step 7: Pre-FFN RMSNorm(x → xb) ---- */
        encode_rmsnorm(enc, g_gpu_x, ffn_norm_buf, g_gpu_xb, dim, rms_eps);

        /* ---- Step 8: FFN gate + up projections ---- */
        encode_q4_matmul(enc, g_gpu_xb, g_gpu_hb,  wg_qs, wg_sc, inter_dim, dim);
        encode_q4_matmul(enc, g_gpu_xb, g_gpu_hb2, wu_qs, wu_sc, inter_dim, dim);

        /* ---- Step 9: Activation + gate multiply ---- */
        if (use_gelu) {
            encode_gelu_tanh(enc, g_gpu_hb, inter_dim);
        } else {
            encode_silu(enc, g_gpu_hb, g_gpu_hb, inter_dim);
        }
        encode_mul(enc, g_gpu_hb, g_gpu_hb2, g_gpu_hb, inter_dim);

        /* ---- Step 10: Down projection (hb → xb2) ---- */
        encode_q4_matmul(enc, g_gpu_hb, g_gpu_xb2, wd_qs, wd_sc, dim, inter_dim);

        /* ---- Step 11: Residual add (x += xb2) ---- */
        encode_add_inplace(enc, g_gpu_x, g_gpu_xb2, dim);

        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status == MTLCommandBufferStatusError) {
            NSLog(@"TurboQuant: GPU graph Phase B error: %@", cmdBuf.error);
            return -1;
        }

        /* Copy result back to CPU */
        memcpy(x, [g_gpu_x contents], (size_t)dim * sizeof(float));

        return 0; /* Success */
    }
}

/* ============================================================
 * Legacy layer forward (backward compat, QKV-only)
 * ============================================================ */
int tq_metal_layer_forward(
    float* xb, float* xb2, float* q, float* k, float* v,
    float* hb, float* hb2,
    const uint8_t* wq_qs, const float* wq_scales,
    const uint8_t* wk_qs, const float* wk_scales,
    const uint8_t* wv_qs, const float* wv_scales,
    const uint8_t* wo_qs, const float* wo_scales,
    const uint8_t* wg_qs, const float* wg_scales,
    const uint8_t* wu_qs, const float* wu_scales,
    const uint8_t* wd_qs, const float* wd_scales,
    int dim, int q_dim, int kv_dim, int inter_dim)
{
    @autoreleasepool {
        if (!tq_metal_available() || !g_gpu_xb) return -1;

        memcpy([g_gpu_xb contents], xb, (size_t)dim * sizeof(float));

        id<MTLCommandBuffer> cmdBuf = [tq_mtl_queue commandBuffer];
        if (!cmdBuf) return -1;
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        if (!enc) return -1;

        if (wq_qs) encode_q4_matmul(enc, g_gpu_xb, g_gpu_q, wq_qs, wq_scales, q_dim, dim);
        if (wk_qs) encode_q4_matmul(enc, g_gpu_xb, g_gpu_k, wk_qs, wk_scales, kv_dim, dim);
        if (wv_qs) encode_q4_matmul(enc, g_gpu_xb, g_gpu_v, wv_qs, wv_scales, kv_dim, dim);

        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status == MTLCommandBufferStatusError) return -1;

        memcpy(q, [g_gpu_q contents], (size_t)q_dim * sizeof(float));
        memcpy(k, [g_gpu_k contents], (size_t)kv_dim * sizeof(float));
        memcpy(v, [g_gpu_v contents], (size_t)kv_dim * sizeof(float));

        return 0;
    }
}

#endif /* __APPLE__ */
