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

/* Reusable input/dimension buffers (shared across batch and immediate modes) */
static id<MTLBuffer> tq_shared_input_buf  = nil;
static uint32_t      tq_shared_input_dim  = 0;
static id<MTLBuffer> tq_shared_indim_buf  = nil;
static id<MTLBuffer> tq_shared_outdim_buf = nil;

/* Threadgroup size for matmul kernels — must match shader constant */
static const uint32_t TQ_MATMUL_TG_SIZE = 256;

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
void tq_metal_batch_flush(void) {
    if (!tq_batch.active) return;

    @autoreleasepool {
        if (tq_batch.encoder) {
            [tq_batch.encoder endEncoding];
            tq_batch.encoder = nil;
        }

        if (tq_batch.cmd_buf && tq_batch.n_copies > 0) {
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
             * Auto-flush if batch is full. */
            if (tq_batch.n_copies >= TQ_BATCH_MAX_OPS) {
                tq_metal_batch_flush();
                /* Restart encoder for next operations */
            }
            output_buf = [tq_mtl_device
                newBufferWithLength:output_size
                            options:MTLResourceStorageModeShared];
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
         * per-dispatch dimension buffers. For simplicity, create small ones. */
        id<MTLBuffer> indim_buf = nil;
        id<MTLBuffer> outdim_buf = nil;
        if (tq_batch.active) {
            /* Allocate small uniform buffers per dispatch in batch mode */
            indim_buf = [tq_mtl_device
                newBufferWithLength:sizeof(uint32_t)
                            options:MTLResourceStorageModeShared];
            outdim_buf = [tq_mtl_device
                newBufferWithLength:sizeof(uint32_t)
                            options:MTLResourceStorageModeShared];
            if (!indim_buf || !outdim_buf) return -1;
        } else {
            indim_buf = tq_shared_indim_buf;
            outdim_buf = tq_shared_outdim_buf;
        }
        *(uint32_t*)[indim_buf contents]  = (uint32_t)in_dim;
        *(uint32_t*)[outdim_buf contents] = (uint32_t)out_dim;

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

        /* --- Create input buffer --- */
        size_t input_bytes = (size_t)hidden_dim * sizeof(float);
        id<MTLBuffer> input_buf = [tq_mtl_device newBufferWithBytes:input
                                                              length:input_bytes
                                                             options:MTLResourceStorageModeShared];
        if (!input_buf) return -1;

        /* --- Create intermediate buffers --- */
        size_t inter_bytes = (size_t)num_active * (size_t)expert_dim * sizeof(float);
        id<MTLBuffer> gate_buf = [tq_mtl_device newBufferWithLength:inter_bytes
                                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> up_buf = [tq_mtl_device newBufferWithLength:inter_bytes
                                                            options:MTLResourceStorageModeShared];
        if (!gate_buf || !up_buf) return -1;

        /* output_buf removed: hybrid mode returns SwiGLU results via hb_output,
         * down projection + accumulate happens on CPU in tq_moe.c */

        /* --- Create params buffer --- */
        id<MTLBuffer> params_buf = [tq_mtl_device newBufferWithBytes:&params
                                                               length:sizeof(MoeGpuParams)
                                                              options:MTLResourceStorageModeShared];
        if (!params_buf) return -1;

        /* --- Create command buffer and encoder --- */
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

            /* One threadgroup per (expert, row): num_active * expert_dim total */
            NSUInteger n_tgs = (NSUInteger)num_active * (NSUInteger)expert_dim;
            MTLSize gridSize = MTLSizeMake(n_tgs, 1, 1);
            MTLSize tgSize   = MTLSizeMake(TQ_MATMUL_TG_SIZE, 1, 1);
            [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
            [enc endEncoding];
        }

        /* --- Phase 1: commit and wait to isolate hang --- */
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status == MTLCommandBufferStatusError) {
            NSLog(@"TurboQuant MoE: Phase 1 (gate+up) FAILED: %@", cmdBuf.error);
            return -1;
        }
        NSLog(@"TurboQuant MoE: Phase 1 (gate+up) completed OK");

#ifdef TQ_MOE_DEBUG_VALIDATE
        /* === Debug: compare GPU gate output for expert 0 vs CPU tq_matmul_gguf === */
        {
            /* tq_matmul_gguf declared in tq_gguf.h (already included) */
            float* gpu_gate = (float*)[gate_buf contents];
            float* cpu_gate = (float*)malloc((size_t)expert_dim * sizeof(float));
            if (cpu_gate) {
                /* CPU matmul for expert 0's gate weights */
                const uint8_t* gate_w = (const uint8_t*)weight_base + gate_offsets[0];
                tq_ggml_dtype gt0 = gate_types_in ? (tq_ggml_dtype)gate_types_in[0]
                                                  : (tq_ggml_dtype)weight_type;
                tq_matmul_gguf(cpu_gate, input, gate_w, gt0, expert_dim, hidden_dim);

                /* Compare first 8 and last 8 values */
                NSLog(@"TurboQuant MoE DEBUG: gate expert 0 comparison (first 8):");
                float max_err = 0.0f;
                for (int i = 0; i < expert_dim; i++) {
                    float err = fabsf(gpu_gate[i] - cpu_gate[i]);
                    if (err > max_err) max_err = err;
                    if (i < 8 || i >= expert_dim - 4) {
                        NSLog(@"  [%d] GPU=%.6f CPU=%.6f err=%.6f", i, gpu_gate[i], cpu_gate[i], err);
                    }
                }
                NSLog(@"TurboQuant MoE DEBUG: gate max_err=%.6f across %d elements", max_err, expert_dim);
                if (max_err > 0.01f) {
                    NSLog(@"TurboQuant MoE DEBUG: *** MISMATCH DETECTED *** — weight offset or decoding bug");
                }
                free(cpu_gate);
            }
        }
#endif /* TQ_MOE_DEBUG_VALIDATE */

        /* --- New command buffer for Phase 2 --- */
        cmdBuf = [tq_mtl_queue commandBuffer];
        if (!cmdBuf) return -1;

        /* ======== Phase 2: SwiGLU ======== */
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            if (!enc) return -1;

            [enc setComputePipelineState:tq_pipe_moe_swiglu];
            [enc setBuffer:gate_buf    offset:0 atIndex:0];
            [enc setBuffer:up_buf      offset:0 atIndex:1];
            [enc setBuffer:params_buf  offset:0 atIndex:2];

            NSUInteger n_threads = (NSUInteger)num_active * (NSUInteger)expert_dim;
            NSUInteger tg = 256;
            NSUInteger n_tgs = (n_threads + tg - 1) / tg;
            MTLSize gridSize = MTLSizeMake(n_tgs, 1, 1);
            MTLSize tgSize   = MTLSizeMake(tg, 1, 1);
            [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
            [enc endEncoding];
        }

        /* --- Phase 2: commit and wait to isolate hang --- */
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status == MTLCommandBufferStatusError) {
            NSLog(@"TurboQuant MoE: Phase 2 (SwiGLU) FAILED: %@", cmdBuf.error);
            return -1;
        }
        NSLog(@"TurboQuant MoE: Phase 2 (SwiGLU) completed OK");

        /* ======== Phase 3: down projection + weighted accumulate (GPU) ========
         * Previously skipped due to IQ2_S shader hanging with constant array.
         * Now fixed: IQ2_S codebook passed as device buffer (buffer 4). */
        {
            /* Create output buffer for hidden_dim results */
            size_t output_bytes = (size_t)hidden_dim * sizeof(float);
            id<MTLBuffer> output_buf = [tq_mtl_device newBufferWithLength:output_bytes
                                                                  options:MTLResourceStorageModeShared];
            if (!output_buf) {
                /* Fallback to hybrid if buffer creation fails */
                memcpy(hb_output, [gate_buf contents], inter_bytes);
                return 1;
            }

            /* New command buffer for Phase 3 */
            cmdBuf = [tq_mtl_queue commandBuffer];
            if (!cmdBuf) {
                memcpy(hb_output, [gate_buf contents], inter_bytes);
                return 1;
            }

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

            /* One threadgroup per output row (hidden_dim total) */
            MTLSize gridSize3 = MTLSizeMake((NSUInteger)hidden_dim, 1, 1);
            MTLSize tgSize3   = MTLSizeMake(TQ_MATMUL_TG_SIZE, 1, 1);
            [enc dispatchThreadgroups:gridSize3 threadsPerThreadgroup:tgSize3];
            [enc endEncoding];

            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            if (cmdBuf.status == MTLCommandBufferStatusError) {
                NSLog(@"TurboQuant MoE: Phase 3 (down+accum) FAILED: %@", cmdBuf.error);
                /* Fallback to hybrid on failure */
                memcpy(hb_output, [gate_buf contents], inter_bytes);
                return 1;
            }
            NSLog(@"TurboQuant MoE: Phase 3 (down+accum) completed OK");

            /* Copy result to output */
            memcpy(output, [output_buf contents], output_bytes);

            /* Also copy hb for potential caller use */
            memcpy(hb_output, [gate_buf contents], inter_bytes);

            return 0; /* Full GPU success */
        }
    }
}

#endif /* __APPLE__ */
