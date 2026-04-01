/**
 * TurboQuant — Metal backend dispatch (Objective-C host code)
 *
 * Loads the .metallib shader library, creates compute pipelines,
 * and provides the dispatch interface for Metal GPU kernels.
 *
 * Includes matmul dispatch for GGUF quantized weight formats
 * (IQ2_XXS, Q8_0, Q4_K) with buffer caching for MoE workloads.
 */
#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "turboquant/tq_gguf.h"

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
static id<MTLComputePipelineState> tq_pipe_matmul_q8_0     = nil;
static id<MTLComputePipelineState> tq_pipe_matmul_q4_k     = nil;

/* ============================================================
 * Buffer cache for matmul dispatch
 *
 * MoE inference issues many small matmuls (e.g. 512x2048) with
 * the same weight tensors. Creating/destroying MTLBuffers per
 * call is expensive. We cache the last-used buffers and reuse
 * them when dimensions match.
 * ============================================================ */

typedef struct {
    id<MTLBuffer> weight_buf;
    id<MTLBuffer> input_buf;
    id<MTLBuffer> output_buf;
    id<MTLBuffer> indim_buf;
    id<MTLBuffer> outdim_buf;
    const void*   last_weight_ptr;
    size_t        last_weight_size;
    uint32_t      last_in_dim;
    uint32_t      last_out_dim;
} tq_matmul_buf_cache_t;

static tq_matmul_buf_cache_t tq_buf_cache = {
    .weight_buf = nil, .input_buf = nil, .output_buf = nil,
    .indim_buf = nil, .outdim_buf = nil,
    .last_weight_ptr = NULL, .last_weight_size = 0,
    .last_in_dim = 0, .last_out_dim = 0
};

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
                opts.fastMathEnabled = YES;
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
        tq_pipe_matmul_q8_0 = makePipe(@"matmul_q8_0");
        tq_pipe_matmul_q4_k = makePipe(@"matmul_q4_k");

        NSLog(@"TurboQuant: Metal backend initialized on %@", tq_mtl_device.name);
        return 0;
    }
}

/**
 * Free Metal resources.
 */
void tq_free_metal_backend(void) {
    /* KV cache pipelines */
    tq_pipe_polar_quantize = nil;
    tq_pipe_polar_attention = nil;
    tq_pipe_qjl_quantize = nil;
    tq_pipe_qjl_attention = nil;
    tq_pipe_value_quantize = nil;

    /* Matmul pipelines */
    tq_pipe_matmul_iq2_xxs = nil;
    tq_pipe_matmul_q8_0 = nil;
    tq_pipe_matmul_q4_k = nil;

    /* Buffer cache */
    tq_buf_cache.weight_buf = nil;
    tq_buf_cache.input_buf = nil;
    tq_buf_cache.output_buf = nil;
    tq_buf_cache.indim_buf = nil;
    tq_buf_cache.outdim_buf = nil;
    tq_buf_cache.last_weight_ptr = NULL;
    tq_buf_cache.last_weight_size = 0;
    tq_buf_cache.last_in_dim = 0;
    tq_buf_cache.last_out_dim = 0;

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
 * Metal matmul dispatch
 *
 * Dispatches fused dequant-matmul on GPU for supported GGUF types.
 * Returns 0 on success, -1 if the type is not supported on Metal.
 *
 * Buffer management:
 *   - Weight buffer: reused if same pointer and size
 *   - Input buffer: reused if in_dim matches, contents updated
 *   - Output buffer: reused if out_dim matches
 *   - Dimension uniform buffers: reused if values match
 *
 * For MoE workloads, the weight pointer changes per expert but
 * dimensions stay the same, so input/output/dim buffers are reused.
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

        /* --- Weight buffer: reuse if same pointer+size --- */
        if (tq_buf_cache.last_weight_ptr != weight ||
            tq_buf_cache.last_weight_size != weight_size) {
            tq_buf_cache.weight_buf = [tq_mtl_device
                newBufferWithBytes:weight
                            length:weight_size
                           options:MTLResourceStorageModeShared];
            if (!tq_buf_cache.weight_buf) return -1;
            tq_buf_cache.last_weight_ptr = weight;
            tq_buf_cache.last_weight_size = weight_size;
        }

        /* --- Input buffer: reuse if size matches, update contents --- */
        if (tq_buf_cache.last_in_dim != (uint32_t)in_dim || !tq_buf_cache.input_buf) {
            tq_buf_cache.input_buf = [tq_mtl_device
                newBufferWithLength:input_size
                            options:MTLResourceStorageModeShared];
            if (!tq_buf_cache.input_buf) return -1;
            tq_buf_cache.last_in_dim = (uint32_t)in_dim;
        }
        memcpy([tq_buf_cache.input_buf contents], x, (size_t)in_dim * sizeof(float));

        /* --- Output buffer: reuse if size matches --- */
        if (tq_buf_cache.last_out_dim != (uint32_t)out_dim || !tq_buf_cache.output_buf) {
            tq_buf_cache.output_buf = [tq_mtl_device
                newBufferWithLength:output_size
                            options:MTLResourceStorageModeShared];
            if (!tq_buf_cache.output_buf) return -1;
            tq_buf_cache.last_out_dim = (uint32_t)out_dim;
        }

        /* --- Dimension uniform buffers --- */
        if (!tq_buf_cache.indim_buf) {
            tq_buf_cache.indim_buf = [tq_mtl_device
                newBufferWithLength:sizeof(uint32_t)
                            options:MTLResourceStorageModeShared];
        }
        if (!tq_buf_cache.outdim_buf) {
            tq_buf_cache.outdim_buf = [tq_mtl_device
                newBufferWithLength:sizeof(uint32_t)
                            options:MTLResourceStorageModeShared];
        }
        *(uint32_t*)[tq_buf_cache.indim_buf contents]  = (uint32_t)in_dim;
        *(uint32_t*)[tq_buf_cache.outdim_buf contents] = (uint32_t)out_dim;

        /* --- Encode and dispatch --- */
        id<MTLCommandBuffer> cmdBuf = [tq_mtl_queue commandBuffer];
        if (!cmdBuf) return -1;

        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        if (!enc) return -1;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:tq_buf_cache.weight_buf offset:0 atIndex:0];
        [enc setBuffer:tq_buf_cache.input_buf  offset:0 atIndex:1];
        [enc setBuffer:tq_buf_cache.output_buf offset:0 atIndex:2];
        [enc setBuffer:tq_buf_cache.indim_buf  offset:0 atIndex:3];
        [enc setBuffer:tq_buf_cache.outdim_buf offset:0 atIndex:4];

        /* Threadgroup shared memory for input caching */
        NSUInteger shared_mem = (NSUInteger)in_dim * sizeof(float);
        /* Cap at device threadgroup memory limit */
        NSUInteger max_shared = [tq_mtl_device maxThreadgroupMemoryLength];
        if (shared_mem > max_shared) {
            /* Input too large for shared memory — still works but
             * shader will read from device memory (slower for MoE) */
            shared_mem = max_shared;
        }
        [enc setThreadgroupMemoryLength:shared_mem atIndex:0];

        /* One threadgroup per output row, TQ_MATMUL_TG_SIZE threads each */
        MTLSize gridSize       = MTLSizeMake((NSUInteger)out_dim, 1, 1);
        MTLSize threadgroupSz  = MTLSizeMake(TQ_MATMUL_TG_SIZE, 1, 1);

        [enc dispatchThreadgroups:gridSize
            threadsPerThreadgroup:threadgroupSz];
        [enc endEncoding];

        /* Synchronous execution — wait for GPU to finish */
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        /* Check for errors */
        if (cmdBuf.status == MTLCommandBufferStatusError) {
            NSLog(@"TurboQuant: Metal matmul error: %@", cmdBuf.error);
            return -1;
        }

        /* Copy result back */
        memcpy(out, [tq_buf_cache.output_buf contents],
               (size_t)out_dim * sizeof(float));

        return 0;
    }
}

#endif /* __APPLE__ */
