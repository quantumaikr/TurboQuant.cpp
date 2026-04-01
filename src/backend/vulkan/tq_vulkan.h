/**
 * TurboQuant -- Vulkan compute backend
 *
 * Provides Vulkan 1.1 compute shader pipelines for all TQ quantization
 * types. Targets AMD GPUs via standard Vulkan, replacing CUDA-specific
 * warp ops with subgroup operations (VK_EXT_subgroup_size_control).
 */
#ifndef TQ_VULKAN_H
#define TQ_VULKAN_H

#ifdef TQ_BUILD_VULKAN

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Constants (matching CUDA backend)
 * ============================================================ */

#define TQ_VK_WORKGROUP_SIZE   256
#define TQ_VK_BK              128
#define TQ_VK_BK_QJL          256
#define TQ_VK_SKETCH_DIM      256
#define TQ_VK_OUTLIERS         4
#define TQ_VK_MAX_PIPELINES   16

/* ============================================================
 * Shader pipeline identifiers
 * ============================================================ */

typedef enum {
    TQ_VK_PIPE_POLAR_QUANTIZE  = 0,
    TQ_VK_PIPE_POLAR_ATTENTION = 1,
    TQ_VK_PIPE_QJL_QUANTIZE    = 2,
    TQ_VK_PIPE_QJL_ATTENTION   = 3,
    TQ_VK_PIPE_TURBO_QUANTIZE  = 4,
    TQ_VK_PIPE_TURBO_ATTENTION = 5,
    TQ_VK_PIPE_VALUE_QUANT_4B  = 6,
    TQ_VK_PIPE_VALUE_QUANT_2B  = 7,
    TQ_VK_PIPE_VALUE_DEQUANT_MATMUL_4B = 8,
    TQ_VK_PIPE_COUNT           = 9
} tq_vk_pipeline_id;

/* ============================================================
 * Push constant structures (passed to shaders)
 * ============================================================ */

typedef struct {
    int32_t n;
    int32_t head_dim;
    int32_t num_blocks;
    int32_t _pad;
} tq_vk_polar_params_t;

typedef struct {
    int32_t num_keys;
    int32_t emb_dim;
    int32_t sketch_dim;
    int32_t _pad;
} tq_vk_qjl_params_t;

typedef struct {
    int32_t n;
    int32_t head_dim;
    int32_t num_blocks;
    int32_t _pad;
} tq_vk_turbo_params_t;

typedef struct {
    int32_t n;
    int32_t num_blocks;
    int32_t quant_bits;  /* 2 or 4 */
    int32_t _pad;
} tq_vk_value_params_t;

typedef struct {
    int32_t seq_len;
    int32_t head_dim;
    int32_t num_blocks;
    int32_t _pad;
} tq_vk_attention_params_t;

typedef struct {
    int32_t seq_len;
    int32_t head_dim;
    int32_t num_dim_blocks;
    int32_t _pad;
} tq_vk_value_matmul_params_t;

/* ============================================================
 * Vulkan backend state
 * ============================================================ */

typedef struct {
    int                    initialized;

    /* Vulkan core objects */
    VkInstance             instance;
    VkPhysicalDevice       physical_device;
    VkDevice               device;
    VkQueue                compute_queue;
    uint32_t               compute_queue_family;

    /* Device properties */
    char                   device_name[256];
    uint32_t               subgroup_size;
    VkPhysicalDeviceMemoryProperties mem_props;

    /* Command infrastructure */
    VkCommandPool          command_pool;
    VkCommandBuffer        command_buffer;
    VkFence                fence;

    /* Descriptor set layout and pool (shared across pipelines) */
    VkDescriptorSetLayout  desc_layout;
    VkDescriptorPool       desc_pool;

    /* Pipeline layout (shared: push constants + descriptor set) */
    VkPipelineLayout       pipeline_layout;

    /* Compute pipelines (one per kernel) */
    VkPipeline             pipelines[TQ_VK_PIPE_COUNT];
    VkShaderModule         shader_modules[TQ_VK_PIPE_COUNT];
} tq_vk_state_t;

/* ============================================================
 * Vulkan buffer wrapper
 * ============================================================ */

typedef struct {
    VkBuffer       buffer;
    VkDeviceMemory memory;
    VkDeviceSize   size;
    void*          mapped;   /* Non-NULL if persistently mapped */
} tq_vk_buffer_t;

/* ============================================================
 * Backend lifecycle
 * ============================================================ */

int  tq_init_vulkan_backend(void);
void tq_shutdown_vulkan_backend(void);
int  tq_vulkan_is_available(void);
const char* tq_vulkan_device_name(void);

/* ============================================================
 * Buffer management
 * ============================================================ */

int  tq_vk_create_buffer(tq_vk_buffer_t* buf, VkDeviceSize size,
                          VkBufferUsageFlags usage,
                          VkMemoryPropertyFlags mem_flags);
void tq_vk_destroy_buffer(tq_vk_buffer_t* buf);
int  tq_vk_upload(tq_vk_buffer_t* buf, const void* data, VkDeviceSize size);
int  tq_vk_download(const tq_vk_buffer_t* buf, void* data, VkDeviceSize size);

/* ============================================================
 * Dispatch functions (host-callable, matching CUDA API pattern)
 * ============================================================ */

void tq_polar_quantize_vulkan(const float* keys, void* out,
                               int n, int head_dim);
void tq_polar_attention_vulkan(const float* query, const void* keys,
                                float* scores, int seq_len, int head_dim);

void tq_qjl_quantize_vulkan(const float* keys, void* out,
                              int num_keys, int emb_dim);
void tq_qjl_attention_vulkan(const float* query, const void* keys,
                               float* scores, int seq_len, int head_dim);

void tq_turbo_quantize_vulkan(const float* keys, void* out,
                                int n, int head_dim);
void tq_turbo_attention_vulkan(const float* query, const void* keys,
                                 float* scores, int seq_len, int head_dim);

void tq_value_quantize_4b_vulkan(const float* values, void* out, int n);
void tq_value_quantize_2b_vulkan(const float* values, void* out, int n);
void tq_value_dequant_matmul_4b_vulkan(const float* attn_weights,
                                         const void* values, float* output,
                                         int seq_len, int head_dim);

/* ============================================================
 * Dispatch table registration (for tq_traits integration)
 * ============================================================ */

void* tq_vulkan_get_quantize_fn(int type_id);
void* tq_vulkan_get_attention_fn(int type_id);

#ifdef __cplusplus
}
#endif

#endif /* TQ_BUILD_VULKAN */
#endif /* TQ_VULKAN_H */
