/**
 * TurboQuant -- Vulkan backend initialization
 *
 * Creates Vulkan instance, selects a compute-capable physical device,
 * creates logical device with compute queue, builds descriptor set layouts,
 * pipeline layouts, and compiles SPIR-V compute pipelines.
 *
 * Requires Vulkan 1.1 for subgroup operations.
 */
#ifdef TQ_BUILD_VULKAN

#include "tq_vulkan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Global state
 * ============================================================ */

static tq_vk_state_t g_vk_state = {0};

/* Internal accessor for dispatch module */
tq_vk_state_t* tq_vk_get_state(void) {
    return &g_vk_state;
}

/* ============================================================
 * SPIR-V shader bytecode (embedded)
 *
 * In production, these would be compiled from .comp files using
 * glslangValidator and embedded via xxd or a CMake shader
 * compilation step. Here we declare them as external symbols
 * that the build system provides.
 * ============================================================ */

extern const uint32_t tq_polar_quantize_spv[];
extern const size_t   tq_polar_quantize_spv_size;
extern const uint32_t tq_polar_attention_spv[];
extern const size_t   tq_polar_attention_spv_size;
extern const uint32_t tq_qjl_quantize_spv[];
extern const size_t   tq_qjl_quantize_spv_size;
extern const uint32_t tq_qjl_attention_spv[];
extern const size_t   tq_qjl_attention_spv_size;
extern const uint32_t tq_turbo_quantize_spv[];
extern const size_t   tq_turbo_quantize_spv_size;
extern const uint32_t tq_turbo_attention_spv[];
extern const size_t   tq_turbo_attention_spv_size;
extern const uint32_t tq_value_quant_4b_spv[];
extern const size_t   tq_value_quant_4b_spv_size;
extern const uint32_t tq_value_quant_2b_spv[];
extern const size_t   tq_value_quant_2b_spv_size;
extern const uint32_t tq_value_dequant_matmul_4b_spv[];
extern const size_t   tq_value_dequant_matmul_4b_spv_size;

/* SPIR-V lookup table */
static const struct {
    const uint32_t** code;
    const size_t*    size;
} g_shader_table[TQ_VK_PIPE_COUNT] = {
    { &tq_polar_quantize_spv,          &tq_polar_quantize_spv_size },
    { &tq_polar_attention_spv,         &tq_polar_attention_spv_size },
    { &tq_qjl_quantize_spv,           &tq_qjl_quantize_spv_size },
    { &tq_qjl_attention_spv,          &tq_qjl_attention_spv_size },
    { &tq_turbo_quantize_spv,         &tq_turbo_quantize_spv_size },
    { &tq_turbo_attention_spv,        &tq_turbo_attention_spv_size },
    { &tq_value_quant_4b_spv,         &tq_value_quant_4b_spv_size },
    { &tq_value_quant_2b_spv,         &tq_value_quant_2b_spv_size },
    { &tq_value_dequant_matmul_4b_spv,&tq_value_dequant_matmul_4b_spv_size },
};

/* ============================================================
 * Helper: find memory type index
 * ============================================================ */

static int tq_vk_find_memory_type(uint32_t type_bits,
                                    VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < g_vk_state.mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (g_vk_state.mem_props.memoryTypes[i].propertyFlags & props) == props) {
            return (int)i;
        }
    }
    return -1;
}

/* ============================================================
 * Helper: create shader module from SPIR-V
 * ============================================================ */

static VkShaderModule tq_vk_create_shader(const uint32_t* code, size_t size) {
    VkShaderModuleCreateInfo ci = {0};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = size;
    ci.pCode    = code;

    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(g_vk_state.device, &ci, NULL, &module) != VK_SUCCESS) {
        fprintf(stderr, "TQ Vulkan: Failed to create shader module\n");
        return VK_NULL_HANDLE;
    }
    return module;
}

/* ============================================================
 * Create Vulkan instance
 * ============================================================ */

static int tq_vk_create_instance(void) {
    VkApplicationInfo app_info = {0};
    app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName   = "TurboQuant";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName        = "TurboQuant.cpp";
    app_info.engineVersion      = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion         = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ci = {0};
    ci.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app_info;

    if (vkCreateInstance(&ci, NULL, &g_vk_state.instance) != VK_SUCCESS) {
        fprintf(stderr, "TQ Vulkan: Failed to create instance\n");
        return -1;
    }
    return 0;
}

/* ============================================================
 * Select physical device with compute queue
 * ============================================================ */

static int tq_vk_select_device(void) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(g_vk_state.instance, &count, NULL);
    if (count == 0) {
        fprintf(stderr, "TQ Vulkan: No Vulkan devices found\n");
        return -1;
    }

    VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(
        count * sizeof(VkPhysicalDevice));
    if (!devices) return -1;
    vkEnumeratePhysicalDevices(g_vk_state.instance, &count, devices);

    /* Prefer discrete GPU, fall back to any compute-capable device */
    int best = -1;
    int best_score = -1;

    for (uint32_t i = 0; i < count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);

        /* Find a compute queue family */
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &qf_count, NULL);
        VkQueueFamilyProperties* qf_props = (VkQueueFamilyProperties*)malloc(
            qf_count * sizeof(VkQueueFamilyProperties));
        if (!qf_props) continue;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &qf_count, qf_props);

        int has_compute = 0;
        for (uint32_t q = 0; q < qf_count; q++) {
            if (qf_props[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                has_compute = 1;
                break;
            }
        }
        free(qf_props);

        if (!has_compute) continue;

        int score = 0;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            score = 100;
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
            score = 50;
        else
            score = 10;

        if (score > best_score) {
            best_score = score;
            best = (int)i;
        }
    }

    if (best < 0) {
        free(devices);
        fprintf(stderr, "TQ Vulkan: No compute-capable device found\n");
        return -1;
    }

    g_vk_state.physical_device = devices[best];
    free(devices);

    /* Store device properties */
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_vk_state.physical_device, &props);
    strncpy(g_vk_state.device_name, props.deviceName,
            sizeof(g_vk_state.device_name) - 1);

    /* Query subgroup properties (Vulkan 1.1) */
    VkPhysicalDeviceSubgroupProperties subgroup_props = {0};
    subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

    VkPhysicalDeviceProperties2 props2 = {0};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup_props;
    vkGetPhysicalDeviceProperties2(g_vk_state.physical_device, &props2);

    g_vk_state.subgroup_size = subgroup_props.subgroupSize;

    /* Memory properties */
    vkGetPhysicalDeviceMemoryProperties(g_vk_state.physical_device,
                                         &g_vk_state.mem_props);

    /* Find compute queue family */
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk_state.physical_device,
                                              &qf_count, NULL);
    VkQueueFamilyProperties* qf_props = (VkQueueFamilyProperties*)malloc(
        qf_count * sizeof(VkQueueFamilyProperties));
    if (!qf_props) return -1;
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk_state.physical_device,
                                              &qf_count, qf_props);

    /* Prefer a dedicated compute queue (no graphics), else any compute */
    g_vk_state.compute_queue_family = UINT32_MAX;
    for (uint32_t q = 0; q < qf_count; q++) {
        if ((qf_props[q].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(qf_props[q].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            g_vk_state.compute_queue_family = q;
            break;
        }
    }
    if (g_vk_state.compute_queue_family == UINT32_MAX) {
        for (uint32_t q = 0; q < qf_count; q++) {
            if (qf_props[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                g_vk_state.compute_queue_family = q;
                break;
            }
        }
    }
    free(qf_props);

    if (g_vk_state.compute_queue_family == UINT32_MAX) {
        fprintf(stderr, "TQ Vulkan: No compute queue family found\n");
        return -1;
    }

    return 0;
}

/* ============================================================
 * Create logical device and queue
 * ============================================================ */

static int tq_vk_create_device(void) {
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_ci = {0};
    queue_ci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_ci.queueFamilyIndex = g_vk_state.compute_queue_family;
    queue_ci.queueCount       = 1;
    queue_ci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dev_ci = {0};
    dev_ci.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_ci.queueCreateInfoCount = 1;
    dev_ci.pQueueCreateInfos    = &queue_ci;

    if (vkCreateDevice(g_vk_state.physical_device, &dev_ci, NULL,
                        &g_vk_state.device) != VK_SUCCESS) {
        fprintf(stderr, "TQ Vulkan: Failed to create logical device\n");
        return -1;
    }

    vkGetDeviceQueue(g_vk_state.device, g_vk_state.compute_queue_family, 0,
                      &g_vk_state.compute_queue);
    return 0;
}

/* ============================================================
 * Create command pool, buffer, and fence
 * ============================================================ */

static int tq_vk_create_command_infra(void) {
    VkCommandPoolCreateInfo pool_ci = {0};
    pool_ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_ci.queueFamilyIndex = g_vk_state.compute_queue_family;

    if (vkCreateCommandPool(g_vk_state.device, &pool_ci, NULL,
                             &g_vk_state.command_pool) != VK_SUCCESS) {
        return -1;
    }

    VkCommandBufferAllocateInfo alloc_info = {0};
    alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool        = g_vk_state.command_pool;
    alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(g_vk_state.device, &alloc_info,
                                  &g_vk_state.command_buffer) != VK_SUCCESS) {
        return -1;
    }

    VkFenceCreateInfo fence_ci = {0};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    if (vkCreateFence(g_vk_state.device, &fence_ci, NULL,
                       &g_vk_state.fence) != VK_SUCCESS) {
        return -1;
    }

    return 0;
}

/* ============================================================
 * Create descriptor set layout and pool
 *
 * All TQ compute shaders use the same binding layout:
 *   binding 0: input  (storage buffer, read-only)
 *   binding 1: output (storage buffer, read-write)
 *   binding 2: aux    (storage buffer, optional - query/weights)
 * Push constants carry per-dispatch parameters.
 * ============================================================ */

static int tq_vk_create_descriptors(void) {
    VkDescriptorSetLayoutBinding bindings[3] = {0};

    /* Binding 0: input buffer */
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    /* Binding 1: output buffer */
    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    /* Binding 2: auxiliary buffer (query, weights, etc.) */
    bindings[2].binding         = 2;
    bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout_ci = {0};
    layout_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_ci.bindingCount = 3;
    layout_ci.pBindings    = bindings;

    if (vkCreateDescriptorSetLayout(g_vk_state.device, &layout_ci, NULL,
                                     &g_vk_state.desc_layout) != VK_SUCCESS) {
        return -1;
    }

    /* Descriptor pool: enough for all concurrent dispatches */
    VkDescriptorPoolSize pool_size = {0};
    pool_size.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 3 * TQ_VK_PIPE_COUNT;

    VkDescriptorPoolCreateInfo pool_ci = {0};
    pool_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_ci.maxSets       = TQ_VK_PIPE_COUNT;
    pool_ci.poolSizeCount = 1;
    pool_ci.pPoolSizes    = &pool_size;

    if (vkCreateDescriptorPool(g_vk_state.device, &pool_ci, NULL,
                                &g_vk_state.desc_pool) != VK_SUCCESS) {
        return -1;
    }

    return 0;
}

/* ============================================================
 * Create pipeline layout (shared)
 *
 * Push constant range: 128 bytes (covers largest param struct)
 * ============================================================ */

static int tq_vk_create_pipeline_layout(void) {
    VkPushConstantRange push_range = {0};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset     = 0;
    push_range.size       = 128; /* generous, covers all param structs */

    VkPipelineLayoutCreateInfo layout_ci = {0};
    layout_ci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount         = 1;
    layout_ci.pSetLayouts            = &g_vk_state.desc_layout;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges    = &push_range;

    if (vkCreatePipelineLayout(g_vk_state.device, &layout_ci, NULL,
                                &g_vk_state.pipeline_layout) != VK_SUCCESS) {
        return -1;
    }

    return 0;
}

/* ============================================================
 * Create compute pipelines from SPIR-V
 * ============================================================ */

static int tq_vk_create_pipelines(void) {
    for (int i = 0; i < TQ_VK_PIPE_COUNT; i++) {
        const uint32_t* code = *g_shader_table[i].code;
        size_t code_size     = *g_shader_table[i].size;

        if (!code || code_size == 0) {
            g_vk_state.pipelines[i]      = VK_NULL_HANDLE;
            g_vk_state.shader_modules[i] = VK_NULL_HANDLE;
            continue;
        }

        VkShaderModule module = tq_vk_create_shader(code, code_size);
        if (module == VK_NULL_HANDLE) {
            fprintf(stderr, "TQ Vulkan: Failed to create shader module %d\n", i);
            return -1;
        }
        g_vk_state.shader_modules[i] = module;

        VkComputePipelineCreateInfo pipe_ci = {0};
        pipe_ci.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipe_ci.stage.sType        = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipe_ci.stage.stage        = VK_SHADER_STAGE_COMPUTE_BIT;
        pipe_ci.stage.module       = module;
        pipe_ci.stage.pName        = "main";
        pipe_ci.layout             = g_vk_state.pipeline_layout;

        if (vkCreateComputePipelines(g_vk_state.device, VK_NULL_HANDLE, 1,
                                      &pipe_ci, NULL,
                                      &g_vk_state.pipelines[i]) != VK_SUCCESS) {
            fprintf(stderr, "TQ Vulkan: Failed to create pipeline %d\n", i);
            return -1;
        }
    }

    return 0;
}

/* ============================================================
 * Public API: init / shutdown / query
 * ============================================================ */

int tq_init_vulkan_backend(void) {
    if (g_vk_state.initialized) return 0;

    if (tq_vk_create_instance() != 0)         return -1;
    if (tq_vk_select_device() != 0)            return -1;
    if (tq_vk_create_device() != 0)            return -1;
    if (tq_vk_create_command_infra() != 0)     return -1;
    if (tq_vk_create_descriptors() != 0)       return -1;
    if (tq_vk_create_pipeline_layout() != 0)   return -1;
    if (tq_vk_create_pipelines() != 0)         return -1;

    printf("TQ Vulkan: Initialized on %s (subgroup size %u)\n",
           g_vk_state.device_name, g_vk_state.subgroup_size);

    g_vk_state.initialized = 1;
    return 0;
}

void tq_shutdown_vulkan_backend(void) {
    if (!g_vk_state.initialized) return;

    vkDeviceWaitIdle(g_vk_state.device);

    /* Destroy pipelines and shader modules */
    for (int i = 0; i < TQ_VK_PIPE_COUNT; i++) {
        if (g_vk_state.pipelines[i] != VK_NULL_HANDLE)
            vkDestroyPipeline(g_vk_state.device, g_vk_state.pipelines[i], NULL);
        if (g_vk_state.shader_modules[i] != VK_NULL_HANDLE)
            vkDestroyShaderModule(g_vk_state.device,
                                   g_vk_state.shader_modules[i], NULL);
    }

    if (g_vk_state.pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(g_vk_state.device,
                                 g_vk_state.pipeline_layout, NULL);
    if (g_vk_state.desc_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(g_vk_state.device, g_vk_state.desc_pool, NULL);
    if (g_vk_state.desc_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(g_vk_state.device,
                                      g_vk_state.desc_layout, NULL);
    if (g_vk_state.fence != VK_NULL_HANDLE)
        vkDestroyFence(g_vk_state.device, g_vk_state.fence, NULL);
    if (g_vk_state.command_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(g_vk_state.device, g_vk_state.command_pool, NULL);
    if (g_vk_state.device != VK_NULL_HANDLE)
        vkDestroyDevice(g_vk_state.device, NULL);
    if (g_vk_state.instance != VK_NULL_HANDLE)
        vkDestroyInstance(g_vk_state.instance, NULL);

    memset(&g_vk_state, 0, sizeof(g_vk_state));
}

int tq_vulkan_is_available(void) {
    VkInstance test_inst = VK_NULL_HANDLE;
    VkApplicationInfo app = {0};
    app.sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ci = {0};
    ci.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;

    if (vkCreateInstance(&ci, NULL, &test_inst) != VK_SUCCESS)
        return 0;

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(test_inst, &count, NULL);
    vkDestroyInstance(test_inst, NULL);

    return (count > 0) ? 1 : 0;
}

const char* tq_vulkan_device_name(void) {
    if (!g_vk_state.initialized) return "N/A";
    return g_vk_state.device_name;
}

/* ============================================================
 * Buffer management implementation
 * ============================================================ */

int tq_vk_create_buffer(tq_vk_buffer_t* buf, VkDeviceSize size,
                          VkBufferUsageFlags usage,
                          VkMemoryPropertyFlags mem_flags) {
    if (!g_vk_state.initialized || !buf) return -1;

    memset(buf, 0, sizeof(*buf));
    buf->size = size;

    VkBufferCreateInfo buf_ci = {0};
    buf_ci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size        = size;
    buf_ci.usage       = usage;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(g_vk_state.device, &buf_ci, NULL,
                        &buf->buffer) != VK_SUCCESS) {
        return -1;
    }

    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(g_vk_state.device, buf->buffer, &mem_req);

    int mem_type = tq_vk_find_memory_type(mem_req.memoryTypeBits, mem_flags);
    if (mem_type < 0) {
        vkDestroyBuffer(g_vk_state.device, buf->buffer, NULL);
        return -1;
    }

    VkMemoryAllocateInfo alloc_info = {0};
    alloc_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize  = mem_req.size;
    alloc_info.memoryTypeIndex = (uint32_t)mem_type;

    if (vkAllocateMemory(g_vk_state.device, &alloc_info, NULL,
                          &buf->memory) != VK_SUCCESS) {
        vkDestroyBuffer(g_vk_state.device, buf->buffer, NULL);
        return -1;
    }

    vkBindBufferMemory(g_vk_state.device, buf->buffer, buf->memory, 0);

    /* Persistently map host-visible memory */
    if (mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        vkMapMemory(g_vk_state.device, buf->memory, 0, size, 0, &buf->mapped);
    }

    return 0;
}

void tq_vk_destroy_buffer(tq_vk_buffer_t* buf) {
    if (!buf) return;
    if (buf->mapped) {
        vkUnmapMemory(g_vk_state.device, buf->memory);
        buf->mapped = NULL;
    }
    if (buf->buffer != VK_NULL_HANDLE)
        vkDestroyBuffer(g_vk_state.device, buf->buffer, NULL);
    if (buf->memory != VK_NULL_HANDLE)
        vkFreeMemory(g_vk_state.device, buf->memory, NULL);
    memset(buf, 0, sizeof(*buf));
}

int tq_vk_upload(tq_vk_buffer_t* buf, const void* data, VkDeviceSize size) {
    if (!buf || !buf->mapped || !data) return -1;
    if (size > buf->size) size = buf->size;
    memcpy(buf->mapped, data, (size_t)size);

    /* Flush for non-coherent memory */
    VkMappedMemoryRange range = {0};
    range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = buf->memory;
    range.offset = 0;
    range.size   = VK_WHOLE_SIZE;
    vkFlushMappedMemoryRanges(g_vk_state.device, 1, &range);

    return 0;
}

int tq_vk_download(const tq_vk_buffer_t* buf, void* data, VkDeviceSize size) {
    if (!buf || !buf->mapped || !data) return -1;
    if (size > buf->size) size = buf->size;

    /* Invalidate for non-coherent memory */
    VkMappedMemoryRange range = {0};
    range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = buf->memory;
    range.offset = 0;
    range.size   = VK_WHOLE_SIZE;
    vkInvalidateMappedMemoryRanges(g_vk_state.device, 1, &range);

    memcpy(data, buf->mapped, (size_t)size);
    return 0;
}

#endif /* TQ_BUILD_VULKAN */
