/**
 * TurboQuant -- Vulkan dispatch wrappers
 *
 * Host-callable functions that allocate Vulkan buffers, upload data,
 * record and submit compute command buffers, then download results.
 * Matches the CUDA wrapper API pattern for drop-in backend switching.
 */
#ifdef TQ_BUILD_VULKAN

#include "tq_vulkan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Block sizes matching tq_types.h / tq_cuda_common.cuh
 * ============================================================ */

#define VK_POLAR_BLOCK_SIZE    (8 + TQ_VK_BK / 2)      /* 72 bytes */
#define VK_QJL_BLOCK_SIZE      (4 + TQ_VK_SKETCH_DIM / 8 + TQ_VK_OUTLIERS) /* 40 bytes */
#define VK_TURBO_BLOCK_SIZE    (VK_POLAR_BLOCK_SIZE + VK_QJL_BLOCK_SIZE) /* 112 bytes */
#define VK_UNIFORM_4B_SIZE     (4 + TQ_VK_BK / 2)      /* 68 bytes */
#define VK_UNIFORM_2B_SIZE     (4 + TQ_VK_BK / 4)      /* 36 bytes */

/* External: get the global Vulkan state from init module */
extern tq_vk_state_t* tq_vk_get_state(void);

/* ============================================================
 * Internal helper: generic compute dispatch
 *
 * Allocates staging buffers, uploads input, records command buffer
 * with pipeline bind + descriptor update + push constants + dispatch,
 * submits, waits, downloads output.
 * ============================================================ */

typedef struct {
    tq_vk_pipeline_id  pipeline;
    const void*        push_constants;
    uint32_t           push_size;
    const void*        input_data;
    VkDeviceSize       input_size;
    const void*        aux_data;       /* optional second input (query, weights) */
    VkDeviceSize       aux_size;
    void*              output_data;
    VkDeviceSize       output_size;
    uint32_t           group_count_x;
    uint32_t           group_count_y;
    uint32_t           group_count_z;
    int                zero_output;    /* memset output to 0 before dispatch */
} tq_vk_dispatch_info_t;

static int tq_vk_dispatch(const tq_vk_dispatch_info_t* info) {
    tq_vk_state_t* vk = tq_vk_get_state();
    if (!vk || !vk->initialized) return -1;

    VkPipeline pipeline = vk->pipelines[info->pipeline];
    if (pipeline == VK_NULL_HANDLE) {
        fprintf(stderr, "TQ Vulkan: Pipeline %d not loaded\n", info->pipeline);
        return -1;
    }

    /* Create staging buffers (host-visible + storage usage) */
    VkBufferUsageFlags staging_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                     | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                     | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags host_props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                     | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    tq_vk_buffer_t buf_in  = {0};
    tq_vk_buffer_t buf_out = {0};
    tq_vk_buffer_t buf_aux = {0};
    int has_aux = (info->aux_data && info->aux_size > 0);

    if (tq_vk_create_buffer(&buf_in, info->input_size, staging_usage, host_props) != 0)
        return -1;
    if (tq_vk_create_buffer(&buf_out, info->output_size, staging_usage, host_props) != 0) {
        tq_vk_destroy_buffer(&buf_in);
        return -1;
    }
    if (has_aux) {
        if (tq_vk_create_buffer(&buf_aux, info->aux_size, staging_usage, host_props) != 0) {
            tq_vk_destroy_buffer(&buf_in);
            tq_vk_destroy_buffer(&buf_out);
            return -1;
        }
    }

    /* Upload data */
    tq_vk_upload(&buf_in, info->input_data, info->input_size);
    if (has_aux) {
        tq_vk_upload(&buf_aux, info->aux_data, info->aux_size);
    }
    if (info->zero_output && buf_out.mapped) {
        memset(buf_out.mapped, 0, (size_t)info->output_size);
    }

    /* Allocate descriptor set */
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo ds_alloc = {0};
    ds_alloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ds_alloc.descriptorPool     = vk->desc_pool;
    ds_alloc.descriptorSetCount = 1;
    ds_alloc.pSetLayouts        = &vk->desc_layout;

    if (vkAllocateDescriptorSets(vk->device, &ds_alloc, &desc_set) != VK_SUCCESS) {
        tq_vk_destroy_buffer(&buf_in);
        tq_vk_destroy_buffer(&buf_out);
        if (has_aux) tq_vk_destroy_buffer(&buf_aux);
        return -1;
    }

    /* Update descriptor set */
    VkDescriptorBufferInfo buf_infos[3] = {0};
    VkWriteDescriptorSet writes[3] = {0};
    uint32_t write_count = 2;

    buf_infos[0].buffer = buf_in.buffer;
    buf_infos[0].offset = 0;
    buf_infos[0].range  = info->input_size;

    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = desc_set;
    writes[0].dstBinding      = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo     = &buf_infos[0];

    buf_infos[1].buffer = buf_out.buffer;
    buf_infos[1].offset = 0;
    buf_infos[1].range  = info->output_size;

    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = desc_set;
    writes[1].dstBinding      = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo     = &buf_infos[1];

    if (has_aux) {
        buf_infos[2].buffer = buf_aux.buffer;
        buf_infos[2].offset = 0;
        buf_infos[2].range  = info->aux_size;

        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = desc_set;
        writes[2].dstBinding      = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo     = &buf_infos[2];
        write_count = 3;
    }

    vkUpdateDescriptorSets(vk->device, write_count, writes, 0, NULL);

    /* Record command buffer */
    vkResetCommandBuffer(vk->command_buffer, 0);

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(vk->command_buffer, &begin_info);

    vkCmdBindPipeline(vk->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                       pipeline);
    vkCmdBindDescriptorSets(vk->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                             vk->pipeline_layout, 0, 1, &desc_set, 0, NULL);

    if (info->push_constants && info->push_size > 0) {
        vkCmdPushConstants(vk->command_buffer, vk->pipeline_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0,
                            info->push_size, info->push_constants);
    }

    vkCmdDispatch(vk->command_buffer,
                   info->group_count_x, info->group_count_y, info->group_count_z);

    /* Memory barrier: compute write -> host read */
    VkMemoryBarrier barrier = {0};
    barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(vk->command_buffer,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_HOST_BIT,
                          0, 1, &barrier, 0, NULL, 0, NULL);

    vkEndCommandBuffer(vk->command_buffer);

    /* Submit and wait */
    vkResetFences(vk->device, 1, &vk->fence);

    VkSubmitInfo submit = {0};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &vk->command_buffer;

    vkQueueSubmit(vk->compute_queue, 1, &submit, vk->fence);
    vkWaitForFences(vk->device, 1, &vk->fence, VK_TRUE, UINT64_MAX);

    /* Download results */
    tq_vk_download(&buf_out, info->output_data, info->output_size);

    /* Cleanup */
    vkFreeDescriptorSets(vk->device, vk->desc_pool, 1, &desc_set);
    tq_vk_destroy_buffer(&buf_in);
    tq_vk_destroy_buffer(&buf_out);
    if (has_aux) tq_vk_destroy_buffer(&buf_aux);

    return 0;
}

/* ============================================================
 * Polar quantize / attention
 * ============================================================ */

void tq_polar_quantize_vulkan(const float* keys, void* out,
                               int n, int head_dim) {
    int num_blocks = (n + TQ_VK_BK - 1) / TQ_VK_BK;

    tq_vk_polar_params_t params = {0};
    params.n          = n;
    params.head_dim   = head_dim;
    params.num_blocks = num_blocks;

    tq_vk_dispatch_info_t info = {0};
    info.pipeline       = TQ_VK_PIPE_POLAR_QUANTIZE;
    info.push_constants = &params;
    info.push_size      = sizeof(params);
    info.input_data     = keys;
    info.input_size     = (VkDeviceSize)n * sizeof(float);
    info.output_data    = out;
    info.output_size    = (VkDeviceSize)num_blocks * VK_POLAR_BLOCK_SIZE;
    info.group_count_x  = (uint32_t)num_blocks;
    info.group_count_y  = 1;
    info.group_count_z  = 1;

    tq_vk_dispatch(&info);
}

void tq_polar_attention_vulkan(const float* query, const void* keys,
                                float* scores, int seq_len, int head_dim) {
    tq_vk_attention_params_t params = {0};
    params.seq_len    = seq_len;
    params.head_dim   = head_dim;
    params.num_blocks = seq_len;

    tq_vk_dispatch_info_t info = {0};
    info.pipeline       = TQ_VK_PIPE_POLAR_ATTENTION;
    info.push_constants = &params;
    info.push_size      = sizeof(params);
    info.input_data     = keys;
    info.input_size     = (VkDeviceSize)seq_len * VK_POLAR_BLOCK_SIZE;
    info.aux_data       = query;
    info.aux_size       = (VkDeviceSize)head_dim * sizeof(float);
    info.output_data    = scores;
    info.output_size    = (VkDeviceSize)seq_len * sizeof(float);
    info.group_count_x  = (uint32_t)seq_len;
    info.group_count_y  = 1;
    info.group_count_z  = 1;

    tq_vk_dispatch(&info);
}

/* ============================================================
 * QJL quantize / attention
 * ============================================================ */

void tq_qjl_quantize_vulkan(const float* keys, void* out,
                              int num_keys, int emb_dim) {
    tq_vk_qjl_params_t params = {0};
    params.num_keys   = num_keys;
    params.emb_dim    = emb_dim;
    params.sketch_dim = TQ_VK_SKETCH_DIM;

    /* Each workgroup handles one key; sketch dimension covered within */
    tq_vk_dispatch_info_t info = {0};
    info.pipeline       = TQ_VK_PIPE_QJL_QUANTIZE;
    info.push_constants = &params;
    info.push_size      = sizeof(params);
    info.input_data     = keys;
    info.input_size     = (VkDeviceSize)num_keys * emb_dim * sizeof(float);
    info.output_data    = out;
    info.output_size    = (VkDeviceSize)num_keys * VK_QJL_BLOCK_SIZE;
    info.group_count_x  = (uint32_t)num_keys;
    info.group_count_y  = 1;
    info.group_count_z  = 1;
    info.zero_output    = 1; /* QJL uses atomicOr on hash bytes */

    tq_vk_dispatch(&info);
}

void tq_qjl_attention_vulkan(const float* query, const void* keys,
                               float* scores, int seq_len, int head_dim) {
    tq_vk_attention_params_t params = {0};
    params.seq_len  = seq_len;
    params.head_dim = head_dim;

    tq_vk_dispatch_info_t info = {0};
    info.pipeline       = TQ_VK_PIPE_QJL_ATTENTION;
    info.push_constants = &params;
    info.push_size      = sizeof(params);
    info.input_data     = keys;
    info.input_size     = (VkDeviceSize)seq_len * VK_QJL_BLOCK_SIZE;
    info.aux_data       = query;
    info.aux_size       = (VkDeviceSize)head_dim * sizeof(float);
    info.output_data    = scores;
    info.output_size    = (VkDeviceSize)seq_len * sizeof(float);
    info.group_count_x  = (uint32_t)seq_len;
    info.group_count_y  = 1;
    info.group_count_z  = 1;

    tq_vk_dispatch(&info);
}

/* ============================================================
 * Turbo (polar + QJL residual) quantize / attention
 * ============================================================ */

void tq_turbo_quantize_vulkan(const float* keys, void* out,
                                int n, int head_dim) {
    int num_blocks = (n + TQ_VK_BK - 1) / TQ_VK_BK;

    tq_vk_turbo_params_t params = {0};
    params.n          = n;
    params.head_dim   = head_dim;
    params.num_blocks = num_blocks;

    tq_vk_dispatch_info_t info = {0};
    info.pipeline       = TQ_VK_PIPE_TURBO_QUANTIZE;
    info.push_constants = &params;
    info.push_size      = sizeof(params);
    info.input_data     = keys;
    info.input_size     = (VkDeviceSize)n * sizeof(float);
    info.output_data    = out;
    info.output_size    = (VkDeviceSize)num_blocks * VK_TURBO_BLOCK_SIZE;
    info.group_count_x  = (uint32_t)num_blocks;
    info.group_count_y  = 1;
    info.group_count_z  = 1;
    info.zero_output    = 1; /* QJL residual uses atomicOr */

    tq_vk_dispatch(&info);
}

void tq_turbo_attention_vulkan(const float* query, const void* keys,
                                 float* scores, int seq_len, int head_dim) {
    tq_vk_attention_params_t params = {0};
    params.seq_len  = seq_len;
    params.head_dim = head_dim;

    tq_vk_dispatch_info_t info = {0};
    info.pipeline       = TQ_VK_PIPE_TURBO_ATTENTION;
    info.push_constants = &params;
    info.push_size      = sizeof(params);
    info.input_data     = keys;
    info.input_size     = (VkDeviceSize)seq_len * VK_TURBO_BLOCK_SIZE;
    info.aux_data       = query;
    info.aux_size       = (VkDeviceSize)head_dim * sizeof(float);
    info.output_data    = scores;
    info.output_size    = (VkDeviceSize)seq_len * sizeof(float);
    info.group_count_x  = (uint32_t)seq_len;
    info.group_count_y  = 1;
    info.group_count_z  = 1;

    tq_vk_dispatch(&info);
}

/* ============================================================
 * Value quantize (4-bit / 2-bit)
 * ============================================================ */

void tq_value_quantize_4b_vulkan(const float* values, void* out, int n) {
    int num_blocks = (n + TQ_VK_BK - 1) / TQ_VK_BK;

    tq_vk_value_params_t params = {0};
    params.n          = n;
    params.num_blocks = num_blocks;
    params.quant_bits = 4;

    tq_vk_dispatch_info_t info = {0};
    info.pipeline       = TQ_VK_PIPE_VALUE_QUANT_4B;
    info.push_constants = &params;
    info.push_size      = sizeof(params);
    info.input_data     = values;
    info.input_size     = (VkDeviceSize)n * sizeof(float);
    info.output_data    = out;
    info.output_size    = (VkDeviceSize)num_blocks * VK_UNIFORM_4B_SIZE;
    info.group_count_x  = (uint32_t)num_blocks;
    info.group_count_y  = 1;
    info.group_count_z  = 1;

    tq_vk_dispatch(&info);
}

void tq_value_quantize_2b_vulkan(const float* values, void* out, int n) {
    int num_blocks = (n + TQ_VK_BK - 1) / TQ_VK_BK;

    tq_vk_value_params_t params = {0};
    params.n          = n;
    params.num_blocks = num_blocks;
    params.quant_bits = 2;

    tq_vk_dispatch_info_t info = {0};
    info.pipeline       = TQ_VK_PIPE_VALUE_QUANT_2B;
    info.push_constants = &params;
    info.push_size      = sizeof(params);
    info.input_data     = values;
    info.input_size     = (VkDeviceSize)n * sizeof(float);
    info.output_data    = out;
    info.output_size    = (VkDeviceSize)num_blocks * VK_UNIFORM_2B_SIZE;
    info.group_count_x  = (uint32_t)num_blocks;
    info.group_count_y  = 1;
    info.group_count_z  = 1;

    tq_vk_dispatch(&info);
}

void tq_value_dequant_matmul_4b_vulkan(const float* attn_weights,
                                         const void* values, float* output,
                                         int seq_len, int head_dim) {
    int num_dim_blocks = (head_dim + TQ_VK_BK - 1) / TQ_VK_BK;
    int num_groups = (head_dim + TQ_VK_WORKGROUP_SIZE - 1) / TQ_VK_WORKGROUP_SIZE;

    tq_vk_value_matmul_params_t params = {0};
    params.seq_len        = seq_len;
    params.head_dim       = head_dim;
    params.num_dim_blocks = num_dim_blocks;

    tq_vk_dispatch_info_t info = {0};
    info.pipeline       = TQ_VK_PIPE_VALUE_DEQUANT_MATMUL_4B;
    info.push_constants = &params;
    info.push_size      = sizeof(params);
    info.input_data     = values;
    info.input_size     = (VkDeviceSize)seq_len * num_dim_blocks * VK_UNIFORM_4B_SIZE;
    info.aux_data       = attn_weights;
    info.aux_size       = (VkDeviceSize)seq_len * sizeof(float);
    info.output_data    = output;
    info.output_size    = (VkDeviceSize)head_dim * sizeof(float);
    info.group_count_x  = (uint32_t)num_groups;
    info.group_count_y  = 1;
    info.group_count_z  = 1;

    tq_vk_dispatch(&info);
}

/* ============================================================
 * Dispatch table (matching CUDA pattern)
 * ============================================================ */

static void tq_polar_quantize_vk_wrap(const float* src, void* dst, int n) {
    tq_polar_quantize_vulkan(src, dst, n, n);
}

static void tq_polar_attention_vk_wrap(const float* query, const void* cache,
                                         float* scores, int seq_len, int hd) {
    tq_polar_attention_vulkan(query, cache, scores, seq_len, hd);
}

static void tq_qjl_quantize_vk_wrap(const float* src, void* dst, int n) {
    tq_qjl_quantize_vulkan(src, dst, 1, n);
}

static void tq_qjl_attention_vk_wrap(const float* query, const void* cache,
                                       float* scores, int seq_len, int hd) {
    tq_qjl_attention_vulkan(query, cache, scores, seq_len, hd);
}

static void tq_turbo_quantize_vk_wrap(const float* src, void* dst, int n) {
    tq_turbo_quantize_vulkan(src, dst, n, n);
}

static void tq_turbo_attention_vk_wrap(const float* query, const void* cache,
                                         float* scores, int seq_len, int hd) {
    tq_turbo_attention_vulkan(query, cache, scores, seq_len, hd);
}

static void tq_uniform_4b_quantize_vk_wrap(const float* src, void* dst, int n) {
    tq_value_quantize_4b_vulkan(src, dst, n);
}

static void tq_uniform_2b_quantize_vk_wrap(const float* src, void* dst, int n) {
    tq_value_quantize_2b_vulkan(src, dst, n);
}

typedef struct {
    void (*quantize)(const float*, void*, int);
    void (*attention)(const float*, const void*, float*, int, int);
} tq_vk_dispatch_entry_t;

static tq_vk_dispatch_entry_t g_vk_dispatch[7] = {
    /* TQ_TYPE_POLAR_3B */
    { tq_polar_quantize_vk_wrap,  tq_polar_attention_vk_wrap },
    /* TQ_TYPE_POLAR_4B */
    { tq_polar_quantize_vk_wrap,  tq_polar_attention_vk_wrap },
    /* TQ_TYPE_QJL_1B */
    { tq_qjl_quantize_vk_wrap,   tq_qjl_attention_vk_wrap },
    /* TQ_TYPE_TURBO_3B */
    { tq_turbo_quantize_vk_wrap,  tq_turbo_attention_vk_wrap },
    /* TQ_TYPE_TURBO_4B */
    { tq_turbo_quantize_vk_wrap,  tq_turbo_attention_vk_wrap },
    /* TQ_TYPE_UNIFORM_4B */
    { tq_uniform_4b_quantize_vk_wrap, NULL },
    /* TQ_TYPE_UNIFORM_2B */
    { tq_uniform_2b_quantize_vk_wrap, NULL },
};

void* tq_vulkan_get_quantize_fn(int type_id) {
    if (type_id < 0 || type_id >= 7) return NULL;
    return (void*)g_vk_dispatch[type_id].quantize;
}

void* tq_vulkan_get_attention_fn(int type_id) {
    if (type_id < 0 || type_id >= 7) return NULL;
    return (void*)g_vk_dispatch[type_id].attention;
}

#endif /* TQ_BUILD_VULKAN */
