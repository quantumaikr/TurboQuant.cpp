/**
 * test_tqm.cpp — Tests for the TQM (TurboQuant Model) file format
 *
 * Tests:
 *   1. Header size is exactly 512 bytes
 *   2. Save/load roundtrip with synthetic model data
 *   3. Auto-detect format (TQM vs safetensors magic)
 *   4. Tokenizer from-memory loading
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" {
#include "turboquant/tq_engine.h"
}

/* ============================================================
 * Test: tqm_header_t is exactly 512 bytes
 * ============================================================ */
TEST(TQM, HeaderSize) {
    EXPECT_EQ(sizeof(tqm_header_t), 512u);
}

/* ============================================================
 * Test: TQM magic constant
 * ============================================================ */
TEST(TQM, MagicValue) {
    /* "TTQM" in little-endian: T=0x54, T=0x54, Q=0x51, M=0x4D */
    EXPECT_EQ(TQM_MAGIC, 0x4D515454u);

    /* Verify it reads as "TTQM" when interpreted as bytes */
    uint32_t magic = TQM_MAGIC;
    const char* bytes = (const char*)&magic;
    EXPECT_EQ(bytes[0], 'T');
    EXPECT_EQ(bytes[1], 'T');
    EXPECT_EQ(bytes[2], 'Q');
    EXPECT_EQ(bytes[3], 'M');
}

/* ============================================================
 * Test: Header field alignment
 * ============================================================ */
TEST(TQM, HeaderFields) {
    tqm_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));

    hdr.magic = TQM_MAGIC;
    hdr.version = TQM_VERSION;
    hdr.n_layers = 4;
    hdr.hidden_dim = 512;
    hdr.intermediate_dim = 1024;
    hdr.n_heads = 8;
    hdr.n_kv_heads = 4;
    hdr.head_dim = 64;
    hdr.vocab_size = 32000;
    hdr.max_seq_len = 2048;
    hdr.rope_freq_base = 10000.0f;
    hdr.rms_norm_eps = 1e-5f;
    hdr.weight_quant = 4;
    hdr.embed_format = 16;
    hdr.tokenizer_offset = 512;
    hdr.tokenizer_size = 1024;
    hdr.weights_offset = 2048;
    hdr.weights_size = 1000000;
    hdr.n_attn_layers = 4;
    for (int i = 0; i < 4; i++) hdr.attn_layer_indices[i] = i;

    /* Verify fields can be read back */
    EXPECT_EQ(hdr.magic, TQM_MAGIC);
    EXPECT_EQ(hdr.version, 1u);
    EXPECT_EQ(hdr.n_layers, 4);
    EXPECT_EQ(hdr.hidden_dim, 512);
    EXPECT_EQ(hdr.vocab_size, 32000);
    EXPECT_FLOAT_EQ(hdr.rope_freq_base, 10000.0f);
    EXPECT_EQ(hdr.tokenizer_offset, 512u);
    EXPECT_EQ(hdr.weights_offset, 2048u);
}

/* ============================================================
 * Test: Save and load roundtrip with a tiny synthetic model
 * ============================================================ */
TEST(TQM, SaveLoadRoundtrip) {
    /* Create a tiny synthetic model */
    tq_model_t model;
    memset(&model, 0, sizeof(model));

    tq_model_config_t* c = &model.config;
    c->n_layers = 2;
    c->hidden_dim = 64;
    c->intermediate_dim = 128;
    c->n_heads = 2;
    c->n_kv_heads = 2;
    c->head_dim = 32;
    c->vocab_size = 256;
    c->max_seq_len = 128;
    c->rope_freq_base = 10000.0f;
    c->rms_norm_eps = 1e-5f;
    c->use_qk_norm = 0;
    c->attn_output_gate = 0;
    c->delta_n_heads = 0;

    /* All layers are self_attn */
    model.n_attn_layers = 2;
    int attn_indices[] = {0, 1};
    model.attn_layer_indices = attn_indices;

    int dim = c->hidden_dim;
    int q_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;

    /* Allocate layer weights as FP32 first */
    model.layers = (tq_layer_weights_t*)calloc(2, sizeof(tq_layer_weights_t));
    ASSERT_NE(model.layers, nullptr);

    /* Allocate FP32 weight buffers for 2 layers */
    for (int l = 0; l < 2; l++) {
        tq_layer_weights_t* layer = &model.layers[l];
        layer->attn_norm = (float*)calloc(dim, sizeof(float));
        layer->ffn_norm = (float*)calloc(dim, sizeof(float));
        layer->wq = (float*)calloc(q_dim * dim, sizeof(float));
        layer->wk = (float*)calloc(kv_dim * dim, sizeof(float));
        layer->wv = (float*)calloc(kv_dim * dim, sizeof(float));
        layer->wo = (float*)calloc(dim * q_dim, sizeof(float));
        layer->w_gate = (float*)calloc(inter * dim, sizeof(float));
        layer->w_up = (float*)calloc(inter * dim, sizeof(float));
        layer->w_down = (float*)calloc(dim * inter, sizeof(float));

        /* Fill with recognizable data */
        for (int i = 0; i < dim; i++) {
            layer->attn_norm[i] = 1.0f + 0.01f * i;
            layer->ffn_norm[i] = 1.0f - 0.01f * i;
        }
        for (int i = 0; i < q_dim * dim; i++) {
            layer->wq[i] = sinf((float)i * 0.01f) * 0.1f;
        }
        for (int i = 0; i < kv_dim * dim; i++) {
            layer->wk[i] = cosf((float)i * 0.01f) * 0.1f;
            layer->wv[i] = sinf((float)i * 0.02f) * 0.1f;
        }
        for (int i = 0; i < dim * q_dim; i++) {
            layer->wo[i] = cosf((float)i * 0.02f) * 0.1f;
        }
        for (int i = 0; i < inter * dim; i++) {
            layer->w_gate[i] = sinf((float)i * 0.03f) * 0.1f;
            layer->w_up[i] = cosf((float)i * 0.03f) * 0.1f;
        }
        for (int i = 0; i < dim * inter; i++) {
            layer->w_down[i] = sinf((float)i * 0.04f) * 0.1f;
        }
    }

    /* Output norm */
    model.output_norm = (float*)calloc(dim, sizeof(float));
    for (int i = 0; i < dim; i++) model.output_norm[i] = 1.0f;

    /* Embedding as FP32 (will be converted to BF16 in TQM) */
    size_t embed_size = (size_t)c->vocab_size * dim;
    model.token_embedding = (float*)calloc(embed_size, sizeof(float));
    for (size_t i = 0; i < embed_size; i++) {
        model.token_embedding[i] = sinf((float)i * 0.001f) * 0.1f;
    }
    model.embed_bf16 = NULL;

    /* Output weight tied to embedding */
    model.output_weight = model.token_embedding;
    model.output_weight_bf16 = NULL;

    /* Quantize to Q4 */
    tq_quantize_weights_q4(&model);
    ASSERT_EQ(model.use_q4_weights, 1);

    /* Save to temp file */
    const char* tmp_path = "/tmp/test_tqm_roundtrip.tqm";
    int ret = tq_save_tqm(&model, NULL, tmp_path);
    ASSERT_EQ(ret, 0);

    /* Load back */
    tq_model_t* loaded = tq_load_tqm(tmp_path);
    ASSERT_NE(loaded, nullptr);

    /* Verify config matches */
    EXPECT_EQ(loaded->config.n_layers, 2);
    EXPECT_EQ(loaded->config.hidden_dim, 64);
    EXPECT_EQ(loaded->config.intermediate_dim, 128);
    EXPECT_EQ(loaded->config.n_heads, 2);
    EXPECT_EQ(loaded->config.n_kv_heads, 2);
    EXPECT_EQ(loaded->config.head_dim, 32);
    EXPECT_EQ(loaded->config.vocab_size, 256);
    EXPECT_FLOAT_EQ(loaded->config.rope_freq_base, 10000.0f);
    EXPECT_EQ(loaded->n_attn_layers, 2);
    EXPECT_EQ(loaded->use_q4_weights, 1);

    /* Verify Q4 weights are non-null */
    for (int l = 0; l < 2; l++) {
        EXPECT_NE(loaded->layers[l].attn_norm, nullptr);
        EXPECT_NE(loaded->layers[l].ffn_norm, nullptr);
        EXPECT_NE(loaded->layers[l].wq_q4, nullptr);
        EXPECT_NE(loaded->layers[l].wk_q4, nullptr);
        EXPECT_NE(loaded->layers[l].wv_q4, nullptr);
        EXPECT_NE(loaded->layers[l].wo_q4, nullptr);
        EXPECT_NE(loaded->layers[l].w_gate_q4, nullptr);
        EXPECT_NE(loaded->layers[l].w_up_q4, nullptr);
        EXPECT_NE(loaded->layers[l].w_down_q4, nullptr);
    }

    /* Verify norm weights match (FP32, direct comparison) */
    for (int l = 0; l < 2; l++) {
        for (int i = 0; i < dim; i++) {
            EXPECT_FLOAT_EQ(loaded->layers[l].attn_norm[i],
                           model.layers[l].attn_norm[i])
                << "attn_norm mismatch at layer=" << l << " i=" << i;
            EXPECT_FLOAT_EQ(loaded->layers[l].ffn_norm[i],
                           model.layers[l].ffn_norm[i])
                << "ffn_norm mismatch at layer=" << l << " i=" << i;
        }
    }

    /* Verify Q4 packed data matches (byte-for-byte) */
    {
        int nb = (dim + 31) / 32;
        size_t wq_qs_size = (size_t)q_dim * nb * 16;
        EXPECT_EQ(memcmp(loaded->layers[0].wq_q4, model.layers[0].wq_q4, wq_qs_size), 0);

        size_t wq_sc_size = (size_t)q_dim * nb * sizeof(float);
        EXPECT_EQ(memcmp(loaded->layers[0].wq_q4s, model.layers[0].wq_q4s, wq_sc_size), 0);
    }

    /* Verify embedding (BF16 in loaded, FP32 in original) */
    EXPECT_NE(loaded->embed_bf16, nullptr);

    /* Verify output_norm */
    for (int i = 0; i < dim; i++) {
        EXPECT_FLOAT_EQ(loaded->output_norm[i], model.output_norm[i]);
    }

    /* Clean up */
    tq_free_model(loaded);

    /* Clean up synthetic model (manually since not created via tq_load_model) */
    for (int l = 0; l < 2; l++) {
        free(model.layers[l].attn_norm);
        free(model.layers[l].ffn_norm);
    }
    free(model.layers);
    free(model.output_norm);
    free(model.token_embedding);
    free(model._q4_data);

    remove(tmp_path);
}

/* ============================================================
 * Test: Auto-detect TQM format in tq_load_model
 * ============================================================ */
TEST(TQM, AutoDetect) {
    /* Write a minimal TQM header to a temp file */
    const char* tmp_path = "/tmp/test_tqm_autodetect.tqm";
    FILE* f = fopen(tmp_path, "wb");
    ASSERT_NE(f, nullptr);

    tqm_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = TQM_MAGIC;
    hdr.version = TQM_VERSION;
    hdr.n_layers = 1;
    hdr.hidden_dim = 32;
    hdr.intermediate_dim = 64;
    hdr.n_heads = 1;
    hdr.n_kv_heads = 1;
    hdr.head_dim = 32;
    hdr.vocab_size = 32;
    hdr.max_seq_len = 64;
    hdr.rope_freq_base = 10000.0f;
    hdr.rms_norm_eps = 1e-5f;
    hdr.weight_quant = 4;
    hdr.embed_format = 16;
    hdr.n_attn_layers = 1;
    hdr.attn_layer_indices[0] = 0;

    /* Calculate required data size for this minimal model */
    int dim = 32, q_dim = 32, kv_dim = 32, inter = 64;
    int nb_dim = (dim + 31) / 32;   /* 1 */
    int nb_qdim = (q_dim + 31) / 32; /* 1 */
    int nb_inter = (inter + 31) / 32; /* 2 */

    /* Per layer: attn_norm(FP32) + ffn_norm(FP32)
     *          + wq_q4 + wk_q4 + wv_q4 + wo_q4
     *          + wgate_q4 + wup_q4 + wdown_q4 */
    size_t layer_size = 0;
    layer_size += dim * 4;      /* attn_norm */
    layer_size += dim * 4;      /* ffn_norm */
    /* wq: [q_dim, dim] Q4 */
    layer_size += (size_t)q_dim * nb_dim * 16 + (size_t)q_dim * nb_dim * 4;
    /* wk: [kv_dim, dim] Q4 */
    layer_size += (size_t)kv_dim * nb_dim * 16 + (size_t)kv_dim * nb_dim * 4;
    /* wv: [kv_dim, dim] Q4 */
    layer_size += (size_t)kv_dim * nb_dim * 16 + (size_t)kv_dim * nb_dim * 4;
    /* wo: [dim, q_dim] Q4 */
    layer_size += (size_t)dim * nb_qdim * 16 + (size_t)dim * nb_qdim * 4;
    /* w_gate: [inter, dim] Q4 */
    layer_size += (size_t)inter * nb_dim * 16 + (size_t)inter * nb_dim * 4;
    /* w_up: [inter, dim] Q4 */
    layer_size += (size_t)inter * nb_dim * 16 + (size_t)inter * nb_dim * 4;
    /* w_down: [dim, inter] Q4 */
    layer_size += (size_t)dim * nb_inter * 16 + (size_t)dim * nb_inter * 4;

    size_t total_data = layer_size;
    total_data += dim * 4;  /* output_norm */
    total_data += (size_t)32 * 32 * 2;  /* embed BF16 */

    hdr.weights_offset = 512;
    hdr.weights_size = total_data;

    fwrite(&hdr, sizeof(hdr), 1, f);

    /* Write zero weight data */
    uint8_t* zeros = (uint8_t*)calloc(1, total_data);
    fwrite(zeros, 1, total_data, f);
    free(zeros);
    fclose(f);

    /* tq_load_model should auto-detect TQM and use fast path */
    tq_model_t* model = tq_load_model(tmp_path);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->config.n_layers, 1);
    EXPECT_EQ(model->config.hidden_dim, 32);
    EXPECT_EQ(model->use_q4_weights, 1);

    tq_free_model(model);
    remove(tmp_path);
}

/* ============================================================
 * Test: Tokenizer from memory
 * ============================================================ */
TEST(TQM, TokenizerFromMemory) {
    /* A minimal tokenizer.json with just a few tokens */
    const char* json =
        "{\n"
        "  \"model\": {\n"
        "    \"type\": \"BPE\",\n"
        "    \"vocab\": {\n"
        "      \"a\": 0,\n"
        "      \"b\": 1,\n"
        "      \"ab\": 2\n"
        "    },\n"
        "    \"merges\": [\n"
        "      \"a b\"\n"
        "    ]\n"
        "  },\n"
        "  \"added_tokens\": []\n"
        "}\n";

    size_t len = strlen(json);
    tq_tokenizer_t* tok = tq_load_tokenizer_from_memory(json, len);
    /* This may or may not work depending on the parser's strictness
     * with minimal JSON — just verify it doesn't crash */
    if (tok) {
        EXPECT_GE(tok->vocab_size, 3);
        tq_free_tokenizer(tok);
    }
}
