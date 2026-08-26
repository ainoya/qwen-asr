/*
 * qwen_asr_decoder.c - Qwen3 LLM decoder
 *
 * Architecture (per layer):
 *   RMSNorm -> QKV (no bias) -> per-head Q/K RMSNorm -> NeoX RoPE
 *   -> Causal GQA attention -> Output proj -> residual
 *   RMSNorm -> SwiGLU MLP (gate/up/down, no bias) -> residual
 *
 * Features: Q/K per-head RMSNorm, NeoX split-half RoPE, GQA 2:1,
 * tied embeddings (tok_embeddings == lm_head).
 */

#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

static float *load_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "decoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

/* Attach to pre-quantized "<name>.q8" / "<name>.q8s" if the model file carries
 * them. Returns 1 when attached, 0 when the file only has the bf16 original. */
static int attach_q8(multi_safetensors_t *ms, const char *name, qwen_q8_mat_t *m) {
    char qn[512];
    safetensors_file_t *sfq = NULL, *sfs = NULL;

    snprintf(qn, sizeof(qn), "%s.q8", name);
    const safetensor_t *tq = multi_safetensors_find(ms, qn, &sfq);
    if (!tq || tq->ndim != 2) return 0;

    snprintf(qn, sizeof(qn), "%s.q8s", name);
    const safetensor_t *ts = multi_safetensors_find(ms, qn, &sfs);
    if (!ts) return 0;

    qwen_q8_attach(m, (int8_t *)safetensors_data(sfq, tq),
                   (float *)safetensors_data(sfs, ts),
                   (int)tq->shape[0], (int)tq->shape[1]);
    return 1;
}

static uint16_t *load_bf16_direct(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "decoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_bf16_direct(sf, t);
}

/* Narrow one already-quantized matrix to 4 bits. Requantizing from Q8 rather
 * than from the original bf16 costs almost nothing at this width: the block
 * scale is already correct and only the 16-level grid matters. Doing it this
 * way means the packed-image and bf16 load paths share one conversion. */
static int narrow_to_q4_scaled(qwen_q8_mat_t *m, const float *colscale) {
    qwen_q8_mat_t q4;
    int rc = colscale ? qwen_q4_from_q8_scaled(&q4, m, colscale)
                      : qwen_q4_from_q8(&q4, m);
    if (rc != 0) return -1;
    if (m->owns) qwen_q8_free(m);   /* only free a copy we made ourselves */
    *m = q4;
    return 0;
}

/* Narrow one layer's six matrices to four bits, optionally rescaling channels
 * (AWQ) on the way.
 *
 * Rescaling needs the matching input channel divided by the same factor. Every
 * such division folds into something already there, so inference is untouched:
 * q/k/v are fed by input_layernorm and gate/up by post_attention_layernorm, so
 * the reciprocal goes into that norm's weight; down's input is the SwiGLU
 * product, so it goes into the `up` rows of the fused gate/up matrix (rows
 * interleave as gate0, up0, gate1, up1, ...), which is exact because only a
 * block scale changes.
 *
 * O is deliberately left unscaled. Under grouped-query attention two of its
 * input channels share one V row, so a per-channel division has nowhere exact
 * to fold - and O measured the smallest gain of the four groups anyway (4.6%,
 * against 19% for down and 16% for q/k/v). */
static int narrow_layer_to_q4(qwen_dec_layer_t *l, int layer, qwen_awq_t *awq,
                              int hidden, int inter) {
    char key[64];
    const float *s;

    snprintf(key, sizeof(key), "L%02d.q", layer);
    s = awq ? qwen_awq_scales(awq, key, hidden) : NULL;
    qwen_q8_mat_t *attn[] = { &l->wq_q8, &l->wk_q8, &l->wv_q8 };
    for (size_t k = 0; k < sizeof(attn) / sizeof(attn[0]); k++) {
        if (narrow_to_q4_scaled(attn[k], s) != 0) return -1;
    }
    if (s && l->input_norm)
        for (int c = 0; c < hidden; c++) l->input_norm[c] /= s[c];

    snprintf(key, sizeof(key), "L%02d.gate_up", layer);
    s = awq ? qwen_awq_scales(awq, key, hidden) : NULL;
    if (narrow_to_q4_scaled(&l->gate_up_q8, s) != 0) return -1;
    if (s && l->post_attn_norm)
        for (int c = 0; c < hidden; c++) l->post_attn_norm[c] /= s[c];

    snprintf(key, sizeof(key), "L%02d.down", layer);
    s = awq ? qwen_awq_scales(awq, key, inter) : NULL;
    if (narrow_to_q4_scaled(&l->down_q8, s) != 0) return -1;
    if (s) qwen_q8_scale_rows(&l->gate_up_q8, s, 1, 2, inter);

    if (narrow_to_q4_scaled(&l->wo_q8, NULL) != 0) return -1;
    return 0;
}

int qwen_decoder_load(qwen_decoder_t *dec, multi_safetensors_t *ms,
                       const qwen_config_t *cfg) {
    char name[512];

    int mode = qwen_weight_quant;
    const char *wenv = getenv("QWEN_WEIGHTS");
    if (wenv && wenv[0]) {
        if (strcmp(wenv, "bf16") == 0) mode = QWEN_WEIGHTS_BF16;
        else if (strcmp(wenv, "q8") == 0) mode = QWEN_WEIGHTS_Q8;
        else if (strcmp(wenv, "q8-lm") == 0) mode = QWEN_WEIGHTS_Q8_LM;
        else if (strcmp(wenv, "q4") == 0) mode = QWEN_WEIGHTS_Q4;
    }
    int use_q8 = (mode != QWEN_WEIGHTS_BF16);
    /* Q8 blocks require the contraction dimension to be a multiple of
     * QWEN_Q8_BLOCK. Both shipped models satisfy this; fall back rather than
     * failing the load if some other checkpoint does not. */
    if (use_q8) {
        int dims[] = { cfg->dec_hidden,
                       cfg->dec_heads * cfg->dec_head_dim,
                       cfg->dec_intermediate };
        for (size_t d = 0; d < sizeof(dims) / sizeof(dims[0]); d++) {
            if (dims[d] % QWEN_Q8_BLOCK != 0) {
                fprintf(stderr,
                        "decoder: dim %d is not a multiple of %d, "
                        "falling back to bf16 weights\n",
                        dims[d], QWEN_Q8_BLOCK);
                use_q8 = 0;
                break;
            }
        }
    }

    int use_q8_embed = use_q8 && (mode == QWEN_WEIGHTS_Q8_LM || mode == QWEN_WEIGHTS_Q4);
    int use_q4_layers = (mode == QWEN_WEIGHTS_Q4);

    dec->quantized = use_q8;
    dec->embed_quantized = use_q8_embed;
    size_t q8_bytes = 0;

    /* Token embeddings: either pre-quantized in the file, or bf16 to quantize. */
    int embed_prepacked = attach_q8(ms, "thinker.model.embed_tokens.weight",
                                    &dec->tok_embeddings_q8);
    if (embed_prepacked) {
        use_q8 = 1;
        use_q8_embed = 1;
        q8_bytes += qwen_q8_bytes(&dec->tok_embeddings_q8);
    } else {
        dec->tok_embeddings_bf16 = load_bf16_direct(ms,
            "thinker.model.embed_tokens.weight");
        if (!dec->tok_embeddings_bf16) return -1;
    }
    dec->quantized = use_q8;
    dec->embed_quantized = use_q8_embed;

    if (use_q8_embed && !embed_prepacked) {
        if (qwen_q8_from_bf16(&dec->tok_embeddings_q8, dec->tok_embeddings_bf16,
                              cfg->vocab_size, cfg->dec_hidden) != 0)
            return -1;
        q8_bytes += qwen_q8_bytes(&dec->tok_embeddings_q8);
    }

    /* Channel rescaling is only a 4-bit concern: it trades resolution between
     * channels sharing a block, and eight bits has enough to go round. */
    qwen_awq_t *awq = NULL;
    const char *awq_env = getenv("QWEN_AWQ");
    const char *awq_file = (awq_env && awq_env[0]) ? awq_env : qwen_awq_path;
    if (use_q4_layers && awq_file) {
        double alpha = qwen_awq_alpha;
        const char *ae = getenv("QWEN_AWQ_ALPHA");
        if (ae && ae[0]) alpha = atof(ae);
        awq = qwen_awq_open(awq_file, alpha);
        if (!awq) {
            fprintf(stderr, "decoder: cannot use AWQ statistics from %s\n", awq_file);
            return -1;
        }
        if (qwen_verbose > 0)
            fprintf(stderr, "AWQ channel rescaling from %s, alpha %.2f\n",
                    awq_file, alpha);
    }

    /* Transformer layers */
    for (int i = 0; i < cfg->dec_layers; i++) {
        qwen_dec_layer_t *l = &dec->layers[i];
        const char *lp = "thinker.model.layers";

        /* Per-head Q/K RMSNorm weights */
        snprintf(name, sizeof(name), "%s.%d.self_attn.q_norm.weight", lp, i);
        l->q_norm_weight = load_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn.k_norm.weight", lp, i);
        l->k_norm_weight = load_f32(ms, name);

        /* RMSNorm weights */
        snprintf(name, sizeof(name), "%s.%d.input_layernorm.weight", lp, i);
        l->input_norm = load_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.post_attention_layernorm.weight", lp, i);
        l->post_attn_norm = load_f32(ms, name);

        if (embed_prepacked) {
            /* Pre-quantized image: point straight at the mapped bytes. */
            struct { const char *suffix; qwen_q8_mat_t *dst; } q8[] = {
                { "self_attn.q_proj.weight", &l->wq_q8 },
                { "self_attn.k_proj.weight", &l->wk_q8 },
                { "self_attn.v_proj.weight", &l->wv_q8 },
                { "self_attn.o_proj.weight", &l->wo_q8 },
                { "mlp.gate_up.weight",      &l->gate_up_q8 },
                { "mlp.down_proj.weight",    &l->down_q8 },
            };
            for (size_t k = 0; k < sizeof(q8) / sizeof(q8[0]); k++) {
                snprintf(name, sizeof(name), "%s.%d.%s", lp, i, q8[k].suffix);
                if (!attach_q8(ms, name, q8[k].dst)) {
                    fprintf(stderr, "decoder: packed model missing %s.q8\n", name);
                    return -1;
                }
            }
            if (use_q4_layers &&
                narrow_layer_to_q4(l, i, awq, cfg->dec_hidden,
                                   cfg->dec_intermediate) != 0) {
                fprintf(stderr, "decoder: 4-bit conversion failed at layer %d\n", i);
                return -1;
            }
            for (size_t k = 0; k < sizeof(q8) / sizeof(q8[0]); k++)
                q8_bytes += qwen_q8_bytes(q8[k].dst);
            continue;
        }

        /* Attention weights (bf16, no bias) */
        snprintf(name, sizeof(name), "%s.%d.self_attn.q_proj.weight", lp, i);
        l->wq_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn.k_proj.weight", lp, i);
        l->wk_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn.v_proj.weight", lp, i);
        l->wv_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn.o_proj.weight", lp, i);
        l->wo_weight_bf16 = load_bf16_direct(ms, name);

        /* SwiGLU MLP weights (bf16, no bias) */
        snprintf(name, sizeof(name), "%s.%d.mlp.gate_proj.weight", lp, i);
        l->gate_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "%s.%d.mlp.up_proj.weight", lp, i);
        l->up_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "%s.%d.mlp.down_proj.weight", lp, i);
        l->down_weight_bf16 = load_bf16_direct(ms, name);

        if (!l->wq_weight_bf16 || !l->wk_weight_bf16 ||
            !l->wv_weight_bf16 || !l->wo_weight_bf16 ||
            !l->gate_weight_bf16 || !l->up_weight_bf16 || !l->down_weight_bf16) {
            fprintf(stderr, "decoder: failed to load layer %d\n", i);
            return -1;
        }

        /* Fuse gate+up: rows interleaved [gate0, up0, gate1, up1, ...] so a
         * single matvec produces the SwiGLU operand pairs contiguously. */
        {
            int inter = cfg->dec_intermediate;
            int hidden = cfg->dec_hidden;
            int q_dim = cfg->dec_heads * cfg->dec_head_dim;
            int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;

            if (use_q8) {
                if (qwen_q8_from_bf16(&l->wq_q8, l->wq_weight_bf16, q_dim, hidden) != 0 ||
                    qwen_q8_from_bf16(&l->wk_q8, l->wk_weight_bf16, kv_dim, hidden) != 0 ||
                    qwen_q8_from_bf16(&l->wv_q8, l->wv_weight_bf16, kv_dim, hidden) != 0 ||
                    qwen_q8_from_bf16(&l->wo_q8, l->wo_weight_bf16, hidden, q_dim) != 0 ||
                    qwen_q8_from_bf16_interleave2(&l->gate_up_q8, l->gate_weight_bf16,
                                                  l->up_weight_bf16, inter, hidden) != 0 ||
                    qwen_q8_from_bf16(&l->down_q8, l->down_weight_bf16, hidden, inter) != 0) {
                    fprintf(stderr, "decoder: quantization failed at layer %d\n", i);
                    return -1;
                }
                if (use_q4_layers &&
                    narrow_layer_to_q4(l, i, awq, hidden, inter) != 0) {
                    fprintf(stderr, "decoder: 4-bit conversion failed at layer %d\n", i);
                    return -1;
                }
                q8_bytes += qwen_q8_bytes(&l->wq_q8) + qwen_q8_bytes(&l->wk_q8) +
                            qwen_q8_bytes(&l->wv_q8) + qwen_q8_bytes(&l->wo_q8) +
                            qwen_q8_bytes(&l->gate_up_q8) + qwen_q8_bytes(&l->down_q8);
            } else {
                size_t row_bytes = (size_t)hidden * sizeof(uint16_t);
                l->gate_up_fused_bf16 = (uint16_t *)malloc(2 * (size_t)inter * row_bytes);
                for (int r = 0; r < inter; r++) {
                    memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r) * hidden,
                           l->gate_weight_bf16 + (size_t)r * hidden, row_bytes);
                    memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r + 1) * hidden,
                           l->up_weight_bf16 + (size_t)r * hidden, row_bytes);
                }
            }
        }
    }

    qwen_awq_close(awq);

    /* Final RMSNorm */
    dec->norm = load_f32(ms, "thinker.model.norm.weight");
    if (!dec->norm) return -1;

    if (use_q8 && qwen_verbose >= 1)
        fprintf(stderr, "Decoder weights: Q8%s, %.2f GB quantized (%.2f GB as bf16)\n",
                use_q8_embed ? " incl. LM head" : " (LM head kept bf16)",
                (double)q8_bytes / 1e9,
                (double)q8_bytes / 1e9 * (2.0 * QWEN_Q8_BLOCK) /
                    (QWEN_Q8_BLOCK + sizeof(float)));

    return 0;
}

/* ========================================================================
 * KV Cache Management
 * ======================================================================== */

static int kv_cache_init(qwen_ctx_t *ctx, int max_seq) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    size_t cache_size = (size_t)ctx->config.dec_layers * max_seq * kv_dim * sizeof(float);
    ctx->kv_cache_k = (float *)calloc(1, cache_size);
    ctx->kv_cache_v = (float *)calloc(1, cache_size);
    ctx->kv_cache_len = 0;
    ctx->kv_cache_max = max_seq;
    if (!ctx->kv_cache_k || !ctx->kv_cache_v) return -1;
    return 0;
}

static int kv_cache_grow(qwen_ctx_t *ctx, int required) {
    if (required <= ctx->kv_cache_max) return 0;

    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    int new_max = ctx->kv_cache_max;
    while (new_max < required) new_max *= 2;

    size_t new_stride = (size_t)new_max * kv_dim;
    size_t old_stride = (size_t)ctx->kv_cache_max * kv_dim;
    size_t total = (size_t)ctx->config.dec_layers * new_stride * sizeof(float);

    float *new_k = (float *)calloc(1, total);
    float *new_v = (float *)calloc(1, total);
    if (!new_k || !new_v) { free(new_k); free(new_v); return -1; }

    size_t copy = (size_t)ctx->kv_cache_len * kv_dim * sizeof(float);
    for (int l = 0; l < ctx->config.dec_layers; l++) {
        memcpy(new_k + l * new_stride, ctx->kv_cache_k + l * old_stride, copy);
        memcpy(new_v + l * new_stride, ctx->kv_cache_v + l * old_stride, copy);
    }

    free(ctx->kv_cache_k);
    free(ctx->kv_cache_v);
    ctx->kv_cache_k = new_k;
    ctx->kv_cache_v = new_v;
    ctx->kv_cache_max = new_max;
    return 0;
}

static float *kv_cache_k_at(qwen_ctx_t *ctx, int layer, int pos) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    return ctx->kv_cache_k + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

static float *kv_cache_v_at(qwen_ctx_t *ctx, int layer, int pos) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    return ctx->kv_cache_v + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

static int ensure_prefill_buffers(qwen_ctx_t *ctx, int seq_len) {
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int q_dim = cfg->dec_heads * cfg->dec_head_dim;
    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;

    if (seq_len <= ctx->pref_seq_cap) return 0;

    int new_cap = ctx->pref_seq_cap > 0 ? ctx->pref_seq_cap : 64;
    while (new_cap < seq_len) new_cap *= 2;

#define REALLOC_PREF(ptr, count) do {                                          \
    void *tmp__ = realloc((ptr), (size_t)(count) * sizeof(float));             \
    if (!tmp__) return -1;                                                      \
    (ptr) = (float *)tmp__;                                                     \
} while (0)

    REALLOC_PREF(ctx->pref_x, new_cap * dim);
    REALLOC_PREF(ctx->pref_x_norm, new_cap * dim);
    REALLOC_PREF(ctx->pref_q, new_cap * q_dim);
    REALLOC_PREF(ctx->pref_k, new_cap * kv_dim);
    REALLOC_PREF(ctx->pref_v, new_cap * kv_dim);
    REALLOC_PREF(ctx->pref_attn_out, new_cap * q_dim);
    REALLOC_PREF(ctx->pref_proj_out, new_cap * dim);
    REALLOC_PREF(ctx->pref_ffn_out, new_cap * dim);
    REALLOC_PREF(ctx->pref_gate, new_cap * intermediate);
    REALLOC_PREF(ctx->pref_gate_up, new_cap * 2 * intermediate);

#undef REALLOC_PREF

    ctx->pref_seq_cap = new_cap;
    return 0;
}

static int ensure_rope_inv_freq(qwen_ctx_t *ctx, int head_dim, float theta) {
    int half = head_dim / 2;
    if (ctx->rope_inv_freq && ctx->rope_inv_freq_half == half) return 0;

    float *inv = (float *)realloc(ctx->rope_inv_freq, (size_t)half * sizeof(float));
    if (!inv) return -1;
    ctx->rope_inv_freq = inv;

    for (int d = 0; d < half; d++) {
        ctx->rope_inv_freq[d] = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
    }
    ctx->rope_inv_freq_half = half;
    return 0;
}

static int ensure_rope_cache(qwen_ctx_t *ctx, int required_pos, int head_dim, float theta) {
    if (required_pos <= ctx->rope_cache_cap) return 0;
    if (ensure_rope_inv_freq(ctx, head_dim, theta) != 0) return -1;

    int new_cap = ctx->rope_cache_cap > 0 ? ctx->rope_cache_cap : 1024;
    while (new_cap < required_pos) new_cap *= 2;

    size_t n = (size_t)new_cap * head_dim;
    float *new_cos = (float *)realloc(ctx->rope_cache_cos, n * sizeof(float));
    if (!new_cos) return -1;
    ctx->rope_cache_cos = new_cos;

    float *new_sin = (float *)realloc(ctx->rope_cache_sin, n * sizeof(float));
    if (!new_sin) return -1;
    ctx->rope_cache_sin = new_sin;

    int half = head_dim / 2;
    for (int pos = ctx->rope_cache_cap; pos < new_cap; pos++) {
        float p = (float)pos;
        float *cos_row = ctx->rope_cache_cos + (size_t)pos * head_dim;
        float *sin_row = ctx->rope_cache_sin + (size_t)pos * head_dim;
        for (int d = 0; d < half; d++) {
            float angle = p * ctx->rope_inv_freq[d];
            float c = cosf(angle);
            float s = sinf(angle);
            cos_row[d] = c;
            cos_row[half + d] = c;
            sin_row[d] = s;
            sin_row[half + d] = s;
        }
    }

    ctx->rope_cache_cap = new_cap;
    return 0;
}

/* ========================================================================
 * Decoder Prefill (Multiple Tokens)
 * ======================================================================== */

void qwen_decoder_prefill(qwen_ctx_t *ctx, const float *input_embeds, int seq_len) {
    qwen_decoder_t *dec = &ctx->decoder;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    float eps = cfg->dec_rms_norm_eps;
    float theta = cfg->dec_rope_theta;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    /* Ensure KV cache */
    if (!ctx->kv_cache_k) {
        if (kv_cache_init(ctx, seq_len + 1024) != 0) return;
    } else if (ctx->kv_cache_len + seq_len > ctx->kv_cache_max) {
        if (kv_cache_grow(ctx, ctx->kv_cache_len + seq_len + 1024) != 0) return;
    }

    if (ensure_prefill_buffers(ctx, seq_len) != 0) return;

    float *x = ctx->pref_x;
    float *x_norm = ctx->pref_x_norm;
    float *q = ctx->pref_q;
    float *k = ctx->pref_k;
    float *v = ctx->pref_v;
    float *attn_out = ctx->pref_attn_out;
    float *proj_out = ctx->pref_proj_out;
    float *ffn_out = ctx->pref_ffn_out;
    float *gate = ctx->pref_gate;
    float *gate_up = ctx->pref_gate_up;

    memcpy(x, input_embeds, (size_t)seq_len * dim * sizeof(float));

    int start_pos = ctx->kv_cache_len;
    if (ensure_rope_cache(ctx, start_pos + seq_len, head_dim, theta) != 0) return;
    const float *rope_cos = ctx->rope_cache_cos + (size_t)start_pos * head_dim;
    const float *rope_sin = ctx->rope_cache_sin + (size_t)start_pos * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < cfg->dec_layers; layer++) {
        qwen_dec_layer_t *l = &dec->layers[layer];

        /* Input RMSNorm */
        qwen_rms_norm(x_norm, x, l->input_norm, seq_len, dim, eps);

        /* QKV projections (no bias) */
        if (dec->quantized) {
            qwen_linear_nobias_q8(q, x_norm, &l->wq_q8, seq_len);
            qwen_linear_nobias_q8(k, x_norm, &l->wk_q8, seq_len);
            qwen_linear_nobias_q8(v, x_norm, &l->wv_q8, seq_len);
        } else {
            qwen_linear_nobias_bf16(q, x_norm, l->wq_weight_bf16, seq_len, dim, q_dim);
            qwen_linear_nobias_bf16(k, x_norm, l->wk_weight_bf16, seq_len, dim, kv_dim);
            qwen_linear_nobias_bf16(v, x_norm, l->wv_weight_bf16, seq_len, dim, kv_dim);
        }

        /* Per-head Q/K RMSNorm */
        qwen_rms_norm_per_head(q, l->q_norm_weight, seq_len, n_heads, head_dim, eps);
        qwen_rms_norm_per_head(k, l->k_norm_weight, seq_len, n_kv_heads, head_dim, eps);

        /* Apply NeoX RoPE */
        qwen_apply_rope_neox(q, rope_cos, rope_sin, seq_len, n_heads, head_dim);
        qwen_apply_rope_neox(k, rope_cos, rope_sin, seq_len, n_kv_heads, head_dim);

        /* Store K, V in cache */
        for (int s = 0; s < seq_len; s++) {
            memcpy(kv_cache_k_at(ctx, layer, start_pos + s),
                   k + s * kv_dim, kv_dim * sizeof(float));
            memcpy(kv_cache_v_at(ctx, layer, start_pos + s),
                   v + s * kv_dim, kv_dim * sizeof(float));
        }

        /* Causal attention */
        int total_seq = start_pos + seq_len;
        float *full_k = kv_cache_k_at(ctx, layer, 0);
        float *full_v = kv_cache_v_at(ctx, layer, 0);
        qwen_causal_attention(attn_out, q, full_k, full_v,
                               seq_len, total_seq, n_heads, n_kv_heads,
                               head_dim, scale, start_pos);

        /* Output projection + residual */
        if (dec->quantized)
            qwen_linear_nobias_q8(proj_out, attn_out, &l->wo_q8, seq_len);
        else
            qwen_linear_nobias_bf16(proj_out, attn_out, l->wo_weight_bf16,
                                     seq_len, q_dim, dim);
        qwen_add_inplace(x, proj_out, seq_len * dim);

        /* Post-attention RMSNorm */
        qwen_rms_norm(x_norm, x, l->post_attn_norm, seq_len, dim, eps);

        /* SwiGLU MLP */
        if (dec->quantized) {
            qwen_linear_nobias_q8(gate_up, x_norm, &l->gate_up_q8, seq_len);
            qwen_swiglu_multiply(gate, gate_up, seq_len, intermediate);
            qwen_linear_nobias_q8(ffn_out, gate, &l->down_q8, seq_len);
        } else {
            qwen_linear_nobias_bf16(gate_up, x_norm, l->gate_up_fused_bf16,
                                     seq_len, dim, 2 * intermediate);
            qwen_swiglu_multiply(gate, gate_up, seq_len, intermediate);
            qwen_linear_nobias_bf16(ffn_out, gate, l->down_weight_bf16,
                                     seq_len, intermediate, dim);
        }

        qwen_add_inplace(x, ffn_out, seq_len * dim);

    }

    ctx->kv_cache_len = start_pos + seq_len;
}

/* ========================================================================
 * Decoder Forward (Single Token Generation)
 * ======================================================================== */

static void ensure_dec_buffers(qwen_ctx_t *ctx) {
    if (ctx->dec_x) return;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int q_dim = cfg->dec_heads * cfg->dec_head_dim;
    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    int head_dim = cfg->dec_head_dim;

    ctx->dec_x        = (float *)malloc(dim * sizeof(float));
    ctx->dec_x_norm   = (float *)malloc(dim * sizeof(float));
    ctx->dec_q        = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_k        = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_v        = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_attn_out = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_proj_out = (float *)malloc(dim * sizeof(float));
    ctx->dec_gate     = (float *)malloc(2 * intermediate * sizeof(float));
    ctx->dec_up       = NULL; /* unused: gate buffer holds fused gate+up */
    ctx->dec_ffn_out  = (float *)malloc(dim * sizeof(float));
    ctx->dec_rope_cos = (float *)malloc(head_dim * sizeof(float));
    ctx->dec_rope_sin = (float *)malloc(head_dim * sizeof(float));
}

int qwen_decoder_forward(qwen_ctx_t *ctx, const float *input_embed) {
    qwen_decoder_t *dec = &ctx->decoder;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    float eps = cfg->dec_rms_norm_eps;
    float theta = cfg->dec_rope_theta;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    ensure_dec_buffers(ctx);
    float *x = ctx->dec_x;
    float *x_norm = ctx->dec_x_norm;
    float *q = ctx->dec_q;
    float *k = ctx->dec_k;
    float *v = ctx->dec_v;
    float *attn_out = ctx->dec_attn_out;
    float *proj_out = ctx->dec_proj_out;
    float *gate_buf = ctx->dec_gate;
    float *ffn_out = ctx->dec_ffn_out;
    memcpy(x, input_embed, dim * sizeof(float));

    int pos = ctx->kv_cache_len;

    /* Grow KV cache if needed */
    if (pos >= ctx->kv_cache_max) {
        if (kv_cache_grow(ctx, pos + 1024) != 0) return QWEN_TOKEN_IM_END;
    }

    if (ensure_rope_cache(ctx, pos + 1, head_dim, theta) != 0) {
        return QWEN_TOKEN_IM_END;
    }
    const float *rope_cos = ctx->rope_cache_cos + (size_t)pos * head_dim;
    const float *rope_sin = ctx->rope_cache_sin + (size_t)pos * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < cfg->dec_layers; layer++) {
        qwen_dec_layer_t *l = &dec->layers[layer];

        qwen_rms_norm(x_norm, x, l->input_norm, 1, dim, eps);
        if (dec->quantized)
            qwen_linear_nobias_q8_qkv(q, k, v, x_norm,
                                      &l->wq_q8, &l->wk_q8, &l->wv_q8);
        else
            qwen_linear_nobias_bf16_qkv(q, k, v, x_norm,
                                        l->wq_weight_bf16,
                                        l->wk_weight_bf16,
                                        l->wv_weight_bf16,
                                        dim, q_dim, kv_dim);

        /* Per-head Q/K RMSNorm */
        qwen_rms_norm_per_head(q, l->q_norm_weight, 1, n_heads, head_dim, eps);
        qwen_rms_norm_per_head(k, l->k_norm_weight, 1, n_kv_heads, head_dim, eps);

        /* Apply NeoX RoPE */
        qwen_apply_rope_neox(q, rope_cos, rope_sin, 1, n_heads, head_dim);
        qwen_apply_rope_neox(k, rope_cos, rope_sin, 1, n_kv_heads, head_dim);

        memcpy(kv_cache_k_at(ctx, layer, pos), k, kv_dim * sizeof(float));
        memcpy(kv_cache_v_at(ctx, layer, pos), v, kv_dim * sizeof(float));

        int total_seq = pos + 1;
        float *full_k = kv_cache_k_at(ctx, layer, 0);
        float *full_v = kv_cache_v_at(ctx, layer, 0);

        qwen_causal_attention(attn_out, q, full_k, full_v,
                               1, total_seq, n_heads, n_kv_heads,
                               head_dim, scale, pos);

        if (dec->quantized)
            qwen_linear_nobias_q8(proj_out, attn_out, &l->wo_q8, 1);
        else
            qwen_linear_nobias_bf16(proj_out, attn_out, l->wo_weight_bf16, 1, q_dim, dim);
        qwen_add_inplace(x, proj_out, dim);

        qwen_rms_norm(x_norm, x, l->post_attn_norm, 1, dim, eps);

        /* Fused gate+up matvec: one pass over x_norm, output interleaved [g0,u0,g1,u1,...] */
        if (dec->quantized)
            qwen_linear_nobias_q8(gate_buf, x_norm, &l->gate_up_q8, 1);
        else
            qwen_linear_nobias_bf16(gate_buf, x_norm, l->gate_up_fused_bf16,
                                     1, dim, 2 * intermediate);
        /* In-place for seq=1: gate_buf[0:inter] receives SwiGLU output. */
        qwen_swiglu_multiply(gate_buf, gate_buf, 1, intermediate);
        if (dec->quantized)
            qwen_linear_nobias_q8(ffn_out, gate_buf, &l->down_q8, 1);
        else
            qwen_linear_nobias_bf16(ffn_out, gate_buf, l->down_weight_bf16, 1, intermediate, dim);
        qwen_add_inplace(x, ffn_out, dim);
    }

    ctx->kv_cache_len = pos + 1;

    /* Final norm + streaming argmax (no logits buffer needed) */
    qwen_rms_norm(x, x, dec->norm, 1, dim, eps);
    if (dec->embed_quantized)
        return qwen_argmax_matvec_q8(x, &dec->tok_embeddings_q8);
    return qwen_argmax_matvec_bf16(x, dec->tok_embeddings_bf16, dim, cfg->vocab_size);
}

/* ========================================================================
 * Batched Decoding
 *
 * One sweep of the decoder weights advances several independent streams.
 * The per-token cost is dominated by reading 1.83 GB of weights, so the
 * second and later streams in a batch are close to free; see qwen_asr.h.
 * ======================================================================== */

qwen_kv_t *qwen_kv_create(qwen_ctx_t *ctx, int max_seq) {
    qwen_kv_t *kv = (qwen_kv_t *)calloc(1, sizeof(*kv));
    if (!kv) return NULL;
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    size_t bytes = (size_t)ctx->config.dec_layers * max_seq * kv_dim * sizeof(float);
    kv->k = (float *)calloc(1, bytes);
    kv->v = (float *)calloc(1, bytes);
    kv->max = max_seq;
    kv->len = 0;
    if (!kv->k || !kv->v) { qwen_kv_free(kv); return NULL; }
    return kv;
}

void qwen_kv_free(qwen_kv_t *kv) {
    if (!kv) return;
    free(kv->k);
    free(kv->v);
    free(kv);
}

/* The context keeps exactly one cache at a time. Bind/unbind swap a stream's
 * cache in so the single-stream prefill can be reused verbatim; prefill may
 * grow it, hence the write-back in unbind(). */
void qwen_kv_bind(qwen_ctx_t *ctx, qwen_kv_t *kv) {
    ctx->kv_cache_k = kv->k;
    ctx->kv_cache_v = kv->v;
    ctx->kv_cache_len = kv->len;
    ctx->kv_cache_max = kv->max;
}

void qwen_kv_unbind(qwen_ctx_t *ctx, qwen_kv_t *kv) {
    kv->k = ctx->kv_cache_k;
    kv->v = ctx->kv_cache_v;
    kv->len = ctx->kv_cache_len;
    kv->max = ctx->kv_cache_max;
    ctx->kv_cache_k = NULL;
    ctx->kv_cache_v = NULL;
    ctx->kv_cache_len = 0;
    ctx->kv_cache_max = 0;
}

static int kv_grow(qwen_ctx_t *ctx, qwen_kv_t *kv, int required) {
    if (required <= kv->max) return 0;
    qwen_kv_bind(ctx, kv);
    int rc = kv_cache_grow(ctx, required);
    qwen_kv_unbind(ctx, kv);
    return rc;
}

int qwen_decoder_forward_batch(qwen_ctx_t *ctx, qwen_kv_t **kvs, int n,
                               const float *embeds, int *out_tokens) {
    if (n <= 0) return 0;
    if (n > QWEN_MAX_BATCH) return -1;

    qwen_decoder_t *dec = &ctx->decoder;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    float eps = cfg->dec_rms_norm_eps;
    float theta = cfg->dec_rope_theta;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    int max_pos = 0;
    for (int s = 0; s < n; s++) {
        if (kv_grow(ctx, kvs[s], kvs[s]->len + 1) != 0) return -1;
        if (kvs[s]->len > max_pos) max_pos = kvs[s]->len;
    }
    if (ensure_prefill_buffers(ctx, n) != 0) return -1;
    if (ensure_rope_cache(ctx, max_pos + 1, head_dim, theta) != 0) return -1;

    float *x = ctx->pref_x;
    float *x_norm = ctx->pref_x_norm;
    float *q = ctx->pref_q;
    float *k = ctx->pref_k;
    float *v = ctx->pref_v;
    float *attn_out = ctx->pref_attn_out;
    float *proj_out = ctx->pref_proj_out;
    float *ffn_out = ctx->pref_ffn_out;
    float *gate = ctx->pref_gate;
    float *gate_up = ctx->pref_gate_up;

    memcpy(x, embeds, (size_t)n * dim * sizeof(float));

    for (int layer = 0; layer < cfg->dec_layers; layer++) {
        qwen_dec_layer_t *l = &dec->layers[layer];

        qwen_rms_norm(x_norm, x, l->input_norm, n, dim, eps);
        if (dec->quantized) {
            qwen_linear_nobias_q8(q, x_norm, &l->wq_q8, n);
            qwen_linear_nobias_q8(k, x_norm, &l->wk_q8, n);
            qwen_linear_nobias_q8(v, x_norm, &l->wv_q8, n);
        } else {
            qwen_linear_nobias_bf16(q, x_norm, l->wq_weight_bf16, n, dim, q_dim);
            qwen_linear_nobias_bf16(k, x_norm, l->wk_weight_bf16, n, dim, kv_dim);
            qwen_linear_nobias_bf16(v, x_norm, l->wv_weight_bf16, n, dim, kv_dim);
        }

        qwen_rms_norm_per_head(q, l->q_norm_weight, n, n_heads, head_dim, eps);
        qwen_rms_norm_per_head(k, l->k_norm_weight, n, n_kv_heads, head_dim, eps);

        /* Each stream sits at its own position, so RoPE and attention are
         * applied per row rather than over the batch. */
        for (int s = 0; s < n; s++) {
            int pos = kvs[s]->len;
            const float *rc = ctx->rope_cache_cos + (size_t)pos * head_dim;
            const float *rs = ctx->rope_cache_sin + (size_t)pos * head_dim;
            qwen_apply_rope_neox(q + (size_t)s * q_dim, rc, rs, 1, n_heads, head_dim);
            qwen_apply_rope_neox(k + (size_t)s * kv_dim, rc, rs, 1, n_kv_heads, head_dim);

            float *dst_k = kvs[s]->k + ((size_t)layer * kvs[s]->max + pos) * kv_dim;
            float *dst_v = kvs[s]->v + ((size_t)layer * kvs[s]->max + pos) * kv_dim;
            memcpy(dst_k, k + (size_t)s * kv_dim, kv_dim * sizeof(float));
            memcpy(dst_v, v + (size_t)s * kv_dim, kv_dim * sizeof(float));

            float *full_k = kvs[s]->k + (size_t)layer * kvs[s]->max * kv_dim;
            float *full_v = kvs[s]->v + (size_t)layer * kvs[s]->max * kv_dim;
            qwen_causal_attention(attn_out + (size_t)s * q_dim, q + (size_t)s * q_dim,
                                  full_k, full_v, 1, pos + 1, n_heads, n_kv_heads,
                                  head_dim, scale, pos);
        }

        if (dec->quantized)
            qwen_linear_nobias_q8(proj_out, attn_out, &l->wo_q8, n);
        else
            qwen_linear_nobias_bf16(proj_out, attn_out, l->wo_weight_bf16, n, q_dim, dim);
        qwen_add_inplace(x, proj_out, n * dim);

        qwen_rms_norm(x_norm, x, l->post_attn_norm, n, dim, eps);

        if (dec->quantized) {
            qwen_linear_nobias_q8(gate_up, x_norm, &l->gate_up_q8, n);
            qwen_swiglu_multiply(gate, gate_up, n, intermediate);
            qwen_linear_nobias_q8(ffn_out, gate, &l->down_q8, n);
        } else {
            qwen_linear_nobias_bf16(gate_up, x_norm, l->gate_up_fused_bf16,
                                    n, dim, 2 * intermediate);
            qwen_swiglu_multiply(gate, gate_up, n, intermediate);
            qwen_linear_nobias_bf16(ffn_out, gate, l->down_weight_bf16,
                                    n, intermediate, dim);
        }
        qwen_add_inplace(x, ffn_out, n * dim);
    }

    for (int s = 0; s < n; s++) kvs[s]->len++;

    qwen_rms_norm(x, x, dec->norm, n, dim, eps);
    if (dec->embed_quantized)
        return qwen_argmax_matvec_q8_batch(x, n, &dec->tok_embeddings_q8, out_tokens);

    for (int s = 0; s < n; s++)
        out_tokens[s] = qwen_argmax_matvec_bf16(x + (size_t)s * dim,
                                                dec->tok_embeddings_bf16,
                                                dim, cfg->vocab_size);
    return 0;
}

/* Prefill several fresh streams in one pass.
 *
 * Prefill measures as ~590 ms of fixed cost plus ~3.6 ms per row on M1 Pro:
 * the fixed part is the sweep that dequantizes every Q8 weight into f32
 * panels for sgemm, and it is paid once per call no matter how few rows
 * follow. Concatenating a batch's segments into one call pays it once
 * instead of n times. Rows keep their own sequence's attention, so this is
 * not a longer context - just a taller matrix.
 *
 * Every stream must be empty (len == 0); embeds[i] is [lens[i]][dec_hidden]. */
int qwen_decoder_prefill_multi(qwen_ctx_t *ctx, qwen_kv_t **kvs,
                               const float *const *embeds, const int *lens, int n) {
    if (n <= 0) return 0;
    if (n == 1) {
        qwen_kv_bind(ctx, kvs[0]);
        qwen_decoder_prefill(ctx, embeds[0], lens[0]);
        qwen_kv_unbind(ctx, kvs[0]);
        return kvs[0]->len == lens[0] ? 0 : -1;
    }

    qwen_decoder_t *dec = &ctx->decoder;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    float eps = cfg->dec_rms_norm_eps;
    float theta = cfg->dec_rope_theta;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    int rows = 0, longest = 0;
    for (int i = 0; i < n; i++) {
        if (kvs[i]->len != 0) return -1;
        if (lens[i] <= 0) return -1;
        if (kv_grow(ctx, kvs[i], lens[i]) != 0) return -1;
        rows += lens[i];
        if (lens[i] > longest) longest = lens[i];
    }
    if (ensure_prefill_buffers(ctx, rows) != 0) return -1;
    if (ensure_rope_cache(ctx, longest, head_dim, theta) != 0) return -1;

    float *x = ctx->pref_x;
    float *x_norm = ctx->pref_x_norm;
    float *q = ctx->pref_q;
    float *k = ctx->pref_k;
    float *v = ctx->pref_v;
    float *attn_out = ctx->pref_attn_out;
    float *proj_out = ctx->pref_proj_out;
    float *ffn_out = ctx->pref_ffn_out;
    float *gate = ctx->pref_gate;
    float *gate_up = ctx->pref_gate_up;

    for (int i = 0, off = 0; i < n; off += lens[i], i++)
        memcpy(x + (size_t)off * dim, embeds[i], (size_t)lens[i] * dim * sizeof(float));

    const float *rope_cos = ctx->rope_cache_cos;
    const float *rope_sin = ctx->rope_cache_sin;

    for (int layer = 0; layer < cfg->dec_layers; layer++) {
        qwen_dec_layer_t *l = &dec->layers[layer];

        qwen_rms_norm(x_norm, x, l->input_norm, rows, dim, eps);
        if (dec->quantized) {
            qwen_linear_nobias_q8(q, x_norm, &l->wq_q8, rows);
            qwen_linear_nobias_q8(k, x_norm, &l->wk_q8, rows);
            qwen_linear_nobias_q8(v, x_norm, &l->wv_q8, rows);
        } else {
            qwen_linear_nobias_bf16(q, x_norm, l->wq_weight_bf16, rows, dim, q_dim);
            qwen_linear_nobias_bf16(k, x_norm, l->wk_weight_bf16, rows, dim, kv_dim);
            qwen_linear_nobias_bf16(v, x_norm, l->wv_weight_bf16, rows, dim, kv_dim);
        }

        qwen_rms_norm_per_head(q, l->q_norm_weight, rows, n_heads, head_dim, eps);
        qwen_rms_norm_per_head(k, l->k_norm_weight, rows, n_kv_heads, head_dim, eps);

        /* Each stream restarts at position 0 and attends only to itself. */
        for (int i = 0, off = 0; i < n; off += lens[i], i++) {
            int len = lens[i];
            float *qi = q + (size_t)off * q_dim;
            float *ki = k + (size_t)off * kv_dim;
            float *vi = v + (size_t)off * kv_dim;

            qwen_apply_rope_neox(qi, rope_cos, rope_sin, len, n_heads, head_dim);
            qwen_apply_rope_neox(ki, rope_cos, rope_sin, len, n_kv_heads, head_dim);

            float *ck = kvs[i]->k + (size_t)layer * kvs[i]->max * kv_dim;
            float *cv = kvs[i]->v + (size_t)layer * kvs[i]->max * kv_dim;
            memcpy(ck, ki, (size_t)len * kv_dim * sizeof(float));
            memcpy(cv, vi, (size_t)len * kv_dim * sizeof(float));

            qwen_causal_attention(attn_out + (size_t)off * q_dim, qi, ck, cv,
                                  len, len, n_heads, n_kv_heads, head_dim, scale, 0);
        }

        if (dec->quantized)
            qwen_linear_nobias_q8(proj_out, attn_out, &l->wo_q8, rows);
        else
            qwen_linear_nobias_bf16(proj_out, attn_out, l->wo_weight_bf16, rows, q_dim, dim);
        qwen_add_inplace(x, proj_out, rows * dim);

        qwen_rms_norm(x_norm, x, l->post_attn_norm, rows, dim, eps);

        if (dec->quantized) {
            qwen_linear_nobias_q8(gate_up, x_norm, &l->gate_up_q8, rows);
            qwen_swiglu_multiply(gate, gate_up, rows, intermediate);
            qwen_linear_nobias_q8(ffn_out, gate, &l->down_q8, rows);
        } else {
            qwen_linear_nobias_bf16(gate_up, x_norm, l->gate_up_fused_bf16,
                                    rows, dim, 2 * intermediate);
            qwen_swiglu_multiply(gate, gate_up, rows, intermediate);
            qwen_linear_nobias_bf16(ffn_out, gate, l->down_weight_bf16,
                                    rows, intermediate, dim);
        }
        qwen_add_inplace(x, ffn_out, rows * dim);
    }

    for (int i = 0; i < n; i++) kvs[i]->len = lens[i];
    return 0;
}
