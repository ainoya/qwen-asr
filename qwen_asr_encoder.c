/*
 * qwen_asr_encoder.c - Audio encoder forward pass
 *
 * Architecture:
 *   Per-chunk Conv2D stem: 3 layers of Conv2D(3x3, stride=2, pad=1) -> GELU
 *     128 mel bins -> 64 -> 32 -> 16 frequency, time/8
 *     Reshape [480, 16, T/8] -> [T/8, 7680], project to d_model
 *   Per-chunk sinusoidal position embeddings
 *   Transformer encoder layers (bidirectional windowed attention):
 *     LayerNorm -> MHA (Q,K,V all have biases) -> residual
 *     LayerNorm -> GELU FFN (fc1,fc2 with biases) -> residual
 *   Final LayerNorm
 *   Projection: proj1 (GELU) -> proj2
 */

#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

#define ENC_PREFIX "thinker.audio_tower."

static float *load_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "encoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

/* Load bf16 weight and convert to f32 at load time.
 * Encoder always processes batches, so pre-converting avoids
 * repeated scratch-buffer conversion during forward pass. */
static float *load_bf16_as_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "encoder: weight not found: %s\n", name);
        return NULL;
    }
    uint16_t *bf16 = safetensors_get_bf16_direct(sf, t);
    if (!bf16) return NULL;

    /* Compute number of elements from tensor shape */
    size_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];

    float *f32 = (float *)malloc(n * sizeof(float));
    if (!f32) return NULL;

    uint32_t *d = (uint32_t *)(void *)f32;
    for (size_t i = 0; i < n; i++)
        d[i] = ((uint32_t)bf16[i]) << 16;

    return f32;
}

/* Load a weight matrix in whichever form the model file carries it: either
 * pre-quantized ("<name>.q8" + "<name>.q8s", used in place) or the original
 * bf16, converted to f32. */
static int load_wmat(multi_safetensors_t *ms, const char *name, qwen_wmat_t *out) {
    char qn[512];
    safetensors_file_t *sf = NULL;

    snprintf(qn, sizeof(qn), "%s.q8", name);
    const safetensor_t *tq = multi_safetensors_find(ms, qn, &sf);
    if (tq && tq->ndim == 2) {
        safetensors_file_t *sfs = NULL;
        snprintf(qn, sizeof(qn), "%s.q8s", name);
        const safetensor_t *ts = multi_safetensors_find(ms, qn, &sfs);
        if (ts) {
            out->rows = (int)tq->shape[0];
            out->cols = (int)tq->shape[1];
            qwen_q8_attach(&out->q8,
                           (int8_t *)safetensors_data(sf, tq),
                           (float *)safetensors_data(sfs, ts),
                           out->rows, out->cols);
            return 0;
        }
    }

    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "encoder: weight not found: %s\n", name);
        return -1;
    }
    out->rows = t->ndim >= 1 ? (int)t->shape[0] : 0;
    out->cols = t->ndim >= 2 ? (int)t->shape[1] : 0;
    out->f32 = load_bf16_as_f32(ms, name);
    return out->f32 ? 0 : -1;
}

int qwen_encoder_load(qwen_encoder_t *enc, multi_safetensors_t *ms,
                       const qwen_config_t *cfg) {
    char name[512];

    /* Conv2D stem (small, f32) */
    snprintf(name, sizeof(name), "%sconv2d1.weight", ENC_PREFIX);
    if (qwen_gpu_resident && !multi_safetensors_find(ms, name, NULL)) {
        /* Reduced image: the whole tower lives on the GPU and arrives through
         * the encoder hook; nothing here to load, nothing here to run. */
        enc->weights_absent = 1;
        return 0;
    }
    enc->conv1_weight = load_f32(ms, name);
    snprintf(name, sizeof(name), "%sconv2d1.bias", ENC_PREFIX);
    enc->conv1_bias = load_f32(ms, name);
    snprintf(name, sizeof(name), "%sconv2d2.weight", ENC_PREFIX);
    enc->conv2_weight = load_f32(ms, name);
    snprintf(name, sizeof(name), "%sconv2d2.bias", ENC_PREFIX);
    enc->conv2_bias = load_f32(ms, name);
    snprintf(name, sizeof(name), "%sconv2d3.weight", ENC_PREFIX);
    enc->conv3_weight = load_f32(ms, name);
    snprintf(name, sizeof(name), "%sconv2d3.bias", ENC_PREFIX);
    enc->conv3_bias = load_f32(ms, name);

    if (!enc->conv1_weight || !enc->conv2_weight || !enc->conv3_weight) return -1;

    /* Conv output projection (bf16, no bias) */
    snprintf(name, sizeof(name), "%sconv_out.weight", ENC_PREFIX);
    if (load_wmat(ms, name, &enc->conv_out_weight) != 0) return -1;

    /* Transformer layers */
    for (int i = 0; i < cfg->enc_layers; i++) {
        qwen_enc_layer_t *l = &enc->layers[i];
        const char *lp = ENC_PREFIX "layers";

        /* Attention weights (bf16) and biases (f32) */
        snprintf(name, sizeof(name), "%s.%d.self_attn.q_proj.weight", lp, i);
        if (load_wmat(ms, name, &l->wq_weight) != 0) return -1;
        snprintf(name, sizeof(name), "%s.%d.self_attn.q_proj.bias", lp, i);
        l->wq_bias = load_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn.k_proj.weight", lp, i);
        if (load_wmat(ms, name, &l->wk_weight) != 0) return -1;
        snprintf(name, sizeof(name), "%s.%d.self_attn.k_proj.bias", lp, i);
        l->wk_bias = load_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn.v_proj.weight", lp, i);
        if (load_wmat(ms, name, &l->wv_weight) != 0) return -1;
        snprintf(name, sizeof(name), "%s.%d.self_attn.v_proj.bias", lp, i);
        l->wv_bias = load_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn.out_proj.weight", lp, i);
        if (load_wmat(ms, name, &l->wo_weight) != 0) return -1;
        snprintf(name, sizeof(name), "%s.%d.self_attn.out_proj.bias", lp, i);
        l->wo_bias = load_f32(ms, name);

        /* Pre-attention LayerNorm */
        snprintf(name, sizeof(name), "%s.%d.self_attn_layer_norm.weight", lp, i);
        l->attn_norm_weight = load_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.self_attn_layer_norm.bias", lp, i);
        l->attn_norm_bias = load_f32(ms, name);

        /* FFN weights (bf16) and biases (f32) */
        snprintf(name, sizeof(name), "%s.%d.fc1.weight", lp, i);
        if (load_wmat(ms, name, &l->fc1_weight) != 0) return -1;
        snprintf(name, sizeof(name), "%s.%d.fc1.bias", lp, i);
        l->fc1_bias = load_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.fc2.weight", lp, i);
        if (load_wmat(ms, name, &l->fc2_weight) != 0) return -1;
        snprintf(name, sizeof(name), "%s.%d.fc2.bias", lp, i);
        l->fc2_bias = load_f32(ms, name);

        /* Pre-FFN LayerNorm */
        snprintf(name, sizeof(name), "%s.%d.final_layer_norm.weight", lp, i);
        l->ffn_norm_weight = load_f32(ms, name);
        snprintf(name, sizeof(name), "%s.%d.final_layer_norm.bias", lp, i);
        l->ffn_norm_bias = load_f32(ms, name);

    }

    /* Final LayerNorm */
    snprintf(name, sizeof(name), "%sln_post.weight", ENC_PREFIX);
    enc->ln_post_weight = load_f32(ms, name);
    snprintf(name, sizeof(name), "%sln_post.bias", ENC_PREFIX);
    enc->ln_post_bias = load_f32(ms, name);

    /* Projection layers */
    snprintf(name, sizeof(name), "%sproj1.weight", ENC_PREFIX);
    if (load_wmat(ms, name, &enc->proj1_weight) != 0) return -1;
    snprintf(name, sizeof(name), "%sproj1.bias", ENC_PREFIX);
    enc->proj1_bias = load_f32(ms, name);
    snprintf(name, sizeof(name), "%sproj2.weight", ENC_PREFIX);
    if (load_wmat(ms, name, &enc->proj2_weight) != 0) return -1;
    snprintf(name, sizeof(name), "%sproj2.bias", ENC_PREFIX);
    enc->proj2_bias = load_f32(ms, name);

    if (!enc->ln_post_weight) return -1;

    return 0;
}

/* ========================================================================
 * Forward Pass
 * ======================================================================== */

/* Coarse phase timing, so the conv stem and the transformer stack can be
 * weighed against each other when deciding what to move to a GPU backend. */
double qwen_enc_conv_ms = 0;
double qwen_enc_layers_ms = 0;

static double enc_now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

int qwen_enc_tap = 0;
float *qwen_enc_tap_mel = NULL;
int qwen_enc_tap_frames = 0;
float *qwen_enc_tap_conv = NULL;
float *qwen_enc_tap_out = NULL;
int qwen_enc_tap_tokens = 0;

/* Replace *dst with a fresh copy of n floats. */
static void enc_tap_store(float **dst, const float *src, size_t n) {
    free(*dst);
    *dst = (float *)malloc(n * sizeof(float));
    if (*dst) memcpy(*dst, src, n * sizeof(float));
}

float *qwen_encoder_forward(qwen_ctx_t *ctx, const float *mel, int mel_frames,
                             int *out_seq_len) {
    double enc_t0 = enc_now_ms();

    /* An external tower, if one is installed. Returning NULL means it could
     * not run, so fall through and encode here. */
    if (ctx->encoder_hook) {
        int seq = 0;
        float *out = ctx->encoder_hook(ctx->encoder_hook_userdata, mel, mel_frames, &seq);
        if (out && seq > 0) {
            qwen_enc_conv_ms = 0;
            qwen_enc_layers_ms = enc_now_ms() - enc_t0;
            *out_seq_len = seq;
            return out;
        }
        free(out);
    }

    if (ctx->encoder.weights_absent) {
        static int warned;
        if (!warned++) fprintf(stderr,
            "encoder: tower weights are GPU-resident and the hook failed; "
            "no CPU fallback is possible\n");
        return NULL;
    }

    const qwen_config_t *cfg = &ctx->config;
    qwen_encoder_t *enc = &ctx->encoder;

    int d_model = cfg->enc_d_model;
    int n_heads = cfg->enc_heads;
    int head_dim = cfg->enc_head_dim;
    int ffn_dim = cfg->enc_ffn_dim;
    int output_dim = cfg->enc_output_dim;
    int chunk_size = cfg->enc_chunk_size;          /* 100 */
    int n_window_infer = cfg->enc_n_window_infer;  /* 800 */


    /* ---- Per-chunk Conv2D stem ---- */
    /* mel: [128, mel_frames] (already in Conv2D-friendly layout)
     * Process chunks of chunk_size frames, each producing tokens_per_chunk tokens */
    int n_chunks = (mel_frames + chunk_size - 1) / chunk_size;
    int tokens_per_chunk = 0; /* computed from first chunk */

    /* First: determine output tokens per chunk from a full chunk */
    {
        int w = chunk_size;
        int w1 = (w + 2 * 1 - 3) / 2 + 1;
        int w2 = (w1 + 2 * 1 - 3) / 2 + 1;
        int w3 = (w2 + 2 * 1 - 3) / 2 + 1;
        tokens_per_chunk = w3; /* 13 for chunk_size=100 */
    }

    /* Collect all chunks' output tokens */
    int total_tokens = 0;

    /* Pre-calculate total tokens */
    for (int c = 0; c < n_chunks; c++) {
        int start = c * chunk_size;
        int end = start + chunk_size;
        if (end > mel_frames) end = mel_frames;
        int chunk_w = end - start;
        int w1 = (chunk_w + 2 - 3) / 2 + 1;
        int w2 = (w1 + 2 - 3) / 2 + 1;
        int w3 = (w2 + 2 - 3) / 2 + 1;
        total_tokens += w3;
    }


    /* Allocate main sequence buffer: [total_tokens, d_model] */
    float *x = (float *)calloc((size_t)total_tokens * d_model, sizeof(float));
    int token_offset = 0;

    /* Process chunks through Conv2D + reshape + project + sinusoidal PE.
     *
     * Chunks are convolved in groups: alone, a 100-frame chunk gives the
     * second conv layer a 480x4320x800 GEMM, too narrow to keep the machine
     * busy, and every layer costs a separate dispatch. The group size is
     * capped by the im2col scratch, which is the widest buffer here. */
    size_t conv_col_budget = 64u << 20; /* bytes; 128 MB buys ~2% more */
    const char *budget_env = getenv("QWEN_CONV_BUDGET_MB");
    if (budget_env && atoi(budget_env) > 0)
        conv_col_budget = (size_t)atoi(budget_env) << 20;

    /* The second conv layer holds the widest im2col: 480*3*3 rows by one
     * chunk's [h2, w2] output positions. */
    int fh1 = (128 + 2 - 3) / 2 + 1, fh2 = (fh1 + 2 - 3) / 2 + 1;
    int fw1 = (chunk_size + 2 - 3) / 2 + 1, fw2 = (fw1 + 2 - 3) / 2 + 1;
    size_t cols_per_chunk = (size_t)QWEN_CONV_HIDDEN * 9 * fh2 * fw2 * sizeof(float);
    int group = (int)(conv_col_budget / (cols_per_chunk ? cols_per_chunk : 1));
    if (group < 1) group = 1;
    if (group > n_chunks) group = n_chunks;

    float *pe = (float *)malloc((size_t)tokens_per_chunk * d_model * sizeof(float));
    qwen_sinusoidal_pe(pe, tokens_per_chunk, d_model);

    for (int c0 = 0; c0 < n_chunks; ) {
        /* Only equal-sized chunks can share a batch; the tail runs alone. */
        int g = 0;
        while (c0 + g < n_chunks && g < group &&
               (c0 + g) * chunk_size + chunk_size <= mel_frames) g++;
        if (g == 0) g = 1;

        int start = c0 * chunk_size;
        int end = start + g * chunk_size;
        if (end > mel_frames) end = mel_frames;
        int chunk_w = (end - start) / g;

        int w1 = (chunk_w + 2 - 3) / 2 + 1;
        int w2 = (w1 + 2 - 3) / 2 + 1;
        int w3 = (w2 + 2 - 3) / 2 + 1;
        int h1 = (128 + 2 - 3) / 2 + 1; /* 64 */
        int h2 = (h1 + 2 - 3) / 2 + 1; /* 32 */
        int h3 = (h2 + 2 - 3) / 2 + 1; /* 16 */

        /* Batched mel: [1][g][128][chunk_w] */
        float *chunk_mel = (float *)malloc((size_t)g * 128 * chunk_w * sizeof(float));
        for (int b = 0; b < g; b++) {
            float *dst = chunk_mel + (size_t)b * 128 * chunk_w;
            const float *src = mel + start + (size_t)b * chunk_w;
            for (int m = 0; m < 128; m++)
                memcpy(dst + (size_t)m * chunk_w, src + (size_t)m * mel_frames,
                       (size_t)chunk_w * sizeof(float));
        }

        /* Conv2D layer 1: [1, g, 128, chunk_w] -> [480, g, 64, w1] */
        float *c1 = (float *)malloc((size_t)QWEN_CONV_HIDDEN * g * h1 * w1 * sizeof(float));
        qwen_conv2d_batch(c1, chunk_mel, enc->conv1_weight, enc->conv1_bias,
                          g, 1, QWEN_CONV_HIDDEN, 128, chunk_w, 3, 3, 2, 1);
        qwen_gelu(c1, QWEN_CONV_HIDDEN * g * h1 * w1);
        free(chunk_mel);

        /* Conv2D layer 2: [480, g, 64, w1] -> [480, g, 32, w2] */
        float *c2 = (float *)malloc((size_t)QWEN_CONV_HIDDEN * g * h2 * w2 * sizeof(float));
        qwen_conv2d_batch(c2, c1, enc->conv2_weight, enc->conv2_bias,
                          g, QWEN_CONV_HIDDEN, QWEN_CONV_HIDDEN, h1, w1, 3, 3, 2, 1);
        qwen_gelu(c2, QWEN_CONV_HIDDEN * g * h2 * w2);
        free(c1);

        /* Conv2D layer 3: [480, g, 32, w2] -> [480, g, 16, w3] */
        float *c3 = (float *)malloc((size_t)QWEN_CONV_HIDDEN * g * h3 * w3 * sizeof(float));
        qwen_conv2d_batch(c3, c2, enc->conv3_weight, enc->conv3_bias,
                          g, QWEN_CONV_HIDDEN, QWEN_CONV_HIDDEN, h2, w2, 3, 3, 2, 1);
        qwen_gelu(c3, QWEN_CONV_HIDDEN * g * h3 * w3);
        free(c2);

        /* Reshape [480, g, 16, w3] -> [g*w3, 480*16=7680] */
        int conv_proj_dim = QWEN_CONV_HIDDEN * h3; /* 480 * 16 = 7680 */
        float *reshaped = (float *)malloc((size_t)g * w3 * conv_proj_dim * sizeof(float));
        for (int b = 0; b < g; b++) {
            for (int t = 0; t < w3; t++) {
                float *dst = reshaped + ((size_t)b * w3 + t) * conv_proj_dim;
                for (int ch = 0; ch < QWEN_CONV_HIDDEN; ch++) {
                    const float *plane = c3 + ((size_t)ch * g + b) * h3 * w3;
                    for (int f = 0; f < h3; f++)
                        dst[ch * h3 + f] = plane[(size_t)f * w3 + t];
                }
            }
        }
        free(c3);

        /* Project: [g*w3, 7680] -> [g*w3, d_model] (no bias) */
        float *projected = x + (size_t)token_offset * d_model;
        qwen_linear_w(projected, reshaped, &enc->conv_out_weight, NULL, g * w3);
        free(reshaped);

        /* Position embeddings restart at 0 in every chunk. */
        if (w3 == tokens_per_chunk) {
            for (int b = 0; b < g; b++)
                qwen_add_inplace(projected + (size_t)b * w3 * d_model, pe, w3 * d_model);
        } else {
            float *pe_tail = (float *)malloc((size_t)w3 * d_model * sizeof(float));
            qwen_sinusoidal_pe(pe_tail, w3, d_model);
            for (int b = 0; b < g; b++)
                qwen_add_inplace(projected + (size_t)b * w3 * d_model, pe_tail, w3 * d_model);
            free(pe_tail);
        }

        token_offset += g * w3;
        c0 += g;
    }
    free(pe);

    if (qwen_enc_tap) {
        enc_tap_store(&qwen_enc_tap_mel, mel, (size_t)128 * mel_frames);
        qwen_enc_tap_frames = qwen_enc_tap_mel ? mel_frames : 0;
        enc_tap_store(&qwen_enc_tap_conv, x, (size_t)total_tokens * d_model);
        qwen_enc_tap_tokens = qwen_enc_tap_conv ? total_tokens : 0;
    }

    qwen_enc_conv_ms = enc_now_ms() - enc_t0;
    enc_t0 = enc_now_ms();

    /* ---- Build attention window boundaries ---- */
    /* Window size = tokens_per_chunk * (n_window_infer / chunk_size) */
    int window_token_size = tokens_per_chunk * (n_window_infer / chunk_size);
    int n_windows = (total_tokens + window_token_size - 1) / window_token_size;
    int *window_starts = (int *)malloc((n_windows + 1) * sizeof(int));
    for (int w = 0; w < n_windows; w++) {
        window_starts[w] = w * window_token_size;
    }
    window_starts[n_windows] = total_tokens;


    /* ---- Transformer layers ---- */
    float *x_norm = (float *)malloc(total_tokens * d_model * sizeof(float));
    float *q = (float *)malloc(total_tokens * d_model * sizeof(float));
    float *k = (float *)malloc(total_tokens * d_model * sizeof(float));
    float *v = (float *)malloc(total_tokens * d_model * sizeof(float));
    float *attn_out = (float *)malloc(total_tokens * d_model * sizeof(float));
    float *proj_out = (float *)malloc(total_tokens * d_model * sizeof(float));
    float *ffn_mid = (float *)malloc(total_tokens * ffn_dim * sizeof(float));
    float *ffn_out = (float *)malloc(total_tokens * d_model * sizeof(float));

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < cfg->enc_layers; layer++) {
        qwen_enc_layer_t *l = &enc->layers[layer];

        /* ---- Self-attention ---- */
        qwen_layer_norm(x_norm, x, l->attn_norm_weight, l->attn_norm_bias,
                        total_tokens, d_model, 1e-5f);

        qwen_linear_w(q, x_norm, &l->wq_weight, l->wq_bias, total_tokens);
        qwen_linear_w(k, x_norm, &l->wk_weight, l->wk_bias, total_tokens);
        qwen_linear_w(v, x_norm, &l->wv_weight, l->wv_bias, total_tokens);

        qwen_bidirectional_attention(attn_out, q, k, v,
                                      total_tokens, n_heads, head_dim, scale,
                                      window_starts, n_windows);

        /* Output projection + residual */
        qwen_linear_w(proj_out, attn_out, &l->wo_weight, l->wo_bias, total_tokens);
        qwen_add_inplace(x, proj_out, total_tokens * d_model);

        /* ---- FFN ---- */
        qwen_layer_norm(x_norm, x, l->ffn_norm_weight, l->ffn_norm_bias,
                        total_tokens, d_model, 1e-5f);

        /* GELU FFN: fc1 -> GELU -> fc2 */
        qwen_linear_w(ffn_mid, x_norm, &l->fc1_weight, l->fc1_bias, total_tokens);
        qwen_gelu(ffn_mid, total_tokens * ffn_dim);
        qwen_linear_w(ffn_out, ffn_mid, &l->fc2_weight, l->fc2_bias, total_tokens);
        qwen_add_inplace(x, ffn_out, total_tokens * d_model);

    }

    /* Final LayerNorm */
    qwen_layer_norm(x, x, enc->ln_post_weight, enc->ln_post_bias,
                    total_tokens, d_model, 1e-5f);

    /* Projection: proj1 (GELU) -> proj2 */
    float *proj_mid = (float *)malloc(total_tokens * d_model * sizeof(float));
    qwen_linear_w(proj_mid, x, &enc->proj1_weight, enc->proj1_bias, total_tokens);
    qwen_gelu(proj_mid, total_tokens * d_model);

    float *enc_output = (float *)malloc(total_tokens * output_dim * sizeof(float));
    qwen_linear_w(enc_output, proj_mid, &enc->proj2_weight, enc->proj2_bias, total_tokens);
    free(proj_mid);

    /* Clean up */
    free(x); free(x_norm); free(q); free(k); free(v);
    free(attn_out); free(proj_out);
    free(ffn_mid); free(ffn_out);
    free(window_starts);

    if (qwen_enc_tap && qwen_enc_tap_tokens)
        enc_tap_store(&qwen_enc_tap_out, enc_output, (size_t)total_tokens * output_dim);

    qwen_enc_layers_ms = enc_now_ms() - enc_t0;
    if (qwen_verbose >= 2)
        fprintf(stderr, "  Encoder: conv stem %.0f ms, transformer %.0f ms\n",
                qwen_enc_conv_ms, qwen_enc_layers_ms);

    *out_seq_len = total_tokens;
    return enc_output;
}
