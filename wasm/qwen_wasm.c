/*
 * qwen_wasm.c - browser entry points for the Qwen3-ASR engine
 *
 * The module is instantiated on the page's main thread (emscripten creates its
 * pthread workers from there), and every call that would block runs on a worker
 * thread instead, so the UI stays responsive:
 *
 *   qwen_wasm_alloc()            page streams the packed Q8 model into wasm
 *   qwen_wasm_init()             attaches the weights (cheap: no conversion)
 *   qwen_wasm_batch_start()      spawns a thread for a whole-file transcription
 *   qwen_wasm_job_done/take()    poll for the result
 *   qwen_wasm_stream_start()     spawns the streaming loop
 *   qwen_wasm_stream_push()      feed microphone chunks as they arrive
 *   qwen_wasm_stream_finish()    end of audio, returns the final transcript
 *
 * Tokens are not pushed to JS. The engine emits them from whichever thread is
 * decoding, so they go into a mutex-guarded buffer that the page drains with
 * qwen_wasm_take_text() on a timer.
 */

#include "../qwen_asr.h"
#include "../qwen_asr_audio.h"
#include "../qwen_asr_kernels.h"

#include <emscripten.h>
#include <emscripten/threading.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static qwen_ctx_t *g_ctx = NULL;
static void *g_model = NULL;

/* ---- token sink ---- */

static pthread_mutex_t g_out_mutex = PTHREAD_MUTEX_INITIALIZER;
static char *g_out = NULL;
static size_t g_out_len = 0, g_out_cap = 0;

static void out_append(const char *piece, void *userdata) {
    (void)userdata;
    size_t n = strlen(piece);
    pthread_mutex_lock(&g_out_mutex);
    if (g_out_len + n + 1 > g_out_cap) {
        size_t cap = g_out_cap ? g_out_cap : 4096;
        while (g_out_len + n + 1 > cap) cap *= 2;
        char *tmp = (char *)realloc(g_out, cap);
        if (!tmp) { pthread_mutex_unlock(&g_out_mutex); return; }
        g_out = tmp;
        g_out_cap = cap;
    }
    memcpy(g_out + g_out_len, piece, n);
    g_out_len += n;
    g_out[g_out_len] = '\0';
    pthread_mutex_unlock(&g_out_mutex);
}

/* The stream's guess for audio it has not committed yet. Unlike the committed
 * text this is not a queue: each chunk replaces it outright, and the page
 * renders it after the committed text and dimmed. */
static char *g_partial = NULL;
static size_t g_partial_cap = 0;

static void out_partial(const char *text, void *userdata) {
    (void)userdata;
    size_t n = strlen(text);
    pthread_mutex_lock(&g_out_mutex);
    if (n + 1 > g_partial_cap) {
        size_t cap = g_partial_cap ? g_partial_cap : 256;
        while (n + 1 > cap) cap *= 2;
        char *tmp = (char *)realloc(g_partial, cap);
        if (!tmp) { pthread_mutex_unlock(&g_out_mutex); return; }
        g_partial = tmp;
        g_partial_cap = cap;
    }
    memcpy(g_partial, text, n + 1);
    pthread_mutex_unlock(&g_out_mutex);
}

/* Current provisional text. Caller frees. */
EMSCRIPTEN_KEEPALIVE
char *qwen_wasm_take_partial(void) {
    pthread_mutex_lock(&g_out_mutex);
    size_t n = g_partial ? strlen(g_partial) : 0;
    char *r = (char *)malloc(n + 1);
    if (r) {
        memcpy(r, g_partial ? g_partial : "", n);
        r[n] = '\0';
    }
    pthread_mutex_unlock(&g_out_mutex);
    return r;
}

/* Hand the page everything emitted since the last call. Caller frees. */
EMSCRIPTEN_KEEPALIVE
char *qwen_wasm_take_text(void) {
    pthread_mutex_lock(&g_out_mutex);
    char *r = (char *)malloc(g_out_len + 1);
    if (r) {
        memcpy(r, g_out ? g_out : "", g_out_len);
        r[g_out_len] = '\0';
    }
    g_out_len = 0;
    if (g_out) g_out[0] = '\0';
    pthread_mutex_unlock(&g_out_mutex);
    return r;
}

/* ---- lifecycle ---- */

EMSCRIPTEN_KEEPALIVE void *qwen_wasm_alloc(unsigned int n) { return malloc(n); }
EMSCRIPTEN_KEEPALIVE void qwen_wasm_release(void *p) { free(p); }

/* Takes ownership of `model` (kept alive for the context's lifetime, weights
 * are read straight out of it). `aux_dir` holds vocab.json / merges.txt in the
 * emscripten in-memory filesystem. */
EMSCRIPTEN_KEEPALIVE
int qwen_wasm_init(void *model, unsigned int model_len, const char *aux_dir,
                   int n_threads, int verbose) {
    if (g_ctx) return 0;

    qwen_verbose = verbose;
    if (n_threads < 1) n_threads = 1;
    qwen_set_threads(n_threads);

    g_model = model;
    g_ctx = qwen_load_memory(model, model_len, aux_dir);
    if (!g_ctx) return -1;

    qwen_set_token_callback(g_ctx, out_append, NULL);
    qwen_set_partial_callback(g_ctx, out_partial, NULL);
    return 0;
}

EMSCRIPTEN_KEEPALIVE
int qwen_wasm_ready(void) { return g_ctx != NULL; }

EMSCRIPTEN_KEEPALIVE
int qwen_wasm_threads(void) { return qwen_get_threads(); }

static int g_pool_parts = 0;
static double g_pool_ms = 0;

EMSCRIPTEN_KEEPALIVE
void qwen_wasm_pool_selftest(int rounds) {
    qwen_pool_selftest(rounds, &g_pool_parts, &g_pool_ms);
}
EMSCRIPTEN_KEEPALIVE int qwen_wasm_pool_parts(void) { return g_pool_parts; }
EMSCRIPTEN_KEEPALIVE double qwen_wasm_pool_ms(void) { return g_pool_ms; }

/* Diagnostic: does a plain worker thread run at all in this environment? */
static volatile int g_selftest = 0;
static void *selftest_main(void *a) { (void)a; g_selftest = 42; return NULL; }

EMSCRIPTEN_KEEPALIVE
int qwen_wasm_selftest_start(void) {
    pthread_t t;
    g_selftest = 0;
    if (pthread_create(&t, NULL, selftest_main, NULL) != 0) return -1;
    pthread_detach(t);
    return 0;
}

EMSCRIPTEN_KEEPALIVE
int qwen_wasm_selftest_value(void) { return g_selftest; }

EMSCRIPTEN_KEEPALIVE
void qwen_wasm_shutdown(void) {
    if (g_ctx) { qwen_free(g_ctx); g_ctx = NULL; }
    free(g_model);
    g_model = NULL;
}

/* ---- tuning knobs the demo exposes ---- */

EMSCRIPTEN_KEEPALIVE
void qwen_wasm_set_language(const char *lang) {
    if (g_ctx) qwen_set_force_language(g_ctx, (lang && lang[0]) ? lang : NULL);
}

EMSCRIPTEN_KEEPALIVE
void qwen_wasm_set_segment_sec(float secs) {
    if (g_ctx) g_ctx->segment_sec = secs;
}

/* Segments decoded in one sweep of the decoder weights; see qwen_asr.h.
 * Only takes effect with a segment size set and past-text conditioning off. */
void qwen_wasm_set_batch_size(int n) {
    if (!g_ctx) return;
    if (n < 1) n = 1;
    if (n > QWEN_MAX_BATCH) n = QWEN_MAX_BATCH;
    g_ctx->batch_size = n;
}

EMSCRIPTEN_KEEPALIVE
/* Reuse previously decoded text as context for the next chunk.
 *
 * The CLI turns this on for --stream and the browser never did, which is not
 * just a conditioning difference: the streaming loop's periodic re-anchor is
 * gated on it, so with it off the decoded prefix grows for the whole session.
 * Per-chunk prefill then grows with it and the stream falls further behind the
 * longer someone talks. */
EMSCRIPTEN_KEEPALIVE
void qwen_wasm_set_past_text(int on) {
    if (g_ctx) g_ctx->past_text_conditioning = on ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE
void qwen_wasm_set_stream_params(float chunk_sec, int max_new_tokens,
                                 float enc_window_sec) {
    if (!g_ctx) return;
    if (chunk_sec > 0) g_ctx->stream_chunk_sec = chunk_sec;
    if (max_new_tokens > 0) g_ctx->stream_max_new_tokens = max_new_tokens;
    if (enc_window_sec > 0) {
        int frames = (int)(enc_window_sec * 100.0f + 0.5f);
        if (frames < 100) frames = 100;
        if (frames > 800) frames = 800;
        g_ctx->config.enc_n_window_infer = frames;
    }
}

/* ---- perf reporting ---- */

EMSCRIPTEN_KEEPALIVE double qwen_wasm_total_ms(void) { return g_ctx ? g_ctx->perf_total_ms : 0; }
EMSCRIPTEN_KEEPALIVE double qwen_wasm_encode_ms(void) { return g_ctx ? g_ctx->perf_encode_ms : 0; }
EMSCRIPTEN_KEEPALIVE double qwen_wasm_decode_ms(void) { return g_ctx ? g_ctx->perf_decode_ms : 0; }
EMSCRIPTEN_KEEPALIVE int qwen_wasm_text_tokens(void) { return g_ctx ? g_ctx->perf_text_tokens : 0; }

static void reset_perf(void) {
    if (!g_ctx) return;
    g_ctx->perf_total_ms = 0;
    g_ctx->perf_encode_ms = 0;
    g_ctx->perf_decode_ms = 0;
    g_ctx->perf_text_tokens = 0;
    g_ctx->perf_audio_ms = 0;
}

/* ---- batch, off the UI thread ---- */

static pthread_t g_job_thread;
static volatile int g_job_running = 0;
static volatile int g_job_done = 0;
static char *g_job_result = NULL;
static float *g_job_samples = NULL;
static int g_job_n = 0;

static void *batch_main(void *arg) {
    (void)arg;
    g_job_result = qwen_transcribe_audio(g_ctx, g_job_samples, g_job_n);
    g_job_done = 1;
    return NULL;
}

/* Copies the samples, so the page may reuse its buffer immediately. */
EMSCRIPTEN_KEEPALIVE
int qwen_wasm_batch_start(const float *samples, int n_samples) {
    if (!g_ctx || g_job_running || n_samples <= 0) return -1;

    reset_perf();
    free(g_job_result);
    g_job_result = NULL;
    free(g_job_samples);
    g_job_samples = (float *)malloc((size_t)n_samples * sizeof(float));
    if (!g_job_samples) return -1;
    memcpy(g_job_samples, samples, (size_t)n_samples * sizeof(float));
    g_job_n = n_samples;
    g_job_done = 0;

    if (pthread_create(&g_job_thread, NULL, batch_main, NULL) != 0) {
        free(g_job_samples);
        g_job_samples = NULL;
        return -1;
    }
    g_job_running = 1;
    return 0;
}

EMSCRIPTEN_KEEPALIVE
int qwen_wasm_job_done(void) { return g_job_running ? g_job_done : 0; }

/* Reaps the thread and hands over the transcript (caller frees). */
EMSCRIPTEN_KEEPALIVE
char *qwen_wasm_job_take(void) {
    if (!g_job_running || !g_job_done) return NULL;
    pthread_join(g_job_thread, NULL);
    g_job_running = 0;
    free(g_job_samples);
    g_job_samples = NULL;
    char *r = g_job_result;
    g_job_result = NULL;
    return r;
}

/* ========================================================================
 * WebGPU backend support
 *
 * The GPU decoder needs three things from here: where the quantized weights
 * live in wasm memory (so JS can upload them once), the model shape, and the
 * input embeddings for an utterance. Generation itself runs entirely in WGSL;
 * only token ids come back, which this turns into text with the existing
 * tokenizer.
 * ======================================================================== */

#include "../qwen_asr_tokenizer.h"

enum {
    QWEN_W_Q = 0, QWEN_W_K, QWEN_W_V, QWEN_W_O, QWEN_W_GATE_UP, QWEN_W_DOWN,
    QWEN_W_EMBED,
};
enum {
    QWEN_N_INPUT = 0, QWEN_N_POST_ATTN, QWEN_N_QNORM, QWEN_N_KNORM, QWEN_N_FINAL,
};

#define DESC_STRIDE 6

/* Fills [kind, layer, rows, cols, quant_ptr, scale_ptr] per quantized matrix.
 * Returns the number of entries written, or -1 if the model is not quantized. */
EMSCRIPTEN_KEEPALIVE
int qwen_wasm_q8_desc(unsigned int *out, int max_entries) {
    if (!g_ctx || !g_ctx->decoder.quantized) return -1;
    const qwen_config_t *cfg = &g_ctx->config;
    int n = 0;

    #define EMIT(kind_, layer_, mat_) do {                                     \
        const qwen_q8_mat_t *m_ = (mat_);                                      \
        if (!m_->q) return -1;                                                 \
        if (n >= max_entries) return -1;                                       \
        unsigned int *e_ = out + (size_t)n * DESC_STRIDE;                      \
        e_[0] = (unsigned int)(kind_);                                         \
        e_[1] = (unsigned int)(layer_);                                        \
        e_[2] = (unsigned int)m_->rows;                                        \
        e_[3] = (unsigned int)m_->cols;                                        \
        e_[4] = (unsigned int)(uintptr_t)m_->q;                                \
        e_[5] = (unsigned int)(uintptr_t)m_->scales;                           \
        n++;                                                                   \
    } while (0)

    for (int l = 0; l < cfg->dec_layers; l++) {
        const qwen_dec_layer_t *lay = &g_ctx->decoder.layers[l];
        EMIT(QWEN_W_Q, l, &lay->wq_q8);
        EMIT(QWEN_W_K, l, &lay->wk_q8);
        EMIT(QWEN_W_V, l, &lay->wv_q8);
        EMIT(QWEN_W_O, l, &lay->wo_q8);
        EMIT(QWEN_W_GATE_UP, l, &lay->gate_up_q8);
        EMIT(QWEN_W_DOWN, l, &lay->down_q8);
    }
    if (!g_ctx->decoder.embed_quantized) return -1;
    EMIT(QWEN_W_EMBED, 0, &g_ctx->decoder.tok_embeddings_q8);
    #undef EMIT

    return n;
}

/* Fills [kind, layer, count, ptr] per f32 norm weight. */
EMSCRIPTEN_KEEPALIVE
int qwen_wasm_f32_desc(unsigned int *out, int max_entries) {
    if (!g_ctx) return -1;
    const qwen_config_t *cfg = &g_ctx->config;
    int n = 0;

    #define EMITF(kind_, layer_, count_, ptr_) do {                            \
        if (!(ptr_) || n >= max_entries) return -1;                            \
        unsigned int *e_ = out + (size_t)n * 4;                                \
        e_[0] = (unsigned int)(kind_);                                         \
        e_[1] = (unsigned int)(layer_);                                        \
        e_[2] = (unsigned int)(count_);                                        \
        e_[3] = (unsigned int)(uintptr_t)(ptr_);                               \
        n++;                                                                   \
    } while (0)

    for (int l = 0; l < cfg->dec_layers; l++) {
        const qwen_dec_layer_t *lay = &g_ctx->decoder.layers[l];
        EMITF(QWEN_N_INPUT, l, cfg->dec_hidden, lay->input_norm);
        EMITF(QWEN_N_POST_ATTN, l, cfg->dec_hidden, lay->post_attn_norm);
        EMITF(QWEN_N_QNORM, l, cfg->dec_head_dim, lay->q_norm_weight);
        EMITF(QWEN_N_KNORM, l, cfg->dec_head_dim, lay->k_norm_weight);
    }
    EMITF(QWEN_N_FINAL, 0, cfg->dec_hidden, g_ctx->decoder.norm);
    #undef EMITF

    return n;
}

/* [layers, hidden, heads, kv_heads, head_dim, intermediate, vocab,
 *  tok_im_end, tok_endoftext, tok_asr_text] */
EMSCRIPTEN_KEEPALIVE
int qwen_wasm_model_shape(int *out) {
    if (!g_ctx) return -1;
    const qwen_config_t *cfg = &g_ctx->config;
    out[0] = cfg->dec_layers;
    out[1] = cfg->dec_hidden;
    out[2] = cfg->dec_heads;
    out[3] = cfg->dec_kv_heads;
    out[4] = cfg->dec_head_dim;
    out[5] = cfg->dec_intermediate;
    out[6] = cfg->vocab_size;
    out[7] = QWEN_TOKEN_IM_END;
    out[8] = QWEN_TOKEN_ENDOFTEXT;
    out[9] = QWEN_TOKEN_ASR_TEXT;
    return 10;
}

EMSCRIPTEN_KEEPALIVE float qwen_wasm_rms_eps(void) {
    return g_ctx ? g_ctx->config.dec_rms_norm_eps : 1e-6f;
}
EMSCRIPTEN_KEEPALIVE float qwen_wasm_rope_theta(void) {
    return g_ctx ? g_ctx->config.dec_rope_theta : 1e6f;
}

/* ---- embeddings for one utterance, built on a worker thread ---- */

static pthread_t g_emb_thread;
static volatile int g_emb_running = 0;
static volatile int g_emb_done = 0;
static float *g_emb = NULL;
static int g_emb_seq = 0;
static double g_emb_mel_ms = 0, g_emb_enc_ms = 0;
static float *g_emb_samples = NULL;
static int g_emb_n = 0;

static void *embeds_main(void *arg) {
    (void)arg;
    free(g_emb);
    g_emb = qwen_build_embeds(g_ctx, g_emb_samples, g_emb_n, &g_emb_seq,
                             &g_emb_mel_ms, &g_emb_enc_ms);
    g_emb_done = 1;
    return NULL;
}

EMSCRIPTEN_KEEPALIVE
int qwen_wasm_embeds_start(const float *samples, int n_samples) {
    if (!g_ctx || g_emb_running || n_samples <= 0) return -1;
    free(g_emb_samples);
    g_emb_samples = (float *)malloc((size_t)n_samples * sizeof(float));
    if (!g_emb_samples) return -1;
    memcpy(g_emb_samples, samples, (size_t)n_samples * sizeof(float));
    g_emb_n = n_samples;
    g_emb_done = 0;
    if (pthread_create(&g_emb_thread, NULL, embeds_main, NULL) != 0) return -1;
    g_emb_running = 1;
    return 0;
}

EMSCRIPTEN_KEEPALIVE int qwen_wasm_embeds_done(void) { return g_emb_running ? g_emb_done : 0; }

/* Reaps the thread; returns the sequence length (0 on failure). */
EMSCRIPTEN_KEEPALIVE
int qwen_wasm_embeds_finish(void) {
    if (!g_emb_running || !g_emb_done) return 0;
    pthread_join(g_emb_thread, NULL);
    g_emb_running = 0;
    free(g_emb_samples);
    g_emb_samples = NULL;
    return g_emb ? g_emb_seq : 0;
}

EMSCRIPTEN_KEEPALIVE float *qwen_wasm_embeds_ptr(void) { return g_emb; }
EMSCRIPTEN_KEEPALIVE double qwen_wasm_embeds_mel_ms(void) { return g_emb_mel_ms; }
EMSCRIPTEN_KEEPALIVE double qwen_wasm_embeds_enc_ms(void) { return g_emb_enc_ms; }


/* ---- Audio encoder: shape, weights and the conv-stem tap ----
 *
 * The decoder's descriptors above split by representation because every
 * decoder matrix is Q8 and every norm is f32. The encoder mixes them: the
 * conv stem's kernels are plain f32, the linear layers are Q8 in a packed
 * model and f32 in an unpacked one, and every linear has a bias. One table
 * with room for both representations keeps the JS side from having to guess.
 *
 * Entry layout (ENC_DESC_STRIDE words):
 *   [0] kind   [1] layer   [2] rows   [3] cols
 *   [4] int8 pointer, or 0 when the matrix is f32
 *   [5] scales pointer, or 0
 *   [6] f32 pointer (the matrix when not Q8, or the vector for norms/biases)
 *   [7] element count for the f32 vector kinds
 */
enum {
    QWEN_E_CONV1 = 0, QWEN_E_CONV1_B,
    QWEN_E_CONV2, QWEN_E_CONV2_B,
    QWEN_E_CONV3, QWEN_E_CONV3_B,
    QWEN_E_CONV_OUT,
    QWEN_E_ATTN_NORM_W, QWEN_E_ATTN_NORM_B,
    QWEN_E_Q, QWEN_E_Q_B,
    QWEN_E_K, QWEN_E_K_B,
    QWEN_E_V, QWEN_E_V_B,
    QWEN_E_O, QWEN_E_O_B,
    QWEN_E_FFN_NORM_W, QWEN_E_FFN_NORM_B,
    QWEN_E_FC1, QWEN_E_FC1_B,
    QWEN_E_FC2, QWEN_E_FC2_B,
    QWEN_E_LN_POST_W, QWEN_E_LN_POST_B,
    QWEN_E_PROJ1, QWEN_E_PROJ1_B,
    QWEN_E_PROJ2, QWEN_E_PROJ2_B,
};

#define ENC_DESC_STRIDE 8

EMSCRIPTEN_KEEPALIVE
int qwen_wasm_enc_shape(int *out) {
    if (!g_ctx) return -1;
    const qwen_config_t *cfg = &g_ctx->config;
    out[0] = cfg->enc_d_model;
    out[1] = cfg->enc_layers;
    out[2] = cfg->enc_heads;
    out[3] = cfg->enc_head_dim;
    out[4] = cfg->enc_ffn_dim;
    out[5] = cfg->enc_output_dim;
    out[6] = cfg->enc_n_window;
    out[7] = cfg->enc_n_window_infer;
    out[8] = cfg->enc_chunk_size;
    out[9] = cfg->enc_conv_proj_dim;
    out[10] = QWEN_CONV_HIDDEN;
    return 11;
}

EMSCRIPTEN_KEEPALIVE
int qwen_wasm_enc_desc(unsigned int *out, int max_entries) {
    if (!g_ctx) return -1;
    const qwen_config_t *cfg = &g_ctx->config;
    const qwen_encoder_t *enc = &g_ctx->encoder;
    int n = 0;

    #define ENC_SLOT()                                                         \
        (n >= max_entries ? NULL : (out + (size_t)(n) * ENC_DESC_STRIDE))

    /* A weight matrix, in whichever representation the model provided. */
    #define ENC_MAT(kind_, layer_, w_) do {                                    \
        const qwen_wmat_t *w = (w_);                                           \
        unsigned int *e = ENC_SLOT();                                          \
        if (!e) return -1;                                                     \
        e[0] = (unsigned int)(kind_);                                          \
        e[1] = (unsigned int)(layer_);                                         \
        e[2] = (unsigned int)w->rows;                                          \
        e[3] = (unsigned int)w->cols;                                          \
        e[4] = (unsigned int)(uintptr_t)w->q8.q;                               \
        e[5] = (unsigned int)(uintptr_t)w->q8.scales;                          \
        e[6] = (unsigned int)(uintptr_t)w->f32;                                \
        e[7] = 0;                                                              \
        if (!e[4] && !e[6]) return -1;                                         \
        n++;                                                                   \
    } while (0)

    /* A plain f32 run: conv kernel, bias or norm vector. */
    #define ENC_VEC(kind_, layer_, count_, ptr_) do {                          \
        unsigned int *e = ENC_SLOT();                                          \
        if (!e || !(ptr_)) return -1;                                          \
        e[0] = (unsigned int)(kind_);                                          \
        e[1] = (unsigned int)(layer_);                                         \
        e[2] = 0; e[3] = 0; e[4] = 0; e[5] = 0;                                \
        e[6] = (unsigned int)(uintptr_t)(ptr_);                                \
        e[7] = (unsigned int)(count_);                                         \
        n++;                                                                   \
    } while (0)

    const int CH = QWEN_CONV_HIDDEN;
    ENC_VEC(QWEN_E_CONV1,   0, CH * 1 * 3 * 3, enc->conv1_weight);
    ENC_VEC(QWEN_E_CONV1_B, 0, CH,             enc->conv1_bias);
    ENC_VEC(QWEN_E_CONV2,   0, CH * CH * 3 * 3, enc->conv2_weight);
    ENC_VEC(QWEN_E_CONV2_B, 0, CH,              enc->conv2_bias);
    ENC_VEC(QWEN_E_CONV3,   0, CH * CH * 3 * 3, enc->conv3_weight);
    ENC_VEC(QWEN_E_CONV3_B, 0, CH,              enc->conv3_bias);
    ENC_MAT(QWEN_E_CONV_OUT, 0, &enc->conv_out_weight);

    for (int l = 0; l < cfg->enc_layers; l++) {
        const qwen_enc_layer_t *lay = &enc->layers[l];
        ENC_VEC(QWEN_E_ATTN_NORM_W, l, cfg->enc_d_model, lay->attn_norm_weight);
        ENC_VEC(QWEN_E_ATTN_NORM_B, l, cfg->enc_d_model, lay->attn_norm_bias);
        ENC_MAT(QWEN_E_Q, l, &lay->wq_weight);
        ENC_VEC(QWEN_E_Q_B, l, cfg->enc_d_model, lay->wq_bias);
        ENC_MAT(QWEN_E_K, l, &lay->wk_weight);
        ENC_VEC(QWEN_E_K_B, l, cfg->enc_d_model, lay->wk_bias);
        ENC_MAT(QWEN_E_V, l, &lay->wv_weight);
        ENC_VEC(QWEN_E_V_B, l, cfg->enc_d_model, lay->wv_bias);
        ENC_MAT(QWEN_E_O, l, &lay->wo_weight);
        ENC_VEC(QWEN_E_O_B, l, cfg->enc_d_model, lay->wo_bias);
        ENC_VEC(QWEN_E_FFN_NORM_W, l, cfg->enc_d_model, lay->ffn_norm_weight);
        ENC_VEC(QWEN_E_FFN_NORM_B, l, cfg->enc_d_model, lay->ffn_norm_bias);
        ENC_MAT(QWEN_E_FC1, l, &lay->fc1_weight);
        ENC_VEC(QWEN_E_FC1_B, l, cfg->enc_ffn_dim, lay->fc1_bias);
        ENC_MAT(QWEN_E_FC2, l, &lay->fc2_weight);
        ENC_VEC(QWEN_E_FC2_B, l, cfg->enc_d_model, lay->fc2_bias);
    }

    ENC_VEC(QWEN_E_LN_POST_W, 0, cfg->enc_d_model, enc->ln_post_weight);
    ENC_VEC(QWEN_E_LN_POST_B, 0, cfg->enc_d_model, enc->ln_post_bias);
    ENC_MAT(QWEN_E_PROJ1, 0, &enc->proj1_weight);
    ENC_VEC(QWEN_E_PROJ1_B, 0, cfg->enc_d_model, enc->proj1_bias);
    ENC_MAT(QWEN_E_PROJ2, 0, &enc->proj2_weight);
    ENC_VEC(QWEN_E_PROJ2_B, 0, cfg->enc_output_dim, enc->proj2_bias);

    #undef ENC_MAT
    #undef ENC_VEC
    #undef ENC_SLOT
    return n;
}

/* Ask for f32 activations everywhere (1) or restore the default (-1). */
EMSCRIPTEN_KEEPALIVE void qwen_wasm_set_q8_batch_max(int n) { qwen_set_q8_batch_max(n); }

/* Conv-stem tap; see qwen_enc_tap in qwen_asr.h. */
EMSCRIPTEN_KEEPALIVE void qwen_wasm_enc_tap_set(int on) { qwen_enc_tap = on ? 1 : 0; }
EMSCRIPTEN_KEEPALIVE float *qwen_wasm_enc_tap_mel(void) { return qwen_enc_tap_mel; }
EMSCRIPTEN_KEEPALIVE int qwen_wasm_enc_tap_frames(void) { return qwen_enc_tap_frames; }
EMSCRIPTEN_KEEPALIVE float *qwen_wasm_enc_tap_ptr(void) { return qwen_enc_tap_conv; }
EMSCRIPTEN_KEEPALIVE float *qwen_wasm_enc_tap_out(void) { return qwen_enc_tap_out; }
EMSCRIPTEN_KEEPALIVE int qwen_wasm_enc_tap_tokens(void) { return qwen_enc_tap_tokens; }



/* ---- Audio tower on the GPU, called from the streaming thread ----
 *
 * The streaming loop runs on a pthread and WebGPU lives on the main thread and
 * is asynchronous, so the two cannot simply call each other. The worker posts
 * the request, then sleeps until the main thread's async continuation has
 * written the result back and flipped a flag. Sleeping rather than spinning
 * matters: the thread pool already spins hard, and a spinning stream thread on
 * top of it starves the very workers the GPU path is trying to leave idle.
 *
 * If the main thread never answers - no tower installed, a GPU error, a page
 * that stopped scheduling work - the wait gives up and the caller falls back
 * to encoding on the CPU rather than hanging the stream.
 */
#define QWEN_ENC_HOOK_TIMEOUT_MS 20000

static volatile int g_enc_hook_state = 0;   /* 0 idle, 1 pending, 2 done, 3 failed */
static float *g_enc_hook_out = NULL;
static int g_enc_hook_seq = 0;

/* Called on the main thread when the GPU is finished. Takes ownership of buf. */
EMSCRIPTEN_KEEPALIVE
void qwen_wasm_enc_hook_done(float *buf, int seq_len) {
    g_enc_hook_out = buf;
    g_enc_hook_seq = seq_len;
    __atomic_store_n(&g_enc_hook_state, (buf && seq_len > 0) ? 2 : 3, __ATOMIC_RELEASE);
}

static float *gpu_encoder_hook(void *ud, const float *mel, int mel_frames, int *out_seq_len) {
    (void)ud;
    g_enc_hook_out = NULL;
    g_enc_hook_seq = 0;
    __atomic_store_n(&g_enc_hook_state, 1, __ATOMIC_RELEASE);

    MAIN_THREAD_ASYNC_EM_ASM({
        if (Module.__gpuEncode) Module.__gpuEncode($0, $1);
        else _qwen_wasm_enc_hook_done(0, 0);
    }, (int)(uintptr_t)mel, mel_frames);

    double t0 = emscripten_get_now();
    for (;;) {
        int st = __atomic_load_n(&g_enc_hook_state, __ATOMIC_ACQUIRE);
        if (st >= 2) break;
        if (emscripten_get_now() - t0 > QWEN_ENC_HOOK_TIMEOUT_MS) {
            __atomic_store_n(&g_enc_hook_state, 3, __ATOMIC_RELEASE);
            return NULL;
        }
        emscripten_thread_sleep(1);
    }

    if (__atomic_load_n(&g_enc_hook_state, __ATOMIC_ACQUIRE) != 2) return NULL;
    *out_seq_len = g_enc_hook_seq;
    return g_enc_hook_out;
}

/* Route the audio tower through Module.__gpuEncode (1) or back to wasm (0). */
EMSCRIPTEN_KEEPALIVE
void qwen_wasm_set_gpu_encoder(int on) {
    if (!g_ctx) return;
    qwen_set_encoder_hook(g_ctx, on ? gpu_encoder_hook : NULL, NULL);
}

/* ---- Audio tower on the GPU: mel here, encoder there, assembly back here ----
 *
 * qwen_build_embeds() runs mel, the encoder and the prompt assembly as one
 * step, which leaves no way to substitute an encoder that lives on the GPU.
 * These split it: compute the mel, hand it out, take an encoder output back,
 * and assemble the decoder's inputs around it. The result lands in the same
 * place qwen_wasm_embeds_finish() leaves it, so the GPU decoder path that
 * follows is unchanged.
 */

static pthread_t g_mel_thread;
static volatile int g_mel_running = 0;
static volatile int g_mel_done = 0;
static float *g_mel_samples = NULL;
static int g_mel_n = 0;
static float *g_mel = NULL;
static int g_mel_frames = 0;
static double g_mel_ms = 0;

static void *mel_main(void *arg) {
    (void)arg;
    double t0 = emscripten_get_now();
    free(g_mel);
    g_mel_frames = 0;
    g_mel = qwen_mel_spectrogram(g_mel_samples, g_mel_n, &g_mel_frames);
    g_mel_ms = emscripten_get_now() - t0;
    g_mel_done = 1;
    return NULL;
}

EMSCRIPTEN_KEEPALIVE
int qwen_wasm_mel_start(const float *samples, int n_samples) {
    if (!g_ctx || g_mel_running || n_samples <= 0) return -1;
    free(g_mel_samples);
    g_mel_samples = (float *)malloc((size_t)n_samples * sizeof(float));
    if (!g_mel_samples) return -1;
    memcpy(g_mel_samples, samples, (size_t)n_samples * sizeof(float));
    g_mel_n = n_samples;
    g_mel_done = 0;
    if (pthread_create(&g_mel_thread, NULL, mel_main, NULL) != 0) return -1;
    g_mel_running = 1;
    return 0;
}

EMSCRIPTEN_KEEPALIVE int qwen_wasm_mel_done(void) { return g_mel_running ? g_mel_done : 0; }

/* Reaps the thread; returns the frame count (0 on failure). */
EMSCRIPTEN_KEEPALIVE
int qwen_wasm_mel_finish(void) {
    if (!g_mel_running || !g_mel_done) return 0;
    pthread_join(g_mel_thread, NULL);
    g_mel_running = 0;
    free(g_mel_samples);
    g_mel_samples = NULL;
    return g_mel ? g_mel_frames : 0;
}

EMSCRIPTEN_KEEPALIVE float *qwen_wasm_mel_ptr(void) { return g_mel; }
EMSCRIPTEN_KEEPALIVE double qwen_wasm_mel_ms(void) { return g_mel_ms; }

/* Assemble the decoder inputs around an encoder output computed on the GPU.
 * Returns the sequence length; the embeddings are at qwen_wasm_embeds_ptr(). */
EMSCRIPTEN_KEEPALIVE
int qwen_wasm_embeds_from_enc(const float *enc_output, int enc_seq_len) {
    if (!g_ctx || !enc_output || enc_seq_len <= 0) return 0;
    char vocab_path[1024];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", g_ctx->model_dir);
    qwen_tokenizer_t *tok = qwen_tokenizer_load(vocab_path);
    if (!tok) return 0;

    int seq = 0;
    free(g_emb);
    g_emb = qwen_assemble_embeds(g_ctx, tok, enc_output, enc_seq_len, NULL, 0, &seq);
    qwen_tokenizer_free(tok);
    if (!g_emb) return 0;
    g_emb_seq = seq;
    return seq;
}

/* ---- CPU prefill, so the GPU only has to run generation ----
 *
 * Mel, encoder and decoder prefill stay on the CPU: prefill is a batched GEMM
 * that wasm handles reasonably, while token generation is the bandwidth-bound
 * part worth moving to the GPU. After this returns, the KV cache in wasm memory
 * holds the whole prompt and the first generated token is known. */

static pthread_t g_pf_thread;
static volatile int g_pf_running = 0;
static volatile int g_pf_done = 0;
static int g_pf_first_token = -1;
static float *g_pf_samples = NULL;
static int g_pf_n = 0;
static double g_pf_prefill_ms = 0;

static void *prefill_main(void *arg) {
    (void)arg;
    int seq = 0;
    double mel_ms = 0, enc_ms = 0;
    float *embeds = qwen_build_embeds(g_ctx, g_pf_samples, g_pf_n, &seq, &mel_ms, &enc_ms);
    if (!embeds) { g_pf_first_token = -1; g_pf_done = 1; return NULL; }

    g_emb_mel_ms = mel_ms;
    g_emb_enc_ms = enc_ms;

    double t0 = emscripten_get_now();
    g_ctx->kv_cache_len = 0;
    qwen_decoder_prefill(g_ctx, embeds, seq - 1);
    g_pf_first_token = qwen_decoder_forward(g_ctx, embeds + (size_t)(seq - 1) * g_ctx->config.dec_hidden);
    g_pf_prefill_ms = emscripten_get_now() - t0;

    free(embeds);
    g_pf_done = 1;
    return NULL;
}

EMSCRIPTEN_KEEPALIVE
int qwen_wasm_prefill_start(const float *samples, int n_samples) {
    if (!g_ctx || g_pf_running || n_samples <= 0) return -1;
    free(g_pf_samples);
    g_pf_samples = (float *)malloc((size_t)n_samples * sizeof(float));
    if (!g_pf_samples) return -1;
    memcpy(g_pf_samples, samples, (size_t)n_samples * sizeof(float));
    g_pf_n = n_samples;
    g_pf_done = 0;
    g_pf_first_token = -1;
    if (pthread_create(&g_pf_thread, NULL, prefill_main, NULL) != 0) return -1;
    g_pf_running = 1;
    return 0;
}

EMSCRIPTEN_KEEPALIVE int qwen_wasm_prefill_done(void) { return g_pf_running ? g_pf_done : 0; }

/* Reaps the thread, returns the first generated token id (-1 on failure). */
EMSCRIPTEN_KEEPALIVE
int qwen_wasm_prefill_finish(void) {
    if (!g_pf_running || !g_pf_done) return -2;
    pthread_join(g_pf_thread, NULL);
    g_pf_running = 0;
    free(g_pf_samples);
    g_pf_samples = NULL;
    return g_pf_first_token;
}

EMSCRIPTEN_KEEPALIVE double qwen_wasm_prefill_ms(void) { return g_pf_prefill_ms; }

/* Non-zero when a forced language put <asr_text> in the prompt, so generation
 * starts already inside the transcript rather than waiting for that marker. */
EMSCRIPTEN_KEEPALIVE
int qwen_wasm_prompt_has_asr_text(void) {
    return g_ctx ? (g_ctx->n_force_prompt_tokens > 0) : 0;
}
EMSCRIPTEN_KEEPALIVE unsigned int qwen_wasm_kv_k_ptr(void) {
    return g_ctx ? (unsigned int)(uintptr_t)g_ctx->kv_cache_k : 0;
}
EMSCRIPTEN_KEEPALIVE unsigned int qwen_wasm_kv_v_ptr(void) {
    return g_ctx ? (unsigned int)(uintptr_t)g_ctx->kv_cache_v : 0;
}
EMSCRIPTEN_KEEPALIVE int qwen_wasm_kv_len(void) { return g_ctx ? g_ctx->kv_cache_len : 0; }
EMSCRIPTEN_KEEPALIVE int qwen_wasm_kv_stride(void) { return g_ctx ? g_ctx->kv_cache_max : 0; }

/* ---- tokenizer ---- */

static qwen_tokenizer_t *g_tok = NULL;

EMSCRIPTEN_KEEPALIVE
const char *qwen_wasm_token_text(int id) {
    if (!g_tok) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/vocab.json", g_ctx ? g_ctx->model_dir : "/model");
        g_tok = qwen_tokenizer_load(path);
        if (!g_tok) return "";
    }
    const char *p = qwen_tokenizer_decode(g_tok, id);
    return p ? p : "";
}

/* ---- live streaming ---- */

static qwen_live_audio_t *g_live = NULL;
static pthread_t g_stream_thread;
static int g_stream_running = 0;
static char *g_stream_result = NULL;


static void *stream_main(void *arg) {
    (void)arg;
    g_stream_result = qwen_transcribe_stream_live(g_ctx, g_live);
    return NULL;
}

EMSCRIPTEN_KEEPALIVE
int qwen_wasm_stream_start(void) {
    if (!g_ctx || g_stream_running) return -1;

    reset_perf();
    free(g_stream_result);
    g_stream_result = NULL;
    pthread_mutex_lock(&g_out_mutex);
    if (g_partial) g_partial[0] = '\0';
    pthread_mutex_unlock(&g_out_mutex);

    g_live = qwen_live_audio_create();
    if (!g_live) return -1;

    /* The engine's stream loop blocks waiting for audio, so it gets its own
     * thread; this worker stays free to accept pushes from the page. */
    if (pthread_create(&g_stream_thread, NULL, stream_main, NULL) != 0) {
        qwen_live_audio_free(g_live);
        g_live = NULL;
        return -1;
    }
    g_stream_running = 1;
    return 0;
}

EMSCRIPTEN_KEEPALIVE
void qwen_wasm_stream_push(const float *samples, int n_samples) {
    if (g_stream_running && g_live) qwen_live_audio_push(g_live, samples, n_samples);
}

/* Signals end of audio, waits for the decoder to drain, returns the final
 * transcript (caller frees). */
EMSCRIPTEN_KEEPALIVE
char *qwen_wasm_stream_finish(void) {
    if (!g_stream_running) return NULL;

    qwen_live_audio_set_eof(g_live);
    pthread_join(g_stream_thread, NULL);
    g_stream_running = 0;

    /* live->thread is 0 here (we never started a reader thread), so free()
     * will not try to join anything. */
    qwen_live_audio_free(g_live);
    g_live = NULL;

    char *r = g_stream_result;
    g_stream_result = NULL;
    return r;
}
