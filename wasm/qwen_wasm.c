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
