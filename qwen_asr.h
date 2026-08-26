/*
 * qwen_asr.h - Qwen3-ASR Pure C Inference Engine
 *
 * Supports both Qwen3-ASR-1.7B and Qwen3-ASR-0.6B models.
 */

#ifndef QWEN_ASR_H
#define QWEN_ASR_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <pthread.h>

#include "qwen_asr_kernels.h"

/* ========================================================================
 * Constants
 * ======================================================================== */

#define QWEN_SAMPLE_RATE      16000
#define QWEN_MEL_BINS         128
#define QWEN_HOP_LENGTH       160
#define QWEN_WINDOW_SIZE      400
#define QWEN_VOCAB_SIZE       151936

/* Maximum layer counts (for static array sizing) */
#define QWEN_MAX_ENC_LAYERS   24
#define QWEN_MAX_DEC_LAYERS   28

/* Special token IDs */
#define QWEN_TOKEN_IM_START     151644
#define QWEN_TOKEN_IM_END       151645
#define QWEN_TOKEN_ENDOFTEXT    151643
#define QWEN_TOKEN_AUDIO_START  151669
#define QWEN_TOKEN_AUDIO_END    151670
#define QWEN_TOKEN_AUDIO_PAD    151676
#define QWEN_TOKEN_ASR_TEXT     151704

/* Conv2D stem constants */
#define QWEN_CONV_HIDDEN      480
#define QWEN_CONV_KERNEL      3

/* ========================================================================
 * Model Configuration (populated from config.json)
 * ======================================================================== */

typedef struct {
    /* Audio encoder */
    int enc_d_model;           /* 1024 or 896 */
    int enc_layers;            /* 24 or 18 */
    int enc_heads;             /* 16 or 14 */
    int enc_head_dim;          /* 64 */
    int enc_ffn_dim;           /* 4096 or 3584 */
    int enc_output_dim;        /* 2048 or 1024 */
    int enc_n_window;          /* 50 */
    int enc_n_window_infer;    /* 800 */
    int enc_chunk_size;        /* n_window * 2 = 100 */
    int enc_conv_proj_dim;     /* CONV_HIDDEN * 16 = 7680 */

    /* LLM decoder */
    int dec_hidden;            /* 2048 or 1024 */
    int dec_layers;            /* 28 */
    int dec_heads;             /* 16 */
    int dec_kv_heads;          /* 8 */
    int dec_head_dim;          /* 128 */
    int dec_intermediate;      /* 6144 or 3072 */
    int vocab_size;            /* 151936 */
    float dec_rms_norm_eps;    /* 1e-6 */
    float dec_rope_theta;      /* 1e6 */
} qwen_config_t;

/* ========================================================================
 * Audio Encoder Layer
 * ======================================================================== */

typedef struct {
    /* Self-attention (ALL have biases) - pre-converted to f32 */
    qwen_wmat_t wq_weight;          /* [d_model, d_model] */
    float *wq_bias;            /* [d_model] */
    qwen_wmat_t wk_weight;          /* [d_model, d_model] */
    float *wk_bias;            /* [d_model] */
    qwen_wmat_t wv_weight;          /* [d_model, d_model] */
    float *wv_bias;            /* [d_model] */
    qwen_wmat_t wo_weight;          /* [d_model, d_model] */
    float *wo_bias;            /* [d_model] */

    /* Pre-attention LayerNorm (with bias) */
    float *attn_norm_weight;   /* [d_model] */
    float *attn_norm_bias;     /* [d_model] */

    /* FFN: GELU(fc1(x)) -> fc2 (ALL have biases) - pre-converted to f32 */
    qwen_wmat_t fc1_weight;         /* [ffn_dim, d_model] */
    float *fc1_bias;           /* [ffn_dim] */
    qwen_wmat_t fc2_weight;         /* [d_model, ffn_dim] */
    float *fc2_bias;           /* [d_model] */

    /* Pre-FFN LayerNorm (with bias) */
    float *ffn_norm_weight;    /* [d_model] */
    float *ffn_norm_bias;      /* [d_model] */
} qwen_enc_layer_t;

typedef struct {
    /* Conv2D stem (3 layers, each 3x3, stride 2) */
    float *conv1_weight;       /* [480, 1, 3, 3] */
    float *conv1_bias;         /* [480] */
    float *conv2_weight;       /* [480, 480, 3, 3] */
    float *conv2_bias;         /* [480] */
    float *conv3_weight;       /* [480, 480, 3, 3] */
    float *conv3_bias;         /* [480] */

    /* Conv output projection - pre-converted to f32 */
    qwen_wmat_t conv_out_weight;    /* [d_model, 7680] */

    /* Transformer layers */
    qwen_enc_layer_t layers[QWEN_MAX_ENC_LAYERS];

    /* Final LayerNorm */
    float *ln_post_weight;     /* [d_model] */
    float *ln_post_bias;       /* [d_model] */

    /* Projection layers - pre-converted to f32 */
    qwen_wmat_t proj1_weight;       /* [d_model, d_model] */
    float *proj1_bias;         /* [d_model] */
    qwen_wmat_t proj2_weight;       /* [output_dim, d_model] */
    float *proj2_bias;         /* [output_dim] */

    /* Non-zero when the tower weights were deliberately not loaded (see
     * qwen_gpu_resident): the GPU owns them, an encoder hook runs them, and
     * the CPU forward refuses cleanly instead of running on nothing. */
    int weights_absent;
} qwen_encoder_t;

/* ========================================================================
 * LLM Decoder Layer
 * ======================================================================== */

typedef struct {
    /* Self-attention (NO biases in decoder) */
    uint16_t *wq_weight_bf16;  /* [n_heads*head_dim, hidden] */
    uint16_t *wk_weight_bf16;  /* [n_kv_heads*head_dim, hidden] */
    uint16_t *wv_weight_bf16;  /* [n_kv_heads*head_dim, hidden] */
    uint16_t *wo_weight_bf16;  /* [hidden, n_heads*head_dim] */

    /* Per-head Q/K RMSNorm */
    float *q_norm_weight;      /* [head_dim] = [128] */
    float *k_norm_weight;      /* [head_dim] = [128] */

    /* RMSNorm (no bias) */
    float *input_norm;         /* [hidden] */
    float *post_attn_norm;     /* [hidden] */

    /* SwiGLU MLP (NO biases) */
    uint16_t *gate_weight_bf16; /* [intermediate, hidden] */
    uint16_t *up_weight_bf16;   /* [intermediate, hidden] */
    uint16_t *down_weight_bf16; /* [hidden, intermediate] */

    /* Fused gate+up weight for single-token matvec [2*intermediate, hidden] */
    uint16_t *gate_up_fused_bf16;

    /* Q8 block-quantized mirrors, used when the decoder runs quantized.
     * Only one of the bf16 / q8 sets is populated. */
    qwen_q8_mat_t wq_q8, wk_q8, wv_q8, wo_q8;
    qwen_q8_mat_t gate_up_q8, down_q8;
} qwen_dec_layer_t;

typedef struct {
    /* Token embeddings (tied with lm_head) */
    uint16_t *tok_embeddings_bf16; /* [vocab_size, hidden] */
    qwen_q8_mat_t tok_embeddings_q8;

    /* Non-zero when the decoder layer weights are stored as Q8 blocks. */
    int quantized;
    /* Non-zero when the transformer layer weights were deliberately not
     * loaded (see qwen_gpu_resident): they live on a GPU that a decoder hook
     * owns, and every CPU decode path refuses cleanly instead of running. */
    int layers_absent;
    /* Non-zero when the tied embedding / LM head is also Q8. Tracked
     * separately because it feeds the input representation directly. */
    int embed_quantized;

    /* Transformer layers */
    qwen_dec_layer_t layers[QWEN_MAX_DEC_LAYERS];

    /* Final RMSNorm */
    float *norm;               /* [hidden] */
} qwen_decoder_t;

/* Weight storage for the decoder. Set before qwen_load(); the
 * QWEN_WEIGHTS=bf16|q8|q8-lm environment variable overrides it.
 *
 *   QWEN_WEIGHTS_BF16   everything straight out of the bf16 mmap
 *   QWEN_WEIGHTS_Q8     transformer layers Q8, tied embedding / LM head bf16
 *   QWEN_WEIGHTS_Q8_LM  the LM head is Q8 as well
 *   QWEN_WEIGHTS_Q4     transformer layers 4-bit, LM head Q8
 *
 * The default is q8-lm. It is output-identical to bf16 on the 1.7B model across
 * the whole regression suite, which is the model this engine is tuned for. On
 * the 0.6B model it measurably degrades quality — logit margins there are small
 * enough that block-int8 LM head weights flip token decisions — so `q8` exists
 * for that case. */
#define QWEN_WEIGHTS_BF16  0
#define QWEN_WEIGHTS_Q8    1
#define QWEN_WEIGHTS_Q8_LM 2
/* Transformer layers at 4 bits, tied embedding / LM head still Q8. The head
 * decides the token and 4 bits there is where quality goes first. */
#define QWEN_WEIGHTS_Q4    3
extern int qwen_weight_quant;

/* When non-zero at load time, a packed model image is allowed to omit the
 * decoder transformer-layer tensors. The browser sets this when the GPU owns
 * those weights: they are uploaded straight from the cached model file, and
 * materializing a second copy in wasm memory - which can never shrink - would
 * hold ~1.5 GB for nothing. The embedding table, norms and encoder still load
 * normally; CPU decode paths fail cleanly if reached. */
extern int qwen_gpu_resident;

/* ========================================================================
 * Token Callback (streaming output)
 * ======================================================================== */

/* Called for each decoded text token during autoregressive generation.
 * 'piece' is the decoded token string (UTF-8). */
typedef void (*qwen_token_cb)(const char *piece, void *userdata);

/* Provisional hypothesis for the audio the stream has not committed yet.
 *
 * Streaming decodes the whole utterance every chunk and only releases the part
 * that has stopped changing; the tail beyond that frontier is a real guess the
 * engine already has, and it is what a UI shows greyed out ahead of the
 * confirmed text. Called once per chunk with that tail, which *replaces*
 * whatever the previous call passed - it is not a delta and it may be revised
 * or disappear. An empty string means there is nothing provisional. */
typedef void (*qwen_partial_cb)(const char *text, void *userdata);

/* Run the audio tower somewhere else.
 *
 * Streaming re-encodes its tail window every chunk, which measurement puts at
 * the bulk of the per-chunk cost - halving the window took a 41s clip from
 * 139 s to 69 s. Moving that work to a GPU is worth more than any tuning of
 * the window, but the streaming loop is C and WebGPU is asynchronous, so the
 * loop asks through this hook rather than calling the encoder directly.
 *
 * Returns a malloc'd [*out_seq_len][enc_output_dim] the caller frees, or NULL
 * to fall back to the built-in encoder. */
typedef float *(*qwen_encoder_hook)(void *userdata, const float *mel,
                                    int mel_frames, int *out_seq_len);

/* Substitute decoder for the streaming loop (a GPU one, in the browser).
 *
 * The hook owns a decoder state that persists across calls: it must prefill
 * embeds[reuse_len .. total_seq-1] against that state - rows before reuse_len
 * are unchanged since the previous call, which is the caller's guarantee -
 * then generate up to max_new tokens greedily and write their ids to
 * out_tokens, including the terminating im_end/endoftext when one is
 * produced. Returns the id count, or -1 to make the caller fall back to the
 * built-in decoder with nothing reused. */
typedef int (*qwen_decoder_hook)(void *userdata, const float *embeds,
                                 int total_seq, int reuse_len, int max_new,
                                 int *out_tokens);

/* ========================================================================
 * Main Context
 * ======================================================================== */

typedef struct {
    qwen_config_t config;
    qwen_encoder_t encoder;
    qwen_decoder_t decoder;

    /* Model files (kept open for mmap) */
    void *safetensors;         /* multi_safetensors_t* */
    char model_dir[512];

    /* KV cache for decoder */
    qwen_f16_t *kv_cache_k;    /* [layers, max_seq, kv_heads * head_dim], f16 */
    qwen_f16_t *kv_cache_v;
    int kv_cache_len;
    int kv_cache_max;

    /* Persistent decoder buffers (single-token generation) */
    float *dec_x, *dec_x_norm, *dec_q, *dec_k, *dec_v;
    float *dec_attn_out, *dec_proj_out;
    float *dec_gate, *dec_up, *dec_ffn_out;
    float *dec_rope_cos, *dec_rope_sin;

    /* Persistent decoder prefill buffers (multi-token prefill) */
    float *pref_x, *pref_x_norm, *pref_q, *pref_k, *pref_v;
    float *pref_attn_out, *pref_proj_out, *pref_ffn_out;
    float *pref_gate, *pref_gate_up;
    int pref_seq_cap;

    /* Cached RoPE tables for decoder positions */
    float *rope_cache_cos, *rope_cache_sin;   /* [pos, head_dim] */
    float *rope_inv_freq;                     /* [head_dim / 2] */
    int rope_cache_cap;                       /* cached positions */
    int rope_inv_freq_half;                   /* cached half-dim */

    /* Token streaming callback (optional) */
    qwen_token_cb token_cb;
    void *token_cb_userdata;
    qwen_partial_cb partial_cb;
    void *partial_cb_userdata;
    qwen_encoder_hook encoder_hook;
    void *encoder_hook_userdata;
    qwen_decoder_hook decoder_hook;
    void *decoder_hook_userdata;

    /* Segmentation settings */
    float segment_sec;             /* 0 = no splitting, default full-audio decode */
    int batch_size;                /* segments decoded in one weight sweep (1 = off) */
    float search_sec;              /* segment-cutting silence search window ± seconds (default 3) */

    /* Streaming settings */
    float stream_chunk_sec;        /* chunk interval in seconds (default 2.0) */
    int stream_rollback;           /* tokens to roll back per chunk (default 5) */
    int stream_unfixed_chunks;     /* cold-start chunks without prefix (default 2) */
    int stream_max_new_tokens;     /* max generated tokens per streaming step (default 32) */
    int past_text_conditioning;    /* 1=enable past text conditioning in -S/--stream (default: off).
                                    * In segmented mode, this also enables boundary cleanup/post-processing. */
    int skip_silence;              /* 1=drop long silent spans before transcription */

    /* Optional prompt/language controls */
    char *prompt;                  /* system prompt text (UTF-8) */
    char *force_language;          /* normalized language name, or NULL */
    int *prompt_tokens;            /* cached token ids for prompt text */
    int n_prompt_tokens;
    int *force_prompt_tokens;      /* cached token ids for "language X" + <asr_text> */
    int n_force_prompt_tokens;
    int prompt_tokens_ready;       /* cache valid flag */

    /* Per-run performance stats (populated by last transcription call) */
    double perf_total_ms;          /* end-to-end inference time in milliseconds */
    int perf_text_tokens;          /* emitted text tokens (after <asr_text>) */
    double perf_audio_ms;          /* input audio duration in milliseconds */
    double perf_encode_ms;         /* mel + encoder time in milliseconds */
    double perf_decode_ms;         /* decoder prefill + decode time in milliseconds */
} qwen_ctx_t;

/* ========================================================================
 * Live Audio (incremental stdin streaming)
 * ======================================================================== */

typedef struct {
    /* Written by reader thread under mutex */
    float *samples;
    int64_t sample_offset;      /* global index of samples[0] */
    int64_t n_samples;          /* number of valid samples in buffer */
    int64_t capacity;           /* allocated capacity (in samples) */
    int eof;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    pthread_t thread;
} qwen_live_audio_t;

/* ========================================================================
 * API Functions
 * ======================================================================== */

/* Write a pre-quantized single-file model image (see qwen_asr_pack.c).
 * Returns 0 on success. */
int qwen_pack_q8(const char *model_dir, const char *out_path);

/* Same, but four_bit narrows the decoder transformer layers to 4 bits and
 * awq_stats (a --calib-out dump, optional but strongly recommended there) bakes
 * channel rescaling into the weights and norms, so the runtime needs neither
 * the statistics nor a conversion pass. */
int qwen_pack(const char *model_dir, const char *out_path, int four_bit,
              const char *awq_stats);

/* Load model from directory */
qwen_ctx_t *qwen_load(const char *model_dir);

/* Load a model whose weights are already in memory (the wasm build fetches the
 * packed image instead of mmapping it). `model_data` must stay alive for the
 * lifetime of the context and is not freed by qwen_free(). `aux_dir` is the
 * directory holding vocab.json / merges.txt. */
qwen_ctx_t *qwen_load_memory(void *model_data, size_t model_size,
                             const char *aux_dir);

/* Free all resources */
void qwen_free(qwen_ctx_t *ctx);

/* Set a callback to receive each decoded token as it's generated.
 * Set cb=NULL to disable. The callback is invoked during transcription. */
void qwen_set_token_callback(qwen_ctx_t *ctx, qwen_token_cb cb, void *userdata);

/* Streaming only. See qwen_partial_cb. */
void qwen_set_partial_callback(qwen_ctx_t *ctx, qwen_partial_cb cb, void *userdata);

/* See qwen_encoder_hook. Applies to every path that runs the audio tower. */
void qwen_set_encoder_hook(qwen_ctx_t *ctx, qwen_encoder_hook fn, void *userdata);

/* See qwen_decoder_hook. Used by the interactive streaming loop only. */
void qwen_set_decoder_hook(qwen_ctx_t *ctx, qwen_decoder_hook fn, void *userdata);

/* Set optional system prompt text (UTF-8). Pass NULL or "" to clear.
 * Returns 0 on success, -1 on allocation/encoding errors. */
int qwen_set_prompt(qwen_ctx_t *ctx, const char *prompt);

/* Set optional forced language. Pass NULL or "" to clear.
 * Returns 0 on success, -1 if language is unsupported. */
int qwen_set_force_language(qwen_ctx_t *ctx, const char *language);

/* Comma-separated supported language names for --language. */
const char *qwen_supported_languages_csv(void);

/* Transcribe a WAV file, returns allocated string (caller must free) */
char *qwen_transcribe(qwen_ctx_t *ctx, const char *wav_path);

/* Transcribe from raw audio samples (mono float32, 16kHz) */
char *qwen_transcribe_audio(qwen_ctx_t *ctx, const float *samples, int n_samples);

/* Transcribe from stdin (auto-detect WAV or raw s16le) */
char *qwen_transcribe_stdin(qwen_ctx_t *ctx);

/* Streaming transcription: process audio in chunks with prefix rollback.
 * Re-encodes growing audio and uses previous text as decoder context.
 * Tokens are emitted via the token callback as they become "fixed". */
char *qwen_transcribe_stream(qwen_ctx_t *ctx, const float *samples, int n_samples);

/* Live streaming transcription from an incrementally-filled audio source.
 * The streaming loop waits for new data instead of terminating at EOF.
 * Tokens are emitted via the token callback as they become "fixed". */
char *qwen_transcribe_stream_live(qwen_ctx_t *ctx, qwen_live_audio_t *live);

/* ========================================================================
 * Alternative decoder backends
 *
 * Everything up to the decoder (mel, encoder, prompt/audio embedding assembly)
 * is backend independent. A different decoder implementation — the WebGPU one
 * under wasm/webgpu/ — consumes the embeddings this produces and drives its own
 * prefill + generation loop.
 * ======================================================================== */

#include "qwen_asr_tokenizer.h"

/* Assemble the decoder's input embeddings around an encoder output computed
 * elsewhere - the WebGPU audio tower runs on the GPU and hands its result
 * back here. enc_output stays owned by the caller; the return is malloc'd. */
float *qwen_assemble_embeds(qwen_ctx_t *ctx, qwen_tokenizer_t *tokenizer,
                            const float *enc_output, int enc_seq_len,
                            const int *past_tokens, int n_past_tokens,
                            int *out_seq);

/* Build the decoder input embeddings for one utterance.
 * Returns a malloc'd [*out_seq_len, dec_hidden] f32 buffer; caller frees.
 * Also fills mel/encoder timings when the pointers are non-NULL. */
float *qwen_build_embeds(qwen_ctx_t *ctx, const float *samples, int n_samples,
                         int *out_seq_len, double *out_mel_ms, double *out_enc_ms);

/* ========================================================================
 * Internal Functions
 * ======================================================================== */

/* Audio encoder forward pass */
float *qwen_encoder_forward(qwen_ctx_t *ctx, const float *mel, int mel_frames,
                             int *out_seq_len);

/* Decoder prefill (multiple tokens) */
void qwen_decoder_prefill(qwen_ctx_t *ctx, const float *input_embeds, int seq_len);

/* ---- Quantization calibration ----
 *
 * qwen_calib_begin() attaches a per-input-channel activation accumulator to
 * every quantized decoder matrix; ordinary transcription then fills it in.
 * qwen_calib_write() dumps it. The dump stores sums rather than means so that
 * runs over separate audio files merge by addition. Returns 0 on success. */
int qwen_calib_begin(qwen_ctx_t *ctx);
int qwen_calib_write(const qwen_ctx_t *ctx, const char *path);

/* Rank every quantized matrix by the output error four bits would cost it,
 * weighted by the activations recorded in `path`. Writes a TSV table to
 * stdout. Reads the weights as currently loaded, so run it on a Q8 model. */
int qwen_calib_rank(const qwen_ctx_t *ctx, const char *path);

/* Search the AWQ channel-scaling exponent per group of matrices that share an
 * input, and report what it saves. Analysis only - it changes no weights. */
int qwen_awq_search(const qwen_ctx_t *ctx, const char *path);

/* AWQ channel scales derived from a statistics dump, for the decoder load path.
 *
 * Set qwen_awq_path (and optionally qwen_awq_alpha) before qwen_load(); the
 * decoder picks them up while narrowing weights to four bits, and folds the
 * matching input-side division into the preceding norm or into the matrix that
 * produced the activation, so inference itself is unchanged. Only meaningful
 * with QWEN_WEIGHTS_Q4. QWEN_AWQ / QWEN_AWQ_ALPHA override both. */
extern const char *qwen_awq_path;
extern double qwen_awq_alpha;

typedef struct qwen_awq qwen_awq_t;
qwen_awq_t *qwen_awq_open(const char *path, double alpha);
void qwen_awq_close(qwen_awq_t *a);

/* Scale vector for the matrix group recorded under `name`, or NULL when the
 * dump has no matching entry of that width. Valid until the next call. */
const float *qwen_awq_scales(qwen_awq_t *a, const char *name, int cols);

/* ---------------------------------------------------------------------------
 * Batched decoding
 *
 * Generating one token streams every decoder weight exactly once, so a single
 * stream runs at the memory wall with the arithmetic units mostly idle.
 * Independent utterances - the segments of a long recording, which under
 * --past-text no do not condition on each other - can share that sweep: N
 * streams cost roughly one stream's bandwidth.
 *
 * Each stream owns its KV cache; qwen_kv_bind() lends one to the context so
 * the ordinary prefill path can fill it.
 * ------------------------------------------------------------------------- */

typedef struct {
    qwen_f16_t *k;             /* [layers, max, kv_heads * head_dim], f16 */
    qwen_f16_t *v;
    int len;
    int max;
} qwen_kv_t;

/* Streams the batched decoder can advance in one sweep. */
#define QWEN_MAX_BATCH 16

/* Rows a combined prefill will stack before splitting into another call. */
#define QWEN_PREFILL_ROW_CAP 2048

qwen_kv_t *qwen_kv_create(qwen_ctx_t *ctx, int max_seq);
void qwen_kv_free(qwen_kv_t *kv);

/* Lend a stream's cache to the context, run qwen_decoder_prefill(), then take
 * it back - prefill may have grown it. */
void qwen_kv_bind(qwen_ctx_t *ctx, qwen_kv_t *kv);
void qwen_kv_unbind(qwen_ctx_t *ctx, qwen_kv_t *kv);

/* Prefill n empty streams in one pass; embeds[i] is [lens[i]][dec_hidden].
 * Cheaper than n separate prefills: the Q8 weight dequantize sweep is paid
 * once. Returns 0 on success. */
int qwen_decoder_prefill_multi(qwen_ctx_t *ctx, qwen_kv_t **kvs,
                               const float *const *embeds, const int *lens, int n);

/* Advance n streams by one token. embeds is [n][dec_hidden] in the same order
 * as kvs; out_tokens receives n ids. Returns 0, or -1 if n exceeds
 * QWEN_MAX_BATCH. */
int qwen_decoder_forward_batch(qwen_ctx_t *ctx, qwen_kv_t **kvs, int n,
                               const float *embeds, int *out_tokens);

/* Decoder forward (single token, uses KV cache, returns greedy token) */
int qwen_decoder_forward(qwen_ctx_t *ctx, const float *input_embed);

/* Test hook for the WebGPU encoder harness.
 *
 * The audio tower is two very different halves - a Conv2D stem and a stack of
 * transformer layers - and porting them to the GPU one at a time needs
 * references for the boundaries between them. When qwen_enc_tap is non-zero,
 * qwen_encoder_forward() leaves copies of:
 *
 *   qwen_enc_tap_mel   [128][qwen_enc_tap_frames]  its mel input
 *   qwen_enc_tap_conv  [qwen_enc_tap_tokens][enc_d_model]
 *                      the stem's output, after the projection and position
 *                      embeddings and before the first transformer layer
 *   qwen_enc_tap_out   [qwen_enc_tap_tokens][enc_output_dim]  its own return
 *
 * The buffers are owned by the encoder and replaced on the next call. */
extern int qwen_enc_tap;
extern float *qwen_enc_tap_mel;
extern int qwen_enc_tap_frames;
extern float *qwen_enc_tap_conv;
extern float *qwen_enc_tap_out;
extern int qwen_enc_tap_tokens;

/* Global verbose flag */
extern int qwen_verbose;

/* Monitor mode: show inline Unicode symbols on stderr for streaming diagnostics.
 * Symbols: ▶ encoder  · prefill  ▪ decode  ▸ slow decode  ⟳ window eviction */
extern int qwen_monitor;

#endif /* QWEN_ASR_H */
