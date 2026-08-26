/*
 * qwen_asr_kernels.h - Math kernels for Qwen3-ASR inference
 *
 * Low-level math operations. All operate on float32 tensors in row-major order.
 * Adapted from voxtral-realtime project.
 */

#ifndef QWEN_ASR_KERNELS_H
#define QWEN_ASR_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/* ========================================================================
 * Basic Operations
 * ======================================================================== */

void qwen_add_inplace(float *a, const float *b, int n);
void qwen_mul_inplace(float *a, const float *b, int n);
void qwen_scale(float *x, float s, int n);
void qwen_copy(float *dst, const float *src, int n);

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

/* C = A @ B^T: A[M,K], B[N,K], C[M,N] */
void qwen_matmul_t(float *C, const float *A, const float *B, int M, int K, int N);

/* y = x @ W^T + b: x[seq,in], W[out,in], b[out], y[seq,out] */
void qwen_linear(float *y, const float *x, const float *W, const float *b,
                 int seq_len, int in_dim, int out_dim);

void qwen_linear_nobias(float *y, const float *x, const float *W,
                         int seq_len, int in_dim, int out_dim);

/* bf16 weight variants */
void qwen_linear_bf16(float *y, const float *x, const uint16_t *W_bf16,
                      const float *b, int seq_len, int in_dim, int out_dim);

void qwen_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                              int seq_len, int in_dim, int out_dim);

/* seq=1 decoder fast path: compute Q/K/V matvecs with one threaded dispatch */
void qwen_linear_nobias_bf16_qkv(float *q, float *k, float *v, const float *x,
                                 const uint16_t *Wq_bf16,
                                 const uint16_t *Wk_bf16,
                                 const uint16_t *Wv_bf16,
                                 int in_dim, int q_dim, int kv_dim);

void qwen_matmul_t_bf16(float *C, const float *A, const uint16_t *B_bf16,
                         int M, int K, int N);

/* ========================================================================
 * Q8 Block-Quantized Weights
 *
 * Decoder token generation streams every decoder weight from RAM once per
 * token, so it is memory-bandwidth bound rather than compute bound.  Storing
 * the weights as blocks of QWEN_Q8_BLOCK int8 values plus one f32 scale cuts
 * the bytes per weight from 2.0 (bf16) to 1.0625, and the dot products run on
 * the integer SDOT/VPDPBUSD units (or i32x4.dot_i16x8 under wasm).
 * ======================================================================== */

#define QWEN_Q8_BLOCK 64

/* Sequence lengths at or below this use a batched matvec instead of the
 * dequantize-into-a-panel path. Override with QWEN_Q8_BATCH_MAX at build time. */
#ifndef QWEN_Q8_BATCH_MAX
#define QWEN_Q8_BATCH_MAX 256
#endif

/* Override the batched-matvec threshold at runtime; -1 restores the default.
 * See the note in qwen_asr_kernels.c - the two paths are not numerically
 * equivalent, so this is how a caller asks for f32 activations throughout. */
void qwen_set_q8_batch_max(int n);

/* Activation rows handled per pass by the batched kernel (register budget). */
#define QWEN_Q8_GROUP 16


/* Block-quantized weights, 8 or 4 bits per value.
 *
 * At 4 bits a block of QWEN_Q8_BLOCK values is 32 bytes plus its f32 scale:
 * 0.5625 B/weight against Q8's 1.0625. Token generation reads every decoder
 * weight once, so that is close to a direct cut in decode time - which is why
 * it is worth the accuracy cost, and why the tied LM head is left at 8 bits
 * where it decides the token.
 *
 * Nibbles are packed so one 16-byte load yields two SIMD vectors: within a
 * block, byte j holds value j in its low nibble and value j+32 in its high
 * nibble. Values are stored biased by 8, so a raw nibble of 0..15 means
 * -8..7 and the dot product needs no correction term. */
/* Per-input-channel activation magnitude, summed over forward passes.
 *
 * Weight quantization error only matters in proportion to the activation it
 * multiplies: a channel the model barely drives can be rounded hard, while a
 * channel carrying large values needs its precision. Both AWQ-style scaling
 * and any per-layer precision choice need this measured on real audio, so the
 * accumulator hangs off the weight matrix and fills in during ordinary
 * transcription runs. Sums (not means) are stored so dumps from separate runs
 * merge by addition. */
typedef struct {
    double *absmean;  /* [cols] sum of |x|        */
    double *sqmean;   /* [cols] sum of x*x        */
    float  *absmax;   /* [cols] largest |x| seen  */
    double  rows;     /* activation rows observed */
} qwen_act_stats_t;

typedef struct {
    int8_t *q;        /* [rows * cols] int8, or [rows * cols / 2] nibbles */
    float  *scales;   /* [rows * cols / QWEN_Q8_BLOCK]           */
    int rows;
    int cols;         /* must be a multiple of QWEN_Q8_BLOCK     */
    int owns;         /* 0 when q/scales point into a mapped model file */
    int bits;         /* 4, or anything else for 8               */
    qwen_act_stats_t *stats;  /* non-NULL only while calibrating     */
} qwen_q8_mat_t;

/* True when the matrix is stored as nibbles. */
#define QWEN_IS_Q4(m) ((m)->bits == 4)

/* Start/stop accumulating input activation statistics for this matrix. */
int  qwen_act_stats_attach(qwen_q8_mat_t *m);
void qwen_act_stats_free(qwen_q8_mat_t *m);

/* Fold seq_len rows of [seq_len][m->cols] activations into the accumulator.
 * A no-op when no accumulator is attached, which is every normal run. */
void qwen_act_stats_observe(const qwen_q8_mat_t *m, const float *x, int seq_len);

/* Point a matrix at pre-quantized data owned by someone else (a packed model
 * file). qwen_q8_free() will not release it. */
void qwen_q8_attach(qwen_q8_mat_t *m, int8_t *q, float *scales, int rows, int cols);

/* Same, for data stored as 4-bit nibbles. `cols` is the real column count, not
 * the halved stored width. */
void qwen_q4_attach(qwen_q8_mat_t *m, int8_t *q, float *scales, int rows, int cols);

/* A weight matrix in whichever representation the model file provided. */
typedef struct {
    float *f32;          /* non-NULL when stored as f32          */
    qwen_q8_mat_t q8;    /* q8.q non-NULL when stored as Q8      */
    int rows;
    int cols;
} qwen_wmat_t;

/* y = x @ W^T (+ bias), dispatching on how W is stored. */
void qwen_linear_w(float *y, const float *x, const qwen_wmat_t *W,
                   const float *bias, int seq_len);

/* y = x @ W^T + bias with a Q8 weight matrix. */
void qwen_linear_bias_q8(float *y, const float *x, const qwen_q8_mat_t *W,
                         const float *bias, int seq_len);

void qwen_wmat_free(qwen_wmat_t *w);

/* Quantize a row-major bf16 weight matrix. Returns 0 on success. */
int qwen_q8_from_bf16(qwen_q8_mat_t *m, const uint16_t *W_bf16, int rows, int cols);

/* Quantize two bf16 matrices into one, interleaving their rows as
 * [A0, B0, A1, B1, ...] (used to fuse the SwiGLU gate/up projections). */
int qwen_q8_from_bf16_interleave2(qwen_q8_mat_t *m, const uint16_t *A,
                                  const uint16_t *B, int rows_each, int cols);

void qwen_q8_free(qwen_q8_mat_t *m);

/* Re-quantize an 8-bit matrix down to 4 bits, in place of `src`. The caller
 * keeps `src` (it may point into a mapped file); `dst` owns its buffers.
 * Requantizing from Q8 rather than the original bf16 costs almost nothing: the
 * block scale is already right and only the 16-level grid matters. */
int qwen_q4_from_q8(qwen_q8_mat_t *dst, const qwen_q8_mat_t *src);

/* Q8 -> Q4, scaling column c by colscale[c] on the way through (AWQ). */
int qwen_q4_from_q8_scaled(qwen_q8_mat_t *dst, const qwen_q8_mat_t *src,
                           const float *colscale);

/* Divide rows row0, row0+stride, ... (n of them) by s[0..n-1], exactly, by
 * scaling the block scales. Folds the input side of AWQ rescaling into the
 * matrix that produced that activation. */
void qwen_q8_scale_rows(qwen_q8_mat_t *m, const float *s, int row0, int stride,
                        int n);

/* Quantize a row-major bf16 weight matrix to 4 bits. */
int qwen_q4_from_bf16(qwen_q8_mat_t *m, const uint16_t *W_bf16, int rows, int cols);

/* Same, fusing two matrices by interleaving rows (see the Q8 version). */
int qwen_q4_from_bf16_interleave2(qwen_q8_mat_t *m, const uint16_t *A,
                                  const uint16_t *B, int rows_each, int cols);

/* Total bytes held by a quantized matrix (for reporting). */
size_t qwen_q8_bytes(const qwen_q8_mat_t *m);

/* Quantize a row of `cols` f32 values to 4-bit blocks and dequantize it back,
 * in place. For the calibration tooling, so that what it measures is the
 * rounding the kernels actually perform. */
void qwen_q4_roundtrip_row(float *w, int cols);

/* Dequantize one row into f32 (used for token embedding lookup). */
void qwen_q8_row_to_f32(float *dst, const qwen_q8_mat_t *m, int row);

/* y = x @ W^T, W quantized. Handles both seq_len==1 (matvec) and prefill. */
void qwen_linear_nobias_q8(float *y, const float *x, const qwen_q8_mat_t *W,
                           int seq_len);

/* seq=1 fast path: Q/K/V matvecs under a single threaded dispatch. */
void qwen_linear_nobias_q8_qkv(float *q, float *k, float *v, const float *x,
                               const qwen_q8_mat_t *Wq, const qwen_q8_mat_t *Wk,
                               const qwen_q8_mat_t *Wv);

/* argmax(W @ x) without materializing logits. */
int qwen_argmax_matvec_q8(const float *x, const qwen_q8_mat_t *W);

/* Same, for m independent hidden states laid out as [m][W->cols]. Writes m
 * token ids to out. Returns -1 if m exceeds QWEN_Q8_MAX_M. */
int qwen_argmax_matvec_q8_batch(const float *x, int m,
                                const qwen_q8_mat_t *W, int *out);

/* ========================================================================
 * 2D Convolution (for audio encoder conv stem)
 * ======================================================================== */

/*
 * 2D Convolution: out = conv2d(in, weight, bias)
 * in: [C_in, H, W]
 * weight: [C_out, C_in, kH, kW]
 * bias: [C_out] (can be NULL)
 * out: [C_out, H_out, W_out]
 * H_out = (H + 2*padding - kH) / stride + 1
 * W_out = (W + 2*padding - kW) / stride + 1
 */
void qwen_conv2d(float *out, const float *in, const float *weight, const float *bias,
                 int c_in, int c_out, int h_in, int w_in,
                 int kh, int kw, int stride, int padding);

/* Same, for `batch` independent images laid out as [channels][batch][h][w].
 * One GEMM covers the whole batch; see the note in qwen_asr_kernels.c. */
void qwen_conv2d_batch(float *out, const float *in, const float *weight,
                       const float *bias, int batch,
                       int c_in, int c_out, int h_in, int w_in,
                       int kh, int kw, int stride, int padding);

/* ========================================================================
 * Normalization
 * ======================================================================== */

/* LayerNorm with bias: out = (x - mean) / sqrt(var + eps) * weight + bias */
void qwen_layer_norm(float *out, const float *x, const float *weight, const float *bias,
                     int seq_len, int hidden, float eps);

/* RMS Normalization: out = x / rms(x) * weight */
void qwen_rms_norm(float *out, const float *x, const float *weight,
                   int seq_len, int hidden, float eps);

/* Per-head RMS Normalization for Q/K norms in decoder
 * x: [seq, n_heads, head_dim], weight: [head_dim]
 * Normalizes each head independently */
void qwen_rms_norm_per_head(float *x, const float *weight,
                             int seq_len, int n_heads, int head_dim, float eps);

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

/* dst[i] = expf(src[i]); vectorized (vForce on Apple, poly elsewhere) */
void qwen_vec_expf(float *dst, const float *src, int n);

void qwen_silu(float *x, int n);
void qwen_gelu(float *x, int n);
void qwen_softmax(float *x, int rows, int cols);
/* out[seq,inter] = SiLU(gate_up[seq,2*inter][:,even]) * gate_up[:,odd] */
void qwen_swiglu_multiply(float *out, const float *gate_up, int seq_len, int intermediate);

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

/*
 * Bidirectional windowed attention (encoder).
 * Q, K, V: [seq, n_heads * head_dim]
 * out: [seq, n_heads * head_dim]
 * window_starts: array of window start positions
 * window_starts[n_windows] = seq (sentinel)
 * All heads have same dimensions (no GQA in encoder).
 */
void qwen_bidirectional_attention(float *out, const float *Q, const float *K,
                                   const float *V, int seq, int n_heads,
                                   int head_dim, float scale,
                                   const int *window_starts, int n_windows);

/*
 * Causal attention with GQA (decoder).
 * Q: [seq_q, n_heads * head_dim]
 * K: [seq_k, n_kv_heads * head_dim]
 * V: [seq_k, n_kv_heads * head_dim]
 * q_offset: global position of first query (for causal mask)
 */
void qwen_causal_attention(float *out, const float *Q, const float *K, const float *V,
                            int seq_q, int seq_k, int n_heads, int n_kv_heads,
                            int head_dim, float scale, int q_offset);

/* ========================================================================
 * Position Embeddings
 * ======================================================================== */

/*
 * Sinusoidal position embeddings (encoder).
 * pe: output [n_pos, d_model]
 * First half = sin, second half = cos.
 */
void qwen_sinusoidal_pe(float *pe, int n_pos, int d_model);

/*
 * NeoX-style RoPE: compute cos/sin for positions.
 * cos_out, sin_out: [seq, head_dim]
 * cos[d] and cos[half+d] are the same (duplicated for full head_dim).
 */
void qwen_compute_rope_neox(float *cos_out, float *sin_out, const int *positions,
                              int seq, int head_dim, float theta);

/*
 * Apply NeoX-style RoPE to Q or K.
 * x: [seq, n_heads * head_dim] (in-place)
 * cos_vals, sin_vals: [seq, head_dim]
 */
void qwen_apply_rope_neox(float *x, const float *cos_vals, const float *sin_vals,
                            int seq, int n_heads, int head_dim);

/* Streaming argmax: finds argmax(W_bf16 @ x) without materializing full logits.
 * Returns the index of the row with highest dot product. */
int qwen_argmax_matvec_bf16(const float *x, const uint16_t *W_bf16,
                             int in_dim, int out_dim);

/* ========================================================================
 * Threading
 * ======================================================================== */

/* Set number of threads for parallel operations (default: 1).
 * Creates a persistent thread pool. Call before inference. */
void qwen_set_threads(int n);

/* Number of threads actually running (may be lower than requested if the
 * platform refused to create them). */
int qwen_get_threads(void);

/* Diagnostic: run `rounds` empty dispatches, report the average number of
 * threads that participated and the total wall time. */
void qwen_pool_selftest(int rounds, int *out_participants, double *out_ms);

/* Get number of available CPU cores */
int qwen_get_num_cpus(void);

/* Global verbose flag */
extern int qwen_verbose;

#endif /* QWEN_ASR_KERNELS_H */
