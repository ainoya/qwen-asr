/*
 * qwen_asr_kernels_generic.c - architecture-generic hot kernels
 */

#include "qwen_asr_kernels_impl.h"

#include <string.h>

void qwen_bf16_matvec_fused_generic(float *y, const float *x, const uint16_t *W_bf16,
                                    const float *bias, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * in_dim;
        float sum = bias ? bias[o] : 0.0f;
        for (int k = 0; k < in_dim; k++) {
            uint32_t f32_bits = ((uint32_t)w_row[k]) << 16;
            float w_val;
            memcpy(&w_val, &f32_bits, sizeof(float));
            sum += w_val * x[k];
        }
        y[o] = sum;
    }
}

void qwen_argmax_bf16_range_generic(const float *x, const uint16_t *W_bf16,
                                    int in_dim, int start, int end,
                                    int *best_out, float *best_val_out) {
    int best = start;
    float best_val = -1e30f;

    for (int o = start; o < end; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * in_dim;
        float sum = 0.0f;
        for (int k = 0; k < in_dim; k++) {
            uint32_t f32_bits = ((uint32_t)w_row[k]) << 16;
            float w_val;
            memcpy(&w_val, &f32_bits, sizeof(float));
            sum += w_val * x[k];
        }
        if (sum > best_val) {
            best_val = sum;
            best = o;
        }
    }

    *best_out = best;
    *best_val_out = best_val;
}

float qwen_dot_f32_generic(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

void qwen_vec_scale_inplace_generic(float *dst, float scale, int n) {
    for (int i = 0; i < n; i++) dst[i] *= scale;
}

void qwen_vec_axpy_inplace_generic(float *dst, const float *src, float alpha, int n) {
    for (int i = 0; i < n; i++) dst[i] += alpha * src[i];
}

void qwen_vec_scale_add_generic(float *dst, const float *src, float correction, int n) {
    for (int i = 0; i < n; i++) dst[i] = dst[i] * correction + src[i];
}

/* ========================================================================
 * Q8 block-quantized kernels (portable reference)
 * ======================================================================== */

#define Q8B 64

void qwen_q8_quantize_row_generic(const float *x, int8_t *qx, float *sx, int n) {
    for (int b = 0; b < n; b += Q8B) {
        float amax = 0.0f;
        for (int i = 0; i < Q8B; i++) {
            float a = x[b + i] < 0.0f ? -x[b + i] : x[b + i];
            if (a > amax) amax = a;
        }
        float scale = amax / 127.0f;
        float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        sx[b / Q8B] = scale;
        for (int i = 0; i < Q8B; i++) {
            float v = x[b + i] * inv;
            int q = (int)(v < 0.0f ? v - 0.5f : v + 0.5f);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            qx[b + i] = (int8_t)q;
        }
    }
}

void qwen_q8_matvec_generic(float *y, const int8_t *qx, const float *sx,
                            const int8_t *W, const float *ws,
                            int in_dim, int rows) {
    int nb = in_dim / Q8B;
    for (int o = 0; o < rows; o++) {
        const int8_t *w = W + (size_t)o * in_dim;
        const float *s = ws + (size_t)o * nb;
        float sum = 0.0f;
        for (int b = 0; b < nb; b++) {
            int32_t acc = 0;
            const int8_t *wb = w + b * Q8B;
            const int8_t *xb = qx + b * Q8B;
            for (int i = 0; i < Q8B; i++) acc += (int32_t)wb[i] * (int32_t)xb[i];
            sum += (float)acc * s[b] * sx[b];
        }
        y[o] = sum;
    }
}

void qwen_q8_argmax_range_generic(const int8_t *qx, const float *sx,
                                  const int8_t *W, const float *ws,
                                  int in_dim, int rows, int row_base,
                                  int *best_out, float *best_val_out) {
    int nb = in_dim / Q8B;
    int best = row_base;
    float best_val = -1e30f;
    for (int o = 0; o < rows; o++) {
        const int8_t *w = W + (size_t)o * in_dim;
        const float *s = ws + (size_t)o * nb;
        float sum = 0.0f;
        for (int b = 0; b < nb; b++) {
            int32_t acc = 0;
            const int8_t *wb = w + b * Q8B;
            const int8_t *xb = qx + b * Q8B;
            for (int i = 0; i < Q8B; i++) acc += (int32_t)wb[i] * (int32_t)xb[i];
            sum += (float)acc * s[b] * sx[b];
        }
        if (sum > best_val) { best_val = sum; best = row_base + o; }
    }
    *best_out = best;
    *best_val_out = best_val;
}


/* Batched Q8 matvec: m activation vectors against one weight matrix.
 *
 * The point is that the weight row is read once and reused for all m outputs.
 * The alternative for a short sequence - dequantize the matrix into a panel and
 * call sgemm - moves ~9 bytes per weight instead of 1.06, which is a bad trade
 * until the sequence is long enough for the GEMM efficiency to pay for it. */
/* Batched Q8 matvec. M must be a compile-time constant so the accumulators
 * stay in registers and the vectorizer can see a fixed trip count; with a
 * runtime count the array is indexed dynamically and lands on the stack. */
#define Q8_MATVEC_M_KERNEL(NAME, M)                                            \
static void NAME(float *y, int ldy, const int8_t *qx, const float *sx,         \
                 const int8_t *W, const float *ws, int in_dim, int rows) {     \
    int nb = in_dim / Q8B;                                                     \
    for (int o = 0; o < rows; o++) {                                           \
        const int8_t *w = W + (size_t)o * in_dim;                              \
        const float *s = ws + (size_t)o * nb;                                  \
        float acc[M];                                                          \
        for (int r = 0; r < M; r++) acc[r] = 0.0f;                             \
        for (int b = 0; b < nb; b++) {                                         \
            const int8_t *wb = w + b * Q8B;                                    \
            float sc = s[b];                                                   \
            for (int r = 0; r < M; r++) {                                      \
                const int8_t *xb = qx + (size_t)r * in_dim + b * Q8B;          \
                int32_t d = 0;                                                 \
                for (int i = 0; i < Q8B; i++) d += (int32_t)wb[i] * (int32_t)xb[i]; \
                acc[r] += (float)d * sc * sx[(size_t)r * nb + b];              \
            }                                                                  \
        }                                                                      \
        for (int r = 0; r < M; r++) y[(size_t)r * ldy + o] = acc[r];           \
    }                                                                          \
}

Q8_MATVEC_M_KERNEL(q8_mv_m1, 1)
Q8_MATVEC_M_KERNEL(q8_mv_m2, 2)
Q8_MATVEC_M_KERNEL(q8_mv_m4, 4)
Q8_MATVEC_M_KERNEL(q8_mv_m8, 8)

void qwen_q8_matvec_m_generic(float *y, int ldy, const int8_t *qx, const float *sx,
                              int m, const int8_t *W, const float *ws,
                              int in_dim, int rows) {
    int nb = in_dim / Q8B;
    int off = 0;
    while (off < m) {
        int take = m - off;
        const int8_t *x = qx + (size_t)off * in_dim;
        const float *xs = sx + (size_t)off * nb;
        float *dst = y + (size_t)off * ldy;
        if (take >= 8)      { q8_mv_m8(dst, ldy, x, xs, W, ws, in_dim, rows); take = 8; }
        else if (take >= 4) { q8_mv_m4(dst, ldy, x, xs, W, ws, in_dim, rows); take = 4; }
        else if (take >= 2) { q8_mv_m2(dst, ldy, x, xs, W, ws, in_dim, rows); take = 2; }
        else                { q8_mv_m1(dst, ldy, x, xs, W, ws, in_dim, rows); take = 1; }
        off += take;
    }
}

/* 4-bit weights. Byte j of a block carries value j in its low nibble and
 * value j+32 in its high, both biased by 8. */
void qwen_q4_matvec_generic(float *y, const int8_t *qx, const float *sx,
                            const int8_t *W, const float *ws,
                            int in_dim, int rows) {
    int nb = in_dim / Q8B;
    int half = Q8B / 2;
    for (int o = 0; o < rows; o++) {
        const unsigned char *w = (const unsigned char *)W + (size_t)o * (in_dim / 2);
        const float *s = ws + (size_t)o * nb;
        float acc = 0.0f;
        for (int b = 0; b < nb; b++) {
            const unsigned char *wb = w + (size_t)b * half;
            const int8_t *xb = qx + (size_t)b * Q8B;
            int32_t d = 0;
            for (int j = 0; j < half; j++) {
                d += ((int32_t)(wb[j] & 0x0F) - 8) * (int32_t)xb[j];
                d += ((int32_t)(wb[j] >> 4) - 8) * (int32_t)xb[j + half];
            }
            acc += (float)d * s[b] * sx[b];
        }
        y[o] = acc;
    }
}
