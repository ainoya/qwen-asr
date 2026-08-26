/*
 * qwen_asr_kernels_neon.c - ARM NEON hot kernels
 */

#include "qwen_asr_kernels_impl.h"

#ifdef __ARM_NEON

#include <arm_neon.h>
#include <string.h>

void qwen_bf16_matvec_fused_neon(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim) {
    int o = 0;

    /* Process 2 output rows at a time, 32 elements/iter, 8 accumulators */
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;
        const uint16_t *w1 = W_bf16 + (size_t)(o + 1) * in_dim;
        float s0 = bias ? bias[o] : 0.0f;
        float s1 = bias ? bias[o + 1] : 0.0f;

        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        float32x4_t b0 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f);
        float32x4_t b2 = vdupq_n_f32(0.0f), b3 = vdupq_n_f32(0.0f);
        int k = 0;

        for (; k + 32 <= in_dim; k += 32) {
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);
            float32x4_t x2 = vld1q_f32(x + k + 8);
            float32x4_t x3 = vld1q_f32(x + k + 12);
            float32x4_t x4 = vld1q_f32(x + k + 16);
            float32x4_t x5 = vld1q_f32(x + k + 20);
            float32x4_t x6 = vld1q_f32(x + k + 24);
            float32x4_t x7 = vld1q_f32(x + k + 28);

            uint16x8_t r0a = vld1q_u16(w0 + k);
            uint16x8_t r0b = vld1q_u16(w0 + k + 8);
            uint16x8_t r0c = vld1q_u16(w0 + k + 16);
            uint16x8_t r0d = vld1q_u16(w0 + k + 24);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0a), 16)), x0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0a), 16)), x1);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0b), 16)), x2);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0b), 16)), x3);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0c), 16)), x4);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0c), 16)), x5);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0d), 16)), x6);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0d), 16)), x7);

            uint16x8_t r1a = vld1q_u16(w1 + k);
            uint16x8_t r1b = vld1q_u16(w1 + k + 8);
            uint16x8_t r1c = vld1q_u16(w1 + k + 16);
            uint16x8_t r1d = vld1q_u16(w1 + k + 24);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1a), 16)), x0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1a), 16)), x1);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1b), 16)), x2);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1b), 16)), x3);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1c), 16)), x4);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1c), 16)), x5);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1d), 16)), x6);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1d), 16)), x7);
        }
        for (; k + 8 <= in_dim; k += 8) {
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);
            uint16x8_t r0 = vld1q_u16(w0 + k);
            uint16x8_t r1 = vld1q_u16(w1 + k);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0), 16)), x0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0), 16)), x1);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1), 16)), x0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1), 16)), x1);
        }
        s0 += vaddvq_f32(vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3)));
        s1 += vaddvq_f32(vaddq_f32(vaddq_f32(b0, b2), vaddq_f32(b1, b3)));

        for (; k < in_dim; k++) {
            uint32_t bits0 = ((uint32_t)w0[k]) << 16;
            uint32_t bits1 = ((uint32_t)w1[k]) << 16;
            float wv0, wv1;
            memcpy(&wv0, &bits0, sizeof(float));
            memcpy(&wv1, &bits1, sizeof(float));
            s0 += wv0 * x[k];
            s1 += wv1 * x[k];
        }
        y[o] = s0;
        y[o + 1] = s1;
    }

    /* Handle remaining odd row */
    for (; o < out_dim; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * in_dim;
        float sum = bias ? bias[o] : 0.0f;
        int k = 0;

        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        for (; k + 8 <= in_dim; k += 8) {
            uint16x8_t bf = vld1q_u16(w_row + k);
            acc0 = vfmaq_f32(acc0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16)),
                             vld1q_f32(x + k));
            acc1 = vfmaq_f32(acc1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16)),
                             vld1q_f32(x + k + 4));
        }
        sum += vaddvq_f32(vaddq_f32(acc0, acc1));

        for (; k < in_dim; k++) {
            uint32_t f32_bits = ((uint32_t)w_row[k]) << 16;
            float w_val;
            memcpy(&w_val, &f32_bits, sizeof(float));
            sum += w_val * x[k];
        }
        y[o] = sum;
    }
}

void qwen_argmax_bf16_range_neon(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out) {
    int best = start;
    float best_val = -1e30f;
    int o = start;

    /* Process 2 rows at a time, 32 elements/iter, 8 accumulators per row */
    for (; o + 1 < end; o += 2) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;
        const uint16_t *w1 = W_bf16 + (size_t)(o + 1) * in_dim;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        float32x4_t b0 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f);
        float32x4_t b2 = vdupq_n_f32(0.0f), b3 = vdupq_n_f32(0.0f);
        int k = 0;

        for (; k + 32 <= in_dim; k += 32) {
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);
            float32x4_t x2 = vld1q_f32(x + k + 8);
            float32x4_t x3 = vld1q_f32(x + k + 12);
            float32x4_t x4 = vld1q_f32(x + k + 16);
            float32x4_t x5 = vld1q_f32(x + k + 20);
            float32x4_t x6 = vld1q_f32(x + k + 24);
            float32x4_t x7 = vld1q_f32(x + k + 28);

            uint16x8_t r0a = vld1q_u16(w0 + k);
            uint16x8_t r0b = vld1q_u16(w0 + k + 8);
            uint16x8_t r0c = vld1q_u16(w0 + k + 16);
            uint16x8_t r0d = vld1q_u16(w0 + k + 24);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0a), 16)), x0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0a), 16)), x1);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0b), 16)), x2);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0b), 16)), x3);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0c), 16)), x4);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0c), 16)), x5);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0d), 16)), x6);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0d), 16)), x7);

            uint16x8_t r1a = vld1q_u16(w1 + k);
            uint16x8_t r1b = vld1q_u16(w1 + k + 8);
            uint16x8_t r1c = vld1q_u16(w1 + k + 16);
            uint16x8_t r1d = vld1q_u16(w1 + k + 24);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1a), 16)), x0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1a), 16)), x1);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1b), 16)), x2);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1b), 16)), x3);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1c), 16)), x4);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1c), 16)), x5);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1d), 16)), x6);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1d), 16)), x7);
        }

        float s0 = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3)));
        float s1 = vaddvq_f32(vaddq_f32(vaddq_f32(b0, b2), vaddq_f32(b1, b3)));

        for (; k < in_dim; k++) {
            uint32_t bits0 = ((uint32_t)w0[k]) << 16;
            uint32_t bits1 = ((uint32_t)w1[k]) << 16;
            float wv0, wv1;
            memcpy(&wv0, &bits0, sizeof(float));
            memcpy(&wv1, &bits1, sizeof(float));
            s0 += wv0 * x[k];
            s1 += wv1 * x[k];
        }

        if (s0 > best_val) { best_val = s0; best = o; }
        if (s1 > best_val) { best_val = s1; best = o + 1; }
    }

    for (; o < end; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * in_dim;
        float sum = 0.0f;
        int k = 0;

        float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
        for (; k + 8 <= in_dim; k += 8) {
            uint16x8_t bf = vld1q_u16(w_row + k);
            acc0 = vfmaq_f32(acc0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16)),
                             vld1q_f32(x + k));
            acc1 = vfmaq_f32(acc1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16)),
                             vld1q_f32(x + k + 4));
        }
        sum += vaddvq_f32(vaddq_f32(acc0, acc1));

        for (; k < in_dim; k++) {
            uint32_t f32_bits = ((uint32_t)w_row[k]) << 16;
            float w_val;
            memcpy(&w_val, &f32_bits, sizeof(float));
            sum += w_val * x[k];
        }
        if (sum > best_val) { best_val = sum; best = o; }
    }

    *best_out = best;
    *best_val_out = best_val;
}

float qwen_dot_f32_neon(const float *a, const float *b, int n) {
    int i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

void qwen_vec_scale_inplace_neon(float *dst, float scale, int n) {
    int i = 0;
    float32x4_t s = vdupq_n_f32(scale);
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        vst1q_f32(dst + i, vfmaq_f32(vdupq_n_f32(0.0f), d0, s));
        vst1q_f32(dst + i + 4, vfmaq_f32(vdupq_n_f32(0.0f), d1, s));
    }
    for (; i < n; i++) dst[i] *= scale;
}

void qwen_vec_axpy_inplace_neon(float *dst, const float *src, float alpha, int n) {
    int i = 0;
    float32x4_t a = vdupq_n_f32(alpha);
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t s0 = vld1q_f32(src + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        float32x4_t s1 = vld1q_f32(src + i + 4);
        vst1q_f32(dst + i, vfmaq_f32(d0, s0, a));
        vst1q_f32(dst + i + 4, vfmaq_f32(d1, s1, a));
    }
    for (; i < n; i++) dst[i] += alpha * src[i];
}

void qwen_vec_scale_add_neon(float *dst, const float *src, float correction, int n) {
    int i = 0;
    float32x4_t c = vdupq_n_f32(correction);
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t s0 = vld1q_f32(src + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        float32x4_t s1 = vld1q_f32(src + i + 4);
        vst1q_f32(dst + i, vfmaq_f32(s0, d0, c));
        vst1q_f32(dst + i + 4, vfmaq_f32(s1, d1, c));
    }
    for (; i < n; i++) dst[i] = dst[i] * correction + src[i];
}


/* ========================================================================
 * Q8 block-quantized kernels
 *
 * One block is 64 int8 weights sharing one f32 scale. SDOT consumes 16 int8
 * pairs per instruction, so a block is 4 SDOTs; four blocks are reduced
 * together with pairwise adds so the scale multiply happens once per 4 blocks.
 * ======================================================================== */

#define Q8B 64

void qwen_q8_quantize_row_neon(const float *x, int8_t *qx, float *sx, int n) {
    for (int b = 0; b < n; b += Q8B) {
        const float *xb = x + b;
        float32x4_t amaxv = vdupq_n_f32(0.0f);
        for (int i = 0; i < Q8B; i += 4)
            amaxv = vmaxq_f32(amaxv, vabsq_f32(vld1q_f32(xb + i)));
        float amax = vmaxvq_f32(amaxv);

        float scale = amax / 127.0f;
        float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        sx[b / Q8B] = scale;

        float32x4_t invv = vdupq_n_f32(inv);
        for (int i = 0; i < Q8B; i += 16) {
            int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xb + i), invv));
            int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xb + i + 4), invv));
            int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xb + i + 8), invv));
            int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xb + i + 12), invv));
            int16x8_t s0 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
            int16x8_t s1 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
            vst1q_s8(qx + b + i, vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1)));
        }
    }
}

#ifdef __ARM_FEATURE_DOTPROD

/* Dot one row of quantized weights against the quantized activation. */
static inline float q8_row_dot(const int8_t *w, const float *s,
                               const int8_t *qx, const float *sx, int nb) {
    float32x4_t accf = vdupq_n_f32(0.0f);
    int b = 0;
    for (; b + 4 <= nb; b += 4) {
        const int8_t *wp = w + (size_t)b * Q8B;
        const int8_t *xp = qx + (size_t)b * Q8B;
        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
        int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
        for (int i = 0; i < Q8B; i += 16) {
            a0 = vdotq_s32(a0, vld1q_s8(wp + i), vld1q_s8(xp + i));
            a1 = vdotq_s32(a1, vld1q_s8(wp + Q8B + i), vld1q_s8(xp + Q8B + i));
            a2 = vdotq_s32(a2, vld1q_s8(wp + 2 * Q8B + i), vld1q_s8(xp + 2 * Q8B + i));
            a3 = vdotq_s32(a3, vld1q_s8(wp + 3 * Q8B + i), vld1q_s8(xp + 3 * Q8B + i));
        }
        /* [sum(a0), sum(a1), sum(a2), sum(a3)] */
        int32x4_t sums = vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3));
        float32x4_t sc = vmulq_f32(vld1q_f32(s + b), vld1q_f32(sx + b));
        accf = vfmaq_f32(accf, vcvtq_f32_s32(sums), sc);
    }
    float sum = vaddvq_f32(accf);
    for (; b < nb; b++) {
        const int8_t *wp = w + (size_t)b * Q8B;
        const int8_t *xp = qx + (size_t)b * Q8B;
        int32x4_t a = vdupq_n_s32(0);
        for (int i = 0; i < Q8B; i += 16)
            a = vdotq_s32(a, vld1q_s8(wp + i), vld1q_s8(xp + i));
        sum += (float)vaddvq_s32(a) * s[b] * sx[b];
    }
    return sum;
}

#else /* NEON without the dot-product extension */

static inline float q8_row_dot(const int8_t *w, const float *s,
                               const int8_t *qx, const float *sx, int nb) {
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const int8_t *wp = w + (size_t)b * Q8B;
        const int8_t *xp = qx + (size_t)b * Q8B;
        int32x4_t a = vdupq_n_s32(0);
        for (int i = 0; i < Q8B; i += 16) {
            int8x16_t wv = vld1q_s8(wp + i);
            int8x16_t xv = vld1q_s8(xp + i);
            int16x8_t lo = vmull_s8(vget_low_s8(wv), vget_low_s8(xv));
            int16x8_t hi = vmull_s8(vget_high_s8(wv), vget_high_s8(xv));
            a = vpadalq_s16(a, lo);
            a = vpadalq_s16(a, hi);
        }
        sum += (float)vaddvq_s32(a) * s[b] * sx[b];
    }
    return sum;
}

#endif

void qwen_q8_matvec_neon(float *y, const int8_t *qx, const float *sx,
                         const int8_t *W, const float *ws,
                         int in_dim, int rows) {
    int nb = in_dim / Q8B;
    for (int o = 0; o < rows; o++)
        y[o] = q8_row_dot(W + (size_t)o * in_dim, ws + (size_t)o * nb, qx, sx, nb);
}

void qwen_q8_argmax_range_neon(const int8_t *qx, const float *sx,
                               const int8_t *W, const float *ws,
                               int in_dim, int rows, int row_base,
                               int *best_out, float *best_val_out) {
    int nb = in_dim / Q8B;
    int best = row_base;
    float best_val = -1e30f;
    for (int o = 0; o < rows; o++) {
        float sum = q8_row_dot(W + (size_t)o * in_dim, ws + (size_t)o * nb, qx, sx, nb);
        if (sum > best_val) { best_val = sum; best = row_base + o; }
    }
    *best_out = best;
    *best_val_out = best_val;
}




/* Batched Q8 matvec: one weight row load feeds m activation vectors. See the
 * generic version for why this beats dequantize-into-a-panel for short
 * sequences. */
/* Batched Q8 matvec: one pass over the weight rows scores M activation rows.
 *
 * M has to be a compile-time constant. With a runtime row count the
 * accumulator array is indexed dynamically, so it lives on the stack instead
 * of in registers and the kernel spends its time spilling - measured at about
 * a quarter of the throughput of the specialised versions below. */
#define Q8_MATVEC_M_KERNEL(NAME, M)                                            \
static void NAME(float *y, int ldy, const int8_t *qx, const float *sx,         \
                 const int8_t *W, const float *ws, int in_dim, int rows) {     \
    int nb = in_dim / Q8B;                                                     \
    for (int o = 0; o < rows; o++) {                                           \
        const int8_t *w = W + (size_t)o * in_dim;                              \
        const float *s = ws + (size_t)o * nb;                                  \
        float32x4_t facc[M];                                                   \
        for (int r = 0; r < M; r++) facc[r] = vdupq_n_f32(0.0f);                \
        for (int b = 0; b < nb; b++) {                                         \
            const int8_t *wp = w + (size_t)b * Q8B;                            \
            int8x16_t w0 = vld1q_s8(wp);                                       \
            int8x16_t w1 = vld1q_s8(wp + 16);                                  \
            int8x16_t w2 = vld1q_s8(wp + 32);                                  \
            int8x16_t w3 = vld1q_s8(wp + 48);                                  \
            float sc = s[b];                                                   \
            for (int r = 0; r < M; r++) {                                      \
                const int8_t *xp = qx + (size_t)r * in_dim + (size_t)b * Q8B;  \
                int32x4_t a0 = vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(xp));    \
                int32x4_t a1 = vdotq_s32(vdupq_n_s32(0), w1, vld1q_s8(xp + 16)); \
                a0 = vdotq_s32(a0, w2, vld1q_s8(xp + 32));                     \
                a1 = vdotq_s32(a1, w3, vld1q_s8(xp + 48));                     \
                facc[r] = vfmaq_n_f32(facc[r], vcvtq_f32_s32(vaddq_s32(a0, a1)), \
                                      sc * sx[(size_t)r * nb + b]);            \
            }                                                                  \
        }                                                                      \
        for (int r = 0; r < M; r++)                                            \
            y[(size_t)r * ldy + o] = vaddvq_f32(facc[r]);                      \
    }                                                                          \
}

#ifdef __ARM_FEATURE_DOTPROD
Q8_MATVEC_M_KERNEL(q8_mv_m1, 1)
Q8_MATVEC_M_KERNEL(q8_mv_m2, 2)
Q8_MATVEC_M_KERNEL(q8_mv_m4, 4)
Q8_MATVEC_M_KERNEL(q8_mv_m8, 8)
#endif

void qwen_q8_matvec_m_neon(float *y, int ldy, const int8_t *qx, const float *sx,
                           int m, const int8_t *W, const float *ws,
                           int in_dim, int rows) {
    int nb = in_dim / Q8B;
#ifdef __ARM_FEATURE_DOTPROD
    int off = 0;
    while (m - off > 0) {
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
#else
    for (int o = 0; o < rows; o++) {
        const int8_t *w = W + (size_t)o * in_dim;
        const float *s = ws + (size_t)o * nb;
        for (int r = 0; r < m; r++) {
            float acc = 0.0f;
            for (int b = 0; b < nb; b++) {
                int32_t d = 0;
                for (int i = 0; i < Q8B; i++)
                    d += (int32_t)w[(size_t)b * Q8B + i] *
                         (int32_t)qx[(size_t)r * in_dim + (size_t)b * Q8B + i];
                acc += (float)d * s[b] * sx[(size_t)r * nb + b];
            }
            y[(size_t)r * ldy + o] = acc;
        }
    }
#endif
}

#endif /* __ARM_NEON */

/* 4-bit weights: one 16-byte load unpacks into two int8x16 vectors, so the
 * weight traffic halves while the SDOT count stays the same. That is the whole
 * point - generation reads every weight once per token and is bound by those
 * bytes, not by the arithmetic. */
void qwen_q4_matvec_neon(float *y, const int8_t *qx, const float *sx,
                         const int8_t *W, const float *ws,
                         int in_dim, int rows) {
    int nb = in_dim / Q8B;
    int half = Q8B / 2;
#ifdef __ARM_FEATURE_DOTPROD
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int8x16_t bias = vdupq_n_s8(8);
#endif
    for (int o = 0; o < rows; o++) {
        const int8_t *w = W + (size_t)o * (in_dim / 2);
        const float *s = ws + (size_t)o * nb;
#ifdef __ARM_FEATURE_DOTPROD
        float32x4_t accf = vdupq_n_f32(0.0f);
        for (int b = 0; b < nb; b++) {
            const int8_t *wb = w + (size_t)b * half;
            const int8_t *xb = qx + (size_t)b * Q8B;
            int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
            for (int j = 0; j < half; j += 16) {
                uint8x16_t raw = vld1q_u8((const uint8_t *)(wb + j));
                int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), bias);
                int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), bias);
                a0 = vdotq_s32(a0, lo, vld1q_s8(xb + j));
                a1 = vdotq_s32(a1, hi, vld1q_s8(xb + half + j));
            }
            float32x4_t f = vcvtq_f32_s32(vaddq_s32(a0, a1));
            accf = vfmaq_n_f32(accf, f, s[b] * sx[b]);
        }
        y[o] = vaddvq_f32(accf);
#else
        float acc = 0.0f;
        for (int b = 0; b < nb; b++) {
            const unsigned char *wb = (const unsigned char *)w + (size_t)b * half;
            const int8_t *xb = qx + (size_t)b * Q8B;
            int32_t d = 0;
            for (int j = 0; j < half; j++) {
                d += ((int32_t)(wb[j] & 0x0F) - 8) * (int32_t)xb[j];
                d += ((int32_t)(wb[j] >> 4) - 8) * (int32_t)xb[j + half];
            }
            acc += (float)d * s[b] * sx[b];
        }
        y[o] = acc;
#endif
    }
}

/* 4-bit weights against M activation rows, NEON.
 *
 * The nibble unpack is the expensive part per weight byte, so it is hoisted out
 * of the row loop: each 16-byte load becomes two int8x16 vectors that stay in
 * registers while all M rows dot against them. At M=4 that is 4 unpacked
 * vectors plus 4 accumulators plus the activation loads, which still fits. */
#ifdef __ARM_FEATURE_DOTPROD
#define Q4_MATVEC_M_NEON(NAME, M)                                              \
static void NAME(float *y, int ldy, const int8_t *qx, const float *sx,         \
                 const int8_t *W, const float *ws, int in_dim, int rows) {     \
    int nb = in_dim / Q8B;                                                     \
    const int half = Q8B / 2;                                                  \
    const uint8x16_t mask = vdupq_n_u8(0x0F);                                  \
    const int8x16_t bias = vdupq_n_s8(8);                                      \
    for (int o = 0; o < rows; o++) {                                           \
        const int8_t *w = W + (size_t)o * (in_dim / 2);                        \
        const float *s = ws + (size_t)o * nb;                                  \
        float32x4_t accf[M];                                                   \
        for (int r = 0; r < M; r++) accf[r] = vdupq_n_f32(0.0f);               \
        for (int b = 0; b < nb; b++) {                                         \
            const int8_t *wb = w + (size_t)b * half;                           \
            int32x4_t a[M];                                                    \
            for (int r = 0; r < M; r++) a[r] = vdupq_n_s32(0);                 \
            for (int j = 0; j < half; j += 16) {                               \
                uint8x16_t raw = vld1q_u8((const uint8_t *)(wb + j));          \
                int8x16_t lo =                                                 \
                    vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), bias);  \
                int8x16_t hi =                                                 \
                    vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), bias);   \
                for (int r = 0; r < M; r++) {                                  \
                    const int8_t *xb = qx + (size_t)r * in_dim + b * Q8B;      \
                    a[r] = vdotq_s32(a[r], lo, vld1q_s8(xb + j));              \
                    a[r] = vdotq_s32(a[r], hi, vld1q_s8(xb + half + j));       \
                }                                                              \
            }                                                                  \
            float sc = s[b];                                                   \
            for (int r = 0; r < M; r++)                                        \
                accf[r] = vfmaq_n_f32(accf[r], vcvtq_f32_s32(a[r]),            \
                                      sc * sx[(size_t)r * nb + b]);            \
        }                                                                      \
        for (int r = 0; r < M; r++)                                            \
            y[(size_t)r * ldy + o] = vaddvq_f32(accf[r]);                      \
    }                                                                          \
}

Q4_MATVEC_M_NEON(q4n_mv_m1, 1)
Q4_MATVEC_M_NEON(q4n_mv_m2, 2)
Q4_MATVEC_M_NEON(q4n_mv_m4, 4)
Q4_MATVEC_M_NEON(q4n_mv_m8, 8)
#endif

void qwen_q4_matvec_m_neon(float *y, int ldy, const int8_t *qx, const float *sx,
                           int m, const int8_t *W, const float *ws,
                           int in_dim, int rows) {
#ifdef __ARM_FEATURE_DOTPROD
    int nb = in_dim / Q8B;
    int off = 0;
    while (off < m) {
        int take = m - off;
        const int8_t *x = qx + (size_t)off * in_dim;
        const float *xs = sx + (size_t)off * nb;
        float *dst = y + (size_t)off * ldy;
        if (take >= 8)      { q4n_mv_m8(dst, ldy, x, xs, W, ws, in_dim, rows); take = 8; }
        else if (take >= 4) { q4n_mv_m4(dst, ldy, x, xs, W, ws, in_dim, rows); take = 4; }
        else if (take >= 2) { q4n_mv_m2(dst, ldy, x, xs, W, ws, in_dim, rows); take = 2; }
        else                { q4n_mv_m1(dst, ldy, x, xs, W, ws, in_dim, rows); take = 1; }
        off += take;
    }
#else
    int nb = in_dim / Q8B;
    for (int r = 0; r < m; r++)
        qwen_q4_matvec_neon(y + (size_t)r * ldy, qx + (size_t)r * in_dim,
                            sx + (size_t)r * nb, W, ws, in_dim, rows);
#endif
}
