/*
 * qwen_asr_kernels_wasm.c - WebAssembly SIMD128 hot kernels
 *
 * The portable C kernels auto-vectorize well enough on NEON (within 1.22x of
 * the hand-written version), but wasm has no int8 dot-product instruction, so
 * clang expands the Q8 inner loop into a long widen/multiply/add sequence.
 * Writing it out with i16x8 widening plus i32x4.dot_i16x8 roughly halves the
 * work: measured 18% off decode and 13% off end-to-end against the
 * auto-vectorized build.
 *
 * Relaxed SIMD's i32x4.relaxed_dot_i8x16_i7x16_add looks like the obvious next
 * step but is not usable here: its second operand is an i7, only defined for
 * values in [-64, 63], while Q8 activations span [-127, 127]. Results outside
 * that range are engine-defined, so the instruction is deliberately not used.
 */

#include "qwen_asr_kernels_impl.h"

#ifdef __wasm_simd128__

#include <wasm_simd128.h>
#include <string.h>

#define Q8B 64

/* One block: 64 int8 pairs -> i32x4 partial sums. */
static inline v128_t q8_block_dot(const int8_t *wp, const int8_t *xp) {
    v128_t acc = wasm_i32x4_splat(0);
    for (int i = 0; i < Q8B; i += 16) {
        v128_t wv = wasm_v128_load(wp + i);
        v128_t xv = wasm_v128_load(xp + i);
        acc = wasm_i32x4_add(acc, wasm_i32x4_dot_i16x8(
            wasm_i16x8_extend_low_i8x16(wv), wasm_i16x8_extend_low_i8x16(xv)));
        acc = wasm_i32x4_add(acc, wasm_i32x4_dot_i16x8(
            wasm_i16x8_extend_high_i8x16(wv), wasm_i16x8_extend_high_i8x16(xv)));
    }
    return acc;
}

static inline float hsum_f32x4(v128_t v) {
    return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1) +
           wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3);
}

/* Scaling stays in the vector domain: each block's i32x4 is converted, scaled
 * and accumulated, so there is one horizontal reduction per row rather than
 * one per block. */
static inline float q8_row_dot(const int8_t *w, const float *s,
                               const int8_t *qx, const float *sx, int nb) {
    v128_t accf = wasm_f32x4_splat(0.0f);
    for (int b = 0; b < nb; b++) {
        v128_t acc = q8_block_dot(w + (size_t)b * Q8B, qx + (size_t)b * Q8B);
        accf = wasm_f32x4_add(accf,
            wasm_f32x4_mul(wasm_f32x4_convert_i32x4(acc),
                           wasm_f32x4_splat(s[b] * sx[b])));
    }
    return hsum_f32x4(accf);
}

void qwen_q8_matvec_wasm(float *y, const int8_t *qx, const float *sx,
                         const int8_t *W, const float *ws,
                         int in_dim, int rows) {
    int nb = in_dim / Q8B;
    for (int o = 0; o < rows; o++)
        y[o] = q8_row_dot(W + (size_t)o * in_dim, ws + (size_t)o * nb, qx, sx, nb);
}


/* Batched matvec: one pass over the weight rows scores M activation rows.
 * M has to be a compile-time constant - with a runtime count the accumulator
 * array is indexed dynamically and lives on the stack rather than in
 * registers, which costs about a quarter of the kernel's throughput. */
#define Q8_MATVEC_M_KERNEL(NAME, M)                                            \
static void NAME(float *y, int ldy, const int8_t *qx, const float *sx,         \
                 const int8_t *W, const float *ws, int in_dim, int rows) {     \
    int nb = in_dim / Q8B;                                                     \
    for (int o = 0; o < rows; o++) {                                           \
        const int8_t *w = W + (size_t)o * in_dim;                              \
        const float *s = ws + (size_t)o * nb;                                  \
        v128_t facc[M];                                                        \
        for (int r = 0; r < M; r++) facc[r] = wasm_f32x4_splat(0.0f);          \
        for (int b = 0; b < nb; b++) {                                         \
            const int8_t *wp = w + (size_t)b * Q8B;                            \
            float sc = s[b];                                                   \
            for (int r = 0; r < M; r++) {                                      \
                const int8_t *xp = qx + (size_t)r * in_dim + (size_t)b * Q8B;  \
                v128_t acc = q8_block_dot(wp, xp);                             \
                facc[r] = wasm_f32x4_add(facc[r],                              \
                    wasm_f32x4_mul(wasm_f32x4_convert_i32x4(acc),              \
                                   wasm_f32x4_splat(sc * sx[(size_t)r * nb + b]))); \
            }                                                                  \
        }                                                                      \
        for (int r = 0; r < M; r++) y[(size_t)r * ldy + o] = hsum_f32x4(facc[r]); \
    }                                                                          \
}

Q8_MATVEC_M_KERNEL(q8_mv_m1, 1)
Q8_MATVEC_M_KERNEL(q8_mv_m2, 2)
Q8_MATVEC_M_KERNEL(q8_mv_m4, 4)
Q8_MATVEC_M_KERNEL(q8_mv_m8, 8)

void qwen_q8_matvec_m_wasm(float *y, int ldy, const int8_t *qx, const float *sx,
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

void qwen_q8_argmax_range_wasm(const int8_t *qx, const float *sx,
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

void qwen_q8_quantize_row_wasm(const float *x, int8_t *qx, float *sx, int n) {
    for (int b = 0; b < n; b += Q8B) {
        const float *xb = x + b;
        v128_t amaxv = wasm_f32x4_splat(0.0f);
        for (int i = 0; i < Q8B; i += 4)
            amaxv = wasm_f32x4_pmax(amaxv, wasm_f32x4_abs(wasm_v128_load(xb + i)));
        float amax = wasm_f32x4_extract_lane(amaxv, 0);
        for (int l = 1; l < 4; l++) {
            float v = l == 1 ? wasm_f32x4_extract_lane(amaxv, 1)
                    : l == 2 ? wasm_f32x4_extract_lane(amaxv, 2)
                             : wasm_f32x4_extract_lane(amaxv, 3);
            if (v > amax) amax = v;
        }

        float scale = amax / 127.0f;
        float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        sx[b / Q8B] = scale;

        v128_t invv = wasm_f32x4_splat(inv);
        for (int i = 0; i < Q8B; i += 16) {
            /* f32x4.nearest is round-half-to-even, matching the NEON path's
             * vcvtnq_s32_f32, so the two SIMD backends agree bit for bit. */
            v128_t i0 = wasm_i32x4_trunc_sat_f32x4(
                wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + i), invv)));
            v128_t i1 = wasm_i32x4_trunc_sat_f32x4(
                wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + i + 4), invv)));
            v128_t i2 = wasm_i32x4_trunc_sat_f32x4(
                wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + i + 8), invv)));
            v128_t i3 = wasm_i32x4_trunc_sat_f32x4(
                wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(xb + i + 12), invv)));
            v128_t s01 = wasm_i16x8_narrow_i32x4(i0, i1);
            v128_t s23 = wasm_i16x8_narrow_i32x4(i2, i3);
            wasm_v128_store(qx + b + i, wasm_i8x16_narrow_i16x8(s01, s23));
        }
    }
}

/* ---- f32 helpers used by attention ---- */

float qwen_dot_f32_wasm(const float *a, const float *b, int n) {
    v128_t acc0 = wasm_f32x4_splat(0.0f), acc1 = wasm_f32x4_splat(0.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(a + i), wasm_v128_load(b + i)));
        acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_v128_load(a + i + 4), wasm_v128_load(b + i + 4)));
    }
    float sum = hsum_f32x4(wasm_f32x4_add(acc0, acc1));
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

void qwen_vec_scale_inplace_wasm(float *dst, float scale, int n) {
    v128_t s = wasm_f32x4_splat(scale);
    int i = 0;
    for (; i + 4 <= n; i += 4)
        wasm_v128_store(dst + i, wasm_f32x4_mul(wasm_v128_load(dst + i), s));
    for (; i < n; i++) dst[i] *= scale;
}

void qwen_vec_axpy_inplace_wasm(float *dst, const float *src, float alpha, int n) {
    v128_t a = wasm_f32x4_splat(alpha);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        wasm_v128_store(dst + i, wasm_f32x4_add(wasm_v128_load(dst + i),
                                                wasm_f32x4_mul(wasm_v128_load(src + i), a)));
        wasm_v128_store(dst + i + 4, wasm_f32x4_add(wasm_v128_load(dst + i + 4),
                                                    wasm_f32x4_mul(wasm_v128_load(src + i + 4), a)));
    }
    for (; i < n; i++) dst[i] += alpha * src[i];
}

void qwen_vec_scale_add_wasm(float *dst, const float *src, float correction, int n) {
    v128_t c = wasm_f32x4_splat(correction);
    int i = 0;
    for (; i + 4 <= n; i += 4)
        wasm_v128_store(dst + i, wasm_f32x4_add(wasm_f32x4_mul(wasm_v128_load(dst + i), c),
                                                wasm_v128_load(src + i)));
    for (; i < n; i++) dst[i] = dst[i] * correction + src[i];
}

/* ---- bf16 path (only reached when loading an unpacked model) ---- */

static inline v128_t bf16_lo_to_f32(v128_t bf) {
    return wasm_i32x4_shl(wasm_u32x4_extend_low_u16x8(bf), 16);
}
static inline v128_t bf16_hi_to_f32(v128_t bf) {
    return wasm_i32x4_shl(wasm_u32x4_extend_high_u16x8(bf), 16);
}

void qwen_bf16_matvec_fused_wasm(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        const uint16_t *w = W_bf16 + (size_t)o * in_dim;
        v128_t acc0 = wasm_f32x4_splat(0.0f), acc1 = wasm_f32x4_splat(0.0f);
        int k = 0;
        for (; k + 8 <= in_dim; k += 8) {
            v128_t bf = wasm_v128_load(w + k);
            acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(bf16_lo_to_f32(bf), wasm_v128_load(x + k)));
            acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(bf16_hi_to_f32(bf), wasm_v128_load(x + k + 4)));
        }
        float sum = (bias ? bias[o] : 0.0f) + hsum_f32x4(wasm_f32x4_add(acc0, acc1));
        for (; k < in_dim; k++) {
            uint32_t bits = ((uint32_t)w[k]) << 16;
            float wv;
            memcpy(&wv, &bits, sizeof(wv));
            sum += wv * x[k];
        }
        y[o] = sum;
    }
}

void qwen_argmax_bf16_range_wasm(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out) {
    int best = start;
    float best_val = -1e30f;
    for (int o = start; o < end; o++) {
        float sum;
        qwen_bf16_matvec_fused_wasm(&sum, x, W_bf16 + (size_t)o * in_dim, NULL, in_dim, 1);
        if (sum > best_val) { best_val = sum; best = o; }
    }
    *best_out = best;
    *best_val_out = best_val;
}

#endif /* __wasm_simd128__ */
