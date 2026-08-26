/*
 * qwen_asr_kernels_impl.h - internal architecture dispatch for hot kernels
 */

#ifndef QWEN_ASR_KERNELS_IMPL_H
#define QWEN_ASR_KERNELS_IMPL_H

#include <stdint.h>

/* Activation rows the batched Q8 kernels accept in one call; they keep one
 * accumulator per row in registers. */
#define QWEN_Q8_MAX_M 16

void qwen_bf16_matvec_fused_generic(float *y, const float *x, const uint16_t *W_bf16,
                                    const float *bias, int in_dim, int out_dim);
void qwen_argmax_bf16_range_generic(const float *x, const uint16_t *W_bf16,
                                    int in_dim, int start, int end,
                                    int *best_out, float *best_val_out);
void qwen_q8_matvec_generic(float *y, const int8_t *qx, const float *sx,
                            const int8_t *W, const float *ws,
                            int in_dim, int rows);
void qwen_q8_argmax_range_generic(const int8_t *qx, const float *sx,
                                  const int8_t *W, const float *ws,
                                  int in_dim, int rows, int row_base,
                                  int *best_out, float *best_val_out);
void qwen_q4_matvec_generic(float *y, const int8_t *qx, const float *sx,
                            const int8_t *W, const float *ws,
                            int in_dim, int rows);
void qwen_q8_matvec_m_generic(float *y, int ldy, const int8_t *qx, const float *sx,
                              int m, const int8_t *W, const float *ws,
                              int in_dim, int rows);
void qwen_q8_quantize_row_generic(const float *x, int8_t *qx, float *sx, int n);
float qwen_dot_f32_generic(const float *a, const float *b, int n);
void qwen_vec_scale_inplace_generic(float *dst, float scale, int n);
void qwen_vec_axpy_inplace_generic(float *dst, const float *src, float alpha, int n);
void qwen_vec_scale_add_generic(float *dst, const float *src, float correction, int n);

#if defined(__wasm_simd128__) && !defined(QWEN_NO_WASM_SIMD)
void qwen_bf16_matvec_fused_wasm(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim);
void qwen_argmax_bf16_range_wasm(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out);
void qwen_q8_matvec_wasm(float *y, const int8_t *qx, const float *sx,
                         const int8_t *W, const float *ws,
                         int in_dim, int rows);
void qwen_q8_argmax_range_wasm(const int8_t *qx, const float *sx,
                               const int8_t *W, const float *ws,
                               int in_dim, int rows, int row_base,
                               int *best_out, float *best_val_out);
void qwen_q8_matvec_m_wasm(float *y, int ldy, const int8_t *qx, const float *sx,
                           int m, const int8_t *W, const float *ws,
                           int in_dim, int rows);
void qwen_q8_quantize_row_wasm(const float *x, int8_t *qx, float *sx, int n);
float qwen_dot_f32_wasm(const float *a, const float *b, int n);
void qwen_vec_scale_inplace_wasm(float *dst, float scale, int n);
void qwen_vec_axpy_inplace_wasm(float *dst, const float *src, float alpha, int n);
void qwen_vec_scale_add_wasm(float *dst, const float *src, float correction, int n);

#define qwen_bf16_matvec_fused_impl qwen_bf16_matvec_fused_wasm
#define qwen_argmax_bf16_range_impl qwen_argmax_bf16_range_wasm
#define qwen_q8_matvec_impl qwen_q8_matvec_wasm
#define qwen_q8_matvec_m_impl qwen_q8_matvec_m_wasm
#define qwen_q4_matvec_impl qwen_q4_matvec_generic
#define qwen_q8_argmax_range_impl qwen_q8_argmax_range_wasm
#define qwen_q8_quantize_row_impl qwen_q8_quantize_row_wasm
#define qwen_dot_f32_impl qwen_dot_f32_wasm
#define qwen_vec_scale_inplace_impl qwen_vec_scale_inplace_wasm
#define qwen_vec_axpy_inplace_impl qwen_vec_axpy_inplace_wasm
#define qwen_vec_scale_add_impl qwen_vec_scale_add_wasm

#elif defined(__ARM_NEON)
void qwen_bf16_matvec_fused_neon(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim);
void qwen_argmax_bf16_range_neon(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out);
void qwen_q8_matvec_neon(float *y, const int8_t *qx, const float *sx,
                        const int8_t *W, const float *ws,
                        int in_dim, int rows);
void qwen_q8_argmax_range_neon(const int8_t *qx, const float *sx,
                              const int8_t *W, const float *ws,
                              int in_dim, int rows, int row_base,
                              int *best_out, float *best_val_out);
void qwen_q4_matvec_neon(float *y, const int8_t *qx, const float *sx,
                         const int8_t *W, const float *ws,
                         int in_dim, int rows);
void qwen_q8_matvec_m_neon(float *y, int ldy, const int8_t *qx, const float *sx,
                           int m, const int8_t *W, const float *ws,
                           int in_dim, int rows);
void qwen_q8_quantize_row_neon(const float *x, int8_t *qx, float *sx, int n);
float qwen_dot_f32_neon(const float *a, const float *b, int n);
void qwen_vec_scale_inplace_neon(float *dst, float scale, int n);
void qwen_vec_axpy_inplace_neon(float *dst, const float *src, float alpha, int n);
void qwen_vec_scale_add_neon(float *dst, const float *src, float correction, int n);

#define qwen_bf16_matvec_fused_impl qwen_bf16_matvec_fused_neon
#define qwen_argmax_bf16_range_impl qwen_argmax_bf16_range_neon
#define qwen_q8_matvec_impl qwen_q8_matvec_neon
#define qwen_q8_matvec_m_impl qwen_q8_matvec_m_neon
#define qwen_q4_matvec_impl qwen_q4_matvec_neon
#define qwen_q8_argmax_range_impl qwen_q8_argmax_range_neon
#define qwen_q8_quantize_row_impl qwen_q8_quantize_row_neon
#define qwen_dot_f32_impl qwen_dot_f32_neon
#define qwen_vec_scale_inplace_impl qwen_vec_scale_inplace_neon
#define qwen_vec_axpy_inplace_impl qwen_vec_axpy_inplace_neon
#define qwen_vec_scale_add_impl qwen_vec_scale_add_neon

#elif defined(__AVX2__) && defined(__FMA__)
void qwen_bf16_matvec_fused_avx(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim);
void qwen_argmax_bf16_range_avx(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out);
void qwen_q8_matvec_avx(float *y, const int8_t *qx, const float *sx,
                        const int8_t *W, const float *ws,
                        int in_dim, int rows);
void qwen_q8_argmax_range_avx(const int8_t *qx, const float *sx,
                              const int8_t *W, const float *ws,
                              int in_dim, int rows, int row_base,
                              int *best_out, float *best_val_out);
void qwen_q8_quantize_row_avx(const float *x, int8_t *qx, float *sx, int n);
float qwen_dot_f32_avx(const float *a, const float *b, int n);
void qwen_vec_scale_inplace_avx(float *dst, float scale, int n);
void qwen_vec_axpy_inplace_avx(float *dst, const float *src, float alpha, int n);
void qwen_vec_scale_add_avx(float *dst, const float *src, float correction, int n);

#define qwen_bf16_matvec_fused_impl qwen_bf16_matvec_fused_avx
#define qwen_argmax_bf16_range_impl qwen_argmax_bf16_range_avx
#define qwen_q8_matvec_impl qwen_q8_matvec_avx
#define qwen_q8_matvec_m_impl qwen_q8_matvec_m_generic
#define qwen_q4_matvec_impl qwen_q4_matvec_generic
#define qwen_q8_argmax_range_impl qwen_q8_argmax_range_avx
#define qwen_q8_quantize_row_impl qwen_q8_quantize_row_avx
#define qwen_dot_f32_impl qwen_dot_f32_avx
#define qwen_vec_scale_inplace_impl qwen_vec_scale_inplace_avx
#define qwen_vec_axpy_inplace_impl qwen_vec_axpy_inplace_avx
#define qwen_vec_scale_add_impl qwen_vec_scale_add_avx

#else
#define qwen_bf16_matvec_fused_impl qwen_bf16_matvec_fused_generic
#define qwen_argmax_bf16_range_impl qwen_argmax_bf16_range_generic
#define qwen_q8_matvec_impl qwen_q8_matvec_generic
#define qwen_q8_matvec_m_impl qwen_q8_matvec_m_generic
#define qwen_q4_matvec_impl qwen_q4_matvec_generic
#define qwen_q8_argmax_range_impl qwen_q8_argmax_range_generic
#define qwen_q8_quantize_row_impl qwen_q8_quantize_row_generic
#define qwen_dot_f32_impl qwen_dot_f32_generic
#define qwen_vec_scale_inplace_impl qwen_vec_scale_inplace_generic
#define qwen_vec_axpy_inplace_impl qwen_vec_axpy_inplace_generic
#define qwen_vec_scale_add_impl qwen_vec_scale_add_generic
#endif

#endif /* QWEN_ASR_KERNELS_IMPL_H */
