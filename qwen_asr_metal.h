/* Optional Apple Silicon Q8 kernels. A negative result requests CPU fallback.
 * Like the existing CPU scratch/thread pool, calls are serialized by the caller.
 */
#ifndef QWEN_ASR_METAL_H
#define QWEN_ASR_METAL_H
#include "qwen_asr.h"

int qwen_metal_available(void);
int qwen_metal_linear(float *y, const float *x, const qwen_q8_mat_t *w,
                      int seq);
void qwen_metal_forget(const qwen_q8_mat_t *w);
/* Experimental resident paths, selected with QWEN_METAL=full or =decode.
 * Decoder calls require the ordinary CPU KV allocation and RoPE tables. */
int qwen_metal_decoder_prefill(qwen_ctx_t *ctx, const float *x, int seq);
int qwen_metal_decoder_step(qwen_ctx_t *ctx, const float *x);
float *qwen_metal_encoder(qwen_ctx_t *ctx, const float *mel, int frames, int *seq);
void qwen_metal_context_free(qwen_ctx_t *ctx);
typedef struct { unsigned long prefills, steps, encoders; } qwen_metal_stats_t;
void qwen_metal_stats_reset(void);
qwen_metal_stats_t qwen_metal_stats(void);
#endif
