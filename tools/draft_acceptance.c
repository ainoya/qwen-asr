/* draft_acceptance - measure 0.6B -> 1.7B token acceptance for speculative
 * decoding, before writing any of it.
 *
 * Speculative decoding lives or dies on one number: when the draft model is
 * conditioned on a correct (target-model) prefix, how often does its greedy
 * next token equal the target's? That is measured here directly: the target
 * context generates its greedy transcript, then the draft context is
 * teacher-forced along that exact token sequence - its own audio tower, its
 * own prefill, the target's tokens as input - and every position records
 * whether the draft's argmax matched. Consecutive-match run lengths give the
 * expected accepted tokens per verification for any draft length K.
 *
 *   ./draft_acceptance <target-model-dir> <draft-model-dir> [--language X] wav...
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../qwen_asr.h"
#include "../qwen_asr_kernels.h"
#include "../qwen_asr_audio.h"

#define MAX_TOKENS 4096

static void embed_row(qwen_ctx_t *ctx, float *dst, int id) {
    qwen_decoder_t *d = &ctx->decoder;
    int dim = ctx->config.dec_hidden;
    if (d->embed_quantized) {
        qwen_q8_row_to_f32(dst, &d->tok_embeddings_q8, id);
        return;
    }
    const uint16_t *r = d->tok_embeddings_bf16 + (size_t)id * dim;
    for (int i = 0; i < dim; i++) {
        union { uint32_t u; float f; } v;
        v.u = (uint32_t)r[i] << 16;
        dst[i] = v.f;
    }
}

/* Greedy generation, returning token ids. */
static int generate(qwen_ctx_t *ctx, const float *samples, int n, int *out) {
    int seq = 0; double mel_ms, enc_ms;
    float *emb = qwen_build_embeds(ctx, samples, n, &seq, &mel_ms, &enc_ms);
    if (!emb || seq < 2) { free(emb); return -1; }
    int dim = ctx->config.dec_hidden;
    ctx->kv_cache_len = 0;                    /* prefill appends; start clean */
    qwen_decoder_prefill(ctx, emb, seq - 1);
    int tok = qwen_decoder_forward(ctx, emb + (size_t)(seq - 1) * dim);
    free(emb);
    float *tmp = (float *)malloc((size_t)dim * sizeof(float));
    int nout = 0;
    while (tok != QWEN_TOKEN_IM_END && nout < MAX_TOKENS) {
        out[nout++] = tok;
        embed_row(ctx, tmp, tok);
        tok = qwen_decoder_forward(ctx, tmp);
    }
    free(tmp);
    return nout;
}

/* Teacher-forced pass: match[i] = draft argmax at position i == tgt[i]. */
static int teacher_forced(qwen_ctx_t *ctx, const float *samples, int n,
                          const int *tgt, int ntgt, unsigned char *match) {
    int seq = 0; double mel_ms, enc_ms;
    float *emb = qwen_build_embeds(ctx, samples, n, &seq, &mel_ms, &enc_ms);
    if (!emb || seq < 2) { free(emb); return -1; }
    int dim = ctx->config.dec_hidden;
    ctx->kv_cache_len = 0;
    qwen_decoder_prefill(ctx, emb, seq - 1);
    int pred = qwen_decoder_forward(ctx, emb + (size_t)(seq - 1) * dim);
    free(emb);
    float *tmp = (float *)malloc((size_t)dim * sizeof(float));
    for (int i = 0; i < ntgt; i++) {
        match[i] = (unsigned char)(pred == tgt[i]);
        if (i + 1 < ntgt) {
            embed_row(ctx, tmp, tgt[i]);      /* feed the TARGET token */
            pred = qwen_decoder_forward(ctx, tmp);
        }
    }
    free(tmp);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <target-dir> <draft-dir> [--language X] wav...\n", argv[0]);
        return 1;
    }
    const char *lang = NULL;
    int argi = 3;
    if (argi < argc && strcmp(argv[argi], "--language") == 0) { lang = argv[argi + 1]; argi += 2; }

    qwen_verbose = 0;
    qwen_ctx_t *tgt = qwen_load(argv[1]);
    qwen_ctx_t *drf = qwen_load(argv[2]);
    if (!tgt || !drf) { fprintf(stderr, "model load failed\n"); return 1; }
    if (lang) { qwen_set_force_language(tgt, lang); qwen_set_force_language(drf, lang); }

    int *ids = (int *)malloc(MAX_TOKENS * sizeof(int));
    unsigned char *match = (unsigned char *)malloc(MAX_TOKENS);
    long total = 0, hits = 0;
    long runhist[64] = {0};                    /* run length, capped at 63 */
    /* effective tokens per verify for a few draft lengths, simulated */
    int Ks[] = {2, 4, 6, 8, 12, 16};
    long adv[6] = {0}, ver[6] = {0};

    printf("file\ttokens\tacc%%\tmeanrun\n");
    for (int a = argi; a < argc; a++) {
        int n = 0;
        float *samples = qwen_load_wav(argv[a], &n);
        if (!samples) { fprintf(stderr, "skip %s\n", argv[a]); continue; }
        int ntgt = generate(tgt, samples, n, ids);
        if (ntgt <= 0) { free(samples); continue; }
        if (teacher_forced(drf, samples, n, ids, ntgt, match) != 0) { free(samples); continue; }
        free(samples);

        long h = 0, runs = 0, runsum = 0;
        for (int i = 0; i < ntgt; ) {
            if (match[i]) {
                int r = 0;
                while (i + r < ntgt && match[i + r]) r++;
                runhist[r < 63 ? r : 63]++;
                runsum += r; runs++;
                i += r;
            } else i++;
        }
        for (int i = 0; i < ntgt; i++) h += match[i];
        /* simulate speculative advance: each verify of K drafts accepts the
         * matching prefix (<=K) plus the verifier's own next token */
        for (size_t k = 0; k < sizeof(Ks)/sizeof(Ks[0]); k++) {
            int p = 0;
            while (p < ntgt) {
                int r = 0;
                while (r < Ks[k] && p + r < ntgt && match[p + r]) r++;
                adv[k] += r + 1; ver[k]++;
                p += r + 1;
            }
        }
        total += ntgt; hits += h;
        const char *base = strrchr(argv[a], '/');
        printf("%s\t%d\t%.1f\t%.1f\n", base ? base + 1 : argv[a], ntgt,
               100.0 * h / ntgt, runs ? (double)runsum / runs : 0.0);
    }

    printf("\n== overall ==\n");
    printf("positions %ld  acceptance %.1f%%\n", total, 100.0 * hits / (total ? total : 1));
    printf("run-length histogram (len:count): ");
    for (int i = 1; i < 64; i++) if (runhist[i]) printf("%d:%ld ", i, runhist[i]);
    printf("\n");
    printf("K\ttokens/verify (draft len K, +1 verifier token)\n");
    for (size_t k = 0; k < sizeof(Ks)/sizeof(Ks[0]); k++)
        if (ver[k]) printf("%d\t%.2f\n", Ks[k], (double)adv[k] / ver[k]);

    qwen_free(tgt); qwen_free(drf);
    free(ids); free(match);
    return 0;
}
