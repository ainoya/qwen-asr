/*
 * qwen_asr_calib.c - Activation measurement for weight-quantization decisions
 *
 * Quantizing a weight matrix hurts in proportion to the activation it
 * multiplies: a channel the model barely drives can be rounded hard, while a
 * channel carrying large values needs its precision. Deciding which matrices
 * can afford four bits, or how to rescale channels before quantizing them,
 * therefore needs activation magnitudes measured on representative audio -
 * not guessed from the weights alone.
 *
 * This file is the tooling half of that: attach accumulators, dump them, and
 * rank matrices by how much quantization error they would actually contribute.
 * It is native-only and never linked into the browser build.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "qwen_asr.h"
#include "qwen_asr_kernels.h"

/* One row of the dump. Fixed-width so a reader needs no schema. */
#define QWEN_CALIB_NAME 48
#define QWEN_CALIB_MAGIC "QACS"
#define QWEN_CALIB_VERSION 1u
#define QWEN_CALIB_MAX (QWEN_MAX_DEC_LAYERS * 6 + 1)

/* Every quantized matrix in the decoder, with the name it gets in the dump.
 * Naming them here rather than in the kernels is the only reason this lives
 * outside qwen_asr_kernels.c. */
static int calib_matrices(const qwen_ctx_t *ctx, qwen_q8_mat_t **out,
                          char names[][QWEN_CALIB_NAME], int max) {
    const qwen_decoder_t *dec = &ctx->decoder;
    int n = 0;
    for (int i = 0; i < ctx->config.dec_layers; i++) {
        const qwen_dec_layer_t *l = &dec->layers[i];
        struct { const char *tag; const qwen_q8_mat_t *m; } row[] = {
            { "q",       &l->wq_q8      },
            { "k",       &l->wk_q8      },
            { "v",       &l->wv_q8      },
            { "o",       &l->wo_q8      },
            { "gate_up", &l->gate_up_q8 },
            { "down",    &l->down_q8    },
        };
        for (size_t j = 0; j < sizeof(row) / sizeof(row[0]); j++) {
            if (!row[j].m->q || n >= max) continue;
            snprintf(names[n], QWEN_CALIB_NAME, "L%02d.%s", i, row[j].tag);
            out[n++] = (qwen_q8_mat_t *)row[j].m;
        }
    }
    if (dec->tok_embeddings_q8.q && n < max) {
        snprintf(names[n], QWEN_CALIB_NAME, "lm_head");
        out[n++] = (qwen_q8_mat_t *)&dec->tok_embeddings_q8;
    }
    return n;
}

int qwen_calib_begin(qwen_ctx_t *ctx) {
    if (!ctx || !ctx->decoder.quantized) {
        fprintf(stderr, "calib: decoder is not quantized, nothing to measure\n");
        return -1;
    }
    qwen_q8_mat_t *mats[QWEN_CALIB_MAX];
    static char names[QWEN_CALIB_MAX][QWEN_CALIB_NAME];
    int n = calib_matrices(ctx, mats, names, QWEN_CALIB_MAX);
    for (int i = 0; i < n; i++) {
        if (qwen_act_stats_attach(mats[i]) != 0) {
            fprintf(stderr, "calib: out of memory attaching %s\n", names[i]);
            return -1;
        }
    }
    fprintf(stderr, "calib: measuring %d matrices\n", n);
    return n > 0 ? 0 : -1;
}

int qwen_calib_write(const qwen_ctx_t *ctx, const char *path) {
    qwen_q8_mat_t *mats[QWEN_CALIB_MAX];
    static char names[QWEN_CALIB_MAX][QWEN_CALIB_NAME];
    int n = calib_matrices(ctx, mats, names, QWEN_CALIB_MAX);

    /* Only matrices that actually saw activations are worth writing: a dump
     * full of zeros would silently pull a merged average down. */
    int live = 0;
    for (int i = 0; i < n; i++)
        if (mats[i]->stats && mats[i]->stats->rows > 0) live++;
    if (live == 0) {
        fprintf(stderr, "calib: no activations observed, not writing %s\n", path);
        return -1;
    }

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "calib: cannot write %s\n", path);
        return -1;
    }
    uint32_t version = QWEN_CALIB_VERSION, count = (uint32_t)live;
    int ok = fwrite(QWEN_CALIB_MAGIC, 1, 4, f) == 4
          && fwrite(&version, sizeof(version), 1, f) == 1
          && fwrite(&count, sizeof(count), 1, f) == 1;

    for (int i = 0; i < n && ok; i++) {
        const qwen_act_stats_t *s = mats[i]->stats;
        if (!s || s->rows <= 0) continue;
        char name[QWEN_CALIB_NAME];
        memset(name, 0, sizeof(name));
        memcpy(name, names[i], strnlen(names[i], QWEN_CALIB_NAME - 1));
        uint32_t cols = (uint32_t)mats[i]->cols, bits = (uint32_t)mats[i]->bits;
        ok = fwrite(name, 1, sizeof(name), f) == sizeof(name)
          && fwrite(&cols, sizeof(cols), 1, f) == 1
          && fwrite(&bits, sizeof(bits), 1, f) == 1
          && fwrite(&s->rows, sizeof(s->rows), 1, f) == 1
          && fwrite(s->absmean, sizeof(double), cols, f) == cols
          && fwrite(s->sqmean, sizeof(double), cols, f) == cols
          && fwrite(s->absmax, sizeof(float), cols, f) == cols;
    }
    if (fclose(f) != 0) ok = 0;
    if (!ok) {
        fprintf(stderr, "calib: short write on %s\n", path);
        return -1;
    }
    fprintf(stderr, "calib: wrote %d matrices to %s\n", live, path);
    return 0;
}

/* ========================================================================
 * Ranking: how much would four bits actually cost this matrix?
 *
 * The useful quantity is not the weight error but the error that reaches the
 * output. For y = W x, narrowing W to W' perturbs the output by (W - W') x, so
 * a channel's contribution scales with the RMS activation measured on that
 * channel. Summing that over the matrix and dividing by the same weighted norm
 * of W gives a dimensionless number comparable across matrices of different
 * shapes - which is what a per-layer precision choice needs.
 *
 * Q8 is the reference rather than the original bf16 on purpose: the question
 * being asked is what the *further* step down to four bits costs, and Q8 is
 * already known to be transcript-identical on this model.
 * ======================================================================== */

typedef struct {
    char name[QWEN_CALIB_NAME];
    uint32_t cols;
    double rows;
    double *sqmean;
} calib_entry_t;

static void calib_entries_free(calib_entry_t *v, int n) {
    for (int i = 0; i < n; i++) free(v[i].sqmean);
    free(v);
}

/* Read a dump, keeping only what the ranking needs (the per-channel energy). */
static calib_entry_t *calib_read(const char *path, int *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "calib: cannot read %s\n", path);
        return NULL;
    }
    char magic[4];
    uint32_t version = 0, count = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, QWEN_CALIB_MAGIC, 4) != 0 ||
        fread(&version, sizeof(version), 1, f) != 1 || version != QWEN_CALIB_VERSION ||
        fread(&count, sizeof(count), 1, f) != 1 || count == 0 ||
        count > QWEN_CALIB_MAX) {
        fprintf(stderr, "calib: %s is not a v%u statistics dump\n",
                path, QWEN_CALIB_VERSION);
        fclose(f);
        return NULL;
    }
    calib_entry_t *v = (calib_entry_t *)calloc(count, sizeof(*v));
    if (!v) { fclose(f); return NULL; }

    int n = 0;
    for (uint32_t i = 0; i < count; i++) {
        calib_entry_t *e = &v[n];
        uint32_t bits = 0;
        if (fread(e->name, 1, QWEN_CALIB_NAME, f) != QWEN_CALIB_NAME ||
            fread(&e->cols, sizeof(e->cols), 1, f) != 1 ||
            fread(&bits, sizeof(bits), 1, f) != 1 ||
            fread(&e->rows, sizeof(e->rows), 1, f) != 1 ||
            e->cols == 0 || e->cols > (1u << 20))
            break;
        e->name[QWEN_CALIB_NAME - 1] = '\0';
        e->sqmean = (double *)malloc((size_t)e->cols * sizeof(double));
        double *absmean = (double *)malloc((size_t)e->cols * sizeof(double));
        if (!e->sqmean || !absmean) { free(absmean); break; }
        /* absmean and absmax are read past; only the energy is used here. */
        int ok = fread(absmean, sizeof(double), e->cols, f) == e->cols
              && fread(e->sqmean, sizeof(double), e->cols, f) == e->cols;
        free(absmean);
        if (!ok || fseek(f, (long)e->cols * (long)sizeof(float), SEEK_CUR) != 0)
            break;
        n++;
    }
    fclose(f);
    if (n == 0) {
        calib_entries_free(v, 0);
        free(v);
        fprintf(stderr, "calib: %s contained no usable entries\n", path);
        return NULL;
    }
    *n_out = n;
    return v;
}

static const calib_entry_t *calib_find(const calib_entry_t *v, int n,
                                       const char *name) {
    for (int i = 0; i < n; i++)
        if (strcmp(v[i].name, name) == 0) return &v[i];
    return NULL;
}

int qwen_calib_rank(const qwen_ctx_t *ctx, const char *path) {
    int n_stats = 0;
    calib_entry_t *stats = calib_read(path, &n_stats);
    if (!stats) return -1;

    qwen_q8_mat_t *mats[QWEN_CALIB_MAX];
    static char names[QWEN_CALIB_MAX][QWEN_CALIB_NAME];
    int n = calib_matrices(ctx, mats, names, QWEN_CALIB_MAX);

    printf("matrix\tcols\trows\tq4_err\tflat_err\toutlier\tMB_q8\tMB_saved\n");

    double *actw = NULL;
    float *r8 = NULL, *r4 = NULL;
    size_t cap = 0;
    int rc = 0;

    for (int i = 0; i < n; i++) {
        qwen_q8_mat_t *m = mats[i];
        const calib_entry_t *st = calib_find(stats, n_stats, names[i]);
        if (!st || (uint32_t)m->cols != st->cols || st->rows <= 0) {
            fprintf(stderr, "calib: no statistics for %s, skipping\n", names[i]);
            continue;
        }
        if ((size_t)m->cols > cap) {
            cap = (size_t)m->cols;
            free(actw); free(r8); free(r4);
            actw = (double *)malloc(cap * sizeof(double));
            r8 = (float *)malloc(cap * sizeof(float));
            r4 = (float *)malloc(cap * sizeof(float));
            if (!actw || !r8 || !r4) { rc = -1; break; }
        }
        /* Per-channel activation energy, and its flat counterpart so the table
         * shows how much the weighting actually changed the verdict. */
        for (uint32_t c = 0; c < st->cols; c++)
            actw[c] = st->sqmean[c] / st->rows;

        qwen_q8_mat_t q4;
        if (qwen_q4_from_q8(&q4, m) != 0) {
            fprintf(stderr, "calib: cannot requantize %s\n", names[i]);
            rc = -1;
            break;
        }
        double err = 0, sig = 0, ferr = 0, fsig = 0;
        for (int r = 0; r < m->rows; r++) {
            qwen_q8_row_to_f32(r8, m, r);
            qwen_q8_row_to_f32(r4, &q4, r);
            for (int c = 0; c < m->cols; c++) {
                double d = (double)r8[c] - (double)r4[c];
                double w = (double)r8[c];
                err  += d * d * actw[c];
                sig  += w * w * actw[c];
                ferr += d * d;
                fsig += w * w;
            }
        }
        qwen_q8_free(&q4);

        /* Outlier severity: how far the loudest channel sits above the mean.
         * This is what AWQ-style rescaling has to work with. */
        double amax = 0, amean = 0;
        for (uint32_t c = 0; c < st->cols; c++) {
            double a = sqrt(actw[c]);
            amean += a;
            if (a > amax) amax = a;
        }
        amean /= st->cols;

        double mb_q8 = (double)qwen_q8_bytes(m) / (1024.0 * 1024.0);
        printf("%s\t%d\t%d\t%.5f\t%.5f\t%.1f\t%.1f\t%.1f\n",
               names[i], m->cols, m->rows,
               sig > 0 ? sqrt(err / sig) : 0.0,
               fsig > 0 ? sqrt(ferr / fsig) : 0.0,
               amean > 0 ? amax / amean : 0.0,
               mb_q8, mb_q8 * (1.0 - 0.5625 / 1.0625));
        fflush(stdout);
    }
    free(actw); free(r8); free(r4);
    calib_entries_free(stats, n_stats);
    return rc;
}
