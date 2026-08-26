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
    double *absmean;
    double *sqmean;
} calib_entry_t;

static void calib_entries_free(calib_entry_t *v, int n) {
    for (int i = 0; i < n; i++) { free(v[i].absmean); free(v[i].sqmean); }
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
        e->absmean = (double *)malloc((size_t)e->cols * sizeof(double));
        if (!e->sqmean || !e->absmean) break;
        /* absmax is read past: nothing here needs the single loudest sample. */
        int ok = fread(e->absmean, sizeof(double), e->cols, f) == e->cols
              && fread(e->sqmean, sizeof(double), e->cols, f) == e->cols;
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

/* ========================================================================
 * AWQ-style channel rescaling: does it buy anything here?
 *
 * The idea (Lin et al., "AWQ: Activation-aware Weight Quantization") is that a
 * weight column feeding a loud activation channel deserves more of the
 * quantization grid than one feeding a quiet channel. Scaling column c of W up
 * by s[c] and dividing x[c] by the same s leaves the product unchanged, but it
 * changes what the block quantizer sees: the loud column now sits higher in its
 * block, and its error, once divided back down by s[c], shrinks. The cost lands
 * on the quiet columns sharing that block, which is the trade being bought.
 *
 * s[c] = (mean|x_c| / geomean) ^ alpha, so alpha=0 is plain quantization. This
 * searches alpha and reports what it saves, per group of matrices that share an
 * input - q/k/v see the same layernorm output, so they must share one s vector,
 * which means one alpha for the group rather than per matrix.
 *
 * Deliberately measure-before-build: applying this at inference means folding
 * 1/s into the preceding norm (for q/k/v and gate/up) and dividing explicitly
 * (for o and down), which is only worth writing if the error actually falls.
 * ======================================================================== */

/* Matrices that share an input, and therefore must share one scale vector. */
typedef struct {
    const char *tag;
    const char *members[3];
    int n;
} awq_group_t;

static double awq_geomean(const double *a, int n) {
    double acc = 0;
    int used = 0;
    for (int c = 0; c < n; c++) {
        if (a[c] <= 0) continue;
        acc += log(a[c]);
        used++;
    }
    return used ? exp(acc / used) : 1.0;
}

/* Weighted relative error of quantizing `mats` to four bits after scaling
 * their columns by s. Writes the error and the reference signal energy. */
static int awq_group_error(qwen_q8_mat_t **mats, int n_mats,
                           const double *actw, const double *s,
                           float *w8, float *scaled,
                           double *err_out, double *sig_out) {
    double err = 0, sig = 0;
    for (int i = 0; i < n_mats; i++) {
        qwen_q8_mat_t *m = mats[i];
        for (int r = 0; r < m->rows; r++) {
            qwen_q8_row_to_f32(w8, m, r);
            for (int c = 0; c < m->cols; c++) scaled[c] = w8[c] * (float)s[c];
            qwen_q4_roundtrip_row(scaled, m->cols);
            for (int c = 0; c < m->cols; c++) {
                /* Undo the scaling: this is the weight the model effectively
                 * multiplies once x has been divided by s. */
                double back = (double)scaled[c] / s[c];
                double d = back - (double)w8[c];
                err += d * d * actw[c];
                sig += (double)w8[c] * (double)w8[c] * actw[c];
            }
        }
    }
    *err_out = err;
    *sig_out = sig;
    return 0;
}

int qwen_awq_search(const qwen_ctx_t *ctx, const char *path) {
    int n_stats = 0;
    calib_entry_t *stats = calib_read(path, &n_stats);
    if (!stats) return -1;

    qwen_q8_mat_t *mats[QWEN_CALIB_MAX];
    static char names[QWEN_CALIB_MAX][QWEN_CALIB_NAME];
    int n = calib_matrices(ctx, mats, names, QWEN_CALIB_MAX);

    static const double alphas[] = { 0.0, 0.25, 0.5, 0.75, 1.0 };
    const int n_alpha = (int)(sizeof(alphas) / sizeof(alphas[0]));

    /* q, k and v share the layernorm output; gate and up are already one
     * matrix. o and down each stand alone. */
    awq_group_t groups[] = {
        { "qkv",     { "q", "k", "v" }, 3 },
        { "o",       { "o" },           1 },
        { "gate_up", { "gate_up" },     1 },
        { "down",    { "down" },        1 },
    };

    printf("group\tcols\talpha\terr_a0\terr_best\tgain_pct\n");

    double *actw = NULL, *s = NULL, *a = NULL;
    float *w8 = NULL, *scaled = NULL;
    size_t cap = 0;
    double tot_a0 = 0, tot_best = 0, tot_sig = 0;
    int rc = 0;

    for (int layer = 0; layer <= ctx->config.dec_layers && rc == 0; layer++) {
        int is_head = (layer == ctx->config.dec_layers);
        int n_groups = is_head ? 1 : (int)(sizeof(groups) / sizeof(groups[0]));

        for (int g = 0; g < n_groups && rc == 0; g++) {
            qwen_q8_mat_t *gm[3];
            int n_gm = 0;
            char label[QWEN_CALIB_NAME];
            const calib_entry_t *st = NULL;

            if (is_head) {
                snprintf(label, sizeof(label), "lm_head");
                for (int i = 0; i < n; i++)
                    if (strcmp(names[i], "lm_head") == 0) gm[n_gm++] = mats[i];
                st = calib_find(stats, n_stats, "lm_head");
            } else {
                snprintf(label, sizeof(label), "L%02d.%s", layer, groups[g].tag);
                for (int k = 0; k < groups[g].n; k++) {
                    char want[QWEN_CALIB_NAME];
                    snprintf(want, sizeof(want), "L%02d.%s", layer, groups[g].members[k]);
                    for (int i = 0; i < n; i++)
                        if (strcmp(names[i], want) == 0) gm[n_gm++] = mats[i];
                    if (!st) st = calib_find(stats, n_stats, want);
                }
            }
            if (n_gm == 0 || !st || st->rows <= 0) continue;

            if ((size_t)st->cols > cap) {
                cap = st->cols;
                free(actw); free(s); free(a); free(w8); free(scaled);
                actw = (double *)malloc(cap * sizeof(double));
                s = (double *)malloc(cap * sizeof(double));
                a = (double *)malloc(cap * sizeof(double));
                w8 = (float *)malloc(cap * sizeof(float));
                scaled = (float *)malloc(cap * sizeof(float));
                if (!actw || !s || !a || !w8 || !scaled) { rc = -1; break; }
            }
            for (uint32_t c = 0; c < st->cols; c++) {
                actw[c] = st->sqmean[c] / st->rows;
                a[c] = st->absmean[c] / st->rows;
            }
            double gmean = awq_geomean(a, (int)st->cols);

            double best_err = 0, best_alpha = 0, err_a0 = 0, sig = 0;
            for (int ai = 0; ai < n_alpha; ai++) {
                for (uint32_t c = 0; c < st->cols; c++) {
                    double ratio = a[c] > 0 ? a[c] / gmean : 1.0;
                    /* An empty channel would otherwise scale to zero and take
                     * its whole block's resolution with it. */
                    if (ratio < 1e-3) ratio = 1e-3;
                    s[c] = pow(ratio, alphas[ai]);
                }
                double err = 0, sg = 0;
                awq_group_error(gm, n_gm, actw, s, w8, scaled, &err, &sg);
                if (ai == 0) { err_a0 = err; best_err = err; sig = sg; }
                else if (err < best_err) { best_err = err; best_alpha = alphas[ai]; }
            }
            tot_a0 += err_a0;
            tot_best += best_err;
            tot_sig += sig;
            printf("%s\t%u\t%.2f\t%.5f\t%.5f\t%.1f\n", label, st->cols, best_alpha,
                   sig > 0 ? sqrt(err_a0 / sig) : 0.0,
                   sig > 0 ? sqrt(best_err / sig) : 0.0,
                   err_a0 > 0 ? 100.0 * (1.0 - sqrt(best_err / err_a0)) : 0.0);
            fflush(stdout);
        }
    }
    free(actw); free(s); free(a); free(w8); free(scaled);
    calib_entries_free(stats, n_stats);
    if (rc == 0 && tot_sig > 0)
        fprintf(stderr, "awq: overall relative error %.5f -> %.5f (%.1f%% better)\n",
                sqrt(tot_a0 / tot_sig), sqrt(tot_best / tot_sig),
                100.0 * (1.0 - sqrt(tot_best / tot_a0)));
    return rc;
}

/* ========================================================================
 * Applying the rescaling: scale vectors for the decoder load path
 * ======================================================================== */

struct qwen_awq {
    calib_entry_t *v;
    int n;
    double alpha;
    float *s;      /* scratch holding the most recently requested group */
    uint32_t cap;
};

qwen_awq_t *qwen_awq_open(const char *path, double alpha) {
    int n = 0;
    calib_entry_t *v = calib_read(path, &n);
    if (!v) return NULL;
    qwen_awq_t *a = (qwen_awq_t *)calloc(1, sizeof(*a));
    if (!a) { calib_entries_free(v, n); return NULL; }
    a->v = v;
    a->n = n;
    a->alpha = alpha;
    return a;
}

void qwen_awq_close(qwen_awq_t *a) {
    if (!a) return;
    calib_entries_free(a->v, a->n);
    free(a->s);
    free(a);
}

const float *qwen_awq_scales(qwen_awq_t *a, const char *name, int cols) {
    if (!a) return NULL;
    const calib_entry_t *e = calib_find(a->v, a->n, name);
    if (!e || (int)e->cols != cols || e->rows <= 0) return NULL;

    if (e->cols > a->cap) {
        float *ns = (float *)realloc(a->s, (size_t)e->cols * sizeof(float));
        if (!ns) return NULL;
        a->s = ns;
        a->cap = e->cols;
    }
    /* Normalising by the geometric mean keeps the scales centred on 1, so the
     * block amax - and with it the quantization step - moves as little as the
     * rescaling allows. */
    double *mean = (double *)malloc((size_t)e->cols * sizeof(double));
    if (!mean) return NULL;
    for (uint32_t c = 0; c < e->cols; c++) mean[c] = e->absmean[c] / e->rows;
    double g = awq_geomean(mean, (int)e->cols);
    for (uint32_t c = 0; c < e->cols; c++) {
        double ratio = mean[c] > 0 ? mean[c] / g : 1.0;
        /* An all-but-silent channel would otherwise scale to nearly zero and
         * take its whole block's resolution down with it. */
        if (ratio < 1e-3) ratio = 1e-3;
        a->s[c] = (float)pow(ratio, a->alpha);
    }
    free(mean);
    return a->s;
}

/* Chosen by --awq-search over 25 minutes of real speech: 0.25 was the best of
 * {0, 0.25, 0.5, 0.75, 1.0} in 107 of 113 matrix groups, so a per-group table
 * would buy nothing over this constant. */
const char *qwen_awq_path = NULL;
double qwen_awq_alpha = 0.25;
