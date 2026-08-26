/*
 * qwen_asr_pack.c - write a pre-quantized model file
 *
 * The engine normally quantizes the bf16 safetensors at load time. That is fine
 * natively (the weights are mmap'd and quantization is a fraction of a second),
 * but a browser build has to *fetch* the model, and 4.7 GB of bf16 is not a
 * download anyone wants. This writes a single safetensors file where every
 * large matrix is already stored as Q8 blocks, so the wasm build can use the
 * bytes it fetched in place with no conversion pass and no second copy.
 *
 * Output layout, per quantized matrix NAME:
 *   NAME.q8   I8  [rows, cols]                 int8 quants
 *   NAME.q8s  F32 [rows, cols/QWEN_Q8_BLOCK]   per-block scales
 * Everything else (norms, biases, the conv stem) is written as F32.
 *
 * The SwiGLU gate and up projections are fused into one interleaved matrix here
 * rather than at load time, which is both smaller on disk and avoids a 750 MB
 * allocation in the browser.
 */

#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PACK_ALIGN 64

typedef enum {
    PLAN_F32,        /* copy source tensor, converted to f32 */
    PLAN_Q8_QUANT,   /* int8 quants of source (or fused pair) */
    PLAN_Q8_SCALE,   /* scales that go with the preceding PLAN_Q8_QUANT */
} plan_kind_t;

/* Bits used for one plan entry's quants. The decoder transformer layers can go
 * to four; the encoder and the tied embedding stay at eight, matching what
 * --weights q4 does at load time. */
#define PACK_Q8 8
#define PACK_Q4 4

typedef struct {
    char name[288];
    plan_kind_t kind;
    int ndim;
    int64_t shape[4];
    size_t nbytes;
    size_t offset;          /* relative to the start of the data section */
    char src_a[256];        /* source tensor name */
    char src_b[256];        /* second source when fusing gate/up ("" if none) */
    int bits;               /* PACK_Q8 or PACK_Q4, for the quant entries */
    int cols;               /* true column count; shape[1] is halved at 4 bits */
    char awq[64];           /* AWQ scale group, "" when none applies */
} plan_entry_t;

typedef struct {
    plan_entry_t *v;
    int n, cap;
} plan_t;

static plan_entry_t *plan_add(plan_t *p) {
    if (p->n == p->cap) {
        int nc = p->cap ? p->cap * 2 : 256;
        void *tmp = realloc(p->v, (size_t)nc * sizeof(plan_entry_t));
        if (!tmp) return NULL;
        p->v = tmp;
        p->cap = nc;
    }
    plan_entry_t *e = &p->v[p->n++];
    memset(e, 0, sizeof(*e));
    return e;
}

static void plan_free(plan_t *p) {
    free(p->v);
}

/* Quantizable = a 2-D matrix whose contraction dimension fits whole Q8 blocks. */
static int is_quantizable(const safetensor_t *t) {
    return t->ndim == 2 && t->shape[1] % QWEN_Q8_BLOCK == 0;
}

static int ends_with(const char *s, const char *suffix) {
    size_t ls = strlen(s), lf = strlen(suffix);
    return ls >= lf && strcmp(s + ls - lf, suffix) == 0;
}

static int add_quant_pair(plan_t *p, const char *out_base, int rows, int cols,
                          const char *src_a, const char *src_b, int bits,
                          const char *awq_key) {
    const char *qs = (bits == PACK_Q4) ? "q4" : "q8";
    plan_entry_t *q = plan_add(p);
    if (!q) return -1;
    snprintf(q->name, sizeof(q->name), "%s.%s", out_base, qs);
    q->kind = PLAN_Q8_QUANT;
    q->ndim = 2;
    q->shape[0] = rows;
    /* Four bits pack two values per byte, so the stored width is halved; the
     * loader doubles shape[1] back to get the real column count. */
    q->shape[1] = (bits == PACK_Q4) ? cols / 2 : cols;
    q->nbytes = (size_t)rows * q->shape[1];
    q->bits = bits;
    q->cols = cols;
    snprintf(q->src_a, sizeof(q->src_a), "%s", src_a);
    if (src_b) snprintf(q->src_b, sizeof(q->src_b), "%s", src_b);
    if (awq_key) snprintf(q->awq, sizeof(q->awq), "%s", awq_key);

    plan_entry_t *s = plan_add(p);
    if (!s) return -1;
    snprintf(s->name, sizeof(s->name), "%s.%ss", out_base, qs);
    s->kind = PLAN_Q8_SCALE;
    s->ndim = 2;
    s->shape[0] = rows;
    s->shape[1] = cols / QWEN_Q8_BLOCK;
    s->nbytes = (size_t)rows * (cols / QWEN_Q8_BLOCK) * sizeof(float);
    s->bits = bits;
    return 0;
}

/* Which AWQ scale vector applies to this tensor, if any.
 *
 * Keyed by the group whose *input* the tensor consumes, because that is what
 * the scaling has to be undone against: q, k and v all read the input_layernorm
 * output, so they share one vector with that norm; gate/up share theirs with
 * post_attention_layernorm; down has its own. O is absent on purpose - under
 * grouped-query attention two of its input channels share one V row, so there
 * is nowhere exact to fold the division. */
static int awq_group_of(const char *name, char *key, size_t cap) {
    const char *at = strstr(name, ".layers.");
    int layer;
    if (!at || sscanf(at, ".layers.%d.", &layer) != 1) return 0;
    static const struct { const char *suffix; const char *tag; } map[] = {
        { "self_attn.q_proj.weight",         "q"       },
        { "self_attn.k_proj.weight",         "q"       },
        { "self_attn.v_proj.weight",         "q"       },
        { "input_layernorm.weight",          "q"       },
        { "mlp.gate_proj.weight",            "gate_up" },
        { "post_attention_layernorm.weight", "gate_up" },
        { "mlp.down_proj.weight",            "down"    },
    };
    for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); i++) {
        if (!ends_with(name, map[i].suffix)) continue;
        snprintf(key, cap, "L%02d.%s", layer, map[i].tag);
        return 1;
    }
    return 0;
}

/* The decoder transformer layers, and only those, are what --weights q4
 * narrows; the encoder and the tied embedding stay at eight bits. */
static int is_decoder_layer(const char *name) {
    return strstr(name, "thinker.model.layers.") != NULL;
}

static int write_padding(FILE *f, size_t n) {
    static const char zeros[PACK_ALIGN] = {0};
    while (n > 0) {
        size_t chunk = n > PACK_ALIGN ? PACK_ALIGN : n;
        if (fwrite(zeros, 1, chunk, f) != chunk) return -1;
        n -= chunk;
    }
    return 0;
}

int qwen_pack(const char *model_dir, const char *out_path, int four_bit,
              const char *awq_stats) {
    multi_safetensors_t *ms = multi_safetensors_open(model_dir);
    if (!ms) {
        fprintf(stderr, "pack: cannot open model in %s\n", model_dir);
        return -1;
    }

    /* Baking the rescaling into the file rather than applying it at load is the
     * whole point: the browser fetches weights that are already correct, and
     * the runtime never needs the statistics. */
    qwen_awq_t *awq = NULL;
    if (four_bit && awq_stats) {
        awq = qwen_awq_open(awq_stats, qwen_awq_alpha);
        if (!awq) {
            multi_safetensors_close(ms);
            return -1;
        }
        if (qwen_verbose >= 1)
            fprintf(stderr, "pack: AWQ rescaling from %s, alpha %.2f\n",
                    awq_stats, qwen_awq_alpha);
    } else if (four_bit && qwen_verbose >= 1) {
        fprintf(stderr, "pack: no --awq statistics given; four bits without "
                        "channel rescaling costs real accuracy\n");
    }

    plan_t plan;
    memset(&plan, 0, sizeof(plan));
    int rc = -1;

    /* ---- Pass 1: decide what goes into the file ---- */
    for (int s = 0; s < ms->num_shards; s++) {
        safetensors_file_t *sf = ms->shards[s];
        for (int i = 0; i < sf->num_tensors; i++) {
            const safetensor_t *t = &sf->tensors[i];

            /* lm_head is tied to embed_tokens; the engine only reads the
             * latter, so shipping both would waste 600 MB. */
            if (strcmp(t->name, "thinker.lm_head.weight") == 0) continue;

            /* up_proj is emitted together with gate_proj below. */
            if (ends_with(t->name, ".mlp.up_proj.weight")) continue;

            char key[64];
            int has_key = awq && awq_group_of(t->name, key, sizeof(key));
            int bits = (four_bit && is_decoder_layer(t->name)) ? PACK_Q4 : PACK_Q8;

            if (ends_with(t->name, ".mlp.gate_proj.weight")) {
                char up[256], fused[256];
                size_t base = strlen(t->name) - strlen("gate_proj.weight");
                snprintf(up, sizeof(up), "%.*sup_proj.weight", (int)base, t->name);
                snprintf(fused, sizeof(fused), "%.*sgate_up.weight", (int)base, t->name);
                if (add_quant_pair(&plan, fused, (int)t->shape[0] * 2,
                                   (int)t->shape[1], t->name, up, bits,
                                   has_key ? key : NULL) != 0)
                    goto done;
                continue;
            }

            if (is_quantizable(t)) {
                if (add_quant_pair(&plan, t->name, (int)t->shape[0],
                                   (int)t->shape[1], t->name, NULL, bits,
                                   has_key ? key : NULL) != 0)
                    goto done;
                continue;
            }

            plan_entry_t *e = plan_add(&plan);
            if (!e) goto done;
            snprintf(e->name, sizeof(e->name), "%s", t->name);
            e->kind = PLAN_F32;
            e->ndim = t->ndim;
            for (int d = 0; d < t->ndim && d < 4; d++) e->shape[d] = t->shape[d];
            e->nbytes = (size_t)safetensor_numel(t) * sizeof(float);
            snprintf(e->src_a, sizeof(e->src_a), "%s", t->name);
            /* The norms carry the input-side half of the rescaling. */
            if (has_key) snprintf(e->awq, sizeof(e->awq), "%s", key);
        }
    }

    /* ---- Lay out the data section ---- */
    size_t off = 0;
    for (int i = 0; i < plan.n; i++) {
        plan.v[i].offset = off;
        off += plan.v[i].nbytes;
        off = (off + PACK_ALIGN - 1) & ~(size_t)(PACK_ALIGN - 1);
    }
    size_t data_bytes = off;

    /* ---- Build the JSON header ---- */
    size_t hcap = (size_t)plan.n * 400 + 1024;
    char *hdr = (char *)malloc(hcap);
    if (!hdr) goto done;
    size_t hl = 0;
    hl += (size_t)snprintf(hdr + hl, hcap - hl, "{");
    for (int i = 0; i < plan.n; i++) {
        plan_entry_t *e = &plan.v[i];
        const char *dt = (e->kind == PLAN_Q8_QUANT) ? "I8" : "F32";
        hl += (size_t)snprintf(hdr + hl, hcap - hl, "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                               i ? "," : "", e->name, dt);
        for (int d = 0; d < e->ndim; d++)
            hl += (size_t)snprintf(hdr + hl, hcap - hl, "%s%lld", d ? "," : "",
                                   (long long)e->shape[d]);
        hl += (size_t)snprintf(hdr + hl, hcap - hl, "],\"data_offsets\":[%llu,%llu]}",
                               (unsigned long long)e->offset,
                               (unsigned long long)(e->offset + e->nbytes));
    }
    hl += (size_t)snprintf(hdr + hl, hcap - hl, "}");

    /* Pad the header so the data section starts 64-byte aligned. */
    size_t hpad = (8 + hl + PACK_ALIGN - 1) / PACK_ALIGN * PACK_ALIGN - 8 - hl;
    for (size_t i = 0; i < hpad; i++) hdr[hl + i] = ' ';
    size_t header_len = hl + hpad;

    FILE *f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "pack: cannot write %s\n", out_path);
        free(hdr);
        goto done;
    }

    uint64_t hlen64 = header_len;
    if (fwrite(&hlen64, 8, 1, f) != 1 ||
        fwrite(hdr, 1, header_len, f) != header_len) {
        fprintf(stderr, "pack: header write failed\n");
        free(hdr);
        fclose(f);
        goto done;
    }
    free(hdr);

    /* ---- Pass 2: emit the data ---- */
    qwen_q8_mat_t pending;
    memset(&pending, 0, sizeof(pending));
    size_t written = 0;

    for (int i = 0; i < plan.n; i++) {
        plan_entry_t *e = &plan.v[i];
        if (written < e->offset && write_padding(f, e->offset - written) != 0)
            goto write_fail;
        written = e->offset;

        if (e->kind == PLAN_F32) {
            safetensors_file_t *sf = NULL;
            const safetensor_t *t = multi_safetensors_find(ms, e->src_a, &sf);
            float *v = t ? safetensors_get_f32(sf, t) : NULL;
            if (!v) { fprintf(stderr, "pack: missing %s\n", e->src_a); goto write_fail; }
            /* A norm carrying the input side of the rescaling: dividing its
             * weight is exact, since RMSNorm's own scale is computed from the
             * unscaled input. */
            if (e->awq[0]) {
                int n = (int)(e->nbytes / sizeof(float));
                const float *sc = qwen_awq_scales(awq, e->awq, n);
                if (sc) for (int c = 0; c < n; c++) v[c] /= sc[c];
            }
            size_t ok = fwrite(v, 1, e->nbytes, f);
            free(v);
            if (ok != e->nbytes) goto write_fail;
        } else if (e->kind == PLAN_Q8_QUANT) {
            safetensors_file_t *sfa = NULL, *sfb = NULL;
            const safetensor_t *ta = multi_safetensors_find(ms, e->src_a, &sfa);
            uint16_t *a = ta ? safetensors_get_bf16_direct(sfa, ta) : NULL;
            if (!a) { fprintf(stderr, "pack: missing %s\n", e->src_a); goto write_fail; }

            qwen_q8_free(&pending);
            if (e->src_b[0]) {
                const safetensor_t *tb = multi_safetensors_find(ms, e->src_b, &sfb);
                uint16_t *b = tb ? safetensors_get_bf16_direct(sfb, tb) : NULL;
                if (!b) { fprintf(stderr, "pack: missing %s\n", e->src_b); goto write_fail; }
                if (qwen_q8_from_bf16_interleave2(&pending, a, b,
                                                  (int)e->shape[0] / 2,
                                                  e->cols) != 0)
                    goto write_fail;
            } else {
                if (qwen_q8_from_bf16(&pending, a, (int)e->shape[0],
                                      e->cols) != 0)
                    goto write_fail;
            }
            if (e->bits == PACK_Q4) {
                /* Go through Q8 so the packed bytes are the same ones
                 * --weights q4 --awq would produce at load time. */
                const float *sc = e->awq[0] ? qwen_awq_scales(awq, e->awq, e->cols) : NULL;
                qwen_q8_mat_t q4;
                int qrc = sc ? qwen_q4_from_q8_scaled(&q4, &pending, sc)
                             : qwen_q4_from_q8(&q4, &pending);
                if (qrc != 0) goto write_fail;
                /* down's input is the SwiGLU product, so its rescaling folds
                 * into the `up` rows here (rows interleave gate0, up0, ...). */
                if (awq && ends_with(e->name, ".mlp.gate_up.weight.q4")) {
                    char dk[64];
                    int layer;
                    const char *at = strstr(e->name, ".layers.");
                    if (at && sscanf(at, ".layers.%d.", &layer) == 1) {
                        int inter = (int)e->shape[0] / 2;
                        snprintf(dk, sizeof(dk), "L%02d.down", layer);
                        const float *ds = qwen_awq_scales(awq, dk, inter);
                        if (ds) qwen_q8_scale_rows(&q4, ds, 1, 2, inter);
                    }
                }
                qwen_q8_free(&pending);
                pending = q4;
            }
            if (fwrite(pending.q, 1, e->nbytes, f) != e->nbytes) goto write_fail;
        } else { /* PLAN_Q8_SCALE */
            if (!pending.scales) { fprintf(stderr, "pack: scales out of order\n"); goto write_fail; }
            if (fwrite(pending.scales, 1, e->nbytes, f) != e->nbytes) goto write_fail;
        }
        written += e->nbytes;

        if (qwen_verbose >= 1 && (i % 64 == 0 || i == plan.n - 1))
            fprintf(stderr, "\rpack: %d/%d tensors, %.2f GB", i + 1, plan.n,
                    (double)written / 1e9);
    }
    if (written < data_bytes && write_padding(f, data_bytes - written) != 0)
        goto write_fail;

    qwen_q8_free(&pending);
    fclose(f);
    if (qwen_verbose >= 1)
        fprintf(stderr, "\rpack: wrote %s (%.2f GB, %d tensors)\n",
                out_path, (double)(8 + header_len + data_bytes) / 1e9, plan.n);
    rc = 0;
    goto done;

write_fail:
    fprintf(stderr, "pack: write failed\n");
    qwen_q8_free(&pending);
    fclose(f);

done:
    plan_free(&plan);
    qwen_awq_close(awq);
    multi_safetensors_close(ms);
    return rc;
}

int qwen_pack_q8(const char *model_dir, const char *out_path) {
    return qwen_pack(model_dir, out_path, 0, NULL);
}
