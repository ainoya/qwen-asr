/* Alternate CPU/Metal with one loaded model. First pair is warmup; subsequent
 * pairs are measured. Does not write audio or transcripts to disk. */
#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "Usage: %s MODEL_DIR WAV [pairs=3] [segment_seconds=0]\n", argv[0]);
        return 2;
    }
    int pairs = argc > 3 ? atoi(argv[3]) : 3;
    if (pairs < 1 || pairs > 20) return 2;
    qwen_verbose = 2; /* per-stage timings, including prefill, on stderr */
    qwen_set_threads(8);
    qwen_ctx_t *ctx = qwen_load(argv[1]);
    if (!ctx) return 2;
    if (argc > 4) ctx->segment_sec = (float)atof(argv[4]);
    char *reference = NULL;
    int mismatches = 0;
    printf("pair,backend,total_ms,encode_ms,decode_ms,exact\n");
    for (int pair = 0; pair <= pairs; pair++) {
        /* Reverse order every pair to limit systematic thermal/order bias. */
        for (int turn = 0; turn < 2; turn++) {
            int metal = turn ^ (pair & 1);
            setenv("QWEN_METAL", metal ? "1" : "0", 1);
            fprintf(stderr, "BENCH pair=%d backend=%s%s\n", pair, metal ? "Metal" : "CPU", pair ? "" : " (warmup)");
            char *text = qwen_transcribe(ctx, argv[2]);
            if (!text) { free(reference); qwen_free(ctx); return 1; }
            if (metal && !qwen_metal_available()) {
                free(text); free(reference); qwen_free(ctx); return 1;
            }
            if (!reference) reference = strdup(text);
            if (!reference) { free(text); qwen_free(ctx); return 1; }
            int exact = strcmp(reference, text) == 0;
            mismatches += !exact;
            printf("%d,%s,%.3f,%.3f,%.3f,%d\n", pair, metal ? "Metal" : "CPU",
                   ctx->perf_total_ms, ctx->perf_encode_ms, ctx->perf_decode_ms, exact);
            fflush(stdout);
            free(text);
        }
    }
    free(reference);
    qwen_free(ctx);
    return mismatches ? 1 : 0;
}
