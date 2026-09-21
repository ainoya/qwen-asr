/* One loaded model, a warmup per backend, then rotated measured rounds.
 * Counters prevent accidentally timing a CPU fallback as resident Metal.
 * No transcripts or audio are written. Exactness is informative; run the
 * independent ASR regression suite for normalized recognition quality. */
#include "qwen_asr.h"
#include "qwen_asr_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr,"Usage: %s MODEL_DIR WAV [rounds=3]\n",argv[0]); return 2;
    }
    int rounds=argc>3?atoi(argv[3]):3;
    if(rounds<1 || rounds>20) return 2;
    const char *modes[]={"0","1","decode","full"};
    qwen_verbose=2; qwen_set_threads(8);
    qwen_ctx_t *ctx=qwen_load(argv[1]); if(!ctx) return 2;
    char *reference=NULL; int bad=0;
    puts("round,mode,total_ms,encode_ms,decode_ms,exact,prefills,steps,encoders");
    for(int r=0;r<=rounds;r++) for(int turn=0;turn<4;turn++) {
        int m=(turn+r)%4;
        setenv("QWEN_METAL",modes[m],1); qwen_metal_stats_reset();
        fprintf(stderr,"BENCH round=%d mode=%s%s\n",r,modes[m],r?"":" (warmup)");
        char *out=qwen_transcribe(ctx,argv[2]);
        if(!out) { bad=1; goto done; }
        if(!reference) reference=strdup(out);
        if(!reference) { free(out); bad=1; goto done; }
        qwen_metal_stats_t s=qwen_metal_stats();
        if((m>=2 && !s.steps) || (m==3 && (!s.prefills || !s.encoders))) bad=1;
        if(m && !qwen_metal_available()) bad=1;
        printf("%d,%s,%.3f,%.3f,%.3f,%d,%lu,%lu,%lu\n",r,modes[m],
            ctx->perf_total_ms,ctx->perf_encode_ms,ctx->perf_decode_ms,
            !strcmp(out,reference),s.prefills,s.steps,s.encoders);
        fflush(stdout); free(out);
        if(bad) { fprintf(stderr,"Required GPU execution did not complete; refusing fallback timings.\n"); goto done; }
    }
done:
    free(reference); qwen_free(ctx); return bad;
}
