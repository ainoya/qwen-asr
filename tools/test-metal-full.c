/* Real-model state and numerical tests for resident Metal, including chunk,
 * tile/window tails, cached prefixes, rollback, and matvec vs dense arithmetic.
 * Run separately from benchmarks; GPU validation layers are encouraged. */
#include "qwen_asr.h"
#include "qwen_asr_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double relative(const float *a,const float *b,size_t n) {
    double err=0,norm=0;
    for(size_t i=0;i<n;i++) {
        if(!isfinite(a[i]) || !isfinite(b[i])) {
            fprintf(stderr,"nonfinite index=%zu reference=%g metal=%g\n",i,a[i],b[i]);
            return INFINITY;
        }
        double d=a[i]-b[i]; err+=d*d; norm+=(double)a[i]*a[i];
    }
    return sqrt(err/fmax(norm,1e-30));
}
static void mode(const char *m) { setenv("QWEN_METAL",m,1); qwen_metal_stats_reset(); }
static int encoder_case(qwen_ctx_t *ctx,int frames) {
    float *mel=malloc((size_t)128*frames*4);
    if(!mel) return 1;
    for(int m=0;m<128;m++) for(int t=0;t<frames;t++)
        mel[m*frames+t]=-0.8f+0.7f*sinf(m*0.031f+t*0.023f)+0.2f*cosf(t*0.21f);
    int na=0,nb=0;
    mode("0"); float *a=qwen_encoder_forward(ctx,mel,frames,&na);
    mode("full"); float *b=qwen_encoder_forward(ctx,mel,frames,&nb);
    double rel=a && b && na==nb?relative(a,b,(size_t)na*ctx->config.enc_output_dim):INFINITY;
    int bad=rel>0.003 || !qwen_metal_stats().encoders;
    printf("encoder frames=%d tokens=%d relative=%.6g %s\n",frames,nb,rel,bad?"FAIL":"PASS"); fflush(stdout);
    free(a); free(b); free(mel); return bad;
}
static int prefill_case(qwen_ctx_t *ctx,const float *input,int seq) {
    int dim=ctx->config.dec_hidden;
    mode("0"); ctx->kv_cache_len=0; qwen_decoder_prefill(ctx,input,seq);
    float *ref=malloc((size_t)seq*dim*4); if(!ref) return 1;
    memcpy(ref,ctx->pref_x,(size_t)seq*dim*4);
    mode("full"); ctx->kv_cache_len=0; qwen_decoder_prefill(ctx,input,seq);
    double rel=relative(ref,ctx->pref_x,(size_t)seq*dim);
    int bad=rel>0.003 || ctx->kv_cache_len!=seq || !qwen_metal_stats().prefills;
    printf("prefill seq=%d relative=%.6g %s\n",seq,rel,bad?"FAIL":"PASS"); fflush(stdout);
    free(ref); return bad;
}
static int cache_case(qwen_ctx_t *ctx,const float *input) {
    int dim=ctx->config.dec_hidden, kvdim=ctx->config.dec_kv_heads*ctx->config.dec_head_dim;
    int layers=ctx->config.dec_layers, prefix=31, suffix=33;
    size_t prefix_bytes=(size_t)layers*prefix*kvdim*2;
    qwen_f16_t *saved=malloc(prefix_bytes*2); float *ref=malloc((size_t)suffix*dim*4);
    if(!saved || !ref) { free(saved); free(ref); return 1; }
    mode("full"); ctx->kv_cache_len=0; qwen_decoder_prefill(ctx,input,prefix+suffix);
    memcpy(ref,ctx->pref_x+(size_t)prefix*dim,(size_t)suffix*dim*4);
    ctx->kv_cache_len=0; qwen_decoder_prefill(ctx,input,prefix);
    for(int l=0;l<layers;l++) {
        memcpy(saved+(size_t)l*prefix*kvdim,ctx->kv_cache_k+(size_t)l*ctx->kv_cache_max*kvdim,(size_t)prefix*kvdim*2);
        memcpy(saved+prefix_bytes/2+(size_t)l*prefix*kvdim,ctx->kv_cache_v+(size_t)l*ctx->kv_cache_max*kvdim,(size_t)prefix*kvdim*2);
    }
    qwen_decoder_prefill(ctx,input+(size_t)prefix*dim,suffix);
    double rel=relative(ref,ctx->pref_x,(size_t)suffix*dim); int bad=rel>0.003;
    for(int l=0;l<layers;l++) {
        bad|=memcmp(saved+(size_t)l*prefix*kvdim,ctx->kv_cache_k+(size_t)l*ctx->kv_cache_max*kvdim,(size_t)prefix*kvdim*2)!=0;
        bad|=memcmp(saved+prefix_bytes/2+(size_t)l*prefix*kvdim,ctx->kv_cache_v+(size_t)l*ctx->kv_cache_max*kvdim,(size_t)prefix*kvdim*2)!=0;
    }
    ctx->kv_cache_len=prefix; // Rollback then overwrite speculative suffix.
    qwen_decoder_prefill(ctx,input+(size_t)prefix*dim,suffix);
    bad|=relative(ref,ctx->pref_x,(size_t)suffix*dim)>0.003;
    bad|=qwen_metal_stats().prefills!=4 || ctx->kv_cache_len!=prefix+suffix;
    printf("cache append/rollback relative=%.6g %s\n",rel,bad?"FAIL":"PASS");
    free(saved); free(ref); return bad;
}
static int matvec_case(qwen_ctx_t *ctx,const float *input) {
    int dim=ctx->config.dec_hidden;
    float *ref=malloc(dim*4); if(!ref) return 1;
    mode("full"); ctx->kv_cache_len=0; qwen_decoder_prefill(ctx,input,2);
    memcpy(ref,ctx->pref_x,dim*4);
    ctx->kv_cache_len=0; qwen_decoder_prefill(ctx,input,1);
    double rel=relative(ref,ctx->pref_x,dim);
    int bad=rel>0.003 || qwen_metal_stats().prefills!=2;
    printf("matvec vs dense relative=%.6g %s\n",rel,bad?"FAIL":"PASS");
    free(ref); return bad;
}
static int growth_case(qwen_ctx_t *ctx,const float *input) {
    int dim=ctx->config.dec_hidden, old_capacity=ctx->kv_cache_max;
    float *ref=malloc((size_t)129*dim*4); if(!ref) return 1;
    mode("full"); ctx->kv_cache_len=0; qwen_decoder_prefill(ctx,input,129);
    memcpy(ref,ctx->pref_x,(size_t)129*dim*4);
    ctx->kv_cache_len=0; qwen_decoder_prefill(ctx,input,1100);
    double rel=relative(ref,ctx->pref_x,(size_t)129*dim);
    int bad=rel>0.003 || ctx->kv_cache_len!=1100 || ctx->kv_cache_max<=old_capacity || qwen_metal_stats().prefills!=2;
    printf("KV/RoPE growth relative=%.6g %s\n",rel,bad?"FAIL":"PASS");
    free(ref); return bad;
}
int main(int argc,char **argv) {
    if(argc!=2) { fprintf(stderr,"Usage: %s MODEL_DIR\n",argv[0]); return 2; }
    qwen_verbose=1; qwen_set_threads(8); qwen_set_q8_batch_max(1);
    qwen_ctx_t *ctx=qwen_load(argv[1]); if(!ctx) return 2;
    int dim=ctx->config.dec_hidden, bad=0;
    float *input=malloc((size_t)1100*dim*4); if(!input) { qwen_free(ctx); return 2; }
    for(int i=0;i<1100;i++) qwen_q8_row_to_f32(input+(size_t)i*dim,&ctx->decoder.tok_embeddings_q8,1000+i);
    int frames[]={9,99,100,101,799,800,801};
    for(size_t i=0;i<sizeof(frames)/sizeof(*frames);i++) bad+=encoder_case(ctx,frames[i]);
    int window=ctx->config.enc_n_window_infer; ctx->config.enc_n_window_infer=100;
    bad+=encoder_case(ctx,801); ctx->config.enc_n_window_infer=window;
    int seqs[]={2,31,32,33,127,128,129,256};
    for(size_t i=0;i<sizeof(seqs)/sizeof(*seqs);i++) bad+=prefill_case(ctx,input,seqs[i]);
    bad+=cache_case(ctx,input); bad+=matvec_case(ctx,input); bad+=growth_case(ctx,input);
    // Disabled and unsupported layouts must decline without advancing KV.
    mode("0"); int before=ctx->kv_cache_len;
    bad+=qwen_metal_decoder_step(ctx,input)!=-1;
    mode("full"); int bits=ctx->decoder.layers[0].wq_q8.bits;
    ctx->decoder.layers[0].wq_q8.bits=4;
    bad+=qwen_metal_decoder_step(ctx,input)!=-1 || ctx->kv_cache_len!=before;
    ctx->decoder.layers[0].wq_q8.bits=bits;
    printf("resident checks: %s (%d failures)\n",bad?"FAIL":"PASS",bad);
    free(input); qwen_free(ctx); return bad?1:0;
}
