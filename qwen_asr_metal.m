/* Q8 Metal acceleration on Apple Silicon. CPU kernels remain the fallback.
 * The shader is embedded at build time; the executable needs no sidecar files.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "qwen_asr_metal.h"
#include <unistd.h>
#include <math.h>
#include <time.h>

extern int qwen_verbose;
static const char *shader_source =
#include "qwen_asr_metal_source.h"
;

@interface QwenMetalWeight : NSObject
@property(nonatomic, strong) id<MTLBuffer> quants;
@property(nonatomic, strong) id<MTLBuffer> scales;
@property(nonatomic) NSUInteger qOffset;
@property(nonatomic) NSUInteger sOffset;
@end
@implementation QwenMetalWeight
@end

typedef struct { uint32_t rows, cols, seq, pad; } Shape;
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> dequant_pipeline;
static NSMutableDictionary<NSValue *, QwenMetalWeight *> *weights;
static id<MTLBuffer> dequant, input, output;
static int initialized, failed;
static id<MTLLibrary> shader_library;

static int enabled(void) {
    const char *env = getenv("QWEN_METAL");
    return !env || strcmp(env, "0") != 0;
}

static int fail(NSString *message) {
    if (!failed && qwen_verbose)
        fprintf(stderr, "Metal: %s; using CPU kernels\n", message.UTF8String);
    failed = 1;
    return -1;
}

static int initialize(void) {
    if (!enabled() || failed) return -1;
    if (initialized) return 0;
    initialized = 1;
    device = MTLCreateSystemDefaultDevice();
    if (!device || !device.hasUnifiedMemory || ![device supportsFamily:MTLGPUFamilyApple7])
        return fail(@"Apple Silicon GPU unavailable");
    NSError *error = nil;
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion2_3;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        options.mathMode = MTLMathModeSafe;
    } else {
#endif
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    }
#endif
    id<MTLLibrary> library = [device newLibraryWithSource:@(shader_source)
                                                options:options error:&error];
    if (!library) return fail(error.localizedDescription);
    shader_library = library;
    dequant_pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"q8_dequant"] error:&error];
    if (!dequant_pipeline) return fail(error.localizedDescription);
    if (dequant_pipeline.maxTotalThreadsPerThreadgroup < 256)
        return fail(@"unsupported threadgroup size");
    queue = [device newCommandQueue];
    if (!queue) return fail(@"command queue allocation failed");
    weights = [NSMutableDictionary new];
    if (qwen_verbose) fprintf(stderr, "Metal: %s (Q8 prefill via MPS)\n", device.name.UTF8String);
    return 0;
}

int qwen_metal_available(void) {
    @autoreleasepool { return initialize() == 0; }
}

// Read-only view over existing unified memory. Round the *view* to VM pages;
// kernels access only the original tensor, and qwen_q8_free drops the view
// before freeing/unmapping its owner. No extra model copy or f32 weight cache.
static id<MTLBuffer> view(const void *ptr, size_t bytes, NSUInteger *offset) {
    size_t page = (size_t)getpagesize();
    uintptr_t base = (uintptr_t)ptr & ~(uintptr_t)(page - 1);
    *offset = (uintptr_t)ptr - base;
    size_t length = (*offset + bytes + page - 1) & ~(page - 1);
    if (length > device.maxBufferLength) return nil;
    return [device newBufferWithBytesNoCopy:(void *)base length:length
                                   options:MTLResourceStorageModeShared deallocator:nil];
}

static QwenMetalWeight *weight(const qwen_q8_mat_t *w) {
    if (!w || !w->q || !w->scales || w->rows <= 0 || w->cols <= 0 ||
        w->cols % 64 || (size_t)w->rows * w->cols > UINT32_MAX ||
        QWEN_IS_Q4(w) || initialize() != 0) return nil;
    NSValue *key = [NSValue valueWithPointer:w->q];
    QwenMetalWeight *entry = weights[key];
    if (entry) return entry;
    entry = [QwenMetalWeight new];
    NSUInteger qo, so;
    size_t count = (size_t)w->rows * w->cols;
    entry.quants = view(w->q, count, &qo);
    entry.scales = view(w->scales, count / 64 * sizeof(float), &so);
    entry.qOffset = qo;
    entry.sOffset = so;
    if (!entry.quants || !entry.scales) {
        fail(@"weight buffer allocation failed");
        return nil;
    }
    weights[key] = entry;
    return entry;
}

static int reserve(id<MTLBuffer> __strong *buffer, size_t bytes) {
    if (*buffer && (*buffer).length >= bytes) return 0;
    if (bytes > device.maxBufferLength) return fail(@"scratch exceeds GPU buffer limit");
    *buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    return *buffer ? 0 : fail(@"scratch allocation failed");
}

static int complete(id<MTLCommandBuffer> command) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        return fail(command.error.localizedDescription ?: @"command failed");
    return 0;
}

int qwen_metal_linear(float *y, const float *x, const qwen_q8_mat_t *w,
                      int seq) {
    if (!y || !x || seq <= 0 || !enabled()) return -1;
    @autoreleasepool {
        QwenMetalWeight *entry = weight(w);
        if (!entry) return -1;
        size_t nx = (size_t)seq * w->cols, ny = (size_t)seq * w->rows;
        if (reserve(&input, nx * sizeof(float)) || reserve(&output, ny * sizeof(float)) ||
            reserve(&dequant, (size_t)w->rows * w->cols * sizeof(float))) return -1;
        memcpy(input.contents, x, nx * sizeof(float));
        Shape p = {w->rows, w->cols, seq, 0};
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return fail(@"command allocation failed");
        [encoder setComputePipelineState:dequant_pipeline];
        [encoder setBuffer:entry.quants offset:entry.qOffset atIndex:0];
        [encoder setBuffer:entry.scales offset:entry.sOffset atIndex:1];
        [encoder setBuffer:dequant offset:0 atIndex:2];
        [encoder setBytes:&p length:sizeof(p) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake((size_t)w->rows * w->cols, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        MPSMatrix *a = [[MPSMatrix alloc] initWithBuffer:input descriptor:
            [MPSMatrixDescriptor matrixDescriptorWithRows:seq columns:w->cols rowBytes:w->cols * 4 dataType:MPSDataTypeFloat32]];
        MPSMatrix *b = [[MPSMatrix alloc] initWithBuffer:dequant descriptor:
            [MPSMatrixDescriptor matrixDescriptorWithRows:w->rows columns:w->cols rowBytes:w->cols * 4 dataType:MPSDataTypeFloat32]];
        MPSMatrix *c = [[MPSMatrix alloc] initWithBuffer:output descriptor:
            [MPSMatrixDescriptor matrixDescriptorWithRows:seq columns:w->rows rowBytes:w->rows * 4 dataType:MPSDataTypeFloat32]];
        MPSMatrixMultiplication *op = [[MPSMatrixMultiplication alloc] initWithDevice:device
            transposeLeft:NO transposeRight:YES resultRows:seq resultColumns:w->rows interiorColumns:w->cols alpha:1 beta:0];
        if (!a || !b || !c || !op) return fail(@"matrix allocation failed");
        [op encodeToCommandBuffer:command leftMatrix:a rightMatrix:b resultMatrix:c];
        if (complete(command)) return -1;
        memcpy(y, output.contents, ny * sizeof(float));
        return 0;
    }
}

void qwen_metal_forget(const qwen_q8_mat_t *w) {
    if (!weights || !w || !w->q) return;
    @autoreleasepool {
        [weights removeObjectForKey:[NSValue valueWithPointer:w->q]];
        if (!weights.count) {
            dequant = input = output = nil;
        }
    }
}

/* Resident decoder. Intermediate activations stay in Metal buffers for the
 * whole forward pass. KV is a temporary no-copy view of the CPU allocation,
 * released before returning, so CPU growth, rollback, batching and fallback
 * retain their existing ownership and layout. Calls remain serialized. */
typedef struct {
    uint32_t seq, dim, heads, kvHeads, hd, pos, stride, total, mode, n, layer, pad;
    float eps, scale;
} DecoderShape;
typedef struct { float value; uint32_t token; } Winner;

@interface QwenMetalContext : NSObject
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLBuffer>> *arena;
@property(nonatomic, strong) NSMutableDictionary<NSValue *, id<MTLBuffer>> *constants;
@end
@implementation QwenMetalContext
- (instancetype)init {
    if ((self = [super init])) {
        _arena = [NSMutableDictionary new]; _constants = [NSMutableDictionary new];
    }
    return self;
}
@end

static NSMutableDictionary<NSValue *, QwenMetalContext *> *contexts;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *resident;
static qwen_metal_stats_t full_stats;
void qwen_metal_stats_reset(void) { memset(&full_stats, 0, sizeof(full_stats)); }
qwen_metal_stats_t qwen_metal_stats(void) { return full_stats; }

void qwen_metal_context_free(qwen_ctx_t *ctx) {
    @autoreleasepool { [contexts removeObjectForKey:[NSValue valueWithPointer:ctx]]; }
}

static int resident_enabled(int prefill) {
    const char *mode = getenv("QWEN_METAL");
    return mode && (!strcmp(mode, "full") || (!prefill && !strcmp(mode, "decode")));
}

static QwenMetalContext *resident_context(qwen_ctx_t *ctx) {
    if (initialize()) return nil;
    if (!resident) {
        NSMutableDictionary *pipes = [NSMutableDictionary new];
        for (NSString *name in @[@"resident_matvec", @"resident_norm", @"resident_rope",
              @"resident_scores", @"resident_softmax", @"resident_apply", @"resident_apply_step",
              @"resident_merge", @"resident_add", @"resident_swiglu", @"resident_argmax",
              @"resident_argmax_final", @"encoder_im2col", @"encoder_bias",
              @"encoder_reshape", @"encoder_position", @"encoder_attention", @"resident_gemm", @"encoder_gemm"]) {
            NSError *error = nil;
            id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:
                [shader_library newFunctionWithName:name] error:&error];
            if (!p || p.threadExecutionWidth != 32 || p.maxTotalThreadsPerThreadgroup < 256) {
                fail(error.localizedDescription ?: @"unsupported resident kernel"); return nil;
            }
            pipes[name] = p;
        }
        resident = pipes;
        contexts = [NSMutableDictionary new];
    }
    NSValue *key = [NSValue valueWithPointer:ctx];
    QwenMetalContext *c = contexts[key];
    if (!c) { c = [QwenMetalContext new]; contexts[key] = c; }
    return c;
}

static id<MTLBuffer> arena(QwenMetalContext *c, NSString *name, size_t floats) {
    id<MTLBuffer> b = c.arena[name];
    if (!b || b.length < floats * sizeof(float)) {
        if (floats > device.maxBufferLength / sizeof(float)) { fail(@"resident buffer limit"); return nil; }
        b = [device newBufferWithLength:floats * sizeof(float) options:MTLResourceStorageModeShared];
        if (!b) { fail(@"resident buffer allocation failed"); return nil; }
        c.arena[name] = b;
    }
    return b;
}

static id<MTLBuffer> constant_buffer(QwenMetalContext *c, const float *src, size_t n) {
    if (!src) return nil;
    NSValue *key = [NSValue valueWithPointer:src];
    id<MTLBuffer> b = c.constants[key];
    if (!b) {
        b = [device newBufferWithBytes:src length:n * sizeof(float) options:MTLResourceStorageModeShared];
        if (!b) { fail(@"constant buffer allocation failed"); return nil; }
        c.constants[key] = b;
    }
    return b;
}

static void dispatch(id<MTLComputeCommandEncoder> e, NSString *name,
                     MTLSize groups, MTLSize threads) {
    if(failed || !e) return;
    [e setComputePipelineState:resident[name]];
    [e dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

static void norm_encode(id<MTLComputeCommandEncoder> e, QwenMetalContext *c,
                        id<MTLBuffer> x, id<MTLBuffer> y, const float *w, const float *bias,
                        int seq, int dim, float eps) {
    [e setComputePipelineState:resident[@"resident_norm"]];
    DecoderShape p = {.seq=seq, .dim=dim, .eps=eps, .mode=bias != NULL};
    [e setBuffer:x offset:0 atIndex:0]; [e setBuffer:y offset:0 atIndex:1];
    [e setBuffer:constant_buffer(c, w, dim) offset:0 atIndex:2];
    [e setBuffer:constant_buffer(c, bias ?: w, dim) offset:0 atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    dispatch(e, @"resident_norm", MTLSizeMake(seq,1,1), MTLSizeMake(256,1,1));
}

static int matvec_encode(id<MTLComputeCommandEncoder> e, id<MTLBuffer> x,
                          id<MTLBuffer> y, const qwen_q8_mat_t *w, int mode) {
    QwenMetalWeight *b = weight(w); if (!b) return -1;
    [e setComputePipelineState:resident[@"resident_matvec"]];
    DecoderShape p = {.dim=w->cols, .mode=mode};
    [e setBuffer:b.quants offset:b.qOffset atIndex:0];
    [e setBuffer:b.scales offset:b.sOffset atIndex:1];
    [e setBuffer:x offset:0 atIndex:2]; [e setBuffer:y offset:0 atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    dispatch(e, @"resident_matvec", MTLSizeMake(mode==2?w->rows/2:w->rows,1,1), MTLSizeMake(64,1,1));
    return 0;
}

// Attention projection uses a direct Q8 tile. Feeding its GPU-produced input
// into MPS produced NaNs on tested M1 Pro boundary cases under shader validation.
static int tiled_dense(id<MTLCommandBuffer> command,id<MTLBuffer> x,id<MTLBuffer> y,
                       const qwen_q8_mat_t *w,int seq) {
    QwenMetalWeight *b=weight(w); if(!b) return -1;
    id<MTLComputeCommandEncoder> e=[command computeCommandEncoder];
    if(!e) return fail(@"GEMM encoder allocation failed");
    [e setComputePipelineState:resident[@"resident_gemm"]];
    DecoderShape p={.seq=seq,.dim=w->cols,.n=w->rows};
    [e setBuffer:x offset:0 atIndex:0]; [e setBuffer:b.quants offset:b.qOffset atIndex:1];
    [e setBuffer:b.scales offset:b.sOffset atIndex:2]; [e setBuffer:y offset:0 atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    dispatch(e,@"resident_gemm",MTLSizeMake((w->rows+31)/32,(seq+31)/32,1),MTLSizeMake(8,8,1));
    [e endEncoding]; return 0;
}

static int mps_gemm(id<MTLCommandBuffer> command,id<MTLBuffer> x,id<MTLBuffer> y,
                    id<MTLBuffer> wb,int rows,int cols,int seq) {
    MPSMatrix *b=[[MPSMatrix alloc] initWithBuffer:wb descriptor:
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols rowBytes:cols*4 dataType:MPSDataTypeFloat32]];
    for(int start=0;start<seq;) {
        int count=seq-start;
        // Shader validation exposed NaNs at 32-row multiples in the resident
        // chain. Two disjoint row ranges retain f32 arithmetic and avoid the
        // affected MPS shape without another GPU submission or CPU readback.
        if(count%32==0) count--;
        MPSMatrix *a=[[MPSMatrix alloc] initWithBuffer:x offset:(size_t)start*cols*4 descriptor:
            [MPSMatrixDescriptor matrixDescriptorWithRows:count columns:cols rowBytes:cols*4 dataType:MPSDataTypeFloat32]];
        MPSMatrix *out=[[MPSMatrix alloc] initWithBuffer:y offset:(size_t)start*rows*4 descriptor:
            [MPSMatrixDescriptor matrixDescriptorWithRows:count columns:rows rowBytes:rows*4 dataType:MPSDataTypeFloat32]];
        MPSMatrixMultiplication *op=[[MPSMatrixMultiplication alloc] initWithDevice:device
            transposeLeft:NO transposeRight:YES resultRows:count resultColumns:rows interiorColumns:cols alpha:1 beta:0];
        if(!a || !b || !out || !op) return fail(@"resident matrix allocation failed");
        [op encodeToCommandBuffer:command leftMatrix:a rightMatrix:b resultMatrix:out];
        start+=count;
    }
    return 0;
}

static int dense_encode(id<MTLCommandBuffer> command,id<MTLBuffer> x,id<MTLBuffer> y,
                        const qwen_q8_mat_t *w,int seq) {
    QwenMetalWeight *b=weight(w); if(!b) return -1;
    if(reserve(&dequant,(size_t)w->rows*w->cols*4)) return -1;
    id<MTLComputeCommandEncoder> e=[command computeCommandEncoder];
    if(!e) return fail(@"dequant encoder allocation failed");
    Shape p={w->rows,w->cols,seq,0};
    [e setComputePipelineState:dequant_pipeline];
    [e setBuffer:b.quants offset:b.qOffset atIndex:0]; [e setBuffer:b.scales offset:b.sOffset atIndex:1];
    [e setBuffer:dequant offset:0 atIndex:2]; [e setBytes:&p length:sizeof(p) atIndex:3];
    [e dispatchThreads:MTLSizeMake((size_t)w->rows*w->cols,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding]; return mps_gemm(command,x,y,dequant,w->rows,w->cols,seq);
}

static void pointwise(id<MTLComputeCommandEncoder> e, NSString *name,
                       id<MTLBuffer> x, id<MTLBuffer> y, uint32_t n) {
    [e setComputePipelineState:resident[name]];
    [e setBuffer:x offset:0 atIndex:0]; [e setBuffer:y offset:0 atIndex:1];
    [e setBytes:&n length:sizeof(n) atIndex:2];
    dispatch(e, name, MTLSizeMake((n+255)/256,1,1), MTLSizeMake(256,1,1));
}

static id<MTLComputeCommandEncoder> compute(id<MTLCommandBuffer> command);

static int decoder_supported(qwen_ctx_t *ctx) {
    const qwen_config_t *f=&ctx->config;
    if (!ctx->decoder.quantized || !ctx->decoder.embed_quantized ||
        f->dec_head_dim!=128 || f->dec_layers<=0 || f->dec_layers>QWEN_MAX_DEC_LAYERS ||
        f->dec_heads!=16 || f->dec_kv_heads!=8 ||
        (f->dec_hidden!=2048 && f->dec_hidden!=1024)) return 0;
    if(f->dec_intermediate<=0 || f->dec_intermediate%64 || !ctx->decoder.norm) return 0;
    int dim=f->dec_hidden, qdim=f->dec_heads*f->dec_head_dim, kvdim=f->dec_kv_heads*f->dec_head_dim;
    for (int i=0;i<f->dec_layers;i++) {
        qwen_dec_layer_t *l=&ctx->decoder.layers[i];
        const qwen_q8_mat_t *all[]={&l->wq_q8,&l->wk_q8,&l->wv_q8,&l->wo_q8,&l->gate_up_q8,&l->down_q8};
        int rows[]={qdim,kvdim,kvdim,dim,2*f->dec_intermediate,dim};
        int cols[]={dim,dim,dim,qdim,dim,f->dec_intermediate};
        if(!l->input_norm || !l->post_attn_norm || !l->q_norm_weight || !l->k_norm_weight) return 0;
        for (int j=0;j<6;j++) if (!all[j]->q || !all[j]->scales || QWEN_IS_Q4(all[j]) ||
            all[j]->stats || all[j]->rows!=rows[j] || all[j]->cols!=cols[j]) return 0;
    }
    const qwen_q8_mat_t *embed=&ctx->decoder.tok_embeddings_q8;
    return embed->q && embed->scales && !embed->stats && !QWEN_IS_Q4(embed) &&
        embed->rows==f->vocab_size && embed->cols==dim;
}

static int finite_values(const float *p,size_t n) {
    for(size_t i=0;i<n;i++) if(!isfinite(p[i])) return 0;
    return 1;
}

static int decoder_run(qwen_ctx_t *ctx, const float *src, int seq, int prefill) {
    if (!resident_enabled(prefill) || !src || seq<1 || !decoder_supported(ctx)) return -1;
    @autoreleasepool {
        QwenMetalContext *c=resident_context(ctx); if (!c) return -1;
        const qwen_config_t *f=&ctx->config;
        int dim=f->dec_hidden, hd=f->dec_head_dim, qdim=f->dec_heads*hd;
        int kvdim=f->dec_kv_heads*hd, inter=f->dec_intermediate;
        int pos=ctx->kv_cache_len, total=pos+seq;
        id<MTLBuffer> x=arena(c,@"x",(size_t)seq*dim), xn=arena(c,@"xn",(size_t)seq*dim);
        id<MTLBuffer> q=arena(c,@"q",(size_t)seq*qdim), k=arena(c,@"k",(size_t)seq*kvdim);
        id<MTLBuffer> v=arena(c,@"v",(size_t)seq*kvdim), att=arena(c,@"att",(size_t)seq*qdim);
        id<MTLBuffer> proj=arena(c,@"proj",(size_t)seq*dim), gu=arena(c,@"gu",(size_t)seq*2*inter);
        id<MTLBuffer> gate=arena(c,@"gate",(size_t)seq*inter), scores=arena(c,@"scores",(size_t)f->dec_heads*seq*total);
        id<MTLBuffer> partial=arena(c,@"partial",8*qdim), logits=arena(c,@"logits",f->vocab_size);
        id<MTLBuffer> winners=arena(c,@"winners",65*2);
        if (!x || !xn || !q || !k || !v || !att || !proj || !gu || !gate || !scores || !partial || !logits || !winners) return -1;
        NSUInteger ko,vo,co,so;
        size_t kvbytes=(size_t)f->dec_layers*ctx->kv_cache_max*kvdim*sizeof(qwen_f16_t);
        id<MTLBuffer> ck=view(ctx->kv_cache_k,kvbytes,&ko), cv=view(ctx->kv_cache_v,kvbytes,&vo);
        id<MTLBuffer> cs=view(ctx->rope_cache_cos,(size_t)total*hd*4,&co);
        id<MTLBuffer> sn=view(ctx->rope_cache_sin,(size_t)total*hd*4,&so);
        if (!ck || !cv || !cs || !sn) return fail(@"resident cache view failed");
        memcpy(x.contents,src,(size_t)seq*dim*4);
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> e=compute(command);
        if (!command || !e) return fail(@"resident command allocation failed");
        for (int l=0;l<f->dec_layers;l++) {
            qwen_dec_layer_t *w=&ctx->decoder.layers[l];
            norm_encode(e,c,x,xn,w->input_norm,NULL,seq,dim,f->dec_rms_norm_eps);
            if (seq==1) {
                if (matvec_encode(e,xn,q,&w->wq_q8,0) || matvec_encode(e,xn,k,&w->wk_q8,0) || matvec_encode(e,xn,v,&w->wv_q8,0)) { [e endEncoding]; return -1; }
            } else {
                [e endEncoding];
                if (dense_encode(command,xn,q,&w->wq_q8,seq) || dense_encode(command,xn,k,&w->wk_q8,seq) || dense_encode(command,xn,v,&w->wv_q8,seq)) return -1;
                e=compute(command);
            }
            DecoderShape p={.seq=seq,.dim=kvdim,.heads=f->dec_heads,.kvHeads=f->dec_kv_heads,
                .hd=hd,.pos=pos,.stride=ctx->kv_cache_max,.total=total,.layer=l,
                .eps=f->dec_rms_norm_eps,.scale=1/sqrtf(hd)};
            for (int which=0;which<2;which++) {
                p.mode=which; p.heads=which?f->dec_kv_heads:f->dec_heads;
                [e setComputePipelineState:resident[@"resident_rope"]];
                [e setBuffer:which?k:q offset:0 atIndex:0];
                [e setBuffer:constant_buffer(c,which?w->k_norm_weight:w->q_norm_weight,hd) offset:0 atIndex:1];
                [e setBuffer:cs offset:co atIndex:2]; [e setBuffer:sn offset:so atIndex:3];
                [e setBuffer:ck offset:ko atIndex:4]; [e setBuffer:cv offset:vo atIndex:5];
                [e setBuffer:v offset:0 atIndex:6]; [e setBytes:&p length:sizeof(p) atIndex:7];
                dispatch(e,@"resident_rope",MTLSizeMake(p.heads,seq,1),MTLSizeMake(128,1,1));
            }
            p.mode=0; p.heads=f->dec_heads;
            [e setComputePipelineState:resident[@"resident_scores"]];
            [e setBuffer:q offset:0 atIndex:0]; [e setBuffer:ck offset:ko atIndex:1];
            [e setBuffer:scores offset:0 atIndex:2]; [e setBytes:&p length:sizeof(p) atIndex:3];
            dispatch(e,@"resident_scores",MTLSizeMake((total+3)/4,p.heads,seq),MTLSizeMake(128,1,1));
            [e setComputePipelineState:resident[@"resident_softmax"]];
            [e setBuffer:scores offset:0 atIndex:0]; [e setBytes:&p length:sizeof(p) atIndex:1];
            dispatch(e,@"resident_softmax",MTLSizeMake(seq,p.heads,1),MTLSizeMake(128,1,1));
            [e setComputePipelineState:resident[seq==1?@"resident_apply_step":@"resident_apply"]];
            [e setBuffer:scores offset:0 atIndex:0];
            [e setBuffer:cv offset:vo atIndex:1]; [e setBuffer:seq==1?partial:att offset:0 atIndex:2];
            [e setBytes:&p length:sizeof(p) atIndex:3];
            if (seq==1) {
                dispatch(e,@"resident_apply_step",MTLSizeMake((hd+63)/64,p.heads,8),MTLSizeMake(64,1,1));
                DecoderShape m={.dim=qdim};
                [e setComputePipelineState:resident[@"resident_merge"]];
                [e setBuffer:partial offset:0 atIndex:0]; [e setBuffer:att offset:0 atIndex:1];
                [e setBytes:&m length:sizeof(m) atIndex:2];
                dispatch(e,@"resident_merge",MTLSizeMake((qdim+63)/64,1,1),MTLSizeMake(64,1,1));
                matvec_encode(e,att,x,&w->wo_q8,1);
            } else {
                dispatch(e,@"resident_apply",MTLSizeMake((seq+31)/32,(hd+31)/32,p.heads),MTLSizeMake(16,16,1));
                [e endEncoding]; if (tiled_dense(command,att,proj,&w->wo_q8,seq)) return -1;
                e=compute(command); pointwise(e,@"resident_add",x,proj,seq*dim);
            }
            norm_encode(e,c,x,xn,w->post_attn_norm,NULL,seq,dim,f->dec_rms_norm_eps);
            if (seq==1) {
                matvec_encode(e,xn,gate,&w->gate_up_q8,2);
                matvec_encode(e,gate,x,&w->down_q8,1);
            } else {
                [e endEncoding]; if (dense_encode(command,xn,gu,&w->gate_up_q8,seq)) return -1;
                e=compute(command); pointwise(e,@"resident_swiglu",gu,gate,seq*inter);
                [e endEncoding]; if (dense_encode(command,gate,proj,&w->down_q8,seq)) return -1;
                e=compute(command); pointwise(e,@"resident_add",x,proj,seq*dim);
            }
        }
        if (!prefill) {
            norm_encode(e,c,x,xn,ctx->decoder.norm,NULL,1,dim,f->dec_rms_norm_eps);
            matvec_encode(e,xn,logits,&ctx->decoder.tok_embeddings_q8,0);
            uint32_t vocab=f->vocab_size;
            [e setComputePipelineState:resident[@"resident_argmax"]];
            [e setBuffer:logits offset:0 atIndex:0]; [e setBuffer:winners offset:0 atIndex:1];
            [e setBytes:&vocab length:sizeof(vocab) atIndex:2];
            dispatch(e,@"resident_argmax",MTLSizeMake(64,1,1),MTLSizeMake(32,1,1));
            [e setComputePipelineState:resident[@"resident_argmax_final"]];
            [e setBuffer:winners offset:0 atIndex:0];
            dispatch(e,@"resident_argmax_final",MTLSizeMake(1,1,1),MTLSizeMake(32,1,1));
        }
        [e endEncoding];
        if (failed || complete(command)) return -1;
        if (prefill) {
            if(!finite_values(x.contents,(size_t)seq*dim)) return fail(@"nonfinite GPU prefill");
            memcpy(ctx->pref_x,x.contents,(size_t)seq*dim*4);
            ctx->kv_cache_len=total; full_stats.prefills++; return 0;
        }
        Winner winner=((Winner *)winners.contents)[64];
        if (!isfinite(winner.value) || winner.token>=(uint32_t)f->vocab_size) return fail(@"invalid GPU logits");
        memcpy(ctx->dec_x,xn.contents,dim*4);
        ctx->kv_cache_len=total; full_stats.steps++; return (int)winner.token;
    }
}

int qwen_metal_decoder_prefill(qwen_ctx_t *ctx, const float *x, int seq) { return decoder_run(ctx,x,seq,1); }
int qwen_metal_decoder_step(qwen_ctx_t *ctx, const float *x) { return decoder_run(ctx,x,1,0); }

typedef struct { uint32_t batch, channels, height, width, oh, ow, start, frames, mode, n; } ConvShape;
extern double qwen_enc_conv_ms, qwen_enc_layers_ms;
static double metal_now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1000.0+t.tv_nsec/1e6;
}

static int dense_f32(id<MTLCommandBuffer> command,QwenMetalContext *c,
                     id<MTLBuffer> x,id<MTLBuffer> y,const float *w,int rows,int cols,int seq) {
    id<MTLBuffer> wb=constant_buffer(c,w,(size_t)rows*cols); if(!wb) return -1;
    // The narrow first convolution (K=9) also needs the explicit tile: MPS
    // produced NaNs after CPU/Metal mode switching on the real-audio test.
    if(cols>=64) return mps_gemm(command,x,y,wb,rows,cols,seq);
    id<MTLComputeCommandEncoder> e=[command computeCommandEncoder];
    if(!e) return fail(@"f32 GEMM encoder allocation failed");
    [e setComputePipelineState:resident[@"encoder_gemm"]];
    DecoderShape p={.seq=seq,.dim=cols,.n=rows};
    [e setBuffer:x offset:0 atIndex:0]; [e setBuffer:wb offset:0 atIndex:1];
    [e setBuffer:wb offset:0 atIndex:2]; [e setBuffer:y offset:0 atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    dispatch(e,@"encoder_gemm",MTLSizeMake((rows+31)/32,(seq+31)/32,1),MTLSizeMake(8,8,1));
    [e endEncoding]; return 0;
}

static id<MTLComputeCommandEncoder> compute(id<MTLCommandBuffer> command) {
    id<MTLComputeCommandEncoder> e=[command computeCommandEncoder];
    if(!e) fail(@"compute encoder allocation failed");
    return e;
}

static void bias_encode(id<MTLComputeCommandEncoder> e,QwenMetalContext *c,
                         id<MTLBuffer> x,const float *bias,int seq,int dim,int gelu) {
    [e setComputePipelineState:resident[@"encoder_bias"]];
    DecoderShape p={.n=seq*dim,.dim=dim,.mode=gelu};
    [e setBuffer:x offset:0 atIndex:0]; [e setBuffer:constant_buffer(c,bias,dim) offset:0 atIndex:1];
    [e setBytes:&p length:sizeof(p) atIndex:2];
    dispatch(e,@"encoder_bias",MTLSizeMake((p.n+255)/256,1,1),MTLSizeMake(256,1,1));
}

static int linear_w_encode(id<MTLCommandBuffer> command,QwenMetalContext *c,
                           id<MTLBuffer> x,id<MTLBuffer> y,const qwen_wmat_t *w,
                           const float *bias,int seq,int gelu) {
    int err=w->f32?dense_f32(command,c,x,y,w->f32,w->rows,w->cols,seq):dense_encode(command,x,y,&w->q8,seq);
    if(err) return -1;
    if(bias) {
        id<MTLComputeCommandEncoder> e=compute(command); if(!e) return -1;
        bias_encode(e,c,y,bias,seq,w->rows,gelu); [e endEncoding];
    }
    return failed?-1:0;
}

static int wmat_supported(const qwen_wmat_t *w,int rows,int cols) {
    return w->rows==rows && w->cols==cols && (w->f32 ||
        (w->q8.q && w->q8.scales && !QWEN_IS_Q4(&w->q8) && !w->q8.stats));
}

float *qwen_metal_encoder(qwen_ctx_t *ctx,const float *mel,int frames,int *seq_out) {
    const char *mode=getenv("QWEN_METAL");
    if(!mode || strcmp(mode,"full") || !mel || frames<1) return NULL;
    const qwen_config_t *f=&ctx->config; qwen_encoder_t *w=&ctx->encoder;
    int dim=f->enc_d_model, inter=f->enc_ffn_dim, outdim=f->enc_output_dim;
    int window=13*(f->enc_n_window_infer/100);
    if(f->enc_chunk_size!=100 || f->enc_head_dim!=64 || dim!=f->enc_heads*64 ||
       window<1 || window>104 || f->enc_layers<1 || f->enc_layers>QWEN_MAX_ENC_LAYERS ||
       !w->conv1_weight || !w->conv2_weight || !w->conv3_weight ||
       !w->conv1_bias || !w->conv2_bias || !w->conv3_bias ||
       !wmat_supported(&w->conv_out_weight,dim,7680) ||
       !wmat_supported(&w->proj1_weight,dim,dim) || !wmat_supported(&w->proj2_weight,outdim,dim)) return NULL;
    for(int l=0;l<f->enc_layers;l++) {
        qwen_enc_layer_t *a=&w->layers[l];
        if(!wmat_supported(&a->wq_weight,dim,dim) || !wmat_supported(&a->wk_weight,dim,dim) ||
           !wmat_supported(&a->wv_weight,dim,dim) || !wmat_supported(&a->wo_weight,dim,dim) ||
           !wmat_supported(&a->fc1_weight,inter,dim) || !wmat_supported(&a->fc2_weight,dim,inter)) return NULL;
    }
    @autoreleasepool {
        QwenMetalContext *c=resident_context(ctx); if(!c) return NULL;
        double begin=metal_now();
        int chunks=(frames+99)/100, seq=(frames/100)*13+((frames%100)+7)/8;
        id<MTLBuffer> m=arena(c,@"enc_mel",(size_t)128*frames), x=arena(c,@"enc_x",(size_t)seq*dim);
        id<MTLBuffer> xn=arena(c,@"enc_xn",(size_t)seq*dim), q=arena(c,@"enc_q",(size_t)seq*dim);
        id<MTLBuffer> k=arena(c,@"enc_k",(size_t)seq*dim), v=arena(c,@"enc_v",(size_t)seq*dim);
        id<MTLBuffer> att=arena(c,@"enc_att",(size_t)seq*dim), proj=arena(c,@"enc_proj",(size_t)seq*dim);
        id<MTLBuffer> mid=arena(c,@"enc_mid",(size_t)seq*inter), out=arena(c,@"enc_out",(size_t)seq*outdim);
        id<MTLBuffer> pe=arena(c,@"enc_pe",13*dim);
        if(!m || !x || !xn || !q || !k || !v || !att || !proj || !mid || !out || !pe) return NULL;
        memcpy(m.contents,mel,(size_t)128*frames*4); qwen_sinusoidal_pe(pe.contents,13,dim);
        id<MTLCommandBuffer> command=[queue commandBuffer]; if(!command) { fail(@"encoder command allocation failed"); return NULL; }
        int offset=0;
        for(int chunk=0;chunk<chunks;) {
            // Four full chunks keep the largest im2col allocation below 64 MiB.
            int g=MIN(4,frames/100-chunk); if(g<1) g=1;
            int width=MIN(100,(frames-chunk*100)/g), h=128, channels=1;
            id<MTLBuffer> prev=m;
            for(int layer=0;layer<3;layer++) {
                int oh=(h+1)/2, ow=(width+1)/2, positions=g*oh*ow;
                id<MTLBuffer> col=arena(c,@"enc_col",(size_t)positions*channels*9);
                NSString *name=layer==0?@"enc_c1":layer==1?@"enc_c2":@"enc_c3";
                id<MTLBuffer> next=arena(c,name,(size_t)positions*480);
                if(!col || !next) return NULL;
                ConvShape p={.batch=g,.channels=channels,.height=h,.width=width,.oh=oh,.ow=ow,
                    .start=chunk*100,.frames=frames,.mode=layer!=0,.n=positions*channels*9};
                id<MTLComputeCommandEncoder> e=compute(command); if(!e) return NULL;
                [e setComputePipelineState:resident[@"encoder_im2col"]];
                [e setBuffer:prev offset:0 atIndex:0]; [e setBuffer:col offset:0 atIndex:1]; [e setBytes:&p length:sizeof(p) atIndex:2];
                dispatch(e,@"encoder_im2col",MTLSizeMake((p.n+255)/256,1,1),MTLSizeMake(256,1,1)); [e endEncoding];
                const float *cw=layer==0?w->conv1_weight:layer==1?w->conv2_weight:w->conv3_weight;
                const float *cb=layer==0?w->conv1_bias:layer==1?w->conv2_bias:w->conv3_bias;
                if(dense_f32(command,c,col,next,cw,480,channels*9,positions)) return NULL;
                e=compute(command); if(!e) return NULL;
                bias_encode(e,c,next,cb,positions,480,1); [e endEncoding];
                prev=next; h=oh; width=ow; channels=480;
            }
            id<MTLBuffer> reshaped=arena(c,@"enc_reshape",(size_t)g*width*7680);
            id<MTLBuffer> projected=arena(c,@"enc_conv_proj",(size_t)g*width*dim);
            if(!reshaped || !projected) return NULL;
            ConvShape p={.ow=width,.n=g*width*7680};
            id<MTLComputeCommandEncoder> e=compute(command); if(!e) return NULL;
            [e setComputePipelineState:resident[@"encoder_reshape"]];
            [e setBuffer:prev offset:0 atIndex:0]; [e setBuffer:reshaped offset:0 atIndex:1]; [e setBytes:&p length:sizeof(p) atIndex:2];
            dispatch(e,@"encoder_reshape",MTLSizeMake((p.n+255)/256,1,1),MTLSizeMake(256,1,1)); [e endEncoding];
            if(linear_w_encode(command,c,reshaped,projected,&w->conv_out_weight,NULL,g*width,0)) return NULL;
            e=compute(command); if(!e) return NULL;
            DecoderShape s={.n=g*width*dim,.seq=width,.pos=offset,.dim=dim};
            [e setComputePipelineState:resident[@"encoder_position"]];
            [e setBuffer:projected offset:0 atIndex:0]; [e setBuffer:pe offset:0 atIndex:1];
            [e setBuffer:x offset:0 atIndex:2]; [e setBytes:&s length:sizeof(s) atIndex:3];
            dispatch(e,@"encoder_position",MTLSizeMake((s.n+255)/256,1,1),MTLSizeMake(256,1,1)); [e endEncoding];
            chunk+=g; offset+=g*width;
        }
        if(failed || complete(command)) return NULL;
        qwen_enc_conv_ms=metal_now()-begin; begin=metal_now();
        command=[queue commandBuffer]; if(!command) { fail(@"encoder command allocation failed"); return NULL; }
        for(int layer=0;layer<f->enc_layers;layer++) {
            qwen_enc_layer_t *a=&w->layers[layer];
            id<MTLComputeCommandEncoder> e=compute(command); if(!e) return NULL;
            norm_encode(e,c,x,xn,a->attn_norm_weight,a->attn_norm_bias,seq,dim,1e-5f); [e endEncoding];
            if(linear_w_encode(command,c,xn,q,&a->wq_weight,a->wq_bias,seq,0) ||
               linear_w_encode(command,c,xn,k,&a->wk_weight,a->wk_bias,seq,0) ||
               linear_w_encode(command,c,xn,v,&a->wv_weight,a->wv_bias,seq,0)) return NULL;
            e=compute(command); if(!e) return NULL;
            DecoderShape p={.seq=seq,.dim=dim,.stride=window};
            [e setComputePipelineState:resident[@"encoder_attention"]];
            [e setBuffer:q offset:0 atIndex:0]; [e setBuffer:k offset:0 atIndex:1]; [e setBuffer:v offset:0 atIndex:2];
            [e setBuffer:att offset:0 atIndex:3]; [e setBytes:&p length:sizeof(p) atIndex:4];
            dispatch(e,@"encoder_attention",MTLSizeMake(seq,f->enc_heads,1),MTLSizeMake(32,1,1)); [e endEncoding];
            if(linear_w_encode(command,c,att,proj,&a->wo_weight,a->wo_bias,seq,0)) return NULL;
            e=compute(command); if(!e) return NULL;
            pointwise(e,@"resident_add",x,proj,seq*dim);
            norm_encode(e,c,x,xn,a->ffn_norm_weight,a->ffn_norm_bias,seq,dim,1e-5f); [e endEncoding];
            if(linear_w_encode(command,c,xn,mid,&a->fc1_weight,a->fc1_bias,seq,1) ||
               linear_w_encode(command,c,mid,proj,&a->fc2_weight,a->fc2_bias,seq,0)) return NULL;
            e=compute(command); if(!e) return NULL;
            pointwise(e,@"resident_add",x,proj,seq*dim); [e endEncoding];
        }
        id<MTLComputeCommandEncoder> e=compute(command); if(!e) return NULL;
        norm_encode(e,c,x,xn,w->ln_post_weight,w->ln_post_bias,seq,dim,1e-5f); [e endEncoding];
        if(linear_w_encode(command,c,xn,proj,&w->proj1_weight,w->proj1_bias,seq,1) ||
           linear_w_encode(command,c,proj,out,&w->proj2_weight,w->proj2_bias,seq,0) || failed || complete(command)) return NULL;
        if(!finite_values(out.contents,(size_t)seq*outdim)) { fail(@"nonfinite GPU encoder"); return NULL; }
        float *result=malloc((size_t)seq*outdim*4); if(!result) return NULL;
        memcpy(result,out.contents,(size_t)seq*outdim*4);
        qwen_enc_layers_ms=metal_now()-begin;
        if(qwen_verbose>=2) fprintf(stderr,"  Metal encoder: conv stem %.0f ms, transformer %.0f ms\n",qwen_enc_conv_ms,qwen_enc_layers_ms);
        full_stats.encoders++; *seq_out=seq; return result;
    }
}
