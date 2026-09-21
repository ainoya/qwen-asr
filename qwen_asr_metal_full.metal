// Resident decoder kernels. Activations are row-major f32; KV uses the same
// f16 representation as the CPU. One submission runs all transformer layers.
struct DecoderShape {
    uint seq, dim, heads, kvHeads, hd, pos, stride, total, mode, n, layer, pad;
    float eps, scale;
};

// Q8 x f32 GEMM: 32 queries by 32 output channels, 4x4 values per lane.
kernel void resident_gemm(device const float *x [[buffer(0)]],
                          device const char *w [[buffer(1)]],
                          device const float *sc [[buffer(2)]], device float *y [[buffer(3)]],
                          constant DecoderShape &p [[buffer(4)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float xs[1024], ws[1024];
    uint tid=lid.y*8+lid.x, q0=group.y*32, o0=group.x*32;
    uint qi=lid.y*4, oi=lid.x*4;
    float4 a0=0,a1=0,a2=0,a3=0;
    for(uint k0=0;k0<p.dim;k0+=32) {
        for(uint i=tid;i<1024;i+=64) {
            uint r=i/32, k=k0+i%32, row=o0+r;
            xs[i]=q0+r<p.seq && k<p.dim?x[(q0+r)*p.dim+k]:0;
            ws[i]=row<p.n && k<p.dim?float(w[row*p.dim+k])*sc[(row*p.dim+k)/64]:0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint k=0;k<32;k++) {
            float4 b=float4(ws[oi*32+k],ws[(oi+1)*32+k],ws[(oi+2)*32+k],ws[(oi+3)*32+k]);
            a0+=xs[qi*32+k]*b; a1+=xs[(qi+1)*32+k]*b;
            a2+=xs[(qi+2)*32+k]*b; a3+=xs[(qi+3)*32+k]*b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint j=0;j<4;j++) if(o0+oi+j<p.n) {
        if(q0+qi<p.seq) y[(q0+qi)*p.n+o0+oi+j]=a0[j];
        if(q0+qi+1<p.seq) y[(q0+qi+1)*p.n+o0+oi+j]=a1[j];
        if(q0+qi+2<p.seq) y[(q0+qi+2)*p.n+o0+oi+j]=a2[j];
        if(q0+qi+3<p.seq) y[(q0+qi+3)*p.n+o0+oi+j]=a3[j];
    }
}

kernel void encoder_gemm(device const float *x [[buffer(0)]],
                          device const float *w [[buffer(1)]],
                          device const float *sc [[buffer(2)]], device float *y [[buffer(3)]],
                          constant DecoderShape &p [[buffer(4)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float xs[1024], ws[1024];
    uint tid=lid.y*8+lid.x, q0=group.y*32, o0=group.x*32;
    uint qi=lid.y*4, oi=lid.x*4;
    float4 a0=0,a1=0,a2=0,a3=0;
    for(uint k0=0;k0<p.dim;k0+=32) {
        for(uint i=tid;i<1024;i+=64) {
            uint r=i/32, k=k0+i%32, row=o0+r;
            xs[i]=q0+r<p.seq && k<p.dim?x[(q0+r)*p.dim+k]:0;
            ws[i]=row<p.n && k<p.dim?w[row*p.dim+k]:0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint k=0;k<32;k++) {
            float4 b=float4(ws[oi*32+k],ws[(oi+1)*32+k],ws[(oi+2)*32+k],ws[(oi+3)*32+k]);
            a0+=xs[qi*32+k]*b; a1+=xs[(qi+1)*32+k]*b;
            a2+=xs[(qi+2)*32+k]*b; a3+=xs[(qi+3)*32+k]*b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint j=0;j<4;j++) if(o0+oi+j<p.n) {
        if(q0+qi<p.seq) y[(q0+qi)*p.n+o0+oi+j]=a0[j];
        if(q0+qi+1<p.seq) y[(q0+qi+1)*p.n+o0+oi+j]=a1[j];
        if(q0+qi+2<p.seq) y[(q0+qi+2)*p.n+o0+oi+j]=a2[j];
        if(q0+qi+3<p.seq) y[(q0+qi+3)*p.n+o0+oi+j]=a3[j];
    }
}

inline float row_dot(device const char4 *w, device const float *sc,
                     device const float4 *x, uint row, uint dim, uint lane) {
    float a = 0;
    for (uint j = lane; j < dim / 4; j += 64)
        a += dot(float4(w[row * (dim / 4) + j]), x[j]) * sc[row * (dim / 64) + j / 16];
    return a;
}

kernel void resident_matvec(device const char4 *w [[buffer(0)]],
                            device const float *sc [[buffer(1)]],
                            device const float4 *x [[buffer(2)]],
                            device float *y [[buffer(3)]],
                            constant DecoderShape &p [[buffer(4)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float r[2], u[2];
    uint wr = p.mode == 2 ? row * 2 : row;
    float a = simd_sum(row_dot(w, sc, x, wr, p.dim, tid));
    float b = p.mode == 2 ? simd_sum(row_dot(w, sc, x, wr + 1, p.dim, tid)) : 0;
    if (lane == 0) { r[sg] = a; u[sg] = b; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        a = r[0] + r[1];
        if (p.mode == 2) a = (a / (1 + exp(-a))) * (u[0] + u[1]);
        y[row] = p.mode == 1 ? y[row] + a : a;
    }
}

kernel void resident_norm(device const float *x [[buffer(0)]],
                          device float *y [[buffer(1)]],
                          device const float *w [[buffer(2)]],
                          device const float *bias [[buffer(3)]],
                          constant DecoderShape &p [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float r[256], means[256];
    float sum = 0, sq = 0;
    for (uint d = tid; d < p.dim; d += 256) {
        float v = x[row * p.dim + d]; sum += v; sq += v * v;
    }
    r[tid] = sq; means[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s; s /= 2) {
        if (tid < s) { r[tid] += r[tid+s]; means[tid] += means[tid+s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = p.mode ? means[0] / p.dim : 0;
    float variance = r[0] / p.dim;
    if (p.mode) {
        // Center before squaring, matching CPU LayerNorm at large offsets.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float centered = 0;
        for (uint d=tid; d<p.dim; d+=256) {
            float z=x[row*p.dim+d]-mean; centered+=z*z;
        }
        r[tid]=centered; threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint s=128;s;s/=2) {
            if(tid<s) r[tid]+=r[tid+s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        variance=r[0]/p.dim;
    }
    float inv = rsqrt(variance + p.eps);
    for (uint d = tid; d < p.dim; d += 256)
        y[row*p.dim+d] = (x[row*p.dim+d] - mean) * inv * w[d] + (p.mode ? bias[d] : 0);
}

kernel void resident_rope(device float *x [[buffer(0)]],
                          device const float *w [[buffer(1)]],
                          device const float *cs [[buffer(2)]],
                          device const float *sn [[buffer(3)]],
                          device half *cacheK [[buffer(4)]],
                          device half *cacheV [[buffer(5)]],
                          device const float *v [[buffer(6)]],
                          constant DecoderShape &p [[buffer(7)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float r[128], a[128];
    uint base = (group.y * p.heads + group.x) * p.hd;
    float z = tid < p.hd ? x[base + tid] : 0;
    r[tid] = z * z;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=64; s; s/=2) {
        if (tid<s) r[tid] += r[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid<p.hd) a[tid] = z * rsqrt(r[0]/p.hd+p.eps) * w[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid<p.hd) {
        uint halfDim = p.hd/2, d = tid % halfDim;
        uint t = (p.pos+group.y)*p.hd+d;
        float out = tid < halfDim ? a[d]*cs[t]-a[d+halfDim]*sn[t]
                                 : a[d+halfDim]*cs[t]+a[d]*sn[t];
        x[base+tid] = out;
        if (p.mode) {
            uint slot = (p.layer*p.stride+p.pos+group.y)*p.dim+group.x*p.hd+tid;
            cacheK[slot] = half(out);
            cacheV[slot] = half(v[base+tid]);
        }
    }
}

kernel void resident_scores(device const float *q [[buffer(0)]],
                            device const half *k [[buffer(1)]],
                            device float *scores [[buffer(2)]],
                            constant DecoderShape &p [[buffer(3)]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
    uint j=group.x*4+sg, h=group.y, i=group.z;
    if (j>p.pos+i || j>=p.total) return;
    uint qb=(i*p.heads+h)*p.hd;
    uint kb=(p.layer*p.stride+j)*p.dim+(h/(p.heads/p.kvHeads))*p.hd;
    float acc=0;
    for (uint d=lane; d<p.hd; d+=32) acc += q[qb+d]*float(k[kb+d]);
    acc=simd_sum(acc);
    if (!lane) scores[(h*p.seq+i)*p.total+j]=acc*p.scale;
}

kernel void resident_softmax(device float *scores [[buffer(0)]],
                             constant DecoderShape &p [[buffer(1)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float r[128];
    uint len=p.pos+group.x+1, base=(group.y*p.seq+group.x)*p.total;
    float a=-INFINITY;
    for (uint j=tid;j<len;j+=128) a=max(a,scores[base+j]);
    r[tid]=a; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=64;s;s/=2) {
        if(tid<s) r[tid]=max(r[tid],r[tid+s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mx=r[0], sum=0;
    for(uint j=tid;j<len;j+=128) { float e=exp(scores[base+j]-mx); scores[base+j]=e; sum+=e; }
    // All lanes must consume the maximum before reusing reduction storage.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    r[tid]=sum; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=64;s;s/=2) {
        if(tid<s) r[tid]+=r[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv=1/max(r[0],1e-30f);
    for(uint j=tid;j<len;j+=128) scores[base+j]*=inv;
}

kernel void resident_apply(device const float *scores [[buffer(0)]],
                           device const half *v [[buffer(1)]],
                           device float *out [[buffer(2)]],
                           constant DecoderShape &p [[buffer(3)]],
                           uint3 group [[threadgroup_position_in_grid]],
                           uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float vt[512], st[512];
    uint q0=group.x*32, d0=group.y*32, h=group.z;
    uint tid=lid.y*16+lid.x, qi=lid.y*2, di=lid.x*2;
    float2 a0=0, a1=0;
    uint end=min(p.total,p.pos+q0+32), kvh=h/(p.heads/p.kvHeads);
    for(uint j0=0;j0<end;j0+=16) {
        for(uint t=0;t<2;t++) {
            uint idx=tid+t*256, j=j0+idx/32, d=d0+idx%32;
            vt[idx]=j<p.total && d<p.hd ? float(v[(p.layer*p.stride+j)*p.dim+kvh*p.hd+d]) : 0;
            uint q=q0+idx/16, key=j0+idx%16;
            st[idx]=q<p.seq && key<=p.pos+q ? scores[(h*p.seq+q)*p.total+key] : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<16;j++) {
            float2 vv=float2(vt[j*32+di],vt[j*32+di+1]);
            a0+=st[qi*16+j]*vv; a1+=st[(qi+1)*16+j]*vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint base=((q0+qi)*p.heads+h)*p.hd+d0+di;
    if(q0+qi<p.seq && d0+di<p.hd) { out[base]=a0.x; out[base+1]=a0.y; }
    if(q0+qi+1<p.seq && d0+di<p.hd) { out[base+p.heads*p.hd]=a1.x; out[base+p.heads*p.hd+1]=a1.y; }
}

kernel void resident_apply_step(device const float *scores [[buffer(0)]],
                                device const half *v [[buffer(1)]],
                                device float *partial [[buffer(2)]],
                                constant DecoderShape &p [[buffer(3)]],
                                uint3 gid [[thread_position_in_grid]]) {
    uint d=gid.x,h=gid.y,slice=gid.z;
    if(d>=p.hd) return;
    uint per=(p.total+7)/8, end=min(p.total,(slice+1)*per);
    uint vb=p.layer*p.stride*p.dim+(h/(p.heads/p.kvHeads))*p.hd+d;
    float a=0;
    for(uint j=slice*per;j<end;j++) a+=scores[h*p.total+j]*float(v[vb+j*p.dim]);
    partial[(slice*p.heads+h)*p.hd+d]=a;
}
kernel void resident_merge(device const float *partial [[buffer(0)]],
                           device float *out [[buffer(1)]],
                           constant DecoderShape &p [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    if(i>=p.dim) return;
    float a=0; for(uint s=0;s<8;s++) a+=partial[s*p.dim+i]; out[i]=a;
}
kernel void resident_add(device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
                         constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i<n) x[i]+=y[i];
}
kernel void resident_swiglu(device const float2 *x [[buffer(0)]], device float *y [[buffer(1)]],
                            constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i<n) { float2 a=x[i]; y[i]=(a.x/(1+exp(-a.x)))*a.y; }
}
struct Winner { float value; uint token; };
kernel void resident_argmax(device const float *x [[buffer(0)]],
                            device Winner *out [[buffer(1)]],
                            constant uint &n [[buffer(2)]],
                            uint group [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    float best=-INFINITY; uint token=0xffffffff;
    for(uint i=group*32+lane;i<n;i+=64*32) {
        float val=x[i]; if(val>best || (val==best && i<token)) { best=val; token=i; }
    }
    float mx=simd_max(best); uint id=simd_min(best==mx?token:0xffffffff);
    if(!lane) out[group]={mx,id};
}
kernel void resident_argmax_final(device Winner *w [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {
    Winner a=w[lane], b=w[lane+32];
    if(b.value>a.value || (b.value==a.value && b.token<a.token)) a=b;
    float mx=simd_max(a.value); uint id=simd_min(a.value==mx?a.token:0xffffffff);
    if(!lane) w[64]={mx,id};
}

// Encoder convolutions use NHWC scratch and [out, in, ky, kx] weights.
struct ConvShape { uint batch, channels, height, width, oh, ow, start, frames, mode, n; };
kernel void encoder_im2col(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                           constant ConvShape &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i>=p.n) return;
    uint col=i%(p.channels*9), row=i/(p.channels*9);
    uint ch=col/9, ky=col%9/3, kx=col%3, b=row/(p.oh*p.ow);
    int y=int(row/p.ow%p.oh)*2-1+int(ky), t=int(row%p.ow)*2-1+int(kx);
    float z=0;
    if(y>=0 && y<int(p.height) && t>=0 && t<int(p.width)) {
        uint index=p.mode ? ((b*p.height+uint(y))*p.width+uint(t))*p.channels+ch
                         : uint(y)*p.frames+p.start+b*p.width+uint(t);
        z=x[index];
    }
    out[i]=z;
}
kernel void encoder_bias(device float *x [[buffer(0)]], device const float *bias [[buffer(1)]],
                         constant DecoderShape &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i>=p.n) return;
    float z=x[i]+bias[i%p.dim];
    if(p.mode) z=z/(1+exp(-1.5957691216057308f*(z+0.044715f*z*z*z)));
    x[i]=z;
}
kernel void encoder_reshape(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                            constant ConvShape &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i>=p.n) return;
    uint d=i%7680, row=i/7680, ch=d/16, f=d%16;
    out[i]=x[((row/p.ow*16+f)*p.ow+row%p.ow)*480+ch];
}
kernel void encoder_position(device const float *x [[buffer(0)]], device const float *pe [[buffer(1)]],
                             device float *out [[buffer(2)]], constant DecoderShape &p [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
    if(i<p.n) out[p.pos*p.dim+i]=x[i]+pe[i%(p.seq*p.dim)];
}
// One SIMD group per query/head; windows are at most 104 keys (eight seconds).
kernel void encoder_attention(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
                              device const float *v [[buffer(2)]], device float *out [[buffer(3)]],
                              constant DecoderShape &p [[buffer(4)]],
                              uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float probs[128];
    uint row=group.x, head=group.y, start=row/p.stride*p.stride, end=min(p.seq,start+p.stride);
    uint qb=row*p.dim+head*64;
    float a=q[qb+lane], b=q[qb+lane+32], mx=-INFINITY;
    for(uint j=start;j<end;j++) {
        uint base=j*p.dim+head*64;
        float score=simd_sum(a*k[base+lane]+b*k[base+lane+32])*0.125f;
        if(!lane) probs[j-start]=score;
        mx=max(mx,score);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum=0;
    for(uint j=lane;j<end-start;j+=32) { float z=exp(probs[j]-mx); probs[j]=z; sum+=z; }
    float inv=1/simd_sum(sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float x=0,y=0;
    for(uint j=start;j<end;j++) {
        float z=probs[j-start]*inv; uint base=j*p.dim+head*64;
        x+=z*v[base+lane]; y+=z*v[base+lane+32];
    }
    out[qb+lane]=x; out[qb+lane+32]=y;
}
