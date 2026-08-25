/*
 * webgpu-decoder.js - Qwen3-ASR token generation on the GPU.
 *
 * Split of work: mel, the audio encoder and the decoder prefill stay in wasm.
 * Prefill is a batched GEMM the CPU handles fine; token generation reads all
 * 1.7 GB of decoder weights once per token, so it is bandwidth-bound and is the
 * part worth moving to the GPU's wider path to memory.
 *
 * Weights are uploaded once as two storage buffers holding exactly the bytes the
 * packed Q8 model already contains (int8 quants + f32 per-block scales).
 * Dequantization in the shader uses unpack4x8snorm, which is core WGSL and maps
 * to a single instruction: q/127, so multiplying by scale*127 recovers the
 * weight. An int8 dot product (dot4I8Packed) would be worse here — Apple GPUs
 * emulate it, measured at 24 GMAC/s by wasm/demo/webgpu-probe.html.
 *
 * The generated token id lives in a GPU buffer, so the embedding lookup for the
 * next step reads it without a round trip through JS.
 */

const W_Q = 0, W_K = 1, W_V = 2, W_O = 3, W_GATE_UP = 4, W_DOWN = 5, W_EMBED = 6;
const N_INPUT = 0, N_POST_ATTN = 1, N_QNORM = 2, N_KNORM = 3, N_FINAL = 4;

const PARAM_STRIDE = 256;   // >= minUniformBufferOffsetAlignment
const PARAM_FIELDS = 16;

/* ------------------------------------------------------------------ shaders */

const HEADER = `
struct Params {
  wordBase  : u32,
  scaleBase : u32,
  rows      : u32,
  cols      : u32,
  xOff      : u32,
  yOff      : u32,
  pos       : u32,
  n         : u32,
  a         : u32,
  b         : u32,
  c         : u32,
  d         : u32,
  fa        : f32,
  fb        : f32,
  fc        : f32,
  fd        : f32,
};
@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read> quants : array<u32>;
@group(0) @binding(2) var<storage, read> scales : array<f32>;
@group(0) @binding(3) var<storage, read> norms  : array<f32>;
@group(0) @binding(4) var<storage, read_write> act : array<f32>;
@group(0) @binding(5) var<storage, read_write> kv  : array<f32>;
@group(0) @binding(6) var<storage, read_write> tok : array<u32>;
@group(0) @binding(7) var<storage, read_write> scratch : array<f32>;

/* Sign-extend the four int8 lanes of a packed u32. */
fn i8x4(v : u32) -> vec4<i32> {
  return vec4<i32>(bitcast<i32>(v << 24u) >> 24u,
                   bitcast<i32>(v << 16u) >> 24u,
                   bitcast<i32>(v << 8u)  >> 24u,
                   bitcast<i32>(v)        >> 24u);
}

/* Row dot product against a *quantized* activation.
 *
 * The activation is quantized to int8 blocks here exactly as the CPU kernel
 * does, rather than kept in f32. Keeping it in f32 is more accurate in
 * isolation, but it makes the GPU disagree with the CPU by ~1e-4 relative — and
 * Japanese comes out of this model as byte-fragment tokens whose logits sit
 * within that margin, so individual CJK glyphs came out corrupted while English
 * matched exactly even at 1170 tokens of context. Matching the CPU's arithmetic
 * is what makes the two backends agree.
 *
 * Lane-per-word keeps adjacent lanes on adjacent u32, which is what Apple GPUs
 * coalesce best; the two scales are re-read per word but are the same 4 bytes
 * for 16 consecutive words and stay in cache.
 */
fn q8row(wordBase : u32, scaleBase : u32, nwords : u32, aqBase : u32, asBase : u32,
         lane : u32, wgSize : u32) -> f32 {
  var acc : f32 = 0.0;
  var w : u32 = lane;
  loop {
    if (w >= nwords) { break; }
    let b = w / 16u;
    let sc = scales[scaleBase + b] * act[asBase + b];
    let d = dot(i8x4(quants[wordBase + w]), i8x4(tok[aqBase + w]));
    acc = acc + f32(d) * sc;
    w = w + wgSize;
  }
  return acc;
}

/* f32-activation variant: skips activation quantization entirely, so it needs
 * one dispatch fewer per matvec and is more accurate in isolation, but it no
 * longer reproduces the CPU's arithmetic. Selectable so the two can be compared
 * on real audio. */
fn q8rowF(wordBase : u32, scaleBase : u32, nwords : u32, xOff : u32,
          lane : u32, wgSize : u32) -> f32 {
  var acc : f32 = 0.0;
  var w : u32 = lane;
  loop {
    if (w >= nwords) { break; }
    let sc = scales[scaleBase + (w / 16u)] * 127.0;
    let q = unpack4x8snorm(quants[wordBase + w]);
    let xi = xOff + w * 4u;
    acc = acc + dot(q, vec4<f32>(act[xi], act[xi + 1u], act[xi + 2u], act[xi + 3u])) * sc;
    w = w + wgSize;
  }
  return acc;
}

`;

/* y[row] = W[row] . x, with `pos` reused as the accumulate flag. */
const MATVEC_WGSL = HEADER + `
const WG : u32 = $WG$u;
var<workgroup> red : array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let row = wid.x + wid.y * P.d;
  if (row >= P.rows) { return; }
  let nwords = P.cols / 4u;
  red[lid.x] = $ROWFN$;
  workgroupBarrier();
  var s : u32 = WG / 2u;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid.x == 0u) {
    let o = P.yOff + row;
    if (P.pos == 1u) { act[o] = act[o] + red[0]; } else { act[o] = red[0]; }
  }
}
`;

/* Same reduction, but the result goes to scratch as a logit. */
const LOGITS_WGSL = HEADER + `
const WG : u32 = $WG$u;
var<workgroup> red : array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let row = wid.x + wid.y * P.d;
  if (row >= P.rows) { return; }
  let nwords = P.cols / 4u;
  red[lid.x] = $ROWFN$;
  workgroupBarrier();
  var s : u32 = WG / 2u;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid.x == 0u) { scratch[row] = red[0]; }
}
`;

const RMSNORM_WGSL = HEADER + `
const WG : u32 = 256u;
var<workgroup> red : array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid : vec3<u32>) {
  var acc : f32 = 0.0;
  var i : u32 = lid.x;
  loop {
    if (i >= P.n) { break; }
    let v = act[P.xOff + i];
    acc = acc + v * v;
    i = i + WG;
  }
  red[lid.x] = acc;
  workgroupBarrier();
  var s : u32 = WG / 2u;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let inv = 1.0 / sqrt(red[0] / f32(P.n) + P.fa);
  i = lid.x;
  loop {
    if (i >= P.n) { break; }
    act[P.yOff + i] = act[P.xOff + i] * inv * norms[P.a + i];
    i = i + WG;
  }
}
`;

/* Per-head RMS norm + NeoX RoPE on Q and K, then append K/V to the cache.
 * Workgroups [0, n) are Q heads, [n, n+rows) are KV heads.
 *   a = q offset, b = k offset, c = v offset, d = headDim
 *   n = qHeads, rows = kvHeads, cols = kvDim
 *   xOff/yOff = q/k norm weight bases
 *   scaleBase = K cache base, wordBase = V cache base (both f32 indices)
 *   pos = position, fa = eps, fb = RoPE table base in `norms`
 *
 * RoPE angles grow with position (pos * freq, and freq is 1 for the first
 * lane), so by a few hundred tokens the argument to cos/sin is in the hundreds
 * of radians. WGSL only guarantees accuracy for small arguments and Metal's
 * fast argument reduction loses enough precision there to flip occasional
 * argmax decisions — it corrupted individual CJK glyphs past ~200 tokens. The
 * table is built with double-precision Math.cos/Math.sin on the CPU instead.
 */
const QKROPE_WGSL = HEADER + `
const WG : u32 = 128u;
var<workgroup> red : array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let head = wid.x;
  let hd = P.d;
  let isQ = head < P.n;
  let base = select(P.b + (head - P.n) * hd, P.a + head * hd, isQ);
  let nbase = select(P.yOff, P.xOff, isQ);

  var acc : f32 = 0.0;
  if (lid.x < hd) { let v = act[base + lid.x]; acc = v * v; }
  red[lid.x] = acc;
  workgroupBarrier();
  var s : u32 = WG / 2u;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let inv = 1.0 / sqrt(red[0] / f32(hd) + P.fa);
  if (lid.x < hd) {
    act[base + lid.x] = act[base + lid.x] * inv * norms[nbase + lid.x];
  }
  workgroupBarrier();

  let half = hd / 2u;
  if (lid.x < half) {
    let t = u32(P.fb) + P.pos * hd + lid.x * 2u;
    let c = norms[t];
    let sn = norms[t + 1u];
    let x1 = act[base + lid.x];
    let x2 = act[base + half + lid.x];
    act[base + lid.x] = x1 * c - x2 * sn;
    act[base + half + lid.x] = x2 * c + x1 * sn;
  }
  workgroupBarrier();

  if (!isQ && lid.x < hd) {
    let kvh = head - P.n;
    let slot = P.pos * P.cols + kvh * hd + lid.x;
    kv[P.scaleBase + slot] = act[base + lid.x];
    kv[P.wordBase + slot] = act[P.c + kvh * hd + lid.x];
  }
}
`;

const ATTN_SCORES_WGSL = HEADER + `
const WG : u32 = 64u;
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let h = gid.y;
  let j = gid.x;
  if (h >= P.rows || j >= P.n) { return; }
  let qb = P.a + h * P.d;
  let kb = P.scaleBase + j * P.cols + (h / P.b) * P.d;
  var acc : f32 = 0.0;
  for (var d : u32 = 0u; d < P.d; d = d + 1u) {
    acc = acc + act[qb + d] * kv[kb + d];
  }
  scratch[P.yOff + h * P.c + j] = acc * P.fa;
}
`;

const ATTN_SOFTMAX_WGSL = HEADER + `
const WG : u32 = 256u;
var<workgroup> rmax : array<f32, WG>;
var<workgroup> rsum : array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let base = P.yOff + wid.x * P.c;
  var m : f32 = -3.0e38;
  var i : u32 = lid.x;
  loop {
    if (i >= P.n) { break; }
    m = max(m, scratch[base + i]);
    i = i + WG;
  }
  rmax[lid.x] = m;
  workgroupBarrier();
  var s : u32 = WG / 2u;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { rmax[lid.x] = max(rmax[lid.x], rmax[lid.x + s]); }
    workgroupBarrier();
    s = s / 2u;
  }
  let mx = rmax[0];

  var sum : f32 = 0.0;
  i = lid.x;
  loop {
    if (i >= P.n) { break; }
    let e = exp(scratch[base + i] - mx);
    scratch[base + i] = e;
    sum = sum + e;
    i = i + WG;
  }
  rsum[lid.x] = sum;
  workgroupBarrier();
  s = WG / 2u;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { rsum[lid.x] = rsum[lid.x] + rsum[lid.x + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let insum = 1.0 / max(rsum[0], 1.0e-30);
  i = lid.x;
  loop {
    if (i >= P.n) { break; }
    scratch[base + i] = scratch[base + i] * insum;
    i = i + WG;
  }
}
`;

/* Weighted sum of V, split over the key axis.
 *
 * One thread per (head, dim) leaves only 2048 threads running a loop over the
 * whole context — far too little to fill an Apple GPU, and it dominated the step
 * once the context passed a few hundred tokens. Each workgroup now covers a
 * slice of keys and writes a partial sum; a second pass adds the slices.
 *   a = partial-sum base in scratch, fa = number of slices
 */
const ATTN_APPLY_WGSL = HEADER + `
const WG : u32 = 64u;
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(workgroup_id) wid : vec3<u32>) {
  let h = gid.y;
  let d = gid.x;
  let slice = wid.z;
  let nslice = u32(P.fa);
  if (h >= P.rows || d >= P.d) { return; }

  let per = (P.n + nslice - 1u) / nslice;
  let j0 = slice * per;
  var j1 = j0 + per;
  if (j1 > P.n) { j1 = P.n; }

  let sb = P.yOff + h * P.c;
  let vb = P.scaleBase + (h / P.b) * P.d + d;
  var acc : f32 = 0.0;
  var j : u32 = j0;
  loop {
    if (j >= j1) { break; }
    acc = acc + scratch[sb + j] * kv[vb + j * P.cols];
    j = j + 1u;
  }
  scratch[P.a + (slice * P.rows + h) * P.d + d] = acc;
}
`;

/* Add the per-slice partials into the attention output. */
const ATTN_MERGE_WGSL = HEADER + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let h = gid.y;
  let d = gid.x;
  if (h >= P.rows || d >= P.d) { return; }
  let nslice = u32(P.fa);
  var acc : f32 = 0.0;
  for (var s : u32 = 0u; s < nslice; s = s + 1u) {
    acc = acc + scratch[P.a + (s * P.rows + h) * P.d + d];
  }
  act[P.xOff + h * P.d + d] = acc;
}
`;

/* n = element count, xOff = source, a = packed-word base in tok,
 * b = scale base in act */
const QUANTACT_WGSL = HEADER + `
var<workgroup> sh : array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let base = P.xOff + wid.x * 64u;
  sh[lid.x] = abs(act[base + lid.x]);
  workgroupBarrier();
  var s : u32 = 32u;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) { sh[lid.x] = max(sh[lid.x], sh[lid.x + s]); }
    workgroupBarrier();
    s = s / 2u;
  }
  let scale = sh[0] / 127.0;
  let inv = select(0.0, 1.0 / scale, scale > 0.0);
  if (lid.x == 0u) { act[P.b + wid.x] = scale; }
  if (lid.x < 16u) {
    var word : u32 = 0u;
    for (var k : u32 = 0u; k < 4u; k = k + 1u) {
      let q = clamp(i32(round(act[base + lid.x * 4u + k] * inv)), -127, 127);
      word = word | ((bitcast<u32>(q) & 0xffu) << (k * 8u));
    }
    tok[P.a + wid.x * 16u + lid.x] = word;
  }
}
`;

const SWIGLU_WGSL = HEADER + `
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let g = act[P.xOff + 2u * i];
  let u = act[P.xOff + 2u * i + 1u];
  act[P.yOff + i] = (g / (1.0 + exp(-g))) * u;
}
`;

const EMBED_WGSL = HEADER + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let w = gid.x;
  let nwords = P.cols / 4u;
  if (w >= nwords) { return; }
  let row = tok[0];
  let sc = scales[P.scaleBase + row * (P.cols / 64u) + (w / 16u)] * 127.0;
  let q = unpack4x8snorm(quants[P.wordBase + row * nwords + w]) * sc;
  let o = P.yOff + w * 4u;
  act[o] = q.x;
  act[o + 1u] = q.y;
  act[o + 2u] = q.z;
  act[o + 3u] = q.w;
}
`;

const ARGMAX_WGSL = HEADER + `
const WG : u32 = 256u;
var<workgroup> rv : array<f32, WG>;
var<workgroup> ri : array<u32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid : vec3<u32>) {
  var bv : f32 = -3.0e38;
  var bi : u32 = 0u;
  var i : u32 = lid.x;
  loop {
    if (i >= P.n) { break; }
    let v = scratch[i];
    if (v > bv) { bv = v; bi = i; }
    i = i + WG;
  }
  rv[lid.x] = bv;
  ri[lid.x] = bi;
  workgroupBarrier();
  var s : u32 = WG / 2u;
  loop {
    if (s == 0u) { break; }
    if (lid.x < s) {
      if (rv[lid.x + s] > rv[lid.x]) {
        rv[lid.x] = rv[lid.x + s];
        ri[lid.x] = ri[lid.x + s];
      }
    }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid.x == 0u) { tok[0] = ri[0]; }
}
`;

/* ==================================================================
 * Prefill kernels
 *
 * Prefill is a batched GEMM, so unlike generation it is compute-bound, not
 * bandwidth-bound: 549 prompt tokens through the 1.7B decoder is ~1.55 TFLOP,
 * which the wasm build grinds through at ~220 GFLOPS while this GPU is good for
 * several TFLOPS. It was 7.2 s of a 15 s transcription.
 *
 * Prefill activations are stored **transposed**, [dim][seqPad]. Every kernel
 * then has its lanes running along the sequence axis, which makes all global
 * reads and writes coalesced, and turns row-wise reductions (RMS norm, per-head
 * norm) into per-lane loops with no cross-lane traffic at all.
 * ================================================================== */

/* C^T[rows][seqPad] = W[rows][cols] (Q8) @ X^T[cols][seqPad]
 *
 * 64x64 output tile per workgroup, 16x16 threads each owning a 4x4 block, with
 * the weight and activation tiles staged in workgroup memory. Tiling in the row
 * dimension is what keeps the weights from being re-read once per sequence
 * position.
 *   a = seqPad, n = seq, rows/cols = W shape, pos = 1 to accumulate
 */
const PRE_MATMUL_WGSL = HEADER + `
const TR : u32 = 64u;
const TS : u32 = 64u;
const TK : u32 = 16u;

var<workgroup> ws : array<f32, 1024>;   // TR x TK
var<workgroup> xs : array<f32, 1024>;   // TK x TS

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let r0 = wid.y * TR;
  let s0 = wid.x * TS;
  let tid = lid.y * 16u + lid.x;
  let nwords = P.cols / 4u;
  let nblocks = P.cols / 64u;

  var acc : array<f32, 16>;
  for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

  var kb : u32 = 0u;
  loop {
    if (kb >= P.cols) { break; }

    // Weight tile: 256 threads, one u32 (4 columns) each.
    {
      let rr = tid / 4u;
      let wq = tid % 4u;
      let row = r0 + rr;
      let col0 = kb + wq * 4u;
      var v = vec4<f32>(0.0);
      if (row < P.rows) {
        let word = quants[P.wordBase + row * nwords + col0 / 4u];
        let sc = scales[P.scaleBase + row * nblocks + col0 / 64u];
        v = vec4<f32>(i8x4(word)) * sc;
      }
      let base = rr * TK + wq * 4u;
      ws[base] = v.x;
      ws[base + 1u] = v.y;
      ws[base + 2u] = v.z;
      ws[base + 3u] = v.w;
    }

    // Activation tile: 256 threads, 4 values each.
    for (var t = 0u; t < 4u; t = t + 1u) {
      let idx = tid + t * 256u;
      let kk = idx / TS;
      let ss = idx % TS;
      let col = kb + kk;
      let sq = s0 + ss;
      var val : f32 = 0.0;
      if (col < P.cols && sq < P.n) { val = act[P.xOff + col * P.a + sq]; }
      xs[kk * TS + ss] = val;
    }
    workgroupBarrier();

    for (var k = 0u; k < TK; k = k + 1u) {
      let xb = k * TS + lid.x * 4u;
      let x0 = xs[xb];
      let x1 = xs[xb + 1u];
      let x2 = xs[xb + 2u];
      let x3 = xs[xb + 3u];
      for (var i = 0u; i < 4u; i = i + 1u) {
        let w = ws[(lid.y * 4u + i) * TK + k];
        acc[i * 4u] = acc[i * 4u] + w * x0;
        acc[i * 4u + 1u] = acc[i * 4u + 1u] + w * x1;
        acc[i * 4u + 2u] = acc[i * 4u + 2u] + w * x2;
        acc[i * 4u + 3u] = acc[i * 4u + 3u] + w * x3;
      }
    }
    workgroupBarrier();
    kb = kb + TK;
  }

  for (var i = 0u; i < 4u; i = i + 1u) {
    let row = r0 + lid.y * 4u + i;
    if (row >= P.rows) { continue; }
    for (var j = 0u; j < 4u; j = j + 1u) {
      let sq = s0 + lid.x * 4u + j;
      if (sq >= P.n) { continue; }
      let o = P.yOff + row * P.a + sq;
      if (P.pos == 1u) { act[o] = act[o] + acc[i * 4u + j]; } else { act[o] = acc[i * 4u + j]; }
    }
  }
}
`;

/* RMS norm down each column. n = dim, a = seqPad, b = seq, c = norm base. */
const PRE_RMSNORM_WGSL = HEADER + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let sq = gid.x;
  if (sq >= P.b) { return; }
  var acc : f32 = 0.0;
  for (var d = 0u; d < P.n; d = d + 1u) {
    let v = act[P.xOff + d * P.a + sq];
    acc = acc + v * v;
  }
  let inv = 1.0 / sqrt(acc / f32(P.n) + P.fa);
  for (var d = 0u; d < P.n; d = d + 1u) {
    act[P.yOff + d * P.a + sq] = act[P.xOff + d * P.a + sq] * inv * norms[P.c + d];
  }
}
`;

/* Per-head RMS norm + RoPE for every position.
 *   workgroup.y selects the head: [0, n) are Q heads, then the KV heads
 *   a = seqPad, b = seq, c = q offset, d = headDim, rows = kvHeads
 *   xOff/yOff = q/k norm bases, wordBase = k offset, fb = RoPE table base
 */
const PRE_QKROPE_WGSL = HEADER + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let sq = gid.x;
  let head = gid.y;
  if (sq >= P.b) { return; }
  let hd = P.d;
  let isQ = head < P.n;
  let base = select(P.wordBase + (head - P.n) * hd, P.c + head * hd, isQ);
  let nbase = select(P.yOff, P.xOff, isQ);

  var acc : f32 = 0.0;
  for (var d = 0u; d < hd; d = d + 1u) {
    let v = act[(base + d) * P.a + sq];
    acc = acc + v * v;
  }
  let inv = 1.0 / sqrt(acc / f32(hd) + P.fa);
  for (var d = 0u; d < hd; d = d + 1u) {
    let i = (base + d) * P.a + sq;
    act[i] = act[i] * inv * norms[nbase + d];
  }

  let half = hd / 2u;
  let tbase = u32(P.fb) + sq * hd;
  for (var d = 0u; d < half; d = d + 1u) {
    let c = norms[tbase + d * 2u];
    let sn = norms[tbase + d * 2u + 1u];
    let i1 = (base + d) * P.a + sq;
    let i2 = (base + half + d) * P.a + sq;
    let x1 = act[i1];
    let x2 = act[i2];
    act[i1] = x1 * c - x2 * sn;
    act[i2] = x2 * c + x1 * sn;
  }
}
`;

/* Copy the transposed K/V into the cache's [pos][kvDim] layout.
 *   a = seqPad, b = seq, cols = kvDim, c = kT offset, d = vT offset
 *   scaleBase = K cache base, wordBase = V cache base
 */
const PRE_KVSTORE_WGSL = HEADER + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let d = gid.x;
  if (d >= P.cols) { return; }
  for (var p = 0u; p < P.b; p = p + 1u) {
    kv[P.scaleBase + p * P.cols + d] = act[(P.c + d) * P.a + p];
    kv[P.wordBase + p * P.cols + d] = act[(P.d + d) * P.a + p];
  }
}
`;

/* Causal scores: one workgroup row per query, lanes along the key axis.
 *   a = seqPad, b = seq, c = qT offset, d = headDim, rows = heads
 *   wordBase = kT offset, scaleBase = score base, n = score stride, fa = scale
 */
const PRE_SCORES_WGSL = HEADER + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let j = gid.x;
  let i = gid.y;
  let h = gid.z;
  if (i >= P.b || j > i) { return; }
  let qb = P.c + h * P.d;
  let kb = P.wordBase + (h / P.rows) * P.d;
  var acc : f32 = 0.0;
  for (var d = 0u; d < P.d; d = d + 1u) {
    acc = acc + act[(qb + d) * P.a + i] * act[(kb + d) * P.a + j];
  }
  scratch[P.scaleBase + (h * P.n + i) * P.n + j] = acc * P.fa;
}
`;

/* Softmax over the causal prefix of each query row. */
const PRE_SOFTMAX_WGSL = HEADER + `
const WG : u32 = 128u;
var<workgroup> rm : array<f32, WG>;
var<workgroup> rs : array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let i = wid.x;
  let h = wid.y;
  if (i >= P.b) { return; }
  let base = P.scaleBase + (h * P.n + i) * P.n;
  let len = i + 1u;

  var m : f32 = -3.0e38;
  var t : u32 = lid.x;
  loop {
    if (t >= len) { break; }
    m = max(m, scratch[base + t]);
    t = t + WG;
  }
  rm[lid.x] = m;
  workgroupBarrier();
  var st : u32 = WG / 2u;
  loop {
    if (st == 0u) { break; }
    if (lid.x < st) { rm[lid.x] = max(rm[lid.x], rm[lid.x + st]); }
    workgroupBarrier();
    st = st / 2u;
  }
  let mx = rm[0];

  var sum : f32 = 0.0;
  t = lid.x;
  loop {
    if (t >= len) { break; }
    let e = exp(scratch[base + t] - mx);
    scratch[base + t] = e;
    sum = sum + e;
    t = t + WG;
  }
  rs[lid.x] = sum;
  workgroupBarrier();
  st = WG / 2u;
  loop {
    if (st == 0u) { break; }
    if (lid.x < st) { rs[lid.x] = rs[lid.x] + rs[lid.x + st]; }
    workgroupBarrier();
    st = st / 2u;
  }
  let insum = 1.0 / max(rs[0], 1.0e-30);
  t = lid.x;
  loop {
    if (t >= len) { break; }
    scratch[base + t] = scratch[base + t] * insum;
    t = t + WG;
  }
}
`;

/* Weighted V sum: one workgroup per (head, query), lanes along head_dim.
 *   a = seqPad, b = seq, c = vT offset, d = headDim, rows = headsPerKv
 *   yOff = attnT offset, scaleBase = score base, n = score stride
 */
const PRE_APPLY_WGSL = HEADER + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(workgroup_id) wid : vec3<u32>) {
  let d = gid.x;
  let i = wid.y;
  let h = wid.z;
  if (d >= P.d || i >= P.b) { return; }
  let base = P.scaleBase + (h * P.n + i) * P.n;
  let vb = P.c + (h / P.rows) * P.d + d;
  var acc : f32 = 0.0;
  for (var j = 0u; j <= i; j = j + 1u) {
    acc = acc + scratch[base + j] * act[vb * P.a + j];
  }
  act[(P.yOff + h * P.d + d) * P.a + i] = acc;
}
`;

/* SwiGLU on the transposed layout: gate/up rows are interleaved. */
const PRE_SWIGLU_WGSL = HEADER + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let sq = gid.x;
  let r = gid.y;
  if (sq >= P.b || r >= P.n) { return; }
  let g = act[P.xOff + (2u * r) * P.a + sq];
  let u = act[P.xOff + (2u * r + 1u) * P.a + sq];
  act[P.yOff + r * P.a + sq] = (g / (1.0 + exp(-g))) * u;
}
`;

/* Pull one column of the transposed prefill state into the generation buffer. */
const PRE_EXTRACT_WGSL = HEADER + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let d = gid.x;
  if (d >= P.n) { return; }
  act[P.yOff + d] = act[P.xOff + d * P.a + P.pos];
}
`;

/* ------------------------------------------------------------------ backend */

export class WebGPUDecoder {
  constructor(Module) {
    this.M = Module;
    this.ready = false;
  }

  static async probe() {
    if (!navigator.gpu) return { ok: false, why: "navigator.gpu is missing" };
    try {
      const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!a) return { ok: false, why: "no adapter" };
      return {
        ok: true,
        maxBuffer: a.limits.maxBufferSize,
        maxBinding: a.limits.maxStorageBufferBindingSize,
        info: a.info || {},
      };
    } catch (e) {
      return { ok: false, why: e.message };
    }
  }

  async init(report = () => {}) {
    const M = this.M;

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const lim = adapter.limits;

    /* ---- shape ---- */
    const shPtr = M._qwen_wasm_alloc(10 * 4) >>> 0;
    if (M._qwen_wasm_model_shape(shPtr) < 0) throw new Error("model shape unavailable");
    const sh = new Int32Array(M.HEAPU8.buffer, shPtr, 10).slice();
    M._qwen_wasm_release(shPtr);
    const cfg = {
      layers: sh[0], hidden: sh[1], heads: sh[2], kvHeads: sh[3], headDim: sh[4],
      inter: sh[5], vocab: sh[6], imEnd: sh[7], endOfText: sh[8], asrText: sh[9],
    };
    cfg.qDim = cfg.heads * cfg.headDim;
    cfg.kvDim = cfg.kvHeads * cfg.headDim;
    cfg.headsPerKv = cfg.heads / cfg.kvHeads;
    this.cfg = cfg;
    this.eps = M._qwen_wasm_rms_eps();
    this.theta = M._qwen_wasm_rope_theta();

    /* ---- weight tables ---- */
    const MAXD = 8 * cfg.layers + 8;
    const dPtr = M._qwen_wasm_alloc(MAXD * 8 * 4) >>> 0;
    const nQ = M._qwen_wasm_q8_desc(dPtr, MAXD);
    if (nQ < 0) {
      M._qwen_wasm_release(dPtr);
      throw new Error("decoder is not Q8 quantized (use the packed model)");
    }
    const qd = new Uint32Array(M.HEAPU8.buffer, dPtr, nQ * 6).slice();
    const nF = M._qwen_wasm_f32_desc(dPtr, MAXD * 2);
    if (nF < 0) {
      M._qwen_wasm_release(dPtr);
      throw new Error("norm weights unavailable");
    }
    const fd = new Uint32Array(M.HEAPU8.buffer, dPtr, nF * 4).slice();
    M._qwen_wasm_release(dPtr);

    let quantBytes = 0, scaleFloats = 0;
    const wmap = new Map();
    for (let i = 0; i < nQ; i++) {
      const [kind, layer, rows, cols, qptr, sptr] = qd.subarray(i * 6, i * 6 + 6);
      const nq = rows * cols;
      wmap.set(`${kind}:${layer}`, {
        rows, cols, qptr, sptr, nq,
        wordBase: quantBytes / 4, scaleBase: scaleFloats,
      });
      quantBytes += nq;
      scaleFloats += nq / 64;
    }
    if (quantBytes > lim.maxStorageBufferBindingSize) {
      throw new Error(`weights need ${(quantBytes / 1e9).toFixed(2)} GB, adapter caps ` +
                      `storage bindings at ${(lim.maxStorageBufferBindingSize / 1e9).toFixed(2)} GB`);
    }

    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: Math.max(quantBytes, 1 << 28),
        maxStorageBufferBindingSize: Math.max(quantBytes, 1 << 28),
      },
    });
    this.device = device;
    this.maxDim = device.limits.maxComputeWorkgroupsPerDimension;
    this.adapterInfo = adapter.info || {};
    /* Surface these: a missing buffer usage flag makes writeBuffer a silent
     * no-op, and the symptom is a kernel that runs fast and returns zeros. */
    this.onError = report;
    device.onuncapturederror = (e) => {
      console.error("gpu:", e.error.message);
      if (this.onError) this.onError("gpu error: " + e.error.message);
    };

    /* ---- upload ---- */
    report(`allocating ${(quantBytes / 1e9).toFixed(2)} GB on the GPU...`);
    this.bufQuant = device.createBuffer({ size: quantBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.bufScale = device.createBuffer({ size: scaleFloats * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

    const CH = 64 << 20;
    let done = 0, lastReport = 0;
    for (const w of wmap.values()) {
      for (let off = 0; off < w.nq; off += CH) {
        const n = Math.min(CH, w.nq - off);
        device.queue.writeBuffer(this.bufQuant, w.wordBase * 4 + off, M.HEAPU8, w.qptr + off, n);
        done += n;
      }
      device.queue.writeBuffer(this.bufScale, w.scaleBase * 4, M.HEAPU8, w.sptr, (w.nq / 64) * 4);
      if (done - lastReport > (256 << 20)) {
        lastReport = done;
        report(`uploading weights ${(done / 1e9).toFixed(2)} / ${(quantBytes / 1e9).toFixed(2)} GB`);
        await device.queue.onSubmittedWorkDone();
      }
    }
    await device.queue.onSubmittedWorkDone();

    let normFloats = 0;
    const nmap = new Map();
    for (let i = 0; i < nF; i++) {
      const [kind, layer, count, ptr] = fd.subarray(i * 4, i * 4 + 4);
      nmap.set(`${kind}:${layer}`, { base: normFloats, count, ptr });
      normFloats += count;
    }
    /* The RoPE cos/sin table lives at the end of this buffer so the shaders do
     * not need a ninth binding (WebGPU guarantees only 8 storage buffers). */
    this.ropeBase = normFloats;
    this.ropeMaxSeq = 8192;
    const normBytes = (normFloats + this.ropeMaxSeq * cfg.headDim) * 4;
    this.bufNorm = device.createBuffer({ size: normBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    for (const n of nmap.values())
      device.queue.writeBuffer(this.bufNorm, n.base * 4, M.HEAPU8, n.ptr, n.count * 4);
    this.uploadRopeTable();

    this.wmap = wmap;
    this.nmap = nmap;
    this.weightBytes = quantBytes + scaleFloats * 4;

    /* ---- activations ---- */
    const A = {};
    let ao = 0;
    const slot = (k, n) => { A[k] = ao; ao += n; };
    slot("x", cfg.hidden);
    slot("xn", cfg.hidden);
    slot("q", cfg.qDim);
    slot("k", cfg.kvDim);
    slot("v", cfg.kvDim);
    slot("attn", cfg.qDim);
    slot("gu", 2 * cfg.inter);
    slot("g", cfg.inter);
    /* Per-block scales for the quantized copies of the vectors that feed a
     * matvec. Packed int8 words live in the `tok` buffer. */
    slot("sxn", cfg.hidden / 64);
    slot("sattn", cfg.qDim / 64);
    slot("sg", cfg.inter / 64);
    this.A = A;
    this.actGenFloats = ao;
    /* Row offsets of the transposed prefill arena, appended to the same buffer
     * once the sequence length is known (see prepareContext). */
    let pr = 0;
    const prow = (k, rows) => { this.PT_[k] = pr; pr += rows; };
    this.PT_ = {};
    prow("x", cfg.hidden);
    prow("xn", cfg.hidden);
    prow("qkv", cfg.qDim + 2 * cfg.kvDim);
    prow("attn", cfg.qDim);
    prow("gu", 2 * cfg.inter);
    prow("g", cfg.inter);
    this.preRows = pr;
    this.Q = {
      xn: 4,
      attn: 4 + cfg.hidden / 4,
      g: 4 + cfg.hidden / 4 + cfg.qDim / 4,
    };
    this.tokWords = this.Q.g + cfg.inter / 4;
    /* Sized for generation only until prepareContext() learns the prompt
     * length and grows it to hold the prefill arena as well. */
    this.bufAct = device.createBuffer({
      size: ao * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.actFloats = ao;

    this.bufTok = device.createBuffer({
      size: this.tokWords * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.bufTokRead = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    /* ---- pipelines ---- */
    const layout = this.pipelineLayout();
    /* Keep the modules so their compilation diagnostics can be surfaced: an
     * invalid pipeline silently turns every dispatch into a no-op, which shows
     * up as "impossibly fast and wrong" rather than as an error. */
    const modules = [];
    const mk = (code, name) => {
      const module = device.createShaderModule({ code, label: name });
      modules.push([name, module]);
      return device.createComputePipeline({
        label: name, layout, compute: { module, entryPoint: "main" },
      });
    };
    /* The best reduction width for a GEMV is hardware dependent, so build a few
     * and let the caller pick (see setMatvecWidth). */
    const ROW_Q8 = "q8row(P.wordBase + row * nwords, P.scaleBase + row * (P.cols / 64u), " +
                   "nwords, P.a, P.b, lid.x, WG)";
    const ROW_F32 = "q8rowF(P.wordBase + row * nwords, P.scaleBase + row * (P.cols / 64u), " +
                    "nwords, P.xOff, lid.x, WG)";
    this.matvecPipes = { q8: {}, f32: {} };
    this.logitsPipes = { q8: {}, f32: {} };
    for (const [mode, rowfn] of [["q8", ROW_Q8], ["f32", ROW_F32]]) {
      for (const wg of [32, 64, 128, 256]) {
        const sub = (code) => code.replace("$WG$", String(wg)).replace("$ROWFN$", rowfn);
        this.matvecPipes[mode][wg] = mk(sub(MATVEC_WGSL), `matvec_${mode}_${wg}`);
        this.logitsPipes[mode][wg] = mk(sub(LOGITS_WGSL), `logits_${mode}_${wg}`);
      }
    }
    this.matvecWidth = 64;
    /* f32 activations by default: one dispatch fewer per matvec (measured 25.6
     * against 30.5 ms/token) and closer to the bf16 reference. The int8 mode
     * exists for bit-comparability with the CPU decoder. */
    this.quantizeActivations = false;

    this.pipe = {
      matvec: this.matvecPipes.f32[64],
      logits: this.logitsPipes.f32[64],
      rmsnorm: mk(RMSNORM_WGSL, "rmsnorm"),
      qkrope: mk(QKROPE_WGSL, "qkrope"),
      scores: mk(ATTN_SCORES_WGSL, "scores"),
      softmax: mk(ATTN_SOFTMAX_WGSL, "softmax"),
      apply: mk(ATTN_APPLY_WGSL, "apply"),
      merge: mk(ATTN_MERGE_WGSL, "merge"),
      swiglu: mk(SWIGLU_WGSL, "swiglu"),
      quantAct: mk(QUANTACT_WGSL, "quantAct"),
      preMatmul: mk(PRE_MATMUL_WGSL, "preMatmul"),
      preRms: mk(PRE_RMSNORM_WGSL, "preRms"),
      preQkRope: mk(PRE_QKROPE_WGSL, "preQkRope"),
      preKvStore: mk(PRE_KVSTORE_WGSL, "preKvStore"),
      preScores: mk(PRE_SCORES_WGSL, "preScores"),
      preSoftmax: mk(PRE_SOFTMAX_WGSL, "preSoftmax"),
      preApply: mk(PRE_APPLY_WGSL, "preApply"),
      preSwiglu: mk(PRE_SWIGLU_WGSL, "preSwiglu"),
      preExtract: mk(PRE_EXTRACT_WGSL, "preExtract"),
      embed: mk(EMBED_WGSL, "embed"),
      argmax: mk(ARGMAX_WGSL, "argmax"),
    };

    let shaderErrors = 0;
    for (const [name, module] of modules) {
      const info = await module.getCompilationInfo();
      for (const m of info.messages) {
        if (m.type !== "error") continue;
        shaderErrors++;
        report(`shader ${name}:${m.lineNum}: ${m.message}`);
        console.error(`shader ${name}:${m.lineNum}:${m.linePos}: ${m.message}`);
      }
    }
    if (shaderErrors) throw new Error(`${shaderErrors} shader compilation error(s)`);

    this.ready = true;
    report("GPU decoder ready");
  }

  setMatvecWidth(wg) {
    if (!this.matvecPipes.q8[wg]) return false;
    this.matvecWidth = wg;
    this.applyMode();
    return true;
  }

  /* q8: quantize the activation like the CPU does (bit-comparable output).
   * f32: keep it in f32 (one dispatch fewer per matvec, slightly different). */
  setQuantizeActivations(on) {
    this.quantizeActivations = !!on;
    this.applyMode();
  }

  applyMode() {
    const mode = this.quantizeActivations ? "q8" : "f32";
    this.pipe.matvec = this.matvecPipes[mode][this.matvecWidth];
    this.pipe.logits = this.logitsPipes[mode][this.matvecWidth];
  }

  /* [pos][d] -> (cos, sin) interleaved, matching the decoder's NeoX layout. */
  uploadRopeTable() {
    const { cfg } = this;
    const half = cfg.headDim / 2;
    const inv = new Float64Array(half);
    for (let d = 0; d < half; d++) inv[d] = 1 / Math.pow(this.theta, (2 * d) / cfg.headDim);

    const table = new Float32Array(this.ropeMaxSeq * cfg.headDim);
    for (let pos = 0; pos < this.ropeMaxSeq; pos++) {
      const row = pos * cfg.headDim;
      for (let d = 0; d < half; d++) {
        const a = pos * inv[d];
        table[row + d * 2] = Math.cos(a);
        table[row + d * 2 + 1] = Math.sin(a);
      }
    }
    this.device.queue.writeBuffer(this.bufNorm, this.ropeBase * 4, table);
  }

  /* Read floats back out of a GPU buffer (debugging aid). */
  async readFloats(buffer, offsetFloats, count) {
    const bytes = count * 4;
    const staging = this.device.createBuffer({
      size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, offsetFloats * 4, staging, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  pipelineLayout() {
    const d = this.device;
    const storage = (t) => ({ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: t } });
    this._bgl = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: PARAM_FIELDS * 4 } },
        { ...storage("read-only-storage"), binding: 1 },
        { ...storage("read-only-storage"), binding: 2 },
        { ...storage("read-only-storage"), binding: 3 },
        { ...storage("storage"), binding: 4 },
        { ...storage("storage"), binding: 5 },
        { ...storage("storage"), binding: 6 },
        { ...storage("storage"), binding: 7 },
      ],
    });
    return d.createPipelineLayout({ bindGroupLayouts: [this._bgl] });
  }

  /* KV cache + scratch for a context, seeded with the prompt state from wasm. */
  prepareContext(kvLen, maxNew, opts = {}) {
    const { cfg, device, M } = this;
    const maxSeq = kvLen + maxNew + 8;
    const seqPad = opts.prefillSeq ? Math.ceil(opts.prefillSeq / 64) * 64 : 0;
    this.seqPad = seqPad;
    this.preSeq = opts.prefillSeq || 0;

    /* Prefill activations live after the generation slots in the same buffer. */
    const wantAct = seqPad
      ? (Math.ceil(this.actGenFloats / seqPad) + this.preRows) * seqPad
      : this.actGenFloats;
    if (this.actFloats !== wantAct) {
      this.bufAct.destroy();
      this.bufAct = device.createBuffer({
        size: wantAct * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      this.actFloats = wantAct;
      this._bind = null;
    }
    /* Some prefill kernels index act[(row) * seqPad + s] and so need row
     * indices; others take an absolute float base. Aligning the arena to a
     * multiple of seqPad lets both forms describe the same memory. */
    this.arenaRow0 = seqPad ? Math.ceil(this.actGenFloats / seqPad) : 0;
    this.PT = {};
    this.PTR = {};
    for (const k of Object.keys(this.PT_)) {
      this.PTR[k] = this.arenaRow0 + this.PT_[k];
      this.PT[k] = this.PTR[k] * seqPad;
    }
    const perLayer = maxSeq * cfg.kvDim;
    const kvFloats = 2 * cfg.layers * perLayer;

    if (!this.bufKV || this.kvCap < kvFloats) {
      if (this.bufKV) this.bufKV.destroy();
      this.bufKV = device.createBuffer({ size: kvFloats * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.kvCap = kvFloats;
      this._bind = null;
    }
    this.maxSeq = maxSeq;
    this.perLayer = perLayer;
    this.vDelta = cfg.layers * perLayer;

    /* scratch holds, in order: logits | attention scores | apply partials. */
    this.scoreBase = cfg.vocab;
    this.attnSlices = 8;
    this.partialBase = cfg.vocab + cfg.heads * maxSeq;
    /* Prefill scores need heads x seq x seq and reuse the same arena. */
    const preScores = seqPad ? cfg.heads * seqPad * seqPad : 0;
    const need = Math.max(this.partialBase + this.attnSlices * cfg.heads * cfg.headDim,
                          preScores);
    if (this.scratchCap !== need) {
      if (this.bufScratch) this.bufScratch.destroy();
      this.bufScratch = device.createBuffer({
        size: need * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      this.scratchCap = need;
      this._bind = null;
    }

    if (!opts.prefillSeq) {
      /* CPU prefilled: copy its KV state across. */
      const kPtr = M._qwen_wasm_kv_k_ptr() >>> 0;
      const vPtr = M._qwen_wasm_kv_v_ptr() >>> 0;
      const wasmStride = M._qwen_wasm_kv_stride();
      const bytes = kvLen * cfg.kvDim * 4;
      for (let l = 0; l < cfg.layers; l++) {
        const src = l * wasmStride * cfg.kvDim * 4;
        device.queue.writeBuffer(this.bufKV, l * perLayer * 4, M.HEAPU8, kPtr + src, bytes);
        device.queue.writeBuffer(this.bufKV, (this.vDelta + l * perLayer) * 4, M.HEAPU8, vPtr + src, bytes);
      }
    }

    this.buildParams();
  }

  bindGroup() {
    if (this._bind) return this._bind;
    this._bind = this.device.createBindGroup({
      layout: this._bgl,
      entries: [
        { binding: 0, resource: { buffer: this.bufParams, size: PARAM_FIELDS * 4 } },
        { binding: 1, resource: { buffer: this.bufQuant } },
        { binding: 2, resource: { buffer: this.bufScale } },
        { binding: 3, resource: { buffer: this.bufNorm } },
        { binding: 4, resource: { buffer: this.bufAct } },
        { binding: 5, resource: { buffer: this.bufKV } },
        { binding: 6, resource: { buffer: this.bufTok } },
        { binding: 7, resource: { buffer: this.bufScratch } },
      ],
    });
    return this._bind;
  }

  buildParams() {
    const { cfg, A } = this;
    const w = (k, l) => this.wmap.get(`${k}:${l}`);
    const nb = (k, l) => this.nmap.get(`${k}:${l}`).base;

    const slots = [];
    const push = (o) => { slots.push(o); return slots.length - 1; };

    const emb = w(W_EMBED, 0);
    const off = {};
    off.embed = push({ wordBase: emb.wordBase, scaleBase: emb.scaleBase, cols: cfg.hidden, yOff: A.x });
    off.layer = [];
    for (let l = 0; l < cfg.layers; l++) {
      const L = {};
      const grid = (rows) => Math.min(rows, this.maxDim);
      L.rms1 = push({ n: cfg.hidden, xOff: A.x, yOff: A.xn, a: nb(N_INPUT, l), fa: this.eps });
      const wqkv = w(W_Q, l);
      const qkvRows = cfg.qDim + 2 * cfg.kvDim;
      L.qxn = push({ n: cfg.hidden, xOff: A.xn, a: this.Q.xn, b: A.sxn });
      L.qkv = push({
        wordBase: wqkv.wordBase, scaleBase: wqkv.scaleBase, rows: qkvRows,
        cols: wqkv.cols, a: this.Q.xn, b: A.sxn, xOff: A.xn, yOff: A.q, d: grid(qkvRows),
      });
      L.qkrope = push({});
      L.scores = push({});
      L.softmax = push({});
      L.apply = push({});
      L.merge = push({});
      const wo = w(W_O, l);
      L.qattn = push({ n: cfg.qDim, xOff: A.attn, a: this.Q.attn, b: A.sattn });
      L.o = push({ wordBase: wo.wordBase, scaleBase: wo.scaleBase, rows: wo.rows, cols: wo.cols, a: this.Q.attn, b: A.sattn, xOff: A.attn, yOff: A.x, pos: 1, d: grid(wo.rows) });
      L.rms2 = push({ n: cfg.hidden, xOff: A.x, yOff: A.xn, a: nb(N_POST_ATTN, l), fa: this.eps });
      const wgu = w(W_GATE_UP, l);
      L.qxn2 = push({ n: cfg.hidden, xOff: A.xn, a: this.Q.xn, b: A.sxn });
      L.gu = push({ wordBase: wgu.wordBase, scaleBase: wgu.scaleBase, rows: wgu.rows, cols: wgu.cols, a: this.Q.xn, b: A.sxn, xOff: A.xn, yOff: A.gu, d: grid(wgu.rows) });
      L.swiglu = push({ n: cfg.inter, xOff: A.gu, yOff: A.g });
      const wd = w(W_DOWN, l);
      L.qg = push({ n: cfg.inter, xOff: A.g, a: this.Q.g, b: A.sg });
      L.down = push({ wordBase: wd.wordBase, scaleBase: wd.scaleBase, rows: wd.rows, cols: wd.cols, a: this.Q.g, b: A.sg, xOff: A.g, yOff: A.x, pos: 1, d: grid(wd.rows) });
      off.layer.push(L);
    }
    off.rmsFinal = push({ n: cfg.hidden, xOff: A.x, yOff: A.xn, a: nb(N_FINAL, 0), fa: this.eps });
    off.qfinal = push({ n: cfg.hidden, xOff: A.xn, a: this.Q.xn, b: A.sxn });
    off.logits = push({
      wordBase: emb.wordBase, scaleBase: emb.scaleBase, rows: cfg.vocab, cols: cfg.hidden,
      a: this.Q.xn, b: A.sxn, xOff: A.xn, d: Math.min(cfg.vocab, this.maxDim),
    });
    off.argmax = push({ n: cfg.vocab });

    /* ---- prefill slots (only when the GPU is doing the prefill) ---- */
    if (this.seqPad) {
      const PT = this.PT;
      const seq = this.preSeq, sp = this.seqPad;
      const eps = this.eps;
      const scale = 1 / Math.sqrt(cfg.headDim);
      off.pre = { layer: [] };

      for (let l = 0; l < cfg.layers; l++) {
        const L = {};
        const kBase = l * this.perLayer;
        L.rms1 = push({ n: cfg.hidden, a: sp, b: seq, c: nb(N_INPUT, l),
                        xOff: PT.x, yOff: PT.xn, fa: eps });
        const wqkv = w(W_Q, l);
        L.qkv = push({ wordBase: wqkv.wordBase, scaleBase: wqkv.scaleBase,
                       rows: cfg.qDim + 2 * cfg.kvDim, cols: cfg.hidden,
                       a: sp, n: seq, xOff: PT.xn, yOff: PT.qkv });
        L.qkrope = push({
          a: sp, b: seq, c: this.PTR.qkv, d: cfg.headDim,
          n: cfg.heads, rows: cfg.kvHeads,
          wordBase: this.PTR.qkv + cfg.qDim,
          xOff: nb(N_QNORM, l), yOff: nb(N_KNORM, l),
          fa: eps, fb: this.ropeBase,
        });
        L.kvstore = push({
          a: sp, b: seq, cols: cfg.kvDim,
          c: this.PTR.qkv + cfg.qDim, d: this.PTR.qkv + cfg.qDim + cfg.kvDim,
          scaleBase: kBase, wordBase: this.vDelta + kBase,
        });
        L.scores = push({
          a: sp, b: seq, c: this.PTR.qkv, d: cfg.headDim, rows: cfg.headsPerKv,
          wordBase: this.PTR.qkv + cfg.qDim, scaleBase: 0, n: sp, fa: scale,
        });
        L.softmax = push({ b: seq, scaleBase: 0, n: sp });
        L.apply = push({
          a: sp, b: seq, c: this.PTR.qkv + cfg.qDim + cfg.kvDim, d: cfg.headDim,
          rows: cfg.headsPerKv, yOff: this.PTR.attn, scaleBase: 0, n: sp,
        });
        const wo = w(W_O, l);
        L.o = push({ wordBase: wo.wordBase, scaleBase: wo.scaleBase, rows: wo.rows,
                     cols: wo.cols, a: sp, n: seq, xOff: PT.attn, yOff: PT.x, pos: 1 });
        L.rms2 = push({ n: cfg.hidden, a: sp, b: seq, c: nb(N_POST_ATTN, l),
                        xOff: PT.x, yOff: PT.xn, fa: eps });
        const wgu = w(W_GATE_UP, l);
        L.gu = push({ wordBase: wgu.wordBase, scaleBase: wgu.scaleBase, rows: wgu.rows,
                      cols: wgu.cols, a: sp, n: seq, xOff: PT.xn, yOff: PT.gu });
        L.swiglu = push({ a: sp, b: seq, n: cfg.inter, xOff: PT.gu, yOff: PT.g });
        const wd = w(W_DOWN, l);
        L.down = push({ wordBase: wd.wordBase, scaleBase: wd.scaleBase, rows: wd.rows,
                        cols: wd.cols, a: sp, n: seq, xOff: PT.g, yOff: PT.x, pos: 1 });
        off.pre.layer.push(L);
      }
      off.pre.extract = push({ n: cfg.hidden, a: sp, pos: seq - 1,
                               xOff: PT.x, yOff: A.x });
    }

    this.off = off;
    this.slotDefs = slots;

    const bytes = slots.length * PARAM_STRIDE;
    if (!this.bufParams || this.paramBytes !== bytes) {
      if (this.bufParams) this.bufParams.destroy();
      this.bufParams = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.paramBytes = bytes;
      this._bind = null;
    }
    this.host = new ArrayBuffer(bytes);
    this.hu = new Uint32Array(this.host);
    this.hf = new Float32Array(this.host);
    for (let i = 0; i < slots.length; i++) this.writeSlot(i, slots[i]);
    /* Upload immediately: the prefill pass reads these slots before any
     * generation step (which is what otherwise refreshes them). */
    this.device.queue.writeBuffer(this.bufParams, 0, this.host);
  }

  writeSlot(i, o) {
    const b = (i * PARAM_STRIDE) / 4;
    const u = this.hu, f = this.hf;
    u[b] = o.wordBase >>> 0; u[b + 1] = o.scaleBase >>> 0;
    u[b + 2] = o.rows >>> 0; u[b + 3] = o.cols >>> 0;
    u[b + 4] = o.xOff >>> 0; u[b + 5] = o.yOff >>> 0;
    u[b + 6] = o.pos >>> 0;  u[b + 7] = o.n >>> 0;
    u[b + 8] = o.a >>> 0;    u[b + 9] = o.b >>> 0;
    u[b + 10] = o.c >>> 0;   u[b + 11] = o.d >>> 0;
    f[b + 12] = o.fa || 0;   f[b + 13] = o.fb || 0;
    f[b + 14] = o.fc || 0;   f[b + 15] = o.fd || 0;
  }

  setStepParams(pos) {
    const { cfg, A } = this;
    const kvLen = pos + 1;
    const scale = 1 / Math.sqrt(cfg.headDim);
    for (let l = 0; l < cfg.layers; l++) {
      const L = this.off.layer[l];
      const kBase = l * this.perLayer;
      this.writeSlot(L.qkrope, {
        a: A.q, b: A.k, c: A.v, d: cfg.headDim,
        n: cfg.heads, rows: cfg.kvHeads, cols: cfg.kvDim,
        xOff: this.nmap.get(`${N_QNORM}:${l}`).base,
        yOff: this.nmap.get(`${N_KNORM}:${l}`).base,
        scaleBase: kBase, wordBase: this.vDelta + kBase,
        pos, fa: this.eps, fb: this.ropeBase,
      });
      this.writeSlot(L.scores, {
        a: A.q, d: cfg.headDim, n: kvLen, rows: cfg.heads, cols: cfg.kvDim,
        scaleBase: kBase, b: cfg.headsPerKv, c: this.maxSeq,
        yOff: this.scoreBase, fa: scale,
      });
      this.writeSlot(L.softmax, { n: kvLen, c: this.maxSeq, yOff: this.scoreBase });
      this.writeSlot(L.apply, {
        xOff: A.attn, d: cfg.headDim, n: kvLen, rows: cfg.heads, cols: cfg.kvDim,
        scaleBase: this.vDelta + kBase, b: cfg.headsPerKv, c: this.maxSeq,
        yOff: this.scoreBase, a: this.partialBase, fa: this.attnSlices,
      });
      this.writeSlot(L.merge, {
        xOff: A.attn, d: cfg.headDim, rows: cfg.heads,
        a: this.partialBase, fa: this.attnSlices,
      });
    }
    this.device.queue.writeBuffer(this.bufParams, 0, this.host);
    this.kvLen = kvLen;
  }

  /* One prefill pass over the whole prompt. Leaves the KV cache filled and the
   * final hidden state of the last position in the generation slot. */
  encodePrefill(pass) {
    const { cfg, pipe } = this;
    const bg = this.bindGroup();
    const use = (p, slot) => { pass.setPipeline(p); pass.setBindGroup(0, bg, [slot * PARAM_STRIDE]); };
    const up = (a, b) => Math.ceil(a / b);
    const seq = this.preSeq, sp = this.seqPad;
    const sBlocks = up(sp, 64);

    for (let l = 0; l < cfg.layers; l++) {
      const L = this.off.pre.layer[l];
      use(pipe.preRms, L.rms1); pass.dispatchWorkgroups(sBlocks);
      use(pipe.preMatmul, L.qkv);
      pass.dispatchWorkgroups(sBlocks, up(cfg.qDim + 2 * cfg.kvDim, 64));
      use(pipe.preQkRope, L.qkrope);
      pass.dispatchWorkgroups(sBlocks, cfg.heads + cfg.kvHeads);
      use(pipe.preKvStore, L.kvstore); pass.dispatchWorkgroups(up(cfg.kvDim, 64));
      use(pipe.preScores, L.scores); pass.dispatchWorkgroups(sBlocks, seq, cfg.heads);
      use(pipe.preSoftmax, L.softmax); pass.dispatchWorkgroups(seq, cfg.heads);
      use(pipe.preApply, L.apply);
      pass.dispatchWorkgroups(up(cfg.headDim, 64), seq, cfg.heads);
      use(pipe.preMatmul, L.o); pass.dispatchWorkgroups(sBlocks, up(cfg.hidden, 64));
      use(pipe.preRms, L.rms2); pass.dispatchWorkgroups(sBlocks);
      use(pipe.preMatmul, L.gu); pass.dispatchWorkgroups(sBlocks, up(2 * cfg.inter, 64));
      use(pipe.preSwiglu, L.swiglu); pass.dispatchWorkgroups(sBlocks, cfg.inter);
      use(pipe.preMatmul, L.down); pass.dispatchWorkgroups(sBlocks, up(cfg.hidden, 64));
    }

    use(pipe.preExtract, this.off.pre.extract);
    pass.dispatchWorkgroups(up(cfg.hidden, 64));

    /* Final norm + LM head on the extracted column gives the first token. */
    use(pipe.rmsnorm, this.off.rmsFinal); pass.dispatchWorkgroups(1);
    if (this.quantizeActivations) {
      use(pipe.quantAct, this.off.qfinal); pass.dispatchWorkgroups(cfg.hidden / 64);
    }
    const rowGrid = (rows) => rows <= this.maxDim ? [rows, 1] : [this.maxDim, up(rows, this.maxDim)];
    use(pipe.logits, this.off.logits); pass.dispatchWorkgroups(...rowGrid(cfg.vocab));
    use(pipe.argmax, this.off.argmax); pass.dispatchWorkgroups(1);
  }

  encodeStep(pass) {
    const { cfg, pipe, maxDim } = this;
    const bg = this.bindGroup();
    const use = (p, slot) => { pass.setPipeline(p); pass.setBindGroup(0, bg, [slot * PARAM_STRIDE]); };
    const up = (a, b) => Math.ceil(a / b);
    const rowGrid = (rows) => rows <= maxDim ? [rows, 1] : [maxDim, up(rows, maxDim)];

    use(pipe.embed, this.off.embed);
    pass.dispatchWorkgroups(up(cfg.hidden / 4, 64));

    for (let l = 0; l < cfg.layers; l++) {
      const L = this.off.layer[l];
      use(pipe.rmsnorm, L.rms1); pass.dispatchWorkgroups(1);
      if (this.quantizeActivations) { use(pipe.quantAct, L.qxn); pass.dispatchWorkgroups(cfg.hidden / 64); }
      use(pipe.matvec, L.qkv);
      pass.dispatchWorkgroups(...rowGrid(cfg.qDim + 2 * cfg.kvDim));
      use(pipe.qkrope, L.qkrope); pass.dispatchWorkgroups(cfg.heads + cfg.kvHeads);
      use(pipe.scores, L.scores); pass.dispatchWorkgroups(up(this.kvLen, 64), cfg.heads);
      use(pipe.softmax, L.softmax); pass.dispatchWorkgroups(cfg.heads);
      use(pipe.apply, L.apply);
      pass.dispatchWorkgroups(up(cfg.headDim, 64), cfg.heads, this.attnSlices);
      use(pipe.merge, L.merge); pass.dispatchWorkgroups(up(cfg.headDim, 64), cfg.heads);
      if (this.quantizeActivations) { use(pipe.quantAct, L.qattn); pass.dispatchWorkgroups(cfg.qDim / 64); }
      use(pipe.matvec, L.o); pass.dispatchWorkgroups(...rowGrid(cfg.hidden));
      use(pipe.rmsnorm, L.rms2); pass.dispatchWorkgroups(1);
      if (this.quantizeActivations) { use(pipe.quantAct, L.qxn2); pass.dispatchWorkgroups(cfg.hidden / 64); }
      use(pipe.matvec, L.gu); pass.dispatchWorkgroups(...rowGrid(2 * cfg.inter));
      use(pipe.swiglu, L.swiglu); pass.dispatchWorkgroups(up(cfg.inter, 256));
      if (this.quantizeActivations) { use(pipe.quantAct, L.qg); pass.dispatchWorkgroups(cfg.inter / 64); }
      use(pipe.matvec, L.down); pass.dispatchWorkgroups(...rowGrid(cfg.hidden));
    }

    use(pipe.rmsnorm, this.off.rmsFinal); pass.dispatchWorkgroups(1);
    if (this.quantizeActivations) { use(pipe.quantAct, this.off.qfinal); pass.dispatchWorkgroups(cfg.hidden / 64); }
    use(pipe.logits, this.off.logits); pass.dispatchWorkgroups(...rowGrid(cfg.vocab));
    use(pipe.argmax, this.off.argmax); pass.dispatchWorkgroups(1);
  }

  /* Prefill on the GPU from wasm-built embeddings, then generate.
   * `embedsPtr` points at [seq, hidden] f32 in wasm memory. */
  async prefillAndGenerate(embedsPtr, seq, maxNew, onPiece) {
    const { device, cfg, M } = this;
    const t0 = performance.now();
    this.prepareContext(seq, maxNew, { prefillSeq: seq });

    /* Upload transposed: the prefill kernels want [dim][seqPad]. */
    const sp = this.seqPad;
    const src = new Float32Array(M.HEAPF32.buffer, embedsPtr, seq * cfg.hidden);
    const col = new Float32Array(sp);
    for (let d = 0; d < cfg.hidden; d++) {
      for (let s2 = 0; s2 < seq; s2++) col[s2] = src[s2 * cfg.hidden + d];
      if (seq < sp) col.fill(0, seq);
      device.queue.writeBuffer(this.bufAct, (this.PT.x + d * sp) * 4, col);
    }

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.encodePrefill(pass);
    pass.end();
    enc.copyBufferToBuffer(this.bufTok, 0, this.bufTokRead, 0, 4);
    device.queue.submit([enc.finish()]);
    await this.bufTokRead.mapAsync(GPUMapMode.READ, 0, 4);
    const first = new Uint32Array(this.bufTokRead.getMappedRange(0, 4).slice(0))[0];
    this.bufTokRead.unmap();
    this.prefillMs = performance.now() - t0;

    return this.generate(first, seq, maxNew, onPiece, { keepContext: true });
  }

  /* Generate from `firstToken` (produced by the CPU prefill). */
  async generate(firstToken, kvLen, maxNew, onPiece, opts = {}) {
    const { device, cfg, M } = this;
    if (!opts.keepContext) this.prepareContext(kvLen, maxNew);

    let text = "";
    /* A forced language puts <asr_text> in the prompt, so the marker never
     * appears in the generated stream and output starts immediately — the same
     * rule the CPU decode loop uses. */
    let pastAsr = M._qwen_wasm_prompt_has_asr_text() !== 0;

    /* Qwen uses byte-level BPE, so a single token can be an incomplete UTF-8
     * sequence — a CJK character often spans two or three tokens. Decoding each
     * piece on its own turns those into replacement characters that can never
     * be recombined (this is what corrupted individual Japanese glyphs while
     * English came out byte-identical). A streaming TextDecoder carries the
     * partial sequence across pieces, like the C path's raw byte append. */
    const utf8 = new TextDecoder("utf-8");
    const emit = (id) => {
      if (id === cfg.asrText) { pastAsr = true; return; }
      if (!pastAsr) return;
      const p = M._qwen_wasm_token_text(id) >>> 0;
      if (!p) return;
      let end = p;
      while (M.HEAPU8[end] !== 0) end++;
      if (end === p) return;
      /* Copy out: TextDecoder refuses views backed by a SharedArrayBuffer. */
      const s = utf8.decode(new Uint8Array(M.HEAPU8.subarray(p, end)), { stream: true });
      if (s) { text += s; if (onPiece) onPiece(s); }
    };

    if (firstToken === cfg.imEnd || firstToken === cfg.endOfText) return { text, tokens: 1 };
    emit(firstToken);

    device.queue.writeBuffer(this.bufTok, 0, new Uint32Array([firstToken]));

    const ids = [firstToken];
    let pos = kvLen;
    let n = 1;
    for (let step = 0; step < maxNew; step++) {
      this.setStepParams(pos);

      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      this.encodeStep(pass);
      pass.end();
      enc.copyBufferToBuffer(this.bufTok, 0, this.bufTokRead, 0, 4);
      device.queue.submit([enc.finish()]);

      await this.bufTokRead.mapAsync(GPUMapMode.READ, 0, 4);
      const id = new Uint32Array(this.bufTokRead.getMappedRange(0, 4).slice(0))[0];
      this.bufTokRead.unmap();

      ids.push(id);
      n++;
      pos++;
      if (id === cfg.imEnd || id === cfg.endOfText) break;
      emit(id);
    }
    return { text, tokens: n, ids };
  }
}
