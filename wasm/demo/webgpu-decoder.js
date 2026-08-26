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

/* Generation steps encoded per submit. One step per submit spends more time in
 * submit + mapAsync than the 15 ms the GPU needs for the step itself once the
 * kernels are fast; batching K steps amortizes that round trip. The token id
 * already stays on the GPU between steps, so chaining costs nothing - the only
 * price is that an end-of-text inside a batch wastes the steps after it. */
const STEP_REGIONS = 8;

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
@group(0) @binding(5) var<storage, read_write> kv  : array<KVT>;
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
  /* Rows are this shard's slice of the vocabulary; P.c is where it starts. */
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

/* Subgroup variant of the row-sum reduction.
 *
 * The tree reduction above costs log2(WG) shared-memory rounds with a barrier
 * each. subgroupAdd does the intra-subgroup part in hardware, leaving one
 * barrier to combine the per-subgroup partials - and none at all when the
 * workgroup is a single subgroup. Google's origin-trial numbers for exactly
 * this shape (matrix-vector reductions) were 2.3-2.9x on some devices; here it
 * mostly trims the fixed cost per row, since the kernel is near the bandwidth
 * floor. Apple's subgroup size is 32; the fallback array is sized for the
 * spec minimum of 4 so the shader stays valid anywhere. */
const MATVEC_SG_BODY = `
const WG : u32 = $WG$u;
var<workgroup> red : array<f32, WG / 4u>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(subgroup_invocation_id) sglane : u32,
        @builtin(subgroup_size) sgsz : u32) {
  let row = wid.x + wid.y * P.d;
  if (row >= P.rows) { return; }
  let nwords = P.cols / 4u;
  let part = subgroupAdd($ROWFN$);
  var total : f32;
  if (WG == 32u && sgsz == 32u) {
    total = part;
  } else {
    if (sglane == 0u) { red[lid.x / sgsz] = part; }
    workgroupBarrier();
    let nsg = (WG + sgsz - 1u) / sgsz;
    var t : f32 = 0.0;
    for (var i : u32 = 0u; i < nsg; i = i + 1u) { t = t + red[i]; }
    total = t;
  }
  if (lid.x == 0u) { $STORE$ }
}
`;

const MATVEC_SG_WGSL = "enable subgroups;\n" + HEADER + MATVEC_SG_BODY
  .replace("$STORE$", "let o = P.yOff + row;\n" +
    "    if (P.pos == 1u) { act[o] = act[o] + total; } else { act[o] = total; }");
const LOGITS_SG_WGSL = "enable subgroups;\n" + HEADER + MATVEC_SG_BODY
  .replace("$STORE$", "scratch[P.c + row] = total;");

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
  if (lid.x == 0u) { scratch[P.c + row] = red[0]; }
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
    kv[P.scaleBase + slot] = KVT(act[base + lid.x]);
    kv[P.wordBase + slot] = KVT(act[P.c + kvh * hd + lid.x]);
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
    acc = acc + act[qb + d] * f32(kv[kb + d]);
  }
  scratch[P.yOff + h * P.c + j] = acc * P.fa;
}
`;

/* Subgroup rewrite of the score pass.
 *
 * The kernel above runs one thread per (head, key) with a 128-long scalar
 * loop: at a 523-token context that is 8k threads on a GPU that wants
 * hundreds of thousands, and it profiled at 21% of the whole step - ten times
 * its share of the bytes. Here a workgroup covers one head and eight keys: the
 * head's q vector is staged in workgroup memory once, each subgroup owns one
 * key, and each lane contributes a vec4 of the dot, folded with subgroupAdd.
 *   dispatch: (ceil(n/8), heads) workgroups of 256
 */
const ATTN_SCORES_SG_WGSL = "enable subgroups;\n" + HEADER + `
var<workgroup> qv : array<f32, 128>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(subgroup_invocation_id) sglane : u32,
        @builtin(subgroup_size) sgsz : u32) {
  let h = wid.y;
  if (lid.x < P.d) { qv[lid.x] = act[P.a + h * P.d + lid.x]; }
  workgroupBarrier();

  /* One subgroup per key; sgsz lanes cover d in vec4 strides. The tail
   * subgroups of the last workgroup fall past the context and contribute
   * nothing - guarded rather than returned, because subgroupAdd must be
   * reached from uniform control flow. */
  let keysPer = 256u / sgsz;
  let j = wid.x * keysPer + lid.x / sgsz;
  let live = j < P.n;
  var acc : f32 = 0.0;
  if (live) {
    let kb = P.scaleBase + j * P.cols + (h / P.b) * P.d;
    var d0 : u32 = sglane * 4u;
    loop {
      if (d0 >= P.d) { break; }
      acc = acc + qv[d0]      * f32(kv[kb + d0])
                + qv[d0 + 1u] * f32(kv[kb + d0 + 1u])
                + qv[d0 + 2u] * f32(kv[kb + d0 + 2u])
                + qv[d0 + 3u] * f32(kv[kb + d0 + 3u]);
      d0 = d0 + sgsz * 4u;
    }
  }
  let total = subgroupAdd(acc);
  if (sglane == 0u && live) { scratch[P.yOff + h * P.c + j] = total * P.fa; }
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
    acc = acc + scratch[sb + j] * f32(kv[vb + j * P.cols]);
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
  /* The token id lives on the GPU, so the shard holding its row cannot be
   * picked on the CPU. Every shard is dispatched and only the one whose range
   * contains the row does anything. */
  let tid = tok[0];
  if (tid < P.b || tid >= P.b + P.c) { return; }
  let row = tid - P.b;
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
const TR : u32 = 128u;
const TS : u32 = 64u;
/* TK = 16 is a measured optimum, not a guess: 32 puts the staged tiles at
 * 24 KB and halves how many workgroups fit a core's threadgroup memory
 * (pre.mm total 1081 -> 1256 ms), and 8 doubles the barrier count for less
 * shared traffic than the occupancy is worth (1127 ms). */
const TK : u32 = 16u;

var<workgroup> ws : array<f32, 2048>;   // TR x TK
var<workgroup> xs : array<f32, 1024>;   // TK x TS

fn store_one(row : u32, sq : u32, val : f32) {
  if (sq >= P.n) { return; }
  let o = P.yOff + row * P.a + sq;
  if (P.pos == 1u) { act[o] = act[o] + val; } else { act[o] = val; }
}

fn store_row(row : u32, sq0 : u32, v : vec4<f32>) {
  if (row >= P.rows) { return; }
  store_one(row, sq0, v.x);
  store_one(row, sq0 + 1u, v.y);
  store_one(row, sq0 + 2u, v.z);
  store_one(row, sq0 + 3u, v.w);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let r0 = wid.y * TR;
  let s0 = wid.x * TS;
  let tid = lid.y * 16u + lid.x;
  let nwords = P.cols / 4u;
  let nblocks = P.cols / 64u;
  let wb = lid.y * 8u;     // 8 rows per thread
  let xb = lid.x * 4u;     // 4 sequence positions per thread

  /* Eight vec4 accumulators, never indexed dynamically. A local array<f32, N>
   * reads the same but lands in thread memory rather than registers — an 8x8
   * tile written that way measured 8x slower. Widening the row tile to 128 is
   * what halves the activation re-reads: the activation tile is re-fetched once
   * per row tile, so its traffic scales with rows/TR. */
  var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0);
  var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
  var a4 = vec4<f32>(0.0); var a5 = vec4<f32>(0.0);
  var a6 = vec4<f32>(0.0); var a7 = vec4<f32>(0.0);

  var kb : u32 = 0u;
  loop {
    if (kb >= P.cols) { break; }

    // Weight tile: TR*TK values as TR*TK/4 u32, two per thread.
    for (var t = 0u; t < 2u; t = t + 1u) {
      let idx = tid + t * 256u;      // 0..511
      let rr = idx / 4u;             // 0..127
      let wq = idx % 4u;
      let row = r0 + rr;
      let col0 = kb + wq * 4u;
      var v = vec4<f32>(0.0);
      if (row < P.rows) {
        let word = quants[P.wordBase + row * nwords + col0 / 4u];
        v = vec4<f32>(i8x4(word)) * scales[P.scaleBase + row * nblocks + col0 / 64u];
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
      let xo = k * TS + xb;
      let x4 = vec4<f32>(xs[xo], xs[xo + 1u], xs[xo + 2u], xs[xo + 3u]);
      let wo = wb * TK + k;
      a0 = a0 + ws[wo] * x4;
      a1 = a1 + ws[wo + TK] * x4;
      a2 = a2 + ws[wo + 2u * TK] * x4;
      a3 = a3 + ws[wo + 3u * TK] * x4;
      a4 = a4 + ws[wo + 4u * TK] * x4;
      a5 = a5 + ws[wo + 5u * TK] * x4;
      a6 = a6 + ws[wo + 6u * TK] * x4;
      a7 = a7 + ws[wo + 7u * TK] * x4;
    }
    workgroupBarrier();
    kb = kb + TK;
  }

  let sq0 = s0 + xb;
  let row0 = r0 + wb;
  store_row(row0, sq0, a0);
  store_row(row0 + 1u, sq0, a1);
  store_row(row0 + 2u, sq0, a2);
  store_row(row0 + 3u, sq0, a3);
  store_row(row0 + 4u, sq0, a4);
  store_row(row0 + 5u, sq0, a5);
  store_row(row0 + 6u, sq0, a6);
  store_row(row0 + 7u, sq0, a7);
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
  /* P.pos is the first absolute position of this batch of rows - zero for a
   * from-scratch prefill, the retained-prefix length for a suffix prefill. */
  let tbase = u32(P.fb) + (P.pos + sq) * hd;
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
    kv[P.scaleBase + (P.pos + p) * P.cols + d] = KVT(act[(P.c + d) * P.a + p]);
    kv[P.wordBase + (P.pos + p) * P.cols + d] = KVT(act[(P.d + d) * P.a + p]);
  }
}
`;

/* ==================================================================
 * Suffix-prefill attention: K and V come from the cache, not the arena.
 *
 * A streaming chunk re-prefills only the rows past the unchanged prefix, and
 * those rows must attend to everything - the retained positions live in the
 * KV cache and the new ones were just stored there by the kvstore pass, so
 * reading the cache serves both uniformly. The from-scratch path keeps its
 * arena-sourced kernels untouched (and byte-identical).
 *
 * Score rows are [head][localRow] with column stride P.n = padded total
 * length; causality is j <= P.pos + s.
 * ================================================================== */

/* q staged per workgroup, one subgroup per key - the generation score pass
 * shape, with a third grid axis for the suffix row. */
const PRE_SCORES_KV_SG_WGSL = "enable subgroups;\n" + HEADER + `
var<workgroup> qv : array<f32, 128>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(subgroup_invocation_id) sglane : u32,
        @builtin(subgroup_size) sgsz : u32) {
  let h = wid.y;
  let sq = wid.z;
  if (lid.x < P.d) { qv[lid.x] = act[(P.c + h * P.d + lid.x) * P.a + sq]; }
  workgroupBarrier();

  let iAbs = P.pos + sq;
  let keysPer = 256u / sgsz;
  let j = wid.x * keysPer + lid.x / sgsz;
  let live = j <= iAbs;
  var acc : f32 = 0.0;
  if (live) {
    let kb = P.wordBase + j * P.cols + (h / P.rows) * P.d;
    var d0 : u32 = sglane * 4u;
    loop {
      if (d0 >= P.d) { break; }
      acc = acc + qv[d0]      * f32(kv[kb + d0])
                + qv[d0 + 1u] * f32(kv[kb + d0 + 1u])
                + qv[d0 + 2u] * f32(kv[kb + d0 + 2u])
                + qv[d0 + 3u] * f32(kv[kb + d0 + 3u]);
      d0 = d0 + sgsz * 4u;
    }
  }
  let total = subgroupAdd(acc);
  if (sglane == 0u && live) {
    scratch[P.scaleBase + (h * P.b + sq) * P.n + j] = total * P.fa;
  }
}
`;

/* Portable fallback: one thread per (key, suffix row, head). */
const PRE_SCORES_KV_WGSL = HEADER + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let j = gid.x;
  let sq = gid.y;
  let h = gid.z;
  let iAbs = P.pos + sq;
  if (sq >= P.b || j > iAbs) { return; }
  let kb = P.wordBase + j * P.cols + (h / P.rows) * P.d;
  var acc : f32 = 0.0;
  for (var d = 0u; d < P.d; d = d + 1u) {
    acc = acc + act[(P.c + h * P.d + d) * P.a + sq] * f32(kv[kb + d]);
  }
  scratch[P.scaleBase + (h * P.b + sq) * P.n + j] = acc * P.fa;
}
`;

/* Softmax over each suffix row's causal prefix, P.pos + row + 1 long. */
const PRE_SOFTMAX_KV_WGSL = HEADER + `
const WG : u32 = 128u;
var<workgroup> rm : array<f32, WG>;
var<workgroup> rs : array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let sq = wid.x;
  let h = wid.y;
  if (sq >= P.b) { return; }
  let base = P.scaleBase + (h * P.b + sq) * P.n;
  let len = P.pos + sq + 1u;

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
  let inv = 1.0 / max(rs[0], 1.0e-30);
  t = lid.x;
  loop {
    if (t >= len) { break; }
    scratch[base + t] = scratch[base + t] * inv;
    t = t + WG;
  }
}
`;

/* The causal V GEMM with V tiles gathered from the cache's [pos][kvDim]
 * layout - coalesced along the dim axis there, against the key axis in the
 * arena-sourced twin. Causality is j <= P.pos + s. */
const PRE_APPLY_KV_WGSL = HEADER + `
const TD : u32 = 32u;
const TQ : u32 = 32u;
const TJ : u32 = 16u;
var<workgroup> vt : array<f32, 512>;    // TJ x TD
var<workgroup> ptile : array<f32, 512>; // TJ x TQ

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let q0 = wid.x * TQ;
  let d0 = wid.y * TD;
  let h = wid.z;
  let tid = lid.y * 16u + lid.x;
  let vb = P.wordBase + (h / P.rows) * P.d + d0;
  let sbase = P.scaleBase + h * P.b * P.n;

  let td = lid.y * 2u;
  let tq = lid.x * 2u;
  var a00 : f32 = 0.0; var a01 : f32 = 0.0;
  var a10 : f32 = 0.0; var a11 : f32 = 0.0;

  let kvTotal = P.pos + P.b;
  let jmax = min(P.pos + q0 + TQ, kvTotal);
  var j0 : u32 = 0u;
  loop {
    if (j0 >= jmax) { break; }

    /* V tile: TJ x TD, consecutive threads on consecutive dims. */
    for (var t = 0u; t < 2u; t = t + 1u) {
      let idx = tid + t * 256u;
      let jj = idx / TD;
      let dd = idx % TD;
      vt[idx] = select(0.0, f32(kv[vb + (j0 + jj) * P.cols + dd]), j0 + jj < kvTotal);
    }
    for (var t = 0u; t < 2u; t = t + 1u) {
      let idx = tid + t * 256u;
      let jj = idx / TQ;
      let qq = idx % TQ;
      let sq = q0 + qq;
      let j = j0 + jj;
      ptile[idx] = select(0.0, scratch[sbase + sq * P.n + j],
                          sq < P.b && j <= P.pos + sq);
    }
    workgroupBarrier();

    for (var jj = 0u; jj < TJ; jj = jj + 1u) {
      let v0 = vt[jj * TD + td];
      let v1 = vt[jj * TD + td + 1u];
      let p0v = ptile[jj * TQ + tq];
      let p1v = ptile[jj * TQ + tq + 1u];
      a00 = a00 + v0 * p0v; a01 = a01 + v0 * p1v;
      a10 = a10 + v1 * p0v; a11 = a11 + v1 * p1v;
    }
    workgroupBarrier();
    j0 = j0 + TJ;
  }

  let orow = (P.yOff + h * P.d + d0 + td) * P.a + q0 + tq;
  if (q0 + tq < P.b) {
    act[orow] = a00;
    act[orow + P.a] = a10;
  }
  if (q0 + tq + 1u < P.b) {
    act[orow + 1u] = a01;
    act[orow + P.a + 1u] = a11;
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
/* Weighted V sum as a causal tiled GEMM: out[d][q] = sum_j V[d][j] P[q][j].
 *
 * Three shapes of this kernel failed before this one, all on the same lesson:
 *
 *   - lanes along d: a warp's V loads sit a seqPad stride apart. 929 ms.
 *   - the same with the j loop reordered per thread: what matters is the set
 *     of addresses one warp instruction gathers, not per-thread order. 911 ms.
 *   - lanes along j with one query per workgroup: coalesced at last, but with
 *     no reuse each workgroup streams the head's whole V - the pass re-reads
 *     V once per query, 68 GB over the layers at seq 549. 734 ms.
 *
 * So it is a GEMM and it wants both tiles staged: 32 dims x 32 queries per
 * workgroup, V and P tiles in shared memory, every load coalesced along j,
 * V read ceil(seq/32) times total instead of seq times. Causality is the tile
 * loader zeroing P entries with j > i, and the j loop stopping at each query
 * tile's own diagonal - the triangle is skipped, not masked away. Profiled
 * 929 ms -> 60 ms at seq 549.
 *   scaleBase = score base, c = V base row, rows = headsPerKv (see caller)
 */
const PRE_APPLY_WGSL = HEADER + `
const TD : u32 = 32u;
const TQ : u32 = 32u;
const TJ : u32 = 16u;
var<workgroup> vt : array<f32, 512>;   // TD x TJ
var<workgroup> ptile : array<f32, 512>; // TJ x TQ

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let q0 = wid.x * TQ;
  let d0 = wid.y * TD;
  let h = wid.z;
  let tid = lid.y * 16u + lid.x;
  let base = P.scaleBase + h * P.n * P.n;
  let vrow = P.c + (h / P.rows) * P.d + d0;

  /* Each thread owns a 2x2 block of the 32x32 output tile. */
  let td = lid.y * 2u;
  let tq = lid.x * 2u;
  var a00 : f32 = 0.0; var a01 : f32 = 0.0;
  var a10 : f32 = 0.0; var a11 : f32 = 0.0;

  /* Causality bounds the j range by the last query in this tile. */
  let jmax = min(q0 + TQ, P.b);
  var j0 : u32 = 0u;
  loop {
    if (j0 >= jmax) { break; }

    /* V tile: TD x TJ, two values per thread, coalesced along j. */
    for (var t = 0u; t < 2u; t = t + 1u) {
      let idx = tid + t * 256u;
      let rr = idx / TJ;
      let jj = idx % TJ;
      /* The tail tile reaches into seq padding, which is never written by the
       * matmuls and may hold a previous, longer run's values. The P tile is
       * zero there, but 0 * garbage must still be a well-behaved zero. */
      vt[idx] = select(0.0, act[(vrow + rr) * P.a + j0 + jj], j0 + jj < P.b);
    }
    /* P tile: TJ x TQ, zero where j overruns the query's causal prefix or
     * the sequence. Stored transposed so the inner loop reads it row-wise. */
    for (var t = 0u; t < 2u; t = t + 1u) {
      let idx = tid + t * 256u;
      let jj = idx / TQ;
      let qq = idx % TQ;
      let i = q0 + qq;
      let j = j0 + jj;
      ptile[idx] = select(0.0, scratch[base + i * P.n + j],
                          i < P.b && j <= i);
    }
    workgroupBarrier();

    for (var jj = 0u; jj < TJ; jj = jj + 1u) {
      let v0 = vt[(td) * TJ + jj];
      let v1 = vt[(td + 1u) * TJ + jj];
      let p0 = ptile[jj * TQ + tq];
      let p1 = ptile[jj * TQ + tq + 1u];
      a00 = a00 + v0 * p0; a01 = a01 + v0 * p1;
      a10 = a10 + v1 * p0; a11 = a11 + v1 * p1;
    }
    workgroupBarrier();
    j0 = j0 + TJ;
  }

  let orow = (P.yOff + h * P.d + d0 + td) * P.a + q0 + tq;
  if (q0 + tq < P.b) {
    act[orow] = a00;
    act[orow + P.a] = a10;
  }
  if (q0 + tq + 1u < P.b) {
    act[orow + 1u] = a01;
    act[orow + P.a + 1u] = a11;
  }
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

    /* Weights are packed into shards rather than one buffer.
     *
     * All of them in one binding is 1.72 GB, and WebGPU only guarantees
     * maxStorageBufferBindingSize of 128 MiB. Chromium exposes tiers above
     * that - roughly 1 GiB, 2 GiB, 4 GiB - so a single binding of 1.72 GB
     * needs the 2 GiB tier and simply cannot be created below it, whatever the
     * GPU is worth. Sharding drops the largest binding to SHARD_BUDGET and
     * brings the 1 GiB tier into range, which is where most desktop GPUs sit.
     *
     * 128 MiB is not a useful target for this model: the KV cache alone is
     * 224 KiB per token, so a 1600-token context needs a 360 MB binding no
     * amount of weight sharding can avoid.
     *
     * A matrix is never split across shards, so the matmul kernels are
     * unchanged and only need the right shard bound. The tied embedding is the
     * exception - 311 MB on its own - and is split by row range, which the
     * logits and embedding kernels handle explicitly. */
    const SHARD_BUDGET = 256 << 20;

    let scaleFloats = 0, quantBytes = 0;
    const wmap = new Map();
    const shards = [{ bytes: 0 }];
    const openShard = (need) => {
      let sh = shards[shards.length - 1];
      if (sh.bytes > 0 && sh.bytes + need > SHARD_BUDGET) {
        shards.push({ bytes: 0 });
        sh = shards[shards.length - 1];
      }
      return sh;
    };

    for (let i = 0; i < nQ; i++) {
      const [kind, layer, rows, cols, qptr, sptr] = qd.subarray(i * 6, i * 6 + 6);
      const nq = rows * cols;
      const scaleBase = scaleFloats;

      if (nq > SHARD_BUDGET) {
        /* Only the embedding gets here. Split by whole rows so each piece is
         * still a contiguous [rows][cols] matrix. */
        const rowsPer = Math.floor(SHARD_BUDGET / cols);
        const pieces = [];
        for (let r0 = 0; r0 < rows; r0 += rowsPer) {
          const n = Math.min(rowsPer, rows - r0);
          const sh = openShard(n * cols);
          pieces.push({
            shard: shards.length - 1,
            wordBase: sh.bytes / 4,
            rowBase: r0, rowCount: n,
            scaleBase: scaleBase + (r0 * cols) / 64,
            qptr: qptr + r0 * cols, nq: n * cols,
          });
          sh.bytes += n * cols;
        }
        wmap.set(`${kind}:${layer}`, { rows, cols, scaleBase, sptr, nq, pieces });
      } else {
        const sh = openShard(nq);
        wmap.set(`${kind}:${layer}`, {
          rows, cols, qptr, sptr, nq, scaleBase,
          shard: shards.length - 1, wordBase: sh.bytes / 4,
        });
        sh.bytes += nq;
      }
      quantBytes += nq;
      scaleFloats += nq / 64;
    }

    const biggestShard = Math.max(...shards.map((s) => s.bytes));
    /* Ask only for what the largest single binding actually needs. The KV
     * cache is allocated later and can be the biggest of them at long
     * contexts, so leave room for it rather than sizing to the shards alone. */
    /* Simulate a lesser device: window.__gpuBindingCap caps what we ask for,
     * which is how the 1 GiB tier gets tested on a machine that offers 4. */
    const cap = (typeof window !== "undefined" && window.__gpuBindingCap) ||
                lim.maxStorageBufferBindingSize;
    const wantBinding = Math.min(cap, lim.maxStorageBufferBindingSize,
                                 Math.max(biggestShard, scaleFloats * 4, 1 << 30));
    if (biggestShard > lim.maxStorageBufferBindingSize) {
      throw new Error(`a weight shard needs ${(biggestShard / 1e6).toFixed(0)} MB, adapter caps ` +
                      `storage bindings at ${(lim.maxStorageBufferBindingSize / 1e6).toFixed(0)} MB`);
    }

    /* Optional features, taken when the adapter has them: subgroups for the
     * matvec reductions, timestamps for the per-kernel profiler. shader-f16 is
     * requested so experiments can use it without a second device. */
    const wantFeatures = ["subgroups", "shader-f16", "timestamp-query"]
      .filter((f) => adapter.features.has(f));
    const device = await adapter.requestDevice({
      requiredFeatures: wantFeatures,
      requiredLimits: {
        maxBufferSize: wantBinding,
        maxStorageBufferBindingSize: wantBinding,
        /* Headroom for tile experiments: the default limit is 16 KB, and an
         * over-limit pipeline does not throw at creation - it fails at
         * dispatch time, silently, as an invalid command buffer. */
        maxComputeWorkgroupStorageSize:
          Math.min(adapter.limits.maxComputeWorkgroupStorageSize, 32768),
      },
    });
    this.hasSubgroups = wantFeatures.includes("subgroups");
    this.hasTimestamps = wantFeatures.includes("timestamp-query");
    this.hasF16 = wantFeatures.includes("shader-f16");
    /* KV cache storage type. f16 halves the biggest allocation this backend
     * makes - the cache is what dominates once weights are sharded - and K/V
     * live after a norm, so their range is tame. Research agrees the format
     * has margin to spare (int8 KV caches measure as quality-neutral; f16 is
     * gentler still). Accumulation stays f32 in every kernel. Set kvF16Pref =
     * false before init() to keep f32 - the CPU-comparison harness does, since
     * bitwise identity with the wasm decoder cannot survive rounded KV. */
    this.kvF16 = this.hasF16 && this.kvF16Pref !== false &&
                 typeof Float16Array !== "undefined";
    this.kvBytes = this.kvF16 ? 2 : 4;
    this.device = device;
    this.maxDim = device.limits.maxComputeWorkgroupsPerDimension;
    this.adapterInfo = adapter.info || {};
    /* Surface these: a missing buffer usage flag makes writeBuffer a silent
     * no-op, and the symptom is a kernel that runs fast and returns zeros. */
    this.onError = report;
    /* A device can be lost at any point - a driver reset, the OS switching
     * GPUs, the tab being discarded. Every call after that fails, so record it
     * and let callers fall back rather than reporting nonsense. */
    device.lost.then((info) => {
      this.lost = info.reason || "unknown";
      const m = `GPU device lost (${this.lost}); falling back to wasm`;
      console.error(m);
      if (this.onError) this.onError(m);
    });
    device.onuncapturederror = (e) => {
      console.error("gpu:", e.error.message);
      if (this.onError) this.onError("gpu error: " + e.error.message);
    };

    /* ---- upload ---- */
    report(`allocating ${(quantBytes / 1e9).toFixed(2)} GB on the GPU in ` +
           `${shards.length} shard${shards.length === 1 ? "" : "s"} of up to ` +
           `${(biggestShard / 1e6).toFixed(0)} MB...`);
    this.bufQuants = shards.map((sh) =>
      device.createBuffer({ size: sh.bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }));
    this.bufScale = device.createBuffer({ size: scaleFloats * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.shardCount = shards.length;

    const CH = 64 << 20;
    let done = 0, lastReport = 0;
    const putQuant = (shard, wordBase, qptr, nq) => {
      for (let off = 0; off < nq; off += CH) {
        const n = Math.min(CH, nq - off);
        device.queue.writeBuffer(this.bufQuants[shard], wordBase * 4 + off, M.HEAPU8, qptr + off, n);
        done += n;
      }
    };
    for (const w of wmap.values()) {
      if (w.pieces) {
        for (const p of w.pieces) putQuant(p.shard, p.wordBase, p.qptr, p.nq);
      } else {
        putQuant(w.shard, w.wordBase, w.qptr, w.nq);
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
    this.bufTokRead = device.createBuffer({ size: STEP_REGIONS * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    /* ---- pipelines ---- */
    const layout = this.pipelineLayout();
    /* Keep the modules so their compilation diagnostics can be surfaced: an
     * invalid pipeline silently turns every dispatch into a no-op, which shows
     * up as "impossibly fast and wrong" rather than as an error. */
    const modules = [];
    const mk = (code, name) => {
      /* Every module sees the KV cache through the KVT alias; enable
       * directives must precede everything else, so hoist them. */
      code = (this.kvF16 ? "enable f16;\nalias KVT = f16;\n"
                         : "alias KVT = f32;\n") + code;
      const enables = [...new Set([...code.matchAll(/^enable [^;]+;$/gm)].map((m) => m[0]))];
      code = enables.join("\n") + "\n" + code.replace(/^enable [^;]+;$/gm, "");
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
    /* Two reduction families: the portable shared-memory tree, and the
     * subgroup one where the feature exists. Both are kept so they can be
     * compared on the same audio (setSubgroups). */
    this.matvecPipes = { q8: {}, f32: {} };
    this.logitsPipes = { q8: {}, f32: {} };
    this.matvecPipesTree = { q8: {}, f32: {} };
    this.logitsPipesTree = { q8: {}, f32: {} };
    for (const [mode, rowfn] of [["q8", ROW_Q8], ["f32", ROW_F32]]) {
      for (const wg of [32, 64, 128, 256]) {
        const sub = (code) => code.replace("$WG$", String(wg)).replace("$ROWFN$", rowfn);
        this.matvecPipesTree[mode][wg] = mk(sub(MATVEC_WGSL), `matvec_${mode}_${wg}`);
        this.logitsPipesTree[mode][wg] = mk(sub(LOGITS_WGSL), `logits_${mode}_${wg}`);
        if (this.hasSubgroups) {
          this.matvecPipes[mode][wg] = mk(sub(MATVEC_SG_WGSL), `matvec_sg_${mode}_${wg}`);
          this.logitsPipes[mode][wg] = mk(sub(LOGITS_SG_WGSL), `logits_sg_${mode}_${wg}`);
        } else {
          this.matvecPipes[mode][wg] = this.matvecPipesTree[mode][wg];
          this.logitsPipes[mode][wg] = this.logitsPipesTree[mode][wg];
        }
      }
    }
    this.useSubgroups = this.hasSubgroups;
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
      scores: this.hasSubgroups ? mk(ATTN_SCORES_SG_WGSL, "scores_sg")
                                : mk(ATTN_SCORES_WGSL, "scores"),
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
      preScoresKv: this.hasSubgroups ? mk(PRE_SCORES_KV_SG_WGSL, "preScoresKv_sg")
                                     : mk(PRE_SCORES_KV_WGSL, "preScoresKv"),
      preSoftmax: mk(PRE_SOFTMAX_WGSL, "preSoftmax"),
      preSoftmaxKv: mk(PRE_SOFTMAX_KV_WGSL, "preSoftmaxKv"),
      preApplyKv: mk(PRE_APPLY_KV_WGSL, "preApplyKv"),
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
    const mv = this.useSubgroups ? this.matvecPipes : this.matvecPipesTree;
    const lg = this.useSubgroups ? this.logitsPipes : this.logitsPipesTree;
    this.pipe.matvec = mv[mode][this.matvecWidth];
    this.pipe.logits = lg[mode][this.matvecWidth];
  }

  /* Flip between the subgroup and shared-memory-tree reductions (A/B). */
  setSubgroups(on) {
    this.useSubgroups = !!on && this.hasSubgroups;
    this.applyMode();
    return this.useSubgroups;
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
    const suffix = opts.prefillBase != null;
    /* A suffix prefill extends a live cache, so the layer stride - and with it
     * every layer's base offset - must not move underneath it. Keep the
     * current layout while it is big enough, and grow in coarse steps so a
     * stream is not re-laying the cache out every chunk. */
    let maxSeq = kvLen + maxNew + 8;
    if (suffix) {
      /* Keep the current layout whenever it actually fits; the 768-step
       * rounding is only for sizing a fresh one, and comparing against the
       * rounded figure would re-lay out a cache that was already big enough. */
      if (this.bufKV && this.maxSeq >= maxSeq) maxSeq = this.maxSeq;
      else maxSeq = Math.ceil(maxSeq / 768) * 768;
    }
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

    if (!this.bufKV || this.kvCap < kvFloats || (suffix && perLayer !== this.perLayer)) {
      const old = this.bufKV;
      this.bufKV = device.createBuffer({ size: kvFloats * this.kvBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      /* Growing under a suffix prefill re-lays the cache out: every layer's
       * region moves, so the retained positions are copied across. */
      if (old && suffix && this.perLayer) {
        const eb = this.kvBytes;
        const enc = device.createCommandEncoder();
        const bytes = Math.min(this.perLayer, perLayer) * eb;
        for (let l = 0; l < cfg.layers; l++) {
          enc.copyBufferToBuffer(old, l * this.perLayer * eb,
                                 this.bufKV, l * perLayer * eb, bytes);
          enc.copyBufferToBuffer(old, (this.vDelta + l * this.perLayer) * eb,
                                 this.bufKV, (cfg.layers * perLayer + l * perLayer) * eb, bytes);
        }
        device.queue.submit([enc.finish()]);
      }
      if (old) old.destroy();
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
    /* Prefill scores need heads x rows x columns and reuse the same arena:
     * square for a from-scratch prefill, nNew x paddedTotal for a suffix. */
    this.preBase = suffix ? opts.prefillBase : null;
    this.preTotalPad = suffix
      ? Math.ceil((opts.prefillBase + this.preSeq) / 64) * 64 : seqPad;
    const preScores = seqPad ? cfg.heads * seqPad * this.preTotalPad : 0;
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
      const srcF16 = !!(M._qwen_wasm_kv_is_f16 && M._qwen_wasm_kv_is_f16());
      const n = kvLen * cfg.kvDim;
      const srcBytes = srcF16 ? 2 : 4;
      for (let l = 0; l < cfg.layers; l++) {
        const src = l * wasmStride * cfg.kvDim;
        let kSrc, vSrc;
        if (srcF16) {
          kSrc = new Uint16Array(M.HEAPU8.buffer, kPtr + src * srcBytes, n);
          vSrc = new Uint16Array(M.HEAPU8.buffer, vPtr + src * srcBytes, n);
          if (this.kvF16) { kSrc = kSrc.slice(); vSrc = vSrc.slice(); }
          else {
            /* Widen through Float16Array's element accessors (exact). */
            kSrc = Float32Array.from(new Float16Array(kSrc.slice().buffer));
            vSrc = Float32Array.from(new Float16Array(vSrc.slice().buffer));
          }
        } else {
          kSrc = new Float32Array(M.HEAPF32.buffer, kPtr + src * srcBytes, n);
          vSrc = new Float32Array(M.HEAPF32.buffer, vPtr + src * srcBytes, n);
          if (this.kvF16) { kSrc = new Float16Array(kSrc); vSrc = new Float16Array(vSrc); }
          else { kSrc = kSrc.slice(); vSrc = vSrc.slice(); }
        }
        device.queue.writeBuffer(this.bufKV, l * perLayer * this.kvBytes, kSrc);
        device.queue.writeBuffer(this.bufKV, (this.vDelta + l * perLayer) * this.kvBytes, vSrc);
      }
    }

    this.buildParams();
  }

  /* One bind group per weight shard; everything else is shared. A dispatch
   * binds the shard holding the matrix it reads. */
  bindGroup(shard = 0) {
    if (!this._bind) this._bind = [];
    if (this._bind[shard]) return this._bind[shard];
    this._bind[shard] = this.device.createBindGroup({
      layout: this._bgl,
      entries: [
        { binding: 0, resource: { buffer: this.bufParams, size: PARAM_FIELDS * 4 } },
        { binding: 1, resource: { buffer: this.bufQuants[shard] } },
        { binding: 2, resource: { buffer: this.bufScale } },
        { binding: 3, resource: { buffer: this.bufNorm } },
        { binding: 4, resource: { buffer: this.bufAct } },
        { binding: 5, resource: { buffer: this.bufKV } },
        { binding: 6, resource: { buffer: this.bufTok } },
        { binding: 7, resource: { buffer: this.bufScratch } },
      ],
    });
    return this._bind[shard];
  }

  /* Rows a slot's dispatch covers, read back from the params it was built
   * with, so the vocabulary pieces do not need a second bookkeeping array. */
  slotRows(slot) {
    return this.hu[(slot * PARAM_STRIDE) / 4 + 2];
  }

  buildParams() {
    const { cfg, A } = this;
    const w = (k, l) => this.wmap.get(`${k}:${l}`);
    const nb = (k, l) => this.nmap.get(`${k}:${l}`).base;

    const slots = [];
    /* Which weight shard each slot reads, so a dispatch can bind it without
     * every call site having to know. */
    const slotShard = [];
    const push = (o) => {
      const { shard = 0, ...rest } = o;
      slots.push(rest);
      slotShard.push(shard);
      return slots.length - 1;
    };
    this.slotShard = slotShard;

    const emb = w(W_EMBED, 0);
    /* The tied embedding is 311 MB, too big for one shard, so it is split by
     * row range. Both kernels that walk the vocabulary get one dispatch per
     * piece: the embedding lookup because the row it wants is a token id that
     * lives on the GPU and only one piece holds it, and the logits pass
     * because every piece contributes its own rows. */
    const embPieces = emb.pieces ||
      [{ shard: emb.shard, wordBase: emb.wordBase, scaleBase: emb.scaleBase,
         rowBase: 0, rowCount: emb.rows }];
    const off = {};
    off.embed = embPieces.map((p) => push({
      shard: p.shard, wordBase: p.wordBase, scaleBase: p.scaleBase,
      cols: cfg.hidden, yOff: A.x, b: p.rowBase, c: p.rowCount,
    }));
    off.layer = [];
    for (let l = 0; l < cfg.layers; l++) {
      const L = {};
      const grid = (rows) => Math.min(rows, this.maxDim);
      L.rms1 = push({ n: cfg.hidden, xOff: A.x, yOff: A.xn, a: nb(N_INPUT, l), fa: this.eps });
      const wqkv = w(W_Q, l);
      const qkvRows = cfg.qDim + 2 * cfg.kvDim;
      L.qxn = push({ n: cfg.hidden, xOff: A.xn, a: this.Q.xn, b: A.sxn });
      L.qkv = push({
        shard: wqkv.shard, wordBase: wqkv.wordBase, scaleBase: wqkv.scaleBase, rows: qkvRows,
        cols: wqkv.cols, a: this.Q.xn, b: A.sxn, xOff: A.xn, yOff: A.q, d: grid(qkvRows),
      });
      L.qkrope = push({});
      L.scores = push({});
      L.softmax = push({});
      L.apply = push({});
      L.merge = push({});
      const wo = w(W_O, l);
      L.qattn = push({ n: cfg.qDim, xOff: A.attn, a: this.Q.attn, b: A.sattn });
      L.o = push({ shard: wo.shard, wordBase: wo.wordBase, scaleBase: wo.scaleBase, rows: wo.rows, cols: wo.cols, a: this.Q.attn, b: A.sattn, xOff: A.attn, yOff: A.x, pos: 1, d: grid(wo.rows) });
      L.rms2 = push({ n: cfg.hidden, xOff: A.x, yOff: A.xn, a: nb(N_POST_ATTN, l), fa: this.eps });
      const wgu = w(W_GATE_UP, l);
      L.qxn2 = push({ n: cfg.hidden, xOff: A.xn, a: this.Q.xn, b: A.sxn });
      L.gu = push({ shard: wgu.shard, wordBase: wgu.wordBase, scaleBase: wgu.scaleBase, rows: wgu.rows, cols: wgu.cols, a: this.Q.xn, b: A.sxn, xOff: A.xn, yOff: A.gu, d: grid(wgu.rows) });
      L.swiglu = push({ n: cfg.inter, xOff: A.gu, yOff: A.g });
      const wd = w(W_DOWN, l);
      L.qg = push({ n: cfg.inter, xOff: A.g, a: this.Q.g, b: A.sg });
      L.down = push({ shard: wd.shard, wordBase: wd.wordBase, scaleBase: wd.scaleBase, rows: wd.rows, cols: wd.cols, a: this.Q.g, b: A.sg, xOff: A.g, yOff: A.x, pos: 1, d: grid(wd.rows) });
      off.layer.push(L);
    }
    off.rmsFinal = push({ n: cfg.hidden, xOff: A.x, yOff: A.xn, a: nb(N_FINAL, 0), fa: this.eps });
    off.qfinal = push({ n: cfg.hidden, xOff: A.xn, a: this.Q.xn, b: A.sxn });
    off.logits = embPieces.map((p) => push({
      shard: p.shard, wordBase: p.wordBase, scaleBase: p.scaleBase,
      rows: p.rowCount, cols: cfg.hidden,
      a: this.Q.xn, b: A.sxn, xOff: A.xn, c: p.rowBase,
      d: Math.min(p.rowCount, this.maxDim),
    }));
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
        L.qkv = push({ shard: wqkv.shard, wordBase: wqkv.wordBase, scaleBase: wqkv.scaleBase,
                       rows: cfg.qDim + 2 * cfg.kvDim, cols: cfg.hidden,
                       a: sp, n: seq, xOff: PT.xn, yOff: PT.qkv });
        const p0 = this.preBase || 0;
        L.qkrope = push({
          a: sp, b: seq, c: this.PTR.qkv, d: cfg.headDim,
          n: cfg.heads, rows: cfg.kvHeads, pos: p0,
          wordBase: this.PTR.qkv + cfg.qDim,
          xOff: nb(N_QNORM, l), yOff: nb(N_KNORM, l),
          fa: eps, fb: this.ropeBase,
        });
        L.kvstore = push({
          a: sp, b: seq, cols: cfg.kvDim, pos: p0,
          c: this.PTR.qkv + cfg.qDim, d: this.PTR.qkv + cfg.qDim + cfg.kvDim,
          scaleBase: kBase, wordBase: this.vDelta + kBase,
        });
        if (this.preBase != null) {
          /* Suffix attention reads K and V from the cache. */
          L.scores = push({
            a: sp, b: seq, c: this.PTR.qkv, d: cfg.headDim, rows: cfg.headsPerKv,
            cols: cfg.kvDim, pos: p0, wordBase: kBase,
            scaleBase: 0, n: this.preTotalPad, fa: scale,
          });
          L.softmax = push({ b: seq, pos: p0, scaleBase: 0, n: this.preTotalPad });
          L.apply = push({
            a: sp, b: seq, d: cfg.headDim, rows: cfg.headsPerKv,
            cols: cfg.kvDim, pos: p0, wordBase: this.vDelta + kBase,
            yOff: this.PTR.attn, scaleBase: 0, n: this.preTotalPad,
          });
        } else {
          L.scores = push({
            a: sp, b: seq, c: this.PTR.qkv, d: cfg.headDim, rows: cfg.headsPerKv,
            wordBase: this.PTR.qkv + cfg.qDim, scaleBase: 0, n: sp, fa: scale,
          });
          L.softmax = push({ b: seq, scaleBase: 0, n: sp });
          L.apply = push({
            a: sp, b: seq, c: this.PTR.qkv + cfg.qDim + cfg.kvDim, d: cfg.headDim,
            rows: cfg.headsPerKv, yOff: this.PTR.attn, scaleBase: 0, n: sp,
          });
        }
        const wo = w(W_O, l);
        L.o = push({ shard: wo.shard, wordBase: wo.wordBase, scaleBase: wo.scaleBase, rows: wo.rows,
                     cols: wo.cols, a: sp, n: seq, xOff: PT.attn, yOff: PT.x, pos: 1 });
        L.rms2 = push({ n: cfg.hidden, a: sp, b: seq, c: nb(N_POST_ATTN, l),
                        xOff: PT.x, yOff: PT.xn, fa: eps });
        const wgu = w(W_GATE_UP, l);
        L.gu = push({ shard: wgu.shard, wordBase: wgu.wordBase, scaleBase: wgu.scaleBase, rows: wgu.rows,
                      cols: wgu.cols, a: sp, n: seq, xOff: PT.xn, yOff: PT.gu });
        L.swiglu = push({ a: sp, b: seq, n: cfg.inter, xOff: PT.gu, yOff: PT.g });
        const wd = w(W_DOWN, l);
        L.down = push({ shard: wd.shard, wordBase: wd.wordBase, scaleBase: wd.scaleBase, rows: wd.rows,
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
      /* One region per batched step; prefill and single-step paths use region
       * 0 and never see the others. */
      this.bufParams = this.device.createBuffer({ size: bytes * STEP_REGIONS,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
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
    this.patchStepSlots(pos);
    this.device.queue.writeBuffer(this.bufParams, 0, this.host);
    this.kvLen = pos + 1;
  }

  patchStepSlots(pos) {
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
    this.kvLen = kvLen;
  }

  /* Patch the step slots for `pos` and upload into params region `region`.
   * Leaves this.kvLen at pos + 1; the encode loop re-sets it per step. */
  writeStepRegion(pos, region) {
    this.patchStepSlots(pos);
    this.device.queue.writeBuffer(this.bufParams, region * this.paramBytes, this.host);
  }

  /* One prefill pass over the whole prompt. Leaves the KV cache filled and the
   * final hidden state of the last position in the generation slot. */
  encodePrefill(pass, phase) {
    const { cfg, pipe } = this;
    const ph = (label) => { if (phase) pass = phase(label); };
    const use = (p, slot) => {
      pass.setPipeline(p);
      pass.setBindGroup(0, this.bindGroup(this.slotShard[slot] | 0), [slot * PARAM_STRIDE]);
    };
    const up = (a, b) => Math.ceil(a / b);
    const seq = this.preSeq, sp = this.seqPad;
    const sBlocks = up(sp, 64);

    for (let l = 0; l < cfg.layers; l++) {
      const L = this.off.pre.layer[l];
      ph("pre.rms"); use(pipe.preRms, L.rms1); pass.dispatchWorkgroups(sBlocks);
      ph("pre.mm.qkv"); use(pipe.preMatmul, L.qkv);
      pass.dispatchWorkgroups(sBlocks, up(cfg.qDim + 2 * cfg.kvDim, 128));
      ph("pre.qkrope"); use(pipe.preQkRope, L.qkrope);
      pass.dispatchWorkgroups(sBlocks, cfg.heads + cfg.kvHeads);
      ph("pre.kvstore"); use(pipe.preKvStore, L.kvstore); pass.dispatchWorkgroups(up(cfg.kvDim, 64));
      if (this.preBase != null) {
        const kvTotal = this.preBase + seq;
        ph("pre.scores"); use(pipe.preScoresKv, L.scores);
        if (this.hasSubgroups) { pass.dispatchWorkgroups(up(kvTotal, 8), cfg.heads, seq); }
        else { pass.dispatchWorkgroups(up(kvTotal, 64), seq, cfg.heads); }
        ph("pre.softmax"); use(pipe.preSoftmaxKv, L.softmax); pass.dispatchWorkgroups(seq, cfg.heads);
        ph("pre.apply"); use(pipe.preApplyKv, L.apply);
        pass.dispatchWorkgroups(up(seq, 32), up(cfg.headDim, 32), cfg.heads);
      } else {
        ph("pre.scores"); use(pipe.preScores, L.scores); pass.dispatchWorkgroups(sBlocks, seq, cfg.heads);
        ph("pre.softmax"); use(pipe.preSoftmax, L.softmax); pass.dispatchWorkgroups(seq, cfg.heads);
        ph("pre.apply"); use(pipe.preApply, L.apply);
        pass.dispatchWorkgroups(up(seq, 32), up(cfg.headDim, 32), cfg.heads);
      }
      ph("pre.mm.o"); use(pipe.preMatmul, L.o); pass.dispatchWorkgroups(sBlocks, up(cfg.hidden, 128));
      ph("pre.rms"); use(pipe.preRms, L.rms2); pass.dispatchWorkgroups(sBlocks);
      ph("pre.mm.gu"); use(pipe.preMatmul, L.gu); pass.dispatchWorkgroups(sBlocks, up(2 * cfg.inter, 128));
      ph("pre.swiglu"); use(pipe.preSwiglu, L.swiglu); pass.dispatchWorkgroups(sBlocks, cfg.inter);
      ph("pre.mm.down"); use(pipe.preMatmul, L.down); pass.dispatchWorkgroups(sBlocks, up(cfg.hidden, 128));
    }

    ph("pre.extract"); use(pipe.preExtract, this.off.pre.extract);
    pass.dispatchWorkgroups(up(cfg.hidden, 64));

    /* Final norm + LM head on the extracted column gives the first token. */
    use(pipe.rmsnorm, this.off.rmsFinal); pass.dispatchWorkgroups(1);
    if (this.quantizeActivations) {
      use(pipe.quantAct, this.off.qfinal); pass.dispatchWorkgroups(cfg.hidden / 64);
    }
    const rowGrid = (rows) => rows <= this.maxDim ? [rows, 1] : [this.maxDim, up(rows, this.maxDim)];
    ph("pre.logits");
    for (const slot of this.off.logits) {
      use(pipe.logits, slot);
      pass.dispatchWorkgroups(...rowGrid(this.slotRows(slot)));
    }
    use(pipe.argmax, this.off.argmax); pass.dispatchWorkgroups(1);
  }

  /* Per-kernel GPU time for the prefill pass; see profileStep. Needs a
   * prefill-capable context (prepareContext with prefillSeq) already set up
   * and the transposed embeddings uploaded - so call it right after a
   * prefillAndGenerate, which leaves both in place. */
  async profilePrefill() {
    if (!this.hasTimestamps) throw new Error("timestamp-query not available");
    const { device } = this;
    const MAXQ = 2048;
    const qs = device.createQuerySet({ type: "timestamp", count: MAXQ });
    const qbuf = device.createBuffer({
      size: MAXQ * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const rbuf = device.createBuffer({
      size: MAXQ * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    const spans = [];
    let cur = null, qi = 0;
    const phase = (label) => {
      if (cur) cur.end();
      if (qi + 2 > MAXQ) throw new Error("profile: query set too small");
      spans.push([label, qi]);
      cur = enc.beginComputePass({ timestampWrites: {
        querySet: qs, beginningOfPassWriteIndex: qi, endOfPassWriteIndex: qi + 1 } });
      qi += 2;
      return cur;
    };
    this.encodePrefill(phase("pre.head"), phase);
    cur.end();
    enc.resolveQuerySet(qs, 0, qi, qbuf, 0);
    enc.copyBufferToBuffer(qbuf, 0, rbuf, 0, qi * 8);
    device.queue.submit([enc.finish()]);
    await rbuf.mapAsync(GPUMapMode.READ, 0, qi * 8);
    const t = new BigUint64Array(rbuf.getMappedRange(0, qi * 8).slice(0));
    rbuf.unmap();
    qs.destroy(); qbuf.destroy(); rbuf.destroy();
    const totals = new Map();
    for (const [label, i] of spans)
      totals.set(label, (totals.get(label) || 0) + Number(t[i + 1] - t[i]));
    const rows = [...totals.entries()].map(([label, ns]) => [label, ns / 1e6])
      .sort((a, b) => b[1] - a[1]);
    const total = rows.reduce((a, [, ms]) => a + ms, 0);
    return { seq: this.preSeq, totalMs: total,
             rows: rows.map(([label, ms]) =>
               ({ label, ms: +ms.toFixed(2), pct: +(100 * ms / total).toFixed(1) })) };
  }

  /* `phase(label)` - when given - is called at each kernel boundary and must
   * return the pass to encode the next kernels into. The profiler uses it to
   * put every kernel kind in its own timestamped pass; normal generation
   * passes no callback and everything lands in the single pass it was given. */
  encodeStep(pass, phase, regionBase = 0) {
    const { cfg, pipe, maxDim } = this;
    const ph = (label) => { if (phase) pass = phase(label); };
    const use = (p, slot) => {
      pass.setPipeline(p);
      pass.setBindGroup(0, this.bindGroup(this.slotShard[slot] | 0),
                        [regionBase + slot * PARAM_STRIDE]);
    };
    const dw = (...a) => pass.dispatchWorkgroups(...a);
    const up = (a, b) => Math.ceil(a / b);
    const rowGrid = (rows) => rows <= maxDim ? [rows, 1] : [maxDim, up(rows, maxDim)];

    /* Every embedding shard is dispatched; only the one holding the token's
     * row writes anything. */
    ph("embed");
    for (const slot of this.off.embed) {
      use(pipe.embed, slot);
      dw(up(cfg.hidden / 4, 64));
    }

    for (let l = 0; l < cfg.layers; l++) {
      const L = this.off.layer[l];
      ph("rmsnorm"); use(pipe.rmsnorm, L.rms1); dw(1);
      if (this.quantizeActivations) { ph("quantAct"); use(pipe.quantAct, L.qxn); dw(cfg.hidden / 64); }
      ph("matvec.qkv"); use(pipe.matvec, L.qkv);
      dw(...rowGrid(cfg.qDim + 2 * cfg.kvDim));
      ph("qkrope"); use(pipe.qkrope, L.qkrope); dw(cfg.heads + cfg.kvHeads);
      ph("attn.scores"); use(pipe.scores, L.scores);
      if (this.hasSubgroups) { dw(up(this.kvLen, 8), cfg.heads); }
      else { dw(up(this.kvLen, 64), cfg.heads); }
      ph("attn.softmax"); use(pipe.softmax, L.softmax); dw(cfg.heads);
      ph("attn.apply"); use(pipe.apply, L.apply);
      dw(up(cfg.headDim, 64), cfg.heads, this.attnSlices);
      ph("attn.merge"); use(pipe.merge, L.merge); dw(up(cfg.headDim, 64), cfg.heads);
      if (this.quantizeActivations) { ph("quantAct"); use(pipe.quantAct, L.qattn); dw(cfg.qDim / 64); }
      ph("matvec.o"); use(pipe.matvec, L.o); dw(...rowGrid(cfg.hidden));
      ph("rmsnorm"); use(pipe.rmsnorm, L.rms2); dw(1);
      if (this.quantizeActivations) { ph("quantAct"); use(pipe.quantAct, L.qxn2); dw(cfg.hidden / 64); }
      ph("matvec.gu"); use(pipe.matvec, L.gu); dw(...rowGrid(2 * cfg.inter));
      ph("swiglu"); use(pipe.swiglu, L.swiglu); dw(up(cfg.inter, 256));
      if (this.quantizeActivations) { ph("quantAct"); use(pipe.quantAct, L.qg); dw(cfg.inter / 64); }
      ph("matvec.down"); use(pipe.matvec, L.down); dw(...rowGrid(cfg.hidden));
    }

    ph("rmsnorm"); use(pipe.rmsnorm, this.off.rmsFinal); dw(1);
    if (this.quantizeActivations) { ph("quantAct"); use(pipe.quantAct, this.off.qfinal); dw(cfg.hidden / 64); }
    ph("logits");
    for (const slot of this.off.logits) {
      use(pipe.logits, slot);
      dw(...rowGrid(this.slotRows(slot)));
    }
    ph("argmax"); use(pipe.argmax, this.off.argmax); dw(1);
  }

  /* Per-kernel GPU time for one generation step, from timestamp queries.
   *
   * Wall clocks are useless here: a background tab throttles the event loop,
   * not the GPU, so only on-GPU timestamps say where a step actually goes.
   * Every kernel kind gets its own pass, which adds pass-transition overhead -
   * the shares are trustworthy, the inflated total is not. Call it after a
   * generate() so a context and a token id are in place. */
  async profileStep(reps = 10) {
    if (this.lost) throw new Error(`GPU device lost (${this.lost})`);
    if (!this.hasTimestamps) throw new Error("timestamp-query not available");
    const { device } = this;
    const MAXQ = 2048;
    const qs = device.createQuerySet({ type: "timestamp", count: MAXQ });
    const qbuf = device.createBuffer({
      size: MAXQ * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const rbuf = device.createBuffer({
      size: MAXQ * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const totals = new Map();
    for (let rep = 0; rep < reps; rep++) {
      this.setStepParams(this.kvLen);
      const enc = device.createCommandEncoder();
      const spans = [];
      let cur = null, qi = 0;
      const phase = (label) => {
        if (cur) cur.end();
        if (qi + 2 > MAXQ) throw new Error("profile: query set too small");
        spans.push([label, qi]);
        cur = enc.beginComputePass({ timestampWrites: {
          querySet: qs, beginningOfPassWriteIndex: qi, endOfPassWriteIndex: qi + 1 } });
        qi += 2;
        return cur;
      };
      this.encodeStep(phase("embed"), phase);
      cur.end();
      enc.resolveQuerySet(qs, 0, qi, qbuf, 0);
      enc.copyBufferToBuffer(qbuf, 0, rbuf, 0, qi * 8);
      device.queue.submit([enc.finish()]);
      await rbuf.mapAsync(GPUMapMode.READ, 0, qi * 8);
      const t = new BigUint64Array(rbuf.getMappedRange(0, qi * 8).slice(0));
      rbuf.unmap();
      for (const [label, i] of spans) {
        const ns = Number(t[i + 1] - t[i]);
        totals.set(label, (totals.get(label) || 0) + ns);
      }
    }
    qs.destroy(); qbuf.destroy(); rbuf.destroy();

    const rows = [...totals.entries()]
      .map(([label, ns]) => [label, ns / reps / 1e6])
      .sort((a, b) => b[1] - a[1]);
    const total = rows.reduce((a, [, ms]) => a + ms, 0);
    return { kvLen: this.kvLen, reps,
             totalMs: total,
             rows: rows.map(([label, ms]) =>
               ({ label, ms: +ms.toFixed(3), pct: +(100 * ms / total).toFixed(1) })) };
  }

  /* Prefill on the GPU from wasm-built embeddings, then generate.
   * `embedsPtr` points at [seq, hidden] f32 in wasm memory.
   *
   * Long prompts prefill in chained chunks through the suffix path rather
   * than one shot. The one-shot path's score scratch is heads x seq x seq -
   * quadratic, 164 MB at a 1600-token prompt and growing - while a chunk's is
   * heads x 512 x seq. The chunks compute identical attention (the suffix
   * kernels read the same cache the one-shot pass would have written; the
   * split harness verified transcripts byte-identical on all 23 goldens), so
   * this trades nothing but a few extra dispatches. */
  async prefillAndGenerate(embedsPtr, seq, maxNew, onPiece) {
    const CHUNK = 512;
    /* One-shot up to 1024 positions (67 MB of score scratch); chunk past it,
     * where the quadratic scratch would keep growing. */
    if (seq > 1024) {
      const t0 = performance.now();
      let p0 = 0;
      while (seq - p0 > CHUNK + 128) {
        const end = p0 + CHUNK;
        /* Reserving the final context up front keeps the cache layout stable
         * across chunks, so the suffix calls never re-lay it out. */
        if (p0 === 0) await this.prefillOneShot(embedsPtr, end, 0, null, seq + maxNew);
        else await this.prefillSuffixAndGenerate(embedsPtr, end, p0, 0, null);
        p0 = end;
      }
      const r = await this.prefillSuffixAndGenerate(embedsPtr, seq, p0, maxNew, onPiece);
      this.prefillMs = performance.now() - t0 - (this.lastProfile?.generateMs || 0);
      return r;
    }
    return this.prefillOneShot(embedsPtr, seq, maxNew, onPiece);
  }

  async prefillOneShot(embedsPtr, seq, maxNew, onPiece, reserve) {
    if (this.lost) throw new Error(`GPU device lost (${this.lost})`);
    const { device, cfg, M } = this;
    const t0 = performance.now();
    this.prepareContext(reserve || seq, maxNew, { prefillSeq: seq });

    /* Upload transposed: the prefill kernels want [dim][seqPad]. One staging
     * array and one writeBuffer - issuing a call per dim measured seconds of
     * pure overhead against a busy tab. */
    const sp = this.seqPad;
    const src = new Float32Array(M.HEAPF32.buffer, embedsPtr, seq * cfg.hidden);
    const stage = new Float32Array(cfg.hidden * sp);
    for (let s2 = 0; s2 < seq; s2++) {
      const row = s2 * cfg.hidden;
      for (let d = 0; d < cfg.hidden; d++) stage[d * sp + s2] = src[row + d];
    }
    device.queue.writeBuffer(this.bufAct, this.PT.x * 4, stage);

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

  /* Prefill only the rows past `p0` - the retained prefix's KV is already in
   * the cache from an earlier call on this decoder - then generate. This is
   * the streaming path: each chunk extends the audio, the caller finds the
   * longest unchanged embedding prefix, and only the tail is re-prefilled.
   * `embedsPtr` still points at the *full* [seq][hidden] embeddings; only
   * columns p0.. are uploaded. */
  async prefillSuffixAndGenerate(embedsPtr, seq, p0, maxNew, onPiece) {
    if (this.lost) throw new Error(`GPU device lost (${this.lost})`);
    if (p0 <= 0) return this.prefillAndGenerate(embedsPtr, seq, maxNew, onPiece);
    const { device, cfg, M } = this;
    const t0 = performance.now();
    const mark = (o, k, t) => { o[k] = +(performance.now() - t).toFixed(1); };
    const lp = this.lastProfile = {};
    const nNew = seq - p0;
    if (nNew < 1) throw new Error("suffix prefill needs at least one new row");
    let tp = performance.now();
    this.prepareContext(seq, maxNew, { prefillSeq: nNew, prefillBase: p0 });
    mark(lp, "prepareMs", tp);

    const sp = this.seqPad;
    const src = new Float32Array(M.HEAPF32.buffer, embedsPtr, seq * cfg.hidden);
    const stage = new Float32Array(cfg.hidden * sp);
    for (let s2 = 0; s2 < nNew; s2++) {
      const row = (p0 + s2) * cfg.hidden;
      for (let d = 0; d < cfg.hidden; d++) stage[d * sp + s2] = src[row + d];
    }
    device.queue.writeBuffer(this.bufAct, this.PT.x * 4, stage);
    mark(lp, "uploadMs", tp);

    tp = performance.now();
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.encodePrefill(pass);
    pass.end();
    enc.copyBufferToBuffer(this.bufTok, 0, this.bufTokRead, 0, 4);
    device.queue.submit([enc.finish()]);
    await this.bufTokRead.mapAsync(GPUMapMode.READ, 0, 4);
    const first = new Uint32Array(this.bufTokRead.getMappedRange(0, 4).slice(0))[0];
    this.bufTokRead.unmap();
    mark(lp, "gpuPrefillMs", tp);
    this.prefillMs = performance.now() - t0;

    tp = performance.now();
    const r = await this.generate(first, seq, maxNew, onPiece, { keepContext: true });
    mark(lp, "generateMs", tp);
    lp.nNew = nNew; lp.seq = seq;
    return r;
  }

  /* Generate from `firstToken` (produced by the CPU prefill). */
  async generate(firstToken, kvLen, maxNew, onPiece, opts = {}) {
    if (this.lost) throw new Error(`GPU device lost (${this.lost})`);
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
    let stop = false;
    for (let step = 0; step < maxNew && !stop; ) {
      /* K chained steps per submit. Every step's argmax leaves the next token
       * id in bufTok on the GPU, so the chain needs no readback; each step
       * copies its id out and one mapAsync at the end returns all K. An
       * end-of-text inside the batch just discards the steps after it. */
      const K = Math.min(STEP_REGIONS, maxNew - step);
      for (let k = 0; k < K; k++) this.writeStepRegion(pos + k, k);

      const enc = device.createCommandEncoder();
      for (let k = 0; k < K; k++) {
        this.kvLen = pos + k + 1;    /* scores/apply dispatch geometry */
        const pass = enc.beginComputePass();
        this.encodeStep(pass, null, k * this.paramBytes);
        pass.end();
        enc.copyBufferToBuffer(this.bufTok, 0, this.bufTokRead, k * 4, 4);
      }
      device.queue.submit([enc.finish()]);

      await this.bufTokRead.mapAsync(GPUMapMode.READ, 0, K * 4);
      const got = new Uint32Array(this.bufTokRead.getMappedRange(0, K * 4).slice(0));
      this.bufTokRead.unmap();

      for (let k = 0; k < K; k++) {
        const id = got[k];
        ids.push(id);
        n++;
        pos++;
        step++;
        if (id === cfg.imEnd || id === cfg.endOfText) { stop = true; break; }
        emit(id);
      }
    }
    return { text, tokens: n, ids };
  }
}
