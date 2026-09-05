import { freshHeap } from "./heap.js";
/*
 * webgpu-encoder.js - Qwen3-ASR audio tower on the GPU.
 *
 * Currently the transformer half: the 24 layers, the final LayerNorm and the
 * two output projections. The Conv2D stem stays in wasm and hands over its
 * output; see qwen_enc_tap in qwen_asr.h for the boundary this is verified
 * against.
 *
 * The tower is not the decoder with different weights, which is why it does
 * not reuse those shaders:
 *
 *   - LayerNorm with a bias, not RMSNorm
 *   - every linear carries a bias
 *   - attention is bidirectional inside a fixed window, with no RoPE and no
 *     grouped KV: 16 heads that each own their K and V
 *   - the FFN is GELU(fc1) -> fc2, not SwiGLU
 *
 * What it does reuse is the layout the decoder prefill proved out: activations
 * live transposed, [dim][seqPad], so every kernel's lanes run along the
 * sequence axis. All global access is then coalesced and the row-wise
 * reductions LayerNorm needs become per-lane loops with no cross-lane traffic.
 */

/* Must match the enum in wasm/qwen_wasm.c. */
const E_CONV1 = 0, E_CONV1_B = 1, E_CONV2 = 2, E_CONV2_B = 3,
      E_CONV3 = 4, E_CONV3_B = 5, E_CONV_OUT = 6,
      E_ATTN_NORM_W = 7, E_ATTN_NORM_B = 8,
      E_Q = 9, E_Q_B = 10, E_K = 11, E_K_B = 12, E_V = 13, E_V_B = 14,
      E_O = 15, E_O_B = 16,
      E_FFN_NORM_W = 17, E_FFN_NORM_B = 18,
      E_FC1 = 19, E_FC1_B = 20, E_FC2 = 21, E_FC2_B = 22,
      E_LN_POST_W = 23, E_LN_POST_B = 24,
      E_PROJ1 = 25, E_PROJ1_B = 26, E_PROJ2 = 27, E_PROJ2_B = 28;

const PARAM_STRIDE = 256;   // >= minUniformBufferOffsetAlignment
const PARAM_FIELDS = 16;

const F_GELU = 1;           // apply GELU to the matmul result
const F_RESIDUAL = 2;       // add into the destination instead of overwriting

const HEADER = `
struct Params {
  wordBase  : u32,
  scaleBase : u32,
  rows      : u32,
  cols      : u32,
  xOff      : u32,
  yOff      : u32,
  stride    : u32,
  n         : u32,
  vecA      : u32,
  vecB      : u32,
  flags     : u32,
  p0        : u32,
  p1        : u32,
  fa        : f32,
  fb        : f32,
  fc        : f32,
};
@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read> quants : array<u32>;
@group(0) @binding(2) var<storage, read> scales : array<f32>;
@group(0) @binding(3) var<storage, read> vecs   : array<f32>;
@group(0) @binding(4) var<storage, read_write> act : array<f32>;
@group(0) @binding(5) var<storage, read_write> scratch : array<f32>;

/* Sign-extend the four int8 lanes of a packed u32. */
fn i8x4(v : u32) -> vec4<i32> {
  return vec4<i32>(bitcast<i32>(v << 24u) >> 24u,
                   bitcast<i32>(v << 16u) >> 24u,
                   bitcast<i32>(v << 8u)  >> 24u,
                   bitcast<i32>(v) >> 24u);
}

/* The exact identity the C path uses: 0.5*(1+tanh(z)) == sigmoid(2z), which
 * keeps this a single exp rather than a tanh. */
fn gelu(x : f32) -> f32 {
  let z = 1.5957691216057308 * (x + 0.044715 * x * x * x);
  return x / (1.0 + exp(-z));
}
`;

/* ---- LayerNorm: one lane per sequence position, looping over dim ---- */
const LN_WGSL = HEADER + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let sq = gid.x;
  if (sq >= P.n) { return; }
  let d = P.rows;

  var sum : f32 = 0.0;
  for (var i = 0u; i < d; i = i + 1u) { sum = sum + act[P.xOff + i * P.stride + sq]; }
  let mean = sum / f32(d);

  var vsum : f32 = 0.0;
  for (var i = 0u; i < d; i = i + 1u) {
    let v = act[P.xOff + i * P.stride + sq] - mean;
    vsum = vsum + v * v;
  }
  let inv = inverseSqrt(vsum / f32(d) + P.fa);

  for (var i = 0u; i < d; i = i + 1u) {
    let v = (act[P.xOff + i * P.stride + sq] - mean) * inv;
    act[P.yOff + i * P.stride + sq] = v * vecs[P.vecA + i] + vecs[P.vecB + i];
  }
}
`;

/* ---- y[rows][n] = W[rows][cols] @ x[cols][n] + bias, optional GELU ----
 *
 * Same shape as the decoder's prefill GEMM: a 128x64 output tile per
 * workgroup, 16x16 threads each owning eight vec4 accumulators. The
 * accumulators are separate variables rather than an array because an array
 * indexed dynamically lands in thread memory instead of registers, which
 * measured 8x slower there.
 */
const MATMUL_WGSL = HEADER + `
const TR : u32 = 128u;
const TS : u32 = 64u;
const TK : u32 = 16u;

var<workgroup> ws : array<f32, 2048>;   // TR x TK
var<workgroup> xs : array<f32, 1024>;   // TK x TS

fn store_one(row : u32, sq : u32, val : f32) {
  if (sq >= P.n) { return; }
  var v = val + vecs[P.vecA + row];
  if ((P.flags & 1u) != 0u) { v = gelu(v); }
  let o = P.yOff + row * P.stride + sq;
  if ((P.flags & 2u) != 0u) { act[o] = act[o] + v; } else { act[o] = v; }
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
  let wb = lid.y * 8u;
  let xb = lid.x * 4u;

  var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0);
  var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
  var a4 = vec4<f32>(0.0); var a5 = vec4<f32>(0.0);
  var a6 = vec4<f32>(0.0); var a7 = vec4<f32>(0.0);

  var kb : u32 = 0u;
  loop {
    if (kb >= P.cols) { break; }

    for (var t = 0u; t < 2u; t = t + 1u) {
      let idx = tid + t * 256u;
      let rr = idx / 4u;
      let wq = idx % 4u;
      let row = r0 + rr;
      let col0 = kb + wq * 4u;
      var v = vec4<f32>(0.0);
      if (row < P.rows) {
        let word = quants[P.wordBase + row * nwords + col0 / 4u];
        let sc = scales[P.scaleBase + row * nblocks + col0 / 64u] * 127.0;
        v = unpack4x8snorm(word) * sc;
      }
      let base = rr * TK + wq * 4u;
      ws[base] = v.x; ws[base + 1u] = v.y; ws[base + 2u] = v.z; ws[base + 3u] = v.w;
    }

    for (var t = 0u; t < 4u; t = t + 1u) {
      let idx = tid + t * 256u;
      let kk = idx / TS;
      let ss = idx % TS;
      let col = kb + kk;
      let sq = s0 + ss;
      var val : f32 = 0.0;
      if (col < P.cols && sq < P.n) { val = act[P.xOff + col * P.stride + sq]; }
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

  let row0 = r0 + wb;
  let sq0 = s0 + xb;
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

/* ---- Windowed bidirectional attention ----
 *
 * Windows are contiguous, equal-sized runs of positions (the last one short),
 * and a query attends to every key in its own window and nothing else. There
 * is no causal mask and no RoPE. p0 is the window size, p1 the head count.
 *
 * One lane per (head, query): the whole softmax and the weighted sum are a
 * per-lane loop over at most p0 keys, which is 104 for the 1.7B model. That is
 * small enough that splitting the reduction the way the decoder has to at long
 * contexts would cost more in dispatches than it saves.
 */
const ATTN_WGSL = HEADER + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= P.n * P.p1) { return; }
  let h = idx / P.n;               // head
  let q = idx % P.n;               // query position

  let hd = P.rows;                 // head_dim
  let win = P.p0;
  let w0 = (q / win) * win;
  var w1 = w0 + win;
  if (w1 > P.n) { w1 = P.n; }

  let headOff = h * hd * P.stride;
  let qBase = P.xOff + headOff;    // Q
  let kBase = P.yOff + headOff;    // K
  let vBase = P.vecA + headOff;    // V
  let oBase = P.vecB + headOff;    // out
  let scoreBase = idx * win;

  var best : f32 = -3.0e38;
  for (var k = w0; k < w1; k = k + 1u) {
    var dot : f32 = 0.0;
    for (var d = 0u; d < hd; d = d + 1u) {
      dot = dot + act[qBase + d * P.stride + q] * act[kBase + d * P.stride + k];
    }
    let s = dot * P.fa;
    scratch[scoreBase + (k - w0)] = s;
    best = max(best, s);
  }

  var sum : f32 = 0.0;
  for (var k = w0; k < w1; k = k + 1u) {
    let e = exp(scratch[scoreBase + (k - w0)] - best);
    scratch[scoreBase + (k - w0)] = e;
    sum = sum + e;
  }
  let inv = 1.0 / sum;

  for (var d = 0u; d < hd; d = d + 1u) {
    var acc : f32 = 0.0;
    for (var k = w0; k < w1; k = k + 1u) {
      acc = acc + scratch[scoreBase + (k - w0)] * act[vBase + d * P.stride + k];
    }
    act[oBase + d * P.stride + q] = acc * inv;
  }
}
`;

/* Subgroup variant: one workgroup of 32 lanes per (query, head), lanes
 * striding the key axis, which is the axis K and V are contiguous in. The
 * portable kernel above runs one *thread* per (query, head) - scores, softmax
 * and the V sum all serial in that thread - which is 8k threads for a 41 s
 * clip and profiled at 26% of the tower. Scores and the V sum fold with
 * subgroupAdd; the probability row lives in workgroup memory.
 *
 * Assumes the workgroup is exactly one subgroup (Apple: size 32, always).
 * The caller checks that and the window bound, and keeps the portable kernel
 * otherwise. */
const ATTN_SG_MAXWIN = 1024;
const ATTN_SG_WGSL = "enable subgroups;\n" + HEADER + `
var<workgroup> pw : array<f32, 1024u>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(subgroup_size) sgsz : u32) {
  let q = wid.x;
  let h = wid.y;
  if (q >= P.n) { return; }

  let hd = P.rows;
  let win = P.p0;
  let w0 = (q / win) * win;
  var w1 = w0 + win;
  if (w1 > P.n) { w1 = P.n; }
  let nk = w1 - w0;

  let headOff = h * hd * P.stride;
  let qBase = P.xOff + headOff;
  let kBase = P.yOff + headOff;
  let vBase = P.vecA + headOff;
  let oBase = P.vecB + headOff;

  /* Scores: each lane owns keys w0+lid.x, w0+lid.x+32, ...
   * (The q vector is re-read per key on purpose: it is a warp-wide broadcast
   * out of L1, and staging it in workgroup memory measured *slower* -
   * 104 -> 158 ms - the barrier and shared traffic cost more than the reads.) */
  var m : f32 = -3.0e38;
  var kk : u32 = lid.x;
  loop {
    if (kk >= nk) { break; }
    var dot : f32 = 0.0;
    for (var d = 0u; d < hd; d = d + 1u) {
      dot = dot + act[qBase + d * P.stride + q] * act[kBase + d * P.stride + w0 + kk];
    }
    let sc = dot * P.fa;
    pw[kk] = sc;
    m = max(m, sc);
    kk = kk + sgsz;
  }
  let mx = subgroupMax(m);

  var sum : f32 = 0.0;
  kk = lid.x;
  loop {
    if (kk >= nk) { break; }
    let e = exp(pw[kk] - mx);
    pw[kk] = e;
    sum = sum + e;
    kk = kk + sgsz;
  }
  let inv = 1.0 / subgroupAdd(sum);
  kk = lid.x;
  loop {
    if (kk >= nk) { break; }
    pw[kk] = pw[kk] * inv;
    kk = kk + sgsz;
  }
  workgroupBarrier();

  /* V sum: lanes on the key axis, one subgroup add per output dim. */
  for (var d = 0u; d < hd; d = d + 1u) {
    var acc : f32 = 0.0;
    let rb = vBase + d * P.stride + w0;
    kk = lid.x;
    loop {
      if (kk >= nk) { break; }
      acc = acc + pw[kk] * act[rb + kk];
      kk = kk + sgsz;
    }
    let total = subgroupAdd(acc);
    if (lid.x == 0u) { act[oBase + d * P.stride + q] = total; }
  }
}
`;

export { HEADER, LN_WGSL, MATMUL_WGSL, ATTN_WGSL, CONV_WGSL, RESHAPE_WGSL, ADD_WGSL,
         E_CONV1, E_CONV1_B, E_CONV2, E_CONV2_B, E_CONV3, E_CONV3_B, E_CONV_OUT,
         E_ATTN_NORM_W, E_ATTN_NORM_B, E_Q, E_Q_B, E_K, E_K_B, E_V, E_V_B,
         E_O, E_O_B, E_FFN_NORM_W, E_FFN_NORM_B, E_FC1, E_FC1_B, E_FC2, E_FC2_B,
         E_LN_POST_W, E_LN_POST_B, E_PROJ1, E_PROJ1_B, E_PROJ2, E_PROJ2_B,
         PARAM_STRIDE, PARAM_FIELDS, F_GELU, F_RESIDUAL };


/* ---- Conv2D stem ----
 *
 * The three stem layers are 3x3, stride 2, pad 1, and the C encoder runs them
 * as im2col plus a GEMM. Materializing the im2col matrix here would cost 55 MB
 * per chunk group for the second layer, so instead this is the same tiled GEMM
 * with the activation tile *gathered*: the tile loader turns a (K, column)
 * pair into (in_channel, tap) and (batch, out_row, out_col) and reads the
 * input pixel directly. Same arithmetic, no intermediate buffer.
 *
 * The kernels are f32, not Q8 - that is how the model stores them - so the
 * weight tile comes from `vecs` rather than quants/scales.
 *
 * Layout for a stem stage is [channel][batch * h * w], with P.stride the
 * padded column count.
 *
 *   wordBase  f32 weight base        scaleBase input row stride
 *   rows      out channels           cols      in_channels * 9
 *   p0        input h                p1        input w
 *   vecA      bias base              flags     F_GELU
 */
const CONV_WGSL = HEADER + `
const TR : u32 = 128u;
const TS : u32 = 64u;
const TK : u32 = 16u;

var<workgroup> ws : array<f32, 2048>;
var<workgroup> xs : array<f32, 1024>;

fn store_one(row : u32, sq : u32, val : f32) {
  if (sq >= P.n) { return; }
  var v = val + vecs[P.vecA + row];
  if ((P.flags & 1u) != 0u) { v = gelu(v); }
  act[P.yOff + row * P.stride + sq] = v;
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
  let hIn = P.p0;
  let wIn = P.p1;
  let hOut = (hIn - 1u) / 2u + 1u;
  let wOut = (wIn - 1u) / 2u + 1u;
  let plane = hOut * wOut;

  let r0 = wid.y * TR;
  let s0 = wid.x * TS;
  let tid = lid.y * 16u + lid.x;
  let wb = lid.y * 8u;
  let xb = lid.x * 4u;

  var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0);
  var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
  var a4 = vec4<f32>(0.0); var a5 = vec4<f32>(0.0);
  var a6 = vec4<f32>(0.0); var a7 = vec4<f32>(0.0);

  var kb : u32 = 0u;
  loop {
    if (kb >= P.cols) { break; }

    /* Weight tile, straight f32. */
    for (var t = 0u; t < 8u; t = t + 1u) {
      let idx = tid + t * 256u;      // 0..2047
      let rr = idx / TK;
      let kk = idx % TK;
      let row = r0 + rr;
      let col = kb + kk;
      var v : f32 = 0.0;
      if (row < P.rows && col < P.cols) { v = vecs[P.wordBase + row * P.cols + col]; }
      ws[rr * TK + kk] = v;
    }

    /* Activation tile, gathered: this is the im2col that is never built. */
    for (var t = 0u; t < 4u; t = t + 1u) {
      let idx = tid + t * 256u;
      let kk = idx / TS;
      let ss = idx % TS;
      let col = kb + kk;
      let sq = s0 + ss;
      var val : f32 = 0.0;
      if (col < P.cols && sq < P.n) {
        let ic = col / 9u;
        let tap = col % 9u;
        let b = sq / plane;
        let rem = sq % plane;
        let oh = rem / wOut;
        let ow = rem % wOut;
        let ih = i32(oh * 2u + tap / 3u) - 1;
        let iw = i32(ow * 2u + tap % 3u) - 1;
        if (ih >= 0 && ih < i32(hIn) && iw >= 0 && iw < i32(wIn)) {
          let pix = b * hIn * wIn + u32(ih) * wIn + u32(iw);
          val = act[P.xOff + ic * P.scaleBase + pix];
        }
      }
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

  let row0 = r0 + wb;
  let sq0 = s0 + xb;
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

/* ---- [channel][batch*h*w] -> [channel*h][token], the stem's reshape ----
 *
 * The C encoder flattens conv3's [480, h3, w3] per chunk into [w3, 480*h3] and
 * projects that. Written straight into the transposed layout the projection
 * GEMM wants, it is a gather: row (ch*h3 + f), column (chunk*w3 + t).
 *
 *   rows h3   p0 w3   p1 tokenBase   scaleBase input row stride
 */
const RESHAPE_WGSL = HEADER + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let tok = gid.x;
  if (tok >= P.n) { return; }
  let row = gid.y;                 // ch * h3 + f
  let h3 = P.rows;
  let w3 = P.p0;
  let ch = row / h3;
  let f = row % h3;
  let b = tok / w3;
  let t = tok % w3;
  let src = P.xOff + ch * P.scaleBase + b * h3 * w3 + f * w3 + t;
  act[P.yOff + row * P.stride + P.p1 + tok] = act[src];
}
`;

/* ---- Add the per-chunk sinusoidal position embeddings ----
 * Uploaded already transposed, [d_model][n], so this is a straight add.
 */
const ADD_WGSL = HEADER + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let sq = gid.x;
  if (sq >= P.n) { return; }
  let row = gid.y;
  act[P.yOff + row * P.stride + sq] =
      act[P.yOff + row * P.stride + sq] + act[P.xOff + row * P.stride + sq];
}
`;

/* ------------------------------------------------------------------ driver */

const ceilDiv = (a, b) => Math.floor((a + b - 1) / b);

export class WebGPUEncoder {
  constructor(Module, opts = {}) {
    this.M = Module;
    this.opts = opts;
    this.ready = false;
  }

  async init(report = () => {}) {
    const M = this.M;

    const shPtr = M._qwen_wasm_alloc(16 * 4) >>> 0;
    if (M._qwen_wasm_enc_shape(shPtr) < 0) throw new Error("encoder shape unavailable");
    const sh = new Int32Array(freshHeap(M).HEAPU8.buffer, shPtr, 16).slice();
    M._qwen_wasm_release(shPtr);
    this.dModel = sh[0];
    this.layers = sh[1];
    this.heads = sh[2];
    this.headDim = sh[3];
    this.ffnDim = sh[4];
    this.outDim = sh[5];
    this.nWindowInfer = sh[7];
    this.chunkSize = sh[8];
    this.convProjDim = sh[9];
    this.convHidden = sh[10];

    /* Window size in tokens, derived the way the C encoder derives it: the
     * conv stem's three stride-2 layers turn a chunk of mel frames into
     * tokensPerChunk tokens, and a window spans nWindowInfer/chunkSize chunks. */
    const down = (w) => Math.floor((w + 2 - 3) / 2) + 1;
    this.tokensPerChunk = down(down(down(this.chunkSize)));
    this.window = this.tokensPerChunk * Math.floor(this.nWindowInfer / this.chunkSize);

    /* Precompute sinusoidal position embedding table for chunk-local positions (0..15).
     * Slices are reused per-chunk directly on CPU/GPU upload without per-chunk trig overhead.
     * Dynamic growth via ensurePeTable() protects against chunk groups with w3 > maxPeT. */
    this.maxPeT = 0;
    this.ensurePeTable(16);

    /* Chunks convolved together. Set here rather than in prepare() because
     * convPlan() runs first and reads it - with it undefined the planner fell
     * back to one chunk per group, which silently overran the uniform buffer
     * on the first call only. */
    this.convGroup = 8;

    if (this.opts?.device) {
      this.device = this.opts.device;
      this.adapter = this.opts.adapter;
      this.hasTimestamps = this.device.features.has("timestamp-query");
      this.sgExact32 = this.device.features.has("subgroups") &&
        (this.adapter?.info?.subgroupMinSize === 32) && (this.adapter?.info?.subgroupMaxSize === 32);
    } else {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) throw new Error("no WebGPU adapter");
      const lim = adapter.limits;
      const cap = (typeof window !== "undefined" && window.__gpuBindingCap) ||
                  lim.maxStorageBufferBindingSize;
      const want = Math.min(cap, lim.maxStorageBufferBindingSize, 1 << 30);
      const wantFeatures = ["timestamp-query", "subgroups"]
        .filter((f) => adapter.features.has(f));
      this.adapter = adapter;
      this.device = await adapter.requestDevice({
        requiredFeatures: wantFeatures,
        requiredLimits: { maxBufferSize: want, maxStorageBufferBindingSize: want },
      });
      this.hasTimestamps = this.device.features.has("timestamp-query");
      this.sgExact32 = this.device.features.has("subgroups") &&
        (adapter.info?.subgroupMinSize === 32) && (adapter.info?.subgroupMaxSize === 32);
    }
    /* See the note in webgpu-decoder.js: a lost device has to be observable,
     * not discovered through a wrong answer. */
    this.device.lost.then((info) => {
      this.lost = info.reason || "unknown";
      console.error(`GPU device lost (${this.lost})`);
      report(`GPU device lost (${this.lost}); falling back to wasm`);
    });
    this.device.addEventListener("uncapturederror", (e) => {
      console.error("gpu error:", e.error.message);
      report("gpu error: " + e.error.message);
    });

    await this.loadWeights(report);
    await this.buildPipelines();
    this.ready = true;
  }

  /* Read the descriptor table and pack every weight into three buffers:
   * int8 quants, their f32 scales, and the plain f32 vectors. */
  async loadWeights(report) {
    const M = this.M;
    /* Either the C descriptor table (weights in the wasm heap) or an injected
     * weightSource whose records carry file offsets and an async reader; see
     * webgpu-decoder.js. Records must arrive in the C emission order. */
    let recs;
    if (this.weightSource) {
      recs = this.weightSource.entries;
    } else {
      const MAXD = 1024;
      const dPtr = M._qwen_wasm_alloc(MAXD * 8 * 4) >>> 0;
      const n = M._qwen_wasm_enc_desc(dPtr, MAXD);
      if (n < 0) { M._qwen_wasm_release(dPtr); throw new Error("encoder weights unavailable"); }
      const d = new Uint32Array(freshHeap(M).HEAPU8.buffer, dPtr, n * 8).slice();
      M._qwen_wasm_release(dPtr);
      recs = [];
      for (let i = 0; i < n; i++) {
        const e = d.subarray(i * 8, i * 8 + 8);
        recs.push({ kind: e[0], layer: e[1], rows: e[2], cols: e[3],
                    qPtr: e[4], sPtr: e[5], fPtr: e[6], count: e[7] });
      }
    }

    /* Same shard budget as the decoder, and for the same reason: one binding
     * holding all 313 MB needs a limit tier that not every device offers. No
     * encoder matrix is larger than 4 MB, so nothing has to be split. */
    const SHARD_BUDGET = 256 << 20;
    const shards = [{ bytes: 0 }];

    let qBytes = 0, sFloats = 0, vFloats = 0;
    const entries = [];
    /* The conv-out projection has no bias, but the GEMM always adds one, so
     * reserve a run of zeros for it rather than branching in the shader. */
    for (const rec of recs) {
      if (rec.qPtr || rec.qoff != null) {
        if (rec.cols % 64 !== 0) throw new Error(`kind ${rec.kind}: cols ${rec.cols} not a multiple of 64`);
        const nq = rec.rows * rec.cols;
        let sh = shards[shards.length - 1];
        if (sh.bytes > 0 && sh.bytes + nq > SHARD_BUDGET) {
          shards.push({ bytes: 0 });
          sh = shards[shards.length - 1];
        }
        rec.shard = shards.length - 1;
        rec.wordBase = sh.bytes / 4;
        rec.scaleBase = sFloats;
        sh.bytes += nq;
        qBytes += nq;
        sFloats += rec.rows * (rec.cols / 64);
      } else if ((rec.fPtr || rec.foff != null) && rec.count) {
        rec.vecBase = vFloats;
        vFloats += rec.count;
      } else {
        throw new Error(`kind ${rec.kind}: f32 matrices are not supported on the GPU path`);
      }
      entries.push(rec);
    }
    this.entries = entries;
    this.zeroVec = vFloats;
    vFloats += Math.max(this.dModel, this.outDim);

    report(`allocating ${((qBytes + sFloats * 4 + vFloats * 4) / 1e9).toFixed(2)} GB on the GPU...`);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.bufQuants = shards.map((sh) =>
      this.device.createBuffer({ size: Math.max(4, sh.bytes), usage }));
    this.shardCount = shards.length;
    this.bufScales = this.device.createBuffer({ size: Math.max(4, sFloats * 4), usage });
    this.bufVecs = this.device.createBuffer({ size: Math.max(4, vFloats * 4), usage });
    this.weightBytes = qBytes + sFloats * 4 + vFloats * 4;

    const src = this.weightSource;
    const put = async (buf, dstOff, off, len) => {
      const chunk = await src.read(off, len);
      const u8 = chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk);
      this.device.queue.writeBuffer(buf, dstOff, u8, 0, len);
    };
    for (const rec of entries) {
      if (rec.qPtr || rec.qoff != null) {
        const qn = rec.rows * rec.cols;
        const sn = rec.rows * (rec.cols / 64);
        if (src) {
          await put(this.bufQuants[rec.shard], rec.wordBase * 4, rec.qoff, qn);
          await put(this.bufScales, rec.scaleBase * 4, rec.soff, sn * 4);
        } else {
          this.device.queue.writeBuffer(this.bufQuants[rec.shard], rec.wordBase * 4,
            freshHeap(M).HEAPU8.buffer, rec.qPtr, qn);
          this.device.queue.writeBuffer(this.bufScales, rec.scaleBase * 4,
            freshHeap(M).HEAPU8.buffer, rec.sPtr, sn * 4);
        }
      } else if (src) {
        await put(this.bufVecs, rec.vecBase * 4, rec.foff, rec.count * 4);
      } else {
        this.device.queue.writeBuffer(this.bufVecs, rec.vecBase * 4,
          freshHeap(M).HEAPU8.buffer, rec.fPtr, rec.count * 4);
      }
    }
    await this.device.queue.onSubmittedWorkDone();
  }

  find(kind, layer = 0) {
    const rec = this.entries.find((e) => e.kind === kind && e.layer === layer);
    if (!rec) throw new Error(`weight kind ${kind} layer ${layer} missing`);
    return rec;
  }

  /* An explicit layout, not "auto". The uniform is bound once and stepped
   * through with a dynamic offset, and an inferred layout declares
   * hasDynamicOffset false - which makes every setBindGroup with an offset a
   * validation error. WebGPU validation errors do not throw: the pass simply
   * stops doing anything, so the symptom is a full encoder run finishing in
   * 3 ms with an all-zero result. */
  pipelineLayout() {
    const d = this.device;
    const storage = (t) => ({ visibility: GPUShaderStage.COMPUTE, buffer: { type: t } });
    this._bgl = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: PARAM_FIELDS * 4 } },
        { ...storage("read-only-storage"), binding: 1 },
        { ...storage("read-only-storage"), binding: 2 },
        { ...storage("read-only-storage"), binding: 3 },
        { ...storage("storage"), binding: 4 },
        { ...storage("storage"), binding: 5 },
      ],
    });
    return d.createPipelineLayout({ bindGroupLayouts: [this._bgl] });
  }

  async buildPipelines() {
    const layout = this.pipelineLayout();
    const mk = async (code, label) => {
      const mod = this.device.createShaderModule({ code });
      const info = await mod.getCompilationInfo?.();
      const errs = (info?.messages || []).filter((m) => m.type === "error");
      if (errs.length) throw new Error(`${label}: ${errs.map((m) => `${m.lineNum}: ${m.message}`).join("; ")}`);
      const pipe = this.device.createComputePipeline({
        layout, compute: { module: mod, entryPoint: "main" },
      });
      pipe.label = label;
      return pipe;
    };
    this.pipeLN = await mk(LN_WGSL, "layernorm");
    this.pipeMatmul = await mk(MATMUL_WGSL, "matmul");
    this.pipeAttn = await mk(ATTN_WGSL, "attention");
    this.pipeAttnSg = this.sgExact32 ? await mk(ATTN_SG_WGSL, "attention_sg") : null;
    this.pipeConv = await mk(CONV_WGSL, "conv2d");
    this.pipeReshape = await mk(RESHAPE_WGSL, "reshape");
    this.pipeAdd = await mk(ADD_WGSL, "add");
  }

  /* Activation regions. The transformer's are all [rows][seqPad] so its
   * kernels' lanes stay on the sequence axis; the conv stem's are sized by the
   * chunk group instead, since the stem works in image space. Offsets are in
   * floats. */
  prepare(tokens) {
    const sp = ceilDiv(tokens, 64) * 64;
    if (this.seqPad === sp) { this.tokens = tokens; return; }
    this.seqPad = sp;
    this.tokens = tokens;

    const d = this.dModel;
    const CH = this.convHidden;
    const cw = this.chunkSize;
    const w1 = ((cw - 1) >> 1) + 1, w2 = ((w1 - 1) >> 1) + 1, w3 = ((w2 - 1) >> 1) + 1;
    const G = this.convGroup;

    let off = 0;
    const region = (n) => { const o = off; off += n; return o; };
    this.oX = region(d * sp);
    this.oXN = region(d * sp);
    this.oQ = region(d * sp);
    this.oK = region(d * sp);
    this.oV = region(d * sp);
    this.oATT = region(d * sp);
    this.oFF = region(this.ffnDim * sp);
    this.oOUT = region(this.outDim * sp);
    this.oCX = region(this.convProjDim * sp);
    this.oPE = region(d * sp);
    /* Every chunk becomes its own image, so the mel is stored once at
     * chunk granularity. Must be an integer: a fractional region size makes
     * every later offset fractional, and writeBuffer rejects those. */
    const maxChunks = ceilDiv(tokens, w3) + G + 1;
    this.oMEL = region(128 * maxChunks * cw);
    this.oC1 = region(CH * G * 64 * w1);
    this.oC2 = region(CH * G * 32 * w2);
    this.oC3 = region(CH * G * 16 * w3);
    this.actFloats = off;

    const wantAct = this.actFloats * 4;
    let rebinding = false;
    if (!this.bufAct || (this.actCapBytes || 0) < wantAct) {
      this.bufAct?.destroy();
      this.bufAct = this.device.createBuffer({
        size: wantAct,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      this.actCapBytes = wantAct;
      rebinding = true;
    }

    const wantScratch = Math.max(4, this.heads * sp * this.window * 4);
    if (!this.bufScratch || (this.scratchCapBytes || 0) < wantScratch) {
      this.bufScratch?.destroy();
      this.bufScratch = this.device.createBuffer({
        size: wantScratch,
        usage: GPUBufferUsage.STORAGE,
      });
      this.scratchCapBytes = wantScratch;
      rebinding = true;
    }

    /* Nine dispatches per layer - two LayerNorms, Q/K/V, attention, O, and the
     * two FFN matmuls - plus the three tail steps, plus the stem: four per
     * chunk group and two to project and add position embeddings. */
    const maxGroups = ceilDiv(ceilDiv(tokens, this.tokensPerChunk), G) + 2;
    this.slots = this.layers * 9 + 3 + maxGroups * 4 + 2;
    const wantParams = this.slots * PARAM_STRIDE;
    if (!this.bufParams || (this.paramsCapBytes || 0) < wantParams) {
      this.bufParams?.destroy();
      this.bufParams = this.device.createBuffer({
        size: wantParams,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.paramsCapBytes = wantParams;
      this.host = new ArrayBuffer(wantParams);
      this.hostU = new Uint32Array(this.host);
      this.hostF = new Float32Array(this.host);
      rebinding = true;
    }

    const wantRead = this.outDim * sp * 4;
    if (!this.readBuf || (this.readCapBytes || 0) < wantRead) {
      this.readBuf?.destroy();
      this.readBuf = this.device.createBuffer({
        size: wantRead,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.readCapBytes = wantRead;
    }

    if (rebinding || !this.bindCache) {
      this.bindCache = new Map();
    }
  }

  bindGroup(shard = 0) {
    let bg = this.bindCache.get(shard);
    if (bg) return bg;
    bg = this.device.createBindGroup({
      layout: this._bgl,
      entries: [
        { binding: 0, resource: { buffer: this.bufParams, size: PARAM_STRIDE } },
        { binding: 1, resource: { buffer: this.bufQuants[shard] } },
        { binding: 2, resource: { buffer: this.bufScales } },
        { binding: 3, resource: { buffer: this.bufVecs } },
        { binding: 4, resource: { buffer: this.bufAct } },
        { binding: 5, resource: { buffer: this.bufScratch } },
      ],
    });
    this.bindCache.set(shard, bg);
    return bg;
  }

  /* Fill one uniform slot. Fields match the Params struct. */
  setSlot(i, f) {
    (this.slotShard || (this.slotShard = []))[i] = f.shard || 0;
    const u = (i * PARAM_STRIDE) / 4;
    this.hostU[u + 0] = f.wordBase || 0;
    this.hostU[u + 1] = f.scaleBase || 0;
    this.hostU[u + 2] = f.rows || 0;
    this.hostU[u + 3] = f.cols || 0;
    this.hostU[u + 4] = f.xOff || 0;
    this.hostU[u + 5] = f.yOff || 0;
    this.hostU[u + 6] = f.stride === undefined ? this.seqPad : f.stride;
    this.hostU[u + 7] = f.n === undefined ? this.tokens : f.n;
    this.hostU[u + 8] = f.vecA || 0;
    this.hostU[u + 9] = f.vecB || 0;
    this.hostU[u + 10] = f.flags || 0;
    this.hostU[u + 11] = f.p0 || 0;
    this.hostU[u + 12] = f.p1 || 0;
    this.hostF[u + 13] = f.fa || 0;
    this.hostF[u + 14] = f.fb || 0;
    this.hostF[u + 15] = f.fc || 0;
  }

  /* Build every uniform slot for one run, then upload them in one write.
   * The decoder learned this the hard way: filling the host array without
   * uploading it left prefill reading zeros. */
  buildParams(slot = 0, plan = []) {
    const d = this.dModel;

    const matmul = (mat, biasKind, layer, xOff, yOff, flags) => {
      const w = this.find(mat, layer);
      const b = this.find(biasKind, layer);
      this.setSlot(slot, {
        shard: w.shard,
        wordBase: w.wordBase, scaleBase: w.scaleBase,
        rows: w.rows, cols: w.cols, xOff, yOff,
        vecA: b.vecBase, flags,
      });
      plan.push({ pipe: this.pipeMatmul, slot,
                  x: ceilDiv(this.tokens, 64), y: ceilDiv(w.rows, 128) });
      slot++;
    };

    const layernorm = (wKind, bKind, layer, xOff, yOff) => {
      const w = this.find(wKind, layer);
      const b = this.find(bKind, layer);
      this.setSlot(slot, { rows: d, xOff, yOff, vecA: w.vecBase, vecB: b.vecBase, fa: 1e-5 });
      plan.push({ pipe: this.pipeLN, slot, x: ceilDiv(this.tokens, 64), y: 1 });
      slot++;
    };

    for (let l = 0; l < this.layers; l++) {
      layernorm(E_ATTN_NORM_W, E_ATTN_NORM_B, l, this.oX, this.oXN);
      matmul(E_Q, E_Q_B, l, this.oXN, this.oQ, 0);
      matmul(E_K, E_K_B, l, this.oXN, this.oK, 0);
      matmul(E_V, E_V_B, l, this.oXN, this.oV, 0);

      this.setSlot(slot, {
        rows: this.headDim, xOff: this.oQ, yOff: this.oK,
        vecA: this.oV, vecB: this.oATT,
        p0: this.window, p1: this.heads,
        fa: 1.0 / Math.sqrt(this.headDim),
      });
      if (this.pipeAttnSg && this.window <= ATTN_SG_MAXWIN) {
        plan.push({ pipe: this.pipeAttnSg, slot, x: this.tokens, y: this.heads });
      } else {
        plan.push({ pipe: this.pipeAttn, slot,
                    x: ceilDiv(this.tokens * this.heads, 64), y: 1 });
      }
      slot++;

      matmul(E_O, E_O_B, l, this.oATT, this.oX, F_RESIDUAL);
      layernorm(E_FFN_NORM_W, E_FFN_NORM_B, l, this.oX, this.oXN);
      matmul(E_FC1, E_FC1_B, l, this.oXN, this.oFF, F_GELU);
      matmul(E_FC2, E_FC2_B, l, this.oFF, this.oX, F_RESIDUAL);
    }

    layernorm(E_LN_POST_W, E_LN_POST_B, 0, this.oX, this.oXN);
    matmul(E_PROJ1, E_PROJ1_B, 0, this.oXN, this.oQ, F_GELU);
    matmul(E_PROJ2, E_PROJ2_B, 0, this.oQ, this.oOUT, 0);
    return { slot, plan };
  }

  /* The stem: conv groups, the projection onto d_model, and the position
   * embeddings. Leaves its result at oX, where the transformer starts. */
  buildStemParams(convPlan) {
    const plan = [];
    const ref = { v: 0 };
    for (const g of convPlan.groups) this.encodeConvGroup(plan, ref, g);

    const w = this.find(E_CONV_OUT);
    let slot = ref.v++;
    this.setSlot(slot, {
      shard: w.shard,
      wordBase: w.wordBase, scaleBase: w.scaleBase,
      rows: w.rows, cols: w.cols,
      xOff: this.oCX, yOff: this.oX,
      vecA: this.zeroVec, flags: 0,
    });
    plan.push({ pipe: this.pipeMatmul, slot,
                x: ceilDiv(this.tokens, 64), y: ceilDiv(w.rows, 128) });

    slot = ref.v++;
    this.setSlot(slot, { xOff: this.oPE, yOff: this.oX });
    plan.push({ pipe: this.pipeAdd, slot,
                x: ceilDiv(this.tokens, 64), y: this.dModel });
    return { slot: ref.v, plan };
  }

  /* mel in, [tokens][output_dim] out - the whole audio tower.
   *
   * Serialized: the streaming loop asks for this from a worker through a
   * main-thread hop, and two overlapping requests would share one readback
   * buffer and one set of activation regions. Queueing is what keeps a slow
   * chunk from corrupting the next one. */
  async runFromMel(melPtr, melFrames) {
    if (this.lost) throw new Error(`GPU device lost (${this.lost})`);
    const mine = this._queue = Promise.resolve(this._queue).then(
      () => this.runFromMelInner(melPtr, melFrames), () => this.runFromMelInner(melPtr, melFrames));
    return mine;
  }

  async runFromMelInner(melPtr, melFrames) {
    this.calls = (this.calls || 0) + 1;
    const cp = this.convPlan(melFrames);
    this.prepare(cp.tokens);
    this.uploadPE(cp);
    this.uploadMel(melPtr, melFrames, cp);

    const stem = this.buildStemParams(cp);
    const rest = this.buildParams(stem.slot, stem.plan.slice());
    this.device.queue.writeBuffer(this.bufParams, 0, this.host, 0, rest.slot * PARAM_STRIDE);
    this.plan = rest.plan;
    return this.submit();
  }


  /* ---- Conv2D stem ----
   *
   * Chunks are convolved in groups of equal width, like the C encoder does,
   * because a single 100-frame chunk gives the second layer a 480x4320x800
   * GEMM - too narrow to keep the GPU busy - and one dispatch per layer. The
   * trailing partial chunk, when there is one, runs on its own.
   */
  convPlan(melFrames) {
    const cw = this.chunkSize;
    const nChunks = Math.ceil(melFrames / cw);
    const groups = [];
    let tokenBase = 0;
    for (let c = 0; c < nChunks; ) {
      /* Only chunks of the same width can share a batch. */
      const full = (c + 1) * cw <= melFrames;
      let g = 1;
      if (full) {
        while (g < this.convGroup && c + g < nChunks && (c + g + 1) * cw <= melFrames) g++;
      }
      const width = full ? cw : melFrames - c * cw;
      const w1 = ((width - 1) >> 1) + 1, w2 = ((w1 - 1) >> 1) + 1, w3 = ((w2 - 1) >> 1) + 1;
      groups.push({ chunk: c, batch: g, width, w1, w2, w3, tokenBase, melStart: c * cw });
      tokenBase += g * w3;
      c += g;
    }
    return { groups, tokens: tokenBase };
  }

  /* Ensure sinusoidal position embedding table covers at least neededT positions.
   * Grows dynamically if any chunk group has w3 > maxPeT. */
  ensurePeTable(neededT) {
    if (this.peTable && this.maxPeT >= neededT) return;
    const d = this.dModel;
    if (!d) return;
    let newCap = Math.max(16, this.maxPeT || 16);
    while (newCap < neededT) newCap *= 2;
    const half = d >> 1;
    const logTs = Math.log(10000.0) / (half - 1);
    const newTable = new Float32Array(newCap * d);
    if (this.peTable) {
      newTable.set(this.peTable);
    }
    const startT = this.peTable ? (this.maxPeT || 0) : 0;
    for (let t = startT; t < newCap; t++) {
      const row = t * d;
      for (let i = 0; i < half; i++) {
        const angle = t * Math.exp(-i * logTs);
        newTable[row + i] = Math.sin(angle);
        newTable[row + half + i] = Math.cos(angle);
      }
    }
    this.peTable = newTable;
    this.maxPeT = newCap;
  }

  /* Sinusoidal position embeddings, restarting at 0 in every chunk, written
   * transposed to match the activation layout. */
  uploadPE(plan) {
    for (const g of plan.groups) {
      if (!this.peTable || g.w3 > (this.maxPeT || 0)) this.ensurePeTable(g.w3);
    }
    const d = this.dModel, sp = this.seqPad, half = d >> 1;
    if (!this.peBuf || this.peBuf.length < d * sp) {
      this.peBuf = new Float32Array(d * sp);
    }
    const pe = this.peBuf;
    pe.subarray(0, d * sp).fill(0);
    const peTable = this.peTable;
    for (const g of plan.groups) {
      for (let b = 0; b < g.batch; b++) {
        for (let t = 0; t < g.w3; t++) {
          const tok = g.tokenBase + b * g.w3 + t;
          const srcRow = t * d;
          for (let i = 0; i < half; i++) {
            pe[i * sp + tok] = peTable[srcRow + i];
            pe[(half + i) * sp + tok] = peTable[srcRow + half + i];
          }
        }
      }
    }
    this.device.queue.writeBuffer(this.bufAct, this.oPE * 4, pe.buffer, 0, d * sp * 4);
  }

  /* mel arrives as [128][melFrames]; the stem wants every chunk as its own
   * image, so it is rewritten to a run of [128][width] blocks. All groups go
   * up in one write - they share the conv scratch but not their input, since
   * the whole stem is queued as a single command buffer. */
  uploadMel(melPtr, melFrames, plan) {
    const src = new Float32Array(freshHeap(this.M).HEAPU8.buffer, melPtr >>> 0, 128 * melFrames);
    let total = 0;
    for (const g of plan.groups) { g.melOff = total; total += g.batch * 128 * g.width; }
    if (!this.melDstBuf || this.melDstBuf.length < total) {
      this.melDstBuf = new Float32Array(total);
    }
    const dst = this.melDstBuf;
    for (const g of plan.groups) {
      for (let b = 0; b < g.batch; b++) {
        const start = g.melStart + b * g.width;
        const bOff = g.melOff + b * 128 * g.width;
        for (let m = 0; m < 128; m++) {
          const sRow = m * melFrames + start;
          const dRow = bOff + m * g.width;
          dst.set(src.subarray(sRow, sRow + g.width), dRow);
        }
      }
    }
    this.device.queue.writeBuffer(this.bufAct, this.oMEL * 4, dst.buffer, 0, total * 4);
  }

  /* Queue one group's three convolutions and its reshape. */
  encodeConvGroup(plan, slotRef, g) {
    const CH = this.convHidden;
    const stage = (wKind, bKind, xOff, yOff, inStride, hIn, wIn, cIn, nOut) => {
      const w = this.find(wKind), b = this.find(bKind);
      const slot = slotRef.v++;
      this.setSlot(slot, {
        wordBase: w.vecBase, scaleBase: inStride,
        rows: CH, cols: cIn * 9, xOff, yOff,
        vecA: b.vecBase, flags: F_GELU,
        p0: hIn, p1: wIn,
        stride: nOut, n: nOut,
      });
      plan.push({ pipe: this.pipeConv, slot, x: ceilDiv(nOut, 64), y: ceilDiv(CH, 128) });
    };

    const B = g.batch;
    const n1 = B * 64 * g.w1, n2 = B * 32 * g.w2, n3 = B * 16 * g.w3;
    stage(E_CONV1, E_CONV1_B, this.oMEL + g.melOff, this.oC1,
          128 * g.width, 128, g.width, 1, n1);
    stage(E_CONV2, E_CONV2_B, this.oC1, this.oC2, n1, 64, g.w1, CH, n2);
    stage(E_CONV3, E_CONV3_B, this.oC2, this.oC3, n2, 32, g.w2, CH, n3);

    const slot = slotRef.v++;
    this.setSlot(slot, {
      rows: 16, p0: g.w3, p1: g.tokenBase,
      scaleBase: n3, xOff: this.oC3, yOff: this.oCX,
      n: B * g.w3,
    });
    plan.push({ pipe: this.pipeReshape, slot,
                x: ceilDiv(B * g.w3, 64), y: this.convProjDim });
  }

  /* Upload the conv stem's output, transposed into [d_model][seqPad]. */
  uploadInput(ptr, tokens) {
    const d = this.dModel, sp = this.seqPad;
    const src = new Float32Array(freshHeap(this.M).HEAPU8.buffer, ptr >>> 0, tokens * d);
    const dst = new Float32Array(d * sp);
    for (let t = 0; t < tokens; t++) {
      const base = t * d;
      for (let i = 0; i < d; i++) dst[i * sp + t] = src[base + i];
    }
    this.device.queue.writeBuffer(this.bufAct, this.oX * 4, dst.buffer, 0, dst.byteLength);
  }

  /* conv-stem output in, [tokens][output_dim] out - the transformer half only,
   * which is what the golden conv tap checks. */
  async run(convPtr, tokens) {
    if (this.lost) throw new Error(`GPU device lost (${this.lost})`);
    this.prepare(tokens);
    this.uploadInput(convPtr, tokens);
    const r = this.buildParams();
    this.device.queue.writeBuffer(this.bufParams, 0, this.host, 0, r.slot * PARAM_STRIDE);
    this.plan = r.plan;
    return this.submit();
  }

  /* Per-kernel GPU time over the recorded plan; same timestamp mechanism and
   * caveats as webgpu-decoder.js profileStep(). Call after a run so this.plan
   * holds a full tower. */
  async profileRun() {
    if (!this.hasTimestamps) throw new Error("timestamp-query not available");
    const { device } = this;
    const MAXQ = Math.min(4096, (this.plan.length + 1) * 2);
    const qs = device.createQuerySet({ type: "timestamp", count: MAXQ });
    const qbuf = device.createBuffer({
      size: MAXQ * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const rbuf = device.createBuffer({
      size: MAXQ * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    const spans = [];
    let qi = 0;
    for (const step of this.plan) {
      if (qi + 2 > MAXQ) throw new Error("profile: query set too small");
      const pass = enc.beginComputePass({ timestampWrites: {
        querySet: qs, beginningOfPassWriteIndex: qi, endOfPassWriteIndex: qi + 1 } });
      pass.setPipeline(step.pipe);
      pass.setBindGroup(0, this.bindGroup(this.slotShard[step.slot] | 0),
                        [step.slot * PARAM_STRIDE]);
      pass.dispatchWorkgroups(step.x, step.y);
      pass.end();
      spans.push([step.pipe.label || "?", qi]);
      qi += 2;
    }
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
    return { tokens: this.tokens, totalMs: total,
             rows: rows.map(([label, ms]) =>
               ({ label, ms: +ms.toFixed(2), pct: +(100 * ms / total).toFixed(1) })) };
  }

  async submit() {
    const t0 = performance.now();
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (const step of this.plan) {
      pass.setPipeline(step.pipe);
      pass.setBindGroup(0, this.bindGroup(this.slotShard[step.slot] | 0),
                        [step.slot * PARAM_STRIDE]);
      pass.dispatchWorkgroups(step.x, step.y);
    }
    pass.end();
    enc.copyBufferToBuffer(this.bufAct, this.oOUT * 4, this.readBuf, 0,
                           this.outDim * this.seqPad * 4);
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    this.runMs = performance.now() - t0;

    /* Unmap even if the copy out throws: a buffer left mapped makes every
     * later mapAsync fail with "already has an outstanding map pending", so
     * one bad call would poison the encoder for the rest of the session. */
    let flat;
    await this.readBuf.mapAsync(GPUMapMode.READ);
    try {
      flat = new Float32Array(this.readBuf.getMappedRange().slice(0));
    } finally {
      this.readBuf.unmap();
    }

    /* Back to [tokens][output_dim] for comparison with the C encoder. */
    const tokens = this.tokens;
    const out = new Float32Array(tokens * this.outDim);
    for (let t = 0; t < tokens; t++)
      for (let i = 0; i < this.outDim; i++)
        out[t * this.outDim + i] = flat[i * this.seqPad + t];
    return out;
  }
}
