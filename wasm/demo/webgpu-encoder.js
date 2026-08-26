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
        v = vec4<f32>(i8x4(word)) * scales[P.scaleBase + row * nblocks + col0 / 64u];
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

export { HEADER, LN_WGSL, MATMUL_WGSL, ATTN_WGSL,
         E_CONV1, E_CONV1_B, E_CONV2, E_CONV2_B, E_CONV3, E_CONV3_B, E_CONV_OUT,
         E_ATTN_NORM_W, E_ATTN_NORM_B, E_Q, E_Q_B, E_K, E_K_B, E_V, E_V_B,
         E_O, E_O_B, E_FFN_NORM_W, E_FFN_NORM_B, E_FC1, E_FC1_B, E_FC2, E_FC2_B,
         E_LN_POST_W, E_LN_POST_B, E_PROJ1, E_PROJ1_B, E_PROJ2, E_PROJ2_B,
         PARAM_STRIDE, PARAM_FIELDS, F_GELU, F_RESIDUAL };

/* ------------------------------------------------------------------ driver */

const ceilDiv = (a, b) => Math.floor((a + b - 1) / b);

export class WebGPUEncoder {
  constructor(Module) {
    this.M = Module;
    this.ready = false;
  }

  async init(report = () => {}) {
    const M = this.M;

    const shPtr = M._qwen_wasm_alloc(16 * 4) >>> 0;
    if (M._qwen_wasm_enc_shape(shPtr) < 0) throw new Error("encoder shape unavailable");
    const sh = new Int32Array(M.HEAPU8.buffer, shPtr, 16).slice();
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

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const lim = adapter.limits;
    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: lim.maxBufferSize,
        maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      },
    });
    this.device.onuncapturederror = (e) => {
      console.error("gpu error:", e.error.message);
      report("gpu error: " + e.error.message);
    };

    await this.loadWeights(report);
    await this.buildPipelines();
    this.ready = true;
  }

  /* Read the descriptor table and pack every weight into three buffers:
   * int8 quants, their f32 scales, and the plain f32 vectors. */
  async loadWeights(report) {
    const M = this.M;
    const MAXD = 1024;
    const dPtr = M._qwen_wasm_alloc(MAXD * 8 * 4) >>> 0;
    const n = M._qwen_wasm_enc_desc(dPtr, MAXD);
    if (n < 0) { M._qwen_wasm_release(dPtr); throw new Error("encoder weights unavailable"); }
    const d = new Uint32Array(M.HEAPU8.buffer, dPtr, n * 8).slice();
    M._qwen_wasm_release(dPtr);

    let qBytes = 0, sFloats = 0, vFloats = 0;
    const entries = [];
    for (let i = 0; i < n; i++) {
      const e = d.subarray(i * 8, i * 8 + 8);
      const rec = { kind: e[0], layer: e[1], rows: e[2], cols: e[3],
                    qPtr: e[4], sPtr: e[5], fPtr: e[6], count: e[7] };
      if (rec.qPtr) {
        if (rec.cols % 64 !== 0) throw new Error(`kind ${rec.kind}: cols ${rec.cols} not a multiple of 64`);
        rec.wordBase = qBytes / 4;
        rec.scaleBase = sFloats;
        qBytes += rec.rows * rec.cols;
        sFloats += rec.rows * (rec.cols / 64);
      } else if (rec.fPtr && rec.count) {
        rec.vecBase = vFloats;
        vFloats += rec.count;
      } else {
        throw new Error(`kind ${rec.kind}: f32 matrices are not supported on the GPU path`);
      }
      entries.push(rec);
    }
    this.entries = entries;

    report(`allocating ${((qBytes + sFloats * 4 + vFloats * 4) / 1e9).toFixed(2)} GB on the GPU...`);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.bufQuants = this.device.createBuffer({ size: Math.max(4, qBytes), usage });
    this.bufScales = this.device.createBuffer({ size: Math.max(4, sFloats * 4), usage });
    this.bufVecs = this.device.createBuffer({ size: Math.max(4, vFloats * 4), usage });
    this.weightBytes = qBytes + sFloats * 4 + vFloats * 4;

    for (const rec of entries) {
      if (rec.qPtr) {
        const qn = rec.rows * rec.cols;
        this.device.queue.writeBuffer(this.bufQuants, rec.wordBase * 4,
          M.HEAPU8.buffer, rec.qPtr, qn);
        const sn = rec.rows * (rec.cols / 64);
        this.device.queue.writeBuffer(this.bufScales, rec.scaleBase * 4,
          M.HEAPU8.buffer, rec.sPtr, sn * 4);
      } else {
        this.device.queue.writeBuffer(this.bufVecs, rec.vecBase * 4,
          M.HEAPU8.buffer, rec.fPtr, rec.count * 4);
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
      return this.device.createComputePipeline({
        layout, compute: { module: mod, entryPoint: "main" },
      });
    };
    this.pipeLN = await mk(LN_WGSL, "layernorm");
    this.pipeMatmul = await mk(MATMUL_WGSL, "matmul");
    this.pipeAttn = await mk(ATTN_WGSL, "attention");
  }

  /* Activation regions, all [rows][seqPad] so the GEMM's lanes stay on the
   * sequence axis. Offsets are in floats. */
  prepare(tokens) {
    const sp = ceilDiv(tokens, 64) * 64;
    if (this.seqPad === sp) { this.tokens = tokens; return; }
    this.seqPad = sp;
    this.tokens = tokens;

    const d = this.dModel;
    let off = 0;
    const region = (rows) => { const o = off; off += rows * sp; return o; };
    this.oX = region(d);
    this.oXN = region(d);
    this.oQ = region(d);
    this.oK = region(d);
    this.oV = region(d);
    this.oATT = region(d);
    this.oFF = region(this.ffnDim);
    this.oOUT = region(this.outDim);
    this.actFloats = off;

    this.bufAct?.destroy();
    this.bufAct = this.device.createBuffer({
      size: this.actFloats * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.bufScratch?.destroy();
    this.bufScratch = this.device.createBuffer({
      size: Math.max(4, this.heads * sp * this.window * 4),
      usage: GPUBufferUsage.STORAGE,
    });

    /* Nine dispatches per layer - two LayerNorms, Q/K/V, attention, O, and
     * the two FFN matmuls - plus the three tail steps. */
    this.slots = this.layers * 9 + 3;
    this.bufParams?.destroy();
    this.bufParams = this.device.createBuffer({
      size: this.slots * PARAM_STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.host = new ArrayBuffer(this.slots * PARAM_STRIDE);
    this.hostU = new Uint32Array(this.host);
    this.hostF = new Float32Array(this.host);

    this.readBuf?.destroy();
    this.readBuf = this.device.createBuffer({
      size: this.outDim * sp * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    this.bindCache = new Map();
  }

  bindGroup(pipe) {
    let bg = this.bindCache.get("shared");
    if (bg) return bg;
    bg = this.device.createBindGroup({
      layout: this._bgl,
      entries: [
        { binding: 0, resource: { buffer: this.bufParams, size: PARAM_STRIDE } },
        { binding: 1, resource: { buffer: this.bufQuants } },
        { binding: 2, resource: { buffer: this.bufScales } },
        { binding: 3, resource: { buffer: this.bufVecs } },
        { binding: 4, resource: { buffer: this.bufAct } },
        { binding: 5, resource: { buffer: this.bufScratch } },
      ],
    });
    this.bindCache.set("shared", bg);
    return bg;
  }

  /* Fill one uniform slot. Fields match the Params struct. */
  setSlot(i, f) {
    const u = (i * PARAM_STRIDE) / 4;
    this.hostU[u + 0] = f.wordBase || 0;
    this.hostU[u + 1] = f.scaleBase || 0;
    this.hostU[u + 2] = f.rows || 0;
    this.hostU[u + 3] = f.cols || 0;
    this.hostU[u + 4] = f.xOff || 0;
    this.hostU[u + 5] = f.yOff || 0;
    this.hostU[u + 6] = this.seqPad;
    this.hostU[u + 7] = this.tokens;
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
  buildParams() {
    const d = this.dModel;
    let slot = 0;
    const plan = [];

    const matmul = (mat, biasKind, layer, xOff, yOff, flags) => {
      const w = this.find(mat, layer);
      const b = this.find(biasKind, layer);
      this.setSlot(slot, {
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
      plan.push({ pipe: this.pipeAttn, slot,
                  x: ceilDiv(this.tokens * this.heads, 64), y: 1 });
      slot++;

      matmul(E_O, E_O_B, l, this.oATT, this.oX, F_RESIDUAL);
      layernorm(E_FFN_NORM_W, E_FFN_NORM_B, l, this.oX, this.oXN);
      matmul(E_FC1, E_FC1_B, l, this.oXN, this.oFF, F_GELU);
      matmul(E_FC2, E_FC2_B, l, this.oFF, this.oX, F_RESIDUAL);
    }

    layernorm(E_LN_POST_W, E_LN_POST_B, 0, this.oX, this.oXN);
    matmul(E_PROJ1, E_PROJ1_B, 0, this.oXN, this.oQ, F_GELU);
    matmul(E_PROJ2, E_PROJ2_B, 0, this.oQ, this.oOUT, 0);

    this.device.queue.writeBuffer(this.bufParams, 0, this.host, 0, slot * PARAM_STRIDE);
    this.plan = plan;
  }

  /* Upload the conv stem's output, transposed into [d_model][seqPad]. */
  uploadInput(ptr, tokens) {
    const d = this.dModel, sp = this.seqPad;
    const src = new Float32Array(this.M.HEAPU8.buffer, ptr >>> 0, tokens * d);
    const dst = new Float32Array(d * sp);
    for (let t = 0; t < tokens; t++) {
      const base = t * d;
      for (let i = 0; i < d; i++) dst[i * sp + t] = src[base + i];
    }
    this.device.queue.writeBuffer(this.bufAct, this.oX * 4, dst.buffer, 0, dst.byteLength);
  }

  /* conv-stem output in, [tokens][output_dim] out. */
  async run(convPtr, tokens) {
    this.prepare(tokens);
    this.uploadInput(convPtr, tokens);
    this.buildParams();

    const t0 = performance.now();
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (const step of this.plan) {
      pass.setPipeline(step.pipe);
      pass.setBindGroup(0, this.bindGroup(step.pipe), [step.slot * PARAM_STRIDE]);
      pass.dispatchWorkgroups(step.x, step.y);
    }
    pass.end();
    enc.copyBufferToBuffer(this.bufAct, this.oOUT * 4, this.readBuf, 0,
                           this.outDim * this.seqPad * 4);
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    this.runMs = performance.now() - t0;

    await this.readBuf.mapAsync(GPUMapMode.READ);
    const flat = new Float32Array(this.readBuf.getMappedRange().slice(0));
    this.readBuf.unmap();

    /* Back to [tokens][output_dim] for comparison with the C encoder. */
    const out = new Float32Array(tokens * this.outDim);
    for (let t = 0; t < tokens; t++)
      for (let i = 0; i < this.outDim; i++)
        out[t * this.outDim + i] = flat[i * this.seqPad + t];
    return out;
  }
}
