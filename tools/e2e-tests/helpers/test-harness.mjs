/**
 * tools/e2e-tests/helpers/test-harness.mjs
 * Authoritative test utilities, specifications, and oracles for Qwen-ASR Mobile Safari E2E Tests.
 */

import { readFileSync, openSync, readSync, closeSync } from "node:fs";
import { join } from "node:path";

// 1. Text Normalization and Character Error Rate (CER)
export const DROP_RE = /^thinker\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate_up|down_proj))\.weight\.q8s?$|^thinker\.model\.embed_tokens\.weight\.q8s?$|^thinker\.audio_tower\./;

export function normalizeText(t) {
  if (typeof t !== "string") return "";
  return t
    .toLowerCase()
    .replace(/[.,!?;:'"()\[\]{}‘’“”]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function editDistance(a, b) {
  const prev = new Array(b.length + 1);
  const cur = new Array(b.length + 1);
  for (let j = 0; j <= b.length; j++) prev[j] = j;
  for (let i = 1; i <= a.length; i++) {
    cur[0] = i;
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(
        prev[j] + 1,
        cur[j - 1] + 1,
        prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)
      );
    }
    for (let j = 0; j <= b.length; j++) prev[j] = cur[j];
  }
  return prev[b.length];
}

export function calculateCer(got, ref) {
  const a = normalizeText(got);
  const b = normalizeText(ref);
  if (!b.length) return { err: a.length, len: 0, cer: a.length > 0 ? 1.0 : 0.0 };
  const err = editDistance(a, b);
  return { err, len: b.length, cer: err / b.length };
}

// 2. Safetensors Parser & Reduced Image Calculator
export function parseSafetensorsHeaderFromBuffer(buf) {
  if (buf.byteLength < 8) throw new Error("Buffer too small for safetensors header length");
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const hlenBig = view.getBigUint64(0, true);
  if (hlenBig > BigInt(buf.byteLength - 8)) {
    throw new Error(`Invalid header length ${hlenBig} exceeds buffer size`);
  }
  const hlen = Number(hlenBig);
  const str = new TextDecoder().decode(new Uint8Array(buf.buffer, buf.byteOffset + 8, hlen));
  const header = JSON.parse(str);
  return { hlen, header, dataBase: 8 + hlen };
}

export function readSafetensorsHeaderFromFile(filePath) {
  const fd = openSync(filePath, "r");
  try {
    const lenBuf = Buffer.alloc(8);
    readSync(fd, lenBuf, 0, 8, 0);
    const hlen = Number(lenBuf.readBigUInt64LE(0));
    const hBuf = Buffer.alloc(hlen);
    readSync(fd, hBuf, 0, hlen, 8);
    const header = JSON.parse(hBuf.toString("utf8"));
    return { hlen, header, dataBase: 8 + hlen };
  } finally {
    closeSync(fd);
  }
}

export function buildReducedHeader(header) {
  const kept = Object.entries(header)
    .filter(([name]) => name !== "__metadata__" && !DROP_RE.test(name))
    .sort((a, b) => a[1].data_offsets[0] - b[1].data_offsets[0]);

  const newHeader = {};
  let dataOff = 0;
  for (const [name, t] of kept) {
    const size = t.data_offsets[1] - t.data_offsets[0];
    dataOff = Math.ceil(dataOff / 64) * 64;
    newHeader[name] = { dtype: t.dtype, shape: t.shape, data_offsets: [dataOff, dataOff + size] };
    dataOff += size;
  }
  const hjson = new TextEncoder().encode(JSON.stringify(newHeader));
  const hpad = Math.ceil(hjson.length / 64) * 64;
  const totalReducedSize = 8 + hpad + dataOff;
  return { kept, newHeader, hpad, dataOff, totalReducedSize };
}

// 3. Dynamic WebGPU Sharding Simulation
export function calculateDynamicShards(entries, shardBudget) {
  const shards = [{ bytes: 0, items: [] }];
  const wmap = new Map();

  const openShard = (need) => {
    let sh = shards[shards.length - 1];
    if (sh.bytes > 0 && sh.bytes + need > shardBudget) {
      shards.push({ bytes: 0, items: [] });
      sh = shards[shards.length - 1];
    }
    return sh;
  };

  for (const ent of entries) {
    const { key, rows, cols } = ent;
    const nq = rows * cols;
    if (nq > shardBudget) {
      const rowsPer = Math.floor(shardBudget / cols);
      if (rowsPer <= 0) {
        throw new Error(`Matrix column width ${cols} exceeds entire shard budget ${shardBudget}`);
      }
      const pieces = [];
      for (let r0 = 0; r0 < rows; r0 += rowsPer) {
        const n = Math.min(rowsPer, rows - r0);
        const sh = openShard(n * cols);
        pieces.push({
          shard: shards.length - 1,
          wordBase: sh.bytes / 4,
          rowBase: r0,
          rowCount: n,
          nq: n * cols,
        });
        sh.bytes += n * cols;
        sh.items.push(`${key}:slice[${r0}:${r0 + n}]`);
      }
      wmap.set(key, { rows, cols, nq, pieces });
    } else {
      const sh = openShard(nq);
      wmap.set(key, {
        rows, cols, nq,
        shard: shards.length - 1,
        wordBase: sh.bytes / 4,
      });
      sh.bytes += nq;
      sh.items.push(key);
    }
  }

  const biggestShard = Math.max(...shards.map((s) => s.bytes));
  return { shards, biggestShard, wmap };
}

// 4. AudioWorklet Resampler Mathematical Reference
export function downsample48kTo16k(input, state = null) {
  // Exact 3:1 decimation with 5-tap anti-aliasing filter
  const outLen = Math.floor(input.length / 3);
  const out = new Float32Array(outLen);
  // Symmetric FIR low-pass filter weights: fc = 0.33, normalized sum = 1.0
  const b = [0.0625, 0.25, 0.375, 0.25, 0.0625];
  const hist = state?.history || new Float32Array(4);

  for (let i = 0; i < outLen; i++) {
    const srcIdx = i * 3;
    let acc = 0;
    for (let j = 0; j < 5; j++) {
      const idx = srcIdx + j - 2;
      let sample = 0;
      if (idx < 0) {
        sample = hist[hist.length + idx] ?? 0;
      } else if (idx < input.length) {
        sample = input[idx];
      }
      acc += sample * b[j];
    }
    out[i] = acc;
  }

  if (state) {
    state.history = new Float32Array([
      input[input.length - 4] ?? 0,
      input[input.length - 3] ?? 0,
      input[input.length - 2] ?? 0,
      input[input.length - 1] ?? 0
    ]);
  }
  return out;
}

export function downsample44kTo16k(input) {
  // Ratio 44100 / 16000 = 2.75625
  const ratio = 44100 / 16000;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const frac = pos - i0;
    out[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return out;
}

// 5. Memory Headroom Tracker (Assertion ceiling: 1.8 GB = 1,800,000,000 bytes)
export const MEMORY_CEILING_BYTES = 1.8 * 1e9; // 1.8 GB Mobile Safari tab ceiling

export class MemoryTracker {
  constructor(ceiling = MEMORY_CEILING_BYTES) {
    this.ceiling = ceiling;
    this.activeGpuBytes = 0;
    this.wasmHeapBytes = 0;
    this.jsTransientBytes = 0;
    this.peakTotalBytes = 0;
    this.history = [];
  }

  recordGpuAllocation(bytes) {
    this.activeGpuBytes += bytes;
    this._checkPeak("gpu_alloc", bytes);
  }

  recordGpuRelease(bytes) {
    this.activeGpuBytes = Math.max(0, this.activeGpuBytes - bytes);
    this._checkPeak("gpu_free", -bytes);
  }

  setWasmHeap(bytes) {
    this.wasmHeapBytes = bytes;
    this._checkPeak("wasm_heap", bytes);
  }

  setJsTransient(bytes) {
    this.jsTransientBytes = bytes;
    this._checkPeak("js_transient", bytes);
  }

  getTotalBytes() {
    return this.activeGpuBytes + this.wasmHeapBytes + this.jsTransientBytes;
  }

  _checkPeak(action, delta) {
    const total = this.getTotalBytes();
    if (total > this.peakTotalBytes) {
      this.peakTotalBytes = total;
    }
    this.history.push({
      action,
      delta,
      total,
      gpu: this.activeGpuBytes,
      wasm: this.wasmHeapBytes,
      js: this.jsTransientBytes,
      timestamp: Date.now()
    });
    if (total > this.ceiling) {
      throw new Error(`[OOM_VIOLATION] Total memory ${(total / 1e9).toFixed(3)} GB exceeded Mobile Safari tab ceiling ${(this.ceiling / 1e9).toFixed(2)} GB`);
    }
  }

  assertWithinCeiling() {
    if (this.peakTotalBytes > this.ceiling) {
      throw new Error(`Peak memory ${(this.peakTotalBytes / 1e9).toFixed(3)} GB exceeded 1.8 GB ceiling`);
    }
    return true;
  }
}

// 6. Mock WebGPU Infrastructure for Lifecycle and Contract Verification
export class MockGPUBuffer {
  constructor(size, usage, label = "") {
    this.size = size;
    this.usage = usage;
    this.label = label;
    this.destroyed = false;
  }

  destroy() {
    this.destroyed = true;
  }
}

export class MockWebGPUDevice {
  constructor(limits = {}) {
    this.limits = {
      maxBufferSize: limits.maxBufferSize ?? (1 << 30),
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize ?? (256 << 20),
      ...limits
    };
    this.buffers = new Set();
    this.queue = {
      writeBuffer: (buf, off, src) => {
        if (buf.destroyed) throw new Error("Cannot write to destroyed buffer");
      },
      onSubmittedWorkDone: async () => Promise.resolve()
    };
  }

  createBuffer(desc) {
    if (desc.size > this.limits.maxBufferSize) {
      throw new Error(`Buffer size ${desc.size} exceeds maxBufferSize ${this.limits.maxBufferSize}`);
    }
    const buf = new MockGPUBuffer(desc.size, desc.usage, desc.label);
    this.buffers.add(buf);
    return buf;
  }

  destroyBuffer(buf) {
    buf.destroy();
    this.buffers.delete(buf);
  }
}

// 7. HTML / DOM UX Contract Checker for Mobile Safari
export function verifyMobileSafariHtmlContracts(htmlContent) {
  const issues = [];
  // 1. Viewport meta tag with viewport-fit=cover
  const viewportRegex = /<meta\s+name=["']viewport["']\s+content=["']([^"']+)["']/i;
  const match = htmlContent.match(viewportRegex);
  if (!match) {
    issues.push("Missing <meta name='viewport'> tag");
  } else {
    const content = match[1];
    if (!content.includes("width=device-width")) issues.push("Viewport missing width=device-width");
    if (!content.includes("viewport-fit=cover")) issues.push("Viewport missing viewport-fit=cover");
  }

  // 2. Safe-area insets in CSS
  if (!htmlContent.includes("safe-area-inset-top") && !htmlContent.includes("safe-area-inset-bottom")) {
    issues.push("Missing env(safe-area-inset-*) declarations in styling");
  }

  // 3. Touch target sizes: check if buttons or interactive elements have min-height >= 44px
  if (htmlContent.includes("padding: 9px 16px") && !htmlContent.includes("min-height: 44px")) {
    issues.push("Touch targets (buttons) may be smaller than 44pt Apple HIG minimum");
  }

  // 4. Input font sizes: check if font-size is at least 16px to prevent iOS auto-zoom
  const inputFontMatches = htmlContent.match(/input[^{]*\{[^}]*font-size:\s*(\d+)px/i);
  if (inputFontMatches && parseInt(inputFontMatches[1], 10) < 16) {
    issues.push(`Input font size is ${inputFontMatches[1]}px (< 16px), will cause iOS Safari auto-zoom`);
  }

  return { ok: issues.length === 0, issues };
}
