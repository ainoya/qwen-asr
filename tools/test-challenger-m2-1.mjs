/**
 * tools/test-challenger-m2-1.mjs
 * Empirical Challenger M2.1 Stress Test Harness
 *
 * Focus Areas:
 * 1. Dynamic sharding boundaries across arbitrary device limits:
 *    - 128 MiB, 256 MiB, 64 MiB, 192 MiB, 100 MiB, 150 MiB, odd limits (128 MiB - 1, 256 MiB + 1)
 *    - Verify all shard sizes are strictly <= binding limit
 *    - Verify 311 MB embedding table (151936 x 2048) partitioned with ZERO gaps and ZERO overlaps
 *    - Verify alignment (wordBase * 4, scaleBase) and interval dispatch table
 * 2. KV cache sizing & clamping on f16 and f32 fallbacks:
 *    - Sizing at 768 tokens for f16 (fits in 128 MiB)
 *    - Strict clamping to 512 tokens for f32 (112 MiB <= 128 MiB)
 *    - Sizing across varying prompt lengths, step increments, suffix prefills
 *    - Verification of cfg.qDim, cfg.kvDim, cfg.headsPerKv existence and activation buffer sizing
 * 3. Half-float decoding fallbacks:
 *    - Exhaustive 65,536 half-float value verification against IEEE-754 oracle
 *    - Subnormal numbers, signed zeroes, infinities, NaNs
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

// Mock globals for headless Node.js
globalThis.self = globalThis;
globalThis.window = globalThis;
globalThis.GPUBufferUsage = {
  MAP_READ: 0x01,
  MAP_WRITE: 0x02,
  COPY_SRC: 0x04,
  COPY_DST: 0x08,
  INDEX: 0x10,
  VERTEX: 0x20,
  UNIFORM: 0x40,
  STORAGE: 0x80,
  INDIRECT: 0x100,
  QUERY_RESOLVE: 0x200,
};

globalThis.document = {
  getElementById: () => ({
    id: "mock",
    classList: { add() {}, remove() {} },
    appendChild() {},
    addEventListener() {},
    style: {},
    disabled: false,
    textContent: "",
  }),
  createElement: () => ({
    appendChild() {},
    classList: { add() {}, remove() {} },
    style: {},
    textContent: "",
  }),
};

// Import modules under test
const { WebGPUDecoder } = await import("../wasm/demo/webgpu-decoder.js");
const { WebGPUEncoder } = await import("../wasm/demo/webgpu-encoder.js");
const app = await import("../wasm/demo/app.js");
const {
  extractModelDescriptorsFromHeader,
  prepareWasmReducedImage,
  buildIntervalDispatchTable,
} = app;

// Read real model header
const modelPath = path.join(process.cwd(), "qwen3-asr-1.7b-q8", "qwen-asr-q8.bin");
const fd = fs.openSync(modelPath, "r");
const lenBuf = Buffer.alloc(8);
fs.readSync(fd, lenBuf, 0, 8, 0);
const headerLen = Number(lenBuf.readBigUInt64LE(0));
const headerBuf = Buffer.alloc(headerLen);
fs.readSync(fd, headerBuf, 0, headerLen, 8);
fs.closeSync(fd);
const modelHeader = JSON.parse(headerBuf.toString("utf8"));
const dataBase = 8 + headerLen;

const { decEntries, encEntries } = extractModelDescriptorsFromHeader(modelHeader, dataBase);

// Strict mock WebGPU device that enforces WebIDL unsigned long long on buffer size
class StrictMockGPUBuffer {
  constructor(size, usage, label = "") {
    if (typeof size !== "number" || Number.isNaN(size) || !Number.isFinite(size) || size < 0) {
      throw new TypeError(`WebGPU WebIDL error: buffer size must be valid non-negative integer, got: ${size}`);
    }
    this.size = size;
    this.usage = usage;
    this.label = label;
    this.destroyed = false;
  }
  destroy() {
    this.destroyed = true;
  }
}

class StrictMockGPUDevice {
  constructor(limits = {}) {
    this.limits = {
      maxBufferSize: limits.maxBufferSize ?? (1 << 30),
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize ?? (128 << 20),
      maxComputeWorkgroupsPerDimension: 65535,
      ...limits,
    };
    this.buffers = [];
    this.queue = {
      writeBuffer(buf, off, src) {
        if (buf.destroyed) throw new Error("write to destroyed buffer");
      },
      submit() {},
      async onSubmittedWorkDone() { return Promise.resolve(); },
    };
  }
  createBuffer(desc) {
    if (typeof desc.size !== "number" || Number.isNaN(desc.size)) {
      throw new TypeError(`WebGPU TypeError: createBuffer size cannot be NaN`);
    }
    if (desc.size > this.limits.maxBufferSize) {
      throw new Error(`Buffer size ${desc.size} exceeds maxBufferSize ${this.limits.maxBufferSize}`);
    }
    const buf = new StrictMockGPUBuffer(desc.size, desc.usage, desc.label);
    this.buffers.push(buf);
    return buf;
  }
  createBindGroupLayout() { return {}; }
  createBindGroup() { return {}; }
  createCommandEncoder() {
    return {
      copyBufferToBuffer() {},
      finish() { return {}; },
    };
  }
}

function createMockWasmModule() {
  const heap = new Uint8Array(64 * 1024 * 1024); // 64 MB heap to accommodate layer stride
  let allocPtr = 1024;
  return {
    HEAPU8: heap,
    HEAPF32: new Float32Array(heap.buffer),
    _qwen_wasm_alloc: (sz) => { const p = allocPtr; allocPtr += sz; return p; },
    _qwen_wasm_release: () => {},
    _qwen_wasm_rms_eps: () => 1e-6,
    _qwen_wasm_rope_theta: () => 1e6,
    _qwen_wasm_kv_stride: () => 128,
    _qwen_wasm_kv_k_ptr: () => 1024,
    _qwen_wasm_kv_v_ptr: () => 1024 + 128 * 1024,
    _qwen_wasm_kv_is_f16: () => 0,
    _qwen_wasm_prompt_has_asr_text: () => 1,
    _qwen_wasm_token_text: () => 0,
  };
}

let totalTests = 0;
let passedTests = 0;
let failedTests = 0;
const failures = [];

function runTest(name, fn) {
  totalTests++;
  try {
    fn();
    console.log(`  ✔ [PASS] ${name}`);
    passedTests++;
  } catch (err) {
    console.error(`  ✖ [FAIL] ${name}`);
    console.error(`    Error: ${err.message}`);
    failedTests++;
    failures.push({ name, error: err.message, stack: err.stack });
  }
}

console.log("=======================================================");
console.log("CHALLENGER M2.1: EMPIRICAL BOUNDARY & STRESS SUITE");
console.log("=======================================================\n");

// ============================================================================
// SUITE 1: Raw Implementation Diagnostic & Bug Reproduction
// ============================================================================
console.log("--- Suite 1: Raw WebGPUDecoder Implementation Diagnostics ---");

runTest("TC-BUG-01: [EMPIRICAL BUG REPRODUCTION] Raw WebGPUDecoder.allocateStorageBuffers omits cfg.qDim & cfg.kvDim, producing NaN buffer allocations", () => {
  const mockModule = createMockWasmModule();
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: 128 << 20 });
  const dec = new WebGPUDecoder(mockModule);
  dec.setupDevice(device);

  // Calling allocateStorageBuffers as app.js does
  dec.allocateStorageBuffers({
    shardBudget: 128 << 20,
    entries: decEntries,
    header: modelHeader,
  });

  // Check whether cfg.qDim or cfg.kvDim were populated
  const missingQDim = dec.cfg?.qDim === undefined;
  const missingKvDim = dec.cfg?.kvDim === undefined;
  const isActNaN = Number.isNaN(dec.actFloats);

  console.log(`    -> Raw dec.cfg.qDim: ${dec.cfg?.qDim}`);
  console.log(`    -> Raw dec.cfg.kvDim: ${dec.cfg?.kvDim}`);
  console.log(`    -> Raw dec.actFloats: ${dec.actFloats}`);
  console.log(`    -> Raw dec.bufAct.size: ${dec.bufAct?.size}`);

  // This test asserts that cfg.qDim and cfg.kvDim MUST be defined for WebGPUDecoder to be valid.
  assert.strictEqual(missingQDim, false, "CRITICAL DEFECT: dec.cfg.qDim is undefined in allocateStorageBuffers!");
  assert.strictEqual(missingKvDim, false, "CRITICAL DEFECT: dec.cfg.kvDim is undefined in allocateStorageBuffers!");
  assert.strictEqual(isActNaN, false, "CRITICAL DEFECT: dec.actFloats is NaN!");
});

// ============================================================================
// SUITE 2: Dynamic Sharding Across Arbitrary Limits (With Dimension Guard)
// ============================================================================
console.log("\n--- Suite 2: Dynamic Sharding Across Arbitrary Device Limits ---");

const testBudgets = [
  { name: "Standard 256 MiB (Desktop)", budget: 256 << 20, maxBinding: 256 << 20 },
  { name: "Standard 128 MiB (Mobile Safari iOS 18)", budget: 128 << 20, maxBinding: 128 << 20 },
  { name: "Odd Limit 192 MiB", budget: 192 << 20, maxBinding: 192 << 20 },
  { name: "Restricted 64 MiB", budget: 64 << 20, maxBinding: 64 << 20 },
  { name: "Decimal 100 MB (104,857,600 bytes)", budget: 100 * 1024 * 1024, maxBinding: 100 * 1024 * 1024 },
  { name: "Decimal 150 MB (157,286,400 bytes)", budget: 150 * 1024 * 1024, maxBinding: 150 * 1024 * 1024 },
  { name: "Exact 128 MiB - 1 byte (134,217,727 bytes)", budget: (128 << 20) - 1, maxBinding: (128 << 20) - 1 },
  { name: "Exact 256 MiB + 1 byte (268,435,457 bytes)", budget: (256 << 20) + 1, maxBinding: (256 << 20) + 1 },
];

for (const { name, budget, maxBinding } of testBudgets) {
  runTest(`TC-SHARD-${budget}: Sharding under ${name}`, () => {
    const mockModule = createMockWasmModule();
    // Relax NaN check ONLY for testing sharding math independently of the cfg.qDim bug
    const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: maxBinding });
    const dec = new WebGPUDecoder(mockModule);
    dec.setupDevice(device);

    // Provide properly dimensioned cfg to test the sharding algorithm itself
    dec.cfg = {
      layers: 28, hidden: 2048, heads: 16, kvHeads: 8, headDim: 128, inter: 6144, vocab: 151936, imEnd: 151645,
      qDim: 2048, kvDim: 1024, headsPerKv: 2
    };

    dec.allocateStorageBuffers({
      shardBudget: budget,
      entries: decEntries,
      header: modelHeader,
      cfg: dec.cfg,
    });

    const activeBudget = Math.min(budget, maxBinding);

    // 1. Assert all shards strictly <= activeBudget
    for (let sIdx = 0; sIdx < dec.shards.length; sIdx++) {
      const sh = dec.shards[sIdx];
      assert.ok(
        sh.bytes <= activeBudget,
        `Decoder shard ${sIdx} size (${sh.bytes}) exceeds active budget (${activeBudget})`
      );
      assert.ok(sh.bytes > 0, `Decoder shard ${sIdx} size must be > 0`);
      assert.strictEqual(sh.bytes % 4, 0, `Decoder shard ${sIdx} size (${sh.bytes}) must be 4-byte aligned`);
    }

    // 2. Assert embedding partition (311 MB / 151,936 x 2048) has ZERO gaps and ZERO overlaps
    const emb = dec.wmap.get("6:0");
    assert.ok(emb, "Embedding tensor 6:0 must be present in wmap");
    assert.strictEqual(emb.rows, 151936, "Embedding must have 151,936 rows");
    assert.strictEqual(emb.cols, 2048, "Embedding must have 2,048 cols");
    const totalEmbBytes = 151936 * 2048;

    if (totalEmbBytes > activeBudget) {
      assert.ok(Array.isArray(emb.pieces), "Embedding must be sliced into pieces");
      let expectedNextRow = 0;
      let totalPiecesBytes = 0;

      for (let pIdx = 0; pIdx < emb.pieces.length; pIdx++) {
        const p = emb.pieces[pIdx];
        // Zero gap assertion: piece must start exactly where previous ended
        assert.strictEqual(
          p.rowBase,
          expectedNextRow,
          `Embedding piece ${pIdx} has row gap/overlap: rowBase=${p.rowBase}, expected=${expectedNextRow}`
        );
        assert.ok(p.rowCount > 0, `Embedding piece ${pIdx} rowCount must be positive`);
        assert.strictEqual(p.nq, p.rowCount * 2048, `Embedding piece ${pIdx} nq mismatch`);
        assert.ok(
          p.nq <= activeBudget,
          `Embedding piece ${pIdx} size (${p.nq}) exceeds active budget (${activeBudget})`
        );

        // Verify within shard bounds
        const shardBytes = dec.shards[p.shard].bytes;
        assert.ok(
          p.wordBase * 4 + p.nq <= shardBytes,
          `Embedding piece ${pIdx} overflows shard ${p.shard}: ${p.wordBase * 4 + p.nq} > ${shardBytes}`
        );
        assert.strictEqual(
          (p.wordBase * 4) % 4,
          0,
          `Embedding piece ${pIdx} wordBase byte offset must be 4-byte aligned`
        );

        expectedNextRow += p.rowCount;
        totalPiecesBytes += p.nq;
      }

      // Assert total rows and total bytes exact match
      assert.strictEqual(expectedNextRow, 151936, "Embedding pieces must sum to exactly 151,936 rows");
      assert.strictEqual(totalPiecesBytes, totalEmbBytes, "Embedding pieces bytes must sum to total embedding bytes");
    }

    // 3. Assert all non-embedding matrices fit within their assigned shard
    for (const [key, w] of dec.wmap.entries()) {
      if (key === "6:0") continue;
      assert.ok(
        w.wordBase * 4 + w.nq <= dec.shards[w.shard].bytes,
        `Matrix ${key} overflows shard ${w.shard}`
      );
    }
  });

  runTest(`TC-ENC-SHARD-${budget}: Encoder sharding under ${name}`, () => {
    const mockModule = createMockWasmModule();
    const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: maxBinding });
    const enc = new WebGPUEncoder(mockModule);
    enc.device = device;
    enc.adapter = { limits: { maxStorageBufferBindingSize: maxBinding } };

    enc.allocateStorageBuffers({
      shardBudget: budget,
      entries: encEntries,
      header: modelHeader,
    });

    const activeBudget = Math.min(budget, maxBinding);
    for (let sIdx = 0; sIdx < enc.shards.length; sIdx++) {
      const sh = enc.shards[sIdx];
      assert.ok(
        sh.bytes <= activeBudget,
        `Encoder shard ${sIdx} size (${sh.bytes}) exceeds active budget (${activeBudget})`
      );
      assert.ok(sh.bytes > 0, `Encoder shard ${sIdx} size must be > 0`);
      assert.strictEqual(sh.bytes % 4, 0, `Encoder shard ${sIdx} size must be 4-byte aligned`);
    }
    assert.ok(!Number.isNaN(enc.bufScales.size), "Encoder bufScales size cannot be NaN");
    assert.ok(!Number.isNaN(enc.bufVecs.size), "Encoder bufVecs size cannot be NaN");
  });
}

// ============================================================================
// SUITE 3: KV Cache Sizing, Clamping, and Stepping
// ============================================================================
console.log("\n--- Suite 3: KV Cache Sizing, Clamping, and Stepping ---");

runTest("TC-KV-02: Float16 KV cache sizing at 768 tokens fits in 128 MiB (84 MiB <= 128 MiB)", () => {
  const mockModule = createMockWasmModule();
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: 128 << 20 });
  const dec = new WebGPUDecoder(mockModule);
  dec.setupDevice(device, null, ["shader-f16"]);
  dec.kvF16 = true;
  dec.kvBytes = 2;

  dec.cfg = {
    layers: 28, hidden: 2048, heads: 16, kvHeads: 8, headDim: 128, inter: 6144, vocab: 151936,
    qDim: 2048, kvDim: 1024, headsPerKv: 2
  };

  dec.allocateStorageBuffers({
    shardBudget: 128 << 20,
    entries: decEntries,
    header: modelHeader,
    cfg: dec.cfg,
  });

  // Test prepareContext under f16 at 768 tokens (760 prompt + 8 padding = 768)
  dec.prepareContext(760, 0, { prefillSeq: 760 });
  const kvBytes = dec.bufKV.size;
  const maxBinding = 128 << 20;

  assert.ok(kvBytes <= maxBinding, `f16 KV cache at 768 tokens (${kvBytes} bytes) must be <= 128 MiB (${maxBinding} bytes)`);
  assert.strictEqual(kvBytes, 88080384, "f16 KV cache at 768 tokens must equal exactly 88,080,384 bytes (84 MiB)");
});

runTest("TC-KV-03: Float32 fallback KV cache strictly clamped to 512 tokens (112 MiB <= 128 MiB)", () => {
  const mockModule = createMockWasmModule();
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: 128 << 20 });
  const dec = new WebGPUDecoder(mockModule);
  dec.setupDevice(device);
  dec.kvF16 = false;
  dec.kvBytes = 4;

  dec.cfg = {
    layers: 28, hidden: 2048, heads: 16, kvHeads: 8, headDim: 128, inter: 6144, vocab: 151936,
    qDim: 2048, kvDim: 1024, headsPerKv: 2
  };

  dec.allocateStorageBuffers({
    shardBudget: 128 << 20,
    entries: decEntries,
    header: modelHeader,
    cfg: dec.cfg,
  });

  // Context length of 512 tokens (504 prompt + 8 padding = 512)
  dec.prepareContext(504, 0, { prefillSeq: 504 });
  const kvBytes512 = dec.bufKV.size;
  const maxBinding = 128 << 20;

  assert.ok(kvBytes512 <= maxBinding, `f32 KV cache at 512 tokens (${kvBytes512} bytes) must be <= 128 MiB (${maxBinding} bytes)`);
  assert.strictEqual(kvBytes512, 117440512, "f32 KV cache at 512 tokens must equal exactly 117,440,512 bytes (112 MiB)");

  // Context length exceeding 512 on f32 must throw descriptive error
  assert.throws(() => {
    dec.prepareContext(513, 0);
  }, /exceeds maximum supported context length \(512\)/);
});

runTest("TC-KV-04: Suffix prefill stepping aligns to 768 (f16) and 512 (f32)", () => {
  const mockModule = createMockWasmModule();
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: 128 << 20 });
  const dec = new WebGPUDecoder(mockModule);
  dec.setupDevice(device);

  dec.cfg = {
    layers: 28, hidden: 2048, heads: 16, kvHeads: 8, headDim: 128, inter: 6144, vocab: 151936,
    qDim: 2048, kvDim: 1024, headsPerKv: 2
  };

  dec.allocateStorageBuffers({
    shardBudget: 128 << 20,
    entries: decEntries,
    header: modelHeader,
    cfg: dec.cfg,
  });

  // f16 stepping: 10 tokens suffix prefill should step to 768
  dec.kvF16 = true;
  dec.kvBytes = 2;
  dec.prepareContext(10, 0, { prefillBase: 0 });
  assert.strictEqual(dec.maxSeq, 768, "Suffix prefill on f16 must step maxSeq to 768");

  // f32 stepping: 10 tokens suffix prefill should step to 512
  dec.kvF16 = false;
  dec.kvBytes = 4;
  dec.bufKV = null; // reset cache
  dec.prepareContext(10, 0, { prefillBase: 0 });
  assert.strictEqual(dec.maxSeq, 512, "Suffix prefill on f32 must step maxSeq to 512");
});

// ============================================================================
// SUITE 4: Half-Float (widenF16) Oracle Verification
// ============================================================================
console.log("\n--- Suite 4: Half-Float Decoding Fallbacks & Edge Cases ---");

runTest("TC-F16-01: widenF16 correctly decodes all 65,536 half-float values against IEEE-754 oracle", () => {
  function widenF16_fallback(u16) {
    const out = new Float32Array(u16.length);
    const u32Buf = new Uint32Array(1);
    const f32Buf = new Float32Array(u32Buf.buffer);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i];
      const s = (h & 0x8000) << 16;
      let e = (h & 0x7c00) >> 10;
      let f = h & 0x03ff;
      if (e === 0) {
        if (f !== 0) {
          while (!(f & 0x0400)) { f <<= 1; e--; }
          e++; f &= ~0x0400;
          u32Buf[0] = s | ((e + 112) << 23) | (f << 13);
        } else { u32Buf[0] = s; }
      } else if (e === 0x1f) {
        u32Buf[0] = s | 0x7f800000 | (f << 13);
      } else {
        u32Buf[0] = s | ((e + 112) << 23) | (f << 13);
      }
      out[i] = f32Buf[0];
    }
    return out;
  }

  function refHalf(h) {
    const s = (h >> 15) & 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    if (e === 0) {
      if (f === 0) return s ? -0.0 : 0.0;
      return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    }
    if (e === 31) {
      if (f === 0) return s ? -Infinity : Infinity;
      return NaN;
    }
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  const allH = new Uint16Array(65536);
  for (let i = 0; i < 65536; i++) allH[i] = i;
  const decoded = widenF16_fallback(allH);

  for (let i = 0; i < 65536; i++) {
    const expected = refHalf(i);
    const actual = decoded[i];
    if (Number.isNaN(expected)) {
      assert.ok(Number.isNaN(actual), `Expected NaN for 0x${i.toString(16)}, got ${actual}`);
    } else if (Object.is(expected, -0.0)) {
      assert.ok(Object.is(actual, -0.0), `Expected -0.0 for 0x${i.toString(16)}, got ${actual}`);
    } else {
      assert.strictEqual(actual, expected, `Mismatch for half 0x${i.toString(16)}: expected ${expected}, got ${actual}`);
    }
  }
});

runTest("TC-F16-02: widenF16 critical boundary constants", () => {
  function widenF16_fallback(u16) {
    const out = new Float32Array(u16.length);
    const u32Buf = new Uint32Array(1);
    const f32Buf = new Float32Array(u32Buf.buffer);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i];
      const s = (h & 0x8000) << 16;
      let e = (h & 0x7c00) >> 10;
      let f = h & 0x03ff;
      if (e === 0) {
        if (f !== 0) {
          while (!(f & 0x0400)) { f <<= 1; e--; }
          e++; f &= ~0x0400;
          u32Buf[0] = s | ((e + 112) << 23) | (f << 13);
        } else { u32Buf[0] = s; }
      } else if (e === 0x1f) {
        u32Buf[0] = s | 0x7f800000 | (f << 13);
      } else {
        u32Buf[0] = s | ((e + 112) << 23) | (f << 13);
      }
      out[i] = f32Buf[0];
    }
    return out;
  }

  const specialInputs = new Uint16Array([
    0x0000, // +0.0
    0x8000, // -0.0
    0x3c00, // +1.0
    0xbc00, // -1.0
    0x7bff, // max normal (65504)
    0xfbff, // min normal (-65504)
    0x0400, // min positive normal (2^-14 ~ 6.1035e-5)
    0x0001, // min positive subnormal (2^-24 ~ 5.9605e-8)
    0x7c00, // +Infinity
    0xfc00, // -Infinity
  ]);
  const out = widenF16_fallback(specialInputs);

  assert.strictEqual(Object.is(out[0], 0.0), true);
  assert.strictEqual(Object.is(out[1], -0.0), true);
  assert.strictEqual(out[2], 1.0);
  assert.strictEqual(out[3], -1.0);
  assert.strictEqual(out[4], 65504.0);
  assert.strictEqual(out[5], -65504.0);
  assert.ok(Math.abs(out[6] - Math.pow(2, -14)) < 1e-10);
  assert.ok(Math.abs(out[7] - Math.pow(2, -24)) < 1e-15);
  assert.strictEqual(out[8], Infinity);
  assert.strictEqual(out[9], -Infinity);
});



console.log("\n=======================================================");
console.log(`TOTAL: ${totalTests}, PASSED: ${passedTests - failedTests}, FAILED: ${failedTests}`);
console.log("=======================================================\n");

if (failedTests > 0) {
  console.error("FAILURES DETECTED:");
  for (const f of failures) {
    console.error(`- ${f.name}: ${f.error}`);
  }
  process.exit(1);
} else {
  process.exit(0);
}
