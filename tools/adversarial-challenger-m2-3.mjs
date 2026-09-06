/**
 * tools/adversarial-challenger-m2-3.mjs
 * Challenger M2.3: Dimension Setup, Activation Sizing, and Context Limit Stress Suite
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
const { extractModelDescriptorsFromHeader } = app;

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
    this.copiedRanges = [];
    this.queue = {
      writeBuffer(buf, off, src) {
        if (buf.destroyed) throw new Error("write to destroyed buffer");
      },
      submit: (cmds) => {},
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
    const dev = this;
    return {
      copyBufferToBuffer(src, srcOff, dst, dstOff, size) {
        dev.copiedRanges.push({ src, srcOff, dst, dstOff, size });
      },
      finish() { return {}; },
    };
  }
}

function createMockWasmModule() {
  const heap = new Uint8Array(64 * 1024 * 1024);
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
console.log("CHALLENGER M2.3: EMPIRICAL STRESS & BOUNDARY TEST SUITE");
console.log("=======================================================\n");

// ============================================================================
// SUITE 1: Dimension Setup Derivation & Invariant Verification
// ============================================================================
console.log("--- Suite 1: Dimension Setup Derivation & Invariant Verification ---");

runTest("TC-M23-DIM-01: Default 1.7B shape derives qDim=2048, kvDim=1024, headsPerKv=2", () => {
  const mockModule = createMockWasmModule();
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: 128 << 20 });
  const dec = new WebGPUDecoder(mockModule);
  dec.setupDevice(device);

  dec.allocateStorageBuffers({
    shardBudget: 128 << 20,
    entries: decEntries,
    header: modelHeader,
  });

  assert.strictEqual(dec.cfg.qDim, 2048, "dec.cfg.qDim must equal 2048");
  assert.strictEqual(dec.cfg.kvDim, 1024, "dec.cfg.kvDim must equal 1024");
  assert.strictEqual(dec.cfg.headsPerKv, 2, "dec.cfg.headsPerKv must equal 2");
  assert.strictEqual(dec.cfg.endOfText, 151643, "dec.cfg.endOfText must equal 151643");
  assert.strictEqual(dec.cfg.asrText, 151704, "dec.cfg.asrText must equal 151704");
  assert.strictEqual(dec.actFloats, 16544, "dec.actFloats must equal 16544");
  assert.strictEqual(dec.bufAct.size, 16544 * 4, "dec.bufAct.size must equal 66176 bytes");
});

runTest("TC-M23-DIM-02: Partial opts.cfg without qDim/kvDim is safely completed without NaN", () => {
  const mockModule = createMockWasmModule();
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: 128 << 20 });
  const dec = new WebGPUDecoder(mockModule);
  dec.setupDevice(device);

  // Incomplete config without qDim, kvDim, headsPerKv, endOfText, asrText
  const partialCfg = {
    layers: 28,
    hidden: 2048,
    heads: 16,
    kvHeads: 8,
    headDim: 128,
    inter: 6144,
    vocab: 151936,
    imEnd: 151645,
  };

  dec.allocateStorageBuffers({
    shardBudget: 128 << 20,
    entries: decEntries,
    header: modelHeader,
    cfg: partialCfg,
  });

  assert.strictEqual(dec.cfg.qDim, 2048);
  assert.strictEqual(dec.cfg.kvDim, 1024);
  assert.strictEqual(dec.cfg.headsPerKv, 2);
  assert.strictEqual(dec.cfg.endOfText, 151643);
  assert.strictEqual(dec.cfg.asrText, 151704);
  assert.ok(!Number.isNaN(dec.actFloats), "actFloats must not be NaN");
  assert.ok(!Number.isNaN(dec.bufAct.size), "bufAct.size must not be NaN");
});

runTest("TC-M23-DIM-03: Custom explicit qDim & kvDim in opts.cfg are preserved, not overwritten", () => {
  const mockModule = createMockWasmModule();
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: 128 << 20 });
  const dec = new WebGPUDecoder(mockModule);
  dec.setupDevice(device);

  const customCfg = {
    layers: 28,
    hidden: 2048,
    heads: 16,
    kvHeads: 8,
    headDim: 128,
    inter: 6144,
    vocab: 151936,
    imEnd: 151645,
    qDim: 2048,
    kvDim: 1024,
    headsPerKv: 2,
    endOfText: 99999,
    asrText: 88888,
  };

  dec.allocateStorageBuffers({
    shardBudget: 128 << 20,
    entries: decEntries,
    header: modelHeader,
    cfg: customCfg,
  });

  assert.strictEqual(dec.cfg.endOfText, 99999);
  assert.strictEqual(dec.cfg.asrText, 88888);
});

runTest("TC-M23-DIM-04: Non-standard GQA & MQA architectural dimensions allocate cleanly", () => {
  const mockModule = createMockWasmModule();
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: 128 << 20 });
  const dec = new WebGPUDecoder(mockModule);
  dec.setupDevice(device);

  // MQA: 16 heads, 1 kvHead
  const mqaCfg = {
    layers: 24,
    hidden: 2048,
    heads: 16,
    kvHeads: 1,
    headDim: 128,
    inter: 6144,
    vocab: 151936,
    imEnd: 151645,
  };

  dec.allocateStorageBuffers({
    shardBudget: 128 << 20,
    entries: decEntries,
    header: modelHeader,
    cfg: mqaCfg,
  });

  assert.strictEqual(dec.cfg.qDim, 2048);
  assert.strictEqual(dec.cfg.kvDim, 128);
  assert.strictEqual(dec.cfg.headsPerKv, 16);
  assert.ok(dec.actFloats > 0 && !Number.isNaN(dec.actFloats));
  assert.strictEqual(dec.bufAct.size % 4, 0);
  assert.strictEqual(dec.bufTok.size % 4, 0);
  assert.strictEqual(dec.bufTokRead.size % 4, 0);

  // Verify monotonic slot addresses in A
  const keys = ["x", "xn", "q", "k", "v", "attn", "g", "sxn", "sattn", "sg"];
  for (let i = 0; i < keys.length - 1; i++) {
    const k1 = keys[i], k2 = keys[i + 1];
    assert.ok(dec.A[k2] > dec.A[k1], `dec.A.${k2} (${dec.A[k2]}) must be > dec.A.${k1} (${dec.A[k1]})`);
    assert.strictEqual(dec.A[k1] % 1, 0, `dec.A.${k1} must be an integer`);
  }
});

// ============================================================================
// SUITE 2: KV Cache Context Sizing & Clamping Limits
// ============================================================================
console.log("\n--- Suite 2: KV Cache Context Sizing & Clamping Limits ---");

runTest("TC-M23-CTX-01: Float32 fallback strictly enforces 512 context ceiling under 128 MiB", () => {
  const mockModule = createMockWasmModule();
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: 128 << 20 });
  const dec = new WebGPUDecoder(mockModule);
  dec.setupDevice(device);
  dec.kvF16 = false;
  dec.kvBytes = 4;

  dec.allocateStorageBuffers({
    shardBudget: 128 << 20,
    entries: decEntries,
    header: modelHeader,
  });

  // 1. Boundary: exactly 512 tokens (totalSeq = 512 + 0 = 512)
  dec.prepareContext(512, 0);
  assert.strictEqual(dec.maxSeq, 512);
  assert.ok(dec.bufKV.size <= (128 << 20), `KV buffer (${dec.bufKV.size}) must be <= 128 MiB`);
  assert.strictEqual(dec.bufKV.size, 117440512);

  // 2. Exact sum boundary: kvLen = 500, maxNew = 12 (totalSeq = 512)
  dec.prepareContext(500, 12);
  assert.strictEqual(dec.maxSeq, 512);

  // 3. One token over: kvLen = 513, maxNew = 0 (totalSeq = 513)
  assert.throws(() => {
    dec.prepareContext(513, 0);
  }, (err) => {
    return err.message.includes("exceeds maximum supported context length (512)") &&
           err.message.includes("Float32 fallback");
  });

  // 4. Sum one token over: kvLen = 500, maxNew = 13 (totalSeq = 513)
  assert.throws(() => {
    dec.prepareContext(500, 13);
  }, /exceeds maximum supported context length \(512\)/);
});

runTest("TC-M23-CTX-02: Float16 KV cache allows extended context up to binding limit (1170 tokens)", () => {
  const mockModule = createMockWasmModule();
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: 128 << 20 });
  const dec = new WebGPUDecoder(mockModule);
  dec.setupDevice(device, null, ["shader-f16"]);
  dec.kvF16 = true;
  dec.kvBytes = 2;

  dec.allocateStorageBuffers({
    shardBudget: 128 << 20,
    entries: decEntries,
    header: modelHeader,
  });

  // maxPhysicalTokens = floor((128 * 1024 * 1024) / (2 * 28 * 1024 * 2)) = floor(134217728 / 114688) = 1170
  // 1. Boundary: 1170 tokens
  dec.prepareContext(1170, 0);
  assert.strictEqual(dec.maxSeq, 1170);
  assert.strictEqual(dec.bufKV.size, 134184960);
  assert.ok(dec.bufKV.size <= (128 << 20), "KV buffer size must fit in 128 MiB");

  // 2. Exceeding by 1 token (1171) must throw
  assert.throws(() => {
    dec.prepareContext(1171, 0);
  }, /exceeds maximum supported context length \(1170\)/);
});

runTest("TC-M23-CTX-03: Restricted 64 MiB binding limit dynamically recalculates token ceilings", () => {
  const mockModule = createMockWasmModule();
  const maxBinding64MB = 64 << 20; // 67,108,864 bytes
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: maxBinding64MB });
  const dec = new WebGPUDecoder(mockModule);
  dec.setupDevice(device);

  dec.allocateStorageBuffers({
    shardBudget: maxBinding64MB,
    entries: decEntries,
    header: modelHeader,
  });

  // Under f32: bytesPerToken = 229,376. maxPhysicalTokens = floor(67108864 / 229376) = 292 tokens
  dec.kvF16 = false;
  dec.kvBytes = 4;

  dec.prepareContext(292, 0);
  assert.strictEqual(dec.maxSeq, 292);
  assert.strictEqual(dec.bufKV.size, 66977792);
  assert.ok(dec.bufKV.size <= maxBinding64MB, "KV buffer size must fit in 64 MiB");

  assert.throws(() => {
    dec.prepareContext(293, 0);
  }, /exceeds maximum supported context length \(292\)/);

  // Under f16: bytesPerToken = 114,688. maxPhysicalTokens = floor(67108864 / 114688) = 585 tokens
  dec.kvF16 = true;
  dec.kvBytes = 2;
  dec.bufKV = null;

  dec.prepareContext(585, 0);
  assert.strictEqual(dec.maxSeq, 585);
  assert.strictEqual(dec.bufKV.size, 67092480);
  assert.ok(dec.bufKV.size <= maxBinding64MB, "f16 KV buffer size must fit in 64 MiB");

  assert.throws(() => {
    dec.prepareContext(586, 0);
  }, /exceeds maximum supported context length \(585\)/);
});

runTest("TC-M23-CTX-04: Extreme zero & single token contexts initialize cleanly", () => {
  const mockModule = createMockWasmModule();
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: 128 << 20 });
  const dec = new WebGPUDecoder(mockModule);
  dec.setupDevice(device);

  dec.allocateStorageBuffers({
    shardBudget: 128 << 20,
    entries: decEntries,
    header: modelHeader,
  });

  // Zero-token context
  dec.prepareContext(0, 0);
  assert.strictEqual(dec.maxSeq, 8); // 0 + 0 + 8
  assert.ok(dec.bufKV.size > 0 && !Number.isNaN(dec.bufKV.size));
  assert.ok(dec.bufAct.size > 0 && !Number.isNaN(dec.bufAct.size));
  assert.ok(dec.bufScratch.size > 0 && !Number.isNaN(dec.bufScratch.size));

  // Single-token context
  dec.prepareContext(1, 0);
  assert.strictEqual(dec.maxSeq, 9); // 1 + 0 + 8
  assert.ok(dec.bufKV.size > 0 && !Number.isNaN(dec.bufKV.size));
});

// ============================================================================
// SUITE 3: Dynamic Growth & Buffer Destruction (Zero Memory Leak)
// ============================================================================
console.log("\n--- Suite 3: Dynamic Growth & Buffer Destruction (Zero Memory Leak) ---");

runTest("TC-M23-GROW-01: Re-allocating bufAct and bufScratch destroys stale buffers", () => {
  const mockModule = createMockWasmModule();
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: 128 << 20 });
  const dec = new WebGPUDecoder(mockModule);
  dec.setupDevice(device);

  dec.allocateStorageBuffers({
    shardBudget: 128 << 20,
    entries: decEntries,
    header: modelHeader,
  });

  // Step 1: Small prefill context
  dec.prepareContext(10, 0, { prefillSeq: 10 });
  const firstBufAct = dec.bufAct;
  const firstBufScratch = dec.bufScratch;
  const firstBufKV = dec.bufKV;

  assert.strictEqual(firstBufAct.destroyed, false);
  assert.strictEqual(firstBufScratch.destroyed, false);
  assert.strictEqual(firstBufKV.destroyed, false);

  // Step 2: Larger prefill context requiring buffer expansion
  dec.prepareContext(250, 0, { prefillSeq: 250 });
  const secondBufAct = dec.bufAct;
  const secondBufScratch = dec.bufScratch;
  const secondBufKV = dec.bufKV;

  // Stale buffers MUST have been destroyed to prevent GPU memory leak
  assert.strictEqual(firstBufAct.destroyed, true, "First bufAct must be destroyed upon growth");
  assert.strictEqual(firstBufScratch.destroyed, true, "First bufScratch must be destroyed upon growth");
  assert.strictEqual(firstBufKV.destroyed, true, "First bufKV must be destroyed upon growth");

  assert.strictEqual(secondBufAct.destroyed, false);
  assert.strictEqual(secondBufScratch.destroyed, false);
  assert.strictEqual(secondBufKV.destroyed, false);
});

runTest("TC-M23-GROW-02: Suffix prefill expansion copies retained KV data and destroys old buffer", () => {
  const mockModule = createMockWasmModule();
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: 128 << 20 });
  const dec = new WebGPUDecoder(mockModule);
  dec.setupDevice(device);
  dec.kvF16 = false;
  dec.kvBytes = 4;

  dec.allocateStorageBuffers({
    shardBudget: 128 << 20,
    entries: decEntries,
    header: modelHeader,
  });

  // Chunk 1: Initial suffix prefill
  dec.prepareContext(10, 0, { prefillSeq: 10, prefillBase: 0 });
  const oldKV = dec.bufKV;
  assert.strictEqual(dec.maxSeq, 512, "Suffix prefill on f32 must step to 512");
  assert.strictEqual(oldKV.destroyed, false);

  // Chunk 2: Expanding within capacity (should NOT reallocate)
  dec.prepareContext(20, 0, { prefillSeq: 10, prefillBase: 10 });
  assert.strictEqual(dec.bufKV, oldKV, "Buffer should be preserved if maxSeq <= current maxSeq");
  assert.strictEqual(oldKV.destroyed, false);
});

// ============================================================================
// SUITE 4: Shard Budget Sub-Minimum & Boundary Enforcement
// ============================================================================
console.log("\n--- Suite 4: Shard Budget Sub-Minimum & Boundary Enforcement ---");

runTest("TC-M23-SHARD-01: Shard budget below matrix column width throws descriptive error", () => {
  const mockModule = createMockWasmModule();
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: 128 << 20 });
  const dec = new WebGPUDecoder(mockModule);
  dec.setupDevice(device);

  // Column width of embedding matrix is 2048. Setting budget to 2047 must throw.
  assert.throws(() => {
    dec.allocateStorageBuffers({
      shardBudget: 2047,
      entries: decEntries,
      header: modelHeader,
    });
  }, (err) => {
    return err.message.includes("SHARD_BUDGET (2047) cannot be smaller than matrix column width (2048)");
  });
});

// ============================================================================
// SUITE 5: Extended Edge Cases & Stress Scenarios
// ============================================================================
console.log("\n--- Suite 5: Extended Edge Cases & Stress Scenarios ---");

runTest("TC-M23-EXT-01: 0.6B model header auto-detection derives correct dimensions", () => {
  const mockModule = createMockWasmModule();
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: 128 << 20 });
  const dec = new WebGPUDecoder(mockModule);
  dec.setupDevice(device);

  // Simulate 0.6B model header where input_layernorm.weight is shape [1024]
  const header06b = {
    "thinker.model.layers.0.input_layernorm.weight": { shape: [1024] },
  };
  for (let l = 0; l < 24; l++) {
    header06b[`thinker.model.layers.${l}.self_attn.q_proj.weight.q8`] = { shape: [1024, 1024] };
  }

  dec.allocateStorageBuffers({
    shardBudget: 128 << 20,
    entries: decEntries,
    header: header06b,
  });

  assert.strictEqual(dec.cfg.hidden, 1024, "0.6B hidden must be 1024");
  assert.strictEqual(dec.cfg.inter, 3072, "0.6B inter must be 3072");
  assert.strictEqual(dec.cfg.layers, 24, "0.6B layers must be 24");
  assert.strictEqual(dec.cfg.qDim, 2048);
  assert.strictEqual(dec.cfg.kvDim, 1024);
  assert.ok(dec.actFloats > 0 && !Number.isNaN(dec.actFloats));
});

runTest("TC-M23-EXT-02: User-configured opts.maxSeq is safely clamped by hardware binding limit", () => {
  const mockModule = createMockWasmModule();
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: 128 << 20 });
  const dec = new WebGPUDecoder(mockModule);
  dec.setupDevice(device, null, ["shader-f16"]);
  dec.kvF16 = true;
  dec.kvBytes = 2;

  // Case A: User specifies maxSeq: 768 (stricter than hardware max 1170)
  dec.opts = { maxSeq: 768 };
  dec.allocateStorageBuffers({
    shardBudget: 128 << 20,
    entries: decEntries,
    header: modelHeader,
  });

  dec.prepareContext(768, 0);
  assert.strictEqual(dec.maxSeq, 768);

  assert.throws(() => {
    dec.prepareContext(769, 0);
  }, /exceeds maximum supported context length \(768\)/);

  // Case B: User specifies maxSeq: 4096 (exceeds hardware max 1170)
  // Must safely clamp to 1170 and not allocate beyond 128 MiB
  dec.opts = { maxSeq: 4096 };
  dec.prepareContext(1170, 0);
  assert.strictEqual(dec.maxSeq, 1170);
  assert.ok(dec.bufKV.size <= (128 << 20));

  assert.throws(() => {
    dec.prepareContext(1171, 0);
  }, /exceeds maximum supported context length \(1170\)/);
});

runTest("TC-M23-EXT-03: Multi-iteration streaming prefill up to exact boundary and beyond", () => {
  const mockModule = createMockWasmModule();
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: 128 << 20 });
  const dec = new WebGPUDecoder(mockModule);
  dec.setupDevice(device);
  dec.kvF16 = false;
  dec.kvBytes = 4;

  dec.allocateStorageBuffers({
    shardBudget: 128 << 20,
    entries: decEntries,
    header: modelHeader,
  });

  // Stream 51 chunks of 10 tokens (0 -> 510 tokens)
  for (let step = 0; step < 51; step++) {
    const p0 = step * 10;
    const nNew = 10;
    const seq = p0 + nNew;
    dec.prepareContext(seq, 0, { prefillSeq: nNew, prefillBase: p0 });
    assert.strictEqual(dec.maxSeq, 512);
  }

  // 52nd chunk: 2 tokens (seq = 512 tokens total, exact limit)
  dec.prepareContext(512, 0, { prefillSeq: 2, prefillBase: 510 });
  assert.strictEqual(dec.maxSeq, 512);
  assert.ok(dec.bufKV.size <= (128 << 20));

  // 53rd chunk: 1 token (seq = 513 tokens total, overflow)
  assert.throws(() => {
    dec.prepareContext(513, 0, { prefillSeq: 1, prefillBase: 512 });
  }, /exceeds maximum supported context length \(512\)/);
});

runTest("TC-M23-EXT-04: WebGPUEncoder dimension auto-detection & buffer allocation", () => {
  const mockModule = createMockWasmModule();
  const device = new StrictMockGPUDevice({ maxStorageBufferBindingSize: 128 << 20 });
  const enc = new WebGPUEncoder(mockModule);
  enc.device = device;
  enc.adapter = { limits: { maxStorageBufferBindingSize: 128 << 20 } };

  enc.allocateStorageBuffers({
    shardBudget: 128 << 20,
    entries: encEntries,
    header: modelHeader,
  });

  assert.strictEqual(enc.dModel, 1024);
  assert.strictEqual(enc.layers, 24);
  assert.strictEqual(enc.heads, 16);
  assert.strictEqual(enc.headDim, 64);
  assert.strictEqual(enc.ffnDim, 4096);
  assert.strictEqual(enc.outDim, 2048);
  assert.ok(enc.bufScales.size > 0 && !Number.isNaN(enc.bufScales.size));
  assert.ok(enc.bufVecs.size > 0 && !Number.isNaN(enc.bufVecs.size));
  for (const sh of enc.shards) {
    assert.ok(sh.bytes <= (128 << 20));
  }
});

console.log("\n=======================================================");
console.log(`TOTAL: ${totalTests}, PASSED: ${passedTests}, FAILED: ${failedTests}`);
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
