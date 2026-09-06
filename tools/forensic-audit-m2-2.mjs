/**
 * tools/forensic-audit-m2-2.mjs
 * Forensic Integrity & Adversarial Stress Suite for Milestone 2 (Round 2)
 *
 * Independent verification script executed by Forensic Auditor M2 (Round 2)
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

globalThis.GPUShaderStage = {
  VERTEX: 0x1,
  FRAGMENT: 0x2,
  COMPUTE: 0x4,
};

globalThis.GPUMapMode = {
  READ: 0x01,
  WRITE: 0x02,
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

class ForensicMockBuffer {
  constructor(size, usage, label = "") {
    if (typeof size !== "number" || Number.isNaN(size) || !Number.isFinite(size) || size < 0) {
      throw new TypeError(`Buffer size must be a non-negative finite number, got: ${size}`);
    }
    this.size = size;
    this.usage = usage;
    this.label = label;
    this.destroyed = false;
    this.mapState = "unmapped";
  }
  destroy() {
    this.destroyed = true;
  }
  async mapAsync() {
    this.mapState = "mapped";
  }
  getMappedRange() {
    return new ArrayBuffer(this.size);
  }
  unmap() {
    this.mapState = "unmapped";
  }
}

class ForensicMockGPUDevice {
  constructor(options = {}) {
    this.limits = {
      maxStorageBufferBindingSize: options.maxBinding || (128 << 20),
      maxBufferSize: options.maxBinding ? options.maxBinding * 4 : (256 << 20),
      maxComputeWorkgroupsPerDimension: 65535,
    };
    this.features = new Set(options.features || []);
    this.buffers = [];
    this.failBufferAt = options.failBufferAt ?? -1;
    this.eventListeners = new Map();
    this.pipelineAsyncCallCount = 0;
    this.pipelineSyncCallCount = 0;
    this.lost = new Promise(() => {});
    this.queue = {
      writeBuffer: () => {},
      submit: () => {},
      onSubmittedWorkDone: async () => {},
    };
  }

  createCommandEncoder() {
    return {
      copyBufferToBuffer: () => {},
      finish: () => ({}),
    };
  }

  createBuffer(desc) {
    if (this.failBufferAt >= 0 && this.buffers.length >= this.failBufferAt) {
      throw new Error(`Forensic Simulated OOM: Buffer allocation #${this.buffers.length + 1} failed`);
    }
    const buf = new ForensicMockBuffer(desc.size, desc.usage, desc.label);
    this.buffers.push(buf);
    return buf;
  }

  createBindGroupLayout() { return {}; }
  createPipelineLayout() { return {}; }
  createShaderModule() {
    return {
      getCompilationInfo: async () => ({ messages: [] }),
    };
  }
  createComputePipeline() {
    this.pipelineSyncCallCount++;
    return {};
  }
  async createComputePipelineAsync() {
    this.pipelineAsyncCallCount++;
    await new Promise((r) => setTimeout(r, 2));
    return {};
  }

  addEventListener(type, listener) {
    if (!this.eventListeners.has(type)) this.eventListeners.set(type, []);
    this.eventListeners.get(type).push(listener);
  }

  removeEventListener(type, listener) {
    if (!this.eventListeners.has(type)) return;
    const list = this.eventListeners.get(type);
    const idx = list.indexOf(listener);
    if (idx !== -1) list.splice(idx, 1);
  }
}

let passed = 0;
let failed = 0;
const results = [];

function check(title, fn) {
  try {
    fn();
    console.log(`  [PASS] ${title}`);
    passed++;
    results.push({ title, status: "PASS" });
  } catch (err) {
    console.error(`  [FAIL] ${title}: ${err.message}`);
    failed++;
    results.push({ title, status: "FAIL", error: err.message, stack: err.stack });
  }
}

async function checkAsync(title, fn) {
  try {
    await fn();
    console.log(`  [PASS] ${title}`);
    passed++;
    results.push({ title, status: "PASS" });
  } catch (err) {
    console.error(`  [FAIL] ${title}: ${err.message}`);
    failed++;
    results.push({ title, status: "FAIL", error: err.message, stack: err.stack });
  }
}

console.log("==================================================================");
console.log("FORENSIC AUDITOR M2.2: INDEPENDENT ADVERSARIAL STRESS SUITE");
console.log("==================================================================");

// ----------------------------------------------------------------------------
// Test 1: Sub-minimum binding constraint enforcement
// ----------------------------------------------------------------------------
check("Check 1: allocateStorageBuffers rejects SHARD_BUDGET < cols with informative error", () => {
  const dev = new ForensicMockGPUDevice({ maxBinding: 1024 }); // cols = 2048 > 1024
  const dec = new WebGPUDecoder({}, { device: dev });
  assert.throws(
    () => dec.allocateStorageBuffers({ header: modelHeader, entries: decEntries, shardBudget: 1024 }),
    /cannot be smaller than matrix column width/
  );
  assert.strictEqual(dec.destroyed, true, "Must destroy state after rejection");
});

// ----------------------------------------------------------------------------
// Test 2: Progressive allocation rollback on shard 0 failure (WebGPUDecoder)
// ----------------------------------------------------------------------------
check("Check 2: WebGPUDecoder handles buffer failure on shard 0 without error in destroy()", () => {
  const dev = new ForensicMockGPUDevice({ failBufferAt: 0 }); // First buffer fails
  const dec = new WebGPUDecoder({}, { device: dev });
  assert.throws(
    () => dec.allocateStorageBuffers({ header: modelHeader, entries: decEntries, shardBudget: 128 << 20 }),
    /Forensic Simulated OOM/
  );
  assert.strictEqual(dec.destroyed, true);
  assert.strictEqual(dec.bufQuants, null);
});

// ----------------------------------------------------------------------------
// Test 3: Progressive allocation rollback on last shard failure (WebGPUDecoder)
// ----------------------------------------------------------------------------
check("Check 3: WebGPUDecoder cleanly destroys all 14 shards if shard 15 fails", () => {
  const dev = new ForensicMockGPUDevice({ failBufferAt: 14 }); // Shard 15 fails (0-indexed: 14)
  const dec = new WebGPUDecoder({}, { device: dev });
  assert.throws(
    () => dec.allocateStorageBuffers({ header: modelHeader, entries: decEntries, shardBudget: 128 << 20 }),
    /Forensic Simulated OOM/
  );
  assert.strictEqual(dev.buffers.length, 14);
  for (const b of dev.buffers) {
    assert.strictEqual(b.destroyed, true, "Buffer must be destroyed");
  }
  assert.strictEqual(dec.destroyed, true);
});

// ----------------------------------------------------------------------------
// Test 4: Progressive allocation rollback on WebGPUEncoder
// ----------------------------------------------------------------------------
check("Check 4: WebGPUEncoder handles mid-allocation OOM and destroys all previous shards", () => {
  const dev = new ForensicMockGPUDevice({ failBufferAt: 2 }); // Shard 2 fails out of 3 shards
  const enc = new WebGPUEncoder({}, { device: dev });
  assert.throws(
    () => enc.allocateStorageBuffers({ device: dev, header: modelHeader, entries: encEntries, shardBudget: 128 << 20 }),
    /Forensic Simulated OOM/
  );
  assert.strictEqual(dev.buffers.length, 2);
  for (const b of dev.buffers) {
    assert.strictEqual(b.destroyed, true);
  }
  assert.strictEqual(enc.destroyed, true);
});

// ----------------------------------------------------------------------------
// Test 5: Mid-init destroy interleaving (early, middle, late)
// ----------------------------------------------------------------------------
await checkAsync("Check 5: WebGPUDecoder mid-init destroy at various compile ticks preserves invariant ready===false", async () => {
  for (const cancelAt of [1, 5, 12, 20, 27]) {
    const dev = new ForensicMockGPUDevice();
    const dec = new WebGPUDecoder({}, { device: dev });
    let count = 0;
    const p = dec.finishInit(() => {
      count++;
      if (count === cancelAt) {
        dec.destroy();
      }
    });
    await p;
    assert.strictEqual(dec.destroyed, true, `Expected destroyed=true at cancelAt=${cancelAt}`);
    assert.strictEqual(dec.ready, false, `Expected ready=false at cancelAt=${cancelAt}`);
  }
});

// ----------------------------------------------------------------------------
// Test 6: Concurrent finishInit deduplication across 10 parallel calls
// ----------------------------------------------------------------------------
await checkAsync("Check 6: Concurrent finishInit across 10 parallel calls produces exactly 28 pipeline compiles", async () => {
  const dev = new ForensicMockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });
  const promises = Array.from({ length: 10 }, () => dec.finishInit());
  await Promise.all(promises);
  assert.strictEqual(dec.ready, true);
  assert.strictEqual(dev.pipelineAsyncCallCount, 28, "Must compile exactly 28 pipelines");
});

// ----------------------------------------------------------------------------
// Test 7: Retry finishInit after failure
// ----------------------------------------------------------------------------
await checkAsync("Check 7: finishInit cleans up _initPromise on failure, permitting clean retry", async () => {
  const dev = new ForensicMockGPUDevice();
  // Fail the first attempt by corrupting device
  const origAsync = dev.createComputePipelineAsync;
  dev.createComputePipelineAsync = async () => {
    throw new Error("Simulated transient shader compile timeout");
  };
  const dec = new WebGPUDecoder({}, { device: dev });

  await assert.rejects(() => dec.finishInit(), /Simulated transient shader compile timeout/);
  assert.strictEqual(dec.ready, false);
  assert.strictEqual(dec._initPromise, null, "_initPromise must be cleared in finally");

  // Restore and retry
  dev.createComputePipelineAsync = origAsync;
  await dec.finishInit();
  assert.strictEqual(dec.ready, true, "Retry must succeed");
});

// ----------------------------------------------------------------------------
// Test 8: Uncaught error listener detachment is idempotent and safe against throwing
// ----------------------------------------------------------------------------
check("Check 8: Uncaptured error listener cleanup handles throwing removeEventListener gracefully", () => {
  const dev = new ForensicMockGPUDevice();
  dev.removeEventListener = () => {
    throw new Error("Simulated device lost removeEventListener failure");
  };
  const dec = new WebGPUDecoder({}, { device: dev });
  assert.doesNotThrow(() => dec.destroy());
  assert.strictEqual(dec.destroyed, true);
});

// ----------------------------------------------------------------------------
// Test 9: Mode mutation guards on destroyed instance
// ----------------------------------------------------------------------------
check("Check 9: All mode mutators safely return on destroyed decoder", () => {
  const dev = new ForensicMockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });
  dec.destroy();
  assert.strictEqual(dec.setMatvecWidth(128), false);
  assert.strictEqual(dec.setSubgroups(true), false);
  assert.doesNotThrow(() => dec.setQuantizeActivations(true));
  assert.doesNotThrow(() => dec.applyMode());
  assert.doesNotThrow(() => dec.ensureMatvecPipes("q8", 128));
  assert.strictEqual(dec.matvecPipesTree, null);
});

// ----------------------------------------------------------------------------
// Test 10: KV cache context ceiling on Float32 vs Float16
// ----------------------------------------------------------------------------
check("Check 10: prepareContext correctly calculates bytesPerToken and enforces context boundaries", () => {
  const heap = new Uint8Array(20 * 1024 * 1024);
  const mockM = {
    HEAPU8: heap,
    HEAPF32: new Float32Array(heap.buffer),
    _qwen_wasm_alloc: () => 1024,
    _qwen_wasm_release: () => {},
    _qwen_wasm_rms_eps: () => 1e-6,
    _qwen_wasm_rope_theta: () => 1e6,
    _qwen_wasm_kv_stride: () => 128,
    _qwen_wasm_kv_k_ptr: () => 1024,
    _qwen_wasm_kv_v_ptr: () => 1024 + 128 * 1024,
    _qwen_wasm_kv_is_f16: () => 0,
  };
  const dev = new ForensicMockGPUDevice({ maxBinding: 128 << 20 });
  const dec = new WebGPUDecoder(mockM, { device: dev });
  dec.allocateStorageBuffers({ header: modelHeader, entries: decEntries, shardBudget: 128 << 20 });

  // On f32 (kvF16 = false): limit is 512
  assert.strictEqual(dec.kvF16, false);
  assert.doesNotThrow(() => dec.prepareContext(512, 0));
  assert.throws(() => dec.prepareContext(513, 0), /exceeds maximum supported context length \(512\)/);

  // If kvF16 is enabled: limit is higher (min(1600, maxPhysical))
  dec.kvF16 = true;
  dec.kvBytes = 2;
  assert.doesNotThrow(() => dec.prepareContext(768, 0));
});

console.log("\n==================================================================");
console.log(`FORENSIC STRESS TEST RESULTS: ${passed} passed, ${failed} failed`);
console.log("==================================================================");

if (failed > 0) process.exit(1);
process.exit(0);


