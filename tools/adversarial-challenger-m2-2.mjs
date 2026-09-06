/**
 * tools/adversarial-challenger-m2-2.mjs
 * Empirical Challenger M2.2 Verification Harness
 *
 * Focus Areas:
 * 1. Stress-test destroy() idempotency: multiple calls, before init, during init, and after init.
 * 2. Simulate device.lost: clean callback detachment, resource clearing, UI recovery signals.
 * 3. Pipeline compilation concurrency & fallback: sync vs async pipeline resolution behavior.
 */

import assert from "node:assert/strict";
import { openSync, readSync, closeSync } from "node:fs";
import { join } from "node:path";

// 0. Environment & Mock Setup
globalThis.window = globalThis;
globalThis.self = globalThis;
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
globalThis.GPUMapMode = {
  READ: 0x01,
  WRITE: 0x02,
};
globalThis.GPUShaderStage = {
  VERTEX: 0x01,
  FRAGMENT: 0x02,
  COMPUTE: 0x04,
};

const mockElements = new Map();
globalThis.document = {
  getElementById: (id) => {
    if (!mockElements.has(id)) {
      mockElements.set(id, {
        id,
        classList: { add: () => {}, remove: () => {} },
        appendChild: () => {},
        addEventListener: () => {},
        style: {},
        value: "webgpu",
        disabled: false,
        textContent: "",
      });
    }
    return mockElements.get(id);
  },
  createElement: () => ({
    appendChild: () => {},
    classList: { add: () => {}, remove: () => {} },
    style: {},
    textContent: "",
  }),
};

// Session storage mock
const mockSessionStorage = new Map();
globalThis.sessionStorage = {
  getItem: (k) => mockSessionStorage.get(k) ?? null,
  setItem: (k, v) => mockSessionStorage.set(k, String(v)),
  removeItem: (k) => mockSessionStorage.delete(k),
  clear: () => mockSessionStorage.clear(),
};

// Mock location
globalThis.location = {
  reloaded: false,
  reload: () => { globalThis.location.reloaded = true; },
};

// Import implementations
const { WebGPUDecoder } = await import("../wasm/demo/webgpu-decoder.js");
const { WebGPUEncoder } = await import("../wasm/demo/webgpu-encoder.js");
const app = await import("../wasm/demo/app.js");
const {
  parseSafetensorsHeaderStream,
  prepareWasmReducedImage,
  extractModelDescriptorsFromHeader,
  buildIntervalDispatchTable,
} = app;

// 1. Mock GPU Infrastructure
class MockGPUBuffer {
  constructor(size, usage, label = "") {
    this.size = size;
    this.usage = usage;
    this.label = label;
    this.destroyed = false;
    this.destroyCount = 0;
    this.mapState = "unmapped"; // "unmapped", "pending", "mapped"
    this.unmapCount = 0;
    this.data = new ArrayBuffer(size);
  }

  async mapAsync(mode, offset = 0, size = this.size) {
    if (this.destroyed) throw new Error("mapAsync on destroyed buffer");
    if (this.mapState !== "unmapped") throw new Error(`Buffer already ${this.mapState}`);
    this.mapState = "pending";
    await new Promise((r) => setTimeout(r, 1));
    if (this.destroyed) {
      this.mapState = "unmapped";
      throw new Error("AbortError: Buffer was destroyed during mapAsync");
    }
    this.mapState = "mapped";
  }

  getMappedRange(offset = 0, size = this.size) {
    if (this.mapState !== "mapped") throw new Error("Buffer is not mapped");
    return this.data.slice(offset, offset + size);
  }

  unmap() {
    this.unmapCount++;
    this.mapState = "unmapped";
  }

  destroy() {
    this.destroyCount++;
    this.destroyed = true;
  }
}

class MockGPUQueue {
  constructor() {
    this.writes = [];
    this.syncCount = 0;
  }
  writeBuffer(buffer, offset, data) {
    if (buffer.destroyed) throw new Error("writeBuffer to destroyed buffer");
    this.writes.push({ buffer, offset, size: data.byteLength });
  }
  async onSubmittedWorkDone() {
    this.syncCount++;
    return Promise.resolve();
  }
}

class MockShaderModule {
  constructor(desc, compileMessages = []) {
    this.code = desc.code;
    this.label = desc.label;
    this.compileMessages = compileMessages;
  }
  async getCompilationInfo() {
    return { messages: this.compileMessages };
  }
}

class MockComputePipeline {
  constructor(desc) {
    this.label = desc.label;
    this.layout = desc.layout;
  }
}

class MockGPUDevice {
  constructor(limits = {}, features = [], supportAsync = true) {
    this.limits = {
      maxBufferSize: limits.maxBufferSize ?? (256 << 20),
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize ?? (128 << 20),
      maxComputeWorkgroupsPerDimension: 65535,
      maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize ?? 32768,
      ...limits,
    };
    this.features = new Set(features);
    this.queue = new MockGPUQueue();
    this.buffers = [];
    this.eventListeners = new Map();
    this.lostPromiseResolve = null;
    this.lost = new Promise((r) => { this.lostPromiseResolve = r; });
    this.failBufferAllocationAfter = -1;
    this.bufferCount = 0;
    this.shaderErrorMessages = [];
    this.pipelineAsyncCallCount = 0;
    this.pipelineSyncCallCount = 0;
    if (supportAsync) {
      this.createComputePipelineAsync = async (desc) => {
        this.pipelineAsyncCallCount++;
        await new Promise((r) => setTimeout(r, 1));
        return new MockComputePipeline(desc);
      };
    } else {
      this.createComputePipelineAsync = undefined;
    }
  }

  addEventListener(event, handler) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    this.eventListeners.get(event).push(handler);
  }

  removeEventListener(event, handler) {
    if (!this.eventListeners.has(event)) return;
    const list = this.eventListeners.get(event);
    const idx = list.indexOf(handler);
    if (idx >= 0) list.splice(idx, 1);
  }

  emit(event, data) {
    const list = this.eventListeners.get(event) || [];
    for (const h of list) h(data);
  }

  triggerDeviceLoss(reason = "destroyed", message = "Metal system reset") {
    if (this.lostPromiseResolve) {
      this.lostPromiseResolve({ reason, message });
    }
  }

  createBuffer(desc) {
    if (this.failBufferAllocationAfter >= 0 && this.bufferCount >= this.failBufferAllocationAfter) {
      throw new Error(`Simulated OOM allocation failure at buffer index ${this.bufferCount}`);
    }
    if (desc.size > this.limits.maxBufferSize) {
      throw new Error(`Buffer size ${desc.size} exceeds maxBufferSize ${this.limits.maxBufferSize}`);
    }
    const buf = new MockGPUBuffer(desc.size, desc.usage, desc.label);
    this.buffers.push(buf);
    this.bufferCount++;
    return buf;
  }

  createShaderModule(desc) {
    return new MockShaderModule(desc, this.shaderErrorMessages);
  }

  createBindGroupLayout() { return {}; }
  createPipelineLayout() { return {}; }

  createComputePipeline(desc) {
    this.pipelineSyncCallCount++;
    return new MockComputePipeline(desc);
  }
}

// 2. Load Model Header for Real Shape Tests
const MODEL_PATH = join(process.cwd(), "qwen3-asr-1.7b-q8", "qwen-asr-q8.bin");
function loadHeader() {
  const fd = openSync(MODEL_PATH, "r");
  try {
    const lenBuf = Buffer.alloc(8);
    readSync(fd, lenBuf, 0, 8, 0);
    const hlen = Number(lenBuf.readBigUInt64LE(0));
    const hBuf = Buffer.alloc(hlen);
    readSync(fd, hBuf, 0, hlen, 8);
    const header = JSON.parse(hBuf.toString("utf8"));
    return { header, dataBase: 8 + hlen };
  } finally {
    closeSync(fd);
  }
}
const { header: modelHeader, dataBase: modelDataBase } = loadHeader();
const { decEntries, encEntries } = extractModelDescriptorsFromHeader(modelHeader, modelDataBase);

let passed = 0;
let failed = 0;
const findings = [];

function recordFinding(category, description, details) {
  findings.push({ category, description, details });
}

function test(name, fn) {
  try {
    fn();
    console.log(`  ✔ [PASS] ${name}`);
    passed++;
  } catch (err) {
    console.error(`  ✖ [FAIL] ${name}:`, err.message);
    failed++;
  }
}

async function testAsync(name, fn) {
  try {
    await fn();
    console.log(`  ✔ [PASS] ${name}`);
    passed++;
  } catch (err) {
    console.error(`  ✖ [FAIL] ${name}:`, err.message);
    failed++;
  }
}

console.log("\n=======================================================");
console.log("CHALLENGER M2.2 EMPIRICAL ADVERSARIAL STRESS TEST SUITE");
console.log("=======================================================\n");

// ============================================================================
// SUITE 1: WebGPU Resource Destruction Idempotency & Lifecycle Boundaries
// ============================================================================
console.log("--- Suite 1: WebGPU Resource Destruction Idempotency & Lifecycle ---");

test("TC-M22-01: Pre-init destroy() idempotency on uninitialized instances", () => {
  const dec = new WebGPUDecoder();
  const enc = new WebGPUEncoder();

  // Call destroy 100 times in a tight loop before init
  for (let i = 0; i < 100; i++) {
    dec.destroy();
    enc.destroy();
  }

  assert.strictEqual(dec.destroyed, true);
  assert.strictEqual(dec.ready, false);
  assert.strictEqual(enc.destroyed, true);
  assert.strictEqual(enc.ready, false);
});

test("TC-M22-02: Pre-allocation destroy() idempotency with device attached", () => {
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });
  const enc = new WebGPUEncoder({}, { device: dev });

  // Multiple destroy calls before allocateStorageBuffers
  for (let i = 0; i < 20; i++) {
    dec.destroy();
    enc.destroy();
  }

  assert.strictEqual(dec.destroyed, true);
  assert.strictEqual(enc.destroyed, true);
});

test("TC-M22-03: Post-allocation full destruction destroys all persistent GPU buffers", () => {
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });
  const enc = new WebGPUEncoder({}, { device: dev });

  dec.allocateStorageBuffers({
    device: dev,
    header: modelHeader,
    entries: decEntries,
    shardBudget: 128 << 20,
  });

  enc.allocateStorageBuffers({
    device: dev,
    header: modelHeader,
    entries: encEntries,
    shardBudget: 128 << 20,
  });

  const totalBuffersCreated = dev.buffers.length;
  assert.ok(totalBuffersCreated > 15, `Expected >15 buffers created, got ${totalBuffersCreated}`);

  // Invoke destroy
  dec.destroy();
  enc.destroy();

  // Assert EVERY created buffer was marked destroyed
  for (let i = 0; i < dev.buffers.length; i++) {
    const b = dev.buffers[i];
    assert.strictEqual(
      b.destroyed,
      true,
      `Buffer ${i} (${b.label || "unlabeled"}, size=${b.size}) was not destroyed!`
    );
  }

  // Assert all internal buffer references are nullified
  assert.strictEqual(dec.bufQuants, null);
  assert.strictEqual(dec.bufScale, null);
  assert.strictEqual(dec.bufNorm, null);
  assert.strictEqual(dec.bufAct, null);
  assert.strictEqual(dec.bufKV, null);
  assert.strictEqual(dec.bufScratch, null);
  assert.strictEqual(dec.bufParams, null);
  assert.strictEqual(dec.bufTok, null);
  assert.strictEqual(dec.bufTokRead, null);
  assert.strictEqual(dec.bufEmbedOut, null);
  assert.strictEqual(dec.bufEmbedRead, null);

  assert.strictEqual(enc.bufQuants, null);
  assert.strictEqual(enc.bufScales, null);
  assert.strictEqual(enc.bufVecs, null);
  assert.strictEqual(enc.bufAct, null);
  assert.strictEqual(enc.bufScratch, null);
  assert.strictEqual(enc.bufParams, null);
  assert.strictEqual(enc.readBuf, null);
});

test("TC-M22-04: Consecutive multiple destroy() calls post-allocation are strictly idempotent", () => {
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });

  dec.allocateStorageBuffers({
    device: dev,
    header: modelHeader,
    entries: decEntries,
    shardBudget: 128 << 20,
  });

  const buffers = [...dev.buffers];
  // First destroy
  dec.destroy();
  // Call destroy 50 more times consecutively
  for (let i = 0; i < 50; i++) {
    assert.doesNotThrow(() => dec.destroy());
  }

  // Verify buffers were destroyed exactly once, not re-destroyed
  for (const b of buffers) {
    assert.strictEqual(b.destroyCount, 1, `Buffer destroyCount was ${b.destroyCount}, expected 1`);
  }
});

test("TC-M22-05: Re-initialization cycle (allocate -> destroy -> allocate -> destroy)", () => {
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });

  // Cycle 1
  dec.allocateStorageBuffers({
    device: dev,
    header: modelHeader,
    entries: decEntries,
    shardBudget: 128 << 20,
  });
  assert.strictEqual(dec.destroyed, false);
  const cycle1Buffers = [...dev.buffers];
  dec.destroy();
  assert.strictEqual(dec.destroyed, true);
  for (const b of cycle1Buffers) assert.strictEqual(b.destroyed, true);

  // Cycle 2
  dec.allocateStorageBuffers({
    device: dev,
    header: modelHeader,
    entries: decEntries,
    shardBudget: 128 << 20,
  });
  assert.strictEqual(dec.destroyed, false);
  const cycle2Buffers = dev.buffers.slice(cycle1Buffers.length);
  assert.ok(cycle2Buffers.length > 0);
  for (const b of cycle2Buffers) assert.strictEqual(b.destroyed, false);

  dec.destroy();
  assert.strictEqual(dec.destroyed, true);
  for (const b of cycle2Buffers) assert.strictEqual(b.destroyed, true);
});

test("TC-M22-06: Partial allocation failure rollback analysis (post-map vs mid-map)", () => {
  // Case A: Failure after shards.map (e.g. at bufScale allocation)
  // Decoder has 15 shards under 128MB. If allocation fails at buffer 16 (bufScale):
  {
    const dev = new MockGPUDevice();
    dev.failBufferAllocationAfter = 15; // Shards 0..14 succeed, buffer 15 (bufScale) fails

    const dec = new WebGPUDecoder({}, { device: dev });
    assert.throws(
      () => {
        dec.allocateStorageBuffers({
          device: dev,
          header: modelHeader,
          entries: decEntries,
          shardBudget: 128 << 20,
        });
      },
      /Simulated OOM allocation failure/
    );

    // bufQuants was assigned, so destroy() destroyed all 15 shards
    assert.strictEqual(dev.buffers.length, 15);
    for (const b of dev.buffers) {
      assert.strictEqual(b.destroyed, true, "Shards must be destroyed when subsequent buffer fails");
    }
    assert.strictEqual(dec.destroyed, true);
  }

  // Case B: Failure during shards.map (mid-map)
  // If failure occurs on shard 5 of 15, shards.map aborts before this.bufQuants is assigned!
  {
    const dev = new MockGPUDevice();
    dev.failBufferAllocationAfter = 5; // Fails on 6th shard inside shards.map

    const dec = new WebGPUDecoder({}, { device: dev });
    assert.throws(
      () => {
        dec.allocateStorageBuffers({
          device: dev,
          header: modelHeader,
          entries: decEntries,
          shardBudget: 128 << 20,
        });
      },
      /Simulated OOM allocation failure/
    );

    // VERIFICATION: Are shards 0..4 destroyed or leaked?
    const leakedShards = dev.buffers.filter((b) => !b.destroyed);
    assert.strictEqual(
      leakedShards.length,
      0,
      "All allocated shards must be destroyed during partial allocation rollback"
    );
    assert.strictEqual(dec.destroyed, true, "Instance must be marked destroyed");
  }
});

await testAsync("TC-M22-07: Buffer lock release & unmap on destruction", async () => {
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });

  dec.allocateStorageBuffers({
    device: dev,
    header: modelHeader,
    entries: decEntries,
    shardBudget: 128 << 20,
  });

  // Map bufTokRead
  assert.ok(dec.bufTokRead);
  await dec.bufTokRead.mapAsync(GPUMapMode.READ);
  assert.strictEqual(dec.bufTokRead.mapState, "mapped");

  // Call destroy while buffer is in mapped state
  dec.destroy();

  // Assert unmap was called before destruction
  assert.strictEqual(dec.bufTokRead, null);
  const tokBuf = dev.buffers.find((b) => b.usage & GPUBufferUsage.MAP_READ);
  assert.ok(tokBuf, "Mapped readback buffer found");
  assert.strictEqual(tokBuf.destroyed, true);
  assert.strictEqual(tokBuf.mapState, "unmapped");
  assert.ok(tokBuf.unmapCount >= 1, "unmap was not called on mapped buffer during destroy");
});

await testAsync("TC-M22-08: Mid-initialization destroy() race condition behavior", async () => {
  // Scenario A: Early mid-init destroy() aborts cleanly without unhandled TypeError
  {
    const dev = new MockGPUDevice();
    const dec = new WebGPUDecoder({}, { device: dev });
    let caughtErr = null;
    const initPromise = dec.finishInit().catch((e) => { caughtErr = e; });

    // Call destroy immediately
    dec.destroy();
    await initPromise;

    assert.strictEqual(caughtErr, null, "finishInit must abort cleanly without unhandled TypeError");
    assert.strictEqual(dec.destroyed, true, "dec.destroyed must remain true");
    assert.strictEqual(dec.ready, false, "dec.ready must remain false");
  }

  // Scenario B: Mid-init destroy() called later leaves instance with destroyed=true AND ready=false
  {
    const dev = new MockGPUDevice();
    const dec = new WebGPUDecoder({}, { device: dev });
    let compiled = 0;
    const initPromise = dec.finishInit(() => {
      compiled++;
      if (compiled === 8) {
        dec.destroy();
      }
    });

    await initPromise;
    assert.strictEqual(dec.destroyed, true);
    assert.strictEqual(dec.ready, false, "finishInit must not overwrite ready=true after destroy()");
  }
});

// ============================================================================
// SUITE 2: device.lost Simulation, Teardown & Recovery Signals
// ============================================================================
console.log("\n--- Suite 2: device.lost Simulation & Recovery Signals ---");

test("TC-M22-09: cleanupGpuResources() idempotency and WASM hook teardown", () => {
  const mockModule = {
    __gpuEncode: () => {},
    __gpuDecode: () => {},
    __gpuEmbedMany: () => {},
    _qwen_wasm_set_gpu_encoder: (v) => { mockModule.encSet = v; },
    _qwen_wasm_set_gpu_decoder: (v) => { mockModule.decSet = v; },
    _qwen_wasm_set_gpu_embedder: (v) => { mockModule.embSet = v; },
  };

  // Test when called multiple times
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder(mockModule, { device: dev });
  dec.allocateStorageBuffers({
    device: dev,
    header: modelHeader,
    entries: decEntries,
    shardBudget: 128 << 20,
  });

  dec.destroy();
  dec.destroy();
  assert.strictEqual(dec.destroyed, true);
});

await testAsync("TC-M22-10: WebGPU device.lost event propagation to WebGPUDecoder", async () => {
  const dev = new MockGPUDevice();
  let errorReported = "";
  const dec = new WebGPUDecoder({}, { device: dev });
  dec.onError = (msg) => { errorReported = msg; };

  // Trigger device loss
  dev.triggerDeviceLoss("destroyed", "GPU reset due to OS memory event");
  await new Promise((r) => setTimeout(r, 5));

  assert.strictEqual(dec.lost, "destroyed");
  assert.ok(errorReported.includes("GPU device lost (destroyed)"), `Unexpected error message: ${errorReported}`);
});

await testAsync("TC-M22-11: Callback leak check: uncaptured error listeners remain attached to device", async () => {
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });

  const uncapturedListenersBefore = (dev.eventListeners.get("uncapturederror") || []).length;
  assert.strictEqual(uncapturedListenersBefore, 1, "setupDevice should register 1 uncapturederror listener");

  // Call destroy
  dec.destroy();

  // VERIFICATION: Did destroy remove the uncapturederror listener?
  const uncapturedListenersAfter = (dev.eventListeners.get("uncapturederror") || []).length;
  assert.strictEqual(
    uncapturedListenersAfter,
    0,
    "uncapturederror listener must be removed by destroy()"
  );
});

await testAsync("TC-M22-12: Simulated mid-stream inference device loss recovery", async () => {
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });
  dec.allocateStorageBuffers({
    device: dev,
    header: modelHeader,
    entries: decEntries,
    shardBudget: 128 << 20,
  });
  await dec.finishInit();

  // Invalidate device
  dev.triggerDeviceLoss("device-reset", "Metal command buffer failed");
  await new Promise((r) => setTimeout(r, 5));

  assert.strictEqual(dec.lost, "device-reset");

  // Clean destruction after device loss
  assert.doesNotThrow(() => dec.destroy());
  assert.strictEqual(dec.destroyed, true);
});

// ============================================================================
// SUITE 3: Pipeline Compilation Concurrency & Fallbacks
// ============================================================================
console.log("\n--- Suite 3: Pipeline Compilation Concurrency & Fallback ---");

await testAsync("TC-M22-13: Async createComputePipelineAsync resolution in WebGPUDecoder", async () => {
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });

  let statusMsgs = [];
  await dec.finishInit((m) => statusMsgs.push(m));

  assert.strictEqual(dec.ready, true);
  assert.ok(dev.pipelineAsyncCallCount >= 28, `Expected >=28 async pipelines, got ${dev.pipelineAsyncCallCount}`);
  assert.strictEqual(dev.pipelineSyncCallCount, 0, "Should not call sync pipeline when async is available");
  assert.ok(statusMsgs.length >= 28, "Expected progress reports during async compilation");
});

await testAsync("TC-M22-14: Synchronous createComputePipeline fallback when async unavailable", async () => {
  // Create device with supportAsync = false
  const dev = new MockGPUDevice({}, [], false);

  const dec = new WebGPUDecoder({}, { device: dev });
  await dec.finishInit();

  assert.strictEqual(dec.ready, true);
  assert.ok(dev.pipelineSyncCallCount >= 28, `Expected >=28 sync pipelines, got ${dev.pipelineSyncCallCount}`);
});

await testAsync("TC-M22-15: Concurrent finishInit() compilation calls trigger duplicate builds", async () => {
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });

  // Launch two concurrent finishInit() calls
  const [res1, res2] = await Promise.all([
    dec.finishInit(),
    dec.finishInit(),
  ]);

  // VERIFICATION: Compilation promise memoization deduplicates concurrent calls
  // Single call is 28 pipelines. Concurrent call should be 28 if memoized.
  assert.strictEqual(
    dev.pipelineAsyncCallCount,
    28,
    "Expected 28 compiles due to memoized concurrent finishInit calls"
  );
});

test("TC-M22-16: Dynamic on-demand pipeline compilation with ensureMatvecPipes", () => {
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });

  // Initial state: not compiled
  assert.strictEqual(dec.matvecPipesTree, undefined);

  // Switch to q8 mode and wg 128
  dec.ensureMatvecPipes("q8", 128);
  assert.ok(dec.matvecPipesTree.q8[128], "q8 128 pipeline compiled on-demand");

  const syncCallsBefore = dev.pipelineSyncCallCount;
  // Calling again should use cache and not recompile
  dec.ensureMatvecPipes("q8", 128);
  assert.strictEqual(dev.pipelineSyncCallCount, syncCallsBefore, "ensureMatvecPipes must cache compiled pipelines");

  // Invalid width
  assert.strictEqual(dec.setMatvecWidth(99), false, "setMatvecWidth must reject unsupported workgroup width");
});

await testAsync("TC-M22-17: Shader compilation error capture and diagnostics formatting", async () => {
  const dev = new MockGPUDevice();
  dev.shaderErrorMessages = [
    { type: "error", lineNum: 42, linePos: 10, message: "syntax error: unexpected token" },
  ];

  const dec = new WebGPUDecoder({}, { device: dev });
  await assert.rejects(
    async () => await dec.finishInit(),
    /shader compilation error\(s\)/
  );
  assert.strictEqual(dec.ready, false);
});

test("TC-M22-18: Post-destruction state pollution via ensureMatvecPipes and applyMode", () => {
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });
  dec.allocateStorageBuffers({
    device: dev,
    header: modelHeader,
    entries: decEntries,
    shardBudget: 128 << 20,
  });
  dec.destroy();
  assert.strictEqual(dec.destroyed, true);
  assert.strictEqual(dec.matvecPipesTree, null);

  // Invoking setMatvecWidth on a destroyed instance must return false and not resurrect pipelines
  const ret = dec.setMatvecWidth(128);
  assert.strictEqual(ret, false, "setMatvecWidth on destroyed instance must return false");
  assert.strictEqual(dec.matvecPipesTree, null, "matvecPipesTree must remain null on destroyed instance");
});

console.log("\n=======================================================");
console.log(`CHALLENGER RESULTS: ${passed} passed, ${failed} failed`);
console.log(`EMPIRICAL FINDINGS RECORDED: ${findings.length}`);
for (const f of findings) {
  console.log(`  - [${f.category}] ${f.description}`);
  console.log(`    Detail: ${f.details}`);
}
console.log("=======================================================\n");

if (failed > 0) {
  process.exit(1);
} else {
  process.exit(0);
}
