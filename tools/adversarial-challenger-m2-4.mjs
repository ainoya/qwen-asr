/**
 * tools/adversarial-challenger-m2-4.mjs
 * Comprehensive Empirical Challenger M2.4 Verification & Stress Harness
 *
 * Focus:
 * 1. Mid-init destroy() at every single shader compilation stage for Decoder AND Encoder.
 * 2. High-concurrency finishInit() (10+ parallel calls) with and without mid-flight destroy().
 * 3. Granular allocation rollback: failure injection at every single buffer allocation index (0..N).
 * 4. Rapid repeated lifecycle cycles (allocate -> finishInit -> destroy) x50.
 * 5. Device loss during active compilation and inference.
 * 6. Post-destruction method call resistance and state invariance.
 */

import assert from "node:assert/strict";
import { openSync, readSync, closeSync } from "node:fs";
import { join } from "node:path";

// Setup WebGPU mock environment
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

// Import implementations
const { WebGPUDecoder } = await import("../wasm/demo/webgpu-decoder.js");
const { WebGPUEncoder } = await import("../wasm/demo/webgpu-encoder.js");
const app = await import("../wasm/demo/app.js");
const { extractModelDescriptorsFromHeader } = app;

class MockGPUBuffer {
  constructor(size, usage, label = "") {
    this.size = size;
    this.usage = usage;
    this.label = label;
    this.destroyed = false;
    this.destroyCount = 0;
    this.mapState = "unmapped";
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
    this.onPipelineCompile = null;

    if (supportAsync) {
      this.createComputePipelineAsync = async (desc) => {
        this.pipelineAsyncCallCount++;
        if (this.onPipelineCompile) await this.onPipelineCompile(desc.label, this.pipelineAsyncCallCount);
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
    if (this.onPipelineCompile) this.onPipelineCompile(desc.label, this.pipelineSyncCallCount);
    return new MockComputePipeline(desc);
  }
}

// Load real model header
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
const failureDetails = [];

function test(name, fn) {
  try {
    fn();
    console.log(`  ✔ [PASS] ${name}`);
    passed++;
  } catch (err) {
    console.error(`  ✖ [FAIL] ${name}:`, err.message);
    failed++;
    failureDetails.push({ name, error: err });
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
    failureDetails.push({ name, error: err });
  }
}

console.log("\n=======================================================");
console.log("CHALLENGER M2.4: DEEP ADVERSARIAL STRESS SUITE");
console.log("=======================================================\n");

// ---------------------------------------------------------------------
// TEST GROUP 1: Exhaustive Allocation Rollback Matrix
// ---------------------------------------------------------------------
console.log("--- Group 1: Exhaustive Allocation Rollback Matrix ---");

// Determine total buffer count for Decoder
{
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });
  dec.allocateStorageBuffers({
    device: dev,
    header: modelHeader,
    entries: decEntries,
    shardBudget: 128 << 20,
  });
  const decoderTotalBuffers = dev.buffers.length;
  dec.destroy();

  // Now test every single failure point from 0 to decoderTotalBuffers - 1
  for (let failIdx = 0; failIdx < decoderTotalBuffers; failIdx++) {
    test(`TC-M24-DEC-ROLLBACK-${failIdx}: Decoder allocation failure at buffer ${failIdx}/${decoderTotalBuffers}`, () => {
      const devTest = new MockGPUDevice();
      devTest.failBufferAllocationAfter = failIdx;
      const decTest = new WebGPUDecoder({}, { device: devTest });

      assert.throws(() => {
        decTest.allocateStorageBuffers({
          device: devTest,
          header: modelHeader,
          entries: decEntries,
          shardBudget: 128 << 20,
        });
      }, /Simulated OOM allocation failure/);

      assert.strictEqual(devTest.buffers.length, failIdx, `Expected ${failIdx} buffers allocated before fail`);
      const leaked = devTest.buffers.filter(b => !b.destroyed);
      assert.strictEqual(leaked.length, 0, `Leaked ${leaked.length} buffers when failing at idx ${failIdx}`);
      assert.strictEqual(decTest.destroyed, true, "Instance must be marked destroyed on rollback");
      assert.strictEqual(decTest.ready, false, "Instance ready must be false");
    });
  }
}

// Determine total buffer count for Encoder
{
  const dev = new MockGPUDevice();
  const enc = new WebGPUEncoder({}, { device: dev });
  enc.allocateStorageBuffers({
    device: dev,
    header: modelHeader,
    entries: encEntries,
    shardBudget: 128 << 20,
  });
  const encoderTotalBuffers = dev.buffers.length;
  enc.destroy();

  // Test every single failure point from 0 to encoderTotalBuffers - 1
  for (let failIdx = 0; failIdx < encoderTotalBuffers; failIdx++) {
    test(`TC-M24-ENC-ROLLBACK-${failIdx}: Encoder allocation failure at buffer ${failIdx}/${encoderTotalBuffers}`, () => {
      const devTest = new MockGPUDevice();
      devTest.failBufferAllocationAfter = failIdx;
      const encTest = new WebGPUEncoder({}, { device: devTest });

      assert.throws(() => {
        encTest.allocateStorageBuffers({
          device: devTest,
          header: modelHeader,
          entries: encEntries,
          shardBudget: 128 << 20,
        });
      }, /Simulated OOM allocation failure/);

      assert.strictEqual(devTest.buffers.length, failIdx, `Expected ${failIdx} buffers allocated before fail`);
      const leaked = devTest.buffers.filter(b => !b.destroyed);
      assert.strictEqual(leaked.length, 0, `Leaked ${leaked.length} buffers when failing at idx ${failIdx}`);
      assert.strictEqual(encTest.destroyed, true, "Instance must be marked destroyed on rollback");
      assert.strictEqual(encTest.ready, false, "Instance ready must be false");
    });
  }
}

// ---------------------------------------------------------------------
// TEST GROUP 2: Mid-Compilation Destroy() at Every Step
// ---------------------------------------------------------------------
console.log("\n--- Group 2: Mid-Compilation Destroy() at Every Stage ---");

// Decoder compiles 28 shaders in standard configuration without subgroups.
// Test destroying at every single step from 1 to 28
for (let step = 1; step <= 28; step++) {
  await testAsync(`TC-M24-DEC-MID-DESTROY-STEP-${step}: Decoder destroy() triggered during shader compile #${step}`, async () => {
    const dev = new MockGPUDevice();
    const dec = new WebGPUDecoder({}, { device: dev });

    let compiledSoFar = 0;
    dev.onPipelineCompile = (name, count) => {
      compiledSoFar = count;
      if (compiledSoFar === step) {
        dec.destroy();
      }
    };

    let errorCaught = null;
    try {
      await dec.finishInit();
    } catch (e) {
      errorCaught = e;
    }

    assert.strictEqual(errorCaught, null, `finishInit() must not reject when destroyed mid-compile at step ${step}`);
    assert.strictEqual(dec.destroyed, true, "dec.destroyed must remain true");
    assert.strictEqual(dec.ready, false, "dec.ready must remain false");
    assert.strictEqual(dec.pipe, null, "dec.pipe must remain null");
  });
}

// Encoder compiles shaders in finishInit -> buildPipelines
// Test destroying at every single step from 1 to 6
for (let step = 1; step <= 6; step++) {
  await testAsync(`TC-M24-ENC-MID-DESTROY-STEP-${step}: Encoder destroy() triggered during shader compile #${step}`, async () => {
    const dev = new MockGPUDevice();
    const enc = new WebGPUEncoder({}, { device: dev });
    enc.allocateStorageBuffers({
      device: dev,
      header: modelHeader,
      entries: encEntries,
      shardBudget: 128 << 20,
    });

    let compiledSoFar = 0;
    dev.onPipelineCompile = (name, count) => {
      compiledSoFar = count;
      if (compiledSoFar === step) {
        enc.destroy();
      }
    };

    let errorCaught = null;
    try {
      await enc.finishInit();
    } catch (e) {
      errorCaught = e;
    }

    assert.strictEqual(errorCaught, null, `finishInit() must not reject when destroyed mid-compile at step ${step}`);
    assert.strictEqual(enc.destroyed, true, "enc.destroyed must remain true");
    assert.strictEqual(enc.ready, false, "enc.ready must remain false");
    assert.ok(!enc.pipeLN, "enc.pipeLN must not be populated");
  });
}

// ---------------------------------------------------------------------
// TEST GROUP 3: High-Concurrency Stress (10+ parallel calls)
// ---------------------------------------------------------------------
console.log("\n--- Group 3: High-Concurrency Stress ---");

await testAsync("TC-M24-CONCURRENCY-01: 10 parallel dec.finishInit() calls without destruction", async () => {
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });

  const callers = Array.from({ length: 10 }, () => dec.finishInit());
  const results = await Promise.all(callers);

  assert.strictEqual(dec.ready, true);
  assert.strictEqual(!!dec.destroyed, false);
  // Total pipeline async calls must be exactly 28 (memoized, not 280)
  assert.strictEqual(dev.pipelineAsyncCallCount, 28, "Expected exactly 28 async pipeline calls across 10 callers");
  for (const r of results) {
    assert.strictEqual(r, undefined);
  }
});

await testAsync("TC-M24-CONCURRENCY-02: 10 parallel enc.finishInit() calls without destruction", async () => {
  const dev = new MockGPUDevice();
  const enc = new WebGPUEncoder({}, { device: dev });
  enc.allocateStorageBuffers({
    device: dev,
    header: modelHeader,
    entries: encEntries,
    shardBudget: 128 << 20,
  });

  const callers = Array.from({ length: 10 }, () => enc.finishInit());
  const results = await Promise.all(callers);

  assert.strictEqual(enc.ready, true);
  assert.strictEqual(!!enc.destroyed, false);
  // Total sync pipeline calls must be 6 (or 7), not 60
  assert.strictEqual(dev.pipelineSyncCallCount, 6, "Expected exactly 6 pipeline calls across 10 callers");
});

await testAsync("TC-M24-CONCURRENCY-03: 10 parallel dec.finishInit() calls interrupted by destroy() mid-flight", async () => {
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });

  dev.onPipelineCompile = (name, count) => {
    if (count === 14) {
      dec.destroy();
    }
  };

  const callers = Array.from({ length: 10 }, () => dec.finishInit());
  const results = await Promise.all(callers);

  assert.strictEqual(dec.ready, false);
  assert.strictEqual(dec.destroyed, true);
  assert.strictEqual(dec.pipe, null);
});

await testAsync("TC-M24-CONCURRENCY-04: Staggered finishInit() callers joining an in-flight compilation", async () => {
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });

  const p1 = dec.finishInit();
  await new Promise((r) => setTimeout(r, 5));
  const p2 = dec.finishInit();
  await new Promise((r) => setTimeout(r, 5));
  const p3 = dec.finishInit();

  await Promise.all([p1, p2, p3]);
  assert.strictEqual(dec.ready, true);
  assert.strictEqual(dev.pipelineAsyncCallCount, 28, "Staggered callers must join single in-flight build");
});

// ---------------------------------------------------------------------
// TEST GROUP 4: Rapid Repeated Cycles & Listener Leak Check
// ---------------------------------------------------------------------
console.log("\n--- Group 4: Rapid Repeated Cycles & Listener Leaks ---");

await testAsync("TC-M24-LIFECYCLE-50CYCLES: 50 repeated cycles of allocate -> finishInit -> destroy", async () => {
  const dev = new MockGPUDevice();

  for (let cycle = 0; cycle < 50; cycle++) {
    const dec = new WebGPUDecoder({}, { device: dev });
    dec.allocateStorageBuffers({
      device: dev,
      header: modelHeader,
      entries: decEntries,
      shardBudget: 128 << 20,
    });
    await dec.finishInit();
    assert.strictEqual(dec.ready, true);
    assert.strictEqual(dec.destroyed, false);

    dec.destroy();
    assert.strictEqual(dec.ready, false);
    assert.strictEqual(dec.destroyed, true);

    // Assert that the uncapturederror listener was cleanly removed each time
    const listeners = dev.eventListeners.get("uncapturederror") || [];
    assert.strictEqual(listeners.length, 0, `Uncaptured error listener leaked on cycle ${cycle}!`);
  }
});

await testAsync("TC-M24-LIFECYCLE-ENC-50CYCLES: 50 repeated encoder cycles of allocate -> finishInit -> destroy", async () => {
  const dev = new MockGPUDevice();

  for (let cycle = 0; cycle < 50; cycle++) {
    const enc = new WebGPUEncoder({}, { device: dev });
    enc.allocateStorageBuffers({
      device: dev,
      header: modelHeader,
      entries: encEntries,
      shardBudget: 128 << 20,
    });
    await enc.finishInit();
    assert.strictEqual(enc.ready, true);
    assert.strictEqual(enc.destroyed, false);

    enc.destroy();
    assert.strictEqual(enc.ready, false);
    assert.strictEqual(enc.destroyed, true);

    const listeners = dev.eventListeners.get("uncapturederror") || [];
    assert.strictEqual(listeners.length, 0, `Uncaptured error listener leaked on encoder cycle ${cycle}!`);
  }
});

// ---------------------------------------------------------------------
// TEST GROUP 5: Post-destruction Method Guards & State Invariance
// ---------------------------------------------------------------------
console.log("\n--- Group 5: Post-Destruction Method Guards ---");

test("TC-M24-GUARD-01: Decoder configuration methods on destroyed instance do not mutate state", () => {
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });
  dec.destroy();

  assert.strictEqual(dec.setMatvecWidth(128), false);
  assert.strictEqual(dec.setQuantizeActivations(true), undefined);
  assert.strictEqual(dec.setSubgroups(true), false);
  assert.strictEqual(dec.applyMode("q8", 128), undefined);
  dec.ensureMatvecPipes("q8", 128);

  assert.strictEqual(dec.matvecPipesTree, null);
  assert.strictEqual(dec.destroyed, true);
  assert.strictEqual(dec.ready, false);
});

await testAsync("TC-M24-GUARD-02: finishInit() on already-destroyed instance returns immediately", async () => {
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });
  dec.destroy();

  await dec.finishInit();
  assert.strictEqual(dev.pipelineAsyncCallCount, 0, "finishInit on destroyed instance must compile 0 pipelines");
  assert.strictEqual(dec.ready, false);
  assert.strictEqual(dec.destroyed, true);
});

await testAsync("TC-M24-GUARD-03: enc.finishInit() on already-destroyed encoder returns immediately", async () => {
  const dev = new MockGPUDevice();
  const enc = new WebGPUEncoder({}, { device: dev });
  enc.destroy();

  await enc.finishInit();
  assert.strictEqual(dev.pipelineSyncCallCount, 0, "finishInit on destroyed encoder must compile 0 pipelines");
  assert.strictEqual(enc.ready, false);
  assert.strictEqual(enc.destroyed, true);
});

// ---------------------------------------------------------------------
// TEST GROUP 6: Device Loss During Active Init
// ---------------------------------------------------------------------
console.log("\n--- Group 6: Device Loss During Active Compilation ---");

await testAsync("TC-M24-DEV-LOSS-MID-INIT: Device loss triggered midway through finishInit()", async () => {
  const dev = new MockGPUDevice();
  const dec = new WebGPUDecoder({}, { device: dev });

  let errorReported = "";
  dec.onError = (msg) => { errorReported = msg; };

  dev.onPipelineCompile = (name, count) => {
    if (count === 10) {
      dev.triggerDeviceLoss("device-reset", "GPU crash");
    }
  };

  await dec.finishInit();
  // Allow lost promise callback to fire
  await new Promise((r) => setTimeout(r, 10));

  assert.strictEqual(dec.lost, "device-reset");
  assert.ok(errorReported.includes("device-reset"));

  // Calling destroy after device loss must succeed cleanly
  assert.doesNotThrow(() => dec.destroy());
  assert.strictEqual(dec.destroyed, true);
});

console.log("\n=======================================================");
console.log(`CHALLENGER M2.4 RESULTS: ${passed} passed, ${failed} failed`);
if (failed > 0) {
  console.log("FAILURES:");
  for (const f of failureDetails) {
    console.log(`  - ${f.name}: ${f.error.message}`);
    console.log(`    Stack: ${f.error.stack}`);
  }
}
console.log("=======================================================\n");

process.exit(failed > 0 ? 1 : 0);
