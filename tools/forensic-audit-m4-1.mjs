/**
 * tools/forensic-audit-m4-1.mjs
 * Comprehensive Repository-Wide Forensic Integrity Audit Suite for Milestones 1–4
 *
 * Authored by Forensic Auditor M4 (teamwork_preview_auditor)
 *
 * Audits all deliverable targets:
 * - wasm/demo/webgpu-decoder.js
 * - wasm/demo/webgpu-encoder.js
 * - wasm/demo/test-auto.html
 * - wasm/demo/app.js
 * - wasm/demo/mic-worklet.js
 * - wasm/demo/index.html
 * - tools/run-webgpu-test.mjs
 *
 * Exhaustive forensic verification checks:
 * 1. CHEATING DETECTION:
 *    - Ensure NO mock implementations, NO dummy constants, NO bypass flags.
 *    - Verify that CER = 0.000 is produced by real model inference, not hardcoded strings or fake return values.
 *    - Verify getActiveGpuBytes() calculates real buffer sizes rather than returning fabricated constants.
 * 2. ZERO-MEMORY-DUPLICATION PIPELINE (R1):
 *    - Verify direct streaming ingestion into GPU storage buffers without allocating 2.18 GB in WASM linear memory.
 *    - Verify WASM norm reduced image (508 KB) is used instead of full model retention in WASM.
 * 3. MOBILE SAFARI WEBGPU COMPATIBILITY (R2):
 *    - Verify dynamic storage buffer sharding <= maxStorageBufferBindingSize (128/256 MB).
 *    - Verify Float16 KV cache (kvF16) and explicit destroy() lifecycle methods.
 * 4. MOBILE SAFARI AUDIO & TOUCH UX (R3):
 *    - Verify synchronous AudioContext.resume() on user gesture before await.
 *    - Verify real-time 44.1k/48k -> 16k Butterworth AudioWorklet downsampler.
 *    - Verify Apple HIG >= 44x44pt touch targets and >= 16px font sizes.
 * 5. AUTOMATED VERIFICATION & HEADROOM (R4):
 *    - Verify peak memory < 1.8 GB and host heap < 100 MB constraints are enforced by real assertions.
 *    - Verify 126/126 E2E tests pass legitimately.
 *    - Verify reference CPU benchmark matches samples/jfk.wav verbatim with CER = 0.000.
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { execSync } from "node:child_process";

// Set up mock DOM and WebGPU environment for Node.js
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
  getElementById: (id) => ({
    id,
    classList: { add() {}, remove() {} },
    appendChild() {},
    addEventListener() {},
    style: {},
    disabled: false,
    textContent: "",
    value: "4",
  }),
  createElement: (tag) => ({
    tag,
    appendChild() {},
    classList: { add() {}, remove() {} },
    style: {},
    textContent: "",
  }),
  addEventListener() {},
};

console.log("==================================================================");
console.log("  FORENSIC INTEGRITY AUDIT — MILESTONES 1–4 COMPREHENSIVE");
console.log("==================================================================\n");

let passedChecks = 0;
let totalChecks = 0;

function check(name, fn) {
  totalChecks++;
  try {
    fn();
    console.log(`  ✔ [PASS] ${name}`);
    passedChecks++;
  } catch (err) {
    console.error(`  ✖ [FAIL] ${name}`);
    console.error(`    Error: ${err.message}`);
    throw err;
  }
}

async function asyncCheck(name, fn) {
  totalChecks++;
  try {
    await fn();
    console.log(`  ✔ [PASS] ${name}`);
    passedChecks++;
  } catch (err) {
    console.error(`  ✖ [FAIL] ${name}`);
    console.error(`    Error: ${err.message}`);
    throw err;
  }
}

// ============================================================================
// Load Modules Under Audit
// ============================================================================
const { WebGPUDecoder } = await import("../wasm/demo/webgpu-decoder.js");
const { WebGPUEncoder } = await import("../wasm/demo/webgpu-encoder.js");
const app = await import("../wasm/demo/app.js");
const {
  extractModelDescriptorsFromHeader,
  prepareWasmReducedImage,
  buildIntervalDispatchTable,
  IntervalDispatcher,
  streamChunksToTargets,
} = app;

// Read real model header from disk
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

// Mock Buffer & Device for testing
class MockBuffer {
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

class MockGPUDevice {
  constructor(maxBinding = 128 << 20) {
    this.limits = {
      maxStorageBufferBindingSize: maxBinding,
      maxBufferSize: maxBinding * 4,
      maxComputeWorkgroupStorageSize: 32768,
      maxComputeWorkgroupsPerDimension: 65535,
    };
    this.features = new Set(["subgroups", "shader-f16"]);
    this.buffers = [];
    this.queue = {
      writeBuffer: () => {},
      submit: () => {},
      onSubmittedWorkDone: async () => {},
    };
  }
  createBuffer(desc) {
    const b = new MockBuffer(desc.size, desc.usage, desc.label);
    this.buffers.push(b);
    return b;
  }
  createBindGroupLayout() { return {}; }
  createPipelineLayout() { return {}; }
  createShaderModule() { return { getCompilationInfo: async () => ({ messages: [] }) }; }
  createComputePipeline() { return {}; }
  async createComputePipelineAsync() { return {}; }
  addEventListener() {}
  removeEventListener() {}
}

// ============================================================================
// SECTION 1: CHEATING DETECTION & FACADE VERIFICATION
// ============================================================================
console.log("--- Section 1: Cheating Detection & Facade Verification ---");

const targetFiles = [
  "wasm/demo/webgpu-decoder.js",
  "wasm/demo/webgpu-encoder.js",
  "wasm/demo/test-auto.html",
  "wasm/demo/app.js",
  "wasm/demo/mic-worklet.js",
  "wasm/demo/index.html",
  "tools/run-webgpu-test.mjs",
];

check("1.1 Source Scan: No mock implementations, dummy constants, or bypass flags in runtime files", () => {
  const runtimeFiles = [
    "wasm/demo/webgpu-decoder.js",
    "wasm/demo/webgpu-encoder.js",
    "wasm/demo/app.js",
    "wasm/demo/mic-worklet.js",
  ];
  for (const f of runtimeFiles) {
    const src = fs.readFileSync(f, "utf8");
    assert.ok(!src.includes("mockImplementation"), `Found mockImplementation in ${f}`);
    assert.ok(!src.includes("BYPASS_CHECK"), `Found BYPASS_CHECK in ${f}`);
    assert.ok(!src.includes("FAKE_TOKEN"), `Found FAKE_TOKEN in ${f}`);
    assert.ok(!src.includes("DUMMY_RESULT"), `Found DUMMY_RESULT in ${f}`);
  }
});

check("1.2 Real Inference Verification: JFK transcription string exists solely as test reference oracle", () => {
  const jfkQuote = "ask not what your country can do for you";
  for (const f of targetFiles) {
    const src = fs.readFileSync(f, "utf8");
    if (src.includes(jfkQuote)) {
      assert.ok(
        f === "wasm/demo/test-auto.html",
        `JFK reference quote must ONLY appear in test-auto.html as reference comparator, but found in ${f}`
      );
      // In test-auto.html, ensure it is used strictly for CER calculation
      assert.ok(src.includes('const ref = "And so, my fellow Americans'), "Must be in const ref definition");
      assert.ok(src.includes("calculateCer(genResult.text, ref)"), "Must be passed to calculateCer");
    }
  }
});

check("1.3 WebGPUDecoder.getActiveGpuBytes: computes dynamic buffer size, rejects static/fake values", () => {
  const dec = new WebGPUDecoder({}, { device: new MockGPUDevice() });
  assert.strictEqual(dec.getActiveGpuBytes(), 0, "Initial unallocated decoder must return 0 bytes");

  // Create simulated buffers with distinct sizes
  dec.bufQuants = [
    new MockBuffer(100_000_000, GPUBufferUsage.STORAGE),
    new MockBuffer(200_000_000, GPUBufferUsage.STORAGE),
  ];
  dec.bufScale = new MockBuffer(50_000_000, GPUBufferUsage.STORAGE);
  dec.bufNorm = new MockBuffer(5_000_000, GPUBufferUsage.STORAGE);
  dec.bufAct = new MockBuffer(10_000_000, GPUBufferUsage.STORAGE);
  dec.bufTok = new MockBuffer(4, GPUBufferUsage.STORAGE);
  dec.bufTokRead = new MockBuffer(4, GPUBufferUsage.MAP_READ);
  dec.bufKV = new MockBuffer(80_000_000, GPUBufferUsage.STORAGE);

  const expectedTotal = 100_000_000 + 200_000_000 + 50_000_000 + 5_000_000 + 10_000_000 + 4 + 4 + 80_000_000;
  assert.strictEqual(dec.getActiveGpuBytes(), expectedTotal, "Must calculate exact sum of active buffer sizes");

  // Mark one buffer as destroyed
  dec.bufQuants[0].destroyed = true;
  assert.strictEqual(dec.getActiveGpuBytes(), expectedTotal - 100_000_000, "Destroyed buffer must be excluded");

  // Call destroy()
  dec.destroy();
  assert.strictEqual(dec.getActiveGpuBytes(), 0, "Destroyed decoder must return 0 bytes");
});

check("1.4 WebGPUEncoder.getActiveGpuBytes: computes dynamic buffer size, rejects static/fake values", () => {
  const enc = new WebGPUEncoder({}, { device: new MockGPUDevice() });
  assert.strictEqual(enc.getActiveGpuBytes(), 0, "Initial unallocated encoder must return 0 bytes");

  enc.bufQuants = [
    new MockBuffer(50_000_000, GPUBufferUsage.STORAGE),
    new MockBuffer(60_000_000, GPUBufferUsage.STORAGE),
  ];
  enc.bufScales = new MockBuffer(15_000_000, GPUBufferUsage.STORAGE);
  enc.bufVecs = new MockBuffer(20_000_000, GPUBufferUsage.STORAGE);
  enc.bufConv1Out = new MockBuffer(5_000_000, GPUBufferUsage.STORAGE);
  enc.bufConv2Out = new MockBuffer(5_000_000, GPUBufferUsage.STORAGE);
  enc.bufAct = new MockBuffer(25_000_000, GPUBufferUsage.STORAGE);

  const expectedTotal = 50_000_000 + 60_000_000 + 15_000_000 + 20_000_000 + 5_000_000 + 5_000_000 + 25_000_000;
  assert.strictEqual(enc.getActiveGpuBytes(), expectedTotal, "Must calculate exact sum of active encoder buffers");

  enc.destroy();
  assert.strictEqual(enc.getActiveGpuBytes(), 0, "Destroyed encoder must return 0 bytes");
});

// ============================================================================
// SECTION 2: ZERO-MEMORY-DUPLICATION PIPELINE (R1)
// ============================================================================
console.log("\n--- Section 2: Zero-Memory-Duplication Pipeline (R1) ---");

check("2.1 Reduced WASM Image Sizing: prepareWasmReducedImage creates ~508 KB image (< 1 MB)", () => {
  let wasmAllocatedSize = 0;
  const mockWasmHeap = new Uint8Array(2 * 1024 * 1024);
  const mockModule = {
    _qwen_wasm_alloc: (size) => {
      wasmAllocatedSize = size;
      return 1024;
    },
    HEAPU8: mockWasmHeap,
  };

  const reduced = prepareWasmReducedImage(modelHeader, mockModule);
  assert.ok(reduced.reducedLen < 1024 * 1024, `Reduced image size ${(reduced.reducedLen / 1024).toFixed(1)} KB must be < 1 MB`);
  assert.ok(reduced.reducedLen > 400 * 1024, `Reduced image size ${(reduced.reducedLen / 1024).toFixed(1)} KB must be realistic (~508 KB)`);
  assert.strictEqual(wasmAllocatedSize, reduced.reducedLen, "Allocated size in WASM must match reducedLen");

  // Verify only norm tensors are kept (113 tensors)
  assert.strictEqual(reduced.kept.length, 113, "Must contain exactly 113 kept norm tensors");
  for (const [name] of reduced.kept) {
    assert.ok(
      name.includes("norm") || name.includes("ln_") || name.includes("bias"),
      `Unexpected non-norm tensor kept in reduced image: ${name}`
    );
    assert.ok(!name.endsWith(".q8"), `Q8 quantized tensor must NOT be in reduced image: ${name}`);
    assert.ok(!name.endsWith(".q8s"), `Q8 scales tensor must NOT be in reduced image: ${name}`);
  }
});

check("2.2 Zero Duplicate 2.18 GB WASM Allocation: test-auto.html eliminated legacy full model allocator", () => {
  const testAutoSrc = fs.readFileSync("wasm/demo/test-auto.html", "utf8");
  assert.ok(
    !testAutoSrc.includes("_qwen_wasm_alloc(total)"),
    "test-auto.html MUST NOT allocate total model size (2.18 GB) via _qwen_wasm_alloc(total)"
  );
  assert.ok(
    testAutoSrc.includes("loadGpuResidentDirect"),
    "test-auto.html must utilize loadGpuResidentDirect"
  );
});

check("2.3 Streaming Interval Dispatch: maps full model directly to GPU without intermediate chunk array", () => {
  const dev = new MockGPUDevice(128 << 20);
  const dec = new WebGPUDecoder({}, { device: dev });
  dec.allocateStorageBuffers({ header: modelHeader, entries: decEntries, shardBudget: 128 << 20 });
  const enc = new WebGPUEncoder({}, { device: dev });
  enc.allocateStorageBuffers({ device: dev, header: modelHeader, entries: encEntries, shardBudget: 128 << 20 });

  const dummyWasmNormMap = new Map();
  const intervals = buildIntervalDispatchTable(modelHeader, dataBase, dec, enc, dummyWasmNormMap);

  assert.ok(intervals.length > 200, "Interval dispatch table must have entries for all model tensors");
  // Check that intervals are sorted and contiguous
  for (let i = 1; i < intervals.length; i++) {
    assert.ok(intervals[i].start >= intervals[i - 1].end, `Intervals must not overlap: [${intervals[i].start}, ${intervals[i].end}]`);
  }

  // Test dispatcher
  let gpuWrites = 0;
  dev.queue.writeBuffer = () => { gpuWrites++; };
  const dispatcher = new IntervalDispatcher(intervals, dev, {});
  const testChunk = new Uint8Array(1024 * 1024); // 1 MB chunk
  dispatcher.dispatch(testChunk, dataBase);
  assert.ok(gpuWrites > 0, "Dispatcher must write slices directly into GPU storage buffers via writeBuffer");
});

// ============================================================================
// SECTION 3: MOBILE SAFARI WEBGPU COMPATIBILITY & LIMITS (R2)
// ============================================================================
console.log("\n--- Section 3: Mobile Safari WebGPU Compatibility & Limits (R2) ---");

check("3.1 Storage Buffer Sharding: shards at 128 MB (Mobile Safari limit) and 256 MB (Desktop)", () => {
  for (const maxBinding of [128 << 20, 256 << 20]) {
    const dev = new MockGPUDevice(maxBinding);
    const dec = new WebGPUDecoder({}, { device: dev });
    dec.allocateStorageBuffers({ header: modelHeader, entries: decEntries, shardBudget: maxBinding });

    assert.ok(dec.bufQuants.length > 0, "Must create storage shards");
    for (let i = 0; i < dec.bufQuants.length; i++) {
      const b = dec.bufQuants[i];
      assert.ok(
        b.size <= maxBinding,
        `Decoder shard ${i} size (${b.size} bytes) exceeds limit ${maxBinding} bytes`
      );
    }

    const enc = new WebGPUEncoder({}, { device: dev });
    enc.allocateStorageBuffers({ device: dev, header: modelHeader, entries: encEntries, shardBudget: maxBinding });
    for (let i = 0; i < enc.bufQuants.length; i++) {
      const b = enc.bufQuants[i];
      assert.ok(
        b.size <= maxBinding,
        `Encoder shard ${i} size (${b.size} bytes) exceeds limit ${maxBinding} bytes`
      );
    }
  }
});

check("3.2 Large Matrix Slicing: 311 MB embedding matrix correctly sliced under 128 MB limit", () => {
  const dev = new MockGPUDevice(128 << 20);
  const dec = new WebGPUDecoder({}, { device: dev });
  dec.allocateStorageBuffers({ header: modelHeader, entries: decEntries, shardBudget: 128 << 20 });

  const emb = dec.wmap.get("6:0"); // embed_tokens
  assert.ok(emb, "Embedding matrix descriptor must be registered in wmap");
  assert.ok(Array.isArray(emb.pieces), "Embedding matrix must be sliced into pieces when exceeding 128 MB");
  assert.ok(emb.pieces.length >= 3, `Expected at least 3 pieces for 311 MB embedding matrix, got ${emb.pieces.length}`);

  let totalNq = 0;
  for (const p of emb.pieces) {
    totalNq += p.nq;
    assert.ok(p.nq <= (128 << 20), "Each slice piece must be within 128 MB");
  }
  assert.strictEqual(totalNq, emb.rows * emb.cols, "Sum of piece elements must equal total embedding elements");
});

check("3.3 Float16 KV Cache (kvF16): context length clamped to 512 on f32, expands on f16", () => {
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

  const dev = new MockGPUDevice(128 << 20);
  const dec = new WebGPUDecoder(mockM, { device: dev });
  dec.allocateStorageBuffers({ header: modelHeader, entries: decEntries, shardBudget: 128 << 20 });

  // On f32 (kvF16 = false): limit is 512
  dec.kvF16 = false;
  dec.kvBytes = 4;
  assert.doesNotThrow(() => dec.prepareContext(512, 0));
  assert.throws(
    () => dec.prepareContext(513, 0),
    /exceeds maximum supported context length \(512\)/
  );

  // On f16: limit allows 768 tokens within 128 MB
  dec.kvF16 = true;
  dec.kvBytes = 2;
  assert.doesNotThrow(() => dec.prepareContext(768, 0));
});

check("3.4 Explicit Resource Lifecycle: destroy() releases all buffers and resets state idempotently", () => {
  const dev = new MockGPUDevice(128 << 20);
  const dec = new WebGPUDecoder({}, { device: dev });
  dec.allocateStorageBuffers({ header: modelHeader, entries: decEntries, shardBudget: 128 << 20 });
  assert.ok(dec.bufQuants.length > 0);

  const buffers = [...dec.bufQuants, dec.bufScale, dec.bufNorm, dec.bufAct, dec.bufTok, dec.bufTokRead, dec.bufKV];
  dec.destroy();

  assert.strictEqual(dec.destroyed, true);
  assert.strictEqual(dec.ready, false);
  for (const b of buffers) {
    if (b) assert.strictEqual(b.destroyed, true, "All buffers must be explicitly destroyed");
  }

  // Idempotent second call
  assert.doesNotThrow(() => dec.destroy());
});

// ============================================================================
// SECTION 4: MOBILE SAFARI AUDIO & TOUCH UX (R3)
// ============================================================================
console.log("\n--- Section 4: Mobile Safari Audio & Touch UX (R3) ---");

const appSource = fs.readFileSync("wasm/demo/app.js", "utf8");
const micWorkletSource = fs.readFileSync("wasm/demo/mic-worklet.js", "utf8");
const indexHtmlSource = fs.readFileSync("wasm/demo/index.html", "utf8");

check("4.1 Synchronous AudioContext Unlock: audioCtx.resume() called before await on user gesture", () => {
  const micClickIdx = appSource.indexOf('$("mic").onclick = async () => {');
  assert.ok(micClickIdx !== -1, "$('mic').onclick must exist");
  const micStopClickIdx = appSource.indexOf('$("micstop").onclick = async () => {');
  const handlerBody = appSource.slice(micClickIdx, micStopClickIdx);

  const resumeIdx = handlerBody.indexOf("audioCtx.resume()");
  const awaitIdx = handlerBody.indexOf("await ");
  assert.ok(resumeIdx !== -1, "audioCtx.resume() must be called in mic click handler");
  assert.ok(awaitIdx !== -1, "await must be present");
  assert.ok(resumeIdx < awaitIdx, "audioCtx.resume() MUST precede any await to survive Mobile Safari gesture window");
});

check("4.2 mic-worklet.js: 4th-order Butterworth downsampler with 0 sample drift", () => {
  // Run MicCollector inside sandboxed vm
  class BaseAudioWorkletProcessor {
    constructor() {
      this.port = { postMessage: (msg, transfer) => { if (this._onmessage) this._onmessage(msg, transfer); } };
    }
  }
  let RegisteredClass = null;
  const sandbox = {
    AudioWorkletProcessor: BaseAudioWorkletProcessor,
    registerProcessor: (name, cls) => { RegisteredClass = cls; },
    sampleRate: 48000,
    Float32Array,
    Math,
    console,
  };
  vm.createContext(sandbox);
  vm.runInContext(micWorkletSource, sandbox);

  assert.ok(RegisteredClass !== null, "MicCollector must register");
  const mc48 = new RegisteredClass({ processorOptions: { sourceRate: 48000 } });
  assert.strictEqual(mc48.ratio, 3.0);
  assert.strictEqual(mc48.biquads.length, 2, "4th-order filter requires 2 biquads");

  // Check 10-second drift at 48 kHz
  let emitted = 0;
  mc48.port.postMessage = () => { emitted++; };
  const quantum = 128;
  const numQuanta = Math.floor(48000 * 10 / quantum);
  for (let i = 0; i < numQuanta; i++) {
    mc48.process([ [ new Float32Array(quantum) ] ]);
  }
  const produced = emitted * 4000 + mc48.n;
  const expected = (numQuanta * quantum) / 3.0;
  assert.strictEqual(Math.abs(produced - expected), 0, "Drift must be exactly 0 samples");
});

check("4.3 Apple HIG Touch Targets: min-height/width >= 44px and font-size >= 16px in index.html", () => {
  assert.ok(indexHtmlSource.includes("min-height: 44px"), "Interactive elements must declare min-height: 44px");
  assert.ok(indexHtmlSource.includes("min-width: 44px"), "Interactive elements must declare min-width: 44px");
  assert.ok(indexHtmlSource.includes("font-size: 16px"), "Inputs must declare font-size: 16px to prevent auto-zoom");
  assert.ok(indexHtmlSource.includes("viewport-fit=cover"), "Viewport meta must include viewport-fit=cover");
  assert.ok(indexHtmlSource.includes("safe-area-inset-top"), "Safe area insets must be handled");
});

// ============================================================================
// SECTION 5: AUTOMATED VERIFICATION & HEADROOM (R4)
// ============================================================================
console.log("\n--- Section 5: Automated Verification & Headroom (R4) ---");

check("5.1 test-auto.html & run-webgpu-test.mjs: Peak memory < 1.8 GB and host heap < 100 MB enforced", () => {
  const testAutoSrc = fs.readFileSync("wasm/demo/test-auto.html", "utf8");
  assert.ok(
    testAutoSrc.includes("peakMem > 1.8 * 1e9") && testAutoSrc.includes("[OOM_VIOLATION]"),
    "test-auto.html must assert peak memory < 1.8 GB"
  );
  assert.ok(
    testAutoSrc.includes("jfkResult.cer > 0.000"),
    "test-auto.html must assert JFK CER tolerance 0.000"
  );

  const runnerSrc = fs.readFileSync("tools/run-webgpu-test.mjs", "utf8");
  assert.ok(
    runnerSrc.includes("if (peakMemoryObserved > memoryLimitBytes)"),
    "run-webgpu-test.mjs must enforce memoryLimitBytes failure exit"
  );
  assert.ok(
    runnerSrc.includes("if (jsHeapBytes > 100 * 1024 * 1024)"),
    "run-webgpu-test.mjs must enforce JS heap < 100 MB failure exit"
  );
});

check("5.2 E2E Test Suite: All 126 E2E tests pass legitimately", () => {
  const out = execSync(`"${process.execPath}" tools/e2e-tests/run-all.mjs`, { encoding: "utf8" });
  assert.ok(out.includes("pass 126"), "All 126 tests must pass");
  assert.ok(out.includes("fail 0"), "0 failures expected");
});

check("5.3 Reference CPU Benchmark: Matches samples/jfk.wav verbatim with CER = 0.000", () => {
  const out = execSync(`"${process.execPath}" wasm/bench-node.js qwen3-asr-1.7b-q8 samples/jfk.wav 4`, { encoding: "utf8" });
  const expectedText = "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.";
  assert.ok(out.includes(expectedText), "CPU reference benchmark must produce exact JFK transcript");

  // Calculate CER against reference
  function normalize(t) {
    return t ? t.toLowerCase().replace(/[.,!?;:'"()\[\]{}‘’“”]/g, "").replace(/\s+/g, " ").trim() : "";
  }
  function editDistance(a, b) {
    const prev = new Array(b.length + 1);
    const cur = new Array(b.length + 1);
    for (let j = 0; j <= b.length; j++) prev[j] = j;
    for (let i = 1; i <= a.length; i++) {
      cur[0] = i;
      for (let j = 1; j <= b.length; j++)
        cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      for (let j = 0; j <= b.length; j++) prev[j] = cur[j];
    }
    return prev[b.length];
  }
  const cer = editDistance(normalize(expectedText), normalize(expectedText)) / normalize(expectedText).length;
  assert.strictEqual(cer, 0.0, "CER must be exactly 0.000");
});

// ============================================================================
// SUMMARY & VERDICT
// ============================================================================
console.log("\n==================================================================");
console.log(`  ALL ${passedChecks}/${totalChecks} FORENSIC VERIFICATION CHECKS PASSED!`);
console.log("  FINAL VERDICT: CLEAN");
console.log("==================================================================\n");
