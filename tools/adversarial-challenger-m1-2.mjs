/**
 * tools/adversarial-challenger-m1-2.mjs
 * Empirical Challenger M1.2 Verification Harness
 *
 * Tests:
 * 1. Stream truncation, malformed headers, and premature closure in parseSafetensorsHeaderStream and streamChunksToTargets.
 * 2. Bit-exact norm offsets in prepareWasmReducedImage verified via C engine.
 * 3. Dynamic sharding at 128 MiB and 256 MiB in WebGPUDecoder and WebGPUEncoder.
 */

import assert from "node:assert/strict";
import { readFileSync, writeFileSync, unlinkSync, existsSync, openSync, readSync, closeSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { join } from "node:path";

// 0. Setup mock globals for WebGPU & DOM
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

globalThis.document = {
  getElementById: (id) => ({
    id,
    classList: { add: () => {}, remove: () => {} },
    appendChild: () => {},
    addEventListener: () => {},
    style: {},
    disabled: false,
    textContent: "",
  }),
  createElement: () => ({
    appendChild: () => {},
    classList: { add: () => {}, remove: () => {} },
    style: {},
    textContent: "",
  }),
};

// Import implementations
const app = await import("../wasm/demo/app.js");
const {
  parseSafetensorsHeaderStream,
  prepareWasmReducedImage,
  extractModelDescriptorsFromHeader,
  buildIntervalDispatchTable,
  IntervalDispatcher,
  streamChunksToTargets,
} = app;

const { WebGPUDecoder } = await import("../wasm/demo/webgpu-decoder.js");
const { WebGPUEncoder } = await import("../wasm/demo/webgpu-encoder.js");

const MODEL_PATH = join(process.cwd(), "qwen3-asr-1.7b-q8", "qwen-asr-q8.bin");

function createMockReader(chunks, failAtChunk = -1, failError = new Error("Network connection reset")) {
  let idx = 0;
  return {
    async read() {
      if (idx === failAtChunk) {
        throw failError;
      }
      if (idx >= chunks.length) {
        return { done: true, value: undefined };
      }
      return { done: false, value: chunks[idx++] };
    }
  };
}

class MockGPUQueue {
  constructor() {
    this.writes = [];
    this.syncCount = 0;
  }
  writeBuffer(buffer, offset, data) {
    if (buffer.destroyed) throw new Error("writeBuffer to destroyed buffer");
    if (offset % 4 !== 0) throw new Error(`writeBuffer offset ${offset} is not 4-byte aligned`);
    if (data.byteLength % 4 !== 0) throw new Error(`writeBuffer size ${data.byteLength} is not 4-byte aligned`);
    if (offset + data.byteLength > buffer.size) {
      throw new Error(`writeBuffer overflow: ${offset} + ${data.byteLength} > ${buffer.size}`);
    }
    const copy = new Uint8Array(data.byteLength);
    copy.set(data);
    this.writes.push({ buffer, offset, data: copy });
    buffer.data.set(data, offset);
  }
  async onSubmittedWorkDone() {
    this.syncCount++;
    return Promise.resolve();
  }
}

class MockGPUBuffer {
  constructor(size, usage, label = "") {
    this.size = size;
    this.usage = usage;
    this.label = label;
    this.destroyed = false;
    this.data = new Uint8Array(size);
  }
  destroy() {
    this.destroyed = true;
  }
}

class MockGPUDevice {
  constructor(limits = {}) {
    this.limits = {
      maxBufferSize: limits.maxBufferSize ?? (1 << 30),
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize ?? (256 << 20),
      ...limits
    };
    this.queue = new MockGPUQueue();
    this.buffers = [];
  }
  createBuffer(desc) {
    if (desc.size > this.limits.maxBufferSize) {
      throw new Error(`Buffer size ${desc.size} exceeds maxBufferSize ${this.limits.maxBufferSize}`);
    }
    const buf = new MockGPUBuffer(desc.size, desc.usage, desc.label);
    this.buffers.push(buf);
    return buf;
  }
}

let passed = 0;
let failed = 0;

function report(testName, fn) {
  try {
    fn();
    console.log(`  ✔ [PASS] ${testName}`);
    passed++;
  } catch (err) {
    console.error(`  ✖ [FAIL] ${testName}:`, err.message);
    failed++;
  }
}

async function reportAsync(testName, fn) {
  try {
    await fn();
    console.log(`  ✔ [PASS] ${testName}`);
    passed++;
  } catch (err) {
    console.error(`  ✖ [FAIL] ${testName}:`, err.message);
    failed++;
  }
}

console.log("\n=======================================================");
console.log("CHALLENGER M1.2 ADVERSARIAL STRESS TEST SUITE");
console.log("=======================================================\n");

// ============================================================================
// SUITE 1: Stream Truncation, Malformed Headers & Premature Connection Closure
// ============================================================================
console.log("--- Suite 1: Stream Truncation & Malformed Headers ---");

await reportAsync("TC-ADV-01: parseSafetensorsHeaderStream rejects empty 0-byte stream", async () => {
  const reader = createMockReader([]);
  await assert.rejects(
    async () => await parseSafetensorsHeaderStream(reader),
    /Stream closed before reading safetensors header length/
  );
});

await reportAsync("TC-ADV-02: parseSafetensorsHeaderStream rejects partial 4-byte stream", async () => {
  const reader = createMockReader([new Uint8Array([1, 2, 3, 4])]);
  await assert.rejects(
    async () => await parseSafetensorsHeaderStream(reader),
    /Stream closed before reading safetensors header length/
  );
});

await reportAsync("TC-ADV-03: parseSafetensorsHeaderStream rejects zero header length", async () => {
  const buf = new Uint8Array(8);
  const reader = createMockReader([buf]);
  await assert.rejects(
    async () => await parseSafetensorsHeaderStream(reader),
    /Invalid safetensors header length: 0/
  );
});

await reportAsync("TC-ADV-04: parseSafetensorsHeaderStream rejects header length > 50MB", async () => {
  const buf = new Uint8Array(8);
  new DataView(buf.buffer).setBigUint64(0, BigInt(51 * 1024 * 1024), true);
  const reader = createMockReader([buf]);
  await assert.rejects(
    async () => await parseSafetensorsHeaderStream(reader),
    /Invalid safetensors header length/
  );
});

await reportAsync("TC-ADV-05: parseSafetensorsHeaderStream rejects premature stream close during JSON header", async () => {
  const hlen = 1000;
  const buf = new Uint8Array(8);
  new DataView(buf.buffer).setBigUint64(0, BigInt(hlen), true);
  const reader = createMockReader([buf, new Uint8Array(100)]);
  await assert.rejects(
    async () => await parseSafetensorsHeaderStream(reader),
    /Stream closed while reading safetensors header \(108\/1008 bytes\)/
  );
});

await reportAsync("TC-ADV-06: parseSafetensorsHeaderStream rejects malformed JSON header syntax", async () => {
  const badJson = new TextEncoder().encode("{ \"foo\": bar ");
  const buf = new Uint8Array(8 + badJson.length);
  new DataView(buf.buffer).setBigUint64(0, BigInt(badJson.length), true);
  buf.set(badJson, 8);
  const reader = createMockReader([buf]);
  await assert.rejects(
    async () => await parseSafetensorsHeaderStream(reader),
    (err) => err instanceof SyntaxError
  );
});

await reportAsync("TC-ADV-07: parseSafetensorsHeaderStream correctly captures excess leftover bytes", async () => {
  const jsonStr = JSON.stringify({ "test.weight": { dtype: "F32", shape: [1], data_offsets: [0, 4] } });
  const jsonBytes = new TextEncoder().encode(jsonStr);
  const excess = new Uint8Array([0xde, 0xad, 0xbe, 0xef, 0x42, 0x42, 0x42, 0x42]);
  const totalBuf = new Uint8Array(8 + jsonBytes.length + excess.length);
  new DataView(totalBuf.buffer).setBigUint64(0, BigInt(jsonBytes.length), true);
  totalBuf.set(jsonBytes, 8);
  totalBuf.set(excess, 8 + jsonBytes.length);

  const reader = createMockReader([totalBuf]);
  const res = await parseSafetensorsHeaderStream(reader);
  assert.strictEqual(res.headerLen, jsonBytes.length);
  assert.strictEqual(res.dataBase, 8 + jsonBytes.length);
  assert.strictEqual(res.leftover.length, excess.length);
  assert.deepStrictEqual(Array.from(res.leftover), Array.from(excess));
});

await reportAsync("TC-ADV-08: parseSafetensorsHeaderStream handles 1-byte incremental stream delivery", async () => {
  const jsonStr = JSON.stringify({ "norm": { dtype: "F32", shape: [2], data_offsets: [0, 8] } });
  const jsonBytes = new TextEncoder().encode(jsonStr);
  const full = new Uint8Array(8 + jsonBytes.length);
  new DataView(full.buffer).setBigUint64(0, BigInt(jsonBytes.length), true);
  full.set(jsonBytes, 8);

  const oneByteChunks = [];
  for (let i = 0; i < full.length; i++) {
    oneByteChunks.push(full.subarray(i, i + 1));
  }
  const reader = createMockReader(oneByteChunks);
  const res = await parseSafetensorsHeaderStream(reader);
  assert.strictEqual(res.headerLen, jsonBytes.length);
  assert.deepStrictEqual(res.header, JSON.parse(jsonStr));
});

await reportAsync("TC-ADV-09: streamChunksToTargets cleanly rejects premature connection closure", async () => {
  const device = new MockGPUDevice();
  const intervals = [
    { start: 100, end: 300, target: { gpuBuffer: device.createBuffer({ size: 200, usage: 0 }), gpuOffset: 0 } }
  ];
  const dispatcher = new IntervalDispatcher(intervals, device, { HEAPU8: new Uint8Array(1024) });

  const reader = createMockReader([new Uint8Array(100)]);
  await assert.rejects(
    async () => await streamChunksToTargets({
      reader,
      dispatcher,
      initialFileOffset: 100,
      totalBytes: 300,
      device,
    }),
    /Model stream truncated: received 200 of 300 bytes/
  );
});

await reportAsync("TC-ADV-10: streamChunksToTargets rejects stream truncated with unaligned remainder", async () => {
  const device = new MockGPUDevice();
  const intervals = [
    { start: 100, end: 300, target: { gpuBuffer: device.createBuffer({ size: 200, usage: 0 }), gpuOffset: 0 } }
  ];
  const dispatcher = new IntervalDispatcher(intervals, device, { HEAPU8: new Uint8Array(1024) });

  const reader = createMockReader([new Uint8Array(99)]);
  await assert.rejects(
    async () => await streamChunksToTargets({
      reader,
      dispatcher,
      initialFileOffset: 100,
      totalBytes: 300,
      device,
    }),
    /Model stream truncated with 3 unaligned bytes/
  );
});

await reportAsync("TC-ADV-11: streamChunksToTargets propagates stream reader network error", async () => {
  const device = new MockGPUDevice();
  const intervals = [
    { start: 0, end: 1000, target: { gpuBuffer: device.createBuffer({ size: 1000, usage: 0 }), gpuOffset: 0 } }
  ];
  const dispatcher = new IntervalDispatcher(intervals, device, { HEAPU8: new Uint8Array(1024) });

  const networkErr = new Error("ECONNRESET: connection abruptly terminated by peer");
  const reader = createMockReader([new Uint8Array(100)], 1, networkErr);
  await assert.rejects(
    async () => await streamChunksToTargets({
      reader,
      dispatcher,
      initialFileOffset: 0,
      totalBytes: 1000,
      device,
    }),
    /ECONNRESET/
  );
});

await reportAsync("TC-ADV-12: streamChunksToTargets reconstructs bit-exact data across arbitrary prime chunk sizes", async () => {
  const device = new MockGPUDevice();
  const gpuBuf = device.createBuffer({ size: 1024, usage: 0 });
  const wasmHeap = new Uint8Array(2048);
  const Module = { HEAPU8: wasmHeap };

  const intervals = [
    { start: 0, end: 256, target: { gpuBuffer: gpuBuf, gpuOffset: 0 } },
    { start: 256, end: 512, target: { wasmPtr: 512 } },
    { start: 512, end: 1024, target: { gpuBuffer: gpuBuf, gpuOffset: 256, wasmPtr: 1024 } },
  ];
  const dispatcher = new IntervalDispatcher(intervals, device, Module);

  const groundTruth = new Uint8Array(1024);
  for (let i = 0; i < 1024; i++) groundTruth[i] = (i * 37 + 13) & 0xff;

  const primeSizes = [3, 7, 11, 13, 17, 23, 29];
  const pathologicalChunks = [];
  let pOff = 0;
  let pIdx = 0;
  while (pOff < 1024) {
    const sz = Math.min(primeSizes[pIdx % primeSizes.length], 1024 - pOff);
    pathologicalChunks.push(groundTruth.subarray(pOff, pOff + sz));
    pOff += sz;
    pIdx++;
  }

  const reader = createMockReader(pathologicalChunks);
  await streamChunksToTargets({
    reader,
    dispatcher,
    initialFileOffset: 0,
    totalBytes: 1024,
    device,
  });

  assert.deepStrictEqual(
    Array.from(gpuBuf.data.subarray(0, 256)),
    Array.from(groundTruth.subarray(0, 256)),
    "GPU target [0, 256) must be bit-exact"
  );

  assert.deepStrictEqual(
    Array.from(wasmHeap.subarray(512, 512 + 256)),
    Array.from(groundTruth.subarray(256, 512)),
    "WASM target [256, 512) must be bit-exact"
  );

  assert.deepStrictEqual(
    Array.from(gpuBuf.data.subarray(256, 256 + 512)),
    Array.from(groundTruth.subarray(512, 1024)),
    "Dual GPU target [512, 1024) must be bit-exact"
  );
  assert.deepStrictEqual(
    Array.from(wasmHeap.subarray(1024, 1024 + 512)),
    Array.from(groundTruth.subarray(512, 1024)),
    "Dual WASM target [512, 1024) must be bit-exact"
  );
});

// ============================================================================
// SUITE 2: Bit-Exact Norm Offsets & C Engine Safetensors Compatibility
// ============================================================================
console.log("\n--- Suite 2: Bit-Exact Norm Offsets & C Engine Compatibility ---");

assert.ok(existsSync(MODEL_PATH), `Model file not found at ${MODEL_PATH}`);

let modelHeader, modelDataBase;
{
  const fd = openSync(MODEL_PATH, "r");
  const lenBuf = Buffer.alloc(8);
  readSync(fd, lenBuf, 0, 8, 0);
  const hlen = Number(lenBuf.readBigUInt64LE(0));
  const hBuf = Buffer.alloc(hlen);
  readSync(fd, hBuf, 0, hlen, 8);
  closeSync(fd);
  modelHeader = JSON.parse(hBuf.toString("utf8"));
  modelDataBase = 8 + hlen;
}

const mockWasmHeap = new Uint8Array(10 * 1024 * 1024);
let allocPointer = 1024;
const mockModule = {
  HEAPU8: mockWasmHeap,
  HEAPF32: new Float32Array(mockWasmHeap.buffer),
  _qwen_wasm_alloc: (size) => {
    const ptr = allocPointer;
    allocPointer += size;
    return ptr;
  },
  _qwen_wasm_release: () => {},
  _qwen_wasm_rms_eps: () => 1e-6,
  _qwen_wasm_rope_theta: () => 1e6,
};

const reduced = prepareWasmReducedImage(modelHeader, mockModule);

// Populate sentinel float values into mockWasmHeap using wasmNormMap pointers
const f32View = new Float32Array(mockWasmHeap.buffer);
for (let l = 0; l < 28; l++) {
  const suffixes = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight"
  ];
  for (let k = 0; k < 4; k++) {
    const name = `thinker.model.layers.${l}.${suffixes[k]}`;
    const ptr = reduced.wasmNormMap.get(name);
    assert.ok(ptr != null, `Pointer for ${name} must exist in wasmNormMap`);
    const floatIdx = ptr / 4;
    f32View[floatIdx] = 1000.0 + l * 10.0 + k;
  }
}
const finalPtr = reduced.wasmNormMap.get("thinker.model.norm.weight");
assert.ok(finalPtr != null, "Pointer for final norm must exist");
f32View[finalPtr / 4] = 9999.0;

report("TC-ADV-13: prepareWasmReducedImage keeps exactly 113 tensors (28*4 layers + 1 final)", () => {
  assert.strictEqual(reduced.kept.length, 113);
  assert.strictEqual(Object.keys(reduced.newHeader).length, 113);

  for (let l = 0; l < 28; l++) {
    assert.ok(reduced.newHeader[`thinker.model.layers.${l}.input_layernorm.weight`], `Missing input_layernorm for layer ${l}`);
    assert.ok(reduced.newHeader[`thinker.model.layers.${l}.post_attention_layernorm.weight`], `Missing post_attention_layernorm for layer ${l}`);
    assert.ok(reduced.newHeader[`thinker.model.layers.${l}.self_attn.q_norm.weight`], `Missing q_norm for layer ${l}`);
    assert.ok(reduced.newHeader[`thinker.model.layers.${l}.self_attn.k_norm.weight`], `Missing k_norm for layer ${l}`);
  }
  assert.ok(reduced.newHeader["thinker.model.norm.weight"], "Missing final norm weight");
});

report("TC-ADV-14: prepareWasmReducedImage enforces strict 64-byte alignment on all relative data offsets", () => {
  for (const [name, t] of Object.entries(reduced.newHeader)) {
    assert.strictEqual(
      t.data_offsets[0] % 64,
      0,
      `Tensor ${name} data_offset start (${t.data_offsets[0]}) must be 64-byte aligned`
    );
  }
});

report("TC-ADV-15: WASM reduced image header size prefix matches padded JSON header length", () => {
  const view = new DataView(mockWasmHeap.buffer, reduced.reducedPtr, 8);
  const storedHlen = Number(view.getBigUint64(0, true));
  assert.strictEqual(storedHlen % 64, 0, "Header JSON must be padded to 64-byte boundary");

  const jsonStr = new TextDecoder().decode(mockWasmHeap.subarray(reduced.reducedPtr + 8, reduced.reducedPtr + 8 + storedHlen));
  const parsedHeader = JSON.parse(jsonStr);
  assert.strictEqual(Object.keys(parsedHeader).length, 113);
});

report("TC-ADV-16: wasmNormMap destination addresses are 4-byte/8-byte aligned and bit-exact", () => {
  assert.strictEqual(reduced.wasmNormMap.size, 113);
  for (const [name, ptr] of reduced.wasmNormMap.entries()) {
    assert.strictEqual(ptr % 4, 0, `Pointer for ${name} must be 4-byte aligned for float32`);
    assert.strictEqual(ptr % 8, 0, `Pointer for ${name} must be 8-byte aligned`);
    assert.strictEqual(reduced.newHeader[name].data_offsets[0] % 64, 0);
  }
});

report("TC-ADV-17: C engine safetensors_open_memory parses the reduced image binary with exact offsets & bit-exact floats", () => {
  const reducedImageBytes = mockWasmHeap.subarray(reduced.reducedPtr, reduced.reducedPtr + reduced.reducedLen);
  const testBinPath = join(process.cwd(), "tools", "test_reduced_image.bin");
  const testExePath = join(process.cwd(), "tools", "verify_reduced_c");

  assert.ok(existsSync(testExePath), "Compiled C verification binary tools/verify_reduced_c must exist");

  try {
    writeFileSync(testBinPath, reducedImageBytes);
    const res = spawnSync(testExePath, [testBinPath], { encoding: "utf8" });
    if (res.status !== 0) {
      console.error("C verification stdout:", res.stdout);
      console.error("C verification stderr:", res.stderr);
      throw new Error(`C verification failed with exit code ${res.status}`);
    }
    assert.match(res.stdout, /SUCCESS: Verified all 113 tensors: correct dtype \(F32\), present, 4\/8-byte aligned, and BIT-EXACT float values confirmed in C engine!/);
  } finally {
    if (existsSync(testBinPath)) unlinkSync(testBinPath);
  }
});

// ============================================================================
// SUITE 3: Dynamic WebGPU Sharding (128 MiB and 256 MiB SHARD_BUDGET)
// ============================================================================
console.log("\n--- Suite 3: Dynamic WebGPU Sharding (128MB vs 256MB) ---");

const { decEntries, encEntries } = extractModelDescriptorsFromHeader(modelHeader, modelDataBase);

function verifyDecoderSharding(shardBudget) {
  const device = new MockGPUDevice({ maxStorageBufferBindingSize: shardBudget });
  const decoder = new WebGPUDecoder(mockModule);
  decoder.setupDevice(device);

  decoder.allocateStorageBuffers({
    shardBudget,
    entries: decEntries,
    header: modelHeader,
  });

  // 1. Assert all shards are within budget
  for (let i = 0; i < decoder.shards.length; i++) {
    const sh = decoder.shards[i];
    assert.ok(
      sh.bytes <= shardBudget,
      `Shard ${i} size ${sh.bytes} exceeds SHARD_BUDGET ${shardBudget}`
    );
    assert.strictEqual(
      decoder.bufQuants[i].size,
      sh.bytes,
      `GPUBuffer ${i} size must match shard size`
    );
  }

  // 2. Assert embedding matrix slicing (kind 6)
  const embKey = "6:0";
  const emb = decoder.wmap.get(embKey);
  assert.ok(emb, "Embedding matrix must be present in wmap");
  assert.strictEqual(emb.rows, 151936);
  assert.strictEqual(emb.cols, 2048);
  const totalEmbBytes = 151936 * 2048; // 311,164,928 bytes

  if (totalEmbBytes > shardBudget) {
    assert.ok(Array.isArray(emb.pieces), "Embedding must be sliced into pieces");
    let accumulatedRows = 0;
    for (let pIdx = 0; pIdx < emb.pieces.length; pIdx++) {
      const p = emb.pieces[pIdx];
      assert.strictEqual(p.rowBase, accumulatedRows, `Piece ${pIdx} rowBase must equal accumulated rows`);
      assert.ok(p.nq <= shardBudget, `Piece ${pIdx} size ${p.nq} must be <= SHARD_BUDGET`);
      assert.strictEqual(p.nq, p.rowCount * 2048);

      // Verify no out-of-bounds in shard buffer
      const shardBytes = decoder.shards[p.shard].bytes;
      assert.ok(
        p.wordBase * 4 + p.nq <= shardBytes,
        `Piece ${pIdx} overflows shard ${p.shard}: ${p.wordBase * 4 + p.nq} > ${shardBytes}`
      );

      // Verify scaleBase within bufScale
      assert.ok(
        p.scaleBase * 4 + (p.nq / 64) * 4 <= decoder.bufScale.size,
        `Piece ${pIdx} scaleBase overflows bufScale`
      );

      accumulatedRows += p.rowCount;
    }
    assert.strictEqual(accumulatedRows, 151936, "All rows of embedding must be accounted for across pieces");
  }

  // 3. Assert all non-sliced matrices fit within shards
  for (const [k, w] of decoder.wmap.entries()) {
    if (!w.pieces) {
      const shBytes = decoder.shards[w.shard].bytes;
      assert.ok(
        w.wordBase * 4 + w.nq <= shBytes,
        `Matrix ${k} overflows shard ${w.shard}: ${w.wordBase * 4 + w.nq} > ${shBytes}`
      );
      assert.ok(
        w.scaleBase * 4 + (w.nq / 64) * 4 <= decoder.bufScale.size,
        `Matrix ${k} scaleBase overflows bufScale`
      );
    }
  }

  return decoder;
}

function verifyEncoderSharding(shardBudget) {
  const device = new MockGPUDevice({ maxStorageBufferBindingSize: shardBudget });
  const encoder = new WebGPUEncoder(mockModule);
  encoder.device = device;
  encoder.adapter = { limits: { maxStorageBufferBindingSize: shardBudget } };

  encoder.allocateStorageBuffers({
    shardBudget,
    entries: encEntries,
    header: modelHeader,
  });

  // Assert all shards within budget
  for (let i = 0; i < encoder.shards.length; i++) {
    const sh = encoder.shards[i];
    assert.ok(
      sh.bytes <= shardBudget,
      `Encoder shard ${i} size ${sh.bytes} exceeds SHARD_BUDGET ${shardBudget}`
    );
  }

  // Assert each encoder entry bounds
  for (const ent of encoder.entries) {
    if (ent.shard != null) {
      const shBytes = encoder.shards[ent.shard].bytes;
      const qBytes = ent.rows * ent.cols;
      assert.ok(
        ent.wordBase * 4 + qBytes <= shBytes,
        `Encoder matrix overflows shard ${ent.shard}`
      );
      assert.ok(
        ent.scaleBase * 4 + (ent.rows * (ent.cols / 64) * 4) <= encoder.bufScales.size,
        "Encoder matrix scaleBase overflows bufScales"
      );
    } else if (ent.vecBase != null) {
      assert.ok(
        ent.vecBase * 4 + ent.count * 4 <= encoder.bufVecs.size,
        "Encoder vector overflows bufVecs"
      );
    }
  }

  return encoder;
}

report("TC-ADV-18: Decoder allocateStorageBuffers under SHARD_BUDGET = 256 MiB (268,435,456 bytes)", () => {
  const dec256 = verifyDecoderSharding(256 << 20);
  console.log(`    -> Decoder 256MB: ${dec256.shardCount} shards allocated`);
  const emb = dec256.wmap.get("6:0");
  assert.strictEqual(emb.pieces.length, 2, "311 MB embedding matrix must split into exactly 2 pieces under 256MB");
  assert.strictEqual(emb.pieces[0].rowCount, 131072);
  assert.strictEqual(emb.pieces[0].nq, 268435456); // exact 256 MiB
  assert.strictEqual(emb.pieces[1].rowCount, 20864);
  assert.strictEqual(emb.pieces[1].nq, 42729472);
});

report("TC-ADV-19: Decoder allocateStorageBuffers under SHARD_BUDGET = 128 MiB (134,217,728 bytes)", () => {
  const dec128 = verifyDecoderSharding(128 << 20);
  console.log(`    -> Decoder 128MB: ${dec128.shardCount} shards allocated`);
  const emb = dec128.wmap.get("6:0");
  assert.strictEqual(emb.pieces.length, 3, "311 MB embedding matrix must split into exactly 3 pieces under 128MB");
  assert.strictEqual(emb.pieces[0].rowCount, 65536);
  assert.strictEqual(emb.pieces[0].nq, 134217728); // exact 128 MiB
  assert.strictEqual(emb.pieces[1].rowCount, 65536);
  assert.strictEqual(emb.pieces[1].nq, 134217728); // exact 128 MiB
  assert.strictEqual(emb.pieces[2].rowCount, 20864);
  assert.strictEqual(emb.pieces[2].nq, 42729472);
});

report("TC-ADV-20: Decoder allocateStorageBuffers stress under SHARD_BUDGET = 64 MiB (67,108,864 bytes)", () => {
  const dec64 = verifyDecoderSharding(64 << 20);
  console.log(`    -> Decoder 64MB: ${dec64.shardCount} shards allocated`);
  const emb = dec64.wmap.get("6:0");
  assert.strictEqual(emb.pieces.length, 5, "311 MB embedding matrix must split into exactly 5 pieces under 64MB");
  assert.strictEqual(emb.pieces[0].rowCount, 32768);
  assert.strictEqual(emb.pieces[0].nq, 67108864); // exact 64 MiB
});

report("TC-ADV-21: Encoder allocateStorageBuffers under SHARD_BUDGET = 256 MiB and 128 MiB", () => {
  const enc256 = verifyEncoderSharding(256 << 20);
  console.log(`    -> Encoder 256MB: ${enc256.shardCount} shards allocated`);
  const enc128 = verifyEncoderSharding(128 << 20);
  console.log(`    -> Encoder 128MB: ${enc128.shardCount} shards allocated`);
  assert.ok(enc128.shardCount >= enc256.shardCount);
});

report("TC-ADV-22: Interval dispatch table for 128MB and 256MB produces strictly monotonic, non-overlapping ranges", () => {
  for (const budget of [128 << 20, 256 << 20]) {
    const dec = verifyDecoderSharding(budget);
    const enc = verifyEncoderSharding(budget);
    const intervals = buildIntervalDispatchTable(modelHeader, modelDataBase, dec, enc, reduced.wasmNormMap);

    const emb = dec.wmap.get("6:0");
    const expectedIntervalCount = 995 + (emb.pieces ? emb.pieces.length - 1 : 0);
    assert.strictEqual(intervals.length, expectedIntervalCount, `Intervals count mismatch for budget ${budget}`);

    for (let i = 0; i < intervals.length; i++) {
      const inv = intervals[i];
      assert.ok(inv.start < inv.end, `Interval ${i} start ${inv.start} must be < end ${inv.end}`);
      if (i > 0) {
        const prev = intervals[i - 1];
        assert.ok(
          inv.start >= prev.end,
          `Interval ${i} [${inv.start}, ${inv.end}) overlaps with previous [${prev.start}, ${prev.end})`
        );
      }
    }
  }
});

console.log("\n=======================================================");
console.log(`CHALLENGER RESULTS: ${passed} passed, ${failed} failed`);
console.log("=======================================================\n");

if (failed > 0) {
  process.exit(1);
} else {
  process.exit(0);
}
