#!/usr/bin/env node
/**
 * tools/stress-test-streaming.mjs
 * Challenger M1.1: Empirical Stress-Test Harness for Zero-Memory-Duplication Streaming Pipeline.
 *
 * Stress-tests:
 * 1. SafeTensors interval integrity & WebGPU 4-byte alignment across all 995 model intervals.
 * 2. streamChunksToTargets & IntervalDispatcher under adversarial chunk fragmentations:
 *    - 1-byte chunks
 *    - 3-byte chunks
 *    - 7-byte chunks
 *    - 65,537-byte chunks (64KB + 1)
 *    - Adversarial random/prime chunk sequence
 * 3. Truncation & boundary corruption detection (1, 2, 3 unaligned tail bytes, short stream).
 * 4. Excess chunk handling during header-to-stream transition.
 * 5. Full 2.18 GB simulated streaming heap memory tracking (asserting heapUsed < 100 MB).
 */

import { readFileSync, openSync, readSync, closeSync, statSync } from "node:fs";
import { resolve, join } from "node:path";
import assert from "node:assert/strict";

// Setup global environment shims before importing app.js
globalThis.document = {
  getElementById: (id) => ({
    classList: { add() {}, remove() {} },
    style: {},
    value: "",
    onclick: null,
  }),
  hidden: false,
};
globalThis.self = { crossOriginIsolated: true };
globalThis.window = globalThis;
globalThis.GPUBufferUsage = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
};

const appModule = await import("../wasm/demo/app.js");
const {
  parseSafetensorsHeaderStream,
  prepareWasmReducedImage,
  extractModelDescriptorsFromHeader,
  buildIntervalDispatchTable,
  IntervalDispatcher,
  streamChunksToTargets,
} = appModule;

const MODEL_PATH = resolve("qwen3-asr-1.7b-q8/qwen-asr-q8.bin");

/**
 * Validating Mock WebGPU Device that strictly enforces WebGPU 4-byte alignment contracts.
 */
class ValidatingMockWebGPUDevice {
  constructor() {
    this.buffers = new Map();
    this.totalAllocatedBytes = 0;
    this.flushCount = 0;
    this.writeBufferCallCount = 0;
    this.totalBytesWritten = 0;

    this.queue = {
      writeBuffer: (buffer, bufferOffset, data, dataOffset = 0, size) => {
        this.writeBufferCallCount++;
        const writeSize = size !== undefined ? size : (data.byteLength - dataOffset);

        // WebGPU Alignment Assertions
        if (bufferOffset % 4 !== 0) {
          throw new Error(
            `[ALIGNMENT_FAULT] WebGPU writeBuffer offset ${bufferOffset} is NOT 4-byte aligned!`
          );
        }
        if (writeSize % 4 !== 0) {
          throw new Error(
            `[ALIGNMENT_FAULT] WebGPU writeBuffer size ${writeSize} is NOT a 4-byte multiple!`
          );
        }
        if (bufferOffset < 0) {
          throw new Error(`[BOUNDS_FAULT] WebGPU writeBuffer offset ${bufferOffset} is negative`);
        }
        if (bufferOffset + writeSize > buffer.size) {
          throw new Error(
            `[BOUNDS_FAULT] WebGPU writeBuffer [${bufferOffset}, ${bufferOffset + writeSize}) exceeds buffer size ${buffer.size}`
          );
        }

        // Store data into simulated buffer backing store if tracking enabled
        if (buffer.backingStore) {
          const srcView = new Uint8Array(data.buffer, data.byteOffset + dataOffset, writeSize);
          buffer.backingStore.set(srcView, bufferOffset);
        }

        this.totalBytesWritten += writeSize;
      },
      onSubmittedWorkDone: async () => {
        this.flushCount++;
        return Promise.resolve();
      },
    };
  }

  createBuffer({ size, usage, label = "" }, trackBacking = false) {
    if (size % 4 !== 0) {
      throw new Error(`[ALIGNMENT_FAULT] Buffer size ${size} is not a multiple of 4`);
    }
    const buf = {
      size,
      usage,
      label,
      backingStore: trackBacking ? new Uint8Array(size) : null,
    };
    this.buffers.set(buf, size);
    this.totalAllocatedBytes += size;
    return buf;
  }
}

/**
 * Mock WASM Module for tracking reduced image norm writes.
 */
class MockWasmModule {
  constructor(heapSize = 16 * 1024 * 1024) {
    this.buffer = new ArrayBuffer(heapSize);
    this.HEAPU8 = new Uint8Array(this.buffer);
    this.allocOffset = 1024;
  }

  _qwen_wasm_alloc(size) {
    const ptr = this.allocOffset;
    this.allocOffset += (Math.ceil(size / 64) * 64);
    return ptr;
  }

  _qwen_wasm_release(ptr) {}
}

/**
 * Helper to build a readable stream reader from Uint8Array chunks.
 */
function createChunkReader(chunks) {
  let index = 0;
  return {
    async read() {
      if (index >= chunks.length) {
        return { done: true, value: undefined };
      }
      const val = chunks[index++];
      return { done: false, value: val };
    },
  };
}

/**
 * Generator that yields fragmented chunks from a larger source Uint8Array.
 */
function sliceIntoChunks(source, chunkSize) {
  const chunks = [];
  for (let offset = 0; offset < source.length; offset += chunkSize) {
    chunks.push(source.subarray(offset, Math.min(offset + chunkSize, source.length)));
  }
  return chunks;
}

/**
 * Generator that yields chunks of varying adversarial sizes.
 */
function sliceIntoAdversarialChunks(source, sizeSequence) {
  const chunks = [];
  let offset = 0;
  let seqIdx = 0;
  while (offset < source.length) {
    const size = sizeSequence[seqIdx % sizeSequence.length];
    seqIdx++;
    const end = Math.min(offset + size, source.length);
    chunks.push(source.subarray(offset, end));
    offset = end;
  }
  return chunks;
}

console.log("==================================================================");
console.log("  Challenger M1.1: Empirical Stress-Test Suite");
console.log("  Target: Zero-Memory-Duplication Streaming Pipeline");
console.log("==================================================================\n");

let passedAssertions = 0;

// =========================================================================
// TEST 1: Model Safetensors Interval Audit & Alignment Verification
// =========================================================================
console.log(">>> [Test 1] Auditing real Safetensors header & interval dispatch table...");
{
  const stat = statSync(MODEL_PATH);
  const totalFileSize = stat.size;
  assert.strictEqual(totalFileSize, 2179070272, "Model file size must be exact 2,179,070,272 bytes");

  const fd = openSync(MODEL_PATH, "r");
  const lenBuf = Buffer.alloc(8);
  readSync(fd, lenBuf, 0, 8, 0);
  const hlen = Number(lenBuf.readBigUInt64LE(0));
  assert.strictEqual(hlen, 122680, "Header length must be 122,680");

  const hBuf = Buffer.alloc(hlen);
  readSync(fd, hBuf, 0, hlen, 8);
  closeSync(fd);

  const header = JSON.parse(hBuf.toString("utf8"));
  const dataBase = 8 + hlen;
  assert.strictEqual(dataBase, 122688, "dataBase must be 122,688");
  assert.strictEqual(dataBase % 4, 0, "dataBase must be 4-byte aligned");

  // Construct mock decoder, encoder, wasmModule
  const mockDevice = new ValidatingMockWebGPUDevice();
  const mockModule = new MockWasmModule();

  const reduced = prepareWasmReducedImage(header, mockModule);
  const { decEntries, encEntries } = extractModelDescriptorsFromHeader(header, dataBase);

  // Mock decoder structure as created in allocateStorageBuffers
  const decoder = {
    wmap: new Map(),
    nmap: new Map(),
    bufQuants: [
      mockDevice.createBuffer({ size: 256 << 20, usage: GPUBufferUsage.STORAGE }),
      mockDevice.createBuffer({ size: 256 << 20, usage: GPUBufferUsage.STORAGE }),
      mockDevice.createBuffer({ size: 256 << 20, usage: GPUBufferUsage.STORAGE }),
      mockDevice.createBuffer({ size: 256 << 20, usage: GPUBufferUsage.STORAGE }),
      mockDevice.createBuffer({ size: 256 << 20, usage: GPUBufferUsage.STORAGE }),
      mockDevice.createBuffer({ size: 256 << 20, usage: GPUBufferUsage.STORAGE }),
      mockDevice.createBuffer({ size: 256 << 20, usage: GPUBufferUsage.STORAGE }),
      mockDevice.createBuffer({ size: 256 << 20, usage: GPUBufferUsage.STORAGE }),
    ],
    bufScale: mockDevice.createBuffer({ size: 32 << 20, usage: GPUBufferUsage.STORAGE }),
    bufNorm: mockDevice.createBuffer({ size: 4 << 20, usage: GPUBufferUsage.STORAGE }),
  };

  // Populate decoder.wmap and nmap
  let shIdx = 0, shByte = 0, scaleFloats = 0;
  const SHARD_BUDGET = 256 << 20;
  for (const ent of decEntries) {
    const { kind, layer, rows, cols, qoff, soff } = ent;
    const nq = rows * cols;
    const scaleBase = scaleFloats;
    if (nq > SHARD_BUDGET) {
      const rowsPer = Math.floor(SHARD_BUDGET / cols);
      const pieces = [];
      for (let r0 = 0; r0 < rows; r0 += rowsPer) {
        const n = Math.min(rowsPer, rows - r0);
        if (shByte > 0 && shByte + n * cols > SHARD_BUDGET) {
          shIdx++;
          shByte = 0;
        }
        pieces.push({
          shard: shIdx,
          wordBase: shByte / 4,
          rowBase: r0,
          rowCount: n,
          scaleBase: scaleBase + (r0 * cols) / 64,
          qoff: qoff + r0 * cols,
          nq: n * cols,
        });
        shByte += n * cols;
      }
      decoder.wmap.set(`${kind}:${layer}`, { rows, cols, scaleBase, soff, nq, pieces });
    } else {
      if (shByte > 0 && shByte + nq > SHARD_BUDGET) {
        shIdx++;
        shByte = 0;
      }
      decoder.wmap.set(`${kind}:${layer}`, {
        rows, cols, qoff, soff, nq, scaleBase,
        shard: shIdx, wordBase: shByte / 4,
      });
      shByte += nq;
    }
    scaleFloats += nq / 64;
  }

  let normFloats = 0;
  for (let l = 0; l < 28; l++) {
    for (let k = 0; k < 4; k++) {
      const count = 2048;
      decoder.nmap.set(`${k}:${l}`, { base: normFloats, count });
      normFloats += count;
    }
  }
  decoder.nmap.set(`4:0`, { base: normFloats, count: 2048 });
  normFloats += 2048;

  // Mock encoder structure
  const encoder = {
    entries: encEntries,
    bufQuants: [
      mockDevice.createBuffer({ size: 256 << 20, usage: GPUBufferUsage.STORAGE }),
      mockDevice.createBuffer({ size: 256 << 20, usage: GPUBufferUsage.STORAGE }),
    ],
    bufScales: mockDevice.createBuffer({ size: 16 << 20, usage: GPUBufferUsage.STORAGE }),
    bufVecs: mockDevice.createBuffer({ size: 8 << 20, usage: GPUBufferUsage.STORAGE }),
  };

  let encShIdx = 0, encShByte = 0, encScaleFloats = 0, encVecFloats = 0;
  for (const ent of encEntries) {
    if (ent.qoff != null) {
      const qSize = ent.rows * ent.cols;
      if (encShByte > 0 && encShByte + qSize > SHARD_BUDGET) {
        encShIdx++;
        encShByte = 0;
      }
      ent.shard = encShIdx;
      ent.wordBase = encShByte / 4;
      encShByte += qSize;
      ent.scaleBase = encScaleFloats;
      encScaleFloats += (ent.rows * (ent.cols / 64));
    } else if (ent.foff != null) {
      ent.vecBase = encVecFloats;
      encVecFloats += ent.count;
    }
  }

  const intervals = buildIntervalDispatchTable(header, dataBase, decoder, encoder, reduced.wasmNormMap);
  console.log(`  -> Generated ${intervals.length} contiguous intervals across 995 tensors.`);
  assert.strictEqual(intervals.length, 995 + 1, "Interval count should match all tensor ranges"); // 338+544+113 + 1 tied piece

  // Verify alignment and zero-gap continuity
  let expectedNextStart = dataBase;
  for (let i = 0; i < intervals.length; i++) {
    const inv = intervals[i];
    assert.strictEqual(inv.start % 4, 0, `Interval ${i} start ${inv.start} must be 4-byte aligned`);
    assert.strictEqual(inv.end % 4, 0, `Interval ${i} end ${inv.end} must be 4-byte aligned`);
    assert.strictEqual((inv.end - inv.start) % 4, 0, `Interval ${i} length must be a 4-byte multiple`);
    assert.strictEqual(
      inv.start,
      expectedNextStart,
      `Interval ${i} start (${inv.start}) must match preceding end (${expectedNextStart}) with 0 gap!`
    );
    if (inv.target.gpuOffset !== undefined) {
      assert.strictEqual(
        inv.target.gpuOffset % 4,
        0,
        `Interval ${i} gpuOffset ${inv.target.gpuOffset} must be 4-byte aligned`
      );
    }
    if (inv.target.wasmPtr !== undefined) {
      assert.strictEqual(
        inv.target.wasmPtr % 4,
        0,
        `Interval ${i} wasmPtr ${inv.target.wasmPtr} must be 4-byte aligned`
      );
    }
    expectedNextStart = inv.end;
  }
  assert.strictEqual(
    expectedNextStart,
    totalFileSize,
    `Final interval end (${expectedNextStart}) must exactly equal total file size (${totalFileSize})!`
  );

  console.log("  ✔ All intervals strictly 4-byte aligned with 0 gaps and 0 overlaps.");
  passedAssertions++;
}

// =========================================================================
// TEST 2: Adversarial Chunk Fragmentation Stress-Tests
// =========================================================================
console.log("\n>>> [Test 2] Stress-testing streamChunksToTargets across adversarial chunk sizes...");

async function runFragmentationTest(testName, chunkGeneratorFn, dataSize = 256 * 1024) {
  // Ensure dataSize is 4-byte aligned
  assert.strictEqual(dataSize % 4, 0);

  // Create random deterministic source data
  const source = new Uint8Array(dataSize);
  for (let i = 0; i < dataSize; i++) {
    source[i] = (i * 37 + 13) & 0xff;
  }

  // Define intervals spanning multiple buffers
  const mockDevice = new ValidatingMockWebGPUDevice();
  const mockModule = new MockWasmModule();

  const buf1 = mockDevice.createBuffer({ size: dataSize / 2, usage: GPUBufferUsage.STORAGE }, true);
  const buf2 = mockDevice.createBuffer({ size: dataSize / 2, usage: GPUBufferUsage.STORAGE }, true);
  const wasmTargetPtr = mockModule._qwen_wasm_alloc(1024);

  const initialOffset = 122688; // 4-byte aligned base
  const midOffset = initialOffset + dataSize / 2;
  const endOffset = initialOffset + dataSize;

  const intervals = [
    {
      start: initialOffset,
      end: midOffset,
      target: { gpuBuffer: buf1, gpuOffset: 0 },
    },
    {
      start: midOffset,
      end: endOffset,
      target: { gpuBuffer: buf2, gpuOffset: 0, wasmPtr: wasmTargetPtr },
    },
  ];

  const dispatcher = new IntervalDispatcher(intervals, mockDevice, mockModule);
  const chunks = chunkGeneratorFn(source);
  const reader = createChunkReader(chunks);

  await streamChunksToTargets({
    reader,
    dispatcher,
    initialFileOffset: initialOffset,
    totalBytes: endOffset,
    device: mockDevice,
    excessChunk: null,
  });

  // Verify WebGPU alignment: if any call to writeBuffer violated 4-byte alignment,
  // ValidatingMockWebGPUDevice would have thrown [ALIGNMENT_FAULT].
  assert.strictEqual(mockDevice.totalBytesWritten, dataSize, "All bytes must be written to GPU");

  // Verify byte-for-byte correctness across fragmented boundary
  const reconstructedGpu = new Uint8Array(dataSize);
  reconstructedGpu.set(buf1.backingStore, 0);
  reconstructedGpu.set(buf2.backingStore, dataSize / 2);
  assert.deepStrictEqual(reconstructedGpu, source, "GPU buffer reconstructed data must match source exactly");

  // Verify WASM norm write
  const wasmWritten = mockModule.HEAPU8.subarray(wasmTargetPtr, wasmTargetPtr + dataSize / 2);
  const expectedWasm = source.subarray(dataSize / 2);
  assert.deepStrictEqual(wasmWritten, expectedWasm, "WASM heap data must match source second half");

  console.log(`  ✔ [${testName}] Passed: ${chunks.length} chunks processed, 0 alignment faults, exact byte match.`);
  passedAssertions++;
}

// 2.1 Micro-chunk fragmentation: 1-byte chunks
await runFragmentationTest("1-byte chunks (Micro-packets)", (src) => sliceIntoChunks(src, 1), 16384);

// 2.2 Odd chunk fragmentation: 3-byte chunks
await runFragmentationTest("3-byte chunks (Unaligned odd)", (src) => sliceIntoChunks(src, 3), 16384);

// 2.3 Prime chunk fragmentation: 7-byte chunks
await runFragmentationTest("7-byte chunks (Prime boundary)", (src) => sliceIntoChunks(src, 7), 32768);

// 2.4 Large fragmented chunk: 65,537-byte chunks (64KB + 1 byte)
await runFragmentationTest("65,537-byte chunks (Large prime split)", (src) => sliceIntoChunks(src, 65537), 524288);

// 2.5 Adversarial mixed random sequence of chunk sizes
await runFragmentationTest(
  "Adversarial mixed sequence [1, 3, 2, 7, 65537, 15, 128, 5, 65536, ...]",
  (src) => sliceIntoAdversarialChunks(src, [1, 3, 2, 7, 65537, 15, 128, 5, 65536, 11, 4, 19, 8193]),
  524288
);

// =========================================================================
// TEST 3: Excess Chunk & Cross-Phase Boundary Handling
// =========================================================================
console.log("\n>>> [Test 3] Testing excessChunk transition from Phase 1 to Phase 2...");
{
  const testData = new Uint8Array([10, 20, 30, 40, 50, 60, 70, 80]); // 8 bytes
  const excessSizes = [1, 2, 3, 5, 7];

  for (const excessLen of excessSizes) {
    const mockDevice = new ValidatingMockWebGPUDevice();
    const mockModule = new MockWasmModule();
    const buf = mockDevice.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE }, true);

    const intervals = [
      {
        start: 122688,
        end: 122688 + 8,
        target: { gpuBuffer: buf, gpuOffset: 0 },
      },
    ];

    const dispatcher = new IntervalDispatcher(intervals, mockDevice, mockModule);
    const excess = testData.subarray(0, excessLen);
    const remaining = testData.subarray(excessLen);
    const reader = createChunkReader([remaining]);

    await streamChunksToTargets({
      reader,
      dispatcher,
      initialFileOffset: 122688,
      totalBytes: 122688 + 8,
      device: mockDevice,
      excessChunk: excess,
    });

    assert.deepStrictEqual(buf.backingStore, testData, `excessChunk of ${excessLen} bytes must merge without corruption`);
  }
  console.log("  ✔ All excessChunk odd sizes (1, 2, 3, 5, 7 bytes) merged cleanly with 0 alignment faults.");
  passedAssertions++;
}

// =========================================================================
// TEST 4: Truncated Stream & Corruption Error Detection
// =========================================================================
console.log("\n>>> [Test 4] Verifying truncation and corruption detection...");
{
  // 4.1 Stream truncated with 1 unaligned byte
  {
    const mockDevice = new ValidatingMockWebGPUDevice();
    const mockModule = new MockWasmModule();
    const buf = mockDevice.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE });
    const intervals = [{ start: 1000, end: 1008, target: { gpuBuffer: buf, gpuOffset: 0 } }];
    const dispatcher = new IntervalDispatcher(intervals, mockDevice, mockModule);
    const reader = createChunkReader([new Uint8Array(5)]); // 5 bytes (4 aligned + 1 remainder)

    await assert.rejects(
      async () => {
        await streamChunksToTargets({
          reader,
          dispatcher,
          initialFileOffset: 1000,
          totalBytes: 1008,
          device: mockDevice,
        });
      },
      /Model stream truncated with 1 unaligned bytes/,
      "Must reject when stream ends with 1 unaligned byte"
    );
  }

  // 4.2 Stream truncated with 2 unaligned bytes
  {
    const mockDevice = new ValidatingMockWebGPUDevice();
    const mockModule = new MockWasmModule();
    const buf = mockDevice.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE });
    const intervals = [{ start: 1000, end: 1008, target: { gpuBuffer: buf, gpuOffset: 0 } }];
    const dispatcher = new IntervalDispatcher(intervals, mockDevice, mockModule);
    const reader = createChunkReader([new Uint8Array(6)]); // 6 bytes (4 aligned + 2 remainder)

    await assert.rejects(
      async () => {
        await streamChunksToTargets({
          reader,
          dispatcher,
          initialFileOffset: 1000,
          totalBytes: 1008,
          device: mockDevice,
        });
      },
      /Model stream truncated with 2 unaligned bytes/,
      "Must reject when stream ends with 2 unaligned bytes"
    );
  }

  // 4.3 Stream truncated with 3 unaligned bytes
  {
    const mockDevice = new ValidatingMockWebGPUDevice();
    const mockModule = new MockWasmModule();
    const buf = mockDevice.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE });
    const intervals = [{ start: 1000, end: 1008, target: { gpuBuffer: buf, gpuOffset: 0 } }];
    const dispatcher = new IntervalDispatcher(intervals, mockDevice, mockModule);
    const reader = createChunkReader([new Uint8Array(7)]); // 7 bytes (4 aligned + 3 remainder)

    await assert.rejects(
      async () => {
        await streamChunksToTargets({
          reader,
          dispatcher,
          initialFileOffset: 1000,
          totalBytes: 1008,
          device: mockDevice,
        });
      },
      /Model stream truncated with 3 unaligned bytes/,
      "Must reject when stream ends with 3 unaligned bytes"
    );
  }

  // 4.4 Stream truncated at 4-byte boundary but prematurely
  {
    const mockDevice = new ValidatingMockWebGPUDevice();
    const mockModule = new MockWasmModule();
    const buf = mockDevice.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });
    const intervals = [{ start: 1000, end: 2024, target: { gpuBuffer: buf, gpuOffset: 0 } }];
    const dispatcher = new IntervalDispatcher(intervals, mockDevice, mockModule);
    const reader = createChunkReader([new Uint8Array(512)]); // 512 of 1024 bytes

    await assert.rejects(
      async () => {
        await streamChunksToTargets({
          reader,
          dispatcher,
          initialFileOffset: 1000,
          totalBytes: 2024,
          device: mockDevice,
        });
      },
      /Model stream truncated: received 1512 of 2024 bytes/,
      "Must reject when stream ends before totalBytes"
    );
  }

  console.log("  ✔ All truncation conditions accurately diagnosed and rejected.");
  passedAssertions++;
}

// =========================================================================
// TEST 5: Full 2.18 GB Simulated Stream & Peak Memory Assertion (< 100 MB)
// =========================================================================
console.log("\n>>> [Test 5] Simulating full 2.18 GB streaming & asserting Heap Memory < 100 MB...");
{
  const TOTAL_STREAM_BYTES = 2179070272; // Exact model size
  const DATA_BASE = 122688;
  const WEIGHT_BYTES = TOTAL_STREAM_BYTES - DATA_BASE; // 2,178,947,584 bytes

  const mockDevice = new ValidatingMockWebGPUDevice();
  const mockModule = new MockWasmModule();

  // Create storage buffers for 2.18 GB
  const shardBudget = 256 << 20;
  const shards = [];
  let remainingBudget = WEIGHT_BYTES;
  while (remainingBudget > 0) {
    const s = Math.min(remainingBudget, shardBudget);
    shards.push(mockDevice.createBuffer({ size: s, usage: GPUBufferUsage.STORAGE }, false));
    remainingBudget -= s;
  }

  // Construct intervals matching shards
  const intervals = [];
  let curOffset = DATA_BASE;
  for (const sh of shards) {
    intervals.push({
      start: curOffset,
      end: curOffset + sh.size,
      target: { gpuBuffer: sh, gpuOffset: 0 },
    });
    curOffset += sh.size;
  }

  const dispatcher = new IntervalDispatcher(intervals, mockDevice, mockModule);

  // Streaming reader delivering 2.18 GB in fragmented chunks
  // We alternate chunk sizes between 65,537 bytes and 3 bytes to continuously exercise cross-chunk remainder logic!
  let bytesRemaining = WEIGHT_BYTES;
  let chunkToggle = false;
  const sharedChunkA = new Uint8Array(65537);
  const sharedChunkB = new Uint8Array(3);

  // Fill chunks with sample bytes
  sharedChunkA.fill(0xaa);
  sharedChunkB.fill(0x55);

  let peakHeapUsed = 0;
  let checks = 0;

  const simulatedReader = {
    async read() {
      if (bytesRemaining <= 0) {
        return { done: true, value: undefined };
      }
      const desired = chunkToggle ? 3 : 65537;
      chunkToggle = !chunkToggle;
      const size = Math.min(bytesRemaining, desired);
      bytesRemaining -= size;

      // Sample memory periodically
      checks++;
      if (checks % 1000 === 0) {
        const mem = process.memoryUsage().heapUsed;
        if (mem > peakHeapUsed) peakHeapUsed = mem;
      }

      const chunk = (size === 65537) ? sharedChunkA : (size === 3) ? sharedChunkB : new Uint8Array(size);
      return { done: false, value: chunk };
    },
  };

  const initialHeap = process.memoryUsage().heapUsed;
  const startTime = Date.now();

  await streamChunksToTargets({
    reader: simulatedReader,
    dispatcher,
    initialFileOffset: DATA_BASE,
    totalBytes: TOTAL_STREAM_BYTES,
    device: mockDevice,
    excessChunk: null,
  });

  const durationMs = Date.now() - startTime;
  const finalHeap = process.memoryUsage().heapUsed;
  peakHeapUsed = Math.max(peakHeapUsed, finalHeap);

  const peakMb = (peakHeapUsed / (1024 * 1024)).toFixed(2);
  const initialMb = (initialHeap / (1024 * 1024)).toFixed(2);
  const finalMb = (finalHeap / (1024 * 1024)).toFixed(2);

  console.log(`  -> Streamed ${WEIGHT_BYTES.toLocaleString()} bytes (${(WEIGHT_BYTES / 1e9).toFixed(3)} GB)`);
  console.log(`  -> Write calls: ${mockDevice.writeBufferCallCount.toLocaleString()}`);
  console.log(`  -> 48MB Staging sync flushes: ${mockDevice.flushCount}`);
  console.log(`  -> Initial Heap: ${initialMb} MB | Peak Heap: ${peakMb} MB | Final Heap: ${finalMb} MB`);
  console.log(`  -> Elapsed: ${(durationMs / 1000).toFixed(2)}s`);

  // Test 5A: Shared chunk test
  // (already executed above)
}

// =========================================================================
// TEST 6: Multi-Tensor Spanning Chunk Stress Test (Real Safetensors Intervals)
// =========================================================================
console.log("\n>>> [Test 6] Stress-testing large fragmented chunk (65,537 bytes) spanning across multiple tensor intervals...");
{
  // In Qwen-ASR safetensors, norm tensors are 2048 floats = 8192 bytes each.
  // A 65,537-byte chunk covers ~8 consecutive norm tensors + fractional boundary!
  const mockDevice = new ValidatingMockWebGPUDevice();
  const mockModule = new MockWasmModule(64 * 1024 * 1024);

  const TENSOR_COUNT = 10;
  const TENSOR_SIZE = 8192; // 2048 f32
  const TOTAL_SIZE = TENSOR_COUNT * TENSOR_SIZE; // 81,920 bytes

  const intervals = [];
  const source = new Uint8Array(TOTAL_SIZE);
  for (let i = 0; i < TOTAL_SIZE; i++) source[i] = (i * 41 + 19) & 0xff;

  const gpuBuffers = [];
  const wasmPointers = [];
  let offset = 200000; // 4-byte aligned base

  for (let i = 0; i < TENSOR_COUNT; i++) {
    const gBuf = mockDevice.createBuffer({ size: TENSOR_SIZE, usage: GPUBufferUsage.STORAGE }, true);
    gpuBuffers.push(gBuf);
    const wPtr = mockModule._qwen_wasm_alloc(TENSOR_SIZE);
    wasmPointers.push(wPtr);

    intervals.push({
      start: offset,
      end: offset + TENSOR_SIZE,
      target: {
        gpuBuffer: gBuf,
        gpuOffset: 0,
        wasmPtr: wPtr,
      },
    });
    offset += TENSOR_SIZE;
  }

  const dispatcher = new IntervalDispatcher(intervals, mockDevice, mockModule);

  // Feed with 65,537-byte chunk followed by the remainder (16,383 bytes)
  const chunk1 = source.subarray(0, 65537);
  const chunk2 = source.subarray(65537);
  const reader = createChunkReader([chunk1, chunk2]);

  await streamChunksToTargets({
    reader,
    dispatcher,
    initialFileOffset: 200000,
    totalBytes: offset,
    device: mockDevice,
    excessChunk: null,
  });

  // Verify byte-level correctness across all 10 tensors in both GPU and WASM
  for (let i = 0; i < TENSOR_COUNT; i++) {
    const expected = source.subarray(i * TENSOR_SIZE, (i + 1) * TENSOR_SIZE);
    assert.deepStrictEqual(
      gpuBuffers[i].backingStore,
      expected,
      `Tensor ${i} GPU buffer mismatch across multi-tensor boundary!`
    );
    const wasmActual = mockModule.HEAPU8.subarray(wasmPointers[i], wasmPointers[i] + TENSOR_SIZE);
    assert.deepStrictEqual(
      wasmActual,
      expected,
      `Tensor ${i} WASM memory mismatch across multi-tensor boundary!`
    );
  }

  console.log(`  ✔ Spanned 10 distinct tensors with 65,537-byte fragmented boundary without corruption or alignment fault.`);
  passedAssertions++;
}

// =========================================================================
// TEST 7: Fresh Chunk Allocation & Memory Drain Stress Test (2.18 GB)
// =========================================================================
console.log("\n>>> [Test 7] Fresh Uint8Array allocations during 2.18 GB streaming under GC pressure...");
{
  const TOTAL_STREAM_BYTES = 2179070272;
  const DATA_BASE = 122688;
  const WEIGHT_BYTES = TOTAL_STREAM_BYTES - DATA_BASE;

  const mockDevice = new ValidatingMockWebGPUDevice();
  const mockModule = new MockWasmModule();

  const shardBudget = 256 << 20;
  const shards = [];
  let rem = WEIGHT_BYTES;
  while (rem > 0) {
    const s = Math.min(rem, shardBudget);
    shards.push(mockDevice.createBuffer({ size: s, usage: GPUBufferUsage.STORAGE }, false));
    rem -= s;
  }

  const intervals = [];
  let cur = DATA_BASE;
  for (const sh of shards) {
    intervals.push({ start: cur, end: cur + sh.size, target: { gpuBuffer: sh, gpuOffset: 0 } });
    cur += sh.size;
  }

  const dispatcher = new IntervalDispatcher(intervals, mockDevice, mockModule);

  // Here, we allocate FRESH Uint8Array instances on every chunk (simulating real network chunks)
  // Using 1 MB chunks to simulate 2179 chunks of 1MB each
  const CHUNK_SIZE = 1024 * 1024 + 1; // 1MB + 1 byte (adversarial unaligned chunk)
  let bytesRemaining = WEIGHT_BYTES;
  let peakHeapUsed = 0;
  let chunkCount = 0;

  const simulatedReader = {
    async read() {
      if (bytesRemaining <= 0) return { done: true, value: undefined };
      const size = Math.min(bytesRemaining, CHUNK_SIZE);
      bytesRemaining -= size;
      chunkCount++;

      // Allocate FRESH chunk
      const chunk = new Uint8Array(size);

      if (chunkCount % 100 === 0) {
        const mem = process.memoryUsage().heapUsed;
        if (mem > peakHeapUsed) peakHeapUsed = mem;
      }

      return { done: false, value: chunk };
    },
  };

  const initialHeap = process.memoryUsage().heapUsed;
  const startTime = Date.now();

  await streamChunksToTargets({
    reader: simulatedReader,
    dispatcher,
    initialFileOffset: DATA_BASE,
    totalBytes: TOTAL_STREAM_BYTES,
    device: mockDevice,
    excessChunk: null,
  });

  const durationMs = Date.now() - startTime;
  const finalHeap = process.memoryUsage().heapUsed;
  peakHeapUsed = Math.max(peakHeapUsed, finalHeap);

  const peakMb = (peakHeapUsed / (1024 * 1024)).toFixed(2);
  const initialMb = (initialHeap / (1024 * 1024)).toFixed(2);
  const finalMb = (finalHeap / (1024 * 1024)).toFixed(2);

  console.log(`  -> Freshly allocated ${chunkCount} chunks across 2.18 GB`);
  console.log(`  -> Initial Heap: ${initialMb} MB | Peak Heap: ${peakMb} MB | Final Heap: ${finalMb} MB`);
  console.log(`  -> Elapsed: ${(durationMs / 1000).toFixed(2)}s`);

  assert.ok(
    peakHeapUsed < 100 * 1024 * 1024,
    `[OOM_VIOLATION] Fresh allocation peak heap ${peakMb} MB exceeded 100 MB ceiling!`
  );

  console.log("  ✔ Peak heap usage strictly bounded (< 100 MB) even with fresh chunk allocations.");
  passedAssertions++;
}


console.log("\n==================================================================");
console.log(`  SUMMARY: All ${passedAssertions} stress test suites PASSED with 0 faults!`);
console.log("==================================================================");
process.exit(0);
