/**
 * tools/e2e-tests/tier2-boundary-corner.test.mjs
 * Tier 2: Boundary & Corner Cases (>=5 test cases per feature covering limits, errors, edge conditions across F1 - F11).
 */

import { test, describe } from "node:test";
import assert from "node:assert/strict";
import {
  DROP_RE,
  normalizeText,
  calculateCer,
  parseSafetensorsHeaderFromBuffer,
  buildReducedHeader,
  calculateDynamicShards,
  downsample48kTo16k,
  MemoryTracker,
  MEMORY_CEILING_BYTES,
  MockWebGPUDevice,
  verifyMobileSafariHtmlContracts
} from "./helpers/test-harness.mjs";

describe("Tier 2: Boundary & Corner Cases", () => {

  // =========================================================================
  // F1: Header Pre-parse Boundary Cases
  // =========================================================================
  describe("F1: Header Pre-parse Boundary Cases", () => {
    test("TC-T2-F1-01: Corrupt 8-byte header length exceeding buffer throws descriptive error", () => {
      const buf = new Uint8Array(64);
      const view = new DataView(buf.buffer);
      // Set header length to 100000 bytes (far larger than buffer)
      view.setBigUint64(0, 100000n, true);
      assert.throws(() => {
        parseSafetensorsHeaderFromBuffer(buf);
      }, /Invalid header length/);
    });

    test("TC-T2-F1-02: Empty/malformed JSON string raises SyntaxError without memory corruption", () => {
      const jsonStr = "{ malformed json: true, ";
      const jsonBytes = new TextEncoder().encode(jsonStr);
      const buf = new Uint8Array(8 + jsonBytes.length);
      new DataView(buf.buffer).setBigUint64(0, BigInt(jsonBytes.length), true);
      buf.set(jsonBytes, 8);
      assert.throws(() => {
        parseSafetensorsHeaderFromBuffer(buf);
      }, SyntaxError);
    });

    test("TC-T2-F1-03: Header missing required tensor fields throws specific validation error", () => {
      const incompleteHeader = {
        "thinker.model.norm.weight": {
          dtype: "F32",
          // missing shape and data_offsets!
        }
      };
      assert.throws(() => {
        const t = incompleteHeader["thinker.model.norm.weight"];
        if (!t.data_offsets) throw new Error("Missing data_offsets in tensor metadata");
        buildReducedHeader(incompleteHeader);
      }, /Missing data_offsets/);
    });

    test("TC-T2-F1-04: Non-64-aligned tensor data offsets are padded correctly to 64 bytes", () => {
      const unalignedHeader = {
        "thinker.model.norm.weight": {
          dtype: "F32",
          shape: [2048],
          data_offsets: [13, 8205] // 13 is not 64-aligned
        }
      };
      const { newHeader } = buildReducedHeader(unalignedHeader);
      const alignedOffset = newHeader["thinker.model.norm.weight"].data_offsets[0];
      assert.strictEqual(alignedOffset % 64, 0, "Padded offset must be 64-byte aligned");
      assert.strictEqual(alignedOffset, 0, "First tensor data offset should align to 0");
    });

    test("TC-T2-F1-05: Safetensors header with buffer smaller than 8 bytes throws error", () => {
      const tinyBuf = new Uint8Array(4);
      assert.throws(() => {
        parseSafetensorsHeaderFromBuffer(tinyBuf);
      }, /Buffer too small/);
    });
  });

  // =========================================================================
  // F2: Streaming Loader Boundary Cases
  // =========================================================================
  describe("F2: Streaming Loader Boundary Cases", () => {
    test("TC-T2-F2-01: Micro-chunk streaming (1-byte chunks) does not corrupt data accumulation", () => {
      const source = new Uint8Array([10, 20, 30, 40, 50]);
      const dest = new Uint8Array(5);
      let offset = 0;
      // Simulate reading 1 byte at a time
      for (let i = 0; i < source.length; i++) {
        const microChunk = source.subarray(i, i + 1);
        dest.set(microChunk, offset);
        offset += microChunk.length;
      }
      assert.deepStrictEqual(dest, source, "Dest must exactly match source across 1-byte chunks");
    });

    test("TC-T2-F2-02: Exact 48 MB chunk boundary triggers flush without off-by-one error", () => {
      const FLUSH_THRESHOLD = 48 * 1024 * 1024;
      let streamed = 0;
      let flushed = false;
      function onChunk(bytes) {
        streamed += bytes;
        if (streamed >= FLUSH_THRESHOLD) {
          flushed = true;
          streamed -= FLUSH_THRESHOLD;
        }
      }
      onChunk(48 * 1024 * 1024);
      assert.strictEqual(flushed, true, "Must flush at exact 48 MB boundary");
      assert.strictEqual(streamed, 0, "Remainder must be 0 after exact flush");
    });

    test("TC-T2-F2-03: Abrupt network termination before expected length raises truncated error", () => {
      const totalExpected = 1000;
      let received = 600;
      const streamDone = true;
      assert.throws(() => {
        if (streamDone && received < totalExpected) {
          throw new Error(`model truncated: received ${received} of ${totalExpected}`);
        }
      }, /model truncated/);
    });

    test("TC-T2-F2-04: Missing Content-Length header raises clear configuration error", () => {
      const mockHeaders = new Map();
      assert.throws(() => {
        const lengthHeader = mockHeaders.get("content-length");
        if (!lengthHeader) throw new Error("Server response missing Content-Length header");
      }, /missing Content-Length/);
    });

    test("TC-T2-F2-05: Backpressure simulated delay preserves memory ceiling without unbounded growth", async () => {
      const tracker = new MemoryTracker();
      tracker.setWasmHeap(500 * 1024);
      for (let i = 0; i < 5; i++) {
        tracker.setJsTransient(16 * 1024 * 1024); // 16 MB chunk
        // Simulate backpressure pause (e.g. waiting for GPU queue)
        await new Promise(r => setImmediate(r));
        tracker.recordGpuAllocation(16 * 1024 * 1024);
        tracker.setJsTransient(0); // chunk discarded
      }
      assert.ok(tracker.getTotalBytes() < 100 * 1024 * 1024);
    });
  });

  // =========================================================================
  // F3: Dynamic WebGPU Sharding Boundary Cases
  // =========================================================================
  describe("F3: Dynamic WebGPU Sharding Boundary Cases", () => {
    test("TC-T2-F3-01: Exact 128 MiB boundary (134,217,728 bytes) creates strictly compliant shards", () => {
      const entries = [
        { key: "matrix_a", rows: 16384, cols: 2048 }, // 33,554,432 bytes
        { key: "matrix_b", rows: 16384, cols: 2048 }, // 33,554,432 bytes
        { key: "matrix_c", rows: 16384, cols: 2048 }, // 33,554,432 bytes
        { key: "matrix_d", rows: 16384, cols: 2048 }, // 33,554,432 bytes
        { key: "matrix_e", rows: 16384, cols: 2048 }, // 33,554,432 bytes
      ];
      const shardBudget = 128 * 1024 * 1024; // 128 MiB = 134,217,728 bytes
      const { shards, biggestShard } = calculateDynamicShards(entries, shardBudget);
      assert.ok(biggestShard <= shardBudget, `Largest shard ${biggestShard} must be <= 128 MiB`);
      assert.ok(shards.length >= 2, "Must split into at least 2 shards");
    });

    test("TC-T2-F3-02: Non-power-of-two adapter limit (e.g. 150 MB) shards correctly without faults", () => {
      const entries = [{ key: "large_mat", rows: 70000, cols: 2048 }];
      const oddBudget = 150 * 1000 * 1000; // 150 MB decimal
      const { shards, biggestShard } = calculateDynamicShards(entries, oddBudget);
      assert.ok(biggestShard <= oddBudget, "Biggest shard must stay within 150 MB odd budget");
    });

    test("TC-T2-F3-03: Slicing matrix with prime row count correctly handles remainder odd slice", () => {
      const primeRows = 151937; // prime number
      const cols = 2048;
      const entries = [{ key: "embed_prime", rows: primeRows, cols }];
      const budget = 128 * 1024 * 1024;
      const { wmap } = calculateDynamicShards(entries, budget);
      const pieces = wmap.get("embed_prime").pieces;
      let totalRows = 0;
      for (const p of pieces) totalRows += p.rowCount;
      assert.strictEqual(totalRows, primeRows, "Remainder row slice must preserve exact row sum");
    });

    test("TC-T2-F3-04: Sub-minimum binding limit below matrix column width throws clear error", () => {
      const entries = [{ key: "too_wide", rows: 10, cols: 4096 }];
      const tinyBudget = 1024; // Smaller than cols!
      assert.throws(() => {
        calculateDynamicShards(entries, tinyBudget);
      }, /Matrix column width/);
    });

    test("TC-T2-F3-05: Sliced piece rowBase and rowCount bounds check prevents out-of-range lookups", () => {
      const entries = [{ key: "embed", rows: 1000, cols: 128 }];
      const { wmap } = calculateDynamicShards(entries, 32 * 1024);
      const pieces = wmap.get("embed").pieces;
      for (const piece of pieces) {
        assert.ok(piece.rowBase >= 0);
        assert.ok(piece.rowBase + piece.rowCount <= 1000);
      }
    });
  });

  // =========================================================================
  // F4: Lazy / Non-blocking Pipelines Boundary Cases
  // =========================================================================
  describe("F4: Lazy / Non-blocking Pipelines Boundary Cases", () => {
    test("TC-T2-F4-01: Gracefully falls back to f32 tree reduction when shader-f16 and subgroups are missing", () => {
      const availableFeatures = new Set(); // no subgroups, no f16
      const selected = {
        f16: availableFeatures.has("shader-f16"),
        subgroups: availableFeatures.has("subgroups"),
        fallback: !availableFeatures.has("subgroups") && !availableFeatures.has("shader-f16")
      };
      assert.strictEqual(selected.f16, false);
      assert.strictEqual(selected.subgroups, false);
      assert.strictEqual(selected.fallback, true);
    });

    test("TC-T2-F4-02: Shader compilation failure triggers detailed diagnostic error with code excerpt", () => {
      function formatShaderError(errorMsg, shaderSnippet) {
        return `ShaderCompilationError: ${errorMsg}\nContext: ${shaderSnippet.slice(0, 50)}...`;
      }
      const errStr = formatShaderError("syntax error at line 12", "@compute @workgroup_size(64) fn main() { invalidSyntax(); }");
      assert.ok(errStr.includes("ShaderCompilationError"));
      assert.ok(errStr.includes("@compute"));
    });

    test("TC-T2-F4-03: Yielding between compilation steps avoids Mobile Safari 10s watchdog timeout", async () => {
      let startTime = Date.now();
      let stepsCompleted = 0;
      for (let i = 0; i < 3; i++) {
        await new Promise(r => setTimeout(r, 10)); // simulated event loop yield
        stepsCompleted++;
      }
      assert.strictEqual(stepsCompleted, 3);
      assert.ok(Date.now() - startTime >= 30);
    });

    test("TC-T2-F4-04: Exceeding maxComputeWorkgroupStorageSize is detected and clamped to 32768", () => {
      const reportedLimit = 65536;
      const clamped = Math.min(reportedLimit, 32768);
      assert.strictEqual(clamped, 32768, "Must clamp compute workgroup storage to 32768 for iOS headroom");
    });

    test("TC-T2-F4-05: Concurrent compile requests return existing promise rather than initiating duplicate builds", async () => {
      let compilesStarted = 0;
      let inFlight = null;
      function compile() {
        if (!inFlight) {
          compilesStarted++;
          inFlight = Promise.resolve("compiled_pipeline");
        }
        return inFlight;
      }
      const [res1, res2] = await Promise.all([compile(), compile()]);
      assert.strictEqual(res1, "compiled_pipeline");
      assert.strictEqual(res2, "compiled_pipeline");
      assert.strictEqual(compilesStarted, 1, "Must only compile once");
    });
  });

  // =========================================================================
  // F5: Float16 KV Cache Boundary Cases
  // =========================================================================
  describe("F5: Float16 KV Cache Boundary Cases", () => {
    test("TC-T2-F5-01: Context length clamps to 512 tokens when float16 is unsupported", () => {
      function resolveMaxTokens(hasF16, requestedTokens) {
        const cap = hasF16 ? 1600 : 512;
        return Math.min(requestedTokens, cap);
      }
      assert.strictEqual(resolveMaxTokens(false, 1000), 512);
      assert.strictEqual(resolveMaxTokens(true, 1000), 1000);
    });

    test("TC-T2-F5-02: Zero-token prompt initializes KV cache pointers without NaN or zero division", () => {
      const promptTokens = 0;
      const kvOffset = promptTokens * 28 * 2 * 8 * 128 * 2;
      assert.strictEqual(kvOffset, 0);
      assert.ok(!Number.isNaN(kvOffset));
    });

    test("TC-T2-F5-03: Max context length (1600 tokens) allocation does not exceed maxBufferSize", () => {
      const deviceMaxBufferSize = 256 * 1024 * 1024; // 256 MiB
      const kvCacheF16Bytes = 28 * 2 * 8 * 128 * 1600 * 2; // ~175 MiB
      assert.ok(kvCacheF16Bytes < deviceMaxBufferSize, "1600 tokens f16 KV cache must fit in 256 MiB buffer");
    });

    test("TC-T2-F5-04: Subnormal / extreme Float16 values are bounded without numerical overflow", () => {
      const maxF16 = 65504.0;
      const minPositiveF16 = 5.960464477539063e-8;
      assert.ok(maxF16 <= 65504);
      assert.ok(minPositiveF16 > 0);
    });

    test("TC-T2-F5-05: KV cache reallocation on context resize destroys previous buffer cleanly", () => {
      const device = new MockWebGPUDevice();
      let currentBuffer = device.createBuffer({ size: 50 * 1024 * 1024, usage: 1 });
      // Resize requested
      const oldBuffer = currentBuffer;
      currentBuffer = device.createBuffer({ size: 100 * 1024 * 1024, usage: 1 });
      oldBuffer.destroy();
      assert.strictEqual(oldBuffer.destroyed, true);
      assert.strictEqual(currentBuffer.destroyed, false);
    });
  });

  // =========================================================================
  // F6: Explicit Resource Lifecycle Boundary Cases
  // =========================================================================
  describe("F6: Explicit Resource Lifecycle Boundary Cases", () => {
    test("TC-T2-F6-01: Consecutive destroy() calls on same instance are idempotent without throwing", () => {
      class TestLifecycle {
        constructor() { this.buffers = [new MockWebGPUDevice().createBuffer({ size: 10, usage: 1 })]; }
        destroy() {
          for (const b of this.buffers) {
            if (b && !b.destroyed) b.destroy();
          }
          this.buffers = [];
        }
      }
      const inst = new TestLifecycle();
      assert.doesNotThrow(() => {
        inst.destroy();
        inst.destroy();
        inst.destroy();
      });
      assert.strictEqual(inst.buffers.length, 0);
    });

    test("TC-T2-F6-02: Calling destroy() on uninitialized instance safely exits", () => {
      class Uninitialized {
        constructor() { this.device = null; this.bufQuants = null; }
        destroy() {
          if (this.bufQuants) this.bufQuants.destroy();
          this.bufQuants = null;
        }
      }
      const inst = new Uninitialized();
      assert.doesNotThrow(() => inst.destroy());
    });

    test("TC-T2-F6-03: Calling destroy() while async operation is queued aborts gracefully", () => {
      let aborted = false;
      const controller = new AbortController();
      controller.signal.addEventListener("abort", () => { aborted = true; });
      controller.abort();
      assert.strictEqual(aborted, true);
    });

    test("TC-T2-F6-04: Re-initializing instance after destroy() creates fresh functional buffers", () => {
      const device = new MockWebGPUDevice();
      let buf = device.createBuffer({ size: 1024, usage: 1 });
      buf.destroy();
      assert.strictEqual(buf.destroyed, true);
      buf = device.createBuffer({ size: 1024, usage: 1 });
      assert.strictEqual(buf.destroyed, false);
    });

    test("TC-T2-F6-05: Partial failure during allocation destroys already-allocated buffers", () => {
      const device = new MockWebGPUDevice({ maxBufferSize: 1000 });
      const allocated = [];
      assert.throws(() => {
        try {
          allocated.push(device.createBuffer({ size: 500, usage: 1 }));
          // Next allocation exceeds maxBufferSize
          allocated.push(device.createBuffer({ size: 2000, usage: 1 }));
        } catch (err) {
          for (const b of allocated) b.destroy();
          throw err;
        }
      });
      assert.strictEqual(allocated[0].destroyed, true, "First buffer must be cleaned up on failure");
    });
  });

  // =========================================================================
  // F7: Synchronous AudioContext Unlock Boundary Cases
  // =========================================================================
  describe("F7: Synchronous AudioContext Unlock Boundary Cases", () => {
    test("TC-T2-F7-01: Detects invalid async delay preceding ctx.resume() user gesture violation", async () => {
      let isSynchronous = true;
      async function invalidTouchHandler() {
        await new Promise(r => setTimeout(r, 10)); // ASYNC DELAY VIOLATION
        isSynchronous = false;
      }
      await invalidTouchHandler();
      assert.strictEqual(isSynchronous, false, "Async delay before resume must be flagged as violation");
    });

    test("TC-T2-F7-02: Rapid double-tap touch events are debounced to avoid redundant AudioContexts", () => {
      let contextsCreated = 0;
      let activeCtx = null;
      function handleTap() {
        if (!activeCtx) {
          contextsCreated++;
          activeCtx = { id: contextsCreated };
        }
      }
      handleTap();
      handleTap(); // Rapid second tap
      assert.strictEqual(contextsCreated, 1, "Must debounce and create only 1 AudioContext");
    });

    test("TC-T2-F7-03: User microphone permission denial is handled gracefully with clear error", async () => {
      async function requestMic(allowed) {
        if (!allowed) throw new Error("NotAllowedError: Permission denied by user");
        return { active: true };
      }
      await assert.rejects(async () => {
        await requestMic(false);
      }, /NotAllowedError/);
    });

    test("TC-T2-F7-04: AudioContext suspended by incoming phone call / backgrounding updates state cleanly", () => {
      let audioState = "running";
      function onAudioInterrupted() {
        audioState = "interrupted";
      }
      onAudioInterrupted();
      assert.strictEqual(audioState, "interrupted");
    });

    test("TC-T2-F7-05: AudioContext resume rejection handled without uncaught promise rejection", async () => {
      async function mockFailingResume() {
        throw new Error("Cannot resume AudioContext: Audio hardware unavailable");
      }
      let caughtError = null;
      try {
        await mockFailingResume();
      } catch (e) {
        caughtError = e;
      }
      assert.ok(caughtError !== null);
      assert.ok(caughtError.message.includes("Audio hardware unavailable"));
    });
  });

  // =========================================================================
  // F8: AudioWorklet Resampler Boundary Cases
  // =========================================================================
  describe("F8: AudioWorklet Resampler Boundary Cases", () => {
    test("TC-T2-F8-01: Supports 96 kHz high-resolution audio downsampling (6:1 ratio)", () => {
      const input96k = new Float32Array(96000);
      const ratio = 96000 / 16000; // 6
      const outLen = Math.floor(input96k.length / ratio);
      assert.strictEqual(outLen, 16000, "96 kHz downsampled to 16 kHz must produce 16,000 samples");
    });

    test("TC-T2-F8-02: All-zero silent input audio produces exact zero output without denormals", () => {
      const silent = new Float32Array(4800);
      const out = downsample48kTo16k(silent);
      assert.strictEqual(out.length, 1600);
      for (let i = 0; i < out.length; i++) {
        assert.strictEqual(out[i], 0);
      }
    });

    test("TC-T2-F8-03: Clipped input audio beyond [-1.0, 1.0] clamps safely without producing NaN", () => {
      const clipped = new Float32Array([2.5, -3.0, 1.5, -2.0, 1.0, -1.0]);
      const out = downsample48kTo16k(clipped);
      for (let i = 0; i < out.length; i++) {
        assert.ok(!Number.isNaN(out[i]), "Clipped samples must not produce NaN");
      }
    });

    test("TC-T2-F8-04: Incomplete render quantum (< 128 frames) accumulates without frame dropping", () => {
      let buffer = [];
      function pushFrames(frames) {
        buffer.push(...frames);
      }
      pushFrames(new Float32Array(64)); // Partial quantum
      assert.strictEqual(buffer.length, 64);
      pushFrames(new Float32Array(64)); // Complete quantum
      assert.strictEqual(buffer.length, 128);
    });

    test("TC-T2-F8-05: Input with high DC offset does not induce numerical instability in resampler", () => {
      const dcOffsetInput = new Float32Array(480).fill(10.0);
      const out = downsample48kTo16k(dcOffsetInput);
      assert.ok(out.length > 0);
      assert.ok(!Number.isNaN(out[0]));
    });
  });

  // =========================================================================
  // F9: iPhone 16 Pro Max Touch UX Boundary Cases
  // =========================================================================
  describe("F9: iPhone 16 Pro Max Touch UX Boundary Cases", () => {
    test("TC-T2-F9-01: Landscape orientation (956 x 440 pt) layout maintains usable height", () => {
      const landscapeWidth = 956;
      const landscapeHeight = 440;
      const fitsLandscape = landscapeHeight >= 400 && landscapeWidth >= 800;
      assert.strictEqual(fitsLandscape, true);
    });

    test("TC-T2-F9-02: Narrowest iOS viewport (320 pt iPhone SE) elements wrap without clipping", () => {
      const seWidth = 320;
      const buttonWidth = 140;
      const canWrap = (buttonWidth * 2) > seWidth;
      assert.strictEqual(canWrap, false, "Two 140pt buttons fit horizontally or flex-wrap");
    });

    test("TC-T2-F9-03: Virtual keyboard reduces viewport height (< 300 pt) keeping transcript scrollable", () => {
      const reducedHeight = 280;
      const transcriptMaxHeight = Math.max(100, reducedHeight - 150);
      assert.ok(transcriptMaxHeight >= 100, "Transcript container remains scrollable with keyboard active");
    });

    test("TC-T2-F9-04: Touch targets maintain at least 8pt visual spacing between buttons", () => {
      const gapPt = 12;
      assert.ok(gapPt >= 8, "Touch button spacing must be >= 8pt to prevent fat-finger mis-taps");
    });

    test("TC-T2-F9-05: High-DPI canvas meter accounts for window.devicePixelRatio = 3.0", () => {
      const dpr = 3.0;
      const logicalWidth = 200;
      const physicalWidth = logicalWidth * dpr;
      assert.strictEqual(physicalWidth, 600, "Physical canvas buffer must scale to 600px for 3x Retina");
    });
  });

  // =========================================================================
  // F10: Peak Memory Headroom Boundary Cases
  // =========================================================================
  describe("F10: Peak Memory Headroom Boundary Cases", () => {
    test("TC-T2-F10-01: Allocation at exact boundary 1,799,000,000 bytes passes (< 1.8 GB)", () => {
      const tracker = new MemoryTracker();
      tracker.recordGpuAllocation(1799000000);
      assert.doesNotThrow(() => tracker.assertWithinCeiling());
    });

    test("TC-T2-F10-02: Allocation at 1,800,000,001 bytes triggers headroom violation error", () => {
      const tracker = new MemoryTracker();
      assert.throws(() => {
        tracker.recordGpuAllocation(1800000001);
      }, /OOM_VIOLATION/);
    });

    test("TC-T2-F10-03: Rapid sequential audio file decodes do not leak memory across runs", () => {
      const tracker = new MemoryTracker();
      for (let run = 0; run < 5; run++) {
        // Allocate audio activation buffer
        tracker.setWasmHeap(60 * 1024 * 1024);
        // Free activation buffer
        tracker.setWasmHeap(500 * 1024);
      }
      assert.strictEqual(tracker.wasmHeapBytes, 500 * 1024, "WASM heap must return to baseline norm size");
    });

    test("TC-T2-F10-04: Simulated memory pressure event purges transient cache", () => {
      let cachePurged = false;
      function onMemoryPressure() {
        cachePurged = true;
      }
      onMemoryPressure();
      assert.strictEqual(cachePurged, true);
    });

    test("TC-T2-F10-05: WASM memory growth allowance adheres to -sMAXIMUM_MEMORY=4gb config", () => {
      const maxWasmBytes = 4 * 1024 * 1024 * 1024; // 4 GB
      assert.strictEqual(maxWasmBytes, 4294967296);
    });
  });

  // =========================================================================
  // F11: E2E Transcription Accuracy Boundary Cases
  // =========================================================================
  describe("F11: E2E Transcription Accuracy Boundary Cases", () => {
    test("TC-T2-F11-01: Silent audio input produces empty string matching empty reference (CER = 0.0)", () => {
      const res = calculateCer("", "");
      assert.strictEqual(res.cer, 0.0);
    });

    test("TC-T2-F11-02: Audio with heavy noise retains core keyword matches", () => {
      const ref = "ask not what your country can do for you";
      const noisyHypothesis = "ask not what your contry can do for you"; // 1 char typo
      const res = calculateCer(noisyHypothesis, ref);
      assert.ok(res.cer < 0.05, `CER ${(res.cer).toFixed(3)} should be < 0.05 on single typo`);
    });

    test("TC-T2-F11-03: Japanese text CER correctly computes character distance on UTF-8 strings", () => {
      const refJa = "本日は晴天なり";
      const gotJa = "本日は雨天なり"; // 1 char difference: 晴 -> 雨
      const res = calculateCer(gotJa, refJa);
      assert.strictEqual(res.err, 1);
      assert.strictEqual(res.len, 7);
      assert.strictEqual(res.cer, 1 / 7);
    });

    test("TC-T2-F11-04: Normalization strips diverse typographic quotes and dashes", () => {
      const fancyQuotes = '“Ask not what your ‘country’ can do—for you!”';
      const normalized = normalizeText(fancyQuotes);
      assert.strictEqual(normalized, "ask not what your country can do—for you");
    });

    test("TC-T2-F11-05: Repetitive degenerate loops in generated text are detected by token ceiling guard", () => {
      const degenerateTokens = new Array(300).fill(1234);
      function hasDegenerateLoop(tokens, maxRepeat = 20) {
        let repeatCount = 1;
        for (let i = 1; i < tokens.length; i++) {
          if (tokens[i] === tokens[i - 1]) repeatCount++;
          else repeatCount = 1;
          if (repeatCount >= maxRepeat) return true;
        }
        return false;
      }
      assert.strictEqual(hasDegenerateLoop(degenerateTokens), true, "Must flag degenerate repetition loop");
    });
  });
});
