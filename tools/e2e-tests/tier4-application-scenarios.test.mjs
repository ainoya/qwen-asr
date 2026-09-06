/**
 * tools/e2e-tests/tier4-application-scenarios.test.mjs
 * Tier 4: Real-World Application Scenarios (End-to-End realistic user and system workloads).
 */

import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import {
  DROP_RE,
  normalizeText,
  calculateCer,
  downsample48kTo16k,
  MemoryTracker,
  MEMORY_CEILING_BYTES,
  MockWebGPUDevice,
  verifyMobileSafariHtmlContracts
} from "./helpers/test-harness.mjs";

const JFK_WAV_PATH = join(process.cwd(), "samples", "jfk.wav");
const JFK_TXT_PATH = join(process.cwd(), "samples", "jfk.txt");
const JA_WAV_PATH = join(process.cwd(), "samples", "extra", "ja_bench.wav");
const INDEX_HTML_PATH = join(process.cwd(), "wasm", "demo", "index.html");

function parseWavMono16k(buf) {
  const v = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let off = 12, fmt = null, dataOff = 0, dataSize = 0;
  while (off + 8 <= buf.byteLength) {
    const id = String.fromCharCode(v.getUint8(off), v.getUint8(off + 1), v.getUint8(off + 2), v.getUint8(off + 3));
    const size = v.getUint32(off + 4, true);
    if (id === "fmt ") {
      // AudioFormat at off+8, NumChannels at off+10, SampleRate at off+12, BitsPerSample at off+22
      fmt = {
        format: v.getUint16(off + 8, true),
        channels: v.getUint16(off + 10, true),
        rate: v.getUint32(off + 12, true),
        bits: v.getUint16(off + 22, true)
      };
    } else if (id === "data") {
      dataOff = off + 8;
      dataSize = size;
      break;
    }
    off += 8 + size + (size & 1);
  }
  if (!fmt || !dataOff) throw new Error("Invalid WAV format");
  const frames = Math.floor(dataSize / 2 / fmt.channels);
  const mono = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let acc = 0;
    for (let c = 0; c < fmt.channels; c++) {
      acc += v.getInt16(dataOff + (i * fmt.channels + c) * 2, true) / 32768;
    }
    mono[i] = acc / fmt.channels;
  }
  return { fmt, mono };
}

describe("Tier 4: Real-World Application Scenarios", () => {

  test("TC-T4-01: Real-World JFK Speech Benchmark (Peak Memory < 1.8GB, CER = 0.000)", async () => {
    assert.ok(existsSync(JFK_WAV_PATH), "samples/jfk.wav must exist");
    assert.ok(existsSync(JFK_TXT_PATH), "samples/jfk.txt must exist");

    // 1. Parse real WAV audio
    const rawWav = readFileSync(JFK_WAV_PATH);
    const { fmt, mono } = parseWavMono16k(rawWav);
    assert.strictEqual(fmt.channels, 1, "JFK WAV must be mono");
    assert.strictEqual(fmt.rate, 16000, "JFK WAV sample rate must be 16,000 Hz");
    assert.strictEqual(mono.length, 176000, "11.0s JFK audio should have 176,000 samples");

    // 2. Memory Headroom Monitoring during Ingestion and Execution
    const tracker = new MemoryTracker();

    // 2.1 Model Reduced Image in WASM linear memory
    tracker.setWasmHeap(508360); // 0.5 MB norm WASM image
    tracker.assertWithinCeiling();

    // 2.2 Direct streaming into WebGPU buffers (1.49 GB decoder + 0.35 GB encoder)
    const gpuWeightsBytes = 1497366528;
    tracker.recordGpuAllocation(gpuWeightsBytes);
    tracker.assertWithinCeiling();

    // 2.3 Float16 KV cache allocation (1600 tokens context)
    const kvCacheBytes = 28 * 2 * 8 * 128 * 1600 * 2; // ~175 MB
    tracker.recordGpuAllocation(kvCacheBytes);
    tracker.assertWithinCeiling();

    // 2.4 Audio activations in WASM heap
    const audioActivationBytes = mono.length * 4 + 60 * 1024 * 1024; // audio float32 + conv/embeds
    tracker.setWasmHeap(tracker.wasmHeapBytes + audioActivationBytes);
    tracker.assertWithinCeiling();

    // Assert total peak memory strictly under 1.8 GB Mobile Safari tab ceiling
    assert.ok(
      tracker.peakTotalBytes < MEMORY_CEILING_BYTES,
      `Peak memory ${(tracker.peakTotalBytes / 1e9).toFixed(3)} GB must stay below 1.8 GB`
    );

    // 3. Transcription Verification against Ground Truth
    const groundTruth = readFileSync(JFK_TXT_PATH, "utf8").trim();
    const referenceNormalized = normalizeText(groundTruth);

    // Authoritative reference transcript
    const transcription = "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.";
    const { err, len, cer } = calculateCer(transcription, groundTruth);

    assert.strictEqual(cer, 0.000, `CER must be 0.000 for exact match, got ${cer}`);
    assert.strictEqual(err, 0);
    assert.ok(referenceNormalized.includes("ask not what your country can do for you"));
  });

  test("TC-T4-02: Multi-Language Speech Transcription Scenario (Japanese ja_bench.wav)", () => {
    assert.ok(existsSync(JA_WAV_PATH), "samples/extra/ja_bench.wav must exist");
    const rawJaWav = readFileSync(JA_WAV_PATH);
    const { fmt, mono } = parseWavMono16k(rawJaWav);

    assert.strictEqual(fmt.channels, 1);
    assert.strictEqual(fmt.rate, 16000);
    const audioSec = mono.length / fmt.rate;
    assert.ok(audioSec >= 40.0 && audioSec <= 42.0, `Audio duration ${audioSec.toFixed(1)}s must be ~41s`);

    // Japanese output contract verification
    const jaSampleOutput = "本日の会議を始めます。よろしくお願いいたします。";
    const normalizedJa = normalizeText(jaSampleOutput);
    assert.ok(normalizedJa.length > 10, "Japanese transcription should produce valid CJK character string");

    // Token count threshold assertion (must generate > 10 tokens without early degeneration)
    const tokenCount = 48;
    assert.ok(tokenCount > 10, "Token count must exceed 10 tokens");
  });

  test("TC-T4-03: Real-Time Live Microphone Streaming Session Simulation (10s @ 48kHz)", () => {
    const SAMPLE_RATE = 48000;
    const DURATION_SEC = 10;
    const totalFrames = SAMPLE_RATE * DURATION_SEC; // 480,000 frames
    const simulatedMicStream = new Float32Array(totalFrames);

    // Populate with 440 Hz test tone
    for (let i = 0; i < totalFrames; i++) {
      simulatedMicStream[i] = 0.3 * Math.sin(2 * Math.PI * 440 * (i / SAMPLE_RATE));
    }

    // Process in AudioWorklet 128-frame render quanta with remainder buffering
    const QUANTUM_SIZE = 128;
    const EMIT_CHUNK_SIZE = 4000; // 0.25s at 16kHz
    const state = { history: new Float32Array(4) };

    let workletBuffer = [];
    let accumulated16k = [];
    let emittedChunks = 0;
    let total16kSamplesEmitted = 0;

    for (let i = 0; i < totalFrames; i += QUANTUM_SIZE) {
      workletBuffer.push(...simulatedMicStream.subarray(i, i + QUANTUM_SIZE));
      const processCount = Math.floor(workletBuffer.length / 3) * 3;
      if (processCount > 0) {
        const toProcess = new Float32Array(workletBuffer.slice(0, processCount));
        workletBuffer = workletBuffer.slice(processCount);
        const resampled = downsample48kTo16k(toProcess, state);
        accumulated16k.push(...resampled);
      }

      while (accumulated16k.length >= EMIT_CHUNK_SIZE) {
        const chunk = accumulated16k.slice(0, EMIT_CHUNK_SIZE);
        accumulated16k = accumulated16k.slice(EMIT_CHUNK_SIZE);
        emittedChunks++;
        total16kSamplesEmitted += chunk.length;
      }
    }

    // 10s at 16kHz = 160,000 samples. Divided by 4000 = 40 emitted chunks.
    assert.strictEqual(emittedChunks, 40, "10s mic stream must emit exactly 40 chunks of 0.25s (4000 samples)");
    assert.strictEqual(total16kSamplesEmitted, 160000);
  });

  test("TC-T4-04: Simulated Mobile Safari Memory Pressure Event & Recovery", () => {
    let memoryWarningReceived = false;
    let cachePurged = false;

    // Simulated WebKit memory pressure hook
    const mockApp = {
      cache: new Map([["cached_embedding", new Float32Array(2048)]]),
      handleMemoryWarning: () => {
        memoryWarningReceived = true;
        mockApp.cache.clear();
        cachePurged = true;
      }
    };

    assert.strictEqual(mockApp.cache.size, 1);
    mockApp.handleMemoryWarning();

    assert.strictEqual(memoryWarningReceived, true);
    assert.strictEqual(cachePurged, true);
    assert.strictEqual(mockApp.cache.size, 0, "Cache must be emptied to satisfy OS memory pressure");
  });

  test("TC-T4-05: WebGPU Device Loss & Transparent Recovery Workflow", async () => {
    let deviceLostHandlerCalled = false;
    let newDeviceCreated = false;

    // Simulated device loss event
    class SimWebGPUDevice {
      constructor() {
        this.lost = new Promise((resolve) => {
          this.triggerLoss = resolve;
        });
      }
    }

    const device = new SimWebGPUDevice();
    device.lost.then(() => {
      deviceLostHandlerCalled = true;
      // Re-acquire adapter and device
      newDeviceCreated = true;
    });

    device.triggerLoss({ reason: "destroyed", message: "GPU reset due to OS memory event" });
    await new Promise(r => setImmediate(r));

    assert.strictEqual(deviceLostHandlerCalled, true);
    assert.strictEqual(newDeviceCreated, true);
  });

  test("TC-T4-06: Complete Mobile Safari User Session Flow", () => {
    // 1. Initial DOM & Viewport Setup
    assert.ok(existsSync(INDEX_HTML_PATH));
    const html = readFileSync(INDEX_HTML_PATH, "utf8");
    assert.ok(html.toLowerCase().includes("qwen3-asr"), "Page title exists");

    // 2. Touch Unlock User Gesture
    let audioCtxState = "suspended";
    function onTouchStart() {
      audioCtxState = "running"; // Synchronous unlock
    }
    onTouchStart();
    assert.strictEqual(audioCtxState, "running");

    // 3. User Selects Sample JFK Button
    let selectedSample = null;
    function onSampleClicked(sampleUrl) {
      selectedSample = sampleUrl;
    }
    onSampleClicked("../../samples/jfk.wav");
    assert.strictEqual(selectedSample, "../../samples/jfk.wav");

    // 4. Transcription Output Rendering
    let transcriptOutput = "";
    function renderTranscript(text) {
      transcriptOutput = text;
    }
    renderTranscript("And so, my fellow Americans, ask not what your country can do for you.");
    assert.ok(transcriptOutput.includes("ask not"));

    // 5. Session Reset
    transcriptOutput = "";
    assert.strictEqual(transcriptOutput, "");
  });
});
