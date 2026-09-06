/**
 * tools/adversarial-challenger-m3-1.mjs
 * Empirical Challenger M3.1 Stress-Testing Harness for wasm/demo/mic-worklet.js
 *
 * Requirements:
 * 1. Test sample rate conversions: 48,000 -> 16,000 Hz, 44,100 -> 16,000 Hz, 96,000 -> 16,000 Hz, 16,000 -> 16,000 Hz passthrough.
 * 2. Test long-running stream (60+ s synthetic audio chunks) to verify sample drift is exactly 0.
 * 3. Test anti-aliasing filter attenuation: inject high-frequency sine waves (> 8 kHz: 10 kHz, 15 kHz, 20 kHz) and verify significant attenuation.
 * 4. Verify zero NaN/Infinity under extreme inputs (silence, full-scale +/-1.0, DC offset, impulse burst).
 * 5. Verify chunk accumulation to CHUNK = 4000 and transferability of buffer.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import vm from "node:vm";
import { MessageChannel } from "node:worker_threads";

const WORKLET_PATH = join(process.cwd(), "wasm", "demo", "mic-worklet.js");
const workletSource = readFileSync(WORKLET_PATH, "utf8");

// Helper to instantiate MicCollector under a mocked AudioWorklet environment
function createCollectorInstance({ sourceRate, mockSampleRate, customPort } = {}) {
  let registeredName = null;
  let registeredClass = null;

  const sandbox = {
    AudioWorkletProcessor: class AudioWorkletProcessor {
      constructor() {
        this.port = customPort || {
          postMessage: () => {}
        };
      }
    },
    registerProcessor: (name, cls) => {
      registeredName = name;
      registeredClass = cls;
    },
    sampleRate: mockSampleRate !== undefined ? mockSampleRate : sourceRate,
    Math,
    Float32Array,
    Float64Array,
    console
  };

  vm.runInNewContext(workletSource, sandbox);

  assert.strictEqual(registeredName, "mic-collector", "Processor must register as 'mic-collector'");
  assert.ok(registeredClass, "Processor class must be registered");

  const options = sourceRate !== undefined ? { processorOptions: { sourceRate } } : undefined;
  const instance = new registeredClass(options);
  if (customPort) instance.port = customPort;
  return { instance, registeredClass, registeredName };
}

// DFT power analysis helper
function analyzeDft(signal, sampleRate) {
  const N = signal.length;
  const binWidth = sampleRate / N;
  let maxPower = 0;
  let peakFreq = 0;
  let peakBin = 0;

  const numBins = Math.floor(N / 2);
  const powers = new Float64Array(numBins);
  for (let k = 1; k < numBins; k++) {
    let re = 0;
    let im = 0;
    const w = (2 * Math.PI * k) / N;
    for (let n = 0; n < N; n++) {
      re += signal[n] * Math.cos(w * n);
      im -= signal[n] * Math.sin(w * n);
    }
    const p = (re * re + im * im) / (N * N);
    powers[k] = p;
    if (p > maxPower) {
      maxPower = p;
      peakFreq = k * binWidth;
      peakBin = k;
    }
  }

  let noisePower = 0;
  for (let k = 1; k < numBins; k++) {
    if (Math.abs(k - peakBin) > 3) {
      noisePower += powers[k];
    }
  }

  const snrDb = 10 * Math.log10(maxPower / (noisePower + 1e-12));
  return { peakFreq, peakBin, binWidth, snrDb, peakPower: maxPower };
}

// Measure RMS attenuation of a sine wave
function measureFilterAttenuation(freq, sourceRate) {
  const collector = createCollectorInstance({ sourceRate }).instance;
  const outputs = [];
  collector.port = {
    postMessage: (chunk) => { outputs.push(chunk); }
  };

  const durationSec = 1.5;
  const numQuanta = Math.floor((durationSec * sourceRate) / 128);
  for (let q = 0; q < numQuanta; q++) {
    const chunk = new Float32Array(128);
    for (let i = 0; i < 128; i++) {
      const t = (q * 128 + i) / sourceRate;
      chunk[i] = Math.sin(2 * Math.PI * freq * t);
    }
    collector.process([[chunk]]);
  }

  const allSamples = [];
  for (const c of outputs) {
    for (let i = 0; i < c.length; i++) allSamples.push(c[i]);
  }
  for (let i = 0; i < collector.n; i++) allSamples.push(collector.buf[i]);

  // Discard first 200ms warmup to allow IIR filter settling
  const warmup = Math.floor(0.2 * 16000);
  const steady = allSamples.slice(warmup);
  assert.ok(steady.length > 1000, "Must have sufficient steady-state samples");

  let sumSq = 0;
  for (const s of steady) sumSq += s * s;
  const rms = Math.sqrt(sumSq / steady.length);
  const inputRms = 1 / Math.SQRT2; // RMS of unit sine = ~0.707107
  const gain = rms / inputRms;
  const db = 20 * Math.log10(gain);

  return { freq, sourceRate, rms, gain, db };
}

let totalTests = 0;
let passedTests = 0;

function runTest(name, fn) {
  totalTests++;
  process.stdout.write(`  [TEST] ${name} ... `);
  try {
    fn();
    passedTests++;
    console.log("PASS");
  } catch (err) {
    console.log("FAIL");
    console.error(err);
    throw err;
  }
}

async function runAsyncTest(name, fn) {
  totalTests++;
  process.stdout.write(`  [TEST] ${name} ... `);
  try {
    await fn();
    passedTests++;
    console.log("PASS");
  } catch (err) {
    console.log("FAIL");
    console.error(err);
    throw err;
  }
}

console.log("================================================================================");
console.log("CHALLENGER M3.1: Empirical Stress Suite for wasm/demo/mic-worklet.js");
console.log("================================================================================\n");

// -----------------------------------------------------------------------------
// Suite 1: Sample Rate Conversions & Frequency Fidelity
// -----------------------------------------------------------------------------
console.log("--- Suite 1: Sample Rate Conversions & Frequency Fidelity ---");

for (const rate of [48000, 44100, 96000, 16000]) {
  runTest(`Preserves 1000 Hz tone when converting ${rate} Hz -> 16,000 Hz`, () => {
    const collector = createCollectorInstance({ sourceRate: rate }).instance;
    const outputs = [];
    collector.port = { postMessage: (c) => outputs.push(c) };

    const durationSec = 2.0;
    const numQuanta = Math.floor((durationSec * rate) / 128);
    for (let q = 0; q < numQuanta; q++) {
      const chunk = new Float32Array(128);
      for (let i = 0; i < 128; i++) {
        const t = (q * 128 + i) / rate;
        chunk[i] = Math.sin(2 * Math.PI * 1000 * t);
      }
      collector.process([[chunk]]);
    }

    const all = [];
    for (const c of outputs) for (let i = 0; i < c.length; i++) all.push(c[i]);
    for (let i = 0; i < collector.n; i++) all.push(collector.buf[i]);

    const window = all.slice(8000, 8000 + 4096);
    const dft = analyzeDft(window, 16000);

    assert.ok(
      Math.abs(dft.peakFreq - 1000) <= dft.binWidth,
      `Peak freq ${dft.peakFreq} Hz must match 1000 Hz within bin resolution (${dft.binWidth} Hz)`
    );
    assert.ok(dft.snrDb > 55.0, `SNR ${dft.snrDb.toFixed(1)} dB must exceed 55 dB threshold`);
  });
}

runTest("Constructor fallback logic respects processorOptions over global sampleRate", () => {
  const { instance } = createCollectorInstance({ sourceRate: 48000, mockSampleRate: 44100 });
  assert.strictEqual(instance.sourceRate, 48000, "Should use processorOptions.sourceRate");
  assert.strictEqual(instance.ratio, 3.0, "Ratio should be 48000/16000 = 3.0");
});

runTest("Constructor fallback logic adopts global sampleRate when processorOptions omitted", () => {
  const { instance } = createCollectorInstance({ sourceRate: undefined, mockSampleRate: 44100 });
  assert.strictEqual(instance.sourceRate, 44100, "Should adopt global sampleRate 44100");
  assert.strictEqual(instance.ratio, 44100 / 16000, "Ratio should be 44100/16000");
});

// -----------------------------------------------------------------------------
// Suite 2: Long-Running Stream & Zero Sample Drift (60+ Seconds)
// -----------------------------------------------------------------------------
console.log("\n--- Suite 2: Long-Running Stream & Zero Sample Drift (60+ Seconds) ---");

runTest("60 seconds at 48,000 Hz produces exactly 960,000 samples (0 drift)", () => {
  const collector = createCollectorInstance({ sourceRate: 48000 }).instance;
  let chunks = 0;
  collector.port = { postMessage: () => { chunks++; } };

  const numQuanta = 22500; // 22500 * 128 = 2,880,000 samples = exactly 60.000s
  const input = new Float32Array(128);
  for (let i = 0; i < numQuanta; i++) {
    collector.process([[input]]);
  }

  const totalSamples = chunks * 4000 + collector.n;
  assert.strictEqual(totalSamples, 960000, "Total samples must be exactly 960,000");
  assert.strictEqual(collector.n, 0, "Remainder buffer should be 0 at exact integer second boundary");
  assert.strictEqual(collector.phase, 0.0, "Phase accumulator must return exactly to 0.0");
});

runTest("60 seconds at 96,000 Hz produces exactly 960,000 samples (0 drift)", () => {
  const collector = createCollectorInstance({ sourceRate: 96000 }).instance;
  let chunks = 0;
  collector.port = { postMessage: () => { chunks++; } };

  const numQuanta = 45000; // 45000 * 128 = 5,760,000 samples = exactly 60.000s
  const input = new Float32Array(128);
  for (let i = 0; i < numQuanta; i++) {
    collector.process([[input]]);
  }

  const totalSamples = chunks * 4000 + collector.n;
  assert.strictEqual(totalSamples, 960000, "Total samples must be exactly 960,000");
  assert.strictEqual(collector.n, 0, "Remainder buffer should be 0");
  assert.strictEqual(collector.phase, 0.0, "Phase accumulator must return exactly to 0.0");
});

runTest("60 seconds at 16,000 Hz passthrough produces exactly 960,000 samples (0 drift)", () => {
  const collector = createCollectorInstance({ sourceRate: 16000 }).instance;
  let chunks = 0;
  collector.port = { postMessage: () => { chunks++; } };

  const numQuanta = 7500; // 7500 * 128 = 960,000 samples = exactly 60.000s
  const input = new Float32Array(128);
  for (let i = 0; i < numQuanta; i++) {
    collector.process([[input]]);
  }

  const totalSamples = chunks * 4000 + collector.n;
  assert.strictEqual(totalSamples, 960000, "Total samples must be exactly 960,000");
  assert.strictEqual(collector.n, 0, "Remainder buffer should be 0");
});

runTest("64+ seconds (22,050 quanta) at 44,100 Hz produces exactly 1,024,000 samples (0 drift)", () => {
  const collector = createCollectorInstance({ sourceRate: 44100 }).instance;
  let chunks = 0;
  collector.port = { postMessage: () => { chunks++; } };

  // 22050 * 128 = 2,822,400 input samples. 2,822,400 / (44100 / 16000) = exactly 1,024,000 samples.
  const numQuanta = 22050;
  const input = new Float32Array(128);
  for (let i = 0; i < numQuanta; i++) {
    collector.process([[input]]);
  }

  const totalSamples = chunks * 4000 + collector.n;
  assert.strictEqual(totalSamples, 1024000, "Total samples must be exactly 1,024,000");
  assert.strictEqual(collector.n, 0, "Remainder buffer should be 0 at exact rational cycle boundary");
  assert.ok(Math.abs(collector.phase) < 1e-8, `Phase accumulator drift (${collector.phase}) must be < 1e-8`);
});

runTest("300 seconds (5 minutes) long-duration stream at 48,000 Hz maintains 0 sample drift", () => {
  const collector = createCollectorInstance({ sourceRate: 48000 }).instance;
  let chunks = 0;
  collector.port = { postMessage: () => { chunks++; } };

  const numQuanta = 112500; // 300s * 48000 / 128 = 112500 quanta = 14,400,000 samples
  const input = new Float32Array(128);
  for (let i = 0; i < numQuanta; i++) {
    collector.process([[input]]);
  }

  const totalSamples = chunks * 4000 + collector.n;
  const expectedSamples = 300 * 16000; // 4,800,000 samples
  assert.strictEqual(totalSamples, expectedSamples, `Total samples must be exactly ${expectedSamples}`);
  assert.strictEqual(collector.n, 0, "Remainder must be 0");
  assert.strictEqual(collector.phase, 0.0, "Phase must remain 0.0");
});

runTest("Variable quantum sizes (33 to 512 samples) maintain exact sample decimation with 0 drift", () => {
  const collector = createCollectorInstance({ sourceRate: 48000 }).instance;
  let chunks = 0;
  collector.port = { postMessage: () => { chunks++; } };

  const sizes = [64, 128, 256, 127, 33, 512, 192, 96];
  let totalInputSamples = 0;
  for (let i = 0; i < 2000; i++) {
    const sz = sizes[i % sizes.length];
    totalInputSamples += sz;
    const ch = new Float32Array(sz);
    collector.process([[ch]]);
  }

  const totalOutputSamples = chunks * 4000 + collector.n;
  const pendingBoundarySample = collector.phase < 0 ? 1 : 0;
  const expectedOutputSamples = Math.floor((totalInputSamples - 1) / 3.0) + 1;
  assert.strictEqual(
    totalOutputSamples + pendingBoundarySample,
    expectedOutputSamples,
    "Variable quantum output must match expected decimation"
  );
});

// -----------------------------------------------------------------------------
// Suite 3: Anti-Aliasing Filter Attenuation (> 8 kHz)
// -----------------------------------------------------------------------------
console.log("\n--- Suite 3: Anti-Aliasing Filter Attenuation (> 8 kHz) ---");

runTest("Passband transparency at 1 kHz and 4 kHz (< 0.3 dB attenuation)", () => {
  const res1k = measureFilterAttenuation(1000, 48000);
  const res4k = measureFilterAttenuation(4000, 48000);

  assert.ok(Math.abs(res1k.db) < 0.01, `1 kHz attenuation ${res1k.db.toFixed(4)} dB must be < 0.01 dB`);
  assert.ok(Math.abs(res4k.db) < 0.3, `4 kHz attenuation ${res4k.db.toFixed(4)} dB must be < 0.3 dB`);
});

runTest("Cutoff frequency at 7.2 kHz exhibits -3.0 dB attenuation (Butterworth half-power)", () => {
  const res7200 = measureFilterAttenuation(7200, 48000);
  assert.ok(
    Math.abs(res7200.db - (-3.01)) < 0.2,
    `7.2 kHz attenuation ${res7200.db.toFixed(2)} dB must match -3.01 dB (+/- 0.2 dB)`
  );
});

runTest("Nyquist edge (8 kHz) exhibits >= 9 dB attenuation at 48 kHz", () => {
  const res8k = measureFilterAttenuation(8000, 48000);
  assert.ok(res8k.db <= -9.0, `8 kHz attenuation ${res8k.db.toFixed(2)} dB must be <= -9.0 dB`);
});

runTest("Stopband attenuation: 10 kHz is attenuated by >= 12 dB across all rates", () => {
  for (const rate of [48000, 44100, 96000]) {
    const res = measureFilterAttenuation(10000, rate);
    assert.ok(res.db <= -12.0, `10 kHz at ${rate} Hz attenuation ${res.db.toFixed(2)} dB must be <= -12.0 dB`);
  }
});

runTest("Stopband attenuation: 15 kHz is attenuated by >= 27 dB across all rates", () => {
  for (const rate of [48000, 44100, 96000]) {
    const res = measureFilterAttenuation(15000, rate);
    assert.ok(res.db <= -27.0, `15 kHz at ${rate} Hz attenuation ${res.db.toFixed(2)} dB must be <= -27.0 dB`);
  }
});

runTest("Stopband attenuation: 20 kHz is attenuated by >= 40 dB across all rates (-69 dB at 48k)", () => {
  const res48k = measureFilterAttenuation(20000, 48000);
  const res44k = measureFilterAttenuation(20000, 44100);
  const res96k = measureFilterAttenuation(20000, 96000);

  assert.ok(res48k.db <= -65.0, `20 kHz at 48 kHz attenuation ${res48k.db.toFixed(2)} dB must be <= -65.0 dB`);
  assert.ok(res44k.db <= -80.0, `20 kHz at 44.1 kHz attenuation ${res44k.db.toFixed(2)} dB must be <= -80.0 dB`);
  assert.ok(res96k.db <= -40.0, `20 kHz at 96 kHz attenuation ${res96k.db.toFixed(2)} dB must be <= -40.0 dB`);
});

// -----------------------------------------------------------------------------
// Suite 4: Extreme Inputs, Numerical Stability & Zero NaN/Infinity
// -----------------------------------------------------------------------------
console.log("\n--- Suite 4: Extreme Inputs, Numerical Stability & Zero NaN/Infinity ---");

const extremePatterns = [
  { name: "silence (all zeros)", gen: () => 0.0 },
  { name: "full-scale positive DC (+1.0)", gen: () => 1.0 },
  { name: "full-scale negative DC (-1.0)", gen: () => -1.0 },
  { name: "half-scale DC offset (+0.5)", gen: () => 0.5 },
  { name: "Nyquist square wave (+/- 1.0 alternate)", gen: (i) => (i % 2 === 0 ? 1.0 : -1.0) },
  { name: "single Dirac impulse burst", gen: (i) => (i === 0 ? 1.0 : 0.0) },
  { name: "repeated Dirac impulse train (every 64 samples)", gen: (i) => (i % 64 === 0 ? 1.0 : 0.0) },
  { name: "alternating Dirac impulses (+1, -1)", gen: (i) => (i % 32 === 0 ? (i % 64 === 0 ? 1.0 : -1.0) : 0.0) },
  { name: "out-of-bounds clipped amplitude (+/- 2.0)", gen: (i) => Math.sin(i / 8) * 2.0 },
  { name: "subnormal / denormal floats (1e-38)", gen: (i) => (i % 2 === 0 ? 1e-38 : -1e-38) }
];

for (const pattern of extremePatterns) {
  runTest(`Zero NaN/Infinity under ${pattern.name} across all sample rates`, () => {
    for (const rate of [48000, 44100, 96000, 16000]) {
      const collector = createCollectorInstance({ sourceRate: rate }).instance;
      let nanCount = 0;
      let nonFiniteCount = 0;

      collector.port = {
        postMessage: (chunk) => {
          for (let i = 0; i < chunk.length; i++) {
            if (Number.isNaN(chunk[i])) nanCount++;
            if (!Number.isFinite(chunk[i])) nonFiniteCount++;
          }
        }
      };

      for (let q = 0; q < 200; q++) {
        const ch = new Float32Array(128);
        for (let i = 0; i < 128; i++) {
          ch[i] = pattern.gen(q * 128 + i);
        }
        collector.process([[ch]]);
      }

      for (let i = 0; i < collector.n; i++) {
        if (Number.isNaN(collector.buf[i])) nanCount++;
        if (!Number.isFinite(collector.buf[i])) nonFiniteCount++;
      }

      assert.strictEqual(nanCount, 0, `NaN detected at rate ${rate} for ${pattern.name}`);
      assert.strictEqual(nonFiniteCount, 0, `Non-finite detected at rate ${rate} for ${pattern.name}`);
    }
  });
}

runTest("Recovers to exact silence (no limit-cycle drift or residual DC) after high-amplitude burst", () => {
  const collector = createCollectorInstance({ sourceRate: 48000 }).instance;
  collector.port = { postMessage: () => {} };

  // Feed 100 quanta of full scale sine
  for (let q = 0; q < 100; q++) {
    const ch = new Float32Array(128);
    for (let i = 0; i < 128; i++) ch[i] = Math.sin(i * 0.1);
    collector.process([[ch]]);
  }

  // Feed 100 quanta of pure silence
  for (let q = 0; q < 100; q++) {
    const ch = new Float32Array(128);
    collector.process([[ch]]);
  }

  // Verify internal filter states and buffer decayed to near-zero
  for (let i = 0; i < collector.n; i++) {
    assert.ok(Math.abs(collector.buf[i]) < 1e-6, `Residual sample ${collector.buf[i]} must be < 1e-6`);
  }
  for (const bq of collector.biquads) {
    assert.ok(Math.abs(bq.s1) < 1e-6, `Biquad state s1 ${bq.s1} must decay to < 1e-6`);
    assert.ok(Math.abs(bq.s2) < 1e-6, `Biquad state s2 ${bq.s2} must decay to < 1e-6`);
  }
});

// -----------------------------------------------------------------------------
// Suite 5: Chunk Accumulation & Buffer Transferability
// -----------------------------------------------------------------------------
console.log("\n--- Suite 5: Chunk Accumulation & Buffer Transferability ---");

await runAsyncTest("Accumulates exactly CHUNK = 4000 samples and transfers buffer via MessageChannel (48 kHz)", async () => {
  const mc = new MessageChannel();
  const collector = createCollectorInstance({ sourceRate: 48000, customPort: mc.port1 }).instance;

  const receivedChunks = [];
  mc.port2.on("message", (chunk) => {
    receivedChunks.push(chunk);
  });

  const input = new Float32Array(128);
  for (let i = 0; i < 128; i++) input[i] = Math.sin(i);

  // Feed 1,000 quanta (= 128,000 input samples @ 48k -> 42,666 output samples -> exactly 10 full chunks)
  for (let q = 0; q < 1000; q++) {
    collector.process([[input]]);
  }

  await new Promise((resolve) => setTimeout(resolve, 50));

  assert.strictEqual(receivedChunks.length, 10, "Should have received exactly 10 chunks");
  for (let i = 0; i < receivedChunks.length; i++) {
    const chunk = receivedChunks[i];
    assert.ok(chunk instanceof Float32Array, "Received chunk must be Float32Array");
    assert.strictEqual(chunk.length, 4000, "Chunk length must be exactly 4000");
    assert.strictEqual(chunk.byteLength, 16000, "Chunk byteLength must be exactly 16000 bytes");
  }

  // Verify internal collector.buf was not detached
  assert.strictEqual(collector.buf.buffer.byteLength, 16000, "Internal collector.buf must remain allocated and not detached");
  const totalDecimated = Math.floor((128000 - 1) / 3.0) + 1; // 42667
  assert.strictEqual(collector.n, totalDecimated - 40000, "Remainder samples in buffer must match decimation remainder");

  mc.port1.close();
  mc.port2.close();
});

await runAsyncTest("Accumulates exactly CHUNK = 4000 samples and transfers buffer in 16 kHz passthrough", async () => {
  const mc = new MessageChannel();
  const collector = createCollectorInstance({ sourceRate: 16000, customPort: mc.port1 }).instance;

  const receivedChunks = [];
  mc.port2.on("message", (chunk) => {
    receivedChunks.push(chunk);
  });

  const input = new Float32Array(128);
  for (let i = 0; i < 128; i++) input[i] = Math.sin(i);

  // Feed 1,000 quanta (= 128,000 input samples @ 16k -> 128,000 output samples -> exactly 32 full chunks)
  for (let q = 0; q < 1000; q++) {
    collector.process([[input]]);
  }

  await new Promise((resolve) => setTimeout(resolve, 50));

  assert.strictEqual(receivedChunks.length, 32, "Should have received exactly 32 chunks in passthrough");
  for (let i = 0; i < receivedChunks.length; i++) {
    const chunk = receivedChunks[i];
    assert.ok(chunk instanceof Float32Array, "Received chunk must be Float32Array");
    assert.strictEqual(chunk.length, 4000, "Chunk length must be exactly 4000");
    assert.strictEqual(chunk.byteLength, 16000, "Chunk byteLength must be exactly 16000 bytes");
  }

  assert.strictEqual(collector.buf.buffer.byteLength, 16000, "Internal collector.buf must remain allocated");
  assert.strictEqual(collector.n, 0, "Remainder samples should be 0 (128000 is divisible by 4000)");

  mc.port1.close();
  mc.port2.close();
});

// -----------------------------------------------------------------------------
// Suite 6: AudioWorklet Input Edge Cases (Empty / Missing / Channels)
// -----------------------------------------------------------------------------
console.log("\n--- Suite 6: AudioWorklet Input Edge Cases ---");

runTest("Gracefully handles empty inputs, missing channels, and zero-length buffers", () => {
  const collector = createCollectorInstance({ sourceRate: 48000 }).instance;

  // Empty inputs array
  assert.strictEqual(collector.process([]), true, "Empty inputs must return true");

  // Input with no channels
  assert.strictEqual(collector.process([[]]), true, "Input with no channels must return true");

  // Input with zero-length Float32Array
  assert.strictEqual(collector.process([[new Float32Array(0)]]), true, "Zero-length channel must return true");

  // Undefined input
  assert.strictEqual(collector.process([undefined]), true, "Undefined channel list must return true");

  // State should remain clean
  assert.strictEqual(collector.n, 0, "Remainder should remain 0");
  assert.strictEqual(collector.phase, 0.0, "Phase should remain 0.0");
});

console.log("\n================================================================================");
console.log(`SUMMARY: ${passedTests} / ${totalTests} tests passed (100%).`);
console.log("================================================================================\n");
