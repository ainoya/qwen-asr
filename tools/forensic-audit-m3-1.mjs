/**
 * tools/forensic-audit-m3-1.mjs
 * Forensic Integrity Verification Suite for Milestone 3 (Worker M3)
 *
 * Authored by Forensic Auditor M3 (teamwork_preview_auditor)
 * Directly tests real implementation files:
 * - wasm/demo/mic-worklet.js
 * - wasm/demo/app.js
 * - wasm/demo/index.html
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

console.log("==================================================================");
console.log("  FORENSIC INTEGRITY AUDIT — MILESTONE 3 (WORKER M3)");
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
// SECTION 1: Forensic Verification of wasm/demo/mic-worklet.js
// ============================================================================
console.log("--- Section 1: MicCollector AudioWorklet Resampler (wasm/demo/mic-worklet.js) ---");

const micWorkletSource = fs.readFileSync("wasm/demo/mic-worklet.js", "utf8");

// Load MicCollector in sandboxed AudioWorkletGlobalScope
class BaseAudioWorkletProcessor {
  constructor() {
    this.port = {
      postMessage: (msg, transfer) => {
        if (this._onmessage) this._onmessage(msg, transfer);
      }
    };
  }
}

let registeredName = null;
let MicCollectorClass = null;

const sandbox = {
  AudioWorkletProcessor: BaseAudioWorkletProcessor,
  registerProcessor: (name, cls) => {
    registeredName = name;
    MicCollectorClass = cls;
  },
  sampleRate: 48000,
  Float32Array,
  Math,
  console
};

vm.createContext(sandbox);
vm.runInContext(micWorkletSource, sandbox);

check("1.1 Registration: MicCollector registers under 'mic-collector'", () => {
  assert.strictEqual(registeredName, "mic-collector");
  assert.ok(MicCollectorClass !== null);
});

check("1.2 Rate Adaptation: dynamically configures sourceRate and ratio", () => {
  const p48k = new MicCollectorClass({ processorOptions: { sourceRate: 48000 } });
  assert.strictEqual(p48k.sourceRate, 48000);
  assert.strictEqual(p48k.targetRate, 16000);
  assert.strictEqual(p48k.ratio, 3.0);

  const p44k = new MicCollectorClass({ processorOptions: { sourceRate: 44100 } });
  assert.strictEqual(p44k.sourceRate, 44100);
  assert.strictEqual(p44k.targetRate, 16000);
  assert.strictEqual(p44k.ratio, 44100 / 16000);

  const p16k = new MicCollectorClass({ processorOptions: { sourceRate: 16000 } });
  assert.strictEqual(p16k.sourceRate, 16000);
  assert.strictEqual(p16k.ratio, 1.0);
  assert.strictEqual(p16k.biquads, null, "16k passthrough must have no biquads");
});

check("1.3 Filter Coefficients: 4th-order Butterworth Q values & biquad math", () => {
  const p = new MicCollectorClass({ processorOptions: { sourceRate: 48000 } });
  assert.ok(Array.isArray(p.biquads));
  assert.strictEqual(p.biquads.length, 2, "4th order filter requires 2 cascaded biquad sections");

  // Ground truth Q values
  const q1Expected = 1 / (2 * Math.cos(3 * Math.PI / 8));
  const q2Expected = 1 / (2 * Math.cos(Math.PI / 8));

  // Compute expected coefficients for section 0
  const fc = 7200;
  const w0 = 2 * Math.PI * (fc / 48000);
  const cosw0 = Math.cos(w0);
  const sinw0 = Math.sin(w0);

  const alpha0 = sinw0 / (2 * q1Expected);
  const a0_0 = 1 + alpha0;
  const b0_0_expected = ((1 - cosw0) / 2) / a0_0;
  const a1_0_expected = (-2 * cosw0) / a0_0;

  const bq0 = p.biquads[0];
  assert.ok(Math.abs(bq0.b0 - b0_0_expected) < 1e-12, "bq0.b0 coefficient matches formula");
  assert.ok(Math.abs(bq0.a1 - a1_0_expected) < 1e-12, "bq0.a1 coefficient matches formula");

  // Verify DC gain is exactly 1.0 across both sections
  for (let s = 0; s < 2; s++) {
    const bq = p.biquads[s];
    const bSum = bq.b0 + bq.b1 + bq.b2;
    const aSum = 1 + bq.a1 + bq.a2;
    const dcGain = bSum / aSum;
    assert.ok(Math.abs(dcGain - 1.0) < 1e-6, `Section ${s} DC gain must be 1.0 (got ${dcGain})`);
  }
});

check("1.4 Filter Frequency Attenuation: passband vs stopband frequency response", () => {
  function measureGainDb(freq, sourceRate = 48000) {
    const mc = new MicCollectorClass({ processorOptions: { sourceRate } });
    const N = 48000 * 2; // 2 seconds
    let out = [];
    mc.port.postMessage = (chunk) => { out.push(...chunk); };

    const quantumSize = 128;
    for (let i = 0; i < N; i += quantumSize) {
      const chunk = new Float32Array(quantumSize);
      for (let j = 0; j < quantumSize; j++) {
        chunk[j] = Math.sin(2 * Math.PI * freq * (i + j) / sourceRate);
      }
      mc.process([ [ chunk ] ]);
    }
    const half = out.slice(Math.floor(out.length / 2));
    let sumSq = 0;
    for (const s of half) sumSq += s * s;
    const rms = Math.sqrt(sumSq / half.length);
    const inRms = 1 / Math.sqrt(2);
    return 20 * Math.log10(rms / inRms);
  }

  const g1k = measureGainDb(1000);
  const g3k = measureGainDb(3000);
  const g7k = measureGainDb(7000);
  const g10k = measureGainDb(10000);
  const g15k = measureGainDb(15000);
  const g20k = measureGainDb(20000);

  assert.ok(Math.abs(g1k) < 0.2, `1 kHz passband gain must be ~0 dB, got ${g1k.toFixed(2)} dB`);
  assert.ok(Math.abs(g3k) < 0.5, `3 kHz passband gain must be ~0 dB, got ${g3k.toFixed(2)} dB`);
  assert.ok(g7k < 0 && g7k > -6, `7 kHz cutoff transition gain must be ~ -3 dB, got ${g7k.toFixed(2)} dB`);
  assert.ok(g10k < -12, `10 kHz stopband attenuation must be < -12 dB, got ${g10k.toFixed(2)} dB`);
  assert.ok(g15k < -35, `15 kHz stopband attenuation must be < -35 dB, got ${g15k.toFixed(2)} dB`);
  assert.ok(g20k < -65, `20 kHz stopband attenuation must be < -65 dB, got ${g20k.toFixed(2)} dB`);
});

check("1.5 Fractional Interpolation Drift: zero sample drift over long duration", () => {
  function verifyDrift(sourceRate, durationSec) {
    const mc = new MicCollectorClass({ processorOptions: { sourceRate } });
    let emitted = 0;
    mc.port.postMessage = () => { emitted++; };

    const totalInputSamples = Math.round(sourceRate * durationSec);
    const quantum = 128;
    const numQuanta = Math.floor(totalInputSamples / quantum);

    const dummyInput = [ [ new Float32Array(quantum) ] ];
    for (let q = 0; q < numQuanta; q++) {
      mc.process(dummyInput);
    }

    const totalProduced = emitted * 4000 + mc.n;
    const expected = (numQuanta * quantum) / (sourceRate / 16000);
    const drift = Math.abs(totalProduced - expected);
    assert.ok(drift <= 1.0, `Drift at ${sourceRate}Hz (${durationSec}s) must be <= 1 sample, got ${drift}`);
    return { totalProduced, expected, drift };
  }

  const d48_10s = verifyDrift(48000, 10);
  assert.strictEqual(d48_10s.drift, 0, "48 kHz 10s must have exactly 0 drift");

  const d48_100s = verifyDrift(48000, 100);
  assert.strictEqual(d48_100s.drift, 0, "48 kHz 100s must have exactly 0 drift");

  const d44_100s = verifyDrift(44100, 100);
  assert.ok(d44_100s.drift < 0.5, "44.1 kHz 100s drift must be < 0.5 sample");

  const d16_10s = verifyDrift(16000, 10);
  assert.strictEqual(d16_10s.drift, 0, "16 kHz 10s passthrough must have exactly 0 drift");
});

check("1.6 Quantum Boundary Continuity: no glitches or phase discontinuities", () => {
  const mc = new MicCollectorClass({ processorOptions: { sourceRate: 48000 } });
  let samples = [];
  mc.port.postMessage = (chunk) => { samples.push(...chunk); };

  const quantum = 128;
  const numQuanta = 100;
  for (let q = 0; q < numQuanta; q++) {
    const chunk = new Float32Array(quantum);
    for (let j = 0; j < quantum; j++) {
      chunk[j] = Math.sin(2 * Math.PI * 440 * (q * quantum + j) / 48000);
    }
    mc.process([ [ chunk ] ]);
  }

  // Max theoretical step for 440 Hz at 16 kHz: 2 * pi * 440 / 16000 = 0.1728
  let maxStep = 0;
  for (let i = 100; i < samples.length; i++) {
    const step = Math.abs(samples[i] - samples[i - 1]);
    if (step > maxStep) maxStep = step;
  }
  assert.ok(maxStep < 0.18, `Max derivative step must be < 0.18 (got ${maxStep.toFixed(4)})`);
});

check("1.7 Buffer Transferability: zero-copy transferable objects emitted", () => {
  const mc = new MicCollectorClass({ processorOptions: { sourceRate: 48000 } });
  let receivedMsg = null;
  let receivedTransfer = null;
  mc.port.postMessage = (msg, transfer) => {
    receivedMsg = msg;
    receivedTransfer = transfer;
  };

  const quantum = 128;
  const input = [ [ new Float32Array(quantum).fill(0.1) ] ];
  // Feed enough quanta to fill 4000 samples (4000 * 3 = 12000 input samples = 94 quanta)
  for (let i = 0; i < 95; i++) {
    mc.process(input);
  }

  assert.ok(receivedMsg instanceof Float32Array, "Emitted message must be Float32Array");
  assert.strictEqual(receivedMsg.length, 4000, "Emitted chunk must be exactly 4000 samples");
  assert.ok(Array.isArray(receivedTransfer), "postMessage transfer array must be provided");
  assert.strictEqual(receivedTransfer[0], receivedMsg.buffer, "Transfer list must contain the chunk buffer");
});

// ============================================================================
// SECTION 2: Forensic Verification of wasm/demo/app.js (Feature F7)
// ============================================================================
console.log("\n--- Section 2: AudioContext Unlock & Lifecycle (wasm/demo/app.js) ---");

const appSource = fs.readFileSync("wasm/demo/app.js", "utf8");

check("2.1 Synchronous Resume Call: audioCtx.resume() called before any await in $('mic').onclick", () => {
  const micClickIdx = appSource.indexOf('$("mic").onclick = async () => {');
  assert.ok(micClickIdx !== -1, "$('mic').onclick must exist");

  const micStopClickIdx = appSource.indexOf('$("micstop").onclick = async () => {');
  const micClickHandlerBody = appSource.slice(micClickIdx, micStopClickIdx);

  // Find first resume call
  const firstResumeIdx = micClickHandlerBody.indexOf("audioCtx.resume()");
  assert.ok(firstResumeIdx !== -1, "audioCtx.resume() must be called in mic click handler");

  // Find first await call
  const firstAwaitIdx = micClickHandlerBody.indexOf("await ");
  assert.ok(firstAwaitIdx !== -1, "await must be present for async operations");

  assert.ok(
    firstResumeIdx < firstAwaitIdx,
    `audioCtx.resume() (pos ${firstResumeIdx}) MUST precede the first await (pos ${firstAwaitIdx})`
  );
});

check("2.2 Context Suspension on Stop: micstop suspends rather than closes context", () => {
  const micStopIdx = appSource.indexOf('$("micstop").onclick = async () => {');
  assert.ok(micStopIdx !== -1, "$('micstop').onclick must exist");

  const stopBody = appSource.slice(micStopIdx, micStopIdx + 800);
  assert.ok(stopBody.includes("audioCtx.suspend()"), "micstop must call audioCtx.suspend()");
  assert.ok(!stopBody.includes("audioCtx.close()"), "micstop MUST NOT call audioCtx.close()");
});

check("2.3 Context Reuse: decodeTo16k uses getOrCreateAudioContext without closing probe contexts", () => {
  const decodeTo16kIdx = appSource.indexOf("async function decodeTo16k(bytes) {");
  assert.ok(decodeTo16kIdx !== -1, "decodeTo16k function must exist");

  const decodeBody = appSource.slice(decodeTo16kIdx, decodeTo16kIdx + 1200);
  assert.ok(decodeBody.includes("getOrCreateAudioContext()"), "decodeTo16k must use getOrCreateAudioContext()");
  assert.ok(!decodeBody.includes("probe.close()"), "decodeTo16k must not create and close temporary probe context");
});

check("2.4 Interruption & Visibility Handling: handles visibilitychange and audio state change", () => {
  assert.ok(appSource.includes('document.addEventListener("visibilitychange"'), "visibilitychange listener must exist");
  assert.ok(appSource.includes("initAudioUnlockGesture()"), "initAudioUnlockGesture must be invoked");
  assert.ok(appSource.includes('pointerdown'), "pointerdown unlock listener must be attached");
  assert.ok(appSource.includes('touchend'), "touchend unlock listener must be attached");
  assert.ok(appSource.includes("onstatechange"), "onstatechange handler must track interrupted state");
});

// ============================================================================
// SECTION 3: Forensic Verification of wasm/demo/index.html (Feature F9)
// ============================================================================
console.log("\n--- Section 3: iPhone 16 Pro Max Touch UX (wasm/demo/index.html) ---");

const indexHtmlSource = fs.readFileSync("wasm/demo/index.html", "utf8");

check("3.1 Viewport Meta: viewport-fit=cover present with width=device-width", () => {
  const viewportMatch = indexHtmlSource.match(/<meta\s+name=["']viewport["']\s+content=["']([^"']+)["']/i);
  assert.ok(viewportMatch, "Viewport meta tag must exist");
  const content = viewportMatch[1];
  assert.ok(content.includes("width=device-width"), "Must include width=device-width");
  assert.ok(content.includes("viewport-fit=cover"), "Must include viewport-fit=cover");
});

check("3.2 Safe Area Insets: env(safe-area-inset-*) dynamically applied", () => {
  assert.ok(indexHtmlSource.includes("safe-area-inset-top"), "Must declare safe-area-inset-top");
  assert.ok(indexHtmlSource.includes("safe-area-inset-bottom"), "Must declare safe-area-inset-bottom");
  assert.ok(indexHtmlSource.includes("safe-area-inset-left"), "Must declare safe-area-inset-left");
  assert.ok(indexHtmlSource.includes("safe-area-inset-right"), "Must declare safe-area-inset-right");
});

check("3.3 Apple HIG 44pt Touch Targets: buttons, selects, inputs have min-height/width >= 44px", () => {
  assert.ok(indexHtmlSource.includes("min-height: 44px"), "Interactive elements must have min-height: 44px");
  assert.ok(indexHtmlSource.includes("min-width: 44px"), "Interactive elements must have min-width: 44px");
});

check("3.4 Form Input Font Size: font-size >= 16px to prevent Mobile Safari auto-zoom", () => {
  const inputRuleMatch = indexHtmlSource.match(/select,\s*input\[type=number\][^{]*\{([^}]+)\}/);
  assert.ok(inputRuleMatch, "CSS rule for select and input must exist");
  assert.ok(inputRuleMatch[1].includes("font-size: 16px"), "Font size must be 16px");
});

check("3.5 Touch Feedback & Layout: touch-action manipulation, 100dvh, and mobile card queries", () => {
  assert.ok(indexHtmlSource.includes("touch-action: manipulation"), "Must set touch-action: manipulation");
  assert.ok(indexHtmlSource.includes("-webkit-tap-highlight-color: transparent"), "Must remove gray tap highlight");
  assert.ok(indexHtmlSource.includes("100dvh"), "Must support dynamic viewport height 100dvh");
  assert.ok(indexHtmlSource.includes("@media (max-width: 640px)"), "Must provide mobile responsive layout");
  assert.ok(indexHtmlSource.includes("@media (max-height: 500px) and (orientation: landscape)"), "Must support compact landscape");
});

// ============================================================================
// SECTION 4: Forensic Absence of Prohibited Patterns (General Profile)
// ============================================================================
console.log("\n--- Section 4: Prohibited Pattern Scan (General Profile) ---");

check("4.1 No hardcoded test outputs or fake tokens in implementation files", () => {
  const files = ["wasm/demo/app.js", "wasm/demo/mic-worklet.js", "wasm/demo/index.html"];
  for (const f of files) {
    const src = fs.readFileSync(f, "utf8");
    // Verify no mock transcriptions
    assert.ok(!src.includes("ask not what your country can do for you"), `No mock transcript in ${f}`);
    assert.ok(!src.includes("dummy"), `No dummy constants in ${f}`);
    assert.ok(!src.includes("fake"), `No fake tokens in ${f}`);
  }
});

check("4.2 No facade or empty stub functions in M3 implementation", () => {
  // Verify MicCollector methods are non-trivial
  assert.ok(micWorkletSource.includes("this.filterSample(ch[i])"));
  assert.ok(micWorkletSource.includes("this.phase += this.ratio"));
  assert.ok(micWorkletSource.includes("this.prevSample = filtered[L - 1]"));
});

// ============================================================================
// Summary
// ============================================================================
console.log("\n==================================================================");
console.log(`  ALL ${passedChecks}/${totalChecks} FORENSIC VERIFICATION CHECKS PASSED!`);
console.log("  VERDICT: CLEAN");
console.log("==================================================================\n");
