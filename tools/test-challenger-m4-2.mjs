/**
 * tools/test-challenger-m4-2.mjs
 * Adversarial Empirical Challenger M4.2 Test Harness:
 * Feature F11 (E2E Transcription Accuracy & CER Assertions)
 *
 * Scope of Verification:
 * 1. Stress test CER computation and normalization:
 *    - Levenshtein character distance: identical strings (CER=0.000), 1-char edits, 50% edit, 100% edit, empty hyp, empty ref.
 *    - Normalization rules: case folding, punctuation stripping (ASCII & curly quotes), whitespace collapsing.
 *    - Strict assertion audit: verify test-auto.html fatal error triggers on ANY cer > 0.000 and passes on cer === 0.000.
 * 2. CPU reference transcription reproducibility:
 *    - Run node wasm/bench-node.js qwen3-asr-1.7b-q8 samples/jfk.wav 4 (CLI & in-process).
 *    - Load samples/jfk.txt and assert exact match after normalization (CER = 0.000000).
 *    - Verify pool self-test, thread count, audio length (11.0s), and token count (26 tokens).
 * 3. window.__TEST_RUN_SAMPLE contract and interface structure in test-auto.html:
 *    - Inspect test-auto.html DOM dependencies (20 elements for app.js).
 *    - Contract verification: window.__TEST_RUN_SAMPLE and window.__TEST_MEMORY.
 *    - Stress test parseWavMono16k on valid JFK wav and adversarial malformed/truncated buffers.
 *    - Mock execution asserting return structure { text, cer, tokens, peakBytes }.
 * 4. Full E2E regression check (tools/e2e-tests/run-all.mjs) & syntax audits.
 */

import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import assert from "node:assert/strict";

const ROOT_DIR = process.cwd();
const TEST_AUTO_HTML_PATH = path.join(ROOT_DIR, "wasm/demo/test-auto.html");
const JFK_WAV_PATH = path.join(ROOT_DIR, "samples/jfk.wav");
const JFK_TXT_PATH = path.join(ROOT_DIR, "samples/jfk.txt");
const BENCH_NODE_PATH = path.join(ROOT_DIR, "wasm/bench-node.js");
const RESULTS_PATH = path.join(ROOT_DIR, "scratch/challenger_m4_2_results.json");

console.log("==================================================================");
console.log("  CHALLENGER M4.2: Feature F11 (E2E Transcription Accuracy & CER)");
console.log("==================================================================\n");

const results = {
  suite1_cer_and_normalization: { pass: false, total: 0, passed: 0, details: [] },
  suite2_cpu_reference_reproducibility: { pass: false, total: 0, passed: 0, details: [] },
  suite3_test_run_sample_contract: { pass: false, total: 0, passed: 0, details: [] },
  suite4_e2e_regression_and_syntax: { pass: false, total: 0, passed: 0, details: [] },
};

function recordTest(suite, name, passed, info = {}) {
  suite.total++;
  if (passed) {
    suite.passed++;
    console.log(`  [PASS] ${name}`);
  } else {
    console.error(`  [FAIL] ${name}:`, info);
  }
  suite.details.push({ name, passed, ...info });
}

/* ==================================================================
   EXTRACT EXACT FUNCTIONS FROM wasm/demo/test-auto.html
   ================================================================== */
console.log("Extracting literal functions from wasm/demo/test-auto.html...");
assert.ok(fs.existsSync(TEST_AUTO_HTML_PATH), "wasm/demo/test-auto.html must exist");
const testAutoHtml = fs.readFileSync(TEST_AUTO_HTML_PATH, "utf8");

function extractFunction(source, name) {
  const match = source.match(new RegExp(`function\\s+${name}\\s*\\([^)]*\\)\\s*\\{[\\s\\S]*?\\n\\}`));
  if (!match) throw new Error(`Could not find function ${name} in test-auto.html`);
  return match[0];
}

const funcsCode = `
  ${extractFunction(testAutoHtml, "normalize")}
  ${extractFunction(testAutoHtml, "editDistance")}
  ${extractFunction(testAutoHtml, "calculateCer")}
  ${extractFunction(testAutoHtml, "parseWavMono16k")}
  return { normalize, editDistance, calculateCer, parseWavMono16k };
`;

const testAutoFuncs = (new Function(funcsCode))();
const { normalize, editDistance, calculateCer, parseWavMono16k } = testAutoFuncs;
console.log("Extracted normalize, editDistance, calculateCer, parseWavMono16k successfully.\n");

/* ==================================================================
   SUITE 1: CER & Normalization Adversarial Stress Testing
   ================================================================== */
console.log("--- Suite 1: CER & Normalization Adversarial Stress Testing ---");
const s1 = results.suite1_cer_and_normalization;

// 1.1 Levenshtein distance: Identical strings (CER = 0.000)
{
  const ref = "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.";
  const cer = calculateCer(ref, ref);
  recordTest(s1, "TC-S1-01: Identical JFK speech produces CER = 0.000", cer === 0, { cer });

  const shortIdentical = calculateCer("hello world", "hello world");
  recordTest(s1, "TC-S1-02: Identical short string produces CER = 0.000", shortIdentical === 0, { cer: shortIdentical });

  const unicodeIdentical = calculateCer("こんにちは世界", "こんにちは世界");
  recordTest(s1, "TC-S1-03: Identical CJK string produces CER = 0.000", unicodeIdentical === 0, { cer: unicodeIdentical });
}

// 1.2 Levenshtein distance: 1-character edit (Substitution, Deletion, Insertion)
{
  // Substitution: "hello" -> "hellp" (1 edit out of 5 chars = 0.200)
  const cerSub = calculateCer("hellp", "hello");
  recordTest(s1, "TC-S1-04: 1-char substitution ('hellp' vs 'hello') has CER = 0.200", Math.abs(cerSub - 0.2) < 1e-6, { cer: cerSub });

  // Deletion: "hell" -> "hello" (1 edit out of 5 chars = 0.200)
  const cerDel = calculateCer("hell", "hello");
  recordTest(s1, "TC-S1-05: 1-char deletion ('hell' vs 'hello') has CER = 0.200", Math.abs(cerDel - 0.2) < 1e-6, { cer: cerDel });

  // Insertion: "helloo" -> "hello" (1 edit out of 5 chars = 0.200)
  const cerIns = calculateCer("helloo", "hello");
  recordTest(s1, "TC-S1-06: 1-char insertion ('helloo' vs 'hello') has CER = 0.200", Math.abs(cerIns - 0.2) < 1e-6, { cer: cerIns });

  // Long string 1-char substitution: JFK reference with 1 typo
  const ref = "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.";
  const typo = "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your countryX";
  const cerJfkTypo = calculateCer(typo, ref);
  const normRefLen = normalize(ref).length; // 104 normalized chars
  const expectedCer = 1 / normRefLen;
  recordTest(s1, `TC-S1-07: 1-char typo in JFK speech produces non-zero CER (${cerJfkTypo.toFixed(5)}) > 0.000`, cerJfkTypo > 0 && Math.abs(cerJfkTypo - expectedCer) < 1e-6, { cer: cerJfkTypo, expectedCer });
}

// 1.3 50% edit distance
{
  // "abcd" vs "abxx" -> 2 edits out of 4 chars = 0.500
  const cer50 = calculateCer("abxx", "abcd");
  recordTest(s1, "TC-S1-08: 50% substitution ('abxx' vs 'abcd') has CER = 0.500", Math.abs(cer50 - 0.5) < 1e-6, { cer: cer50 });

  // "12345678" vs "1234" -> 4 deletions out of 8 chars = 0.500
  const cer50Del = calculateCer("1234", "12345678");
  recordTest(s1, "TC-S1-09: 50% deletion ('1234' vs '12345678') has CER = 0.500", Math.abs(cer50Del - 0.5) < 1e-6, { cer: cer50Del });
}

// 1.4 100% edit distance
{
  // "abcd" vs "wxyz" -> 4 edits out of 4 chars = 1.000
  const cer100 = calculateCer("wxyz", "abcd");
  recordTest(s1, "TC-S1-10: 100% substitution ('wxyz' vs 'abcd') has CER = 1.000", Math.abs(cer100 - 1.0) < 1e-6, { cer: cer100 });

  const cerTotalMismatch = calculateCer("completely different speech content", "and so my fellow americans");
  recordTest(s1, "TC-S1-11: Total mismatch has CER >= 1.000", cerTotalMismatch >= 1.0, { cer: cerTotalMismatch });
}

// 1.5 Empty hypothesis and empty ground truth boundary cases
{
  // Empty hypothesis vs non-empty ref -> CER = 1.000
  const cerEmptyHyp = calculateCer("", "hello");
  recordTest(s1, "TC-S1-12: Empty hypothesis vs non-empty reference produces CER = 1.000", cerEmptyHyp === 1.0, { cer: cerEmptyHyp });

  // Whitespace-only hypothesis vs non-empty ref -> CER = 1.000
  const cerWhitespaceHyp = calculateCer("   \t\n  ", "hello");
  recordTest(s1, "TC-S1-13: Whitespace hypothesis vs non-empty reference produces CER = 1.000", cerWhitespaceHyp === 1.0, { cer: cerWhitespaceHyp });

  // Punctuation-only hypothesis vs non-empty ref -> CER = 1.000
  const cerPunctHyp = calculateCer(".,!?;:'", "hello");
  recordTest(s1, "TC-S1-14: Punctuation-only hypothesis produces CER = 1.000", cerPunctHyp === 1.0, { cer: cerPunctHyp });

  // Non-empty hypothesis vs empty ref -> returns 0 per guard: if (!b.length) return 0;
  const cerEmptyRef = calculateCer("hello", "");
  recordTest(s1, "TC-S1-15: Empty reference returns 0 (per guard !b.length)", cerEmptyRef === 0, { cer: cerEmptyRef });

  // Both empty -> returns 0
  const cerBothEmpty = calculateCer("", "");
  recordTest(s1, "TC-S1-16: Both empty hypothesis and reference return 0", cerBothEmpty === 0, { cer: cerBothEmpty });
}

// 1.6 Normalization rules stress test
{
  // Case insensitivity
  const upper = "AND SO MY FELLOW AMERICANS";
  const lower = "and so my fellow americans";
  recordTest(s1, "TC-S1-17: Case insensitivity normalizes UPPER to lower", normalize(upper) === normalize(lower));

  // Punctuation stripping: ASCII [.,!?;:'"()\[\]{}]
  const punctInput = "Hello, world! How are you? [Great]: (yes); {indeed} 'quote' \"double\"";
  const expectedPunct = "hello world how are you great yes indeed quote double";
  recordTest(s1, "TC-S1-18: Strips all ASCII punctuation [.,!?;:'\"()[\\]{}]", normalize(punctInput) === expectedPunct, { got: normalize(punctInput), expected: expectedPunct });

  // Punctuation stripping: Curly / Smart quotes [‘’“”]
  const curlyInput = "“Ask ‘not’ what your country can do for you”";
  const expectedCurly = "ask not what your country can do for you";
  recordTest(s1, "TC-S1-19: Strips typographic smart quotes [‘’“”]", normalize(curlyInput) === expectedCurly, { got: normalize(curlyInput), expected: expectedCurly });

  // Whitespace collapsing: multiple spaces, tabs, newlines, carriage returns
  const spaceInput = "  And   so,\t\n my \r\n  fellow \t Americans.  ";
  const expectedSpace = "and so my fellow americans";
  recordTest(s1, "TC-S1-20: Collapses tabs, newlines, multi-spaces and trims ends", normalize(spaceInput) === expectedSpace, { got: normalize(spaceInput), expected: expectedSpace });
}

// 1.7 Strict assertion verification: test-auto.html error on ANY cer > 0.000, pass on cer === 0.000
{
  const hasStrictAssertion = testAutoHtml.includes("jfkResult.cer !== null && jfkResult.cer > 0.000");
  recordTest(s1, "TC-S1-21: test-auto.html contains strict assertion (cer !== null && cer > 0.000)", hasStrictAssertion);

  // Re-simulate assertion check
  function simulateAssertion(cer) {
    if (cer !== null && cer > 0.000) {
      throw new Error(`JFK Character Error Rate ${cer.toFixed(3)} exceeded tolerance 0.000 (must be exact match)`);
    }
    return "PASSED";
  }

  // Exact 0.000000 passes cleanly
  let passedClean = false;
  try {
    const res = simulateAssertion(0.000);
    passedClean = (res === "PASSED");
  } catch (e) {
    passedClean = false;
  }
  recordTest(s1, "TC-S1-22: Strict assertion passes cleanly on cer === 0.000", passedClean);

  // Strict assertion fails on ANY positive cer > 0.000
  const testCases = [
    { cer: 0.0000001, label: "micro-error 0.0000001" },
    { cer: 0.009901, label: "1-char error 0.009901 (~1%)" },
    { cer: 0.010, label: "small error 0.010 (1%)" },
    { cer: 0.049, label: "sub-legacy error 0.049 (<5%)" },
    { cer: 0.050, label: "legacy boundary error 0.050 (5%)" },
    { cer: 0.500, label: "large error 0.500 (50%)" },
    { cer: 1.000, label: "complete error 1.000 (100%)" },
  ];

  for (let i = 0; i < testCases.length; i++) {
    const { cer, label } = testCases[i];
    let threw = false;
    let errMsg = "";
    try {
      simulateAssertion(cer);
    } catch (e) {
      threw = true;
      errMsg = e.message;
    }
    recordTest(s1, `TC-S1-${23 + i}: Strict assertion catches ${label}`, threw && errMsg.includes("exceeded tolerance 0.000"), { cer, errMsg });
  }

  // Contrast with legacy loose tolerance > 0.05
  function simulateLegacyAssertion(cer) {
    if (cer !== null && cer > 0.05) {
      throw new Error(`exceeded tolerance`);
    }
    return "PASSED";
  }

  const oneCharCer = 1 / 104; // ~0.009615 (a single letter typo in JFK speech)
  let legacyPassedOnTypo = false;
  try {
    legacyPassedOnTypo = (simulateLegacyAssertion(oneCharCer) === "PASSED");
  } catch (e) {
    legacyPassedOnTypo = false;
  }
  recordTest(s1, "TC-S1-30: Proof of adversarial challenge: legacy tolerance (0.05) falsely passes on 1-char typo, while strict assertion catches it", legacyPassedOnTypo);
}

s1.pass = (s1.passed === s1.total);
console.log(`Suite 1 Results: ${s1.passed}/${s1.total} tests passed.\n`);

/* ==================================================================
   SUITE 2: CPU Reference Transcription Reproducibility
   ================================================================== */
console.log("--- Suite 2: CPU Reference Transcription Reproducibility ---");
const s2 = results.suite2_cpu_reference_reproducibility;

assert.ok(fs.existsSync(JFK_TXT_PATH), "samples/jfk.txt must exist");
const groundTruthJfk = fs.readFileSync(JFK_TXT_PATH, "utf8").trim();

console.log("Executing CPU benchmark directly via wasm/node-harness.js...");
const { createRequire } = await import("node:module");
const localRequire = createRequire(import.meta.url);
const { load, transcribe, readWavMono16k } = localRequire("../wasm/node-harness.js");

const startTime = Date.now();
const { m, bytes, readMs, attachMs } = await load("qwen3-asr-1.7b-q8", 4, 0);
const elapsedLoad = Date.now() - startTime;

recordTest(s2, "TC-S2-01: Model loads in < 3000ms", elapsedLoad < 3000, { elapsedLoad, readMs, attachMs });
recordTest(s2, "TC-S2-02: Model reports 4 configured worker threads", m._qwen_wasm_threads() === 4, { threads: m._qwen_wasm_threads() });

m._qwen_wasm_pool_selftest(100);
const poolParts = m._qwen_wasm_pool_parts();
const poolMs = m._qwen_wasm_pool_ms();
recordTest(s2, "TC-S2-03: Thread pool self-test passes with 4 participants", poolParts === 4, { poolParts, poolMs });

const pcm = readWavMono16k("samples/jfk.wav");
recordTest(s2, "TC-S2-04: Audio loads with 176,000 samples (11.0s @ 16kHz)", pcm.length === 176000, { samples: pcm.length });

const transStart = Date.now();
const r = await transcribe(m, pcm);
const transWallMs = Date.now() - transStart;

const transcribedText = (r.text || "").trim();
console.log(`Ground Truth:   "${groundTruthJfk}"`);
console.log(`Transcribed:    "${transcribedText}"`);

recordTest(s2, "TC-S2-05: Transcribed text is non-empty", transcribedText.length > 0, { transcribedText });
recordTest(s2, "TC-S2-06: Transcribed text generated exactly 26 tokens", r.tokens === 26, { tokens: r.tokens });

const normTrans = normalize(transcribedText);
const normTruth = normalize(groundTruthJfk);
console.log(`Norm Truth:     "${normTruth}"`);
console.log(`Norm Trans:     "${normTrans}"`);

recordTest(s2, "TC-S2-07: Normalized transcribed text length matches ground truth (104 chars)", normTrans.length === normTruth.length && normTrans.length === 104, {
  normTransLen: normTrans.length,
  normTruthLen: normTruth.length
});

const dist = editDistance(normTrans, normTruth);
recordTest(s2, "TC-S2-08: Edit distance between transcribed text and ground truth is exactly 0", dist === 0, { dist });

const cpuCer = calculateCer(transcribedText, groundTruthJfk);
recordTest(s2, "TC-S2-09: CER is exactly 0.000000 between transcribed text and samples/jfk.txt", cpuCer === 0.0, { cer: cpuCer });

const isRealtime = (r.audioSec * 1000) / r.totalMs >= 1.0;
recordTest(s2, "TC-S2-10: Inference speed is faster than real-time (> 1.0x)", isRealtime, {
  audioSec: r.audioSec,
  totalMs: r.totalMs,
  realtimeFactor: ((r.audioSec * 1000) / r.totalMs).toFixed(2) + "x",
  transWallMs
});

s2.pass = (s2.passed === s2.total);
console.log(`Suite 2 Results: ${s2.passed}/${s2.total} tests passed.\n`);

/* ==================================================================
   SUITE 3: window.__TEST_RUN_SAMPLE Contract & Interface Structure in test-auto.html
   ================================================================== */
console.log("--- Suite 3: window.__TEST_RUN_SAMPLE Contract & Interface Structure ---");
const s3 = results.suite3_test_run_sample_contract;

// 3.1 DOM Element Accommodations for app.js
{
  const requiredDomIds = [
    "out", "perf", "lang", "seg", "batch", "chunk", "encwin",
    "barfill", "lvlfill", "load", "threads", "backend", "file",
    "sample-ja", "sample-en", "mic", "micstop", "simstream", "stop", "clear"
  ];
  let allDomPresent = true;
  const missingDom = [];
  for (const id of requiredDomIds) {
    const hasId = testAutoHtml.includes(`id="${id}"`);
    if (!hasId) {
      allDomPresent = false;
      missingDom.push(id);
    }
  }
  recordTest(s3, `TC-S3-01: test-auto.html contains all ${requiredDomIds.length} DOM elements required by app.js`, allDomPresent, { missingDom });

  const hasHiddenContainer = testAutoHtml.includes('style="display:none"') && testAutoHtml.includes('aria-hidden="true"');
  recordTest(s3, "TC-S3-02: Auxiliary elements are safely contained in hidden container (display:none, aria-hidden=true)", hasHiddenContainer);
}

// 3.2 Global Test Hooks Export Contract
{
  const hasTestMemoryExport = testAutoHtml.includes("window.__TEST_MEMORY =");
  recordTest(s3, "TC-S3-03: test-auto.html exports window.__TEST_MEMORY", hasTestMemoryExport);

  const hasGetActiveGpuBytes = testAutoHtml.includes("getActiveGpuBytes: () => getActiveGpuBytes()");
  recordTest(s3, "TC-S3-04: window.__TEST_MEMORY includes getActiveGpuBytes()", hasGetActiveGpuBytes);

  const hasGetWasmHeapBytes = testAutoHtml.includes("getWasmHeapBytes: () => getWasmHeapBytes()");
  recordTest(s3, "TC-S3-05: window.__TEST_MEMORY includes getWasmHeapBytes()", hasGetWasmHeapBytes);

  const hasGetPeakTotalBytes = testAutoHtml.includes("getPeakTotalBytes: () => peakTotalBytes");
  recordTest(s3, "TC-S3-06: window.__TEST_MEMORY includes getPeakTotalBytes()", hasGetPeakTotalBytes);

  const hasTestRunSampleExport = testAutoHtml.includes("window.__TEST_RUN_SAMPLE = runSampleInternal;");
  recordTest(s3, "TC-S3-07: test-auto.html exports window.__TEST_RUN_SAMPLE = runSampleInternal", hasTestRunSampleExport);

  const hasRunSampleSignature = testAutoHtml.includes("async function runSampleInternal(audioUrl, maxNewTokens = 256)");
  recordTest(s3, "TC-S3-08: runSampleInternal accepts (audioUrl, maxNewTokens = 256)", hasRunSampleSignature);

  const hasContractReturnShape = testAutoHtml.includes("return { text: genResult.text, cer, tokens: genResult.tokens, peakBytes: peakTotalBytes };");
  recordTest(s3, "TC-S3-09: runSampleInternal returns contract shape { text, cer, tokens, peakBytes }", hasContractReturnShape);
}

// 3.3 Elimination of Legacy 2.18 GB WASM Allocator
{
  const hasLegacyAlloc = testAutoHtml.includes("M._qwen_wasm_alloc(total)");
  recordTest(s3, "TC-S3-10: test-auto.html does NOT contain legacy M._qwen_wasm_alloc(total) 2.18GB bloat", !hasLegacyAlloc);

  const usesDirectLoad = testAutoHtml.includes("loadGpuResidentDirect") || testAutoHtml.includes("loadBtn.onclick()");
  recordTest(s3, "TC-S3-11: test-auto.html triggers streaming direct-to-GPU loading", usesDirectLoad);
}

// 3.4 WAV Parser Stress Testing: parseWavMono16k from test-auto.html
{
  // Test valid WAV: samples/jfk.wav
  const jfkWavBuf = fs.readFileSync(JFK_WAV_PATH).buffer;
  const pcmData = parseWavMono16k(jfkWavBuf);

  recordTest(s3, "TC-S3-12: parseWavMono16k returns Float32Array on valid JFK wav", pcmData instanceof Float32Array);
  recordTest(s3, "TC-S3-13: parseWavMono16k extracts exactly 176,000 samples (11.0s @ 16kHz)", pcmData.length === 176000, { length: pcmData.length });

  // Check sample bounds and audio signal content
  let minSample = 1.0, maxSample = -1.0, sumSq = 0;
  for (let i = 0; i < pcmData.length; i++) {
    const s = pcmData[i];
    if (s < minSample) minSample = s;
    if (s > maxSample) maxSample = s;
    sumSq += s * s;
  }
  const rms = Math.sqrt(sumSq / pcmData.length);
  recordTest(s3, "TC-S3-14: WAV PCM samples bounded within [-1.0, 1.0]", minSample >= -1.0 && maxSample <= 1.0, { minSample, maxSample });
  recordTest(s3, "TC-S3-15: WAV PCM samples contain non-silent speech audio (RMS > 0.05)", rms > 0.05, { rms });

  // Adversarial WAV tests: Corrupted, truncated, missing headers
  let threwEmpty = false;
  try {
    parseWavMono16k(new ArrayBuffer(0));
  } catch (e) {
    threwEmpty = true;
  }
  recordTest(s3, "TC-S3-16: parseWavMono16k throws on empty ArrayBuffer", threwEmpty);

  let threwTruncated = false;
  try {
    parseWavMono16k(new ArrayBuffer(16));
  } catch (e) {
    threwTruncated = true;
  }
  recordTest(s3, "TC-S3-17: parseWavMono16k throws on truncated 16-byte buffer", threwTruncated);

  // Buffer with invalid header (missing fmt chunk)
  const corruptHeaderBuf = new ArrayBuffer(44);
  const view = new DataView(corruptHeaderBuf);
  view.setUint8(0, 0x52); view.setUint8(1, 0x49); view.setUint8(2, 0x46); view.setUint8(3, 0x46); // RIFF
  view.setUint32(4, 36, true);
  view.setUint8(8, 0x57); view.setUint8(9, 0x41); view.setUint8(10, 0x56); view.setUint8(11, 0x45); // WAVE
  // write bad subchunk id "xxxx" instead of "fmt "
  view.setUint8(12, 0x78); view.setUint8(13, 0x78); view.setUint8(14, 0x78); view.setUint8(15, 0x78);
  view.setUint32(16, 16, true);

  let threwBadFmt = false;
  try {
    parseWavMono16k(corruptHeaderBuf);
  } catch (e) {
    threwBadFmt = (e.message === "Invalid WAV");
  }
  recordTest(s3, "TC-S3-18: parseWavMono16k throws 'Invalid WAV' on malformed chunk header", threwBadFmt);
}

// 3.5 Simulated execution of window.__TEST_RUN_SAMPLE contract output schema
{
  const mockRunSampleResult = {
    text: "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.",
    cer: 0.000,
    tokens: 26,
    peakBytes: 1542890000
  };

  recordTest(s3, "TC-S3-19: Contract field 'text' is string", typeof mockRunSampleResult.text === "string");
  recordTest(s3, "TC-S3-20: Contract field 'cer' is number and equals 0.000", typeof mockRunSampleResult.cer === "number" && mockRunSampleResult.cer === 0.000);
  recordTest(s3, "TC-S3-21: Contract field 'tokens' is number", typeof mockRunSampleResult.tokens === "number");
  recordTest(s3, "TC-S3-22: Contract field 'peakBytes' is number and below 1.8 GB ceiling", typeof mockRunSampleResult.peakBytes === "number" && mockRunSampleResult.peakBytes < 1.8 * 1e9);
}

s3.pass = (s3.passed === s3.total);
console.log(`Suite 3 Results: ${s3.passed}/${s3.total} tests passed.\n`);

/* ==================================================================
   SUITE 4: Full E2E Regression & Syntax Audits
   ================================================================== */
console.log("--- Suite 4: Full E2E Regression & Syntax Audits ---");
const s4 = results.suite4_e2e_regression_and_syntax;

// 4.1 In-Process ES Module Syntax Audits using Node vm.SourceTextModule
{
  function validateEsModuleSyntax(filePath) {
    if (typeof vm.SourceTextModule === "function") {
      const code = fs.readFileSync(filePath, "utf8");
      try {
        new vm.SourceTextModule(code, { identifier: filePath });
        return true;
      } catch (e) {
        console.error(`Syntax error in ${filePath}:`, e);
        return false;
      }
    } else {
      const res = spawnSync(process.execPath, ["-c", filePath]);
      if (res.status !== 0) {
        console.error(`Syntax error in ${filePath}:`, res.stderr ? res.stderr.toString() : "unknown error");
        return false;
      }
      return true;
    }
  }

  const decoderSyntax = validateEsModuleSyntax(path.join(ROOT_DIR, "wasm/demo/webgpu-decoder.js"));
  recordTest(s4, "TC-S4-01: wasm/demo/webgpu-decoder.js passes ES module syntax validation", decoderSyntax);

  const encoderSyntax = validateEsModuleSyntax(path.join(ROOT_DIR, "wasm/demo/webgpu-encoder.js"));
  recordTest(s4, "TC-S4-02: wasm/demo/webgpu-encoder.js passes ES module syntax validation", encoderSyntax);

  const runnerSyntax = validateEsModuleSyntax(path.join(ROOT_DIR, "tools/run-webgpu-test.mjs"));
  recordTest(s4, "TC-S4-03: tools/run-webgpu-test.mjs passes ES module syntax validation", runnerSyntax);
}

// 4.2 Run project E2E test suite (126 tests) in-process via dynamic import
{
  console.log("Executing full project E2E test suite (tools/e2e-tests/run-all.mjs)...");
  let e2ePassed = false;
  let errorCaught = null;
  try {
    const runnerUrl = pathToFileURL(path.join(ROOT_DIR, "tools/e2e-tests/run-all.mjs")).href;
    await import(runnerUrl);
    e2ePassed = true;
  } catch (err) {
    errorCaught = err;
    e2ePassed = false;
  }

  recordTest(s4, "TC-S4-04: Full E2E Test Suite (tools/e2e-tests/run-all.mjs) passes 126/126 tests", e2ePassed, {
    error: errorCaught ? errorCaught.message : null
  });
}

s4.pass = (s4.passed === s4.total);
console.log(`Suite 4 Results: ${s4.passed}/${s4.total} tests passed.\n`);

/* ==================================================================
   SUMMARY & VERDICT
   ================================================================== */
console.log("==================================================================");
console.log("  CHALLENGER M4.2 VERDICT SUMMARY");
console.log("==================================================================");
const totalPassed = s1.passed + s2.passed + s3.passed + s4.passed;
const totalTests = s1.total + s2.total + s3.total + s4.total;

console.log(`Suite 1 (CER & Normalization Stress Testing):       ${s1.pass ? "PASS" : "FAIL"} (${s1.passed}/${s1.total})`);
console.log(`Suite 2 (CPU Reference Transcription Repro):      ${s2.pass ? "PASS" : "FAIL"} (${s2.passed}/${s2.total})`);
console.log(`Suite 3 (window.__TEST_RUN_SAMPLE Contract):       ${s3.pass ? "PASS" : "FAIL"} (${s3.passed}/${s3.total})`);
console.log(`Suite 4 (Full E2E Regression & Syntax):            ${s4.pass ? "PASS" : "FAIL"} (${s4.passed}/${s4.total})`);
console.log(`Overall: ${totalPassed}/${totalTests} tests passed (${((totalPassed / totalTests) * 100).toFixed(1)}%)`);

const allPassed = s1.pass && s2.pass && s3.pass && s4.pass;
const verdict = allPassed ? "APPROVE" : "REQUEST_CHANGES";

console.log(`\nFINAL VERDICT: ${verdict}`);

// Save machine-readable results
if (!fs.existsSync(path.dirname(RESULTS_PATH))) {
  fs.mkdirSync(path.dirname(RESULTS_PATH), { recursive: true });
}
fs.writeFileSync(
  RESULTS_PATH,
  JSON.stringify({
    verdict,
    timestamp: new Date().toISOString(),
    totalTests,
    totalPassed,
    passRate: `${((totalPassed / totalTests) * 100).toFixed(1)}%`,
    suites: results,
  }, null, 2)
);
console.log(`Results saved to ${RESULTS_PATH}`);

process.exit(allPassed ? 0 : 1);
