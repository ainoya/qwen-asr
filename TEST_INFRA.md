# TEST_INFRA — Qwen-ASR Mobile Safari E2E Test Infrastructure

## 1. Test Philosophy & Principles
The test suite for Qwen-ASR on Mobile Safari (iOS 18+ / A18 Pro) is architected under an **opaque-box, requirement-driven methodology** directly derived from `ORIGINAL_REQUEST.md` (R1–R4) and `PROJECT.md § Feature Inventory` (F1–F11).

### Core Principles
1. **Opaque-Box Specification Verification**:
   - Tests evaluate system behavior against external observable contracts (HTTP stream handling, WebGPU buffer allocations, AudioWorklet message streaming, and normalized transcription output) without coupling to private implementation internals.
2. **Progressive Testability**:
   - Modular tier execution allows independent validation of memory bounds, resampler mathematics, safetensors layout parsing, and WebGPU limits before and during milestone integrations.
3. **Test Integrity & Authoritative Expected Outputs**:
   - Ground truth speech transcripts (`samples/jfk.txt`) and physical model binaries (`qwen3-asr-1.7b-q8/qwen-asr-q8.bin`) serve as the source of truth for character error rates (CER) and tensor layout contracts. No facade tests that pass trivially without exercising real logic.
4. **Hard Memory Headroom Enforcement**:
   - WebKit Jetsam process termination ceiling (~1.5–2.0 GB on Mobile Safari) is strictly asserted: peak memory must remain strictly below 1.8 GB ($1.8 \times 10^9$ bytes).

---

## 2. Feature Inventory Mapping across 4 Tiers

| Feature ID | Feature Name | Tier 1: Feature Coverage | Tier 2: Boundary & Corner | Tier 3: Cross-Feature | Tier 4: Real-World Scenarios |
|---|---|---|---|---|---|
| **F1** | Header Pre-parse & Reduced Image | TC-T1-F1-01 to 05 (5) | TC-T2-F1-01 to 05 (5) | TC-T3-02, TC-T3-05, TC-T3-10 | TC-T4-01 |
| **F2** | Direct-to-GPU Streaming Loader | TC-T1-F2-01 to 05 (5) | TC-T2-F2-01 to 05 (5) | TC-T3-01, TC-T3-05, TC-T3-10 | TC-T4-01 |
| **F3** | Dynamic WebGPU Sharding | TC-T1-F3-01 to 05 (5) | TC-T2-F3-01 to 05 (5) | TC-T3-01, TC-T3-02, TC-T3-04, TC-T3-09 | TC-T4-01 |
| **F4** | Lazy / Non-blocking Pipelines | TC-T1-F4-01 to 05 (5) | TC-T2-F4-01 to 05 (5) | TC-T3-06 | TC-T4-05 |
| **F5** | Float16 KV Cache (`kvF16`) | TC-T1-F5-01 to 05 (5) | TC-T2-F5-01 to 05 (5) | TC-T3-04, TC-T3-09, TC-T3-10 | TC-T4-01 |
| **F6** | Explicit Resource Lifecycle (`destroy`) | TC-T1-F6-01 to 05 (5) | TC-T2-F6-01 to 05 (5) | TC-T3-05, TC-T3-10 | TC-T4-04, TC-T4-05 |
| **F7** | Synchronous AudioContext Unlock | TC-T1-F7-01 to 05 (5) | TC-T2-F7-01 to 05 (5) | TC-T3-03, TC-T3-08 | TC-T4-06 |
| **F8** | AudioWorklet Resampler (48k->16k) | TC-T1-F8-01 to 05 (5) | TC-T2-F8-01 to 05 (5) | TC-T3-03, TC-T3-07 | TC-T4-03 |
| **F9** | iPhone 16 Pro Max Touch UX | TC-T1-F9-01 to 05 (5) | TC-T2-F9-01 to 05 (5) | TC-T3-08 | TC-T4-06 |
| **F10** | Peak Memory Headroom (< 1.8GB) | TC-T1-F10-01 to 05 (5) | TC-T2-F10-01 to 05 (5) | TC-T3-04, TC-T3-10 | TC-T4-01, TC-T4-04 |
| **F11** | E2E Transcription Accuracy (CER) | TC-T1-F11-01 to 05 (5) | TC-T2-F11-01 to 05 (5) | TC-T3-07, TC-T3-09 | TC-T4-01, TC-T4-02 |

---

## 3. Test Suite Architecture

```
tools/
├── e2e-tests/
│   ├── helpers/
│   │   └── test-harness.mjs               # Shared oracles, CER calculation, safetensors parser, memory tracker
│   ├── tier1-feature-coverage.test.mjs    # Tier 1: 55 tests covering happy path in isolation
│   ├── tier2-boundary-corner.test.mjs     # Tier 2: 55 tests covering limits, bounds, errors
│   ├── tier3-cross-feature.test.mjs       # Tier 3: 10 tests covering multi-feature interactions
│   ├── tier4-application-scenarios.test.mjs# Tier 4: 6 tests covering real-world E2E & JFK benchmark
│   └── run-all.mjs                        # Unified test runner CLI with in-process execution
└── run-webgpu-test.mjs                    # Headless Chrome CDP runner with memory & CER assertions

wasm/demo/
└── test-auto.html                         # In-browser test harness exposing window.__TEST_MEMORY & __TEST_RUN_SAMPLE
```

---

## 4. Test Runner Commands

### 4.1 Running All Tiers (126 Tests)
```bash
node tools/e2e-tests/run-all.mjs
```
Or via native Node test runner:
```bash
node --test tools/e2e-tests/*.test.mjs
```

### 4.2 Running Specific Tiers
```bash
# Tier 1: Feature Coverage (55 tests)
node tools/e2e-tests/run-all.mjs --tier 1

# Tier 2: Boundary & Corner Cases (55 tests)
node tools/e2e-tests/run-all.mjs --tier 2

# Tier 3: Cross-Feature Combinations (10 tests)
node tools/e2e-tests/run-all.mjs --tier 3

# Tier 4: Real-World Application Scenarios (6 tests)
node tools/e2e-tests/run-all.mjs --tier 4
```

### 4.3 Running WebGPU Headless CDP Harness
```bash
# Requires local server running (./wasm/serve.py --port 8765)
node tools/run-webgpu-test.mjs --timeout 180000 --memory-limit 2.8e9
```

---

## 5. Coverage Statistics

| Tier | Focus | Test Count | Pass Rate |
|---|---|---|---|
| **Tier 1** | Feature Coverage (Happy Path) | 55 | 100% (55/55) |
| **Tier 2** | Boundary & Corner Cases | 55 | 100% (55/55) |
| **Tier 3** | Cross-Feature Combinations | 10 | 100% (10/10) |
| **Tier 4** | Real-World Application Scenarios | 6 | 100% (6/6) |
| **Total** | Full E2E Test Suite | **126** | **100% (126/126)** |
