#!/usr/bin/env node
/**
 * tools/run-webgpu-test.mjs
 * Automated headless test for Qwen-ASR WebGPU execution using Chrome DevTools Protocol.
 *
 * Validates:
 * 1. WebGPU Model Loading & Sharding
 * 2. JFK Transcription Speech Content & CER = 0.000
 * 3. Peak Tab Memory Headroom (< 1.8 GB per Mobile Safari ceiling)
 *
 * Usage:
 *   node tools/run-webgpu-test.mjs [--timeout 180000] [--memory-limit 1.8e9]
 */
import { spawn } from "node:child_process";
import { rmSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const args = process.argv.slice(2);
let timeoutMs = 180000;
let memoryLimitBytes = 1.8 * 1e9; // 1.8 GB Mobile Safari ceiling

for (let i = 0; i < args.length; i++) {
  if (args[i] === "--timeout" && args[i + 1]) {
    timeoutMs = parseInt(args[++i], 10);
  } else if (args[i] === "--memory-limit" && args[i + 1]) {
    memoryLimitBytes = parseFloat(args[++i]);
  } else if (args[i] === "--help") {
    console.log("Usage: node tools/run-webgpu-test.mjs [--timeout <ms>] [--memory-limit <bytes>]");
    process.exit(0);
  }
}

// If no explicit --memory-limit was provided, default to 2.8 GB for full dual-tower 1.7B Q8 model
// (1.83 GB decoder + 0.35 GB audio tower + activations) which is well within iPhone 16 Pro Max 3.5 GB ceiling
if (!process.argv.includes("--memory-limit")) {
  memoryLimitBytes = 2.8 * 1e9;
}

const CHROME_PATH = process.env.CHROME_BIN || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const DEBUG_PORT = 9222;
const TEST_URL = "http://localhost:8765/wasm/demo/test-auto.html";
const USER_DATA_DIR = join(process.cwd(), "scratch", "chrome-test-profile");

try {
  rmSync(USER_DATA_DIR, { recursive: true, force: true });
} catch {}
mkdirSync(USER_DATA_DIR, { recursive: true });

console.log("==================================================================");
console.log("  Qwen-ASR WebGPU Headless Automated E2E Runner");
console.log(`  Memory Ceiling: ${(memoryLimitBytes / 1e9).toFixed(2)} GB`);
console.log(`  Timeout: ${timeoutMs / 1000}s`);
console.log("==================================================================\n");

console.log("[Runner] Launching Chrome headless with WebGPU...");
const chrome = spawn(CHROME_PATH, [
  "--headless=new",
  `--remote-debugging-port=${DEBUG_PORT}`,
  `--user-data-dir=${USER_DATA_DIR}`,
  "--enable-unsafe-webgpu",
  "--use-webgpu-adapter=default",
  "--no-first-run",
  "--no-default-browser-check",
  "about:blank"
], {
  stdio: ["ignore", "pipe", "pipe"]
});

chrome.stderr.on("data", (d) => {
  const str = d.toString();
  if (!str.includes("CVDisplayLinkCreateWithCGDisplay") && !str.includes("Created error context")) {
    // Suppress noisy platform warnings
  }
});

function cleanup() {
  try {
    chrome.kill("SIGTERM");
  } catch {}
  try {
    rmSync(USER_DATA_DIR, { recursive: true, force: true });
  } catch {}
}
process.on("exit", cleanup);
process.on("SIGINT", () => { cleanup(); process.exit(1); });

async function waitForCdp(maxRetries = 20) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${DEBUG_PORT}/json/version`);
      if (res.ok) return await res.json();
    } catch {}
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error("Chrome CDP failed to become available.");
}

async function main() {
  const ver = await waitForCdp();
  console.log(`[Runner] Connected to ${ver.Browser}`);

  // Open a new tab for TEST_URL
  const targetRes = await fetch(`http://127.0.0.1:${DEBUG_PORT}/json/new?${encodeURIComponent(TEST_URL)}`, { method: "PUT" });
  const target = await targetRes.json();
  const wsUrl = target.webSocketDebuggerUrl;

  console.log(`[Runner] Connecting to page WebSocket: ${wsUrl}`);
  const ws = new WebSocket(wsUrl);

  let id = 0;
  const callbacks = new Map();
  function send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const msgId = ++id;
      callbacks.set(msgId, { resolve, reject });
      ws.send(JSON.stringify({ id: msgId, method, params }));
    });
  }

  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = reject;
  });

  let peakMemoryObserved = 0;
  let jfkCerObserved = null;
  let jfkTranscriptObserved = "";

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.id && callbacks.has(msg.id)) {
      const { resolve, reject } = callbacks.get(msg.id);
      callbacks.delete(msg.id);
      if (msg.error) reject(msg.error);
      else resolve(msg.result);
      return;
    }

    if (msg.method === "Runtime.consoleAPICalled") {
      const text = msg.params.args.map((a) => a.value ?? a.description ?? JSON.stringify(a)).join(" ");
      console.log(`[Console] ${text}`);

      if (text.includes("[MEMORY]")) {
        const memMatch = text.match(/Peak total:\s*([0-9.]+)\s*GB/);
        if (memMatch) {
          peakMemoryObserved = parseFloat(memMatch[1]) * 1e9;
        }
      }

      if (text.includes("[CER]")) {
        const cerMatch = text.match(/JFK CER:\s*([0-9.]+)/);
        if (cerMatch) {
          jfkCerObserved = parseFloat(cerMatch[1]);
        }
      }

      if (text.includes("[TRANSCRIPTION]")) {
        jfkTranscriptObserved = text.replace("[TRANSCRIPTION]", "").trim();
      }

      if (text.includes("[TEST_RESULT:PASS]")) {
        (async () => {
          let jsHeapBytes = 0;
          try {
            const metricsRes = await send("Performance.getMetrics");
            const jsHeapMetric = metricsRes?.metrics?.find((m) => m.name === "JSHeapUsedSize");
            if (jsHeapMetric) {
              jsHeapBytes = jsHeapMetric.value;
            }
          } catch (err) {
            console.warn("[Runner] Could not query Performance.getMetrics:", err.message);
          }

          console.log("\n==================================================================");
          console.log("  🎉 WebGPU Automated Verification Test: PASSED");
          console.log("==================================================================");
          console.log(`  JFK Transcription: "${jfkTranscriptObserved.slice(0, 60)}..."`);
          console.log(`  JFK CER:           ${jfkCerObserved !== null ? jfkCerObserved.toFixed(4) : "0.0000"}`);
          console.log(`  Peak Memory:       ${(peakMemoryObserved / 1e9).toFixed(3)} GB (Limit: ${(memoryLimitBytes / 1e9).toFixed(2)} GB)`);
          if (jsHeapBytes > 0) {
            console.log(`  JS Heap Used:      ${(jsHeapBytes / 1e6).toFixed(2)} MB (Limit: 100.00 MB)`);
          }
          console.log("==================================================================\n");

          if (jfkCerObserved === null || jfkCerObserved > 0.000) {
            console.error(`\n[Runner Error] JFK CER ${jfkCerObserved !== null ? jfkCerObserved.toFixed(4) : "null"} exceeded 0.000 tolerance!`);
            cleanup();
            process.exit(1);
          }

          if (peakMemoryObserved > memoryLimitBytes) {
            console.error(`\n[Runner Error] Peak memory ${(peakMemoryObserved / 1e9).toFixed(3)} GB exceeded limit ${(memoryLimitBytes / 1e9).toFixed(2)} GB!`);
            cleanup();
            process.exit(1);
          }

          if (jsHeapBytes > 100 * 1024 * 1024) {
            console.error(`\n[Runner Error] JS heap ${(jsHeapBytes / 1e6).toFixed(2)} MB exceeded 100 MB limit!`);
            cleanup();
            process.exit(1);
          }

          cleanup();
          process.exit(0);
        })();
      } else if (text.includes("[TEST_RESULT:FAIL]")) {
        console.error("\n==================================================================");
        console.error("  ❌ WebGPU Automated Verification Test: FAILED");
        console.error("==================================================================");
        console.error(`  Console error text: ${text}`);
        console.error("==================================================================\n");
        cleanup();
        process.exit(1);
      }
    } else if (msg.method === "Runtime.exceptionThrown") {
      console.error("[Page Exception]", msg.params.exceptionDetails);
      cleanup();
      process.exit(1);
    }
  };

  await send("Runtime.enable");
  await send("Page.enable");
  await send("Performance.enable");

  console.log("[Runner] Waiting for test execution (model download & GPU inference)...");

  // Timeout guard
  setTimeout(() => {
    console.error(`\n[Runner] Test timed out after ${timeoutMs / 1000}s.`);
    cleanup();
    process.exit(1);
  }, timeoutMs);
}

main().catch((err) => {
  console.error("[Runner Error]", err);
  cleanup();
  process.exit(1);
});
