#!/usr/bin/env node
/**
 * tools/run-webgpu-test.mjs
 * Automated headless test for Qwen-ASR WebGPU execution using Chrome DevTools Protocol.
 */
import { spawn } from 'node:child_process';
import { rmSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const DEBUG_PORT = 9222;
const TEST_URL = 'http://localhost:8765/wasm/demo/test-auto.html';
const USER_DATA_DIR = join(process.cwd(), 'scratch', 'chrome-test-profile');

try {
  rmSync(USER_DATA_DIR, { recursive: true, force: true });
} catch {}
mkdirSync(USER_DATA_DIR, { recursive: true });

console.log('[Runner] Launching Chrome headless with WebGPU...');
const chrome = spawn(CHROME_PATH, [
  '--headless=new',
  `--remote-debugging-port=${DEBUG_PORT}`,
  `--user-data-dir=${USER_DATA_DIR}`,
  '--enable-unsafe-webgpu',
  '--use-webgpu-adapter=default',
  '--no-first-run',
  '--no-default-browser-check',
  'about:blank'
], {
  stdio: ['ignore', 'pipe', 'pipe']
});

chrome.stderr.on('data', (d) => {
  const str = d.toString();
  // Filter out noisy CoreAnimation / CVDisplayLink errors on mac headless
  if (!str.includes('CVDisplayLinkCreateWithCGDisplay')) {
    // console.error('[Chrome Stderr]', str.trim());
  }
});

function cleanup() {
  try {
    chrome.kill('SIGTERM');
  } catch {}
  try {
    rmSync(USER_DATA_DIR, { recursive: true, force: true });
  } catch {}
}
process.on('exit', cleanup);
process.on('SIGINT', () => { cleanup(); process.exit(1); });

async function waitForCdp(maxRetries = 20) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${DEBUG_PORT}/json/version`);
      if (res.ok) return await res.json();
    } catch {}
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error('Chrome CDP failed to become available.');
}

async function main() {
  const ver = await waitForCdp();
  console.log(`[Runner] Connected to ${ver.Browser}`);

  // Open a new tab for TEST_URL
  const targetRes = await fetch(`http://127.0.0.1:${DEBUG_PORT}/json/new?${encodeURIComponent(TEST_URL)}`, { method: 'PUT' });
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

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.id && callbacks.has(msg.id)) {
      const { resolve, reject } = callbacks.get(msg.id);
      callbacks.delete(msg.id);
      if (msg.error) reject(msg.error);
      else resolve(msg.result);
      return;
    }

    if (msg.method === 'Runtime.consoleAPICalled') {
      const text = msg.params.args.map((a) => a.value ?? a.description ?? JSON.stringify(a)).join(' ');
      console.log(`[Console] ${text}`);

      if (text.includes('[TEST_RESULT:PASS]')) {
        console.log('\n========================================');
        console.log('🎉 WebGPU Verification Test PASSED!');
        console.log('========================================\n');
        cleanup();
        process.exit(0);
      } else if (text.includes('[TEST_RESULT:FAIL]')) {
        console.error('\n========================================');
        console.error('❌ WebGPU Verification Test FAILED!');
        console.error('========================================\n');
        cleanup();
        process.exit(1);
      }
    } else if (msg.method === 'Runtime.exceptionThrown') {
      console.error('[Page Exception]', msg.params.exceptionDetails);
    }
  };

  await send('Runtime.enable');
  await send('Page.enable');

  console.log('[Runner] Waiting for test execution (model download & GPU inference)...');

  // Set timeout of 180 seconds
  setTimeout(() => {
    console.error('[Runner] Test timed out after 180s.');
    cleanup();
    process.exit(1);
  }, 180000);
}

main().catch((err) => {
  console.error('[Runner Error]', err);
  cleanup();
  process.exit(1);
});
