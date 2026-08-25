/*
 * node-harness.js - shared loader for running the wasm build under Node.
 *
 * Browsers throttle how many cores a renderer gets (an embedded webview gave
 * this engine ~1.3), so in-page timings say more about the browser than about
 * the build. Node runs the same qwen_asr.wasm on real worker_threads, which is
 * what the numbers in the README come from.
 */
const fs = require('fs');
const path = require('path');

const P = (x) => x >>> 0;

function cstr(m, s) {
  const len = m.lengthBytesUTF8(s) + 1;
  const p = P(m._qwen_wasm_alloc(len));
  m.stringToUTF8(s, p, len);
  return p;
}

/* 16-bit PCM WAV -> mono float32 at 16 kHz. */
function readWavMono16k(file) {
  const b = fs.readFileSync(file);
  if (b.toString('ascii', 0, 4) !== 'RIFF') throw new Error(`${file}: not RIFF`);
  let off = 12, fmt = null, data = null;
  while (off + 8 <= b.length) {
    const id = b.toString('ascii', off, off + 4);
    const size = b.readUInt32LE(off + 4);
    if (id === 'fmt ') {
      fmt = { channels: b.readUInt16LE(off + 10), rate: b.readUInt32LE(off + 12), bits: b.readUInt16LE(off + 22) };
    } else if (id === 'data') {
      data = b.subarray(off + 8, off + 8 + size);
      break;
    }
    off += 8 + size + (size & 1);
  }
  if (!fmt || !data) throw new Error(`${file}: missing fmt/data chunk`);
  if (fmt.bits !== 16) throw new Error(`${file}: need 16-bit PCM`);

  const frames = Math.floor(data.length / 2 / fmt.channels);
  const mono = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let acc = 0;
    for (let c = 0; c < fmt.channels; c++) acc += data.readInt16LE((i * fmt.channels + c) * 2) / 32768;
    mono[i] = acc / fmt.channels;
  }
  if (fmt.rate === 16000) return mono;

  const n = Math.round(frames * 16000 / fmt.rate);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = i * fmt.rate / 16000, i0 = Math.floor(x), t = x - i0;
    out[i] = mono[i0] * (1 - t) + (mono[Math.min(i0 + 1, frames - 1)] || 0) * t;
  }
  return out;
}

async function load(modelDir, threads, verbose) {
  const createQwenASR = require(path.join(__dirname, 'demo', 'qwen_asr.js'));
  const m = await createQwenASR();

  m.FS.mkdirTree('/model');
  for (const f of ['vocab.json', 'merges.txt'])
    m.FS.writeFile('/model/' + f, fs.readFileSync(path.join(modelDir, f)));

  const modelPath = path.join(modelDir, 'qwen-asr-q8.bin');
  const bytes = fs.statSync(modelPath).size;
  const ptr = P(m._qwen_wasm_alloc(bytes));
  if (!ptr) throw new Error(`cannot allocate ${bytes} bytes of wasm memory`);

  const t0 = Date.now();
  const fd = fs.openSync(modelPath, 'r');
  const CHUNK = 32 << 20;
  const buf = Buffer.allocUnsafe(CHUNK);
  for (let off = 0; off < bytes;) {
    const n = fs.readSync(fd, buf, 0, Math.min(CHUNK, bytes - off), off);
    m.HEAPU8.set(buf.subarray(0, n), ptr + off);
    off += n;
  }
  fs.closeSync(fd);
  const readMs = Date.now() - t0;

  const dp = cstr(m, '/model');
  const t1 = Date.now();
  if (m._qwen_wasm_init(ptr, bytes, dp, threads, verbose || 0) !== 0)
    throw new Error('qwen_wasm_init failed');
  m._qwen_wasm_release(dp);

  return { m, bytes, readMs, attachMs: Date.now() - t1, threads: m._qwen_wasm_threads() };
}

function setLanguage(m, lang) {
  const p = cstr(m, lang || '');
  m._qwen_wasm_set_language(p);
  m._qwen_wasm_release(p);
}

async function transcribe(m, samples) {
  const sp = P(m._qwen_wasm_alloc(samples.length * 4));
  m.HEAPF32.set(samples, sp / 4);
  const t0 = Date.now();
  if (m._qwen_wasm_batch_start(sp, samples.length) !== 0) throw new Error('batch start failed');
  while (!m._qwen_wasm_job_done()) await new Promise((r) => setTimeout(r, 15));
  const rp = P(m._qwen_wasm_job_take());
  const text = rp ? m.UTF8ToString(rp) : '';
  if (rp) m._free(rp);
  m._qwen_wasm_release(sp);
  return {
    text,
    wallMs: Date.now() - t0,
    totalMs: m._qwen_wasm_total_ms(),
    encodeMs: m._qwen_wasm_encode_ms(),
    decodeMs: m._qwen_wasm_decode_ms(),
    tokens: m._qwen_wasm_text_tokens(),
    audioSec: samples.length / 16000,
  };
}

module.exports = { load, transcribe, readWavMono16k, setLanguage, P, cstr };
