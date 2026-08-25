/*
 * stream-node.js - exercise the wasm streaming API outside a browser.
 *
 * Pushes a WAV through qwen_wasm_stream_push() in 0.25 s chunks, the same
 * granularity the demo's microphone worklet uses, and prints tokens as they are
 * committed. Use --realtime to pace the pushes to wall clock (what a live mic
 * looks like); without it the audio is fed as fast as the engine accepts it,
 * which is the useful mode for checking correctness.
 *
 *   node wasm/stream-node.js [model-dir] [audio.wav] [threads] [--realtime]
 */
const { load, readWavMono16k, setLanguage, P } = require('./node-harness');

const CHUNK = 4000; /* 0.25 s at 16 kHz */

(async () => {
  const dir = process.argv[2] || 'qwen3-asr-1.7b-q8';
  const wav = process.argv[3] || 'samples/jfk.wav';
  const threads = Number(process.argv[4] || 8);
  const realtime = process.argv.includes('--realtime');

  const { m } = await load(dir, threads, 0);
  setLanguage(m, process.env.QWEN_LANGUAGE || '');

  const samples = readWavMono16k(wav);
  const audioSec = samples.length / 16000;
  console.log(`streaming ${wav} (${audioSec.toFixed(1)}s), threads=${m._qwen_wasm_threads()}` +
              (realtime ? ', paced to wall clock' : ', fed as fast as accepted'));

  if (m._qwen_wasm_stream_start() !== 0) throw new Error('stream start failed');

  const drain = () => {
    const p = P(m._qwen_wasm_take_text());
    if (!p) return;
    const s = m.UTF8ToString(p);
    m._free(p);
    if (s) process.stdout.write(s);
  };
  const ticker = setInterval(drain, 120);

  const t0 = Date.now();
  const sp = P(m._qwen_wasm_alloc(CHUNK * 4));
  for (let off = 0; off < samples.length; off += CHUNK) {
    const part = samples.subarray(off, Math.min(off + CHUNK, samples.length));
    m.HEAPF32.set(part, sp / 4);
    m._qwen_wasm_stream_push(sp, part.length);
    if (realtime) await new Promise((r) => setTimeout(r, 250));
    else await new Promise((r) => setImmediate(r));
  }

  const rp = P(m._qwen_wasm_stream_finish());
  clearInterval(ticker);
  drain();
  const wall = Date.now() - t0;
  const text = rp ? m.UTF8ToString(rp) : '';
  if (rp) m._free(rp);

  console.log(`\n\nfinal: ${text}`);
  console.log(`audio ${audioSec.toFixed(1)}s | wall ${(wall / 1000).toFixed(2)}s ` +
              `(${(audioSec * 1000 / wall).toFixed(2)}x realtime) | ` +
              `inference ${(m._qwen_wasm_total_ms() / 1000).toFixed(2)}s | ${m._qwen_wasm_text_tokens()} tokens`);
  process.exit(0);
})().catch((e) => { console.error(e); process.exit(1); });
