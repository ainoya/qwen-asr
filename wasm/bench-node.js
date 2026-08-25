/*
 * bench-node.js - time one file through the wasm build.
 *   node wasm/bench-node.js [model-dir] [audio.wav] [threads]
 */
const { load, transcribe, readWavMono16k } = require('./node-harness');

(async () => {
  const dir = process.argv[2] || 'qwen3-asr-1.7b-q8';
  const wav = process.argv[3] || 'samples/jfk.wav';
  const threads = Number(process.argv[4] || 8);

  const { m, bytes, readMs, attachMs } = await load(dir, threads, Number(process.env.QWEN_VERBOSE || 0));
  console.log(`model ${(bytes / 1e9).toFixed(2)} GB read in ${readMs} ms, attached in ${attachMs} ms, ` +
              `threads=${m._qwen_wasm_threads()}`);

  m._qwen_wasm_pool_selftest(100);
  console.log(`pool: ${m._qwen_wasm_pool_parts()} participants, ` +
              `${(m._qwen_wasm_pool_ms() * 1000 / 100).toFixed(0)} us/dispatch`);

  const r = await transcribe(m, readWavMono16k(wav));
  console.log('\n' + (r.text || '(empty)') + '\n');
  console.log(`audio ${r.audioSec.toFixed(1)}s | inference ${(r.totalMs / 1000).toFixed(2)}s ` +
              `(${(r.audioSec * 1000 / r.totalMs).toFixed(2)}x realtime) | ` +
              `encode ${Math.round(r.encodeMs)}ms decode ${Math.round(r.decodeMs)}ms | ` +
              `${r.tokens} tokens | wall ${r.wallMs}ms`);
  process.exit(0);
})().catch((e) => { console.error(e); process.exit(1); });
