/*
 * dump-golden.js - produce reference tensors for the browser GPU harness.
 *
 *   node wasm/dump-golden.js [model-dir] [samples-dir-or-wav ...] [--threads N]
 *
 * Why this exists: a browser tab that is not in front has its renderer
 * descheduled to roughly a core and a bit, so running the wasm reference
 * inside the harness makes a comparison take minutes instead of seconds and
 * the page looks hung. Node has no such limit, so the CPU side is computed
 * here and the page only runs the GPU.
 *
 * For each sample this writes the decoder's input embeddings - mel, audio
 * encoder and prompt assembly, everything up to the decoder - plus the
 * transcript the wasm decoder produces from them. The harness feeds the
 * embeddings straight to the GPU decoder and diffs the text.
 *
 * Output goes to wasm/demo/golden/, which the demo server already serves.
 */
const fs = require('fs');
const path = require('path');
const { load, readWavMono16k, P } = require('./node-harness');

const OUT_DIR = path.join(__dirname, 'demo', 'golden');

function collectWavs(args) {
  const wavs = [];
  const walk = (d) => {
    for (const e of fs.readdirSync(d, { withFileTypes: true }).sort((a, b) => a.name < b.name ? -1 : 1)) {
      const full = path.join(d, e.name);
      if (e.isDirectory()) walk(full);
      else if (e.name.endsWith('.wav')) wavs.push(full);
    }
  };
  for (const a of args) {
    if (fs.statSync(a).isDirectory()) walk(a);
    else wavs.push(a);
  }
  return wavs;
}

(async () => {
  const argv = process.argv.slice(2);
  let threads = 8;
  const ti = argv.indexOf('--threads');
  if (ti >= 0) { threads = Number(argv[ti + 1]) || 8; argv.splice(ti, 2); }

  const dir = argv.shift() || 'qwen3-asr-1.7b-q8';
  const wavs = collectWavs(argv.length ? argv : ['samples']);
  if (!wavs.length) { console.error('no wavs'); process.exit(1); }

  const { m } = await load(dir, threads, 0);
  console.log(`model ${dir}, threads=${m._qwen_wasm_threads()}, ${wavs.length} samples`);

  const shPtr = P(m._qwen_wasm_alloc(10 * 4));
  if (m._qwen_wasm_model_shape(shPtr) < 0) throw new Error('model shape unavailable');
  /* Only HEAPU8/HEAPF32 are exported, so read the int fields through a view. */
  const dim = new DataView(m.HEAPU8.buffer).getInt32(shPtr + 4, true);
  m._qwen_wasm_release(shPtr);

  fs.mkdirSync(OUT_DIR, { recursive: true });
  const index = { model: path.basename(dir), dim, samples: [] };

  for (const wav of wavs) {
    const name = path.basename(wav, '.wav');
    const samples = readWavMono16k(wav);
    const sp = P(m._qwen_wasm_alloc(samples.length * 4));
    m.HEAPF32.set(samples, sp / 4);

    /* Embeddings: mel + encoder + prompt assembly. */
    if (m._qwen_wasm_embeds_start(sp, samples.length) !== 0) throw new Error(`${name}: embeds start`);
    while (!m._qwen_wasm_embeds_done()) await new Promise((r) => setImmediate(r));
    const seq = m._qwen_wasm_embeds_finish();
    if (!seq) throw new Error(`${name}: embeds failed`);
    const embPtr = P(m._qwen_wasm_embeds_ptr());
    /* Copy out before anything else can reuse the buffer. */
    const emb = new Float32Array(m.HEAPF32.subarray(embPtr / 4, embPtr / 4 + seq * dim));
    fs.writeFileSync(path.join(OUT_DIR, `${name}.f32`), Buffer.from(emb.buffer));

    /* Reference transcript from the wasm decoder on the same audio. */
    if (m._qwen_wasm_batch_start(sp, samples.length) !== 0) throw new Error(`${name}: batch start`);
    while (!m._qwen_wasm_job_done()) await new Promise((r) => setImmediate(r));
    const rp = P(m._qwen_wasm_job_take());
    const text = rp ? m.UTF8ToString(rp) : '';
    if (rp) m._free(rp);
    m._qwen_wasm_release(sp);

    index.samples.push({
      name,
      wav: path.relative(path.join(__dirname, 'demo'), wav).split(path.sep).join('/'),
      seq,
      audioSec: samples.length / 16000,
      text,
    });
    console.log(`  ${name}: seq=${seq}, ${(samples.length / 16000).toFixed(1)}s, ` +
                `${(seq * dim * 4 / 1e6).toFixed(1)} MB embeddings`);
  }

  fs.writeFileSync(path.join(OUT_DIR, 'index.json'), JSON.stringify(index, null, 2));
  console.log(`\nwrote ${index.samples.length} samples to ${path.relative(process.cwd(), OUT_DIR)}`);
  process.exit(0);
})().catch((e) => { console.error(e); process.exit(1); });
