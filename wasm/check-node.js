/*
 * check-node.js - quality check for the wasm build.
 *
 * Runs every sample that has a sibling .txt reference and reports the same
 * normalized character error rate asr_regression.py uses, so the wasm build can
 * be held to the same bar as the native one.
 *
 *   node wasm/check-node.js [model-dir] [samples-dir] [threads]
 */
const fs = require('fs');
const path = require('path');
const { load, transcribe, readWavMono16k, setLanguage } = require('./node-harness');

const THRESHOLD = 0.15;

function normalize(s) {
  return s.toLowerCase().replace(/[.,!?;:'"()\[\]{}‘’“”]/g, '')
          .replace(/\s+/g, ' ').trim();
}

function editDistance(a, b) {
  const prev = new Array(b.length + 1);
  const cur = new Array(b.length + 1);
  for (let j = 0; j <= b.length; j++) prev[j] = j;
  for (let i = 1; i <= a.length; i++) {
    cur[0] = i;
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    for (let j = 0; j <= b.length; j++) prev[j] = cur[j];
  }
  return prev[b.length];
}

function findWavs(dir) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...findWavs(p));
    else if (e.name.endsWith('.wav') && fs.existsSync(p.replace(/\.wav$/, '.txt'))) out.push(p);
  }
  return out.sort();
}

(async () => {
  const dir = process.argv[2] || 'qwen3-asr-1.7b-q8';
  const samplesDir = process.argv[3] || 'samples';
  const threads = Number(process.argv[4] || 8);

  const { m } = await load(dir, threads, 0);
  setLanguage(m, process.env.QWEN_LANGUAGE || '');

  const wavs = findWavs(samplesDir);
  let fails = 0, totErr = 0, totLen = 0, totAudio = 0, totMs = 0;

  for (let i = 0; i < wavs.length; i++) {
    const wav = wavs[i];
    const ref = fs.readFileSync(wav.replace(/\.wav$/, '.txt'), 'utf8');
    const r = await transcribe(m, readWavMono16k(wav));
    const a = normalize(r.text), b = normalize(ref);
    const err = editDistance(a, b);
    const rate = b.length ? err / b.length : 0;
    const ok = rate <= THRESHOLD;
    if (!ok) fails++;
    totErr += err; totLen += b.length; totAudio += r.audioSec; totMs += r.totalMs;
    console.log(`[${ok ? ' OK ' : 'FAIL'} ${i + 1}/${wavs.length}] ${path.basename(wav)} | ` +
                `norm ${err}/${b.length} (${rate.toFixed(3)}) | ${(r.totalMs / 1000).toFixed(1)}s ` +
                `(${(r.audioSec * 1000 / r.totalMs).toFixed(2)}x)`);
  }

  console.log(`\nwasm check: ${wavs.length - fails}/${wavs.length} within ${THRESHOLD}, ` +
              `aggregate norm error ${(totErr / totLen).toFixed(4)}, ` +
              `${totAudio.toFixed(0)}s audio in ${(totMs / 1000).toFixed(0)}s ` +
              `(${(totAudio * 1000 / totMs).toFixed(2)}x realtime)`);
  process.exit(fails ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
