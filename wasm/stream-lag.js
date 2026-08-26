/*
 * stream-lag.js - how far behind live does streaming text land?
 *
 *   node wasm/stream-lag.js [model] [audio.wav] [threads] [chunkSec] [encWindowSec] [maxNew]
 *
 * Feeds audio in real time, the way a microphone does, and records when each
 * new piece of text first appears against the audio position it describes.
 * "provisional" is the first guess covering that audio; "committed" is when it
 * stops changing. Those two numbers are the streaming experience - throughput
 * measured by feeding as fast as the engine accepts says nothing about them.
 */
const { load, readWavMono16k, P } = require('./node-harness');

const CHUNK_MS = 250;   // same granularity as the demo's mic worklet

(async () => {
  const dir = process.argv[2] || 'qwen3-asr-1.7b-q8';
  const wav = process.argv[3] || 'samples/jfk.wav';
  const threads = Number(process.argv[4] || 8);
  const chunkSec = Number(process.argv[5] || 2);
  const encWin = Number(process.argv[6] || 8);
  const maxNew = Number(process.argv[7] || 32);

  const { m } = await load(dir, threads, 0);
  m._qwen_wasm_set_stream_params(chunkSec, maxNew, encWin);
  const pastText = process.env.QWEN_PAST_TEXT === undefined ? 1 : Number(process.env.QWEN_PAST_TEXT);
  m._qwen_wasm_set_past_text(pastText);

  const samples = readWavMono16k(wav);
  const audioSec = samples.length / 16000;
  const step = 16000 * CHUNK_MS / 1000;
  const sp = P(m._qwen_wasm_alloc(step * 4));

  if (m._qwen_wasm_stream_start() !== 0) throw new Error('stream start failed');

  const t0 = Date.now();
  let committed = '', partial = '';
  const events = [];

  const poll = () => {
    const tp = P(m._qwen_wasm_take_text());
    if (tp) { committed += m.UTF8ToString(tp); m._free(tp); }
    const pp = P(m._qwen_wasm_take_partial());
    let np = '';
    if (pp) { np = m.UTF8ToString(pp); m._free(pp); }
    const shown = committed + np;
    if (shown !== events.at(-1)?.shown) {
      events.push({ at: (Date.now() - t0) / 1000, fed: fedSec(), shown, committed });
    }
    partial = np;
  };

  let fedSamples = 0;
  const fedSec = () => fedSamples / 16000;

  /* Poll on a timer rather than busy-waiting: the thread pool spins about a
   * million iterations before parking, so a main thread that also spins turns
   * an 8-worker pool on 8 cores into 9 runnable threads and the barrier eats
   * the machine. Measured: busy-polling made an 11s clip take 112s. */
  const poller = setInterval(poll, 100);
  for (let off = 0; off < samples.length; off += step) {
    const part = samples.subarray(off, Math.min(off + step, samples.length));
    m.HEAPF32.set(part, sp / 4);
    m._qwen_wasm_stream_push(sp, part.length);
    fedSamples += part.length;
    const due = t0 + (fedSamples / 16000) * 1000;
    const wait = due - Date.now();
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  }
  clearInterval(poller);
  poll();

  const rp = P(m._qwen_wasm_stream_finish());
  const final = rp ? m.UTF8ToString(rp) : '';
  if (rp) m._free(rp);
  const wall = (Date.now() - t0) / 1000;

  /* Lag: for each growth in the shown text, how far behind the audio it is. */
  const lags = [];
  for (const e of events) if (e.shown) lags.push(e.at - e.fed);
  const first = events.find((e) => e.shown);
  const firstCommit = events.find((e) => e.committed);

  console.log(`\nchunk ${chunkSec}s | enc window ${encWin}s | maxNew ${maxNew} | ` +
              `past-text ${pastText} | threads ${threads}`);
  console.log(`audio ${audioSec.toFixed(1)}s, wall ${wall.toFixed(1)}s ` +
              `(${wall > audioSec + 0.5 ? 'FELL BEHIND by ' + (wall - audioSec).toFixed(1) + 's' : 'kept up'})`);
  console.log(`first text on screen   : ${first ? first.at.toFixed(1) + 's' : 'never'}`);
  console.log(`first committed text   : ${firstCommit ? firstCommit.at.toFixed(1) + 's' : 'never'}`);
  if (lags.length) {
    lags.sort((a, b) => a - b);
    const med = lags[Math.floor(lags.length / 2)];
    console.log(`lag behind live: median ${med.toFixed(1)}s, worst ${lags.at(-1).toFixed(1)}s ` +
                `over ${lags.length} updates`);
  }
  console.log(`\nfinal: ${final}\n`);
  process.exit(0);
})().catch((e) => { console.error(e); process.exit(1); });
