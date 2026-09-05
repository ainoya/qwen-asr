/*
 * app.js - demo UI for the WebAssembly build.
 *
 * The wasm module is instantiated on this (main) thread, because emscripten
 * creates its pthread workers from wherever the module lives and nesting
 * workers inside another worker is fragile. Nothing here blocks: transcription
 * runs on a wasm thread and the page polls for tokens and completion.
 */

import { WebGPUDecoder } from "./webgpu-decoder.js";
import { freshHeap } from "./heap.js";
import { tick, until, sleep } from "./tick.js";
import { WebGPUEncoder } from "./webgpu-encoder.js";

const $ = (id) => document.getElementById(id);
/* A hosted playground overrides these via playground-config.js; the local
 * dev server (wasm/serve.py) uses the repo-relative defaults. */
const CFG = (typeof window !== "undefined" && window.QWEN_PLAYGROUND) || {};
const MODEL_BASE = CFG.modelBase || "../../qwen3-asr-1.7b-q8";
const SAMPLE_EN = CFG.sampleEn || "../../samples/jfk.wav";
const SAMPLE_JA = CFG.sampleJa || "../../samples/extra/ja_bench.wav";

let Module = null;
let ready = false;
let busy = false;
let sampleBuf = 0;
let sampleCap = 0;
let poller = null;
let gpu = null, encoder = null;
/* Set when the wasm image was reduced because the GPU owns the transformer
 * weights; a GPU failure then means reloading with the full image, since the
 * CPU has nothing to decode with. */
let gpuWeightSource = null, gpuResidentActive = false;
/* Exposed so the harness can read hook counters without a rebuild. */
if (typeof window !== "undefined") window.__asr = () => ({ gpu, encoder, Module });

function log(msg, cls) {
  const d = document.createElement("div");
  if (cls) d.className = cls;
  d.textContent = msg;
  $("log").appendChild(d);
  $("log").scrollTop = $("log").scrollHeight;
}

const setStatus = (s) => { $("status").textContent = s; };

function cstr(s) {
  const len = Module.lengthBytesUTF8(s) + 1;
  const p = P(Module._qwen_wasm_alloc(len));
  Module.stringToUTF8(s, p, len);
  return p;
}

/* wasm exports return pointers as signed i32. The packed model is >2 GB, so
 * every allocation after it sits above 2^31 and comes back negative unless we
 * reinterpret it as unsigned. Same reason `>> 2` must never be used to turn a
 * byte pointer into a Float32Array index. */
const P = (ptr) => ptr >>> 0;

/* hardwareConcurrency counts efficiency cores, and a browser may hand out fewer
 * cores than it reports. Spinning pool threads are counter-productive in that
 * case, so start conservative and let the user raise it. */
const defaultThreads = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 2));
const f32idx = (ptr) => P(ptr) / 4;

function ensureSampleBuf(n) {
  const bytes = n * 4;
  if (bytes > sampleCap) {
    if (sampleBuf) Module._qwen_wasm_release(sampleBuf);
    sampleBuf = P(Module._qwen_wasm_alloc(bytes));
    sampleCap = sampleBuf ? bytes : 0;
  }
  return sampleBuf;
}

/* Committed text is append-only; the provisional guess replaces itself every
 * chunk, so the two are separate nodes and only the second one is rewritten. */
let committedText = "";

function renderStream(partial) {
  const out = $("out");
  out.textContent = committedText;
  if (partial) {
    const span = document.createElement("span");
    span.className = "partial";
    span.textContent = partial;
    out.appendChild(span);
  }
}

function drainText() {
  const p = P(Module._qwen_wasm_take_text());
  if (p) {
    const s = Module.UTF8ToString(p);
    Module._free(p);
    if (s) committedText += s;
  }
  let partial = "";
  const q = P(Module._qwen_wasm_take_partial());
  if (q) {
    partial = Module.UTF8ToString(q);
    Module._free(q);
  }
  renderStream(partial);
}

function showPerf(audioSec) {
  const totalMs = Module._qwen_wasm_total_ms();
  const rt = audioSec > 0 && totalMs > 0 ? (audioSec * 1000 / totalMs) : 0;
  $("perf").innerHTML =
    `audio <b>${audioSec.toFixed(1)}s</b> · inference <b>${(totalMs / 1000).toFixed(2)}s</b> ` +
    `(<b>${rt.toFixed(2)}x</b> realtime) · encode <b>${Math.round(Module._qwen_wasm_encode_ms())}ms</b> ` +
    `· decode <b>${Math.round(Module._qwen_wasm_decode_ms())}ms</b> ` +
    `· <b>${Module._qwen_wasm_text_tokens()}</b> tokens`;
}

function sendSettings() {
  const lp = cstr($("lang").value || "");
  Module._qwen_wasm_set_language(lp);
  Module._qwen_wasm_release(lp);
  Module._qwen_wasm_set_segment_sec(Number($("seg").value) || 0);
  Module._qwen_wasm_set_batch_size(Number($("batch").value) || 1);
  /* Streaming re-encodes its window every chunk and re-prefills the changed
   * suffix, which is the bulk of the per-chunk cost, so route both to the GPU
   * whenever there is one. Not tied to the decoder dropdown: a GPU that is
   * present serves the stream regardless of how batch jobs are decoded. */
  Module._qwen_wasm_set_gpu_encoder(encoder ? 1 : 0);
  Module._qwen_wasm_set_gpu_decoder(gpu ? 1 : 0);
  Module._qwen_wasm_set_past_text(1);
  Module._qwen_wasm_set_stream_params(Number($("chunk").value) || 0, 32,
                                      Number($("encwin").value) || 0);
}

/* ---------------- model loading ---------------- */

/* Stream the packed model straight into wasm memory: response.arrayBuffer()
 * would hold a second ~2 GB copy in the JS heap first. */
/* Keep the packed model in the origin's private file system.
 *
 * It is 2.18 GB and the dev server sends Cache-Control: no-store, so every
 * reload refetched the whole thing - about three minutes on a 100 Mbit line.
 * OPFS survives reloads and is per-origin, so the second visit starts in
 * seconds. The file is named by its byte length: a different model is a
 * different file rather than a stale hit, and no separate metadata to keep
 * consistent.
 *
 * Every step degrades to plain fetching. OPFS is missing in some private
 * modes, writes fail on a full disk, and neither is a reason not to run.
 */
async function opfsHandle(total, create) {
  if (!navigator.storage?.getDirectory) return null;
  try {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle("qwen-asr", { create });
    return await dir.getFileHandle(`model-${total}.bin`, { create });
  } catch {
    return null;   /* not found, or no OPFS at all */
  }
}

async function copyStreamInto(stream, ptr, total, label, alsoWrite) {
  const reader = stream.getReader();
  let off = 0, lastReport = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    // Re-read HEAPU8 every time: growing memory swaps the backing buffer.
    freshHeap(Module).HEAPU8.set(value, ptr + off);
    if (alsoWrite) {
      try { await alsoWrite.write(value); } catch { alsoWrite = null; }
    }
    off += value.length;
    if (off - lastReport > 48 * 1024 * 1024) {
      lastReport = off;
      $("barfill").style.width = (off / total * 100).toFixed(1) + "%";
      setStatus(`${label} ${(off / 1e9).toFixed(2)} / ${(total / 1e9).toFixed(2)} GB`);
      await tick();
    }
  }
  return { off, writer: alsoWrite };
}

async function fetchModelInto(url, total) {
  const ptr = P(Module._qwen_wasm_alloc(total));
  if (!ptr) throw new Error(`could not allocate ${total} bytes of wasm memory`);

  /* Cached copy first. A size mismatch means a different build; refetch. */
  const cached = await opfsHandle(total, false);
  if (cached) {
    try {
      const file = await cached.getFile();
      if (file.size === total) {
        const { off } = await copyStreamInto(file.stream(), ptr, total, "loading cached model", null);
        if (off === total) {
          $("barfill").style.width = "100%";
          log("model loaded from the local cache");
          return ptr;
        }
      }
    } catch (e) {
      log(`cache read failed (${e.message}); refetching`, "err");
    }
  }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);

  /* Write the cache as the bytes stream past, so caching costs no extra pass. */
  let writer = null, handle = null;
  try {
    handle = await opfsHandle(total, true);
    if (handle) writer = await handle.createWritable();
  } catch { writer = null; }

  const { off, writer: stillWriting } =
    await copyStreamInto(res.body, ptr, total, "downloading model", writer);

  if (off !== total) {
    if (stillWriting) { try { await stillWriting.abort(); } catch {} }
    throw new Error(`model truncated: ${off} of ${total}`);
  }
  if (stillWriting) {
    try { await stillWriting.close(); log("model cached for next time"); }
    catch (e) { log(`could not cache the model (${e.message})`, "err"); }
  }
  $("barfill").style.width = "100%";
  return ptr;
}

/* Stream the model into OPFS without touching wasm memory. */
async function fetchModelToOpfs(url, total) {
  const cached = await opfsHandle(total, false);
  if (cached) {
    try { if ((await cached.getFile()).size === total) return cached; } catch {}
  }
  const handle = await opfsHandle(total, true);
  if (!handle) throw new Error("OPFS unavailable");
  const writer = await handle.createWritable();
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const reader = res.body.getReader();
  let off = 0, lastReport = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    await writer.write(value);
    off += value.length;
    if (off - lastReport > 48 * 1024 * 1024) {
      lastReport = off;
      $("barfill").style.width = (off / total * 100).toFixed(1) + "%";
      setStatus(`downloading model ${(off / 1e9).toFixed(2)} / ${(total / 1e9).toFixed(2)} GB`);
      await tick();
    }
  }
  await writer.close();
  if (off !== total) throw new Error(`model truncated: ${off} of ${total}`);
  log("model cached for next time");
  return opfsHandle(total, false);
}

/* GPU-resident load: the transformer-layer weights never enter wasm memory.
 *
 * wasm memory cannot shrink, so the classic flow - materialize the whole
 * 2.18 GB image in the heap, upload 1.72 GB of it to the GPU - holds both
 * copies forever. Here the safetensors header is parsed in JS, a reduced
 * image containing only the decoder norms is built for wasm. Decoder layers,
 * the audio tower and the tied embedding / LM head upload straight to the GPU;
 * prompt assembly reads a small cached set of embedding rows back on demand. */
const DROP_RE = /^thinker\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate_up|down_proj))\.weight\.q8s?$|^thinker\.model\.embed_tokens\.weight\.q8s?$|^thinker\.audio_tower\./;

/* Random-access source for the packed model: the OPFS cache when it can hold
 * the file, else a JS ArrayBuffer held only until the GPU upload finishes -
 * JS memory, unlike wasm memory, is returned when dropped. (Some embedded
 * profiles cap OPFS below the model size; this machine's pane refuses writes
 * past ~2.08 GB with a 2.18 GB model.) */
async function acquireModelSource(url, total) {
  try {
    const handle = await fetchModelToOpfs(url, total);
    const file = await handle.getFile();
    if (file.size === total)
      return { read: (o, n) => file.slice(o, o + n).arrayBuffer(), transient: null };
  } catch (e) {
    log(`OPFS cache unavailable (${e.message}); holding the model in JS memory during load`, "err");
  }
  /* One ArrayBuffer cannot hold the model (Chrome caps them near 2^31), so
   * the transient copy is an array of 256 MB chunks. */
  const CB = 256 << 20;
  const chunks = [];
  for (let o = 0; o < total; o += CB) chunks.push(new Uint8Array(Math.min(CB, total - o)));
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const reader = res.body.getReader();
  let off = 0, lastReport = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    let vo = 0;
    while (vo < value.length) {
      const ci = Math.floor((off + vo) / CB), co = (off + vo) % CB;
      const n = Math.min(value.length - vo, CB - co);
      chunks[ci].set(value.subarray(vo, vo + n), co);
      vo += n;
    }
    off += value.length;
    if (off - lastReport > 48 * 1024 * 1024) {
      lastReport = off;
      $("barfill").style.width = (off / total * 100).toFixed(1) + "%";
      setStatus(`downloading model ${(off / 1e9).toFixed(2)} / ${(total / 1e9).toFixed(2)} GB`);
      await tick();
    }
  }
  if (off !== total) throw new Error(`model truncated: ${off} of ${total}`);
  const read = (o, n) => {
    const ci = Math.floor(o / CB), co = o % CB;
    if (co + n <= CB) return Promise.resolve(chunks[ci].subarray(co, co + n));
    const out = new Uint8Array(n);          /* spans a chunk boundary: assemble */
    let done2 = 0;
    while (done2 < n) {
      const c = Math.floor((o + done2) / CB), cc = (o + done2) % CB;
      const m = Math.min(n - done2, CB - cc);
      out.set(chunks[c].subarray(cc, cc + m), done2);
      done2 += m;
    }
    return Promise.resolve(out);
  };
  return { read, transient: chunks };
}

async function loadGpuResident(url, total, threads) {
  const src = await acquireModelSource(url, total);
  const read = src.read;

  const asU8 = (b) => b instanceof Uint8Array ? b : new Uint8Array(b);
  const h8 = asU8(await read(0, 8));
  const hlen = Number(new DataView(h8.buffer, h8.byteOffset, 8).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(asU8(await read(8, hlen))));
  const dataBase = 8 + hlen;

  /* Reduced image: same format, GPU-owned tensors left out. */
  const kept = Object.entries(header)
    .filter(([name]) => name !== "__metadata__" && !DROP_RE.test(name))
    .sort((a, b) => a[1].data_offsets[0] - b[1].data_offsets[0]);
  const newHeader = {};
  let dataOff = 0;
  for (const [name, t] of kept) {
    const size = t.data_offsets[1] - t.data_offsets[0];
    dataOff = Math.ceil(dataOff / 64) * 64;
    newHeader[name] = { dtype: t.dtype, shape: t.shape, data_offsets: [dataOff, dataOff + size] };
    dataOff += size;
  }
  let hjson = new TextEncoder().encode(JSON.stringify(newHeader));
  const hpad = Math.ceil(hjson.length / 64) * 64;   /* keep the data 64-aligned */
  const padded = new Uint8Array(hpad).fill(0x20);
  padded.set(hjson);
  hjson = padded;

  const reducedLen = 8 + hjson.length + dataOff;
  const reducedSize = reducedLen < 100e6
    ? `${(reducedLen / 1e6).toFixed(1)} MB`
    : `${(reducedLen / 1e9).toFixed(2)} GB`;
  setStatus(`building the reduced image (${reducedSize} in wasm)...`);
  const ptr = P(Module._qwen_wasm_alloc(reducedLen));
  if (!ptr) throw new Error(`could not allocate ${reducedLen} bytes of wasm memory`);
  /* No cached view: every heap access re-validates (freshHeap). */
  new DataView(freshHeap(Module).HEAPU8.buffer, ptr, 8)
    .setBigUint64(0, BigInt(hjson.length), true);
  freshHeap(Module).HEAPU8.set(hjson, ptr + 8);
  let copied = 0, lastReport = 0;
  for (const [name, t] of kept) {
    const size = t.data_offsets[1] - t.data_offsets[0];
    const srcOff = dataBase + t.data_offsets[0];
    const dst = ptr + 8 + hjson.length + newHeader[name].data_offsets[0];
    for (let off = 0; off < size; off += 64 << 20) {
      const n = Math.min(64 << 20, size - off);
      freshHeap(Module).HEAPU8.set(asU8(await read(srcOff + off, n)), dst + off);
      copied += n;
      if (copied - lastReport > 128 << 20) {
        lastReport = copied;
        $("barfill").style.width = (copied / dataOff * 100).toFixed(1) + "%";
        await tick();
      }
    }
  }
  $("barfill").style.width = "100%";

  /* Decoder weight source: entries in exactly the order the C descriptor
   * table would emit them - layers 0..N x (q, k, v, o, gate_up, down), then
   * the tied embedding. Offsets are absolute file positions. */
  const abs = (t) => dataBase + t.data_offsets[0];
  const len = (t) => t.data_offsets[1] - t.data_offsets[0];
  const MATS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "mlp.gate_up", "mlp.down_proj"];
  let layers = 0;
  while (header[`thinker.model.layers.${layers}.self_attn.q_proj.weight.q8`]) layers++;
  if (!layers) throw new Error("no decoder layers in the model header");
  const entries = [];
  for (let l = 0; l < layers; l++) {
    for (let kind = 0; kind < 6; kind++) {
      const base = `thinker.model.layers.${l}.${MATS[kind]}.weight`;
      const q = header[`${base}.q8`], sc = header[`${base}.q8s`];
      if (!q || !sc) throw new Error(`missing ${base}.q8 in the model header`);
      entries.push({ kind, layer: l, rows: q.shape[0], cols: q.shape[1],
                     qoff: abs(q), soff: abs(sc) });
    }
  }
  const emb = header["thinker.model.embed_tokens.weight.q8"];
  const embS = header["thinker.model.embed_tokens.weight.q8s"];
  if (!emb || !embS) throw new Error("missing embed_tokens.q8 in the model header");
  entries.push({ kind: 6, layer: 0, rows: emb.shape[0], cols: emb.shape[1],
                 qoff: abs(emb), soff: abs(embS) });

  /* Encoder records, in the C emission order (see qwen_wasm_enc_desc).
   * Kinds are positional: vec entries carry a count, mats carry rows/cols. */
  const T = "thinker.audio_tower.";
  const need = (n) => {
    const t = header[n];
    if (!t) throw new Error(`missing ${n} in the model header`);
    return t;
  };
  const numel = (t) => t.shape.reduce((a, b) => a * b, 1);
  let ek = 0;
  const encEntries = [];
  const vec = (name) => { const t = need(name);
    encEntries.push({ kind: ek++, layer: 0, count: numel(t), foff: abs(t) }); };
  const mat = (name) => { const q = need(name + ".q8"), sc = need(name + ".q8s");
    encEntries.push({ kind: ek++, layer: 0, rows: q.shape[0], cols: q.shape[1],
                      qoff: abs(q), soff: abs(sc) }); };
  vec(T + "conv2d1.weight"); vec(T + "conv2d1.bias");
  vec(T + "conv2d2.weight"); vec(T + "conv2d2.bias");
  vec(T + "conv2d3.weight"); vec(T + "conv2d3.bias");
  mat(T + "conv_out.weight");
  let encLayers = 0;
  while (header[`${T}layers.${encLayers}.self_attn.q_proj.weight.q8`]) encLayers++;
  if (!encLayers) throw new Error("no encoder layers in the model header");
  for (let l = 0; l < encLayers; l++) {
    const L = `${T}layers.${l}.`;
    ek = 7;   /* per-layer kinds restart at E_ATTN_NORM_W */
    const lv = (name) => { const t = need(name);
      encEntries.push({ kind: ek++, layer: l, count: numel(t), foff: abs(t) }); };
    const lm = (name) => { const q = need(name + ".q8"), sc = need(name + ".q8s");
      encEntries.push({ kind: ek++, layer: l, rows: q.shape[0], cols: q.shape[1],
                        qoff: abs(q), soff: abs(sc) }); };
    lv(L + "self_attn_layer_norm.weight"); lv(L + "self_attn_layer_norm.bias");
    lm(L + "self_attn.q_proj.weight"); lv(L + "self_attn.q_proj.bias");
    lm(L + "self_attn.k_proj.weight"); lv(L + "self_attn.k_proj.bias");
    lm(L + "self_attn.v_proj.weight"); lv(L + "self_attn.v_proj.bias");
    lm(L + "self_attn.out_proj.weight"); lv(L + "self_attn.out_proj.bias");
    lv(L + "final_layer_norm.weight"); lv(L + "final_layer_norm.bias");
    lm(L + "fc1.weight"); lv(L + "fc1.bias");
    lm(L + "fc2.weight"); lv(L + "fc2.bias");
  }
  ek = 23;
  vec(T + "ln_post.weight"); vec(T + "ln_post.bias");
  mat(T + "proj1.weight"); vec(T + "proj1.bias");
  mat(T + "proj2.weight"); vec(T + "proj2.bias");

  gpuWeightSource = { entries, encEntries, read, transient: src.transient };

  setStatus("attaching the reduced image...");
  Module._qwen_wasm_set_gpu_resident(1);
  const dir = cstr("/model");
  const rc = Module._qwen_wasm_init(ptr, reducedLen, dir, threads, 0);
  Module._qwen_wasm_release(dir);
  if (rc !== 0) throw new Error("qwen_wasm_init failed on the reduced image");
  gpuResidentActive = true;
  log(`gpu-resident load: ${reducedSize} in wasm instead of ${(total / 1e9).toFixed(2)} GB`);
}

$("load").onclick = async () => {
  $("load").disabled = true;
  const t0 = performance.now();
  try {
    setStatus("instantiating wasm...");
    log("instantiating wasm module");
    Module = await createQwenASR();

    setStatus("fetching tokenizer...");
    Module.FS.mkdirTree("/model");
    for (const name of ["vocab.json", "merges.txt"]) {
      const r = await fetch(`${MODEL_BASE}/${name}`);
      if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
      Module.FS.writeFile(`/model/${name}`, new Uint8Array(await r.arrayBuffer()));
    }

    let total = 0;
    try {
      const head = await fetch(`${MODEL_BASE}/qwen-asr-q8.bin`, { method: "HEAD" });
      if (head.ok) total = Number(head.headers.get("content-length")) || 0;
    } catch {}
    /* Some CDNs answer HEAD without an exposed Content-Length; the deploy
     * config carries the size so progress and the OPFS cache still work. */
    if (!total) total = Number(CFG.modelSize) || 0;
    if (!total) throw new Error("could not determine the model size");

    const threads = Number($("threads").value) || 8;
    /* GPU backend: probe first, and if the GPU is real, keep the transformer
     * weights out of wasm memory entirely. sessionStorage flag forces the
     * classic full image after a mid-session GPU failure. */
    let probe = null;
    let loaded = false;
    if ($("backend").value === "gpu") {
      probe = await WebGPUDecoder.probe();
      if (!probe.ok) {
        log(`WebGPU unavailable (${probe.why}); falling back to the wasm decoder`, "err");
        $("backend").value = "cpu";
      } else if (!sessionStorage.getItem("qwenFullImage")) {
        try {
          log(`fetching packed model, ${(total / 1e9).toFixed(2)} GB`);
          await loadGpuResident(`${MODEL_BASE}/qwen-asr-q8.bin`, total, threads);
          loaded = true;
        } catch (e) {
          log(`gpu-resident load failed (${e.message}); using the full image`, "err");
          gpuWeightSource = null;
          gpuResidentActive = false;
        }
      }
    }

    if (!loaded) {
      log(`fetching packed model, ${(total / 1e9).toFixed(2)} GB`);
      const ptr = await fetchModelInto(`${MODEL_BASE}/qwen-asr-q8.bin`, total);
      setStatus("attaching weights...");
      const dir = cstr("/model");
      const rc = Module._qwen_wasm_init(ptr, total, dir, threads, 0);
      Module._qwen_wasm_release(dir);
      if (rc !== 0) throw new Error("qwen_wasm_init failed");
    }

    if ($("backend").value === "gpu") {
      {
        setStatus("uploading weights to the GPU...");
        gpu = new WebGPUDecoder(Module);
        if (gpuWeightSource) gpu.weightSource = gpuWeightSource;
        await gpu.init((m) => setStatus(m));
        gpu.weightSource = null;
        log(`GPU decoder ready (${(gpu.weightBytes / 1e9).toFixed(2)} GB resident on the GPU)`);

        /* The input embedding and LM head are one tied Qwen matrix. Keep that
         * 311 MB table only on the GPU and cache the few rows C asks for while
         * assembling prompts. The cache is bounded (~2 MB for the 1.7B model)
         * and an LRU refresh keeps streaming-prefix vocabulary hot. */
        gpu.prepareEmbeddingLookup();
        const embedCache = new Map();
        const EMBED_CACHE_ROWS = 256;
        Module.__gpuEmbedMany = async (idsPtr, n, dst, dim, req) => {
          try {
            idsPtr = idsPtr >>> 0;
            dst = dst >>> 0;
            /* Copy before await: the ids live in a worker-owned stack frame. */
            const ids = new Int32Array(freshHeap(Module).HEAPU8.buffer, idsPtr, n).slice();
            const missing = [];
            const seen = new Set();
            for (const id of ids) {
              if (!embedCache.has(id) && !seen.has(id)) {
                seen.add(id);
                missing.push(id);
              }
            }
            const fresh = new Map();
            if (missing.length) {
              const rows = await gpu.readTokenEmbeddings(missing);
              if (rows.length !== missing.length * dim)
                throw new Error(`embedding batch width ${rows.length}, expected ${missing.length * dim}`);
              for (let i = 0; i < missing.length; i++)
                fresh.set(missing[i], rows.slice(i * dim, (i + 1) * dim));
            }
            const base = f32idx(dst);
            for (let i = 0; i < ids.length; i++) {
              const id = ids[i];
              const row = embedCache.get(id) || fresh.get(id);
              if (!row) throw new Error(`embedding row ${id} unavailable`);
              freshHeap(Module).HEAPF32.set(row, base + i * dim);
              if (embedCache.has(id)) embedCache.delete(id);
              embedCache.set(id, row);
              if (embedCache.size > EMBED_CACHE_ROWS)
                embedCache.delete(embedCache.keys().next().value);
            }
            Module._qwen_wasm_embed_hook_done(req, 1);
          } catch (err) {
            console.error("gpu embedding hook:", err);
            Module._qwen_wasm_embed_hook_done(req, 0);
          }
        };
        Module._qwen_wasm_set_gpu_embedder(1);

        /* The audio tower is optional: if it will not start, mel and the
         * encoder stay in wasm and only the decoder moves. */
        try {
          setStatus("uploading the audio tower to the GPU...");
          const e = new WebGPUEncoder(Module, { device: gpu.device, adapter: gpu.adapter });
          if (gpuWeightSource) e.weightSource =
            { entries: gpuWeightSource.encEntries, read: gpuWeightSource.read };
          await e.init(() => {});
          e.weightSource = null;
          encoder = e;
          log(`GPU audio tower ready (${(e.weightBytes / 1e6).toFixed(0)} MB)`);

          /* The streaming loop runs on a worker and asks for the tower through
           * this, because WebGPU is main-thread and asynchronous. Errors are
           * reported as a failure so the loop falls back to the wasm encoder
           * rather than stalling. */
          Module.__gpuEncode = async (melPtr, frames) => {
            const hs = window.__hookStats || (window.__hookStats = {dec: 0, decMs: 0, decArrive: 0, enc: 0, encMs: 0});
            const et0 = performance.now();
            hs.enc++;
            try {
              encoder.hookCalls = (encoder.hookCalls || 0) + 1;
              const out = await encoder.runFromMel(melPtr, frames);
              const p = P(Module._qwen_wasm_alloc(out.byteLength));
              if (!p) throw new Error("out of wasm memory for the encoder output");
              freshHeap(Module).HEAPF32.set(out, f32idx(p));
              Module._qwen_wasm_enc_hook_done(p, encoder.tokens);
              hs.encMs += performance.now() - et0;
              (hs.encProfiles || (hs.encProfiles = [])).push(
                {runMs: Math.round(encoder.runMs), wallMs: Math.round(performance.now() - et0)});
            } catch (err) {
              encoder.hookFails = (encoder.hookFails || 0) + 1;
              if (encoder.hookFails <= 2)
                log(`GPU tower failed mid-stream (${err.message}); falling back`, "err");
              Module._qwen_wasm_enc_hook_done(0, 0);
            }
          };
        } catch (err) {
          if (gpuResidentActive) {
            /* The reduced image has no tower weights for a CPU fallback. */
            log(`GPU tower failed on a reduced image (${err.message}); reloading with the full image`, "err");
            sessionStorage.setItem("qwenFullImage", "1");
            location.reload();
            return;
          }
          log(`GPU audio tower unavailable (${err.message}); mel and the encoder stay on the cpu`, "err");
        }
        /* Both uploads done: drop the source so a transient JS copy of the
         * model can be collected. */
        gpuWeightSource = null;

        /* Streaming decode: one call covers the suffix prefill against the
         * KV the GPU already holds from the previous chunk, plus the whole
         * token loop. reuseLen rows are unchanged - the C loop guarantees it -
         * and a failure report makes the loop fall back to the wasm decoder. */
        Module.__gpuDecode = async (embedsPtr, totalSeq, reuseLen, maxNew) => {
          const hs = window.__hookStats || (window.__hookStats = {dec: 0, decMs: 0, decArrive: 0, enc: 0, encMs: 0});
          const t0 = performance.now();
          hs.dec++;
          try {
            if (!gpu) throw new Error("no GPU decoder");
            /* The pointer crosses EM_ASM as a signed int and the packed model
             * puts the heap well past 2 GB, so restore the unsigned value. */
            embedsPtr = embedsPtr >>> 0;
            const r = reuseLen > 0
              ? await gpu.prefillSuffixAndGenerate(embedsPtr, totalSeq, reuseLen, maxNew - 1, null)
              : await gpu.prefillAndGenerate(embedsPtr, totalSeq, maxNew - 1, null);
            const ids = (r.ids || []).slice(0, maxNew);
            const p = P(Module._qwen_wasm_alloc(Math.max(ids.length, 1) * 4));
            if (!p) throw new Error("out of wasm memory for the token ids");
            /* HEAP32 is not exported by this build; any heap view's buffer
             * is the same SharedArrayBuffer. */
            new Int32Array(freshHeap(Module).HEAPU8.buffer, p, ids.length).set(ids);
            Module._qwen_wasm_dec_hook_done(p, ids.length);
            Module._qwen_wasm_release(p);
            hs.decMs += performance.now() - t0;
            (hs.decProfiles || (hs.decProfiles = [])).push(gpu.lastProfile ||
              {prefillMs: Math.round(gpu.prefillMs)});
          } catch (err) {
            console.error("gpu decode hook:", err);
            gpu && (gpu.hookFails = (gpu.hookFails || 0) + 1);
            if (!gpu || gpu.hookFails <= 2)
              log(`GPU decode failed mid-stream (${err && err.message}); falling back`, "err");
            Module._qwen_wasm_dec_hook_done(0, -1);
          }
        };
      }
    }

    ready = true;
    window.__gpu = gpu; window.__enc = encoder; window.__M = Module;  /* debug */
    /* GPU decode makes 1-second chunks sustainable (measured ~0.8 s per chunk
     * at a 41 s context in a throttled tab), and halving the chunk halves how
     * long words sit unconfirmed on screen. CPU decode cannot keep that pace,
     * so only the GPU path gets the faster default - and only if the user has
     * not chosen a chunk size themselves. */
    if (gpu && $("chunk").value === "2") {
      $("chunk").value = "1";
      log("GPU decode active: streaming chunk defaults to 1 s");
    }
    sendSettings();
    for (const id of ["file", "sample-ja", "sample-en", "mic", "simstream"])
      $(id).disabled = false;
    setStatus(`ready — ${((performance.now() - t0) / 1000).toFixed(1)}s, ${threads} threads`);
    log("model ready");
  } catch (e) {
    if (gpuResidentActive && !(gpu && gpu.ready)) {
      /* wasm attached a reduced image but the GPU never came up: nothing can
       * decode. Reload once with the full image (it is in the OPFS cache). */
      log(`load failed after a reduced attach (${e.message}); reloading with the full image`, "err");
      sessionStorage.setItem("qwenFullImage", "1");
      location.reload();
      return;
    }
    log("error: " + e.message, "err");
    setStatus("failed");
    $("load").disabled = false;
  }
};

/* ---------------- batch ---------------- */

async function decodeTo16k(bytes) {
  const probe = new AudioContext();
  const decoded = await probe.decodeAudioData(bytes.slice(0));
  probe.close();
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const mono = await off.startRendering();
  return mono.getChannelData(0);
}

async function runBatch(bytes, label) {
  if (!ready || busy) return;
  busy = true;
  $("out").textContent = "";
  committedText = "";
  $("perf").textContent = "";
  setStatus("decoding audio...");
  sendSettings();

  let samples;
  try {
    samples = await decodeTo16k(bytes);
  } catch (e) {
    busy = false;
    log("audio decode failed: " + e.message, "err");
    setStatus("failed");
    return;
  }

  const audioSec = samples.length / 16000;
  setStatus(`transcribing ${label} (${audioSec.toFixed(1)}s)...`);

  const ptr = ensureSampleBuf(samples.length);
  if (!ptr) { busy = false; log("out of wasm memory for the audio buffer", "err"); return; }

  if (gpu && $("backend").value === "gpu") {
    /* Everything but mel on the GPU when the audio tower is available:
     * mel here, tower and decoder there, prompt assembly back here. */
    try {
      freshHeap(Module).HEAPF32.set(samples, f32idx(ptr));
      const t0 = performance.now();
      let seq, melMs = 0;
      if (encoder) {
        if (Module._qwen_wasm_mel_start(ptr, samples.length) !== 0)
          throw new Error("mel failed to start");
        await until(() => Module._qwen_wasm_mel_done());
        const frames = Module._qwen_wasm_mel_finish();
        if (!frames) throw new Error("mel failed");
        melMs = Module._qwen_wasm_mel_ms();
        setStatus(`running the audio tower on the GPU (${frames} frames)...`);

        const encOut = await encoder.runFromMel(P(Module._qwen_wasm_mel_ptr()), frames);
        const ep = P(Module._qwen_wasm_alloc(encOut.byteLength));
        if (!ep) throw new Error("out of wasm memory for the encoder output");
        freshHeap(Module).HEAPF32.set(encOut, f32idx(ep));
        if (Module._qwen_wasm_embeds_from_enc_start(ep, encoder.tokens) !== 0) {
          Module._qwen_wasm_release(ep);
          throw new Error("prompt assembly failed to start");
        }
        Module._qwen_wasm_release(ep);
        await until(() => Module._qwen_wasm_embeds_from_enc_done());
        seq = Module._qwen_wasm_embeds_from_enc_finish();
        if (!seq) throw new Error("prompt assembly failed");
      } else {
        if (Module._qwen_wasm_embeds_start(ptr, samples.length) !== 0)
          throw new Error("encoder failed to start");
        await until(() => Module._qwen_wasm_embeds_done());
        seq = Module._qwen_wasm_embeds_finish();
        if (!seq) throw new Error("encoder failed");
      }
      const encMs = performance.now() - t0;
      setStatus(`prefilling ${seq} tokens on the GPU...`);

      const t1 = performance.now();
      const r = await gpu.prefillAndGenerate(
        P(Module._qwen_wasm_embeds_ptr()), seq, 1024,
        (piece) => { $("out").textContent += piece; });
      const decMs = performance.now() - t1;
      const wall = encMs + decMs;

      $("out").textContent = r.text;
      $("perf").innerHTML =
        `audio <b>${audioSec.toFixed(1)}s</b> · total <b>${(wall / 1000).toFixed(2)}s</b> ` +
        `(<b>${(audioSec * 1000 / wall).toFixed(2)}x</b> realtime) · ` +
        (encoder
          ? `mel <b>${Math.round(melMs)}ms</b> (cpu) · gpu tower <b>${Math.round(encoder.runMs)}ms</b> · `
          : `mel+encoder <b>${Math.round(encMs)}ms</b> (cpu) · `) +
        `gpu prefill <b>${Math.round(gpu.prefillMs)}ms</b> · ` +
        `gpu generation <b>${Math.round(decMs - gpu.prefillMs)}ms</b> for <b>${r.tokens}</b> tokens`;
      setStatus("done");
    } catch (e) {
      /* A lost device is recoverable by giving up on the GPU, so do that and
       * finish the job rather than leaving the user with nothing. */
      if (gpu?.lost || encoder?.lost) {
        if (gpuResidentActive) {
          /* The CPU has no transformer weights to retry with; reload with the
           * full image (it comes straight from the OPFS cache). */
          log("GPU lost with gpu-resident weights; reloading with the full image", "err");
          sessionStorage.setItem("qwenFullImage", "1");
          location.reload();
          return;
        }
        log(`GPU unavailable (${gpu?.lost || encoder?.lost}); retrying on the cpu`, "err");
        gpu = null; encoder = null;
        Module.__gpuEncode = null;
        Module.__gpuDecode = null;
        Module.__gpuEmbedMany = null;
        Module._qwen_wasm_set_gpu_encoder(0);
        Module._qwen_wasm_set_gpu_decoder(0);
        Module._qwen_wasm_set_gpu_embedder(0);
        $("backend").value = "cpu";
        busy = false;
        return runBatch(bytes, label);
      }
      log("error: " + e.message, "err");
      setStatus("failed");
    }
    busy = false;
    return;
  }

  try {
    freshHeap(Module).HEAPF32.set(samples, f32idx(ptr));
    if (Module._qwen_wasm_batch_start(ptr, samples.length) !== 0)
      throw new Error("could not start transcription");
  } catch (e) {
    busy = false;
    log("error: " + e.message, "err");
    setStatus("failed");
    return;
  }

  poller = setInterval(() => {
    drainText();
    if (!Module._qwen_wasm_job_done()) return;
    clearInterval(poller);
    poller = null;
    const rp = P(Module._qwen_wasm_job_take());
    drainText();
    if (rp) {
      /* The final text supersedes both the committed run and the guess. */
      committedText = Module.UTF8ToString(rp);
      $("out").textContent = committedText;
      Module._free(rp);
    }
    showPerf(audioSec);
    setStatus("done");
    busy = false;
  }, 120);
}

$("file").onchange = async (e) => {
  const f = e.target.files[0];
  if (f) runBatch(await f.arrayBuffer(), f.name);
};

async function runSample(url) {
  setStatus("fetching sample...");
  const r = await fetch(url);
  if (!r.ok) { log(`sample ${url}: HTTP ${r.status}`, "err"); return; }
  runBatch(await r.arrayBuffer(), url.split("/").pop());
}
$("sample-ja").onclick = () => runSample(SAMPLE_JA);
$("sample-en").onclick = () => runSample(SAMPLE_EN);

/* ---------------- streaming ---------------- */

let audioCtx = null, micStream = null, micNode = null, streamedSamples = 0;

$("mic").onclick = async () => {
  if (!ready || busy) return;

  busy = true;
  streamedSamples = 0;
  $("out").textContent = "";
  committedText = "";
  $("perf").textContent = "";
  sendSettings();

  try {
    audioCtx = new AudioContext({ sampleRate: 16000 });
    await audioCtx.audioWorklet.addModule("mic-worklet.js");
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });

    if (Module._qwen_wasm_stream_start() !== 0) throw new Error("stream start failed");

    const src = audioCtx.createMediaStreamSource(micStream);
    micNode = new AudioWorkletNode(audioCtx, "mic-collector");
    micNode.port.onmessage = (ev) => {
      const s = ev.data;
      let peak = 0;
      for (let i = 0; i < s.length; i += 8) peak = Math.max(peak, Math.abs(s[i]));
      $("lvlfill").style.width = Math.min(100, peak * 180) + "%";
      const ptr = ensureSampleBuf(s.length);
      freshHeap(Module).HEAPF32.set(s, f32idx(ptr));
      Module._qwen_wasm_stream_push(ptr, s.length);
      streamedSamples += s.length;
    };
    src.connect(micNode);
    const sink = audioCtx.createGain();
    sink.gain.value = 0;
    micNode.connect(sink).connect(audioCtx.destination);

    poller = setInterval(drainText, 120);
    $("mic").classList.add("hide");
    $("micstop").classList.remove("hide");
    setStatus("listening");
  } catch (e) {
    busy = false;
    log("mic error: " + e.message, "err");
    setStatus("failed");
  }
};

$("micstop").onclick = async () => {
  $("micstop").classList.add("hide");
  $("mic").classList.remove("hide");
  setStatus("finishing...");
  if (micNode) micNode.disconnect();
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (audioCtx) await audioCtx.close();
  micNode = micStream = audioCtx = null;
  $("lvlfill").style.width = "0";

await (async () => {
    /* stream_finish() joins the stream thread; done naively that join blocks
     * the main thread, which is also the only thread that can service the GPU
     * hooks the stream thread may be waiting on. Signal end-of-audio first and
     * poll for the drain so the hooks keep flowing. */
    Module._qwen_wasm_stream_eof();
    while (!Module._qwen_wasm_stream_drained()) await sleep(30);
  })();
  const rp = P(Module._qwen_wasm_stream_finish());
  if (poller) { clearInterval(poller); poller = null; }
  drainText();
  if (rp) {
    /* The final text supersedes both the committed run and the guess. */
    committedText = Module.UTF8ToString(rp);
    $("out").textContent = committedText;
    Module._free(rp);
  }
  showPerf(streamedSamples / 16000);
  setStatus("done");
  busy = false;
};

/* Feed a file through the *streaming* path at wall-clock speed. Same code path
 * as the microphone, so the streaming behaviour can be checked without one. */
$("simstream").onclick = async () => {
  if (!ready || busy) return;
  busy = true;
  streamedSamples = 0;
  $("out").textContent = "";
  committedText = "";
  $("perf").textContent = "";
  sendSettings();

  try {
    const url = $("lang").value === "English" ? SAMPLE_EN : SAMPLE_JA;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    const samples = await decodeTo16k(await r.arrayBuffer());

    if (Module._qwen_wasm_stream_start() !== 0) throw new Error("stream start failed");
    poller = setInterval(drainText, 120);
    setStatus(`streaming ${url.split("/").pop()} (${(samples.length / 16000).toFixed(1)}s)`);

    const CHUNK = 4000; // 0.25 s, same granularity as the mic worklet
    for (let off = 0; off < samples.length; off += CHUNK) {
      const part = samples.subarray(off, Math.min(off + CHUNK, samples.length));
      const ptr = ensureSampleBuf(part.length);
      freshHeap(Module).HEAPF32.set(part, f32idx(ptr));
      Module._qwen_wasm_stream_push(ptr, part.length);
      streamedSamples += part.length;
      await sleep(250);
    }

    setStatus("draining...");
    Module._qwen_wasm_stream_eof();
    while (!Module._qwen_wasm_stream_drained()) await sleep(30);
    const rp = P(Module._qwen_wasm_stream_finish());
    if (poller) { clearInterval(poller); poller = null; }
    drainText();
    if (rp) {
      /* The final text supersedes both the committed run and the guess. */
      committedText = Module.UTF8ToString(rp);
      $("out").textContent = committedText;
      Module._free(rp);
    }
    showPerf(streamedSamples / 16000);
    setStatus("done");
  } catch (e) {
    if (poller) { clearInterval(poller); poller = null; }
    log("error: " + e.message, "err");
    setStatus("failed");
  }
  busy = false;
};

/* ---------------- tabs ---------------- */

$("tab-batch").onclick = () => {
  $("tab-batch").classList.add("on"); $("tab-live").classList.remove("on");
  $("pane-batch").classList.remove("hide"); $("pane-live").classList.add("hide");
};
$("tab-live").onclick = () => {
  $("tab-live").classList.add("on"); $("tab-batch").classList.remove("on");
  $("pane-live").classList.remove("hide"); $("pane-batch").classList.add("hide");
};

$("threads").value = String(defaultThreads);

if (!self.crossOriginIsolated) {
  log("warning: page is not cross-origin isolated, so SharedArrayBuffer (and " +
      "therefore multithreading) is unavailable. Serve with wasm/serve.py.", "err");
}
