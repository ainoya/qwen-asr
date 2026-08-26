/*
 * app.js - demo UI for the WebAssembly build.
 *
 * The wasm module is instantiated on this (main) thread, because emscripten
 * creates its pthread workers from wherever the module lives and nesting
 * workers inside another worker is fragile. Nothing here blocks: transcription
 * runs on a wasm thread and the page polls for tokens and completion.
 */

import { WebGPUDecoder } from "./webgpu-decoder.js";
import { tick, until, sleep } from "./tick.js";
import { WebGPUEncoder } from "./webgpu-encoder.js";

const $ = (id) => document.getElementById(id);
const MODEL_BASE = "../../qwen3-asr-1.7b-q8";

let Module = null;
let ready = false;
let busy = false;
let sampleBuf = 0;
let sampleCap = 0;
let poller = null;
let gpu = null, encoder = null;

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

function drainText() {
  const p = P(Module._qwen_wasm_take_text());
  if (!p) return;
  const s = Module.UTF8ToString(p);
  Module._free(p);
  if (s) $("out").textContent += s;
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
  Module._qwen_wasm_set_stream_params(Number($("chunk").value) || 0, 32,
                                      Number($("encwin").value) || 0);
}

/* ---------------- model loading ---------------- */

/* Stream the packed model straight into wasm memory: response.arrayBuffer()
 * would hold a second ~2 GB copy in the JS heap first. */
async function fetchModelInto(url, total) {
  const ptr = P(Module._qwen_wasm_alloc(total));
  if (!ptr) throw new Error(`could not allocate ${total} bytes of wasm memory`);

  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const reader = res.body.getReader();

  let off = 0, lastReport = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    // Re-read HEAPU8 every time: growing memory swaps the backing buffer.
    Module.HEAPU8.set(value, ptr + off);
    off += value.length;
    if (off - lastReport > 48 * 1024 * 1024) {
      lastReport = off;
      const pct = (off / total * 100);
      $("barfill").style.width = pct.toFixed(1) + "%";
      setStatus(`downloading model ${(off / 1e9).toFixed(2)} / ${(total / 1e9).toFixed(2)} GB`);
      await tick();
    }
  }
  if (off !== total) throw new Error(`model truncated: ${off} of ${total}`);
  $("barfill").style.width = "100%";
  return ptr;
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

    const head = await fetch(`${MODEL_BASE}/qwen-asr-q8.bin`, { method: "HEAD" });
    if (!head.ok) throw new Error(`model: HTTP ${head.status}`);
    const total = Number(head.headers.get("content-length"));
    if (!total) throw new Error("server did not report Content-Length for the model");

    log(`fetching packed model, ${(total / 1e9).toFixed(2)} GB`);
    const ptr = await fetchModelInto(`${MODEL_BASE}/qwen-asr-q8.bin`, total);

    setStatus("attaching weights...");
    const threads = Number($("threads").value) || 8;
    const dir = cstr("/model");
    const rc = Module._qwen_wasm_init(ptr, total, dir, threads, 0);
    Module._qwen_wasm_release(dir);
    if (rc !== 0) throw new Error("qwen_wasm_init failed");

    if ($("backend").value === "gpu") {
      const probe = await WebGPUDecoder.probe();
      if (!probe.ok) {
        log(`WebGPU unavailable (${probe.why}); falling back to the wasm decoder`, "err");
        $("backend").value = "cpu";
      } else {
        setStatus("uploading weights to the GPU...");
        gpu = new WebGPUDecoder(Module);
        await gpu.init((m) => setStatus(m));
        log(`GPU decoder ready (${(gpu.weightBytes / 1e9).toFixed(2)} GB resident on the GPU)`);

        /* The audio tower is optional: if it will not start, mel and the
         * encoder stay in wasm and only the decoder moves. */
        try {
          setStatus("uploading the audio tower to the GPU...");
          const e = new WebGPUEncoder(Module);
          await e.init(() => {});
          encoder = e;
          log(`GPU audio tower ready (${(e.weightBytes / 1e6).toFixed(0)} MB)`);
        } catch (err) {
          log(`GPU audio tower unavailable (${err.message}); mel and the encoder stay on the cpu`, "err");
        }
      }
    }

    ready = true;
    sendSettings();
    for (const id of ["file", "sample-ja", "sample-en", "mic", "simstream"])
      $(id).disabled = false;
    setStatus(`ready — ${((performance.now() - t0) / 1000).toFixed(1)}s, ${threads} threads`);
    log("model ready");
  } catch (e) {
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
      Module.HEAPF32.set(samples, f32idx(ptr));
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
        Module.HEAPF32.set(encOut, f32idx(ep));
        seq = Module._qwen_wasm_embeds_from_enc(ep, encoder.tokens);
        Module._qwen_wasm_release(ep);
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
      log("error: " + e.message, "err");
      setStatus("failed");
    }
    busy = false;
    return;
  }

  try {
    Module.HEAPF32.set(samples, f32idx(ptr));
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
      $("out").textContent = Module.UTF8ToString(rp);
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
$("sample-ja").onclick = () => runSample("../../samples/extra/ja_bench.wav");
$("sample-en").onclick = () => runSample("../../samples/jfk.wav");

/* ---------------- streaming ---------------- */

let audioCtx = null, micStream = null, micNode = null, streamedSamples = 0;

$("mic").onclick = async () => {
  if (!ready || busy) return;
  if ($("backend").value === "gpu")
    log("streaming uses the wasm decoder; the GPU backend handles batch only", "err");
  busy = true;
  streamedSamples = 0;
  $("out").textContent = "";
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
      Module.HEAPF32.set(s, f32idx(ptr));
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

  const rp = P(Module._qwen_wasm_stream_finish());
  if (poller) { clearInterval(poller); poller = null; }
  drainText();
  if (rp) {
    $("out").textContent = Module.UTF8ToString(rp);
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
  $("perf").textContent = "";
  sendSettings();

  try {
    const url = $("lang").value === "English" ? "../../samples/jfk.wav"
                                              : "../../samples/extra/ja_bench.wav";
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
      Module.HEAPF32.set(part, f32idx(ptr));
      Module._qwen_wasm_stream_push(ptr, part.length);
      streamedSamples += part.length;
      await sleep(250);
    }

    setStatus("draining...");
    const rp = P(Module._qwen_wasm_stream_finish());
    if (poller) { clearInterval(poller); poller = null; }
    drainText();
    if (rp) {
      $("out").textContent = Module.UTF8ToString(rp);
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
