import "./app.js";
import { freshHeap } from "./heap.js";

const $ = (id) => document.getElementById(id);
const report = (value) => { $("results").textContent += "\n" + JSON.stringify(value); };
let gpu, M, golden;

async function action(fn) {
  for (const id of ["prepare", "profile", "regression", "kernels", "fallback"]) $(id).disabled = true;
  try { await fn(); } catch (e) { report({ error: e.stack || String(e) }); }
  finally {
    const ready = !!gpu?.ready;
    $("prepare").disabled = ready;
    $("profile").disabled = $("regression").disabled = !ready;
    $("kernels").disabled = $("fallback").disabled = !ready;
  }
}

$("prepare").onclick = () => action(async () => {
  await $("load").onclick();
  ({ gpu, Module: M } = window.__asr());
  if (!gpu?.ready) throw new Error("Model failed to load: " + $("out").textContent);
  // Long golden fixtures need more than the production 1600-token default.
  // prepareContext still enforces the actual device's buffer limit.
  gpu.opts.maxSeq = 2300;
  golden = await (await fetch("./golden/index.json")).json();
  const info = gpu.adapterInfo;
  report({ ready: true, adapter: {vendor: info.vendor, architecture: info.architecture,
    device: info.device, description: info.description, subgroupMinSize: info.subgroupMinSize,
    subgroupMaxSize: info.subgroupMaxSize}, tiled: gpu.tiledPrefillSupported,
    features: [...gpu.device.features], kvF16: gpu.kvF16 });
});

async function withEmbeddings(entry, fn) {
  const response = await fetch(`./golden/${entry.name}.f32`);
  if (!response.ok) throw new Error(`Missing fixture ${entry.name}`);
  const data = new Float32Array(await response.arrayBuffer());
  if (data.length !== entry.seq * golden.dim) throw new Error("Wrong embedding shape");
  const ptr = M._qwen_wasm_alloc(data.byteLength) >>> 0;
  if (!ptr) throw new Error("Embedding allocation failed");
  try {
    freshHeap(M).HEAPF32.set(data, ptr / 4);
    return await fn(ptr);
  } finally { M._qwen_wasm_release(ptr); }
}

$("profile").onclick = () => action(async () => {
  if (!gpu.pipe.preScoresTiled) throw new Error("Tiled kernel is not enabled on this adapter");
  const entry = golden.samples[0];
  await withEmbeddings(entry, async (ptr) => {
    for (let run = 0; run < 4; run++) {
      for (const tiled of run % 2 ? [true, false] : [false, true]) {
        gpu.useTiledPrefillScores = tiled;
        await gpu.prefillAndGenerate(ptr, entry.seq, 0, null);
        report({ run, tiled, prefillWallMs: gpu.prefillMs, profile: await gpu.profilePrefill() });
      }
    }
  });
  report({ profileDone: true });
});

const normalize = (s) => s.toLowerCase().replace(/[.,!?;:'"()\[\]{}‘’“”]/g, "")
  .replace(/\s+/g, " ").trim();
function cer(a, b) {
  a = normalize(a); b = normalize(b);
  const row = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    let diagonal = row[0]; row[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const old = row[j];
      row[j] = Math.min(row[j] + 1, row[j - 1] + 1, diagonal + (a[i - 1] !== b[j - 1]));
      diagonal = old;
    }
  }
  return row[b.length] / Math.max(1, b.length);
}

$("regression").onclick = () => action(async () => {
  if (!gpu.pipe.preScoresTiled) throw new Error("Tiled kernel is not enabled on this adapter");
  let passed = 0;
  for (const entry of golden.samples) {
    await withEmbeddings(entry, async (ptr) => {
      gpu.useTiledPrefillScores = false;
      const baseline = await gpu.prefillAndGenerate(ptr, entry.seq, 640, null);
      gpu.useTiledPrefillScores = true;
      const tuned = await gpu.prefillAndGenerate(ptr, entry.seq, 640, null);
      const error = cer(tuned.text, entry.text);
      const exactIds = baseline.ids.join() === tuned.ids.join();
      if (error <= 0.15 && exactIds) passed++;
      report({ name: entry.name, seq: entry.seq, cpuCer: error,
        cpuExact: tuned.text === entry.text, baselineExactIds: exactIds });
    });
  }
  report({ regressionDone: true, passed, total: golden.samples.length });
  if (passed !== golden.samples.length) throw new Error("Golden regression failed");
});

$("kernels").onclick = () => action(async () => {
  if (!gpu.pipe.preScoresTiled) throw new Error("Tiled kernel is not enabled on this adapter");
  const { device, cfg } = gpu;
  for (const seq of [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 549, 1024]) {
    gpu.prepareContext(seq, 0, { prefillSeq: seq });
    const sp = gpu.seqPad;
    const input = Float32Array.from({ length: (cfg.qDim + cfg.kvDim) * sp },
      (_, i) => ((i * 17 + Math.floor(i / sp) * 13) % 101 - 50) / 37);
    device.queue.writeBuffer(gpu.bufAct, gpu.PT.qkv * 4, input);
    const sentinel = -12345;
    const initial = new Float32Array(cfg.heads * sp * sp).fill(sentinel);
    const read = device.createBuffer({ size: initial.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const outputs = [];
    try {
      for (const tiled of [false, true]) {
        device.queue.writeBuffer(gpu.bufScratch, 0, initial);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(tiled ? gpu.pipe.preScoresTiled : gpu.pipe.preScores);
        pass.setBindGroup(0, gpu.bindGroup(0), [gpu.off.pre.layer[0].scores * 256]);
        pass.dispatchWorkgroups(Math.ceil(seq / (tiled ? 32 : 64)),
          tiled ? Math.ceil(seq / 32) : seq, cfg.heads);
        pass.end();
        enc.copyBufferToBuffer(gpu.bufScratch, 0, read, 0, initial.byteLength);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        outputs.push(new Float32Array(read.getMappedRange().slice(0)));
        read.unmap();
      }
      let maxDiff = 0, cpuMaxDiff = 0;
      for (let h = 0; h < cfg.heads; h++) for (let i = 0; i < sp; i++) {
        for (let j = 0; j < sp; j++) {
          const index = (h * sp + i) * sp + j;
          const actual = outputs[1][index];
          if (!Number.isFinite(actual)) throw new Error(`Nonfinite score at ${seq}:${index}`);
          maxDiff = Math.max(maxDiff, Math.abs(actual - outputs[0][index]));
          if ((i >= seq || j > i) && actual !== sentinel) throw new Error("Causal mask/padding overwritten");
          if (i < seq && (j === 0 || j === i)) {
            let expected = 0;
            for (let d = 0; d < cfg.headDim; d++) {
              expected += input[(h * cfg.headDim + d) * sp + i] *
                input[(cfg.qDim + Math.floor(h / cfg.headsPerKv) * cfg.headDim + d) * sp + j];
            }
            expected /= Math.sqrt(cfg.headDim);
            cpuMaxDiff = Math.max(cpuMaxDiff, Math.abs(expected - actual));
          }
        }
      }
      report({ kernelSeq: seq, maxDiff, cpuMaxDiff });
      if (maxDiff > 1e-5 || cpuMaxDiff > 1e-4) throw new Error("Score accuracy check failed");
    } finally { read.destroy(); }
  }
  report({ kernelsDone: true });
});

$("fallback").onclick = () => action(async () => {
  // Compile only portable f32 shaders on the same real device. This exercises
  // the fallback kernels, not a claim of testing physical non-Apple hardware.
  gpu.ready = false;
  gpu.hasSubgroups = gpu.hasScoreSubgroups = gpu.hasF16 = false;
  gpu.tiledPrefillSupported = gpu.useTiledPrefillScores = gpu.kvF16 = false;
  gpu.kvBytes = 4;
  gpu.bufKV?.destroy(); gpu.bufKV = null; gpu.kvCap = 0; gpu._bind = null;
  await gpu.finishInit();
  const entry = golden.samples.find((e) => e.name === "jfk");
  await withEmbeddings(entry, async (ptr) => {
    const full = await gpu.prefillAndGenerate(ptr, entry.seq, 256, null);
    const split = 65;
    await gpu.prefillOneShot(ptr, split, 0, null, entry.seq + 256);
    const suffix = await gpu.prefillSuffixAndGenerate(ptr, entry.seq, split, 256, null);
    const error = cer(full.text, entry.text);
    const exactIds = full.ids.join() === suffix.ids.join();
    report({ fallbackDone: true, cpuCer: error, suffixExactIds: exactIds,
      subgroups: gpu.hasSubgroups, kvF16: gpu.kvF16 });
    if (error > 0.15 || !exactIds) throw new Error("Portable fallback failed");
  });
  report({ note: "Reload the page before running optimized checks again." });
});
