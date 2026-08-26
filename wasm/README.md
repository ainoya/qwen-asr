# Browser (WebAssembly) build

The same C engine, compiled to WebAssembly, running the 1.7B model entirely
inside a tab. No server-side inference, no audio leaving the machine.

```bash
# 1. build the engine
git clone https://github.com/emscripten-core/emsdk && cd emsdk
./emsdk install latest && ./emsdk activate latest && source ./emsdk_env.sh && cd -
./wasm/build.sh

# 2. pack the model (bf16 safetensors -> pre-quantized Q8 image, ~2.2 GB)
mkdir -p qwen3-asr-1.7b-q8
cp qwen3-asr-1.7b/{config.json,generation_config.json,vocab.json,merges.txt} qwen3-asr-1.7b-q8/
./qwen_asr -d qwen3-asr-1.7b --pack-q8 qwen3-asr-1.7b-q8/qwen-asr-q8.bin

# 3. serve and open
./wasm/serve.py            # http://localhost:8765/wasm/demo/
```

The demo has a **batch** tab (audio file or one of the bundled samples) and a
**streaming** tab (microphone, or "Stream sample file" to watch the incremental
path without a mic).

## Why a packed model file

The engine normally quantizes the bf16 safetensors at load time, which is free
when the weights are mmap'd. A browser has to *fetch* them, and 4.7 GB of bf16
is not a download anyone wants. `--pack-q8` writes a single safetensors image
where every large matrix is already stored as Q8 blocks:

| | bf16 safetensors | packed Q8 |
|---|---|---|
| file size | 4.70 GB | 2.18 GB |
| wasm memory | would not fit | ~2.2 GB, used in place |
| load time | quantization pass | pointer attach, ~15 ms |

The page streams the file straight into wasm memory and the weights are read
from those bytes directly — no second copy, no conversion pass. The encoder is
quantized too (f32 encoder weights alone would be 1.2 GB); transcripts are
unchanged.

Native builds pick the packed file up automatically if it is present in the
model directory, which also makes native startup instant.

## Measured performance

Numbers from `wasm/bench-node.js`, which runs the exact same `qwen_asr.wasm` on
Node's worker_threads. **Browser timings are not usable for benchmarking**: a
hidden or unfocused tab is throttled hard (the same 11 s clip took 3.2 s with a
visible tab's worth of CPU and 55 s in a hidden one), and an embedded webview
gave the engine ~1.3 cores no matter how many threads were requested.

| | native (`make blas`) | wasm |
|---|---|---|
| 11s English clip | 1.9 s (5.9x realtime) | 3.2 s (3.5x realtime) |
| 41s Japanese clip | 5.7 s (7.1x) | 12.9 s (3.2x) |
| whole sample set (575 s audio) | — | 166 s (3.46x) |

So wasm lands at roughly 2.2x the native time — normal for SIMD128 without
Accelerate's AMX-backed sgemm.

The encoder's conv stem now convolves mel chunks in groups rather than one at
a time (see the top level README): 1653 ms -> 1159 ms on the 41s clip. What
remains in wasm on the 41s clip, with the GPU decoder handling prefill and
generation, is mel 254 ms, conv stem 1159 ms and the encoder transformer
1254 ms.

For recordings long enough to segment, `_qwen_wasm_set_batch_size(n)` turns on
batched segment decoding (the demo exposes it as the "batch" control next to
"segment"). Generating a token reads every decoder weight once, so decoding
several segments in lockstep costs barely more than one; it needs a segment
size set and past-text conditioning off. See the `--batch` section of the top
level README for the native measurements.

Quality is checked with the same character-error metric as the native suite:

```bash
node wasm/check-node.js qwen3-asr-1.7b-q8 samples 8
# wasm check: 22/22 within 0.15, aggregate norm error 0.0076
```

## WebGPU backend

`wasm/demo/webgpu-decoder.js` runs **the whole decoder** — prefill and token
generation — on the GPU. Only mel and the audio encoder stay in wasm. Pick it
with the *decoder* dropdown in the demo, or check it against the CPU path with
`wasm/demo/webgpu-test.html` (its `Run all samples` button diffs every sample
against the wasm decoder).

The two halves are on the GPU for different reasons:

- **Generation** reads all 1.7 GB of decoder weights once per token, so it is
  bandwidth-bound. `wasm/demo/webgpu-probe.html` measures 122 GB/s from a
  compute shader against 87 GB/s for the CPU path.
- **Prefill** is a batched GEMM and compute-bound: 549 prompt tokens is
  ~1.55 TFLOP, which wasm grinds through at ~220 GFLOPS.

Measured on an M1 Pro, 1.7B, in a throttled background tab (so these are floors
and useful mainly as ratios). Output is **identical to the wasm decoder on every
sample**, English and Japanese, at contexts from 145 to 1170 tokens.

| clip | all wasm | wasm encoder + GPU decoder | prefill: wasm → GPU |
|------|----------|----------------------------|---------------------|
| 10s English | 3.14 s (3.2x realtime) | **2.14 s (4.7x)** | 1.77 s → 0.59 s |
| 11s English | 3.47 s (3.2x) | **2.39 s (4.6x)** | 1.95 s → 0.63 s |
| 41s Japanese | 14.7 s (2.8x) | **11.4 s (3.6x)** | 7.03 s → 2.44 s |
| 89s English | 36.7 s (2.4x) | **26.0 s (3.4x)** | 16.3 s → 7.32 s |

So GPU prefill is **2.2–3.1x** faster than the wasm one and the whole pipeline
gains 1.3–1.5x. Generation lands at 22–27 ms/token roughly independent of
context length.

With that done the encoder is the largest remaining piece — for the 41s clip the
split is mel+encoder 3.9 s (wasm), GPU prefill 2.5 s, GPU generation 3.4 s.

### Design notes

- **Weights** are uploaded once into two storage buffers holding exactly the
  bytes the packed Q8 file already contains. `maxStorageBufferBindingSize` is
  4.29 GB on Apple silicon, so the 1.72 GB of quants binds in one piece.
- **Dequantization** uses `unpack4x8snorm` (core WGSL, one instruction, gives
  `q/127`) so multiplying by `scale * 127` recovers the weight. `dot4I8Packed`
  is *not* used: Apple GPUs emulate it, and the probe measures 24 GMAC/s against
  the 1.72 GMAC every token needs.
- **Prefill activations are stored transposed**, `[dim][seqPad]`. Every kernel's
  lanes then run along the sequence axis, which makes all global access
  coalesced and turns row-wise reductions (RMS norm, per-head norm) into
  per-lane loops with no cross-lane traffic.
- **The prefill GEMM** is a 64x64 output tile per workgroup, 16x16 threads each
  owning a 4x4 block, with the weight and activation tiles staged in workgroup
  memory. Tiling in the row dimension is what stops the weights being re-read
  once per sequence position.
- **The token id stays on the GPU** between generation steps, so the embedding
  lookup for step N+1 reads the argmax of step N with no round trip through JS.
- **The RoPE cos/sin table is precomputed on the CPU.** Angles grow with
  position and WGSL only guarantees accuracy for small arguments.
- **Activations stay in f32** in generation by default (one dispatch fewer per
  matvec, ~15% faster); `setQuantizeActivations(true)` reproduces the CPU's int8
  activation quantization instead. Both were verified to match.
- **The attention V-sum is split over the key axis.** One thread per (head, dim)
  is only 2048 threads and dominated the step at long contexts: 42 ms/token at a
  1170-token context against 26.5 after splitting into 8 slices plus a merge.

### Will this machine run it?

`wasm/demo/webgpu-report.html` answers that in a few seconds **without
downloading the model**. The requirement is a property of the model's shape, so
it is known up front; the old flow only discovered a machine could not run the
GPU path after fetching 2.18 GB.

The number that decides it: the decoder packs every Q8 matrix into **one
storage buffer of 1.72 GB**, and WebGPU only guarantees
`maxStorageBufferBindingSize` of 128 MiB. A machine that offers the baseline
cannot create that binding at all, whatever its GPU is worth. The report prints
the requirement against the adapter's limits, then actually attempts the
allocation, because a limit is a promise and an allocation is a fact.

It also measures coalesced read bandwidth and projects a decode step from it -
one step reads the whole decoder once, so that is the floor. On an M1 Pro in a
background tab: 72.7 GB/s, projecting 25.1 ms/token, against 22-27 ms/token
actually observed. The projection is worth trusting as a screen.

Output ends with a one-line JSON to paste back, which is how to collect
coverage from machines this repo has never run on.

### The audio tower on the GPU

`wasm/demo/webgpu-encoder.js` runs the whole audio tower - the Conv2D stem and
the 24 transformer layers - leaving only mel in wasm. Check it with
`wasm/demo/webgpu-encoder-test.html`, which has two buttons: the transformer
half alone, fed the C encoder's conv-stem output, and the whole tower from mel.
Both score against tensors tapped out of the C encoder by
`node wasm/dump-golden.js … --enc`.

    transformer half: 23/23 within 2e-3 relative error, worst 1.5e-5
    whole tower:      23/23 within 2e-3 relative error, worst 9.6e-4

Measured in a background tab, so these are floors:

| clip | wasm stem + transformer | GPU tower |
|------|-------------------------|-----------|
| 11s | 1.4 s | **0.35 s** |
| 41s | 2.4 s | **0.88 s** |
| 89s | 5.0 s | **1.51 s** |

The demo uses it automatically when the GPU backend is selected and the tower
starts; if it does not, mel and the encoder stay in wasm and only the decoder
moves. mel is still wasm either way - it is 254 ms on the 41s clip and not
worth a port.

Running the whole pipeline that way changes nothing measurable in the output:

```text
wasm encoder + GPU decoder: 23/23 within 0.15, cer gpu 0.0062 vs cpu 0.0024
GPU tower   + GPU decoder: 23/23 within 0.15, cer gpu 0.0062 vs cpu 0.0024
```

It does not share the decoder's shaders because the tower is a different
network: LayerNorm with a bias rather than RMSNorm, a bias on every linear,
attention that is bidirectional inside a fixed 104-token window with no RoPE
and no grouped KV, and a GELU FFN rather than SwiGLU. What it does share is the
transposed `[dim][seqPad]` activation layout and the 128x64 tiled GEMM.

The stem is the same tiled GEMM with the activation tile **gathered** rather
than read contiguously: the loader turns a (K, column) pair into
(in_channel, tap) and (batch, out_row, out_col) and reads the input pixel
directly. That is im2col without ever building it - materializing it would cost
55 MB per chunk group for the second layer. Chunks are convolved in groups of
eight for the same reason the C encoder groups them: one 100-frame chunk alone
gives the second layer a 480x4320x800 GEMM, too narrow to be worth a dispatch.

Three bugs worth remembering, on top of the four below:

5. **`layout: "auto"` infers `hasDynamicOffset: false`**, so stepping a uniform
   with `setBindGroup(0, bg, [offset])` is a validation error - and validation
   errors do not throw. A 24-layer run finished in 3 ms with an all-zero
   result. Declare the bind group layout explicitly.
6. **A fractional buffer region size** poisons every offset after it.
   `region(128 * (tokens / w3 + G + 1) * cw)` is not an integer unless tokens
   divides w3, and `writeBuffer` rejects fractional offsets - on one sample
   only, which is what made it look like a data bug.
7. **Reading a field that a later call initializes.** `convGroup` was set in
   `prepare()` but read by `convPlan()`, which runs first; undefined made the
   planner fall back to one chunk per group and overrun the uniform buffer,
   again on the first sample only.

### Verifying GPU work from a background tab

`webgpu-test.html` runs the wasm decoder and the GPU decoder on the same audio
and diffs them, which is the right check but only usable with the window in
front. Measured on a tab that is not in front:

| | background | in front |
|---|---|---|
| GPU storage-buffer read | 92 GB/s | 122 GB/s |
| wasm inference | ~1.2 cores' worth | all threads |
| `setTimeout(0)` yield | ~1000 ms | ~0 ms |
| `MessageChannel` yield | 0.005 ms | 0.005 ms |

So GPU work is barely affected, wasm inference is unusable, and any poll loop
written with a timer runs at the clamp rate rather than the work's rate. That
last one is fixed for good in `tick.js`, which all the harnesses now use.

The other two say where the CPU reference belongs: not in the page.
`wasm/dump-golden.js` runs it under Node, which has no such limits, and writes
each sample's decoder input embeddings plus the transcript the wasm decoder
produces from them:

```bash
node wasm/dump-golden.js qwen3-asr-1.7b-q8 samples   # -> wasm/demo/golden/
```

`webgpu-golden.html` then feeds those embeddings straight to the GPU decoder
and scores the result, running **no wasm inference at all**. It reports the
same normalized character error the native suite and `check-node.js` use, for
both paths side by side:

```text
golden check: 23/23 within 0.15, aggregate cer gpu 0.0062 vs cpu 0.0024
              (18/23 byte-identical to the wasm decoder)
```

Byte-identical output is **not** the bar and is not a stable property: the two
paths differ numerically and greedy decoding flips on near-tied logits, so a
sample can swap sides without either path being wrong. The five that differ are
all one-word coin flips that then propagate — "back out on the road" against
"back down the road", "you can't let them" against "you're gonna let them" —
and in at least one case the GPU matches the reference and the CPU does not.
What matters is that the GPU stays as close to the reference as the CPU does,
which at 0.6% against 0.24% character error it does, both far under the 15%
threshold. It works in a
background tab, and because the page stays open, the 2.18 GB model load and
1.72 GB weight upload are paid once per session rather than once per edit.
For a kernel edit loop, `await reload()` re-imports `webgpu-decoder.js` with a
fresh cache key against the weights already resident, and `await runAll()`
re-checks — seconds per iteration.

One decoder context is shared across samples, so `runAll()` refuses to start
while another run is in flight; two overlapping runs corrupt each other's KV
state and every sample comes out DIFF.

### Four bugs worth remembering

All four presented as "the GPU math is wrong" and none of them were.

1. **Byte-level BPE broke per-token string decoding.** A token is often *part* of
   a UTF-8 sequence; a CJK character routinely spans two or three. Decoding each
   piece with `UTF8ToString` produces replacement characters that can never be
   recombined — English came out byte-identical while individual Japanese glyphs
   corrupted. Fix: a streaming `TextDecoder`, matching what the C path does by
   appending raw bytes.
2. **A missing `COPY_DST` on the activation buffer** made `writeBuffer` a silent
   no-op, so the prefill ran on zeros. Symptom: impossibly fast and wrong. Fix
   the flag, and route `onuncapturederror` somewhere visible — WebGPU validation
   errors do not throw.
3. **Uniform parameters were filled but never uploaded.** Generation happened to
   work because its per-step update re-uploaded them; prefill reads them first
   and got zeros.
4. **A shader helper that was never actually inserted** left one pipeline
   referencing an undefined function. An invalid pipeline makes every dispatch a
   silent no-op, so `getCompilationInfo()` is now checked at init.

## Harnesses

| script | what it does |
|---|---|
| `wasm/bench-node.js` | time one file, print the phase breakdown |
| `wasm/check-node.js` | run every sample with a reference, report error rate |
| `wasm/stream-node.js` | push a file through the streaming API (`--realtime` to pace it) |
| `wasm/demo/webgpu-probe.html` | measure WebGPU bandwidth and int8 throughput on this machine |
| `wasm/demo/webgpu-test.html` | run the same audio through both decoders and diff the transcripts |

## Requirements and gotchas

- **Cross-origin isolation.** The thread pool needs `SharedArrayBuffer`, which
  needs COOP/COEP headers. `wasm/serve.py` sets them; a plain
  `python3 -m http.server` will not, and the demo will warn and run
  single-threaded.
- **Pointers above 2 GB.** With a >2 GB heap every allocation after the model
  sits above 2^31 and comes back from wasm as a negative i32. The demo funnels
  every pointer through `P(x) = x >>> 0` and never uses `>> 2` to index
  `HEAPF32`.
- **Spin budget.** A futex round trip between Web Workers measured ~4 ms on
  Chrome, and a decoder token issues ~110 parallel dispatches. The pool spins
  ~1M iterations before parking, which is what keeps wasm decode at 88 ms/token
  instead of 990.
- **Thread count.** Ask for the performance-core count, not
  `hardwareConcurrency`. Spinning threads are counter-productive when the
  browser hands out fewer cores than requested.
