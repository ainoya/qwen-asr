# What this fork adds to antirez/qwen-asr

This repository is a fork of [antirez/qwen-asr](https://github.com/antirez/qwen-asr),
Salvatore Sanfilippo's pure-C inference engine for the Qwen3-ASR speech-to-text
models. The fork keeps the upstream engine's design and contracts intact and
extends it in four directions, over ~50 commits on the
`perf/q8-wasm-webgpu` branch:

1. **Native performance**: ~1.9x faster inference on Apple Silicon, with the
   remaining ceilings measured and documented rather than guessed.
2. **A WebAssembly port** of the whole engine, running in the browser and in
   Node with SIMD128 + pthreads.
3. **A WebGPU backend** that runs the entire model - audio tower, decoder
   prefill and generation - on the GPU, with wasm reduced to mel extraction
   and orchestration, and browser memory cut roughly in half.
4. **Deployment**: a public playground on GitHub Pages fetching a packed
   model from Hugging Face, plus quantization-calibration tooling and a
   Japanese evaluation suite used to make (and reject) quantization
   decisions with measurements.

Try it: **[ainoya.github.io/qwen-asr](https://ainoya.github.io/qwen-asr/)**
(WebGPU-capable Chrome/Edge; ~2.2 GB one-time model download, cached in OPFS).
Model: **[ainoya/qwen3-asr-1.7b-q8-packed](https://huggingface.co/ainoya/qwen3-asr-1.7b-q8-packed)**.

Everything below was measured on an Apple M1 Pro with the 1.7B model unless
stated otherwise. `AGENT.md` is the engineering log with the full details and
the list of things that were tried and did not survive measurement.

---

## 1. Native engine

### Packed Q8 model image (`--pack-q8`)

Upstream loads bf16 safetensors and quantizes the decoder at load time. The
fork adds a pre-quantized image, `qwen-asr-q8.bin`: one safetensors-format
file with the decoder layers, the tied embedding / LM head *and* the audio
tower stored as block-int8 (64 weights per f32 scale, ~1.06 bytes/weight),
the SwiGLU gate/up pair pre-fused into one matrix. It attaches by mmap in
~15 ms instead of converting at every load, and it is **2.18 GB instead of
4.70 GB** - which is also what makes the browser build feasible. The LM head
deliberately stays bf16 by default (`--weights q8`): on the 0.6B model an
int8 head flips near-tied logits and drops phrases; `--weights q8-lm`
quantizes it too, which is output-identical on 1.7B.

### Batched segment decoding (`--batch`, default 4)

In segmented mode (`-S <secs>`), independent segments are decoded in
lockstep: one multi-stream prefill, then batched single-token steps that
sweep the weights once per step for all streams. Token generation is
memory-bandwidth-bound, so amortizing the weight read is the one lever that
matters: generation time on 25 minutes of real speech went **55.5 s → 27.8 s
(2.0x)**. Batch 4 was re-measured against 8 and 16 and wins - beyond 4 the
batched step is attention-bound, not weight-bound.

### Other native work

- **Panel dequantization for prefill**: prefill dequantizes Q8 weights into
  large f32 panels and calls `sgemm` on them, reaching ~1.4 TFLOPS - which
  is exactly the M1 Pro's two AMX blocks at their measured load-issue bound
  (610-680 GFLOPS each, Bhan, arXiv:2606.25426). A hand-written AMX kernel
  has nothing structural left to win here; this is documented so nobody
  tries.
- **Batched Q8 matvec** for short sequences, and a **grouped conv2d stem**
  (mel chunks convolved in groups so the GEMM is wide enough to keep the
  machine busy; the conv2 GEMM runs at ~2.4 TFLOPS on Accelerate).
- **f16 KV cache** (native and GPU): KV RAM halves (458 → 229 MB on the
  25-minute workload), batched decode gets ~2.7% faster because it is
  attention-bound and reads half the bytes. Cost measured honestly:
  Japanese CER 0.1639 → 0.1694 (+3.4% relative), no degeneration. The
  f16→f32 read is branchless on purpose - a branchy converter stopped
  clang auto-vectorizing the wasm attention loops and cost 42%.
- **Runaway-segment guard**: bounded detection of degenerate repetition
  loops in segmented decoding, so a collapsing segment costs a truncation
  rather than the whole run.
- **Thread pool**: hybrid spin/park barrier with work stealing. Budgets are
  iteration counts, not wall-clock - a worker reading a clock goes through
  JS under wasm and was measurably worse.

**Native result**: a 25-minute real-speech file transcribes in ~93 s
(**~16x realtime**) at `-S 30 --batch 4`, with every phase at a measured
hardware ceiling (AMX for prefill, DRAM bandwidth for generation - the Q8
matvec reads at 86.6 GB/s against a 94.7 GB/s pure-read ceiling).

## 2. WebAssembly port

`wasm/` builds the unmodified engine with Emscripten (SIMD128 + pthreads):

- `qwen_asr_kernels_wasm.c` - SIMD128 hot kernels (Q8 matvec/argmax/
  quantize). The portable C kernels are kept auto-vectorizable on purpose;
  the wasm Q8 matvec is within 1.22x of the hand-written NEON one.
- Browser demo (`wasm/demo/`) with batch, simulated-stream and microphone
  modes, provisional-text rendering, and OPFS model caching.
- Node harnesses (`bench-node.js`, `check-node.js`, `stream-node.js`) -
  browser tabs are throttled and useless for benchmarking, so the same
  `qwen_asr.wasm` runs under Node for regression and timing. The English
  suite runs 22/22 at ~3.5x realtime in Node on 8 threads.
- Emscripten-specific traps are documented in `AGENT.md` (worker futex
  round-trips cost ~4 ms, so the pool spins; pointers above 2 GB need
  `>>> 0` at every EM_ASM boundary; heap views on the main thread go stale
  when a worker grows memory - see `wasm/demo/heap.js`).

## 3. WebGPU backend

The entire model runs on the GPU: the audio tower (conv stem + windowed
bidirectional transformer), decoder prefill, and autoregressive generation.
wasm keeps mel extraction and prompt orchestration. All of it is
feature-gated with clean fallbacks (wasm decode when WebGPU is absent, full
reload on device loss).

Highlights, each measured (details and the full trial-and-error log in
`AGENT.md`):

- **Prefill as transposed GEMMs**: activations stored `[dim][seqPad]` so
  every kernel's lanes run along the sequence axis; 64x64-tile GEMM with
  both operands staged in workgroup memory, ~1.45 TFLOPS - already above
  the ~17-20%-of-peak that published WebGPU matmul work reaches.
- **Generation matvec at the bandwidth floor**: lane-per-word Q8 matvec at
  ~80% of the device's pure-read probe (157 GB/s); ~15 ms/token at short
  contexts. Weights are sharded into ≤256 MB bindings so the 1 GiB
  `maxStorageBufferBindingSize` tier (most desktop GPUs) suffices.
- **Subgroup reductions** everywhere they help: matvec row sums 1.35x, the
  generation score pass 6x, encoder attention 1.9x - with tree fallbacks
  kept for devices without the feature.
- **Batched generation steps**: 8 steps per submit with the token id kept in
  a GPU buffer (no readback between steps); cut ~8 ms/token of submit
  overhead to 1-2 ms.
- **f16 KV cache** on the GPU too; the largest KV binding halves.
- **Suffix prefill for streaming**: KV-sourced attention kernels extend a
  retained cache instead of re-prefilling from scratch. Three-stage split
  prefills match one-shot transcripts byte-for-byte on all 23 golden
  samples.
- **Chunked long prefills**: past 1024 tokens the prefill proceeds in
  512-row chunks through the suffix path, so the attention score scratch is
  linear, not quadratic (164 MB → 52 MB at a 1600-token context).
- **Fused SwiGLU**: the packed gate/up matrix interleaves each FFN pair, so
  the workgroup that computed both dots applies silu(g)·u at writeback -
  one dispatch per layer less on both paths, prefill activation arena
  66 → 38 MB, bit-identical outputs.
- **RoPE tables precomputed on the CPU** (WGSL trig accuracy is only
  guaranteed for small arguments), `dot4I8Packed` avoided (emulated on
  Apple GPUs), and GPU timestamp profiling built in
  (`profileStep`/`profilePrefill`/`profileRun`).

**Verification discipline**: `webgpu-golden.html` scores the GPU pipeline
against 23 reference transcripts dumped by the native engine -
**23/23 pass, 18/23 byte-identical to the wasm decoder**, and that
byte-identity count survived every optimization above; changes that moved
it were rejected or explained.

## 4. Browser memory: GPU-resident weights

wasm memory can never shrink, so the only way to stop the heap from holding
weights the GPU also holds is to never materialize them in wasm at all. With
the GPU backend:

- JS parses the safetensors header and builds a **reduced image for wasm
  containing only the decoder norms - 0.5 MB instead of 2.18 GB**.
- Decoder layers, the audio tower and the tied embedding / LM head upload
  to the GPU straight from a random-access source: the OPFS cache when it
  can hold the file, else a transient JS-side copy in 256 MB chunks that is
  dropped after upload (JS memory, unlike wasm memory, is actually
  returned).
- Prompt assembly fetches the few embedding rows it needs *back* from the
  GPU through a batched, LRU-cached hook (`qwen_set_token_embed_hook`); a
  warm run reads nothing back. Every CPU decode path refuses cleanly when
  weights are absent; a GPU failure reloads once with the full image.

Steady-state browser memory: **~2.3 GB (GPU 2.2 GB + wasm ~0.1 GB) instead
of ~4.4 GB**.

Two bugs this surfaced are documented because they will bite anyone doing
the same: model-variant detection must not depend on tensors the reduced
image drops (it silently mis-detected 0.6B and halved every dimension), and
main-thread heap views go stale when a worker grows pthreads memory - reads
throw, but writes through a stale view *silently vanish*
(`wasm/demo/heap.js`).

## 5. Streaming

- **Provisional text**: `qwen_set_partial_callback()` surfaces the
  hypothesis tail the loop already computes; the CLI draws it dimmed
  (`--partial`), the demo renders it as a live span.
- **GPU streaming decode** through a pluggable decoder hook with suffix
  prefill (no KV restart per chunk).
- **1-second chunks by default on the GPU path**: GPU decode brought the
  per-chunk cost to ~0.8 s at a 41 s context in a *throttled* tab, so
  halving the chunk (2 s → 1 s) halves how long words sit unconfirmed on
  screen while staying ahead of realtime (measured 1.19x realtime over a
  full 41 s Japanese stream, zero hook failures).

## 6. Quantization research tooling (and the 4-bit verdict)

The fork adds the machinery to make quantization decisions from
measurements, and used it to *reject* 4-bit as a default:

- **Activation calibration** (`--calib-out` / `--calib-rank`): per-input-
  channel accumulators on all 169 quantized decoder matrices, dumped and
  merged across runs. Activation-weighted error re-ranks matrices by 15.7%
  on average (75.8% worst case) versus weight-only analysis - guessing from
  weights alone misranks.
- **AWQ channel rescaling** (`--awq`, `--awq-search`): the search lands on
  a global alpha of 0.25, which recovers 88% of the Japanese CER gap
  between 4-bit and Q8 on short clips.
- **Opt-in 4-bit weights** (`--weights q4`, `--pack-q4`) exist, and the
  documentation is blunt about why they are not the default: on 25 minutes
  of real long-form Japanese, 4-bit+AWQ *collapses into repetition loops*
  (52% of output as repeated 12-grams; 1.74x slower and wrong), and once
  segment batching is on, 4-bit buys no speed anyway - batched decode is
  attention-bound, so halving weight bytes has nothing left to save. Short
  clean clips hide all of this, which is exactly why the long-form
  measurement is the one that decides.

## 7. Japanese evaluation suite

`samples/ja_say_eval/`: a synthetic Japanese ASR test set built from macOS
`say` voices (three speakers × normal/fast/slow across daily, technical,
figures, names, schedule and context categories), with references,
a manifest and a CER scorer. It exists because the English suite alone
could not see the quality effects that decided the f16-KV and 4-bit
questions above.

## 8. Deployment

- **GitHub Pages playground** (`.github/workflows/pages.yml`): CI builds
  the wasm and deploys the demo. Pages cannot send COOP/COEP headers, so
  the demo ships a from-scratch `coi-serviceworker.js` that stamps them on
  (one automatic reload on first visit).
- **Model on Hugging Face**: 2.18 GB exceeds GitHub's 2 GB release-asset
  limit; the HF CDN answers CORS with the requesting origin echoed and
  serves range requests, which is exactly what a cross-origin-isolated page
  needs. `playground-config.js` points the demo at the model repo; the
  tracked copy is a no-op so local serving is unchanged.
- First visit downloads the model once and caches it in OPFS; subsequent
  visits are ready in a few seconds.

## Measured end-to-end numbers

| Path | Workload | Result |
|---|---|---|
| Native, M1 Pro, `-S 30 --batch 4` | 25 min real Japanese speech | ~93 s, **~16x realtime** |
| Native, M1 Pro | 41 s clip, one context | ~6 s (encoder 0.6 s, prefill 1.5 s, generation 4.1 s at 24 ms/tok) |
| Browser WebGPU, throttled tab | 11 s English clip | ~1.5-2.1 s (**5-7x realtime**) |
| Browser WebGPU, throttled tab | 41 s Japanese clip | ~4.8-5.2 s (**~8x realtime**) |
| Browser WebGPU streaming, 1 s chunks | 41 s Japanese stream | 1.19x realtime, zero fallbacks |
| Browser wasm-only (no WebGPU), Node 8 threads | 575 s English suite | 22/22, ~3.5x realtime |
| Browser memory (GPU backend) | steady state | ~2.3 GB total; wasm heap 0.5 MB |

## Provenance and license

Upstream engine: [antirez/qwen-asr](https://github.com/antirez/qwen-asr),
MIT. Model weights: [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B),
Apache 2.0; the packed Q8 derivative is published at
[ainoya/qwen3-asr-1.7b-q8-packed](https://huggingface.co/ainoya/qwen3-asr-1.7b-q8-packed).
All fork changes are MIT like the upstream code.

For the full engineering log - including everything that was measured and
*rejected*, which is half the value - read `AGENT.md` in the repository
root and the commit messages on this branch: each records what was tried,
what it measured, and why it stayed or died.
