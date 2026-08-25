This file is the practical guide for agents working on this repository.
It is intentionally implementation-oriented: what to change, where, how to test,
and which behaviors are considered contractually stable.

## Project Scope

Pure C inference engine for Qwen3-ASR speech-to-text models:
- `Qwen3-ASR-0.6B`
- `Qwen3-ASR-1.7B`

Primary target is CPU inference (BLAS + architecture-specific SIMD paths).

## Source Of Truth

When docs and code disagree, trust these files first:
- CLI behavior and options: `main.c`
- Public API and runtime state: `qwen_asr.h`
- Offline + segmented + streaming orchestration: `qwen_asr.c`
- Encoder math + load path: `qwen_asr_encoder.c`
- Decoder math + KV cache path: `qwen_asr_decoder.c`
- Kernel dispatch and hot loops: `qwen_asr_kernels*.c`, `qwen_asr_kernels_impl.h`
- Test harness: `asr_regression.py`
- Build targets: `Makefile`

Architecture/background references:
- `MODEL.md`
- `MODEL_CARD_OFFICIAL.md`

## Build Targets

- `make blas` — Accelerate (macOS) / OpenBLAS (Linux). The fast native path.
- `make noblas` — portable kernels only; this is the configuration the wasm
  build compiles, so it is the one to check when touching generic kernels.
- `./wasm/build.sh` — WebAssembly (SIMD128 + pthreads) for the browser demo.
  Requires emsdk on PATH. See `wasm/README.md`.

## Supported Runtime Modes

- Offline full-context (default): `-S 0`
- Offline segmented: `-S <secs>`
- Streaming: `--stream`
- Input from file: `-i file.wav`
- Input from stdin: `--stdin` (WAV or raw s16le 16k mono)
- Browser: `wasm/demo/` (batch + streaming), via `qwen_load_memory()`

## Model Files

Two accepted layouts:

1. Original HuggingFace bf16 safetensors (`model.safetensors` or shards). The
   decoder quantizes at load.
2. A pre-quantized image, `qwen-asr-q8.bin` in the model directory, written by
   `--pack-q8`. If present it is used *instead of* the originals, for both the
   native and wasm builds: weights are attached in place (~15 ms), nothing is
   converted, and it is 2.18 GB instead of 4.70 GB. Written and read by
   `qwen_asr_pack.c` / `attach_q8()` in the decoder and `load_wmat()` in the
   encoder; quantized matrices appear as `NAME.q8` (I8) plus `NAME.q8s` (F32
   scales), and the SwiGLU gate/up pair is pre-fused into `mlp.gate_up.weight`.

   The packed image also quantizes the *encoder*, which costs ~100 ms natively
   (the panel-dequantize path is slower than plain f32 sgemm) but takes encoder
   weights from 1.21 GB to 0.32 GB, which is what makes the browser build fit.

## User-Facing Behavior Contract (Do Not Break)

- `--silent` must still print transcription to stdout.
- `--silent` suppresses status/debug noise (stderr), not the text output.
- Without `--debug`, stderr should be concise:
  - model loading info
  - final inference summary lines
- `--debug` enables verbose internal diagnostics.
- `--language` is the only language forcing flag (no `--force-language`).
- `--past-text` accepted values are exactly `yes|no|auto`.
- `--past-text auto` means:
  - `yes` for `--stream`
  - `no` for non-stream modes
- `--weights` accepted values are exactly `q8|q8-lm|bf16` (default `q8`),
  overridable by `QWEN_WEIGHTS=q8|q8-lm|bf16`.
- `q8` quantizes the 28 transformer layers only; the tied embedding / LM head
  stays bf16. `q8-lm` quantizes the LM head too.
- `--weights q8` must not change transcripts: the regression references were
  generated on the bf16 path and both must stay within threshold on both models.

## Model + Inference Facts

- Model variant is auto-detected from weights (0.6B vs 1.7B).
- Encoder uses per-chunk Conv2D + windowed attention.
- Decoder uses causal Qwen3 with KV cache and prefill reuse.
- Encoder weights are loaded as f32 (converted at load where needed).
- Decoder large weights are bf16 mmapped. By default the 28 transformer layers
  are block-quantized to Q8 at load (`QWEN_Q8_BLOCK` = 64 int8 + one f32 scale
  = 1.0625 B/weight) and consumed via integer-dot kernels; `--weights bf16`
  keeps the bf16 path, which is still fully maintained.
- The tied embedding / LM head is deliberately NOT quantized by default. It is
  only ~18% of the bytes read per token but it decides the token: on the 0.6B
  model Q8 there flips near-tied logits and drops whole phrases (normalized
  error over the sample set goes 3.1% -> 5.0%). On 1.7B it is output-identical,
  which is what `--weights q8-lm` is for. Measured: quantizing the *activation*
  is not the cause — an f32-activation argmax variant changed nothing and was
  ~30% slower, so it was removed.
- Token generation is memory-bandwidth bound, not compute bound. Measured on an
  M1 Pro the Q8 matvec runs at 86.6 GB/s against a 94.7 GB/s pure-read ceiling,
  and decode time is flat from 5 to 8 threads. Do not expect gains from more
  arithmetic-side tuning of the decode matvecs; only fewer bytes per weight
  helps.

## Important Defaults

From `qwen_load()` and CLI:
- Threads: performance-core count (`hw.perflevel0.logicalcpu` on Apple silicon,
  else `_SC_NPROCESSORS_ONLN`). Adding efficiency cores is measurably slower
  because every parallel dispatch is barrier-synchronised.
- Decoder weights: `--weights q8`
- Segment mode default: `-S 0` (full-audio decode)
- Segment batch size: `--batch 4` (only used when `-S > 0`, past-text off, weights quantized)
- Segment batch size: `--batch 4` (only used when `-S > 0`, past-text off, weights quantized)
- Segment cut search window: `-W 3.0`
- Stream chunk: `2.0s`
- Stream rollback: `5` tokens
- Stream unfixed chunks: `2`
- Stream max new tokens/chunk: `32`
- Encoder infer attention window: `8s` (`--enc-window-sec` in `[1,8]`)

## Repository Map

- `main.c`
  - CLI parsing, defaults, reporting, callback wiring
- `qwen_asr.c`
  - high-level transcription flows
  - segmented logic + optional past-text cleanup path
  - streaming chunk loop, encoder-window cache, rollback commit logic
- `qwen_asr_encoder.c`
  - audio tower load + forward
- `qwen_asr_decoder.c`
  - decoder load + prefill + token step + KV cache
- `qwen_asr_audio.c`
  - WAV/stdin decoding, resampling, mel prep helpers
- `qwen_asr_tokenizer.c`
  - tokenizer encode/decode
- `qwen_asr_safetensors.c`
  - safetensors loading and mmap
- `qwen_asr_kernels.c`
  - common math, threading, BLAS paths
  - thread pool: hybrid spin/park barrier plus `qwen_claim_chunk()`
    work-stealing; workers are `QOS_CLASS_USER_INTERACTIVE` on Apple
  - Q8 quantization, threaded Q8 matvec/argmax, and the panel-dequantize
    prefill path
  - portable blocked GEMM (`qwen_gemm_t_generic` / `qwen_gemm_nn_generic`)
    used when `USE_BLAS` is not defined (wasm/browser targets)
- `qwen_asr_kernels_generic.c`
  - generic hot kernels
- `qwen_asr_kernels_neon.c`
  - ARM NEON hot kernels
- `qwen_asr_kernels_avx.c`
  - x86 AVX hot kernels
- `qwen_asr_kernels_impl.h`
  - architecture dispatch macros
- `asr_regression.py`
  - quality + focused regression checks
- `qwen_asr_pack.c`
  - writes the pre-quantized `qwen-asr-q8.bin` image (`--pack-q8`)
- `qwen_asr_kernels_wasm.c`
  - WebAssembly SIMD128 hot kernels (Q8 matvec/argmax/quantize, f32 helpers)
- `wasm/`
  - `qwen_wasm.c` browser entry points, `build.sh`, `serve.py`,
    node harnesses (`bench-node.js`, `check-node.js`, `stream-node.js`),
    and `demo/` (batch + streaming demo, `webgpu-decoder.js`,
    `webgpu-test.html`, `webgpu-probe.html`)
- `download_model.sh`
  - interactive small/large model downloader

## Build + Run

Build:
```bash
make blas     # Accelerate / OpenBLAS
make noblas   # portable kernels only (what a wasm/browser build compiles)
```

Smoke run:
```bash
./qwen_asr -d qwen3-asr-0.6b -i samples/jfk.wav
```

Stdin path:
```bash
cat samples/jfk.wav | ./qwen_asr -d qwen3-asr-0.6b --stdin
```

## wasm Regression Workflow

Browser timings are useless for benchmarking (hidden/unfocused tabs are
throttled hard, and embedded webviews cap the renderer at ~1.3 cores). Run the
same `qwen_asr.wasm` under Node instead:

```bash
node wasm/bench-node.js  qwen3-asr-1.7b-q8 samples/jfk.wav 8
node wasm/check-node.js  qwen3-asr-1.7b-q8 samples 8        # 22/22, norm error 0.0076
node wasm/stream-node.js qwen3-asr-1.7b-q8 samples/jfk.wav 8
```

`check-node.js` must stay at 22/22. It is the only thing that catches wasm-only
kernel bugs — an inverted `wasm_v128_bitselect` in the activation quantizer got
through everything else and made one sample emit nothing at all.

## Regression Workflow

Primary suite:
```bash
make test
# equivalent to:
./asr_regression.py --binary ./qwen_asr --model-dir qwen3-asr-1.7b
```

Focused checks:
```bash
./asr_regression.py --segment-check-only --binary ./qwen_asr --model-dir qwen3-asr-1.7b
./asr_regression.py --stream-check-only --binary ./qwen_asr --model-dir qwen3-asr-1.7b
./asr_regression.py --stream-cache-check-only --binary ./qwen_asr --stream-cache-model-dir qwen3-asr-0.6b
```

Notes:
- Quality regression only runs on WAVs that already have sibling `.txt` refs.
- `make test` includes stream-cache equivalence check by default.
- This means both model dirs are typically required:
  - main model (`--model-dir`, default `qwen3-asr-1.7b`)
  - stream-cache model (`--stream-cache-model-dir`, default `qwen3-asr-0.6b`)

Reference management:
```bash
./asr_regression.py --generate-missing --binary ./qwen_asr --model-dir qwen3-asr-1.7b
./asr_regression.py --refresh-refs --binary ./qwen_asr --model-dir qwen3-asr-1.7b
```

## Streaming Implementation Notes

Current streaming behavior in `qwen_transcribe_stream()`:
- Chunk-by-chunk audio growth (default 2s)
- Encoder cache for completed local-attention windows
- Re-encode only current partial tail window
- Decoder prefill reuse by longest unchanged embedding prefix
- Prefix rollback policy for token stability
- Monotonic commit frontier (no retracting already-emitted text)

Debug/env switch:
- `QWEN_STREAM_NO_ENC_CACHE=1` disables encoder window cache (debug/regression only)

Important caveat:
- In streaming mode, if no token callback is installed (for example CLI `--silent`),
  the code uses direct final refinement instead of interactive chunk emission.
  This path is not representative of interactive stream throughput.

## Segmented Mode Notes

When `-S > 0`:
- split points are chosen near low-energy regions inside `-W`
- default emission is token-by-token ASAP

When `--past-text no` (the default) and weights are quantized, segments are
decoded `--batch` at a time through `decode_segment_group()` in `qwen_asr.c`:
one `qwen_decoder_prefill_multi()` for the whole group, then lockstep
`qwen_decoder_forward_batch()` steps. Each stream owns a `qwen_kv_t`. Output is
emitted a segment at a time rather than a token at a time; `--batch 1` restores
the old per-segment path.

When `--past-text no` (the default) and weights are quantized, segments are
decoded `--batch` at a time through `decode_segment_group()` in `qwen_asr.c`:
one `qwen_decoder_prefill_multi()` for the whole group, then lockstep
`qwen_decoder_forward_batch()` steps. Each stream owns a `qwen_kv_t`. Output is
emitted a segment at a time rather than a token at a time; `--batch 1` restores
the old per-segment path.

When `--past-text yes` in segmented mode:
- boundary cleanup/post-processing path is enabled
- output is buffered per segment before emission
- collapse guardrails can retry segments unconditioned and disable conditioning after repeated collapses

## Performance Reporting Contract

Final stderr summary line format is:
```text
Inference: <ms> ms, <tokens> text tokens (<tok/s> tok/s, encoding: <ms>ms, decoding: <ms>ms)
Audio: <audio_s> s processed in <infer_s> s (<x>x realtime)
```

`encoding` = mel + encoder time
`decoding` = decoder prefill + autoregressive decode

## Kernel/Optimization Rules

- Architecture dispatch is centralized in `qwen_asr_kernels_impl.h`.
- Keep generic/NEON/AVX variants functionally equivalent.
- If you optimize one path, verify no regression on others.
- Favor meaningful speedups; avoid complexity for tiny wins.
- The generic kernels are the wasm/browser path. Keep their inner loops
  auto-vectorizable (no libm calls, no aliasing ambiguity) rather than adding
  target-specific intrinsics: the portable Q8 matvec is within 1.22x of the
  hand-written NEON one because clang vectorizes it.
- Avoid scalar libm in hot loops. `qwen_vec_expf()` is vForce on Apple and an
  auto-vectorizable polynomial elsewhere; GELU/SiLU/softmax all go through it.
- Verify both backends after kernel work: `make blas` and `make noblas`.
  Transcripts must stay identical between them.

## Measured Hot Spots (Apple M1 Pro, 1.7B, 41s audio)

Useful when deciding where effort pays off:

| Phase | Time | Status |
|-------|------|--------|
| encoder | ~0.63 s | mostly Accelerate `sgemm` |
| decoder prefill | ~1.48 s | `sgemm` at ~1.4 TFLOPS, near the AMX ceiling |
| token generation | ~4.1 s (24 ms/tok) | at the DRAM read bandwidth wall |

Known remaining opportunities, in rough value order:

1. **f16 KV cache.** The cache is f32 today. At a 1500-token context it is
   ~15% of the bytes read per generated token, and it dominates RAM for very
   long files. Halving it is worth ~7% of decode there. The catch is that the
   prefill attention path hands K/V straight to `cblas_sgemm`, so that path
   would need an f32 scratch conversion; only the seq_q==1 path benefits
   directly.
2. **Opt-in 4-bit weights.** Would roughly halve both the browser download and
   the decode bandwidth, at a measured quality cost (see below). Should never
   become the default.
3. **Encoder conv stem.** `im2col` still allocates per chunk and runs
   single-threaded; the conv2 GEMM is the largest single conv cost.
4. **WebGPU encoder.** The decoder (prefill and generation) already runs on the
   GPU; mel and the audio encoder are still wasm and are now the largest single
   piece in the browser (3.9 s of an 11.4 s run on the 41s clip). It cannot
   reuse the decoder shaders: LayerNorm has a bias, the FFN is GELU rather than
   SwiGLU, attention is windowed and bidirectional, and there is a conv2d stem.

## WebGPU Notes

`wasm/demo/webgpu-decoder.js` runs the whole decoder on the GPU — prefill and
generation. wasm supplies the input embeddings (`qwen_wasm_embeds_*`, built by
`qwen_build_embeds()` which is the mel + encoder + prompt assembly split out of
`transcribe_segment`) and the weight tables (`qwen_wasm_q8_desc`,
`qwen_wasm_f32_desc`, `qwen_wasm_model_shape`). The CPU-prefill handoff
(`qwen_wasm_prefill_*`, `qwen_wasm_kv_*`) is still there and still used by
`webgpu-test.html` to isolate the generation kernels.

Prefill activations are stored **transposed**, `[dim][seqPad]`, so every kernel's
lanes run along the sequence axis: all access is coalesced and row-wise
reductions become per-lane loops. The prefill GEMM is a 64x64 tile per
workgroup, 16x16 threads each owning 4x4, with both operand tiles in workgroup
memory — tiling in the row dimension is what stops the weights being re-read
once per sequence position.

Things that were learned the hard way and should not be re-litigated:

- **Byte-level BPE breaks per-token string decoding.** A token is often part of a
  UTF-8 sequence; a CJK character spans two or three. `UTF8ToString` per token
  produces replacement characters that cannot be recombined — English matched
  byte for byte while Japanese glyphs corrupted, which reads exactly like a
  numerical bug. Use a streaming `TextDecoder`.
- **`dot4I8Packed` is emulated on Apple GPUs** (24 GMAC/s measured). Unpack with
  `unpack4x8snorm` and use f32 FMA.
- **RoPE cos/sin must be precomputed on the CPU.** WGSL only guarantees accuracy
  for small arguments and the angles grow with position.
- **Keep the next token id in a GPU buffer** so the embedding lookup does not
  need a readback between steps.
- **Lane-per-word beats lane-per-block** in the Q8 matvec (26 vs 40 ms/token).
  Re-reading the block scale per word is cheap because it stays in cache;
  giving each lane its own byte run is not.
- **Split the attention V-sum over the key axis.** One thread per (head, dim) is
  only 2048 threads and dominated the step at long contexts (42 vs 26.5
  ms/token at a 1170-token context).
- **WebGPU validation errors do not throw.** A missing `COPY_DST` made
  `writeBuffer` a silent no-op and the prefill ran on zeros; an invalid pipeline
  (a shader helper that was not actually inserted) made every dispatch a no-op.
  Both looked like "impossibly fast and wrong". `getCompilationInfo()` is checked
  at init and `onuncapturederror` is routed to the page — keep it that way.
- **Uniform params must be uploaded in `buildParams()`**, not only in the
  per-step update: prefill reads them before any generation step runs.
- Verify with `wasm/demo/webgpu-test.html`, which runs the wasm decoder, the
  GPU-generation-only path and the full GPU decoder on the same audio and diffs
  the transcripts. All must report IDENTICAL.

## Threading Notes

The pool is a hybrid spin/park barrier with work stealing
(`qwen_claim_chunk()`), and both halves matter:

- A decoder token issues ~110 tiny parallel dispatches. Natively a futex round
  trip is tens of microseconds; between Web Workers it measured ~4 ms. Parking
  between dispatches took wasm decode from 88 ms/token to 990. Hence
  `QWEN_WORKER_SPINS` / `QWEN_JOIN_SPINS` of ~1M iterations.
- Budgets are iteration counts on purpose. A wall-clock spin was tried and was
  *worse* under wasm: reading a clock from a worker goes through JS.
- `qwen_set_threads()` counts threads that actually started and lowers
  `n_threads` if the platform refused some — a barrier waiting on a thread that
  was never created hangs forever.
- `qwen_pool_selftest()` reports participants and per-dispatch cost. Use it
  first when a new platform looks "slow": a barrier silently running
  single-threaded looks exactly like a slow kernel.
- Single-threaded paths must still go through `parallel_for()`, which resets the
  work-stealing cursor. Calling a worker function directly leaves a stale cursor
  and the worker silently processes nothing (this was a real SIGSEGV, caught by
  `--stream-cache-check` which runs at `threads=1`).

Research note: published evaluations of llama.cpp quantization on ASR models
report Q8_0 matching FP16 word error rate on a 1.7B model, while Q4_K raises
character error rate by ~8.6% relative. That is why Q8 is the default and why a
4-bit mode, if it is ever added, should stay opt-in.

## Change Checklist For Agents

Before editing:
1. Identify behavioral contract impacted (CLI, output, speed, quality, memory).
2. Read corresponding source-of-truth file(s).

After editing:
1. Build: `make blas`
2. Run focused sanity command(s) for changed area.
3. Run regression:
   - at minimum relevant focused checks
   - ideally full `make test` for non-trivial changes
   - for kernel changes also `node wasm/check-node.js` (rebuild wasm first)
4. Update `README.md` (and `wasm/README.md` for browser-facing changes) if
   CLI/runtime behavior changed.
5. Keep `AGENT.md` aligned if workflow/test defaults changed.

## Local-Only Artifacts (Do Not Depend On In Commits)

Common local directories/files are intentionally ignored:
- `qwen3-asr-0.6b/`, `qwen3-asr-1.7b/`, `qwen3-asr-1.7b-q8/`
- `wasm/demo/qwen_asr.js`, `wasm/demo/qwen_asr.wasm` (build output)
- `Qwen3-ASR/`
- `samples/extra/`
- `TODO.md`
- virtualenv folders

Do not make code rely on these being present unless guarded by checks.
