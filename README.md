# Qwen3-ASR — pure C, WebAssembly, WebGPU

Speech-to-text with the Qwen3-ASR models (0.6B / 1.7B) in a single
dependency-light C engine — and, in this fork, the same engine running
entirely in the browser, with the whole model on the GPU via WebGPU.

**► Try it now: [ainoya.github.io/qwen-asr](https://ainoya.github.io/qwen-asr/)** —
batch, streaming and microphone transcription in a WebGPU-capable browser.
One ~2.2 GB model download on first visit, cached locally (OPFS); later
visits are ready in a few seconds.

**Model:** [ainoya/qwen3-asr-1.7b-q8-packed](https://huggingface.co/ainoya/qwen3-asr-1.7b-q8-packed)
on Hugging Face — a pre-quantized int8 image (2.18 GB instead of 4.70 GB of
bf16) that both the native and browser builds attach directly.

This is a fork of **[antirez/qwen-asr](https://github.com/antirez/qwen-asr)**
by Salvatore Sanfilippo — read the
[original README](https://github.com/antirez/qwen-asr#readme) for the
upstream project's own write-up. The engine's design and behavior contracts
are upstream's; this fork extends it with native performance work, the
WebAssembly port, the WebGPU backend, and the deployment around them.

## What this fork adds

Full details with measurements: **[docs/EXTENSIONS.md](docs/EXTENSIONS.md)**.
The short version:

- **~1.9x faster native inference** (Apple Silicon measured): a packed Q8
  model image that attaches by mmap in ~15 ms, batched segment decoding
  that sweeps the weights once for several streams (generation 2.0x), an
  f16 KV cache, and prefill/conv paths tuned until they sit at measured
  hardware ceilings (AMX load-issue bound, DRAM bandwidth).
- **A WebAssembly port** — SIMD128 + pthreads, same engine, same
  transcripts; Node harnesses for benchmarking and a 22/22 regression
  suite.
- **A WebGPU backend** running the *entire* model on the GPU: audio tower,
  decoder prefill and generation, with subgroup reductions, batched
  generation steps, suffix prefill for streaming, and SwiGLU fused into
  the gate/up matmuls. Verified against 23 golden transcripts —
  18/23 byte-identical to the CPU decoder.
- **GPU-resident weights**: the wasm heap keeps **0.5 MB** of the model
  instead of 2.18 GB; browser memory is ~2.3 GB total instead of ~4.4 GB.
- **Quantization research tooling**: activation calibration, AWQ channel
  rescaling, and an opt-in 4-bit mode whose documentation explains, with
  long-form measurements, why it is *not* the default.
- **Deployment**: the GitHub Pages playground (cross-origin isolation via a
  service-worker shim), the Hugging Face model repo, and a synthetic
  Japanese evaluation suite.

## Measured performance

| Path | Workload | Result |
|---|---|---|
| Native (M1 Pro, `-S 30 --batch 4`) | 25 min of real speech | **~16x realtime** |
| Browser WebGPU (throttled tab) | 41 s Japanese clip | **~8x realtime** |
| Browser WebGPU streaming (1 s chunks) | 41 s stream | 1.19x realtime |
| Browser wasm-only (no WebGPU) | English suite in Node | 22/22 at ~3.5x realtime |

## Quick start (native)

```bash
make blas                      # Accelerate (macOS) / OpenBLAS (Linux)
./download_model.sh            # interactive: small=0.6B, large=1.7B
./qwen_asr -d qwen3-asr-1.7b -i samples/jfk.wav
```

Streaming, stdin/ffmpeg piping, segmented long-form decoding, weight
precision options and the C API are documented in the full manual:
**[docs/MANUAL.md](docs/MANUAL.md)**.

## Quick start (browser, local)

```bash
./wasm/build.sh                # needs emsdk on PATH
python3 wasm/serve.py --port 8765
# open http://localhost:8765/wasm/demo/
```

See [wasm/README.md](wasm/README.md) for the browser architecture, the
WebGPU verification pages, and the Node harnesses.

## Documentation

| Document | Contents |
|---|---|
| [docs/MANUAL.md](docs/MANUAL.md) | Full usage manual: every CLI option, modes, benchmarks, memory, C API |
| [docs/EXTENSIONS.md](docs/EXTENSIONS.md) | What this fork adds, with the measurements behind each decision |
| [wasm/README.md](wasm/README.md) | Browser/wasm/WebGPU architecture and workflows |
| [AGENT.md](AGENT.md) | Engineering log: everything tried, measured, kept or rejected |
| [Original README](https://github.com/antirez/qwen-asr#readme) | The upstream project's own write-up |

## License

MIT (© Salvatore Sanfilippo and contributors), same as upstream. Model
weights are Apache 2.0 from
[Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B).
