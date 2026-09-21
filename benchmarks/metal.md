# Native Apple Silicon Metal/MPS prefill

`make metal` builds a hybrid backend for large Q8 matrix multiplications.
On the measured M1 Pro it reduced warm prefill time by 28–34%, and whole
inference time by 10–17% on the three clips below. This does not move the
entire model onto the GPU: single-token generation and short sequences
continue to use the existing CPU kernels.

## Implementation

- Metal reads the existing Q8 quants/scales through shared, page-aligned
  buffer views. No second copy of the quantized model is uploaded.
- A Metal kernel dequantizes one matrix into reusable float32 scratch;
  MPS multiplies it by the float32 activations. Weights, activations,
  accumulation, and results stay float32 in this path.
- The CPU panel path dispatches to Metal only for at least 256 activation
  rows and at least 1,048,576 weights. The existing short-sequence
  quantized-activation path keeps its arithmetic and crossover settings.
- BF16 and Q4 matrices stay on the CPU. Both models and both original
  model files and packed Q8 images use the same kernel dispatch.
- Buffers grow as needed and are released when the last cached matrix is
  freed. GPU views are released before the owning weight memory is freed.
  Scratch holds one dequantized matrix, not a float32 copy of the model.
- `QWEN_METAL=0` disables GPU dispatch. An unavailable device, allocation
  failure, or failed command falls back to CPU computation. Like the CPU
  thread pool and scratch, this backend expects serialized inference calls.

For this hybrid path, small GPU matvecs submitted individually and custom
float32 prefill tiles did not beat the CPU/MPS alternatives. The subsequent
[resident Metal experiment](metal-full.md) batches all decoder layers into
one submission per token and measures that different tradeoff. Neither path
changes quantization or uses half-precision GEMM operands.

## Measurements

Measured 2026-09-20, Apple M1 Pro, 32 GiB RAM, macOS 26.6.2, 8 CPU threads,
Qwen3-ASR-1.7B packed Q8, full-audio mode. One model load per clip, one
warmup pair, then three measured CPU/Metal pairs with order reversed every
pair. Values are medians. No other inference tests ran concurrently.
Raw runs, including warmups, are in
[metal-m1-pro-2026-09-20.json](metal-m1-pro-2026-09-20.json).

| Audio | CPU total | Metal total | Total reduction | CPU prefill | Metal prefill | Prefill reduction |
|---|---:|---:|---:|---:|---:|---:|
| 45 s English, `45s_right_through_the_billboard.wav` | 4.061 s | 3.654 s | 10.0% | 1.417 s | 1.013 s | 28.5% |
| 119 s English, `119s_theres_supposed_to_be_another_broadcast.wav` | 13.193 s | 10.940 s | 17.1% | 3.784 s | 2.727 s | 27.9% |
| 41 s Japanese, local clip | 5.860 s | 5.004 s | 14.6% | 1.369 s | 0.910 s | 33.5% |

All four Metal runs per clip, including warmup, produced byte-identical
transcripts to that clip's CPU reference. This is a result for these clips,
not a guarantee of bitwise equivalence for every input/device. The Japanese
clip is local and is not distributed with this benchmark.

**Cold start matters.** The first GPU use creates buffers and prepares Metal
and MPS pipelines. On the 119 s clip the first Metal pass took 12.569 s
versus the first CPU pass's 11.863 s. Warm figures do not imply a faster
first CLI invocation. Clips with short prefills usually stay on the CPU.
Timing varies with GPU/CPU load, temperature, and power state; measure the
intended workload instead of extrapolating these percentages to other Macs.

## Reproduce

```bash
make metal
./tools/test-metal
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./tools/test-metal

# CSV on stdout; stage timings (including prefill) on stderr.
# pair 0 is warmup. The optional fourth argument selects segment seconds.
./tools/bench-metal qwen3-asr-1.7b-q8 \
  samples/night_of_the_living_dead_1968/45s_right_through_the_billboard.wav \
  > /tmp/metal-45.csv 2> /tmp/metal-45.log
./tools/bench-metal qwen3-asr-1.7b-q8 \
  samples/night_of_the_living_dead_1968/119s_theres_supposed_to_be_another_broadcast.wav \
  > /tmp/metal-119.csv 2> /tmp/metal-119.log

# Include first-use costs when comparing independent CLI invocations.
QWEN_METAL=0 ./qwen_asr -d qwen3-asr-1.7b-q8 -i recording.wav --debug
./qwen_asr -d qwen3-asr-1.7b-q8 -i recording.wav --debug
```

The benchmark fails if Metal is unavailable or a transcript differs; it
does not mistake CPU fallback for a successful GPU measurement. Model/audio
loading is outside the inference timers. Run timing measurements with the
Metal validation layers disabled. Tests under a sandbox without GPU access
must be rerun with actual device access to validate Metal execution.

## Validation

`tools/test-metal` compares against the CPU float32 panel path, using a
relative L2 tolerance of `2e-5`. It covers irregular row counts, multiple
block widths, 1.7B projection shapes, explicit disable, Q4/invalid-input
fallback, and free/reload of cached weight views. Metal API and shader
validation passed with no failures; measured relative errors were at most
`1.13e-7` for the tiny case and zero for the tested model-sized cases.

The 22 public English regressions passed, together with segmented conditioning,
stdin streaming, and both 0.6B stream-cache equivalence cases. All 18 Japanese
synthetic fixtures matched the CPU transcripts exactly (normalized CER
0.163898 for both backends). These short fixtures mostly exercise the CPU
fallback; the 41 s Japanese benchmark above exercises Metal.

Clean `make blas` and `make noblas` builds passed and produced identical JFK
transcripts. A final `make metal` restored the opt-in binary. Running that
binary without GPU access produced the same 45 s transcript as the CPU and
zero stderr bytes with `--silent`.
