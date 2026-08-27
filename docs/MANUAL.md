# Qwen3-ASR Pure C Implementation


This is a C implementation of the inference pipeline for [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) speech-to-text models (both 0.6B and 1.7B). It has zero external dependencies beyond the C standard library and a BLAS implementation (Accelerate on macOS, OpenBLAS on Linux). Tokens stream to stdout as they are generated. The implementation runs at speed multiple of the file length even in very modest hardware, like low end Intel or AMD processor.

**Important**: this implementation explicitly **avoids implementing support for MPS**. Transcription systems are very important pieces of infrastructure, and are often run on remote Linux servers. Adding the MPS target would focus the efforts too much on Apple hardware, so for now I'm skipping it. The code runs very well anyway on Apple hardware (NEON optimized). Please, **don't send pull requests** about this feature, fork the code instead, in order to add MPS support. I'll add it much later when the other optimizations are already mature.

## Supported modes and models

Both normal (offline) and streaming (online) modes are supported. Normal mode defaults to full offline decode (`-S 0`), so the whole audio is encoded at once. Streaming mode processes audio in 2-second chunks with prefix rollback (it keeps the last few decoded tokens as context for the decoder/LLM when transcribing the next chunk).

*Important practical note*: in this implementation, interactive `--stream` prioritizes incremental token stability over throughput and can be much slower than normal mode when you process an already-recorded file end-to-end.

Audio can be piped from stdin (`--stdin`), making it easy to transcode and transcribe any format via ffmpeg. Language is usually auto-detected from audio, and can be forced with `--language`. A system prompt can bias the model toward specific terms or spellings.

Both the 0.6B and 1.7B parameters models are supported. While the 1.7B model is generally more powerful, the 0.6B model seems the sweet spot for CPU inference, however the speed difference is not huge, so you may want to try both and decide what to use depending on your use case.

## Quick Start

```bash
# Build
make blas

# Download a model (interactive selector: small=0.6B, large=1.7B)
./download_model.sh

# Transcribe audio (tokens stream to stdout as generated)
./qwen_asr -d qwen3-asr-0.6b -i audio.wav

# Pipe any format via ffmpeg
ffmpeg -i audio.mp3 -f s16le -ar 16000 -ac 1 - 2>/dev/null | \
    ./qwen_asr -d qwen3-asr-0.6b --stdin

# Browser demo (WebAssembly) - see wasm/README.md
./wasm/build.sh && ./qwen_asr -d qwen3-asr-1.7b --pack-q8 qwen3-asr-1.7b-q8/qwen-asr-q8.bin
./wasm/serve.py     # then open http://localhost:8765/wasm/demo/

# Streaming mode (incremental output for live audio)
./qwen_asr -d qwen3-asr-0.6b -i long_recording.wav --stream
```

## Features

- **Almost zero dependencies**: Pure C implementation. Only needs BLAS (Accelerate on macOS, OpenBLAS on Linux).
- **Both models**: Automatically detects Qwen3-ASR-0.6B or 1.7B from the weight files.
- **Streaming output**: Tokens are printed to stdout as they are generated, word by word, even in offline mode (no `--stream`).
- **Streaming mode**: `--stream` processes audio in chunks with prefix rollback. A sliding window bounds encoder and decoder context for indefinite streaming.
- **Monitor mode**: `--monitor` shows inline Unicode symbols on stderr for real-time pipeline diagnostics.
- **Language control**: `--language Italian` forces the target language (otherwise it is usually auto-detected).
- **Prompt biasing**: `--prompt` injects a system prompt to bias the model toward specific terms or spellings. Note that prompt biasing is very soft. The models may or may not care about your instructions. Usually spelling instructions are followed decently.
- **Optional silence skipping**: `--skip-silence` drops long silent spans before inference (off by default). It may use less CPU for the same file.
- **Memory-mapped weights**: BF16 weights are mmap'd directly from safetensors files — loading is near-instant.
- **Q8 quantized decoder**: the decoder's transformer weights are block-quantized to int8 at load time (`--weights q8`, the default). Token generation is memory-bandwidth bound, so cutting the bytes per weight is close to a direct speedup: ~1.8x end-to-end on an M1 Pro. Quality is unchanged on the regression suite. `--weights q8-lm` also quantizes the LM head for a bit more speed, `--weights bf16` disables quantization.
- **Runs in a browser**: `wasm/build.sh` compiles the same engine to WebAssembly with SIMD128 and pthreads; `wasm/demo/` is a batch + streaming demo that loads the 1.7B model into a tab. See [wasm/README.md](wasm/README.md).
- **Pre-quantized model images**: `--pack-q8` writes a single 2.2 GB file (from 4.7 GB of bf16) that both the native and wasm builds use in place, which makes startup instant and is what makes the browser build feasible at all. `--pack-q4 --awq` writes a 1.5 GB one with the decoder layers at four bits and channel rescaling already folded in.
- **Performance-core aware threading**: the default thread count is the number of performance cores. On an Apple M1 Pro (8P+2E), including the efficiency cores in a barrier-synchronised split makes every dispatch wait for the slowest core and is measurably slower.
- **WAV input**: Supports 16-bit PCM WAV files at any sample rate (auto-resampled to 16kHz).
- **Stdin input**: Reads from stdin with auto-detection (WAV header or raw s16le 16kHz mono).
- **Optional segment splitting**: use `-S 20` / `-S 30` for large files with segment-cutting silence search (`-W 3`).

## Usage

### Normal Mode (Default)

```bash
./qwen_asr -d qwen3-asr-0.6b -i recording.wav
```

This is the default mode, and defaults to `-S 0` (full-audio offline decode).
The model sees the entire recording in one shot, which is usually best for short/medium files.
For long files, memory/time grow with sequence length, so segmented mode (`-S 20` or similar) is often preferable.

Tokens stream to stdout as they are generated. By default, timing info is printed to stderr (`Inference: ...` and `Audio: ... (Xx realtime)`). Use `--silent` or `--debug` to control verbosity:

```bash
./qwen_asr -d qwen3-asr-0.6b -i audio.wav --silent    # no stderr output
./qwen_asr -d qwen3-asr-0.6b -i audio.wav --debug      # per-layer/per-chunk details
```

### Weight Precision (`--weights`)

```bash
./qwen_asr -d qwen3-asr-1.7b -i audio.wav                    # q8 (default)
./qwen_asr -d qwen3-asr-1.7b -i audio.wav --weights q8-lm    # also quantize the LM head
./qwen_asr -d qwen3-asr-1.7b -i audio.wav --weights q4       # 4-bit layers, opt-in
./qwen_asr -d qwen3-asr-1.7b -i audio.wav --weights bf16     # no quantization
```

Token generation reads every decoder weight from RAM once per token, so it is
memory-bandwidth bound rather than compute bound. `q8` stores each weight matrix
as blocks of 64 int8 values with one f32 scale — 1.0625 bytes per weight against
2.0 for bf16 — and the dot products run on the integer SDOT / VPDPBUSD units.

| Mode | What is quantized | 1.7B decoder bytes | Notes |
|------|-------------------|--------------------|-------|
| `bf16` | nothing | 3.44 GB | reference precision |
| `q8` (default) | the 28 transformer layers | 1.50 GB + 0.62 GB bf16 LM head | quality-neutral on both models |
| `q8-lm` | layers **and** the tied embedding / LM head | 1.83 GB | identical output on 1.7B, degrades 0.6B |
| `q4` | layers at 4 bits, LM head Q8 | 1.01 GB | 1.43x on decode, but see below |

The LM head is left in bf16 by default on purpose. It is only 18% of the bytes,
but it decides the token: on the 0.6B model, block-int8 weights there flip
near-tied logits and drop whole phrases from quiet passages (normalized error on
the sample set rises from 3.1% to 5.0%). On the 1.7B model `q8-lm` is
output-identical across the whole regression suite, so it is a reasonable choice
if you only run the large model. Quantizing the activation instead of the weight
was ruled out as the cause — an f32-activation argmax made no difference.

#### Four bits (`--weights q4`)

Opt-in, and it should stay that way. Blocks of 64 nibbles plus one f32 scale is
0.5625 bytes per weight, and since token generation is bandwidth bound that is
close to a direct cut: decode over the Japanese eval set drops from 15.1 s to
10.5 s (1.43x). Two things spoil it.

Prefill gets *slower*, 9.9 s to 11.9 s. Sequences past the batched-matvec
threshold take the panel path, which dequantizes into an f32 panel for `sgemm`,
and four bits means more unpacking for the same multiply. The net over a whole
run is only 1.10x.

Quality drops about as much as the published INT4 numbers for this model family
predict. Japanese CER goes 0.164 to 0.209; the English regression suite fails 2
of 22 samples where `q8-lm` passes all 22.

#### Channel rescaling (`--awq`)

The 4-bit loss is uneven across input channels: a weight column feeding a loud
activation channel deserves more of the quantization grid than one feeding a
quiet channel. Scaling that column up and dividing the activation back down
leaves the product unchanged but moves what the block quantizer sees.

```bash
tools/prep-calib.sh ~/recordings /tmp/calib 30 45      # 16 kHz mono clips
sox /tmp/calib/*.wav /tmp/all.wav                      # or any concatenation
./qwen_asr -d qwen3-asr-1.7b -i /tmp/all.wav -S 30 --calib-out act.qacs
./qwen_asr -d qwen3-asr-1.7b -i audio.wav --weights q4 --awq act.qacs
```

Calibration needs audio only - no transcripts - and 25 minutes is plenty. Use
recordings resembling what you will transcribe. Every division folds into a norm
weight or into the block scales of the matrix that produced the activation, so
inference itself costs nothing extra.

It recovers most of what four bits gives up. Measured on 25 minutes of real
Japanese speech, then evaluated on the Japanese set:

| Weights | CER | Decode | Whole run |
|---------|-----|--------|-----------|
| `q8-lm` | 0.164 | 15.97 s | 31.2 s |
| `q4` | 0.209 | 10.64 s | 26.5 s |
| `q4 --awq` | 0.169 | 8.70 s | 24.2 s |

`--awq-alpha` (default 0.25) sets the exponent and the optimum is sharp - 0.15
gives 0.187 and 0.40 gives 0.236, worse than not rescaling at all. `--awq-search`
reports what each value buys in weighted error, and picked the same 0.25.

**Those numbers are short clips, and they do not carry over.** The table above
is 18 clips of a few seconds each, decoded one at a time in full context. Two
things go wrong on real long-form audio.

Four bits collapses into repetition loops on some segments. On 25 minutes of
real Japanese speech at `-S 30`, two of fifty segments looped; before the loop
guard existed they ran to the token limit and turned a 142 s job into 247 s.
Rescaling does not prevent it, and neither do the two English regression samples
it fails (normalized error 0.245 and 0.437 without rescaling, 0.265 and 0.437
with). Cutting shorter avoids it — `-S 15` collapsed none of 100 segments on the
same audio — at the price of more prefill work.

And once segment batching is on, four bits stops being faster at all.
Generation is only weight-bound at batch 1, where it is a real 1.32x (42.0 s
against Q8's 55.5 s on that 25-minute file). `--batch 4` takes Q8's generation
from 55.5 s to 27.8 s on its own, and four bits then measures 28.2 s — nothing,
because batched decode is bound by per-stream attention rather than by the
weight read. Whole-run totals on that file: **Q8 at `--batch 4` is 94.0 s and
every four-bit configuration is slower.**

So `q4` is worth trying only for single-stream decode, or where the download
size matters more than the time. Check it against a long recording of your own
first.

To avoid carrying the statistics around at run time, bake them into a model
image instead:

```bash
./qwen_asr -d qwen3-asr-1.7b --pack-q4 qwen3-asr-1.7b-q4/qwen-asr-q8.bin --awq act.qacs
cp qwen3-asr-1.7b/{config.json,generation_config.json,merges.txt,vocab.json} qwen3-asr-1.7b-q4/
./qwen_asr -d qwen3-asr-1.7b-q4 -i audio.wav
```

That writes 1.47 GB against 2.18 GB for `--pack-q8`, attaches in place with no
conversion pass, and needs no flags to use - which is what the browser build
wants. The scaled weights are the same bytes `--weights q4 --awq` would build at
load time, so the transcripts match.

Quantization happens at load time from the mmap'd bf16 tensors, so no separate
model file is needed; it adds a few hundred milliseconds to startup.
`QWEN_WEIGHTS=bf16|q8|q8-lm` overrides the flag, which is handy when driving the
binary from a script or test harness.

### Runaway Segments

A greedy decoder that enters a repetition loop cannot leave it, so a segment
that starts looping keeps emitting until it runs out of budget. Two things bound
that. The per-segment token budget is tied to the audio rather than being a flat
limit — speech tops out near 3.4 text tokens per second, so 12/s plus a floor
leaves wide margin while still cutting a runaway off. And a detector watches the
generated ids for a short cycle repeating past a threshold that scales with the
cycle length (a single token repeated needs far more evidence than an eight-token
phrase), then stops and drops the repetitions from the text.

Without `--silent` a truncated segment reports itself:

```text
  Segment collapsed into a 4-token loop after 55 tokens; truncating
```

This never fires on the regression suite or on 25 minutes of real speech at Q8,
and Q8 output is byte-identical with and without it. It exists because four-bit
weights do trip it, and because a flat 2048-token limit on a 30-second segment
was a latent hazard regardless of precision.

### Which Mode To Use (By File Length)

- **Up to ~60s**: use `-S 0` (which is the default) for best quality if speed is acceptable.
- **Large prerecorded files**: use segmented offline mode, e.g. `-S 20` (or `-S 30`, or even more).
- **Long live/continuous audio or low-latency UI needs**: use `--stream`.
- **Batch/offline file transcription**: prefer `-S 20`/`-S 30`; it is usually much faster than interactive `--stream`.
- **If segmented output drops/warps around boundaries**: try a different segment size and keep default `--past-text auto`.
- **If you want stronger continuity across segments/chunks**: try `--past-text yes` (can help continuity, can also cause drift on some files).

Large-file tradeoff summary:
- `-S 20`: offline segmented decode, usually best throughput on long files, stable memory, and token-by-token output.
- `-S 20 --past-text yes`: buffered per-segment output with boundary cleanup and continuity bias.
- `--stream`: incremental output while audio arrives, lower interaction latency, but usually higher total compute for full prerecorded files.

### Streaming Mode (`--stream`)

```bash
./qwen_asr -d qwen3-asr-0.6b -i long_recording.wav --stream
```

Streaming mode processes audio in **2-second chunks** with rollback text conditioning:

1. Audio arrives chunk by chunk.
2. Encoder uses local windows (`--enc-window-sec`, default `8s`); completed windows are cached, only the current partial tail window is re-encoded.
3. Decoder prompt includes previous output minus a rollback suffix (5 tokens by default) to stabilize chunk boundaries.
4. Per chunk decode is bounded by `--stream-max-new-tokens` (default `32`).
5. Only stable text is emitted; final chunk flushes remaining text.

This keeps encoder recomputation under control. For long streams, a **sliding window** automatically bounds both encoder and decoder context so that memory and compute stay flat indefinitely:

- **Encoder window eviction**: only the most recent 4 encoder attention windows (~32 s of audio context) are kept; older windows are freed.
- **Decoder prefix capping**: only the most recent ~150 tokens of previously decoded text are fed as decoder prefix context. Internal token history is still tracked for streaming commit/dedup decisions, while only the capped tail is embedded into the decoder input.

These limits activate automatically when the stream is long enough to exceed them. For short files or live sessions under ~40 s, they have no effect.

`--stream --silent` has a special non-interactive behavior for file input: it skips chunk-by-chunk streaming and runs one direct final refinement pass. (For live stdin streaming, chunked mode is still used.)

Default stream settings:
- `chunk_size`: 2s
- `encoder_window`: 8s (`--enc-window-sec`, range `1..8`)
- `rollback`: 5 tokens
- `unfixed_chunks`: 2
- `max_new_tokens`: 32 (`--stream-max-new-tokens`)
- `past_text`: `auto` by default (effectively `yes` for `--stream`, `no` otherwise)

Streaming tuning:

```bash
# default streaming
./qwen_asr -d qwen3-asr-0.6b -i audio.wav --stream

# lower-latency encoder window (may reduce quality)
./qwen_asr -d qwen3-asr-0.6b -i audio.wav --stream --enc-window-sec 4

# allow more text generation per chunk
./qwen_asr -d qwen3-asr-0.6b -i audio.wav --stream --stream-max-new-tokens 64
```

### Provisional Text (`--partial`)

Streaming holds text back until it stops changing, which is what keeps the
output stable but also what makes it feel late: on an 11s clip the first
committed words arrive about four seconds in. The engine has a guess long
before that — it decodes the whole utterance every chunk and only releases the
part behind its commit frontier — so `--partial` shows the rest, dimmed, and
rewrites it as it firms up:

```bash
cat speech.wav | ./qwen_asr -d qwen3-asr-1.7b --stdin --stream --partial
```

What that looks like chunk by chunk, committed text in plain type and the
guess in brackets:

```text
                                                    [And so, my fellow Americans.]
                                                    [And so, my fellow Americans, ask.]
And so, my fellow Americans,                        [ ask not what your country]
And so, my fellow Americans, ask not what your country  [ can do for you.]
```

The guess is a real hypothesis, not a partial word: it can be revised or
disappear entirely, as "Americans." becomes "Americans, ask." above. Committed
text is never retracted.

The provisional text is drawn on **stderr** and erased before the next
committed piece lands, so stdout still carries only the transcription and
`--silent` still works. It needs a terminal — with stderr redirected the flag
is ignored rather than writing escape codes into a log.

Through the C API this is `qwen_set_partial_callback()`, called once per chunk
with the text after the frontier; each call replaces the previous one. The
browser demo renders it the same way, greyed after the confirmed text.

### Monitor Mode (`--monitor`)

```bash
./qwen_asr -d qwen3-asr-0.6b -i audio.wav --stream --monitor
```

Shows inline Unicode symbols on stderr alongside the transcription, useful for diagnosing streaming pipeline behavior in real time:

| Symbol | Meaning |
|--------|---------|
| `▶` | Encoder chunk processed |
| `·` | Decoder prefill completed |
| `▪` | Decode step (normal speed) |
| `▸` | Decode step (slow, >30 ms/token) |
| `⟳` | Encoder window evicted (sliding window) |

Example output (stderr + stdout interleaved):
```
▶·▪▶·▪▶·▪And so, my fellow Americans,▶·▪ ask not what your country...
```

Monitor output goes to stderr and does not affect the transcription text on stdout. It can be combined with `--debug` for full diagnostics, or used alone for a lightweight visual heartbeat.

### Segment Splitting (`-S`)

```bash
./qwen_asr -d qwen3-asr-0.6b -i long_recording.wav -S 20
```

Splits audio into segments of ~N seconds, finding segment-cutting silence boundaries within a search window (`-W`, default 3 seconds). Segments are transcribed and concatenated.

Default segmented behavior (`-S > 0`) decodes segments in batches of 4 (see
`--batch` below) and emits a segment at a time. `--batch 1` restores
token-by-token emission.

#### Batched segment decoding (`--batch`)

Without past-text conditioning, segments do not depend on each other, so
several of them can be decoded in lockstep. Generating one token reads every
decoder weight exactly once — 1.83 GB for the 1.7B model in Q8 — which
saturates memory bandwidth while leaving the arithmetic units idle, so the
second and later streams in a batch are close to free.

```bash
./qwen_asr -d qwen3-asr-1.7b -i long_recording.wav -S 30 --batch 8
```

Applies when all of these hold: `-S > 0`, past-text conditioning off (the
default), and quantized weights (`--weights q8-lm`, the default). Otherwise the
flag is ignored and segments are decoded one at a time.

Measured on M1 Pro, 1.7B Q8, a 246s Japanese recording at `-S 30`:

| `--batch` | wall | realtime | peak RSS |
|-----------|------|----------|----------|
| 1 | 32.6 s | 7.5x | 2.7 GB |
| 4 (default) | 19.5 s | 12.6x | 3.4 GB |
| 8 | 17.9 s | 13.8x | 4.0 GB |
| 12 | 16.9 s | 14.6x | 4.1 GB |

Each stream keeps its own KV cache, which is where the extra memory goes;
raise `--batch` if you have the headroom. The trade-off is emission
granularity, not accuracy: a segment's text is released once every earlier
segment in its group has finished.

When `--past-text yes` is used, segmented mode switches to buffered per-segment emission and enables boundary post-processing:
- Split points are chosen near low-energy (silence-like) regions within the `-W` window to avoid cutting in the middle of words.
- If past-text conditioning causes a segment collapse (too short for its duration) or large duplicate span, that segment is retried without conditioning.
- If collapses keep happening, past-text conditioning is disabled for the remainder of the run.
- Boundary whitespace and spacing are normalized when segments are appended.

By default (`--past-text auto`), segmented mode does **not** use past-text conditioning. This is usually more stable on long files.
If you want extra continuity bias across boundaries, enable conditioning explicitly:

```bash
./qwen_asr -d qwen3-asr-0.6b -i lecture.wav -S 20 --past-text yes
```

If repeated conditioned segment collapses are detected, conditioning is disabled automatically for the rest of the run (fail-open behavior).

The same flags apply to `--stream` mode:
- `--past-text auto` (default) enables text-prefix conditioning in streaming mode.
- `--past-text yes` forces conditioning on.
- `--past-text no` forces conditioning off.

```bash
# 20-second segments with default segment-cutting silence search window
./qwen_asr -d qwen3-asr-0.6b -i lecture.wav -S 20

# Same segmentation with past-text conditioning (auto boundary cleanup)
./qwen_asr -d qwen3-asr-0.6b -i lecture.wav -S 20 --past-text yes
```

### Silence Skipping (`--skip-silence`)

```bash
./qwen_asr -d qwen3-asr-0.6b -i recording.wav --skip-silence
```

When enabled, long silent spans are removed before transcription (short pauses are kept). This reduces compute on recordings with long dead-air sections.

Tradeoffs:
- Useful for podcasts, meetings, and captured audio with long pauses.
- Can slightly alter timing-sensitive boundary behavior and punctuation.
- Disabled by default to preserve baseline behavior.

### Language (`--language`)

```bash
./qwen_asr -d qwen3-asr-0.6b -i audio.wav --language Italian
```

Forces the model to transcribe (or translate) into the specified language by adding language tokens into the decoder prompt. If omitted, language is usually auto-detected from the audio. The 0.6b and 1.7b models can behave differently, with the smaller model being more likely to translate when forced into a different language than the source audio.

Example of this behavior:

```
$ ./qwen_asr -d qwen3-asr-0.6b -i samples/jfk.wav --language Italian --silent
E così, miei amici americani, chiedete non ciò che il vostro paese può
fare per voi, chiedete ciò che voi possiate fare per il vostro paese.
```

Supported languages: Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian.

### System Prompt (`--prompt`)

```bash
./qwen_asr -d qwen3-asr-0.6b -i audio.wav --prompt "Preserve spelling: PostgreSQL, Redis, CUDA"
```

Injects a system prompt into the model's chat template. This slightly biases the model without changing the fundamental transcription behavior. Useful for:

- **Preserving technical terms**: `--prompt "Preserve spelling: PostgreSQL, CUDA, FFmpeg"`
- **Domain context**: `--prompt "This is a medical consultation about cardiology."`
- **Style hints**: `--prompt "Use formal punctuation and capitalization."`

The prompt is encoded once and prepended to every segment/chunk. Its effect is subtle — it nudges the model's token probabilities rather than forcing specific output.

### Reading Audio from Stdin

The **`--stdin` flag** reads audio from standard input. The format is auto-detected: if the data starts with a RIFF header it is parsed as WAV, otherwise it is treated as **raw signed 16-bit little-endian, 16 kHz, mono** (`s16le`).

```bash
# Transcribe an MP3 file
ffmpeg -i podcast.mp3 -f s16le -ar 16000 -ac 1 - 2>/dev/null | \
    ./qwen_asr -d qwen3-asr-0.6b --stdin

# Pipe a WAV directly
cat recording.wav | ./qwen_asr -d qwen3-asr-0.6b --stdin

# Live transcription of a web radio stream
curl -sL http://stream.live.vc.bbcmedia.co.uk/bbc_world_service | \
    ffmpeg -i pipe:0 -ar 16000 -ac 1 -f s16le pipe:1 2>/dev/null | \
    ./qwen_asr -d qwen3-asr-0.6b --stdin --stream --monitor

# Same flow, but keep WAV framing on stdin
curl -sL http://stream.live.vc.bbcmedia.co.uk/bbc_world_service | \
    ffmpeg -i pipe:0 -ar 16000 -ac 1 -f wav pipe:1 2>/dev/null | \
    ./qwen_asr -d qwen3-asr-0.6b --stdin --stream --monitor
```

To convert files to WAV format, just use ffmpeg:

    ffmpeg -i input.ogg output.wav

There are two example WAV files under the `samples/` directory.

### C API

The library exposes a simple callback-based API:

**Offline transcription:**

```c
#include "qwen_asr.h"

qwen_ctx_t *ctx = qwen_load("qwen3-asr-0.6b");

/* Optional: set a callback to receive tokens as they are decoded */
qwen_set_token_callback(ctx, my_token_handler, userdata);

/* Optional: force language or set system prompt */
qwen_set_force_language(ctx, "Italian");
qwen_set_prompt(ctx, "Preserve spelling: PostgreSQL, Redis");

/* Transcribe — returns malloc'd string */
char *text = qwen_transcribe(ctx, "audio.wav");
printf("%s\n", text);
free(text);

/* Or from raw samples */
char *text2 = qwen_transcribe_audio(ctx, samples, n_samples);

qwen_free(ctx);
```

**Streaming transcription:**

```c
/* Load audio first */
float *samples = qwen_load_wav("long_audio.wav", &n_samples);

/* Stream-transcribe with prefix rollback */
qwen_set_token_callback(ctx, my_token_handler, userdata);
char *text = qwen_transcribe_stream(ctx, samples, n_samples);
free(text);
free(samples);
```

Tokens are emitted via the callback as they become "fixed" (past the rollback window). The returned string contains the full concatenated text.

## Regression Tests

The repository includes `asr_regression.py` (repo root), a stdlib-only regression harness.
It scans `samples/**/*.wav` recursively:
- quality regression runs on WAV files that already have a sibling `.txt` reference
- focused checks (segmented conditioning, streaming, stream-cache equivalence) use fixed targets

Generate references (using the larger model and full-context decode):

```bash
./asr_regression.py --generate-missing \
    --binary ./qwen_asr --model-dir qwen3-asr-1.7b
```

Run regression checks:

```bash
./asr_regression.py \
    --binary ./qwen_asr --model-dir qwen3-asr-1.7b
```

Or run the default regression profile via make:

```bash
make test
```

Streaming cache equivalence regression (cache on vs off):

```bash
./asr_regression.py --stream-cache-check-only \
    --binary ./qwen_asr --stream-cache-model-dir qwen3-asr-0.6b
```

Or via make:

```bash
make test-stream-cache
```

Output format:
- Each sample starts with a progress line: `START i/N`.
- Live model text is shown while that sample is transcribed.
- The sample closes with `DONE: OK i/N` (only `OK` is green) or `DONE: FAIL i/N` (status in red).

Example:

```text
[START 1/22] jfk.wav ...
And so, my fellow Americans, ask not what your country can do for you...
[DONE: OK 1/22] jfk.wav | exact 0/108 (0.000) | norm 0/104 (0.000) | 2.6s
```

Per sample, the tool reports two distances:
- `exact`: character-level Levenshtein distance on raw text.
- `norm`: character-level Levenshtein distance after normalization
  (punctuation -> spaces, lowercase, whitespace collapsed).

## Building

```bash
make blas       # BLAS acceleration (Accelerate on macOS, OpenBLAS on Linux)
make test       # Run regression checks (requires built binary + model files)
make test-stream-cache  # Check stream cache on/off equivalence
make clean      # Clean build artifacts
```

For Linux, install OpenBLAS first:
```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# Fedora
sudo dnf install openblas-devel
```

## How Fast Is It?

### Optimization Notes (Apple M1 Pro, 8P+2E, `make blas`)

Measured back-to-back on the same machine with the 1.7B model and default flags,
best of three runs. "before" is the tree prior to the Q8 / threading / kernel
work; that build defaulted to 10 threads, the new one to the 8 performance cores.

| Audio | Phase | before | after (`q8`) | after (`q8-lm`) |
|-------|-------|--------|--------------|-----------------|
| 41s (Japanese) | encoder | 1672 ms | 635 ms | 635 ms |
| | decoder prefill | 1665 ms | 1476 ms | 1472 ms |
| | token generation | 7593 ms (44.9 ms/tok) | 4116 ms (24.4 ms/tok) | 3527 ms (20.9 ms/tok) |
| | **total inference** | **11031 ms** (3.7x realtime) | **6206 ms** (6.6x) | **5737 ms** (7.1x) |
| 89s (English) | **total inference** | **27203 ms** (3.3x realtime) | **15004 ms** (5.9x) | **14467 ms** (6.1x) |

That is **1.8x** end-to-end on the default settings, or 1.9x with `q8-lm`. With
`--weights bf16` — i.e. only the threading and kernel work, no quantization —
the 41s clip runs in 8558 ms, so roughly 1.3x of the gain is independent of
quantization.

Where the time went, and what changed:

- **Token generation is memory-bandwidth bound**, not compute bound. Every
  decoder weight is read once per token: 3.44 GB for the 1.7B model in bf16.
  A standalone microbenchmark of the Q8 matvec reaches 86.6 GB/s against
  94.7 GB/s for a pure read of the same bytes, and decode throughput is flat
  from 5 to 8 threads — the kernel sits at the memory wall, so the only
  remaining lever is reading fewer bytes.
- **The thread pool was the second bottleneck.** Each generated token issues
  ~110 tiny parallel dispatches, and every one went through a mutex + condvar
  round trip. A hybrid barrier (spin on an atomic generation counter, park only
  if the spin runs out) plus work-stealing chunks instead of a fixed row split
  was worth ~1.5x on decode by itself.
- **The default thread count included the efficiency cores.** With a
  barrier-synchronised split that makes every dispatch wait for the slowest
  core: 10 threads were 1.35x *slower* than 8 here, and it stays slower even
  with work stealing, because the E-cluster has much less memory bandwidth. The
  default is now the performance-core count, and workers are created at
  `QOS_CLASS_USER_INTERACTIVE`.
- **The encoder spent 57% of its time in `qwen_gelu`**, one scalar `tanhf` per
  element. Rewriting it through the exact identity
  `0.5 * (1 + tanh(z)) == sigmoid(2z)` turns it into a single vectorized `exp`
  (vForce on Apple, an auto-vectorizable polynomial elsewhere), and it is now
  threaded. The encoder went from 1672 ms to 635 ms.
- **Attention was an online-softmax loop with a scalar `expf` per key**, and in
  the encoder it was not threaded at all. It is now a two-pass kernel (scores,
  vectorized softmax, weighted V sum) threaded over heads, and decoder prefill
  routes attention through `sgemm`, which matters for long audio where it is
  O(seq²).

Decoder prefill is already close to the hardware ceiling: its `sgemm` calls run
at ~1.4 TFLOPS, about what the M1 AMX blocks sustain.

#### Reading the weights once for several rows

Being at the memory wall means the only remaining levers are reading fewer
bytes, or getting more work out of the bytes already read. Two changes take the
second route.

- **A batched Q8 matvec for short sequences.** The Q8 linear path had only two
  modes: a single-row matvec, and a prefill path that dequantizes weight row
  panels into f32 and hands them to `sgemm`. The panel path moves ~9 bytes per
  weight (read the int8, write the f32, read it back) to buy AMX throughput.
  That pays for a full-audio prefill of several hundred rows and is pure waste
  below that. A kernel that reads each weight row once and accumulates against
  every activation row moves 1.06 bytes per weight instead. Crossover measured
  near 500 rows; the default threshold is 256. Streaming, which re-prefills a
  few dozen rows per chunk, went from 18.6 s to 11.0 s on a 45s clip.

  The kernel is instantiated for a compile-time row count. With a runtime count
  the accumulator array is indexed dynamically, so it lives on the stack rather
  than in registers; specialising for 1/2/4/8 rows and splitting the batch
  across them was worth another 24% on batched decode.

- **The conv stem convolves chunks in groups.** The encoder splits mel into
  100-frame chunks and runs the three-layer Conv2D stem on each one
  separately. Per chunk the second layer is a 480x4320x800 GEMM — too narrow
  to keep the machine busy — and each layer costs its own dispatch. Since the
  chunks are independent, their im2col columns concatenate along the spatial
  axis and one GEMM covers a group of them; the group size is capped by the
  im2col scratch (64 MB). The conv stem went from 285 ms to 163 ms native and
  1653 ms to 1159 ms in wasm on a 41s clip.

  Blocking the portable `gemm_nn` over N was tried here and is *not* used: it
  keeps B in cache across the M loop, but it also shortens the inner run to
  the panel width, and the four strided A loads per k no longer amortize over
  it. A 128-column panel measured 30% slower.

- **Batched segment decoding** (`--batch`, above) applies the same idea to
  whole utterances, and **one prefill for the whole batch** applies it to the
  prefill: prefill costs roughly 590 ms of fixed dequantize sweep plus 3.6 ms
  per row, and concatenating a batch's segments into a single call pays the
  fixed part once. Rows keep their own sequence's attention, so this is a
  taller matrix, not a longer context.

### Compared To Other Runtimes

Worth being blunt about where this sits.

| Runtime | Hardware | 1.7B single-stream | Notes |
|---------|----------|--------------------|-------|
| vLLM (bf16, CUDA graphs) | datacenter GPU | **~67x realtime** (RTF 0.0148) | from the [Qwen3-ASR technical report](https://arxiv.org/html/2601.21337v2), Table 2; ~2000x realtime at concurrency 128 |
| this engine (`make blas`) | Apple M1 Pro laptop | **6.6–7.1x realtime** | Q8, single stream, no GPU |
| this engine (wasm) | same laptop, in a tab | **3.2–3.5x realtime** | 2.2 GB model fetched into the page |
| llama.cpp + GGUF Q8_0 | Apple M3 Air 8 GB | ~2.1x realtime | [reported here](https://github.com/shershah1024/qwen3-asr-llamacpp): 3.1 s for 6.6 s clips |

For the last row the closest thing this repo has is a 7.1 s clip, which runs in
1.64 s here (4.3x realtime) — but that is a different machine, a different clip
set, and their figure may include model load, so it is a rough marker rather
than a head-to-head.

So: **no, this is not faster than vLLM on a GPU** — it is roughly an order of
magnitude slower per stream, and far more than that under batching. That gap is
mostly physics rather than implementation quality. Token generation reads every
decoder weight once per token, so it is bandwidth-bound; this machine gives ~95
GB/s to the CPU and the Q8 kernel already reaches 87 of them, while an
H100-class part has ~3.3 TB/s. A ~30x bandwidth advantage turning into a ~10x
end-to-end advantage is about what you would predict.

What this engine offers instead is the other axis: no GPU, no CUDA, no Python,
~2 GB of RAM, a single binary (or a browser tab), fully offline, and audio that
never leaves the machine — at a speed that is still comfortably faster than
real time on a laptop.

### WebGPU Backend

`wasm/demo/webgpu-decoder.js` runs **the whole decoder on the GPU** — prefill and
token generation — leaving only mel and the audio encoder in wasm. Pick it with
the *decoder* dropdown in the demo. Design notes are in
[wasm/README.md](wasm/README.md).

Each half is there for a different reason, and the split follows measurement
rather than intuition. `wasm/demo/webgpu-probe.html` reports, on this M1 Pro:

| | measured |
|---|---|
| WebGPU compute shader, coalesced reads | **122 GB/s** |
| CPU (Q8 matvec, 8 threads) | 87 GB/s |
| `maxStorageBufferBindingSize` | 4.29 GB — the 1.72 GB of quants binds in one piece |
| `dot4I8Packed` | **emulated** on Apple GPUs, 24 GMAC/s |

Generation reads every decoder weight once per token, so it is bandwidth-bound
and the GPU's wider path to memory is what helps. Prefill is a batched GEMM and
compute-bound, where the GPU's arithmetic throughput is what helps. An int8 dot
product is the wrong tool for either — Apple GPUs have no DP4a equivalent, and
24 GMAC/s is well under the 1.72 GMAC each token needs, so the shaders unpack
int8 to f32 with `unpack4x8snorm` and use ordinary FMA.

Results, 1.7B, in a throttled background tab (floors, useful as ratios).
**Output is identical to the wasm decoder on every sample tried**, English and
Japanese, at contexts from 145 to 1170 tokens:

| clip | all wasm | wasm encoder + GPU decoder | speedup |
|------|----------|----------------------------|---------|
| 10s English | 3.14 s (3.2x realtime) | **2.14 s (4.7x)** | 1.47x |
| 11s English | 3.47 s (3.2x) | **2.39 s (4.6x)** | 1.45x |
| 41s Japanese | 14.7 s (2.8x) | **11.4 s (3.6x)** | 1.29x |
| 89s English | 36.7 s (2.4x) | **26.0 s (3.4x)** | 1.41x |

GPU prefill alone is 2.2–3.1x faster than the wasm prefill (7.03 s → 2.44 s on
the 41s clip), and generation lands at 22–27 ms/token roughly independent of
context length. What is left is the audio encoder: on the 41s clip the split is
now mel+encoder 3.9 s in wasm, GPU prefill 2.5 s, GPU generation 3.4 s. Moving
the encoder over is the obvious next piece — it needs its own shaders (LayerNorm
with bias, GELU FFN, windowed bidirectional attention, the conv2d stem) rather
than reusing the decoder's.

### Portable Build (no BLAS)

The non-BLAS path matters for wasm/browser targets, where neither Accelerate nor
OpenBLAS is available. It used to fall back to naive triple loops; it now uses a
blocked, threaded GEMM, and the Q8 kernels auto-vectorize well enough that the
portable Q8 matvec is within 1.22x of the hand-written NEON one.

```bash
make noblas
```

| Build | 41s Japanese clip, 1.7B |
|-------|--------------------------|
| `make blas` (Accelerate) | 6.2 s |
| `make noblas` (portable kernels) | 9.7 s |
| `make noblas`, before this work | 20.0 s — and that was for an 11s clip |

Transcripts are byte-identical across all three.

### Earlier Benchmarks (Apple M3 Max)

These predate the optimization work above and were measured on **Apple M3 Max** (128GB RAM) with `make blas` (single run per row).
`Inference`/`Audio` are from program summary. `wall` includes model-load and process overhead.

### Offline Mode (Full + Segmented)

| Setup | Audio | 0.6B (`Inference`, realtime, wall) | 1.7B (`Inference`, realtime, wall) |
|-------|-------|-------------------------------------|-------------------------------------|
| `samples/jfk.wav -S 0` | `11.0s` | `1.4s`, `7.99x`, `1.83s` | `2.6s`, `4.29x`, `3.17s` |
| `45s_dont_be_afraid_of_me.wav -S 30 -W 3` | `45.0s` | `3.4s`, `13.38x`, `3.64s` | `5.9s`, `7.63x`, `6.52s` |
| `89s_ill_come_back_down_as_soon_as.wav -S 30 -W 3` | `88.9s` | `13.1s`, `6.78x`, `13.39s` | `26.6s`, `3.34x`, `27.22s` |

### Streaming Mode (45s clip, interactive `--stream`)

| Setup | 0.6B (`Inference`, realtime) | 1.7B (`Inference`, realtime) |
|-------|--------------------------------|--------------------------------|
| cache ON (default, with prefill KV reuse) | `9.6s`, `4.69x` | `17.7s`, `2.54x` |
| cache OFF (`QWEN_STREAM_NO_ENC_CACHE=1`) | `22.0s`, `2.05x` | `34.3s`, `1.31x` |

Outputs were exact matches between cache ON/OFF in this benchmark.

### Streaming Non-Interactive Path (`--stream --silent`, file input)

For file input, `--stream --silent` does not run interactive chunk commits: it executes one direct final refinement pass and prints only the final transcript. This is useful for quiet batch runs, but its timing is not directly comparable to interactive `--stream`.

### Long-file Example (`/tmp/nirvana.wav`, 135s, 0.6B)

| Mode | Result |
|------|--------|
| `--stream` | `141.3s` inference (`0.96x` realtime) |
| offline segmented mode (`-S 30` in this measurement) | `14.0s` inference (`9.64x` realtime) |

## Model Architecture

Qwen3-ASR is a speech-to-text model available in 0.6B and 1.7B parameter variants:

**Pipeline:**
```
WAV -> 16kHz -> Mel Spectrogram -> Conv2D Stem -> Encoder -> Projection -> Decoder -> Tokens
```

| Component | Architecture |
|-----------|-------------|
| Conv2D Stem | 3 layers (480 channels, 3x3, stride 2), 8x time downsampling |
| Audio Encoder | Transformer with bidirectional windowed attention, sinusoidal PE |
| Projection | Linear -> GELU -> Linear (encoder dim -> decoder dim) |
| LLM Decoder | Qwen3 with GQA, per-head Q/K RMSNorm, NeoX split-half RoPE, SwiGLU |

| Parameter | 0.6B | 1.7B |
|-----------|------|------|
| Encoder layers | 18 | 24 |
| Encoder dim | 896 | 1024 |
| Decoder layers | 28 | 28 |
| Decoder dim | 1024 | 2048 |
| GQA heads | 16 Q / 8 KV | 16 Q / 8 KV |
| Vocab size | 151,936 | 151,936 |
| Weight format | BF16 | BF16 |
| Supported languages | 30 (see `--language`) |

## Memory Requirements

Memory usage has two parts:
- Static model footprint (allocated once at load time).
- Runtime footprint (depends on input length and decoding mode).

### Static Footprint (Model Load)

These numbers come from the current implementation and model files:
- Safetensors are memory-mapped.
- Encoder BF16 weights are converted to F32 and kept in heap memory.
- Decoder builds a fused gate/up matrix copy for faster decode.

| Component | 0.6B | 1.7B |
|-----------|------|------|
| safetensors mmap files | 1.747 GiB | 4.376 GiB |
| encoder copied F32 weights | 0.694 GiB | 1.183 GiB |
| decoder extra heap (fused + norms) | 0.328 GiB | 1.313 GiB |
| static total (theoretical) | 2.770 GiB | 6.871 GiB |

### Runtime Scaling (Why `-S 0` Grows)

For one segment, dominant runtime allocations scale with sequence length:

- `mel_frames ~= floor(audio_seconds * 100)`
- `enc_tokens ~= 13 * floor(mel_frames / 100) + ceil((mel_frames % 100) / 8)`
- `total_seq = enc_tokens + 15` (plus prompt/language/past-text tokens if used)
- `prefill_len = total_seq - 1`
- `pref_cap = next_pow2(prefill_len)`
- `kv_max = prefill_len + 1024`

Main growing buffers:
- KV cache: `2 * 28 * kv_max * 1024 * 4` bytes
- Prefill buffers:
  - 0.6B: `77,824 * pref_cap` bytes
  - 1.7B: `131,072 * pref_cap` bytes

Implications:
- `-S 0` (full-audio decode) lets `total_seq` grow with audio duration, so peak memory increases with file length.
- `-S 20` (or any segmented mode) bounds per-segment `total_seq`, so memory stays nearly flat as file length increases.
- Enabling `--past-text yes` adds previous text tokens to each segment/chunk prompt and can increase memory again.

### Measured Peak RSS (`--silent`)

Measured on Apple M3 Max using the current codebase:

| Audio length | 0.6B `-S 0` | 0.6B `-S 20` | 1.7B `-S 0` | 1.7B `-S 20` |
|--------------|-------------:|-------------:|-------------:|-------------:|
| 10.000s | 2.695 GiB | 2.688 GiB | 6.573 GiB | 6.573 GiB |
| 45.000s | 2.861 GiB | 2.757 GiB | 6.783 GiB | 6.700 GiB |
| 88.890s | 3.173 GiB | 2.815 GiB | 7.113 GiB | 6.742 GiB |
| 119.262s | 3.254 GiB | 2.789 GiB | 7.288 GiB | 6.706 GiB |

In practice:
- For long files, segmented mode is safer for both speed and memory.
- Default is `-S 0`, so for large files explicitly pick segmented mode (`-S 20` or `-S 30`).
- Use `-S 0` mainly for short files where full-context quality is worth the extra memory/time.

## License

MIT
