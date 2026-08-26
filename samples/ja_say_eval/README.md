# Japanese Say evaluation set

This is a deterministic synthetic Japanese regression set for Qwen3-ASR 1.7B.
It is intended to catch output changes caused by quantization, kernels, prompt
assembly, and native/WebGPU parity work. It is not a substitute for a real-world
Japanese CER benchmark: macOS Say audio is clean and covers only synthetic voices.

The set crosses six text cases with three Japanese voice/rate profiles, producing
18 mono 16 kHz PCM WAV files. Cases cover ordinary prose, dates and times, numeric
expressions, Japanese proper nouns, mixed Japanese/technical English, and
context-dependent homophones.

## Regenerate

```sh
./samples/ja_say_eval/generate.sh
```

Generation requires macOS voices `Kyoko`, `Eddy (日本語（日本）)`, and
`Grandpa (日本語（日本）)`, plus `/usr/bin/afconvert`. The script fails instead
of silently substituting voices when one is unavailable.

## Evaluate the full set

```sh
./samples/ja_say_eval/evaluate.py \
  --binary ./qwen_asr \
  --model-dir ./qwen3-asr-1.7b \
  --weights q8-lm \
  --output /tmp/qwen-ja-say-q8-lm.json
```

For a faster six-clip smoke run, select one profile:

```sh
./samples/ja_say_eval/evaluate.py --profile kyoko_normal
```

For quantization and kernel changes, compare against a saved Q8 result. This is
the primary regression metric because Japanese number and name orthography can
have multiple valid transcriptions even when the spoken content is unchanged:

```sh
./samples/ja_say_eval/evaluate.py \
  --weights q8-lm \
  --baseline /tmp/qwen-ja-say-q8-lm.json \
  --output /tmp/qwen-ja-say-candidate.json
```

The evaluator reports normalized Japanese character error rate and sums the
engine's mel, encoder, prefill, decode, and inference timers. Each invocation
starts a new process, so wall time includes model loading while stage totals do
not. Keep the same voice files, thread count, power state, and model image when
comparing optimization branches.

The existing generic harness can also run the directory:

```sh
./asr_regression.py \
  --samples-root samples/ja_say_eval/audio \
  --model-dir qwen3-asr-1.7b \
  --skip-segment-check --skip-stream-check --skip-stream-cache-check \
  --arg=--weights --arg=q8-lm --arg=--language --arg=Japanese
```
