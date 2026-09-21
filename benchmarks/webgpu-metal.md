# Apple GPU WebGPU prefill tuning

WebGPU can be tuned for Apple GPUs while keeping standard WGSL and portable
fallbacks. The browser owns its graphics backend; this change does not call
Metal APIs from JavaScript or require a Metal-only browser extension.

## Change and compatibility

The decoder's Q/K score pass now stages 32 query positions and 32 key positions
in 4 KiB of workgroup memory. Each load serves multiple output scores. Loads
follow the contiguous sequence axis, and whole tiles above the causal diagonal
are skipped. Activations, accumulation, and scores remain f32; the pre-existing
optional f16 KV cache is unchanged.

- Enabled automatically for adapters reporting vendor `apple`, with sufficient
  workgroup limits, on one-shot prefills of at least 128 tokens. Chunked prompts
  use it for their first chunk; suffix attention retains its existing kernels.
- Other vendors, missing adapter information, and short prefills retain the
  scalar score kernel. No optional WebGPU feature was added as a requirement.
- The existing subgroup score kernels assume eight keys per workgroup. Both
  generation and suffix-prefill dispatch now require a guaranteed subgroup
  width of 32, rather than feature presence alone. Unknown, variable, or other
  widths select the portable score shaders, preventing incomplete coverage on
  wider subgroups.
- `new WebGPUDecoder(Module, { tiledPrefillScores: false })` disables tiling.
  The harness toggles `gpu.useTiledPrefillScores` to compare both compiled paths
  on the same loaded model.

## M1 Pro measurement, 2026-09-20

Apple M1 Pro, 32 GiB, Chromium WebGPU reporting `apple` / `metal-3`, Qwen3-ASR
1.7B Q8, 549 prompt embeddings from a local 41 s Japanese fixture. One model
load, one warmup pair, then three measured pairs with alternating order.

| Median | Scalar scores | Tiled scores | Reduction |
|---|---:|---:|---:|
| GPU Q/K score passes, all decoder layers | 92.08 ms | 19.99 ms | 78.3% |
| GPU prefill, sum of timestamped kernel passes | 1264.98 ms | 1192.49 ms | 5.7% |
| Prefill wall time | 1266.80 ms | 1196.47 ms | 5.5% |

These are **decoder prefill** improvements, not end-to-end transcription
speedups. Encoder and generation time are outside the GPU prefill measurement.
Browser scheduling, power state, and other workloads affect wall time.
Raw paired measurements are in
[webgpu-metal-m1-pro-2026-09-20.json](webgpu-metal-m1-pro-2026-09-20.json).

## Validation and reproduction

Use the production GPU-resident loader, which keeps the 2.18 GB model out of
the 2 GiB browser WASM heap:

```bash
source /path/to/emsdk/emsdk_env.sh
bash wasm/build.sh
python3 wasm/serve.py
# Open /wasm/demo/webgpu-attention-test.html
node --experimental-default-type=module --test tools/test-webgpu-attention.mjs
```

Prepare local goldens using the existing `wasm/dump-golden.js` workflow. The
fixtures and model are not distributed by this change. The check page prints
metrics and pass/fail results without printing transcripts.
A CPU-only Node run that loads the full 1.7B model needs a separate WASM build
with `-sMAXIMUM_MEMORY=4gb`; the production browser build stays at 2 GiB and
uses the reduced GPU-resident image.

1. **Load model**, then **Check kernels**: 13 lengths from 1 through 1024,
   including both sides of the 32/64/128 boundaries. Every score matched the
   scalar GPU result exactly; maximum checked CPU f64 reference difference was
   `3.69e-6`. Padding and the causal mask retained their sentinels.
2. **Compare all goldens**: all 23 fixtures (62–1566 prompt tokens, English and
   Japanese) produced identical token IDs with tiling off/on. All were within
   normalized CER 0.15 against CPU goldens. Long fixtures also exercise chained
   suffix prefills; the harness raises only its own context limit to 2300 and
   still enforces device buffer limits.
3. **Check portable fallback**: recompiles with subgroup and f16 paths disabled
   on the same device. JFK had normalized CER 0 and identical token IDs between
   one-shot and split prefill. Reload before optimized checks afterward.
4. **Profile prefill**: emits one warmup pair and three alternating pairs using
   GPU timestamps. Full-audio browser timing was not recorded for this change.

No WebGPU validation errors were reported. Adapter-selection tests cover
non-Apple/redacted vendors, insufficient limits, and subgroup widths 4–128.
Physical Intel, AMD, NVIDIA, and mobile GPUs were not available for this run;
feature-disabled testing on an M1 Pro does not establish their performance.

The production 2 GiB WASM build passed. A separate temporary 4 GiB build for
the CPU-only Node 1.7B loader passed all 22 English regressions (aggregate
normalized CER 0.0062). The modified C file's preprocessed WASM source was
identical to HEAD, confirming that the native Metal dispatch is excluded.
An additional 0.6B Node run scored 21/22: `15s_there_are_two_of_them_out_there`
had CER 0.231 against its reference. This smaller-model quality limitation is
outside the changed WASM code; it is not included in the 1.7B pass count.

Native Metal/MPS validation and its separate end-to-end measurements are in
[metal.md](metal.md).
