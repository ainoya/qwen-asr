#!/bin/bash
# Build the browser (WebAssembly) engine.
#
# Requires the Emscripten SDK on PATH:
#   git clone https://github.com/emscripten-core/emsdk
#   cd emsdk && ./emsdk install latest && ./emsdk activate latest
#   source ./emsdk_env.sh
#
# Output: wasm/demo/qwen_asr.js + qwen_asr.wasm
#
# Notes on the flags that matter:
#   -msimd128      the Q8 int8 dot products and the vectorized exp rely on it;
#                  without it decoding is several times slower
#   -pthread       the engine's thread pool maps onto Web Workers
#   MAXIMUM_MEMORY the 1.7B packed model needs ~2.2 GB resident plus activations
#   ALLOW_MEMORY_GROWTH so we don't reserve 4 GB up front

set -e
cd "$(dirname "$0")/.."

if ! command -v emcc >/dev/null 2>&1; then
    echo "error: emcc not found. Source emsdk_env.sh first." >&2
    exit 1
fi

OUT=wasm/demo
mkdir -p "$OUT"

SRCS="qwen_asr.c qwen_asr_kernels.c qwen_asr_kernels_generic.c \
qwen_asr_kernels_neon.c qwen_asr_kernels_avx.c qwen_asr_kernels_wasm.c qwen_asr_audio.c \
qwen_asr_encoder.c qwen_asr_decoder.c qwen_asr_tokenizer.c \
qwen_asr_safetensors.c qwen_asr_pack.c wasm/qwen_wasm.c"

EXPORTS='["_qwen_wasm_alloc","_qwen_wasm_release","_qwen_wasm_init","_qwen_wasm_ready","_qwen_wasm_threads","_qwen_wasm_pool_selftest","_qwen_wasm_pool_parts","_qwen_wasm_pool_ms","_qwen_wasm_shutdown","_qwen_wasm_selftest_start","_qwen_wasm_selftest_value",
"_qwen_wasm_batch_start","_qwen_wasm_job_done","_qwen_wasm_job_take","_qwen_wasm_take_text","_qwen_wasm_set_language",
"_qwen_wasm_set_segment_sec","_qwen_wasm_set_batch_size","_qwen_wasm_set_stream_params",
"_qwen_wasm_enc_shape","_qwen_wasm_enc_desc","_qwen_wasm_enc_tap_set",
"_qwen_wasm_enc_tap_ptr","_qwen_wasm_enc_tap_out","_qwen_wasm_enc_tap_tokens",
"_qwen_wasm_enc_tap_mel","_qwen_wasm_enc_tap_frames",
"_qwen_wasm_stream_start","_qwen_wasm_stream_push","_qwen_wasm_stream_finish",
"_qwen_wasm_total_ms","_qwen_wasm_encode_ms","_qwen_wasm_decode_ms",
"_qwen_wasm_text_tokens",
"_qwen_wasm_q8_desc","_qwen_wasm_f32_desc","_qwen_wasm_model_shape",
"_qwen_wasm_rms_eps","_qwen_wasm_rope_theta",
"_qwen_wasm_embeds_start","_qwen_wasm_embeds_done","_qwen_wasm_embeds_finish",
"_qwen_wasm_embeds_ptr","_qwen_wasm_embeds_mel_ms","_qwen_wasm_embeds_enc_ms",
"_qwen_wasm_token_text",
"_qwen_wasm_prefill_start","_qwen_wasm_prefill_done","_qwen_wasm_prefill_finish",
"_qwen_wasm_prefill_ms","_qwen_wasm_prompt_has_asr_text","_qwen_wasm_kv_k_ptr","_qwen_wasm_kv_v_ptr",
"_qwen_wasm_kv_len","_qwen_wasm_kv_stride",
"_malloc","_free"]'

emcc $SRCS -o "$OUT/qwen_asr.js" \
    -O3 -ffast-math -msimd128 -pthread \
    -Wall \
    -sMODULARIZE=1 \
    -sEXPORT_NAME=createQwenASR \
    -sEXPORT_ES6=0 \
    -sENVIRONMENT=web,worker,node \
    -sALLOW_MEMORY_GROWTH=1 \
    -sMAXIMUM_MEMORY=4gb \
    -sINITIAL_MEMORY=64mb \
    -sSTACK_SIZE=8mb \
    -sPTHREAD_POOL_SIZE=12 \
    -sPTHREAD_POOL_SIZE_STRICT=0 \
    -sEXPORTED_FUNCTIONS="$EXPORTS" \
    -sEXPORTED_RUNTIME_METHODS='["ccall","cwrap","UTF8ToString","stringToUTF8","lengthBytesUTF8","HEAPU8","HEAPF32","FS"]' \
    -sFORCE_FILESYSTEM=1 \
    -sASSERTIONS=0

ls -la "$OUT"/qwen_asr.js "$OUT"/qwen_asr.wasm
echo "built $OUT/qwen_asr.{js,wasm}"
