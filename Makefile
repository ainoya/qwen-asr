# qwen_asr — Qwen3-ASR Pure C Inference Engine
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm -lpthread

# Platform detection
UNAME_S := $(shell uname -s)

# Source files
SRCS = qwen_asr.c qwen_asr_kernels.c qwen_asr_kernels_generic.c qwen_asr_kernels_neon.c qwen_asr_kernels_avx.c qwen_asr_kernels_wasm.c qwen_asr_audio.c qwen_asr_encoder.c qwen_asr_decoder.c qwen_asr_tokenizer.c qwen_asr_safetensors.c qwen_asr_pack.c qwen_asr_calib.c
OBJS = $(SRCS:.c=.o)
ifeq ($(METAL),1)
OBJS += qwen_asr_metal.o
endif
MAIN = main.c
TARGET = qwen_asr
METAL_CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK -DUSE_METAL
METAL_LDFLAGS = $(LDFLAGS) -framework Accelerate -framework Foundation -framework Metal -framework MetalPerformanceShaders

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

.PHONY: all clean debug info help blas metal noblas test test-metal test-stream-cache bench bench-plot bench-record

# Default: show available targets
all: help

help:
	@echo "qwen_asr — Qwen3-ASR Pure C Inference - Build Targets"
	@echo ""
	@echo "Choose a backend:"
	@echo "  make blas     - With BLAS acceleration (Accelerate/OpenBLAS)"
	@echo "  make metal    - Apple Silicon Q8 GPU kernels + Accelerate"
	@echo "  make noblas   - Portable kernels only, no BLAS dependency"
	@echo "                  (what a wasm/browser build compiles; ~1.6x slower here)"
	@echo ""
	@echo "Other targets:"
	@echo "  make debug    - Debug build with AddressSanitizer"
	@echo "  make test     - Run regression suite (requires ./qwen_asr and model files)"
	@echo "  make test-metal - Build Metal and check GPU kernels against CPU"
	@echo "  make test-stream-cache - Run stream cache on/off equivalence check"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make info     - Show build configuration"
	@echo ""
	@echo "Example: make blas && ./qwen_asr -d model_dir -i audio.wav"

# =============================================================================
# Backend: blas (Accelerate on macOS, OpenBLAS on Linux)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas: LDFLAGS += -framework Accelerate
else
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
blas: LDFLAGS += -lopenblas
endif
blas:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"
	@echo ""
	@echo "Built with BLAS backend"

# =============================================================================
# Backend: noblas (portable blocked GEMM + Q8 kernels, no external BLAS)
#
# This is the configuration a wasm/browser build compiles: no Accelerate, no
# OpenBLAS, everything through the in-tree kernels. Transcripts match the BLAS
# build exactly.
# =============================================================================
noblas: CFLAGS = $(CFLAGS_BASE)
noblas:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"
	@echo ""
	@echo "Built without BLAS (portable kernels)"

# Metal is opt-in; the other targets do not link Apple GPU frameworks.
metal:
ifeq ($(UNAME_S),Darwin)
	@$(MAKE) clean
	@$(MAKE) $(TARGET) tools/test-metal tools/test-metal-full tools/bench-metal tools/bench-metal-full METAL=1 CFLAGS="$(METAL_CFLAGS)" LDFLAGS="$(METAL_LDFLAGS)"
else
	@echo "Metal requires macOS on Apple Silicon"; exit 1
endif

qwen_asr_metal_source.h: qwen_asr_metal.metal qwen_asr_metal_full.metal
	python3 -c 'import json,sys; print(json.dumps("\n".join(open(p).read() for p in sys.argv[1:])))' $^ > $@

qwen_asr_metal.o: qwen_asr_metal.m qwen_asr_metal_source.h qwen_asr_metal.h qwen_asr.h qwen_asr_kernels.h qwen_asr_kernels_impl.h
	$(CC) $(CFLAGS) -fno-fast-math -fobjc-arc -c -o $@ $<

tools/test-metal: tools/test-metal.c $(OBJS)
	$(CC) $(CFLAGS) -fno-fast-math -I. -o $@ $^ $(LDFLAGS)

tools/bench-metal: tools/bench-metal.c $(OBJS)
	$(CC) $(CFLAGS) -I. -o $@ $^ $(LDFLAGS)

tools/bench-metal-full: tools/bench-metal-full.c $(OBJS)
	$(CC) $(CFLAGS) -I. -o $@ $^ $(LDFLAGS)

tools/test-metal-full: tools/test-metal-full.c $(OBJS)
	$(CC) $(CFLAGS) -fno-fast-math -I. -o $@ $^ $(LDFLAGS)

test-metal: metal
	./tools/test-metal

# =============================================================================
# Build rules
# =============================================================================
$(TARGET): $(OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c qwen_asr.h qwen_asr_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

# =============================================================================
# Utilities
# =============================================================================
clean:
	rm -f $(OBJS) qwen_asr_metal.o qwen_asr_metal_source.h main.o $(TARGET) tools/test-metal tools/test-metal-full tools/bench-metal tools/bench-metal-full

info:
	@echo "Platform: $(UNAME_S)"
	@echo "Compiler: $(CC)"
	@echo ""
ifeq ($(UNAME_S),Darwin)
	@echo "Backends: blas (Apple Accelerate), metal (Q8 prefill + Accelerate), noblas (portable)"
else
	@echo "Backends: blas (OpenBLAS), noblas (portable)"
endif

test:
	./asr_regression.py --binary ./qwen_asr --model-dir qwen3-asr-1.7b

bench:
	python3 tools/benchmark.py --run-wasm

bench-plot:
	python3 tools/benchmark.py --plot

bench-record:
	python3 tools/benchmark.py --record

test-webgpu:
	node tools/run-webgpu-test.mjs

# =============================================================================
# Dependencies
# =============================================================================
qwen_asr.o: qwen_asr.c qwen_asr.h qwen_asr_kernels.h qwen_asr_safetensors.h qwen_asr_audio.h qwen_asr_tokenizer.h
qwen_asr_kernels.o: qwen_asr_kernels.c qwen_asr_kernels.h qwen_asr_kernels_impl.h qwen_asr_metal.h
qwen_asr_kernels_generic.o: qwen_asr_kernels_generic.c qwen_asr_kernels_impl.h
qwen_asr_kernels_neon.o: qwen_asr_kernels_neon.c qwen_asr_kernels_impl.h
qwen_asr_kernels_avx.o: qwen_asr_kernels_avx.c qwen_asr_kernels_impl.h
qwen_asr_kernels_wasm.o: qwen_asr_kernels_wasm.c qwen_asr_kernels_impl.h
qwen_asr_audio.o: qwen_asr_audio.c qwen_asr_audio.h
qwen_asr_encoder.o: qwen_asr_encoder.c qwen_asr.h qwen_asr_kernels.h qwen_asr_safetensors.h
qwen_asr_decoder.o: qwen_asr_decoder.c qwen_asr.h qwen_asr_kernels.h qwen_asr_safetensors.h
qwen_asr_tokenizer.o: qwen_asr_tokenizer.c qwen_asr_tokenizer.h
qwen_asr_safetensors.o: qwen_asr_safetensors.c qwen_asr_safetensors.h
qwen_asr_pack.o: qwen_asr_pack.c qwen_asr.h qwen_asr_kernels.h qwen_asr_safetensors.h
qwen_asr_calib.o: qwen_asr_calib.c qwen_asr.h qwen_asr_kernels.h
main.o: main.c qwen_asr.h qwen_asr_kernels.h
