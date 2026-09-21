/* Native Metal numerical checks and isolated timings; no model download. */
#include "qwen_asr_metal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern int qwen_verbose;
static unsigned rng = 42;
static float random_value(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((rng >> 8) % 2001 - 1000.0f) / 1000.0f;
}
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}
static qwen_q8_mat_t matrix(int rows, int cols) {
    qwen_q8_mat_t w = {0};
    size_t n = (size_t)rows * cols;
    int8_t *q = malloc(n);
    float *s = malloc(n / 64 * sizeof(float));
    if (!q || !s) exit(2);
    for (size_t i = 0; i < n; i++) q[i] = (int8_t)(random_value() * 127);
    for (size_t i = 0; i < n / 64; i++) s[i] = (1 + random_value()) / 127;
    qwen_q8_attach(&w, q, s, rows, cols);
    w.owns = 1;
    return w;
}

static int linear_case(int rows, int cols, int seq) {
    qwen_q8_mat_t w = matrix(rows, cols);
    size_t ny = (size_t)rows * seq;
    float *x = malloc((size_t)seq * cols * sizeof(float));
    float *cpu = malloc(ny * sizeof(float)), *gpu = malloc(ny * sizeof(float));
    for (int i = 0; i < seq * cols; i++) x[i] = random_value();
    setenv("QWEN_METAL", "0", 1);
    qwen_linear_nobias_q8(cpu, x, &w, seq);
    setenv("QWEN_METAL", "1", 1);
    if (qwen_metal_linear(gpu, x, &w, seq)) return 1;
    /* Alternate backends after warming both. These are kernel smoke timings;
     * use bench-metal for whole-inference performance and cold-start costs. */
    double cpu_ms = 0, gpu_ms = 0;
    for (int repeat = 0; repeat < 3; repeat++) {
        setenv("QWEN_METAL", "0", 1);
        double t = now();
        qwen_linear_nobias_q8(cpu, x, &w, seq);
        cpu_ms += now() - t;
        setenv("QWEN_METAL", "1", 1);
        t = now();
        if (qwen_metal_linear(gpu, x, &w, seq)) return 1;
        gpu_ms += now() - t;
    }
    cpu_ms /= 3; gpu_ms /= 3;
    double error = 0, norm = 0;
    for (size_t i = 0; i < ny; i++) {
        if (!isfinite(gpu[i])) return 1;
        double d = cpu[i] - gpu[i]; error += d * d; norm += (double)cpu[i] * cpu[i];
    }
    double rel = sqrt(error / fmax(norm, 1e-30));
    printf("gemm %dx%d seq=%d rel=%.3g CPU=%.3f Metal=%.3f ms (%.2fx)\n", rows, cols, seq, rel, cpu_ms, gpu_ms, cpu_ms / gpu_ms);
    qwen_q8_free(&w); free(x); free(cpu); free(gpu);
    return rel > 2e-5;
}

static int fallback_cases(void) {
    qwen_q8_mat_t w = matrix(3, 64);
    float x[128] = {0}, y[6];
    for (int i = 0; i < 6; i++) y[i] = 123;
    int bad = 0;
    setenv("QWEN_METAL", "0", 1);
    bad += qwen_metal_linear(y, x, &w, 2) != -1;
    setenv("QWEN_METAL", "1", 1);
    w.bits = 4;
    bad += qwen_metal_linear(y, x, &w, 2) != -1;
    w.bits = 8;
    bad += qwen_metal_linear(y, x, &w, 0) != -1;
    bad += qwen_metal_linear(y, x, NULL, 2) != -1;
    for (int i = 0; i < 6; i++) bad += y[i] != 123;
    qwen_q8_free(&w);
    printf("fallbacks: %s\n", bad ? "FAIL" : "PASS");
    return bad;
}

int main(void) {
    qwen_verbose = 1;
    qwen_set_threads(8);
    qwen_set_q8_batch_max(1); /* compare against the f32 panel arithmetic */
    int bad = fallback_cases();
    bad += linear_case(1, 64, 2);
    bad += linear_case(37, 64, 257) + linear_case(65, 128, 513);
    bad += linear_case(2048, 2048, 549);
    bad += linear_case(6144, 2048, 549);
    bad += linear_case(12288, 2048, 549);
    bad += linear_case(2048, 6144, 549);
    /* All views have been freed; reload a small allocation to catch stale
     * buffer reuse when the allocator recycles tensor addresses. */
    bad += linear_case(37, 64, 257);
    printf("Metal tests: %s (%d failures)\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}
