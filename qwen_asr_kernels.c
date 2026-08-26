/*
 * qwen_asr_kernels.c - Math kernels for Qwen3-ASR inference
 * Adapted from voxtral-realtime project.
 */

#include "qwen_asr_kernels.h"
#include "qwen_asr_kernels_impl.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#include <time.h>
#if (defined(__AVX512F__) || defined(__AVX2__)) && (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#include <pthread/qos.h>
#else
#include <unistd.h>
#endif

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================
 * Thread Pool
 *
 * Hybrid barrier: workers spin briefly on an atomic generation counter
 * before parking on a condvar, and the dispatcher spins on an atomic
 * completion counter.  Decoder token steps issue ~110 tiny parallel
 * dispatches each, so futex round-trips dominated the sync cost with the
 * previous mutex/condvar-only barrier.
 * ======================================================================== */

#define QWEN_MAX_THREADS 16

/* How long a thread spins before parking on the condvar.
 *
 * Parking is what costs: a futex round trip is tens of microseconds natively,
 * but in a browser it goes through Atomics.wait/notify between Web Workers and
 * measured ~4 ms per hop on Chrome. A decoder token issues ~110 dispatches, so
 * parking between them turned a 22 ms/token native step into 990 ms/token under
 * wasm. Spinning for a couple of milliseconds instead keeps the pool hot across
 * back-to-back dispatches while still releasing the cores when inference stops.
 */
/* Budgets are iteration counts, deliberately not wall-clock: reading a clock
 * from a wasm worker goes through JS and is far too expensive to do inside a
 * spin loop. ~1M iterations is on the order of a millisecond either side of the
 * native/wasm divide, which is enough to bridge back-to-back dispatches. */
#define QWEN_WORKER_SPINS    1000000
#define QWEN_JOIN_SPINS      1000000
#define QWEN_PARK_TIMEOUT_MS 2

typedef void (*parallel_fn_t)(int tid, int n_threads, void *arg);

static struct {
    pthread_t threads[QWEN_MAX_THREADS - 1];
    int tids[QWEN_MAX_THREADS - 1];
    int n_threads;
    _Atomic int shutdown;

    parallel_fn_t fn;
    void *arg;
    _Atomic int generation;
    _Atomic int n_done;
    _Atomic int n_parked;
    _Atomic int cursor;      /* dynamic work-stealing cursor for the current job */

    pthread_mutex_t mutex;
    pthread_cond_t cond_work;
} tp = {
    .n_threads = 1,
    .shutdown = 0,
    .generation = 0,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cond_work = PTHREAD_COND_INITIALIZER,
};

static inline void qwen_cpu_relax(void) {
#if defined(__ARM_ARCH) || defined(__aarch64__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    /* wasm and friends: nothing useful to emit, the atomic load is the fence */
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

/* Block until tp.generation moves past my_gen (or shutdown). Returns the new
 * generation. Spins first, then parks on the condvar with a bounded timeout so
 * a missed wakeup can never hang the pool. */
static int worker_wait_for_work(int my_gen) {
    for (int spins = 0; spins < QWEN_WORKER_SPINS; spins++) {
        int gen = atomic_load(&tp.generation);
        if (gen != my_gen || atomic_load(&tp.shutdown)) return gen;
        qwen_cpu_relax();
    }

    pthread_mutex_lock(&tp.mutex);
    atomic_fetch_add(&tp.n_parked, 1);
    for (;;) {
        int gen = atomic_load(&tp.generation);
        if (gen != my_gen || atomic_load(&tp.shutdown)) {
            atomic_fetch_sub(&tp.n_parked, 1);
            pthread_mutex_unlock(&tp.mutex);
            return gen;
        }
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += QWEN_PARK_TIMEOUT_MS * 1000 * 1000; /* safety net */
        if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
        pthread_cond_timedwait(&tp.cond_work, &tp.mutex, &ts);
    }
}

static void *worker_loop(void *arg) {
    int tid = *(int *)arg;
    int my_gen = 0;

    for (;;) {
        int gen = worker_wait_for_work(my_gen);
        if (atomic_load(&tp.shutdown)) return NULL;

        my_gen = gen;
        parallel_fn_t fn = tp.fn;
        void *a = tp.arg;
        int nt = tp.n_threads;

        fn(tid, nt, a);

        atomic_fetch_add(&tp.n_done, 1);
    }
}

/* Number of *performance* cores. On Apple silicon the efficiency cores are
 * several times slower than the P cores, so handing them an equal share of a
 * barrier-synchronised split makes every dispatch wait for the slowest core.
 * Measured on an M1 Pro (8P+2E): using all 10 logical CPUs is ~1.35x slower
 * than using the 8 P cores alone. */
int qwen_get_num_cpus(void) {
#ifdef __APPLE__
    int nlevels = 0;
    size_t len = sizeof(nlevels);
    if (sysctlbyname("hw.nperflevels", &nlevels, &len, NULL, 0) == 0 && nlevels > 1) {
        int p = 0;
        len = sizeof(p);
        /* perflevel0 is the fastest level. */
        if (sysctlbyname("hw.perflevel0.logicalcpu", &p, &len, NULL, 0) == 0 && p > 0)
            return p;
    }
    int n = 0;
    len = sizeof(n);
    sysctlbyname("hw.ncpu", &n, &len, NULL, 0);
    return n > 0 ? n : 1;
#else
    int n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? n : 1;
#endif
}

void qwen_set_threads(int n) {
    if (n < 1) n = 1;
    if (n > QWEN_MAX_THREADS) n = QWEN_MAX_THREADS;

    /* Shutdown existing workers */
    if (tp.n_threads > 1) {
        pthread_mutex_lock(&tp.mutex);
        atomic_store(&tp.shutdown, 1);
        atomic_fetch_add(&tp.generation, 1);
        pthread_cond_broadcast(&tp.cond_work);
        pthread_mutex_unlock(&tp.mutex);
        for (int i = 0; i < tp.n_threads - 1; i++)
            pthread_join(tp.threads[i], NULL);
        atomic_store(&tp.shutdown, 0);
        atomic_store(&tp.generation, 0);
        atomic_store(&tp.n_parked, 0);
    }

    tp.n_threads = n;
    if (n <= 1) return;

    pthread_attr_t attr;
    pthread_attr_init(&attr);
#if defined(__APPLE__)
    /* Ask the scheduler to keep the pool on performance cores. */
    pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    /* Count what actually started: a barrier that waits on a worker which was
     * never created would hang forever. Browsers in particular can refuse to
     * spawn more workers than the preallocated pool. */
    int started = 0;
    for (int i = 0; i < n - 1; i++) {
        tp.tids[i] = started + 1;
        if (pthread_create(&tp.threads[started], &attr, worker_loop,
                           &tp.tids[started]) == 0)
            started++;
    }
    pthread_attr_destroy(&attr);

    if (started != n - 1) {
        fprintf(stderr, "Thread pool: could only start %d of %d threads\n",
                started + 1, n);
        tp.n_threads = started + 1;
    }

    if (qwen_verbose >= 1)
        fprintf(stderr, "Thread pool: %d threads\n", tp.n_threads);
}

int qwen_get_threads(void) { return tp.n_threads; }

/* Claim the next chunk index of the current job.
 *
 * Static row splits leave the dispatcher waiting on whichever worker the OS
 * happened to preempt; handing out small chunks on demand keeps every core
 * busy until the job is actually finished. */
static inline int qwen_claim_chunk(void) {
    return atomic_fetch_add_explicit(&tp.cursor, 1, memory_order_relaxed);
}

/* Chunk size that gives each thread several bites at the work. */
static inline int qwen_chunk_size(int total, int n_threads, int min_chunk) {
    if (n_threads <= 1) return total > 0 ? total : 1;
    int c = (total + n_threads * 4 - 1) / (n_threads * 4);
    if (c < min_chunk) c = min_chunk;
    return c > 0 ? c : 1;
}

/* Dispatch work to all threads; main thread is tid=0 */
static void parallel_for(parallel_fn_t fn, void *arg) {
    /* Reset before the single-thread shortcut too: workers that hand out work
     * with qwen_claim_chunk() would otherwise see a cursor left over from the
     * previous job and silently skip every chunk. */
    atomic_store(&tp.cursor, 0);

    if (tp.n_threads <= 1) {
        fn(0, 1, arg);
        return;
    }

    int expect = tp.n_threads - 1;
    tp.fn = fn;
    tp.arg = arg;
    atomic_store(&tp.n_done, 0);
    atomic_fetch_add(&tp.generation, 1);

    /* Only pay for a futex wake if somebody actually parked. The seq_cst
     * ordering between the generation store and this load guarantees a worker
     * that is about to park has already been counted. */
    if (atomic_load(&tp.n_parked) > 0) {
        pthread_mutex_lock(&tp.mutex);
        pthread_cond_broadcast(&tp.cond_work);
        pthread_mutex_unlock(&tp.mutex);
    }

    fn(0, tp.n_threads, arg);

    for (int spins = 0; spins < QWEN_JOIN_SPINS; spins++) {
        if (atomic_load(&tp.n_done) >= expect) return;
        qwen_cpu_relax();
    }
    /* Fallback: a worker got descheduled. Yield instead of burning the core. */
    while (atomic_load(&tp.n_done) < expect)
        sched_yield();
}

/* Diagnostic: how many pool threads actually pick up a dispatch, and how long
 * a round trip costs. Useful when porting to a new threading substrate (wasm
 * workers in particular), where a barrier that silently runs single-threaded
 * looks like a slow kernel. */
typedef struct {
    _Atomic int participants;
    _Atomic int spins;
} pool_test_t;

static void pool_test_worker(int tid, int n_threads, void *arg) {
    (void)tid; (void)n_threads;
    pool_test_t *t = (pool_test_t *)arg;
    atomic_fetch_add(&t->participants, 1);
    /* A little work so the dispatcher cannot finish before workers start. */
    volatile int acc = 0;
    for (int i = 0; i < 200000; i++) acc += i;
    atomic_fetch_add(&t->spins, acc & 1);
}

void qwen_pool_selftest(int rounds, int *out_participants, double *out_ms) {
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    int total = 0;
    for (int r = 0; r < rounds; r++) {
        pool_test_t t;
        atomic_store(&t.participants, 0);
        atomic_store(&t.spins, 0);
        parallel_for(pool_test_worker, &t);
        total += atomic_load(&t.participants);
    }
    clock_gettime(CLOCK_MONOTONIC, &b);
    if (out_participants) *out_participants = rounds > 0 ? total / rounds : 0;
    if (out_ms) *out_ms = (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;
}

/* ========================================================================
 * Basic Element-wise Operations
 * ======================================================================== */

void qwen_add_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

void qwen_mul_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] *= b[i];
}

void qwen_scale(float *x, float s, int n) {
    for (int i = 0; i < n; i++) x[i] *= s;
}

void qwen_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, n * sizeof(float));
}

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

#ifndef USE_BLAS
/* Portable blocked GEMM used when no BLAS is linked (wasm builds, or Linux
 * without OpenBLAS). C[M,N] = A[M,K] @ B[N,K]^T, optionally + bias and with a
 * caller-supplied row stride for C.
 *
 * A 4x4 register tile keeps 16 running dot products alive over the shared k
 * loop, which clang vectorizes to NEON / SSE / wasm-simd128. The naive triple
 * loop it replaces reloaded both operands for every single multiply. */
#define QWEN_GEMM_MR 4
#define QWEN_GEMM_NR 4

/* 4x4 register tile over C = A @ B^T. Explicit scalar accumulators (rather
 * than an array) keep all 16 values in registers across the k loop. */
static void gemm_t_tile(float *C, int ldc, const float *restrict A,
                        const float *restrict B, const float *bias,
                        int M, int n0, int n1, int K) {
    for (int m = 0; m < M; m += QWEN_GEMM_MR) {
        int mr = M - m < QWEN_GEMM_MR ? M - m : QWEN_GEMM_MR;
        for (int n = n0; n < n1; n += QWEN_GEMM_NR) {
            int nr = n1 - n < QWEN_GEMM_NR ? n1 - n : QWEN_GEMM_NR;

            if (mr == 4 && nr == 4) {
                const float *restrict a0 = A + (size_t)(m + 0) * K;
                const float *restrict a1 = A + (size_t)(m + 1) * K;
                const float *restrict a2 = A + (size_t)(m + 2) * K;
                const float *restrict a3 = A + (size_t)(m + 3) * K;
                const float *restrict b0 = B + (size_t)(n + 0) * K;
                const float *restrict b1 = B + (size_t)(n + 1) * K;
                const float *restrict b2 = B + (size_t)(n + 2) * K;
                const float *restrict b3 = B + (size_t)(n + 3) * K;
                float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
                float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
                float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
                float c30 = 0, c31 = 0, c32 = 0, c33 = 0;
                for (int k = 0; k < K; k++) {
                    float av0 = a0[k], av1 = a1[k], av2 = a2[k], av3 = a3[k];
                    float bv0 = b0[k], bv1 = b1[k], bv2 = b2[k], bv3 = b3[k];
                    c00 += av0 * bv0; c01 += av0 * bv1; c02 += av0 * bv2; c03 += av0 * bv3;
                    c10 += av1 * bv0; c11 += av1 * bv1; c12 += av1 * bv2; c13 += av1 * bv3;
                    c20 += av2 * bv0; c21 += av2 * bv1; c22 += av2 * bv2; c23 += av2 * bv3;
                    c30 += av3 * bv0; c31 += av3 * bv1; c32 += av3 * bv2; c33 += av3 * bv3;
                }
                float b0v = bias ? bias[n + 0] : 0.0f, b1v = bias ? bias[n + 1] : 0.0f;
                float b2v = bias ? bias[n + 2] : 0.0f, b3v = bias ? bias[n + 3] : 0.0f;
                float *r0 = C + (size_t)(m + 0) * ldc + n;
                float *r1 = C + (size_t)(m + 1) * ldc + n;
                float *r2 = C + (size_t)(m + 2) * ldc + n;
                float *r3 = C + (size_t)(m + 3) * ldc + n;
                r0[0] = c00 + b0v; r0[1] = c01 + b1v; r0[2] = c02 + b2v; r0[3] = c03 + b3v;
                r1[0] = c10 + b0v; r1[1] = c11 + b1v; r1[2] = c12 + b2v; r1[3] = c13 + b3v;
                r2[0] = c20 + b0v; r2[1] = c21 + b1v; r2[2] = c22 + b2v; r2[3] = c23 + b3v;
                r3[0] = c30 + b0v; r3[1] = c31 + b1v; r3[2] = c32 + b2v; r3[3] = c33 + b3v;
            } else {
                for (int i = 0; i < mr; i++) {
                    const float *restrict a = A + (size_t)(m + i) * K;
                    for (int j = 0; j < nr; j++) {
                        const float *restrict b = B + (size_t)(n + j) * K;
                        float sum = 0.0f;
                        for (int k = 0; k < K; k++) sum += a[k] * b[k];
                        C[(size_t)(m + i) * ldc + n + j] =
                            sum + (bias ? bias[n + j] : 0.0f);
                    }
                }
            }
        }
    }
}

typedef struct {
    float *C;
    int ldc;
    const float *A;
    const float *B;
    const float *bias;
    int M, K, N;
} gemm_task_t;

static void gemm_worker(int tid, int n_threads, void *arg) {
    (void)tid;
    gemm_task_t *t = (gemm_task_t *)arg;
    int chunk = qwen_chunk_size(t->N, n_threads, QWEN_GEMM_NR);
    chunk = ((chunk + QWEN_GEMM_NR - 1) / QWEN_GEMM_NR) * QWEN_GEMM_NR;

    for (int c = qwen_claim_chunk(); ; c = qwen_claim_chunk()) {
        int n0 = c * chunk;
        if (n0 >= t->N) return;
        int n1 = n0 + chunk;
        if (n1 > t->N) n1 = t->N;
        gemm_t_tile(t->C, t->ldc, t->A, t->B, t->bias, t->M, n0, n1, t->K);
    }
}

/* C[M,N] = A[M,K] @ B[K,N], both row-major (conv2d's im2col shape).
 *
 * Four output rows share each B read. Two variants were tried and are both
 * slower, so this stays simple: a register-tiled 4x8 block was 20% slower
 * under wasm (the accumulators do not survive the engine's register
 * allocation), and blocking over N was 30% slower on the conv stem's
 * 480x4320x3200 shapes - it keeps B in cache across the M loop but shortens
 * the inner run to the panel width, and the four A loads per k are strided by
 * K and stop amortizing over it. */
static void gemm_nn_tile(float *restrict C, const float *restrict A,
                         const float *restrict B, int K, int N,
                         int m0, int m1) {
    for (int m = m0; m < m1; m += 4) {
        int mr = m1 - m < 4 ? m1 - m : 4;
        for (int j = 0; j < mr; j++)
            memset(C + (size_t)(m + j) * N, 0, (size_t)N * sizeof(float));

        for (int k = 0; k < K; k++) {
            const float *restrict brow = B + (size_t)k * N;
            if (mr == 4) {
                float a0 = A[(size_t)(m + 0) * K + k];
                float a1 = A[(size_t)(m + 1) * K + k];
                float a2 = A[(size_t)(m + 2) * K + k];
                float a3 = A[(size_t)(m + 3) * K + k];
                float *restrict r0 = C + (size_t)(m + 0) * N;
                float *restrict r1 = C + (size_t)(m + 1) * N;
                float *restrict r2 = C + (size_t)(m + 2) * N;
                float *restrict r3 = C + (size_t)(m + 3) * N;
                for (int n = 0; n < N; n++) {
                    float b = brow[n];
                    r0[n] += a0 * b; r1[n] += a1 * b;
                    r2[n] += a2 * b; r3[n] += a3 * b;
                }
            } else {
                for (int j = 0; j < mr; j++) {
                    float a = A[(size_t)(m + j) * K + k];
                    float *restrict r = C + (size_t)(m + j) * N;
                    for (int n = 0; n < N; n++) r[n] += a * brow[n];
                }
            }
        }
    }
}

static void gemm_nn_worker(int tid, int n_threads, void *arg) {
    (void)tid;
    gemm_task_t *t = (gemm_task_t *)arg;
    int chunk = qwen_chunk_size(t->M, n_threads, 4);
    chunk = ((chunk + 3) / 4) * 4;
    for (int c = qwen_claim_chunk(); ; c = qwen_claim_chunk()) {
        int m0 = c * chunk;
        if (m0 >= t->M) return;
        int m1 = m0 + chunk;
        if (m1 > t->M) m1 = t->M;
        gemm_nn_tile(t->C, t->A, t->B, t->K, t->N, m0, m1);
    }
}

static void qwen_gemm_nn_generic(float *C, const float *A, const float *B,
                                 int M, int K, int N) {
    gemm_task_t task = { C, N, A, B, NULL, M, K, N };
    if (tp.n_threads > 1 && M >= 8) {
        parallel_for(gemm_nn_worker, &task);
        return;
    }
    gemm_nn_tile(C, A, B, K, N, 0, M);
}

static void qwen_gemm_t_generic(float *C, int ldc, const float *A, const float *B,
                                const float *bias, int M, int K, int N) {
    gemm_task_t task = { C, ldc, A, B, bias, M, K, N };
    if (tp.n_threads > 1 && N >= 32) {
        parallel_for(gemm_worker, &task);
        return;
    }
    gemm_t_tile(C, ldc, A, B, bias, M, 0, N, K);
}
#endif /* !USE_BLAS */

void qwen_matmul_t(float *C, const float *A, const float *B, int M, int K, int N) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
#else
    qwen_gemm_t_generic(C, N, A, B, NULL, M, K, N);
#endif
}

void qwen_linear(float *y, const float *x, const float *W, const float *b,
                 int seq_len, int in_dim, int out_dim) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, out_dim, in_dim,
                1.0f, x, in_dim, W, in_dim,
                0.0f, y, out_dim);
    if (b != NULL) {
        for (int s = 0; s < seq_len; s++) {
            for (int o = 0; o < out_dim; o++) {
                y[s * out_dim + o] += b[o];
            }
        }
    }
#else
    qwen_gemm_t_generic(y, out_dim, x, W, b, seq_len, in_dim, out_dim);
#endif
}

void qwen_linear_nobias(float *y, const float *x, const float *W,
                         int seq_len, int in_dim, int out_dim) {
    qwen_linear(y, x, W, NULL, seq_len, in_dim, out_dim);
}

/* Convert bf16 buffer to f32 buffer */
static void bf16_to_f32_buf(float *dst, const uint16_t *src, size_t n) {
    uint32_t *d = (uint32_t *)(void *)dst;
    for (size_t i = 0; i < n; i++)
        d[i] = ((uint32_t)src[i]) << 16;
}

/* Reusable scratch buffer for bf16->f32 conversion */
static float *bf16_scratch = NULL;
static size_t bf16_scratch_cap = 0;

static float *bf16_get_scratch(size_t n) {
    if (n > bf16_scratch_cap) {
        free(bf16_scratch);
        bf16_scratch = (float *)malloc(n * sizeof(float));
        bf16_scratch_cap = bf16_scratch ? n : 0;
    }
    return bf16_scratch;
}

typedef struct {
    const uint16_t *src;
    size_t n;
    float *dst_f32;
} bf16_cache_entry_t;

static bf16_cache_entry_t *bf16_cache = NULL;
static int bf16_cache_len = 0;
static int bf16_cache_cap = 0;
static size_t bf16_cache_bytes = 0;
static size_t bf16_cache_limit_bytes = 0;
static int bf16_cache_limit_init = 0;

static void bf16_cache_init_limit(void) {
    if (bf16_cache_limit_init) return;
    bf16_cache_limit_init = 1;

    /* Default OFF. Override with QWEN_BF16_CACHE_MB=<n> to enable. */
    unsigned long long mb = 0;
    const char *env = getenv("QWEN_BF16_CACHE_MB");
    if (env && env[0] != '\0') {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env) mb = v;
    }
    bf16_cache_limit_bytes = (size_t)(mb * 1024ULL * 1024ULL);

    if (qwen_verbose >= 2) {
        fprintf(stderr, "BF16 cache: limit=%llu MB\n", mb);
    }
}

static const float *bf16_get_cached_f32(const uint16_t *src, size_t n) {
    bf16_cache_init_limit();

    for (int i = 0; i < bf16_cache_len; i++) {
        if (bf16_cache[i].src == src && bf16_cache[i].n == n) {
            return bf16_cache[i].dst_f32;
        }
    }

    if (bf16_cache_limit_bytes == 0) return NULL;

    size_t bytes = n * sizeof(float);
    if (bytes > bf16_cache_limit_bytes) return NULL;
    if (bf16_cache_bytes + bytes > bf16_cache_limit_bytes) return NULL;

    float *dst = (float *)malloc(bytes);
    if (!dst) return NULL;
    bf16_to_f32_buf(dst, src, n);

    if (bf16_cache_len == bf16_cache_cap) {
        int new_cap = bf16_cache_cap > 0 ? bf16_cache_cap * 2 : 256;
        bf16_cache_entry_t *tmp = (bf16_cache_entry_t *)realloc(
            bf16_cache, (size_t)new_cap * sizeof(bf16_cache_entry_t));
        if (!tmp) {
            free(dst);
            return NULL;
        }
        bf16_cache = tmp;
        bf16_cache_cap = new_cap;
    }

    bf16_cache[bf16_cache_len].src = src;
    bf16_cache[bf16_cache_len].n = n;
    bf16_cache[bf16_cache_len].dst_f32 = dst;
    bf16_cache_len++;
    bf16_cache_bytes += bytes;
    return dst;
}

static const float *bf16_get_f32_view(const uint16_t *src, size_t n) {
    const float *cached = bf16_get_cached_f32(src, n);
    if (cached) return cached;

    float *scratch = bf16_get_scratch(n);
    if (!scratch) return NULL;
    bf16_to_f32_buf(scratch, src, n);
    return scratch;
}

/*
 * Fused BF16 matvec: y[out_dim] = W_bf16[out_dim, in_dim] @ x[in_dim] + bias
 * Processes 2 output rows at a time to amortize x vector loads.
 */
static void bf16_matvec_fused(float *y, const float *x, const uint16_t *W_bf16,
                               const float *bias, int in_dim, int out_dim) {
    qwen_bf16_matvec_fused_impl(y, x, W_bf16, bias, in_dim, out_dim);
}

/* Threaded matvec: split output rows across threads */
typedef struct {
    float *y;
    const float *x;
    const uint16_t *W_bf16;
    const float *bias;
    int in_dim;
    int out_dim;
} matvec_task_t;

static void matvec_worker(int tid, int n_threads, void *arg) {
    matvec_task_t *t = (matvec_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->out_dim) end = t->out_dim;
    if (start >= end) return;

    bf16_matvec_fused(t->y + start, t->x,
                      t->W_bf16 + (size_t)start * t->in_dim,
                      t->bias ? t->bias + start : NULL,
                      t->in_dim, end - start);
}

static void bf16_matvec_threaded(float *y, const float *x, const uint16_t *W_bf16,
                                  const float *bias, int in_dim, int out_dim) {
    if (tp.n_threads <= 1) {
        bf16_matvec_fused(y, x, W_bf16, bias, in_dim, out_dim);
        return;
    }
    matvec_task_t task = { y, x, W_bf16, bias, in_dim, out_dim };
    parallel_for(matvec_worker, &task);
}

typedef struct {
    float *q;
    float *k;
    float *v;
    const float *x;
    const uint16_t *Wq_bf16;
    const uint16_t *Wk_bf16;
    const uint16_t *Wv_bf16;
    int in_dim;
    int q_dim;
    int kv_dim;
    int total_dim;
} qkv_matvec_task_t;

static void qkv_matvec_worker(int tid, int n_threads, void *arg) {
    qkv_matvec_task_t *t = (qkv_matvec_task_t *)arg;
    int chunk = (t->total_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->total_dim) end = t->total_dim;
    if (start >= end) return;

    int q_end = t->q_dim;
    int k_end = q_end + t->kv_dim;
    int v_end = k_end + t->kv_dim;

    if (start < q_end) {
        int s = start;
        int e = end < q_end ? end : q_end;
        if (s < e) {
            bf16_matvec_fused(t->q + s, t->x,
                              t->Wq_bf16 + (size_t)s * t->in_dim,
                              NULL, t->in_dim, e - s);
        }
    }

    if (end > q_end && start < k_end) {
        int s = start > q_end ? start - q_end : 0;
        int e_abs = end < k_end ? end : k_end;
        int e = e_abs - q_end;
        if (s < e) {
            bf16_matvec_fused(t->k + s, t->x,
                              t->Wk_bf16 + (size_t)s * t->in_dim,
                              NULL, t->in_dim, e - s);
        }
    }

    if (end > k_end && start < v_end) {
        int s = start > k_end ? start - k_end : 0;
        int e_abs = end < v_end ? end : v_end;
        int e = e_abs - k_end;
        if (s < e) {
            bf16_matvec_fused(t->v + s, t->x,
                              t->Wv_bf16 + (size_t)s * t->in_dim,
                              NULL, t->in_dim, e - s);
        }
    }
}

void qwen_linear_nobias_bf16_qkv(float *q, float *k, float *v, const float *x,
                                 const uint16_t *Wq_bf16,
                                 const uint16_t *Wk_bf16,
                                 const uint16_t *Wv_bf16,
                                 int in_dim, int q_dim, int kv_dim) {
    if (tp.n_threads <= 1) {
        bf16_matvec_fused(q, x, Wq_bf16, NULL, in_dim, q_dim);
        bf16_matvec_fused(k, x, Wk_bf16, NULL, in_dim, kv_dim);
        bf16_matvec_fused(v, x, Wv_bf16, NULL, in_dim, kv_dim);
        return;
    }

    qkv_matvec_task_t task = {
        .q = q,
        .k = k,
        .v = v,
        .x = x,
        .Wq_bf16 = Wq_bf16,
        .Wk_bf16 = Wk_bf16,
        .Wv_bf16 = Wv_bf16,
        .in_dim = in_dim,
        .q_dim = q_dim,
        .kv_dim = kv_dim,
        .total_dim = q_dim + 2 * kv_dim,
    };
    parallel_for(qkv_matvec_worker, &task);
}

void qwen_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                              int seq_len, int in_dim, int out_dim) {
    if (seq_len == 1) {
        bf16_matvec_threaded(y, x, W_bf16, NULL, in_dim, out_dim);
        return;
    }
    size_t n = (size_t)out_dim * in_dim;
    const float *W_f32 = bf16_get_f32_view(W_bf16, n);
    if (!W_f32) return;
    qwen_linear_nobias(y, x, W_f32, seq_len, in_dim, out_dim);
}

void qwen_linear_bf16(float *y, const float *x, const uint16_t *W_bf16,
                      const float *b, int seq_len, int in_dim, int out_dim) {
    if (seq_len == 1) {
        bf16_matvec_threaded(y, x, W_bf16, b, in_dim, out_dim);
        return;
    }
    size_t n = (size_t)out_dim * in_dim;
    const float *W_f32 = bf16_get_f32_view(W_bf16, n);
    if (!W_f32) return;
    qwen_linear(y, x, W_f32, b, seq_len, in_dim, out_dim);
}

/* Find argmax over a range of output rows [start, end).
 * Uses 2-row processing to amortize x vector loads (same as bf16_matvec_fused). */
static void argmax_bf16_range(const float *x, const uint16_t *W_bf16,
                               int in_dim, int start, int end,
                               int *best_out, float *best_val_out) {
    qwen_argmax_bf16_range_impl(x, W_bf16, in_dim, start, end, best_out, best_val_out);
}

typedef struct {
    const float *x;
    const uint16_t *W_bf16;
    int in_dim;
    int out_dim;
    int best_idx[QWEN_MAX_THREADS];
    float best_val[QWEN_MAX_THREADS];
} argmax_task_t;

static void argmax_worker(int tid, int n_threads, void *arg) {
    argmax_task_t *t = (argmax_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->out_dim) end = t->out_dim;
    if (start >= end) {
        t->best_val[tid] = -1e30f;
        t->best_idx[tid] = 0;
        return;
    }
    argmax_bf16_range(t->x, t->W_bf16, t->in_dim, start, end,
                      &t->best_idx[tid], &t->best_val[tid]);
}

int qwen_argmax_matvec_bf16(const float *x, const uint16_t *W_bf16,
                             int in_dim, int out_dim) {
    if (tp.n_threads <= 1) {
        int best;
        float best_val;
        argmax_bf16_range(x, W_bf16, in_dim, 0, out_dim, &best, &best_val);
        return best;
    }

    argmax_task_t task;
    task.x = x;
    task.W_bf16 = W_bf16;
    task.in_dim = in_dim;
    task.out_dim = out_dim;
    parallel_for(argmax_worker, &task);

    int best = task.best_idx[0];
    float best_val = task.best_val[0];
    for (int i = 1; i < tp.n_threads; i++) {
        if (task.best_val[i] > best_val) {
            best_val = task.best_val[i];
            best = task.best_idx[i];
        }
    }
    return best;
}

void qwen_matmul_t_bf16(float *C, const float *A, const uint16_t *B_bf16,
                         int M, int K, int N) {
    if (M == 1) {
        bf16_matvec_threaded(C, A, B_bf16, NULL, K, N);
    } else {
        size_t n = (size_t)N * K;
        const float *B_f32 = bf16_get_f32_view(B_bf16, n);
        if (!B_f32) return;
        qwen_matmul_t(C, A, B_f32, M, K, N);
    }
}

/* ========================================================================
 * Q8 Block-Quantized Weights
 * ======================================================================== */

static inline float q8_bf16_to_f32(uint16_t h) {
    uint32_t bits = ((uint32_t)h) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

size_t qwen_q8_bytes(const qwen_q8_mat_t *m) {
    if (!m || !m->q) return 0;
    if (QWEN_IS_Q4(m))
        return (size_t)m->rows * m->cols / 2 +
               (size_t)m->rows * (m->cols / QWEN_Q8_BLOCK) * sizeof(float);
    size_t n = (size_t)m->rows * m->cols;
    return n + (n / QWEN_Q8_BLOCK) * sizeof(float);
}

void qwen_q8_free(qwen_q8_mat_t *m) {
    if (!m) return;
    qwen_act_stats_free(m);
    if (m->owns) {
        free(m->q);
        free(m->scales);
    }
    m->q = NULL;
    m->scales = NULL;
    m->rows = m->cols = 0;
    m->owns = 0;
}

/* ---- Activation statistics for quantization calibration ---- */

int qwen_act_stats_attach(qwen_q8_mat_t *m) {
    if (!m || m->stats) return 0;
    qwen_act_stats_t *s = (qwen_act_stats_t *)calloc(1, sizeof(*s));
    if (!s) return -1;
    s->absmean = (double *)calloc((size_t)m->cols, sizeof(double));
    s->sqmean  = (double *)calloc((size_t)m->cols, sizeof(double));
    s->absmax  = (float  *)calloc((size_t)m->cols, sizeof(float));
    if (!s->absmean || !s->sqmean || !s->absmax) {
        free(s->absmean); free(s->sqmean); free(s->absmax); free(s);
        return -1;
    }
    m->stats = s;
    return 0;
}

void qwen_act_stats_free(qwen_q8_mat_t *m) {
    if (!m || !m->stats) return;
    free(m->stats->absmean);
    free(m->stats->sqmean);
    free(m->stats->absmax);
    free(m->stats);
    m->stats = NULL;
}

void qwen_act_stats_observe(const qwen_q8_mat_t *m, const float *x, int seq_len) {
    qwen_act_stats_t *s = m->stats;
    if (!s || !x || seq_len <= 0) return;
    int cols = m->cols;
    /* Serial on the calling thread. Calibration runs are not timed, and this
     * keeps the accumulator free of per-thread partials. */
    for (int r = 0; r < seq_len; r++) {
        const float *row = x + (size_t)r * cols;
        for (int c = 0; c < cols; c++) {
            float v = row[c];
            float a = fabsf(v);
            s->absmean[c] += a;
            s->sqmean[c]  += (double)v * (double)v;
            if (a > s->absmax[c]) s->absmax[c] = a;
        }
    }
    s->rows += seq_len;
}

void qwen_q8_attach(qwen_q8_mat_t *m, int8_t *q, float *scales, int rows, int cols) {
    m->q = q;
    m->scales = scales;
    m->rows = rows;
    m->cols = cols;
    m->owns = 0;
    m->bits = 8;
    m->stats = NULL;
}

void qwen_q4_attach(qwen_q8_mat_t *m, int8_t *q, float *scales, int rows, int cols) {
    qwen_q8_attach(m, q, scales, rows, cols);
    m->bits = 4;
}

void qwen_wmat_free(qwen_wmat_t *w) {
    if (!w) return;
    free(w->f32);
    w->f32 = NULL;
    qwen_q8_free(&w->q8);
    w->rows = w->cols = 0;
}

/* Quantize one row of bf16 weights into int8 blocks with per-block f32 scale. */
static void q8_quantize_bf16_row(int8_t *q, float *s, const uint16_t *w, int cols) {
    for (int b = 0; b < cols; b += QWEN_Q8_BLOCK) {
        float amax = 0.0f;
        for (int i = 0; i < QWEN_Q8_BLOCK; i++) {
            float v = q8_bf16_to_f32(w[b + i]);
            float a = v < 0.0f ? -v : v;
            if (a > amax) amax = a;
        }
        float scale = amax / 127.0f;
        float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        s[b / QWEN_Q8_BLOCK] = scale;
        for (int i = 0; i < QWEN_Q8_BLOCK; i++) {
            float v = q8_bf16_to_f32(w[b + i]) * inv;
            int qi = (int)(v < 0.0f ? v - 0.5f : v + 0.5f);
            if (qi > 127) qi = 127;
            if (qi < -127) qi = -127;
            q[b + i] = (int8_t)qi;
        }
    }
}

typedef struct {
    qwen_q8_mat_t *m;
    const uint16_t *A;
    const uint16_t *B; /* NULL unless interleaving two sources */
    int rows_each;
} q8_quant_task_t;

static void q8_quant_worker(int tid, int n_threads, void *arg) {
    q8_quant_task_t *t = (q8_quant_task_t *)arg;
    qwen_q8_mat_t *m = t->m;
    int cols = m->cols;
    int nb = cols / QWEN_Q8_BLOCK;
    int chunk = (m->rows + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > m->rows) end = m->rows;

    for (int r = start; r < end; r++) {
        const uint16_t *src;
        if (t->B) {
            /* rows alternate A0, B0, A1, B1, ... */
            const uint16_t *base = (r & 1) ? t->B : t->A;
            src = base + (size_t)(r >> 1) * cols;
        } else {
            src = t->A + (size_t)r * cols;
        }
        q8_quantize_bf16_row(m->q + (size_t)r * cols,
                             m->scales + (size_t)r * nb, src, cols);
    }
}

static int q8_alloc(qwen_q8_mat_t *m, int rows, int cols) {
    if (cols % QWEN_Q8_BLOCK != 0) {
        fprintf(stderr, "q8: cols=%d not a multiple of %d\n", cols, QWEN_Q8_BLOCK);
        return -1;
    }
    size_t n = (size_t)rows * cols;
    m->q = (int8_t *)malloc(n);
    m->scales = (float *)malloc((n / QWEN_Q8_BLOCK) * sizeof(float));
    if (!m->q || !m->scales) {
        qwen_q8_free(m);
        return -1;
    }
    m->rows = rows;
    m->cols = cols;
    m->owns = 1;
    m->bits = 8;
    m->stats = NULL;
    return 0;
}

int qwen_q8_from_bf16(qwen_q8_mat_t *m, const uint16_t *W_bf16, int rows, int cols) {
    if (q8_alloc(m, rows, cols) != 0) return -1;
    q8_quant_task_t task = { m, W_bf16, NULL, rows };
    parallel_for(q8_quant_worker, &task);
    return 0;
}

int qwen_q8_from_bf16_interleave2(qwen_q8_mat_t *m, const uint16_t *A,
                                  const uint16_t *B, int rows_each, int cols) {
    if (q8_alloc(m, 2 * rows_each, cols) != 0) return -1;
    q8_quant_task_t task = { m, A, B, rows_each };
    parallel_for(q8_quant_worker, &task);
    return 0;
}


/* ---- 4-bit weights ----
 *
 * Same block size and scale layout as Q8; only the payload changes. Within a
 * block, byte j carries value j in its low nibble and value j+32 in its high,
 * biased by 8, so one 16-byte load unpacks straight into two SIMD vectors and
 * the dot product needs no correction term. See qwen_asr_kernels.h. */

#define Q4_HALF (QWEN_Q8_BLOCK / 2)

static int q4_alloc(qwen_q8_mat_t *m, int rows, int cols) {
    if (cols % QWEN_Q8_BLOCK) return -1;
    size_t nq = (size_t)rows * cols / 2;
    size_t ns = (size_t)rows * (cols / QWEN_Q8_BLOCK);
    m->q = (int8_t *)malloc(nq);
    m->scales = (float *)malloc(ns * sizeof(float));
    m->rows = rows;
    m->cols = cols;
    m->owns = 1;
    m->bits = 4;
    m->stats = NULL;
    if (!m->q || !m->scales) { qwen_q8_free(m); return -1; }
    return 0;
}

/* Quantize one block of already-dequantized values. */
static void q4_pack_block(int8_t *dst, float *scale_out, const float *w) {
    float amax = 0.0f;
    for (int i = 0; i < QWEN_Q8_BLOCK; i++) {
        float a = fabsf(w[i]);
        if (a > amax) amax = a;
    }
    /* amax/8 rather than amax/7: -8 is representable, so all 16 levels are
     * used. One extra level is 12% more resolution, which is not nothing at
     * four bits. */
    float scale = amax / 8.0f;
    float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    *scale_out = scale;
    for (int j = 0; j < Q4_HALF; j++) {
        int lo = (int)lrintf(w[j] * inv);
        int hi = (int)lrintf(w[j + Q4_HALF] * inv);
        if (lo < -8) lo = -8; else if (lo > 7) lo = 7;
        if (hi < -8) hi = -8; else if (hi > 7) hi = 7;
        dst[j] = (int8_t)(((lo + 8) & 0x0F) | (((hi + 8) & 0x0F) << 4));
    }
}

/* Quantize a row to 4-bit blocks and dequantize it straight back.
 *
 * Exists so the calibration tooling measures the rounding the kernels really
 * do rather than its own reimplementation of it - the whole point of that
 * tooling is to predict what quantizing will cost. */
void qwen_q4_roundtrip_row(float *w, int cols) {
    int nb = cols / QWEN_Q8_BLOCK;
    for (int b = 0; b < nb; b++) {
        float *blk = w + b * QWEN_Q8_BLOCK;
        int8_t packed[Q4_HALF];
        float scale;
        q4_pack_block(packed, &scale, blk);
        for (int j = 0; j < Q4_HALF; j++) {
            blk[j]            = (float)(((packed[j] & 0x0F)) - 8) * scale;
            blk[j + Q4_HALF]  = (float)(((packed[j] >> 4) & 0x0F) - 8) * scale;
        }
    }
}

int qwen_q4_from_q8(qwen_q8_mat_t *dst, const qwen_q8_mat_t *src) {
    if (!src->q || QWEN_IS_Q4(src)) return -1;
    if (q4_alloc(dst, src->rows, src->cols) != 0) return -1;
    int nb = src->cols / QWEN_Q8_BLOCK;
    float blk[QWEN_Q8_BLOCK];
    for (int r = 0; r < src->rows; r++) {
        const int8_t *q = src->q + (size_t)r * src->cols;
        const float *s = src->scales + (size_t)r * nb;
        for (int b = 0; b < nb; b++) {
            float sc = s[b];
            const int8_t *qb = q + b * QWEN_Q8_BLOCK;
            for (int i = 0; i < QWEN_Q8_BLOCK; i++) blk[i] = (float)qb[i] * sc;
            q4_pack_block(dst->q + ((size_t)r * nb + b) * Q4_HALF,
                          &dst->scales[(size_t)r * nb + b], blk);
        }
    }
    return 0;
}

/* Q8 -> Q4, scaling column c by colscale[c] on the way through.
 *
 * This is the weight half of AWQ channel rescaling: the caller divides the
 * matching input channel by the same factor - see the folding in
 * qwen_asr_decoder.c - so the product is unchanged and only what the block
 * quantizer sees moves. */
int qwen_q4_from_q8_scaled(qwen_q8_mat_t *dst, const qwen_q8_mat_t *src,
                           const float *colscale) {
    if (!src->q || QWEN_IS_Q4(src) || !colscale) return -1;
    if (q4_alloc(dst, src->rows, src->cols) != 0) return -1;
    int nb = src->cols / QWEN_Q8_BLOCK;
    float blk[QWEN_Q8_BLOCK];
    for (int r = 0; r < src->rows; r++) {
        const int8_t *q = src->q + (size_t)r * src->cols;
        const float *s = src->scales + (size_t)r * nb;
        for (int b = 0; b < nb; b++) {
            float sc = s[b];
            const int8_t *qb = q + b * QWEN_Q8_BLOCK;
            const float *cs = colscale + b * QWEN_Q8_BLOCK;
            for (int i = 0; i < QWEN_Q8_BLOCK; i++) blk[i] = (float)qb[i] * sc * cs[i];
            q4_pack_block(dst->q + ((size_t)r * nb + b) * Q4_HALF,
                          &dst->scales[(size_t)r * nb + b], blk);
        }
    }
    return 0;
}

/* Divide rows row0, row0+stride, ... by s[0], s[1], ... - exactly, by touching
 * only the block scales.
 *
 * This is how the *input* side of AWQ rescaling gets folded away for the O and
 * down projections, which are not fed by a norm: O's input channel c is V's
 * output row c, and down's input channel c is the fused gate/up matrix's `up`
 * row c (rows interleave as gate0, up0, gate1, up1, ...), so scaling those rows
 * scales the activation with no work at inference time. */
void qwen_q8_scale_rows(qwen_q8_mat_t *m, const float *s, int row0, int stride,
                        int n) {
    if (!m->q || !m->scales || !s) return;
    int nb = m->cols / QWEN_Q8_BLOCK;
    for (int i = 0; i < n; i++) {
        int r = row0 + i * stride;
        if (r < 0 || r >= m->rows) break;
        float inv = s[i] != 0.0f ? 1.0f / s[i] : 1.0f;
        float *sc = m->scales + (size_t)r * nb;
        for (int b = 0; b < nb; b++) sc[b] *= inv;
    }
}

static float bf16_to_f32_(uint16_t h) {
    uint32_t bits = ((uint32_t)h) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

int qwen_q4_from_bf16(qwen_q8_mat_t *m, const uint16_t *W_bf16, int rows, int cols) {
    if (q4_alloc(m, rows, cols) != 0) return -1;
    int nb = cols / QWEN_Q8_BLOCK;
    float blk[QWEN_Q8_BLOCK];
    for (int r = 0; r < rows; r++) {
        const uint16_t *w = W_bf16 + (size_t)r * cols;
        for (int b = 0; b < nb; b++) {
            for (int i = 0; i < QWEN_Q8_BLOCK; i++)
                blk[i] = bf16_to_f32_(w[b * QWEN_Q8_BLOCK + i]);
            q4_pack_block(m->q + ((size_t)r * nb + b) * Q4_HALF,
                          &m->scales[(size_t)r * nb + b], blk);
        }
    }
    return 0;
}

int qwen_q4_from_bf16_interleave2(qwen_q8_mat_t *m, const uint16_t *A,
                                  const uint16_t *B, int rows_each, int cols) {
    if (q4_alloc(m, rows_each * 2, cols) != 0) return -1;
    int nb = cols / QWEN_Q8_BLOCK;
    float blk[QWEN_Q8_BLOCK];
    for (int r = 0; r < rows_each; r++) {
        for (int half = 0; half < 2; half++) {
            const uint16_t *w = (half ? B : A) + (size_t)r * cols;
            int dr = r * 2 + half;
            for (int b = 0; b < nb; b++) {
                for (int i = 0; i < QWEN_Q8_BLOCK; i++)
                    blk[i] = bf16_to_f32_(w[b * QWEN_Q8_BLOCK + i]);
                q4_pack_block(m->q + ((size_t)dr * nb + b) * Q4_HALF,
                              &m->scales[(size_t)dr * nb + b], blk);
            }
        }
    }
    return 0;
}

void qwen_q8_row_to_f32(float *dst, const qwen_q8_mat_t *m, int row) {
    int nb = m->cols / QWEN_Q8_BLOCK;
    if (QWEN_IS_Q4(m)) {
        /* Every path that dequantizes a whole matrix - the prefill panel, the
         * embedding lookup - comes through here, so 4-bit needs no separate
         * plumbing anywhere else. */
        const int8_t *q = m->q + (size_t)row * (m->cols / 2);
        const float *s = m->scales + (size_t)row * nb;
        for (int b = 0; b < nb; b++) {
            float sc = s[b];
            const uint8_t *qb = (const uint8_t *)q + b * Q4_HALF;
            float *db = dst + b * QWEN_Q8_BLOCK;
#if defined(__ARM_NEON)
            /* This is the prefill path's only route to f32, so a scalar loop
             * here shows up directly as slower prefill - measured at 29% on a
             * 41s clip before this was vectorized. */
            float32x4_t scv = vdupq_n_f32(sc);
            const uint8x16_t mask = vdupq_n_u8(0x0F);
            const int8x16_t bias = vdupq_n_s8(8);
            for (int j = 0; j < Q4_HALF; j += 16) {
                uint8x16_t raw = vld1q_u8(qb + j);
                int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), bias);
                int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), bias);
                const int8x16_t parts[2] = { lo, hi };
                for (int h = 0; h < 2; h++) {
                    int16x8_t w0 = vmovl_s8(vget_low_s8(parts[h]));
                    int16x8_t w1 = vmovl_s8(vget_high_s8(parts[h]));
                    float *o = db + j + h * Q4_HALF;
                    vst1q_f32(o,     vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))),  scv));
                    vst1q_f32(o + 4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))), scv));
                    vst1q_f32(o + 8, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))),  scv));
                    vst1q_f32(o + 12,vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))), scv));
                }
            }
#else
            for (int j = 0; j < Q4_HALF; j++) {
                db[j]           = (float)((qb[j] & 0x0F) - 8) * sc;
                db[j + Q4_HALF] = (float)((qb[j] >> 4) - 8) * sc;
            }
#endif
        }
        return;
    }
    const int8_t *q = m->q + (size_t)row * m->cols;
    const float *s = m->scales + (size_t)row * nb;
    for (int b = 0; b < nb; b++) {
        float sc = s[b];
        const int8_t *qb = q + b * QWEN_Q8_BLOCK;
        float *db = dst + b * QWEN_Q8_BLOCK;
#if defined(__ARM_NEON)
        float32x4_t scv = vdupq_n_f32(sc);
        for (int i = 0; i < QWEN_Q8_BLOCK; i += 16) {
            int8x16_t v = vld1q_s8(qb + i);
            int16x8_t lo = vmovl_s8(vget_low_s8(v));
            int16x8_t hi = vmovl_s8(vget_high_s8(v));
            vst1q_f32(db + i,      vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),  scv));
            vst1q_f32(db + i + 4,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), scv));
            vst1q_f32(db + i + 8,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),  scv));
            vst1q_f32(db + i + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), scv));
        }
#elif defined(__AVX2__)
        __m256 scv = _mm256_set1_ps(sc);
        for (int i = 0; i < QWEN_Q8_BLOCK; i += 8) {
            __m256i wi = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qb + i)));
            _mm256_storeu_ps(db + i, _mm256_mul_ps(_mm256_cvtepi32_ps(wi), scv));
        }
#else
        for (int i = 0; i < QWEN_Q8_BLOCK; i++) db[i] = (float)qb[i] * sc;
#endif
    }
}

/* ---- Activation quantization scratch (single-threaded producer) ---- */

static int8_t *q8_act_q = NULL;
static float *q8_act_s = NULL;
static int q8_act_cap = 0;

/* Quantize m activation rows into the shared scratch. */
static int q8_act_prepare_m(const float *x, int n, int m) {
    int need = n * m;
    if (need > q8_act_cap) {
        free(q8_act_q);
        free(q8_act_s);
        q8_act_q = (int8_t *)malloc((size_t)need);
        q8_act_s = (float *)malloc(((size_t)need / QWEN_Q8_BLOCK + 1) * sizeof(float));
        q8_act_cap = (q8_act_q && q8_act_s) ? need : 0;
        if (!q8_act_cap) return -1;
    }
    int nb = n / QWEN_Q8_BLOCK;
    for (int r = 0; r < m; r++)
        qwen_q8_quantize_row_impl(x + (size_t)r * n,
                                  q8_act_q + (size_t)r * n,
                                  q8_act_s + (size_t)r * nb, n);
    return 0;
}

static int q8_act_prepare(const float *x, int n) {
    if (n > q8_act_cap) {
        free(q8_act_q);
        free(q8_act_s);
        q8_act_q = (int8_t *)malloc((size_t)n);
        q8_act_s = (float *)malloc(((size_t)n / QWEN_Q8_BLOCK + 1) * sizeof(float));
        q8_act_cap = (q8_act_q && q8_act_s) ? n : 0;
        if (!q8_act_cap) return -1;
    }
    qwen_q8_quantize_row_impl(x, q8_act_q, q8_act_s, n);
    return 0;
}

/* ---- Threaded matvec ---- */

typedef struct {
    float *y;
    const int8_t *qx;
    const float *sx;
    const qwen_q8_mat_t *W;
} q8_matvec_task_t;

static void q8_matvec_worker(int tid, int n_threads, void *arg) {
    (void)tid;
    q8_matvec_task_t *t = (q8_matvec_task_t *)arg;
    const qwen_q8_mat_t *W = t->W;
    int nb = W->cols / QWEN_Q8_BLOCK;
    int chunk = qwen_chunk_size(W->rows, n_threads, 32);

    for (int c = qwen_claim_chunk(); ; c = qwen_claim_chunk()) {
        int start = c * chunk;
        if (start >= W->rows) return;
        int end = start + chunk;
        if (end > W->rows) end = W->rows;

        qwen_q8_matvec_impl(t->y + start, t->qx, t->sx,
                            W->q + (size_t)start * W->cols,
                            W->scales + (size_t)start * nb,
                            W->cols, end - start);
    }
}

/* ---- Short sequences: batched matvec, no dequantize ---- */

static int q8_batch_max_override = -1;

/* Setting this to 1 turns the batched path off, which is how a caller asks for
 * the f32-activation panel path everywhere. That matters for more than
 * benchmarking: the two paths are numerically different. The panel path
 * dequantizes the weights and multiplies in f32, while the batched matvec
 * quantizes the activations to int8, which costs about 2% relative error on
 * the encoder's output. It buys 40% off the encoder's transformer stack on a
 * short clip, so it stays on by default - but a reference dump that wants to
 * be compared against f32 arithmetic needs to be able to ask for f32. */
void qwen_set_q8_batch_max(int n) {
    q8_batch_max_override = n < 0 ? -1 : n;
}

static int qwen_q8_batch_max(void) {
    if (q8_batch_max_override >= 0) return q8_batch_max_override;
    static int v = -1;
    if (v < 0) {
        v = QWEN_Q8_BATCH_MAX;
        const char *env = getenv("QWEN_BATCH_MAX");
        if (env && env[0]) {
            int n = atoi(env);
            if (n >= 0) v = n;
        }
    }
    return v;
}

typedef struct {
    float *y;
    int ldy;
    const int8_t *qx;
    const float *sx;
    int m;
    const qwen_q8_mat_t *W;
} q8_matvec_m_task_t;

static void q8_matvec_m_worker(int tid, int n_threads, void *arg) {
    (void)tid;
    q8_matvec_m_task_t *t = (q8_matvec_m_task_t *)arg;
    const qwen_q8_mat_t *W = t->W;
    int nb = W->cols / QWEN_Q8_BLOCK;
    int chunk = qwen_chunk_size(W->rows, n_threads, 32);

    for (int c = qwen_claim_chunk(); ; c = qwen_claim_chunk()) {
        int start = c * chunk;
        if (start >= W->rows) return;
        int end = start + chunk;
        if (end > W->rows) end = W->rows;
        int rows = end - start;
        const int8_t *wq = W->q + (size_t)start * W->cols;
        const float *wsc = W->scales + (size_t)start * nb;

        /* The kernel holds one accumulator per activation row in registers, so
         * long sequences are walked in groups. The group loop is innermost so
         * this chunk's weight rows are read from DRAM once and stay in cache
         * for every group; hoisting it out would multiply weight traffic by
         * the group count. */
        for (int off = 0; off < t->m; off += QWEN_Q8_GROUP) {
            int m = t->m - off;
            if (m > QWEN_Q8_GROUP) m = QWEN_Q8_GROUP;
            qwen_q8_matvec_m_impl(t->y + (size_t)off * t->ldy + start, t->ldy,
                                  t->qx + (size_t)off * W->cols,
                                  t->sx + (size_t)off * nb, m,
                                  wq, wsc, W->cols, rows);
        }
    }
}

/* ---- Prefill: dequantize row panels and hand them to sgemm ---- */

static float *q8_panel = NULL;
static size_t q8_panel_cap = 0;

typedef struct {
    float *dst;
    const qwen_q8_mat_t *W;
    int row0;
    int nrows;
} q8_dequant_task_t;

static void q8_dequant_worker(int tid, int n_threads, void *arg) {
    q8_dequant_task_t *t = (q8_dequant_task_t *)arg;
    int chunk = (t->nrows + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->nrows) end = t->nrows;
    for (int r = start; r < end; r++)
        qwen_q8_row_to_f32(t->dst + (size_t)r * t->W->cols, t->W, t->row0 + r);
}

static void q4_matvec_worker(int tid, int n_threads, void *arg) {
    (void)tid;
    q8_matvec_task_t *t = (q8_matvec_task_t *)arg;
    const qwen_q8_mat_t *W = t->W;
    int nb = W->cols / QWEN_Q8_BLOCK;
    int chunk = qwen_chunk_size(W->rows, n_threads, 32);

    for (int c = qwen_claim_chunk(); ; c = qwen_claim_chunk()) {
        int start = c * chunk;
        if (start >= W->rows) return;
        int end = start + chunk;
        if (end > W->rows) end = W->rows;

        qwen_q4_matvec_impl(t->y + start, t->qx, t->sx,
                            W->q + (size_t)start * (W->cols / 2),
                            W->scales + (size_t)start * nb,
                            W->cols, end - start);
    }
}

static void q4_matvec_m_worker(int tid, int n_threads, void *arg) {
    (void)tid;
    q8_matvec_m_task_t *t = (q8_matvec_m_task_t *)arg;
    const qwen_q8_mat_t *W = t->W;
    int nb = W->cols / QWEN_Q8_BLOCK;
    int chunk = qwen_chunk_size(W->rows, n_threads, 32);

    for (int c = qwen_claim_chunk(); ; c = qwen_claim_chunk()) {
        int start = c * chunk;
        if (start >= W->rows) return;
        int end = start + chunk;
        if (end > W->rows) end = W->rows;

        qwen_q4_matvec_m_impl(t->y + start, t->ldy, t->qx, t->sx, t->m,
                              W->q + (size_t)start * (W->cols / 2),
                              W->scales + (size_t)start * nb,
                              W->cols, end - start);
    }
}

static void q8_linear_prefill(float *y, const float *x, const qwen_q8_mat_t *W,
                              int seq_len) {
    int cols = W->cols;
    int rows = W->rows;

    /* Dequantize in row panels, then let BLAS do the multiply.
     *
     * The panel wants to be *large*, not L2-sized: sgemm does its own blocking
     * and a big B lets it pick good shapes, while the dequantize itself becomes
     * one long streaming pass. Measured on a 549-token prefill (M1 Pro, 1.7B):
     * 2 M floats 2087 ms, 8 M 1514 ms, 32 M 1406 ms. 32 M covers the largest
     * matrix here (12288x2048) in a single panel, and panel_rows is capped at
     * the row count below, so nothing is over-allocated for smaller ones.
     * QWEN_PANEL_KF (in units of 1024 floats) overrides it for experiments. */
    static int panel_floats = 0;
    if (!panel_floats) {
        panel_floats = 32 * 1024 * 1024;
        const char *env = getenv("QWEN_PANEL_KF");
        if (env && env[0]) {
            int kf = atoi(env);
            if (kf > 0) panel_floats = kf * 1024;
        }
    }
    int panel_rows = panel_floats / cols;
    if (panel_rows < 64) panel_rows = 64;
    if (panel_rows > rows) panel_rows = rows;

    size_t need = (size_t)panel_rows * cols;
    if (need > q8_panel_cap) {
        free(q8_panel);
        q8_panel = (float *)malloc(need * sizeof(float));
        q8_panel_cap = q8_panel ? need : 0;
        if (!q8_panel_cap) return;
    }

    for (int r0 = 0; r0 < rows; r0 += panel_rows) {
        int nr = rows - r0;
        if (nr > panel_rows) nr = panel_rows;

        q8_dequant_task_t task = { q8_panel, W, r0, nr };
        parallel_for(q8_dequant_worker, &task);

#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, nr, cols,
                    1.0f, x, cols, q8_panel, cols,
                    0.0f, y + r0, rows);
#else
        qwen_gemm_t_generic(y + r0, rows, x, q8_panel, NULL, seq_len, cols, nr);
#endif
    }
}

void qwen_linear_bias_q8(float *y, const float *x, const qwen_q8_mat_t *W,
                         const float *bias, int seq_len) {
    qwen_linear_nobias_q8(y, x, W, seq_len);
    if (!bias) return;
    int out_dim = W->rows;
    for (int s = 0; s < seq_len; s++) {
        float *row = y + (size_t)s * out_dim;
        for (int o = 0; o < out_dim; o++) row[o] += bias[o];
    }
}

void qwen_linear_w(float *y, const float *x, const qwen_wmat_t *W,
                   const float *bias, int seq_len) {
    if (W->q8.q)
        qwen_linear_bias_q8(y, x, &W->q8, bias, seq_len);
    else
        qwen_linear(y, x, W->f32, bias, seq_len, W->cols, W->rows);
}

void qwen_linear_nobias_q8(float *y, const float *x, const qwen_q8_mat_t *W,
                           int seq_len) {
    if (W->stats) qwen_act_stats_observe(W, x, seq_len);

    /* Short sequences (streaming chunks, the token right after a prefill) are
     * badly served by the panel path: it moves ~9 bytes per weight where a
     * batched matvec moves 1.06. Crossover measured near 500 rows on M1 Pro
     * (where sgemm on AMX finally outruns SDOT); the default is set well below
     * it so a full-audio prefill still takes the sgemm path. */
    /* 4 bits: the single-row matvec is the whole point (generation reads every
     * weight once per token). Longer sequences fall to the panel path, which
     * dequantizes through qwen_q8_row_to_f32 and so needs no 4-bit variant. */
    if (QWEN_IS_Q4(W)) {
        if (seq_len == 1) {
            if (q8_act_prepare(x, W->cols) != 0) return;
            q8_matvec_task_t task = { y, q8_act_q, q8_act_s, W };
            parallel_for(q4_matvec_worker, &task);
            return;
        }
        /* Batched segment decode asks for 2-8 rows. Sending those to the panel
         * path would dequantize the whole matrix into f32 for a handful of
         * rows, which measured ten times slower than Q8 on a segmented run. */
        if (seq_len <= qwen_q8_batch_max()) {
            if (q8_act_prepare_m(x, W->cols, seq_len) != 0) return;
            q8_matvec_m_task_t task = { y, W->rows, q8_act_q, q8_act_s, seq_len, W };
            parallel_for(q4_matvec_m_worker, &task);
            return;
        }
        q8_linear_prefill(y, x, W, seq_len);
        return;
    }

    if (seq_len > 1 && seq_len <= qwen_q8_batch_max()) {
        if (q8_act_prepare_m(x, W->cols, seq_len) != 0) return;
        q8_matvec_m_task_t task = { y, W->rows, q8_act_q, q8_act_s, seq_len, W };
        parallel_for(q8_matvec_m_worker, &task);
        return;
    }
    if (seq_len != 1) {
        q8_linear_prefill(y, x, W, seq_len);
        return;
    }

    if (q8_act_prepare(x, W->cols) != 0) return;

    q8_matvec_task_t task = { y, q8_act_q, q8_act_s, W };
    parallel_for(q8_matvec_worker, &task);
}

/* ---- Fused Q/K/V matvec: one activation quantization, one dispatch ---- */

typedef struct {
    float *y[3];
    const qwen_q8_mat_t *W[3];
    const int8_t *qx;
    const float *sx;
    int total_rows;
} q8_qkv_task_t;

static void q8_qkv_worker(int tid, int n_threads, void *arg) {
    (void)tid;
    q8_qkv_task_t *t = (q8_qkv_task_t *)arg;
    int chunk = qwen_chunk_size(t->total_rows, n_threads, 32);

    for (int c = qwen_claim_chunk(); ; c = qwen_claim_chunk()) {
        int start = c * chunk;
        if (start >= t->total_rows) return;
        int end = start + chunk;
        if (end > t->total_rows) end = t->total_rows;

        int base = 0;
        for (int m = 0; m < 3; m++) {
            const qwen_q8_mat_t *W = t->W[m];
            int lo = start > base ? start - base : 0;
            int hi = end - base;
            if (hi > W->rows) hi = W->rows;
            if (lo < hi) {
                int nb = W->cols / QWEN_Q8_BLOCK;
                if (QWEN_IS_Q4(W)) {
                    qwen_q4_matvec_impl(t->y[m] + lo, t->qx, t->sx,
                                        W->q + (size_t)lo * (W->cols / 2),
                                        W->scales + (size_t)lo * nb,
                                        W->cols, hi - lo);
                } else {
                    qwen_q8_matvec_impl(t->y[m] + lo, t->qx, t->sx,
                                        W->q + (size_t)lo * W->cols,
                                        W->scales + (size_t)lo * nb,
                                        W->cols, hi - lo);
                }
            }
            base += W->rows;
        }
    }
}

void qwen_linear_nobias_q8_qkv(float *q, float *k, float *v, const float *x,
                               const qwen_q8_mat_t *Wq, const qwen_q8_mat_t *Wk,
                               const qwen_q8_mat_t *Wv) {
    if (Wq->stats) qwen_act_stats_observe(Wq, x, 1);
    if (Wk->stats) qwen_act_stats_observe(Wk, x, 1);
    if (Wv->stats) qwen_act_stats_observe(Wv, x, 1);

    if (q8_act_prepare(x, Wq->cols) != 0) return;

    q8_qkv_task_t task;
    task.y[0] = q; task.y[1] = k; task.y[2] = v;
    task.W[0] = Wq; task.W[1] = Wk; task.W[2] = Wv;
    task.qx = q8_act_q;
    task.sx = q8_act_s;
    task.total_rows = Wq->rows + Wk->rows + Wv->rows;

    parallel_for(q8_qkv_worker, &task);
}

/* ---- Streaming argmax over the (quantized) tied LM head ---- */

typedef struct {
    const int8_t *qx;
    const float *sx;
    const qwen_q8_mat_t *W;
    int best_idx[QWEN_MAX_THREADS];
    float best_val[QWEN_MAX_THREADS];
} q8_argmax_task_t;

static void q8_argmax_worker(int tid, int n_threads, void *arg) {
    q8_argmax_task_t *t = (q8_argmax_task_t *)arg;
    const qwen_q8_mat_t *W = t->W;
    int nb = W->cols / QWEN_Q8_BLOCK;
    int chunk = qwen_chunk_size(W->rows, n_threads, 256);

    float best_val = -1e30f;
    int best_idx = 0;

    for (int c = qwen_claim_chunk(); ; c = qwen_claim_chunk()) {
        int start = c * chunk;
        if (start >= W->rows) break;
        int end = start + chunk;
        if (end > W->rows) end = W->rows;

        int idx;
        float val;
        qwen_q8_argmax_range_impl(t->qx, t->sx,
                                  W->q + (size_t)start * W->cols,
                                  W->scales + (size_t)start * nb,
                                  W->cols, end - start, start, &idx, &val);
        if (val > best_val) { best_val = val; best_idx = idx; }
    }

    t->best_val[tid] = best_val;
    t->best_idx[tid] = best_idx;
}

int qwen_argmax_matvec_q8(const float *x, const qwen_q8_mat_t *W) {
    if (W->stats) qwen_act_stats_observe(W, x, 1);
    if (q8_act_prepare(x, W->cols) != 0) return 0;

    q8_argmax_task_t task;
    task.qx = q8_act_q;
    task.sx = q8_act_s;
    task.W = W;

    if (tp.n_threads <= 1) {
        parallel_for(q8_argmax_worker, &task);
        return task.best_idx[0];
    }
    parallel_for(q8_argmax_worker, &task);

    int best = task.best_idx[0];
    float best_val = task.best_val[0];
    for (int i = 1; i < tp.n_threads; i++) {
        if (task.best_val[i] > best_val) {
            best_val = task.best_val[i];
            best = task.best_idx[i];
        }
    }
    return best;
}


/* ---- Batched argmax: B hidden states scored against one LM head sweep ----
 *
 * Decoding a token is memory bound: the 151936x2048 head alone is 330 MB of
 * the 1.83 GB read per token. Scoring several independent sequences in one
 * sweep costs the same reads, so the head becomes nearly free per extra
 * sequence. */

typedef struct {
    const int8_t *qx;
    const float *sx;
    int m;
    const qwen_q8_mat_t *W;
    int best_idx[QWEN_MAX_THREADS][QWEN_Q8_MAX_M];
    float best_val[QWEN_MAX_THREADS][QWEN_Q8_MAX_M];
} q8_argmax_m_task_t;

#define ARGMAX_BLOCK 256

static void q8_argmax_m_worker(int tid, int n_threads, void *arg) {
    q8_argmax_m_task_t *t = (q8_argmax_m_task_t *)arg;
    const qwen_q8_mat_t *W = t->W;
    int nb = W->cols / QWEN_Q8_BLOCK;
    int m = t->m;
    int chunk = qwen_chunk_size(W->rows, n_threads, ARGMAX_BLOCK);
    float buf[QWEN_Q8_MAX_M * ARGMAX_BLOCK];

    float best_val[QWEN_Q8_MAX_M];
    int best_idx[QWEN_Q8_MAX_M];
    for (int r = 0; r < m; r++) { best_val[r] = -1e30f; best_idx[r] = 0; }

    for (int c = qwen_claim_chunk(); ; c = qwen_claim_chunk()) {
        int start = c * chunk;
        if (start >= W->rows) break;
        int end = start + chunk;
        if (end > W->rows) end = W->rows;

        for (int o = start; o < end; o += ARGMAX_BLOCK) {
            int rows = end - o;
            if (rows > ARGMAX_BLOCK) rows = ARGMAX_BLOCK;
            qwen_q8_matvec_m_impl(buf, ARGMAX_BLOCK, t->qx, t->sx, m,
                                  W->q + (size_t)o * W->cols,
                                  W->scales + (size_t)o * nb,
                                  W->cols, rows);
            for (int r = 0; r < m; r++) {
                const float *row = buf + (size_t)r * ARGMAX_BLOCK;
                float bv = best_val[r];
                int bi = best_idx[r];
                for (int i = 0; i < rows; i++)
                    if (row[i] > bv) { bv = row[i]; bi = o + i; }
                best_val[r] = bv;
                best_idx[r] = bi;
            }
        }
    }

    for (int r = 0; r < m; r++) {
        t->best_val[tid][r] = best_val[r];
        t->best_idx[tid][r] = best_idx[r];
    }
}

int qwen_argmax_matvec_q8_batch(const float *x, int m,
                                const qwen_q8_mat_t *W, int *out) {
    if (m <= 0) return 0;
    if (m > QWEN_Q8_MAX_M) return -1;
    if (m == 1) { out[0] = qwen_argmax_matvec_q8(x, W); return 0; }
    if (q8_act_prepare_m(x, W->cols, m) != 0) return -1;

    q8_argmax_m_task_t task;
    task.qx = q8_act_q;
    task.sx = q8_act_s;
    task.m = m;
    task.W = W;
    parallel_for(q8_argmax_m_worker, &task);

    int n = tp.n_threads > 0 ? tp.n_threads : 1;
    for (int r = 0; r < m; r++) {
        int best = task.best_idx[0][r];
        float bv = task.best_val[0][r];
        for (int i = 1; i < n; i++)
            if (task.best_val[i][r] > bv) { bv = task.best_val[i][r]; best = task.best_idx[i][r]; }
        out[r] = best;
    }
    return 0;
}

/* ========================================================================
 * 2D Convolution (im2col + BLAS sgemm)
 * ======================================================================== */

/*
 * im2col: Unroll input patches into a column matrix for GEMM-based conv2d.
 * Input: [C_in, H_in, W_in]
 * Output columns: [C_in * kH * kW, H_out * W_out]
 */
/* im2col for one patch row (one (channel, ky, kx) triple).
 *
 * The source row index is fixed for the whole row, so the vertical bounds check
 * hoists out; the horizontal one splits the output into a left pad, an interior
 * run and a right pad, which removes the per-element branch from the hot loop. */
static void im2col_row(const float *plane, float *col_ptr,
                       int ki, int kj,
                       int h_in, int w_in, int stride, int padding,
                       int h_out, int w_out) {
    /* Horizontal range of ow where iw = ow*stride - padding + kj is in bounds. */
    int ow_lo = 0;
    while (ow_lo < w_out && ow_lo * stride - padding + kj < 0) ow_lo++;
    int ow_hi = w_out;
    while (ow_hi > ow_lo && (ow_hi - 1) * stride - padding + kj >= w_in) ow_hi--;

    for (int oh = 0; oh < h_out; oh++) {
        float *dst = col_ptr + (size_t)oh * w_out;
        int ih = oh * stride - padding + ki;
        if (ih < 0 || ih >= h_in) {
            memset(dst, 0, (size_t)w_out * sizeof(float));
            continue;
        }
        const float *src = plane + (size_t)ih * w_in - padding + kj;
        for (int ow = 0; ow < ow_lo; ow++) dst[ow] = 0.0f;
        for (int ow = ow_lo; ow < ow_hi; ow++) dst[ow] = src[ow * stride];
        for (int ow = ow_hi; ow < w_out; ow++) dst[ow] = 0.0f;
    }
}

typedef struct {
    const float *in;
    float *cols;
    int batch;
    int c_in, h_in, w_in, kh, kw, stride, padding, h_out, w_out;
} im2col_task_t;

static void im2col_worker(int tid, int n_threads, void *arg) {
    (void)tid;
    im2col_task_t *t = (im2col_task_t *)arg;
    int rows = t->c_in * t->kh * t->kw;
    int spatial = t->h_out * t->w_out;
    int col_len = t->batch * spatial;
    size_t plane_in = (size_t)t->h_in * t->w_in;
    int chunk = qwen_chunk_size(rows, n_threads, 1);

    for (int c = qwen_claim_chunk(); ; c = qwen_claim_chunk()) {
        int start = c * chunk;
        if (start >= rows) return;
        int end = start + chunk;
        if (end > rows) end = rows;
        for (int r = start; r < end; r++) {
            int kj = r % t->kw;
            int ki = (r / t->kw) % t->kh;
            int ic = r / (t->kw * t->kh);
            float *dst = t->cols + (size_t)r * col_len;
            for (int b = 0; b < t->batch; b++)
                im2col_row(t->in + ((size_t)ic * t->batch + b) * plane_in,
                           dst + (size_t)b * spatial, ki, kj,
                           t->h_in, t->w_in, t->stride, t->padding,
                           t->h_out, t->w_out);
        }
    }
}

/* Reused across calls: the encoder runs this once per audio chunk, and the
 * largest buffer here is ~14 MB, so re-allocating it 120 times per utterance is
 * pure overhead. */
static float *conv_cols = NULL;
static size_t conv_cols_cap = 0;

/* Convolve `batch` independent images in one call.
 *
 * Layout is [channels][batch][h][w] throughout, so the im2col columns of every
 * image concatenate along the spatial axis and a single GEMM covers the batch.
 * The encoder's conv stem runs on 100-frame chunks, which alone give a GEMM of
 * 480x4320x800 - too narrow to keep the machine busy - and one dispatch per
 * chunk per layer. Batching widens N and cuts the dispatch count by the same
 * factor. */
void qwen_conv2d_batch(float *out, const float *in, const float *weight,
                       const float *bias, int batch,
                       int c_in, int c_out, int h_in, int w_in,
                       int kh, int kw, int stride, int padding) {
    int h_out = (h_in + 2 * padding - kh) / stride + 1;
    int w_out = (w_in + 2 * padding - kw) / stride + 1;
    int patch_size = c_in * kh * kw;
    int spatial_out = h_out * w_out;
    int n_cols = batch * spatial_out;

    size_t need = (size_t)patch_size * n_cols;
    if (need > conv_cols_cap) {
        free(conv_cols);
        conv_cols = (float *)malloc(need * sizeof(float));
        conv_cols_cap = conv_cols ? need : 0;
        if (!conv_cols_cap) return;
    }

    im2col_task_t task = { in, conv_cols, batch, c_in, h_in, w_in, kh, kw,
                           stride, padding, h_out, w_out };
    parallel_for(im2col_worker, &task);

    /* GEMM: weight[c_out, patch_size] @ cols[patch_size, batch*spatial_out] */
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                c_out, n_cols, patch_size,
                1.0f, weight, patch_size, conv_cols, n_cols,
                0.0f, out, n_cols);
#else
    qwen_gemm_nn_generic(out, weight, conv_cols, c_out, patch_size, n_cols);
#endif

    if (bias) {
        for (int oc = 0; oc < c_out; oc++) {
            float b = bias[oc];
            float *row = out + (size_t)oc * n_cols;
            for (int s = 0; s < n_cols; s++) row[s] += b;
        }
    }
}

void qwen_conv2d(float *out, const float *in, const float *weight, const float *bias,
                 int c_in, int c_out, int h_in, int w_in,
                 int kh, int kw, int stride, int padding) {
    qwen_conv2d_batch(out, in, weight, bias, 1, c_in, c_out,
                      h_in, w_in, kh, kw, stride, padding);
}

/* ========================================================================
 * Normalization
 * ======================================================================== */

void qwen_layer_norm(float *out, const float *x, const float *weight, const float *bias,
                     int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        /* Compute mean */
#if defined(__AVX512F__)
        __m512 sumv = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= hidden; i += 16) {
            sumv = _mm512_add_ps(sumv, _mm512_loadu_ps(x_row + i));
        }
        float mean = _mm512_reduce_add_ps(sumv);
        for (; i < hidden; i++) mean += x_row[i];
#elif defined(__AVX2__)
        __m256 sumv = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            sumv = _mm256_add_ps(sumv, _mm256_loadu_ps(x_row + i));
        }
        __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sumv), _mm256_extractf128_ps(sumv, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float mean = _mm_cvtss_f32(sum128);
        for (; i < hidden; i++) mean += x_row[i];
#else
        float mean = 0.0f;
        for (int i = 0; i < hidden; i++) mean += x_row[i];
#endif
        mean /= hidden;

        /* Compute variance */
#if defined(__AVX512F__) && defined(__FMA__)
        __m512 meanv = _mm512_set1_ps(mean);
        __m512 accv = _mm512_setzero_ps();
        int j = 0;
        for (; j + 16 <= hidden; j += 16) {
            __m512 v = _mm512_sub_ps(_mm512_loadu_ps(x_row + j), meanv);
            accv = _mm512_fmadd_ps(v, v, accv);
        }
        float var = _mm512_reduce_add_ps(accv);
        for (; j < hidden; j++) {
            float d = x_row[j] - mean;
            var += d * d;
        }
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 meanv = _mm256_set1_ps(mean);
        __m256 accv = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            __m256 v = _mm256_sub_ps(_mm256_loadu_ps(x_row + j), meanv);
            accv = _mm256_fmadd_ps(v, v, accv);
        }
        __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
        acc128 = _mm_hadd_ps(acc128, acc128);
        acc128 = _mm_hadd_ps(acc128, acc128);
        float var = _mm_cvtss_f32(acc128);
        for (; j < hidden; j++) {
            float d = x_row[j] - mean;
            var += d * d;
        }
#else
        float var = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float d = x_row[i] - mean;
            var += d * d;
        }
#endif
        var /= hidden;

        float inv_std = 1.0f / sqrtf(var + eps);
#if defined(__AVX512F__) && defined(__FMA__)
        __m512 meanv2 = _mm512_set1_ps(mean);
        __m512 invv = _mm512_set1_ps(inv_std);
        int k = 0;
        for (; k + 16 <= hidden; k += 16) {
            __m512 vx = _mm512_sub_ps(_mm512_loadu_ps(x_row + k), meanv2);
            __m512 vw = _mm512_loadu_ps(weight + k);
            __m512 vb = _mm512_loadu_ps(bias + k);
            __m512 v = _mm512_mul_ps(_mm512_mul_ps(vx, invv), vw);
            v = _mm512_add_ps(v, vb);
            _mm512_storeu_ps(out_row + k, v);
        }
        for (; k < hidden; k++) {
            out_row[k] = (x_row[k] - mean) * inv_std * weight[k] + bias[k];
        }
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 meanv2 = _mm256_set1_ps(mean);
        __m256 invv = _mm256_set1_ps(inv_std);
        int k = 0;
        for (; k + 8 <= hidden; k += 8) {
            __m256 vx = _mm256_sub_ps(_mm256_loadu_ps(x_row + k), meanv2);
            __m256 vw = _mm256_loadu_ps(weight + k);
            __m256 vb = _mm256_loadu_ps(bias + k);
            __m256 v = _mm256_mul_ps(_mm256_mul_ps(vx, invv), vw);
            v = _mm256_add_ps(v, vb);
            _mm256_storeu_ps(out_row + k, v);
        }
        for (; k < hidden; k++) {
            out_row[k] = (x_row[k] - mean) * inv_std * weight[k] + bias[k];
        }
#else
        for (int i = 0; i < hidden; i++) {
            out_row[i] = (x_row[i] - mean) * inv_std * weight[i] + bias[i];
        }
#endif
    }
}

void qwen_rms_norm(float *out, const float *x, const float *weight,
                   int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

#if defined(__AVX512F__) && defined(__FMA__)
        __m512 accv = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= hidden; i += 16) {
            __m512 v = _mm512_loadu_ps(x_row + i);
            accv = _mm512_fmadd_ps(v, v, accv);
        }
        float sum_sq = _mm512_reduce_add_ps(accv);
        for (; i < hidden; i++) sum_sq += x_row[i] * x_row[i];
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 accv = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            __m256 v = _mm256_loadu_ps(x_row + i);
            accv = _mm256_fmadd_ps(v, v, accv);
        }
        __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
        acc128 = _mm_hadd_ps(acc128, acc128);
        acc128 = _mm_hadd_ps(acc128, acc128);
        float sum_sq = _mm_cvtss_f32(acc128);
        for (; i < hidden; i++) sum_sq += x_row[i] * x_row[i];
#else
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; i++) {
            sum_sq += x_row[i] * x_row[i];
        }
#endif
        float rms_inv = 1.0f / sqrtf(sum_sq / hidden + eps);

#if defined(__AVX512F__)
        __m512 scale = _mm512_set1_ps(rms_inv);
        int j = 0;
        for (; j + 16 <= hidden; j += 16) {
            __m512 vx = _mm512_loadu_ps(x_row + j);
            __m512 vw = _mm512_loadu_ps(weight + j);
            _mm512_storeu_ps(out_row + j, _mm512_mul_ps(_mm512_mul_ps(vx, vw), scale));
        }
        for (; j < hidden; j++) out_row[j] = x_row[j] * rms_inv * weight[j];
#elif defined(__AVX2__)
        __m256 scale = _mm256_set1_ps(rms_inv);
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            __m256 vx = _mm256_loadu_ps(x_row + j);
            __m256 vw = _mm256_loadu_ps(weight + j);
            _mm256_storeu_ps(out_row + j, _mm256_mul_ps(_mm256_mul_ps(vx, vw), scale));
        }
        for (; j < hidden; j++) out_row[j] = x_row[j] * rms_inv * weight[j];
#else
        for (int i = 0; i < hidden; i++) {
            out_row[i] = x_row[i] * rms_inv * weight[i];
        }
#endif
    }
}

void qwen_rms_norm_per_head(float *x, const float *weight,
                             int seq_len, int n_heads, int head_dim, float eps) {
    /* x is [seq, n_heads * head_dim] - normalize each [head_dim] segment */
    int hidden = n_heads * head_dim;
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

#if defined(__AVX512F__) && defined(__FMA__)
            __m512 accv = _mm512_setzero_ps();
            int d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 v = _mm512_loadu_ps(vec + d);
                accv = _mm512_fmadd_ps(v, v, accv);
            }
            float sum_sq = _mm512_reduce_add_ps(accv);
            for (; d < head_dim; d++) sum_sq += vec[d] * vec[d];
#elif defined(__AVX2__) && defined(__FMA__)
            __m256 accv = _mm256_setzero_ps();
            int d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 v = _mm256_loadu_ps(vec + d);
                accv = _mm256_fmadd_ps(v, v, accv);
            }
            __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
            acc128 = _mm_hadd_ps(acc128, acc128);
            acc128 = _mm_hadd_ps(acc128, acc128);
            float sum_sq = _mm_cvtss_f32(acc128);
            for (; d < head_dim; d++) sum_sq += vec[d] * vec[d];
#else
            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum_sq += vec[d] * vec[d];
            }
#endif
            float rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);

#if defined(__AVX512F__)
            __m512 scale = _mm512_set1_ps(rms_inv);
            int j = 0;
            for (; j + 16 <= head_dim; j += 16) {
                __m512 v = _mm512_loadu_ps(vec + j);
                __m512 w = _mm512_loadu_ps(weight + j);
                _mm512_storeu_ps(vec + j, _mm512_mul_ps(_mm512_mul_ps(v, w), scale));
            }
            for (; j < head_dim; j++) vec[j] = vec[j] * rms_inv * weight[j];
#elif defined(__AVX2__)
            __m256 scale = _mm256_set1_ps(rms_inv);
            int j = 0;
            for (; j + 8 <= head_dim; j += 8) {
                __m256 v = _mm256_loadu_ps(vec + j);
                __m256 w = _mm256_loadu_ps(weight + j);
                _mm256_storeu_ps(vec + j, _mm256_mul_ps(_mm256_mul_ps(v, w), scale));
            }
            for (; j < head_dim; j++) vec[j] = vec[j] * rms_inv * weight[j];
#else
            for (int d = 0; d < head_dim; d++) {
                vec[d] = vec[d] * rms_inv * weight[d];
            }
#endif
        }
    }
}

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

/* Vectorized exp over a buffer.
 *
 * On Apple this is vForce; elsewhere it is a branch-free 2^k * poly(f)
 * expansion that clang auto-vectorizes to NEON/SSE/wasm-simd. Both are far
 * cheaper than a scalar libm call, and the encoder evaluates one per GELU. */
void qwen_vec_expf(float *dst, const float *src, int n) {
#if defined(__APPLE__) && defined(USE_BLAS)
    int cnt = n;
    vvexpf(dst, src, &cnt);
#else
    const float log2e = 1.44269504088896341f;
    for (int i = 0; i < n; i++) {
        float x = src[i];
        if (x < -87.0f) x = -87.0f;
        if (x > 88.0f) x = 88.0f;
        float t = x * log2e;
        int k = (int)(t + (t >= 0.0f ? 0.5f : -0.5f));
        float f = t - (float)k;
        /* degree-5 minimax for 2^f on [-0.5, 0.5] */
        float p = 0.0013333558f;
        p = p * f + 0.0096181291f;
        p = p * f + 0.0555041087f;
        p = p * f + 0.2402265069f;
        p = p * f + 0.6931471805f;
        p = p * f + 1.0f;
        union { uint32_t u; float f; } sc;
        sc.u = (uint32_t)((k + 127) << 23);
        dst[i] = p * sc.f;
    }
#endif
}

typedef struct {
    float *x;
    int n;
} elemwise_task_t;

/* GELU (tanh approximation), rewritten through the exact identity
 *   0.5 * (1 + tanh(z)) == sigmoid(2z)
 * so one vectorized exp replaces a scalar tanhf per element. */
static void gelu_range(float *x, int n) {
    enum { TILE = 1024 };
    float tmp[TILE];
    for (int off = 0; off < n; off += TILE) {
        int m = n - off;
        if (m > TILE) m = TILE;
        float *xp = x + off;
        for (int i = 0; i < m; i++) {
            float v = xp[i];
            tmp[i] = -1.5957691216057308f * (v + 0.044715f * v * v * v);
        }
        qwen_vec_expf(tmp, tmp, m);
        for (int i = 0; i < m; i++) xp[i] = xp[i] / (1.0f + tmp[i]);
    }
}

static void gelu_worker(int tid, int n_threads, void *arg) {
    elemwise_task_t *t = (elemwise_task_t *)arg;
    int chunk = (t->n + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->n) end = t->n;
    if (start >= end) return;
    gelu_range(t->x + start, end - start);
}

void qwen_gelu(float *x, int n) {
    if (tp.n_threads > 1 && n >= 8192) {
        elemwise_task_t task = { x, n };
        parallel_for(gelu_worker, &task);
        return;
    }
    gelu_range(x, n);
}

void qwen_silu(float *x, int n) {
    enum { TILE = 1024 };
    float tmp[TILE];
    for (int off = 0; off < n; off += TILE) {
        int m = n - off;
        if (m > TILE) m = TILE;
        float *xp = x + off;
        for (int i = 0; i < m; i++) tmp[i] = -xp[i];
        qwen_vec_expf(tmp, tmp, m);
        for (int i = 0; i < m; i++) xp[i] = xp[i] / (1.0f + tmp[i]);
    }
}

typedef struct {
    float *out;
    const float *gate_up;
    int seq_len;
    int intermediate;
} swiglu_task_t;

static void swiglu_worker(int tid, int n_threads, void *arg) {
    swiglu_task_t *t = (swiglu_task_t *)arg;
    int chunk = (t->seq_len + n_threads - 1) / n_threads;
    int s0 = tid * chunk;
    int s1 = s0 + chunk;
    if (s1 > t->seq_len) s1 = t->seq_len;
    if (s0 >= s1) return;

    int inter = t->intermediate;
    int alias_inplace = (t->out == t->gate_up);
    for (int s = s0; s < s1; s++) {
        const float *gu = t->gate_up + (size_t)s * 2 * inter;
        float *o = t->out + (size_t)s * inter;
        if (!alias_inplace) {
#if defined(__APPLE__) && defined(USE_BLAS)
            /* Fast path for prefill: vectorized exp(-g) using Accelerate/vForce. */
            for (int j = 0; j < inter; j++) o[j] = -gu[2 * j];
            int n = inter;
            vvexpf(o, o, &n);
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                o[j] = (g / (1.0f + o[j])) * u;
            }
#else
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g)); /* SiLU */
                o[j] = g * u;
            }
#endif
        } else {
            /* In-place mode (decode seq=1): keep single-pass scalar to avoid alias hazards. */
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g)); /* SiLU */
                o[j] = g * u;
            }
        }
    }
}

void qwen_swiglu_multiply(float *out, const float *gate_up, int seq_len, int intermediate) {
    swiglu_task_t task = {
        .out = out,
        .gate_up = gate_up,
        .seq_len = seq_len,
        .intermediate = intermediate
    };

    if (tp.n_threads > 1 && seq_len >= 2 && intermediate >= 256) {
        parallel_for(swiglu_worker, &task);
    } else {
        swiglu_worker(0, 1, &task);
    }
}

void qwen_softmax(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;
        float max_val = row[0];
        for (int c = 1; c < cols; c++) {
            if (row[c] > max_val) max_val = row[c];
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - max_val);
            sum += row[c];
        }
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            row[c] *= inv_sum;
        }
    }
}

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

static inline float qwen_dot_f32(const float *a, const float *b, int n) {
    return qwen_dot_f32_impl(a, b, n);
}

/* dst = dst * scale */
static inline void qwen_vec_scale_inplace(float *dst, float scale, int n) {
    qwen_vec_scale_inplace_impl(dst, scale, n);
}

/* dst += alpha * src */
static inline void qwen_vec_axpy_inplace(float *dst, const float *src, float alpha, int n) {
    qwen_vec_axpy_inplace_impl(dst, src, alpha, n);
}

/* Per-thread scratch for attention scores. Only one parallel_for runs at a
 * time, so a flat per-tid table is enough. */
static float *attn_scores[QWEN_MAX_THREADS];
static int attn_scores_cap[QWEN_MAX_THREADS];

static float *attn_scores_get(int tid, int n) {
    if (n > attn_scores_cap[tid]) {
        free(attn_scores[tid]);
        attn_scores[tid] = (float *)malloc((size_t)n * sizeof(float));
        attn_scores_cap[tid] = attn_scores[tid] ? n : 0;
    }
    return attn_scores[tid];
}

/* softmax over scores[0..n) in place; returns 1/sum. */
static float attn_softmax_inplace(float *scores, int n) {
    float max_score = scores[0];
    for (int j = 1; j < n; j++)
        if (scores[j] > max_score) max_score = scores[j];

#if defined(__APPLE__) && defined(USE_BLAS)
    /* vForce exp is several times faster than a scalar expf loop, and decoder
     * attention evaluates ~300k of them per generated token. */
    for (int j = 0; j < n; j++) scores[j] -= max_score;
    int cnt = n;
    vvexpf(scores, scores, &cnt);
#else
    for (int j = 0; j < n; j++) scores[j] = expf(scores[j] - max_score);
#endif

    float sum = 0.0f;
    for (int j = 0; j < n; j++) sum += scores[j];
    return sum > 0.0f ? 1.0f / sum : 0.0f;
}

/* out_row = sum_j w[j] * V[j*stride], four keys at a time for ILP. */
static void attn_weighted_sum(float *o_row, const float *V, int v_stride,
                              const float *w, int n, int head_dim) {
    memset(o_row, 0, (size_t)head_dim * sizeof(float));
    int j = 0;
    for (; j + 4 <= n; j += 4) {
        qwen_vec_axpy_inplace(o_row, V + (size_t)j * v_stride, w[j], head_dim);
        qwen_vec_axpy_inplace(o_row, V + (size_t)(j + 1) * v_stride, w[j + 1], head_dim);
        qwen_vec_axpy_inplace(o_row, V + (size_t)(j + 2) * v_stride, w[j + 2], head_dim);
        qwen_vec_axpy_inplace(o_row, V + (size_t)(j + 3) * v_stride, w[j + 3], head_dim);
    }
    for (; j < n; j++)
        qwen_vec_axpy_inplace(o_row, V + (size_t)j * v_stride, w[j], head_dim);
}

static void qwen_bidirectional_attention_heads(float *out, const float *Q, const float *K,
                                               const float *V, int n_heads, int head_dim,
                                               float scale, const int *window_starts,
                                               int n_windows, int head_start, int head_end,
                                               int tid) {
    int hidden = n_heads * head_dim;

    for (int h = head_start; h < head_end; h++) {
        for (int w = 0; w < n_windows; w++) {
            int ws = window_starts[w];
            int we = window_starts[w + 1];
            int n = we - ws;
            if (n <= 0) continue;

            float *scores = attn_scores_get(tid, n);
            if (!scores) return;

            const float *k_base = K + (size_t)ws * hidden + h * head_dim;
            const float *v_base = V + (size_t)ws * hidden + h * head_dim;

            for (int i = ws; i < we; i++) {
                const float *q_row = Q + (size_t)i * hidden + h * head_dim;
                float *o_row = out + (size_t)i * hidden + h * head_dim;

                for (int j = 0; j < n; j++)
                    scores[j] = qwen_dot_f32(q_row, k_base + (size_t)j * hidden, head_dim) * scale;

                float inv_sum = attn_softmax_inplace(scores, n);
                attn_weighted_sum(o_row, v_base, hidden, scores, n, head_dim);
                qwen_vec_scale_inplace(o_row, inv_sum, head_dim);
            }
        }
    }
}

typedef struct {
    float *out;
    const float *Q;
    const float *K;
    const float *V;
    int n_heads, head_dim;
    float scale;
    const int *window_starts;
    int n_windows;
} bidir_attn_task_t;

static void bidir_attn_worker(int tid, int n_threads, void *arg) {
    bidir_attn_task_t *t = (bidir_attn_task_t *)arg;
    int chunk = (t->n_heads + n_threads - 1) / n_threads;
    int h0 = tid * chunk;
    int h1 = h0 + chunk;
    if (h1 > t->n_heads) h1 = t->n_heads;
    if (h0 >= h1) return;

    qwen_bidirectional_attention_heads(t->out, t->Q, t->K, t->V, t->n_heads,
                                       t->head_dim, t->scale, t->window_starts,
                                       t->n_windows, h0, h1, tid);
}

void qwen_bidirectional_attention(float *out, const float *Q, const float *K,
                                   const float *V, int seq __attribute__((unused)),
                                   int n_heads, int head_dim, float scale,
                                   const int *window_starts, int n_windows) {
    if (tp.n_threads > 1 && n_heads >= 2) {
        bidir_attn_task_t task = {
            .out = out, .Q = Q, .K = K, .V = V,
            .n_heads = n_heads, .head_dim = head_dim, .scale = scale,
            .window_starts = window_starts, .n_windows = n_windows
        };
        parallel_for(bidir_attn_worker, &task);
        return;
    }
    qwen_bidirectional_attention_heads(out, Q, K, V, n_heads, head_dim, scale,
                                       window_starts, n_windows, 0, n_heads, 0);
}

static void qwen_causal_attention_heads(float *out, const float *Q, const float *K,
                                        const float *V, int seq_q, int seq_k,
                                        int n_heads, int n_kv_heads, int head_dim,
                                        float scale, int q_offset,
                                        int head_start, int head_end, int tid) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    for (int h = head_start; h < head_end; h++) {
        int kv_h = h / heads_per_kv;
        const float *k_base = K + (size_t)kv_h * head_dim;
        const float *v_base = V + (size_t)kv_h * head_dim;

        for (int i = 0; i < seq_q; i++) {
            const float *q_row = Q + (size_t)i * q_hidden + h * head_dim;
            float *o_row = out + (size_t)i * q_hidden + h * head_dim;
            int global_pos = q_offset + i;
            int k_end = global_pos + 1;
            if (k_end > seq_k) k_end = seq_k;
            if (k_end <= 0) {
                memset(o_row, 0, (size_t)head_dim * sizeof(float));
                continue;
            }

            float *scores = attn_scores_get(tid, k_end);
            if (!scores) return;

            for (int j = 0; j < k_end; j++)
                scores[j] = qwen_dot_f32(q_row, k_base + (size_t)j * kv_hidden, head_dim) * scale;

            float inv_sum = attn_softmax_inplace(scores, k_end);
            attn_weighted_sum(o_row, v_base, kv_hidden, scores, k_end, head_dim);
            qwen_vec_scale_inplace(o_row, inv_sum, head_dim);
        }
    }
}

typedef struct {
    float *out;
    const float *Q;
    const float *K;
    const float *V;
    int seq_q, seq_k;
    int n_heads, n_kv_heads;
    int head_dim;
    float scale;
    int q_offset;
} causal_attn_task_t;

static void causal_attn_worker(int tid, int n_threads, void *arg) {
    causal_attn_task_t *t = (causal_attn_task_t *)arg;
    int chunk = (t->n_heads + n_threads - 1) / n_threads;
    int h0 = tid * chunk;
    int h1 = h0 + chunk;
    if (h1 > t->n_heads) h1 = t->n_heads;
    if (h0 >= h1) return;

    qwen_causal_attention_heads(t->out, t->Q, t->K, t->V,
                                t->seq_q, t->seq_k, t->n_heads, t->n_kv_heads,
                                t->head_dim, t->scale, t->q_offset, h0, h1, tid);
}

#ifdef USE_BLAS
/* Prefill attention through GEMM.
 *
 * The row-at-a-time path costs O(seq_q * seq_k) dot products on the NEON unit;
 * for a 1170-token prefill that is ~200 GFLOP of the total. Scores and the
 * probability-weighted V sum are both plain matrix products with strided
 * operands, which cblas_sgemm handles directly (and runs on the AMX blocks).
 * Query tiling keeps the score buffer small for long audio. */
#define QWEN_ATTN_QTILE 256

static float *attn_gemm_scores = NULL;
static size_t attn_gemm_cap = 0;

static int causal_attention_gemm(float *out, const float *Q, const float *K, const float *V,
                                 int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                 int head_dim, float scale, int q_offset) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    size_t need = (size_t)QWEN_ATTN_QTILE * seq_k;
    if (need > attn_gemm_cap) {
        float *tmp = (float *)realloc(attn_gemm_scores, need * sizeof(float));
        if (!tmp) return -1;
        attn_gemm_scores = tmp;
        attn_gemm_cap = need;
    }
    float *S = attn_gemm_scores;

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;
        const float *k_head = K + (size_t)kv_h * head_dim;
        const float *v_head = V + (size_t)kv_h * head_dim;

        for (int q0 = 0; q0 < seq_q; q0 += QWEN_ATTN_QTILE) {
            int qb = seq_q - q0 < QWEN_ATTN_QTILE ? seq_q - q0 : QWEN_ATTN_QTILE;

            /* Widest key range any query in this tile can attend to. */
            int k_max = q_offset + q0 + qb;
            if (k_max > seq_k) k_max = seq_k;
            if (k_max <= 0) {
                for (int i = 0; i < qb; i++)
                    memset(out + (size_t)(q0 + i) * q_hidden + h * head_dim, 0,
                           (size_t)head_dim * sizeof(float));
                continue;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        qb, k_max, head_dim,
                        scale,
                        Q + (size_t)q0 * q_hidden + h * head_dim, q_hidden,
                        k_head, kv_hidden,
                        0.0f, S, k_max);

            for (int i = 0; i < qb; i++) {
                float *row = S + (size_t)i * k_max;
                int valid = q_offset + q0 + i + 1;
                if (valid > k_max) valid = k_max;
                if (valid <= 0) {
                    memset(row, 0, (size_t)k_max * sizeof(float));
                    continue;
                }
                float inv_sum = attn_softmax_inplace(row, valid);
                for (int j = 0; j < valid; j++) row[j] *= inv_sum;
                if (valid < k_max)
                    memset(row + valid, 0, (size_t)(k_max - valid) * sizeof(float));
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        qb, head_dim, k_max,
                        1.0f, S, k_max,
                        v_head, kv_hidden,
                        0.0f, out + (size_t)q0 * q_hidden + h * head_dim, q_hidden);
        }
    }
    return 0;
}
#endif /* USE_BLAS */

void qwen_causal_attention(float *out, const float *Q, const float *K, const float *V,
                            int seq_q, int seq_k, int n_heads, int n_kv_heads,
                            int head_dim, float scale, int q_offset) {
#ifdef USE_BLAS
    if (seq_q >= 8 &&
        causal_attention_gemm(out, Q, K, V, seq_q, seq_k, n_heads, n_kv_heads,
                              head_dim, scale, q_offset) == 0)
        return;
#endif
    if (tp.n_threads > 1 && n_heads >= 2 && (seq_q >= 2 || seq_k >= 128)) {
        causal_attn_task_t task = {
            .out = out, .Q = Q, .K = K, .V = V,
            .seq_q = seq_q, .seq_k = seq_k,
            .n_heads = n_heads, .n_kv_heads = n_kv_heads,
            .head_dim = head_dim, .scale = scale, .q_offset = q_offset
        };
        parallel_for(causal_attn_worker, &task);
        return;
    }

    qwen_causal_attention_heads(out, Q, K, V,
                                seq_q, seq_k, n_heads, n_kv_heads,
                                head_dim, scale, q_offset, 0, n_heads, 0);
}

/* ========================================================================
 * Position Embeddings
 * ======================================================================== */

void qwen_sinusoidal_pe(float *pe, int n_pos, int d_model) {
    int half = d_model / 2;
    float log_timescale = logf(10000.0f) / (float)(half - 1);

    for (int p = 0; p < n_pos; p++) {
        float *row = pe + p * d_model;
        for (int d = 0; d < half; d++) {
            float inv_timescale = expf(-(float)d * log_timescale);
            float angle = (float)p * inv_timescale;
            row[d] = sinf(angle);          /* first half: sin */
            row[half + d] = cosf(angle);   /* second half: cos */
        }
    }
}

void qwen_compute_rope_neox(float *cos_out, float *sin_out, const int *positions,
                              int seq, int head_dim, float theta) {
    int half = head_dim / 2;

    for (int s = 0; s < seq; s++) {
        float pos = (float)positions[s];
        for (int d = 0; d < half; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
            float angle = pos * freq;
            float c = cosf(angle);
            float sn = sinf(angle);
            /* Duplicate for full head_dim */
            cos_out[s * head_dim + d] = c;
            cos_out[s * head_dim + half + d] = c;
            sin_out[s * head_dim + d] = sn;
            sin_out[s * head_dim + half + d] = sn;
        }
    }
}

void qwen_apply_rope_neox(float *x, const float *cos_vals, const float *sin_vals,
                            int seq, int n_heads, int head_dim) {
    /*
     * NeoX split-half style:
     *   x1 = x[..., :half], x2 = x[..., half:]
     *   rotated = cat(-x2, x1)
     *   result = x * cos + rotated * sin
     */
    int half = head_dim / 2;
    int hidden = n_heads * head_dim;

    for (int s = 0; s < seq; s++) {
        const float *c = cos_vals + s * head_dim;
        const float *sn = sin_vals + s * head_dim;

        for (int h = 0; h < n_heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

#if defined(__AVX512F__) && defined(__FMA__)
            int d = 0;
            for (; d + 16 <= half; d += 16) {
                __m512 x1 = _mm512_loadu_ps(vec + d);
                __m512 x2 = _mm512_loadu_ps(vec + half + d);
                /* RoPE cache duplicates cos/sin across halves. */
                __m512 cc = _mm512_loadu_ps(c + d);
                __m512 ss = _mm512_loadu_ps(sn + d);
                __m512 new1 = _mm512_fmsub_ps(x1, cc, _mm512_mul_ps(x2, ss));
                __m512 new2 = _mm512_fmadd_ps(x2, cc, _mm512_mul_ps(x1, ss));
                _mm512_storeu_ps(vec + d, new1);
                _mm512_storeu_ps(vec + half + d, new2);
            }
            for (; d < half; d++) {
                float x1 = vec[d];
                float x2 = vec[half + d];
                vec[d]        = x1 * c[d]        + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
#elif defined(__AVX2__) && defined(__FMA__)
            int d = 0;
            for (; d + 8 <= half; d += 8) {
                __m256 x1 = _mm256_loadu_ps(vec + d);
                __m256 x2 = _mm256_loadu_ps(vec + half + d);
                __m256 cc = _mm256_loadu_ps(c + d);
                __m256 ss = _mm256_loadu_ps(sn + d);
                __m256 new1 = _mm256_fmsub_ps(x1, cc, _mm256_mul_ps(x2, ss));
                __m256 new2 = _mm256_fmadd_ps(x2, cc, _mm256_mul_ps(x1, ss));
                _mm256_storeu_ps(vec + d, new1);
                _mm256_storeu_ps(vec + half + d, new2);
            }
            for (; d < half; d++) {
                float x1 = vec[d];
                float x2 = vec[half + d];
                vec[d]        = x1 * c[d]        + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
#else
            for (int d = 0; d < half; d++) {
                float x1 = vec[d];           /* first half */
                float x2 = vec[half + d];    /* second half */
                vec[d]        = x1 * c[d]        + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
#endif
        }
    }
}
