#include "matmult.h"

#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <sys/sysctl.h>

// Row-major storage: A is MxK, B is KxN, C is MxN.
// C(i,j) += A(i,p) * B(p,j)
//
// GotoBLAS-style GEMM applied to the transposed problem C^T = B^T * A^T,
// so the outer loop blocks over rows and packed A slivers stay in L1.

// Micro-kernel tile: MR rows x NR columns of C
#define MR 6
#define NR 8

// Cache block sizes can be overridden at compile time, e.g. -DKC=512
#ifndef KC
#define KC 512 // terms per kernel call: slivers of A and B must fit in L1
#endif
#ifndef RC
#define RC 1998 // rows per block (multiple of MR): packed A sits in L2
#endif
#ifndef CC
#define CC 1024 // columns per block (multiple of NR): packed B sits in L2
#endif

// Worker threads, 0 = one per active core. Override e.g. -DNUM_THREADS=1
#ifndef NUM_THREADS
#define NUM_THREADS 0
#endif

#define CACHE_LINE 128

static inline int64_t min64(const int64_t a, const int64_t b) {
  return a < b ? a : b;
}

#define LOAD_ROW(r)                                                            \
  float64x2_t c##r##0 = vld1q_f64(c + r * rs + 0);                             \
  float64x2_t c##r##1 = vld1q_f64(c + r * rs + 2);                             \
  float64x2_t c##r##2 = vld1q_f64(c + r * rs + 4);                             \
  float64x2_t c##r##3 = vld1q_f64(c + r * rs + 6);

#define STORE_ROW(r)                                                           \
  vst1q_f64(c + r * rs + 0, c##r##0);                                          \
  vst1q_f64(c + r * rs + 2, c##r##1);                                          \
  vst1q_f64(c + r * rs + 4, c##r##2);                                          \
  vst1q_f64(c + r * rs + 6, c##r##3);

#define FMA_ROW(r, vec, lane)                                                  \
  c##r##0 = vfmaq_laneq_f64(c##r##0, b0, vec, lane);                           \
  c##r##1 = vfmaq_laneq_f64(c##r##1, b1, vec, lane);                           \
  c##r##2 = vfmaq_laneq_f64(c##r##2, b2, vec, lane);                           \
  c##r##3 = vfmaq_laneq_f64(c##r##3, b3, vec, lane);

// C(0:6, 0:8) += a * b, with a and b packed; rs is the row stride of c.
// Uses 24 registers for C, 4 for b and 3 for a.
static inline void kernel_6x8(const int64_t kc, const double *restrict a,
                              const double *restrict b, double *restrict c,
                              const int64_t rs) {
  LOAD_ROW(0) LOAD_ROW(1) LOAD_ROW(2) LOAD_ROW(3) LOAD_ROW(4) LOAD_ROW(5)

  for (int64_t p = 0; p < kc; p++) {
    const float64x2_t b0 = vld1q_f64(b + 0);
    const float64x2_t b1 = vld1q_f64(b + 2);
    const float64x2_t b2 = vld1q_f64(b + 4);
    const float64x2_t b3 = vld1q_f64(b + 6);
    const float64x2_t a01 = vld1q_f64(a + 0);
    const float64x2_t a23 = vld1q_f64(a + 2);
    const float64x2_t a45 = vld1q_f64(a + 4);

    FMA_ROW(0, a01, 0) FMA_ROW(1, a01, 1)
    FMA_ROW(2, a23, 0) FMA_ROW(3, a23, 1)
    FMA_ROW(4, a45, 0) FMA_ROW(5, a45, 1)

    a += MR;
    b += NR;
  }

  STORE_ROW(0) STORE_ROW(1) STORE_ROW(2) STORE_ROW(3) STORE_ROW(4) STORE_ROW(5)
}

// Edge tile: run the full kernel on a zeroed buffer, then add the valid part.
static void kernel_edge(const int64_t mr, const int64_t nr, const int64_t kc,
                        const double *restrict a, const double *restrict b,
                        double *restrict c, const int64_t rs) {
  double tmp[MR * NR] = {0};
  kernel_6x8(kc, a, b, tmp, NR);

  for (int64_t i = 0; i < mr; i++) {
    for (int64_t j = 0; j < nr; j++) {
      c[i * rs + j] += tmp[i * NR + j];
    }
  }
}

// Packs rows x kc of A into slivers of MR rows, zero-padding the last one.
static void pack_A(const int64_t rows, const int64_t kc,
                   const double *restrict src, const int64_t lda,
                   double *restrict dst) {
  for (int64_t i = 0; i < rows; i += MR) {
    const int64_t mr = min64(MR, rows - i);
    for (int64_t p = 0; p < kc; p++) {
      for (int64_t t = 0; t < mr; t++) {
        dst[t] = src[(i + t) * lda + p];
      }
      for (int64_t t = mr; t < MR; t++) {
        dst[t] = 0.0;
      }
      dst += MR;
    }
  }
}

// Packs kc x cols of B into slivers of NR columns, zero-padding the last one.
static void pack_B(const int64_t kc, const int64_t cols,
                   const double *restrict src, const int64_t ldb,
                   double *restrict dst) {
  for (int64_t j = 0; j < cols; j += NR) {
    const int64_t nr = min64(NR, cols - j);
    for (int64_t p = 0; p < kc; p++) {
      for (int64_t t = 0; t < nr; t++) {
        dst[t] = src[p * ldb + j + t];
      }
      for (int64_t t = nr; t < NR; t++) {
        dst[t] = 0.0;
      }
      dst += NR;
    }
  }
}

static double *alloc_pack(const int64_t count) {
  void *p = NULL;
  if (posix_memalign(&p, CACHE_LINE, count * sizeof(double)) != 0) {
    abort();
  }
  return p;
}

static int64_t thread_count(void) {
  if (NUM_THREADS > 0) {
    return NUM_THREADS;
  }

  int count = 0;
  size_t size = sizeof(count);
  if (sysctlbyname("hw.activecpu", &count, &size, NULL, 0) != 0 || count < 1) {
    return 1;
  }
  return count;
}

typedef void (*task_fn)(const void *args, int64_t idx);

typedef struct {
  task_fn fn;
  const void *args;
  int64_t count;
  _Atomic int64_t next;
} parallel_job;

// Tasks come from a shared counter, so slower cores simply take fewer of them.
static void parallel_worker(void *ctx, size_t worker) {
  (void)worker;
  parallel_job *const job = ctx;

  for (;;) {
    const int64_t idx =
        atomic_fetch_add_explicit(&job->next, 1, memory_order_relaxed);
    if (idx >= job->count) {
      return;
    }
    job->fn(job->args, idx);
  }
}

static void parallel_for(const int64_t threads, const int64_t count,
                         const task_fn fn, const void *const args) {
  parallel_job job = {fn, args, count, 0};
  dispatch_apply_f(threads,
                   dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                   &job, parallel_worker);
}

// Current block, shared read-only by the tasks of one parallel region.
typedef struct {
  int64_t kc, rows, cols;
  int64_t ld_a, ld_b, ld_c;
  const double *A_src, *B_src;
  double *A_pack, *B_pack, *C;
} gemm_block;

static void pack_A_task(const void *args, const int64_t idx) {
  const gemm_block *const g = args;
  const int64_t i = idx * MR;
  pack_A(min64(MR, g->rows - i), g->kc, g->A_src + i * g->ld_a, g->ld_a,
         g->A_pack + i * g->kc);
}

static void pack_B_task(const void *args, const int64_t idx) {
  const gemm_block *const g = args;
  const int64_t j = idx * NR;
  pack_B(g->kc, min64(NR, g->cols - j), g->B_src + j, g->ld_b,
         g->B_pack + j * g->kc);
}

// One task is one MR-row sliver of A: tasks write disjoint rows of C.
static void kernel_task(const void *args, const int64_t idx) {
  const gemm_block *const g = args;
  const int64_t ir = idx * MR;
  const int64_t mr = min64(MR, g->rows - ir);
  const double *const a = &g->A_pack[ir * g->kc];

  for (int64_t jr = 0; jr < g->cols; jr += NR) {
    const int64_t nr = min64(NR, g->cols - jr);
    const double *const b = &g->B_pack[jr * g->kc];
    double *const c = &g->C[ir * g->ld_c + jr];

    if (mr == MR && nr == NR) {
      kernel_6x8(g->kc, a, b, c, g->ld_c);
    } else {
      kernel_edge(mr, nr, g->kc, a, b, c, g->ld_c);
    }
  }
}

void matmult(const int64_t M, const int64_t N, const int64_t K,
             const double *const restrict A, const double *const restrict B,
             double *const restrict C) {
  if (M <= 0 || N <= 0 || K <= 0) {
    return;
  }

  const int64_t max_kc = min64(KC, K);
  const int64_t max_rows = (min64(RC, M) + MR - 1) / MR * MR;
  const int64_t max_cols = (min64(CC, N) + NR - 1) / NR * NR;
  double *const A_pack = alloc_pack(max_rows * max_kc);
  double *const B_pack = alloc_pack(max_cols * max_kc);

  const int64_t threads = thread_count();
  gemm_block g = {.ld_a = K, .ld_b = N, .ld_c = N,
                  .A_pack = A_pack, .B_pack = B_pack};

  for (int64_t ii = 0; ii < M; ii += RC) {
    g.rows = min64(RC, M - ii);

    for (int64_t pp = 0; pp < K; pp += KC) {
      g.kc = min64(KC, K - pp);
      g.A_src = &A[ii * K + pp];
      parallel_for(threads, (g.rows + MR - 1) / MR, pack_A_task, &g);

      for (int64_t jj = 0; jj < N; jj += CC) {
        g.cols = min64(CC, N - jj);
        g.B_src = &B[pp * N + jj];
        g.C = &C[ii * N + jj];
        parallel_for(threads, (g.cols + NR - 1) / NR, pack_B_task, &g);
        parallel_for(threads, (g.rows + MR - 1) / MR, kernel_task, &g);
      }
    }
  }

  free(A_pack);
  free(B_pack);
}
