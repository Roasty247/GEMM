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

// Cache block sizes are upper bounds: block_size() picks the actual lengths.
// They can be overridden at compile time, e.g. -DKC=512
#ifndef KC
#define KC 512 // terms per kernel call: slivers of A and B must fit in L1
#endif
#ifndef RC
// Rows per block (multiple of MR). Packed A is only streamed, one sliver per
// kernel task, so it does not have to fit in L2: this just bounds its size.
// It is large on purpose, because every extra row block repacks all of B.
#define RC 12288
#endif
#ifndef CC
#define CC 1024 // columns per block (multiple of NR): packed B sits in L2
#endif

// Worker threads, 0 = one per active core. Override e.g. -DNUM_THREADS=1
#ifndef NUM_THREADS
#define NUM_THREADS 0
#endif

// Alignment of the pack buffers in bytes: one cache line on Apple Silicon.
#define CACHE_LINE 128

// Purpose: minimum of two 64-bit integers.
// Input:   a, b - the values to compare.
// Output:  the smaller of a and b.
static inline int64_t min64(const int64_t a, const int64_t b) {
  return a < b ? a : b;
}

// Purpose: choose the length of the blocks a dimension is split into. Uses the
//          fewest blocks that respect max_block and makes them about equally
//          long, so there is never a tiny tail block: n = 2050 with
//          max_block = 512 gives 5 blocks of 410, not 4 x 512 + 2.
// Input:   n         - length of the dimension (>= 1).
//          max_block - longest block allowed (the cache block size).
//          unit      - kernel tile size along this dimension (MR, NR, or 1).
// Output:  the block length, at most max_block. It is rounded up to a multiple
//          of unit when the cap allows it, so that only the last block can end
//          in an edge tile. It can therefore exceed n; callers clamp it.
static int64_t block_size(const int64_t n, const int64_t max_block,
                          const int64_t unit) {
  const int64_t count = (n + max_block - 1) / max_block;
  const int64_t even = (n + count - 1) / count;
  const int64_t whole = (even + unit - 1) / unit * unit;
  return whole <= max_block ? whole : even;
}

// Purpose: load row r of the 6x8 tile of C into four NEON registers.
// Input:   r - row index 0..5. It must be a literal digit, because it is
//              pasted into the variable names.
//          Also reads, from the enclosing scope, `c` (top-left element of the
//          tile) and `rs` (row stride of c, in doubles).
// Output:  declares c<r>0 .. c<r>3 in the enclosing scope, each holding two
//          adjacent doubles of the row (LOAD_ROW(2) declares c20 .. c23).
#define LOAD_ROW(r)                                                            \
  float64x2_t c##r##0 = vld1q_f64(c + r * rs + 0);                             \
  float64x2_t c##r##1 = vld1q_f64(c + r * rs + 2);                             \
  float64x2_t c##r##2 = vld1q_f64(c + r * rs + 4);                             \
  float64x2_t c##r##3 = vld1q_f64(c + r * rs + 6);

// Purpose: write row r of the tile from its registers back to C.
// Input:   r - row index 0..5, a literal digit.
//          Also reads c<r>0 .. c<r>3, `c` and `rs` from the enclosing scope.
// Output:  overwrites the 8 doubles c[r * rs] .. c[r * rs + 7].
#define STORE_ROW(r)                                                           \
  vst1q_f64(c + r * rs + 0, c##r##0);                                          \
  vst1q_f64(c + r * rs + 2, c##r##1);                                          \
  vst1q_f64(c + r * rs + 4, c##r##2);                                          \
  vst1q_f64(c + r * rs + 6, c##r##3);

// Purpose: add one term to row r of the tile: C(r, 0:8) += a(r) * b(0:8),
//          done as four fused multiply-adds of two doubles each.
// Input:   r    - row index 0..5, a literal digit.
//          vec  - register holding two consecutive entries of the A sliver.
//          lane - 0 or 1: which of the two entries in vec is a(r).
//          Also reads b0 .. b3 from the enclosing scope: the 8 entries of the
//          B sliver for the current term.
// Output:  updates c<r>0 .. c<r>3 in place.
#define FMA_ROW(r, vec, lane)                                                  \
  c##r##0 = vfmaq_laneq_f64(c##r##0, b0, vec, lane);                           \
  c##r##1 = vfmaq_laneq_f64(c##r##1, b1, vec, lane);                           \
  c##r##2 = vfmaq_laneq_f64(c##r##2, b2, vec, lane);                           \
  c##r##3 = vfmaq_laneq_f64(c##r##3, b3, vec, lane);

// Purpose: the micro-kernel. Computes C(0:6, 0:8) += a * b for one full tile.
//          The tile lives in 24 registers for the whole loop (4 more hold b
//          and 3 hold a), so C is read once and written once.
// Input:   kc - number of terms to accumulate.
//          a  - packed A sliver: kc groups of MR doubles, one group per term.
//          b  - packed B sliver: kc groups of NR doubles, one group per term.
//          c  - top-left element of the tile in C. MR rows of NR doubles are
//               read and written.
//          rs - row stride of c: doubles between the starts of two rows.
// Output:  the 6x8 tile at c is updated in place.
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

// Purpose: micro-kernel for a partial tile at the bottom or right edge of a
//          block, where a full 6x8 tile would write outside the valid part of
//          C. Runs the full kernel into a zeroed buffer, then adds only the
//          valid entries to C. The slivers are zero-padded by the packing
//          routines, so the unused entries of the buffer are harmless.
// Input:   mr, nr - valid rows (1..MR) and valid columns (1..NR) of the tile.
//          kc, a, b - as in kernel_6x8.
//          c  - top-left element of the tile in C. Only mr rows of nr doubles
//               are read and written.
//          rs - row stride of c.
// Output:  the mr x nr tile at c is updated in place.
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

// Purpose: copy a rows x kc piece of A into the layout the kernel reads:
//          slivers of MR rows, each stored term by term (the MR entries of
//          column p, then those of column p + 1, ...), so the kernel walks it
//          with stride 1. A last sliver with fewer than MR rows is padded
//          with zeros.
// Input:   rows - rows to pack.
//          kc   - columns (terms) to pack.
//          src  - top-left element of the piece in A, row-major.
//          lda  - row stride of A in doubles (its number of columns).
//          dst  - destination, with room for ceil(rows / MR) * MR * kc doubles.
// Output:  dst is filled. Nothing is returned.
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

// Purpose: copy a kc x cols piece of B into slivers of NR columns, each
//          stored term by term (the NR entries of row p, then those of row
//          p + 1, ...). A last sliver with fewer than NR columns is padded
//          with zeros.
// Input:   kc   - rows (terms) to pack.
//          cols - columns to pack.
//          src  - top-left element of the piece in B, row-major.
//          ldb  - row stride of B in doubles (its number of columns).
//          dst  - destination, with room for kc * ceil(cols / NR) * NR doubles.
// Output:  dst is filled. Nothing is returned.
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

// Purpose: allocate a pack buffer aligned to a cache line.
// Input:   count - number of doubles the buffer must hold.
// Output:  pointer to the buffer, to be released with free(). It is never
//          NULL: the program aborts if the allocation fails.
static double *alloc_pack(const int64_t count) {
  void *p = NULL;
  if (posix_memalign(&p, CACHE_LINE, count * sizeof(double)) != 0) {
    abort();
  }
  return p;
}

// Purpose: decide how many worker threads to use.
// Input:   none. Reads NUM_THREADS and the system's count of active cores.
// Output:  NUM_THREADS if it is positive, otherwise the number of active
//          cores, or 1 if that cannot be queried.
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

// One piece of parallel work. args describes the whole job and is shared by
// every call; idx says which piece this call handles.
typedef void (*task_fn)(const void *args, int64_t idx);

// State of one parallel_for call, shared by all its workers.
typedef struct {
  task_fn fn;           // the task to run
  const void *args;     // passed unchanged to every task
  int64_t count;        // number of tasks: idx runs over 0 .. count - 1
  _Atomic int64_t next; // lowest task index nobody has claimed yet
} parallel_job;

// Purpose: body of one worker thread. Claims the next unclaimed task index
//          and runs it, over and over, until none are left. Tasks come from a
//          shared counter, so slower cores simply take fewer of them.
// Input:   ctx    - the parallel_job being executed.
//          worker - worker number supplied by dispatch_apply_f. Unused.
// Output:  none. Returns once every task has been claimed.
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

// Purpose: run fn(args, idx) for every idx in 0 .. count - 1, spread over
//          worker threads, and wait for all of them.
// Input:   threads - number of workers to start.
//          count   - number of tasks.
//          fn      - the task. It is called from several threads at once, so
//                    different idx values must not write the same memory.
//          args    - passed unchanged to every call of fn.
// Output:  none. Returns once every task has finished.
static void parallel_for(const int64_t threads, const int64_t count,
                         const task_fn fn, const void *const args) {
  parallel_job job = {fn, args, count, 0};
  dispatch_apply_f(threads,
                   dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                   &job, parallel_worker);
}

// Current block, shared read-only by the tasks of one parallel region.
typedef struct {
  int64_t kc, rows, cols;      // block size: terms, rows, columns
  int64_t ld_a, ld_b, ld_c;    // row strides of A, B and C, in doubles
  const double *A_src, *B_src; // top-left element of the block in A and in B
  double *A_pack, *B_pack, *C; // pack buffers; top-left of the block in C
} gemm_block;

// Purpose: parallel task that packs one MR-row sliver of the current A block.
// Input:   args - the gemm_block describing the block.
//          idx  - sliver index: rows idx * MR .. idx * MR + MR - 1.
// Output:  fills that sliver's MR * kc doubles in A_pack. Different idx values
//          write disjoint parts of the buffer.
static void pack_A_task(const void *args, const int64_t idx) {
  const gemm_block *const g = args;
  const int64_t i = idx * MR;
  pack_A(min64(MR, g->rows - i), g->kc, g->A_src + i * g->ld_a, g->ld_a,
         g->A_pack + i * g->kc);
}

// Purpose: parallel task that packs one NR-column sliver of the current B
//          block.
// Input:   args - the gemm_block describing the block.
//          idx  - sliver index: columns idx * NR .. idx * NR + NR - 1.
// Output:  fills that sliver's NR * kc doubles in B_pack. Different idx values
//          write disjoint parts of the buffer.
static void pack_B_task(const void *args, const int64_t idx) {
  const gemm_block *const g = args;
  const int64_t j = idx * NR;
  pack_B(g->kc, min64(NR, g->cols - j), g->B_src + j, g->ld_b,
         g->B_pack + j * g->kc);
}

// Purpose: parallel task that updates one strip of C: the MR rows of sliver
//          idx, across every column of the block. Its A sliver stays in L1
//          while the B slivers stream past, one kernel call per tile.
// Input:   args - the gemm_block describing the block. A_pack and B_pack must
//                 already hold the packed block.
//          idx  - sliver index: rows idx * MR .. idx * MR + MR - 1.
// Output:  adds this block's kc terms of A * B to that strip of C. Different
//          idx values write disjoint rows of C.
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

// Purpose: C += A * B, the entry point declared in matmult.h. Splits the
//          problem into cache-sized blocks. For each block it packs A and B,
//          then runs the micro-kernel over every tile, using all cores.
// Input:   M, N, K - the dimensions. Nothing is done unless all are positive.
//          A - M x K matrix, row-major, read only.
//          B - K x N matrix, row-major, read only.
//          C - M x N matrix, row-major. It must not overlap A or B.
// Output:  C is updated in place: C(i,j) += sum over p of A(i,p) * B(p,j).
void matmult(const int64_t M, const int64_t N, const int64_t K,
             const double *const restrict A, const double *const restrict B,
             double *const restrict C) {
  if (M <= 0 || N <= 0 || K <= 0) {
    return;
  }

  const int64_t rc = block_size(M, RC, MR);
  const int64_t kc = block_size(K, KC, 1);
  const int64_t cc = block_size(N, CC, NR);

  // Packing pads the last sliver, so round the buffers up to whole slivers.
  double *const A_pack = alloc_pack((rc + MR - 1) / MR * MR * kc);
  double *const B_pack = alloc_pack((cc + NR - 1) / NR * NR * kc);

  const int64_t threads = thread_count();
  gemm_block g = {.ld_a = K, .ld_b = N, .ld_c = N,
                  .A_pack = A_pack, .B_pack = B_pack};

  for (int64_t ii = 0; ii < M; ii += rc) {
    g.rows = min64(rc, M - ii);

    for (int64_t pp = 0; pp < K; pp += kc) {
      g.kc = min64(kc, K - pp);
      g.A_src = &A[ii * K + pp];
      parallel_for(threads, (g.rows + MR - 1) / MR, pack_A_task, &g);

      for (int64_t jj = 0; jj < N; jj += cc) {
        g.cols = min64(cc, N - jj);
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
