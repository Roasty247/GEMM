#include "matmult.h"

// Row-major storage: A is MxK, B is KxN, C is MxN.
// C(i,j) += A(i,p) * B(p,j)

// Block size can be overridden at compile time, e.g. -DBS=128
#ifndef BS
#define BS 64
#endif

static inline int64_t min64(const int64_t a, const int64_t b) {
  return a < b ? a : b;
}

void matmult(const int64_t M, const int64_t N, const int64_t K,
             const double *const restrict A, const double *const restrict B, double *const restrict C) {

  // TODO: play with the block size
  const int64_t BLOCK_SIZE = BS;

  // Loop over blocks: C(ii,jj) += A(ii,pp) * B(pp,jj)
  for (int64_t ii = 0; ii < M; ii += BLOCK_SIZE) {
    const int64_t i_end = min64(ii + BLOCK_SIZE, M);

    for (int64_t jj = 0; jj < N; jj += BLOCK_SIZE) {
      const int64_t j_end = min64(jj + BLOCK_SIZE, N);

      for (int64_t pp = 0; pp < K; pp += BLOCK_SIZE) {
        const int64_t p_end = min64(pp + BLOCK_SIZE, K);

        // Block kernel in i-p-j order: the inner j loop streams
        // contiguously through a row of B and a row of C.
        for (int64_t i = ii; i < i_end; i++) {

          for (int64_t p = pp; p < p_end; p++) {
            const double a = A[i * K + p];

            for (int64_t j = jj; j < j_end; j++) {
              C[i * N + j] += a * B[p * N + j];
            }
          }
        }
      }
    }
  }
}
