#include "matmult.h"

// Row-major storage: A is MxK, B is KxN, C is MxN.
// C(i,j) += A(i,p) * B(p,j)

void matmult(const int64_t M, const int64_t N, const int64_t K,
             const double *const A, const double *const B, double *const C) {

  // IJP: innermost loop is a dot product of row i of A with column j of B.
  // A is read contiguously, B is read with stride N.
  for (int64_t i = 0; i < M; i++) {
    for (int64_t j = 0; j < N; j++) {
      for (int64_t p = 0; p < K; p++) {
        C[i * N + j] += A[i * K + p] * B[p * N + j];
      }
    }
  }
}
