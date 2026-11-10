#include "matmult.h"

// Row-major storage: A is MxK, B is KxN, C is MxN.
// C(i,j) += A(i,p) * B(p,j)

void matmult(const int64_t M, const int64_t N, const int64_t K,
             const double *const A, const double *const B, double *const C) {

  // JPI: innermost loop is an axpy on column j of C:
  //   C(:,j) += B(p,j) * A(:,p)
  // In row-major both C(:,j) and A(:,p) are strided (N and K), so this
  // ordering is the least cache-friendly of the three.
  for (int64_t j = 0; j < N; j++) {
    for (int64_t p = 0; p < K; p++) {
      const double b = B[p * N + j];
      for (int64_t i = 0; i < M; i++) {
        C[i * N + j] += A[i * K + p] * b;
      }
    }
  }
}
