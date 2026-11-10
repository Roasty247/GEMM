#include "matmult.h"

// Row-major storage: A is MxK, B is KxN, C is MxN.
// C(i,j) += A(i,p) * B(p,j)

void matmult(const int64_t M, const int64_t N, const int64_t K,
             const double *const A, const double *const B, double *const C) {

  // PJI: a sequence of rank-1 updates C += A(:,p) * B(p,:), each done
  // column by column as an axpy C(:,j) += B(p,j) * A(:,p).
  // Like JPI, the inner loop walks A and C with strides K and N.
  for (int64_t p = 0; p < K; p++) {
    for (int64_t j = 0; j < N; j++) {
      const double b = B[p * N + j];
      for (int64_t i = 0; i < M; i++) {
        C[i * N + j] += A[i * K + p] * b;
      }
    }
  }
}
