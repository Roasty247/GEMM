#include "matmult.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// EDIT HERE with your name and lastname
const char *name = "Ofir";
const char *lastname = "Yaffe";

// NO need to edit anything beyond this line (unless you want to)

double now_in_seconds() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    fprintf(stderr, "Usage: %s <M> <N> <K> <out.csv>\n", argv[0]);
    return EXIT_FAILURE;
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  char *out_csv = argv[4];

  double *A = (double *)malloc(M * K * sizeof(double));
  double *B = (double *)malloc(K * N * sizeof(double));
  double *C = (double *)calloc(M * N, sizeof(double));

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      A[i * K + j] = (double)rand() / RAND_MAX;
    }
  }

  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      B[i * N + j] = (double)rand() / RAND_MAX;
    }
  }

  double *C_expected = (double *)calloc(M * N, sizeof(double));
  double tbaseline;
  { // Baseline implementation
    double start_time = now_in_seconds();
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
          C_expected[i * N + j] += A[i * K + k] * B[k * N + j];
        }
      }
    }

    double end_time = now_in_seconds();
    tbaseline = end_time - start_time;
  }

  double tmatmult;
  { // Student implementation
    double start_time = now_in_seconds();

    matmult(M, N, K, A, B, C);

    double end_time = now_in_seconds();
    tmatmult = end_time - start_time;
  }

  double gflops = 1e-9 * 2 * M * N * K;

  printf("matmult:  %.6f [s], %.6f [GFLOP/s]\n", tmatmult, gflops/tmatmult);
  printf("baseline: %.6f [s], %.6f [GFLOP/s]\n", tbaseline, gflops/tbaseline);
  printf("Speedup: %.2f\n", tbaseline / tmatmult);

  { // Check for correctness
    int valid = 1;
    for (int i = 0; i < M && valid; i++) {
      for (int j = 0; j < N && valid; j++) {
        if (fabs(C[i * N + j] - C_expected[i * N + j]) > 1e-8) {
          printf("Error at position (%d, %d): %f != %f\n", i, j, C[i * N + j],
                 C_expected[i * N + j]);
          valid = 0;
          break;
        }
      }
    }
  }

  FILE *fp = fopen(out_csv, "w");
  fprintf(fp, "name,lastname,variant,M,N,K,seconds,throughput\n");
  fprintf(fp, "%s,%s,%s,%d,%d,%d,%f,%f\n", name, lastname, argv[0], M, N, K,
          tmatmult, gflops/tmatmult);
  fclose(fp);

  free(A);
  free(B);
  free(C);
  free(C_expected);

  return EXIT_SUCCESS;
}