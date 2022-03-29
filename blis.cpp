#include <cblas.h>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <unistd.h>

#include <iostream>

#define MATRIX_FORMAT CblasRowMajor

#define MDIM 1024
#define NDIM 1024
#define KDIM 1024

#ifdef FLOAT32
  #define BLAS_GEMM cblas_sgemm
  #define DTYPE float
#else
  #define BLAS_GEMM cblas_dgemm
  #define DTYPE double
#endif

void init_matrix(DTYPE *a, int nrows, int ncols) {
  for (int j = 0; j < ncols; j++) {
    for (int i = 0; i < nrows; i++) {
      a[i + j * nrows] = ((DTYPE) rand() / (DTYPE) RAND_MAX);
    }
  }
}

void naive_matmul(const DTYPE *a, const DTYPE *b, DTYPE *c, size_t m, size_t k, size_t n) {
  // correctness check
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      size_t ci = i*n + j;
      c[ci] = 0.0f;
      for (size_t p = 0; p < k; p++) {
        c[ci] += a[i*k + p] * b[p*n + j];
      }
    }
  }
}

static void BenchmarkFunction() {
  DTYPE *A, *B, *C;

  A = (DTYPE *) malloc(MDIM * KDIM * sizeof(DTYPE));
  B = (DTYPE *) malloc(KDIM * NDIM * sizeof(DTYPE));
  C = (DTYPE *) malloc(MDIM * NDIM * sizeof(DTYPE));

  init_matrix(A, MDIM, KDIM);
  init_matrix(B, KDIM, NDIM);
  init_matrix(C, MDIM, NDIM);

  int LDA = KDIM;
  int LDB = NDIM;
  int LDC = NDIM;
  DTYPE alpha = 1.0;
  DTYPE beta = 0.0;

  BLAS_GEMM(MATRIX_FORMAT, CblasNoTrans, CblasNoTrans, MDIM, NDIM, KDIM, alpha,
                A, LDA, B, LDB, beta, C, LDC);

  struct timespec time_start={0, 0},time_end={0, 0};
  clock_gettime(CLOCK_REALTIME, &time_start);
  for (int i = 0; i < 500; i++) {
    BLAS_GEMM(MATRIX_FORMAT, CblasNoTrans, CblasNoTrans, MDIM, NDIM, KDIM, alpha,
                A, LDA, B, LDB, beta, C, LDC);
  }
  clock_gettime(CLOCK_REALTIME, &time_end);
  std::cout << "BLIS duration: " << ((time_end.tv_sec-time_start.tv_sec) * 1000000000.0 + time_end.tv_nsec-time_start.tv_nsec)/500.0/1000000000.0 << std::endl;

  DTYPE *C2 = (DTYPE *) malloc(MDIM * NDIM * sizeof(DTYPE));
  size_t errors = 0;
  naive_matmul(A,B,C2,MDIM,KDIM,NDIM);
  for (size_t i = 0; i < MDIM; i++) {
    for (size_t j = 0; j < NDIM; j++) {
      size_t ci = i + j*MDIM;
      if (std::abs(C[ci] - C2[ci]) > 0.01f) {
        fprintf(stderr, "Incorrect result at index %ld,%ld: C=%0.2f C2=%0.2f\n", i, j, C[ci], C2[ci]);
        errors++;
      }
    }
  }
  printf("Detected %ld errors.\n", errors);

  free(A);
  free(B);
  free(C);
}

int main(int argc, char **argv) {
  BenchmarkFunction();
  return 0;
}
