#include "batch.h"
#include <cassert>
#include <cstddef>
#include <algorithm>


#define BLOCK_M 32
#define BLOCK_N 32
#define BLOCK_K 128

void batchMatrixMultiplication(
    const double* A,
    const double* B,
    double* C,
    int batchCount, // Anzahl Matrizen in Batch
    int M, // Anzahl der Zeilen in A und C
    int K, // Anzahl der Spalten in A und Zeilen in B
    int N  // Anzahl der Spalten in B und C
)
{
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int b = 0; b < batchCount; b++) {
        for (int iBlock = 0; iBlock < M; iBlock += BLOCK_M) {
            for (int kBlock = 0; kBlock < K; kBlock += BLOCK_K) {
                for (int jBlock = 0; jBlock < N; jBlock += BLOCK_N) {
                    // Iterate over the submatrix (tile)
                    int max_i = std::min(iBlock + BLOCK_M, M);
                    for (int i = iBlock; i < max_i; i++) {
                        int iC = b * M + i;
                        int iA = b * M + i;
                        
                        int max_j = std::min(kBlock + BLOCK_K, K);
                        for (int j = kBlock; j < max_j; j++) {
                            int jA = iA * K + j;
                            int jB = b * K + j;
                            
                            int max_k = std::min(jBlock + BLOCK_N, N);
                            for (int k = jBlock; k < max_k; k++) {
                                int kC = iC * N + k;
                                int kB = jB * N + k;
                                C[kC] += A[jA] * B[kB];
                            }
                        }
                    }
                }
            }
        }
    }
}
