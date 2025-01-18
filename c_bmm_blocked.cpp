#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#define DTYPE double

extern "C" {
typedef enum { DOUBLE, INT } DataType;
typedef struct {
    int64_t      order;         // tensor order (number of modes)
    int64_t*     dimensions;    // tensor dimensions
    void*     vals;
    DataType     data_type;          // tensor values
} taco_tensor_t;

int compute(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
    // Extract tensor dimensions
    int64_t C1_dimension = (int64_t)(C->dimensions[0]);
    int64_t C2_dimension = (int64_t)(C->dimensions[1]);
    int64_t C3_dimension = (int64_t)(C->dimensions[2]);
    void* C_vals = (void*)(C->vals);
    int64_t A1_dimension = (int64_t)(A->dimensions[0]);
    int64_t A2_dimension = (int64_t)(A->dimensions[1]);
    int64_t A3_dimension = (int64_t)(A->dimensions[2]);
    void* A_vals = (void*)(A->vals);
    int64_t B1_dimension = (int64_t)(B->dimensions[0]);
    int64_t B2_dimension = (int64_t)(B->dimensions[1]);
    int64_t B3_dimension = (int64_t)(B->dimensions[2]);
    void* B_vals = (void*)(B->vals);

    int64_t border = ((C1_dimension * C2_dimension) * C3_dimension);

    // Initialize tensor C in parallel
#pragma omp parallel for schedule(static)
    for (int64_t pC = 0; pC < border; pC++) {
        if (C->data_type == DOUBLE) {
            ((double*)C_vals)[pC] = 0.0;
        } else if (C->data_type == INT) {
            ((int*)C_vals)[pC] = 0;
        }
    }

    // Block sizes for tiling (adjust based on L1/L2 cache sizes)
    const int blockB = 64; // Block size for B1_dimension
    const int blockA = 64; // Block size for A2_dimension
    const int blockC = 64; // Block size for C3_dimension

    // Perform tensor computation with blocking
#pragma omp parallel for collapse(3) schedule(static)
    for (int64_t b = 0; b < B1_dimension; b += blockB) {
        for (int64_t i = 0; i < A2_dimension; i += blockA) {
            for (int64_t j = 0; j < C3_dimension; j += blockC) {
                // Blocked computation
                for (int64_t bb = b; bb < b + blockB && bb < B1_dimension; bb++) {
                    for (int64_t ii = i; ii < i + blockA && ii < A2_dimension; ii++) {
                        int64_t iC = bb * C2_dimension + ii;
                        int64_t iA = bb * A2_dimension + ii;
                        for (int64_t jj = j; jj < j + blockC && jj < C3_dimension; jj++) {
                            int64_t jC = iC * C3_dimension + jj;
                            for (int64_t k = 0; k < B2_dimension; k++) {
                                int64_t kA = iA * A3_dimension + k;
                                int64_t kB = bb * B2_dimension + k;
                                int64_t jB = kB * B3_dimension + jj;

                                // Update tensor C
                                if (C->data_type == DOUBLE && A->data_type == DOUBLE && B->data_type == DOUBLE) {
                                    ((double*)C_vals)[jC] += ((double*)A_vals)[kA] * ((double*)B_vals)[jB];
                                } else if (C->data_type == INT && A->data_type == INT && B->data_type == INT) {
                                    ((int*)C_vals)[jC] += ((int*)A_vals)[kA] * ((int*)B_vals)[jB];
                                } else {
                                    fprintf(stderr, "Error: Data type mismatch.\n");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}
}
