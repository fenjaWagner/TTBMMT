#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#define DTYPE double

typedef enum { DOUBLE, INT } DataType;
typedef struct {
    int64_t      order;         // tensor order (number of modes)
    int64_t*     dimensions;    // tensor dimensions
    void*     vals;
    DataType     data_type;          // tensor values
} taco_tensor_t;

int compute(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
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
#pragma omp parallel for schedule(static)
    for (int64_t pC = 0; pC < border ; pC++) {
        if (C->data_type == DOUBLE) {
            ((double*)C_vals)[pC] = 0.0;
        } else if (C->data_type == INT) {
            ((int*)C_vals)[pC] = 0;
        }
    }

#pragma omp parallel for schedule(runtime)
    for (int64_t b = 0; b < B1_dimension; b++) {
        for (int64_t i = 0; i < A2_dimension; i++) {
            int64_t iC = b * C2_dimension + i;
            int64_t iA = b * A2_dimension + i;
            for (int64_t k = 0; k < B2_dimension; k++) {
                int64_t kA = iA * A3_dimension + k;
                int64_t kB = b * B2_dimension + k;
                for (int64_t j = 0; j < B3_dimension; j++) {
                    int64_t jC = iC * C3_dimension + j;
                    int64_t jB = kB * B3_dimension + j;
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
    return 0;
}

// Helper function to allocate a tensor
taco_tensor_t* create_tensor(int order, int64_t* dimensions, DataType data_type) {
  taco_tensor_t* tensor = (taco_tensor_t*)malloc(sizeof(taco_tensor_t));
  tensor->order = order;
  tensor->dimensions = (int64_t*)malloc(order * sizeof(int64_t));
  memcpy(tensor->dimensions, dimensions, order * sizeof(int64_t));

  int64_t total_size = 1;
  for (int i = 0; i < order; i++) {
    total_size *= dimensions[i];
  }
  if (data_type == DOUBLE) {
        tensor->vals = (void*)calloc(total_size, sizeof(double));
    } else if (data_type == INT) {
        tensor->vals = (void*)calloc(total_size, sizeof(int));
    } else {
        fprintf(stderr, "Error: Unsupported data type.\n");
        free(tensor->dimensions);
        free(tensor);
        return NULL;
    }
    return tensor;
}

// Helper function to free a tensor
void free_tensor(taco_tensor_t* tensor) {
  free(tensor->dimensions);
  free(tensor->vals);
  free(tensor);
}

// Helper function to print a 3D tensor (flattened as 1D array)
void print_tensor(taco_tensor_t* tensor) {
    if (tensor->order != 3) {
        fprintf(stderr, "Error: print_tensor only supports 3D tensors.\n");
        return;
    }

    int d1 = tensor->dimensions[0];
    int d2 = tensor->dimensions[1];
    int d3 = tensor->dimensions[2];

    if (tensor->data_type == DOUBLE) {
        double* vals = (double*)tensor->vals;
        for (int i = 0; i < d1; i++) {
            printf("Slice %d:\n", i);
            for (int j = 0; j < d2; j++) {
                for (int k = 0; k < d3; k++) {
                    printf("%6.2f ", vals[i * d2 * d3 + j * d3 + k]);
                }
                printf("\n");
            }
            printf("\n");
        }
    } else if (tensor->data_type == INT) {
        int* vals = (int*)tensor->vals;
        for (int i = 0; i < d1; i++) {
            printf("Slice %d:\n", i);
            for (int j = 0; j < d2; j++) {
                for (int k = 0; k < d3; k++) {
                    printf("%6d ", vals[i * d2 * d3 + j * d3 + k]);
                }
                printf("\n");
            }
            printf("\n");
        }
    } else {
        fprintf(stderr, "Error: Unsupported data type in print_tensor.\n");
    }
}


int main() {
  // Tensor dimensions: C(b, i, j) = A(b, i, k) * B(b, k, j)
  int64_t C_dimensions[] = {2, 3, 4}; // Dimensions for tensor C
  int64_t A_dimensions[] = {2, 3, 5}; // Dimensions for tensor A
  int64_t B_dimensions[] = {2, 5, 4}; // Dimensions for tensor B

  // Create tensors
  taco_tensor_t* C = create_tensor(3, C_dimensions, DOUBLE);
  taco_tensor_t* A = create_tensor(3, A_dimensions, DOUBLE);
  taco_tensor_t* B = create_tensor(3, B_dimensions, DOUBLE);

  // Initialize A and B with some values
  double* A_vals = (double*)A->vals;
  double* B_vals = (double*)B->vals;

  // Fill A with values: A(b, i, k) = b + i + k
  for (int64_t b = 0; b < A_dimensions[0]; b++) {
    for (int64_t i = 0; i < A_dimensions[1]; i++) {
      for (int64_t k = 0; k < A_dimensions[2]; k++) {
        A_vals[b * A_dimensions[1] * A_dimensions[2] + i * A_dimensions[2] + k] = 1+b;
      }
    }
  }

  // Fill B with values: B(b, k, j) = b * k * j
  for (int64_t b = 0; b < B_dimensions[0]; b++) {
    for (int64_t k = 0; k < B_dimensions[1]; k++) {
      for (int64_t j = 0; j < B_dimensions[2]; j++) {
        B_vals[b * B_dimensions[1] * B_dimensions[2] + k * B_dimensions[2] + j] = 1;
      }
    }
  }

  // Print tensors A and B
  printf("Tensor A:\n");
  print_tensor(A);
  printf("Tensor B:\n");
  print_tensor(B);

  // Call the compute function
  compute(C, A, B);

  // Print the result tensor C
  printf("Result Tensor C:\n");
  print_tensor(C);

  // Free tensors
  free_tensor(C);
  free_tensor(A);
  free_tensor(B);

  return 0;
}