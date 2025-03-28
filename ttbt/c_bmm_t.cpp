#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include <iostream>
using namespace std;


//typedef enum { DOUBLE, INT } DataType;
typedef struct {
    int64_t      order;         // tensor order (number of modes)
    int64_t*     dimensions;    // tensor dimensions
    void*     vals;
    int     data_type;          // tensor values
} taco_tensor_t;

template <class T>
class Calc_Bmm {
   private:
    taco_tensor_t *A, *B, *C;

   public:
    Calc_Bmm(taco_tensor_t *A_t, taco_tensor_t *B_t, taco_tensor_t *C_t) {
    A = A_t;
    B = B_t;
    C = C_t;
}  // constructor

    int run_bmm() {
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
            ((T *)C_vals)[pC] = 0.0;
    }

#pragma omp parallel for schedule(static) collapse(2)
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
                        ((T *)C_vals)[jC] += ((T *)A_vals)[kA] * ((T *)B_vals)[jB];
                }
            }
        }
    }
    return 0;
    }
};


extern "C" {
int compute(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
    switch (C->data_type) {
        case 0: {
            Calc_Bmm<float> calculator(A, B, C);
            calculator.run_bmm();
            break;
        }
        case 1: {
            Calc_Bmm<double> calculator(A, B, C);
            calculator.run_bmm();
            break;
        }
        case 10: {
            Calc_Bmm<int16_t> calculator(A, B, C);
            calculator.run_bmm();
            break;
        }
        case 11: {
            Calc_Bmm<int32_t> calculator(A, B, C);
            calculator.run_bmm();
            break;
        }
        case 12: {
            Calc_Bmm<int64_t> calculator(A, B, C);
            calculator.run_bmm();
            break;
        }
        default: {
            fprintf(stderr, "Error: Unsupported data type %d\n", C->data_type);
            return -1;
        }
    }
    return 0;
}

}
