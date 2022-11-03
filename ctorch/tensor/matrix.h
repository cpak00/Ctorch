
#include "cblas.h"


template <typename T>
void gemm(CBLAS_ORDER order, CBLAS_TRANSPOSE Atrans, CBLAS_TRANSPOSE Btrans, int m, int n, int k, double alpha, T* A, int An, T* B, int Bn, double beta, T* C, int Cn) {
    if (std::is_same<T, float>::value) {
        cblas_sgemm(order, Atrans, Btrans, m, n, k, (float) alpha, (float*) A, An, (float*) B, Bn, (float) beta, (float*) C, Cn);
    }
    else if (std::is_same<T, double>::value) {
        cblas_dgemm(order, Atrans, Btrans, m, n, k, (double) alpha, (double*) A, An, (double*) B, Bn, (double) beta, (double*) C, Cn);
    }
    else {
        throw "Only support float and double";
    }
}
