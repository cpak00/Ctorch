#ifndef _MATRIX_H_
#define _MATRIX_H_


extern"C"
{
    #include<cblas.h>
}


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

template <typename T>
void transpose(T* data, T* dest, int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            dest[i * width + j] = data[j * height + i];
        }
    }
    
}

template <typename T>
void transpose(Tensor_<T> & data, Tensor_<T> & dest, int sel0, int sel1) {

    int* size0 = data.size();
    int* size1 = dest.size();
    int nsize = data.ndim();

    int nelement = 1;

    for (int i=0; i<nsize; i++) {
        if (i == sel0) size1[i] = size0[sel1];
        else if (i == sel1) size1[i] = size0[sel0];
        else size1[i] = size0[i];
        nelement *= size0[i];
    }

    dest.reshape(size1, nsize);

    int* index0 = new int[nsize];
    int* index1 = new int[nsize];

    for (int i=0; i<nelement; i++) {
        int _i = i;
        for (int j=nsize-1; j>=0; j--) {
            index0[j] = _i % size0[j];
            index1[j] = index0[j];
            _i /= size0[j];
        }
        index1[sel0] = index0[sel1];
        index1[sel1] = index0[sel0];

        dest.index(index1) = data.get(index0);
    }

    delete[] index1;
    delete[] index0;

}

#endif
