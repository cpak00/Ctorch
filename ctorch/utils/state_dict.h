#ifndef _STATEDICT_H
#define _STATEDICT_H
#include "../tensor/tensor.h"
#include <iostream>


template <typename T>
int save(Tensor_<T>** parameters, int nparameter, const char* filepath) {
    FILE* fp;

    fp = fopen(filepath, "wb");
    if (fp == NULL) {
        perror("file open: ");  
        return -1;
    }

    int end_symbol = 0;

    // write the nparameer
    fwrite(&nparameter, sizeof(int), 1, fp);

    // write the parameters
    for (int i=0; i<nparameter; i++) {
        Tensor_<T>* tensor = parameters[i];
        // write the size of tensor
        int tensor_size = tensor->nelement();
        fwrite(&tensor_size, sizeof(int), 1, fp);
        // write the data
        for (int j=0; j<tensor->nelement(); j++) {
            T value = tensor->get(j);
            fwrite(&value, sizeof(T), 1, fp);
        }
        // write the end symbol
        fwrite(&end_symbol, sizeof(int), 1, fp);
    }
    fwrite(&end_symbol, sizeof(int), 1, fp);

    fclose(fp);

    return 0;
}

template <typename T>
int load(Tensor_<T>** parameters, int nparameter, const char* filepath) {
    FILE* fp;

    fp = fopen(filepath, "rb");
    if (fp == NULL) {
        perror("file open: ");  
        return -1;
    }

    int end_symbol;

    // write the nparameer
    int read_nparameter;
    fread(&read_nparameter, sizeof(int), 1, fp);
    if (read_nparameter != nparameter) {
        printf("file format error");
        return -2;
    }

    // write the parameters
    for (int i=0; i<nparameter; i++) {
        Tensor_<T>* tensor = parameters[i];
        // write the size of tensor
        int tensor_size;
        fread(&tensor_size, sizeof(int), 1, fp);
        if (tensor_size != tensor->nelement()) {
            printf("file format error");
            return -2;
        }
        // write the data
        for (int j=0; j<tensor->nelement(); j++) {
            T value;
            fread(&value, sizeof(T), 1, fp);
            tensor->index(j) = value;
        }
        // write the end symbol
        
        fread(&end_symbol, sizeof(int), 1, fp);
        if (end_symbol != 0) {
            printf("file format error\n");
            return -2;
        }
    }
    fread(&end_symbol, sizeof(int), 1, fp);
    if (end_symbol != 0) {
        printf("file format error\n");
        return -2;
    }

    fclose(fp);

    return 0;
}

#endif
