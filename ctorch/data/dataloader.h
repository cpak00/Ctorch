#ifndef _DATALOADER_H
#define _DATALOADER_H
#include "../tensor/tensor.h"
#include <opencv2/opencv.hpp>
#include "dataset.h"

template <class T>
class DataLoader {
private:
    Dataset<T>* dataset;
    int batch_size;
    int* size;
    int nsize;

public:
    DataLoader(Dataset<T>& dataset, int batch_size);
    ~DataLoader() {};

    bool next(Tensor_<T> & data, Tensor_<T> & label);
    void reset() {
        dataset->reset();
    }
};

template<class T>
DataLoader<T>::DataLoader(Dataset<T>& dataset, int batch_size) {
    this->dataset = &dataset;
    this->batch_size = batch_size;
    this->size = dataset._size();
    this->nsize = dataset._nsize();
}

template<class T>
bool DataLoader<T>::next(Tensor_<T> & data, Tensor_<T> & label) {

    int* tensor_size = new int[nsize + 1]; // safely deleted
    tensor_size[0] = batch_size;
    for (int i=1; i<=nsize; i++) {
        tensor_size[i] = size[i-1];
    }
    data = Tensor_<T>(tensor_size, nsize + 1);

    int label_size[] = {batch_size};
    label = Tensor_<T>(label_size, 1);

    bool is_next = true;
    int ind = 0;
    while (is_next && ind < this->batch_size) {
        Tensor_<T> file;
        Tensor_<T> clazz;
        is_next = dataset->next(file, clazz);

        for (int i=0; i<file.nelement(); i++) {
            data.index(ind * file.nelement() + i) = file.get(i);
        }

        for (int i=0; i<clazz.nelement(); i++) {
            label.index(ind * clazz.nelement() + i) = clazz.get(i);
        }

        ind ++;
    }
    
    if (ind < this->batch_size) {
        data.cutoff(ind);
        label.cutoff(ind);
    }

    delete_s(tensor_size);

    return is_next;
}

#endif
