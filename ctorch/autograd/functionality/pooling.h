#ifndef _POOLING_H_
#define _POOLING_H_

#include "../../tensor/tensor.h"
#include "../../tensor/matrix.h"

template <class T>
class MaxPooling_f: public Autograd<T> {
protected:
    int window;
    Tensor_<T> mask;

    Tensor_<T> _forward(Tensor_<T>** input, int ninput);
    void _backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren);
public:
    MaxPooling_f(int window): window(window) {};
};

template <class T>
Tensor_<T> MaxPooling_f<T>::_forward(Tensor_<T>** input, int ninput) {
    assert (ninput == 1);
    mask.zeros_like(input[0]);

    int* new_size = new int[input[0]->ndim()];
    for (int i=0; i<input[0]->ndim(); i++) {
        if (i >= input[0]->ndim()-2) {
            new_size[i] = (input[0]->size()[i] + 1) / 2;
        }
        else {
            new_size[i] = input[0]->size()[i];
        }
    }

    Tensor_<T> output(new_size, input[0]->ndim());

    int* index = new int[input[0]->ndim()];
    for (int n=0; n<output.nelement(); n++) {
        output.get_index(n, index);
        int max = input[0]->index(n), max_i = -1, max_j = -1;

        int* in_index = new int[input[0]->ndim()];
        for (int i=0; i<input[0]->ndim(); i++) in_index[i] = index[i];
        for (int i=0; i<this->window; i++) {
            for (int j=0; j<this->window; j++) {
                in_index[input[0]->ndim()-1] = index[input[0]->ndim()-1] * window + i;
                in_index[input[0]->ndim()-2] = index[input[0]->ndim()-2] * window + j;
                if (max < input[0]->index(in_index)) {
                    max = input[0]->index(in_index);
                    max_i = i; max_j = j;
                }
            }
        }
        in_index[input[0]->ndim()-1] = index[input[0]->ndim()-1] * window + max_i;
        in_index[input[0]->ndim()-2] = index[input[0]->ndim()-2] * window + max_j;
        output.index(n) = max;
        mask.index(in_index) = 1;
        delete[] in_index;
    }

    delete[] index;
    delete[] new_size;
    return output;
}

template <class T>
void MaxPooling_f<T>::_backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren) {
    assert (nchildren == 1);

    int* index = new int[children[0]->ndim()];
    for (int n=0; n<grad.nelement(); n++) {
        grad.get_index(n, index);
        int max = children[0]->index(n), max_i = -1, max_j = -1;

        int* in_index = new int[children[0]->ndim()];
        for (int i=0; i<children[0]->ndim(); i++) in_index[i] = index[i];
        for (int i=0; i<this->window; i++) {
            for (int j=0; j<this->window; j++) {
                in_index[children[0]->ndim()-1] = index[children[0]->ndim()-1] * window + i;
                in_index[children[0]->ndim()-2] = index[children[0]->ndim()-2] * window + j;
                int sel = children[0]->get_index(in_index);
                children[0]->grad[sel] = mask.get(in_index) * grad.get(n);
            }
        }
        delete[] in_index;
    }
    delete[] index;
}

#endif
