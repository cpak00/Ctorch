#ifndef _RESHAPE_H_
#define _RESHAPE_H_

#include "../../tensor/tensor.h"
#include "../../tensor/matrix.h"

template <class T>
class Reshape_f: public Autograd<T> {
private:
    int nsize_for, nsize_bac;
    int *size_for, *size_bac;
public:
    Reshape_f(int *size_bac, int nsize_bac): size_for(NULL) {
        this->nsize_bac = nsize_bac;
        this->size_bac = new int[nsize_bac];
        for (int i=0; i<nsize_bac; i++) this->size_bac[i] = size_bac[i];
    }

    ~Reshape_f() {
        delete_s(size_for);
        delete_s(size_bac);
    }

    Tensor_<T> _forward(Tensor_<T>** input, int ninput, bool is_training=true);
    void _backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren);
};

template <class T>
Tensor_<T> Reshape_f<T>::_forward(Tensor_<T>** input, int ninput, bool is_training) {
    assert (ninput == 1);

    int* size_new = new int[nsize_bac]; // safely deleted

    for (int i=0; i<nsize_bac; i++) {
        if (size_bac[i] < 0) {
            size_new[i] = input[0]->size()[i];
        }
        else {
            size_new[i] = size_bac[i];
        }
    }

    Tensor_<T> output;
    output.clone(input[0]);

    nsize_for = output.ndim();
    delete_s(size_for);
    size_for = new int[nsize_for]; // safely deleted
    for (int i=0; i<nsize_for; i++) size_for[i] = output.size()[i];

    // output.cutoff(size_new[0]);
    output.reshape(size_new, nsize_bac);

    delete_s(size_new);
    return output;
}

template <class T>
void Reshape_f<T>::_backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren) {
    assert (nchildren == 1);

    for (int i=0; i<grad.nelement(); i++) {
        children[0]->grad[i] += grad.get(i);
    }

    children[0]->reshape(size_for ,nsize_for);

}

#endif
