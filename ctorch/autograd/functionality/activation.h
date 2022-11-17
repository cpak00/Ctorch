#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include "../../tensor/tensor.h"
#include "../../tensor/matrix.h"

template <class T>
class ReLU_f: public Autograd<T> {
    Tensor_<T> mask;

    Tensor_<T> _forward(Tensor_<T>** input, int ninput, bool is_training=true);
    void _backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren);
};

template <class T>
Tensor_<T> ReLU_f<T>::_forward(Tensor_<T>** input, int ninput, bool is_training) {
    assert (ninput == 1);
    mask.zeros_like(input[0]);
    Tensor_<T> output;
    output.clone(input[0]);

    for (int i=0; i<input[0]->nelement(); i++) {
        if (input[0]->get(i) > 0) {
            mask.index(i) = 1.f;
        } else {
            output.index(i) = 0.f;
        }
    }

    return output;
}

template <class T>
void ReLU_f<T>::_backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren) {
    assert (nchildren == 1);

    for (int i=0; i<grad.nelement(); i++) {
        children[0]->grad[i] += grad.get(i) * mask.get(i);
    }

}

#endif
