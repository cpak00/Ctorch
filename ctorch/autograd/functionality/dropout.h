#ifndef _DROPOUT_H_
#define _DROPOUT_H_

#include "../../tensor/tensor.h"
#include "../../tensor/matrix.h"

template <class T>
class Dropout_f: public Autograd<T> {
protected:
    float p;
    Tensor_<T> mask;
    Tensor_<T> _forward(Tensor_<T>** input, int ninput, bool is_training=true);
    void _backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren);

public:
    Dropout_f(float p): p(p) {}
};

template <class T>
Tensor_<T> Dropout_f<T>::_forward(Tensor_<T>** input, int ninput, bool is_training) {
    assert (ninput == 1);
    Tensor_<T> output;
    output.clone(input[0]);

    if (is_training) {
        mask.zeros_like(input[0]);
        mask.uniform(0, 1);
        for (int i=0; i<input[0]->nelement(); i++) {
            if (mask.get(i) > p) {
                // mask.index(i) = 1.f;
            } else {
                output.index(i) = 0.f;
            }
        }
    }
    
    return output;
}

template <class T>
void Dropout_f<T>::_backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren) {
    assert (nchildren == 1);

    for (int i=0; i<grad.nelement(); i++) {
        if (mask.get(i) > p) {
            children[0]->grad[i] = grad.get(i) / (1 - p);
        } else {
            children[0]->grad[i] = 0.f;
        } 
    }
}

#endif
