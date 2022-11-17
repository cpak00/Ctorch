#ifndef _ADD_H_
#define _ADD_H_

#include "../../tensor/tensor.h"
#include "../../tensor/matrix.h"

template <class T>
class Add_f: public Autograd<T> {
    Tensor_<T> _forward(Tensor_<T>** input, int ninput, bool is_training=true);
    void _backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren);
};

template <class T>
Tensor_<T> Add_f<T>::_forward(Tensor_<T>** input, int ninput, bool is_training) {
    assert (ninput > 1);

    Tensor_<T> output;
    output.clone(input[0]);

    for (int i=1; i<ninput; i++) {
        for (int n=0; n<input[0]->nelement(); n++) {
            output.data[n] += input[i]->data[n];
        }
    }

    return output;
}

template <class T>
void Add_f<T>::_backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren) {
    assert (nchildren > 1);

    for (int i=0; i<nchildren; i++) {
        for (int n=0; n<grad.nelement(); n++) {
            children[i]->grad[n] += grad.data[n];
        }
    }

}

#endif
