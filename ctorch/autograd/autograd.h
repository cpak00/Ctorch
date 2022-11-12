#ifndef _AUTOGRAD_H_
#define _AUTOGRAD_H_

#include "../tensor/tensor.h"

template <class T> class Tensor_;

template <class T>
class Autograd {
public:
    Tensor_<T> forward(Tensor_<T>** input, int ninput);
    void backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren);

protected:
    virtual Tensor_<T> _forward(Tensor_<T>** input, int ninput) = 0;
    virtual void _backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren) = 0;
};

template <class T>
Tensor_<T> Autograd<T>::forward(Tensor_<T>** input, int ninput) {
    Tensor_<T> output = this->_forward(input, ninput);
    
    /* autograd: build the operator stream */
    output.children = new Tensor_<T>*[ninput];
    for (int i=0; i<ninput; i++) output.children[i] = input[i];
    output.nchildren = ninput;
    output.grad_fn = this;
    /* autograd */

    return output;
}

template <class T>
void Autograd<T>::backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren) {
    this->_backward(grad, children, nchildren);
}

#endif
