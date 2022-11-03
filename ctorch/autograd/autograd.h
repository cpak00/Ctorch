#ifndef _AUTOGRAD_H_
#define _AUTOGRAD_H_

#include "../tensor/tensor.h"

template <class T> class Tensor_;

template <class T>
class Autograd {
public:
    static Tensor_<T> forward(Tensor_<T>* input, int ninput);
    static void backward(Tensor_<T> & grad, Tensor_<T>* children, int nchildren);
};

#endif
