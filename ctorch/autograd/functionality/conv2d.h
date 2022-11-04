#ifndef _LINEARF_H_
#define _LINEARF_H_

#include "../../tensor/tensor.h"
#include "../../tensor/matrix.h"

template <class T>
class Conv2d_f: public Autograd<T> {
public:
    static Tensor_<T> forward(Tensor_<T>* input, int ninput);
    static void backward(Tensor_<T> & grad, Tensor_<T>* children, int nchildren);
};

template <class T>
Tensor_<T> Conv2d_f<T>::forward(Tensor_<T>* input, int ninput) {
    assert(ninput == 2);


}

template <class T>
void Conv2d_f<T>::backward(Tensor_<T> & grad, Tensor_<T>* children, int nchildren) {
    assert(nchildren == 2);


}

#endif
