#ifndef _MODULE_H_
#define _MODULE_H_

#include "../tensor/tensor.h"

template <class T>
class Module {
public:
    Tensor_<T> weight;
    Tensor_<T> bias;
    Tensor_<T> & forward(Tensor_<T> & input);
};

#endif
