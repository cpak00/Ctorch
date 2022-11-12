#ifndef _SGD_H
#define _SGD_H

#include "../tensor/tensor.h"
#include "optimizer.h"

template <class T>
class SGD: public Optimizer<T> {
private:
    Tensor_<T>** parameters;
    T lr;
    T momentum;

public:
    SGD(Tensor_<T>** parameters, T lr, T momentum);

    void step();
};

template <class T>
SGD<T>::SGD(Tensor_<T>** parameters, T lr, T momentum): parameters(parameters), lr(lr), momentum(momentum) {}

template <class T>
void SGD<T>::step() {

}

#endif