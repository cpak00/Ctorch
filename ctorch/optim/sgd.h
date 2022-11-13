#ifndef _SGD_H
#define _SGD_H

#include "../tensor/tensor.h"
#include "optimizer.h"

template <class T>
class SGD: public Optimizer<T> {
public:
    T lr;
    T momentum;

    Tensor_<T>* history_grad;

    SGD(Tensor_<T>** parameters, int nparameters, T lr, T momentum);
    ~SGD();

    void step();
};

template <class T>
SGD<T>::SGD(Tensor_<T>** parameters, int nparameters, T lr, T momentum): Optimizer<T>(parameters, nparameters), lr(lr), momentum(momentum) {
    history_grad = new Tensor_<T>[nparameters];
    for (int i=0; i<nparameters; i++) {
        history_grad[i].zeros_like(parameters[i]);
    }
}

template <class T>
SGD<T>::~SGD() {
    delete[] history_grad;
}

template <class T>
void SGD<T>::step() {
    for (int i=0; i<this->nparameters; i++) {
        if (this->parameters[i]->requires_grad) {
            for (int n=0; n<this->parameters[i]->nelement(); n++) {
                this->parameters[i]->data[n] -= this->lr *  this->parameters[i]->grad[n];
            }
        }
    }
}

#endif