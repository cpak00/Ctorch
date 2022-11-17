#ifndef _SGD_H
#define _SGD_H

#include "../tensor/tensor.h"
#include "optimizer.h"

template <class T>
class SGD: public Optimizer<T> {
public:
    T lr;
    T momentum;

    Tensor_<T>* history_grad = NULL;

    SGD(Tensor_<T>** parameters, int nparameters, T lr, T momentum);
    ~SGD();

    void step();
};

template <class T>
SGD<T>::SGD(Tensor_<T>** parameters, int nparameters, T lr, T momentum): Optimizer<T>(parameters, nparameters), lr(lr), momentum(momentum) {
    history_grad = new Tensor_<T>[nparameters]; // safely deleted
}

template <class T>
SGD<T>::~SGD() {
    delete_s(this->history_grad);
}

template <class T>
void SGD<T>::step() {
    for (int i=0; i<this->nparameters; i++) {
        if (this->parameters[i]->requires_grad) {

            if (history_grad[i].ndim() == 0) {
                history_grad[i].zeros_like(this->parameters[i]);
            }

            for (int n=0; n<this->parameters[i]->nelement(); n++) {
                history_grad[i].index(n) = momentum * history_grad[i].get(n) + (1 - momentum) * this->parameters[i]->grad[n];
                this->parameters[i]->data[n] -= this->lr * history_grad[i].get(n); // this->parameters[i]->grad[n];
            }
        }
    }
}

#endif