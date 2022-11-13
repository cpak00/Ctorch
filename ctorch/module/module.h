#ifndef _MODULE_H_
#define _MODULE_H_

#include "../tensor/tensor.h"

template <class T>
class Module {
private:
    Tensor_<T>** _parameters;
    int _nparameters;

public:
    Module();
    ~Module();

    Tensor_<T> weight;
    Tensor_<T> bias;
    virtual Tensor_<T> & forward(Tensor_<T> & input) = 0;

    virtual Tensor_<T>** parameters();
    virtual int nparameters();
};

template <class T>
Module<T>::Module() {
    _parameters = new Tensor_<T>*[2]; // safely deleted
    _parameters[0] = &(this->weight);
    _parameters[1] = &(this->bias);
    _nparameters = 2;
}

template <class T>
Module<T>::~Module() {
    delete_s(_parameters);
}

template <class T>
Tensor_<T>** Module<T>::parameters() {
    return _parameters;
}

template <class T>
int Module<T>::nparameters() {
    return _nparameters;
}

#endif
