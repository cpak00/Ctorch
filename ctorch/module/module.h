#ifndef _MODULE_H_
#define _MODULE_H_

#include "../tensor/tensor.h"

template <class T>
class Module {
protected:
    Tensor_<T>** _parameters = 0;
    int _nparameters;
    bool is_training;

public:
    Module(int nparam=2);
    virtual ~Module();

    Tensor_<T> weight;
    Tensor_<T> bias;
    virtual Tensor_<T> & forward(Tensor_<T> & input) = 0;

    virtual Tensor_<T>** parameters();
    virtual int nparameters();

    void train() {is_training = true;}
    void eval() {is_training = false;}
};

template <class T>
Module<T>::Module(int nparam) {
    if (nparam > 0) {
        _parameters = new Tensor_<T>*[nparam]; // safely deleted
        _parameters[0] = &(this->weight);
        _parameters[1] = &(this->bias);
    } else {
        _parameters = NULL;
    }
    _nparameters = nparam;
    is_training = true;
}

template <class T>
Module<T>::~Module() {
    for (int i=0; i<_nparameters; i++) {
        _parameters[i] = NULL;
    }
    if (_nparameters > 0)
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
