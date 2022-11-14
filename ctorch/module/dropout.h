#ifndef _DROPOUT_MODULE_H_
#define _DROPOUT_MODULE_H_
#include "module.h"
#include "../autograd/functionality/dropout.h"

template <class T>
class Dropout: public Module<T> {
private:
    Dropout_f<T> dropout_f;
    bool is_bias;
    Tensor_<T> output;

    int ninput;
    Tensor_<T>** inputs;

public:
    Dropout(float p);
    ~Dropout() {
        delete_s(inputs);
    };

    Tensor_<T> & forward(Tensor_<T>& input);
};

template <class T>
Dropout<T>::Dropout(float p): Module<T>(0), dropout_f(p) {
    ninput = 1;
    inputs = new Tensor_<T>*[ninput]; // safely deleted
}

template <class T>
Tensor_<T> & Dropout<T>::forward(Tensor_<T> & x) {
    inputs[0] = &x;

    output = dropout_f.forward(inputs, ninput, this->is_training);
    return output;
}

#endif
