#ifndef _POOLING_MODULE_H_
#define _POOLING_MODULE_H_
#include "module.h"
#include "../autograd/functionality/pooling.h"

template <class T>
class MaxPooling: public Module<T> {
private:
    MaxPooling_f<T> maxpooling_f;
    bool is_bias;
    Tensor_<T> output;

    int ninput;
    Tensor_<T>** inputs;

public:
    MaxPooling(int window);
    ~MaxPooling() {
        delete_s(inputs);
    };

    Tensor_<T> & forward(Tensor_<T>& input);
};

template <class T>
MaxPooling<T>::MaxPooling(int window): Module<T>(0), maxpooling_f(window) {
    ninput = 1;
    inputs = new Tensor_<T>*[ninput]; // safely deleted
}

template <class T>
Tensor_<T> & MaxPooling<T>::forward(Tensor_<T> & x) {
    inputs[0] = &x;

    output = maxpooling_f.forward(inputs, ninput);
    return output;
}

#endif
