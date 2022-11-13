#ifndef _RESHAPE_MODULE_H_
#define _RESHAPE_MODULE_H_
#include "module.h"
#include "../autograd/functionality/reshape.h"

template <class T>
class Reshape: public Module<T> {
private:
    Reshape_f<T> reshape_f;
    bool is_bias;
    Tensor_<T> output;

    int ninput;
    Tensor_<T>** inputs;

public:
    Reshape(int* size, int nsize);
    ~Reshape() {
        delete_s(inputs);
    };

    Tensor_<T> & forward(Tensor_<T>& input);
};

template <class T>
Reshape<T>::Reshape(int* size, int nsize): reshape_f(size, nsize) {
    ninput = 1;
    inputs = new Tensor_<T>*[ninput]; // safely deleted
}

template <class T>
Tensor_<T> & Reshape<T>::forward(Tensor_<T> & x) {
    inputs[0] = &x;

    output = reshape_f.forward(inputs, ninput);
    return output;
}

#endif
