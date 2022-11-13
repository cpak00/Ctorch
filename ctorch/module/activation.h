#ifndef _RELU_MODULE_H_
#define _RELU_MODULE_H_
#include "module.h"
#include "../autograd/functionality/activation.h"

template <class T>
class ReLU: public Module<T> {
private:
    ReLU_f<T> relu_f;
    bool is_bias;
    Tensor_<T> output;

    int ninput;
    Tensor_<T>** inputs;

public:
    ReLU();
    ~ReLU() {
        delete_s(inputs);
    };

    Tensor_<T> & forward(Tensor_<T>& input);
};

template <class T>
ReLU<T>::ReLU() {
    ninput = 1;
    inputs = new Tensor_<T>*[ninput]; // safely deleted
}

template <class T>
Tensor_<T> & ReLU<T>::forward(Tensor_<T> & x) {
    inputs[0] = &x;

    output = relu_f.forward(inputs, ninput);
    return output;
}

#endif
