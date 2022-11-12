#ifndef _CONV2D_MODULE_H_
#define _CONV2D_MODULE_H_
#include "module.h"
#include "../autograd/functionality/conv2d.h"

template <class T>
class Conv2d: public Module<T> {
private:
    Conv2d_f<T> conv2d_f;
    bool is_bias;
    Tensor_<T> output;

    int ninput;
    Tensor_<T>** inputs;

public:
    Conv2d(int input_channel, int output_channel, int kernal_size, int stride=1, int padding=1, bool is_bias=true);
    ~Conv2d() {
        delete[] inputs;
    };

    Tensor_<T> & forward(Tensor_<T>& input);
};

template <class T>
Conv2d<T>::Conv2d(int input_channel, int output_channel, int kernal_size, int stride, int padding, bool is_bias): conv2d_f(kernal_size, stride, padding) {
    int weight_size[] = {output_channel, input_channel, kernal_size, kernal_size};
    this->weight = Tensor_<T>(weight_size, 4);
    this->is_bias = is_bias;

    ninput = (is_bias)? 3: 2;
    inputs = new Tensor_<T>*[ninput];
    if (is_bias) {
        int bias_size[] = {output_channel};
        this->bias = Tensor_<T>(bias_size, 1);
    }
}

template <class T>
Tensor_<T> & Conv2d<T>::forward(Tensor_<T>& x) {
    inputs[0] = &x;
    inputs[1] = &(this->weight);
    if (is_bias) {
        inputs[2] = &(this->bias);
    }

    output = conv2d_f.forward(inputs, ninput);
    return output;
}

#endif
