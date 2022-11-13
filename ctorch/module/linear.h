#ifndef _LINEAR_MODULE_H_
#define _LINEAR_MODULE_H_
#include "module.h"
#include "../autograd/functionality/linear.h"

template <class T>
class Linear: public Module<T> {
private:
    Linear_f<T> linear_f;
    bool is_bias;
    Tensor_<T> output;

    int ninput;
    Tensor_<T>** inputs;

public:
    Linear(int input_channel, int output_channel, bool is_bias=true);
    ~Linear() {
        delete_s(inputs);
    };

    Tensor_<T> & forward(Tensor_<T>& input);
};

template <class T>
Linear<T>::Linear(int input_channel, int output_channel, bool is_bias) {
    int weight_size[] = {output_channel, input_channel};
    this->weight = Tensor_<T>(weight_size, 2);
    this->is_bias = is_bias;

    ninput = (is_bias)? 3: 2;
    inputs = new Tensor_<T>*[ninput]; // safely deleted
    if (is_bias) {
        int bias_size[] = {output_channel};
        this->bias = Tensor_<T>(bias_size, 1);
    }
}

template <class T>
Tensor_<T> & Linear<T>::forward(Tensor_<T> & x) {
    inputs[0] = &x;
    inputs[1] = &(this->weight);
    if (is_bias) {
        inputs[2] = &(this->bias);
    }

    output = linear_f.forward(inputs, ninput);
    return output;
}

#endif
