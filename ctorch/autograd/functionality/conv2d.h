#ifndef _CONV2D_H_
#define _CONV2D_H_

#include "../../tensor/tensor.h"
#include "../../tensor/matrix.h"
#include "../../utils/im2col.h"

template <class T>
class Conv2d_f: public Autograd<T> {
private:
    int kern_size;
    int stride;
    int padding;

    int expand_size;
    int output_ch;
    int act_size;

    Tensor_<T> input_col;

public:
    Conv2d_f(int kern_size, int stride, int padding);
    Tensor_<T> _forward(Tensor_<T>* input, int ninput);
    void _backward(Tensor_<T> & grad, Tensor_<T>* children, int nchildren);
};

template<class T>
Conv2d_f<T>::Conv2d_f(int kern_size, int stride, int padding):kern_size(kern_size), stride(stride), padding(padding) {};

template <class T>
Tensor_<T> Conv2d_f<T>::_forward(Tensor_<T>* input, int ninput) {
    assert(ninput == 2);
    assert(input[0].ndim() == 4 && input[1].ndim() == 4);

    input_col = im2col(input[0], this->kern_size, this->stride, this->padding);
    Tensor_<T> weights = input[1];

    output_ch = input[1].size()[0];
    int input_ch = input[1].size()[1];
    int k_size = input[1].size()[2];

    int batch_size = input_col.size()[0];
    expand_size = input_col.size()[3];
    act_size = input_col.size()[0] * input_col.size()[1] * input_col.size()[2];

    int output_size[] = {batch_size, output_ch, input_col.size()[1], input_col.size()[2]};
    Tensor_<T> output(output_size, 4, input[0].requires_grad);

    Tensor_<T> output_t;
    output_t.zeros_like(output);
    
    gemm<T>(CblasRowMajor, CblasNoTrans, CblasTrans, output_ch, act_size, expand_size, 1., weights.data, expand_size, input_col.data, expand_size, 0., output_t.data, act_size);

    transpose(output_t, output, 0, 1);

    output_t.clear();

    return output;
}   

template <class T>
void Conv2d_f<T>::_backward(Tensor_<T> & grad, Tensor_<T>* children, int nchildren) {
    assert(nchildren == 2);

    for (int i = 0; i<nchildren; i++) {
        if (children[i].grad == NULL) {
            children[i].grad = new T[children[i].nelement()];
        }
    }

    gemm<float>(CblasRowMajor, CblasTrans, CblasNoTrans, output_ch, expand_size, act_size, 1., grad.data, output_ch, input_col.data, expand_size, 0., children[1].grad, expand_size);

}

#endif
