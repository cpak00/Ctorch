#ifndef _LINEARF_H_
#define _LINEARF_H_

#include "../../tensor/tensor.h"
#include "../../tensor/matrix.h"

template <class T>
class Linear_f: public Autograd<T> {
public:
    static Tensor_<T> forward(Tensor_<T>* input, int ninput);
    static void backward(Tensor_<T> & grad, Tensor_<T>* children, int nchildren);
};

template <class T>
Tensor_<T> Linear_f<T>::forward(Tensor_<T>* input, int ninput) {
    assert(ninput == 2);

    int batch_ch = input[0].size()[0];
    int input_ch = input[0].size()[1];
    int output_ch = input[1].size()[1];

    int output_size[] = {batch_ch, output_ch};

    Tensor_<T> output(output_size, 2, input[0].requires_grad);

    gemm<T>(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_ch, output_ch, input_ch, 
        1., input[0].data, input_ch, input[1].data, output_ch, 0., output.data, output_ch);

    /* autograd: build the operator stream */
    output.children = input;
    output.nchildren = ninput;
    output.grad_fn = Linear_f<T>::backward;
    /* autograd */
    
    return output;
}

template <class T>
void Linear_f<T>::backward(Tensor_<T> & grad, Tensor_<T>* children, int nchildren) {
    assert(nchildren == 2);

    for (int i = 0; i<nchildren; i++) {
        if (children[i].grad == NULL) {
            children[i].grad = new T[children[i].nelement()];
        }
    }

    int batch_ch = children[0].size()[0];
    int input_ch = children[0].size()[1];
    int output_ch = children[1].size()[1];

    gemm<T>(CblasRowMajor, CblasNoTrans, CblasTrans, batch_ch, input_ch, output_ch,
        1., grad.data, output_ch, children[1].data, output_ch, 0., children[0].grad, input_ch);

    gemm<T>(CblasRowMajor, CblasTrans, CblasNoTrans, input_ch, output_ch, batch_ch,
        1., children[0].data, input_ch, grad.data, output_ch, 0., children[1].grad, output_ch);
}

#endif
