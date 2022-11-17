#ifndef _LINEAR_H_
#define _LINEAR_H_

#include "../../tensor/tensor.h"
#include "../../tensor/matrix.h"

template <class T>
class Linear_f: public Autograd<T> {
protected:
    Tensor_<T> _forward(Tensor_<T>** input, int ninput, bool is_training=true);
    void _backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren);
};

template <class T>
Tensor_<T> Linear_f<T>::_forward(Tensor_<T>** input, int ninput, bool is_training) {
    assert(ninput == 2 || ninput == 3);

    int batch_ch = input[0]->size()[0];
    int input_ch = input[0]->size()[1];
    int output_ch = input[1]->size()[0];

    int output_size[] = {batch_ch, output_ch};

    Tensor_<T> output(output_size, 2, input[0]->requires_grad);

    for (int i=0; i<output.nelement(); i++) output.data[i] = 0;

    if (ninput == 3) {
        // bias
        for (int i=0; i<output.nelement(); i++) {
            int size[2];
            output.get_index(i, size);
            output.index(i) += input[2]->get(size[1]);
        }
    }

    gemm<T>(CblasRowMajor, CblasNoTrans, CblasTrans, batch_ch, output_ch, input_ch, 
        1., input[0]->data, input_ch, input[1]->data, input_ch, 1., output.data, output_ch);
    
    return output;
}

template <class T>
void Linear_f<T>::_backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren) {
    assert(nchildren == 2 || nchildren == 3);

     if (nchildren == 3) {
        // bias grad
        for (int i=0; i<grad.nelement(); i++) {
            int bias_ind[2];
            grad.get_index(i, bias_ind);
            children[2]->grad[bias_ind[1]] += grad.get(i);
        }
    }
    /*
    for (int i = 0; i<nchildren; i++) {
        if (children[i]->grad == NULL) {
            children[i]->grad = new T[children[i]->nelement()];
        }
    }
    */

    int batch_ch = children[0]->size()[0];
    int input_ch = children[0]->size()[1];
    int output_ch = children[1]->size()[0];

    assert(children[0]->nelement() == batch_ch * input_ch);

    gemm<T>(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_ch, input_ch, output_ch,
        1., grad.data, output_ch, children[1]->data, input_ch, 0., children[0]->grad, input_ch);

    gemm<T>(CblasRowMajor, CblasTrans, CblasNoTrans, output_ch, input_ch, batch_ch,
        1., grad.data, output_ch, children[0]->data, input_ch,  0., children[1]->grad, input_ch);
}

#endif
