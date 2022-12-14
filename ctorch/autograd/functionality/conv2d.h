#ifndef _CONV2D_H_
#define _CONV2D_H_

#include "../../tensor/tensor.h"
#include "../../tensor/matrix.h"
#include "../../utils/im2col.h"

template <class T>
class Conv2d_f: public Autograd<T> {
protected:
    int kern_size;
    int stride;
    int padding;

    int batch_size;
    int expand_size;
    int output_ch;
    int act_size;

    int height_col;
    int width_col;

    Tensor_<T> input_col;

    Tensor_<T> _forward(Tensor_<T>** input, int ninput, bool is_training=true);
    void _backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren);

public:
    Conv2d_f(int kern_size, int stride, int padding);
};

template<class T>
Conv2d_f<T>::Conv2d_f(int kern_size, int stride, int padding): kern_size(kern_size), stride(stride), padding(padding) {};

template <class T>
Tensor_<T> Conv2d_f<T>::_forward(Tensor_<T>** input, int ninput, bool is_training) {
    assert(ninput == 2 || ninput == 3);
    assert(input[0]->ndim() == 4 && input[1]->ndim() == 4);

    // image to column
    input_col = im2col(input[0], this->kern_size, this->stride, this->padding);
    Tensor_<T>* weights = input[1];

    output_ch = input[1]->size()[0];
    int input_ch = input[1]->size()[1];
    int k_size = input[1]->size()[2];

    batch_size = input_col.size()[0];
    expand_size = input_col.size()[3];
    act_size = input_col.size()[1] * input_col.size()[2];

    int act_h = input[0]->size()[2];
    int act_w = input[0]->size()[3];

    // calculate the size of output
    height_col = (act_h + 2*padding - k_size) / stride + 1;
    width_col = (act_w + 2*padding - k_size) / stride + 1;

    int output_t_size[] = {output_ch, batch_size, input_col.size()[1], input_col.size()[2]};
    Tensor_<T> output_t(output_t_size, 4, input[0]->requires_grad);

    for (int i=0; i<output_t.nelement(); i++) output_t.index(i) = 0;

    if (ninput == 3) {
        // bias
        for (int i=0; i<output_t.nelement(); i++) {
            output_t.index(i) += input[2]->get(i / (batch_size * input_col.size()[1] * input_col.size()[2]));
        }
    }

    // matrix multiplication
    gemm<T>(CblasRowMajor, CblasNoTrans, CblasTrans, output_ch, batch_size * act_size, expand_size, 1., weights->data, expand_size, input_col.data, expand_size, 1., output_t.data, batch_size * act_size);

    Tensor_<T> output;
    output.zeros_like(output_t);
    transpose(output_t, output, 0, 1);

    return output;
}   

template <class T>
void Conv2d_f<T>::_backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren) {
    assert(nchildren == 2 || nchildren == 3);

    if (nchildren == 3) {
        // bias grad
        for (int i=0; i<grad.nelement(); i++) {
            int bias_ind[4];
            grad.get_index(i, bias_ind);
            children[2]->grad[bias_ind[1]] += grad.get(i);
        }
    }

    Tensor_<T> grad_tran;
    grad_tran.zeros_like(grad);
    int trans_grad[] = {1, 2, 3, 0};
    transpose(grad, grad_tran, trans_grad, 4);
    
    for (int i = 0; i<nchildren; i++) {
        if (children[i]->grad == NULL) {
            children[i]->grad = new T[children[i]->nelement()];
        }
    }

    int x_grad_size[] = {expand_size, act_size * batch_size};
    Tensor_<T> x_grad_column(x_grad_size, 2);

    int input_col_shape[] = {batch_size, act_size, expand_size};
    input_col.reshape(input_col_shape, 3);

    Tensor_<T> x_hat;
    x_hat.zeros_like(input_col);
    transpose(input_col, x_hat, 0, 1);

    // calculate the grad
    gemm<float>(CblasRowMajor, CblasNoTrans, CblasNoTrans, output_ch, expand_size, batch_size * act_size, 1., grad_tran.data, batch_size * act_size, x_hat.data, expand_size, 1., children[1]->grad, expand_size);
    gemm<float>(CblasRowMajor, CblasTrans, CblasNoTrans, expand_size, act_size * batch_size, output_ch, 1., children[1]->data, expand_size, grad_tran.data, act_size * batch_size, 0., x_grad_column.data, act_size * batch_size);

    int x_grad_column_size[] = {expand_size, input_col.size()[1], batch_size};
    x_grad_column.reshape(x_grad_column_size, 3);

    Tensor_<T> x_grad_column_tran;
    x_grad_column_tran.zeros_like(x_grad_column);
    int tran_x_grad_column[] = {2, 1, 0};
    transpose(x_grad_column, x_grad_column_tran, tran_x_grad_column, 3);

    int _size[] = {batch_size, height_col, width_col, expand_size};
    x_grad_column_tran.reshape(_size, 4);

    Tensor_<T> x_grad;
    x_grad.zeros_like(children[0]);

    // column to image, recover the shape of input
    col2im(x_grad_column_tran, x_grad, kern_size, stride, padding);

    for (int i=0; i<children[0]->nelement(); i++) children[0]->grad[i] += x_grad.data[i];
    
}

#endif
