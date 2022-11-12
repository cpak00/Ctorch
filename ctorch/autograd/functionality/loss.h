#ifndef _LOSS_H_
#define _LOSS_H_

#include "../../tensor/tensor.h"
#include "../../tensor/matrix.h"

#include <math.h>

template <class T>
class SoftmaxLoss_f: public Autograd<T> {
protected:
    Tensor_<T> mask;
    Tensor_<T> true_label;

public:
    Tensor_<T> _forward(Tensor_<T>** input, int ninput);
    void _backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren);
};

template <class T>
Tensor_<T> SoftmaxLoss_f<T>::_forward(Tensor_<T>** input, int ninput) {
    assert (ninput == 2);
    assert (input[0]->ndim() == 2);
    assert (input[1]->ndim() == 1);

    true_label.clone(input[1]);

    int size[] = {1};
    Tensor_<T> output(size, 1);

    T sum = 0.;
    T true_ground;

    for (int i = 0; i < input[0]->size()[0]; i++) {
        T partial_sum = 0.;
        for (int j = 0; j < input[0]->size()[1]; j++) {
            int ind[] = {i, j};
            partial_sum += exp(input[0]->get(ind));
        }
        sum += log(partial_sum);
    }
    for (int i = 0; i < input[0]->size()[0]; i++) {
        int ind[] = {i, (int)input[1]->get(i)};
        true_ground += input[0]->get(ind);
    }

    output.index(0) = ((sum - true_ground)/input[0]->size()[0]);

    return output;
}

template <class T>
void SoftmaxLoss_f<T>::_backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren) {

    int batch_size = children[0]->size()[0];
    T* exp_sum = new T[batch_size];
    for (int i=0; i<batch_size; i++) exp_sum[i] = 0;

    for (int i=0; i<children[0]->nelement(); i++) {
        children[0]->grad[i] = exp(children[0]->data[i]);
        exp_sum[i / children[0]->size()[1]] += children[0]->grad[i];
    }

    for (int i=0; i<children[0]->nelement(); i++) {
        children[0]->grad[i] /= exp_sum[i / children[0]->size()[1]];
    }

    for (int i=0; i<batch_size; i++) {
        int ind[] = {i, (int) true_label.get(i)};
        int n = children[0]->get_index(ind);
        children[0]->grad[n] -= 1;
    }

    for (int i=0; i<children[0]->nelement(); i++) {
        children[0]->grad[i] /= batch_size;
    }

    delete[] exp_sum;
}

#endif
