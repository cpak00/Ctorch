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
    Tensor_<T> _forward(Tensor_<T>** input, int ninput, bool is_training=true);
    void _backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren);
};

template <class T>
Tensor_<T> SoftmaxLoss_f<T>::_forward(Tensor_<T>** input, int ninput, bool is_training) {
    assert (ninput == 2);
    assert (input[0]->ndim() == 2);
    assert (input[1]->ndim() == 1);

    // remove the max value from the input[0]
    for (int i=0; i<input[0]->size()[0]; i++) {
        T max_value = 0;
        for (int n=0; n<input[0]->size()[1]; n++) {
            int ind[] = {i, n};
            T value = input[0]->get(ind);
            max_value = (value > max_value)? value: max_value;
        }
        for (int n=0; n<input[0]->size()[1]; n++) {
            int ind[] = {i, n};
            input[0]->index(ind) -= max_value;
        }
    }

    true_label.clone(input[1]);

    int size[] = {1};
    Tensor_<T> output(size, 1);

    T sum = 0.;
    T true_ground;

    // calculate the loss: sigma(x) = exp(x)/Sigma(exp(xi))
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
    T* exp_sum = new T[batch_size]; // safely deleted
    for (int i=0; i<batch_size; i++) exp_sum[i] = 0;

    Tensor_<T> children_grad;
    children_grad.zeros_like(children[0]);

    // calculate the grad: sigma(xi) - yi

    for (int i=0; i<children[0]->nelement(); i++) {
        children_grad.data[i] = exp(children[0]->data[i]);
        exp_sum[i / children[0]->size()[1]] += children_grad.data[i];
    }

    for (int i=0; i<children[0]->nelement(); i++) {
        children_grad.data[i] /= exp_sum[i / children[0]->size()[1]];
    }

    for (int i=0; i<batch_size; i++) {
        int ind[] = {i, (int) true_label.get(i)};
        int n = children[0]->get_index(ind);
        children_grad.data[n] -= 1;
    }

    for (int i=0; i<children[0]->nelement(); i++) {
        children_grad.data[i] /= batch_size;
    }

    for (int i=0; i<children_grad.nelement(); i++) {
        children[0]->grad[i] += children_grad.get(i);
    }

    delete_s(exp_sum);
}


template <class T>
class L2Regular_f: public Autograd<T> {
    Tensor_<T> _forward(Tensor_<T>** input, int ninput, bool is_training=true);
    void _backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren);
private:
    float lambda;

public:
    L2Regular_f(float lambda): lambda(lambda) {}
};

template <class T>
Tensor_<T> L2Regular_f<T>::_forward(Tensor_<T>** input, int ninput, bool is_training) {
    assert (ninput > 0);
    
    int o_size[] = {1};
    Tensor_<T> output(o_size, 1);

    for (int i=0; i<ninput; i++) {
        Tensor_<T>* tensor = input[i];
        if (tensor->ndim() < 2) continue;
        for (int j=0; j<tensor->nelement(); j++) {
            output.index(0) += lambda / 2. * tensor->get(j) * tensor->get(j);
        }
    }

    return output;
}

template <class T>
void L2Regular_f<T>::_backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren) {
    assert (nchildren > 0);

    for (int i=0; i<nchildren; i++) {
        Tensor_<T>* tensor = children[i];
        if (tensor->ndim() < 2) continue;
        for (int j=0; j<tensor->nelement(); j++) {
            // weight decay
            tensor->grad[j] += lambda * tensor->get(j);
        }
    }
    
}

#endif
