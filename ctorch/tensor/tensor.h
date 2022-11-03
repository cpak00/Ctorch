#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <assert.h>
#include <cblas.h>
#include <algorithm> 
#include <stdio.h>

#include "../utils/def.h"
#include "../autograd/autograd.h"

template <class T> class Autograd;

template <class T>
class Tensor_ {
private:
    int* _size;
    int _ndim;
    int _nelement;
public:
    T* data;
    T* grad;
    
    bool requires_grad;
    Tensor_<T>* children;
    int nchildren;
    void (*grad_fn)(Tensor_<T> & grad, Tensor_<T>* children, int nchildren);
public:
    Tensor_(bool requires_grad = true);
    Tensor_(int* size, int dim, bool requires_grad = true);
    Tensor_<T> clone(Tensor_<T> & tensor);
    // Tensor_(Tensor_<T>&& other);
    // Tensor_<T>& operator=(Tensor_<T>&& other);
    ~Tensor_();

    int ndim();
    int nelement();
    int* size();

    void backward(Tensor_<T> & grad);

    void clear();
};

typedef Tensor_<float> FloatTensor;
// typedef Tensor_<float> Tensor;

/* Implementation */

template<class T>
Tensor_<T>::Tensor_(bool requires_grad): _size(NULL), data(NULL), grad(NULL), grad_fn(NULL), children(NULL) {
    this->requires_grad = requires_grad;
    this->_nelement = 0;
}

template<class T>
Tensor_<T>::Tensor_(int* size, int dim, bool requires_grad): data(NULL), grad(NULL), grad_fn(NULL), children(NULL) {
    this->_ndim = dim;
    this->_size = new int[this->_ndim];
    this->_nelement = 1;
    for (int i=0; i<dim; i++) {
        this->_size[i] = size[i];
        this->_nelement *= size[i];
    }
    this->requires_grad = requires_grad;

    // TODO new a space for data
    this->data = new T[this->_nelement];
    if (this->requires_grad) {
        this->grad = new T[this->_nelement];
    }
}

template<class T>
Tensor_<T> Tensor_<T>::clone(Tensor_<T>& tensor)  {
    this->_ndim = tensor._ndim;
    this->_size = new int[this->_ndim];
    this->_nelement = 1;
    for (int i=0; i<this->_ndim; i++) {
        this->_size[i] = tensor._size[i];
        this->_nelement *= this->_size[i];
    }
    this->requires_grad = tensor.requires_grad;
    // this->grad_fn = tensor.grad_fn;
    // this->children = tensor.children;
    // this->nchildren = tensor.nchildren;

    // copy for data
    this->data = new T[this->_nelement];
    for (int i=0; i<this->_nelement; i++) {
        this->data[i] = tensor.data[i];
    }

    if (tensor.grad != NULL) {
        this->grad = new T[this->_nelement];
        for (int i=0; i<this->_nelement; i++) {
            this->grad[i] = tensor.grad[i];
        }
    }
}

/*
template<class T>
Tensor_<T>::Tensor_(Tensor_<T>&& other) {
    this->_ndim = other._ndim;
    this->_nelement = other._nelement;
    this->requires_grad = other.requires_grad;
    this->grad_fn = other.grad_fn;
    this->children = other.children;
    this->nchildren = other.nchildren;
    std::swap(_size, other._size);
    std::swap(data, other.data);
    std::swap(grad, other.grad);
}

template<class T>
Tensor_<T>& Tensor_<T>::operator=(Tensor_<T>&& other) {
    if (this == &other) {
        return *this;
    }
    this->_ndim = other._ndim;
    this->_nelement = other._nelement;
    this->requires_grad = other.requires_grad;
    this->grad_fn = other.grad_fn;
    this->children = other.children;
    this->nchildren = other.nchildren;
    std::swap(_size, other._size);
    std::swap(data, other.data);
    std::swap(grad, other.grad);
    return *this;
}
*/
template<class T>
Tensor_<T>::~Tensor_() {
    // delete_s(this->_size);
    // delete_s(this->data);
    // delete_s(this->grad);
}

template<class T>
void Tensor_<T>::clear() {
    delete_s(this->_size);
    delete_s(this->data);
    delete_s(this->grad);
}

template<class T>
int Tensor_<T>::ndim() {
    return this->_ndim;
}

template<class T>
int Tensor_<T>::nelement() {
    return this->_nelement;
}

template<class T>
int* Tensor_<T>::size() {
    return this->_size;
}

template<class T>
void Tensor_<T>::backward(Tensor_<T> & grad) {
    if (this->requires_grad && this->grad_fn) {
        grad_fn(grad, this->children, this->nchildren);
    }
    for (int i = 0; i < this->nchildren; i++) {
        if (this->children[i].grad_fn != NULL) {
            Tensor_<T> new_grad(this->children[i]);
            new_grad.data = new_grad.grad;
            new_grad.grad = NULL;
            this->children[i].backward(new_grad);
        }
    }
}

#endif
