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
    T padding;
    
    bool requires_grad;
    Tensor_<T>* children;
    int nchildren;
    // void (*grad_fn)(Tensor_<T> & grad, Tensor_<T>* children, int nchildren);
    Autograd<T>* grad_fn;
public:
    Tensor_(bool requires_grad = true);
    Tensor_(int* size, int dim, bool requires_grad = true);
    void clone(Tensor_<T> & tensor);
    void zeros_like(Tensor_<T> & tensor);
    // Tensor_(Tensor_<T>&& other);
    // Tensor_<T>& operator=(Tensor_<T>&& other);
    ~Tensor_();

    int ndim();
    int nelement();
    int* size();
    void reshape(int* new_size, int nsize);

    T& index(int* ind);
    T& index(int ind);
    T get(int* ind);
    T get(int ind);
    void pretty_print(T* printed_data = NULL);

    void backward(Tensor_<T> & grad);

    void clear();
};

typedef Tensor_<float> FloatTensor;
// typedef Tensor_<float> Tensor;

/* Implementation */

template<class T>
Tensor_<T>::Tensor_(bool requires_grad): _size(NULL), data(NULL), grad(NULL), grad_fn(NULL), children(NULL), padding(0) {
    this->requires_grad = requires_grad;
    this->_nelement = 0;
}

template<class T>
Tensor_<T>::Tensor_(int* size, int dim, bool requires_grad): data(NULL), grad(NULL), grad_fn(NULL), children(NULL), padding(0) {
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
void Tensor_<T>::clone(Tensor_<T>& tensor)  {
    this->_ndim = tensor._ndim;
    this->_size = new int[this->_ndim];
    this->_nelement = 1;
    for (int i=0; i<this->_ndim; i++) {
        this->_size[i] = tensor._size[i];
        this->_nelement *= this->_size[i];
    }
    this->requires_grad = tensor.requires_grad;
    this->padding = tensor.padding;
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

template<class T>
void Tensor_<T>::zeros_like(Tensor_<T>& tensor)  {
    this->_ndim = tensor._ndim;
    this->_size = new int[this->_ndim];
    this->_nelement = 1;
    this->padding = tensor.padding;
    for (int i=0; i<this->_ndim; i++) {
        this->_size[i] = tensor._size[i];
        this->_nelement *= this->_size[i];
    }
    // this->requires_grad = tensor.requires_grad;
    this->data = new T[this->_nelement];
    for (int i=0; i<this->_nelement; i++) {
        this->data[i] = 0;
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
void Tensor_<T>::reshape(int* new_size, int nsize) {
    this->_ndim = nsize;
    this->_size = new int[nsize];
    for (int i=0; i<nsize; i++) {
        this->_size[i] = new_size[i];
    }
}


template<class T>
T& Tensor_<T>::index(int* ind) {
    int sel = 0;
    int size = 1;
    
    for (int i = this->ndim() - 1; i >= 0; i--) {  
        if (ind[i] < 0 || ind[i] >= this->size()[i]) return this->padding;
        sel += ind[i] * size;
        size *= this->size()[i];
    }

    return this->data[sel];
}

template<class T>
T Tensor_<T>::get(int* ind) {
    int sel = 0;
    int size = 1;
    
    for (int i = this->ndim() - 1; i >= 0; i--) {
        if (ind[i] < 0 || ind[i] >= this->size()[i]) return this->padding;
        sel += ind[i] * size;
        size *= this->size()[i];
    }

    return this->data[sel];
}

template<class T>
T& Tensor_<T>::index(int ind) {
    if (ind < 0 || ind >= this->nelement()) return this->padding;
    return this->data[ind];
}

template<class T>
T Tensor_<T>::get(int ind) {
    if (ind < 0 || ind >= this->nelement()) return this->padding;
    return this->data[ind];
}


template<class T>
void Tensor_<T>::pretty_print(T* printed_data) {

    T* printed_ptr = (printed_data == NULL)? this->data:printed_data;


    int* ind = new int[this->ndim()];

    int size = 1;
    for (int i = this->ndim() - 1; i >= 0; i--) {
        size *= this->size()[i];
        ind[this->ndim()-1-i] = size;
    }

    int row_ch = this->size()[this->ndim() - 1];
    for (int k = 0; k < this->ndim(); k++) printf("[");
    for (int i = 0; i < this->nelement(); i++) {
        printf("%.2f ", printed_ptr[i]);
        for (int j = this->ndim()-1; j >= 0; j--) {
            if ((i + 1) % ind[j] == 0) {
                for (int k = 0; k < j+1; k++) printf("]");
                printf("\n");
                for (int k = 0; k < this->ndim()-1; k++) printf(" ");
                if (i != this->nelement()-1) printf("[");
                break;
            }
        }
    }
    
    delete []ind;

    /*
    int length = 7;
    printf("[");
    for (int i = 0; i < this->nelement(); i++) {
        if (i == length && i < this->nelement() - 2) {
            printf(" ... ");
            continue;
        }
        if (i > length && i < this->nelement() - 2) continue;
        printf("%.2f, ", this->data[i]);
    }
    printf("]\n");
    */

}

template<class T>
void Tensor_<T>::backward(Tensor_<T> & grad) {
    if (this->requires_grad && this->grad_fn) {
        grad_fn->backward(grad, this->children, this->nchildren);
    }
    for (int i = 0; i < this->nchildren; i++) {
        if (this->children[i].grad_fn != NULL) {
            Tensor_<T> new_grad;
            new_grad.clone(this->children[i]);
            new_grad.data = new_grad.grad;
            new_grad.grad = NULL;
            this->children[i].backward(new_grad);
            new_grad.clear();
        }
    }
}

#endif
