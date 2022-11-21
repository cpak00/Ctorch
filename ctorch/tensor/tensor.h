#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <assert.h>
#include <cblas.h>
#include <algorithm> 
#include <stdio.h>
#include <random>
#include <limits>

#include "../utils/def.h"
#include "../autograd/autograd.h"

template <class T> class Autograd;

template <class T>
class Tensor_ {
private:
    int* _size;
    int _ndim;
    int _nelement;

    void clear_children();

public:
    bool is_root;

    T* data = NULL;
    T* grad = NULL;
    T padding;
    
    bool requires_grad;
    Tensor_<T>** children;
    int nchildren;
    // void (*grad_fn)(Tensor_<T> & grad, Tensor_<T>* children, int nchildren);
    Autograd<T>* grad_fn;

    Tensor_(bool requires_grad = true);
    Tensor_(int* size, int dim, bool requires_grad = true);
    Tensor_(const Tensor_<T> & other);
    Tensor_(Tensor_<T> && other);
    ~Tensor_();

    Tensor_<T>& operator=(Tensor_<T>& other);
    Tensor_<T>& operator=(Tensor_<T>&& other);
    
    void clone(const Tensor_<T> & tensor);
    void zeros_like(const Tensor_<T> & tensor);
    void clone(const Tensor_<T> * tensor);
    void zeros_like(const Tensor_<T> * tensor);

    int ndim() const;
    int nelement() const;
    int* size();
    void reshape(int* new_size, int nsize);
    Tensor_<T> rotate180();
    void get_index(int n, int* index);
    int get_index(int* index);

    T& index(int* ind);
    T& index(int ind);
    T get(int* ind);
    T get(int ind);

    void pretty_print(T* printed_data = NULL);

    void backward(Tensor_<T> & grad);

    void cutoff(int batch) {assert(_ndim >= 1), _size[0] = batch; _nelement = 1; for(int i=0;i<_ndim;i++) {_nelement *= _size[i];}};
    void normal(T mean, T var);
    void uniform(T mean, T var);
    void argmax(int dim, Tensor_<T> & arg, Tensor_<T> & max);
};

typedef Tensor_<float> FloatTensor;
// typedef Tensor_<float> Tensor;

/* Implementation */

template<class T>
Tensor_<T>::Tensor_(bool requires_grad): is_root(true), _size(NULL), data(NULL), grad(NULL), grad_fn(NULL), children(NULL), padding(0) {
    this->requires_grad = requires_grad;
    this->_nelement = 0;
    this->_ndim = 0;
    this->nchildren = 0;
}

template<class T>
Tensor_<T>::Tensor_(int* size, int dim, bool requires_grad): is_root(true), data(NULL), grad(NULL), grad_fn(NULL), children(NULL), padding(0) {
    this->_ndim = dim;
    this->_size = new int[this->_ndim]; // safely deleted
    this->_nelement = 1;
    this->nchildren = 0;
    for (int i=0; i<dim; i++) {
        this->_size[i] = size[i];
        this->_nelement *= size[i];
    }
    this->requires_grad = requires_grad;

    this->data = new T[this->_nelement]; // safely deleted
    if (this->requires_grad) {
        this->grad = new T[this->_nelement]; // safely deleted
        for (int i=0; i<this->_nelement; i++) this->grad[i] = 0.f;
    } else {
        this->grad = NULL;
    }
}

template<class T>
Tensor_<T>::Tensor_(const Tensor_<T>& other): is_root(true), _size(NULL), data(NULL), grad(NULL), grad_fn(NULL), children(NULL), padding(0) {
    this->clone(other);
}

template<class T>
Tensor_<T>::Tensor_(Tensor_<T> && other): is_root(true), _size(NULL), data(NULL), grad(NULL), grad_fn(NULL), children(NULL), padding(0) {
    this->clone(other);
}

template<class T>
Tensor_<T>& Tensor_<T>::operator=(Tensor_<T>& other) {
    if (this == &other) {
        return *this;
    }
    this->clone(other);
    return *this;
}

template<class T>
Tensor_<T>& Tensor_<T>::operator=(Tensor_<T>&& other) {
    if (this == &other) {
        return *this;
    }
    this->clone(other);

    return *this;
}

template<class T>
void Tensor_<T>::clone(const Tensor_<T>& other)  {
    this->clone(&other);
}

template<class T>
void Tensor_<T>::clone(const Tensor_<T>* other)  {

    this->_ndim = other->_ndim;
    this->_nelement = other->_nelement;
    this->requires_grad = other->requires_grad;
    this->grad_fn = other->grad_fn;
    this->nchildren = other->nchildren;
    this->is_root = other->is_root;

    if (other->nchildren > 0) {
        delete_s(this->children);
        this->children = new Tensor_<T>*[other->nchildren];
        for (int i=0; i<other->nchildren; i++) this->children[i] = other->children[i];
    }
    
    if (other->ndim() > 0) {
        delete_s(this->_size);
        this->_size = new int[other->ndim()]; // safely deleted
        for (int i=0; i<other->ndim(); i++) this->_size[i] = other->_size[i];
    }
    
    if (other->nelement() > 0) {
        delete_s(this->data);
        this->data = new T[other->nelement()]; // safely deleted
        for (int i=0; i<other->nelement(); i++) this->data[i] = other->data[i];
    }
    
    if (other->grad != NULL) {
        delete_s(this->grad);
        this->grad = new T[other->nelement()]; // safely deleted
        for (int i=0; i<other->nelement(); i++) this->grad[i] = other->grad[i];
    } else {
        this->grad = NULL;
    }

}

template<class T>
void Tensor_<T>::zeros_like(const Tensor_<T>& tensor)  {

    this->zeros_like(&tensor);
}

template<class T>
void Tensor_<T>::zeros_like(const Tensor_<T> * tensor)  {
    this->_ndim = tensor->_ndim;
    delete_s(this->_size);
    this->_size = new int[tensor->ndim()]; // safely deleted
    this->_nelement = tensor->nelement();
    this->padding = tensor->padding;
    this->is_root = true;
    
    for (int i=0; i<this->_ndim; i++) {
        this->_size[i] = tensor->_size[i];
    }
    // this->requires_grad = tensor->requires_grad;
    delete_s(this->data);
    this->data = new T[tensor->nelement()]; // safely deleted
    for (int i=0; i<this->_nelement; i++) {
        this->data[i] = 0;
    }
    delete_s(this->grad);
    this->grad = new T[tensor->nelement()]; // safely deleted
    for (int i=0; i<this->_nelement; i++) {
        this->grad[i] = 0;
    }
}

template<class T>
Tensor_<T>::~Tensor_() {
    delete_s(this->grad);
    delete_s(this->_size);
    delete_s(this->data);
    delete_s(this->children);
}

template<class T>
void Tensor_<T>::normal(T mean, T var) {
    std::random_device rd;  // random_device 是一个“真随机数”发生器，它的缺点就是耗能太大，所以尽量别奢侈地一直用它
    std::mt19937 gen(rd()); 
    std::normal_distribution<T> dis(mean, var);

    for (int i=0; i<this->nelement(); i++) {
        T rand_data = dis(gen);
        // rand_data = (rand_data > 1)? 1: rand_data;
        // this->data[i] = (rand_data < -1)? -1: rand_data;
        this->data[i] = rand_data;
    }
}

template<class T>
void Tensor_<T>::uniform(T min, T max) {
    std::random_device rd;  // random_device 是一个“真随机数”发生器，它的缺点就是耗能太大，所以尽量别奢侈地一直用它
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<T> dis(min, max);

    for (int i=0; i<this->nelement(); i++) {
        T rand_data = dis(gen);
        // rand_data = (rand_data > 1)? 1: rand_data;
        // this->data[i] = (rand_data < -1)? -1: rand_data;
        this->data[i] = rand_data;
    }
}

template<class T>
int Tensor_<T>::ndim() const {
    return this->_ndim;
}

template<class T>
int Tensor_<T>::nelement() const {
    return this->_nelement;
}

template<class T>
int* Tensor_<T>::size() {
    return this->_size;
}

template<class T>
void Tensor_<T>::reshape(int* new_size, int nsize) {
    int ele = 1;
    for (int i=0; i<nsize; i++) {
        ele *= new_size[i];
    }
    assert(this->nelement() == ele);

    delete_s(this->_size); // TODO
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
Tensor_<T> Tensor_<T>::rotate180() {
    int* new_size = new int[this->ndim()];
    for (int i=0; i<this->ndim()-2; i++) new_size[i] = this->size()[i];
    new_size[this->ndim()-1] = this->size()[this->ndim()-2];
    new_size[this->ndim()-2] = this->size()[this->ndim()-1];

    int* index0 = new int[this->ndim()];
    int* index1 = new int[this->ndim()];

    Tensor_<T> output(new_size, this->ndim());

    for(int i=0; i<nelement(); i++) {
        int _i = i;
        for (int j=this->ndim()-1; j>=0; j--) {
            index0[j] = _i % this->size()[j];
            index1[j] = index0[j];
            _i /= this->size()[j];
        }
        index1[ndim()-2] = this->size()[ndim()-2] - index0[ndim()-2] - 1;
        index1[ndim()-1] = this->size()[ndim()-1] -index0[ndim()-1] - 1;

        output.index(index1) = this->get(index0);
    }


    delete_s(index0);
    delete_s(index1);
    delete_s(new_size);

    return output;
}

template<class T>
void Tensor_<T>::get_index(int n, int* index) {
    for (int j=this->ndim()-1; j>=0; j--) {
        index[j] = n % this->size()[j];
        n /= this->size()[j];
    }
}

template<class T>
int Tensor_<T>::get_index(int* ind) {
     int sel = 0;
    int size = 1;
    
    for (int i = this->ndim() - 1; i >= 0; i--) {
        if (ind[i] < 0 || ind[i] >= this->size()[i]) return this->padding;
        sel += ind[i] * size;
        size *= this->size()[i];
    }

    return sel;
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
    
    delete_s(ind);

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
void Tensor_<T>::clear_children() {
    for (int i = 0; i < this->nchildren; i++) {
        children[i]->clear_children();        
    }
    delete_s(this->children);
    this->nchildren = 0;
}

template<class T>
void Tensor_<T>::backward(Tensor_<T> & grad) {
    if (this->requires_grad && this->grad_fn) {
        grad_fn->backward(grad, this->children, this->nchildren);
    }
    for (int i = 0; i < this->nchildren; i++) {
        if (this->children[i]->grad_fn != NULL) {
            Tensor_<T> new_grad;
            new_grad.clone(this->children[i]);
            delete_s(new_grad.data);
            new_grad.data = new_grad.grad; // safely deleted
            new_grad.grad = NULL;
            this->children[i]->backward(new_grad);
        }
    }
    // the children needs to be released after the total backward completed
    // otherwise the residual connection will break
    if (this->is_root) {
        clear_children();
    }
}

template <class T>
void Tensor_<T>:: argmax(int dim, Tensor_<T> & arg, Tensor_<T> & max) {

    int o_size[] = {this->_size[dim]};

    arg = Tensor_<T>(o_size, 1);
    max = Tensor_<T>(o_size, 1);

    for (int i=0; i<max.nelement(); i++) {
        max.index(i) = -std::numeric_limits<T>::max();
    }

    int* size_ptr = new int[this->ndim()];

    for (int n=0; n<this->_nelement; n++) {
        this->get_index(n, size_ptr);
        T dim_max = max.get(size_ptr[dim]);
        if (this->get(n) > dim_max) {
            max.index(size_ptr[dim]) = this->get(n);
            arg.index(size_ptr[dim]) = 1;
            for (int i=0; i<this->ndim(); i++) {
                if (i == dim) continue;
                arg.index(size_ptr[dim]) *= size_ptr[i];
            }
        }
    }

    delete_s(size_ptr);
}

#endif
