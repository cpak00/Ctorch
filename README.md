# Ctorch
A completely C/C++ implementation of a convolutional neural network training and inference framework.

- [Ctorch](#ctorch)
  - [Introduction](#introduction)
    - [Dependencies](#dependencies)
    - [File Structure](#file-structure)
  - [Document](#document)
    - [Tensor](#tensor)
    - [Autograd](#autograd)
    - [Module](#module)
    - [Model](#model)
    - [Dataloader](#dataloader)
    - [Optimizer](#optimizer)
  - [Sample](#sample)
  - [Acknowledgement#](#acknowledgement)
    - [Library](#library)
    - [Dataset](#dataset)

## Introduction

This project implements a convolutional neural network training and inference framework implemented entirely in C++ code. This project is based on UNIX system. The file system related code (utils/dir.h) could be modified for cross-platform use.

The source code is located in the folder `ctorch`. Sample code for training and inference is located in the folder `sample`.

This work uses CMake for building. run `./build.sh` can build the total project.

### Dependencies
- OpenBLAS:
`sudo apt-get install libopenblas-openmp-dev libgfortran-9-dev`
- OpenCV:
`sudo apt-get install libopencv-dev`

### File Structure
```
ctorch
├─ctorch.h                      -- include file
├─utils                         -- miscellaneous
|   ├─def.h                       -- global definition
|   ├─dir.h                       -- directy walk used in dataset
|   ├─im2col.h                    -- image to column converter used in conv2d
|   ├─init.h                      -- kaiming_init function
|   ├─metric.h                    -- metric used in training (accuracy, ...)
|   ├─norm.h                      -- normalized function
|   └state_dict.h                 -- save and load parameters in file systems.
├─tensor                        -- tensor related
|   ├─matrix.h                    -- gemm (by OpenBLAS)
|   └tensor.h                     -- tensor implementation
├─optim                         -- optimizer
|   ├─optimizer.h                 -- abstract class of optimizer
|   └sgd.h                        -- an implementation of SGD algorithm
├─module                        -- module classes
|   ├─activation.h
|   ├─conv2d.h
|   ├─dropout.h
|   ├─linear.h
|   ├─model.h
|   ├─module.h
|   ├─pooling.h
|   └reshape.h
├─data                          -- data provider for training
|  ├─dataloader.h                 -- minibatch
|  └dataset.h                     -- ImageFolder implementation
├─autograd                      -- autograd core functions
|    ├─autograd.h                 -- abstract calss of functionality
|    ├─functionality              -- functionality implementation
|    |       ├─activation.h
|    |       ├─add.h
|    |       ├─conv2d.h
|    |       ├─dropout.h
|    |       ├─linear.h
|    |       ├─loss.h
|    |       ├─pooling.h
|    |       └reshape.h
```

## Document

### Tensor

This work implements a simple Tensor class for representing activation values and parameters. 

The data stored in the `Tensor_` class mainly consists of `data` and `grad`. `Tensor_` is implemented using template classes `Tensor_<T>`. `data` and `grad` are stored as pointers `T*`. These pointers will be set to `NULL` by default construction to avoid the problem of wild pointer release.

Suggested constructor is:
```
Tensor_(int* size, int dim, bool requires_grad = true);
```

List of common used functions

- void clone(const Tensor_<T> & tensor);
- void zeros_like(const Tensor_<T> & tensor);
- int ndim() const;
- int nelement() const;
- int* size();
- void reshape(int* new_size, int nsize);
- void get_index(int n, int* index);
- T& index(int* ind);
- T get(int* ind);
- void pretty_print(T* printed_data = NULL);
- void backward(Tensor_<T> & grad);
- void normal(T mean, T var);
- void uniform(T mean, T var);
- void argmax(int dim, Tensor_<T> & arg, Tensor_<T> & max);

### Autograd

The Autograd feature is one of the major reasons why Pytorch came to the fore in the past few years.
The User only needs to write code for forwarding, and the whole system can automatically build the operator graph based on the forward process.

This project implements the function of automatic derivation independently.

The entire functionality is built on an abstract class `Autograd<T>`

```C++
template <class T>
class Autograd {
public:
    Tensor_<T> forward(Tensor_<T>** input, int ninput, bool is_training=true);
    void backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren);

protected:
    virtual Tensor_<T> _forward(Tensor_<T>** input, int ninput, bool is_training) = 0;
    virtual void _backward(Tensor_<T> & grad, Tensor_<T>** children, int nchildren) = 0;
};
```

All operators in a convolutional neural network inherit from the `Aurograd` class to implement.
They need to implement the purely imaginary functions `_forward` and `_backward`. 
these two functions are used to define the forward computation and the gradient backpropagation of this operator, respectively. 
The `forward` function, after generating the output by `_forward`, binds the input to the output tensor for subsequent auto-derivation.
The `backward` function of the parent class implements the automatic chain rule in training.

Since the number of inputs varies from function to function, a pointer approach is used in `forward` and `backward`.


### Module

It is too complicated to build a forward pass directly in the model with various parameters and activation values, and some operations and parameters need to be encapsulated. `Module` is designed for this goal.

```C++
template <class T>
class Module {
protected:
    Tensor_<T>** _parameters = 0;
    int _nparameters;
    bool is_training;

public:
    Module(int nparam=2);
    virtual ~Module();

    Tensor_<T> weight;
    Tensor_<T> bias;
    virtual Tensor_<T> & forward(Tensor_<T> & input) = 0;

    virtual Tensor_<T>** parameters();
    virtual int nparameters();

    void train() {is_training = true;}
    void eval() {is_training = false;}
};
```

`Module` class stores an array of parameters pointers `**_parameters` and the number of parameters `_nparameters`. `forward` is a pure virtual function to be implemented in the children classes.

The `parameters()` and `nparameters()` functions are used to pass out all the parameters in the model.

### Model

The Model class, as a parent class, can simplify the implementation of a specific model to some extent.

It mainly includes the implementation of some public interfaces that allow easy handling of parameters.

`build()` function need to be called at the end of the constructor of the children classes. This function is used to implement the aggregation of the parameters of the various layers constructed in the constructor.

```C++
template<class T>
class Model {
public:
    Tensor_<T>** _parameters;
    int _nparameters;

    Module<float>** layer;
    int nlayer;
public:
    Model(int nlayer):;
    virtual ~Model();
    void train();
    void eval();

    virtual Tensor_<T> forward(Tensor_<T> & x) = 0;
    virtual Tensor_<T>** parameters();
    virtual int nparameters();
    virtual void save_state_dict(const char* filepath);
    virtual void load_state_dict(const char* filepath);

    void build();
};
```

### Dataloader

In order to achieve universal data loading for CNN, this project mainly implements ImageFolder and a kind of dataloader. An example is shown as below:

```C++
ImageFolder<float> mnist_train(size, 3, "/home/chenym/Code/Project/Ctorch/datasets/mnist_png/training", true);
ImageFolder<float> mnist_test(size, 3, "/home/chenym/Code/Project/Ctorch/datasets/mnist_png/testing", true);

DataLoader<float> trainloader(mnist_train, 512);
DataLoader<float> testloader(mnist_test, 512);
```

### Optimizer

Optimizer relies mainly on the model class and the array of parameters pointers in the module class.
When the backpropagation is completed, the value of data will be updated based on the value of grad in the Tensor class. This project realizes SGD with momentum. An example is shown as below:

```C++
SGD<float> optimizer(model.parameters(), model.nparameters(), 0.01, 0.9);
optimizer.zero_grad();
FloatTensor grad;
loss.backward(grad);
optimizer.step();
```

## Sample

This work provides four network as samples: MLP, AlexNet, VGGNet, and ResNet in the folder `sample/`.

Here provide the implementation of an alternative AlexNet with small conv and linear module:

```C++
template <class T>
class ConvNet: public Model<T> {
private:

    MaxPooling<T> pool[1] = {2};

    Tensor_<T> x[8];

    ReLU<T> relu[2];

    Reshape<T>* reshape;

public:
    ConvNet(int hidden_size = 4): Model<T>(4) {

        this->layer[0] = new Conv2d<T>(1, hidden_size, 3, 1, 1);

        int linear_size[] = {-1, hidden_size * 16 * 16};
        reshape = new Reshape<T>(linear_size, 2);

        this->layer[1] = new Linear<T>(hidden_size * 16 * 16, hidden_size * 4);
        this->layer[2] = new Dropout<T>(0.2);
        this->layer[3] = new Linear<T>(hidden_size * 4, 10);

        this->build();
        
    }

    Tensor_<T> forward(Tensor_<T> & input) {

        x[0] = this->layer[0]->forward(input);
        x[1] = this->relu[0].forward(x[0]);
        x[2] = this->pool[0].forward(x[1]);

        x[3] = reshape->forward(x[2]);

        x[4] = this->layer[1]->forward(x[3]);
        x[5] = this->relu[1].forward(x[4]);
        x[6] = this->layer[2]->forward(x[5]);
        x[7] = this->layer[3]->forward(x[6]);

        // printf("(%d %d %d %d)\n", x[8].size()[0], x[8].size()[1], x[8].size()[2], x[8].size()[3]);
        return x[7];
    }
};
```

## Acknowledgement#

### Library
- [OpenBLAS](https://github.com/xianyi/OpenBLAS)

### Dataset
- [Mnist_png](https://github.com/myleott/mnist_png)
