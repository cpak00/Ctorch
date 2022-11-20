# Ctorch
A completely C/C++ implementation of a convolutional neural network training and inference framework.

- [Ctorch](#ctorch)
  - [## Introduction](#-introduction)
    - [Dependencies](#dependencies)
    - [File Structure](#file-structure)
  - [## Document](#-document)
    - [Tensor](#tensor)
    - [Autograd](#autograd)
    - [Module](#module)
    - [Dataloader](#dataloader)
    - [Optimizer](#optimizer)
  - [Acknowledgement#](#acknowledgement)
    - [Dataset](#dataset)

## Introduction
---

### Dependencies
- OpenBLAS:
`sudo apt-get install libopenblas-openmp-dev libgfortran-9-dev`
- OpenCV:
`sudo apt-get install libopencv-dev`

### File Structure
```
ctorch
├─ctorch.h                      -- include file
├─utils                         -- misc
|   ├─def.h
|   ├─dir.h
|   ├─im2col.h
|   ├─init.h
|   ├─metric.h
|   ├─norm.h
|   └state_dict.h
├─tensor
|   ├─matrix.h
|   └tensor.h
├─optim
|   ├─optimizer.h
|   └sgd.h
├─module
|   ├─activation.h
|   ├─conv2d.h
|   ├─dropout.h
|   ├─linear.h
|   ├─model.h
|   ├─module.h
|   ├─pooling.h
|   └reshape.h
├─data
|  ├─dataloader.h
|  └dataset.h
├─autograd
|    ├─autograd.h
|    ├─functionality
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
---

### Tensor

### Autograd

### Module

### Dataloader

### Optimizer

## Acknowledgement#

### Dataset
- [Mnist_png](https://github.com/myleott/mnist_png)
