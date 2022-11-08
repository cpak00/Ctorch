#include "../ctorch/ctorch.h"


int main()
{
    int size[] = {1, 27, 27};
    ImageFolder<float> mnist_train(size, 3, "/home/chenym/Code/Project/Ctorch/datasets/mnist_png/training", true);
    ImageFolder<float> mnist_test(size, 3, "/home/chenym/Code/Project/Ctorch/datasets/mnist_png/testing", false);

    // FloatTensor img;
    // FloatTensor _label;
    // mnist_train.next(img, _label);
    // mnist_train.next(img, _label);

    // img.pretty_print();
    // abel.pretty_print();

    DataLoader<float> trainloader(mnist_train, 10);

    FloatTensor data;
    FloatTensor label;
    trainloader.next(data, label);

    data.pretty_print();
    label.pretty_print();

}

