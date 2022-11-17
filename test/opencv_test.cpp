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

    DataLoader<float> trainloader(mnist_train, 512);

    bool is_next = true;

    int step = 0;

    while (is_next) {
        FloatTensor data;
        FloatTensor label;
        is_next = trainloader.next(data, label);

        step++;
        printf("step: %d\n", step);

    }

}

