#include "../ctorch/tensor/tensor.h"
#include "../ctorch/autograd/functionality/pooling.h"
#include "../ctorch/autograd/functionality/loss.h"

int main() {
    SoftmaxLoss_f<float> softmax;

    int size_x[] = {2, 4};
    FloatTensor x(size_x, 2, true);

    x.data[0] = -0.1;
    x.data[1] = 0.2;
    x.data[2] = 0.3;
    x.data[3] = 0.4;

    x.data[4] = 0.2;
    x.data[5] = 0.1;
    x.data[6] = -0.3;
    x.data[7] = 0.1;

    int size_label[] = {2};
    FloatTensor label(size_label, 1, false);

    label.data[0] = 2;
    label.data[1] = 1;

    FloatTensor* input[2] = {&x, &label};
    FloatTensor l = softmax.forward(input, 2);

    printf("loss: %f\n", l.data[0]);

    FloatTensor grad;
    l.backward(grad);

    x.pretty_print(x.grad);
}
