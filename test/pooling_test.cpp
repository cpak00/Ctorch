#include "../ctorch/tensor/tensor.h"
#include "../ctorch/autograd/functionality/pooling.h"
#include "../ctorch/autograd/functionality/activation.h"

int main() {
    MaxPooling_f<float> pool(2);
    ReLU_f<float> relu;


    int size_x[] = {2, 4, 4};
    FloatTensor x(size_x, 3, true);

    for (int i = 0; i < x.nelement(); i++) {
        x.data[i] = rand() % 10 - 5;
    }

    FloatTensor input[] = {x};
    FloatTensor output1 = pool.forward(input, 1);

    FloatTensor input2[] = {output1};
    FloatTensor output2 = relu.forward(input2, 1);

    x.pretty_print();
    output1.pretty_print();
    output2.pretty_print();

    FloatTensor grad;
    grad.zeros_like(output2);
    for (int i = 0; i < output2.nelement(); i++) {
        grad.data[i] = 1.f;
    }

    output2.backward(grad);
    x.pretty_print(x.grad);
}