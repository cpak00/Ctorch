#include "../ctorch/tensor/tensor.h"
#include "../ctorch/autograd/functionality/linear.h"

int main() {
    Linear_f<float> linear;

    int size_x[] = {2, 3};
    FloatTensor x(size_x, 2, true);

    for (int i = 0; i < x.nelement(); i++) {
        x.data[i] = i;
    }

    int size_y[] = {3, 2};
    FloatTensor y(size_y, 2, true);

    for (int i = 0; i < y.nelement(); i++) {
        y.data[i] = i;
    }

    FloatTensor input[] = {x, y};
    FloatTensor z;
    z = linear.forward(input, 2);

    printf("z data: [");
    for (int i = 0; i < z.nelement(); i++) {
        printf("%f ", z.data[i]);
    }
    printf("]\n");

    // Linear_f<float>::backward(grad, input, 2);
    // z.backward(grad);

    int size_w[] = {2, 1};
    FloatTensor w = FloatTensor(size_w, 2);
    for (int i = 0; i < w.nelement(); i++) {
        w.data[i] = i;
    }

    FloatTensor input2[] = {z, w};
    FloatTensor output;
    output = linear.forward(input2, 2);
    printf("out data: [");
    for (int i = 0; i < output.nelement(); i++) {
        printf("%f ", output.data[i]);
    }
    printf("]\n");

    FloatTensor grad;
    grad.zeros_like(output);
    for (int i = 0; i < z.nelement(); i++) {
        grad.data[i] = 1.f;
    }

    output.backward(grad);

    printf("x grad: \n");
    x.pretty_print(x.grad);

    printf("y grad: \n");
    y.pretty_print(y.grad);

    printf("z grad: \n");
    z.pretty_print(z.grad);

    printf("w grad: \n");
    w.pretty_print(w.grad);
}