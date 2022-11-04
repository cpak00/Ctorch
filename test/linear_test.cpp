#include "../ctorch/tensor/tensor.h"
#include "../ctorch/autograd/functionality/linear.h"

int main() {
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
    z = Linear_f<float>::forward(input, 2);

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
    output = Linear_f<float>::forward(input2, 2);
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

    printf("x grad: [");
    for (int i = 0; i < x.nelement(); i++) {
        printf("%f ", x.grad[i]);
    }
    printf("]\n");

    printf("y grad: [");
    for (int i = 0; i < y.nelement(); i++) {
        printf("%f ", y.grad[i]);
    }
    printf("]\n");

    printf("z grad: [");
    for (int i = 0; i < z.nelement(); i++) {
        printf("%f ", z.grad[i]);
    }
    printf("]\n");

     printf("w grad: [");
    for (int i = 0; i < w.nelement(); i++) {
        printf("%f ", w.grad[i]);
    }
    printf("]\n");
}