#include "../ctorch/ctorch.h"

int main() {
    int size[] = {2, 3};
    FloatTensor x(size, 2, true);
    printf("x nelement: %d\n", x.nelement());

    for (int i=0; i<x.nelement(); i++) {
        x.data[i] = i;
    }

    FloatTensor y;
    y.clone(x);

    x.pretty_print();

    Add_f<float> add;

    FloatTensor* input[] = {&x, &y};

    FloatTensor z = add.forward(input, 2);

    z.pretty_print();

    FloatTensor grad;
    grad.zeros_like(x);
    for (int i=0; i<grad.nelement(); i++) {
        grad.data[i] = 1.;
    }

    z.backward(grad);

    x.pretty_print(x.grad);

}