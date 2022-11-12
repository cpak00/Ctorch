#include "../ctorch/utils/im2col.h"
#include "../ctorch/tensor/tensor.h"
#include "../ctorch/autograd/functionality/conv2d.h"
#include "../ctorch/module/conv2d.h"

int main() {
    int size_x[] = {2, 3, 4, 4};
    FloatTensor x(size_x, 4, true);

    for (int i=0; i<x.nelement(); i++) {
        x.index(i) = i;
    }

    x.pretty_print();

    Conv2d<float> layer1(3, 5, 3, 1, 1, true);
    for (int i=0; i<layer1.weight.nelement(); i++) {
        layer1.weight.index(i) = i;
    }
    for (int i=0; i<layer1.bias.nelement(); i++) {
        layer1.bias.index(i) = i;
    }

    layer1.weight.pretty_print();

    FloatTensor o = layer1.forward(x);
    o.pretty_print();

    FloatTensor grad;
    grad.zeros_like(o);
    for (int i = 0; i < o.nelement(); i++) {
        grad.data[i] = 1.f;
    }
    // grad.data[1 * grad.size()[2]] = 1.f;

    o.backward(grad);

    layer1.weight.pretty_print(layer1.weight.grad);
    x.pretty_print(x.grad);
    layer1.bias.pretty_print(layer1.bias.grad);
}
