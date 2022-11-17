#include "../ctorch/ctorch.h"

int main() {
    Linear<float> layer1(3, 2, true);
    Linear<float> layer2(2, 1, true);

    // Linear_f<float> linear;

    int size_x[] = {2, 3};
    FloatTensor x(size_x, 2, true);

    for (int i = 0; i < x.nelement(); i++) {
        x.data[i] = i;
    }

    // int size_y[] = {3, 2};
    // FloatTensor y(size_y, 2, true);

    for (int i = 0; i < layer1.weight.nelement(); i++) {
        layer1.weight.data[i] = i;
    }

    for (int i = 0; i < layer1.bias.nelement(); i++) {
        layer1.bias.data[i] = i + 1;
    }

    FloatTensor z = layer1.forward(x);

    z.pretty_print();

    // Linear_f<float>::backward(grad, input, 2);
    // z.backward(grad);

    // int size_w[] = {2, 1};
    // FloatTensor w = FloatTensor(size_w, 2);
    for (int i = 0; i < layer2.weight.nelement(); i++) {
        layer2.weight.data[i] = i;
    }

    for (int i = 0; i < layer2.bias.nelement(); i++) {
        layer2.bias.data[i] = i + 1;
    }

    // FloatTensor input2[] = {z, w};
    FloatTensor output = layer2.forward(z);
    printf("out data: \n");
    output.pretty_print();

    printf("grad\n");
    FloatTensor grad;
    grad.zeros_like(output);
    for (int i = 0; i < grad.nelement(); i++) {
        grad.data[i] = 1.f;
    }

    output.backward(grad);

    printf("x grad: \n");
    x.pretty_print(x.grad);

    printf("y grad: \n");
    layer1.weight.pretty_print(layer1.weight.grad);

    printf("z grad: \n");
    z.pretty_print(z.grad);

    printf("w grad: \n");
    layer2.weight.pretty_print(layer2.weight.grad);

    layer1.bias.pretty_print(layer1.bias.grad);
    layer2.bias.pretty_print(layer2.bias.grad);

    FloatTensor** parameters = new FloatTensor*[4];
    parameters[0] = &(layer1.weight);
    parameters[1] = &(layer1.bias);
    parameters[2] = &(layer2.weight);
    parameters[3] = &(layer2.bias);

    SGD<float> optimizer(layer1.parameters(), layer1.nparameters(), 0.01, 0);
    
    optimizer.step();
    optimizer.zero_grad();

    layer1.weight.pretty_print();
    layer2.weight.pretty_print();

    layer1.bias.pretty_print();
    layer2.bias.pretty_print();

    kaiming_normal_(layer1.parameters(), layer1.nparameters());
    layer1.weight.pretty_print();
    layer1.bias.pretty_print();

    delete_s(parameters);
}