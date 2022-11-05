
#include "../ctorch/utils/im2col.h"
#include "../ctorch/tensor/tensor.h"
#include "../ctorch/autograd/functionality/conv2d.h"

int main() {
    int size_x[] = {2, 1, 2, 2};
    FloatTensor x(size_x, 4, true);

    int size_w[] = {2, 1, 3, 3};
    FloatTensor w(size_w, 4, true);

    for (int i=0; i<x.nelement(); i++) {
        x.index(i) = i;
    }

    for (int i=0; i<w.nelement(); i++) {
        w.index(i) = i;
    }

    int ind[] = {0, 0, 0, 0};
    printf("%f\n", x.get(ind));
    
    int ind2[] = {0, 0, 0, 1};
    printf("%f\n", x.get(ind2));

    int ind3[] = {0, 0, 1, 0};
    printf("%f\n", x.get(ind3));

    int ind4[] = {0, 0, 1, 1};
    printf("%f\n", x.get(ind4));

    x.pretty_print();

    printf("\n");

    FloatTensor x_col = im2col(x, 3, 1, 1);
    x_col.pretty_print();

    Conv2d_f<float> conv1(3, 1, 1);
    
    FloatTensor input[] = {x, w};
    FloatTensor output = conv1.forward(input, 2);

    printf("(%d, %d, %d, %d)\n", output.size()[0], output.size()[1], output.size()[2], output.size()[3]);
    output.pretty_print();

    FloatTensor grad;
    grad.zeros_like(output);
    for (int i = 0; i < output.nelement(); i++) {
        grad.data[i] = 1.f;
    }

    FloatTensor o_col = im2col(output, 3, 1, 1);
    o_col.pretty_print();

    int size_ox[] = {2, 1, 3, 3};
    FloatTensor o_x(size_ox, 4);

    gemm<float>(CblasRowMajor, CblasTrans, CblasNoTrans, 2, 9, 8, 1., grad.data, 2, x_col.data, 9, 0., o_x.data, 9);

    o_x.pretty_print();

    output.backward(grad);

    w.pretty_print(w.grad);
}