#include "../ctorch/tensor/tensor.h"
#include <stdio.h>

int main() {
    int size[] = {2, 3};
    FloatTensor x(size, 2, true);
    printf("x nelement: %d\n", x.nelement());

    x.data[0] = 1.f;

    FloatTensor y;
    y.clone(x);
    printf("y nelement: %d\n", y.nelement());
    printf("x element 0: %f\n", x.data[0]);
    printf("y element 0: %f\n", y.data[0]);

    x.data[0] = 2.f;

    printf("x element 0: %f\n", x.data[0]);
    printf("y element 0: %f\n", y.data[0]);
}