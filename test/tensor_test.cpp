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

    // x.data[0] = 2.f;
    for (int i = 0; i<2; i++) {
        for (int j = 0; j<3; j++) {
            int ind[] = {i, j};
            x.index(ind) = j + i * 2;
        }
    }

    printf("x element 0: %f\n", x.data[0]);
    printf("y element 0: %f\n", y.data[0]);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            int ind[] = {i, j};
            printf("%f ", x.index(ind));
        }
        printf("\n");
    }
}