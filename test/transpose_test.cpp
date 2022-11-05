#include <stdio.h>
#include "../ctorch/tensor/tensor.h"
#include "../ctorch/tensor/matrix.h"

int main() {
    int size1[] = {2, 1, 2, 2};
    int size2[] = {1, 2, 2, 2};

    int index1[] = {0, 0, 0, 0};
    int index2[4];

    for (int x=0; x<8; x++){
        int n = x % 2;
        int k = (x/2) % 2;
        int j = (x/2/2) % 1;
        int i = (x/2/2/1) % 2;

        printf("%d %d %d %d\n", i, j, k, n);
    }

    FloatTensor x(size1, 4);
    FloatTensor y(size2, 4);

    for (int i=0; i<x.nelement(); i++) x.index(i) = i;

    transpose<float>(x, y, 2, 3);

    y.pretty_print();
}