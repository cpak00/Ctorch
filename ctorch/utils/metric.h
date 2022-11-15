#ifndef _METRIC_H
#define _METRIC_H

#include "../tensor/tensor.h"

template <typename T>
float accuracy(Tensor_<T> out, Tensor_<T> label) {
    Tensor_<T> arg, max;
    out.argmax(0, arg, max);

    int accuracy_sum = 0;
    for (int i=0; i<label.nelement(); i++) {
        if ((int)arg.get(i) == (int)label.get(i)) {
            accuracy_sum += 1;
        }
    }

    return (float) accuracy_sum / (float) label.nelement();
}

#endif
