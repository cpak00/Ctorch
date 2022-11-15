#include "../ctorch/ctorch.h"

int main() {
    Conv2d<float> layer1(3, 5, 3, 1, 1, true);
    for (int i=0; i<layer1.weight.nelement(); i++) {
        layer1.weight.index(i) = i;
    }
    for (int i=0; i<layer1.bias.nelement(); i++) {
        layer1.bias.index(i) = i;
    }

    save<float>(layer1.parameters(), layer1.nparameters(), "load_test.pth.tar");

    Conv2d<float> layer2(3, 5, 3, 1, 1, true);

    int nparameters;
    load<float>(layer2.parameters(), layer2.nparameters(), "load_test.pth.tar");

    layer2.weight.pretty_print();
    layer2.bias.pretty_print();
}