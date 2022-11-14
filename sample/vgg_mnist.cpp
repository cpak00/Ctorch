#include "../ctorch/ctorch.h"

template <class T>
class VGG: public Model<T> {
private:
    MaxPooling<T> pool1;
    MaxPooling<T> pool2;
    MaxPooling<T> pool3;

    Tensor_<T> x[20];

    ReLU<T> relu[7];

    Reshape<T>* reshape;

public:
    VGG(int hidden_size = 32): Model<T>(9), pool1(2), pool2(2), pool3(2) {
        this->layer[0] = new Conv2d<T>(1, hidden_size, 3);
        this->layer[1] = new Conv2d<T>(hidden_size, hidden_size, 3);

        this->layer[2] = new Conv2d<T>(hidden_size, hidden_size * 2, 3);
        this->layer[3] = new Conv2d<T>(hidden_size * 2, hidden_size * 2, 3);
        
        this->layer[4] = new Conv2d<T>(hidden_size * 2, hidden_size * 4, 3);
        this->layer[5] = new Conv2d<T>(hidden_size * 4, hidden_size * 4, 3);

        int linear_size[] = {-1, hidden_size * 4 * 4 * 4};
        reshape = new Reshape<T>(linear_size, 2);

        this->layer[6] = new Linear<T>((hidden_size * 4) * (4 * 4), hidden_size * 4);
        this->layer[7] = new Dropout<T>(0.5);
        this->layer[8] = new Linear<T>(hidden_size * 4, 10);


        this->build();
        
    }
    ~VGG() {
        for (int i=0; i<this->nlayer; i++) {
            delete this->layer[i];
        }
        delete reshape;
    }

    Tensor_<T> forward(Tensor_<T> & input) {
        x[0] = this->layer[0]->forward(input);
        x[1] = this->relu[0].forward(x[0]);
        x[2] = this->layer[1]->forward(x[1]);
        x[3] = this->relu[1].forward(x[2]);
        x[4] = pool1.forward(x[3]);
        x[5] = this->layer[2]->forward(x[4]);
        x[6] = this->relu[2].forward(x[5]);
        x[7] = this->layer[3]->forward(x[6]);
        x[8] = this->relu[3].forward(x[7]);
        x[9] = pool2.forward(x[8]);
        x[10] = this->layer[4]->forward(x[9]);
        x[11] = this->relu[4].forward(x[10]);
        x[12] = this->layer[5]->forward(x[11]);
        x[13] = this->relu[5].forward(x[12]);
        x[14] = pool3.forward(x[13]);

        x[15] = reshape->forward(x[14]);

        x[16] = this->layer[6]->forward(x[15]);
        x[17] = this->relu[6].forward(x[16]);
        x[18] = this->layer[7]->forward(x[17]);
        x[19] = this->layer[7]->forward(x[18]);

        // printf("(%d %d %d %d)\n", x[8].size()[0], x[8].size()[1], x[8].size()[2], x[8].size()[3]);
        return x[19];
    }
};

int main() {
    VGG<float> model;
    SoftmaxLoss_f<float> criterion;
    SGD<float> optimizer(model.parameters(), model.nparameters(), 0.01, 0.9);
    int epoch_size = 100;

    kaiming_normal_(model.parameters(), model.nparameters());

    int size[] = {1, 32, 32};
    ImageFolder<float> mnist_train(size, 3, "/home/chenym/Code/Project/Ctorch/datasets/mnist_png/training", true);
    ImageFolder<float> mnist_test(size, 3, "/home/chenym/Code/Project/Ctorch/datasets/mnist_png/testing", true);

    DataLoader<float> trainloader(mnist_train, 512);
    DataLoader<float> testloader(mnist_test, 512);

    bool is_next = true;
    float sum_loss = 0.f; int step = 0;
    for (int i = 0; i < epoch_size; i++) {
        model.train();
        sum_loss = 0.f; step = 0;

        is_next = true;
        trainloader.reset();
        while (is_next) {
            FloatTensor data;
            FloatTensor label;
            is_next = trainloader.next(data, label);

            FloatTensor out = model.forward(data);
            FloatTensor* criterion_input[] = {&out, &label};
            FloatTensor loss = criterion.forward(criterion_input, 2);

            sum_loss += loss.data[0];
            step++;
            printf("Epoch %3d.%3d loss: %2.4f (avg: %2.2f)\n", i, step, loss.data[0], sum_loss / step);

            optimizer.zero_grad();
            FloatTensor grad;
            loss.backward(grad);
            optimizer.step();
            // out.pretty_print();
        }
        printf("Epoch %3d Loss: %2.2f\n", i, sum_loss / step);

        model.eval();
        sum_loss = 0.f; step = 0;

        is_next = true;
        testloader.reset();
        while (is_next) {
            FloatTensor data;
            FloatTensor label;
            is_next = testloader.next(data, label);

            FloatTensor out = model.forward(data);
            FloatTensor* criterion_input[] = {&out, &label};
            FloatTensor loss = criterion.forward(criterion_input, 2);

            sum_loss += loss.data[0];
            step++;
            printf("Test %3d.%3d loss: %2.4f (avg: %2.2f)\n", i, step, loss.data[0], sum_loss / step);
        }
    }
}

