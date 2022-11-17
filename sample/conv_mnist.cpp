#include "../ctorch/ctorch.h"
#include <unistd.h>

template <class T>
class ConvNet: public Model<T> {
private:

    MaxPooling<T> pool[1] = {2};

    Tensor_<T> x[8];

    ReLU<T> relu[2];

    Reshape<T>* reshape;

public:
    ConvNet(int hidden_size = 4): Model<T>(4) {

        this->layer[0] = new Conv2d<T>(1, hidden_size, 3, 1, 1);

        int linear_size[] = {-1, hidden_size * 16 * 16};
        reshape = new Reshape<T>(linear_size, 2);

        this->layer[1] = new Linear<T>(hidden_size * 8 * 8, hidden_size * 4);
        this->layer[2] = new Dropout<T>(0.2);
        this->layer[3] = new Linear<T>(hidden_size * 4, 10);

        this->build();
        
    }

    Tensor_<T> forward(Tensor_<T> & input) {

        x[0] = this->layer[0]->forward(input);
        x[1] = this->relu[0].forward(x[0]);
        x[2] = this->pool[0].forward(x[1]);

        x[3] = reshape->forward(x[2]);

        x[4] = this->layer[1]->forward(x[3]);
        x[5] = this->relu[1].forward(x[4]);
        x[6] = this->layer[2]->forward(x[5]);
        x[7] = this->layer[3]->forward(x[6]);

        // printf("(%d %d %d %d)\n", x[8].size()[0], x[8].size()[1], x[8].size()[2], x[8].size()[3]);
        return x[7];
    }
};

int main(int argc, char** argv) {

    ConvNet<float> model;

    kaiming_normal_(model.parameters(), model.nparameters());

    int opt;
    bool is_evaluate = false;
     while ((opt = getopt(argc, argv, "e:r:")) != -1) {
        switch ( opt ) {
            case 'e':
                is_evaluate = true;
                printf("== Evaluate\n");
                printf("== Model parameters load: %s\n", optarg); 
                model.load_state_dict(optarg);
                break;
            case 'r':
                is_evaluate = false;
                printf("== Resume\n");
                printf("== Model parameters load: %s\n", optarg); 
                model.load_state_dict(optarg);
                break;
            case '?':
                    cerr << "Unknown option: '" << char(optopt) << "'!" << endl;
                break;
        }
    }
    
    SoftmaxLoss_f<float> criterion;
    L2Regular_f<float> regularization(0.005);
    SGD<float> optimizer(model.parameters(), model.nparameters(), 0.01, 0.9);
    int epoch_size = 100;

    int size[] = {1, 32, 32};
    ImageFolder<float> mnist_train(size, 3, "/home/chenym/Code/Project/Ctorch/datasets/mnist_png/training", true);
    ImageFolder<float> mnist_test(size, 3, "/home/chenym/Code/Project/Ctorch/datasets/mnist_png/testing", true);

    DataLoader<float> trainloader(mnist_train, 512);
    DataLoader<float> testloader(mnist_test, 512);

    bool is_next = true;
    float sum_loss = 0.f; float sum_acc = 0.f; int step = 0;
    if (!is_evaluate) {
        // training
        for (int i = 0; i < epoch_size; i++) {
            model.train();
            sum_loss = 0.f; sum_acc = 0.f; step = 0;

            is_next = true;
            trainloader.reset();
            while (is_next) {
                FloatTensor data;
                FloatTensor label;
                is_next = trainloader.next(data, label);

                FloatTensor out = model.forward(data);
                FloatTensor* criterion_input[] = {&out, &label};

                float acc = accuracy(out, label);
                FloatTensor loss = criterion.forward(criterion_input, 2);
                FloatTensor reg = regularization.forward(model.parameters(), model.nparameters());

                sum_acc += acc;
                sum_loss += loss.data[0];
                step++;
                printf("Epoch %3d.%3d loss: %2.4f (avg: %2.2f)\t acc: %2.1f%% (avg: %2.1f%%) \n", i, step, loss.data[0], sum_loss / step, acc * 100.0, sum_acc / step * 100.0);

                // optimizer.zero_grad();
                FloatTensor grad;
                loss.backward(grad);
                reg.backward(grad);
                // optimizer.step();
                // out.pretty_print();

                model.save_state_dict("conv_mnist_checkpoint.pth.tar");
            }
            printf("Epoch %3d Loss: %2.2f\n", i, sum_loss / step);

            if (i % 5 == 4) {
                model.eval();
                sum_loss = 0.f; sum_acc = 0.f; step = 0;

                is_next = true;
                testloader.reset();
                while (is_next) {
                    FloatTensor data;
                    FloatTensor label;
                    is_next = testloader.next(data, label);

                    FloatTensor out = model.forward(data);
                    FloatTensor* criterion_input[] = {&out, &label};

                    float acc = accuracy(out, label);
                    FloatTensor loss = criterion.forward(criterion_input, 2);

                    sum_acc += acc;
                    sum_loss += loss.data[0];
                    step++;
                    printf("Eval %3d.%3d loss: %2.4f (avg: %2.2f)\t acc: %2.1f%% (avg: %2.1f%%) \n", i, step, loss.data[0], sum_loss / step, acc * 100.0, sum_acc / step * 100.0);
                }
            }
        }
    } else {
        model.eval();
        sum_loss = 0.f; sum_acc = 0.f; step = 0;

        is_next = true;
        testloader.reset();
        while (is_next) {
            FloatTensor data;
            FloatTensor label;
            is_next = testloader.next(data, label);

            FloatTensor out = model.forward(data);
            FloatTensor* criterion_input[] = {&out, &label};

            float acc = accuracy(out, label);
            FloatTensor loss = criterion.forward(criterion_input, 2);

            sum_acc += acc;
            sum_loss += loss.data[0];
            step++;
            printf("Eval %3d loss: %2.4f (avg: %2.2f)\t acc: %2.1f%% (avg: %2.1f%%) \n", step, loss.data[0], sum_loss / step, acc * 100.0, sum_acc / step * 100.0);
        }
    }
}

