#include "../ctorch/ctorch.h"
#include <unistd.h>

template <class T>
class ResNet: public Model<T> {
private:
    Add_f<T> add[2];
    MaxPooling<T> pool[4] = {2, 2, 2, 2};
    Tensor_<T> x[24];
    ReLU<T> relu[8];
    Reshape<T>* reshape;

public:
    ResNet(int hidden_size = 4): Model<T>(9) {
        this->layer[0] = new Conv2d<T>(1, hidden_size, 3);
        this->layer[1] = new Conv2d<T>(hidden_size, hidden_size * 2, 3);

        this->layer[2] = new Conv2d<T>(hidden_size * 2, hidden_size * 2, 3);
        this->layer[3] = new Conv2d<T>(hidden_size * 2, hidden_size * 2, 3);
        
        this->layer[4] = new Conv2d<T>(hidden_size * 2, hidden_size * 4, 3);
        this->layer[5] = new Conv2d<T>(hidden_size * 4, hidden_size * 4, 3);

        this->layer[6] = new Conv2d<T>(hidden_size * 4, hidden_size * 4, 3);
        this->layer[7] = new Conv2d<T>(hidden_size * 4, hidden_size * 4, 3);

        int linear_size[] = {-1, hidden_size * 4 * 2 * 2};
        reshape = new Reshape<T>(linear_size, 2);
       
        this->layer[8] = new Linear<T>(hidden_size * 4 * 2 * 2, 10);

        this->build();
        
    }
    ~ResNet() {
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

        x[4] = pool[0].forward(x[3]);

        x[5] = this->layer[2]->forward(x[4]);
        x[6] = this->relu[2].forward(x[5]);
        x[7] = this->layer[3]->forward(x[6]);
        x[8] = this->relu[3].forward(x[7]);

        Tensor_<T>* add_0[] = {x+4, x+8};
        x[9] = add[0].forward(add_0, 2);

        x[10] = this->layer[4]->forward(x[9]);
        x[11] = this->relu[4].forward(x[10]);

        x[12] = pool[1].forward(x[11]);
        
        x[13] = this->layer[5]->forward(x[12]);
        x[14] = this->relu[5].forward(x[13]);

        x[15] = pool[2].forward(x[14]);

        x[16] = this->layer[6]->forward(x[15]);
        x[17] = this->relu[6].forward(x[16]);
        x[18] = this->layer[7]->forward(x[17]);
        x[19] = this->relu[7].forward(x[18]);

        Tensor_<T>* add_1[] = {x+15, x+19};
        x[20] = add[1].forward(add_1, 2);

        x[21] = pool[3].forward(x[20]);

        x[22] = reshape->forward(x[21]);
        
        x[23] = this->layer[8]->forward(x[22]);

        // printf("(%d %d %d %d)\n", x[8].size()[0], x[8].size()[1], x[8].size()[2], x[8].size()[3]);
        return x[23];
    }
};

int main(int argc, char** argv) {

    ResNet<float> model;

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

                optimizer.zero_grad();
                FloatTensor grad;
                loss.backward(grad);
                reg.backward(grad);
                optimizer.step();
                // out.pretty_print();

                model.save_state_dict("resnet_mnist_checkpoint.pth.tar");
            }
            printf("Epoch %3d Loss: %2.2f\n", i, sum_loss / step);

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

