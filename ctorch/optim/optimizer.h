#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H

template <class T>
class Optimizer {
protected:
    Tensor_<T>** parameters;
    int nparameters;
public:
    Optimizer(Tensor_<T>** parameters, int nparameters): parameters(parameters), nparameters(nparameters) {};

    virtual void step() = 0;
    virtual void zero_grad();
};

template<class T>
void Optimizer<T>::zero_grad() {
    // clear grad of all parameters
    for (int i=0; i<nparameters; i++) {
        if (parameters[i]->requires_grad) {
            for (int n=0; n<parameters[i]->nelement(); n++) {
                parameters[i]->grad[n] = 0.f;
            }
        }
    }
}

#endif