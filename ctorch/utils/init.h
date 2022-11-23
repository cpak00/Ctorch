#include "../tensor/tensor.h"


template <typename T>
void kaiming_normal_(Tensor_<T>** parameters, int nparameters) {
    // initial all the parameters by normal(0, 1)/sqrt(fan_in/2)
    for (int i=0; i<nparameters; i++) {
        if (parameters[i]->ndim() > 1) {
            parameters[i]->normal(0., 1.);
            
            for (int j=0; j<parameters[i]->nelement(); j++) {
                int fan_in = 1;
                for (int k=1; k<parameters[i]->ndim(); k++) fan_in *= parameters[i]->size()[k];
                parameters[i]->data[j] *= sqrt(2./fan_in);
            }
            
        }
        else if (parameters[i]->ndim() == 1) {
            parameters[i]->normal(0, 1);
        }
    }
}
