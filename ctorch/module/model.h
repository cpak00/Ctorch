#include "module.h"
#include "../utils/state_dict.h"

template<class T>
class Model {
public:
    Tensor_<T>** _parameters;
    int _nparameters;

    Module<float>** layer;
    int nlayer;
public:
    Model(int nlayer): nlayer(nlayer), _parameters(NULL) {
        layer = new Module<float>*[nlayer]; // safely deleted
    };
    virtual ~Model() {
        delete_s(this->layer);
    }

    void train() {
        for (int i=0; i<nlayer; i++) {
            layer[i]->train();
        }
    }
    void eval() {
        for (int i=0; i<nlayer; i++) {
            layer[i]->eval();
        }
    }

    virtual Tensor_<T> forward(Tensor_<T> & x) = 0;

    virtual Tensor_<T>** parameters() {return _parameters;};
    virtual int nparameters() {return _nparameters;};

    virtual void save_state_dict(const char* filepath) {
        save<T>(_parameters, _nparameters, filepath);
    }
    virtual void load_state_dict(const char* filepath) {
        load<T>(_parameters, _nparameters, filepath);
    }

    void build() {
        _nparameters = 0;
        for (int i=0; i<nlayer; i++) {
            _nparameters += layer[i]->nparameters();
        }
        delete_s(_parameters);
        _parameters = new Tensor_<T>* [_nparameters]; // safely deleted
        
        int ind = 0;
        for (int i=0; i<nlayer; i++) {
            for (int j=0; j<layer[i]->nparameters(); j++) {
                assert(ind < _nparameters);
                _parameters[ind] = layer[i]->parameters()[j];
                ind ++;
            }
        }
    }
};


