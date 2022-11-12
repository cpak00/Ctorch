#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H

template <class T>
class Optimizer {
    virtual void step() = 0;
};

#endif