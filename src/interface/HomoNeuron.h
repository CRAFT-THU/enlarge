#if !defined(HOMONEURON_H)
#define HOMONEURON_H

#include "../interface/Neuron.h"

class HomoNeuron : public Neuron {
   public:
    HomoNeuron(Type type, size_t num, size_t offset = 0, int buffer_size = 2)
        : Neuron(type, num, offset, buffer_size) {}
    virtual ~HomoNeuron() = 0;
    virtual bool is_equal(const HomoNeuron *n) = 0;
};

#endif  // HOMONEURON_H
