
#ifndef ARRAY_H
#define ARRAY_H

#include "GArrayNeurons.h"

__global__ void update_array_neuron(GArrayNeurons *d_neurons, int num, int start_id);

#endif // ARRAY_H
