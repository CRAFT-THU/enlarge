#include <cstdlib>

#include "helper.h"
#include "units.h"

int main(int argc, char const* argv[]) {
#if !defined(DEBUG)
    system("rm *.csv");
#endif

    size_t tsteps = 50;
    size_t num_neurons = 100;
    size_t level = 2;

    float dt = 1.;

    std::vector<std::vector<LIFNeuron>> pops;
    for (int i = 0; i < level; ++i) pops.push_back(create_lif(num_neurons));
    auto synapses = connect_exp(pops[0], pops[1]);

#ifdef DEBUG
    auto inputs = get_input();
#endif

    // * in a BSP manner
    for (size_t t = 0; t < tsteps; ++t) {
        // ! use index instead of iterator
        for (int i = 0; i < level; ++i)
            for (int j = 0; j < num_neurons; ++j) pops[i][j].update(dt);
        for (int i = 0; i < synapses.size(); ++i) synapses[i].update();

        for (int i = 0; i < level; ++i) log_nrn_volt(pops[i], i, t);

#ifdef DEBUG
        auto input = inputs[t];
#else
        auto input = gen_input(num_neurons, t, 600.0);
#endif

        for (int i = 0; i < num_neurons; ++i)
            pops[0][i].receiveCurrent(input[i]);
        for (int i = 0; i < synapses.size(); ++i) synapses[i].calCurrent(dt);
        log_syn_conduct(synapses, 0, t);
    }

    return 0;
}
