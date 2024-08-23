#pragma once

#include <cstring>
#include <random>
#include <vector>

#include "units.h"

std::vector<float> gen_input(size_t size, size_t t, float mean = 400.0f) {
    std::vector<float> result(size);
    std::default_random_engine generator(std::random_device{}());
    std::poisson_distribution<int> pd(mean);
    for (int i = 0; i < size; i++) result[i] = pd(generator);

    // * auto-log inputs
    FILE* f = fopen("input.csv", "a");
    for (auto& val : result) {
        fprintf(f, "%f", val);
        if (&val != &result.back()) fprintf(f, ",");
    }
    fprintf(f, "\n");
    fclose(f);

    return result;
}

// ^ All records are comma-separated

std::vector<std::vector<float>> get_input() {
    std::vector<std::vector<float>> result;
    FILE* f = fopen("input.csv", "r");
    if (f == NULL) return result;

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        std::vector<float> input;
        char* p = strtok(line, ",");
        while (p) {
            input.push_back(atof(p));
            p = strtok(NULL, ",");
        }
        result.push_back(input);
    }

    fclose(f);
    return result;
}

void log_nrn_volt(std::vector<LIFNeuron>& neurons, std::string pop_name,
                  size_t t) {
    FILE* f = fopen((pop_name + ".csv").c_str(), "a");
    for (auto& nrn : neurons) {
        // fprintf(f, "%d", nrn.hasFired());
        fprintf(f, "%f", nrn.V);
        if (&nrn != &neurons.back()) fprintf(f, ",");
    }
    fprintf(f, "\n");
    fclose(f);
}

template <typename SYNTYPE>
void log_syn_conduct(std::vector<SYNTYPE>& synapses, std::string pop_name,
                     size_t t) {
    FILE* f = fopen((pop_name + ".csv").c_str(), "a");
    for (auto& syn : synapses) {
        fprintf(f, "%f", syn.s);
        if (&syn != &synapses.back()) fprintf(f, ",");
    }
    fprintf(f, "\n");
    fclose(f);
}