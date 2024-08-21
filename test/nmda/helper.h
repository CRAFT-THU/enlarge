#pragma once

#include <cstring>
#include <random>
#include <vector>

// Generate a N*T sparse matrix as inputs
std::vector<std::vector<float>> gen_input(size_t size, size_t tsteps,
                                          float mean = 100.0f,
                                          std::string filename = "input.csv") {
    std::vector<std::vector<float>> result(size);
    std::default_random_engine generator(std::random_device{}());
    std::poisson_distribution<int> pd(mean);
    for (int t = 0; t < tsteps; ++t)
        for (int i = 0; i < size; ++i) result[i].emplace_back(pd(generator));

    // * log inputs
    FILE *f = fopen(filename.c_str(), "w");
    for (auto &nrn : result) {
        fprintf(f, "%f", nrn[0]);
        for (int i = 1; i < nrn.size(); ++i) fprintf(f, ",%f", nrn[i]);
    }
    fprintf(f, "\n");
    fclose(f);

    return result;
}

// ^ All records are comma-separated
std::vector<std::vector<float>> get_input(std::string filename = "input.csv") {
    std::vector<std::vector<float>> result;
    FILE *f = fopen(filename.c_str(), "r");
    if (f == NULL) return result;

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        std::vector<float> input;
        char *p = strtok(line, ",");
        while (p) {
            input.push_back(atof(p));
            p = strtok(NULL, ",");
        }
        result.push_back(input);
    }

    fclose(f);
    return result;
}