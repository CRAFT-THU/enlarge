/**
 * @ref ../cpu/poisson_cpu.cpp
 */

#include <stdlib.h>
#include <time.h>

#include <iostream>

#include "../../include/BSim.h"  // 引入snn加速器的所有库
#include "helper.h"

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int node_id = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);

    const int N = 10;  // 单个神经元测试
                       // * run 100 tsteps
    const real run_time = 5e-3;
    const real dt = 0.5e-4;
    Network c(dt);

    if (node_id == 0) {
        // parameter for LIF model
        const real fthreshold = -54e-3;
        const real freset = -60e-3;
        const real c_m = 0.2e-9;        //*nF
        const real v_rest = -74e-3;     //*mV
        const real tau_syn_e = 2e-3;    //*ms
        const real tau_syn_i = 0.6e-3;  //*ms

        // # 论文没有这些参数
        const real tau_m = 10e-3;        //*ms
        const real i_offset = 4.6e-10;   //*nA
        const real i_offset2 = 4.6e-10;  //*nA
        const real frefractory = 0;
        const real fv = -74e-3;

        real pmean = 4;
        // real w = 2.4 * (real)(1e-9) / N;
        real w = 1.0;
        auto input = gen_input(N, round(run_time / dt), pmean * w);

        Population* g[5];
        g[0] = c.createPopulation(
            1, N,
            LIFNeuron(fv, v_rest, freset, c_m, tau_m, frefractory, tau_syn_e,
                      tau_syn_i, fthreshold, 0, dt));
        g[1] = c.createPopulation(
            1, N,
            LIFNeuron(fv, v_rest, freset, c_m, tau_m, frefractory, tau_syn_e,
                      tau_syn_i, fthreshold, 0, dt));

        for (int i = 0; i < N; ++i)
            c.set_input(g[0], i, input[i].data(), input[i].size());

        real* delay = getConstArray((real)1e-4, N * N);
        real* weight = getConstArray(w, N * N);
        c.connect(g[0], g[1], weight, delay, NULL, N * N);

        c.log_graph();
    }

    // MNSim sim(&c, dt);
    // sim.run(run_time);

    int num_thread = 2;
    MLSim sim(&c, dt);
    sim.run(run_time, num_thread);

    return 0;
}
