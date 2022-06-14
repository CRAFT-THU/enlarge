#include "../../include/BSim.h" // 引入snn加速器的所有库
#include <stdlib.h>
#include <time.h>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
	if (argc != 4)
	{
		printf("Need 3 paras. For example\n FR 1%%: %s depth num_neuron thread_num\n", argv[0]);
		return 0;
	}
	
	const int depth = atoi(argv[1]);  // 网络深度  
	const int N = atoi(argv[2]);  // 每层神经元的数量
    const int thread_num = atoi(argv[3]);  // 总线程数

	const real run_time = 10.0;
	const real dt = 1e-4;
	Network c(dt);

    Population *g[depth + 1];

    const int diff_num = 5000;
    g[1] = c.createPopulation(1, N, IAFNeuron(dt, 1, 10.0, 250.0, 2e-3, -70.0, 376.0, 
            -55.0, -70.0, 2.0, 2.0, 0.01,
            0.0, -68.56875477, 0.0, 0.0, 0.0, 0.0));

    for (int i = 2; i <= depth; i++) {
        if (i % 4 == 1) {
            g[i] = c.createPopulation(i, N, IAFNeuron(dt, 1, 10.0, 250.0, 2e-3, -70.0, 376.0, 
            -55.0, -70.0, 2.0, 2.0, 0.01,
            0.0, -68.56875477, 0.0, 0.0, 0.0, 0.0));
        }
        else {
            g[i] = c.createPopulation(1, N, IAFNeuron(dt, 1, 10.0, 250.0, 2e-3, -70.0, 376.0, 
                -55.0, -70.0, 2.0, 2.0, 0.01,
                0.0, -68.56875477, 0.0, 0.0, 0.0, 0.0));
        }
    }	

    real *delay = getConstArray((real)2*dt, N * N);

    const real scale = 1e3;
    const real w1_2 = 2.4 * scale;
    const real w2_3 = 2.4 * scale;
    const real w3_4 = 1.5 * scale;
    const real w4_5 = -0.1 * scale;
    const real w1_4 = 0.85 * scale;  // inh

    real *weight1_2 = getConstArray((real)w1_2 / N, N * N);
    real *weight2_3 = getConstArray((real)w2_3 / N, N * N);
    real *weight3_4 = getConstArray((real)w3_4 / N, N * N);
    real *weight4_5 = getConstArray((real)w4_5 / N, N * N);

    real *weight1_4 = getConstArray((real)w1_4 / N, N * N);

    enum SpikeType type = Inh;
    SpikeType *inh_con = getConstArray(type, N * N);

    for (int i = 1; i <= depth; i += 4) {
        if (i + 1 <= depth)
            c.connect(g[i], g[i+1], weight1_2, delay, NULL, N * N);

        if (i + 2 <= depth) 
            c.connect(g[i+1], g[i+2], weight2_3, delay, NULL, N * N);

        if (i + 3 <= depth) {
            c.connect(g[i+2], g[i+3], weight3_4, delay, NULL, N * N);
            c.connect(g[1], g[i+3], weight1_4, delay, NULL, N * N);
        }

        if (i + 4 <= depth) {
            c.connect(g[i+3], g[i+4], weight4_5, delay, inh_con, N * N);
        }
    }

    delArray(weight1_2);
    delArray(weight2_3);
    delArray(weight3_4);
    delArray(weight4_5);
    delArray(weight1_4);
    delArray(inh_con);

    MTSim mt(&c, dt);	// cpu
	mt.run(run_time, thread_num);

    return 0;
}