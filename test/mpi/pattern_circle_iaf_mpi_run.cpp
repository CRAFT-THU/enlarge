#include <stdlib.h>
#include <time.h>
#include <fstream>
#include "../../include/BSim.h"

using namespace std;

const real dt = 1e-4;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int node_id = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);

    if (argc != 2)
    {
		printf("Need 1 paras. For example\n FR 1%%: %s num_neuron \n", argv[0]);
		return 0;
	}
	const size_t N = atoi(argv[1]);  // 每层神经元的数量
    
    
    const real run_time = 10.0;

    string file_name = "pattern_circle_iaf_mpi_" + to_string(N);
    const char * name = file_name.c_str();	

    const int thread_num = 2;
    MLSim mn(name, dt, thread_num);	//gpu
	mn.run(run_time, thread_num, 1);	

    // MNSim mn(name, dt);
    // mn.run(run_time, 1);

    return 0;
}
