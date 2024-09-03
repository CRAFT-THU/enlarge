
#ifndef HELPER_PARALLEL_H
#define HELPER_PARALLEL_H

#include <mpi.h>

#include "helper_array.h"

template<typename T>
void mpi_print_array(T *array, size_t size, int rank, int proc_num, const char *msg="")
{
	for (int i=0; i<proc_num; i++) {
		if (i == rank) {
			fprintf(stdout, "%s ", msg);
			log_array_noendl(stdout, array, size);
			fprintf(stdout, "\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

#ifdef USE_GPU

#include "helper_gpu.h"

template<typename T>
void mpi_print_gpu_array(T *array, size_t size, int rank, int proc_num, const char *msg="")
{
	T *tmp = FROMGPU(array, size);
	for (int i=0; i<proc_num; i++) {
		if (i == rank) {
			fprintf(stdout, "%s ", msg);
			log_array_noendl(stdout, tmp, size);
			fprintf(stdout, "\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	free_c(tmp);
}
#endif


#endif // HELPER_PARALLEL_H
