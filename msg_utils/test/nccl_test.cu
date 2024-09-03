#ifdef NCCL
#include <stdio.h>

#include "nccl.h"
#include "mpi.h"

#include "../helper/helper_gpu.h"

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
#endif

int main(int argc, char **argv)
{
#ifdef NCCL
	const int NGPU = 2;
	int rank = 0, size = 0;

	MPICHECK(MPI_Init(&argc, &argv));
	MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

	MPI_Comm comm_mpi;

	MPICHECK(MPI_Comm_split(MPI_COMM_WORLD, rank/NGPU, rank, &comm_mpi));


	int rank_gpu = 0, size_gpu = 0;

	MPICHECK(MPI_Comm_rank(comm_mpi, &rank_gpu));
	MPICHECK(MPI_Comm_size(comm_mpi, &size_gpu));

	printf("Rank %d out of %d, gpu %d out of %d\n", rank, size, rank_gpu, size_gpu);

	gpuSetDevice(rank_gpu);

	ncclUniqueId id;
	ncclComm_t comm_gpu;
	cudaStream_t s;

	checkCudaErrors(cudaStreamCreate(&s));

	if (0 == rank_gpu) {
		ncclGetUniqueId(&id);
	}

	MPI_Request request;
	MPI_Status status;
	double ts = MPI_Wtime();
	MPICHECK(MPI_Ibcast(&id, sizeof(id), MPI_BYTE, 0, comm_mpi, &request));
	double tm = MPI_Wtime();
	MPICHECK(MPI_Wait(&request, &status));
	double te = MPI_Wtime();
	printf("Time: %lf %lf\n", tm - ts, te - tm);

	NCCLCHECK(ncclCommInitRank(&comm_gpu, size_gpu, id, rank_gpu));

	int sc[2] = {0, 0};
	int rc[2] = {0, 0};

	float *sb = NULL;
	float *rb = NULL;

	float *sb_gpu = NULL;
	float *rb_gpu = NULL;

	if (0 == rank_gpu) {
		sc[1] = 2;
		rc[1] = 1;

		sb = new float[2];
		rb = new float[1];

		sb[0] = rank;
		sb[1] = rank + 0.1;
		rb[0] = 0.0;

		sb_gpu = TOGPU(sb, 2);
		rb_gpu = TOGPU(rb, 1);

	} else {
		sc[0] = 1;
		rc[0] = 2;
		
		sb = new float[1];
		rb = new float[2];

		sb[0] = rank;
		rb[0] = 0.0;
		rb[1] = 0.0;

		sb_gpu = TOGPU(sb, 1);
		rb_gpu = TOGPU(rb, 2);
	}

	double t1 = MPI_Wtime();
	ncclGroupStart();
	double t2 = MPI_Wtime();
	for (int r=0; r<size_gpu; r++) {
		if (sc[r] > 0) {
			ncclSend(sb_gpu, sc[r], ncclFloat, r, comm_gpu, s);
		}
		if (rc[r] > 0) {
			ncclRecv(rb_gpu, rc[r], ncclFloat, r, comm_gpu, s);
		}
	}
	double t3 = MPI_Wtime();
	ncclGroupEnd();
	double t4 = MPI_Wtime();
	checkCudaErrors(cudaStreamSynchronize(s));
	double t5 = MPI_Wtime();

	printf("Time: %lf %lf %lf %lf\n", t2-t1, t3-t2, t4-t3, t5-t4);

	if (0 == rank_gpu) {
		COPYFROMGPU(rb, rb_gpu, 1);
	} else {
		COPYFROMGPU(rb, rb_gpu, 2);
	}

	if (0 == rank_gpu) {
		printf("Rank %d out of %d, gpu %d out of %d, %lf\n", rank, size, rank_gpu, size_gpu, rb[0]);
	} else {
		printf("Rank %d out of %d, gpu %d out of %d, %lf, %lf\n", rank, size, rank_gpu, size_gpu, rb[0], rb[1]);
	}

	delete [] sb;
	delete [] rb;
	NCCLCHECK(ncclCommDestroy(comm_gpu));
	MPICHECK(MPI_Comm_free(&comm_mpi));
	MPICHECK(MPI_Finalize());
#endif
	return 0;
}
