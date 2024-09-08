
#include <assert.h>
#include "mpi.h"
#include "../../utils/utils.h"
#include "LIFHomoData.h"

int sendLIFHomo(void *data_, int dest, int tag, MPI_Comm comm)
{
	LIFHomoData * data = (LIFHomoData *)data_;
	int ret = 0;
	ret = MPI_Send(&(data->num), 1, MPI_INT, dest, tag, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pRefracStep, data->num, MPI_INT, dest, tag+1, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pI_e, data->num, MPI_U_REAL, dest, tag+2, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pI_i, data->num, MPI_U_REAL, dest, tag+3, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pV_m, data->num, MPI_U_REAL, dest, tag+4, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(&(data->cRefracTime), 1, MPI_INT, dest, tag+5, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&(data->cV_reset), 1, MPI_U_REAL, dest, tag+6, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&(data->cV_tmp), 1, MPI_U_REAL, dest, tag+7, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&(data->cV_thresh), 1, MPI_U_REAL, dest, tag+8, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&(data->cCe), 1, MPI_U_REAL, dest, tag+9, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&(data->cCi), 1, MPI_U_REAL, dest, tag+10, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&(data->cC_e), 1, MPI_U_REAL, dest, tag+11, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&(data->cC_m), 1, MPI_U_REAL, dest, tag+12, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&(data->cC_i), 1, MPI_U_REAL, dest, tag+13, comm);
	assert(ret == MPI_SUCCESS);

    ret = MPI_Send(&(data->input_sz), 1, MPI_INT, dest, tag+14, comm);
    assert(ret == MPI_SUCCESS);
    ret = MPI_Send(data->pInput_start, data->num, MPI_INT, dest, tag+15, comm);
    assert(ret == MPI_SUCCESS);
    ret = MPI_Send(data->pInput, data->input_sz, MPI_U_REAL, dest, tag+16, comm);
    assert(ret == MPI_SUCCESS);

	return ret;
}

void * recvLIFHomo(int src, int tag, MPI_Comm comm)
{
	LIFHomoData *net = (LIFHomoData *)mallocLIFHomo();
	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(&(net->num), 1, MPI_INT, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);

	allocLIFHomoPara(net, net->num);

	ret = MPI_Recv(net->pRefracStep, net->num, MPI_INT, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pI_e, net->num, MPI_U_REAL, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pI_i, net->num, MPI_U_REAL, src, tag+3, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pV_m, net->num, MPI_U_REAL, src, tag+4, comm, &status);
	assert(ret==MPI_SUCCESS);

	ret = MPI_Recv(&(net->cRefracTime), 1, MPI_INT, src, tag+5, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(&(net->cV_reset), 1, MPI_U_REAL, src, tag+6, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(&(net->cV_tmp), 1, MPI_U_REAL, src, tag+7, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(&(net->cV_thresh), 1, MPI_U_REAL, src, tag+8, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(&(net->cCe), 1, MPI_U_REAL, src, tag+9, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(&(net->cCi), 1, MPI_U_REAL, src, tag+10, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(&(net->cC_e), 1, MPI_U_REAL, src, tag+11, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(&(net->cC_m), 1, MPI_U_REAL, src, tag+12, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(&(net->cC_i), 1, MPI_U_REAL, src, tag+13, comm, &status);
	assert(ret==MPI_SUCCESS);

    ret = MPI_Recv(&(net->input_sz), 1, MPI_INT, src, tag+14, comm, &status);
	assert(ret==MPI_SUCCESS);
    ret = MPI_Recv(net->pInput_start, net->num, MPI_INT, src, tag+15, comm, &status);
	assert(ret==MPI_SUCCESS);
    net->pInput = malloc_c<real>(net->input_sz);
    ret = MPI_Recv(net->pInput, net->input_sz, MPI_U_REAL, src, tag+16, comm, &status);
    assert(ret==MPI_SUCCESS);

	return net;
}
