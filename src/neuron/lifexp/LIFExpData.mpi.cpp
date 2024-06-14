
#include <assert.h>
#include "mpi.h"

#include "LIFExpData.h"

int sendLIFExp(void *data_, int dest, int tag, MPI_Comm comm)
{
	LIFExpData * data = (LIFExpData *)data_;
	int ret = 0;
	ret = MPI_Send(&(data->num), 1, MPI_INT, dest, tag, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(data->pRefracTime, data->num, MPI_INT, dest, tag+1, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pRefracStep, data->num, MPI_INT, dest, tag+2, comm);
	assert(ret == MPI_SUCCESS);


	ret = MPI_Send(data->pV, data->num, MPI_U_REAL, dest, tag+3, comm);
	assert(ret == MPI_SUCCESS);
    ret = MPI_Send(data->pV_tmp, data->num, MPI_U_REAL, dest, tag+4, comm);
	assert(ret == MPI_SUCCESS);
    ret = MPI_Send(data->pV_thresh, data->num, MPI_U_REAL, dest, tag+5, comm);
	assert(ret == MPI_SUCCESS);
    ret = MPI_Send(data->pV_reset, data->num, MPI_U_REAL, dest, tag+6, comm);
	assert(ret == MPI_SUCCESS);

    ret = MPI_Send(data->pR, data->num, MPI_U_REAL, dest, tag+7, comm);
	assert(ret == MPI_SUCCESS);
    ret = MPI_Send(data->pC_m, data->num, MPI_U_REAL, dest, tag+8, comm);
	assert(ret == MPI_SUCCESS);
    ret = MPI_Send(data->pE, data->num, MPI_U_REAL, dest, tag+9, comm);
	assert(ret == MPI_SUCCESS);

    ret = MPI_Send(data->pI, data->num, MPI_U_REAL, dest, tag+10, comm);
	assert(ret == MPI_SUCCESS);

	return ret;
}

void * recvLIFExp(int src, int tag, MPI_Comm comm)
{
	LIFExpData *net = (LIFExpData *)mallocLIFExp();
	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(&(net->num), 1, MPI_INT, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);

	allocLIFExpPara(net, net->num);

	ret = MPI_Recv(net->pRefracTime, net->num, MPI_INT, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pRefracStep, net->num, MPI_INT, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);

	ret = MPI_Recv(net->pV, net->num, MPI_U_REAL, src, tag+3, comm, &status);
	assert(ret==MPI_SUCCESS);
    ret = MPI_Recv(net->pV_tmp, net->num, MPI_U_REAL, src, tag+4, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pV_thresh, net->num, MPI_U_REAL, src, tag+5, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pV_reset, net->num, MPI_U_REAL, src, tag+6, comm, &status);
	assert(ret==MPI_SUCCESS);
	
    ret = MPI_Recv(net->pR, net->num, MPI_U_REAL, src, tag+7, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pC_m, net->num, MPI_U_REAL, src, tag+8, comm, &status);
	assert(ret==MPI_SUCCESS);
    ret = MPI_Recv(net->pE, net->num, MPI_U_REAL, src, tag+9, comm, &status);
	assert(ret==MPI_SUCCESS);
	
    ret = MPI_Recv(net->pI, net->num, MPI_U_REAL, src, tag+10, comm, &status);
	assert(ret==MPI_SUCCESS);

	return net;
}
