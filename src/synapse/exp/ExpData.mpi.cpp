
#include <assert.h>
#include "mpi.h"

#include "ExpData.h"

int sendExp(void *data_, int dest, int tag, MPI_Comm comm)
{
	ExpData *data = (ExpData *)data_;
	int ret = 0;
	ret = MPI_Send(&(data->num), 1, MPI_INT, dest, tag, comm);
	assert(ret == MPI_SUCCESS);

    ret = MPI_Send(data->s, data->num, MPI_U_REAL, dest, tag+1, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->weight, data->num, MPI_U_REAL, dest, tag+2, comm);
	assert(ret == MPI_SUCCESS);
    ret = MPI_Send(data->g, data->num, MPI_U_REAL, dest, tag+3, comm);
	assert(ret == MPI_SUCCESS);

	return ret;
}

void * recvExp(int src, int tag, MPI_Comm comm)
{
	ExpData *net = (ExpData *)mallocExp();
	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(&(net->num), 1, MPI_INT, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);

	allocExpPara(net, net->num);

    ret = MPI_Recv(net->weight, net->num, MPI_U_REAL, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->weight, net->num, MPI_U_REAL, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);
    ret = MPI_Recv(net->weight, net->num, MPI_U_REAL, src, tag+3, comm, &status);
	assert(ret==MPI_SUCCESS);

	return net;
}
