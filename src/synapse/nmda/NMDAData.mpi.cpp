
#include <assert.h>
#include "mpi.h"

#include "NMDAData.h"

int sendNMDA(void *data_, int dest, int tag, MPI_Comm comm)
{
	NMDAData *data = (NMDAData *)data_;
	int ret = 0;
	ret = MPI_Send(&(data->num), 1, MPI_INT, dest, tag, comm);
	assert(ret == MPI_SUCCESS);

    ret = MPI_Send(data->pS, data->num, MPI_U_REAL, dest, tag+1, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->pX, data->num, MPI_U_REAL, dest, tag+2, comm);
	assert(ret == MPI_SUCCESS);
    ret = MPI_Send(data->pC_decay, data->num, MPI_U_REAL, dest, tag+3, comm);
    assert(ret == MPI_SUCCESS);
    ret = MPI_Send(data->pC_rise, data->num, MPI_U_REAL, dest, tag+4, comm);
    assert(ret == MPI_SUCCESS);
    ret = MPI_Send(data->pG, data->num, MPI_U_REAL, dest, tag+5, comm);
	assert(ret == MPI_SUCCESS);

	return ret;
}

void * recvNMDA(int src, int tag, MPI_Comm comm)
{
	NMDAData *net = (NMDAData *)mallocNMDA();
	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(&(net->num), 1, MPI_INT, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);

	allocNMDAPara(net, net->num);

    ret = MPI_Recv(net->pS, net->num, MPI_U_REAL, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(net->pX, net->num, MPI_U_REAL, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);
    ret = MPI_Recv(net->pC_decay, net->num, MPI_U_REAL, src, tag+3, comm, &status);
    assert(ret==MPI_SUCCESS);
    ret = MPI_Recv(net->pC_rise, net->num, MPI_U_REAL, src, tag+4, comm, &status);
    assert(ret==MPI_SUCCESS);
    ret = MPI_Recv(net->pG, net->num, MPI_U_REAL, src, tag+5, comm, &status);
	assert(ret==MPI_SUCCESS);

	return net;
}