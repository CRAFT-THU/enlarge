
#include <assert.h>
#include "mpi.h"

#include "NMDASynData.h"

int sendNMDASyn(void *data_, int dest, int tag, MPI_Comm comm)
{
	NMDASynData *data = (NMDASynData *)data_;
	int ret = 0;
	ret = MPI_Send(&(data->num), 1, MPI_INT, dest, tag, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(data->g, data->num, MPI_U_REAL, dest, tag+1, comm);
	assert(ret == MPI_SUCCESS);
    ret = MPI_Send(data->M_ca_coeff, data->num, MPI_U_REAL, dest, tag+2, comm);
    assert(ret == MPI_SUCCESS);
    ret = MPI_Send(data->M_c, data->num, MPI_U_REAL, dest, tag+3, comm);
    assert(ret == MPI_SUCCESS);
    ret = MPI_Send(data->E_syn, data->num, MPI_U_REAL, dest, tag+4, comm);
    assert(ret == MPI_SUCCESS);

	return ret;
}

void * recvNMDASyn(int src, int tag, MPI_Comm comm)
{
	NMDASynData *net = (NMDASynData *)mallocNMDASyn();
	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(&(net->num), 1, MPI_INT, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);

	allocNMDASynPara(net, net->num);

	ret = MPI_Recv(net->g, net->num, MPI_U_REAL, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);
    ret = MPI_Recv(net->M_ca_coeff, net->num, MPI_U_REAL, src, tag+2, comm, &status);
    assert(ret==MPI_SUCCESS);
    ret = MPI_Recv(net->M_c, net->num, MPI_U_REAL, src, tag+3, comm, &status);
    assert(ret==MPI_SUCCESS);
    ret = MPI_Recv(net->E_syn, net->num, MPI_U_REAL, src, tag+4, comm, &status);
    assert(ret==MPI_SUCCESS);

	return net;
}
