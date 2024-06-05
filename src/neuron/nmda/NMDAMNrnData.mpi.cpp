
#include <assert.h>
#include "mpi.h"

#include "NMDANrnData.h"

int sendNMDANrn(void *data_, int dest, int tag, MPI_Comm comm)
{
	NMDANrnData * data = (NMDANrnData *)data_;
	int ret = 0;
	ret = MPI_Send(&(data->num), 1, MPI_INT, dest, tag, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(data->s, data->num, MPI_U_REAL, dest, tag+1, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(data->x, data->num, MPI_U_REAL, dest, tag+2, comm);
	assert(ret == MPI_SUCCESS);


	ret = MPI_Send(data->coeff, data->num, MPI_U_REAL, dest, tag+3, comm);
    assert(ret == MPI_SUCCESS);
    ret = MPI_Send(data->tau_decay_rcpl, data->num, MPI_U_REAL, dest, tag+4, comm);
    assert(ret == MPI_SUCCESS);
    ret = MPI_Send(data->tau_rise_compl, data->num, MPI_U_REAL, dest, tag+5, comm);
	assert(ret == MPI_SUCCESS);

	return ret;
}

void * recvNMDANrn(int src, int tag, MPI_Comm comm)
{
	NMDANrnData *net = (NMDANrnData *)mallocNMDANrn();
	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(&(net->num), 1, MPI_INT, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);

	allocNMDANrnPara(net, net->num);

	ret = MPI_Recv(net->s, net->num, MPI_U_REAL, src, tag+1, comm, &status);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Recv(net->x, net->num, MPI_U_REAL, src, tag+2, comm, &status);
	assert(ret == MPI_SUCCESS);


	ret = MPI_Recv(net->coeff, net->num, MPI_U_REAL, src, tag+3, comm, &status);
    assert(ret == MPI_SUCCESS);
    ret = MPI_Recv(net->tau_decay_rcpl, net->num, MPI_U_REAL, src, tag+4, comm, &status);
    assert(ret == MPI_SUCCESS);
    ret = MPI_Recv(net->tau_rise_compl, net->num, MPI_U_REAL, src, tag+5, comm, &status);
    assert(ret == MPI_SUCCESS);

	return net;
}
