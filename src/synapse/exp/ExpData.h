
#ifndef EXPDATA_H
#define EXPDATA_H

#include <stdio.h>
#include "mpi.h"

#include "../../net/Connection.h"

#include "../../base/type.h"
#include "../../utils/BlockSize.h"

struct ExpData {
	bool is_view;
	size_t num;

	real *s;

	real *weight; // (1 - dt / tau)
    real *g;
};


size_t getExpSize();
void *mallocExp();
void *allocExp(size_t num);
int allocExpPara(void *pCPU, size_t num);
int freeExp(void *pCPU);
int freeExpPara(void *pCPU);
void updateExp(Connection *, void *, real *, uinteger_t *, uinteger_t*, size_t, size_t, size_t, int);
int saveExp(void *pCPU, size_t num, const string &path);
void *loadExp(size_t num, const string &path);
bool isEqualExp(void *p1, void *p2, size_t num, uinteger_t *shuffle1=NULL, uinteger_t *shuffle2=NULL);

int shuffleExp(void *p, uinteger_t *shuffle, size_t num);

void *cudaMallocExp();
void *cudaAllocExp(void *pCPU, size_t num);
void *cudaAllocExpPara(void *pCPU, size_t num);
int cudaFreeExp(void *pGPU);
int cudaFreeExpPara(void *pGPU);
int cudaFetchExp(void *pCPU, void *pGPU, size_t num);
int cudaExpParaToGPU(void *pCPU, void *pGPU, size_t num);
int cudaExpParaFromGPU(void *pCPU, void *pGPU, size_t num);
void cudaUpdateExp(Connection *conn, void *data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int t, BlockSize *pSize);

int sendExp(void *data, int dest, int tag, MPI_Comm comm);
void * recvExp(int src, int tag, MPI_Comm comm);

#endif /* EXPDATA_H */
