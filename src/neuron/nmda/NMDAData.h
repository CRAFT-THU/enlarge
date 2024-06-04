
#ifndef NMDADATA_H
#define NMDADATA_H

#include <stdio.h>
#include "mpi.h"

#include "../../net/Connection.h"
#include "../../base/type.h"
#include "../../utils/BlockSize.h"

struct NMDAData {
	bool is_view;
	size_t num;

	int *pRefracTime;
	int *pRefracStep;

	real *pI_e;
	// real *pV_i;
	real *pCe;
	real *pV_reset;
	real *pV_tmp;
	real *pI_i;
	real *pV_thresh;
	real *pCi;
	real *pV_m;
	real *pC_e;
	real *pC_m;
	real *pC_i;
	// real *pV_e;
	
	int *_fire_count;
};


size_t getNMDASize();
void *mallocNMDA();
void *allocNMDA(size_t num);
int allocNMDAPara(void *pCPU, size_t num);
int freeNMDA(void *pCPU);
int freeNMDAPara(void *pCPU);
void updateNMDA(Connection *, void *, real *, uinteger_t *, uinteger_t*, size_t, size_t, size_t, int);
int saveNMDA(void *pCPU, size_t num, const string &path);
void *loadNMDA(size_t num, const string &path);
bool isEqualNMDA(void *p1, void *p2, size_t num, uinteger_t *shuffle1=NULL, uinteger_t *shuffle2=NULL);
int copyNMDA(void *src, size_t s_off, void *dst, size_t d_off);
int logRateNMDA(void *data, const char *name);
// Type castNMDA(void *data);
real * getVNMDA(void *data);

void *cudaMallocNMDA();
void *cudaAllocNMDA(void *pCPU, size_t num);
void *cudaAllocNMDAPara(void *pCPU, size_t num);
int cudaFreeNMDA(void *pGPU);
int cudaFreeNMDAPara(void *pGPU);
int cudaFetchNMDA(void *pCPU, void *pGPU, size_t num);
int cudaNMDAParaToGPU(void *pCPU, void *pGPU, size_t num);
int cudaNMDAParaFromGPU(void *pCPU, void *pGPU, size_t num);
void cudaUpdateNMDA(Connection *conn, void *data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int t, BlockSize *pSize);
int cudaLogRateNMDA(void *cpu, void *gpu, const char *name);
real * cudaGetVNMDA(void *data);

int sendNMDA(void *data, int dest, int tag, MPI_Comm comm);
void * recvNMDA(int src, int tag, MPI_Comm comm);

#endif /* NMDADATA_H */
