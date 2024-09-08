
#ifndef LIFHOMODATA_H
#define LIFHOMODATA_H

#include <stdio.h>
#include "mpi.h"

#include "../../net/Connection.h"
#include "../../base/type.h"
#include "../../utils/BlockSize.h"

struct LIFHomoData {
	bool is_view;
	size_t num;

// vars
	int *pRefracStep;
	real *pI_e;
	real *pI_i;
	real *pV_m;

// consts
	int cRefracTime;
	real cV_reset;
	real cV_tmp;
	real cV_thresh;
	real cCe;
	real cCi;
	real cC_e;
	real cC_m;
	real cC_i;

	/* refer to LIFData.h */
    int input_sz;
	int *pInput_start;
	real *pInput;

	int *_fire_count;
};


size_t getLIFHomoSize();
void *mallocLIFHomo();
void *allocLIFHomo(size_t num);
int allocLIFHomoPara(void *pCPU, size_t num);
int freeLIFHomo(void *pCPU);
int freeLIFHomoPara(void *pCPU);
void updateLIFHomo(Connection *, void *, real *, uinteger_t *, uinteger_t*, size_t, size_t, size_t, int);
int saveLIFHomo(void *pCPU, size_t num, const string &path);
void *loadLIFHomo(size_t num, const string &path);
bool isEqualLIFHomo(void *p1, void *p2, size_t num, uinteger_t *shuffle1=NULL, uinteger_t *shuffle2=NULL);
int copyLIFHomo(void *src, size_t s_off, void *dst, size_t d_off);
int logRateLIFHomo(void *data, const char *name);
// Type castLIFHomo(void *data);
real * getVLIFHomo(void *data);

void *cudaMallocLIFHomo();
void *cudaAllocLIFHomo(void *pCPU, size_t num);
void *cudaAllocLIFHomoPara(void *pCPU, size_t num);
int cudaFreeLIFHomo(void *pGPU);
int cudaFreeLIFHomoPara(void *pGPU);
int cudaFetchLIFHomo(void *pCPU, void *pGPU, size_t num);
int cudaLIFHomoParaToGPU(void *pCPU, void *pGPU, size_t num);
int cudaLIFHomoParaFromGPU(void *pCPU, void *pGPU, size_t num);
void cudaUpdateLIFHomo(Connection *conn, void *data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int t, BlockSize *pSize);
int cudaLogRateLIFHomo(void *cpu, void *gpu, const char *name);
real * cudaGetVLIFHomo(void *data);

int sendLIFHomo(void *data, int dest, int tag, MPI_Comm comm);
void * recvLIFHomo(int src, int tag, MPI_Comm comm);

#endif /* LIFHOMODATA_H */
