
#ifndef LIFEXPDATA_H
#define LIFEXPDATA_H

#include <stdio.h>
#include "mpi.h"

#include "../../net/Connection.h"
#include "../../base/type.h"
#include "../../utils/BlockSize.h"

struct LIFExpData {
	bool is_view;
	size_t num;
    
	int *pRefracTime; // ? refractory time (constant)
    int *pRefracStep;

	real* pV;
    real* pV_tmp; // ? v_rest
    real* pV_thresh;
    real* pV_reset;

    real* pR;
    real* pC_m; // ? dt / tau
    real* pE; // ? E_syn

    real* pI;
	
	int *_fire_count;
};


size_t getLIFExpSize();
void *mallocLIFExp();
void *allocLIFExp(size_t num);
int allocLIFExpPara(void *pCPU, size_t num);
int freeLIFExp(void *pCPU);
int freeLIFExpPara(void *pCPU);
void updateLIFExp(Connection *, void *, real *, uinteger_t *, uinteger_t*, size_t, size_t, size_t, int);
int saveLIFExp(void *pCPU, size_t num, const string &path);
void *loadLIFExp(size_t num, const string &path);
bool isEqualLIFExp(void *p1, void *p2, size_t num, uinteger_t *shuffle1=NULL, uinteger_t *shuffle2=NULL);
int copyLIFExp(void *src, size_t s_off, void *dst, size_t d_off);
int logRateLIFExp(void *data, const char *name);
// Type castLIFExp(void *data);
real * getVLIFExp(void *data);

void *cudaMallocLIFExp();
void *cudaAllocLIFExp(void *pCPU, size_t num);
void *cudaAllocLIFExpPara(void *pCPU, size_t num);
int cudaFreeLIFExp(void *pGPU);
int cudaFreeLIFExpPara(void *pGPU);
int cudaFetchLIFExp(void *pCPU, void *pGPU, size_t num);
int cudaLIFExpParaToGPU(void *pCPU, void *pGPU, size_t num);
int cudaLIFExpParaFromGPU(void *pCPU, void *pGPU, size_t num);
void cudaUpdateLIFExp(Connection *conn, void *data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int t, BlockSize *pSize);
int cudaLogRateLIFExp(void *cpu, void *gpu, const char *name);
real * cudaGetVLIFExp(void *data);

int sendLIFExp(void *data, int dest, int tag, MPI_Comm comm);
void * recvLIFExp(int src, int tag, MPI_Comm comm);

#endif /* LIFEXPDATA_H */
