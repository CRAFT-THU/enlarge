
#ifndef LIFNMDADATA_H
#define LIFNMDADATA_H

#include <stdio.h>

#include "../../base/type.h"
#include "../../net/Connection.h"
#include "../../utils/BlockSize.h"
#include "mpi.h"

struct LIFNmdaData {
    bool is_view;
    size_t num;

    int *pRefracTime;  // ? refractory time (constant)
    int *pRefracStep;

    real *pV;
    real *pV_tmp;  // ? v_rest
    real *pV_thresh;
    real *pV_reset;

    real *pR;
    real *pC_m;  // ? dt / tau
    real *pE;    // ? E_syn
    real *pM_c;  // ? Mg / 3.57

    real *pI;

    int *_fire_count;
};

size_t getLIFNmdaSize();
void *mallocLIFNmda();
void *allocLIFNmda(size_t num);
int allocLIFNmdaPara(void *pCPU, size_t num);
int freeLIFNmda(void *pCPU);
int freeLIFNmdaPara(void *pCPU);
void updateLIFNmda(Connection *, void *, real *, uinteger_t *, uinteger_t *,
                  size_t, size_t, size_t, int);
int saveLIFNmda(void *pCPU, size_t num, const string &path);
void *loadLIFNmda(size_t num, const string &path);
bool isEqualLIFNmda(void *p1, void *p2, size_t num, uinteger_t *shuffle1 = NULL,
                   uinteger_t *shuffle2 = NULL);
int copyLIFNmda(void *src, size_t s_off, void *dst, size_t d_off);
int logRateLIFNmda(void *data, const char *name);
// Type castLIFNmda(void *data);
real *getVLIFNmda(void *data);

void *cudaMallocLIFNmda();
void *cudaAllocLIFNmda(void *pCPU, size_t num);
void *cudaAllocLIFNmdaPara(void *pCPU, size_t num);
int cudaFreeLIFNmda(void *pGPU);
int cudaFreeLIFNmdaPara(void *pGPU);
int cudaFetchLIFNmda(void *pCPU, void *pGPU, size_t num);
int cudaLIFNmdaParaToGPU(void *pCPU, void *pGPU, size_t num);
int cudaLIFNmdaParaFromGPU(void *pCPU, void *pGPU, size_t num);
void cudaUpdateLIFNmda(Connection *conn, void *data, real *buffer,
                      uinteger_t *firedTable, uinteger_t *firedTableSizes,
                      size_t firedTableCap, size_t num, size_t start_id, int t,
                      BlockSize *pSize);
int cudaLogRateLIFNmda(void *cpu, void *gpu, const char *name);
real *cudaGetVLIFNmda(void *data);

int sendLIFNmda(void *data, int dest, int tag, MPI_Comm comm);
void *recvLIFNmda(int src, int tag, MPI_Comm comm);

#endif /* LIFNMDADATA_H */
