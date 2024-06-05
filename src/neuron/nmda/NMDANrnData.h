
#ifndef NMDANRNDATA_H
#define NMDANRNDATA_H

#include <stdio.h>

#include "../../base/type.h"
#include "../../net/Connection.h"
#include "../../utils/BlockSize.h"
#include "mpi.h"

struct NMDANrnData {
    bool is_view;
    size_t num;

    /* variables */
    real *s;
    real *x;

    /* constants */
    real *coeff;           // ? dt / 2
    real *tau_decay_rcpl;  // ? dt / tau_decay
    real *tau_rise_compl;  // ? 1 - (dt / tau_rise)

    /**
     * s = s + coeff * x - s * (coeff * x + tau_decay_rcpl)
     * x = x * tau_rise_compl
     */

    // int *_fire_count; // ! debug use
};

size_t getNMDANrnSize();
void *mallocNMDANrn();
void *allocNMDANrn(size_t num);
int allocNMDANrnPara(void *pCPU, size_t num);
int freeNMDANrn(void *pCPU);
int freeNMDANrnPara(void *pCPU);

// TODO: 传入的是 前驱神经元的firedTable （只读）
void updateNMDANrn(Connection *, void *, real *, uinteger_t *, uinteger_t *,
                size_t, size_t, size_t, int);

int saveNMDANrn(void *pCPU, size_t num, const string &path);
void *loadNMDANrn(size_t num, const string &path);
bool isEqualNMDANrn(void *p1, void *p2, size_t num, uinteger_t *shuffle1 = NULL,
                 uinteger_t *shuffle2 = NULL);
// int copyNMDANrn(void *src, size_t s_off, void *dst, size_t d_off);
// int logRateNMDANrn(void *data, const char *name);
// Type castNMDANrn(void *data);
real *getSNMDANrn(void *data); // ? get NMDANrnData.s

void *cudaMallocNMDANrn();
void *cudaAllocNMDANrn(void *pCPU, size_t num);
void *cudaAllocNMDANrnPara(void *pCPU, size_t num);
int cudaFreeNMDANrn(void *pGPU);
int cudaFreeNMDANrnPara(void *pGPU);
int cudaFetchNMDANrn(void *pCPU, void *pGPU, size_t num);
int cudaNMDANrnParaToGPU(void *pCPU, void *pGPU, size_t num);
int cudaNMDANrnParaFromGPU(void *pCPU, void *pGPU, size_t num);

// TODO: 传入的是 前驱神经元的firedTable （只读）
void cudaUpdateNMDANrn(Connection *conn, void *data, real *buffer,
                    uinteger_t *firedTable, uinteger_t *firedTableSizes,
                    size_t firedTableCap, size_t num, size_t start_id, int t,
                    BlockSize *pSize);

// int cudaLogRateNMDANrn(void *cpu, void *gpu, const char *name);
real *cudaGetVNMDANrn(void *data);

int sendNMDANrn(void *data, int dest, int tag, MPI_Comm comm);
void *recvNMDANrn(int src, int tag, MPI_Comm comm);

#endif /* NMDANRNDATA_H */
