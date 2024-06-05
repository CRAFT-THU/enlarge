
#ifndef NMDASYNDATA_H
#define NMDASYNDATA_H

#include <stdio.h>

#include "../../base/type.h"
#include "../../net/Connection.h"
#include "../../utils/BlockSize.h"
#include "mpi.h"

struct NMDASynData {
    bool is_view;
    size_t num;

    /* constant */
    real *g;           // ? g_max
    real *M_ca_coeff;  // ? Mg / 3.57 * -0.062
    real *M_c;         // ? 1 + Mg / 3.57
    real *E_syn;       // ? E_syn

    /**
     * I = g * (E_syn - V) / (M_c + M_ca_coeff * V)
     */
};

size_t getNMDASynSize();
void *mallocNMDASyn();
void *allocNMDASyn(size_t num);
int allocNMDASynPara(void *pCPU, size_t num);
int freeNMDASyn(void *pCPU);
int freeNMDASynPara(void *pCPU);
void updateNMDASyn(Connection *, void *, real *, uinteger_t *, uinteger_t *,
                   size_t, size_t, size_t, int);
int saveNMDASyn(void *pCPU, size_t num, const string &path);
void *loadNMDASyn(size_t num, const string &path);
bool isEqualNMDASyn(void *p1, void *p2, size_t num, uinteger_t *shuffle1 = NULL,
                    uinteger_t *shuffle2 = NULL);

// int shuffleNMDASyn(void *p, uinteger_t *shuffle, size_t num);

void *cudaMallocNMDASyn();
void *cudaAllocNMDASyn(void *pCPU, size_t num);
void *cudaAllocNMDASynPara(void *pCPU, size_t num);
int cudaFreeNMDASyn(void *pGPU);
int cudaFreeNMDASynPara(void *pGPU);
int cudaFetchNMDASyn(void *pCPU, void *pGPU, size_t num);
int cudaNMDASynParaToGPU(void *pCPU, void *pGPU, size_t num);
int cudaNMDASynParaFromGPU(void *pCPU, void *pGPU, size_t num);
void cudaUpdateNMDASyn(Connection *conn, void *data, real *buffer,
                       uinteger_t *firedTable, uinteger_t *firedTableSizes,
                       size_t firedTableCap, size_t num, size_t start_id, int t,
                       BlockSize *pSize);

int sendNMDASyn(void *data, int dest, int tag, MPI_Comm comm);
void *recvNMDASyn(int src, int tag, MPI_Comm comm);

#endif /* NMDASYNDATA_H */
