/* This header file is writen by qp09
 * usually just for fun
 * Sat December 24 2016
 */
#ifndef MACROS_H
#define MACROS_H

#include <stddef.h>
#include "BlockSize.h"

#define NEURON_GPU_FUNC_DEFINE(name) \
	void* create##name(); \
	size_t get##name##Size(); \
	int alloc##name(void *pCpu, int N); \
	int free##name(void *pCpu); \
	int mpiSend##name(void *data, int rank, int offset, int size); \
	int mpiRecv##name(void **data, int rank, int size); \
	int cudaAlloc##name(void *pCpu, void *pGpu, int num); \
	void cudaUpdate##name(void *data, real *currentE, real *currentI, int *fireTable, int num, int start_id, int t, BlockSize *pSize); \
	int cudaFree##name(void *pGpu); 

#define SYNAPSE_GPU_FUNC_DEFINE(name) \
	void *create##name(); \
	size_t get##name##Size(); \
	int alloc##name(void *pSynapses, int S); \
	int free##name(void *pSynapses); \
	int add##name##Connection(void *pCpu, int *pSynapsesDst); \
	int mpiSend##name(void *data, int rank, int offset, int size); \
	int mpiRecv##name(void **data, int rank, int size); \
	int cudaAlloc##name(void *pCpu, void *pGpu, int num); \
	void cudaUpdate##name(void *data, real *currentE, real *currentI, int *fireTable, int num, int start_id, int t, BlockSize *pSize); \
	int cudaFree##name(void *pGpu);

#define NEURON_GPU_FUNC_BASIC(name) \
void* create##name() \
{ \
	return malloc(sizeof(G##name##Neurons)); \
} \
 \
size_t get##name##Size() \
{ \
	return sizeof(G##name##Neurons); \
} \

#define SYNAPSE_GPU_FUNC_BASIC(name) \
void *create##name() \
{ \
	return malloc(sizeof(G##name##Synapses)); \
} \
 \
size_t get##name##Size() \
{ \
	return sizeof(G##name##Synapses); \
} \
 \
int add##name##Connection(void *pCpu, int *pSynapsesDst) \
{ \
	G##name##Synapses *p = (G##name##Synapses*)pCpu; \
	p->p_dst = pSynapsesDst; \
	\
	return 0; \
}


#endif /* MACROS_H */

