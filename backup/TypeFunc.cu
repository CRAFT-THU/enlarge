/* This file is generated by scripts automatively.
 * do not change it by hand.
 */

#include "../utils/template.h"
#include "../utils/TypeFunc.h"

#include "../neuron/lif/LIFData.h"
#include "../synapse/static/StaticData.h"

void *(*cudaAllocType[])(void *pCPU, size_t num) = {
	cudaAllocLIF,
	cudaAllocStatic
};

int (*cudaFetchType[])(void *pCPU, void *pGPU, size_t num) = {
	cudaFetchLIF,
	cudaFetchStatic
};

int (*cudaFreeType[])(void *) = {
	cudaFreeLIF,
	cudaFreeStatic
};

void (*cudaUpdateType[])(Connection *, void *, real *, real *, uinteger_t *, uinteger_t*, size_t, size_t, size_t, int, BlockSize *) = {
	cudaUpdateLIF,
	cudaUpdateStatic
};
