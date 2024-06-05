#include <stdlib.h>
#include <string.h>
#include "../../../msg_utils/helper/helper_gpu.h"
#include "NMDASynData.h"
#include <helper_c.h>

void *cudaMallocNMDASyn()
{
	void *ret = NULL;
	checkCudaErrors(cudaMalloc((void**)&(ret), sizeof(NMDASynData)*1));
	checkCudaErrors(cudaMemset(ret, 0, sizeof(NMDASynData)*1));
	return ret;
}

void *cudaAllocNMDASyn(void *pCPU, size_t num)
{
	void *ret = cudaMallocNMDASyn();
	void *tmp = cudaAllocNMDASynPara(pCPU, num);
	checkCudaErrors(cudaMemcpy(ret, tmp, sizeof(NMDASynData)*1, cudaMemcpyHostToDevice));
	free(tmp);
	tmp = NULL;
	return ret;
}

void *cudaAllocNMDASynPara(void *pCPU, size_t num)
{
	NMDASynData *p = (NMDASynData*)pCPU;
	NMDASynData *ret = (NMDASynData*)malloc(sizeof(NMDASynData)*1);
	memset(ret, 0, sizeof(NMDASynData)*1);

	checkCudaErrors(cudaMalloc((void**)&(ret->g), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->g, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->g, p->g, sizeof(real)*num, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&(ret->M_ca_coeff), sizeof(real)*num));
    checkCudaErrors(cudaMemset(ret->M_ca_coeff, 0, sizeof(real)*num));
    checkCudaErrors(cudaMemcpy(ret->M_ca_coeff, p->M_ca_coeff, sizeof(real)*num, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&(ret->M_c), sizeof(real)*num));
    checkCudaErrors(cudaMemset(ret->M_c, 0, sizeof(real)*num));
    checkCudaErrors(cudaMemcpy(ret->M_c, p->M_c, sizeof(real)*num, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&(ret->E_syn), sizeof(real)*num));
    checkCudaErrors(cudaMemset(ret->E_syn, 0, sizeof(real)*num));
    checkCudaErrors(cudaMemcpy(ret->E_syn, p->E_syn, sizeof(real)*num, cudaMemcpyHostToDevice));

	return ret;
}

int cudaFetchNMDASyn(void *pCPU, void *pGPU, size_t num)
{
	NMDASynData *pTmp = (NMDASynData*)malloc(sizeof(NMDASynData)*1);
	memset(pTmp, 0, sizeof(NMDASynData)*1);
	checkCudaErrors(cudaMemcpy(pTmp, pGPU, sizeof(NMDASynData)*1, cudaMemcpyDeviceToHost));

	cudaNMDASynParaFromGPU(pCPU, pTmp, num);
	return 0;
}

int cudaNMDASynParaToGPU(void *pCPU, void *pGPU, size_t num)
{
	NMDASynData *pC = (NMDASynData*)pCPU;
	NMDASynData *pG = (NMDASynData*)pGPU;

	checkCudaErrors(cudaMemcpy(pG->g, pC->g, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pG->M_ca_coeff, pC->M_ca_coeff, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pG->M_c, pC->M_c, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pG->E_syn, pC->E_syn, sizeof(real)*num, cudaMemcpyHostToDevice));

	return 0;
}

int cudaNMDASynParaFromGPU(void *pCPU, void *pGPU, size_t num)
{
	NMDASynData *pC = (NMDASynData*)pCPU;
	NMDASynData *pG = (NMDASynData*)pGPU;

	checkCudaErrors(cudaMemcpy(pC->g, pG->g, sizeof(real)*num, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pC->M_ca_coeff, pG->M_ca_coeff, sizeof(real)*num, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pC->M_c, pG->M_c, sizeof(real)*num, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pC->E_syn, pG->E_syn, sizeof(real)*num, cudaMemcpyDeviceToHost));

	return 0;
}

int cudaFreeNMDASyn(void *pGPU)
{
	NMDASynData *tmp = (NMDASynData*)malloc(sizeof(NMDASynData)*1);
	memset(tmp, 0, sizeof(NMDASynData)*1);
	checkCudaErrors(cudaMemcpy(tmp, pGPU, sizeof(NMDASynData)*1, cudaMemcpyDeviceToHost));
	cudaFreeNMDASynPara(tmp);

	tmp = free_c(tmp);
	cudaFree(pGPU); pGPU = NULL;

	return 0;
}

int cudaFreeNMDASynPara(void *pGPU)
{
	NMDASynData *p = (NMDASynData*)pGPU;

	cudaFree(p->g); p->g = NULL;
    cudaFree(p->M_ca_coeff); p->M_ca_coeff = NULL;
    cudaFree(p->M_c); p->M_c = NULL;
    cudaFree(p->E_syn); p->E_syn = NULL;

	return 0;
}

