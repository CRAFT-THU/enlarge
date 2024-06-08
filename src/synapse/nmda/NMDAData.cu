#include <stdlib.h>
#include <string.h>
#include "../../../msg_utils/helper/helper_c.h"
#include "../../../msg_utils/helper/helper_gpu.h"
#include "NMDAData.h"

void *cudaMallocNMDA()
{
	void *ret = NULL;
	checkCudaErrors(cudaMalloc((void**)&(ret), sizeof(NMDAData)*1));
	checkCudaErrors(cudaMemset(ret, 0, sizeof(NMDAData)*1));
	return ret;
}

void *cudaAllocNMDA(void *pCPU, size_t num)
{
	void *ret = cudaMallocNMDA();
	void *tmp = cudaAllocNMDAPara(pCPU, num);
	checkCudaErrors(cudaMemcpy(ret, tmp, sizeof(NMDAData)*1, cudaMemcpyHostToDevice));
	tmp = free_c(tmp);
	return ret;
}

void *cudaAllocNMDAPara(void *pCPU, size_t num)
{
	NMDAData *p = (NMDAData*)pCPU;
	NMDAData *ret = (NMDAData*)malloc(sizeof(NMDAData)*1);
	memset(ret, 0, sizeof(NMDAData)*1);

    checkCudaErrors(cudaMalloc((void**)&(ret->pS), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pS, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pS, p->pS, sizeof(real)*num, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&(ret->pX), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pX, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pX, p->pX, sizeof(real)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(ret->pC_decay), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pC_decay, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pC_decay, p->pC_decay, sizeof(real)*num, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&(ret->pC_rise), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pC_rise, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pC_rise, p->pC_rise, sizeof(real)*num, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&(ret->pG), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pG, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pG, p->pG, sizeof(real)*num, cudaMemcpyHostToDevice));

	return ret;
}

int cudaFetchNMDA(void *pCPU, void *pGPU, size_t num)
{
	NMDAData *pTmp = (NMDAData*)malloc(sizeof(NMDAData)*1);
	memset(pTmp, 0, sizeof(NMDAData)*1);
	checkCudaErrors(cudaMemcpy(pTmp, pGPU, sizeof(NMDAData)*1, cudaMemcpyDeviceToHost));

	cudaNMDAParaFromGPU(pCPU, pTmp, num);
	return 0;
}

int cudaNMDAParaToGPU(void *pCPU, void *pGPU, size_t num)
{
	NMDAData *pC = (NMDAData*)pCPU;
	NMDAData *pG = (NMDAData*)pGPU;

    checkCudaErrors(cudaMemcpy(pG->pS, pC->pS, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pX, pC->pX, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pG->pC_decay, pC->pC_decay, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pG->pC_rise, pC->pC_rise, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pG->pG, pC->pG, sizeof(real)*num, cudaMemcpyHostToDevice));

	return 0;
}

int cudaNMDAParaFromGPU(void *pCPU, void *pGPU, size_t num)
{
	NMDAData *pC = (NMDAData*)pCPU;
	NMDAData *pG = (NMDAData*)pGPU;

    checkCudaErrors(cudaMemcpy(pC->pS, pG->pS, sizeof(real)*num, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pC->pX, pG->pX, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pC_decay, pG->pC_decay, sizeof(real)*num, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pC->pC_rise, pG->pC_rise, sizeof(real)*num, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pC->pG, pG->pG, sizeof(real)*num, cudaMemcpyDeviceToHost));

	return 0;
}

int cudaFreeNMDA(void *pGPU)
{
	NMDAData *tmp = (NMDAData*)malloc(sizeof(NMDAData)*1);
	memset(tmp, 0, sizeof(NMDAData)*1);
	checkCudaErrors(cudaMemcpy(tmp, pGPU, sizeof(NMDAData)*1, cudaMemcpyDeviceToHost));
	cudaFreeNMDAPara(tmp);
	tmp = free_c(tmp);
	cudaFree(pGPU);
	pGPU = NULL;
	return 0;
}

int cudaFreeNMDAPara(void *pGPU)
{
	NMDAData *p = (NMDAData*)pGPU;

    cudaFree(p->pS);
    p->pS = NULL;
    cudaFree(p->pX);
    p->pX = NULL;
	cudaFree(p->pC_decay);
	p->pC_decay = NULL;
    cudaFree(p->pC_rise);
    p->pC_rise = NULL;
    cudaFree(p->pG);
    p->pG = NULL;

	return 0;
}

