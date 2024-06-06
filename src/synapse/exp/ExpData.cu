#include <stdlib.h>
#include <string.h>
#include "../../../msg_utils/helper/helper_c.h"
#include "../../../msg_utils/helper/helper_gpu.h"
#include "ExpData.h"

void *cudaMallocExp()
{
	void *ret = NULL;
	checkCudaErrors(cudaMalloc((void**)&(ret), sizeof(ExpData)*1));
	checkCudaErrors(cudaMemset(ret, 0, sizeof(ExpData)*1));
	return ret;
}

void *cudaAllocExp(void *pCPU, size_t num)
{
	void *ret = cudaMallocExp();
	void *tmp = cudaAllocExpPara(pCPU, num);
	checkCudaErrors(cudaMemcpy(ret, tmp, sizeof(ExpData)*1, cudaMemcpyHostToDevice));
	tmp = free_c(tmp);
	return ret;
}

void *cudaAllocExpPara(void *pCPU, size_t num)
{
	ExpData *p = (ExpData*)pCPU;
	ExpData *ret = (ExpData*)malloc(sizeof(ExpData)*1);
	memset(ret, 0, sizeof(ExpData)*1);

    checkCudaErrors(cudaMalloc((void**)&(ret->pS), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pS, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pS, p->pS, sizeof(real)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(ret->pWeight), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pWeight, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pWeight, p->pWeight, sizeof(real)*num, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&(ret->pG), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pG, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pG, p->pG, sizeof(real)*num, cudaMemcpyHostToDevice));

	return ret;
}

int cudaFetchExp(void *pCPU, void *pGPU, size_t num)
{
	ExpData *pTmp = (ExpData*)malloc(sizeof(ExpData)*1);
	memset(pTmp, 0, sizeof(ExpData)*1);
	checkCudaErrors(cudaMemcpy(pTmp, pGPU, sizeof(ExpData)*1, cudaMemcpyDeviceToHost));

	cudaExpParaFromGPU(pCPU, pTmp, num);
	return 0;
}

int cudaExpParaToGPU(void *pCPU, void *pGPU, size_t num)
{
	ExpData *pC = (ExpData*)pCPU;
	ExpData *pG = (ExpData*)pGPU;

    checkCudaErrors(cudaMemcpy(pG->pS, pC->pS, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pWeight, pC->pWeight, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pG->pG, pC->pG, sizeof(real)*num, cudaMemcpyHostToDevice));

	return 0;
}

int cudaExpParaFromGPU(void *pCPU, void *pGPU, size_t num)
{
	ExpData *pC = (ExpData*)pCPU;
	ExpData *pG = (ExpData*)pGPU;

    checkCudaErrors(cudaMemcpy(pC->pS, pG->pS, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pWeight, pG->pWeight, sizeof(real)*num, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pC->pG, pG->pG, sizeof(real)*num, cudaMemcpyDeviceToHost));

	return 0;
}

int cudaFreeExp(void *pGPU)
{
	ExpData *tmp = (ExpData*)malloc(sizeof(ExpData)*1);
	memset(tmp, 0, sizeof(ExpData)*1);
	checkCudaErrors(cudaMemcpy(tmp, pGPU, sizeof(ExpData)*1, cudaMemcpyDeviceToHost));
	cudaFreeExpPara(tmp);
	tmp = free_c(tmp);
	cudaFree(pGPU);
	pGPU = NULL;
	return 0;
}

int cudaFreeExpPara(void *pGPU)
{
	ExpData *p = (ExpData*)pGPU;

    cudaFree(p->pS);
    p->pS = NULL;
	cudaFree(p->pWeight);
	p->pWeight = NULL;
    cudaFree(p->pG);
    p->pG = NULL;

	return 0;
}

