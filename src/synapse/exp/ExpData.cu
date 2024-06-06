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

    checkCudaErrors(cudaMalloc((void**)&(ret->s), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->s, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->s, p->s, sizeof(real)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(ret->weight), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->weight, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->weight, p->weight, sizeof(real)*num, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&(ret->g), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->g, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->g, p->g, sizeof(real)*num, cudaMemcpyHostToDevice));

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

    checkCudaErrors(cudaMemcpy(pG->s, pC->s, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->weight, pC->weight, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pG->g, pC->g, sizeof(real)*num, cudaMemcpyHostToDevice));

	return 0;
}

int cudaExpParaFromGPU(void *pCPU, void *pGPU, size_t num)
{
	ExpData *pC = (ExpData*)pCPU;
	ExpData *pG = (ExpData*)pGPU;

    checkCudaErrors(cudaMemcpy(pC->s, pG->s, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->weight, pG->weight, sizeof(real)*num, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pC->g, pG->g, sizeof(real)*num, cudaMemcpyDeviceToHost));

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

    cudaFree(p->s);
    p->s = NULL;
	cudaFree(p->weight);
	p->weight = NULL;
    cudaFree(p->g);
    p->g = NULL;

	return 0;
}

