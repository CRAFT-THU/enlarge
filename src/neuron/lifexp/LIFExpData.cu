#include <stdlib.h>
#include <string.h>
#include "../../../msg_utils/helper/helper_gpu.h"
#include "../../../msg_utils/helper/helper_c.h"
#include "LIFExpData.h"

void *cudaMallocLIFExp()
{
	void *ret = NULL;
	checkCudaErrors(cudaMalloc((void**)&(ret), sizeof(LIFExpData)*1));
	checkCudaErrors(cudaMemset(ret, 0, sizeof(LIFExpData)*1));
	return ret;
}

void *cudaAllocLIFExp(void *pCPU, size_t num)
{
	void *ret = cudaMallocLIFExp();
	void *tmp = cudaAllocLIFExpPara(pCPU, num);
	checkCudaErrors(cudaMemcpy(ret, tmp, sizeof(LIFExpData)*1, cudaMemcpyHostToDevice));
	free(tmp);
	tmp = NULL;
	return ret;
}

void *cudaAllocLIFExpPara(void *pCPU, size_t num)
{
	LIFExpData *p = (LIFExpData*)pCPU;
	LIFExpData *ret = (LIFExpData*)malloc(sizeof(LIFExpData)*1);
	memset(ret, 0, sizeof(LIFExpData)*1);

	checkCudaErrors(cudaMalloc((void**)&(ret->pRefracTime), sizeof(int)*num));
	checkCudaErrors(cudaMemset(ret->pRefracTime, 0, sizeof(int)*num));
	checkCudaErrors(cudaMemcpy(ret->pRefracTime, p->pRefracTime, sizeof(int)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&(ret->pRefracStep), sizeof(int)*num));
	checkCudaErrors(cudaMemset(ret->pRefracStep, 0, sizeof(int)*num));
	checkCudaErrors(cudaMemcpy(ret->pRefracStep, p->pRefracStep, sizeof(int)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(ret->pV), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pV, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pV, p->pV, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&(ret->pV_tmp), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pV_tmp, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pV_tmp, p->pV_tmp, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&(ret->pV_thresh), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pV_thresh, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pV_thresh, p->pV_thresh, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&(ret->pV_reset), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pV_reset, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pV_reset, p->pV_reset, sizeof(real)*num, cudaMemcpyHostToDevice));

    
    checkCudaErrors(cudaMalloc((void**)&(ret->pR), sizeof(real)*num));
    checkCudaErrors(cudaMemset(ret->pR, 0, sizeof(real)*num));
    checkCudaErrors(cudaMemcpy(ret->pR, p->pR, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&(ret->pC_m), sizeof(real)*num));
    checkCudaErrors(cudaMemset(ret->pC_m, 0, sizeof(real)*num));
    checkCudaErrors(cudaMemcpy(ret->pC_m, p->pC_m, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&(ret->pE), sizeof(real)*num));
    checkCudaErrors(cudaMemset(ret->pE, 0, sizeof(real)*num));
    checkCudaErrors(cudaMemcpy(ret->pE, p->pE, sizeof(real)*num, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&(ret->pI), sizeof(real)*num));
    checkCudaErrors(cudaMemset(ret->pI, 0, sizeof(real)*num));
    checkCudaErrors(cudaMemcpy(ret->pI, p->pI, sizeof(real)*num, cudaMemcpyHostToDevice));

	ret->_fire_count = TOGPU(p->_fire_count, num);

	return ret;
}

int cudaFetchLIFExp(void *pCPU, void *pGPU, size_t num)
{
	LIFExpData *pTmp = (LIFExpData*)malloc(sizeof(LIFExpData)*1);
	memset(pTmp, 0, sizeof(LIFExpData)*1);
	checkCudaErrors(cudaMemcpy(pTmp, pGPU, sizeof(LIFExpData)*1, cudaMemcpyDeviceToHost));

	cudaLIFExpParaFromGPU(pCPU, pTmp, num);
	return 0;
}

int cudaLIFExpParaToGPU(void *pCPU, void *pGPU, size_t num)
{
	LIFExpData *pC = (LIFExpData*)pCPU;
	LIFExpData *pG = (LIFExpData*)pGPU;

	checkCudaErrors(cudaMemcpy(pG->pRefracTime, pC->pRefracTime, sizeof(int)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pRefracStep, pC->pRefracStep, sizeof(int)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(pG->pV, pC->pV, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pG->pV_tmp, pC->pV_tmp, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pG->pV_thresh, pC->pV_thresh, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pG->pV_reset, pC->pV_reset, sizeof(real)*num, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(pG->pR, pC->pR, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pG->pC_m, pC->pC_m, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pG->pE, pC->pE, sizeof(real)*num, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(pG->pI, pC->pI, sizeof(real)*num, cudaMemcpyHostToDevice));

	COPYTOGPU(pG->_fire_count, pC->_fire_count, num);

	return 0;
}

int cudaLIFExpParaFromGPU(void *pCPU, void *pGPU, size_t num)
{
	LIFExpData *pC = (LIFExpData*)pCPU;
	LIFExpData *pG = (LIFExpData*)pGPU;

	checkCudaErrors(cudaMemcpy(pC->pRefracTime, pG->pRefracTime, sizeof(int)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pRefracStep, pG->pRefracStep, sizeof(int)*num, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(pC->pV, pG->pV, sizeof(real)*num, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pC->pV_tmp, pG->pV_tmp, sizeof(real)*num, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pC->pV_thresh, pG->pV_thresh, sizeof(real)*num, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pC->pV_reset, pG->pV_reset, sizeof(real)*num, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(pC->pR, pG->pR, sizeof(real)*num, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pC->pC_m, pG->pC_m, sizeof(real)*num, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pC->pE, pG->pE, sizeof(real)*num, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(pC->pI, pG->pI, sizeof(real)*num, cudaMemcpyDeviceToHost));

	COPYFROMGPU(pC->_fire_count, pG->_fire_count, num);

	return 0;
}

int cudaFreeLIFExp(void *pGPU)
{
	LIFExpData *tmp = (LIFExpData*)malloc(sizeof(LIFExpData)*1);
	memset(tmp, 0, sizeof(LIFExpData)*1);
	checkCudaErrors(cudaMemcpy(tmp, pGPU, sizeof(LIFExpData)*1, cudaMemcpyDeviceToHost));
	cudaFreeLIFExpPara(tmp);
	tmp = free_c(tmp);
	cudaFree(pGPU);
	pGPU = NULL;
	return 0;
}

int cudaFreeLIFExpPara(void *pGPU)
{
	LIFExpData *p = (LIFExpData*)pGPU;
	cudaFree(p->pRefracTime);
	p->pRefracTime = NULL;
	cudaFree(p->pRefracStep);
	p->pRefracStep = NULL;

	cudaFree(p->pV);
	p->pV = NULL;
    cudaFree(p->pV_tmp);
    p->pV_tmp = NULL;
    cudaFree(p->pV_thresh);
    p->pV_thresh = NULL;
    cudaFree(p->pV_reset);
    p->pV_reset = NULL;

    cudaFree(p->pR);
    p->pR = NULL;
    cudaFree(p->pC_m);
    p->pC_m = NULL;
    cudaFree(p->pE);
    p->pE = NULL;

    cudaFree(p->pI);
    p->pI = NULL;

	gpuFree(p->_fire_count);

	return 0;
}

int cudaLogRateLIFExp(void *cpu, void *gpu, const char *name)
{
	LIFExpData *c = static_cast<LIFExpData *>(cpu);
	LIFExpData *g = static_cast<LIFExpData *>(gpu);

	LIFExpData *t = FROMGPU(g, 1);
	COPYFROMGPU(c->_fire_count, t->_fire_count, c->num);
	return logRateLIFExp(cpu, name);
}

real * cudaGetVLIFExp(void *data) {
	LIFExpData *c_g_lif = FROMGPU(static_cast<LIFExpData *>(data), 1);
	return c_g_lif->pV;
}
