#include <stdlib.h>
#include <string.h>
#include "../../utils/utils.h"
#include "../../../msg_utils/helper/helper_gpu.h"
#include "LIFHomoData.h"

void *cudaMallocLIFHomo()
{
	void *ret = NULL;
	checkCudaErrors(cudaMalloc((void**)&(ret), sizeof(LIFHomoData)*1));
	checkCudaErrors(cudaMemset(ret, 0, sizeof(LIFHomoData)*1));
	return ret;
}

void *cudaAllocLIFHomo(void *pCPU, size_t num)
{
	void *ret = cudaMallocLIFHomo();
	void *tmp = cudaAllocLIFHomoPara(pCPU, num);
	checkCudaErrors(cudaMemcpy(ret, tmp, sizeof(LIFHomoData)*1, cudaMemcpyHostToDevice));
	free(tmp);
	tmp = NULL;
	return ret;
}

void *cudaAllocLIFHomoPara(void *pCPU, size_t num)
{
	LIFHomoData *p = (LIFHomoData*)pCPU;
	LIFHomoData *ret = (LIFHomoData*)malloc(sizeof(LIFHomoData)*1);
	memset(ret, 0, sizeof(LIFHomoData)*1);

	ret->pRefracStep = TOGPU(p->pRefracStep, num);
	ret->pI_e = TOGPU(p->pI_e, num);
	ret->pI_i = TOGPU(p->pI_i, num);
	ret->pV_m = TOGPU(p->pV_m, num);

	// TODO: check if these are necessary
	// ret->cRefracTime = p->cRefracTime;
	// ret->cV_reset = p->cV_reset;
	// ret->cV_tmp = p->cV_tmp;
	// ret->cV_thresh = p->cV_thresh;
	// ret->cCe = p->cCe;
	// ret->cCi = p->cCi;
	// ret->cC_e = p->cC_e;
	// ret->cC_m = p->cC_m;
	// ret->cC_i = p->cC_i;
	// end TODO
	
	ret->pInput_start = TOGPU(p->pInput_start, num);
	ret->pInput = TOGPU(p->pInput, p->input_sz);
	ret->_fire_count = TOGPU(p->_fire_count, num);

	return ret;
}

int cudaFetchLIFHomo(void *pCPU, void *pGPU, size_t num)
{
	LIFHomoData *pTmp = (LIFHomoData*)malloc(sizeof(LIFHomoData)*1);
	memset(pTmp, 0, sizeof(LIFHomoData)*1);
	checkCudaErrors(cudaMemcpy(pTmp, pGPU, sizeof(LIFHomoData)*1, cudaMemcpyDeviceToHost));

	cudaLIFHomoParaFromGPU(pCPU, pTmp, num);
	return 0;
}

int cudaLIFHomoParaToGPU(void *pCPU, void *pGPU, size_t num)
{
	LIFHomoData *pC = (LIFHomoData*)pCPU;
	LIFHomoData *pG = (LIFHomoData*)pGPU;

	COPYTOGPU(pG->pRefracStep, pC->pRefracStep, num);
	COPYTOGPU(pG->pI_e, pC->pI_e, num);
	COPYTOGPU(pG->pI_i, pC->pI_i, num);
	COPYTOGPU(pG->pV_m, pC->pV_m, num);

	pG->cRefracTime = pC->cRefracTime;
	pG->cV_reset = pC->cV_reset;
	pG->cV_tmp = pC->cV_tmp;
	pG->cV_thresh = pC->cV_thresh;
	pG->cCe = pC->cCe;
	pG->cCi = pC->cCi;
	pG->cC_e = pC->cC_e;
	pG->cC_m = pC->cC_m;
	pG->cC_i = pC->cC_i;

    pG->input_sz = pC->input_sz;
	COPYTOGPU(pG->pInput_start, pC->pInput_start, num);
	COPYTOGPU(pG->pInput, pC->pInput, pC->input_sz);

	COPYTOGPU(pG->_fire_count, pC->_fire_count, num);

	return 0;
}

int cudaLIFHomoParaFromGPU(void *pCPU, void *pGPU, size_t num)
{
	LIFHomoData *pC = (LIFHomoData*)pCPU;
	LIFHomoData *pG = (LIFHomoData*)pGPU;

	COPYFROMGPU(pC->pRefracStep, pG->pRefracStep, num);
	COPYFROMGPU(pC->pI_e, pG->pI_e, num);
	COPYFROMGPU(pC->pI_i, pG->pI_i, num);
	COPYFROMGPU(pC->pV_m, pG->pV_m, num);

	pC->cRefracTime = pG->cRefracTime;
	pC->cV_reset = pG->cV_reset;
	pC->cV_tmp = pG->cV_tmp;
	pC->cV_thresh = pG->cV_thresh;
	pC->cCe = pG->cCe;
	pC->cCi = pG->cCi;
	pC->cC_e = pG->cC_e;
	pC->cC_m = pG->cC_m;
	pC->cC_i = pG->cC_i;
	
    pC->input_sz = pG->input_sz;
	COPYFROMGPU(pC->pInput_start, pG->pInput_start, num);
	COPYFROMGPU(pC->pInput, pG->pInput, pG->input_sz);

	COPYFROMGPU(pC->_fire_count, pG->_fire_count, num);

	return 0;
}

int cudaFreeLIFHomo(void *pGPU)
{
	LIFHomoData *tmp = (LIFHomoData*)malloc(sizeof(LIFHomoData)*1);
	memset(tmp, 0, sizeof(LIFHomoData)*1);
	checkCudaErrors(cudaMemcpy(tmp, pGPU, sizeof(LIFHomoData)*1, cudaMemcpyDeviceToHost));
	cudaFreeLIFHomoPara(tmp);
	tmp = free_c(tmp);
	pGPU = gpuFree(pGPU);
	return 0;
}

int cudaFreeLIFHomoPara(void *pGPU)
{
	LIFHomoData *p = (LIFHomoData*)pGPU;

	p->pRefracStep = gpuFree(p->pRefracStep);
	p->pI_e = gpuFree(p->pI_e);
	p->pI_i = gpuFree(p->pI_i);
	p->pV_m = gpuFree(p->pV_m);

	p->pInput_start = gpuFree(p->pInput_start);
	p->pInput = gpuFree(p->pInput);

	p->_fire_count = gpuFree(p->_fire_count);

	return 0;
}

int cudaLogRateLIFHomo(void *cpu, void *gpu, const char *name)
{
	LIFHomoData *c = static_cast<LIFHomoData *>(cpu);
	LIFHomoData *g = static_cast<LIFHomoData *>(gpu);

	LIFHomoData *t = FROMGPU(g, 1);
	COPYFROMGPU(c->_fire_count, t->_fire_count, c->num);
	return logRateLIFHomo(cpu, name);
}

real * cudaGetVLIFHomo(void *data) {
	LIFHomoData *c_g_lifHomo = FROMGPU(static_cast<LIFHomoData *>(data), 1);
	return c_g_lifHomo->pV_m;
}
