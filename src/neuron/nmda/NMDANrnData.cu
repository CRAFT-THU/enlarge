#include <stdlib.h>
#include <string.h>
#include "../../../msg_utils/helper/helper_gpu.h"
#include "NMDANrnData.h"

void *cudaMallocNMDANrn()
{
	void *ret = NULL;
	checkCudaErrors(cudaMalloc((void**)&(ret), sizeof(NMDANrnData)*1));
	checkCudaErrors(cudaMemset(ret, 0, sizeof(NMDANrnData)*1));
	return ret;
}

void *cudaAllocNMDANrn(void *pCPU, size_t num)
{
	void *ret = cudaMallocNMDANrn();
	void *tmp = cudaAllocNMDANrnPara(pCPU, num);
	checkCudaErrors(cudaMemcpy(ret, tmp, sizeof(NMDANrnData)*1, cudaMemcpyHostToDevice));
	free(tmp);
	tmp = NULL;
	return ret;
}

void *cudaAllocNMDANrnPara(void *pCPU, size_t num)
{
	NMDANrnData *p = (NMDANrnData*)pCPU;
	NMDANrnData *ret = (NMDANrnData*)malloc(sizeof(NMDANrnData)*1);
	memset(ret, 0, sizeof(NMDANrnData)*1);

	checkCudaErrors(cudaMalloc((void**)&(ret->s), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->s, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->s, p->s, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&(ret->x), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->x, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->x, p->x, sizeof(real)*num, cudaMemcpyHostToDevice));
	
    checkCudaErrors(cudaMalloc((void**)&(ret->coeff), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->coeff, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->coeff, p->coeff, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&(ret->tau_decay_rcpl), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->tau_decay_rcpl, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->tau_decay_rcpl, p->tau_decay_rcpl, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&(ret->tau_rise_compl), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->tau_rise_compl, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->tau_rise_compl, p->tau_rise_compl, sizeof(real)*num, cudaMemcpyHostToDevice));


	// ret->_fire_count = TOGPU(p->_fire_count, num);

	return ret;
}

int cudaFetchNMDANrn(void *pCPU, void *pGPU, size_t num)
{
	NMDANrnData *pTmp = (NMDANrnData*)malloc(sizeof(NMDANrnData)*1);
	memset(pTmp, 0, sizeof(NMDANrnData)*1);
	checkCudaErrors(cudaMemcpy(pTmp, pGPU, sizeof(NMDANrnData)*1, cudaMemcpyDeviceToHost));

	cudaNMDANrnParaFromGPU(pCPU, pTmp, num);
	return 0;
}

int cudaNMDANrnParaToGPU(void *pCPU, void *pGPU, size_t num)
{
	NMDANrnData *pC = (NMDANrnData*)pCPU;
	NMDANrnData *pG = (NMDANrnData*)pGPU;

    checkCudaErrors(cudaMemcpy(pG->s, pC->s, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pG->x, pC->x, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pG->coeff, pC->coeff, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pG->tau_decay_rcpl, pC->tau_decay_rcpl, sizeof(real)*num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pG->tau_rise_compl, pC->tau_rise_compl, sizeof(real)*num, cudaMemcpyHostToDevice));

	// COPYTOGPU(pG->_fire_count, pC->_fire_count, num);

	return 0;
}

int cudaNMDANrnParaFromGPU(void *pCPU, void *pGPU, size_t num)
{
	NMDANrnData *pC = (NMDANrnData*)pCPU;
	NMDANrnData *pG = (NMDANrnData*)pGPU;

    checkCudaErrors(cudaMemcpy(pC->s, pG->s, sizeof(real)*num, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pC->x, pG->x, sizeof(real)*num, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pC->coeff, pG->coeff, sizeof(real)*num, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pC->tau_decay_rcpl, pG->tau_decay_rcpl, sizeof(real)*num, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pC->tau_rise_compl, pG->tau_rise_compl, sizeof(real)*num, cudaMemcpyDeviceToHost));
	
	// COPYFROMGPU(pC->_fire_count, pG->_fire_count, num);

	return 0;
}

int cudaFreeNMDANrn(void *pGPU)
{
	NMDANrnData *tmp = (NMDANrnData*)malloc(sizeof(NMDANrnData)*1);
	memset(tmp, 0, sizeof(NMDANrnData)*1);
	checkCudaErrors(cudaMemcpy(tmp, pGPU, sizeof(NMDANrnData)*1, cudaMemcpyDeviceToHost));
	cudaFreeNMDANrnPara(tmp);
	free(tmp);
	tmp = NULL;
	cudaFree(pGPU);
	pGPU = NULL;
	return 0;
}

int cudaFreeNMDANrnPara(void *pGPU)
{
	NMDANrnData *p = (NMDANrnData*)pGPU;
	cudaFree(p->s);
	p->s = NULL;
	cudaFree(p->x);
	p->x = NULL;
	cudaFree(p->coeff);
	p->coeff = NULL;
	cudaFree(p->tau_decay_rcpl);
    p->tau_decay_rcpl = NULL;
    cudaFree(p->tau_rise_compl);
    p->tau_rise_compl = NULL;

	// gpuFree(p->_fire_count);

	return 0;
}

// int cudaLogRateNMDANrn(void *cpu, void *gpu, const char *name)
// {
// 	NMDANrnData *c = static_cast<NMDANrnData *>(cpu);
// 	NMDANrnData *g = static_cast<NMDANrnData *>(gpu);

// 	NMDANrnData *t = FROMGPU(g, 1);
// 	COPYFROMGPU(c->_fire_count, t->_fire_count, c->num);
// 	return logRateNMDANrn(cpu, name);
// }

real * cudaGetSNMDANrn(void *data) {
	NMDANrnData *c_g_nmda = FROMGPU(static_cast<NMDANrnData *>(data), 1);
	return c_g_nmda->s;
}
