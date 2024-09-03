/* This header file is writen by qp09
 * usually just for fun
 * Thu February 16 2017
 */
#ifndef HELPER_GPU_H
#define HELPER_GPU_H

#include <assert.h>

#include "helper_cuda.h"

const int MAX_BLOCK_SIZE = 512;

#define GPU_CHECK(val) check ( (val), #val, __FILE__, __LINE__)
#define _CHECK(val, a, b) check ( (val), #val, (a), (b))

#define gpu_free_clear(var) var = gpuFree(var)
#define gpuFreeClear gpu_free_clear

#ifdef DEBUG
#define TOGPU(cpu, size) toGPU( (cpu), size, "TOGPU", #cpu, __FILE__, __LINE__)
#define COPYTOGPU(gpu, cpu, size) toGPU( (gpu), (cpu), size, "COPYTOGPU", #gpu, #cpu, __FILE__, __LINE__)

#define FROMGPU(gpu, size) fromGPU( (gpu), size,"TOGPU", #gpu, __FILE__, __LINE__)
#define COPYFROMGPU(cpu, gpu, size) fromGPU( (cpu), (gpu), size, "COPYTOGPU", #cpu, #gpu, __FILE__, __LINE__)
#else
#define TOGPU(cpu, size) toGPU( (cpu), size)
#define COPYTOGPU(gpu, cpu, size) toGPU( (gpu), (cpu), size)

#define FROMGPU(gpu, size) fromGPU( (gpu), size)
#define COPYFROMGPU(cpu, gpu, size) fromGPU( (cpu), (gpu), size)
#endif

inline void gpuSetDevice(int device = 0) {
	assert(device >= 0);
    checkCudaErrors(cudaSetDevice(device));
}

inline int gpuGetDevice() {
	int device = -1;
    checkCudaErrors(cudaGetDevice(&device));
	return device;
}

template<typename T>
T* hostMalloc(size_t size = 1)
{
	T * ret;
	checkCudaErrors(cudaMallocHost((void**)&(ret), sizeof(T) * size));
	checkCudaErrors(cudaMemset(ret, 0, sizeof(T)*(size)));
	return ret;
}

template<typename T>
T* gpuMalloc(size_t size = 1)
{ 
	T * ret;
	checkCudaErrors(cudaMalloc((void**)&(ret), sizeof(T) * size));
	checkCudaErrors(cudaMemset(ret, 0, sizeof(T)*(size)));
	return ret;
}

template<typename T>
void gpuMemset(T* array, int c, size_t size = 1)
{ 
	assert(array);
	checkCudaErrors(cudaMemset(array, c, sizeof(T)*(size)));
}

template<typename T>
T* hostFree(T* cpu)
{
	checkCudaErrors(cudaFreeHost(cpu));
	return NULL;
}

template<typename T>
T* gpuFree(T* gpu)
{
	checkCudaErrors(cudaFree(gpu));
	return NULL;
}

template<typename T>
void gpuMemcpyPeer(T*data_d, int dst, T*data_s, int src, size_t size = 1)
{
	assert(data_d);
	assert(data_s);
	checkCudaErrors(cudaMemcpyPeer(data_d, dst, data_s, src, sizeof(T)*(size)));
}

template<typename T>
void gpuMemcpy(T*data_d, T*data_s, size_t size = 1)
{
	assert(data_d);
	assert(data_s);
	checkCudaErrors(cudaMemcpy(data_d, data_s, sizeof(T)*(size), cudaMemcpyDeviceToDevice));
}

template<typename T>
T* toGPU(T* cpu, size_t size, const char *const func, const char *const c_name, const char *const file, int const line)
{
	if (!cpu) {
		printf("Warn: null cpu pointer at %s:%d %s\n", file, line, c_name);
		return NULL;
	}

	if (size <= 0) {
		return NULL;
	}

	T * ret;

	check(cudaMalloc((void**)&(ret), sizeof(T) * size), func, file, line);
	check(cudaMemcpy(ret, cpu, sizeof(T)*size, cudaMemcpyHostToDevice), func, file, line);

	return ret;
}


template<typename T>
T* toGPU(T* cpu, size_t size)
{
	if (size <= 0) {
		return NULL;
	}

	assert(cpu);

	T * ret;
	checkCudaErrors(cudaMalloc((void**)&(ret), sizeof(T) * size));
	checkCudaErrors(cudaMemcpy(ret, cpu, sizeof(T)*size, cudaMemcpyHostToDevice));

	return ret;
}

template<typename T>
void toGPU(T *gpu, T *cpu, size_t size, const char *const func, const char *const g_name, const char *const c_name, const char *const file, int const line)
{
	if (!cpu) {
		printf("Warn: null cpu pointer at %s:%d %s\n", file, line, c_name);
		return;
	}

	if (!gpu) {
		printf("Warn: null gpu pointer at %s:%d %s\n", file, line, g_name);
		return;
	}

	if (size <= 0) {
		return;
	}

	check(cudaMemcpy(gpu, cpu, sizeof(T)*size, cudaMemcpyHostToDevice), func, file, line);

	return;
}

template<typename T>
void toGPU(T* gpu, T* cpu, size_t size)
{
	if (size <= 0) {
		return;
	}

	assert(cpu);
	assert(gpu);
	checkCudaErrors(cudaMemcpy(gpu, cpu, sizeof(T)*size, cudaMemcpyHostToDevice));
}

template<typename T>
T* fromGPU(T* gpu, size_t size, const char *const func, const char *const g_name, const char *const file, int const line)
{
	if (!gpu) {
		printf("Warn: null gpu pointer at %s:%d %s\n", file, line, g_name);
		return NULL;
	}

	if (size <= 0) {
		return NULL;
	}

	T * ret = static_cast<T*>(malloc(sizeof(T)*size));
	assert(ret);
	check(cudaMemcpy(ret, gpu, sizeof(T)*size, cudaMemcpyDeviceToHost), func, file, line);

	return ret;
}

template<typename T>
T* fromGPU(T* gpu, size_t size)
{
	if (size <= 0) {
		return NULL;
	}

	assert(gpu);
	T * ret = static_cast<T*>(malloc(sizeof(T)*size));
	assert(ret);
	checkCudaErrors(cudaMemcpy(ret, gpu, sizeof(T)*size, cudaMemcpyDeviceToHost));

	return ret;
}

template<typename T>
void fromGPU(T* cpu, T* gpu, size_t size, const char *const func, const char *const c_name, const char *const g_name, const char *const file, int const line)
{
	if (!cpu) {
		printf("Warn: null cpu pointer at %s:%d %s\n", file, line, c_name);
		return;
	}

	if (!gpu) {
		printf("Warn: null gpu pointer at %s:%d %s\n", file, line, g_name);
		return;
	}

	if (size <= 0) {
		return;
	}

	check(cudaMemcpy(cpu, gpu, sizeof(T)*size, cudaMemcpyDeviceToHost), func, file, line);
}

template<typename T>
void fromGPU(T* cpu, T* gpu, size_t size)
{
	if (size <= 0) {
		return;
	}

	assert(cpu);
	assert(gpu);
	checkCudaErrors(cudaMemcpy(cpu, gpu, sizeof(T)*size, cudaMemcpyDeviceToHost));
}


template<typename DATA, typename SIZE>
__device__ int merge2array(DATA *src, const SIZE size, DATA *dst, SIZE * dst_size, const SIZE dst_offset) 
{
	__shared__ volatile SIZE start_loc;
	if (threadIdx.x == 0) {
		start_loc = atomicAdd(dst_size, size);
	}
	__syncthreads();

	for (SIZE idx=threadIdx.x; idx<size; idx+=blockDim.x) {
		dst[dst_offset + start_loc + idx] = src[idx];
	}

	return 0;
}

#endif /* HELPER_GPU_H */

