
#include "../third_party/cuda/helper_cuda.h"

#include "CrossNodeData.h"

CrossNodeData * copyCNDtoGPU(CrossNodeData *data)
{
	CrossNodeData *gpu = (CrossNodeData *)malloc(sizeof(CrossNodeData));
	assert(gpu != NULL);

	gpu->_node_num = data->_node_num;

	checkCudaErrors(cudaMalloc((void**)&(gpu->_recv_offset), sizeof(int)*(data->_node_num + 1)));
	checkCudaErrors(cudaMemcpy(gpu->_recv_offset, data->_recv_offset, sizeof(int)*(data->_node_num + 1), cudaMemcpyHostToDevice));


	checkCudaErrors(cudaMalloc((void**)&(gpu->_recv_num), sizeof(int)*(data->_node_num)));
	checkCudaErrors(cudaMemset(data->_recv_num, 0, sizeof(int)*(data->_node_num)));

	checkCudaErrors(cudaMalloc((void**)&(gpu->_recv_data), sizeof(int)*(data->_recv_offset[data->_node_num])));
	checkCudaErrors(cudaMemset(data->_recv_data, 0, sizeof(int)*(data->_recv_offset[data->_node_num])));

	checkCudaErrors(cudaMalloc((void**)&(gpu->_send_offset), sizeof(int)*(data->_node_num + 1)));
	checkCudaErrors(cudaMemcpy(gpu->_send_offset, data->_send_offset, sizeof(int)*(data->_node_num + 1), cudaMemcpyHostToDevice));


	checkCudaErrors(cudaMalloc((void**)&(gpu->_send_num), sizeof(int)*(data->_node_num)));
	checkCudaErrors(cudaMemset(data->_send_num, 0, sizeof(int)*(data->_node_num)));

	checkCudaErrors(cudaMalloc((void**)&(gpu->_send_data), sizeof(int)*(data->_send_offset[data->_node_num])));
	checkCudaErrors(cudaMemset(data->_send_data, 0, sizeof(int)*(data->_send_offset[data->_node_num])));

	return gpu;
}

int freeCNDGPU(CrossNodeData *data) 
{
	cudaFree(data->_recv_offset);
	cudaFree(data->_recv_num);
	cudaFree(data->_recv_data);

	cudaFree(data->_send_offset);
	cudaFree(data->_send_num);
	cudaFree(data->_send_data);

	data->_node_num = 0;
	free(data);
	data = NULL;
}
