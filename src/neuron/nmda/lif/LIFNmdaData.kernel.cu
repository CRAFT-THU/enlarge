
#include "LIFNmdaData.h"

#include "../../../msg_utils/helper/helper_gpu.h"
#include "../../gpu_utils/runtime.h"
#include "../../net/Connection.h"

// __global__ void find_lif_neuron(LIFData *data, real * currentE, real * currentI, int num, int offset)
// {
// 	__shared__ uinteger_t tActiveTable[MAX_BLOCK_SIZE];
// 	__shared__ volatile uinteger_t activeCnt;
// 
// 	if (threadIdx.x == 0) {
// 		activeCnt = 0;
// 	}
// 	__syncthreads();
// 
// 	uinteger_t tid = blockIdx.x * blockDim.x + threadIdx.x;
// 	for (uinteger_t idx = tid; idx < num; idx += blockDim.x * gridDim.x) {
// 		//bool actived = false;
// 		int testLoc = 0;
// 		bool actived = data->pRefracStep[idx] <= 0;
// 		if (actived) {
// 			testLoc = atomicAdd((int*)&activeCnt, 1);
// 			if (testLoc < MAX_BLOCK_SIZE) {
// 				tActiveTable[testLoc] = idx;
// 				actived = false;
// 			}
// 		} else {
// 			currentE[offset + idx] = 0;
// 			currentI[offset + idx] = 0;
// 			data->pRefracStep[idx] = data->pRefracStep[idx] - 1;
// 		}
// 		__syncthreads();
// 
// 		if (activeCnt >= MAX_BLOCK_SIZE) {
// 			commit2globalTable(tActiveTable, MAX_BLOCK_SIZE, gActiveTable, &gActiveTableSize, 0);
// 			if (threadIdx.x == 0) {
// 				activeCnt = 0;
// 			}
// 		}
// 		__syncthreads();
// 
// 		if (actived) {
// 			testLoc = atomicAdd((int*)&activeCnt, 1);
// 			if (testLoc < MAX_BLOCK_SIZE) {
// 				tActiveTable[testLoc] = idx;
// 				actived = false;
// 			}
// 		}
// 		__syncthreads();
// 
// 		if (activeCnt >= MAX_BLOCK_SIZE) {
// 			commit2globalTable(tActiveTable, MAX_BLOCK_SIZE, gActiveTable, &gActiveTableSize, 0);
// 			if (threadIdx.x == 0) {
// 				activeCnt = 0;
// 			}
// 		}
// 		__syncthreads();
// 
// 		if (activeCnt > 0) {
// 			commit2globalTable(tActiveTable, activeCnt, gActiveTable, &gActiveTableSize, 0);
// 			if (threadIdx.x == 0) {
// 				activeCnt = 0;
// 			}
// 		}
// 		__syncthreads();
// 	}
// }

// __global__ void update_lif_neuron(Connection *connection, LIFData *data, real *currentE, real *currentI, int *firedTable, int *firedTableSizes, int num, int offset, int time)
// {
// 	int currentIdx = time % (connection->maxDelay+1);
// 	__shared__ int fire_table_t[MAX_BLOCK_SIZE];
// 	__shared__ volatile int fire_cnt;
// 	if (threadIdx.x == 0) {
// 		fire_cnt = 0;
// 	}
// 	__syncthreads();
// 
// 	int tid = blockIdx.x * blockDim.x + threadIdx.x;
// 	for (int idx = tid; idx < gActiveTableSize; idx +=blockDim.x*gridDim.x) {
// 		bool fired = false;
// 		int testLoc = 0;
// 
// 		int nid = gActiveTable[idx];
// 		int gnid = offset + nid; 
// 
// 		//real I = sqrtf(data->pCe[nid]) * data->pI_e[nid] + sqrtf(data->pCi[nid]) * data->pI_i[nid] + data->p_i_tmp[nid];
// 
// 		//real I = currentE[gnid] + data->p_i_tmp[nid];
// 		//data->pV_m[nid] = data->pV_m[nid] * data->p_C1[nid] + data->p_C2[nid] * I;
// 
// 		data->pV_m[nid] = data->pC_m[nid] * data->pV_m[nid] + data->pV_tmp[nid] + data->pI_e[nid] * data->pC_e[nid] + data->pI_i[nid] * data->pC_i[nid];
// 
// 		//data->p_i_syn[nid] = 0;
// 
// 		data->pI_e[nid] *= data->pCe[nid];
// 		data->pI_i[nid] *= data->pCi[nid];
// 
// 		fired = data->pV_m[nid] >= data->pV_thresh[nid];
// 
// 		gFireCount[gnid] += fired;
// 
// 		if (fired) {
// 			testLoc = atomicAdd((int*)&fire_cnt, 1);
// 			if (testLoc < MAX_BLOCK_SIZE) {
// 				fire_table_t[testLoc] = gnid;
// 				fired = false;
// 			}
// 
// 			data->pRefracStep[nid] = data->pRefracTime[nid] - 1;
// 			data->pV_m[nid] = data->pV_reset[nid];
// 		} else {
// 			gXInput[gnid] += currentE[gnid] + currentI[gnid];
// 			data->pI_e[nid] += currentE[gnid];
// 			data->pI_i[nid] += currentI[gnid];
// 		}
// 
// 		currentE[gnid] = 0;
// 		currentI[gnid] = 0;
// 
// 		__syncthreads();
// 		if (fire_cnt >= MAX_BLOCK_SIZE) {
// 			commit2globalTable(fire_table_t, MAX_BLOCK_SIZE, firedTable, &firedTableSizes[currentIdx], gFiredTableCap*currentIdx);
// 			if (threadIdx.x == 0) {
// 				fire_cnt = 0;
// 			}
// 		}
// 
// 		__syncthreads();
// 
// 		if (fired) {
// 			testLoc = atomicAdd((int*)&fire_cnt, 1);
// 			if (testLoc < MAX_BLOCK_SIZE) {
// 				fire_table_t[testLoc] = gnid;
// 				fired = false;
// 			}
// 		}
// 		__syncthreads();
// 		if (fire_cnt >= MAX_BLOCK_SIZE) {
// 			commit2globalTable(fire_table_t, MAX_BLOCK_SIZE, firedTable, &firedTableSizes[currentIdx], gFiredTableCap*currentIdx);
// 			if (threadIdx.x == 0) {
// 				fire_cnt = 0;
// 			}
// 		}
// 		__syncthreads();
// 
// 		if (fire_cnt > 0) {
// 			commit2globalTable(fire_table_t, fire_cnt, firedTable, &firedTableSizes[currentIdx], gFiredTableCap*currentIdx);
// 			if (threadIdx.x == 0) {
// 				fire_cnt = 0;
// 			}
// 		}
// 
// 	}
// 	//__syncthreads();
// 	//if (threadIdx.x == 0 && blockIdx.x == 0) {
// 	//	gActiveTableSize = 0;
// 	//}
// }

__global__ void update_all_lif_nmda_neuron(Connection *connection, LIFNmdaData *data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t offset, int time)
// __global__ void update_all_lif_neuron(LIFData *data, int num, int offset, int time)
{
	int currentIdx = time % (connection->maxDelay + 1);
	__shared__ uinteger_t fire_table_t[MAX_BLOCK_SIZE];
	__shared__ volatile uinteger_t fire_cnt;

	if (threadIdx.x == 0) {
		fire_cnt = 0;
	}

	__syncthreads();

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (size_t idx = 0; idx < num; idx +=blockDim.x*gridDim.x) {
		bool fired = false;
		uinteger_t testLoc = 0;

		size_t nid = idx + tid;
		size_t gnid = offset + nid; 
		if (nid < num) {
			bool actived = data->pRefracStep[nid] <= 0;

			if (actived) {
                // ! 先更新电流、电压再判断是否发放
                real V = data->pV[nid];
                data->pI[nid] += buffer[gnid] * (data->pE[nid] - V) / (1 + data->pM_c[nid] * exp(-0.062 * V));
				data->pV[nid] = (1 - data->pC_m[nid]) * data->pV[nid] + data->pC_m[nid] * (data->pV_tmp[nid] + data->pR[nid] * data->pI[nid]);

				fired = data->pV[nid] >= data->pV_thresh[nid];

				data->_fire_count[gnid] += fired;

				if (fired) {
					testLoc = atomicAdd((uinteger_t *)&fire_cnt, 1);
					if (testLoc < MAX_BLOCK_SIZE) {
						fire_table_t[testLoc] = gnid;
						fired = false;
					}

					// data->pRefracStep[nid] = data->pRefracTime[nid] - 1; // ! 按需求严格打 time次拍 后再恢复
                    data->pRefracStep[nid] = data->pRefracTime[nid];
					data->pV[nid] = data->pV_reset[nid];
				}
			} else {
				data->pRefracStep[nid] -= 1;
			}

			buffer[gnid] = 0;
			buffer[gnid+num] = 0;
		}

		__syncthreads();
		if (fire_cnt >= MAX_BLOCK_SIZE) {
			commit2globalTable(fire_table_t, static_cast<uinteger_t>(MAX_BLOCK_SIZE), firedTable, &firedTableSizes[currentIdx], static_cast<uinteger_t>(firedTableCap*currentIdx));
			if (threadIdx.x == 0) {
				fire_cnt = 0;
			}
		}

		__syncthreads();

		if (fired) {
			testLoc = atomicAdd((uinteger_t*)&fire_cnt, 1);
			if (testLoc < MAX_BLOCK_SIZE) {
				fire_table_t[testLoc] = gnid;
				fired = false;
			}
		}
		__syncthreads();
		if (fire_cnt >= MAX_BLOCK_SIZE) {
			commit2globalTable(fire_table_t, static_cast<uinteger_t>(MAX_BLOCK_SIZE), firedTable, &firedTableSizes[currentIdx], static_cast<uinteger_t>(firedTableCap*currentIdx));
			if (threadIdx.x == 0) {
				fire_cnt = 0;
			}
		}
		__syncthreads();

	}
	if (fire_cnt > 0) {
		commit2globalTable(fire_table_t, fire_cnt, firedTable, &firedTableSizes[currentIdx], static_cast<uinteger_t>(firedTableCap*currentIdx));
		if (threadIdx.x == 0) {
			fire_cnt = 0;
		}
	}
	__syncthreads();
}

// __global__ void update_dense_lif_neuron(Connection *connection, LIFData *data, real *buffer, int *firedTable, int *firedTableSizes, int firedTableCap, int num, int offset, int time)
// {
// 	//__shared__ int fire_table_t[MAX_BLOCK_SIZE];
// 	//__shared__ volatile int fire_cnt;
// 
// 	//if (threadIdx.x == 0) {
// 	//	fire_cnt = 0;
// 	//}
// 	//__syncthreads();
// 
// 	int tid = blockIdx.x * blockDim.x + threadIdx.x;
// 	int currentIdx = time % (connection->maxDelay+1);
// 	for (int idx = tid; idx < num; idx +=blockDim.x*gridDim.x) {
// 		//bool fired = false;
// 		//int testLoc = 0;
// 
// 		int nid = idx;
// 		int gnid = offset + idx; 
// 		bool actived = data->pRefracStep[idx] <= 0;
// 
// 		if (actived) {
// 			data->pV_m[nid] = data->pC_m[nid] * data->pV_m[nid] + data->pV_tmp[nid] + data->pI_e[nid] * data->pC_e[nid] + data->pI_i[nid] * data->pC_i[nid];
// 
// 			data->pI_e[nid] *= data->pCe[nid];
// 			data->pI_i[nid] *= data->pCi[nid];
// 
// 			bool fired = data->pV_m[nid] >= data->pV_thresh[nid];
// 
// 			firedTable[firedTableCap*currentIdx + gnid] = fired;
// 
// 			data->_fire_count[gnid] += fired;
// 
// 			if (fired) {
// 				data->pRefracStep[nid] = data->pRefracTime[nid] - 1;
// 				data->pV_m[nid] = data->pV_reset[nid];
// 
// 			} else {
// 				// gXInput[gnid] += currentE[gnid] + currentI[gnid];
// 				data->pI_e[nid] += buffer[gnid];
// 				data->pI_i[nid] += buffer[gnid+num];
// 				//real input = 0, input_I = 0;
// 				//for (int i=data->p_start_E[nid]; i<data->p_start_I[nid]; i++) {
// 				//	input += currentE[i];
// 				//}
// 				//for (int i=data->p_start_I[nid]; i<data->p_end[nid]; i++) {
// 				//	input_I += currentE[i];
// 				//}
// 				//data->pI_e[nid] += input;
// 				//data->pI_i[nid] += input_I;
// 				//gXInput[gnid] += input + input_I;
// 			}
// 
// 		} else {
// 			data->pRefracStep[idx] = data->pRefracStep[idx] - 1;
// 			firedTable[firedTableCap*currentIdx + gnid] = 0;
// 		}
// 		buffer[gnid] = 0;
// 		buffer[gnid+num] = 0;
// 	}
// 	__syncthreads();
// }

void cudaUpdateLIFNmda(Connection *conn, void *data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t offset, int time, BlockSize *pSize)
{
	// find_lif_neuron<<<pSize->gridSize, pSize->blockSize>>>((LIFData*)data, currentE, currentI, num, offset);
	// update_lif_neuron<<<pSize->gridSize, pSize->blockSize>>>(conn, (LIFData*)data, currentE, currentI, firedTable, firedTableSizes, num, offset, time);
	update_all_lif_nmda_neuron<<<pSize->gridSize, pSize->blockSize>>>(conn, (LIFNmdaData*)data, buffer, firedTable, firedTableSizes, firedTableCap, num, offset, time);

}
