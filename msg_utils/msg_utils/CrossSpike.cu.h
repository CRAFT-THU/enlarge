
#ifndef CROSSSPIKE_CU_H
#define CROSSSPIKE_CU_H

#include "../helper/helper_gpu.h"

template<typename TID, typename TSIZE, typename SIZE>
__global__ void fetch_kernel(TID *data, integer_t *offset, integer_t *num, const integer_t *idx2index, const integer_t *index2ridx, const TID *fired_table, const TSIZE *fired_sizes, const SIZE fired_cap, const int proc_num, const int delay_idx, const int min_delay, const int curr_delay)
{
	__shared__ TID cross_neuron_id[MAX_BLOCK_SIZE];
	__shared__ volatile int cross_cnt;

	if (threadIdx.x == 0) {
		cross_cnt = 0;
	}
	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	TSIZE fired_size = fired_sizes[delay_idx];
	for (int proc = 0; proc < proc_num; proc++) {
		for (size_t i = 0; i < fired_size; i += blockDim.x * gridDim.x) {
			size_t idx = i + tid;
			if (idx < fired_size) {
				TID nid = static_cast<TID>(fired_table[fired_cap*delay_idx + idx]);
				integer_t tmp = idx2index[nid];
				if (tmp >= 0) {
					integer_t map_nid = index2ridx[tmp*proc_num + proc];
					if (map_nid >= 0) {
						size_t test_loc = static_cast<size_t>(atomicAdd(const_cast<int*>(&cross_cnt), 1));
						if (test_loc < MAX_BLOCK_SIZE) {
							cross_neuron_id[test_loc] = static_cast<TID>(map_nid);
						}
					}
				}
			}
			__syncthreads();

			if (cross_cnt > 0) {
				int idx_t = proc * (min_delay + 1) + curr_delay + 1;
				merge2array(cross_neuron_id, cross_cnt, data, &(num[idx_t]), offset[proc]);
				if (threadIdx.x == 0) {
					cross_cnt = 0;
				}
			}
			__syncthreads();
		}
		__syncthreads();
	}
}

__global__ void update_kernel(integer_t *start, int proc_num, int min_delay, int curr_delay)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// for (int i=tid; i<proc_num; i++) {
	if (tid < proc_num) {
		start[tid*(min_delay+1)+curr_delay+2] = start[tid*(min_delay+1)+curr_delay+1];
	}
	// }
}

#endif // CROSSSPIKE_CU_H
