
#include "../helper/helper_c.h"
#include "../helper/helper_gpu.h"
#include "GPUManager.h"
#include "CrossSpike.h"
#include "CrossSpike.cu.h"


CrossSpike::CrossSpike(int proc_rank, int proc_num, int delay, int gpu_num, const MPI_Comm &comm) : CrossSpike(proc_rank, proc_num, delay)
{
	assert(gpu_num >= 0);
	if (gpu_num > 0) {
		_gpu_group = proc_rank / gpu_num;
	} else {
		_gpu_group = proc_rank;
	}

	MPI_Comm_split(comm, _gpu_group, _proc_rank, &_grp_comm);


	MPI_Comm_rank(_grp_comm, &_gpu_rank);
	MPI_Comm_size(_grp_comm, &_gpu_num);

	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);
	printf("Processor %s, rank %d out of %d processes, rank %d out of %d GPUs\n", processor_name, proc_rank, proc_num, _gpu_rank, gpu_num);

	if (gpu_num > 0) {
		gm.set(_gpu_rank);
		gm.lock();
	}

	if (gpu_num > 1) {
#ifdef NCCL
		ncclUniqueId id;
		checkCudaErrors(cudaStreamCreate(&_stream));

		if (0 == _gpu_rank) {
			ncclGetUniqueId(&id);
		}

		MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, _grp_comm);
		ncclCommInitRank(&_gpu_comm, _gpu_num, id, _gpu_rank);
#endif
	}
}

CrossSpike::~CrossSpike()
{
	if (_proc_num > 0) {
		free_clear(_recv_offset);
		free_clear(_recv_start);
		free_clear(_recv_num);
		free_clear(_recv_data);

		free_clear(_send_offset);
		free_clear(_send_start);
		free_clear(_send_num);
		free_clear(_send_data);
	}

	if (_gpu_array) {
		gpuFreeClear(_gpu_array->_recv_offset);
		gpuFreeClear(_gpu_array->_recv_start);
		gpuFreeClear(_gpu_array->_recv_num);
		gpuFreeClear(_gpu_array->_recv_data);

		gpuFreeClear(_gpu_array->_send_offset);
		gpuFreeClear(_gpu_array->_send_start);
		gpuFreeClear(_gpu_array->_send_num);
		gpuFreeClear(_gpu_array->_send_data);

		_gpu_array->_proc_num = 0;
		_gpu_array->_min_delay = 0;
		_gpu_array->_gpu_array = NULL;

		delete _gpu_array;
	}

	_proc_num = 0;
	_min_delay = 0;
}

int CrossSpike::to_gpu()
{
	size_t size = _min_delay * _proc_num;
	size_t num_p_1 = _proc_num + 1;

	if (!_gpu_array) {
		_gpu_array = new CrossSpike;
		_gpu_array->_proc_num = _proc_num;
		_gpu_array->_min_delay = _min_delay;

		_gpu_array->_recv_offset = TOGPU(_recv_offset, num_p_1);
		_gpu_array->_recv_start = TOGPU(_recv_start, size+_proc_num);
		_gpu_array->_recv_num = TOGPU(_recv_num, _proc_num);

		_gpu_array->_send_offset = TOGPU(_send_offset, num_p_1);
		_gpu_array->_send_start = TOGPU(_send_start, size+_proc_num);
		_gpu_array->_send_num = TOGPU(_send_num, _proc_num);

		_gpu_array->_recv_data = TOGPU(_recv_data, _recv_offset[_proc_num]);

		_gpu_array->_send_data = TOGPU(_send_data, _send_offset[_proc_num]);
	} else {
		assert(_gpu_array->_proc_num == _proc_num);
		assert(_gpu_array->_min_delay == _min_delay);

		COPYTOGPU(_gpu_array->_recv_offset, _recv_offset, num_p_1);
		COPYTOGPU(_gpu_array->_recv_start, _recv_start, size+_proc_num);
		COPYTOGPU(_gpu_array->_recv_num, _recv_num, _proc_num);

		COPYTOGPU(_gpu_array->_send_offset, _send_offset, num_p_1);
		COPYTOGPU(_gpu_array->_send_start, _send_start, size+_proc_num);
		COPYTOGPU(_gpu_array->_send_num, _send_num, _proc_num);

		COPYTOGPU(_gpu_array->_recv_data, _recv_data, _recv_offset[_proc_num]);

		COPYTOGPU(_gpu_array->_send_data, _send_data, _send_offset[_proc_num]);
	}

	return 0;
}

int CrossSpike::from_gpu()
{
	assert(_gpu_array);
	assert(_gpu_array->_proc_num == _proc_num);
	assert(_gpu_array->_min_delay == _min_delay);

	size_t size = _min_delay * _proc_num;
	size_t num_p_1 = _proc_num + 1;

	COPYFROMGPU(_recv_offset, _gpu_array->_recv_offset, num_p_1);
	COPYFROMGPU(_recv_start, _gpu_array->_recv_start, size+_proc_num);
	COPYFROMGPU(_recv_num, _gpu_array->_recv_num, _proc_num);

	COPYFROMGPU(_send_offset, _gpu_array->_send_offset, num_p_1);
	COPYFROMGPU(_send_start, _gpu_array->_send_start, size+_proc_num);
	COPYFROMGPU(_send_num, _gpu_array->_send_num, _proc_num);

	COPYFROMGPU(_recv_data, _gpu_array->_recv_data, _recv_offset[_proc_num]);
	COPYFROMGPU(_send_data, _gpu_array->_send_data, _send_offset[_proc_num]);

	return 0;
}

int CrossSpike::update_gpu(const int &time)
{
	int curr_delay = time % _min_delay;
	if (curr_delay >= _min_delay -1) {
#ifdef PROF
		double ts = MPI_Wtime();
#endif
		COPYFROMGPU(_send_start, _gpu_array->_send_start, _proc_num * (_min_delay + 1));
		if (_send_offset[_proc_num] > 0) {
			COPYFROMGPU(_send_data, _gpu_array->_send_data, _send_offset[_proc_num]);
		}
#ifdef PROF
		double te = MPI_Wtime();
		_cpu_wait_gpu += te - ts;
#endif
		msg_gpu();
	} else {
		cudaDeviceSynchronize();
		update_kernel<<<1, _proc_num>>>(_gpu_array->_send_start, _proc_num, _min_delay, curr_delay);
	}

	return 0;
}

int CrossSpike::msg_gpu()
{
#ifdef PROF
	double ts = MPI_Wtime();
#endif

	for (int i=0; i<_proc_num; i++) {
		if (_gpu_num > 1 && i/_gpu_num == _gpu_group) {
			_send_num[i] = 0;
		} else {
			_send_num[i] = _send_start[i*(_min_delay+1)+_min_delay];
		}
	}

	// int num_size = _min_delay * _proc_num;
	// print_mpi_x32(_send_num, num_size, "Send Num");
	// print_mpi_x32(_recv_num, num_size, "To Recv Num");

	int size = _min_delay + 1;
	int r_offset = _gpu_group * _gpu_num;

	if (_gpu_num > 1) {
		cudaDeviceSynchronize();
#ifdef NCCL
		ncclGroupStart();
		for (int r=0; r<_gpu_num; r++) {
			if (r != _gpu_rank) {
				ncclSend(_gpu_array->_send_start + ((r_offset + r)*size), size, NCCL_INTEGER_T, r, _gpu_comm, _stream);
				ncclRecv(_gpu_array->_recv_start + ((r_offset + r)*size), size, NCCL_INTEGER_T, r, _gpu_comm, _stream);
			}
		}
		ncclGroupEnd();
#endif
	}


	MPI_Alltoall(_send_start, _min_delay+1, MPI_INTEGER_T, _recv_start, _min_delay+1, MPI_INTEGER_T, MPI_COMM_WORLD);

	for (int i=0; i<_proc_num; i++) {
		if (_gpu_num > 1 && i/_gpu_num == _gpu_group) {
			_recv_num[i] = 0;
		} else {
			_recv_num[i] = _recv_start[i*(_min_delay+1)+_min_delay];
		}
	}
#ifdef PROF
		double te = MPI_Wtime();
		_cpu_time += te - ts;
#endif

#ifdef ASYNC
	int ret = MPI_Ialltoallv(_send_data, _send_num, _send_offset , MPI_NID_T, _recv_data, _recv_num, _recv_offset, MPI_INTEGER_T, MPI_COMM_WORLD, &_request);
	assert(ret == MPI_SUCCESS);
#else
	int ret = MPI_Alltoallv(_send_data, _send_num, _send_offset, MPI_NID_T, _recv_data, _recv_num, _recv_offset, MPI_INTEGER_T, MPI_COMM_WORLD);
	assert(ret == MPI_SUCCESS);
#endif

	if (_gpu_num > 1) {
		cudaDeviceSynchronize();

#ifdef PROF
		ts = MPI_Wtime();
#endif
#ifdef NCCL
		ncclGroupStart();
		for (int r=0; r<_gpu_num; r++) {
			int idx = r_offset + r;
			int num = _send_start[idx*(_min_delay+1)+_min_delay];
			if (num > 0) {
				ncclSend(_gpu_array->_send_data + _send_offset[idx], num, NCCL_NID_T, r, _gpu_comm, _stream);
			}
			num = _recv_start[idx*(_min_delay+1)+_min_delay];
			if (num > 0) {
				ncclRecv(_gpu_array->_recv_data + _recv_offset[idx], num, NCCL_NID_T, r, _gpu_comm, _stream);
			}
		}
		ncclGroupEnd();
#endif
#ifdef PROF
		te = MPI_Wtime();
		_gpu_time += te - ts;
#endif
	}


	// print_mpi_x32(_recv_num, num_size, "Recv Num");


	return 0;
}


int CrossSpike::fetch_gpu(const CrossMap *map, const nid_t *tables, const nsize_t *table_sizes, const size_t &table_cap, const int &proc_num, const int &max_delay, const int &time, const int &grid, const int &block)
{
	int delay_idx = time % (max_delay + 1);
	int curr_delay = time % _min_delay;
	fetch_kernel<<<grid, block>>>(_gpu_array->_send_data, _gpu_array->_send_offset, _gpu_array->_send_start, map->_gpu_array->_idx2index, map->_gpu_array->_index2ridx, tables, table_sizes, table_cap, proc_num, delay_idx, _min_delay, curr_delay);
	return 0;
}

int CrossSpike::upload_gpu(nid_t *tables, nsize_t *table_sizes, nsize_t *c_table_sizes, const size_t &table_cap, const int &max_delay, const int &time, const int &grid, const int &block)
{
#ifdef PROF
	double ts = 0, te = 0;
#endif
	int curr_delay = time % _min_delay;
	if (curr_delay >= _min_delay -1) {
#ifdef PROF
		ts = MPI_Wtime();
#endif
		COPYFROMGPU(c_table_sizes, table_sizes, max_delay+1);
#ifdef PROF
		te = MPI_Wtime();
		_gpu_wait += te - ts;
#endif

		if (_gpu_num > 1) {
			for (int d=0; d<_min_delay; d++) {
				int delay_idx = (time-_min_delay+2+d+max_delay)%(max_delay+1);
				for (int g=0; g<_gpu_num; g++) {
					int p = _gpu_group * _gpu_num + g;
					int start = _recv_start[p*(_min_delay+1)+d];
					int end = _recv_start[p*(_min_delay+1)+d+1];
					if (end > start) {
						gpuMemcpy(tables + table_cap * delay_idx + c_table_sizes[delay_idx], _gpu_array->_recv_data + _recv_offset[p] + start, end - start);
						c_table_sizes[delay_idx] += end - start;
					}
				}
			}
		}

#ifdef ASYNC
#ifdef PROF
		ts = MPI_Wtime();
#endif 
		MPI_Status status_t;
		int ret = MPI_Wait(&_request, &status_t);
		assert(ret == MPI_SUCCESS);
#ifdef PROF
		te = MPI_Wtime();
		_comm_time += te - ts;
#endif
#endif

#ifdef PROF
		ts = MPI_Wtime();
#endif 
		for (int d=0; d < _min_delay; d++) {
			int delay_idx = (time-_min_delay+2+d+max_delay)%(max_delay+1);
			for (int p = 0; p<_proc_num; p++) {
				int start = _recv_start[p*(_min_delay+1)+d];
				int end = _recv_start[p*(_min_delay+1)+d+1];
				if (end > start && (p/_gpu_num != _gpu_group)) {
					assert(c_table_sizes[delay_idx] + end - start <= table_cap);
					COPYTOGPU(tables + table_cap*delay_idx + c_table_sizes[delay_idx], _recv_data + _recv_offset[p] + start, end-start);
					c_table_sizes[delay_idx] += end - start;
				}
			}
		}
		COPYTOGPU(table_sizes, c_table_sizes, max_delay+1);
#ifdef PROF
		te = MPI_Wtime();
		_gpu_time += te - ts;
#endif

		{ // Reset
			gpuMemset(_gpu_array->_recv_start, 0, _min_delay * _proc_num + _proc_num);
			gpuMemset(_gpu_array->_send_start, 0, _min_delay * _proc_num + _proc_num);

			memset_c(_recv_num, 0, _proc_num);
			memset_c(_send_num, 0, _proc_num);
		}
	}

	return 0;
}

int CrossSpike::log_gpu(int time, const char *name)
{
	log_cpu(time, name);

	string s(name);

	integer_t *start = FROMGPU(_gpu_array->_send_start, _min_delay * _proc_num + _proc_num);

	integer_t *num = FROMGPU(_gpu_array->_send_num, _proc_num);

	{
		FILE *sf = fopen_c((s+".gpu.send").c_str(), time == 0 ? "w+" : "a+");
		fprintf(sf, "Time %d: \n", time);


		fprintf(sf, "Send start: ");
		for (size_t i=0; i<_proc_num * (_min_delay+1); i++) {
			fprintf(sf, FT_INTEGER_T " ", start[i]);
		}
		fprintf(sf, "\n");

		fprintf(sf, "Send num: ");
		for (size_t i=0; i<_proc_num; i++) {
			fprintf(sf, FT_INTEGER_T " ", num[i]);
		}
		fprintf(sf, "\n");

		nid_t * data = NULL;
		if (_send_offset[_proc_num] > 0) {
			data = FROMGPU(_gpu_array->_send_data, _send_offset[_proc_num]);
		}
		fprintf(sf, "Send data: ");
		for (size_t i=0; i<_send_offset[_proc_num]; i++) {
			fprintf(sf, FT_NID_T " ", data[i]);
		}
		fprintf(sf, "\n");

		for (int d=0; d<_min_delay; d++) {
			fprintf(sf, "Delay %d: \n", d);
			for (int g=0; g<_gpu_num; g++) {
				fprintf(sf, "GPU %d: ", g);
				int idx = _gpu_group * _gpu_num + g;
				int st = start[idx*(_min_delay+1)+d];
				int end = start[idx*(_min_delay+1)+d+1];
				for (int k=st; k<end; k++) {
					fprintf(sf, FT_NID_T " ", data[_send_offset[idx] + k]);
				}
				fprintf(sf, "\n");
			}
			fprintf(sf, "\n");
		}
		fprintf(sf, "\n");
		fflush(sf);
		fclose_c(sf);
		free_c(data);
	}

	{
		COPYFROMGPU(start, _gpu_array->_recv_start, _min_delay * _proc_num + _proc_num);

		COPYFROMGPU(num, _gpu_array->_recv_num, _proc_num);
		FILE *rf = fopen_c((s+".gpu.recv").c_str(), time == 0 ? "w+" : "a+");

		fprintf(rf, "Time %d: \n", time);

		fprintf(rf, "Recv start: ");
		for (size_t i=0; i<_proc_num * (_min_delay+1); i++) {
			fprintf(rf, FT_INTEGER_T " ", start[i]);
		}
		fprintf(rf, "\n");

		fprintf(rf, "Recv num: ");
		for (size_t i=0; i<_proc_num; i++) {
			fprintf(rf, FT_INTEGER_T " ", num[i]);
		}
		fprintf(rf, "\n");

		nid_t *data = NULL;
		if (_recv_offset[_proc_num] > 0) {
			data = FROMGPU(_gpu_array->_recv_data, _recv_offset[_proc_num]);
		}
		fprintf(rf, "Recv data: ");
		for (size_t i=0; i<_recv_offset[_proc_num]; i++) {
			fprintf(rf, FT_NID_T " ", data[i]);
		}
		fprintf(rf, "\n");

		for (int d=0; d<_min_delay; d++) {
			fprintf(rf, "Delay %d: \n", d);
			for (int g=0; g<_gpu_num; g++) {
				fprintf(rf, "GPU %d: ", g);
				int idx = _gpu_group * _gpu_num + g;
				int st = start[idx*(_min_delay+1)+d];
				int end = start[idx*(_min_delay+1)+d+1];
				for (int k=st; k<end; k++) {
					fprintf(rf, FT_NID_T " ", data[_recv_offset[idx] + k]);
				}
				// log_array_noendl(rf, _recv_data + _recv_offset[n]+start, end-start);
				fprintf(rf, "\n");
			}
			fprintf(rf, "\n");
		}
		fprintf(rf, "\n");
		fflush(rf);
		fclose_c(rf);
		free_c(data);
	}

	free_c(start);
	free_c(num);

	return 0;
}


