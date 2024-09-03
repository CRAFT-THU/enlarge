
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <string>

#include "../helper/helper_c.h"
#include "../helper/helper_array.h"
#include "msg_utils.h"
#include "CrossSpike.h"

using std::string;

CrossSpike::CrossSpike()
{
	_proc_num = 0;
	_proc_rank = 0;

	_gpu_num = 0;
	_gpu_rank = 0;
	_gpu_group = 0;

	_min_delay = 0;

	_recv_offset = NULL;
	_recv_start = NULL;
	_recv_num = NULL;
	_recv_data = NULL;

	_send_offset = NULL;
	_send_start = NULL;
	_send_num = NULL;
	_send_data = NULL;

	_gpu_array = NULL;

#ifdef PROF
	_cpu_wait_gpu = 0;
	_gpu_wait = 0;
	_gpu_time = 0;
	_cpu_time = 0;
	_comm_time = 0;
#endif
}

CrossSpike::CrossSpike(int proc_rank, int proc_num, int delay)
{
	assert(delay > 0);
	assert(proc_num > 0);
	// printf("Delay: %d\n", delay);
	// printf("Node: %d\n", proc_num);
	_proc_rank = proc_rank;
	_proc_num = proc_num;

	_gpu_rank = 0;
	_gpu_num = 0;
	_gpu_group = 0;

	_min_delay = delay;

	size_t size = delay * proc_num;
	size_t num_p_1 = proc_num + 1;

	_recv_offset = malloc_c<integer_t>(num_p_1);
	_recv_start = malloc_c<integer_t>(size+proc_num);
	_recv_num = malloc_c<integer_t>(proc_num);
	_recv_data = NULL;

	_send_offset = malloc_c<integer_t>(num_p_1);
	_send_start = malloc_c<integer_t>(size+proc_num);
	_send_num = malloc_c<integer_t>(proc_num);
	_send_data = NULL;

	_gpu_array = NULL;

	reset();

#ifdef PROF
	_cpu_wait_gpu = 0;
	_gpu_wait = 0;
	_gpu_time = 0;
	_cpu_time = 0;
#endif
}

void CrossSpike::reset()
{
	int proc_num = _proc_num;
	int size = _min_delay * _proc_num;
	memset_c(_recv_start, 0, size + proc_num);
	memset_c(_recv_num, 0, proc_num);

	memset_c(_send_start, 0, size + proc_num);
	memset_c(_send_num, 0, proc_num);
}

void CrossSpike::alloc()
{
	int data_size = _recv_offset[_proc_num];
	// printf("Data Size1: %d\n", data_size);
	if (data_size > 0) {
		// printf("Size_t: %lu\n", sizeof(int)*data_size);
		_recv_data = malloc_c<nid_t>(data_size);
	}

	data_size = _send_offset[_proc_num];
	// printf("Data Size2: %d\n", data_size);
	if (data_size > 0) {
		// printf("Size_t: %lu\n", sizeof(int)*data_size);
		_send_data = malloc_c<nid_t>(data_size);
	}
}


int CrossSpike::send(int dst, int tag, MPI_Comm comm) 
{
	int ret = 0;
	ret = MPI_Send(&(_proc_rank), 1, MPI_INT, dst, tag+1, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&(_proc_num), 1, MPI_INT, dst, tag, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(&(_gpu_rank), 1, MPI_INT, dst, tag+3, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&(_gpu_num), 1, MPI_INT, dst, tag+2, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&(_gpu_group), 1, MPI_INT, dst, tag+4, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(&(_min_delay), 1, MPI_INT, dst, tag+5, comm);
	assert(ret == MPI_SUCCESS);

	int size = _min_delay * _proc_num;
	int num_p_1 = _proc_num + 1;

	ret = MPI_Send(_recv_offset, num_p_1, MPI_INTEGER_T, dst, tag+6, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(_recv_start, size+_proc_num, MPI_INTEGER_T, dst, tag+7, comm);
	ret = MPI_Send(_recv_num, _proc_num, MPI_INTEGER_T, dst, tag+8, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(_send_offset, num_p_1, MPI_INTEGER_T, dst, tag+9, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(_send_start, size+_proc_num, MPI_INTEGER_T, dst, tag+10, comm);
	ret = MPI_Send(_send_num, _proc_num, MPI_INTEGER_T, dst, tag+11, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(_recv_data, _recv_offset[_proc_num], MPI_NID_T, dst, tag+12, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(_send_data, _send_offset[_proc_num], MPI_NID_T, dst, tag+13, comm);
	assert(ret == MPI_SUCCESS);

	return 14;
}

int CrossSpike::recv(int src, int tag, MPI_Comm comm)
{
	assert(0 == _proc_num);
	assert(0 == _min_delay);

	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(&(_proc_rank), 1, MPI_INT, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(&(_proc_num), 1, MPI_INT, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);

	ret = MPI_Recv(&(_gpu_rank), 1, MPI_INT, src, tag+3, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(&(_gpu_num), 1, MPI_INT, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(&(_gpu_group), 1, MPI_INT, src, tag+4, comm, &status);
	assert(ret==MPI_SUCCESS);


	ret = MPI_Recv(&(_min_delay), 1, MPI_INT, src, tag+5, comm, &status);
	assert(ret==MPI_SUCCESS);

	int size = _min_delay * _proc_num;
	int num_p_1 = _proc_num + 1;

	_recv_offset = malloc_c<integer_t>(num_p_1);

	_send_offset = malloc_c<integer_t>(num_p_1);

	_recv_start = malloc_c<integer_t>(size + _proc_num);
	_recv_num = malloc_c<integer_t>(_proc_num);

	_send_start = malloc_c<integer_t>(size + _proc_num);
	_send_num = malloc_c<integer_t>(_proc_num);

	// reset();

	ret = MPI_Recv(_recv_offset, num_p_1, MPI_INTEGER_T, src, tag+6, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(_recv_start, size+_proc_num, MPI_INTEGER_T, src, tag+7, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(_recv_num, _proc_num, MPI_INTEGER_T, src, tag+8, comm, &status);
	assert(ret==MPI_SUCCESS);

	ret = MPI_Recv(_send_offset, num_p_1, MPI_INTEGER_T, src, tag+9, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(_send_start, size+_proc_num, MPI_INTEGER_T, src, tag+10, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(_send_num, _proc_num, MPI_INTEGER_T, src, tag+11, comm, &status);
	assert(ret==MPI_SUCCESS);

	alloc();

	ret = MPI_Recv(_recv_data, _recv_offset[_proc_num], MPI_NID_T, src, tag+12, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(_send_data, _send_offset[_proc_num], MPI_NID_T, src, tag+13, comm, &status);
	assert(ret==MPI_SUCCESS);

	return 14;
}

int CrossSpike::save(const string &path)
{
	string name = path + "/cross.data";
	FILE *f = fopen_c(name.c_str(), "w");

	fwrite_c(&(_proc_rank), 1, f);
	fwrite_c(&(_proc_num), 1, f);

	fwrite_c(&(_gpu_rank), 1, f);
	fwrite_c(&(_gpu_num), 1, f);
	fwrite_c(&(_gpu_group), 1, f);

	fwrite_c(&(_min_delay), 1, f);

	int size = _min_delay * _proc_num;
	int num_p_1 = _proc_num + 1;

	fwrite_c(_recv_offset, num_p_1, f);
	fwrite_c(_recv_start, size+_proc_num, f);
	fwrite_c(_recv_num, _proc_num, f);


	fwrite_c(_send_offset, num_p_1, f);
	fwrite_c(_send_start, size+_proc_num, f);
	fwrite_c(_send_num, _proc_num, f);

	fwrite_c(_recv_data, _recv_offset[_proc_num], f);
	fwrite_c(_send_data, _send_offset[_proc_num], f);

	fclose_c(f);

	return 0;
}

int CrossSpike::load(const string &path)
{
	string name = path + "/cross.data";
	FILE *f = fopen_c(name.c_str(), "r");

	fread_c(&(_proc_rank), 1, f);
	fread_c(&(_proc_num), 1, f);

	fread_c(&(_gpu_rank), 1, f);
	fread_c(&(_gpu_num), 1, f);
	fread_c(&(_gpu_group), 1, f);

	fread_c(&(_min_delay), 1, f);

	int size = _min_delay * _proc_num;
	int num_p_1 = _proc_num + 1;

	_recv_offset = malloc_c<integer_t>(num_p_1);
	_recv_start = malloc_c<integer_t>(size+_proc_num);
	_recv_num = malloc_c<integer_t>(_proc_num);
	_recv_data = NULL;

	_send_offset = malloc_c<integer_t>(num_p_1);
	_send_start = malloc_c<integer_t>(size+_proc_num);
	_send_num = malloc_c<integer_t>(_proc_num);
	_send_data = NULL;

	// reset();

	fread_c(_recv_offset, _proc_num+1, f);
	fread_c(_recv_start, size+_proc_num, f);
	fread_c(_recv_num, _proc_num, f);

	fread_c(_send_offset, _proc_num+1, f);
	fread_c(_send_start, size+_proc_num, f);
	fread_c(_send_num, _proc_num, f);

	alloc();
	fread_c(_recv_data, _recv_offset[_proc_num], f);
	fread_c(_send_data, _send_offset[_proc_num], f);

	fclose_c(f);

	return 0;
}

int CrossSpike::msg_cpu()
{
	for (int i=0; i<_proc_num; i++) {
		_send_num[i] = _send_start[i*(_min_delay+1)+_min_delay];
	}

	// int num_size = _min_delay * _proc_num;
	// print_mpi_x32(_send_num, num_size, "Send Num");
	// print_mpi_x32(_recv_num, num_size, "To Recv Num");

	MPI_Alltoall(_send_start, _min_delay+1, MPI_INTEGER_T, _recv_start, _min_delay+1, MPI_INTEGER_T, MPI_COMM_WORLD);

	// print_mpi_x32(_recv_num, num_size, "Recv Num");

	for (int i=0; i<_proc_num; i++) {
		_recv_num[i] = _recv_start[i*(_min_delay+1)+_min_delay];
	}

#ifdef ASYNC
	int ret = MPI_Ialltoallv(_send_data, _send_num, _send_offset , MPI_NID_T, _recv_data, _recv_num, _recv_offset, MPI_INTEGER_T, MPI_COMM_WORLD, &_request);
	assert(ret == MPI_SUCCESS);
#else
	int ret = MPI_Alltoallv(_send_data, _send_num, _send_offset, MPI_NID_T, _recv_data, _recv_num, _recv_offset, MPI_INTEGER_T, MPI_COMM_WORLD);
	assert(ret == MPI_SUCCESS);
#endif

	return ret;
}

int CrossSpike::update_cpu(const int &time) 
{
	int curr_delay = time % _min_delay;
	if (curr_delay >= _min_delay - 1) {
		msg_cpu();
	} else {
		for (int i=0; i<_proc_num; i++) {
			_send_start[i*(_min_delay+1)+curr_delay+2] = _send_start[i*(_min_delay+1)+curr_delay+1];
		}
	}
	return 0;
}

bool CrossSpike::equal(const CrossSpike &m)
{
	bool ret = true;
	ret = ret && (_proc_rank == m._proc_rank);
	ret = ret && (_proc_num == m._proc_num);

	ret = ret && (_gpu_rank == m._gpu_rank);
	ret = ret && (_gpu_num == m._gpu_num);
	ret = ret && (_gpu_group == m._gpu_group);

	ret = ret && (_min_delay == m._min_delay);

	size_t size = _min_delay * _proc_num;
	size_t num_p_1 = _proc_num + 1;

	ret = ret && is_equal_array(_recv_offset, m._recv_offset, num_p_1);
	ret = ret && is_equal_array(_recv_start, m._recv_start, size+_proc_num);
	ret = ret && is_equal_array(_recv_num, m._recv_num, _proc_num);
	ret = ret && is_equal_array(_recv_data, m._recv_data, _recv_offset[_proc_num]);

	ret = ret && is_equal_array(_send_offset, m._send_offset, num_p_1);
	ret = ret && is_equal_array(_send_start, m._send_start, size+_proc_num);
	ret = ret && is_equal_array(_send_num, m._send_num, _proc_num);
	ret = ret && is_equal_array(_send_data, m._send_data, _send_offset[_proc_num]);

	return ret;
}

int CrossSpike::log_cpu(int time, const char *name)
{
	string s(name);

	if (time == 0) {
		FILE *f = fopen_c((s+".cs").c_str(), "w+");
		fprintf(f, "Proc rank: %d\n", _proc_rank);
		fprintf(f, "Proc num:  %d\n", _proc_num);
		fprintf(f, "GPU rank:  %d\n", _gpu_rank);
		fprintf(f, "GPU num:   %d\n", _gpu_num);
		fprintf(f, "GPU group: %d\n", _gpu_group);
		fprintf(f, "Min delay: " FT_INTEGER_T "\n", _min_delay);

		fprintf(f, "Recv offset: ");
		for (int i=0; i<_proc_num+1; i++) {
			fprintf(f, FT_INTEGER_T " ", _recv_offset[i]);
		}
		fprintf(f, "\n");

		fprintf(f, "Send offset: ");
		for (int i=0; i<_proc_num+1; i++) {
			fprintf(f, FT_INTEGER_T " ", _send_offset[i]);
		}
		fprintf(f, "\n");
		fclose_c(f);
	}

	{
		FILE *sf = fopen_c((s+".send").c_str(), time == 0 ? "w+" : "a+");
		fprintf(sf, "Time %d: \n", time);


		fprintf(sf, "Send start: ");
		for (int i=0; i<_proc_num * (_min_delay+1); i++) {
			fprintf(sf, FT_INTEGER_T " ", _send_start[i]);
		}
		fprintf(sf, "\n");

		fprintf(sf, "Send num: ");
		for (int i=0; i<_proc_num; i++) {
			fprintf(sf, FT_INTEGER_T " ", _send_num[i]);
		}
		fprintf(sf, "\n");

		fprintf(sf, "Send data: ");
		for (int i=0; i<_send_offset[_proc_num]; i++) {
			fprintf(sf, FT_NID_T " ", _send_data[i]);
		}
		fprintf(sf, "\n");

		for (int d=0; d<_min_delay; d++) {
			fprintf(sf, "Delay %d: \n", d);
			for (int n=0; n<_proc_num; n++) {
				fprintf(sf, "Proc %d: ", n);
				int start = _send_start[n*(_min_delay+1)+d];
				int end = _send_start[n*(_min_delay+1)+d+1];
				for (int k=start; k<end; k++) {
					fprintf(sf, FT_NID_T " ", _send_data[_send_offset[n] + k]);
				}
				fprintf(sf, "\n");
			}
			fprintf(sf, "\n");
		}
		fprintf(sf, "\n");
		fflush(sf);
		fclose_c(sf);
	}

	{
		FILE *rf = fopen_c((s+".recv").c_str(), time == 0 ? "w+" : "a+");

		fprintf(rf, "Time %d: \n", time);

		fprintf(rf, "Recv start: ");
		for (int i=0; i<_proc_num * (_min_delay+1); i++) {
			fprintf(rf, FT_INTEGER_T " ", _recv_start[i]);
		}
		fprintf(rf, "\n");

		fprintf(rf, "Recv num: ");
		for (int i=0; i<_proc_num; i++) {
			fprintf(rf, FT_INTEGER_T " ", _recv_num[i]);
		}
		fprintf(rf, "\n");

		fprintf(rf, "Recv data: ");
		for (int i=0; i<_recv_offset[_proc_num]; i++) {
			fprintf(rf, FT_NID_T " ", _recv_data[i]);
		}
		fprintf(rf, "\n");

		for (int d=0; d<_min_delay; d++) {
			fprintf(rf, "Delay %d: \n", d);
			for (int n=0; n<_proc_num; n++) {
				fprintf(rf, "Proc %d: ", n);
				int start = _recv_start[n*(_min_delay+1)+d];
				int end = _recv_start[n*(_min_delay+1)+d+1];
				for (int k=start; k<end; k++) {
					fprintf(rf, FT_NID_T " ", _recv_data[_recv_offset[n] + k]);
				}
				// log_array_noendl(rf, _recv_data + _recv_offset[n]+start, end-start);
				fprintf(rf, "\n");
			}
			fprintf(rf, "\n");
		}
		fprintf(rf, "\n");
		fflush(rf);
		fclose_c(rf);
	}
	return 0;
}

int CrossSpike::fetch_cpu(const CrossMap *map, const nid_t *tables, const nsize_t *table_sizes, const size_t &table_cap, const int &proc_num, const int &max_delay, const int &time)
{
	int delay_idx = time % (max_delay+1);
	int curr_delay = time % _min_delay;
	size_t fired_size = table_sizes[delay_idx];

	for (int proc=0; proc<proc_num; proc++) {
		for (size_t idx=0; idx<fired_size; idx++) {
			nid_t nid = tables[table_cap * delay_idx + idx];
			integer_t tmp = map->_idx2index[nid];
			if (tmp >= 0) {
				integer_t map_nid = map->_index2ridx[tmp*proc_num+proc];
				if (map_nid >= 0) {
					integer_t idx_t = proc * (_min_delay+1) + curr_delay + 1;
					assert(idx_t >= 0);
					_send_data[_send_offset[proc] + _send_start[idx_t]]= map_nid;
					_send_start[idx_t]++;
				}
			}
		}
	}
	return 0;
}

int CrossSpike::upload_cpu(nid_t *tables, nsize_t *table_sizes, const size_t &table_cap, const int &max_delay, const int &time)
{
	int curr_delay = time % _min_delay;
	if (curr_delay >= _min_delay - 1) {
#ifdef ASYNC
		MPI_Status status_t;
		int ret = MPI_Wait(&_request, &status_t);
		assert(ret == MPI_SUCCESS);
#endif
		for (int d = 0; d < _min_delay; d++) {
			int delay_idx = (time-_min_delay+2+d+max_delay)%(max_delay+1);
			for (int p = 0; p < _proc_num; p++) {
				int start = _recv_start[p*(_min_delay+1)+d];
				int end = _recv_start[p*(_min_delay+1)+d+1];
				for (int i=start; i<end; i++) {
					tables[table_cap*delay_idx + table_sizes[delay_idx] + i-start] = _recv_data[_recv_offset[p]+i];
				}
				table_sizes[delay_idx] += static_cast<nsize_t>(end - start);
			}
		}

		reset();
	}

	return 0;
}

void CrossSpike::prof()
{
#ifdef PROF
	printf("ProcBuf prof: %lf:%lf:%lf:%lf:%lf\n", _cpu_wait_gpu, _cpu_time, _comm_time, _gpu_time, _gpu_wait);
#endif
}
