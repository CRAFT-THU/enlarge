#ifndef PROCBUF_H
#define PROCBUF_H

#include "CrossSpike.h"

// #define ASYNC

class ProcBuf {
public:
	ProcBuf(CrossSpike **cs, pthread_barrier_t *barrier, int proc_rank, int proc_num, int thread_num, int min_delay);
	~ProcBuf();

	void log_cpu(const int &thread_id, const int &time, const char *name);
	void prof();

	inline int fetch_cpu(const int &thread_id, const CrossMap *cm, const nid_t *tables, const nsize_t *table_sizes, const size_t &table_cap, const int &max_delay, const int &time) {
		return _cs[thread_id]->fetch_cpu(cm, tables, table_sizes, table_cap, _proc_num * _thread_num, max_delay, time);
	}

	int update_cpu(const int &thread_id, const int &time);

	int upload_cpu(const int &thread_id, nid_t *tables, nsize_t *table_sizes, const size_t &table_cap, const int &max_delay, const int &time);

	void to_gpu(const int &thread_id);

	inline int fetch_gpu(const int &thread_id, const CrossMap *cm, const nid_t *tables, const nsize_t *table_sizes, const size_t &table_cap, const int &max_delay, const int &time, const int &grid, const int &block) {
		return _cs[thread_id]->fetch_gpu(cm, tables, table_sizes, table_cap, _proc_num * _thread_num, max_delay, time, grid, block);
	}

	int update_gpu(const int &thread_id, const int &time);

	int upload_gpu(const int &thread_id, nid_t *tables, nsize_t *table_sizes, nsize_t *c_table_sizes, const size_t &table_cap, const int &max_delay, const int &time, const int &grid, const int &block);

	void print();

	// Cap _thread_num * _proc_num * _thread_num * (_min_delay + 1)
	integer_t *_recv_start;
	integer_t *_send_start;

	// Cap _proc_num + 1
	integer_t *_recv_offset;
	integer_t *_send_offset;

	// instance level offset view for sender
    // Cap _proc_num * dst_thread_num * src_thread_num;
	integer_t *_data_offset;
	integer_t *_data_r_offset;

	// instance level offset view for receiver
	// Cap _proc_num * dst_thread_num
	// integer_t *_rdata_offset;
	// integer_t *_sdata_offset;


	// Cap _proc_num
	integer_t *_recv_num;
	integer_t *_send_num;

	// Cap _rdata_size[_thread_num]
	nid_t *_recv_data;
	// Cap _sdata_size[_thread_num]
	nid_t *_send_data;

	// Cap _thread_num
	CrossSpike **_cs;

	// Cap _thread_num + 1;
	size_t *_rdata_size;
	size_t *_sdata_size;

	int _proc_rank;
	int _proc_num;
	int _thread_num;
	int _min_delay;

#ifdef PROF
	double _cpu_wait_gpu;
	double _gpu_wait;
	double _comm_time;
	double _cpu_time;
	double _gpu_time;
#endif

protected:
	pthread_barrier_t *_barrier;
#ifdef ASYNC
	MPI_Request _request;
#endif
};

#endif // PROCBUF_H
