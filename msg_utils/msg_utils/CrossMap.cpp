
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>

#include "../helper/helper_c.h"
#include "../helper/helper_array.h"
#include "CrossMap.h"

using std::string;

CrossMap::CrossMap()
{
	_idx2index = NULL;
	_index2ridx = NULL;
	_cross_size = 0;
	_num = 0;

	_gpu_array = NULL;
}

CrossMap::CrossMap(size_t num, size_t cross_num, size_t node_num) : CrossMap(num, cross_num * node_num)
{
}

CrossMap:: CrossMap(size_t num, size_t cross_size)
{
	_num = num;

	_idx2index = malloc_c<integer_t>(num);
	std::fill(_idx2index, _idx2index + num, -1);

	_cross_size = cross_size;
	if (_cross_size > 0) {
		_index2ridx = malloc_c<integer_t>(cross_size);
		std::fill(_index2ridx, _index2ridx + cross_size, -1);
	} else {
		_index2ridx = NULL;
	}

	_gpu_array = NULL;
}

int CrossMap::send(int dest, int tag, MPI_Comm comm) 
{
	int ret = 0;
	ret = MPI_Send(&_num, 1, MPI_SIZE_T, dest, tag, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&_cross_size, 1, MPI_SIZE_T, dest, tag, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(_idx2index, _num, MPI_INTEGER_T, dest, tag+1, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(_index2ridx, _cross_size, MPI_INTEGER_T, dest, tag+2, comm);
	assert(ret == MPI_SUCCESS);
	return ret;
}

int CrossMap::recv(int src, int tag, MPI_Comm comm)
{
	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(&_num, 1, MPI_SIZE_T, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(&_cross_size, 1, MPI_SIZE_T, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);
	if (_idx2index) {
		free_clear(_idx2index);
	}
	_idx2index = malloc_c<integer_t>(_num);
	ret = MPI_Recv(_idx2index, _num, MPI_INTEGER_T, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);
	_index2ridx = malloc_c<integer_t>(_cross_size);
	ret = MPI_Recv(_index2ridx, _cross_size, MPI_INTEGER_T, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);

	return 0;
}

int CrossMap::save(FILE *f)
{
	fwrite_c(&(_num), 1, f);
	fwrite_c(&(_cross_size), 1, f);
	fwrite_c(_idx2index, _num, f);
	fwrite_c(_index2ridx, _cross_size, f);

	return 0;
}

int CrossMap::load(FILE *f)
{
	fread_c(&(_num), 1, f);
	fread_c(&(_cross_size), 1, f);
	_idx2index = malloc_c<integer_t>(_num);
	_index2ridx = malloc_c<integer_t>(_cross_size);
	fwrite_c(_idx2index, _num, f);
	fwrite_c(_index2ridx, _cross_size, f);

	return 0;
}

int CrossMap::compare(CrossMap &m)
{
	bool equal = true;
	equal = (_num == m._num) && equal;
	equal = (_cross_size== m._cross_size) && equal;
	equal = is_equal_array(_idx2index, m._idx2index, _num) && equal;
	equal = is_equal_array(_index2ridx, m._index2ridx, _cross_size) && equal;
	
	return equal;
}

int CrossMap::log(const char *name)
{
	string s(name);
	FILE *f = fopen_c((s+".cm").c_str(), "w+");
	fprintf(f, "%ld\n", _num);
	fprintf(f, "%ld\n", _cross_size);
	size_t cross_num = 0;
	for (size_t i=0; i<_num; i++) {
		fprintf(f, PT_INTEGER_T("", " "), _idx2index[i]);
		if (_idx2index[i]>=0) {
			cross_num++;
		}
	}
	fprintf(f, "\n");
	fprintf(f, "\n");

	fprintf(f, "%ld\n", cross_num);

	if (cross_num <= 0) {
		assert(_cross_size <= 0);
	} else {
		assert(_cross_size % cross_num == 0);
		size_t node_num = _cross_size / cross_num;
		for (size_t i=0; i<cross_num; i++) {
			for (size_t j=0; j<node_num; j++) {
				fprintf(f, PT_INTEGER_T("", " "), _index2ridx[i*node_num+j]);
			}
			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}
	fclose_c(f);

	return 0;
}

