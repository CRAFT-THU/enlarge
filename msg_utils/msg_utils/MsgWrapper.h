
#ifndef MSGWRAPPER_H
#define MSGWRAPPER_H

#include <iostream>
#include <assert.h>
#include <mpi.h>

#include "helper/helper_type.h"

using std::cout;
using std::endl;

template<typename T>
class MsgWrapper {
public:
	MsgWrapper(size_t num);
	MsgWrapper();
	~MsgWrapper();

	int send_meta(int dst, int tag);
	int recv_meta(int dst, int tag);

	int send_meta(int dst, int tag, MPI_Comm comm);
	int recv_meta(int dst, int tag, MPI_Comm comm);

	int send(int dst, int tag);
	int recv(int src, int tag);

	int send(int dst, int tag, MPI_Comm comm);
	int recv(int src, int tag, MPI_Comm comm);

	T * get_value(size_t idx);

	void print();

protected:
	T *_data;
	size_t _num;
};

template<typename T>
MsgWrapper<T>::MsgWrapper()
{
	_num = 0;
	_data = NULL; 
}

template<typename T>
MsgWrapper<T>::MsgWrapper(size_t num)
{
	_num = num;
	if (num > 0) {
		_data = new T[num];
	} else {
		_data = NULL;
	}
}

template<typename T>
MsgWrapper<T>::~MsgWrapper()
{
	_num = 0;
	delete [] _data;
}

template<typename T>
int MsgWrapper<T>::send_meta(int dst, int tag)
{
	return send_meta(dst, tag, MPI_COMM_WORLD);
}

template<typename T>
int MsgWrapper<T>::send_meta(int dst, int tag, MPI_Comm comm)
{
	int ret = 0;
	ret = MPI_Send(&_num, 1, MPI_SIZE_T, dst, tag, comm);
	assert(ret == MPI_SUCCESS);
	return 1;
}

template<typename T>
int MsgWrapper<T>::recv_meta(int src, int tag)
{
	return recv_meta(src, tag, MPI_COMM_WORLD);
}

template<typename T>
int MsgWrapper<T>::recv_meta(int src, int tag, MPI_Comm comm)
{
	int ret = 0;
	size_t num;
	MPI_Status status;
	ret = MPI_Recv(&num, 1, MPI_SIZE_T, src, tag, comm, &status);
	assert(ret == MPI_SUCCESS);
	if (num > _num) {
		delete[] _data;
		_num = num;
		_data = new T[num];
	}
	return 1;
}

template<typename T>
int MsgWrapper<T>::send(int dst, int tag)
{
	return send(dst, tag, MPI_COMM_WORLD);
}

template<typename T>
int MsgWrapper<T>::send(int dst, int tag, MPI_Comm comm)
{
	int ret = 0;
	ret = MPI_Send(_data, sizeof(T) * _num, MPI_UNSIGNED_CHAR, dst, tag, comm);
	assert(ret == MPI_SUCCESS);
	return 1;
}

template<typename T>
int MsgWrapper<T>::recv(int src, int tag)
{
	return recv(src, tag, MPI_COMM_WORLD);
}

template<typename T>
int MsgWrapper<T>::recv(int src, int tag, MPI_Comm comm)
{
	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(_data, sizeof(T) * _num, MPI_UNSIGNED_CHAR, src, tag, comm, &status);
	assert(ret == MPI_SUCCESS);
	return ret;
}

template<typename T>
T * MsgWrapper<T>::get_value(size_t idx)
{
	return _data + idx;
}

template<typename T>
void MsgWrapper<T>::print()
{
	cout << _num << endl;
	for (size_t i=0; i<_num; i++) {
		cout << _data[i] << " ";
	}
	cout << endl;
}

#endif // MSG_WRAPPER_H
