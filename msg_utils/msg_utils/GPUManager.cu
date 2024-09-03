
#include <stdio.h>
#include <assert.h>

#include "GPUManager.h"
#include "../helper/helper_gpu.h"

thread_local GPUManager gm;

thread_local int GPUManager::_id = 0;
thread_local atomic<char> GPUManager::_lock(0);


GPUManager::GPUManager()
{
}

GPUManager::~GPUManager()
{
}

int GPUManager::set(int id)
{
	assert(id >= 0);
	bool locked = _lock.load() > 0;
	if (locked && id != _id) {
		printf("Warn: try to change a locked gpu device!\n");
		return _id;
	}

	gpuSetDevice(id);
	_id = gpuGetDevice();

	return _id;
}

int GPUManager::get()
{
	int id = gpuGetDevice();
	if (_id !=  id) {
		printf("Error: gpu device variance, usually due to calling setDevice driectly!\n");
		_id = id;
		return -1;
	}
	return id;
}

void GPUManager::check()
{
	int id = gpuGetDevice();
	if (_id !=  id) {
		printf("Warn: gpu device variance, usually due to calling setDevice driectly!\n");
		_id = id;
	}
	printf("Now on GPU device %d\n", _id);
}

int GPUManager::lock()
{
	_lock.store(1);
	return _lock.load();
}

int GPUManager::unlock()
{
	_lock.store(0);
	return _lock.load();
}
