
#include "../helper/helper_c.h"
#include "../helper/helper_gpu.h"
#include "CrossMap.h"

CrossMap::~CrossMap()
{
	if (_num > 0) {
		free_clear(_idx2index);
	}

	if (_cross_size > 0) {
		free_clear(_index2ridx);
	}

	if (_gpu_array) {
		if (_gpu_array->_idx2index) {
			gpuFreeClear(_gpu_array->_idx2index);
		}

		if (_gpu_array->_index2ridx) {
			gpuFreeClear(_gpu_array->_index2ridx);
		}

		_gpu_array->_num = 0;
		_gpu_array->_cross_size = 0;
		_gpu_array->_gpu_array = NULL;

		delete _gpu_array;
	}

	_num  = 0;
	_cross_size = 0;
}

int CrossMap::to_gpu()
{
	if (!_gpu_array) {
		_gpu_array = new CrossMap;

		_gpu_array->_num = _num;
		_gpu_array->_cross_size = _cross_size;

		_gpu_array->_idx2index = TOGPU(_idx2index, _num);
		_gpu_array->_index2ridx = TOGPU(_index2ridx, _cross_size);
	} else {
		assert(_gpu_array->_num == _num);
		assert(_gpu_array->_cross_size == _cross_size);

		COPYTOGPU(_gpu_array->_idx2index, _idx2index, _num);
		COPYTOGPU(_gpu_array->_index2ridx, _index2ridx, _cross_size);
	}

	return 0;
}
