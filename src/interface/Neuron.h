/* This header file is writen by qp09
 * usually just for fun
 * Thu October 22 2015
 */
#ifndef NEURON_H
#define NEURON_H

#include "../interface/Model.h"
// #include <cassert>

class Neuron : public Model {
public:
	Neuron(Type type, size_t num, bool use_input=false, size_t offset=0, int buffer_size=2) : Model(type, num, offset, buffer_size) {
		this->_use_input = use_input;
		if (use_input) {
			_input.resize(num);
			_input_sz.resize(num);
		}
	}

	virtual ~Neuron() = 0;

	virtual int append(const Neuron *n, size_t num=0) = 0;

	void set_input(size_t idx, real* input, int sz) {
		// assert(this->_use_input == true);
		_input[idx] = input;
		_input_sz[idx] = sz;
	}
	void set_input(vector<real*>& input, vector<int>& input_sz) {
		this->_use_input = true;
		_input = std::move(input);
		_input_sz = std::move(input_sz);
	}

protected:
	bool _use_input; // ? 是否使用初始输入
	vector<int> _input_sz; // ? 初始输入每一项的长度
	vector<real*> _input; // ? 初始输入，按"邻接表"方式压缩
};

#endif /* NEURON_H */

