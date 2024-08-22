/* This header file is writen by qp09
 * usually just for fun
 * Thu October 22 2015
 */
#ifndef NEURON_H
#define NEURON_H

#include "../interface/Model.h"

class Neuron : public Model {
public:
	Neuron(Type type, size_t num, size_t offset=0, int buffer_size=2) : Model(type, num, offset, buffer_size) {}

	virtual ~Neuron() = 0;

	virtual int append(const Neuron *n, size_t num=0) = 0;

	void set_input(size_t idx, real* input, int sz) {
		if (_input_size.empty()) {
			_input_list.resize(_num, nullptr);
			_input_size.resize(_num, 0);
		}
		_input_list[idx] = input;
		_input_size[idx] = sz;
	}
	void unset_input(size_t idx) {
		_input_size[idx] = 0;
	}

public:
// protected:
	vector<int> _input_size; // ? 初始输入每一项的长度（0表示接受外部输入，否则接受初始输入）
	vector<real*> _input_list; // ? 初始输入，按"邻接表"方式压缩

    int _input_sz;
	vector<int> _input_start; // ? 初始输入的起始位置（由input_sz得到）
	vector<real> _input; // ? 压缩后的初始输入

	void parse_input() { // ! CALLED AFTER ALL INPUTS ARE SET
        if (_input_start.size() == _num) return; // * have been parsed
        
        _input_sz = 0;
		_input_start.resize(_num, 0);
        for (int i = 0; i < _num; ++i) {
            _input_start[i] = _input_sz;
            _input_sz += _input_size[i];
        }
        printf("input_sz: %d\n", _input_sz);

		_input.clear();
		for (int i = 0; i < _num; ++i)
			if (_input_size[i] > 0)
				_input.insert(_input.end(), _input_list[i], _input_list[i] + _input_size[i]);
	}
};

#endif /* NEURON_H */

