/* This header file is writen by qp09
 * usually just for fun
 * Thu October 22 2015
 */
#ifndef NEURON_H
#define NEURON_H

#include "../interface/Model.h"

class Neuron : public Model {
public:
	Neuron(Type type, size_t num, size_t offset=0, int buffer_size=2) : Model(type, num, offset, buffer_size) {
		is_input_parsed = false;
	}

	virtual ~Neuron() = 0;

	virtual int append(const Neuron *n, size_t num=0) = 0;

	void set_input(size_t idx, real* input, int sz);
	void unset_input(size_t idx);
	void parse_input();

protected:
	bool is_input_parsed = false;
	vector<int> _input_size;
	vector<real*> _input_list;

	// TODO: optimization needed
    int _input_sz;
	vector<int> _input_start;
	vector<real> _input;
};

#endif /* NEURON_H */

