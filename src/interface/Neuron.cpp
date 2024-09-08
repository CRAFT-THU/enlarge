
#include "Neuron.h"

Neuron::~Neuron() {}

void Neuron::set_input(size_t idx, real* input, int sz) {
    if (_input_size.empty()) {
        _input_list.resize(_num, nullptr);
        _input_size.resize(_num, 0);
    }
    if (0 > idx || idx >= _num) return;
    _input_list[idx] = input;
    _input_size[idx] = sz;
    is_input_parsed = false;
}

void Neuron::unset_input(size_t idx) {
    if (0 > idx || idx >= _num) return;
    _input_size[idx] = 0;
    is_input_parsed = false;
}

void Neuron::parse_input() {      // ! CALLED AFTER ALL INPUTS ARE SET
    if (is_input_parsed) return;  // * have been parsed
    _input_sz = 0;
    _input_start.resize(_num, 0);
    for (int i = 0; i < _num; ++i) {
        _input_start[i] = _input_sz;
        _input_sz += _input_size[i];
    }
    _input.clear();
    for (int i = 0; i < _num; ++i)
        if (_input_size[i] > 0)
            _input.insert(_input.end(), _input_list[i],
                          _input_list[i] + _input_size[i]);
    is_input_parsed = true;
}
