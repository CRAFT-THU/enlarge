#ifndef LIFEXPNEURON_H
#define LIFEXPNEURON_H

#include <stdio.h>
#include "../../interface/Neuron.h"

class LIFExpNeuron : public Neuron {
public:
	LIFExpNeuron(real v_init, real v_rest, real v_reset, real v_th, real r, real tau_m, real tau_ref, real E_syn, real dt, size_t num=1);
	LIFExpNeuron(const LIFExpNeuron &n, size_t num=0);
	~LIFExpNeuron();

	virtual int append(const Neuron *n, size_t num=0) override;
	virtual void * packup() override;
	int packup(void *data, size_t dst, size_t src) override;

protected:
	vector<int> _refract_time; // ? tau_ref
    vector<int> _refract_step;

	vector<real> _v;
	vector<real> _V_tmp; // ? a.k.a. V_rest
	vector<real> _V_thresh;
	vector<real> _V_reset;

    vector<real> _R;
    vector<real> _C_m; // ? dt / tau_m
    vector<real> _E;

	vector<real> _i; // ? input_current
	
};

#endif /* LIFEXPNEURON_H */
