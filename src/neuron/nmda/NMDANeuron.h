#ifndef NMDANEURON_H
#define NMDANEURON_H

#include <stdio.h>
#include "../../interface/Neuron.h"

class NMDANeuron : public Neuron {
public:
	NMDANeuron(real tau_rise, real tau_decay, real dt, size_t num=1);
	NMDANeuron(const NMDANeuron &n, size_t num=0);
	~NMDANeuron();

	virtual int append(const Neuron *n, size_t num=0) override;
	virtual void * packup() override;
	int packup(void *data, size_t dst, size_t src) override;

protected:
    /* refers to struct NMDANrnData */
	vector<real> _s; // ? defaults to 0
    vector<real> _x; // ? defaults to 0
    
    vector<real> _coeff;
    vector<real> _tau_decay_rcpl;
    vector<real> _tau_rise_compl;
};

#endif /* NMDANEURON_H */
