#ifndef NMDASYNAPSE_H
#define NMDASYNAPSE_H

#include <stdio.h>
#include <list>
#include "../../interface/Synapse.h"

using std::list;

class NMDASynapse : public Synapse {
public:
	NMDASynapse(real weight, real delay, real tau_syn, real dt, size_t num=1);
	NMDASynapse(const real *weight, const real *delay, const real *tau_syn, real dt, size_t num=1);
	NMDASynapse(const real *weight, const real *delay, const real tau_syn, real dt, size_t num=1);
	NMDASynapse(const real *weight, const real *delay, real dt, size_t num=1);
	NMDASynapse(const NMDASynapse &s, size_t num=0);
	~NMDASynapse();

	virtual int append(const Synapse *s, size_t num=0) override;

	virtual void * packup() override;
	virtual int packup(void *data, size_t dst, size_t src) override;

	virtual real weight(size_t idx) override {
		return _weight[idx];
	}

protected:
	vector<real> _weight;
	// const static Type type;
	// real _weight;
	// real _delay;
	// real _tau_syn;
	// list<int> delay_queue;
	// NeuronBase *pDest;
};

#endif /* NMDASYNAPSE_H */
