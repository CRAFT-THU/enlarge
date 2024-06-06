#ifndef EXPSYNAPSE_H
#define EXPSYNAPSE_H

#include <stdio.h>
#include <list>
#include "../../interface/Synapse.h"

using std::list;

class ExpSynapse : public Synapse {
public:
    // * g_max 和 tau_syn 是可变的
	ExpSynapse(real g_max, real tau_rise, real dt, size_t num=1);
    ExpSynapse(const real *g_max, const real *tau_rise, real dt, size_t num=1);
    ExpSynapse(const real *g_max, const real tau_rise, real dt, size_t num=1);
    ExpSynapse(const real g_max, const real *tau_rise, real dt, size_t num=1);
    ExpSynapse(real dt, size_t num=1); // ? g_max = 0.01, tau_rise = 2.0
	ExpSynapse(const ExpSynapse &s, size_t num=0);
	~ExpSynapse();

	virtual int append(const Synapse *s, size_t num=0) override;

	virtual void * packup() override;
	virtual int packup(void *data, size_t dst, size_t src) override;

	virtual real weight(size_t idx) override {
		return _weight[idx];
	}

protected:
    vector<real> _s;
	vector<real> _weight;
    vector<real> _g;
};

#endif /* EXPSYNAPSE_H */
