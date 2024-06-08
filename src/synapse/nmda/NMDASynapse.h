#ifndef NMDASYNAPSE_H
#define NMDASYNAPSE_H

#include <stdio.h>
#include <list>
#include "../../interface/Synapse.h"

using std::list;

class NMDASynapse : public Synapse {
public:
    // TODO: 假定 tau_rise 和 tau_decay 同时指定
	NMDASynapse(real g_max, real tau_rise, real tau_decay, real dt, size_t num=1);
    NMDASynapse(const real *g_max, const real *tau_rise, const real *tau_decay, real dt, size_t num=1);
    NMDASynapse(const real *g_max, const real tau_rise, const real tau_decay, real dt, size_t num=1);
    NMDASynapse(const real g_max, const real *tau_rise, const real *tau_decay, real dt, size_t num=1);
    // NMDASynapse(real dt, size_t num=1); // ? g_max = 0.5, tau_rise = 2.0, tau_decay = 100.0
	NMDASynapse(const NMDASynapse &s, size_t num=0);
	~NMDASynapse();

	virtual int append(const Synapse *s, size_t num=0) override;

	virtual void * packup() override;
	virtual int packup(void *data, size_t dst, size_t src) override;

    // TODO: 暂定是NMDA的_C_decay（和_s）关联的系数
	virtual real weight(size_t idx) override {
		// return _weight[idx];
        return _C_decay[idx];
	}

protected:
    vector<real> _s;
    vector<real> _x;
    vector<real> _C_decay;
	vector<real> _C_rise;
    vector<real> _g;
};

#endif /* NMDASYNAPSE_H */
