#ifndef LIFHOMONEURON_H
#define LIFHOMONEURON_H

#include <stdio.h>

#include "../../interface/HomoNeuron.h"

class LIFHomoNeuron : public HomoNeuron {
   public:
    LIFHomoNeuron(real v_init, real v_rest, real v_reset, real cm, real tau_m,
                  real tau_refrac, real tau_syn_E, real tau_syn_I,
                  real v_thresh, real i_offset, real dt, size_t num = 1);
    LIFHomoNeuron(const LIFHomoNeuron &n, size_t num = 0);
    ~LIFHomoNeuron();

	virtual bool is_equal(const HomoNeuron *n) override;
    virtual int append(const Neuron *n, size_t num = 0) override;
    virtual void *packup() override;
    int packup(void *data, size_t dst, size_t src) override;

   protected:
   	// vars
    vector<int> _refract_step;
    vector<real> _v;
    vector<real> _i_i;
    vector<real> _i_e;
    
	// consts
    int _refract_time;
	real _Ci;
	real _Ce;
	real _Cm;
	real _C_i;
	real _C_e;
	real _v_tmp;
	real _V_thresh;
	real _V_reset;
};

#endif /* LIFHOMONEURON_H */
