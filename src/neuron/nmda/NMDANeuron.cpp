
#include <assert.h>

#include "math.h"

#include "../../third_party/json/json.h"
#include "../../../msg_utils/helper/helper_c.h"
#include "NMDANeuron.h"
#include "NMDANrnData.h"


NMDANeuron::NMDANeuron(real tau_rise, real tau_decay, real dt, size_t num) : Neuron(NMDA_NRN, num){

	real coeff = dt / 2;
	real tau_decay_rcpl = dt / tau_decay;
	real tau_rise_compl = 1 - dt / tau_rise;

	_s.insert(_s.end(), num, 0);
	_x.insert(_x.end(), num, 0);

	_coeff.insert(_coeff.end(), num, coeff);
	_tau_decay_rcpl.insert(_tau_decay_rcpl.end(), num, tau_decay_rcpl);
	_tau_rise_compl.insert(_tau_rise_compl.end(), num, tau_rise_compl);

	assert(_num == _s.size());
}

NMDANeuron::NMDANeuron(const NMDANeuron &n, size_t num) : Neuron(NMDA_NRN, 0)
{
	append(dynamic_cast<const Neuron *>(&n), num);
}

NMDANeuron::~NMDANeuron()
{
	_num = 0;
	_s.clear();
	_x.clear();
	_coeff.clear();
	_tau_decay_rcpl.clear();
	_tau_rise_compl.clear();
}

int NMDANeuron::append(const Neuron * neuron, size_t num)
{
	const NMDANeuron *n = dynamic_cast<const NMDANeuron *>(neuron);
	int ret = 0;
	if ((num > 0) && (num != n->size())) {
		ret = num;
		_s.insert(_s.end(), num, 0);
		_x.insert(_x.end(), num, n->_x[0]);

		_coeff.insert(_coeff.end(), num, n->_coeff[0]);
		_tau_decay_rcpl.insert(_tau_decay_rcpl.end(), num, n->_tau_decay_rcpl[0]);
		_tau_rise_compl.insert(_tau_rise_compl.end(), num, n->_tau_rise_compl[0]);

	} else {
		ret = n->size();
		_s.insert(_s.end(), n->_s.begin(), n->_s.end());
		_x.insert(_x.end(), n->_x.begin(), n->_x.end());

		_coeff.insert(_coeff.end(), n->_coeff.begin(), n->_coeff.end());
		_tau_decay_rcpl.insert(_tau_decay_rcpl.end(), n->_tau_decay_rcpl.begin(), n->_tau_decay_rcpl.end());
		_tau_rise_compl.insert(_tau_rise_compl.end(), n->_tau_rise_compl.begin(), n->_tau_rise_compl.end());
	}

	_num += ret;
	assert(_num == _s.size());

	return ret;
}

void * NMDANeuron::packup()
{
	NMDANrnData *p = static_cast<NMDANrnData*>(mallocNMDANrn());
	assert(p != nullptr);

	p->num = _num;
	p->s = _s.data();
	p->x = _x.data();
	p->coeff = _coeff.data();
	p->tau_decay_rcpl = _tau_decay_rcpl.data();
	p->tau_rise_compl = _tau_rise_compl.data();

	// p->_fire_count = malloc_c<int>(_num);
	p->is_view = true;

	return p;
}

int NMDANeuron::packup(void *data, size_t dst, size_t src)
{
	NMDANrnData *p = static_cast<NMDANrnData*>(data);
	assert (p != nullptr);

	p->s[dst] = _s[src];
	p->x[dst] = _x[src];
	p->coeff[dst] = _coeff[src];
	p->tau_decay_rcpl[dst] = _tau_decay_rcpl[src];
	p->tau_rise_compl[dst] = _tau_rise_compl[src];

	return 0;
}
