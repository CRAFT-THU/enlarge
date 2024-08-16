
#include <assert.h>
#include <numeric>

#include "math.h"

#include "../../third_party/json/json.h"
#include "../../../msg_utils/helper/helper_c.h"
#include "LIFNeuron.h"
#include "LIFData.h"


LIFNeuron::LIFNeuron(real v_init, real v_rest, real v_reset, real cm, real tau_m, real tau_refrac, real tau_syn_E, real tau_syn_I, real v_thresh, real i_offset, real dt, size_t num, bool use_input) : Neuron(LIF, num, use_input){

	real rm = (fabs(cm) > ZERO)?(tau_m/cm):1.0;			// 
	real Cm = (tau_m>0)?exp(-dt/tau_m):0.0;
	real Ce = (tau_syn_E > 0)?exp(-dt/tau_syn_E):0.0;
	real Ci = (tau_syn_I > 0)?exp(-dt/tau_syn_I):0.0;

	real v_tmp = i_offset * rm + v_rest;
	v_tmp *= (1-Cm);

	real C_e = rm * tau_syn_E/(tau_syn_E - tau_m);
	real C_i = rm * tau_syn_I/(tau_syn_I - tau_m);

	C_e = C_e * (Ce - Cm);
	C_i = C_i * (Ci - Cm);
	
	int refract_time = static_cast<int>(tau_refrac/dt);

	_refract_step.insert(_refract_step.end(), num, 0);
	_refract_time.insert(_refract_time.end(), num, refract_time);

	_v.insert(_v.end(), num, v_init);
	_Ci.insert(_Ci.end(), num, Ci);
	_Ce.insert(_Ce.end(), num, Ce);
	_C_i.insert(_C_i.end(), num, C_i);
	_C_e.insert(_C_e.end(), num, C_e);
	_Cm.insert(_Cm.end(), num, Cm);
	_v_tmp.insert(_v_tmp.end(), num, v_tmp);
	_V_thresh.insert(_V_thresh.end(), num, v_thresh);
	_V_reset.insert(_V_reset.end(), num, v_reset);

	_i_e.insert(_i_e.end(), num, 0);
	_i_i.insert(_i_i.end(), num, 0);

	assert(_num == _v.size());
}

LIFNeuron::LIFNeuron(const LIFNeuron &n, size_t num) : Neuron(LIF, 0)
{
	append(dynamic_cast<const Neuron *>(&n), num);
}

LIFNeuron::~LIFNeuron()
{
	_num = 0;
	_refract_step.clear();
	_refract_time.clear();

	_v.clear();
	_Ci.clear();
	_Ce.clear();
	_Cm.clear();
	_C_i.clear();
	_C_e.clear();
	_v_tmp.clear();
	_V_thresh.clear();
	_V_reset.clear();

	_i_i.clear();
	_i_e.clear();

	_use_input = false;

	for (auto& p : _input) p = free_c(p);
	_input.clear();
	_input_sz.clear();
}

int LIFNeuron::append(const Neuron * neuron, size_t num)
{
	const LIFNeuron *n = dynamic_cast<const LIFNeuron *>(neuron);
	assert(this->_use_input == n->_use_input);
	int ret = 0;
	if ((num > 0) && (num != n->size())) {
		ret = num;
		_refract_step.insert(_refract_step.end(), num, 0);
		_refract_time.insert(_refract_time.end(), num, n->_refract_time[0]);

		_v.insert(_v.end(), num, n->_v[0]);
		_Ci.insert(_Ci.end(), num, n->_Ci[0]);
		_Ce.insert(_Ce.end(), num, n->_Ce[0]);
		_Cm.insert(_Cm.end(), num, n->_Cm[0]);
		_C_i.insert(_C_i.end(), num, n->_C_i[0]);
		_C_e.insert(_C_e.end(), num, n->_C_e[0]);
		_v_tmp.insert(_v_tmp.end(), num, n->_v_tmp[0]);
		_V_thresh.insert(_V_thresh.end(), num, n->_V_thresh[0]);
		_V_reset.insert(_V_reset.end(), num, n->_V_reset[0]);

		_i_e.insert(_i_e.end(), num, 0);
		_i_i.insert(_i_i.end(), num, 0);

		if (this->_use_input && n->_use_input) { // inputs无法对齐就置零
			_input_sz.insert(_input_sz.end(), num, 0);
			_input.insert(_input.end(), num, nullptr);
		}
	} else {
		ret = n->size();
		_refract_step.insert(_refract_step.end(), n->_refract_step.begin(), n->_refract_step.end());
		_refract_time.insert(_refract_time.end(), n->_refract_time.begin(), n->_refract_time.end());

		_v.insert(_v.end(), n->_v.begin(), n->_v.end());
		_Ci.insert(_Ci.end(), n->_Ci.begin(), n->_Ci.end());
		_Ce.insert(_Ce.end(), n->_Ce.begin(), n->_Ce.end());
		_Cm.insert(_Cm.end(), n->_Cm.begin(), n->_Cm.end());
		_C_i.insert(_C_i.end(), n->_C_i.begin(), n->_C_i.end());
		_C_e.insert(_C_e.end(), n->_C_e.begin(), n->_C_e.end());
		_v_tmp.insert(_v_tmp.end(), n->_v_tmp.begin(), n->_v_tmp.end());
		_V_thresh.insert(_V_thresh.end(), n->_V_thresh.begin(), n->_V_thresh.end());
		_V_reset.insert(_V_reset.end(), n->_V_reset.begin(), n->_V_reset.end());

		_i_e.insert(_i_e.end(), n->_i_e.begin(), n->_i_e.end());
		_i_i.insert(_i_i.end(), n->_i_i.begin(), n->_i_i.end());

		if (this->_use_input && n->_use_input) {
			_input_sz.insert(_input_sz.end(), n->_input_sz.begin(), n->_input_sz.end());
			_input.insert(_input.end(), n->_input.begin(), n->_input.end());
		}
	}

	_num += ret;
	assert(_num == _v.size());

	return ret;
}

void * LIFNeuron::packup()
{
	LIFData *p = static_cast<LIFData*>(mallocLIF());
	assert(p != NULL);
	p->num = _num;
	p->pRefracTime = _refract_time.data();
	p->pRefracStep = _refract_step.data();

	p->pI_e = _i_e.data();
	p->pI_i = _i_i.data();
	p->pCe = _Ce.data();
	p->pV_reset = _V_reset.data();
	// p->pV_e = _v_.data();
	p->pV_tmp = _v_tmp.data();
	// p->pI_i = _i_i.data();
	p->pV_thresh = _V_thresh.data();
	p->pCi = _Ci.data();
	p->pV_m = _v.data();
	p->pC_e = _C_e.data();
	p->pC_m = _Cm.data();
	p->pC_i = _C_i.data();

	// * 存储神经元初始输入的起始索引
	p->use_input = _use_input;
	if (_use_input) {
		vector<int> inputStart(_num + 1, 0);
		std::partial_sum(_input_sz.begin(), _input_sz.end(), inputStart.begin() + 1);
		p->pInput_start = inputStart.data();

		// * 压缩神经元的初始输入
		vector<real> input;
		for (int i = 0; i < _num; ++i)
			if (_input_sz[i] > 0)
				input.insert(input.end(), _input[i], _input[i] + _input_sz[i]);
		p->pInput = input.data();
	}

	p->_fire_count = malloc_c<int>(_num);
	p->is_view = true;

	return p;
}

int LIFNeuron::packup(void *data, size_t dst, size_t src)
{
	LIFData *p = static_cast<LIFData*>(data);
	p->pRefracTime[dst] = _refract_time[src];
	p->pRefracStep[dst] = _refract_step[src];

	p->pI_e[dst] = _i_e[src];
	p->pI_i[dst] = _i_i[src];
	p->pCe[dst] = _Ce[src];
	p->pV_reset[dst] = _V_reset[src];
	p->pV_tmp[dst] = _v_tmp[src];
	p->pV_thresh[dst] = _V_thresh[src];
	p->pCi[dst] = _Ci[src];
	p->pV_m[dst] = _v[src];
	p->pC_e[dst] = _C_e[src];
	p->pC_m[dst] = _Cm[src];
	p->pC_i[dst] = _C_i[src];

	// TODO: 是否记录pInput_start和pInput？

	return 0;
}
