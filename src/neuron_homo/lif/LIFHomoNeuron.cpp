
#include "LIFHomoNeuron.h"

#include <assert.h>

#include <numeric>

#include "../../../msg_utils/helper/helper_c.h"
#include "../../third_party/json/json.h"
#include "LIFHomoData.h"
#include "math.h"

LIFHomoNeuron::LIFHomoNeuron(real v_init, real v_rest, real v_reset, real cm,
                             real tau_m, real tau_refrac, real tau_syn_E,
                             real tau_syn_I, real v_thresh, real i_offset,
                             real dt, size_t num)
    : HomoNeuron(LIFHomo, num, false) {
    real rm = (fabs(cm) > ZERO) ? (tau_m / cm) : 1.0;
    real Cm = (tau_m > 0) ? exp(-dt / tau_m) : 0.0;
    real Ce = (tau_syn_E > 0) ? exp(-dt / tau_syn_E) : 0.0;
    real Ci = (tau_syn_I > 0) ? exp(-dt / tau_syn_I) : 0.0;

    real v_tmp = i_offset * rm + v_rest;
    v_tmp *= (1 - Cm);

    real C_e = rm * tau_syn_E / (tau_syn_E - tau_m);
    real C_i = rm * tau_syn_I / (tau_syn_I - tau_m);

    C_e = C_e * (Ce - Cm);
    C_i = C_i * (Ci - Cm);

    int refract_time = static_cast<int>(tau_refrac / dt);

    _refract_time = refract_time;
    _Ci = Ci;
    _Ce = Ce;
    _Cm = Cm;
    _C_i = C_i;
    _C_e = C_e;
    _V_thresh = v_thresh;
    _v_tmp = v_tmp;
    _V_reset = v_reset;

    _refract_step.insert(_refract_step.end(), num, 0);

    _v.insert(_v.end(), num, v_init);

    _i_e.insert(_i_e.end(), num, 0);
    _i_i.insert(_i_i.end(), num, 0);

    _input_size.insert(_input_size.end(), num, 0);
    _input_list.insert(_input_list.end(), num, nullptr);

    assert(_num == _v.size());
}

LIFHomoNeuron::LIFHomoNeuron(const LIFHomoNeuron &n, size_t num)
    : HomoNeuron(LIF, 0) {
    append(dynamic_cast<const Neuron *>(&n), num);
}

LIFHomoNeuron::~LIFHomoNeuron() {
    _num = 0;
    _refract_step.clear();
    _v.clear();
    _i_i.clear();
    _i_e.clear();

    // for (auto& p : _input_list) p = free_c(p); // FIXME: leave it to user
    _input_list.clear();
    _input_size.clear();
    _input_start.clear();
    _input.clear();
}

bool LIFHomoNeuron::is_equal(const HomoNeuron *neuron) {
    const LIFHomoNeuron *n = dynamic_cast<const LIFHomoNeuron *>(neuron);
    bool check = (n != nullptr);
    check = check && (_refract_time == n->_refract_time);
    check = check && (fabs(_Ci - n->_Ci) < ZERO);
    check = check && (fabs(_Ce - n->_Ce) < ZERO);
    check = check && (fabs(_Cm - n->_Cm) < ZERO);
    check = check && (fabs(_C_i - n->_C_i) < ZERO);
    check = check && (fabs(_C_e - n->_C_e) < ZERO);
    check = check && (fabs(_v_tmp - n->_v_tmp) < ZERO);
    check = check && (fabs(_V_thresh - n->_V_thresh) < ZERO);
    check = check && (fabs(_V_reset - n->_V_reset) < ZERO);
    return check;
}

int LIFHomoNeuron::append(const Neuron *neuron, size_t num) {
    const HomoNeuron *n = dynamic_cast<const HomoNeuron *>(neuron);
    if (!is_equal(n)) return 0;

    int ret = 0;
    const LIFHomoNeuron *nrn = dynamic_cast<const LIFHomoNeuron *>(n);
    if ((num > 0) && (num != n->size())) {
        ret = num;
        _refract_step.insert(_refract_step.end(), num, 0);
        _v.insert(_v.end(), num, nrn->_v[0]);
        _i_e.insert(_i_e.end(), num, 0);
        _i_i.insert(_i_i.end(), num, 0);

        _input_size.insert(_input_size.end(), num, 0);
        _input_list.insert(_input_list.end(), num, nullptr);
    } else {
        ret = n->size();
        _refract_step.insert(_refract_step.end(), nrn->_refract_step.begin(),
                             nrn->_refract_step.end());
        _v.insert(_v.end(), nrn->_v.begin(), nrn->_v.end());
        _i_e.insert(_i_e.end(), nrn->_i_e.begin(), nrn->_i_e.end());
        _i_i.insert(_i_i.end(), nrn->_i_i.begin(), nrn->_i_i.end());

        _input_size.insert(_input_size.end(), nrn->_input_size.begin(),
                           nrn->_input_size.end());
        _input_list.insert(_input_list.end(), nrn->_input_list.begin(),
                           nrn->_input_list.end());
    }

    _num += ret;
    assert(_num == _v.size());

    return ret;
}

void *LIFHomoNeuron::packup() {
    parse_input();

    LIFHomoData *p = static_cast<LIFHomoData *>(mallocLIFHomo());
    assert(p != NULL);
    p->num = _num;
    p->pRefracStep = _refract_step.data();
    p->pI_e = _i_e.data();
    p->pI_i = _i_i.data();
    p->pV_m = _v.data();

    p->cRefracTime = _refract_time;
    p->cV_reset = _V_reset;
    p->cV_tmp = _v_tmp;
    p->cV_thresh = _V_thresh;
    p->cCe = _Ce;
    p->cCi = _Ci;
    p->cC_e = _C_e;
    p->cC_m = _Cm;
    p->cC_i = _C_i;

    // * 存储神经元初始输入的起始索引
    p->input_sz = _input_sz;
    p->pInput_start = _input_start.data();
    p->pInput = _input.data();

    p->_fire_count = malloc_c<int>(_num);
    p->is_view = true;

    return p;
}

int LIFHomoNeuron::packup(void *data, size_t dst, size_t src) {
    parse_input();

    LIFHomoData *p = static_cast<LIFHomoData *>(data);
    p->pRefracStep[dst] = _refract_step[src];
    p->pI_e[dst] = _i_e[src];
    p->pI_i[dst] = _i_i[src];
    p->pV_m[dst] = _v[src];

    p->cRefracTime = _refract_time;
    p->cV_reset = _V_reset;
    p->cV_tmp = _v_tmp;
    p->cV_thresh = _V_thresh;
    p->cCe = _Ce;
    p->cCi = _Ci;
    p->cC_e = _C_e;
    p->cC_m = _Cm;
    p->cC_i = _C_i;

    p->input_sz = _input_sz;
    p->pInput_start[dst] = _input_start[src];

    if (!p->pInput) p->pInput = _input.data();
    // assert(p->pInput != nullptr);
    return 0;
}
