
#include "LIFNmdaNeuron.h"

#include <assert.h>
#include <math.h>

#include "../../../msg_utils/helper/helper_c.h"
#include "../../third_party/json/json.h"
#include "LIFNmdaData.h"

LIFNmdaNeuron::LIFNmdaNeuron(real v_init, real v_rest, real v_reset, real v_th,
                           real r, real tau_m, real tau_ref, real E_syn, real Mg, real dt,
                           size_t num)
    : Neuron(LIFNmda, num) {
    real cm = dt / tau_m;
    real m_c = Mg / 3.57;

    _refract_time.insert(_refract_time.end(), num,
                         static_cast<int>(tau_ref / dt));  // * 转换为步数
    _refract_step.insert(_refract_step.end(), num, 0);

    _v.insert(_v.end(), num, v_init);
    _V_tmp.insert(_V_tmp.end(), num, v_rest);
    _V_thresh.insert(_V_thresh.end(), num, v_th);
    _V_reset.insert(_V_reset.end(), num, v_reset);

    _R.insert(_R.end(), num, r);
    _C_m.insert(_C_m.end(), num, cm);
    _E.insert(_E.end(), num, E_syn);
    _M_c.insert(_M_c.end(), num, m_c);

    _i.insert(_i.end(), num, 0);

    assert(_num == _v.size());
}

LIFNmdaNeuron::LIFNmdaNeuron(const LIFNmdaNeuron &n, size_t num)
    : Neuron(LIFNmda, 0) {
    append(dynamic_cast<const Neuron *>(&n), num);
}

LIFNmdaNeuron::~LIFNmdaNeuron() {
    _num = 0;
    _refract_time.clear();
    _refract_step.clear();
    _v.clear();
    _V_tmp.clear();
    _V_thresh.clear();
    _V_reset.clear();
    _R.clear();
    _C_m.clear();
    _E.clear();
    _M_c.clear();
    _i.clear();
}

int LIFNmdaNeuron::append(const Neuron *neuron, size_t num) {
    const LIFNmdaNeuron *n = dynamic_cast<const LIFNmdaNeuron *>(neuron);
    int ret = 0;
    if ((num > 0) && (num != n->size())) {
        ret = num;
        _refract_step.insert(_refract_step.end(), num, 0);
        _refract_time.insert(_refract_time.end(), num, n->_refract_time[0]);

        _v.insert(_v.end(), num, n->_v[0]);
        _V_tmp.insert(_V_tmp.end(), num, n->_V_tmp[0]);
        _V_thresh.insert(_V_thresh.end(), num, n->_V_thresh[0]);
        _V_reset.insert(_V_reset.end(), num, n->_V_reset[0]);

        _R.insert(_R.end(), num, n->_R[0]);
        _C_m.insert(_C_m.end(), num, n->_C_m[0]);
        _E.insert(_E.end(), num, n->_E[0]);
        _M_c.insert(_M_c.end(), num, n->_M_c[0]);

        _i.insert(_i.end(), num, 0);
    } else {
        ret = n->size();
        _refract_step.insert(_refract_step.end(), n->_refract_step.begin(),
                             n->_refract_step.end());
        _refract_time.insert(_refract_time.end(), n->_refract_time.begin(),
                             n->_refract_time.end());

        _v.insert(_v.end(), n->_v.begin(), n->_v.end());
        _V_tmp.insert(_V_tmp.end(), n->_V_tmp.begin(), n->_V_tmp.end());
        _V_thresh.insert(_V_thresh.end(), n->_V_thresh.begin(),
                         n->_V_thresh.end());
        _V_reset.insert(_V_reset.end(), n->_V_reset.begin(), n->_V_reset.end());

        _R.insert(_R.end(), n->_R.begin(), n->_R.end());
        _C_m.insert(_C_m.end(), n->_C_m.begin(), n->_C_m.end());
        _E.insert(_E.end(), n->_E.begin(), n->_E.end());
        _M_c.insert(_M_c.end(), n->_M_c.begin(), n->_M_c.end());

        _i.insert(_i.end(), n->_i.begin(), n->_i.end());
    }

    _num += ret;
    assert(_num == _v.size());

    return ret;
}

void *LIFNmdaNeuron::packup() {
    LIFNmdaData *p = static_cast<LIFNmdaData *>(mallocLIFNmda());
    assert(p != NULL);
    p->num = _num;
    p->pRefracTime = _refract_time.data();
    p->pRefracStep = _refract_step.data();
    
    p->pV = _v.data();
    p->pV_tmp = _V_tmp.data();
    p->pV_thresh = _V_thresh.data();
    p->pV_reset = _V_reset.data();
    
    p->pR = _R.data();
    p->pC_m = _C_m.data();
    p->pE = _E.data();
    p->pM_c = _M_c.data();
    
    p->pI = _i.data();
    
    p->_fire_count = malloc_c<int>(_num);
    p->is_view = true;

    return p;
}

int LIFNmdaNeuron::packup(void *data, size_t dst, size_t src) {
    LIFNmdaData *p = static_cast<LIFNmdaData *>(data);

    p->pRefracTime[dst] = _refract_time[src];
    p->pRefracStep[dst] = _refract_step[src];

    p->pV[dst] = _v[src];
    p->pV_tmp[dst] = _V_tmp[src];
    p->pV_thresh[dst] = _V_thresh[src];
    p->pV_reset[dst] = _V_reset[src];

    p->pR[dst] = _R[src];
    p->pC_m[dst] = _C_m[src];
    p->pE[dst] = _E[src];
    p->pM_c[dst] = _M_c[src];

    p->pI[dst] = _i[src];

    return 0;
}
