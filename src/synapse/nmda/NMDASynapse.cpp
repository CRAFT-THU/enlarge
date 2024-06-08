#include "NMDASynapse.h"

#include <assert.h>
#include <math.h>

#include "../../third_party/json/json.h"
#include "NMDAData.h"

// const Type NMDASynapse::type = NMDA;

NMDASynapse::NMDASynapse(real g_max, real tau_rise, real tau_decay, real dt, size_t num)
    : Synapse(NMDA, num) {
    real c_rise = (1 - dt / tau_rise);
    real c_decay = (1 - dt / tau_decay);

    _s.insert(_s.end(), num, 0);
    _x.insert(_x.end(), num, 0);
    _C_decay.insert(_C_decay.end(), num, c_decay);
    _C_rise.insert(_C_rise.end(), num, c_rise);
    _g.insert(_g.end(), num, g_max);
    assert(_num == _C_decay.size());
}

NMDASynapse::NMDASynapse(const real *g_max, const real *tau_rise, const real* tau_decay, real dt,
                       size_t num)
    : Synapse(NMDA, num) {
    _s.insert(_s.end(), num, 0);
    _x.insert(_x.end(), num, 0);
    _C_decay.resize(num);
    _C_rise.resize(num);
    _g.resize(num);

    for (size_t i = 0; i < num; i++) {
        assert(fabs(tau_decay[i]) > ZERO);
        _C_decay[i] = (1 - dt / tau_decay[i]);
        assert(fabs(tau_rise[i]) > ZERO);
        _C_rise[i] = (1 - dt / tau_rise[i]);
        _g[i] = g_max[i];
    }

    assert(_num == _C_decay.size());
}

NMDASynapse::NMDASynapse(const real *g_max, const real tau_rise, const real tau_decay, real dt,
                       size_t num)
    : Synapse(NMDA, num) {
    real c_rise = (1 - dt / tau_rise);
    real c_decay = (1 - dt / tau_decay);

    _s.insert(_s.end(), num, 0);
    _x.insert(_x.end(), num, 0);
    _C_decay.insert(_C_decay.end(), num, c_decay);
    _C_rise.insert(_C_rise.end(), num, c_rise);
    _g.insert(_g.end(), g_max, g_max + num);

    assert(_num == _g.size());
}

NMDASynapse::NMDASynapse(const real g_max, const real *tau_rise, const real *tau_decay, real dt,
                       size_t num)
    : Synapse(NMDA, num) {
    _s.insert(_s.end(), num, 0);
    _x.insert(_x.end(), num, 0);
    _C_decay.resize(num);
    _C_rise.resize(num);
    _g.insert(_g.end(), num, g_max);

    for (size_t i = 0; i < num; i++) {
        _C_decay[i] = (1 - dt / tau_decay[i]);
        _C_rise[i] = (1 - dt / tau_rise[i]);
    }

    assert(_num == _C_decay.size());
}

NMDASynapse::NMDASynapse(const NMDASynapse &s, size_t num) : Synapse(NMDA, 0) {
    append(dynamic_cast<const Synapse *>(&s), num);
}

NMDASynapse::~NMDASynapse() {
    _num = 0;
    _delay.clear();
    _s.clear();
    _x.clear();
    _C_decay.clear();
    _C_rise.clear();
    _g.clear();
}

int NMDASynapse::append(const Synapse *syn, size_t num) {
    const NMDASynapse *that = dynamic_cast<const NMDASynapse *>(syn);
    size_t ret = 0;
    if ((num > 0) && (num != that->size())) {
        ret = num;
        _s.insert(_s.end(), num, that->_s[0]);
        _x.insert(_x.end(), num, that->_x[0]);
        _C_decay.insert(_C_decay.end(), num, that->_C_decay[0]);
        _C_rise.insert(_C_rise.end(), num, that->_C_rise[0]);
        _g.insert(_g.end(), num, that->_g[0]);
        _delay.insert(_delay.end(), num, that->_delay[0]);
    } else {
        ret = that->_num;
        _s.insert(_s.end(), that->_s.begin(), that->_s.end());
        _x.insert(_x.end(), that->_x.begin(), that->_x.end());
        _C_decay.insert(_C_decay.end(), that->_C_decay.begin(), that->_C_decay.end());
        _C_rise.insert(_C_rise.end(), that->_C_rise.begin(), that->_C_rise.end());
        _g.insert(_g.end(), that->_g.begin(), that->_g.end());
        _delay.insert(_delay.end(), that->_delay.begin(), that->_delay.end());
    }
    _num += ret;
    assert(_num == _C_decay.size());

    return ret;
}

void *NMDASynapse::packup() {
    NMDAData *p = static_cast<NMDAData *>(mallocNMDA());

    p->num = _num;

    p->pS = _s.data();
    p->pX = _x.data();
    p->pC_decay = _C_decay.data();
    p->pC_rise = _C_rise.data();
    p->pG = _g.data();

    p->is_view = true;

    return p;
}

int NMDASynapse::packup(void *data, size_t dst, size_t src) {
    NMDAData *p = static_cast<NMDAData *>(data);

    p->pS[dst] = _s[src];
    p->pX[dst] = _x[src];
    p->pC_decay[dst] = _C_decay[src];
    p->pC_rise[dst] = _C_rise[src];
    p->pG[dst] = _g[src];

    return 0;
}
