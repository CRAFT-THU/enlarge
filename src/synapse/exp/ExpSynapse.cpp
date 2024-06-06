#include "ExpSynapse.h"

#include <assert.h>
#include <math.h>

#include "../../third_party/json/json.h"
#include "ExpData.h"

// const Type ExpSynapse::type = Exp;

ExpSynapse::ExpSynapse(real g_max, real tau_rise, real dt, size_t num)
    : Synapse(Exp, num) {
    real weight = (1 - dt / tau_rise);

    _s.insert(_s.end(), num, 0);
    _weight.insert(_weight.end(), num, weight);
    _g.insert(_g.end(), num, g_max);
    assert(_num == _weight.size());
}

ExpSynapse::ExpSynapse(const real *g_max, const real *tau_rise, real dt,
                       size_t num)
    : Synapse(Exp, num) {
    _s.insert(_s.end(), num, 0);
    _weight.resize(num);
    _g.resize(num);

    for (size_t i = 0; i < num; i++) {
        assert(fabs(tau_rise[i]) > ZERO);
        _weight[i] = (1 - dt / tau_rise[i]);
        _g[i] = g_max[i];
    }

    assert(_num == _weight.size());
}

ExpSynapse::ExpSynapse(const real *g_max, const real tau_rise, real dt,
                       size_t num)
    : Synapse(Exp, num) {
    real weight = (1 - dt / tau_rise);

    _s.insert(_s.end(), num, 0);
    _weight.insert(_weight.end(), num, weight);
    _g.insert(_g.end(), g_max, g_max + num);

    assert(_num == _g.size());
}

ExpSynapse::ExpSynapse(const real g_max, const real *tau_rise, real dt,
                       size_t num)
    : Synapse(Exp, num) {
    _s.insert(_s.end(), num, 0);
    _g.insert(_g.end(), num, g_max);

    for (size_t i = 0; i < num; i++) _weight[i] = (1 - dt / tau_rise[i]);

    assert(_num == _weight.size());
}

ExpSynapse::ExpSynapse(const ExpSynapse &s, size_t num) : Synapse(Exp, 0) {
    append(dynamic_cast<const Synapse *>(&s), num);
}

ExpSynapse::~ExpSynapse() {
    _num = 0;
    _delay.clear();
    _weight.clear();
}

int ExpSynapse::append(const Synapse *syn, size_t num) {
    const ExpSynapse *that = dynamic_cast<const ExpSynapse *>(syn);
    size_t ret = 0;
    if ((num > 0) && (num != that->size())) {
        ret = num;
        _s.insert(_s.end(), num, that->_s[0]);
        _weight.insert(_weight.end(), num, that->_weight[0]);
        _g.insert(_g.end(), num, that->_g[0]);
        _delay.insert(_delay.end(), num, that->_delay[0]);
    } else {
        ret = that->_num;
        _s.insert(_s.end(), that->_s.begin(), that->_s.end());
        _weight.insert(_weight.end(), that->_weight.begin(),
                       that->_weight.end());
        _g.insert(_g.end(), that->_g.begin(), that->_g.end());
        _delay.insert(_delay.end(), that->_delay.begin(), that->_delay.end());
    }
    _num += ret;
    assert(_num == _weight.size());

    return ret;
}

void *ExpSynapse::packup() {
    ExpData *p = static_cast<ExpData *>(mallocExp());

    p->num = _num;

    p->pS = _s.data();
    p->pWeight = _weight.data();
    p->pG = _g.data();

    p->is_view = true;

    return p;
}

int ExpSynapse::packup(void *data, size_t dst, size_t src) {
    ExpData *p = static_cast<ExpData *>(data);

    p->pS[dst] = _s[src];
    p->pWeight[dst] = _weight[src];
    p->pG[dst] = _g[src];

    return 0;
}
