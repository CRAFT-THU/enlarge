
#include "NMDANrnData.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "../../../msg_utils/helper/helper_c.h"
#include "../../utils/utils.h"

size_t getNMDANrnSize() { return sizeof(NMDANrnData); }

void *mallocNMDANrn() {
    NMDANrnData *p = (NMDANrnData *)malloc(sizeof(NMDANrnData) * 1);
    memset(p, 0, sizeof(NMDANrnData) * 1);
    return (void *)p;
}

int allocNMDANrnPara(void *pCPU, size_t num) {
    NMDANrnData *p = (NMDANrnData *)pCPU;

    p->num = num;

    p->s = malloc_c<real>(num);
    p->x = malloc_c<real>(num);

    p->coeff = malloc_c<real>(num);
    p->tau_decay_rcpl = malloc_c<real>(num);
    p->tau_rise_compl = malloc_c<real>(num);

    // p->_fire_count = malloc_c<int>(num);

    p->is_view = false;

    return 0;
}

void *allocNMDANrn(size_t num) {
    assert(num > 0);
    void *p = mallocNMDANrn();
    allocNMDANrnPara(p, num);
    return p;
}

int freeNMDANrnPara(void *pCPU) {
    NMDANrnData *p = (NMDANrnData *)pCPU;

    p->num = 0;

    if (!p->is_view) {
        free_c(p->s);
        free_c(p->x);
        free_c(p->coeff);
        free_c(p->tau_decay_rcpl);
        free_c(p->tau_rise_compl);
    }

    // free_c(p->_fire_count);

    return 0;
}

int freeNMDANrn(void *pCPU) {
    NMDANrnData *p = (NMDANrnData *)pCPU;

    freeNMDANrnPara(p);
    free_c(p);
    return 0;
}

int saveNMDANrn(void *pCPU, size_t num, const string &path) {
    string name = path + "/nmda.neuron";
    FILE *f = fopen(name.c_str(), "w");

    NMDANrnData *p = (NMDANrnData *)pCPU;
    assert(num <= p->num);
    if (num <= 0) num = p->num;

    fwrite(&num, sizeof(size_t), 1, f);

    fwrite(p->s, sizeof(real), num, f);
    fwrite(p->x, sizeof(real), num, f);
    fwrite(p->coeff, sizeof(real), num, f);
    fwrite(p->tau_decay_rcpl, sizeof(real), num, f);
    fwrite(p->tau_rise_compl, sizeof(real), num, f);
    // fwrite_c(p->_fire_count, num, f);

    fclose_c(f);

    return 0;
}

void *loadNMDANrn(size_t num, const string &path) {
    string name = path + "/nmda.neuron";
    FILE *f = fopen(name.c_str(), "r");

    NMDANrnData *p = (NMDANrnData *)allocNMDANrn(num);

    fread_c(&(p->num), 1, f);

    assert(num == p->num);

    fread_c(p->s, num, f);
    fread_c(p->x, num, f);
    fread_c(p->coeff, num, f);
    fread_c(p->tau_decay_rcpl, num, f);
    fread_c(p->tau_rise_compl, num, f);

    fclose_c(f);

    return p;
}

bool isEqualNMDANrn(void *p1, void *p2, size_t num, uinteger_t *shuffle1,
                 uinteger_t *shuffle2) {
    NMDANrnData *t1 = (NMDANrnData *)p1;
    NMDANrnData *t2 = (NMDANrnData *)p2;

    bool ret = t1->num == t2->num;
    ret = ret && isEqualArray(t1->s, t2->s, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->x, t2->x, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->coeff, t2->coeff, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->tau_decay_rcpl, t2->tau_decay_rcpl, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->tau_rise_compl, t2->tau_rise_compl, num, shuffle1, shuffle2);

    return ret;
}

// int copyNMDANrn(void *p_src, size_t s_off, void *p_dst, size_t d_off) {
//     NMDANrnData *src = static_cast<NMDANrnData *>(p_src);
//     NMDANrnData *dst = static_cast<NMDANrnData *>(p_dst);

//     dst->pRefracTime[d_off] = src->pRefracTime[s_off];
//     return 0;
// }

// int logRateNMDANrn(void *data, const char *name) {
//     char filename[512];
//     sprintf(filename, "rate_%s.%s.log", name, "NMDANrn");
//     FILE *f = fopen_c(filename, "w+");
//     NMDANrnData *d = static_cast<NMDANrnData *>(data);
//     log_array(f, d->_fire_count, d->num);
//     fclose_c(f);
//     return 0;
// }

real *getSNMDANrn(void *data) {
    NMDANrnData *p = static_cast<NMDANrnData *>(data);
    return p->s;
}