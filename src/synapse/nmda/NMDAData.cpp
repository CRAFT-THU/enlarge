#include "NMDAData.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "../../../msg_utils/helper/helper_c.h"
#include "../../utils/utils.h"

size_t getNMDASize() { return sizeof(NMDAData); }

void *mallocNMDA() {
    NMDAData *p = (NMDAData *)malloc(sizeof(NMDAData) * 1);
    memset(p, 0, sizeof(NMDAData) * 1);
    return (void *)p;
}

int allocNMDAPara(void *pCPU, size_t num) {
    NMDAData *p = (NMDAData *)pCPU;

    p->num = num;

    p->pS = malloc_c<real>(num);
    p->pX = malloc_c<real>(num);
    p->pC_decay = malloc_c<real>(num);
    p->pC_rise = malloc_c<real>(num);
    p->pG = malloc_c<real>(num);

    p->is_view = false;

    return 0;
}

void *allocNMDA(size_t num) {
    assert(num > 0);
    void *p = mallocNMDA();
    allocNMDAPara(p, num);
    return p;
}

int freeNMDAPara(void *pCPU) {
    NMDAData *p = (NMDAData *)pCPU;

    if (!p->is_view) {
        p->pS = free_c(p->pS);
        p->pX = free_c(p->pX);
        p->pC_decay = free_c(p->pC_decay);
        p->pC_rise = free_c(p->pC_rise);
        p->pG = free_c(p->pG);
    }

    return 0;
}

int freeNMDA(void *pCPU) {
    NMDAData *p = (NMDAData *)pCPU;

    freeNMDAPara(p);
    p = free_c(p);
    return 0;
}

int saveNMDA(void *pCPU, size_t num, const string &path) {
    string name = path + "/nmda.synapse";
    FILE *f = fopen_c(name.c_str(), "w");

    NMDAData *p = (NMDAData *)pCPU;
    assert(num <= p->num);
    if (num <= 0) {
        num = p->num;
    }

    fwrite_c(&(num), 1, f);
    fwrite_c(p->pS, num, f);
    fwrite_c(p->pX, num, f);
    fwrite_c(p->pC_decay, num, f);
    fwrite_c(p->pC_rise, num, f);
    fwrite_c(p->pG, num, f);

    fclose_c(f);

    return 0;
}

void *loadNMDA(size_t num, const string &path) {
    string name = path + "/nmda.synapse";
    FILE *f = fopen_c(name.c_str(), "r");

    NMDAData *p = (NMDAData *)allocNMDA(num);

    fread_c(&(p->num), 1, f);
    assert(num == p->num);

    fread_c(p->pS, num, f);
    fread_c(p->pX, num, f);
    fread_c(p->pC_decay, num, f);
    fread_c(p->pC_rise, num, f);
    fread_c(p->pG, num, f);

    fclose_c(f);

    return p;
}

bool isEqualNMDA(void *p1, void *p2, size_t num, uinteger_t *shuffle1,
                uinteger_t *shuffle2) {
    NMDAData *t1 = (NMDAData *)p1;
    NMDAData *t2 = (NMDAData *)p2;

    bool ret = t1->num == t2->num;
    ret = ret && isEqualArray(t1->pS, t2->pS, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->pX, t2->pX, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->pC_decay, t2->pC_decay, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->pC_rise, t2->pC_rise, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->pG, t2->pG, num, shuffle1, shuffle2);

    return ret;
}

int shuffleNMDA(void *p, uinteger_t *shuffle, size_t num) {
    NMDAData *d = static_cast<NMDAData *>(p);
    assert(num == d->num);

    real *tmp_s = malloc_c<real>(d->num);
    memcpy_c(tmp_s, d->pS, d->num);
    real *tmp_x = malloc_c<real>(d->num);
    memcpy_c(tmp_x, d->pX, d->num);
    real *tmp_c_decay = malloc_c<real>(d->num);
    memcpy_c(tmp_c_decay, d->pC_decay, d->num);
    real *tmp_c_rise = malloc_c<real>(d->num);
    memcpy_c(tmp_c_rise, d->pC_rise, d->num);
    real *tmp_g = malloc_c<real>(d->num);
    memcpy_c(tmp_g, d->pG, d->num);

    for (size_t i = 0; i < num; i++) {
        d->pS[i] = tmp_s[shuffle[i]];
        d->pX[i] = tmp_x[shuffle[i]];
        d->pC_decay[i] = tmp_c_decay[shuffle[i]];
        d->pC_rise[i] = tmp_c_rise[shuffle[i]];
        d->pG[i] = tmp_g[shuffle[i]];
    }

    return num;
}
