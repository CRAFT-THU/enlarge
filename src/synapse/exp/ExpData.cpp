#include "ExpData.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "../../../msg_utils/helper/helper_c.h"
#include "../../utils/utils.h"

size_t getExpSize() { return sizeof(ExpData); }

void *mallocExp() {
    ExpData *p = (ExpData *)malloc(sizeof(ExpData) * 1);
    memset(p, 0, sizeof(ExpData) * 1);
    return (void *)p;
}

int allocExpPara(void *pCPU, size_t num) {
    ExpData *p = (ExpData *)pCPU;

    p->num = num;

    p->s = malloc_c<real>(num);
    p->weight = malloc_c<real>(num);
    p->g = malloc_c<real>(num);

    p->is_view = false;

    return 0;
}

void *allocExp(size_t num) {
    assert(num > 0);
    void *p = mallocExp();
    allocExpPara(p, num);
    return p;
}

int freeExpPara(void *pCPU) {
    ExpData *p = (ExpData *)pCPU;

    if (!p->is_view) {
        p->s = free_c(p->s);
        p->weight = free_c(p->weight);
        p->g = free_c(p->g);
    }

    return 0;
}

int freeExp(void *pCPU) {
    ExpData *p = (ExpData *)pCPU;

    freeExpPara(p);
    p = free_c(p);
    return 0;
}

int saveExp(void *pCPU, size_t num, const string &path) {
    string name = path + "/exp.synapse";
    FILE *f = fopen_c(name.c_str(), "w");

    ExpData *p = (ExpData *)pCPU;
    assert(num <= p->num);
    if (num <= 0) {
        num = p->num;
    }

    fwrite_c(&(num), 1, f);
    fwrite_c(p->s, num, f);
    fwrite_c(p->weight, num, f);
    fwrite_c(p->g, num, f);

    fclose_c(f);

    return 0;
}

void *loadExp(size_t num, const string &path) {
    string name = path + "/exp.synapse";
    FILE *f = fopen_c(name.c_str(), "r");

    ExpData *p = (ExpData *)allocExp(num);

    fread_c(&(p->num), 1, f);
    assert(num == p->num);

    fread_c(p->s, num, f);
    fread_c(p->weight, num, f);
    fread_c(p->g, num, f);

    fclose_c(f);

    return p;
}

bool isEqualExp(void *p1, void *p2, size_t num, uinteger_t *shuffle1,
                uinteger_t *shuffle2) {
    ExpData *t1 = (ExpData *)p1;
    ExpData *t2 = (ExpData *)p2;

    bool ret = isEqualArray(t1->s, t2->s, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->weight, t2->weight, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->g, t2->g, num, shuffle1, shuffle2);

    return ret;
}

int shuffleExp(void *p, uinteger_t *shuffle, size_t num) {
    ExpData *d = static_cast<ExpData *>(p);
    assert(num == d->num);

    real *tmp_s = malloc_c<real>(d->num);
    memcpy_c(tmp_s, d->s, d->num);
    real *tmp_w = malloc_c<real>(d->num);
    memcpy_c(tmp_w, d->weight, d->num);
    real *tmp_g = malloc_c<real>(d->num);
    memcpy_c(tmp_g, d->g, d->num);

    for (size_t i = 0; i < num; i++) {
        d->s[i] = tmp_s[shuffle[i]];
        d->weight[i] = tmp_w[shuffle[i]];
        d->g[i] = tmp_g[shuffle[i]];
    }

    return num;
}
