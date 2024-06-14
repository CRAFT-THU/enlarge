
#include "LIFExpData.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "../../../msg_utils/helper/helper_c.h"
#include "../../utils/utils.h"

size_t getLIFExpSize() { return sizeof(LIFExpData); }

void *mallocLIFExp() {
    LIFExpData *p = (LIFExpData *)malloc(sizeof(LIFExpData) * 1);
    memset(p, 0, sizeof(LIFExpData) * 1);
    return (void *)p;
}

int allocLIFExpPara(void *pCPU, size_t num) {
    LIFExpData *p = (LIFExpData *)pCPU;

    p->num = num;

    p->pRefracTime = malloc_c<int>(num);
    p->pRefracStep = malloc_c<int>(num);

    p->pV = malloc_c<real>(num);
    p->pV_tmp = malloc_c<real>(num);
    p->pV_thresh = malloc_c<real>(num);
    p->pV_reset = malloc_c<real>(num);

    p->pR = malloc_c<real>(num);
    p->pC_m = malloc_c<real>(num);
    p->pE = malloc_c<real>(num);

    p->pI = malloc_c<real>(num);

    p->_fire_count = malloc_c<int>(num);

    p->is_view = false;

    return 0;
}

void *allocLIFExp(size_t num) {
    assert(num > 0);
    void *p = mallocLIFExp();
    allocLIFExpPara(p, num);
    return p;
}

int freeLIFExpPara(void *pCPU) {
    LIFExpData *p = (LIFExpData *)pCPU;

    p->num = 0;

    if (!p->is_view) {
        p->pRefracTime = free_c(p->pRefracTime);
        p->pRefracStep = free_c(p->pRefracStep);

        p->pV = free_c(p->pV);
        p->pV_tmp = free_c(p->pV_tmp);
        p->pV_thresh = free_c(p->pV_thresh);
        p->pV_reset = free_c(p->pV_reset);

        p->pR = free_c(p->pR);
        p->pC_m = free_c(p->pC_m);
        p->pE = free_c(p->pE);

        p->pI = free_c(p->pI);
    }
    free_c(p->_fire_count);

    return 0;
}

int freeLIFExp(void *pCPU) {
    LIFExpData *p = (LIFExpData *)pCPU;

    freeLIFExpPara(p);
    p = free_c(p);
    return 0;
}

int saveLIFExp(void *pCPU, size_t num, const string &path) {
    string name = path + "/lifexp.neuron";
    FILE *f = fopen(name.c_str(), "w");

    LIFExpData *p = (LIFExpData *)pCPU;
    assert(num <= p->num);
    if (num <= 0) num = p->num;

    fwrite(&num, sizeof(size_t), 1, f);

    // ! 注意是先Time后Step，和LIFData保持一致
    fwrite(p->pRefracTime, sizeof(int), num, f);
    fwrite(p->pRefracStep, sizeof(int), num, f);

    fwrite(p->pV, sizeof(real), num, f);
    fwrite(p->pV_tmp, sizeof(real), num, f);
    fwrite(p->pV_thresh, sizeof(real), num, f);
    fwrite(p->pV_reset, sizeof(real), num, f);

    fwrite(p->pR, sizeof(real), num, f);
    fwrite(p->pC_m, sizeof(real), num, f);
    fwrite(p->pE, sizeof(real), num, f);

    fwrite(p->pI, sizeof(real), num, f);

    fwrite_c(p->_fire_count, num, f);

    fclose_c(f);

    return 0;
}

void *loadLIFExp(size_t num, const string &path) {
    string name = path + "/lif.neuron";
    FILE *f = fopen(name.c_str(), "r");

    LIFExpData *p = (LIFExpData *)allocLIFExp(num);

    fread_c(&(p->num), 1, f);

    assert(num == p->num);

    // ! 注意是先Time后Step，和LIFData保持一致
    fread_c(p->pRefracTime, num, f);
    fread_c(p->pRefracStep, num, f);

    fread_c(p->pV, num, f);
    fread_c(p->pV_tmp, num, f);
    fread_c(p->pV_thresh, num, f);
    fread_c(p->pV_reset, num, f);

    fread_c(p->pR, num, f);
    fread_c(p->pC_m, num, f);
    fread_c(p->pE, num, f);

    fread_c(p->pI, num, f);

    fread_c(p->_fire_count, num, f);

    fclose_c(f);

    return p;
}

bool isEqualLIFExp(void *p1, void *p2, size_t num, uinteger_t *shuffle1,
                uinteger_t *shuffle2) {
    LIFExpData *t1 = (LIFExpData *)p1;
    LIFExpData *t2 = (LIFExpData *)p2;

    bool ret = t1->num == t2->num;
    ret = ret && isEqualArray(t1->pRefracTime, t2->pRefracTime, num, shuffle1,
                              shuffle2);
    ret = ret && isEqualArray(t1->pRefracStep, t2->pRefracStep, num, shuffle1,
                              shuffle2);

    ret = ret && isEqualArray(t1->pV, t2->pV, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->pV_tmp, t2->pV_tmp, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->pV_thresh, t2->pV_thresh, num, shuffle1,
                              shuffle2);
    ret = ret && isEqualArray(t1->pV_reset, t2->pV_reset, num, shuffle1, shuffle2);

    ret = ret && isEqualArray(t1->pR, t2->pR, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->pC_m, t2->pC_m, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->pE, t2->pE, num, shuffle1, shuffle2);

    ret = ret && isEqualArray(t1->pI, t2->pI, num, shuffle1, shuffle2);

    return ret;
}

int copyLIFExp(void *p_src, size_t s_off, void *p_dst, size_t d_off) {
    LIFExpData *src = static_cast<LIFExpData *>(p_src);
    LIFExpData *dst = static_cast<LIFExpData *>(p_dst);

    dst->pRefracTime[d_off] = src->pRefracTime[s_off];
    dst->pRefracStep[d_off] = src->pRefracStep[s_off];

    dst->pV[d_off] = src->pV[s_off];
    dst->pV_tmp[d_off] = src->pV_tmp[s_off];
    dst->pV_thresh[d_off] = src->pV_thresh[s_off];
    dst->pV_reset[d_off] = src->pV_reset[s_off];

    dst->pR[d_off] = src->pR[s_off];
    dst->pC_m[d_off] = src->pC_m[s_off];
    dst->pE[d_off] = src->pE[s_off];

    dst->pI[d_off] = src->pI[s_off];

    return 0;
}

int logRateLIFExp(void *data, const char *name) {
    char filename[512];
    sprintf(filename, "rate_%s.%s.log", name, "LIFExp");
    FILE *f = fopen_c(filename, "w+");
    LIFExpData *d = static_cast<LIFExpData *>(data);
    log_array(f, d->_fire_count, d->num);
    fclose_c(f);
    return 0;
}

real *getVLIFExp(void *data) {
    LIFExpData *p = static_cast<LIFExpData *>(data);
    return p->pV;
}