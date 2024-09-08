
#include "LIFHomoData.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "../../../msg_utils/helper/helper_c.h"
#include "../../utils/utils.h"

size_t getLIFHomoSize() { return sizeof(LIFHomoData); }

void *mallocLIFHomo() {
    LIFHomoData *p = (LIFHomoData *)malloc(sizeof(LIFHomoData) * 1);
    memset(p, 0, sizeof(LIFHomoData) * 1);
    return (void *)p;
}

int allocLIFHomoPara(void *pCPU, size_t num) {
    LIFHomoData *p = (LIFHomoData *)pCPU;

    p->num = num;

    p->pRefracStep = malloc_c<int>(num);
    p->pI_e = malloc_c<real>(num);
    p->pI_i = malloc_c<real>(num);
    p->pV_m = malloc_c<real>(num);

    p->input_sz = 0;
    p->pInput_start = malloc_c<int>(num);
    p->pInput = nullptr;

    p->_fire_count = malloc_c<int>(num);

    p->is_view = false;

    return 0;
}

void *allocLIFHomo(size_t num) {
    assert(num > 0);
    void *p = mallocLIFHomo();
    allocLIFHomoPara(p, num);
    return p;
}

int freeLIFHomoPara(void *pCPU) {
    LIFHomoData *p = (LIFHomoData *)pCPU;

    if (!p->is_view) {
        p->pRefracStep = free_c(p->pRefracStep);
        p->pI_e = free_c(p->pI_e);
        p->pI_i = free_c(p->pI_i);
        p->pV_m = free_c(p->pV_m);

        p->pInput_start = free_c(p->pInput_start);
        p->pInput = free_c(p->pInput);
    }

    p->_fire_count = free_c(p->_fire_count);
    p->num = 0;
    p->input_sz = 0;

    return 0;
}

int freeLIFHomo(void *pCPU) {
    LIFHomoData *p = (LIFHomoData *)pCPU;

    freeLIFHomoPara(p);
    p = free_c(p);
    return 0;
}

int saveLIFHomo(void *pCPU, size_t num, const string &path) {
    string name = path + "/lif.neuron";
    FILE *f = fopen(name.c_str(), "w");

    LIFHomoData *p = (LIFHomoData *)pCPU;
    assert(num <= p->num);
    if (num <= 0) num = p->num;

    fwrite_c(&num, 1, f);
    fwrite_c(p->pRefracStep, num, f);
    fwrite_c(p->pI_e, num, f);
    fwrite_c(p->pI_i, num, f);
    fwrite_c(p->pV_m, num, f);

    fwrite_c(&(p->cRefracTime), 1, f);
    fwrite_c(&(p->cV_reset), 1, f);
    fwrite_c(&(p->cV_tmp), 1, f);
    fwrite_c(&(p->cV_thresh), 1, f);
    fwrite_c(&(p->cCe), 1, f);
    fwrite_c(&(p->cCi), 1, f);
    fwrite_c(&(p->cC_e), 1, f);
    fwrite_c(&(p->cC_m), 1, f);
    fwrite_c(&(p->cC_i), 1, f);

    fwrite_c(&(p->input_sz), 1, f);
    fwrite_c(p->pInput_start, num, f);
    fwrite_c(p->pInput, p->input_sz, f);

    fwrite_c(p->_fire_count, num, f);

    fclose_c(f);

    return 0;
}

void *loadLIFHomo(size_t num, const string &path) {
    string name = path + "/lif.neuron";
    FILE *f = fopen(name.c_str(), "r");

    LIFHomoData *p = (LIFHomoData *)allocLIFHomo(num);

    fread_c(&(p->num), 1, f);

    assert(num == p->num);

    fread_c(p->pRefracStep, num, f);
    fread_c(p->pI_e, num, f);
    fread_c(p->pI_i, num, f);
    fread_c(p->pV_m, num, f);

    fread_c(&(p->cRefracTime), 1, f);
    fread_c(&(p->cV_reset), 1, f);
    fread_c(&(p->cV_tmp), 1, f);
    fread_c(&(p->cV_thresh), 1, f);
    fread_c(&(p->cCe), 1, f);
    fread_c(&(p->cCi), 1, f);
    fread_c(&(p->cC_e), 1, f);
    fread_c(&(p->cC_m), 1, f);
    fread_c(&(p->cC_i), 1, f);

    fread_c(&(p->input_sz), 1, f);
    fread_c(p->pInput_start, num, f);

    p->pInput = malloc_c<real>(p->input_sz);
    fread_c(p->pInput, p->input_sz, f);

    fread_c(p->_fire_count, num, f);

    fclose_c(f);

    return p;
}

bool isEqualLIFHomo(void *p1, void *p2, size_t num, uinteger_t *shuffle1,
                    uinteger_t *shuffle2) {
    LIFHomoData *t1 = (LIFHomoData *)p1;
    LIFHomoData *t2 = (LIFHomoData *)p2;

    bool ret = t1->num == t2->num;
    ret = ret && isEqualArray(t1->pRefracStep, t2->pRefracStep, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->pI_e, t2->pI_e, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->pI_i, t2->pI_i, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->pV_m, t2->pV_m, num, shuffle1, shuffle2);
    
	ret = ret && (t1->cRefracTime == t2->cRefracTime);
	ret = ret && (fabs(t1->cV_reset - t2->cV_reset) < ZERO);
	ret = ret && (fabs(t1->cV_tmp - t2->cV_tmp) < ZERO);
	ret = ret && (fabs(t1->cV_thresh - t2->cV_thresh) < ZERO);
	ret = ret && (fabs(t1->cCe - t2->cCe) < ZERO);
	ret = ret && (fabs(t1->cCi - t2->cCi) < ZERO);
	ret = ret && (fabs(t1->cC_e - t2->cC_e) < ZERO);
	ret = ret && (fabs(t1->cC_m - t2->cC_m) < ZERO);
	ret = ret && (fabs(t1->cC_i - t2->cC_i) < ZERO);

    return ret;
}

int copyLIFHomo(void *p_src, size_t s_off, void *p_dst, size_t d_off) {
    LIFHomoData *src = static_cast<LIFHomoData *>(p_src);
    LIFHomoData *dst = static_cast<LIFHomoData *>(p_dst);

    dst->pRefracStep[d_off] = src->pRefracStep[s_off];
	dst->pI_e[d_off] = src->pI_e[s_off];
	dst->pI_i[d_off] = src->pI_i[s_off];
	dst->pV_m[d_off] = src->pV_m[s_off];

    return 0;
}

int logRateLIFHomo(void *data, const char *name) {
    char filename[512];
    sprintf(filename, "rate_%s.%s.log", name, "LIFHomo");
    FILE *f = fopen_c(filename, "w+");
    LIFHomoData *d = static_cast<LIFHomoData *>(data);
    log_array(f, d->_fire_count, d->num);
    fclose_c(f);
    return 0;
}

real *getVLIFHomo(void *data) {
    LIFHomoData *p = static_cast<LIFHomoData *>(data);
    return p->pV_m;
}