#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "../../utils/utils.h"
#include "../../../msg_utils/helper/helper_c.h"


#include "NMDASynData.h"

size_t getNMDASynSize()
{
	return sizeof(NMDASynData);
}

void *mallocNMDASyn()
{
	NMDASynData *p = (NMDASynData*)malloc(sizeof(NMDASynData)*1);
	memset(p, 0, sizeof(NMDASynData)*1);
	return (void*)p;
}

int allocNMDASynPara(void *pCPU, size_t num)
{
	NMDASynData *p = (NMDASynData*)pCPU;

	p->num = num;
	p->g = malloc_c<real>(num);
    p->M_ca_coeff = malloc_c<real>(num);
    p->M_c = malloc_c<real>(num);
    p->E_syn = malloc_c<real>(num);

	p->is_view = false;

	return 0;
}

void *allocNMDASyn(size_t num)
{
	assert(num > 0);
	void *p = mallocNMDASyn();
	allocNMDASynPara(p, num);
	return p;
}

int freeNMDASynPara(void *pCPU)
{
	NMDASynData *p = (NMDASynData*)pCPU;

	// free(p->pDst);
	// p->pDst = NULL;

	if (!p->is_view) {
		p->g = free_c(p->g);
        p->M_ca_coeff = free_c(p->M_ca_coeff);
        p->M_c = free_c(p->M_c);
        p->E_syn = free_c(p->E_syn);
	}

	return 0;
}

int freeNMDASyn(void *pCPU)
{
	NMDASynData *p = (NMDASynData*)pCPU;

	freeNMDASynPara(p);
	p = free_c(p);
	return 0;
}

int saveNMDASyn(void *pCPU, size_t num, const string &path)
{
	string name = path + "/nmda.synapse";
	FILE *f = fopen_c(name.c_str(), "w");

	NMDASynData *p = (NMDASynData*)pCPU;
	assert(num <= p->num);
	if (num <= 0) {
		num = p->num;
	}

	fwrite_c(&(num), 1, f);
	
	fwrite_c(p->g, num, f);
    fwrite_c(p->M_ca_coeff, num, f);
    fwrite_c(p->M_c, num, f);
    fwrite_c(p->E_syn, num, f);

	fclose_c(f);

	return 0;
}

void *loadNMDASyn(size_t num, const string &path)
{
	string name = path + "/static.synapse";
	FILE *f = fopen_c(name.c_str(), "r");

	NMDASynData *p = (NMDASynData*)allocNMDASyn(num);


	fread_c(&(p->num), 1, f);
	assert(num == p->num);

	fread_c(p->g, num, f);
    fread_c(p->M_ca_coeff, num, f);
    fread_c(p->M_c, num, f);
    fread_c(p->E_syn, num, f);

	fclose_c(f);

	return p;
}

bool isEqualNMDASyn(void *p1, void *p2, size_t num, uinteger_t *shuffle1, uinteger_t *shuffle2)
{
	NMDASynData *t1 = (NMDASynData*)p1;
	NMDASynData *t2 = (NMDASynData*)p2;

	bool ret = true;

	ret = ret && isEqualArray(t1->g, t2->g, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->M_ca_coeff, t2->M_ca_coeff, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->M_c, t2->M_c, num, shuffle1, shuffle2);
    ret = ret && isEqualArray(t1->E_syn, t2->E_syn, num, shuffle1, shuffle2);

	return ret;
}

// int shuffleNMDASyn(void *p, uinteger_t *shuffle, size_t num)
// {
// 	NMDASynData *d = static_cast<NMDASynData *>(p);
// 	assert(num == d->num);

// 	real *tmp = malloc_c<real>(d->num);
// 	memcpy_c(tmp, d->pWeight, d->num);

// 	for (size_t i=0; i<num; i++) {
// 		d->pWeight[i] = tmp[shuffle[i]];
// 	}

// 	return num;
// }