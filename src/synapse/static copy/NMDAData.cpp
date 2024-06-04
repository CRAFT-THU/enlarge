#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "../../utils/utils.h"
#include "../../../msg_utils/helper/helper_c.h"


#include "NMDAData.h"

size_t getNMDASize()
{
	return sizeof(NMDAData);
}

void *mallocNMDA()
{
	NMDAData *p = (NMDAData*)malloc(sizeof(NMDAData)*1);
	memset(p, 0, sizeof(NMDAData)*1);
	return (void*)p;
}

int allocNMDAPara(void *pCPU, size_t num)
{
	NMDAData *p = (NMDAData*)pCPU;

	p->num = num;
	p->pWeight = malloc_c<real>(num);
	p->is_view = false;

	return 0;
}

void *allocNMDA(size_t num)
{
	assert(num > 0);
	void *p = mallocNMDA();
	allocNMDAPara(p, num);
	return p;
}

int freeNMDAPara(void *pCPU)
{
	NMDAData *p = (NMDAData*)pCPU;

	// free(p->pDst);
	// p->pDst = NULL;

	if (!p->is_view) {
		free(p->pWeight);
		p->pWeight = NULL;
	}

	return 0;
}

int freeNMDA(void *pCPU)
{
	NMDAData *p = (NMDAData*)pCPU;

	freeNMDAPara(p);
	free(p);
	p = NULL;
	return 0;
}

int saveNMDA(void *pCPU, size_t num, const string &path)
{
	string name = path + "/static.synapse";
	FILE *f = fopen_c(name.c_str(), "w");

	NMDAData *p = (NMDAData*)pCPU;
	assert(num <= p->num);
	if (num <= 0) {
		num = p->num;
	}

	fwrite_c(&(num), 1, f);
	// fwrite(p->pDst, sizeof(int), num, f);
	fwrite_c(p->pWeight, num, f);

	fclose_c(f);

	return 0;
}

void *loadNMDA(size_t num, const string &path)
{
	string name = path + "/static.synapse";
	FILE *f = fopen_c(name.c_str(), "r");

	NMDAData *p = (NMDAData*)allocNMDA(num);


	fread_c(&(p->num), 1, f);
	assert(num == p->num);

	// fread(p->pDst, sizeof(int), num, f);
	fread_c(p->pWeight, num, f);

	fclose_c(f);

	return p;
}

bool isEqualNMDA(void *p1, void *p2, size_t num, uinteger_t *shuffle1, uinteger_t *shuffle2)
{
	NMDAData *t1 = (NMDAData*)p1;
	NMDAData *t2 = (NMDAData*)p2;

	bool ret = true;
	// ret = ret && isEqualArray(t1->pDst, t2->pDst, num);

	ret = ret && isEqualArray(t1->pWeight, t2->pWeight, num, shuffle1, shuffle2);

	return ret;
}

int shuffleNMDA(void *p, uinteger_t *shuffle, size_t num)
{
	NMDAData *d = static_cast<NMDAData *>(p);
	assert(num == d->num);

	real *tmp = malloc_c<real>(d->num);
	memcpy_c(tmp, d->pWeight, d->num);

	for (size_t i=0; i<num; i++) {
		d->pWeight[i] = tmp[shuffle[i]];
	}

	return num;
}


