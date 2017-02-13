/* This program is writen by qp09.
 * usually just for fun.
 * Thu February 02 2017
 */

#include <stdlib.h>

#include "GPossionNeurons.h"

#include "../utils/macros.h"
#include "../utils/IDPool.h"

NEURON_GPU_FUNC_BASIC(Possion)

int allocPossionNeurons(void *pCpu, int N)
{
	GPossionNeurons *p = (GPossionNeurons *)pCpu;
	p->p_rate = (real*)malloc(N*sizeof(real));
	p->p_fire_cycle = (int*)malloc(N*sizeof(int));
	p->p_end_cycle = (int*)malloc(N*sizeof(int));
	p->p_refrac_step = (int*)malloc(N*sizeof(int));

	return 0;
}


