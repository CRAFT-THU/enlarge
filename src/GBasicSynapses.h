/* This header file is writen by qp09
 * usually just for fun
 * Tue December 27 2016
 */
#ifndef GBASICSYNAPSES_H
#define GBASICSYNAPSES_H

#include "macros.h"
#include "constant.h"

struct GBasicSynapses {
	real *p_weight;
	int *p_delay_steps;

	SYNAPSE_CONNECT_PARA
};

SYNAPSE_GPU_FUNC_DEFINE(Basic)

#endif /* GBASICSYNAPSES_H */

