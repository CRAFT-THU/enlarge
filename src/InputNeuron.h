/* This header file is writen by qp09
 * usually just for fun
 * Wed January 06 2016
 */
#ifndef INPUTNEURON_H
#define INPUTNEURON_H

#include <stdio.h>
#include <vector>
#include <deque>

#include "NeuronBase.h"

using std::vector;
using std::deque;

class InputNeuron: public NeuronBase {
public:
	InputNeuron(ID id);
	InputNeuron(const InputNeuron &templ, ID id);
	~InputNeuron();

	virtual ID getID();

	virtual int fire();
	virtual int recv(real I);

	virtual int reset(SimInfo &info);
	virtual int update(SimInfo &info);
	virtual void monitor(SimInfo &info);

	virtual size_t getSize();
	virtual int getData(void *data);
	virtual int hardCopy(void *data, int idx);
	virtual SynapseBase *addSynapse(real weight, real delay, SpikeType type, real tau, NeuronBase *pDest);

	int addFireTime(int cycle);
protected:
	deque<int> fireTime;
	real _dt;
	real tau_syn_E;
	real tau_syn_I;
	FILE *file;
	ID m_id;
};

#endif /* INPUTNEURON_H */

