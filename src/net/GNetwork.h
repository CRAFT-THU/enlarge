/* This header file is writen by qp09
 * usually just for fun
 * Mon January 18 2016
 */
#ifndef GNETWORK_H
#define GNETWORK_H

#include "../utils/type.h"

#include "Connection.h"

struct GNetwork {
	//Numbers of types
	int nTypeNum;
	int sTypeNum;

	// Delay info moved into connection
	// int maxDelay;
	// int minDelay;

	//Type 
	Type * pNTypes;
	Type * pSTypes;

	//Index for each type
	int *pNeuronNums;
	int *pSynapseNums;

	//Pointers to neurons
	void **ppNeurons;
	//Pointers to synapses
	void **ppSynapses;

	//Neuron to Synapse Connection
	Connection *pConnection;

};


// init and free
// This func just set pConnection to NULL
GNetwork * allocGNetwork(int nTypeNum, int sTypeNum);
GNetwork * deepcopyGNetwork(GNetwork *net);
// TODO freeGNetwork
void freeGNetwork(GNetwork * network);

// Save and Load
int saveGNetwork(GNetwork *net, FILE *f);
GNetwork *loadGNetwork(FILE *f);
bool compareGNetwork(GNetwork *n1, GNetwork *n2);

// Transfer GNetwork between CPU and GPU
// Only copy inside data arrays to GPU, the info data is left on CPU
GNetwork* copyGNetworkToGPU(GNetwork *);
int fetchGNetworkFromGPU(GNetwork *, GNetwork*);
int freeGNetworkGPU(GNetwork *);

// MPI
int copyGNetwork(GNetwork *dNet, GNetwork *sNet, int rank, int rankSize);
int sendGNetwork(GNetwork *network, int dst, int tag, MPI_Comm comm);
GNetwork * recvGNetwork(int src, int tag, MPI_Comm comm);


// Other utils
int printGNetwork(GNetwork *net, int rank = 0);

#endif /* GNETWORK_H */

