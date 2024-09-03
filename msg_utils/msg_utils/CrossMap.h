
#ifndef CROSSMAP_H
#define CROSSMAP_H

#include "mpi.h"

#include "../helper/helper_type.h"

class CrossMap {
public:
	CrossMap();
	CrossMap(size_t num, size_t cross_size);
	CrossMap(size_t num, size_t cross_num, size_t node_num);
	~CrossMap();

	int send(int dest, int tag, MPI_Comm comm);
	int recv(int src, int tag, MPI_Comm comm);
	int save(FILE *f);
	int load(FILE *f);
	int compare(CrossMap &m);

	int to_gpu();

	int log(const char *name);

public:
	// ID of neurons on this node to index in this map 
	// index = _idx2index[id]
	integer_t *_idx2index;
	// idx in this map to ID of shadow neurons on node j
	// id = _index2ridx[index * node_num + j], -1 means no shadow neuron on node j
	integer_t *_index2ridx;

	CrossMap * _gpu_array;

	// _num number of neurons on this node
	// _cross_size = node_num * number_of_the_neurons_on_this_node_which_have_crossnode_connections
	size_t _cross_size;
	size_t _num;
};


// int saveCNM(CrossMap *map, FILE *f);
// CrossMap *loadCNM(FILE *f); 
// int compareCNM(CrossMap *m1, CrossMap *m2);
// 
// int sendMap(CrossMap * network, int dest, int tag, MPI_Comm comm);
// CrossMap * recvMap(int src, int tag, MPI_Comm comm);


#endif // CROSSMAP_H
